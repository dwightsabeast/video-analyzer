#!/usr/bin/env python3
"""
va_quality - reference-based quality metrics (PSNR / SSIM / VMAF).

Compares a distorted file against a reference (e.g. an encode vs its master)
using ffmpeg's psnr, ssim and libvmaf filters. PSNR/SSIM run in this build;
VMAF is gated on libvmaf being present in the ffmpeg binary.

Stats logs are written into a private temp directory and referenced by a
RELATIVE filename with ``cwd=`` pointing at that directory: absolute paths
break inside lavfi filtergraphs on Windows (the drive-letter ``:`` reads as
an option separator). Mismatched resolutions are scaled to the reference
before comparison and mismatched durations stop at the shorter input
(``shortest=1``) - both are surfaced in the result notes.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import tempfile
import subprocess

from va_ffmpeg import find_ffmpeg, has_filter, probe

try:
    import va_perf
except ImportError:    # engine modules stay importable standalone
    va_perf = None

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
_TIMEOUT = 900  # seconds; full-file null-mux passes on long movies are slow


def vmaf_available() -> bool:
    return has_filter("libvmaf")


def xpsnr_available() -> bool:
    """XPSNR ships in ffmpeg >= 7.1 (psychovisually weighted PSNR, ITU-T."""
    return has_filter("xpsnr")


# libvmaf >= 2 embeds these models; no .json files needed.
VMAF_MODELS = {"default": None, "4k": "vmaf_4k_v0.6.1", "neg": "vmaf_v0.6.1neg"}


def vmaf_model_for(choice, ref_info) -> "str | None":
    """Resolve a model choice ('auto'|'default'|'4k'|'neg') to a libvmaf
    embedded-model version string (None = library default, 1080p model).
    'auto' picks the 4K model when the reference is UHD-class - the default
    model assumes ~1080p viewing distance and over-scores 4K sources."""
    if choice in VMAF_MODELS:
        return VMAF_MODELS[choice]
    h = (ref_info or {}).get("height") or 0
    w = (ref_info or {}).get("width") or 0
    return VMAF_MODELS["4k"] if (h >= 1620 or w >= 2880) else None


def _run(args, cwd=None, timeout=_TIMEOUT):
    """ffmpeg runner: UTF-8 decode (ffmpeg output is UTF-8 even on Windows),
    bounded runtime, no console flash. Returns CompletedProcess or None."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, cwd=cwd,
            creationflags=CREATIONFLAGS)
    except subprocess.TimeoutExpired:
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def _chain(filt_expr, dist_info, ref_info):
    """Build the lavfi graph, scaling distorted to the reference geometry
    when they differ. Returns (graph, note_or_None)."""
    dw, dh = dist_info.get("width") or 0, dist_info.get("height") or 0
    rw, rh = ref_info.get("width") or 0, ref_info.get("height") or 0
    if dw and rw and (dw, dh) != (rw, rh):
        graph = ("[0:v]scale=%d:%d:flags=bicubic[d];[d][1:v]" % (rw, rh)) + filt_expr
        return graph, "distorted scaled %dx%d -> %dx%d to match reference" % (dw, dh, rw, rh)
    return "[0:v][1:v]" + filt_expr, None


def _err_tail(r):
    if r is None:
        return "ffmpeg timed out or could not run"
    tail = [ln for ln in (r.stderr or "").splitlines() if ln.strip()]
    return tail[-1][:200] if tail else "ffmpeg reported no detail"


def _metric(distorted, reference, dist_info, ref_info, filt, log_name, parse_log,
            stderr_rx=None):
    """Shared PSNR/SSIM runner. Returns dict with average/per_frame and
    optional note/error keys, or None when ffmpeg is absent."""
    exe = find_ffmpeg()
    if not exe:
        return None
    tmpdir = tempfile.mkdtemp(prefix="va_qm_")
    try:
        graph, note = _chain(filt + "=stats_file=" + log_name + ":shortest=1",
                             dist_info, ref_info)
        args = [exe, "-hide_banner", "-nostdin", "-i", distorted, "-i", reference,
                "-lavfi", graph, "-f", "null", "-"]
        r = _run(args, cwd=tmpdir)
        per = []
        try:
            with open(os.path.join(tmpdir, log_name), encoding="utf-8",
                      errors="replace") as fh:
                per = parse_log(fh)
        except OSError:
            pass
        avg = None
        if r is not None and stderr_rx:
            m = re.search(stderr_rx, r.stderr or "")
            if m:
                avg = float("inf") if m.group(1) == "inf" else float(m.group(1))
        if avg is None and not per:
            return {"average": None, "per_frame": [], "error": _err_tail(r)}
        out = {"average": avg, "per_frame": per}
        if note:
            out["note"] = note
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_psnr_log(fh):
    per = []
    for line in fh:
        m = re.search(r"psnr_avg:([-\d.]+|inf)", line)
        if m:
            per.append(float("inf") if m.group(1) == "inf" else float(m.group(1)))
    return per


def _parse_ssim_log(fh):
    per = []
    for line in fh:
        m = re.search(r"\bAll:([-\d.]+)", line)
        if m:
            per.append(float(m.group(1)))
    return per


def _psnr(distorted, reference, dist_info, ref_info):
    return _metric(distorted, reference, dist_info, ref_info, "psnr",
                   "psnr.log", _parse_psnr_log, r"average:([-\d.]+|inf)")


def _ssim(distorted, reference, dist_info, ref_info):
    return _metric(distorted, reference, dist_info, ref_info, "ssim",
                   "ssim.log", _parse_ssim_log, r"SSIM.*All:([-\d.]+)")


def _vmaf(distorted, reference, dist_info, ref_info, model="auto"):
    exe = find_ffmpeg()
    if not exe or not vmaf_available():
        return None
    mname = vmaf_model_for(model, ref_info)
    opts = "libvmaf=log_path=vmaf.json:log_fmt=json:shortest=1"
    if mname:
        opts += ":model=version=" + mname
    tmpdir = tempfile.mkdtemp(prefix="va_qm_")
    try:
        graph, note = _chain(opts, dist_info, ref_info)
        args = [exe, "-hide_banner", "-nostdin", "-i", distorted, "-i", reference,
                "-lavfi", graph, "-f", "null", "-"]
        r = _run(args, cwd=tmpdir)
        try:
            with open(os.path.join(tmpdir, "vmaf.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            return {"average": None, "per_frame": [], "error": _err_tail(r)}
        per = [f.get("metrics", {}).get("vmaf") for f in data.get("frames", [])]
        pooled = data.get("pooled_metrics", {}).get("vmaf", {})
        avg = pooled.get("mean")
        if avg is None and per:
            vals = [v for v in per if v is not None]
            avg = sum(vals) / len(vals) if vals else None
        out = {"average": avg, "per_frame": per,
               "min": pooled.get("min"), "max": pooled.get("max"),
               "model": mname or "vmaf_v0.6.1 (default)"}
        if note:
            out["note"] = note
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_xpsnr_log(fh):
    """Per-frame luma XPSNR from the stats file (one 'XPSNR y: <v>' per line)."""
    per = []
    for line in fh:
        m = re.search(r"XPSNR\s+y:\s*(inf|[-\d.]+)", line) or \
            re.search(r"\by:\s*(inf|[-\d.]+)", line)
        if m:
            per.append(float("inf") if m.group(1) == "inf" else float(m.group(1)))
    return per


def _xpsnr(distorted, reference, dist_info, ref_info):
    """Luma XPSNR (u/v reported alongside). Needs ffmpeg >= 7.1."""
    exe = find_ffmpeg()
    if not exe or not xpsnr_available():
        return None
    tmpdir = tempfile.mkdtemp(prefix="va_qm_")
    try:
        graph, note = _chain("xpsnr=stats_file=xpsnr.log:shortest=1",
                             dist_info, ref_info)
        args = [exe, "-hide_banner", "-nostdin", "-i", distorted, "-i", reference,
                "-lavfi", graph, "-f", "null", "-"]
        r = _run(args, cwd=tmpdir)
        per = []
        try:
            with open(os.path.join(tmpdir, "xpsnr.log"), encoding="utf-8",
                      errors="replace") as fh:
                per = _parse_xpsnr_log(fh)
        except OSError:
            pass
        avg = u = v = None
        if r is not None:
            m = re.search(r"XPSNR(?:\s+average[^y]*)?\s+y:\s*(inf|[-\d.]+)"
                          r"(?:\s+u:\s*(inf|[-\d.]+))?(?:\s+v:\s*(inf|[-\d.]+))?",
                          r.stderr or "")
            if m:
                conv = lambda g: None if g is None else (
                    float("inf") if g == "inf" else float(g))
                avg, u, v = conv(m.group(1)), conv(m.group(2)), conv(m.group(3))
        if avg is None and per:
            fin = [p for p in per if p != float("inf")]
            avg = (sum(fin) / len(fin)) if fin else float("inf")
        if avg is None and not per:
            return {"average": None, "per_frame": [], "error": _err_tail(r)}
        out = {"average": avg, "per_frame": per, "u": u, "v": v}
        if note:
            out["note"] = note
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Single-decode combined run -------------------------------------------------
#
# The per-metric helpers above each decode BOTH files in full, so a 4-metric
# compare costs 8 decodes. _combined_compare() chains the metrics in one
# process: the distorted stream flows THROUGH psnr -> ssim -> xpsnr -> libvmaf
# (each filter passes its primary input along unmodified) while the reference
# is split once per metric. Two decodes total, identical numbers, and libvmaf
# gets n_threads (it is single-threaded by default - on a many-core box this
# is the difference between VMAF crawling and VMAF keeping up with the decode).

def _combined_compare(distorted, reference, dist_info, ref_info, vmaf, vmaf_model):
    """All available metrics in one ffmpeg run (2 decodes). None -> caller
    falls back to the per-metric path."""
    exe = find_ffmpeg()
    if not exe:
        return None
    want = [("psnr", "psnr=stats_file=psnr.log:shortest=1"),
            ("ssim", "ssim=stats_file=ssim.log:shortest=1")]
    if xpsnr_available():
        want.append(("xpsnr", "xpsnr=stats_file=xpsnr.log:shortest=1"))
    mname = vmaf_model_for(vmaf_model, ref_info) if vmaf and vmaf_available() else None
    if vmaf and vmaf_available():
        opts = "libvmaf=log_path=vmaf.json:log_fmt=json:shortest=1"
        if va_perf is not None:
            try:
                opts += ":n_threads=%d" % max(1, va_perf.vmaf_threads())
            except Exception:
                pass
        if mname:
            opts += ":model=version=" + mname
        want.append(("vmaf", opts))

    dw, dh = dist_info.get("width") or 0, dist_info.get("height") or 0
    rw, rh = ref_info.get("width") or 0, ref_info.get("height") or 0
    note = None
    if dw and rw and (dw, dh) != (rw, rh):
        head = "[0:v]scale=%d:%d:flags=bicubic[d0];" % (rw, rh)
        note = "distorted scaled %dx%d -> %dx%d to match reference" % (dw, dh, rw, rh)
        cur = "[d0]"
    else:
        head, cur = "", "[0:v]"
    n = len(want)
    graph = head + "[1:v]split=%d%s;" % (n, "".join("[r%d]" % i for i in range(n)))
    for i, (_, expr) in enumerate(want):
        out_lbl = "" if i == n - 1 else "[m%d]" % i
        graph += "%s[r%d]%s%s;" % (cur, i, expr, out_lbl)
        cur = "[m%d]" % i
    graph = graph.rstrip(";")

    tmpdir = tempfile.mkdtemp(prefix="va_qm_")
    try:
        args = [exe, "-hide_banner", "-nostdin", "-i", distorted, "-i", reference,
                "-lavfi", graph, "-an", "-f", "null", "-"]
        r = _run(args, cwd=tmpdir)
        if r is None or r.returncode != 0:
            return None

        def log(name, parser):
            try:
                with open(os.path.join(tmpdir, name), encoding="utf-8",
                          errors="replace") as fh:
                    return parser(fh)
            except OSError:
                return []

        keys = [k for k, _ in want]
        res = {}
        if "psnr" in keys:
            per = log("psnr.log", _parse_psnr_log)
            m = re.search(r"average:([-\d.]+|inf)", r.stderr or "")
            avg = (float("inf") if m.group(1) == "inf" else float(m.group(1))) if m else None
            res["psnr"] = {"average": avg, "per_frame": per}
        if "ssim" in keys:
            per = log("ssim.log", _parse_ssim_log)
            m = re.search(r"SSIM.*All:([-\d.]+)", r.stderr or "")
            res["ssim"] = {"average": float(m.group(1)) if m else None, "per_frame": per}
        if "xpsnr" in keys:
            per = log("xpsnr.log", _parse_xpsnr_log)
            m = re.search(r"XPSNR(?:\s+average[^y]*)?\s+y:\s*(inf|[-\d.]+)"
                          r"(?:\s+u:\s*(inf|[-\d.]+))?(?:\s+v:\s*(inf|[-\d.]+))?",
                          r.stderr or "")
            conv = lambda g: None if g is None else (
                float("inf") if g == "inf" else float(g))
            res["xpsnr"] = {"average": conv(m.group(1)) if m else None,
                            "per_frame": per,
                            "u": conv(m.group(2)) if m else None,
                            "v": conv(m.group(3)) if m else None}
            if res["xpsnr"]["average"] is None and per:
                fin = [p for p in per if p != float("inf")]
                res["xpsnr"]["average"] = (sum(fin) / len(fin)) if fin else float("inf")
        if "vmaf" in keys:
            try:
                with open(os.path.join(tmpdir, "vmaf.json"), encoding="utf-8") as fh:
                    data = json.load(fh)
                per = [f.get("metrics", {}).get("vmaf") for f in data.get("frames", [])]
                pooled = data.get("pooled_metrics", {}).get("vmaf", {})
                avg = pooled.get("mean")
                if avg is None and per:
                    vals = [v for v in per if v is not None]
                    avg = sum(vals) / len(vals) if vals else None
                res["vmaf"] = {"average": avg, "per_frame": per,
                               "min": pooled.get("min"), "max": pooled.get("max"),
                               "model": mname or "vmaf_v0.6.1 (default)"}
            except (OSError, json.JSONDecodeError, ValueError):
                res["vmaf"] = None
        # a run that parsed nothing at all is a failed run, not a result
        if not any(v and v.get("average") is not None or (v and v.get("per_frame"))
                   for v in res.values()):
            return None
        if note:
            for v in res.values():
                if v:
                    v["note"] = note
        res["_passes"] = "single-pass (%d metrics, 2 decodes)" % n
        return res
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compare(distorted, reference, vmaf=True, vmaf_model="auto") -> dict:
    """Quality of ``distorted`` vs ``reference``. VMAF (model selectable:
    auto/default/4k/neg) and XPSNR included when the ffmpeg build has them.

    Result entries may carry ``note`` (e.g. auto-scaling applied) or ``error``
    (why a metric could not be computed) so callers can tell "perfect match"
    from "nothing measured"."""
    dist_info = probe(distorted)
    ref_info = probe(reference)
    combined = _combined_compare(distorted, reference, dist_info, ref_info,
                                 vmaf, vmaf_model)
    if combined is not None:
        out = {
            "distorted": os.path.basename(distorted),
            "reference": os.path.basename(reference),
            "psnr": combined.get("psnr"),
            "ssim": combined.get("ssim"),
            "xpsnr": combined.get("xpsnr") if xpsnr_available() else None,
            "vmaf": (combined.get("vmaf") if vmaf else None),
            "engine": combined.get("_passes"),
            "vmaf_available": vmaf_available(),
            "xpsnr_available": xpsnr_available(),
        }
    else:
        out = {
            "distorted": os.path.basename(distorted),
            "reference": os.path.basename(reference),
            "psnr": _psnr(distorted, reference, dist_info, ref_info),
            "ssim": _ssim(distorted, reference, dist_info, ref_info),
            "xpsnr": _xpsnr(distorted, reference, dist_info, ref_info),
            "vmaf": (_vmaf(distorted, reference, dist_info, ref_info, model=vmaf_model)
                     if vmaf else None),
            "engine": "per-metric (fallback)",
            "vmaf_available": vmaf_available(),
            "xpsnr_available": xpsnr_available(),
        }
    notes = []
    if not dist_info.get("ok"):
        notes.append("distorted file unreadable")
    if not ref_info.get("ok"):
        notes.append("reference file unreadable")
    dd, rd = dist_info.get("duration") or 0, ref_info.get("duration") or 0
    if dd and rd and abs(dd - rd) > 0.5:
        notes.append("durations differ (%.1fs vs %.1fs) - compared the overlap only"
                     % (dd, rd))
    if notes:
        out["notes"] = notes
    return out
