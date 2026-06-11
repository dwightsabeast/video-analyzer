#!/usr/bin/env python3
"""
va_dynhdr - Dolby Vision & HDR10+ dynamic-metadata intelligence.

  * inspect()            - static structure: DV profile/level/layers/cross-compat,
                           HDR10+ presence, MDCV + MaxCLL/MaxFALL, with QC flags.
  * hdr10plus_metadata() - per-scene HDR10+ params (ffprobe side-data, or the
                           optional hdr10plus_tool for full detail).
  * dovi_metadata()      - per-frame DV RPU L1 (min/max/avg nits) + L2 trims via the
                           optional dovi_tool.
  * verify_metadata()    - declared dynamic metadata vs MEASURED nits per scene.
  * dynamic_vs_static()  - libplacebo A/B: dynamic-metadata tonemap vs static HDR10.

Built-in parts use ffprobe only. dovi_tool / hdr10plus_tool are optional binaries
(placed beside the script or on PATH); their features degrade gracefully if absent.
"""

from __future__ import annotations

import os
import json
import base64
import shutil
import tempfile
import subprocess

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from va_ffmpeg import find_ffmpeg, find_tool, has_filter, ffprobe_json, probe
import va_hdr
import va_metrics

DV_PROFILE = {4: "4 - single-layer (legacy)", 5: "5 - single-layer PQ (NOT HDR10-compatible)",
              7: "7 - dual-layer BL+EL (UHD Blu-ray; limited TV playback)",
              8: "8 - single-layer, cross-compatible"}
DV_COMPAT = {0: "none (incompatible base layer)", 1: "HDR10 (profile 8.1)",
             2: "SDR (profile 8.2)", 4: "HLG (profile 8.4)", 6: "HDR10 (8.1)"}
CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _demux_into(exe, path, tool_args, timeout):
    """ffmpeg-demux ``path`` to raw HEVC on stdout, piped into an external tool.

    Deadlock-safe: the parent closes its copy of the pipe so a tool that dies
    early EPIPEs ffmpeg instead of leaving it blocked forever (which also kept
    the source file locked on Windows), and both children get bounded waits
    with kill fallbacks. Returns the tool's returncode (None on timeout)."""
    p1 = subprocess.Popen([exe, "-nostdin", "-i", path, "-c:v", "copy",
                           "-bsf:v", "hevc_mp4toannexb", "-f", "hevc", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          creationflags=CREATIONFLAGS)
    p2 = subprocess.Popen(tool_args, stdin=p1.stdout, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, creationflags=CREATIONFLAGS)
    try:
        p1.stdout.close()  # our copy; p2 holds its own dup
    except OSError:
        pass
    rc = None
    try:
        rc = p2.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p2.kill()
    for p in (p2, p1):
        try:
            p.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            p.kill()
            try:
                p.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
    return rc


def _frame_time(f):
    """Frame timestamp across ffprobe versions (pkt_pts_time died in 5.0)."""
    for k in ("pts_time", "pkt_pts_time", "best_effort_timestamp_time"):
        v = f.get(k)
        if v not in (None, "N/A"):
            return v
    return None


def _side_data(pj):
    blocks = []
    if not isinstance(pj, dict):
        return blocks
    for s in pj.get("streams") or []:
        if isinstance(s, dict) and s.get("codec_type") == "video":
            blocks += s.get("side_data_list") or []
    for f in pj.get("frames") or []:
        if isinstance(f, dict):
            blocks += f.get("side_data_list") or []
    return [b for b in blocks if isinstance(b, dict)]


def parse_hdr_metadata(pj) -> dict:
    """Pure parser over an ffprobe JSON. Returns DV/HDR10+/MDCV/CLL + QC flags."""
    out = {"dolby_vision": None, "hdr10plus": False, "mdcv": False, "cll": None, "flags": []}
    for sd in _side_data(pj):
        t = str(sd.get("side_data_type", "")).lower()
        if "dovi" in t or "dolby vision" in t:
            prof = sd.get("dv_profile")
            comp = sd.get("dv_bl_signal_compatibility_id")
            out["dolby_vision"] = {
                "profile": prof, "level": sd.get("dv_level"),
                "rpu": bool(sd.get("rpu_present_flag")), "bl": bool(sd.get("bl_present_flag")),
                "el": bool(sd.get("el_present_flag")), "compatibility_id": comp,
                "profile_desc": DV_PROFILE.get(prof, str(prof)),
                "compatibility": DV_COMPAT.get(comp, str(comp))}
        elif "2094-40" in t or "hdr10+" in t or "hdr dynamic metadata" in t:
            out["hdr10plus"] = True
        elif "mastering display" in t:
            out["mdcv"] = True
        elif "content light level" in t:
            out["cll"] = {"max_content": sd.get("max_content"), "max_average": sd.get("max_average")}
    dv = out["dolby_vision"]
    if dv:
        if dv["profile"] == 7:
            out["flags"].append(("warn", "DV profile 7 (dual-layer): won't play on most consumer TVs - consider profile 8.1"))
        if dv["profile"] == 5:
            out["flags"].append(("warn", "DV profile 5: no HDR10 fallback - non-DV players may render wrong colours"))
        compat = dv.get("compatibility_id")
        if not out["mdcv"] and not out["cll"] and compat in (1, "1"):
            # only the HDR10-compatible flavour promises MDCV/CLL; HLG (4) and
            # SDR (2) fallbacks don't carry HDR10 static metadata by design
            out["flags"].append(("warn", "Dolby Vision signals HDR10 compatibility "
                                         "but carries no MDCV/CLL static metadata"))
    if out["hdr10plus"] and not out["mdcv"]:
        out["flags"].append(("warn", "HDR10+ present but no mastering-display (MDCV) base metadata"))
    return out


def inspect(path) -> dict:
    return parse_hdr_metadata(ffprobe_json(path, frames=True) or {})


def summary_lines(meta) -> list:
    """Human-readable lines for a report / dialog."""
    lines = []
    dv = meta.get("dolby_vision")
    if dv:
        lines.append("Dolby Vision: profile %s, level %s" % (dv["profile_desc"], dv["level"]))
        lines.append("  layers: %s%s%s   base compatibility: %s" % (
            "BL " if dv["bl"] else "", "EL " if dv["el"] else "", "RPU" if dv["rpu"] else "",
            dv["compatibility"]))
    else:
        lines.append("Dolby Vision: none")
    lines.append("HDR10+: %s" % ("present" if meta.get("hdr10plus") else "none"))
    lines.append("Mastering display (MDCV): %s" % ("present" if meta.get("mdcv") else "none"))
    cll = meta.get("cll")
    if cll and (cll.get("max_content") is not None or cll.get("max_average") is not None):
        lines.append("MaxCLL/MaxFALL: %s / %s nits" % (
            cll.get("max_content", "?"), cll.get("max_average", "?")))
    elif cll:
        lines.append("MaxCLL/MaxFALL: declared, values unknown")
    else:
        lines.append("MaxCLL/MaxFALL: none")
    return lines


def hdr10plus_metadata(path) -> dict:
    """Per-scene HDR10+ params. Uses hdr10plus_tool if present (rich), else parses
    ffprobe frame side-data (limited). Returns {source, available, scenes:[...]}."""
    tool = find_tool("hdr10plus_tool")
    exe = find_ffmpeg()
    if tool and exe:
        tmp = tempfile.mkdtemp(prefix="va_h10p_")
        try:
            out = os.path.join(tmp, "meta.json")
            try:
                r = subprocess.run([tool, "extract", path, "-o", out],
                                   capture_output=True, timeout=300,
                                   creationflags=CREATIONFLAGS)
                direct_ok = r.returncode == 0
            except subprocess.TimeoutExpired:
                direct_ok = False
            if not direct_ok or not os.path.isfile(out):
                _demux_into(exe, path, [tool, "extract", "-", "-o", out], 300)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            scenes = data.get("SceneInfo") or []
            series = []
            for sc in scenes:
                lp = sc.get("LuminanceParameters", {})
                mx = lp.get("MaxScl") or [0]
                series.append({"frame": sc.get("SceneFrameIndex"),
                               "avg_maxrgb": lp.get("AverageRGB"),
                               "maxscl": max(mx) if mx else None,
                               "target_nits": sc.get("TargetedSystemDisplayMaximumLuminance")})
            return {"source": "hdr10plus_tool", "available": bool(series), "scenes": series}
        except Exception:
            pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    pj = ffprobe_json(path, frames=True) or {}
    frames = []
    for f in pj.get("frames", []):
        for sd in f.get("side_data_list", []) or []:
            if "2094-40" in str(sd.get("side_data_type", "")).lower():
                frames.append({"t": _frame_time(f),
                               **{k: v for k, v in sd.items() if k != "side_data_type"}})
    return {"source": "ffprobe" if frames else "none", "available": bool(frames), "scenes": frames}


def dovi_metadata(path) -> dict:
    """Per-frame Dolby Vision RPU L1 (min/max/avg nits) via the optional dovi_tool.
    Best-effort: dovi_tool's export JSON shape varies by version; verify on your box."""
    tool = find_tool("dovi_tool")
    exe = find_ffmpeg()
    if not tool or not exe:
        return {"source": "none", "available": False, "frames": []}
    tmp = tempfile.mkdtemp(prefix="va_dovi_")
    try:
        rpu = os.path.join(tmp, "rpu.bin")
        js = os.path.join(tmp, "rpu.json")
        _demux_into(exe, path, [tool, "extract-rpu", "-", "-o", rpu], 600)
        if not os.path.isfile(rpu):
            return {"source": "none", "available": False, "frames": []}
        subprocess.run([tool, "export", "-i", rpu, "-o", js], capture_output=True,
                       timeout=300, creationflags=CREATIONFLAGS)
        with open(js, encoding="utf-8") as fh:
            data = json.load(fh)
        rows = []
        for i, fr in enumerate(data if isinstance(data, list) else data.get("frames", [])):
            dm = fr.get("vdr_dm_data", fr)
            l1 = None
            for blk in (dm.get("ext_metadata_blocks", []) or []):
                if blk.get("level") == 1 or "Level1" in str(blk):
                    l1 = blk
            src = l1 or dm
            mn, mx, av = src.get("min_pq"), src.get("max_pq"), src.get("avg_pq")
            def nits(pq):
                return None if pq is None else round(float(va_hdr.pq_eotf(np.array(pq / 4095.0))))
            rows.append({"frame": i, "min_nits": nits(mn), "max_nits": nits(mx), "avg_nits": nits(av)})
        return {"source": "dovi_tool", "available": bool(rows), "frames": rows}
    except Exception:
        return {"source": "none", "available": False, "frames": []}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify_metadata(path, scenes_limit=30) -> dict:
    """Compare declared dynamic metadata (or static CLL) against MEASURED nits per scene."""
    info = probe(path)
    fps = info["fps"] or 25.0
    cuts = va_metrics.scene_cuts(path)
    bounds = [0] + [int(c * fps) for c in cuts] + [info.get("nb_frames") or 0]
    bounds = sorted(set(b for b in bounds if b >= 0))
    dv = dovi_metadata(path)
    h10 = hdr10plus_metadata(path)
    rows = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 1:
            continue
        mid = (a + b) // 2
        nm = va_hdr.nits_map(path, mid, cap=480)
        if nm is None:
            continue
        mmax, mavg = round(float(nm.max())), round(float(nm.mean()))
        dmax = None
        if dv["available"]:
            seg = [f["max_nits"] for f in dv["frames"] if a <= f["frame"] < b and f["max_nits"]]
            dmax = max(seg) if seg else None
        elif h10["available"]:
            def _sframe(s):
                if s.get("frame") is not None:
                    return s["frame"]
                try:
                    return int(float(s.get("t") or 0) * fps)
                except (TypeError, ValueError):
                    return 0
            seg = [s.get("maxscl") for s in h10["scenes"] if a <= _sframe(s) < b and s.get("maxscl")]
            dmax = max(seg) if seg else None
        status = "ok"
        if dmax is not None and mmax > dmax * 1.15 + 50:
            status = "content peak %d exceeds declared %d" % (mmax, dmax)
        rows.append({"scene": i, "t": round(mid / fps, 2), "measured_max": mmax,
                     "measured_avg": mavg, "declared_max": dmax, "status": status})
        if len(rows) >= scenes_limit:
            break
    return {"has_dynamic": dv["available"] or h10["available"],
            "source": dv["source"] if dv["available"] else h10["source"],
            "scenes": rows, "static": va_hdr.maxcll_maxfall(path)}


def dynamic_vs_static(path, idx=None, target=1000):
    """libplacebo A/B: dynamic-metadata tonemap vs static HDR10, side by side (RGB)."""
    if not has_filter("libplacebo"):
        return None, "libplacebo not available (needs the full ffmpeg build)"
    exe = find_ffmpeg()
    info = probe(path)
    if idx is None:
        idx = max(0, (info.get("nb_frames") or 2) // 2)
    ss = idx / (info["fps"] or 25.0)
    w, h = info["width"], info["height"]
    s = 640 / max(w, h) if max(w, h) > 640 else 1.0
    ow, oh = max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)

    def render(dyn):
        vf = ("libplacebo=w=%d:h=%d:target_peak=%d:apply_dolbyvision=%d:colorspace=bt709:"
              "color_primaries=bt709:color_trc=bt709:tonemapping=bt.2390,format=bgr24"
              % (ow, oh, target, 1 if dyn else 0))
        args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-init_hw_device", "vulkan=v",
                "-filter_hw_device", "v", "-ss", "%.3f" % ss, "-i", path, "-vf", vf,
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        try:
            out = subprocess.run(args, capture_output=True, timeout=90,
                                 creationflags=CREATIONFLAGS).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        if len(out) < ow * oh * 3:
            return None
        return np.frombuffer(out[:ow * oh * 3], np.uint8).reshape(oh, ow, 3)

    dyn = render(True)
    sta = render(False)
    if dyn is None or sta is None:
        return None, "libplacebo render failed (needs a real HDR/DV file on a GPU build)"

    def lab(img, txt):
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 18), (0, 0, 0), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    lab(dyn, "dynamic metadata (%d nits)" % target)
    lab(sta, "static HDR10 (%d nits)" % target)
    return cv2.cvtColor(cv2.hconcat([dyn, sta]), cv2.COLOR_BGR2RGB), "ok"


def render_dynamic_timeline(verify_result, w=720, h=180):
    """Chart of measured vs declared peak nits per scene."""
    w, h = max(200, int(w)), max(100, int(h))
    img = np.full((h, w, 3), 24, np.uint8)
    rows = verify_result.get("scenes", [])
    if not rows:
        cv2.putText(img, "no scene data", (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return img
    mx = max([r["measured_max"] for r in rows] + [r["declared_max"] or 0 for r in rows] + [100])
    n = len(rows)
    for series, color in (("measured_max", (78, 201, 176)), ("declared_max", (240, 192, 64))):
        pts = []
        for i, r in enumerate(rows):
            v = r.get(series)
            if v is None:
                continue
            x = int(8 + i / max(1, n - 1) * (w - 16))
            y = int(h - 20 - (v / mx) * (h - 36))
            pts.append((x, y))
        for j in range(1, len(pts)):
            cv2.line(img, pts[j - 1], pts[j], color, 1, cv2.LINE_AA)
    cv2.putText(img, "peak nits/scene: measured=teal declared=gold (max %d)" % round(mx),
                (8, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
    return img


# --- External-tool reports & plots (dovi_tool / hdr10plus_tool / mediainfo / mp4dump)

def _extract_rpu(path, tmp) -> "str | None":
    """Demux HEVC via ffmpeg and pull the DV RPU with dovi_tool. Returns path or None."""
    tool = find_tool("dovi_tool")
    exe = find_ffmpeg()
    if not tool or not exe:
        return None
    rpu = os.path.join(tmp, "rpu.bin")
    _demux_into(exe, path, [tool, "extract-rpu", "-", "-o", rpu], 600)
    return rpu if os.path.isfile(rpu) and os.path.getsize(rpu) else None


def _png_rgb(png_path):
    if cv2 is None or not os.path.isfile(png_path):
        return None
    bgr = cv2.imread(png_path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_dovi_l1(path):
    """dovi_tool's L1 brightness plot for the file. Returns (rgb_image|None, message)."""
    tool = find_tool("dovi_tool")
    if not tool:
        return None, "dovi_tool not installed (Tools dialog)"
    if not find_ffmpeg():
        return None, "ffmpeg is required to demux the HEVC stream"
    tmp = tempfile.mkdtemp(prefix="va_dviplot_")
    try:
        rpu = _extract_rpu(path, tmp)
        if not rpu:
            return None, "no Dolby Vision RPU found in this file"
        png = os.path.join(tmp, "l1_plot.png")
        try:
            r = subprocess.run([tool, "plot", rpu, "-t", os.path.basename(path), "-o", png],
                               capture_output=True, timeout=300, creationflags=CREATIONFLAGS)
        except (OSError, subprocess.SubprocessError) as e:
            return None, "dovi_tool plot failed: %s" % e
        img = _png_rgb(png)
        if img is None:
            err = (r.stderr or b"").decode("utf-8", "replace").strip()
            return None, "dovi_tool plot failed: %s" % (err[:160] or "no output")
        return img, "ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def plot_hdr10plus(path):
    """hdr10plus_tool's brightness plot for the file. Returns (rgb_image|None, message)."""
    tool = find_tool("hdr10plus_tool")
    if not tool:
        return None, "hdr10plus_tool not installed (Tools dialog)"
    exe = find_ffmpeg()
    if not exe:
        return None, "ffmpeg is required to demux the HEVC stream"
    tmp = tempfile.mkdtemp(prefix="va_h10plot_")
    try:
        meta = os.path.join(tmp, "meta.json")
        try:
            r = subprocess.run([tool, "extract", path, "-o", meta],
                               capture_output=True, timeout=300,
                               creationflags=CREATIONFLAGS)
            direct_ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            direct_ok = False
        if not direct_ok or not os.path.isfile(meta):
            _demux_into(exe, path, [tool, "extract", "-", "-o", meta], 300)
        if not os.path.isfile(meta) or not os.path.getsize(meta):
            return None, "no HDR10+ metadata found in this file"
        png = os.path.join(tmp, "h10p_plot.png")
        try:
            r = subprocess.run([tool, "plot", meta, "-t", os.path.basename(path), "-o", png],
                               capture_output=True, timeout=300, creationflags=CREATIONFLAGS)
        except (OSError, subprocess.SubprocessError) as e:
            return None, "hdr10plus_tool plot failed: %s" % e
        img = _png_rgb(png)
        if img is None:
            err = (r.stderr or b"").decode("utf-8", "replace").strip()
            return None, "hdr10plus_tool plot failed: %s" % (err[:160] or "no output")
        return img, "ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mediainfo_report(path):
    """Full MediaInfo text report. Returns (text|None, message)."""
    tool = find_tool("mediainfo")
    if not tool:
        return None, "mediainfo not installed (Tools dialog)"
    try:
        r = subprocess.run([tool, path], capture_output=True, timeout=120,
                           creationflags=CREATIONFLAGS)
    except (OSError, subprocess.SubprocessError) as e:
        return None, "mediainfo failed: %s" % e
    text = (r.stdout or b"").decode("utf-8", "replace").strip()
    if not text:
        return None, "mediainfo produced no output"
    return text, "ok"


def mp4_report(path):
    """Bento4 mp4dump box tree, with a DV/HDR signaling summary up top.
    Returns (text|None, message). MP4/MOV family only."""
    tool = find_tool("mp4dump")
    if not tool:
        return None, "mp4dump not installed (Tools dialog)"
    if os.path.splitext(path)[1].lower() not in (".mp4", ".m4v", ".mov", ".m4a", ".isma", ".ismv"):
        return None, "mp4dump reads MP4/MOV containers only (this is %s)" % \
            (os.path.splitext(path)[1] or "unknown")
    try:
        r = subprocess.run([tool, "--verbosity", "1", path], capture_output=True,
                           timeout=120, creationflags=CREATIONFLAGS)
    except (OSError, subprocess.SubprocessError) as e:
        return None, "mp4dump failed: %s" % e
    text = (r.stdout or b"").decode("utf-8", "replace")
    if not text.strip():
        return None, (r.stderr or b"").decode("utf-8", "replace").strip()[:160] or "no output"
    keys = ("dvcc", "dvvc", "dvwc", "hvcc", "hvce", "hev1", "hvc1", "dvh1", "dvhe", "clli", "mdcv")
    hits = [ln.rstrip() for ln in text.splitlines() if any(k in ln.lower() for k in keys)]
    head = ["DV / HDR signaling boxes found:"] + (["  " + h.strip() for h in hits] or
                                                  ["  (none - no dvcC/dvvC/hvcC markers)"])
    return "\n".join(head) + "\n\n" + "-" * 60 + "\n\n" + text, "ok"


# --- Matroska / WebM (MKVToolNix: mkvmerge -J + mkvinfo) -----------------------

_MKV_EXTS = (".mkv", ".mka", ".mks", ".mk3d", ".webm")
# Matroska BlockAdditionMapping types carrying Dolby Vision configuration
# (the MKV equivalent of the MP4 dvcC/dvvC boxes; what players key DV off).
_DV_BLOCK_IDS = {0x64766343: "dvcC", 0x64766643: "dvvC", 0x68766345: "hvcE"}


def _dv_config_bits(data) -> "dict | None":
    """Decode a dvcC/dvvC payload (Dolby Vision ISOBMFF configuration record)."""
    if not data or len(data) < 5:
        return None
    return {"profile": data[2] >> 1,
            "level": ((data[2] & 1) << 5) | (data[3] >> 3),
            "rpu": bool((data[3] >> 2) & 1), "el": bool((data[3] >> 1) & 1),
            "bl": bool(data[3] & 1), "compatibility_id": data[4] >> 4}


def _mkv_summary(j) -> list:
    """Readable lines from a ``mkvmerge -J`` identification payload, with the
    Dolby Vision block-addition mappings decoded. Pure (testable offline)."""
    lines = []
    if not isinstance(j, dict):
        return lines
    cont = j.get("container") or {}
    cprops = cont.get("properties") or {}
    dur = cprops.get("duration")
    lines.append("Container : %s%s" % (
        cont.get("type", "?"),
        "  (%.1f s)" % (dur / 1e9) if isinstance(dur, (int, float)) else ""))
    dv_seen = False
    for t in j.get("tracks") or []:
        if not isinstance(t, dict):
            continue
        p = t.get("properties") or {}
        bits = [str(t.get("id", "?")), str(t.get("type", "?")),
                str(t.get("codec", p.get("codec_id", "?")))]
        if p.get("pixel_dimensions"):
            bits.append(p["pixel_dimensions"])
        if p.get("language") and p["language"] != "und":
            bits.append(p["language"])
        if p.get("default_track"):
            bits.append("default")
        lines.append("Track #%s" % "  ".join(bits))
        for bam in p.get("block_addition_mappings") or []:
            if not isinstance(bam, dict):
                continue
            id_type = bam.get("id_type")
            name = _DV_BLOCK_IDS.get(id_type)
            if not name:
                lines.append("  block-addition mapping: id_type %s" % id_type)
                continue
            dv_seen = True
            cfg = None
            if name in ("dvcC", "dvvC"):
                try:
                    cfg = _dv_config_bits(base64.b64decode(bam.get("id_extra_data") or ""))
                except (ValueError, TypeError):
                    cfg = None
            if cfg:
                layers = "".join(s for s, on in (("BL+", cfg["bl"]), ("EL+", cfg["el"]),
                                                 ("RPU", cfg["rpu"])) if on).rstrip("+")
                lines.append("  %s: Dolby Vision profile %d.%d, level %d  [%s]  base: %s"
                             % (name, cfg["profile"], cfg["compatibility_id"], cfg["level"],
                                layers or "?", DV_COMPAT.get(cfg["compatibility_id"],
                                                             str(cfg["compatibility_id"]))))
            else:
                lines.append("  %s: Dolby Vision configuration present" % name)
    if not dv_seen:
        lines.append("(no Dolby Vision block-addition mapping signaled - DV-in-MKV"
                     " players will not engage DV for this file)")
    n_att = len(j.get("attachments") or [])
    n_chp = len(j.get("chapters") or [])
    if n_att or n_chp:
        lines.append("Attachments: %d   Chapter editions: %d" % (n_att, n_chp))
    return lines


def mkv_report(path):
    """MKVToolNix container report: mkvmerge -J summary (with Dolby Vision
    block-addition mappings decoded - the MKV signaling mp4dump cannot see)
    plus the mkvinfo element tree. Returns (text|None, message)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MKV_EXTS:
        return None, "mkvinfo reads Matroska/WebM containers only (this is %s)" % (ext or "unknown")
    mkvmerge = find_tool("mkvmerge")
    mkvinfo = find_tool("mkvinfo")
    if not mkvmerge and not mkvinfo:
        return None, "MKVToolNix not installed (Tools dialog)"
    parts = []
    if mkvmerge:
        try:
            r = subprocess.run([mkvmerge, "-J", path], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120,
                               creationflags=CREATIONFLAGS)
            j = json.loads(r.stdout or "null")
        except (OSError, subprocess.SubprocessError):
            j = None
        except ValueError:
            j = None
        if isinstance(j, dict):
            parts.append("\n".join(["DV / HDR signaling (mkvmerge -J):"] +
                                   ["  " + ln for ln in _mkv_summary(j)]))
    if mkvinfo:
        try:
            r = subprocess.run([mkvinfo, path], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120,
                               creationflags=CREATIONFLAGS)
            tree = (r.stdout or "").splitlines()
            if len(tree) > 500:
                tree = tree[:500] + ["... (%d more lines truncated)" % (len(tree) - 500)]
            if tree:
                parts.append("\n".join(["Element tree (mkvinfo):"] + tree))
        except (OSError, subprocess.SubprocessError) as e:
            parts.append("mkvinfo failed: %s" % e)
    if not parts:
        return None, "MKVToolNix produced no output for this file"
    return ("\n\n" + "-" * 60 + "\n\n").join(parts), "ok"
