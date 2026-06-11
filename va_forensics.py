#!/usr/bin/env python3
"""
va_forensics - provenance & integrity signals, new territory for a QC tool.

Bitstream / timing:
  * recompression() - 8x8 blockiness + DCT double-quantization periodicity, a
                      relative indicator of how heavily / how many times a file has
                      been (re)compressed - useful for spotting laundered "masters".
  * splice_points() - legacy single-signal splice list (coding-cost jumps only).
  * splice_scan()   - corroborated splice detection: coding-cost jumps, off-cadence
                      keyframes, PTS discontinuities and audio discontinuities are
                      merged per timestamp and scored - agreement across independent
                      signals is what makes a candidate suspicious.

Container / metadata:
  * container_report()     - pure-python MP4 box walk (edit lists, free-space gaps,
                             multiple mdat, moov/mdat order, 1904-epoch times) or
                             MKV writing/muxing apps via mkvmerge; edit-history tells.
  * encoder_fingerprint()  - x264/x265 SEI settings strings, Lavf/Lavc and other
                             writing-library markers byte-scanned from the stream,
                             cross-checked against container tags.
  * metadata_consistency() - creation-time / handler / duration coherence audit.

Pixel level:
  * ela_map()   - error-level-analysis residual for one frame (recompress + diff).
  * noise_map() - per-block sensor-noise deviation map for one frame.
  * noise_consistency() - noise-floor drift across sampled timeline segments.

Temporal:
  * frame_loop_scan() - perceptual-hash search for repeated frame SEQUENCES
                        (looped/recycled footage), distinct from freeze detection.

Audio:
  * enf_trace()  - mains-hum (50/60 Hz ENF) presence + continuity trace.
  * render_enf() - plot of the ENF track for the GUI.

Provenance / reporting:
  * content_credentials() / content_credentials_report() - C2PA scan + c2patool
                      validation (now with ingredient-chain rendering).
  * c2pa_summary() - compact machine-readable C2PA status for reports.
  * sha256_file()  - chain-of-custody hash.
  * forensics_report() / render_report() / report_marks() - run everything,
                      merge findings, render text, surface timeline markers.

Indicators, not proof. Pure ffmpeg + cv2/numpy (+ stdlib; c2patool optional).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from va_ffmpeg import VideoSource, find_tool, find_ffmpeg, ffprobe_json, probe
import va_metrics
import va_audio

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _blockiness(luma, block=8):
    """Boundary-vs-interior gradient ratio, or None when the frame has too
    little texture for the ratio to mean anything (flat/synthetic frames)."""
    h, w = luma.shape
    cols = np.arange(block, w, block)
    icols = np.array([c for c in range(1, w) if c % block != 0])
    rows = np.arange(block, h, block)
    irows = np.array([r for r in range(1, h) if r % block != 0])
    if len(cols) == 0 or len(rows) == 0 or len(icols) == 0 or len(irows) == 0:
        return None
    bnd = (np.mean(np.abs(luma[:, cols] - luma[:, cols - 1])) +
           np.mean(np.abs(luma[rows, :] - luma[rows - 1, :]))) / 2.0
    interior = (np.mean(np.abs(luma[:, icols] - luma[:, icols - 1])) +
                np.mean(np.abs(luma[irows, :] - luma[irows - 1, :]))) / 2.0
    if interior < 0.05:          # essentially no interior detail
        return None
    return float(bnd / (interior + 1e-6) - 1.0)


def _dct_periodicity(luma, block=8):
    """Comb-like structure in a mid-frequency DCT coefficient histogram = a tell of
    double quantization (re-encoding)."""
    h, w = luma.shape
    h -= h % block
    w -= w % block
    if h < block or w < block:
        return 0.0
    coeffs = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            d = cv2.dct(luma[y:y + block, x:x + block])
            coeffs.append(d[2, 2])
    coeffs = np.array(coeffs)
    if coeffs.size < 16:
        return 0.0
    hist, _ = np.histogram(coeffs, bins=64, range=(-40, 40))
    hist = hist - hist.mean()
    spec = np.abs(np.fft.rfft(hist))
    if spec.size < 3 or spec[1:].sum() == 0:
        return 0.0
    return float(spec[2:].max() / (spec[1:].sum() + 1e-6))   # 0..1 periodic-energy fraction


def recompression(path, samples=5) -> dict:
    # storage geometry: SAR/rotation resampling would smear the 8x8 DCT grid
    # this analysis depends on
    vs = VideoSource(path, display_ar=False)
    try:
        info = vs.info
        if not info.get("ok"):
            return {"blockiness": None, "dct_double_quant": None,
                    "note": "file unreadable — no analysis performed"}
        n = info.get("nb_frames") or 0
        idxs = [max(0, int(n * (i + 1) / (samples + 1))) for i in range(samples)] if n else list(range(samples))
        blocks, dqs, decoded = [], [], 0
        for i in idxs:
            fr = vs.frame_at(i)
            if fr is None:
                continue
            decoded += 1
            luma = (0.2126 * fr[:, :, 2] + 0.7152 * fr[:, :, 1] + 0.0722 * fr[:, :, 0]).astype(np.float32)
            b = _blockiness(luma)
            if b is not None:
                blocks.append(b)
            dqs.append(_dct_periodicity(luma))
    finally:
        vs.close()
    if not decoded:
        return {"blockiness": None, "dct_double_quant": None,
                "note": "file unreadable — no analysis performed"}
    dq = round(float(np.mean(dqs)) if dqs else 0.0, 4)
    if not blocks:
        return {"blockiness": 0.0, "dct_double_quant": dq,
                "note": "insufficient texture for blockiness measurement; "
                        "double-quantization structure %.2f - relative indicator "
                        "(compare across files; treat as a hint, not proof)." % dq}
    block = round(float(np.mean(blocks)), 4)
    note = ("double-quantization structure %.2f, blockiness index %.2f - relative "
            "indicators (compare across files; H.264 in-loop deblocking masks part "
            "of the signal, so treat as a hint, not proof)." % (dq, block))
    return {"blockiness": block, "dct_double_quant": dq, "note": note}


def splice_points(path, z=4.0) -> dict:
    """Frames where coding cost jumps abnormally (possible splice / inserted clip)."""
    fs = va_metrics.frame_sizes(path)
    sizes = np.array(fs.get("size", []), dtype=float)
    times = fs.get("t", [])
    if sizes.size < 8:
        return {"candidates": [], "note": "too short"}
    d = np.abs(np.diff(sizes))
    thr = d.mean() + z * d.std()
    cuts = va_metrics.scene_cuts(path)
    cand = []
    for i in np.where(d > thr)[0]:
        t = times[i + 1] if i + 1 < len(times) else float(i + 1)
        at_cut = any(abs(t - c) < 0.2 for c in cuts)
        cand.append({"t": round(float(t), 2), "at_scene_cut": at_cut})
    return {"candidates": cand[:50], "scene_cuts": len(cuts),
            "note": "coding-cost discontinuities; those not at a scene cut are more suspicious"}


def content_credentials(path, scan_bytes=4_000_000) -> dict:
    """Detect embedded C2PA / JUMBF provenance manifests (presence only).
    Scans the first AND last ``scan_bytes`` (C2PA boxes usually trail mdat)."""
    found = {}
    try:
        with open(path, "rb") as fh:
            data = fh.read(scan_bytes)
            fh.seek(0, 2)
            size = fh.tell()
            if size > scan_bytes:                     # tail scan
                fh.seek(max(scan_bytes, size - scan_bytes))
                data += fh.read(scan_bytes)
    except OSError:
        return {"c2pa": False, "jumbf": False, "xmp": False, "present": False,
                "note": "file unreadable — no analysis performed"}
    found["c2pa"] = (b"c2pa" in data) or (b"urn:c2pa" in data)
    found["jumbf"] = (b"jumb" in data) or (b"jumd" in data)
    found["xmp"] = b"http://ns.adobe.com/xap/" in data
    found["present"] = any(found.values())
    return found


_C2PA_FAIL_HINTS = ("mismatch", "untrusted", "revoked", "expired", "missing",
                    "invalid", "error", "failure")


def _c2pa_state(data) -> str:
    """Validation outcome of a c2patool manifest store: 'VALID', 'INVALID' or
    'present (validation state not reported)'. Newer c2patool emits a top-level
    ``validation_state``; older builds list ``validation_status`` entries whose
    codes (e.g. signingCredential.untrusted) flag failures."""
    state = data.get("validation_state")
    if isinstance(state, str) and state:
        low = state.lower()
        if low in ("valid", "trusted"):
            return "VALID"
        if low == "invalid":
            return "INVALID"
        return "present (validation state not reported)"
    codes = [str(e.get("code", "")).lower()
             for e in (data.get("validation_status") or []) if isinstance(e, dict)]
    codes = [c for c in codes if c]
    if codes:
        bad = any(h in c for c in codes for h in _C2PA_FAIL_HINTS)
        return "INVALID" if bad else "VALID"
    return "present (validation state not reported)"


def _c2pa_report_text(data, max_lines=300) -> str:
    """Readable report: validation header, active-manifest summary, pretty JSON."""
    lines = ["Content Credentials (C2PA) - c2patool manifest store",
             "Validation: %s" % _c2pa_state(data)]
    manifests = data.get("manifests")
    active = manifests.get(data.get("active_manifest")) if isinstance(manifests, dict) else None
    if isinstance(active, dict):
        if active.get("title"):
            lines.append("Active manifest: %s" % active["title"])
        if active.get("claim_generator"):
            lines.append("Generator: %s" % active["claim_generator"])
        sig = active.get("signature_info")
        if isinstance(sig, dict) and (sig.get("issuer") or sig.get("time")):
            lines.append("Signed by: %s%s" % (sig.get("issuer") or "unknown issuer",
                                              "  (%s)" % sig["time"] if sig.get("time") else ""))
    chain = _c2pa_ingredients(data)
    if chain:
        lines.append("Ingredients (provenance chain):")
        lines.extend(chain)
    body = json.dumps(data, indent=2, ensure_ascii=False).splitlines()
    if len(body) > max_lines:
        extra = len(body) - max_lines
        body = body[:max_lines] + ["... (%d more JSON lines truncated)" % extra]
    return "\n".join(lines) + "\n\n" + "\n".join(body)


def content_credentials_report(path, timeout=120):
    """Content Credentials (C2PA) report. With c2patool installed: run it, parse the
    manifest store and report the validation outcome; without it: fall back to the
    content_credentials() byte scan (presence only). Returns (text|None, message)."""
    tool = find_tool("c2patool")
    if not tool:
        scan = content_credentials(path)
        if scan.get("note"):                                   # unreadable file
            return None, scan["note"]
        if not scan.get("present"):
            return None, ("no C2PA markers found (presence scan; install c2patool "
                          "for validation - Tools dialog)")
        names = {"c2pa": "C2PA box / urn:c2pa identifier ('c2pa')",
                 "jumbf": "JUMBF superbox ('jumb'/'jumd')",
                 "xmp": "XMP packet (Adobe namespace)"}
        text = ("Provenance markers found (byte scan of file head/tail):\n" +
                "\n".join("  * " + names[k] for k in ("c2pa", "jumbf", "xmp") if scan.get(k)) +
                "\n\nPresence only - the manifest was not read or validated.")
        return text, "ok (presence scan only - install c2patool for validation, Tools dialog)"
    try:
        r = subprocess.run([tool, path], capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace",
                           creationflags=CREATIONFLAGS)
    except subprocess.TimeoutExpired:
        return None, "c2patool timed out after %ds" % timeout
    except (OSError, subprocess.SubprocessError) as e:
        return None, "c2patool failed: %s" % str(e)[:160]
    out = (r.stdout or "").strip()
    if out:
        try:
            data = json.loads(out)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return _c2pa_report_text(data), "ok"
    blob = ((r.stderr or "") + "\n" + (r.stdout or "")).lower()
    if "no claim" in blob or "no jumbf" in blob:
        return None, "no Content Credentials (C2PA) found in this file"
    err_lines = (r.stderr or "").strip().splitlines()
    first = err_lines[0].strip() if err_lines else (
        out.splitlines()[0].strip() if out else "exit code %s, no output" % r.returncode)
    return None, "c2patool failed: %s" % first[:160]


# === C2PA ingredient chain ====================================================

def _c2pa_ingredients(data, _depth=0, _manifest=None, _seen=None) -> list:
    """Indented text lines describing the active manifest's ingredient chain
    (each ingredient = an earlier asset this one was derived from). Recurses
    into ingredient manifests present in the same store; cycle-guarded."""
    if not isinstance(data, dict) or _depth > 6:
        return []
    manifests = data.get("manifests")
    if not isinstance(manifests, dict):
        return []
    _seen = _seen if _seen is not None else set()
    label = _manifest if _manifest is not None else data.get("active_manifest")
    man = manifests.get(label) if label else None
    if not isinstance(man, dict) or label in _seen:
        return []
    _seen.add(label)
    out = []
    for ing in (man.get("ingredients") or []):
        if not isinstance(ing, dict):
            continue
        bits = [ing.get("title") or "(untitled ingredient)"]
        if ing.get("relationship"):
            bits.append(ing["relationship"])
        if ing.get("format"):
            bits.append(ing["format"])
        out.append("  " * (_depth + 1) + "* " + "  ".join(str(b) for b in bits))
        ref = ing.get("active_manifest") or (ing.get("c2pa_manifest") or {}).get("url") \
            if isinstance(ing.get("c2pa_manifest"), dict) else ing.get("active_manifest")
        if isinstance(ref, str):
            ref = ref.split("/")[-1]
            out.extend(_c2pa_ingredients(data, _depth + 1, ref, _seen))
    return out


def c2pa_summary(path, timeout=120) -> dict:
    """Compact C2PA status for machine reports: {present, validation, generator,
    issuer, ingredients, source}. Uses c2patool when installed, else byte scan."""
    tool = find_tool("c2patool")
    if not tool:
        scan = content_credentials(path)
        return {"present": bool(scan.get("present")), "validation": None,
                "generator": None, "issuer": None, "ingredients": 0,
                "source": "byte scan (install c2patool to validate)"}
    try:
        r = subprocess.run([tool, path], capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace",
                           creationflags=CREATIONFLAGS)
    except (OSError, subprocess.SubprocessError):
        return {"present": False, "validation": None, "generator": None,
                "issuer": None, "ingredients": 0, "source": "c2patool failed"}
    try:
        data = json.loads((r.stdout or "").strip() or "null")
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"present": False, "validation": None, "generator": None,
                "issuer": None, "ingredients": 0, "source": "c2patool"}
    manifests = data.get("manifests")
    active = manifests.get(data.get("active_manifest")) if isinstance(manifests, dict) else None
    gen = iss = None
    if isinstance(active, dict):
        gen = active.get("claim_generator")
        sig = active.get("signature_info")
        iss = sig.get("issuer") if isinstance(sig, dict) else None
    return {"present": True, "validation": _c2pa_state(data), "generator": gen,
            "issuer": iss, "ingredients": len(_c2pa_ingredients(data)),
            "source": "c2patool"}


# === Chain of custody =========================================================

def sha256_file(path, chunk=1 << 22) -> "str | None":
    """SHA-256 of the file (chunked; None when unreadable)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


# === Audio discontinuity features =============================================

_HANN = None


def _audio_features(path, sr=8000, win=512, hop=256, cancel=None) -> "dict | None":
    """Windowed RMS (dB) + spectral flux from the first audio stream, streamed
    through an ffmpeg f32le pipe (memory stays flat on long files).
    Returns {t, rms_db, flux} or None when there is no usable audio."""
    global _HANN
    exe = find_ffmpeg()
    if not exe:
        return None
    if _HANN is None or len(_HANN) != win:
        _HANN = np.hanning(win).astype(np.float32)
    args = [exe, "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0",
            "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=CREATIONFLAGS)
    except OSError:
        return None
    va_metrics._track(proc)
    rms_db, flux = [], []
    prev_mag = None
    buf = b""
    try:
        while True:
            if cancel is not None and cancel():
                break
            chunkb = proc.stdout.read(1 << 16)
            if not chunkb:
                break
            buf += chunkb
            while len(buf) >= win * 4:
                x = np.frombuffer(buf[:win * 4], np.float32)
                buf = buf[hop * 4:]
                if not np.all(np.isfinite(x)):
                    x = np.nan_to_num(x)
                rms = float(np.sqrt(np.mean(x * x)))
                rms_db.append(20.0 * np.log10(rms + 1e-7))
                mag = np.abs(np.fft.rfft(x * _HANN))
                if prev_mag is None:
                    flux.append(0.0)
                else:
                    flux.append(float(np.sqrt(np.sum((mag - prev_mag) ** 2)) /
                                      (np.sum(prev_mag) + 1e-9)))
                prev_mag = mag
    finally:
        try:
            proc.stdout.close()
            proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
        va_metrics._untrack(proc)
    if len(rms_db) < 8:
        return None
    return {"t": np.arange(len(rms_db)) * (hop / float(sr)),
            "rms_db": np.asarray(rms_db), "flux": np.asarray(flux)}


def _audio_jumps(path, cancel=None) -> list:
    """Times (s) of abrupt audio discontinuities: spectral-flux spikes (robust
    z-score) or hard RMS steps. A corroborating signal - musical transients can
    fire it, so it is never used alone."""
    feats = _audio_features(path, cancel=cancel)
    if feats is None:
        return []
    fx = feats["flux"]
    med = float(np.median(fx))
    mad = float(np.median(np.abs(fx - med))) * 1.4826 + 1e-9
    zhit = np.where((fx - med) / mad > 9.0)[0]
    rstep = np.abs(np.diff(feats["rms_db"]))
    rhit = np.where(rstep > 16.0)[0] + 1
    times = sorted(set(np.concatenate([zhit, rhit]).tolist()))
    out = []
    for i in times:
        t = float(feats["t"][min(i, len(feats["t"]) - 1)])
        if not out or t - out[-1] > 0.3:
            out.append(round(t, 3))
        if len(out) >= 200:
            break
    return out


# === Corroborated splice detection ===========================================

def splice_scan(path, z=4.0, cancel=None) -> dict:
    """Multi-signal splice detection. Independent signals are merged per
    timestamp and scored:

        size_jump  (+1)  abnormal per-frame coding-cost discontinuity
        pts_gap    (+2)  presentation-timestamp gap / reversal mid-stream
        idr_offcadence (+1) forced keyframe breaking the file's GOP rhythm
        audio_jump (+1)  audio spectral/level discontinuity at the same time
        at_scene_cut (-1) coincides with a detected content cut (normal editing)

    Returns {candidates: [{t, score, signals, at_scene_cut}], stats, note}.
    Candidates are sorted by score (strongest corroboration first). Indicators,
    not proof."""
    fs = va_metrics.frame_sizes(path, cancel=cancel)
    sizes = np.asarray(fs.get("size", []), dtype=float)
    times = np.asarray(fs.get("t", []), dtype=float)
    keys = list(fs.get("key", []))
    if not keys and fs.get("type"):
        keys = [1 if tp == "I" else 0 for tp in fs["type"]]
    n = sizes.size
    if n < 16:
        return {"candidates": [], "stats": {"frames": int(n)},
                "note": "too short for splice analysis"}

    events = {}   # rounded-time bucket -> {"t", "signals", set}

    def hit(t, signal):
        key = round(float(t) / 0.25)
        e = events.setdefault(key, {"t": float(t), "signals": set()})
        e["signals"].add(signal)

    # 1. coding-cost discontinuities
    d = np.abs(np.diff(sizes))
    thr = d.mean() + z * d.std()
    for i in np.where(d > thr)[0]:
        hit(times[i + 1] if i + 1 < n else float(i + 1), "size_jump")

    # 2. PTS gaps / reversals (skip first frame; needs a stable median duration)
    dts = np.diff(times)
    good = dts[np.isfinite(dts)]
    med_dt = float(np.median(good)) if good.size else 0.0
    if med_dt > 0:
        for i in np.where((dts <= 0) | (dts > 1.75 * med_dt))[0]:
            hit(times[i + 1], "pts_gap")

    # 3. off-cadence keyframes: an early IDR inside an otherwise regular GOP
    kf = [i for i, k in enumerate(keys) if k]
    med_gop = 0.0
    if len(kf) >= 4:
        iv = np.diff(kf)
        med_gop = float(np.median(iv))
        if med_gop >= 8:                      # irregular-GOP files prove nothing
            for j, gap in enumerate(iv):
                if gap < 0.7 * med_gop:
                    hit(times[kf[j + 1]], "idr_offcadence")

    # 4. audio discontinuities
    for t in _audio_jumps(path, cancel=cancel):
        key = round(t / 0.25)
        for k2 in (key - 1, key, key + 1):
            if k2 in events:
                events[k2]["signals"].add("audio_jump")
                break
        else:
            hit(t, "audio_jump")

    # 5. scene cuts lower suspicion (an edit at a cut is normal editing)
    cuts = va_metrics.scene_cuts(path, cancel=cancel)

    weights = {"size_jump": 1, "pts_gap": 2, "idr_offcadence": 1, "audio_jump": 1}
    end_t = float(times[-1]) if n else 0.0
    cand = []
    for e in events.values():
        if e["t"] < 0.3 or (end_t > 1.0 and e["t"] > end_t - 0.25):
            continue                          # encoder start/flush artifacts
        at_cut = any(abs(e["t"] - c) < 0.3 for c in cuts)
        score = sum(weights[s] for s in e["signals"]) - (1 if at_cut else 0)
        if score >= 1:
            cand.append({"t": round(e["t"], 3), "score": int(score),
                         "signals": sorted(e["signals"]), "at_scene_cut": at_cut})
    cand.sort(key=lambda c: (-c["score"], -len(c["signals"]), c["t"]))
    return {
        "candidates": cand[:50],
        "stats": {"cut_times": [round(float(c), 2) for c in cuts[:400]],
                  "frames": int(n), "keyframes": len(kf),
                  "median_gop_frames": round(med_gop, 1) if med_gop else None,
                  "scene_cuts": len(cuts)},
        "note": ("signals agreeing at one timestamp raise the score; score>=3 is "
                 "strongly corroborated, 1-2 is weak. A candidate at a scene cut "
                 "is usually just editing. Indicators, not proof."),
    }


# === Container forensics ======================================================

_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts",
                   b"udta", b"moof", b"traf", b"mvex", b"mfra"}
_EPOCH_1904 = -2082844800           # 1904-01-01 as a unix timestamp


def _mp4_time(v) -> "str | None":
    """MP4 1904-epoch seconds -> ISO string (None for 0 / absurd values)."""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    unix = v + _EPOCH_1904
    if unix < 0 or unix > 4102444800:        # past year 2100 = garbage
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(unix)) + " UTC"


def _walk_mp4(fh, start, end, depth=0, out=None, limit=6000):
    """Collect (depth, type, offset, size, payload_head) for every box.
    Defensive: malformed sizes terminate the current level, never raise."""
    out = out if out is not None else []
    pos = start
    while pos + 8 <= end and len(out) < limit:
        fh.seek(pos)
        head = fh.read(8)
        if len(head) < 8:
            break
        size = struct.unpack(">I", head[:4])[0]
        btype = head[4:8]
        hdr = 8
        if size == 1:
            big = fh.read(8)
            if len(big) < 8:
                break
            size = struct.unpack(">Q", big)[0]
            hdr = 16
        elif size == 0:                       # runs to end of enclosing space
            size = end - pos
        if size < hdr or pos + size > end:
            break
        payload = fh.read(min(64, size - hdr)) if size > hdr else b""
        out.append((depth, btype, pos, size, payload))
        if btype in _MP4_CONTAINERS and depth < 8:
            _walk_mp4(fh, pos + hdr, pos + size, depth + 1, out, limit)
        elif btype == b"meta" and depth < 8 and size >= hdr + 4:
            _walk_mp4(fh, pos + hdr + 4, pos + size, depth + 1, out, limit)
        pos += size
    return out


def _parse_elst(fh, off, size) -> list:
    """Edit-list entries [(segment_duration, media_time)] (version 0/1)."""
    try:
        fh.seek(off + 8)
        data = fh.read(min(size - 8, 16 + 20 * 64))
        if len(data) < 8:
            return []
        ver = data[0]
        count = struct.unpack(">I", data[4:8])[0]
        out = []
        p = 8
        for _ in range(min(count, 64)):
            if ver == 1:
                if p + 20 > len(data):
                    break
                dur, mt = struct.unpack(">Qq", data[p:p + 16])
                p += 20
            else:
                if p + 12 > len(data):
                    break
                dur, mt = struct.unpack(">Ii", data[p:p + 8])
                p += 12
            out.append((int(dur), int(mt)))
        return out
    except (OSError, struct.error):
        return []


def _parse_mvhd(payload) -> dict:
    """creation/modification times out of an mvhd payload head."""
    try:
        ver = payload[0]
        if ver == 1 and len(payload) >= 20:
            cre, mod = struct.unpack(">QQ", payload[4:20])
        elif len(payload) >= 12:
            cre, mod = struct.unpack(">II", payload[4:12])
        else:
            return {}
        return {"created": _mp4_time(cre), "modified": _mp4_time(mod),
                "_cre_raw": cre, "_mod_raw": mod}
    except (IndexError, struct.error):
        return {}


def container_report(path) -> dict:
    """Container-level edit-history tells.

    MP4/MOV (pure python box walk): multiple mdat, large free/skip gaps,
    moov-vs-mdat order, edit lists, mvhd creation vs modification, uuid boxes.
    MKV (mkvmerge -J when installed): muxing/writing application + date.
    Every finding is {severity: info|warn, text}; severities stay conservative -
    container layout alone is rarely proof of anything."""
    rep = {"format": "unknown", "findings": [], "top_level": [], "details": {}}
    f = rep["findings"]
    try:
        size = os.path.getsize(path)
        fh = open(path, "rb")
    except OSError:
        rep["note"] = "file unreadable"
        return rep
    with fh:
        head = fh.read(12)
        if len(head) >= 8 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"skip", b"free", b"styp"):
            rep["format"] = "mp4"
            boxes = _walk_mp4(fh, 0, size)
            top = [(b[1].decode("latin-1"), b[3]) for b in boxes if b[0] == 0]
            rep["top_level"] = top
            order = [t for t, _ in top]
            n_mdat = order.count("mdat")
            if n_mdat > 1:
                f.append({"severity": "warn", "text":
                          "%d mdat boxes - media data was appended after the "
                          "original write (editing/concatenation tell)" % n_mdat})
            if "moov" in order and "mdat" in order:
                rep["details"]["moov_position"] = (
                    "before mdat (faststart - post-processed for streaming)"
                    if order.index("moov") < order.index("mdat")
                    else "after mdat (typical straight recording)")
            free_total = sum(s for t, s in top if t in ("free", "skip"))
            for i, (t, s) in enumerate(top):
                if t in ("free", "skip") and s >= 65536:
                    f.append({"severity": "info", "text":
                              "large %s box (%.1f KB) at top level - often left "
                              "behind by an in-place metadata edit" % (t, s / 1024.0)})
                    break
            rep["details"]["free_bytes_top_level"] = int(free_total)
            uuids = sum(1 for b in boxes if b[1] == b"uuid")
            if uuids:
                rep["details"]["uuid_boxes"] = uuids
            for b in boxes:
                if b[1] == b"mvhd":
                    mv = _parse_mvhd(b[4])
                    if mv:
                        rep["details"]["created"] = mv.get("created")
                        rep["details"]["modified"] = mv.get("modified")
                        cre, mod = mv.get("_cre_raw") or 0, mv.get("_mod_raw") or 0
                        if cre and mod and abs(mod - cre) > 2:
                            f.append({"severity": "info", "text":
                                      "container modified %s after creation (created %s, "
                                      "modified %s)" % ("%.0fs" % abs(mod - cre),
                                                        mv.get("created"), mv.get("modified"))})
                        if not cre:
                            f.append({"severity": "info", "text":
                                      "creation time zeroed in mvhd (metadata stripped "
                                      "or written by a sanitising muxer)"})
                    break
            elsts = [b for b in boxes if b[1] == b"elst"]
            for b in elsts[:4]:
                entries = _parse_elst(fh, b[2], b[3])
                # a single small media_time offset is normal codec priming
                # (AAC encoder delay), not an edit - only bigger offsets or
                # multi-entry lists are tells
                if (len(entries) > 1) or any(e[1] > 4096 for e in entries):
                    f.append({"severity": "info", "text":
                              "edit list with %d entr%s (media_time offsets %s) - "
                              "playback range was trimmed/shifted after encode" % (
                                  len(entries), "y" if len(entries) == 1 else "ies",
                                  ",".join(str(e[1]) for e in entries[:4]))})
                    break
            hdlrs = []
            for b in boxes:
                if b[1] == b"hdlr" and len(b[4]) > 24:
                    name = b[4][24:].split(b"\x00")[0].decode("latin-1", "replace").strip()
                    if name:
                        hdlrs.append(name)
            if hdlrs:
                rep["details"]["handlers"] = hdlrs[:6]
        elif head[:4] == b"\x1aE\xdf\xa3":
            rep["format"] = "mkv"
            tool = find_tool("mkvmerge")
            if tool:
                try:
                    r = subprocess.run([tool, "-J", path], capture_output=True,
                                       timeout=60, encoding="utf-8", errors="replace",
                                       creationflags=CREATIONFLAGS)
                    data = json.loads(r.stdout or "null")
                except (OSError, subprocess.SubprocessError, ValueError):
                    data = None
                props = (data or {}).get("container", {}).get("properties", {}) \
                    if isinstance(data, dict) else {}
                for k, label in (("muxing_application", "muxing app"),
                                 ("writing_application", "writing app"),
                                 ("date_utc", "muxed")):
                    if props.get(k):
                        rep["details"][label] = props[k]
                wa = str(props.get("writing_application", ""))
                if "lavf" in wa.lower() or "ffmpeg" in wa.lower():
                    f.append({"severity": "info", "text":
                              "written by ffmpeg (%s) - re-muxed at least once; "
                              "original recorder metadata is likely gone" % wa})
                if not props.get("date_utc"):
                    f.append({"severity": "info",
                              "text": "no muxing date stored in the segment"})
            else:
                rep["note"] = "mkvmerge not installed - container detail skipped (Tools dialog)"
        else:
            rep["note"] = "not an MP4/MOV or MKV - container walk skipped"
    return rep


# === Encoder fingerprinting & metadata audit ==================================

_BITSTREAM_MARKERS = [
    (b"x264 - core", "x264"), (b"x265 (build", "x265"),
    (b"Lavf", "ffmpeg-mux (Lavf)"), (b"Lavc", "ffmpeg-encode (Lavc)"),
    (b"HandBrake", "HandBrake"), (b"VirtualDub", "VirtualDub"),
    (b"Ambarella", "Ambarella (action cam)"), (b"GoPro", "GoPro"),
    (b"DJI ", "DJI"), (b"Adobe", "Adobe"),
    (b"Apple", "Apple"), (b"SVT-AV1", "SVT-AV1"), (b"libaom", "libaom-AV1"),
    (b"Kvazaar", "Kvazaar"), (b"DivX", "DivX"), (b"Xvid", "Xvid"),
]


def _extract_settings(data, anchor) -> str:
    """The printable run following an encoder marker (x264/x265 SEI text)."""
    i = data.find(anchor)
    if i < 0:
        return ""
    end = i
    while end < len(data) and end - i < 900 and 32 <= data[end] < 127:
        end += 1
    return data[i:end].decode("latin-1", "replace")


def encoder_fingerprint(path, scan_bytes=3_000_000) -> dict:
    """What actually wrote these bytes. Byte-scans the head + tail for encoder
    signatures (x264/x265 settings SEI, Lavf/Lavc, camera vendors) and compares
    with the container's claimed encoder tags."""
    rep = {"markers": [], "settings": None, "tags": {}, "findings": []}
    try:
        with open(path, "rb") as fh:
            data = fh.read(scan_bytes)
            fh.seek(0, 2)
            size = fh.tell()
            if size > scan_bytes:
                fh.seek(max(scan_bytes, size - 1_000_000))
                data += fh.read(1_000_000)
    except OSError:
        rep["note"] = "file unreadable"
        return rep
    seen = []
    for marker, label in _BITSTREAM_MARKERS:
        if marker in data and label not in seen:
            seen.append(label)
    rep["markers"] = seen
    if b"x264 - core" in data:
        s = _extract_settings(data, b"x264 - core")
        rep["settings"] = s[:700]
        m = None
        for token in ("crf=", "bitrate="):
            i = s.find(token)
            if i >= 0:
                m = s[i:s.find(" ", i) if s.find(" ", i) > 0 else None]
                break
        rep["findings"].append({"severity": "info", "text":
                                "x264 settings embedded in the bitstream%s" %
                                (" (%s)" % m if m else "")})
    elif b"x265 (build" in data:
        rep["settings"] = _extract_settings(data, b"x265 (build")[:700]

    pd = ffprobe_json(path) or {}
    fmt_tags = (pd.get("format") or {}).get("tags") or {}
    rep["tags"]["format_encoder"] = fmt_tags.get("encoder")
    rep["tags"]["major_brand"] = fmt_tags.get("major_brand")
    handlers, senc = [], []
    for s in pd.get("streams", []):
        tg = s.get("tags") or {}
        if tg.get("handler_name"):
            handlers.append(tg["handler_name"])
        if tg.get("encoder"):
            senc.append(tg["encoder"])
    rep["tags"]["handlers"] = handlers
    rep["tags"]["stream_encoders"] = senc

    lavf = any("Lavf" in str(v) for v in [fmt_tags.get("encoder", "")] + senc) or \
        ("ffmpeg-mux (Lavf)" in seen)
    camera = [m for m in seen if m in ("Ambarella (action cam)", "GoPro", "DJI", "Apple")]
    if lavf:
        rep["findings"].append({"severity": "info", "text":
                                "ffmpeg (Lavf/Lavc) traces present - the file has been "
                                "re-muxed or re-encoded since the original recording"})
    if camera and lavf:
        rep["findings"].append({"severity": "info", "text":
                                "camera-vendor strings (%s) AND ffmpeg traces coexist - "
                                "camera original processed through ffmpeg" % ", ".join(camera)})
    claimed = str(fmt_tags.get("encoder") or "")
    if claimed and "x264" in seen and "264" not in claimed and "Lavf" not in claimed:
        rep["findings"].append({"severity": "warn", "text":
                                "container encoder tag says %r but the bitstream carries "
                                "x264 settings - the tag was edited or copied" % claimed})
    return rep


def metadata_consistency(path) -> dict:
    """Cheap coherence audit of ffprobe-visible metadata: creation times across
    streams, default ffmpeg handler names, duration mismatches."""
    rep = {"findings": [], "details": {}}
    pd = ffprobe_json(path)
    if not pd:
        rep["note"] = "no ffprobe data"
        return rep
    f = rep["findings"]
    fmt = pd.get("format") or {}
    fmt_tags = fmt.get("tags") or {}
    fct = fmt_tags.get("creation_time")
    rep["details"]["creation_time"] = fct
    stimes = []
    handlers = []
    for s in pd.get("streams", []):
        tg = s.get("tags") or {}
        if tg.get("creation_time"):
            stimes.append((s.get("codec_type", "?"), tg["creation_time"]))
        if tg.get("handler_name"):
            handlers.append(tg["handler_name"])
    if fct is None and not stimes:
        f.append({"severity": "info", "text":
                  "no creation_time anywhere - metadata stripped or written by a "
                  "tool that does not record it (ffmpeg default)"})
    elif fct and stimes and any(t != fct for _, t in stimes):
        f.append({"severity": "warn", "text":
                  "creation_time differs between container (%s) and stream(s) (%s) - "
                  "streams were combined from different sources or times edited" % (
                      fct, ", ".join("%s:%s" % p for p in stimes[:3]))})
    generic = [h for h in handlers if h in ("VideoHandler", "SoundHandler",
                                            "VideoHandler\x00", "ISO Media file produced by Google Inc.")]
    if generic:
        f.append({"severity": "info", "text":
                  "generic handler names (%s) - ffmpeg/server defaults rather than "
                  "camera-original handlers" % ", ".join(sorted(set(g.strip("\x00") for g in generic)))})
    rep["details"]["handlers"] = handlers
    try:
        fdur = float(fmt.get("duration") or 0)
        for s in pd.get("streams", []):
            sd = float(s.get("duration") or 0)
            if fdur and sd and abs(fdur - sd) > max(1.0, 0.05 * fdur):
                f.append({"severity": "info", "text":
                          "%s stream duration %.2fs differs from container %.2fs - "
                          "trailing data trimmed or appended" % (s.get("codec_type", "?"), sd, fdur)})
                break
    except (TypeError, ValueError):
        pass
    return rep


# === Pixel-level maps =========================================================

def ela_map(bgr, quality=75):
    """Error-level analysis for one frame: re-encode as JPEG at a fixed quality
    and show |original - recompressed|. Regions that respond differently from
    their surroundings have a different compression history (paste/overlay
    tells). Returns (heatmap 0..1, mean_residual). cv2 required."""
    if cv2 is None or bgr is None:
        return None, 0.0
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None, 0.0
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None or dec.shape != bgr.shape:
        return None, 0.0
    resid = np.abs(bgr.astype(np.float32) - dec.astype(np.float32)).mean(axis=2)
    p99 = float(np.percentile(resid, 99)) or 1.0
    heat = np.clip(resid / p99, 0.0, 1.0)
    heat = cv2.GaussianBlur(heat, (0, 0), 1.0)
    return heat, round(float(resid.mean()), 3)


def _noise_residual(bgr):
    gray = (0.2126 * bgr[:, :, 2] + 0.7152 * bgr[:, :, 1] +
            0.0722 * bgr[:, :, 0]).astype(np.float32)
    return gray - cv2.medianBlur(gray, 3)


def noise_map(bgr, block=16):
    """Per-block sensor-noise deviation for one frame. Pasted/composited content
    usually carries a different noise floor than its surroundings. Returns
    (heatmap 0..1 of |block sigma - frame median sigma|, stats dict)."""
    if cv2 is None or bgr is None:
        return None, {}
    resid = _noise_residual(bgr)
    h, w = resid.shape
    bh, bw = h // block, w // block
    if bh < 2 or bw < 2:
        return None, {}
    r = resid[:bh * block, :bw * block].reshape(bh, block, bw, block)
    sig = r.std(axis=(1, 3))
    med = float(np.median(sig))
    den = max(med, 0.15)                     # noiseless content: ratios bottom out
    dev = np.abs(sig - med) / den
    heat = np.clip(dev / 2.0, 0.0, 1.0).astype(np.float32)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    q1, q3 = np.percentile(sig, [25, 75])
    stats = {"sigma_median": round(med, 3),
             "sigma_iqr": round(float(q3 - q1), 3),
             "deviation_p95": round(float(np.percentile(dev, 95)), 3)}
    if med < 0.15:
        stats["note"] = "noise floor near zero (synthetic/denoised) - map is weakly informative"
    return heat, stats


def noise_consistency(path, samples=24, cancel=None) -> dict:
    """Noise-floor drift across the timeline: sample frames evenly, measure the
    global noise sigma of each, flag robust-z outliers (a segment whose sensor
    noise does not match the rest of the file was likely sourced elsewhere)."""
    if cv2 is None:
        return {"t": [], "sigma": [], "outliers": [], "note": "cv2 missing"}
    vs = VideoSource(path)
    out_t, out_s = [], []
    try:
        if not vs.info.get("ok"):
            return {"t": [], "sigma": [], "outliers": [], "note": "unreadable"}
        n = vs.info.get("nb_frames") or 0
        fps = vs.fps or 25.0
        idxs = [int(n * (i + 0.5) / samples) for i in range(samples)] if n else list(range(samples))
        for i in idxs:
            if cancel is not None and cancel():
                break
            fr = vs.frame_at(i)
            if fr is None:
                continue
            gray = (0.2126 * fr[:, :, 2] + 0.7152 * fr[:, :, 1] +
                    0.0722 * fr[:, :, 0])
            if float(gray.std()) < 2.0:       # flat/black frame - no signal
                continue
            resid = _noise_residual(fr)
            out_t.append(round(i / fps, 2))
            out_s.append(float(np.median(np.abs(resid))) * 1.4826)
    finally:
        vs.close()
    if len(out_s) < 6:
        return {"t": out_t, "sigma": [round(s, 3) for s in out_s], "outliers": [],
                "note": "too few decodable samples"}
    arr = np.asarray(out_s)
    med = float(np.median(arr))
    if med < 0.05:
        return {"t": out_t, "sigma": [round(s, 3) for s in out_s],
                "median_sigma": round(med, 3), "outliers": [],
                "note": "noise floor near zero (synthetic/denoised content) - "
                        "consistency analysis not informative"}
    mad = float(np.median(np.abs(arr - med))) * 1.4826 + 1e-4
    outliers = [{"t": out_t[i], "sigma": round(float(arr[i]), 3),
                 "z": round(float((arr[i] - med) / mad), 1)}
                for i in np.where(np.abs(arr - med) / mad > 3.0)[0]]
    return {"t": out_t, "sigma": [round(s, 3) for s in out_s],
            "median_sigma": round(med, 3), "outliers": outliers,
            "note": "outlier = noise floor inconsistent with the rest of the file "
                    "(z>3); scene content changes can also shift it - corroborate."}


# === Repeated-sequence (loop) detection =======================================

def frame_loop_scan(path, min_seconds=0.5, cancel=None) -> dict:
    """Find frame SEQUENCES that repeat elsewhere in the timeline (looped or
    recycled footage - the classic doctored-surveillance tell). A 480-bit
    perceptual gradient hash per frame (16x16 gray, horizontal+vertical dHash)
    via an ffmpeg pipe, then fuzzy runs of matching hashes at a constant
    offset. The hash size matters: true re-encoded copies land ~5-15 bits
    apart while cyclic content (clocks, spinners, test patterns) stays >70
    bits, so tolerance 24 separates them. Freezes (static repeats) are
    excluded - those are va_temporal's job."""
    exe = find_ffmpeg()
    if not exe:
        return {"loops": [], "note": "no ffmpeg"}
    info = probe(path)
    fps = info.get("fps") or 25.0
    args = [exe, "-v", "error", "-nostdin", "-i", path,
            "-vf", "scale=16:16:flags=area,format=gray", "-f", "rawvideo", "-"]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=CREATIONFLAGS)
    except OSError:
        return {"loops": [], "note": "ffmpeg failed to start"}
    va_metrics._track(proc)
    rows = []
    try:
        while True:
            if cancel is not None and cancel():
                break
            buf = proc.stdout.read(256)
            if not buf or len(buf) < 256:
                break
            img = np.frombuffer(buf, np.uint8).reshape(16, 16).astype(np.int16)
            bits = np.concatenate([(img[:, 1:] > img[:, :-1]).flatten(),
                                   (img[1:, :] > img[:-1, :]).flatten()])
            rows.append(np.packbits(bits))          # 60 bytes / 480 bits
            if len(rows) > 500_000:
                break
    finally:
        try:
            proc.stdout.close()
            proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
        va_metrics._untrack(proc)
    n = len(rows)
    min_run = max(3, int(round(fps * min_seconds)))
    if n < 2 * min_run:
        return {"loops": [], "frames": n, "note": "too short"}

    TOL = 24
    bview = np.stack(rows)                          # (n, 60) uint8
    pop = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)

    def close(a, b):
        return int(pop[bview[a] ^ bview[b]].sum()) <= TOL

    def _periodic(a, ln, delta):
        """True when the matched region is self-similar at a sub-period of
        delta (rotating/blinking/cyclic content, not a copied sequence)."""
        for div in (2, 3, 4):
            dd = delta // div
            if dd < 2:
                continue
            pts = [p for p in range(a, min(a + ln, n), max(1, ln // 8))
                   if p - dd >= 0]
            if pts and sum(1 for p in pts if close(p, p - dd)) >= 0.7 * len(pts):
                return True
        return False

    stride = max(min_run // 3, n // 4000, 1)
    raw = []
    seen_pairs = set()
    for i in range(min_run, n, stride):
        if cancel is not None and cancel():
            break
        d = pop[bview ^ bview[i]].sum(axis=1)
        match = np.where(d <= TOL)[0]
        if match.size > max(64, n // 4):      # static/ambient content - ambiguous
            continue
        for j in match:
            delta = i - int(j)
            if delta < min_run:
                continue
            key = (i // min_run, delta // min_run)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            a = i                              # extend backward then forward
            while a - 1 >= delta and close(a - 1, a - 1 - delta):
                a -= 1
            b = i
            while b + 1 < n and close(b + 1, b + 1 - delta):
                b += 1
            ln = b - a + 1
            if ln >= min_run:
                for s2 in range(a, b + 1, min_run):
                    seen_pairs.add((s2 // min_run, delta // min_run))
                if ln <= delta and not _periodic(a, ln, delta):
                    raw.append((a, delta, ln))

    # Classify: a delta whose matches TILE the timeline (>=2.5 cycles, >=25%
    # of the possible span matching) is periodicity - benign cyclic content,
    # or a feed looped continuously to cover a span. One-off repeats stay
    # individual loop reports.
    groups = {}
    for s, delta, ln in raw:
        groups.setdefault(round(delta / float(max(1, min_run))), []).append((s, delta, ln))
    periodic = []
    loops = []
    for runs in groups.values():
        delta = int(np.median([d for _, d, _ in runs]))
        total = sum(ln for _, _, ln in runs)
        if n >= 2.5 * delta and total >= 0.25 * max(1, n - delta):
            periodic.append({"period_s": round(delta / fps, 2),
                             "coverage_pct": round(100.0 * total / max(1, n - delta), 1),
                             "cycles": round(n / float(delta), 1)})
            continue
        for s, d2, ln in sorted(runs, key=lambda r: -r[2]):
            seg = {bview[k].tobytes() for k in range(s, s + ln)}
            if len(seg) < max(3, ln // 4):    # a freeze, not a loop
                continue
            if any(abs(s - L["_s"]) < min_run and abs(d2 - L["_d"]) < min_run for L in loops):
                continue
            loops.append({"src_t": round((s - d2) / fps, 2), "repeat_t": round(s / fps, 2),
                          "duration_s": round(ln / fps, 2), "frames": int(ln),
                          "_s": s, "_d": d2})
            if len(loops) >= 20:
                break
    for L in loops:
        L.pop("_s"), L.pop("_d")
    loops.sort(key=lambda L: -L["duration_s"])
    return {"loops": loops, "periodic": periodic[:6], "frames": n,
            "note": "a repeat means content at repeat_t replays content from src_t; "
                    "periodic = the timeline tiles at one offset (cyclic content OR a "
                    "continuously looped feed); freezes are excluded (needs >=%d "
                    "matching frames)" % min_run}


# === ENF (mains hum) ==========================================================

def enf_trace(path, cancel=None) -> dict:
    """Electrical-network-frequency trace: track the 50/60 Hz mains hum that
    real-room recordings often pick up. A continuous hum supports continuity;
    frequency steps or dropouts at one timestamp corroborate an audio edit.
    Many clean/denoised recordings have no hum at all - absence proves nothing."""
    exe = find_ffmpeg()
    if not exe:
        return {"present": False, "note": "no ffmpeg"}
    sr = 400
    args = [exe, "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0",
            "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        r = subprocess.run(args, capture_output=True, timeout=600,
                           creationflags=CREATIONFLAGS)
    except (OSError, subprocess.SubprocessError):
        return {"present": False, "note": "audio decode failed"}
    pcm = np.frombuffer(r.stdout or b"", np.float32)
    win, hop = 8 * sr, 2 * sr
    if pcm.size < win:
        return {"present": False, "note": "no/too little audio"}
    nfft = 4 * win
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    hann = np.hanning(win).astype(np.float32)
    band = {50: (freqs >= 49.0) & (freqs <= 51.0), 60: (freqs >= 59.0) & (freqs <= 61.0)}
    noise_mask = ((freqs >= 35) & (freqs <= 75) &
                  ~((freqs >= 48) & (freqs <= 52)) & ~((freqs >= 58) & (freqs <= 62)))
    rows = {50: [], 60: []}
    times = []
    for s in range(0, pcm.size - win + 1, hop):
        if cancel is not None and cancel():
            break
        spec = np.abs(np.fft.rfft(pcm[s:s + win] * hann, nfft))
        floor = float(np.median(spec[noise_mask])) + 1e-12
        times.append(s / sr + win / (2.0 * sr))
        for hz in (50, 60):
            idx = np.where(band[hz])[0]
            k = idx[np.argmax(spec[idx])]
            # parabolic interpolation for sub-bin frequency
            if 0 < k < spec.size - 1 and spec[k] > 0:
                a, b, c = spec[k - 1], spec[k], spec[k + 1]
                denom = a - 2 * b + c
                shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
                fpk = freqs[k] + shift * (freqs[1] - freqs[0])
            else:
                fpk = freqs[k]
            rows[hz].append((float(fpk), 20.0 * np.log10(float(spec[k]) / floor)))
    if not times:
        return {"present": False, "note": "no analysis windows"}
    best_hz, best_med = None, -1e9
    for hz in (50, 60):
        med = float(np.median([snr for _, snr in rows[hz]]))
        if med > best_med:
            best_hz, best_med = hz, med
    track = rows[best_hz]
    present = best_med >= 10.0 and \
        sum(1 for _, snr in track if snr >= 6.0) >= 0.7 * len(track)
    out = {"present": bool(present), "base_hz": best_hz if present else None,
           "median_snr_db": round(best_med, 1),
           "t": [round(t, 2) for t in times],
           "freq": [round(f, 4) for f, _ in track],
           "snr_db": [round(snr, 1) for _, snr in track], "jumps": [], "gaps": []}
    if present:
        for i in range(1, len(track)):
            if track[i][1] >= 6.0 and track[i - 1][1] >= 6.0 and \
                    abs(track[i][0] - track[i - 1][0]) > 0.08:
                out["jumps"].append(round(times[i], 2))
            if track[i][1] < 6.0:
                out["gaps"].append(round(times[i], 2))
        out["note"] = ("mains hum at %d Hz tracked; %d frequency jump(s), %d dropout "
                       "window(s). Steps/dropouts can corroborate edits." %
                       (best_hz, len(out["jumps"]), len(out["gaps"])))
    else:
        out["note"] = ("no reliable mains hum (median SNR %.1f dB) - common for clean "
                       "or denoised audio; proves nothing by itself." % best_med)
    return out


def render_enf(res, w=900, h=280):
    """Plot the ENF track (frequency over time, SNR-shaded) as an RGB image."""
    if cv2 is None or not res or not res.get("t"):
        return None
    img = np.full((h, w, 3), 16, np.uint8)
    t = res["t"]
    fq = res["freq"]
    snr = res.get("snr_db") or [0.0] * len(t)
    if len(t) < 2:
        return None
    base = res.get("base_hz") or (50 if np.median(fq) < 55 else 60)
    lo, hi = base - 0.5, base + 0.5
    for gy, val in ((0.25, hi - 0.25), (0.5, float(base)), (0.75, lo + 0.25)):
        y = int(gy * (h - 30)) + 15
        cv2.line(img, (50, y), (w - 10, y), (40, 40, 44), 1)
        cv2.putText(img, "%.2f Hz" % val, (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (120, 120, 128), 1, cv2.LINE_AA)
    span_t = t[-1] - t[0] or 1.0
    prev = None
    for i in range(len(t)):
        x = 50 + int((t[i] - t[0]) / span_t * (w - 64))
        fr = max(lo, min(hi, fq[i]))
        y = 15 + int((hi - fr) / (hi - lo) * (h - 30))
        good = snr[i] >= 6.0
        color = (96, 200, 168) if good else (70, 70, 76)
        if prev is not None:
            cv2.line(img, prev, (x, y), color, 1, cv2.LINE_AA)
        prev = (x, y)
    for tj in res.get("jumps", []):
        x = 50 + int((tj - t[0]) / span_t * (w - 64))
        cv2.line(img, (x, 10), (x, h - 10), (95, 108, 224), 1)
    label = ("ENF %s Hz  median SNR %.1f dB  jumps:%d  dropouts:%d" %
             (res.get("base_hz"), res.get("median_snr_db", 0),
              len(res.get("jumps", [])), len(res.get("gaps", [])))) \
        if res.get("present") else "no reliable mains hum detected"
    cv2.putText(img, label, (50, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 205), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# === Report orchestrator ======================================================

def _stage(rep, key, fn, on_progress, label):
    if on_progress is not None:
        try:
            on_progress(label)
        except Exception:
            pass
    try:
        rep[key] = fn()
    except Exception as exc:  # one broken stage must not kill the report
        rep[key] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}


def forensics_report(path, deep=True, cancel=None, on_progress=None) -> dict:
    """Run the whole forensic battery and merge everything into one report dict
    with a unified `findings` list [{severity, area, t, text}] (t=None for
    file-level findings). `deep=False` skips the slow whole-file passes
    (noise consistency, loop scan, ENF). Indicators, not proof."""
    rep = {"file": os.path.basename(path), "size_bytes": None,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        rep["size_bytes"] = os.path.getsize(path)
    except OSError:
        pass
    stages = [
        ("sha256", lambda: sha256_file(path), "hashing (SHA-256)"),
        ("recompression", lambda: recompression(path), "recompression analysis"),
        ("splices", lambda: splice_scan(path, cancel=cancel), "splice scan"),
        ("container", lambda: container_report(path), "container walk"),
        ("fingerprint", lambda: encoder_fingerprint(path), "encoder fingerprint"),
        ("metadata", lambda: metadata_consistency(path), "metadata audit"),
        ("c2pa", lambda: c2pa_summary(path), "Content Credentials"),
        ("audio", lambda: va_audio.forensic_battery(
            path, deep=deep, cancel=cancel,
            on_progress=on_progress), "audio forensics"),
    ]
    if deep:
        stages += [
            ("noise", lambda: noise_consistency(path, cancel=cancel), "noise consistency"),
            ("loops", lambda: frame_loop_scan(path, cancel=cancel), "loop scan"),
            ("enf", lambda: enf_trace(path, cancel=cancel), "ENF trace"),
        ]
    for key, _fn, _lbl in stages:      # canonical key order regardless of
        rep.setdefault(key, None)      # which stage finishes first
    workers = 1
    try:
        import va_perf
        workers = va_perf.pool_workers(len(stages))
    except Exception:   # noqa: BLE001 - battery must run without va_perf
        workers = 1
    if workers > 1:
        # Every stage is an independent measurement (own subprocesses / numpy
        # work, distinct rep key), so they fan out onto a thread pool sized by
        # va_perf - the wall-clock win is roughly the pool width, and the CPU
        # spends its time computing instead of waiting on one child at a time.
        from concurrent.futures import ThreadPoolExecutor
        lock = threading.Lock()

        def _one(item):
            key, fn, label = item
            if on_progress is not None:
                with lock:
                    try:
                        on_progress(label)
                    except Exception:   # noqa: BLE001
                        pass
            try:
                rep[key] = fn()
            except Exception as exc:  # one broken stage must not kill the report
                rep[key] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, stages))
    else:
        for key, fn, label in stages:
            _stage(rep, key, fn, on_progress, label)

    finds = []

    def add(severity, area, t, text):
        finds.append({"severity": severity, "area": area,
                      "t": (round(float(t), 2) if t is not None else None), "text": text})

    for src_key in ("container", "fingerprint", "metadata"):
        d = rep.get(src_key) or {}
        for f in (d.get("findings") or []):
            add(f.get("severity", "info"), src_key, None, f.get("text", ""))

    sp = rep.get("splices") or {}
    for c in (sp.get("candidates") or [])[:12]:
        if c["score"] >= 3:
            add("warn", "splice", c["t"], "possible splice @ %.2fs (score %d: %s)%s" % (
                c["t"], c["score"], "+".join(c["signals"]),
                " at scene cut" if c.get("at_scene_cut") else ""))
        elif c["score"] >= 2 or len(c["signals"]) >= 2:
            add("info", "splice", c["t"], "weak splice signal @ %.2fs (score %d: %s)%s" % (
                c["t"], c["score"], "+".join(c["signals"]),
                " at scene cut" if c.get("at_scene_cut") else ""))

    rc = rep.get("recompression") or {}
    dq = rc.get("dct_double_quant")
    if isinstance(dq, (int, float)) and dq > 0.35:
        add("info", "recompression", None,
            "double-quantization structure %.2f - consistent with re-encoding "
            "(relative indicator)" % dq)

    for o in ((rep.get("noise") or {}).get("outliers") or [])[:8]:
        add("warn", "noise", o["t"],
            "noise floor inconsistent @ %.1fs (sigma %.2f, z %.1f) - content "
            "possibly sourced elsewhere" % (o["t"], o["sigma"], o["z"]))

    for p in ((rep.get("loops") or {}).get("periodic") or [])[:4]:
        add("warn", "loops", None,
            "timeline is hash-periodic every %.2fs (%.0f%% coverage, %.1f cycles) - "
            "cyclic synthetic content, or a feed looped to cover a span" % (
                p["period_s"], p["coverage_pct"], p["cycles"]))
    for L in ((rep.get("loops") or {}).get("loops") or [])[:8]:
        add("warn", "loops", L["repeat_t"],
            "repeated sequence: %.2fs..%.2fs replays %.2fs..%.2fs (%.2fs long)" % (
                L["repeat_t"], L["repeat_t"] + L["duration_s"],
                L["src_t"], L["src_t"] + L["duration_s"], L["duration_s"]))

    enf = rep.get("enf") or {}
    if enf.get("present"):
        for tj in (enf.get("jumps") or [])[:8]:
            add("warn", "enf", tj, "mains-hum frequency step @ %.1fs - corroborates "
                                   "an audio edit at this point" % tj)

    au = rep.get("audio") or {}
    cutv = ((rep.get("splices") or {}).get("stats") or {}).get("cut_times") or []
    for f in (au.get("findings") or []):
        sev, txt, t = f.get("severity", "info"), f.get("text", ""), f.get("t")
        if (t is not None and sev == "warn" and "splice" in txt
                and any(abs(t - c) <= 0.5 for c in cutv)):
            sev = "info"
            txt += " - at a picture cut (normal sound editing)"
        add(sev, "audio", t, txt)

    c2 = rep.get("c2pa") or {}
    if c2.get("present"):
        v = c2.get("validation")
        if v == "INVALID":
            add("warn", "provenance", None,
                "Content Credentials present but INVALID - content was modified "
                "after signing, or the signature is untrusted")
        else:
            add("info", "provenance", None, "Content Credentials present (%s)" %
                (v or "not validated - install c2patool"))

    sev_rank = {"warn": 0, "info": 1}
    finds.sort(key=lambda f: (sev_rank.get(f["severity"], 2),
                              f["t"] if f["t"] is not None else 1e12))
    rep["findings"] = finds
    warns = sum(1 for f in finds if f["severity"] == "warn")
    rep["summary"] = ("%d corroborated concern(s), %d informational note(s) - "
                      "indicators, not proof" % (warns, len(finds) - warns)) if finds \
        else "no integrity concerns surfaced by the battery"
    return rep


def report_marks(rep) -> list:
    """[(t_seconds, label)] for timeline marking / issue navigation."""
    marks = []
    for f in (rep or {}).get("findings", []):
        if f.get("t") is not None:
            marks.append((float(f["t"]), "%s (%s)" % (f.get("area", "forensic"),
                                                      f.get("severity", ""))))
    seen, out = set(), []
    for t, lab in sorted(marks):
        k = round(t, 1)
        if k not in seen:
            seen.add(k)
            out.append((t, lab))
    return out


def _fmt_kv(k, v, pad=22) -> str:
    return "  %-*s %s" % (pad, k, v)


def render_report(rep) -> str:
    """Readable multi-section text rendering of a forensics_report() dict."""
    L = ["FORENSICS / INTEGRITY REPORT - %s" % rep.get("file", "?"),
         "=" * 64,
         _fmt_kv("generated", rep.get("generated", "")),
         _fmt_kv("size", "{:,} bytes".format(rep["size_bytes"]) if rep.get("size_bytes") else "?"),
         _fmt_kv("sha256", rep.get("sha256") or "unreadable"),
         "",
         "VERDICT: %s" % rep.get("summary", ""), ""]
    finds = rep.get("findings") or []
    if finds:
        L.append("FINDINGS (%d)" % len(finds))
        for f in finds:
            ts = ("@ %7.2fs " % f["t"]) if f.get("t") is not None else "          "
            L.append("  [%-4s] %s%s" % (f["severity"].upper(), ts, f["text"]))
        L.append("")

    rc = rep.get("recompression") or {}
    L += ["RECOMPRESSION", _fmt_kv("blockiness", rc.get("blockiness")),
          _fmt_kv("double-quantization", rc.get("dct_double_quant")), ""]

    sp = rep.get("splices") or {}
    st = sp.get("stats") or {}
    L.append("SPLICE SCAN  (%d candidate(s); %s frames, %s keyframes, median GOP %s)" % (
        len(sp.get("candidates") or []), st.get("frames", "?"),
        st.get("keyframes", "?"), st.get("median_gop_frames", "?")))
    for c in (sp.get("candidates") or [])[:15]:
        L.append("  score %d  @ %8.2fs  %-40s%s" % (
            c["score"], c["t"], "+".join(c["signals"]),
            "  (at scene cut)" if c.get("at_scene_cut") else ""))
    L.append("")

    co = rep.get("container") or {}
    L.append("CONTAINER  (%s)" % co.get("format", "?"))
    if co.get("top_level"):
        L.append(_fmt_kv("top-level boxes", " ".join("%s(%d)" % (t, s) for t, s in co["top_level"][:10])))
    for k, v in (co.get("details") or {}).items():
        L.append(_fmt_kv(k, v))
    if co.get("note"):
        L.append(_fmt_kv("note", co["note"]))
    L.append("")

    fp = rep.get("fingerprint") or {}
    L.append("ENCODER FINGERPRINT")
    L.append(_fmt_kv("bitstream markers", ", ".join(fp.get("markers") or []) or "none found"))
    tags = fp.get("tags") or {}
    if tags.get("format_encoder"):
        L.append(_fmt_kv("container encoder tag", tags["format_encoder"]))
    if tags.get("handlers"):
        L.append(_fmt_kv("handlers", ", ".join(tags["handlers"])))
    if fp.get("settings"):
        s = fp["settings"]
        L.append(_fmt_kv("settings string", s[:160] + ("..." if len(s) > 160 else "")))
    L.append("")

    nz = rep.get("noise")
    if nz is not None:
        L.append("NOISE CONSISTENCY  (median sigma %s, %d outlier(s))" % (
            nz.get("median_sigma", "?"), len(nz.get("outliers") or [])))
    lp = rep.get("loops")
    if lp is not None:
        L.append("LOOP SCAN  (%s frames hashed, %d repeated sequence(s), %d periodic "
                 "pattern(s))" % (lp.get("frames", "?"), len(lp.get("loops") or []),
                                  len(lp.get("periodic") or [])))
    enf = rep.get("enf")
    if enf is not None:
        L.append("ENF (MAINS HUM)  %s" % (enf.get("note") or ""))

    au = rep.get("audio")
    if au is not None and au.get("present"):
        L += ["", va_audio.render_battery(au)]

    c2 = rep.get("c2pa") or {}
    L += ["", "CONTENT CREDENTIALS (C2PA)",
          _fmt_kv("present", c2.get("present")),
          _fmt_kv("validation", c2.get("validation") or "-"),
          _fmt_kv("source", c2.get("source", ""))]
    if c2.get("generator"):
        L.append(_fmt_kv("generator", c2["generator"]))
    L += ["", "Every signal above is an INDICATOR to be corroborated, not proof of",
          "tampering. Clean results do not certify authenticity either."]
    return "\n".join(str(x) for x in L)
