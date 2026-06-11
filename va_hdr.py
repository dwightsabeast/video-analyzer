#!/usr/bin/env python3
"""
va_hdr - HDR light-level intelligence.

  * nits_map()        - decode the PQ-encoded frame and apply the SMPTE ST 2084
                        EOTF to get a per-pixel luminance (cd/m^2) map.
  * maxcll_maxfall()  - measure MaxCLL / MaxFALL from the pixels and compare to the
                        values DECLARED in the file's metadata (a common, rarely
                        caught delivery error).
  * multi_display()   - preview how the HDR frame clips on 400 / 1000 / 4000-nit
                        displays (clipped pixels marked), beside the SDR tonemap.

PQ only (HDR10). Pure ffmpeg + cv2/numpy.
"""

from __future__ import annotations

import subprocess

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from va_ffmpeg import find_ffmpeg, ffprobe_json, probe, VideoSource, CREATIONFLAGS

_M1 = 0.1593017578125
_M2 = 78.84375
_C1 = 0.8359375
_C2 = 18.8515625
_C3 = 18.6875


def pq_eotf(e):
    """SMPTE ST 2084 EOTF: normalised PQ signal (0..1) -> luminance in cd/m^2."""
    ep = np.power(np.clip(e, 0.0, 1.0), 1.0 / _M2)
    num = np.clip(ep - _C1, 0.0, None)
    den = np.clip(_C2 - _C3 * ep, 1e-6, None)
    return 10000.0 * np.power(num / den, 1.0 / _M1)


def _decode_pq_rgb(path, idx, w, h, fps, cap=None):
    exe = find_ffmpeg()
    if not exe or w <= 0 or h <= 0:
        return None
    vf = "format=rgb48le"
    ow, oh = w, h
    if cap and max(w, h) > cap:
        s = cap / max(w, h)
        ow, oh = max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)
        vf = "scale=%d:%d,format=rgb48le" % (ow, oh)
    args = [exe, "-hide_banner", "-loglevel", "error", "-ss", "%.4f" % (idx / (fps or 25.0)),
            "-i", path, "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    try:
        out = subprocess.run(args, capture_output=True, timeout=60,
                             creationflags=CREATIONFLAGS).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    need = ow * oh * 3
    arr = np.frombuffer(out, np.uint16)
    if arr.size < need:
        return None
    return arr[:need].reshape(oh, ow, 3).astype(np.float32) / 65535.0


def nits_map(path, idx=None, cap=None):
    """Per-pixel max-RGB luminance (cd/m^2) for one frame.

    PQ (SMPTE ST 2084) streams only: the EOTF applied here is meaningless for
    HLG/SDR pixels, so unreadable or non-PQ inputs return None."""
    info = probe(path)
    if not info.get("ok") or not info.get("is_pq"):
        return None
    if idx is None:
        idx = max(0, (info.get("nb_frames") or 2) // 2)
    rgb = _decode_pq_rgb(path, idx, info["width"], info["height"], info["fps"], cap)
    if rgb is None:
        return None
    return pq_eotf(rgb).max(axis=2)


def declared_cll(path):
    """Declared (MaxCLL, MaxFALL) from metadata, or (None, None)."""
    data = ffprobe_json(path, frames=True)
    if not data:
        return None, None
    blocks = []
    for s in data.get("streams", []):
        blocks += s.get("side_data_list", []) or []
    for f in data.get("frames", []):
        blocks += f.get("side_data_list", []) or []
    for sd in blocks:
        if "content light" in str(sd.get("side_data_type", "")).lower():
            return sd.get("max_content"), sd.get("max_average")
    return None, None


def _declared_limit(v):
    """CTA-861.3 semantics: MaxCLL/MaxFALL of 0 (or absent/junk) means UNKNOWN,
    not a 0-nit limit. Returns a usable int limit, or None."""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def maxcll_maxfall(path, samples=6) -> dict:
    info = probe(path)
    result = {"measured_maxcll": None, "measured_maxfall": None,
              "declared_maxcll": None, "declared_maxfall": None, "notes": []}
    if not info.get("ok"):
        result["notes"].append("file unreadable — no analysis performed")
        return result
    if not info.get("is_pq"):
        dcll, dfall = declared_cll(path)
        result["declared_maxcll"], result["declared_maxfall"] = dcll, dfall
        if info.get("is_hlg"):
            result["notes"].append("HLG transfer — PQ-only measurement not applicable "
                                   "(HLG OOTF not implemented)")
        else:
            result["notes"].append("not a PQ stream — light levels not measured")
        return result
    n = info.get("nb_frames") or 0
    idxs = [max(0, int(n * (i + 1) / (samples + 1))) for i in range(samples)] if n else [0]
    peak, falls = 0.0, []
    for i in idxs:
        nm = nits_map(path, i, cap=480)
        if nm is None:
            continue
        peak = max(peak, float(nm.max()))
        falls.append(float(nm.mean()))
    measured_cll = round(peak)
    measured_fall = round(max(falls)) if falls else 0
    dcll, dfall = declared_cll(path)
    lim_cll, lim_fall = _declared_limit(dcll), _declared_limit(dfall)
    notes = []
    if lim_cll is None and lim_fall is None:
        notes.append("no MaxCLL/MaxFALL declared in metadata (0 = unknown)"
                     if (dcll is not None or dfall is not None)
                     else "no MaxCLL/MaxFALL declared in metadata")
    else:
        if lim_cll is not None and measured_cll > lim_cll * 1.1 + 50:
            notes.append("measured MaxCLL %d exceeds declared %d" % (measured_cll, lim_cll))
        if lim_fall is not None and measured_fall > lim_fall * 1.1 + 20:
            notes.append("measured MaxFALL %d exceeds declared %d" % (measured_fall, lim_fall))
        if not notes:
            notes.append("measured light levels within declared metadata")
    result.update({"measured_maxcll": measured_cll, "measured_maxfall": measured_fall,
                   "declared_maxcll": dcll, "declared_maxfall": dfall, "notes": notes})
    return result


def multi_display(path, idx=None, targets=(400, 1000, 4000)) -> "tuple":
    """RGB montage: SDR tonemap + each target-nit display with clipped pixels marked.
    Returns (montage_rgb, clip_stats); (None, {}) when the file is not PQ."""
    info = probe(path)
    if not info.get("ok") or not info.get("is_pq"):
        return None, {}
    vs = VideoSource(path)
    if idx is None:
        idx = max(0, (vs.nb_frames or 2) // 2)
    base = vs.frame_at(idx)          # tonemapped SDR preview (BGR)
    vs.close()
    nm = nits_map(path, idx)
    if base is None or nm is None:
        return None, {}
    nm = cv2.resize(nm, (base.shape[1], base.shape[0]))
    stats = {}

    def label(img, txt):
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 18), (0, 0, 0), -1)
        cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    tiles = []
    sdr = base.copy()
    label(sdr, "SDR tonemap")
    tiles.append(sdr)
    for t in targets:
        tile = base.copy()
        clip = nm > t
        pct = round(float(clip.mean()) * 100.0, 2)
        stats["%dnit_clip_pct" % t] = pct
        tile[clip] = (0, 0, 255)     # mark clipped (BGR red)
        label(tile, "%d nits: %.1f%% clipped" % (t, pct))
        tiles.append(tile)
    montage = cv2.cvtColor(cv2.hconcat(tiles), cv2.COLOR_BGR2RGB)
    return montage, stats
