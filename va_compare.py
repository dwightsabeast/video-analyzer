#!/usr/bin/env python3
"""
va_compare - A/B visual comparison + encode-ladder analysis.

difference(a, b, mode): render two BGR frames as a difference / heatmap / split /
blend image (RGB out). encode_ladder(): encode a source at several bitrates
(codec h264/hevc/av1) and score each against the source (PSNR/SSIM, VMAF when
available) to find the knee (per-title bitrate). Pure compute (ffmpeg + cv2/numpy).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import subprocess

from functools import lru_cache

import cv2
import numpy as np

from va_ffmpeg import find_ffmpeg
import va_quality

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def difference(a: np.ndarray, b: np.ndarray, mode: str = "heatmap", amplify: int = 4) -> np.ndarray:
    """Return an RGB comparison of two BGR frames. mode: a|b|diff|heatmap|split|blend."""
    if a is None:
        return None
    if b is not None and b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if mode == "a" or b is None:
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    if mode == "b":
        return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    if mode == "split":
        out = a.copy()
        w = a.shape[1]
        out[:, w // 2:] = b[:, w // 2:]
        cv2.line(out, (w // 2, 0), (w // 2, a.shape[0]), (255, 255, 255), 1)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if mode == "blend":
        return cv2.cvtColor(cv2.addWeighted(a, 0.5, b, 0.5, 0), cv2.COLOR_BGR2RGB)
    diff = cv2.absdiff(a, b)
    if mode == "diff":
        amp = np.clip(diff.astype(np.int32) * amplify, 0, 255).astype(np.uint8)
        return cv2.cvtColor(amp, cv2.COLOR_BGR2RGB)
    mag = np.clip(diff.max(axis=2).astype(np.int32) * amplify, 0, 255).astype(np.uint8)
    hm = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


_CODECS = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}
_CODEC_ARGS = {"hevc": ["-preset", "fast", "-tag:v", "hvc1"], "av1": ["-preset", "8"]}

_LAST_ERROR = None   # why the most recent ladder came back empty (GUI hint)


def last_error() -> "str | None":
    """Reason the most recent encode_ladder() returned []: 'no ffmpeg',
    "unknown codec 'x'", 'encoder libx265 not in this ffmpeg build' or None
    (all good). Lets callers tell a missing encoder from an encode failure."""
    return _LAST_ERROR


def _fail(reason):
    global _LAST_ERROR
    _LAST_ERROR = reason


@lru_cache(maxsize=1)
def _encoders() -> frozenset:
    """Encoder names exposed by the bundled/PATH ffmpeg (e.g. 'libx265')."""
    exe = find_ffmpeg()
    if not exe:
        return frozenset()
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=15,
                             creationflags=CREATIONFLAGS).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return frozenset(names)


def encoder_available(codec: str) -> bool:
    """True when this ffmpeg build can encode ``codec`` ('h264'|'hevc'|'av1')."""
    enc = _CODECS.get(codec)
    return enc is not None and enc in _encoders()


def encode_ladder(source, bitrates_kbps, on_progress=None, cancel=None, codec="h264") -> list:
    """Encode ``source`` at each bitrate (``codec``: 'h264'|'hevc'|'av1') and score
    vs source. Returns a list of {bitrate_kbps, size_bytes, psnr, ssim, vmaf, codec};
    [] with the reason in last_error() when the codec/encoder is unusable."""
    _fail(None)
    exe = find_ffmpeg()
    results = []
    if not exe:
        _fail("no ffmpeg")
        return results
    encoder = _CODECS.get(codec)
    if encoder is None:
        _fail("unknown codec '%s'" % codec)
        return results
    if encoder not in _encoders():
        _fail("encoder %s not in this ffmpeg build" % encoder)
        return results
    extra = _CODEC_ARGS.get(codec, [])
    tmpdir = tempfile.mkdtemp(prefix="va_ladder_")
    try:
        for i, br in enumerate(bitrates_kbps):
            if cancel is not None and cancel():
                break
            enc = os.path.join(tmpdir, "enc_%dk.mp4" % br)
            try:
                subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", source,
                                "-c:v", encoder, "-b:v", "%dk" % br] + extra + ["-an", enc],
                               capture_output=True, timeout=600, creationflags=CREATIONFLAGS)
            except (OSError, subprocess.SubprocessError):
                continue
            if not os.path.isfile(enc):
                continue
            q = va_quality.compare(enc, source, vmaf=va_quality.vmaf_available())
            results.append({
                "bitrate_kbps": br, "size_bytes": os.path.getsize(enc),
                "psnr": (q["psnr"] or {}).get("average"),
                "ssim": (q["ssim"] or {}).get("average"),
                "vmaf": (q["vmaf"] or {}).get("average"),
                "codec": codec,
            })
            try:
                os.remove(enc)
            except OSError:
                pass
            if on_progress is not None:
                on_progress(i + 1, len(bitrates_kbps))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def optimal_bitrate(results, metric=None, frac=0.98) -> "dict | None":
    """Lowest bitrate reaching ``frac`` of the best score (per-title 'knee')."""
    if not results:
        return None
    if metric is None:
        for m in ("vmaf", "ssim", "psnr"):
            if all(r.get(m) is not None for r in results):
                metric = m
                break
    if metric is None:
        return None
    finite = [r[metric] for r in results if r[metric] not in (None, float("inf"))]
    if not finite:
        return None
    best = max(finite)
    target = best * frac
    cands = [r for r in results if r[metric] is not None and
             (r[metric] == float("inf") or r[metric] >= target)]
    opt = min(cands, key=lambda r: r["bitrate_kbps"]) if cands else results[-1]
    return {"metric": metric, "bitrate_kbps": opt["bitrate_kbps"], "value": opt[metric]}
