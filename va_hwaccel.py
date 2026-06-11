#!/usr/bin/env python3
"""
va_hwaccel - pick a hardware-accelerated ffmpeg decode/tonemap pipeline.

Vendor-agnostic and CODEC-AWARE. A decode method is accepted only if it can truly
hardware-decode THIS file's codec (cuda/qsv via their explicit decoders; others by
forcing the hw output format) - not merely that a GPU device exists, because e.g.
AV1 hardware decode needs a recent GPU. HDR tonemap tries libplacebo (3 option
variants) then tonemap_opencl, else software zscale/tonemap. Output is downscaled
to a cap so the raw pipe stays cheap. Recipes cached per session.

Env: VA_NO_HWACCEL=1 forces software; VA_HWACCEL_DEBUG=1 logs the chosen path.
Run `python hwinfo.py <file>` for a full per-path report with ffmpeg errors.
"""

from __future__ import annotations

import os
import sys
import subprocess
from functools import lru_cache

from va_ffmpeg import find_ffmpeg, has_filter, probe

DECODE_MAX = 1280
_DECODE_PRIORITY = ["cuda", "qsv", "d3d11va", "dxva2", "vaapi", "videotoolbox"]
_DEV_TYPE = {"cuda": "cuda", "qsv": "qsv", "d3d11va": "d3d11va", "dxva2": "dxva2",
             "vaapi": "vaapi", "videotoolbox": "videotoolbox"}
_HW_FMT = {"cuda": "cuda", "qsv": "qsv", "d3d11va": "d3d11",
           "dxva2": "dxva2_vld", "vaapi": "vaapi"}
_LP_VARIANTS = (0, 1, 2)
# ffmpeg decoder names that differ from ffprobe codec names
_DECODER_NAME = {"mpeg2video": "mpeg2", "mpeg1video": "mpeg1"}
CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

_RECIPE: dict = {}
_DECODE_BY_CODEC: dict = {}   # codec -> verified hw decode method (or None)
_LP_VARIANT = 0


@lru_cache(maxsize=1)
def hwaccels() -> frozenset:
    exe = find_ffmpeg()
    if not exe:
        return frozenset()
    try:
        out = subprocess.run([exe, "-hide_banner", "-hwaccels"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=15,
                             creationflags=CREATIONFLAGS).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(s.strip() for s in out.splitlines()
                     if s.strip() and " " not in s.strip() and ":" not in s.strip())


def _decode_methods() -> list:
    if os.environ.get("VA_NO_HWACCEL"):
        return []
    have = hwaccels()
    return [m for m in _DECODE_PRIORITY if m in have]


def _fit(w, h, cap):
    m = max(w, h)
    if m == 0 or m <= cap:
        return max(2, (w // 2) * 2), max(2, (h // 2) * 2)
    s = cap / m
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def _sw_tonemap_vf(w, h) -> str:
    return ("zscale=w=%d:h=%d:t=linear:npl=100,tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:p=bt709:r=tv,format=bgr24" % (w, h))


def _libplacebo_vf(w, h, variant=None, dovi=False) -> str:
    v = _LP_VARIANT if variant is None else variant
    base = ("libplacebo=w=%d:h=%d:colorspace=bt709:color_primaries=bt709:"
            "color_trc=bt709:tonemapping=bt.2390" % (w, h))
    if dovi:
        # Explicit so builds lacking DV support fail the probe (and fall
        # through to the software IPT decode) instead of silently producing
        # the magenta misread.
        base += ":apply_dolbyvision=true"
    if v == 1:
        return base + ",hwdownload,format=bgr24"
    if v == 2:
        return base + ":format=yuv420p,format=bgr24"
    return base + ",format=bgr24"


def _opencl_vf(w, h) -> str:
    return ("format=p010le,hwupload,tonemap_opencl=tonemap=hable:t=bt709:m=bt709:p=bt709:"
            "format=nv12,hwdownload,format=nv12,scale=%d:%d,format=bgr24" % (w, h))


def _scale_vf(w, h, src_w, src_h) -> str:
    if (w, h) == (src_w, src_h):
        return "format=bgr24"
    return "scale=%d:%d:flags=fast_bilinear,format=bgr24" % (w, h)


def _run_capture(pre, vf, path, lavfi_input=None):
    exe = find_ffmpeg()
    if not exe:
        return False, "no ffmpeg found"
    args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error"] + list(pre)
    args += ["-f", "lavfi", "-i", lavfi_input] if lavfi_input else ["-i", path]
    if vf:
        args += ["-vf", vf]
    args += ["-frames:v", "1", "-f", "null", "-"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=40,
                           creationflags=CREATIONFLAGS)
        errs = [ln for ln in (r.stderr or "").splitlines() if ln.strip()]
        return r.returncode == 0, (errs[-1] if errs else "")
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)


def _probe_ok(pre, vf, path) -> bool:
    return _run_capture(pre, vf, path)[0]


def _probe_device(method):
    typ = _DEV_TYPE.get(method, method)
    return _run_capture(["-init_hw_device", typ + "=d"], None, None,
                        lavfi_input="nullsrc=s=64x64:d=0.04")


def _hw_decode_ok(method, path, codec):
    """True iff `method` actually hardware-decodes this file's codec. cuda/qsv use
    explicit decoders (clean, system output); others force the hw output format."""
    if not _probe_device(method)[0]:
        return False
    if method == "cuda" and codec:
        return _run_capture(["-c:v", _DECODER_NAME.get(codec, codec) + "_cuvid"], None, path)[0]
    if method == "qsv" and codec:
        return _run_capture(["-c:v", _DECODER_NAME.get(codec, codec) + "_qsv"], None, path)[0]
    fmt = _HW_FMT.get(method)
    if not fmt:
        return False
    return _run_capture(["-hwaccel", method, "-hwaccel_output_format", fmt],
                        "hwdownload,format=nv12|p010le|yuv420p", path)[0]


def _first_ok_decode(path, codec):
    for m in _decode_methods():
        if _hw_decode_ok(m, path, codec):
            return m
    return None


def _first_ok_tonemap(path, w, h):
    global _LP_VARIANT
    if os.environ.get("VA_NO_HWACCEL"):
        return "sw"
    if has_filter("libplacebo"):
        for v in _LP_VARIANTS:
            if _probe_ok(["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"],
                         _libplacebo_vf(w, h, v), path):
                _LP_VARIANT = v
                return "libplacebo"
    if "opencl" in hwaccels() and has_filter("tonemap_opencl") and _probe_ok(
            ["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"], _opencl_vf(w, h), path):
        return "opencl"
    return "sw"


_HW_SCALE = {"cuda": "scale_cuda", "qsv": "scale_qsv"}


def _color_in(info) -> str:
    """Explicit HDR input characteristics so the tonemap stays correct even if frame
    metadata is dropped (e.g. through a hardware scaler + hwdownload)."""
    if info.get("is_hlg"):
        return "tin=arib-std-b67:min=bt2020nc:pin=bt2020"
    return "tin=smpte2084:min=bt2020nc:pin=bt2020"


def _sw_tonemap_tail(info) -> str:
    return ("zscale=%s:t=linear:npl=100,tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:p=bt709:r=tv,format=bgr24" % _color_in(info))


def _hdr_pipelines(dm, info, w, h):
    """Ordered (pre, vf, label) HDR candidates, best first; probed in turn."""
    no_hw = bool(os.environ.get("VA_NO_HWACCEL"))
    tail = _sw_tonemap_tail(info)
    out = []
    if not no_hw and has_filter("libplacebo"):          # full GPU scale + tonemap
        for v in _LP_VARIANTS:
            pre = ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"]
            if dm:
                pre = pre + ["-hwaccel", dm]
            out.append((pre, _libplacebo_vf(w, h, v), "%s+libplacebo" % (dm or "cpu")))
    if not no_hw and dm in _HW_SCALE and has_filter(_HW_SCALE[dm]):   # GPU decode+scale, tiny SW tonemap
        pre = ["-hwaccel", dm, "-hwaccel_output_format", _HW_FMT[dm]]
        vf = "%s=%d:%d,hwdownload,format=p010le,%s" % (_HW_SCALE[dm], w, h, tail)
        out.append((pre, vf, "%s+gpuscale+sw" % dm))
    if not no_hw and "opencl" in hwaccels() and has_filter("tonemap_opencl"):
        pre = ["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"]
        if dm:
            pre = pre + ["-hwaccel", dm]
        out.append((pre, _opencl_vf(w, h), "%s+opencl" % (dm or "cpu")))
    if dm:                                               # GPU decode, CPU scale+tonemap
        out.append((["-hwaccel", dm],
                    "scale=%d:%d:flags=fast_bilinear,%s" % (w, h, tail), "%s+sw" % dm))
    out.append(([], "scale=%d:%d:flags=fast_bilinear,%s" % (w, h, tail), "cpu+sw"))
    return out


def _ipt_vf(w, h) -> str:
    # raw 10-bit planar pipe; the python side (va_ipt) does IPT->BGR
    return "scale=%d:%d:flags=fast_bilinear,format=yuv420p10le" % (w, h)


def _dovi_metadata_visible(path) -> bool:
    """True when this ffmpeg/ffprobe pair decodes the DV RPU into side data -
    the precondition for libplacebo's apply_dolbyvision to actually fire."""
    try:
        import va_ipt
        return va_ipt.dovi_meta(path) is not None
    except Exception:   # noqa: BLE001 - capability probe must never raise
        return False


def _ipt_pipelines(dm, info, w, h):
    """Ordered candidates for Dolby Vision profile 5 (IPT-PQ base layer).
    Only DV-aware paths are eligible: libplacebo applies the RPU exactly;
    the numpy IPT decode (va_ipt) is the always-available fallback. The
    generic zscale/tonemap chains would misread IPT as YCbCr, so they are
    deliberately NOT offered here."""
    no_hw = bool(os.environ.get("VA_NO_HWACCEL"))
    out = []
    if not no_hw and has_filter("libplacebo") and _dovi_metadata_visible(info.get("path") or ""):
        for v in _LP_VARIANTS:
            pre = ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"]
            if dm:
                pre = pre + ["-hwaccel", dm]
            out.append((pre, _libplacebo_vf(w, h, v, dovi=True),
                        "%s+libplacebo-dovi" % (dm or "cpu")))
    if dm and not no_hw:
        out.append((["-hwaccel", dm], _ipt_vf(w, h), "%s+ipt-np" % dm))
    out.append(([], _ipt_vf(w, h), "cpu+ipt-np"))
    return out


def _codec_of(path, info=None):
    if info and info.get("codec"):
        return info["codec"]
    try:
        return probe(path).get("codec", "")
    except Exception:
        return ""


def decode_method(path, info=None):
    """Cached hardware decode method that truly decodes this file's codec, or None.
    Keyed by codec since AV1 / HEVC / H.264 hw support differs across GPUs."""
    codec = _codec_of(path, info)
    if codec not in _DECODE_BY_CODEC:
        _DECODE_BY_CODEC[codec] = _first_ok_decode(path, codec)
    return _DECODE_BY_CODEC[codec]


def decode_args(path, info=None) -> list:
    """ffmpeg input flags to hardware-accelerate decoding ([] if software)."""
    m = decode_method(path, info)
    return ["-hwaccel", m] if m else []


_PIPELINE_CACHE: dict = {}


def choose_video_pipeline(path, info, cap=DECODE_MAX, native_ar=True) -> dict:
    """``native_ar=True`` sizes the output to the file's DISPLAY geometry
    (SAR-corrected, rotation-aware - ffmpeg auto-rotates on decode, the scale
    then bakes square pixels), so anamorphic/vertical files render at their
    true shape. ``native_ar=False`` keeps storage geometry untouched."""
    w, h = info.get("width", 0), info.get("height", 0)
    rot = int(info.get("rotation") or 0) if native_ar else 0
    if native_ar:
        tw = info.get("display_width") or w
        th = info.get("display_height") or h
    else:
        tw, th = w, h
    ow, oh = _fit(tw, th, cap)
    # frames reaching the filter graph are already auto-rotated, so compare
    # the no-op fast path against the rotated storage shape
    src_w, src_h = (h, w) if rot in (90, 270) else (w, h)
    is_hdr = bool(info.get("is_hdr"))
    is_ipt = bool(info.get("dovi_ipt"))
    key = (_codec_of(path, info), is_hdr, is_ipt, ow, oh, rot)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    dm = decode_method(path, info)
    if is_ipt:
        cands = _ipt_pipelines(dm, info, ow, oh)
    elif is_hdr:
        cands = _hdr_pipelines(dm, info, ow, oh)
        if rot in (90, 270):
            # hw-frame paths cannot auto-rotate (no transpose on GPU frames);
            # keep only candidates whose frames are in system memory
            cands = [c for c in cands if "gpuscale" not in c[2]]
    else:
        cands = ([(["-hwaccel", dm], _scale_vf(ow, oh, src_w, src_h), dm)] if dm else [])
        cands.append(([], _scale_vf(ow, oh, src_w, src_h), "cpu"))
    chosen = None
    for pre, vf, label in cands:
        if _probe_ok(pre, vf, path):
            chosen = {"pre": pre, "vf": vf, "out_w": ow, "out_h": oh, "label": label}
            break
    if chosen is None:
        if is_ipt:
            chosen = {"pre": [], "vf": _ipt_vf(ow, oh), "out_w": ow, "out_h": oh,
                      "label": "cpu+ipt-np"}
        else:
            vf = _sw_tonemap_vf(ow, oh) if is_hdr else _scale_vf(ow, oh, src_w, src_h)
            chosen = {"pre": [], "vf": vf, "out_w": ow, "out_h": oh,
                      "label": "cpu+sw" if is_hdr else "cpu"}
    if "ipt-np" in chosen["label"]:
        chosen["pipe_fmt"] = "yuv420p10le"
        chosen["post"] = "ipt"
    if os.environ.get("VA_HWACCEL_DEBUG"):
        sys.stderr.write("[hwaccel] %s -> %s  vf=%s\n" % (
            os.path.basename(path), chosen["label"], chosen["vf"]))
    _PIPELINE_CACHE[key] = chosen
    return chosen


def diagnostics(path) -> dict:
    codec = _codec_of(path)
    rep = {
        "ffmpeg": find_ffmpeg(), "codec": codec, "hwaccels": sorted(hwaccels()),
        "filters": {f: has_filter(f) for f in
                    ("libplacebo", "tonemap_opencl", "tonemap_vaapi", "zscale", "scale_cuda")},
        "decode": [], "tonemap": [],
    }
    for m in _decode_methods():
        dev_ok, dev_err = _probe_device(m)
        dec_ok = _hw_decode_ok(m, path, codec) if dev_ok else False
        rep["decode"].append({"method": m, "device_ok": dev_ok,
                              "decode_ok": dec_ok, "error": dev_err})
    w, h = 1280, 720
    if has_filter("libplacebo"):
        for v in _LP_VARIANTS:
            ok, err = _run_capture(["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"],
                                   _libplacebo_vf(w, h, v), path)
            rep["tonemap"].append({"method": "libplacebo[v%d]" % v, "ok": ok, "error": err})
    if "opencl" in hwaccels() and has_filter("tonemap_opencl"):
        ok, err = _run_capture(["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"],
                               _opencl_vf(w, h), path)
        rep["tonemap"].append({"method": "tonemap_opencl", "ok": ok, "error": err})
    ok, err = _run_capture([], _sw_tonemap_vf(w, h), path)
    rep["tonemap"].append({"method": "software (zscale)", "ok": ok, "error": err})
    return rep


def reset_cache():
    global _LP_VARIANT
    _RECIPE.clear()
    _DECODE_BY_CODEC.clear()
    _PIPELINE_CACHE.clear()
    _LP_VARIANT = 0
    hwaccels.cache_clear()
    try:
        import va_ipt
        va_ipt.reset_cache()
    except Exception:   # noqa: BLE001
        pass
