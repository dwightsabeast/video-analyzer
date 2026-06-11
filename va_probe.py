#!/usr/bin/env python3
"""
va_probe - HDR/colour metadata classification and ffprobe text formatting.

Extracted unchanged from the original single-file app so the proven HDR logic
(HDR10 / HDR10+ / HLG / Dolby Vision detection, side-data parsing, pretty
formatting) is reusable by the new GUI. No Tkinter here.
"""

from __future__ import annotations

import os
import json
import time
import subprocess
from collections import OrderedDict

try:
    import cv2
except ImportError:
    cv2 = None

from va_ffmpeg import find_ffprobe, CREATIONFLAGS


COLOR_PRIMARIES = {
    "1":  "BT.709 (SDR)",
    "4":  "BT.470M",
    "5":  "BT.470BG (PAL/SECAM)",
    "6":  "SMPTE 170M (NTSC)",
    "7":  "SMPTE 240M",
    "8":  "Film",
    "9":  "BT.2020 (HDR / Wide Gamut)",
    "10": "SMPTE ST 428-1 (CIE XYZ)",
    "11": "SMPTE RP 431-2 (DCI-P3)",
    "12": "SMPTE EG 432-1 (Display P3)",
    "22": "EBU Tech. 3213-E",
}

TRANSFER_CHARACTERISTICS = {
    "1":  "BT.709 (SDR Gamma ≈2.4)",
    "4":  "BT.470M (Gamma 2.2)",
    "5":  "BT.470BG (Gamma 2.8)",
    "6":  "SMPTE 170M",
    "7":  "SMPTE 240M",
    "8":  "Linear",
    "9":  "Logarithmic (100:1)",
    "10": "Logarithmic (316:1)",
    "11": "IEC 61966-2-4",
    "12": "BT.1361",
    "13": "sRGB / sYCC",
    "14": "BT.2020 (10-bit SDR)",
    "15": "BT.2020 (12-bit SDR)",
    "16": "SMPTE ST 2084 (PQ / HDR10)",
    "17": "SMPTE ST 428-1",
    "18": "ARIB STD-B67 (HLG)",
}

MATRIX_COEFFICIENTS = {
    "0":  "Identity / GBR",
    "1":  "BT.709",
    "5":  "BT.470BG",
    "6":  "SMPTE 170M",
    "7":  "SMPTE 240M",
    "8":  "YCgCo",
    "9":  "BT.2020 NCL (Non-Constant Luminance)",
    "10": "BT.2020 CL (Constant Luminance)",
    "11": "SMPTE ST 2085 (Y'D'zD'x)",
    "12": "Chroma-derived NCL",
    "13": "Chroma-derived CL",
    "14": "ICtCp",
}


_UNTAGGED = "not tagged (BT.709 assumed)"


def _safe_int(val, default=None):
    """int(val) or default — never raises (handles 'N/A', None, junk)."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=None):
    """float(val) or default — never raises (handles 'N/A', None, junk)."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt_hms(sec) -> str:
    """Seconds -> HH:MM:SS; 'n/a' for negative, absurd or unparseable values."""
    sec = _safe_float(sec)
    if sec is None or sec < 0 or sec > 86400 * 36500:   # > 100 years: absurd
        return "n/a"
    try:
        return time.strftime("%H:%M:%S", time.gmtime(sec))
    except (ValueError, OverflowError, OSError):
        return "n/a"


def _colour_tag(table: dict, raw: str) -> str:
    """Friendly name for a colour tag; absent/unknown tags display explicitly."""
    raw = str(raw or "").strip()
    if not raw or raw.lower() in ("none", "unknown", "n/a", "unspecified", "2"):
        return _UNTAGGED
    return table.get(raw, raw)


def classify_hdr(stream: dict, side_data: "list | None" = None) -> dict:
    """Classify the HDR format from stream metadata and side data."""
    info: "OrderedDict[str, str]" = OrderedDict()

    transfer = str(stream.get("color_transfer", ""))
    primaries = str(stream.get("color_primaries", ""))
    matrix = str(stream.get("color_space", ""))
    pix_fmt = stream.get("pix_fmt", "")
    bits = stream.get("bits_per_raw_sample", "")

    info["Color Primaries"] = _colour_tag(COLOR_PRIMARIES, primaries)
    info["Transfer Curve"] = _colour_tag(TRANSFER_CHARACTERISTICS, transfer)
    info["Matrix Coefficients"] = _colour_tag(MATRIX_COEFFICIENTS, matrix)
    info["Pixel Format"] = pix_fmt
    if bits:
        info["Bit Depth"] = f"{bits}-bit"

    hdr_type = "SDR"
    is_wide_gamut = primaries in ("9", "bt2020nc", "bt2020c", "bt2020")
    is_pq = transfer in ("16", "smpte2084")
    is_hlg = transfer in ("18", "arib-std-b67")
    is_high_bit = bool(pix_fmt) and ("10" in pix_fmt or "12" in pix_fmt or "16" in pix_fmt)

    if is_pq:
        hdr_type = "HDR10 (PQ)"
    elif is_hlg:
        hdr_type = "HLG"
    elif is_wide_gamut and is_high_bit:
        hdr_type = "Wide Color Gamut (possibly HDR)"

    has_mdcv = has_cll = has_dovi = has_hdr10plus = False
    dovi_sd: dict = {}

    if side_data:
        for sd in side_data:
            if not isinstance(sd, dict):
                continue
            sd_type = str(sd.get("side_data_type", ""))
            t = sd_type.lower()

            # Mastering Display Colour Volume (MDCV) — HDR10 static metadata
            if "mastering display" in t:
                has_mdcv = True
                info["── Mastering Display ──"] = ""
                for key in ("red_x", "red_y", "green_x", "green_y",
                            "blue_x", "blue_y", "white_point_x", "white_point_y"):
                    if key in sd:
                        info[f"  {key}"] = _rational(sd[key], "{:.5f}")
                for key in ("min_luminance", "max_luminance"):
                    if key in sd:
                        info[f"  {key}"] = _rational(sd[key], "{:.4f}") + " cd/m² (nits)"

            # Content Light Level (CLL) — HDR10
            elif "content light level" in t:
                has_cll = True
                info["── Content Light Level ──"] = ""
                if "max_content" in sd:
                    info["  MaxCLL"] = f"{sd['max_content']} nits"
                if "max_average" in sd:
                    info["  MaxFALL"] = f"{sd['max_average']} nits"

            # Dolby Vision (ffprobe: "DOVI configuration record")
            elif "dovi" in t or "dolby vision" in t:
                has_dovi = True
                dovi_sd = sd
                info["── Dolby Vision ──"] = ""
                for k, v in sd.items():
                    if k != "side_data_type":
                        info[f"  {k}"] = str(v)

            # HDR10+ (ffprobe: "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)")
            elif "hdr10+" in t or "smpte2094" in t or "smpte 2094" in t or "hdr dynamic metadata" in t:
                has_hdr10plus = True
                info["── HDR10+ Dynamic Metadata ──"] = ""
                for k, v in sd.items():
                    if k != "side_data_type":
                        info[f"  {k}"] = str(v)

    if has_dovi:
        # Base-layer compatibility decides what non-DV players get
        # (IDs as in va_dynhdr.DV_COMPAT: 0=none, 1/6=HDR10, 2=SDR, 4=HLG).
        dv_profile = _safe_int(dovi_sd.get("dv_profile"))
        dv_compat = _safe_int(dovi_sd.get("dv_bl_signal_compatibility_id"))
        if dv_profile == 5 or dv_compat == 0:
            hdr_type = ("Dolby Vision P5 (IPT-PQ)" if dv_profile == 5
                        else "Dolby Vision (no fallback)")
            info["  pixel encoding"] = ("IPT-PQ, not YCbCr - needs a DV-aware "
                                        "decoder (plain decode looks magenta)")
        elif dv_compat in (1, 6):
            hdr_type = "Dolby Vision + HDR10"
        elif dv_compat == 2:
            hdr_type = "Dolby Vision + SDR base"
        elif dv_compat == 4:
            hdr_type = "Dolby Vision + HLG base"
        elif is_hlg:
            hdr_type = "Dolby Vision + HLG"
        else:
            hdr_type = "Dolby Vision"
    elif has_hdr10plus and is_pq:
        hdr_type = "HDR10+"
    elif has_hdr10plus:
        hdr_type = "HDR10+ (non-PQ stream — check transfer tags)" + \
                   (" + HLG" if is_hlg else "")
    elif is_pq and has_mdcv:
        hdr_type = "HDR10"
    elif is_pq:
        hdr_type = "PQ (HDR10 — no static metadata)"

    info["HDR Format"] = hdr_type

    if is_wide_gamut:
        info["Wide Gamut"] = "Yes (BT.2020)"
    if has_mdcv:
        info["Static Metadata"] = "SMPTE ST 2086 (MDCV) present"
    if has_cll:
        info["Light Level"] = "CTA-861.3 (CLL) present"

    return info


def _rational(raw, fmt: str) -> str:
    """ffprobe gives some values as 'N/M' rationals — render them as decimals."""
    if isinstance(raw, str) and "/" in raw:
        try:
            num, den = raw.split("/")
            den_i = int(den)
            if den_i != 0:
                return fmt.format(int(num) / den_i)
        except (ValueError, ZeroDivisionError):
            return str(raw)
    return str(raw)


# ─── Stream Info Extraction ──────────────────────────────────────────────────

def extract_with_ffprobe(path: str) -> "dict | None":
    """Get rich metadata via ffprobe (bundled or on PATH), if available."""
    exe = find_ffprobe()
    if not exe:
        return None

    base = [exe, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams"]
    # First attempt grabs the first frame's side_data (HDR dynamic metadata);
    # the second is a lighter fallback without frame decoding.
    attempts = [
        (base + ["-show_frames", "-read_intervals", "%+#1", path], 30),
        (base + [path], 15),
    ]
    for args, tmo in attempts:
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=tmo,
                                    creationflags=CREATIONFLAGS)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            continue
    return None


def extract_with_opencv(path: str) -> dict:
    """Fallback: extract what OpenCV can tell us (no HDR metadata)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "Cannot open file with OpenCV"}

    props: "OrderedDict[str, str]" = OrderedDict()
    props["File"] = os.path.basename(path)
    props["File Size"] = f"{os.path.getsize(path) / (1024 * 1024):.2f} MB"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    duration = total / fps if fps > 0 else 0

    props["Resolution"] = f"{w}x{h}"
    props["Frame Rate"] = f"{fps:.3f} fps"
    props["Total Frames"] = str(total)
    props["Duration"] = f"{duration:.2f}s ({_fmt_hms(duration)})"
    props["FourCC Codec"] = fourcc_str
    props["Backend"] = cap.getBackendName()
    if find_ffprobe():
        props["HDR Note"] = "ffprobe found but returned no data for this file"
    else:
        props["HDR Note"] = "Bundle ffprobe beside this script for full HDR metadata"
    cap.release()
    return {"opencv": props}


def build_summary(probe: dict) -> "list[tuple[str, str]]":
    """Pull the headline facts out of an ffprobe payload."""
    out: "list[tuple[str, str]]" = []
    fmt = probe.get("format", {})
    if not isinstance(fmt, dict):
        fmt = {}
    streams = probe.get("streams", [])
    if not isinstance(streams, list):
        streams = []

    long_name = fmt.get("format_long_name") or fmt.get("format_name")
    if long_name:
        out.append(("Container", long_name))

    v = next((s for s in streams
              if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if v:
        codec = v.get("codec_name", "?")
        if v.get("profile"):
            codec += f" ({v['profile']})"
        out.append(("Codec", codec))
        if v.get("width"):
            res = f"{v.get('width')}x{v.get('height')}"
            sar = str(v.get("sample_aspect_ratio") or "")
            rot = 0
            tags = v.get("tags") or {}
            try:
                rot = int(round(float(tags.get("rotate")))) % 360
            except (TypeError, ValueError):
                for sd in v.get("side_data_list") or []:
                    if "rotation" in sd:
                        try:
                            rot = int(round(-float(sd["rotation"]))) % 360
                        except (TypeError, ValueError):
                            pass
                        break
            extra = []
            if sar and sar not in ("1:1", "0:1", "N/A"):
                extra.append(f"anamorphic SAR {sar}")
            if rot:
                extra.append(f"rotated {rot}°")
            if extra:
                res += "  (" + ", ".join(extra) + ")"
            out.append(("Resolution", res))
            try:
                w, h = int(v["width"]), int(v["height"])
                if sar and ":" in sar and sar not in ("1:1", "0:1"):
                    n, d = sar.split(":")
                    w = int(round(w * int(n) / max(1, int(d))))
                if rot in (90, 270):
                    w, h = h, w
                if (w, h) != (int(v["width"]), int(v["height"])) and h:
                    from math import gcd
                    g = gcd(w, h) or 1
                    out.append(("Display AR", "%d:%d (%.2f:1, shown %dx%d)" % (
                        w // g, h // g, w / h, w, h)))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        afr = str(v.get("avg_frame_rate") or v.get("r_frame_rate") or "")
        if "/" in afr:
            try:
                n, d = afr.split("/")
                if int(d):
                    out.append(("Frame Rate", f"{int(n) / int(d):.3f} fps"))
            except (ValueError, ZeroDivisionError):
                pass
        if v.get("pix_fmt"):
            out.append(("Pixel Format", v["pix_fmt"]))
        if v.get("bits_per_raw_sample"):
            out.append(("Bit Depth", f"{v['bits_per_raw_sample']}-bit"))

    if fmt.get("duration"):
        out.append(("Duration", _fmt_hms(fmt["duration"])))
    br = _safe_float(fmt.get("bit_rate")) if fmt.get("bit_rate") else None
    if br is not None:
        out.append(("Bit Rate", f"{br / 1000:.0f} kbps"))
    return out


def format_ffprobe(data: dict) -> str:
    """Pretty-format ffprobe JSON into readable text."""
    lines: "list[str]" = []

    frame_side_data: "dict[int, list]" = {}
    frames = data.get("frames", [])
    for frame in (frames if isinstance(frames, list) else []):
        if not isinstance(frame, dict):
            continue
        sidx = frame.get("stream_index", 0)
        sd = frame.get("side_data_list", [])
        if sd and isinstance(sd, list):
            frame_side_data.setdefault(sidx, []).extend(sd)

    if isinstance(data.get("format"), dict):
        fmt = data["format"]
        lines.append("═" * 62)
        lines.append("  CONTAINER / FORMAT")
        lines.append("═" * 62)
        for key in ("filename", "format_name", "format_long_name",
                    "duration", "size", "bit_rate", "nb_streams", "nb_programs"):
            if key in fmt:
                val = fmt[key]
                if key == "size":
                    sz = _safe_float(val)
                    if sz is not None:
                        val = f"{sz / (1024 * 1024):.2f} MB ({val} bytes)"
                elif key == "bit_rate":
                    br = _safe_float(val)
                    if br is not None:
                        val = f"{br / 1000:.1f} kbps"
                elif key == "duration":
                    sec = _safe_float(val)
                    if sec is not None:
                        val = f"{sec:.3f}s ({_fmt_hms(sec)})"
                lines.append(f"  {key:25s} : {val}")
        if isinstance(fmt.get("tags"), dict):
            lines.append("  ── tags ──")
            for k, v in fmt["tags"].items():
                lines.append(f"    {k:23s} : {v}")

    streams = data.get("streams", [])
    for i, s in enumerate(streams if isinstance(streams, list) else []):
        if not isinstance(s, dict):
            continue
        codec_type = str(s.get("codec_type", "unknown")).upper()
        lines.append("")
        lines.append("═" * 62)
        lines.append(f"  STREAM #{i}  —  {codec_type}")
        lines.append("═" * 62)

        display_keys = [
            "codec_name", "codec_long_name", "codec_type", "codec_tag_string",
            "profile", "level",
            "width", "height", "coded_width", "coded_height",
            "pix_fmt", "bits_per_raw_sample",
            "color_range", "color_space", "color_transfer", "color_primaries",
            "sample_rate", "channels", "channel_layout", "bits_per_sample",
            "r_frame_rate", "avg_frame_rate", "time_base",
            "duration", "duration_ts", "nb_frames",
            "bit_rate", "max_bit_rate",
            "sample_fmt", "sample_aspect_ratio", "display_aspect_ratio",
            "start_time", "start_pts", "field_order",
        ]
        for key in display_keys:
            if key in s and s[key] not in (None, "unknown", "N/A", "0/0"):
                val = s[key]
                if key == "bit_rate":
                    br = _safe_float(val)
                    if br is not None:
                        val = f"{br / 1000:.1f} kbps"
                lines.append(f"  {key:25s} : {val}")

        stream_side_data = s.get("side_data_list", [])
        if not isinstance(stream_side_data, list):
            stream_side_data = []
        all_side_data = stream_side_data + frame_side_data.get(i, [])

        if codec_type == "VIDEO":
            lines.append("")
            lines.append("  ┌─────────────────────────────────────────────────────┐")
            lines.append("  │              HDR / COLOR ANALYSIS                     │")
            lines.append("  └─────────────────────────────────────────────────────┘")
            hdr_info = classify_hdr(s, all_side_data if all_side_data else None)
            for k, v in hdr_info.items():
                if v == "":
                    lines.append(f"  {k}")
                else:
                    lines.append(f"  {k:30s} : {v}")

        if all_side_data:
            lines.append("")
            lines.append("  ── side_data (raw) ──")
            for sd in all_side_data:
                if not isinstance(sd, dict):
                    continue
                sd_type = sd.get("side_data_type", "unknown")
                lines.append(f"    [{sd_type}]")
                for k, v in sd.items():
                    if k != "side_data_type":
                        lines.append(f"      {k:23s} : {v}")

        disp = s.get("disposition", {})
        if not isinstance(disp, dict):
            disp = {}
        active = [k for k, v in disp.items() if v == 1]
        if active:
            lines.append(f"  {'disposition':25s} : {', '.join(active)}")

        if isinstance(s.get("tags"), dict):
            lines.append("  ── tags ──")
            for k, v in s["tags"].items():
                lines.append(f"    {k:23s} : {v}")

    return "\n".join(lines)


def format_opencv(data: dict) -> str:
    lines = ["═" * 62, "  STREAM INFO  (OpenCV — bundle ffprobe for full HDR detail)", "═" * 62]
    for k, v in data.items():
        lines.append(f"  {k:25s} : {v}")
    return "\n".join(lines)


