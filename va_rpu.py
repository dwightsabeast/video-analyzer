#!/usr/bin/env python3
"""
va_rpu - minimal Dolby Vision RPU reader (pure Python, no dovi_tool needed).

Reads the first RPU NAL (HEVC NAL type 62) straight out of the bitstream and
parses rpu_data_header -> rpu_data_mapping -> vdr_dm_data far enough to
recover everything the Profile 5 software decode needs:

    ycc_to_rgb_matrix / offset   (IPT -> PQ'd LMS, /2^13 and /2^28)
    rgb_to_lms_matrix            (crosstalk undo, /2^14)
    source_min_pq / source_max_pq
    per-component reshaping curves (polynomial + MMR, /2^coef_log2_denom)

Bit layout follows dovi_tool's dolby_vision crate (rpu_data_header.rs,
rpu_data_mapping.rs, vdr_dm_data.rs) and ffmpeg's dovi_rpudec.c field
scaling. Curves are returned in va_ipt's shape so the two plug together:
[{"lo", "hi", "pieces": [(plo, phi, "poly", [coef...], None) |
                         (plo, phi, "mmr", constant, ndarray(order, 7))]}]

This is the fallback for ffmpeg/ffprobe builds too old to expose
"Dolby Vision Metadata" frame side data (< 5.1); newer builds keep using
ffprobe (one subprocess, already JSON). Static: first RPU only, like the
ffprobe path - per-scene reshape drift is accepted and labelled approximate.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from va_ffmpeg import find_ffmpeg, CREATIONFLAGS


class _Bits:
    """MSB-first bit reader with ue(v)/se(v) Exp-Golomb."""

    def __init__(self, data: bytes):
        self.d = data
        self.pos = 0          # bit position

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.d[self.pos >> 3]
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def flag(self) -> bool:
        return bool(self.u(1))

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 32:
                raise ValueError("bad Exp-Golomb")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)


def _strip_ep(b: bytes) -> bytes:
    """Remove HEVC emulation-prevention bytes (00 00 03 xx -> 00 00 xx)."""
    out = bytearray()
    i, n = 0, len(b)
    while i < n:
        if i + 2 < n and b[i] == 0 and b[i + 1] == 0 and b[i + 2] == 3:
            out += b[i:i + 2]
            i += 3
        else:
            out.append(b[i])
            i += 1
    return bytes(out)


def _first_rpu_nal(path, max_bytes=6 << 20, timeout=30) -> "bytes | None":
    """First NAL of type 62 (UNSPEC62 = DV RPU) from the Annex-B HEVC stream.
    Reads only the head of the file - the first RPU arrives with frame 1."""
    exe = find_ffmpeg()
    if not exe:
        return None
    args = [exe, "-v", "error", "-i", path, "-map", "0:v:0", "-c:v", "copy",
            "-f", "hevc", "-"]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=CREATIONFLAGS)
        data = proc.stdout.read(max_bytes)
        proc.kill()
        proc.wait(timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    # scan Annex-B start codes for nal_unit_type 62
    i, n = 0, len(data)
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0 or j + 4 >= n:
            return None
        start = j + 3
        ntype = (data[start] >> 1) & 0x3F
        end = data.find(b"\x00\x00\x01", start)
        if end < 0:
            end = n
        elif end > start and data[end - 1] == 0:   # 4-byte start code
            end -= 1
        if ntype == 62:
            return _strip_ep(data[start + 2:end])  # skip 2-byte NAL header
        i = start


def _read_coef(bits, denom_len: int) -> float:
    """One RPU coefficient: se(v) integer part + u(denom_len) fraction.
    value = (int_part << denom_len | frac) / 2^denom_len (ffmpeg dovi_rpudec)."""
    ipart = bits.se()
    frac = bits.u(denom_len)
    return float((ipart << denom_len) + frac) / float(1 << denom_len)


def parse_rpu(payload: bytes) -> "dict | None":
    """Parse one RPU payload (emulation-prevention already stripped, starting
    at the 0x19 prefix byte) -> va_ipt-shaped meta dict, or None."""
    if not payload or payload[0] != 0x19:
        return None
    bits = _Bits(payload[1:])

    # ---- rpu_data_header ----------------------------------------------------
    if bits.u(6) != 2:                      # rpu_type
        return None
    rpu_format = bits.u(11)
    bits.u(4)                               # vdr_rpu_profile
    bits.u(4)                               # vdr_rpu_level
    denom = 23
    bl_bit_depth = 10
    reserved3 = 0
    disable_residual = True
    if bits.flag():                         # vdr_seq_info_present
        bits.flag()                         # chroma_resampling_explicit_filter
        coef_type = bits.u(2)
        if coef_type == 0:
            denom = bits.ue()
        denom_len = denom if coef_type == 0 else 32
        if coef_type > 1:
            return None
        bits.u(2)                           # vdr_rpu_normalized_idc
        bits.flag()                         # bl_video_full_range
        if rpu_format & 0x700 == 0:
            bl_bit_depth = bits.ue() + 8
            bits.ue()                       # el_bit_depth (+ ext_mapping_idc)
            bits.ue()                       # vdr_bit_depth
            bits.flag()                     # spatial_resampling_filter
            reserved3 = bits.u(3)
            bits.flag()                     # el_spatial_resampling_filter
            disable_residual = bits.flag()
    else:
        denom_len = denom
    dm_present = bits.flag()
    if bits.flag():                         # use_prev_vdr_rpu
        bits.ue()                           # prev_vdr_rpu_id - no payload follows
        return None
    if not dm_present:
        return None

    # ---- rpu_data_mapping ---------------------------------------------------
    bits.ue()                               # vdr_rpu_id
    bits.ue()                               # mapping_color_space
    bits.ue()                               # mapping_chroma_format_idc
    pivots_all = []
    for _c in range(3):
        npiv = bits.ue() + 2
        pivots_all.append([bits.u(bl_bit_depth) for _ in range(npiv)])
    if rpu_format & 0x700 == 0 and not disable_residual:    # profile 7 NLQ
        if bits.u(3) != 0:
            return None
        for _ in range(2):                  # NLQ_NUM_PIVOTS = 2
            bits.u(bl_bit_depth)
        has_nlq = True
    else:
        has_nlq = False
    bits.ue()                               # num_x_partitions_minus1
    bits.ue()                               # num_y_partitions_minus1

    bl_max = float((1 << bl_bit_depth) - 1)
    curves = []
    for c in range(3):
        piv = [p / bl_max for p in pivots_all[c]]
        pieces = []
        for i in range(len(piv) - 1):
            idc = bits.ue()
            if idc == 0:                    # polynomial
                order = bits.ue() + 1       # poly_order_minus1 + 1
                if order - 1 == 0 and bits.flag():
                    return None             # linear_interp: unsupported (rare)
                coef = [_read_coef(bits, denom_len) for _ in range(order + 1)]
                pieces.append((piv[i], piv[i + 1], "poly", coef, None))
            elif idc == 1:                  # MMR
                order = bits.u(2) + 1
                const = _read_coef(bits, denom_len)
                rows = []
                for _j in range(order):
                    rows.append([_read_coef(bits, denom_len) for _ in range(7)])
                pieces.append((piv[i], piv[i + 1], "mmr", const,
                               np.array(rows, dtype=np.float64)))
            else:
                return None
        curves.append({"lo": piv[0], "hi": piv[-1], "pieces": pieces})

    if has_nlq:                             # profile 7: skip NLQ payload
        for _p in range(1):                 # nlq_num_pivots_minus2 == 0 -> 1 piece
            for _c in range(3):
                bits.ue()                   # num_nlq_param_predictors? (simplified)
        return None                         # P7 EL streams: matrices still follow,
                                            # but NLQ skip here is approximate - bail

    # ---- vdr_dm_data_payload ------------------------------------------------
    if reserved3 == 1:                      # compressed DM data - no matrices
        return None
    bits.ue()                               # affected_dm_metadata_id
    bits.ue()                               # current_dm_metadata_id
    bits.ue()                               # scene_refresh_flag
    def _i16(v):
        return v - 65536 if v >= 32768 else v
    ycc = [_i16(bits.u(16)) / 8192.0 for _ in range(9)]
    off = [bits.u(32) / 268435456.0 for _ in range(3)]
    lms = [_i16(bits.u(16)) / 16384.0 for _ in range(9)]
    bits.u(16)                              # signal_eotf
    bits.u(16); bits.u(16); bits.u(32)      # signal_eotf_param0/1/2
    bits.u(5)                               # signal_bit_depth
    bits.u(2)                               # signal_color_space
    bits.u(2)                               # signal_chroma_format
    bits.u(2)                               # signal_full_range_flag
    src_min_pq = bits.u(12)
    src_max_pq = bits.u(12)
    bits.u(10)                              # source_diagonal

    identity = all(len(c["pieces"]) == 1 and c["pieces"][0][2] == "poly"
                   and c["pieces"][0][3][:2] in ([0.0, 1.0],)
                   and all(abs(x) < 1e-9 for x in c["pieces"][0][3][2:])
                   for c in curves)
    return {
        "nonlinear": [ycc[0:3], ycc[3:6], ycc[6:9]],
        "offset": off,
        "linear": [lms[0:3], lms[3:6], lms[6:9]],
        "source_min_pq": src_min_pq,
        "source_max_pq": src_max_pq or 3696,
        "curves": None if identity else curves,
        "exact": True,
        "via": "rpu-parse",
    }


def dovi_meta_from_stream(path) -> "dict | None":
    """First-RPU Dolby Vision colour metadata read directly from the file.
    None when there is no RPU NAL or the payload shape is unsupported."""
    nal = _first_rpu_nal(path)
    if not nal:
        return None
    try:
        return parse_rpu(nal)
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    import json
    import sys
    m = dovi_meta_from_stream(sys.argv[1])
    if not m:
        print("no RPU metadata found")
        sys.exit(1)
    show = dict(m, curves="%d component curve(s)" % len(m["curves"] or []))
    print(json.dumps(show, indent=2, default=str))
