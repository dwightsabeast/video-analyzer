"""Compressed-domain hidden-data indicators (best-effort).

True coefficient/motion-vector steganalysis needs full entropy decoding of the
bitstream, which this does NOT do. Two grounded proxies instead:

  * SEI / user-data NAL scan - H.264 and HEVC carry optional SEI NAL units, and
    `user_data_unregistered` (payload type 5: a 16-byte UUID + arbitrary bytes)
    is a documented, common place to stash a payload or watermark. We demux the
    Annex-B elementary stream and parse SEI NALs directly (no full decode),
    reporting count and total size. A few hundred bytes (an x264 version string)
    is normal; kilobytes of user-data is not;
  * frame-size vs complexity residual - data hidden by inflating the compressed
    stream (QDCT/MV embedding) makes coded frames larger than their visual
    complexity warrants. We regress coded size on a complexity proxy and flag
    frames with a large positive residual.

Indicators, not proof. numpy only for the math; ffmpeg demux for the NAL scan.
The pure helpers (parse_sei, size_residuals) are unit-tested directly."""

from __future__ import annotations

import subprocess

import numpy as np

from va_ffmpeg import find_ffmpeg

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if hasattr(
    subprocess, "CREATE_NO_WINDOW") else 0


def _strip_ep(b: bytes) -> bytes:
    """Remove emulation-prevention bytes (00 00 03 xx -> 00 00 xx)."""
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


def _nal_split(stream: bytes):
    """Yield (nal_bytes) between Annex-B start codes (00 00 01 / 00 00 00 01)."""
    idx = []
    i, n = 0, len(stream)
    while i + 3 <= n:
        if stream[i] == 0 and stream[i + 1] == 0 and stream[i + 2] == 1:
            idx.append(i + 3)
            i += 3
        else:
            i += 1
    for k, start in enumerate(idx):
        end = (idx[k + 1] - 3 if k + 1 < len(idx) else len(stream))
        # trim a trailing 00 that belongs to the next start code
        while end > start and stream[end - 1] == 0:
            end -= 1
        yield stream[start:end]


def _parse_sei_payloads(rbsp: bytes):
    """Walk SEI message(s) in an (already EP-stripped) SEI RBSP. Yields
    (payload_type, payload_size)."""
    i, n = 0, len(rbsp)
    while i < n:
        ptype = 0
        while i < n and rbsp[i] == 0xFF:
            ptype += 255
            i += 1
        if i >= n:
            break
        ptype += rbsp[i]
        i += 1
        psize = 0
        while i < n and rbsp[i] == 0xFF:
            psize += 255
            i += 1
        if i >= n:
            break
        psize += rbsp[i]
        i += 1
        yield ptype, psize
        i += psize
        if rbsp[i:i + 1] == b"\x80":            # rbsp_trailing
            break


def parse_sei(stream: bytes, codec="h264") -> list:
    """All SEI messages in an Annex-B stream: [{type, size, nal_offset}]."""
    out = []
    is_hevc = codec in ("hevc", "h265")
    for nal in _nal_split(stream):
        if not nal:
            continue
        if is_hevc:
            nal_type = (nal[0] >> 1) & 0x3F
            if nal_type not in (39, 40):        # PREFIX_SEI / SUFFIX_SEI
                continue
            body = nal[2:]
        else:
            nal_type = nal[0] & 0x1F
            if nal_type != 6:                   # SEI
                continue
            body = nal[1:]
        for ptype, psize in _parse_sei_payloads(_strip_ep(body)):
            out.append({"type": int(ptype), "size": int(psize)})
    return out


def scan_sei(path, codec="h264", max_bytes=24 << 20, timeout=120) -> dict:
    """Demux the Annex-B elementary stream and tally SEI messages, with special
    attention to user_data_unregistered (type 5)."""
    exe = find_ffmpeg()
    if not exe:
        return {"ok": False, "error": "ffmpeg required"}
    bsf = "hevc_mp4toannexb" if codec in ("hevc", "h265") else "h264_mp4toannexb"
    fmt = "hevc" if codec in ("hevc", "h265") else "h264"
    try:
        p = subprocess.Popen(
            [exe, "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
             "-c:v", "copy", "-bsf:v", bsf, "-f", fmt, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=CREATIONFLAGS)
        data = p.stdout.read(max_bytes)
        p.stdout.close()
        p.kill()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": "demux failed: %s" % exc}
    if not data:
        return {"ok": False, "error": "no elementary stream (not H.264/HEVC?)"}
    seis = parse_sei(data, codec=codec)
    user = [s for s in seis if s["type"] == 5]
    user_bytes = sum(s["size"] for s in user)
    return {"ok": True, "sei_count": len(seis), "user_data_count": len(user),
            "user_data_bytes": int(user_bytes),
            "types": sorted({s["type"] for s in seis}),
            "bytes_scanned": len(data)}


def size_residuals(sizes, complexity):
    """Robust-regress coded size on a complexity proxy; return standardized
    positive residuals (how much bigger each frame is than its complexity
    predicts, in MAD units)."""
    s = np.asarray(sizes, np.float64)
    x = np.asarray(complexity, np.float64)
    n = min(s.size, x.size)
    if n < 16:
        return np.zeros(n)
    s, x = s[:n], x[:n]
    # least-squares line size ~ a*complexity + b
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9
    return (resid - med) / (1.4826 * mad)


def analyze_sizes(frame_sizes, complexity, z=6.0) -> dict:
    """Flag intra/predicted frames whose coded size is anomalously high for
    their complexity. frame_sizes/complexity are per-frame parallel arrays."""
    zr = size_residuals(frame_sizes, complexity)
    hits = [int(i) for i in np.flatnonzero(zr > z)]
    return {"residual_z": zr.tolist(), "anomaly_frames": hits[:200],
            "max_z": round(float(zr.max()), 2) if zr.size else 0.0}


def render_report(rep: dict) -> str:
    lines = ["COMPRESSED-DOMAIN INDICATORS (best-effort - not full bitstream "
             "steganalysis)", ""]
    sei = rep.get("sei") or {}
    if sei.get("ok"):
        lines.append("SEI messages       : %d (payload types %s)"
                     % (sei["sei_count"], sei.get("types")))
        lines.append("user-data SEI      : %d message(s), %d bytes total"
                     % (sei["user_data_count"], sei["user_data_bytes"]))
    elif sei:
        lines.append("SEI scan           : %s" % sei.get("error"))
    sz = rep.get("sizes") or {}
    if sz:
        lines.append("frame-size anomalies: %d frame(s) over threshold "
                     "(max %.1f MAD)" % (len(sz.get("anomaly_frames", [])),
                                         sz.get("max_z", 0.0)))
    lines.append("")
    for f in rep.get("findings", []):
        lines.append("[%s] %s" % (f["severity"].upper(), f["text"]))
    if not rep.get("findings"):
        lines.append("No compressed-domain anomalies surfaced.")
    lines += ["", "SEI user-data is a documented carrier (UUID + bytes); a few "
                  "hundred bytes is a normal encoder tag, kilobytes is not. The "
                  "size residual is a coarse proxy for coefficient/MV embedding "
                  "and also rises for genuinely complex frames."]
    return "\n".join(lines)
