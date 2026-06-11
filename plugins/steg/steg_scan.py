"""Orchestrator: run all hidden-data layers and fuse into one report.

  container  - always (appended data, polyglots, padding atoms, mdat gaps)
  frame LSB  - only on a lossless 8-bit source (pixel LSB can't survive lossy
               coding); samples frames at native resolution
  codec      - SEI/user-data NAL scan + frame-size-vs-complexity residual

Returns {ok, stats, findings[], entropy{}, residual{}, gated{}}."""

from __future__ import annotations

import numpy as np

import va_ffmpeg
import steg_container
import steg_bitplane
import steg_codec

LOSSLESS = {"ffv1", "huffyuv", "ffvhuff", "utvideo", "magicyuv", "rawvideo",
            "qtrle", "msrle", "png", "apng", "tiff", "v210", "r210", "ljpeg",
            "gif"}
H26X = {"h264", "avc", "h265", "hevc"}
LSB_FRAMES = 40          # frames sampled for LSB (native res)
CX_FRAMES = 200          # frames sampled (downscaled) for the size residual


def _luma_native(path, nframes):
    """Sample up to nframes at native resolution; yields (index, luma float)."""
    vs = va_ffmpeg.VideoSource(path, decode_max=10000)   # no downscale = real LSBs
    try:
        total = vs.nb_frames or 0
        if total <= 0:
            vs.start(0)
            idxs = range(nframes)
        else:
            step = max(1, total // nframes)
            idxs = range(0, total, step)
        for i in idxs:
            fr = vs.frame_at(i) if total > 0 else vs.read()
            if fr is None:
                break
            g = np.asarray(fr)
            yield i, (g.mean(axis=2) if g.ndim == 3 else g).astype(np.float32)
    finally:
        vs.close()


def _complexity(path):
    """(indices, sizes, complexity) sampled at downscale for the size residual.
    complexity = spatial detail + temporal change."""
    import va_metrics
    fs = va_metrics.frame_sizes(path)
    sizes_all = fs.get("size") or []
    if len(sizes_all) < 16:
        return [], [], []
    vs = va_ffmpeg.VideoSource(path, decode_max=200)
    idxs, sizes, cx = [], [], []
    try:
        total = vs.nb_frames or len(sizes_all)
        step = max(1, total // CX_FRAMES)
        prev = None
        for i in range(0, total, step):
            fr = vs.frame_at(i)
            if fr is None:
                break
            g = np.asarray(fr)
            g = (g.mean(axis=2) if g.ndim == 3 else g).astype(np.float32)
            spatial = float(np.abs(np.diff(g, axis=1)).mean()
                            + np.abs(np.diff(g, axis=0)).mean())
            temporal = 0.0 if prev is None or prev.shape != g.shape \
                else float(np.abs(g - prev).mean())
            prev = g
            if i < len(sizes_all):
                idxs.append(i)
                sizes.append(sizes_all[i])
                cx.append(spatial + temporal)
    finally:
        vs.close()
    return idxs, sizes, cx


def scan(path, force_lsb=False, on_progress=None, cancel=None) -> dict:
    say = on_progress or (lambda m: None)
    info = va_ffmpeg.probe(path) or {}
    codec = str(info.get("codec") or "").lower()
    depth = int(info.get("bit_depth") or 8)
    findings, gated = [], {}

    say("container forensics")
    cont = steg_container.scan(path, on_progress=say, cancel=cancel)
    findings += cont.get("findings", [])
    stats = {"container": cont.get("stats", {}), "codec": codec, "bit_depth": depth}
    entropy = cont.get("entropy", {})

    lossless = codec in LOSSLESS
    lsb = None
    if lossless or force_lsb:
        if depth != 8:
            gated["lsb"] = "skipped: %d-bit source (LSB analysis assumes 8-bit)" % depth
        else:
            say("LSB steganalysis on sampled frames")
            rates, susp = [], 0
            nseen = 0
            for _i, luma in _luma_native(path, LSB_FRAMES):
                if cancel is not None and cancel():
                    break
                a = steg_bitplane.analyze_channel(luma)
                rates.append(a["rs"]["rate"])
                susp += 1 if a["suspicious"] else 0
                nseen += 1
            if nseen:
                lsb = {"frames": nseen, "suspicious_frames": susp,
                       "max_rate": round(float(np.max(rates)), 3),
                       "mean_rate": round(float(np.mean(rates)), 3)}
                if susp > max(1, nseen // 10):
                    findings.append({"kind": "lsb", "severity": "warn",
                                     "offset": 0, "size": 0,
                                     "text": "LSB anomaly on %d/%d sampled frames "
                                             "(max RS rate %.2f) - possible "
                                             "pixel-LSB payload" % (susp, nseen,
                                                                    lsb["max_rate"])})
    else:
        gated["lsb"] = ("skipped: lossy codec '%s' - pixel-LSB stego cannot "
                        "survive lossy coding (use force to scan anyway)" % codec)

    say("compressed-domain indicators")
    codec_rep = {"findings": []}
    if codec in H26X:
        sei = steg_codec.scan_sei(path, codec=codec)
        codec_rep["sei"] = sei
        if sei.get("ok") and sei.get("user_data_bytes", 0) > 2048:
            findings.append({"kind": "sei_userdata", "severity": "warn",
                             "offset": 0, "size": sei["user_data_bytes"],
                             "text": "%d bytes of user-data SEI across %d message"
                                     "(s) - well beyond a normal encoder tag"
                                     % (sei["user_data_bytes"],
                                        sei["user_data_count"])})
    idxs, sizes, cx = _complexity(path)
    residual = {}
    if idxs:
        sz = steg_codec.analyze_sizes(sizes, cx, z=6.0)
        codec_rep["sizes"] = sz
        residual = {"t": [i / (info.get("fps") or 30.0) for i in idxs],
                    "v": sz["residual_z"]}
        if sz["anomaly_frames"]:
            real = [idxs[i] for i in sz["anomaly_frames"] if i < len(idxs)]
            findings.append({"kind": "size_anomaly", "severity": "info",
                             "offset": 0, "size": 0,
                             "text": "%d frame(s) coded far larger than their "
                                     "visual complexity (max %.1f MAD) - possible "
                                     "coefficient/MV embedding"
                                     % (len(real), sz["max_z"])})

    warns = sum(1 for f in findings if f["severity"] == "warn")
    return {"ok": True, "findings": findings, "stats": stats, "gated": gated,
            "lsb": lsb, "codec": codec_rep, "entropy": entropy,
            "residual": residual, "warn": warns}


def marks(rep) -> list:
    if not rep or not rep.get("ok"):
        return []
    out = []
    for f in rep["findings"]:
        if f["kind"] in ("appended_data", "embedded_file", "padding_atom",
                         "mdat_gap", "sei_userdata"):
            # offset-based findings have no timecode; pin them at t=0 marker
            out.append((0.0, f["kind"]))
    return out[:1]        # one container marker; frame/size hits aren't time-mapped here


def render_report(rep) -> str:
    if not rep.get("ok"):
        return "Hidden-data scan failed: %s" % rep.get("error", "?")
    st = rep["stats"]
    lines = ["HIDDEN-DATA / STEGANOGRAPHY SCAN", "",
             "container: %s   codec: %s   depth: %d-bit"
             % (st["container"].get("container", "?"), st["codec"], st["bit_depth"]),
             "file size: %d bytes   entropy mean %.2f / max %.2f bits/byte"
             % (st["container"].get("file_size", 0),
                st["container"].get("entropy_mean", 0),
                st["container"].get("entropy_max", 0)), ""]
    if rep.get("lsb"):
        l = rep["lsb"]
        lines.append("frame LSB: %d frames sampled, %d suspicious, max RS rate %.2f"
                     % (l["frames"], l["suspicious_frames"], l["max_rate"]))
    for k, why in rep.get("gated", {}).items():
        lines.append("%s: %s" % (k, why))
    sei = (rep.get("codec") or {}).get("sei") or {}
    if sei.get("ok"):
        lines.append("SEI: %d message(s), %d user-data bytes"
                     % (sei["sei_count"], sei["user_data_bytes"]))
    lines.append("")
    if not rep["findings"]:
        lines.append("No hidden-data indicators found.")
    for f in rep["findings"]:
        lines.append("[%s] %-14s %s" % (f["severity"].upper(), f["kind"], f["text"]))
    lines += ["", "Container findings (appended data, polyglots, stuffed atoms) "
                  "are reliable. LSB needs a lossless source and over-reads on "
                  "noisy content. SEI user-data and size anomalies are "
                  "indicators, not proof."]
    return "\n".join(lines)


# --- QC checks for the "steg" profile ------------------------------------------

def _rep(ctx):
    return (ctx.get("steg") or (ctx.get("adv") or {}).get("steg")
            or ctx.get("verify") or {})


def _has(ctx, kind):
    return [f for f in (_rep(ctx).get("findings") or []) if f["kind"] == kind]


def chk_appended():
    def f(ctx):
        if not _rep(ctx):
            return "pass", "not measured (run the hidden-data scan)"
        h = _has(ctx, "appended_data")
        return ("warn", h[0]["text"]) if h else ("pass", "no trailing data")
    return ("steg_appended", "Appended data", f)


def chk_embedded():
    def f(ctx):
        if not _rep(ctx):
            return "pass", "not measured"
        h = [x for x in _rep(ctx).get("findings", [])
             if x["kind"] == "embedded_file" and x["severity"] == "warn"]
        return ("warn", "%d embedded-file signature(s)" % len(h)) if h \
            else ("pass", "no embedded archives/documents")
    return ("steg_embedded", "Embedded files", f)


def chk_padding():
    def f(ctx):
        if not _rep(ctx):
            return "pass", "not measured"
        h = _has(ctx, "padding_atom")
        return ("warn", h[0]["text"]) if h else ("pass", "padding atoms clean")
    return ("steg_padding", "Padding atoms", f)


def chk_mdat_gap():
    def f(ctx):
        if not _rep(ctx):
            return "pass", "not measured"
        h = _has(ctx, "mdat_gap")
        return ("warn", h[0]["text"]) if h else ("pass", "no unreferenced media data")
    return ("steg_mdat", "Unreferenced media data", f)


def chk_lsb():
    def f(ctx):
        rep = _rep(ctx)
        if not rep:
            return "pass", "not measured"
        if rep.get("lsb") is None:
            return "pass", rep.get("gated", {}).get("lsb", "LSB not applicable")
        h = _has(ctx, "lsb")
        return ("warn", h[0]["text"]) if h else (
            "pass", "frame LSBs consistent with clean content")
    return ("steg_lsb", "Pixel LSB", f)


def chk_sei():
    def f(ctx):
        if not _rep(ctx):
            return "pass", "not measured"
        h = _has(ctx, "sei_userdata")
        return ("warn", h[0]["text"]) if h else ("pass", "no oversized SEI user-data")
    return ("steg_sei", "SEI user-data", f)


def qc_checks():
    return [chk_appended(), chk_embedded(), chk_padding(), chk_mdat_gap(),
            chk_lsb(), chk_sei()]
