#!/usr/bin/env python3
"""
va_ipt - Dolby Vision Profile 5 (IPTPQc2) software decode.

DV Profile 5 stores pixels in Dolby's IPT colour space inside an ordinary
yuv420p10 stream with NO colour tags (no valid VUI exists for IPT). Decoded
as plain YCbCr it shows the classic magenta/violet wash. The first-choice
path is ffmpeg's libplacebo filter (applies the per-frame RPU exactly); this
module is the zero-extra-dependency fallback so the preview still looks right
with ANY ffmpeg build: numpy + the file's own colour metadata (read once via
ffprobe) reproduce libplacebo's decode chain. Per-scene reshaping curves are
applied statically from the first frame, so the result is labelled
"approximate" - tone/saturation can drift slightly across scene changes.

Decode chain (mirrors libplacebo src/shaders/{dolbyvision,colorspace}.c,
PL_COLOR_SYSTEM_DOLBYVISION):

    sig   = code / 1023                       (DV base layer is full range)
    sig   = reshape(sig)                      (RPU poly / MMR curves)
    lms'  = ycc_to_rgb_matrix @ (sig - ycc_to_rgb_offset)
    lms   = PQ_EOTF(lms')                     (linear light, 1.0 = 10^4 nits)
    rgb   = (DOVI_LMS2RGB @ rgb_to_lms_matrix) @ lms     (linear BT.2020)
    rgb   = hable_tonemap(rgb, peak)          (peak from RPU source_max_pq)
    bgr24 = BT709_OETF(BT2020_TO_BT709 @ rgb)

References: libplacebo colorspace.c / dolbyvision.c (decode + hard-coded
LMS->RGB), ffmpeg libavcodec/dovi_rpudec.c (field scaling: matrices /2^13 and
/2^14, offsets /2^28, curve coefficients /2^coef_log2_denom, pivots are
bl_bit_depth code values), fftools/ffprobe.c print_dovi_metadata (JSON keys).
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dep of the GUI
    cv2 = None

from va_ffmpeg import ffprobe_json

# libplacebo: "Dolby Vision always outputs BT.2020-referred HPE LMS" - this is
# its hard-coded LMS->RGB finisher, composed with the RPU's rgb_to_lms matrix.
DOVI_LMS2RGB = np.array([
    [3.06441879, -2.16597676,  0.10155818],
    [-0.65612108,  1.78554118, -0.12943749],
    [0.01736321, -0.04725154,  1.03004253]], dtype=np.float64)

BT2020_TO_BT709 = np.array([
    [1.6604910, -0.5876411, -0.0728499],
    [-0.1245505,  1.1328999, -0.0083494],
    [-0.0181508, -0.1005789,  1.1187297]], dtype=np.float64)

# SMPTE ST 2084 (PQ) constants
_M1 = 2610.0 / 16384.0
_M2 = 2523.0 / 4096.0 * 128.0
_C1 = 3424.0 / 4096.0
_C2 = 2413.0 / 4096.0 * 32.0
_C3 = 2392.0 / 4096.0 * 32.0

# Last-resort colour metadata if the file's RPU cannot be read at all: the
# canonical Profile 5 IPTPQc2 constants (dovi_tool profiles/profile5.rs;
# ycc /2^13, offsets /2^28, rgb_to_lms /2^14 - the 2% crosstalk inverse).
# Every real P5 stream observed carries exactly these, but per-file values
# from the RPU/ffprobe still take precedence. NOT the BT.2100 ICtCp matrix -
# decoding real P5 with ICtCp swings greens to magenta (overnight 2026-06-10).
_FALLBACK_META = {
    "nonlinear": [[1.0, 799 / 8192.0, 1681 / 8192.0],
                  [1.0, -933 / 8192.0, 1091 / 8192.0],
                  [1.0, 267 / 8192.0, -5545 / 8192.0]],
    "offset": [0.0, 0.5, 0.5],
    "linear": [[17081 / 16384.0, -349 / 16384.0, -349 / 16384.0],
               [-349 / 16384.0, 17081 / 16384.0, -349 / 16384.0],
               [-349 / 16384.0, -349 / 16384.0, 17081 / 16384.0]],
    "source_max_pq": 3696,          # ffmpeg's default (~2000 nits)
    "curves": None,
    "exact": False,
}

_META_CACHE: dict = {}


def pq_eotf(e):
    """PQ signal (0..1) -> linear light, 1.0 = 10,000 nits."""
    e = np.clip(e, 0.0, 1.0)
    ep = np.power(e, 1.0 / _M2)
    num = np.clip(ep - _C1, 0.0, None)
    den = np.clip(_C2 - _C3 * ep, 1e-7, None)
    return np.power(num / den, 1.0 / _M1)


def pq_oetf(x):
    """Linear light (1.0 = 10,000 nits) -> PQ signal (0..1)."""
    x = np.clip(x, 0.0, None)
    xp = np.power(x, _M1)
    return np.power((_C1 + _C2 * xp) / (1.0 + _C3 * xp), _M2)


def bt709_oetf(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099)


def _hable(x):
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def _fracs(val, n):
    """ffprobe prints rational lists as 'num/den num/den ...' - parse n of them."""
    if isinstance(val, (list, tuple)):
        parts = [str(p) for p in val]
    else:
        parts = str(val or "").split()
    out = []
    for p in parts[:n]:
        try:
            if "/" in p:
                num, den = p.split("/")
                out.append(float(num) / (float(den) or 1.0))
            else:
                out.append(float(p))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return out if len(out) == n else None


def _ints(val):
    if isinstance(val, (list, tuple)):
        parts = [str(p) for p in val]
    else:
        parts = str(val or "").split()
    try:
        return [int(float(p)) for p in parts]
    except (TypeError, ValueError):
        return []


def _parse_curves(sd):
    """RPU reshaping curves from ffprobe's component list (None = identity).
    Coefficients are fixed-point with denominator 2^coef_log2_denom; pivots
    are raw bl_bit_depth code values."""
    comps = sd.get("components") or sd.get("component_list")
    if not isinstance(comps, list) or len(comps) < 3:
        return None
    denom = float(1 << int(sd.get("coef_log2_denom") or 23))
    bl_max = float((1 << int(sd.get("bl_bit_depth") or 10)) - 1)
    curves = []
    try:
        for comp in comps[:3]:
            pivots = [p / bl_max for p in _ints(comp.get("pivots"))]
            pieces_in = comp.get("pieces") or comp.get("piece_list") or []
            if len(pivots) < 2 or len(pieces_in) != len(pivots) - 1:
                return None
            pieces = []
            for i, pc in enumerate(pieces_in):
                kind = str(pc.get("mapping_idc_name") or "")
                if not kind:
                    kind = "polynomial" if str(pc.get("mapping_idc")) == "0" else "mmr"
                if kind == "polynomial":
                    coef = [c / denom for c in _ints(pc.get("poly_coef"))]
                    if not coef:
                        return None
                    pieces.append((pivots[i], pivots[i + 1], "poly", coef, None))
                else:
                    order = int(pc.get("mmr_order") or 0)
                    const = float(pc.get("mmr_constant") or 0) / denom
                    coef = [c / denom for c in _ints(pc.get("mmr_coef"))]
                    if order < 1 or order > 3 or len(coef) != order * 7:
                        return None
                    mmr = np.array(coef, dtype=np.float64).reshape(order, 7)
                    pieces.append((pivots[i], pivots[i + 1], "mmr", const, mmr))
            curves.append({"lo": pivots[0], "hi": pivots[-1], "pieces": pieces})
    except (TypeError, ValueError):
        return None
    return curves


def dovi_meta(path) -> "dict | None":
    """Per-file Dolby Vision colour metadata: ffprobe frame side data first
    (ffmpeg >= 5.1), then a direct RPU-NAL parse (va_rpu) for older builds.
    None only when neither path can read an RPU."""
    if path in _META_CACHE:
        return _META_CACHE[path]
    meta = None
    data = ffprobe_json(path, frames=True)
    for fr in (data or {}).get("frames") or []:
        for sd in fr.get("side_data_list") or []:
            t = str(sd.get("side_data_type", "")).lower()
            if "dolby vision metadata" not in t:
                continue
            mat = _fracs(sd.get("ycc_to_rgb_matrix"), 9)
            off = _fracs(sd.get("ycc_to_rgb_offset"), 3)
            lms = _fracs(sd.get("rgb_to_lms_matrix"), 9)
            if not (mat and off and lms):
                continue
            meta = {
                "nonlinear": [mat[0:3], mat[3:6], mat[6:9]],
                "offset": off,
                "linear": [lms[0:3], lms[3:6], lms[6:9]],
                "source_max_pq": int(sd.get("source_max_pq") or 3696),
                "curves": _parse_curves(sd),
                "exact": True,
            }
            break
        if meta:
            break
    if meta is None:                # old ffprobe - read the RPU NAL directly
        try:
            import va_rpu
            meta = va_rpu.dovi_meta_from_stream(path)
        except Exception:   # noqa: BLE001 - fallback must never break decode
            meta = None
    _META_CACHE[path] = meta
    return meta


def reset_cache():
    _META_CACHE.clear()


class IPTDecoder:
    """Streamed yuv420p10le (IPT) frames -> display BGR, one frame at a time."""

    def __init__(self, path, npl=100.0):
        meta = dovi_meta(path) or _FALLBACK_META
        self._npl = float(npl)
        self.exact = bool(meta.get("exact"))
        self.curves = meta.get("curves")
        self._m_ycc = np.array(meta["nonlinear"], dtype=np.float32)
        self._off = np.array(meta["offset"], dtype=np.float32)
        self._m_post = (DOVI_LMS2RGB @ np.array(meta["linear"], dtype=np.float64)
                        ).astype(np.float32)
        self._m_709 = BT2020_TO_BT709.astype(np.float32)
        # 1.0 after the EOTF LUT = npl (SDR reference white, like zscale npl=100)
        self._eotf_lut = (pq_eotf(np.linspace(0.0, 1.0, 4096)) *
                          (10000.0 / npl)).astype(np.float32)
        peak_nits = float(pq_eotf(min(max(int(meta.get("source_max_pq") or 3696),
                                          0), 4095) / 4095.0) * 10000.0)
        self._peak = max(1.0, min(peak_nits, 10000.0) / npl)
        self._hable_peak = float(_hable(np.float64(self._peak)))
        out = bt709_oetf(np.linspace(0.0, 1.0, 2048))
        self._oetf_lut = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # -- reshaping ------------------------------------------------------------

    def _reshape(self, sig):
        if not self.curves:
            return sig
        src = sig.copy()
        s = (src[..., 0], src[..., 1], src[..., 2])
        for c, curve in enumerate(self.curves):
            pieces = curve["pieces"]
            x = np.clip(src[..., c], curve["lo"], curve["hi"])
            res = x if len(pieces) == 1 else x.copy()
            for i, (plo, phi, kind, a, b) in enumerate(pieces):
                if len(pieces) == 1:
                    m = slice(None)
                    xv = x
                else:
                    m = (x >= plo) & ((x < phi) if i < len(pieces) - 1 else (x <= phi))
                    if not m.any():
                        continue
                    xv = x[m]
                if kind == "poly":
                    r = np.zeros_like(xv)
                    for coef in reversed(a):
                        r = r * xv + np.float32(coef)
                else:  # mmr - terms s0,s1,s2,s0s1,s0s2,s1s2,s0s1s2 per order
                    t = [s[0] if m == slice(None) else s[0][m],
                         s[1] if m == slice(None) else s[1][m],
                         s[2] if m == slice(None) else s[2][m]]
                    base = [t[0], t[1], t[2], t[0] * t[1], t[0] * t[2],
                            t[1] * t[2], t[0] * t[1] * t[2]]
                    r = np.full_like(xv, np.float32(a))
                    cur = base
                    for o in range(b.shape[0]):
                        if o > 0:
                            cur = [cb * bb for cb, bb in zip(cur, base)]
                        for k in range(7):
                            r += np.float32(b[o, k]) * cur[k]
                if m == slice(None):
                    res = r
                else:
                    res[m] = r
            src[..., c] = np.clip(res, curve["lo"], curve["hi"])
        return src

    # -- frame decode ---------------------------------------------------------

    def _linear2020(self, buf, w, h):
        """yuv420p10le bytes -> linear BT.2020 RGB float32, 1.0 = npl nits."""
        cw, ch = w // 2, h // 2
        plane = np.frombuffer(buf, dtype="<u2")
        y = plane[:w * h].reshape(h, w).astype(np.float32)
        u = plane[w * h:w * h + cw * ch].reshape(ch, cw).astype(np.float32)
        v = plane[w * h + cw * ch:w * h + 2 * cw * ch].reshape(ch, cw).astype(np.float32)
        inv = np.float32(1.0 / 1023.0)
        sig = np.empty((h, w, 3), dtype=np.float32)
        sig[..., 0] = y * inv
        if cv2 is not None:
            sig[..., 1] = cv2.resize(u * inv, (w, h), interpolation=cv2.INTER_LINEAR)
            sig[..., 2] = cv2.resize(v * inv, (w, h), interpolation=cv2.INTER_LINEAR)
        else:  # nearest-neighbour fallback
            sig[..., 1] = np.repeat(np.repeat(u * inv, 2, 0), 2, 1)[:h, :w]
            sig[..., 2] = np.repeat(np.repeat(v * inv, 2, 0), 2, 1)[:h, :w]
        np.clip(sig, 0.0, 1.0, out=sig)
        sig = self._reshape(sig)
        # IPT -> PQ'd LMS  (cv2.transform = SIMD 3x3, ~3x faster than numpy)
        sig -= self._off
        if cv2 is not None:
            img = cv2.transform(sig, self._m_ycc)
        else:
            img = sig @ self._m_ycc.T
        # PQ EOTF (LUT) -> linear LMS, 1.0 = npl (SDR reference white)
        idx = np.clip(img * 4095.0, 0.0, 4095.0).astype(np.int32)
        img = np.take(self._eotf_lut, idx)
        # -> linear BT.2020 RGB
        img = cv2.transform(img, self._m_post) if cv2 is not None else img @ self._m_post.T
        np.clip(img, 0.0, None, out=img)
        return img

    def decode(self, buf, w, h):
        """yuv420p10le bytes -> (h, w, 3) BGR uint8 (display: tonemap + BT.709)."""
        img = self._linear2020(buf, w, h)
        # hable tonemap on max channel (ffmpeg tonemap=hable:desat=0 behaviour)
        sig_t = np.maximum(img.max(axis=2), 1e-6)
        ratio = (_hable(np.minimum(sig_t, self._peak)) / self._hable_peak) / sig_t
        img *= ratio[..., None]
        # BT.2020 -> BT.709, encode, BGR
        img = cv2.transform(img, self._m_709) if cv2 is not None else img @ self._m_709.T
        oidx = np.clip(img * 2047.0, 0.0, 2047.0).astype(np.int32)
        return np.take(self._oetf_lut, oidx)[..., ::-1]

    def decode_native(self, buf, w, h):
        """yuv420p10le bytes -> (h, w, 3) float32 RGB, PQ-encoded BT.2020.

        The scope feed: the RPU-reshaped signal WITHOUT tonemapping or gamut
        conversion, re-encoded to PQ (0..1 = 0..10,000 nits) so scopes can
        analyse it exactly like a plain HDR10 stream."""
        img = self._linear2020(buf, w, h)           # 1.0 = npl nits
        return pq_oetf(img * (self._npl / 10000.0)).astype(np.float32)


def encode_iptpqc2(rgb_linear_2020, meta=None):
    """Inverse of the decode chain (no reshape) - test/fixture helper.
    rgb_linear_2020: float array (..., 3), 1.0 = 10,000 nits.
    Returns 10-bit IPT code values (..., 3) as uint16."""
    meta = meta or _FALLBACK_META
    m_post = DOVI_LMS2RGB @ np.array(meta["linear"], dtype=np.float64)
    lms = rgb_linear_2020 @ np.linalg.inv(m_post).T
    lmsp = pq_oetf(lms)
    m_ycc = np.array(meta["nonlinear"], dtype=np.float64)
    ipt = lmsp @ np.linalg.inv(m_ycc).T + np.array(meta["offset"])
    return np.clip(np.round(ipt * 1023.0), 0, 1023).astype(np.uint16)
