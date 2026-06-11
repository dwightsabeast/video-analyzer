#!/usr/bin/env python3
"""
va_scopes - numpy/cv2 scope renderers (no matplotlib, no Pillow).

Every function takes a BGR frame (as decoded) and returns an RGB uint8 image,
ready to hand to Tk via a PPM. Scopes: vectorscope, luma waveform, RGB parade,
false colour, luma histogram, and a CIE 1931 chromaticity / gamut plot.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

BG = (24, 24, 24)
GRID = (85, 85, 85)
SPOKE = (60, 60, 60)
DIM = (136, 136, 136)
TEXT = (170, 170, 170)
TEAL = (78, 201, 176)
GOLD = (240, 192, 64)
RED = (255, 96, 96)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _luma(bgr: np.ndarray, wide: bool = False) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    if wide:
        return 0.2627 * r + 0.6780 * g + 0.0593 * b
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# --- Vectorscope (hue x saturation) ------------------------------------------

@lru_cache(maxsize=8)
def _vscope_bg(size: int) -> bytes:
    img = np.empty((size, size, 3), np.uint8)
    img[:] = BG
    cx = cy = size // 2
    radius = max(4, size // 2 - 26)
    for sat in (64, 128, 192, 255):
        cv2.circle(img, (cx, cy), int(radius * sat / 255.0), GRID, 1, cv2.LINE_AA)
    labels = {0: "Red", 30: "Org", 60: "Yel", 90: "Cht", 120: "Grn", 150: "Spr",
              180: "Cyn", 210: "Azu", 240: "Blu", 270: "Vio", 300: "Mag", 330: "Rose"}
    for deg, name in labels.items():
        a = np.deg2rad(deg)
        dx, dy = np.sin(a), -np.cos(a)
        cv2.line(img, (cx, cy), (int(cx + radius * dx), int(cy + radius * dy)),
                 SPOKE, 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(name, FONT, 0.34, 1)
        lx = int(cx + (radius + 12) * dx) - tw // 2
        ly = int(cy + (radius + 12) * dy) + th // 2
        cv2.putText(img, name, (lx, ly), FONT, 0.34, TEXT, 1, cv2.LINE_AA)
    return img.tobytes() + bytes([size & 0xFF, (size >> 8) & 0xFF])


def vectorscope(bgr: np.ndarray, size: int, samples: int = 6000) -> np.ndarray:
    size = max(120, int(size))
    raw = _vscope_bg(size)
    img = np.frombuffer(raw[:-2], np.uint8).reshape(size, size, 3).copy()
    cx = cy = size // 2
    radius = max(4, size // 2 - 26)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if hsv.shape[0] == 0:
        return img
    n = min(samples, hsv.shape[0])
    rng = np.random.default_rng(0)          # fixed seed: deterministic renders
    idx = rng.integers(0, hsv.shape[0], size=n)
    s = hsv[idx]
    ang = np.deg2rad(s[:, 0].astype(np.float32) * 2.0)
    r = s[:, 1].astype(np.float32) / 255.0 * radius
    xs = (cx + r * np.sin(ang)).astype(np.int32)
    ys = (cy - r * np.cos(ang)).astype(np.int32)
    rgb = cv2.cvtColor(s.reshape(n, 1, 3), cv2.COLOR_HSV2RGB).reshape(n, 3)
    for ox in (0, 1):
        for oy in (0, 1):
            xx = np.clip(xs + ox, 0, size - 1)
            yy = np.clip(ys + oy, 0, size - 1)
            img[yy, xx] = rgb
    return img


# --- Luma histogram ----------------------------------------------------------

def histogram(bgr: np.ndarray, w: int, h: int, wide: bool = False,
              is_hdr: bool = False) -> np.ndarray:
    w, h = max(120, int(w)), max(120, int(h))
    luma = _luma(bgr, wide)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    left, right, top, bottom = 40, 12, 26, 22
    pw = max(1, w - left - right)
    ph = max(1, h - top - bottom)
    baseline = h - bottom
    bins = 128
    counts, edges = np.histogram(luma, bins=bins, range=(0, 255))
    peak = counts.max()
    norm = counts / peak if peak > 0 else counts
    if is_hdr:
        low = np.array([51, 26, 102], np.float32)
        high = np.array([255, 217, 38], np.float32)
        title = "Luma (Rec.2020, 8-bit)" if wide else "Luma (HDR, 8-bit)"
    else:
        low = np.array([38, 77, 128], np.float32)
        high = np.array([102, 217, 255], np.float32)
        title = "Luma (Rec.709)"
    barw = pw / bins
    for i in range(bins):
        t = (edges[i] + edges[i + 1]) / 2 / 255.0
        col = (low + (high - low) * t).astype(np.uint8)
        x0 = int(left + i * barw)
        x1 = max(x0 + 1, int(left + (i + 1) * barw))
        bh = int(norm[i] * ph)
        if bh > 0:
            img[baseline - bh:baseline, x0:x1] = col
    cv2.line(img, (left, baseline), (w - right, baseline), (68, 68, 68), 1, cv2.LINE_AA)
    for v in (0, 64, 128, 192, 255):
        x = int(left + v / 255.0 * pw)
        cv2.line(img, (x, baseline), (x, baseline + 3), DIM, 1, cv2.LINE_AA)
        cv2.putText(img, str(v), (x - 8, baseline + 15), FONT, 0.3, DIM, 1, cv2.LINE_AA)
    mean_v = float(luma.mean())
    p99 = float(np.percentile(luma, 99))
    mean_col = GOLD if is_hdr else TEAL
    mx = int(left + np.clip(mean_v, 0, 255) / 255.0 * pw)
    px = int(left + np.clip(p99, 0, 255) / 255.0 * pw)
    ys = np.arange(top, baseline)
    dash = (ys // 3) % 2 == 0
    if 0 <= mx < w:
        img[ys[dash], mx] = mean_col
    if 0 <= px < w:
        img[ys[dash], px] = RED
    cv2.putText(img, title, (left, 16), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "mean %.0f" % mean_v, (max(left, w - right - 150), 16),
                FONT, 0.35, mean_col, 1, cv2.LINE_AA)
    cv2.putText(img, "99%% %.0f" % p99, (max(left, w - right - 64), 16),
                FONT, 0.35, RED, 1, cv2.LINE_AA)
    return img


# --- Waveform & RGB parade ---------------------------------------------------

def _accum(plane: np.ndarray, out_w: int, levels: int = 256) -> np.ndarray:
    src_h = plane.shape[0]
    pw = cv2.resize(plane, (out_w, src_h), interpolation=cv2.INTER_AREA)
    lv = np.clip(pw, 0, levels - 1).astype(np.int32)
    cols = np.tile(np.arange(out_w), (src_h, 1))
    a = np.zeros((levels, out_w), np.float32)
    np.add.at(a, (lv, cols), 1.0)
    return a


def _trace(a: np.ndarray, out_w: int, out_h: int, tint) -> np.ndarray:
    m = a.max() or 1.0
    inten = np.log1p(a) / np.log1p(m)
    grid = cv2.resize(inten, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    grid = np.flipud(grid)
    canvas = np.zeros((out_h, out_w, 3), np.float32)
    for c in range(3):
        canvas[:, :, c] = grid * tint[c]
    return canvas


def _graticule(img: np.ndarray, x0: int, x1: int):
    h = img.shape[0]
    for ire, lab in ((0, "0"), (64, "25"), (128, "50"), (192, "75"), (255, "100")):
        y = int(h - 1 - ire / 255.0 * (h - 1))
        cv2.line(img, (x0, y), (x1, y), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(img, lab, (x0 + 2, max(10, y - 2)), FONT, 0.3, DIM, 1, cv2.LINE_AA)


def waveform(bgr: np.ndarray, w: int, h: int, wide: bool = False) -> np.ndarray:
    w, h = max(120, int(w)), max(80, int(h))
    luma = _luma(bgr, wide)
    base = np.empty((h, w, 3), np.float32)
    base[:] = BG
    base += _trace(_accum(luma, w), w, h, (110, 255, 150))
    img = np.clip(base, 0, 255).astype(np.uint8)
    _graticule(img, 0, w - 1)
    cv2.putText(img, "Waveform (luma)", (6, 14), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
    return img


def rgb_parade(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    w, h = max(150, int(w)), max(80, int(h))
    gap = 6
    cw = (w - 2 * gap) // 3
    base = np.empty((h, w, 3), np.float32)
    base[:] = BG
    tints = ((255, 60, 60), (60, 255, 60), (90, 90, 255))   # R, G, B
    planes = (bgr[:, :, 2].astype(np.float32), bgr[:, :, 1].astype(np.float32),
              bgr[:, :, 0].astype(np.float32))
    for i, (plane, tint) in enumerate(zip(planes, tints)):
        x0 = i * (cw + gap)
        base[:, x0:x0 + cw, :] += _trace(_accum(plane, cw), cw, h, tint)
    img = np.clip(base, 0, 255).astype(np.uint8)
    for i in range(3):
        x0 = i * (cw + gap)
        _graticule(img, x0, x0 + cw - 1)
    cv2.putText(img, "RGB parade", (6, 14), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
    return img


# --- False colour ------------------------------------------------------------

@lru_cache(maxsize=1)
def _false_lut() -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    stops = [(0, (40, 30, 90)), (16, (30, 60, 180)), (45, (40, 160, 160)),
             (110, (60, 170, 60)), (128, (150, 150, 150)), (180, (210, 200, 70)),
             (235, (235, 140, 40)), (254, (240, 80, 40)), (255, (255, 40, 40))]
    for j in range(len(stops) - 1):
        v0, c0 = stops[j]
        v1, c1 = stops[j + 1]
        for v in range(v0, v1 + 1):
            t = (v - v0) / max(1, (v1 - v0))
            lut[v] = [int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3)]
    return lut


def false_color(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    w, h = max(120, int(w)), max(80, int(h))
    luma = np.clip(_luma(bgr), 0, 255).astype(np.uint8)
    colored = _false_lut()[luma]
    sh, sw = colored.shape[:2]
    scale = min(w / sw, h / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(colored, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.empty((h, w, 3), np.uint8)
    canvas[:] = BG
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# --- CIE 1931 chromaticity / gamut -------------------------------------------

_PRIMS = {
    "709": [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)],
    "P3": [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
    "2020": [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
}
_PRIM_COL = {"709": (120, 200, 255), "P3": (240, 200, 90), "2020": (120, 255, 150)}


# Tolerance for the gamut inside/outside test, in CIE xy units. Legitimate
# on-gamut colours (pure primaries/secondaries) sit exactly on the triangle
# edge, so float rounding alone could flip them outside; 1e-3 absorbs that
# (plus 8-bit/codec noise) while staying far below both the old pixel-grid
# quantisation step (~2.5e-3 at size 320, which caused a ~2% false floor) and
# any real gamut difference (709 vs P3 primaries differ by ~0.04-0.2).
_GAMUT_EPS = 1e-3


def _frac_outside(xs, ys, prim, eps=_GAMUT_EPS):
    """Fraction of float chromaticity points outside the primaries' triangle.

    Tested in xy space *before* pixel quantisation; points within ``eps`` of
    an edge count as inside."""
    if xs.shape[0] == 0:
        return 0.0
    (x0, y0), (x1, y1), (x2, y2) = prim
    sgn = 1.0 if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) >= 0 else -1.0
    inside = np.ones(xs.shape[0], bool)
    for i in range(3):
        ax, ay = prim[i]
        bx, by = prim[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        d = sgn * (ex * (ys - ay) - ey * (xs - ax)) / float(np.hypot(ex, ey))
        inside &= d >= -eps
    return float(1.0 - inside.mean())


def _xy_to_px(x, y, size):
    return int(np.clip(x / 0.8, 0, 1) * (size - 1)), int((1 - np.clip(y / 0.9, 0, 1)) * (size - 1))


def cie_gamut(bgr: np.ndarray, size: int, samples: int = 8000) -> "tuple[np.ndarray, dict]":
    size = max(160, int(size))
    img = np.empty((size, size, 3), np.uint8)
    img[:] = BG
    flat = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0
    n = min(samples, flat.shape[0])
    if n > 0:
        rng = np.random.default_rng(0)      # fixed seed: deterministic renders
        idx = rng.integers(0, flat.shape[0], size=n)
        s = flat[idx]
        lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
        r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
        X = 0.4124 * r + 0.3576 * g + 0.1805 * b
        Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
        tot = X + Y + Z
        ok = tot > 1e-6
        cx = (X[ok] / tot[ok]); cy = (Y[ok] / tot[ok])
        pxs = np.clip(cx / 0.8, 0, 1) * (size - 1)
        pys = (1 - np.clip(cy / 0.9, 0, 1)) * (size - 1)
        pts = np.stack([pxs, pys], 1).astype(np.int32)
        img[np.clip(pts[:, 1], 0, size - 1), np.clip(pts[:, 0], 0, size - 1)] = (200, 200, 200)
    else:
        pts = np.zeros((0, 2), np.int32)
    for name, prim in _PRIMS.items():
        poly = np.array([_xy_to_px(x, y, size) for x, y in prim], np.int32)
        cv2.polylines(img, [poly], True, _PRIM_COL[name], 1, cv2.LINE_AA)
        cv2.putText(img, name, tuple(poly[0] + np.array([3, -3])), FONT, 0.32,
                    _PRIM_COL[name], 1, cv2.LINE_AA)
    cov = {"coverage_2020_pct": 0.0, "outside_709_pct": 0.0, "source": "display"}
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts.astype(np.int32))
        hull_area = cv2.contourArea(hull)
        tri2020 = np.array([_xy_to_px(x, y, size) for x, y in _PRIMS["2020"]], np.int32)
        a2020 = cv2.contourArea(tri2020)
        cov["coverage_2020_pct"] = round(min(100.0, hull_area / a2020 * 100.0), 1) if a2020 else 0.0
        cov["outside_709_pct"] = round(_frac_outside(cx, cy, _PRIMS["709"]) * 100.0, 1)
    cv2.putText(img, "CIE 1931  cover2020 %.0f%%  out709 %.0f%%" %
                (cov["coverage_2020_pct"], cov["outside_709_pct"]),
                (6, size - 8), FONT, 0.34, TEXT, 1, cv2.LINE_AA)
    return img, cov

# === Native-signal scopes =====================================================
# Variants that take float32 RGB frames in 0..1 SIGNAL domain straight off
# va_ffmpeg.NativeTap: the file's own transfer (PQ/HLG/gamma) and primaries
# (BT.2020/P3/BT.709) - no tonemap, no gamut conversion, no 8-bit crush. The
# display-referred functions above keep working for SDR files and fallbacks.

_M1, _M2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
_C1, _C2, _C3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0

_PRIM_LABEL = {"709": "BT.709", "P3": "P3", "2020": "BT.2020"}

# RGB -> XYZ (D65) per primaries set (rows: X, Y, Z).
_RGB2XYZ = {
    "709": np.array([[0.4124, 0.3576, 0.1805],
                     [0.2126, 0.7152, 0.0722],
                     [0.0193, 0.1192, 0.9505]], np.float64),
    "P3": np.array([[0.4866, 0.2657, 0.1982],
                    [0.2290, 0.6917, 0.0793],
                    [0.0000, 0.0451, 1.0439]], np.float64),
    "2020": np.array([[0.6370, 0.1446, 0.1689],
                      [0.2627, 0.6780, 0.0593],
                      [0.0000, 0.0281, 1.0610]], np.float64),
}

# Y'CbCr (non-constant-luminance) coefficients: (luma weights, Cb div, Cr div).
# P3-tagged streams are almost always coded with the BT.709 matrix.
_YCC = {"709": ((0.2126, 0.7152, 0.0722), 1.8556, 1.5748),
        "P3": ((0.2126, 0.7152, 0.0722), 1.8556, 1.5748),
        "2020": ((0.2627, 0.6780, 0.0593), 1.8814, 1.4746)}


def _pq_eotf(e):
    """PQ signal 0..1 -> linear light, 1.0 = 10,000 nits."""
    e = np.clip(e, 0.0, 1.0)
    ep = np.power(e, 1.0 / _M2)
    return np.power(np.clip(ep - _C1, 0.0, None) /
                    np.clip(_C2 - _C3 * ep, 1e-7, None), 1.0 / _M1)


def _pq_oetf(x):
    """Linear light (1.0 = 10,000 nits) -> PQ signal 0..1."""
    x = np.clip(x, 0.0, None)
    xp = np.power(x, _M1)
    return np.power((_C1 + _C2 * xp) / (1.0 + _C3 * xp), _M2)


def _nits_to_pq(n):
    return float(_pq_oetf(np.float64(n) / 10000.0))


def _hlg_inv_oetf(e):
    """HLG signal 0..1 -> scene-linear light (relative, 1.0 = peak)."""
    e = np.clip(e, 0.0, 1.0)
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.where(e <= 0.5, e * e / 3.0, (np.exp((e - c) / a) + b) / 12.0)


def _linearize(rgb, transfer):
    if transfer == "pq":
        return _pq_eotf(rgb)
    if transfer == "hlg":
        return _hlg_inv_oetf(rgb)
    return np.power(np.clip(rgb, 0.0, 1.0), 2.4)   # SDR video: BT.1886 display gamma


@lru_cache(maxsize=4)
def _to709_matrix(primaries: str) -> np.ndarray:
    """Linear RGB src-primaries -> linear BT.709 (for cosmetic dot colours)."""
    if primaries not in _RGB2XYZ or primaries == "709":
        return np.eye(3, dtype=np.float32)
    m = np.linalg.inv(_RGB2XYZ["709"]) @ _RGB2XYZ[primaries]
    return m.astype(np.float32)


def _luma_signal(rgbf: np.ndarray, primaries: str) -> np.ndarray:
    """Broadcast luma Y' on the signal (non-linear) values, 0..1."""
    cf = _YCC.get(primaries, _YCC["709"])[0]
    return cf[0] * rgbf[..., 0] + cf[1] * rgbf[..., 1] + cf[2] * rgbf[..., 2]


def _tag(transfer: str, primaries: str) -> str:
    return "%s/%s" % (transfer.upper(), _PRIM_LABEL.get(primaries, primaries))


# --- CIE gamut (native) -------------------------------------------------------

def cie_gamut_native(rgbf: np.ndarray, size: int, transfer: str = "pq",
                     primaries: str = "2020",
                     samples: int = 8000) -> "tuple[np.ndarray, dict]":
    """CIE 1931 plot of the NATIVE signal: linearised with the file's own
    transfer and projected with its own primaries matrix, so wide-gamut points
    really land outside BT.709 instead of being tonemapped into it first."""
    size = max(160, int(size))
    img = np.empty((size, size, 3), np.uint8)
    img[:] = BG
    flat = rgbf.reshape(-1, 3).astype(np.float32)
    n = min(samples, flat.shape[0])
    cov = {"coverage_2020_pct": 0.0, "outside_709_pct": 0.0, "source": "native",
           "transfer": transfer, "primaries": primaries}
    pts = np.zeros((0, 2), np.int32)
    cx = cy = None
    if n > 0:
        rng = np.random.default_rng(0)      # fixed seed: deterministic renders
        idx = rng.integers(0, flat.shape[0], size=n)
        lin = _linearize(flat[idx], transfer)
        m = _RGB2XYZ.get(primaries, _RGB2XYZ["709"]).astype(np.float32)
        xyz = lin @ m.T
        tot = xyz.sum(axis=1)
        ok = tot > 1e-8
        cx = xyz[ok, 0] / tot[ok]
        cy = xyz[ok, 1] / tot[ok]
        pxs = np.clip(cx / 0.8, 0, 1) * (size - 1)
        pys = (1 - np.clip(cy / 0.9, 0, 1)) * (size - 1)
        pts = np.stack([pxs, pys], 1).astype(np.int32)
        img[np.clip(pts[:, 1], 0, size - 1), np.clip(pts[:, 0], 0, size - 1)] = (200, 200, 200)
    for name, prim in _PRIMS.items():
        poly = np.array([_xy_to_px(x, y, size) for x, y in prim], np.int32)
        cv2.polylines(img, [poly], True, _PRIM_COL[name], 1, cv2.LINE_AA)
        cv2.putText(img, name, tuple(poly[0] + np.array([3, -3])), FONT, 0.32,
                    _PRIM_COL[name], 1, cv2.LINE_AA)
    if pts.shape[0] >= 3 and cx is not None:
        hull = cv2.convexHull(pts.astype(np.int32))
        hull_area = cv2.contourArea(hull)
        tri2020 = np.array([_xy_to_px(x, y, size) for x, y in _PRIMS["2020"]], np.int32)
        a2020 = cv2.contourArea(tri2020)
        cov["coverage_2020_pct"] = round(min(100.0, hull_area / a2020 * 100.0), 1) if a2020 else 0.0
        cov["outside_709_pct"] = round(_frac_outside(cx, cy, _PRIMS["709"]) * 100.0, 1)
    cv2.putText(img, "CIE 1931 NATIVE %s" % _tag(transfer, primaries),
                (6, 16), FONT, 0.34, GOLD, 1, cv2.LINE_AA)
    cv2.putText(img, "cover2020 %.0f%%  out709 %.0f%%" %
                (cov["coverage_2020_pct"], cov["outside_709_pct"]),
                (6, size - 8), FONT, 0.34, TEXT, 1, cv2.LINE_AA)
    return img, cov


# --- Waveform / parade (native) ------------------------------------------------

def _graticule_native(img: np.ndarray, x0: int, x1: int, transfer: str):
    h = img.shape[0]
    if transfer == "pq":
        marks = [(0.0, "0"), (_nits_to_pq(100), "100"), (_nits_to_pq(203), "203"),
                 (_nits_to_pq(1000), "1k"), (_nits_to_pq(4000), "4k"), (1.0, "10k nit")]
    else:
        marks = [(v / 100.0, ("%d%%" % v) if v == 100 else str(v))
                 for v in (0, 25, 50, 75, 100)]
    for f, lab in marks:
        y = int(round((h - 1) - f * (h - 1)))
        cv2.line(img, (x0, y), (x1, y), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(img, lab, (x0 + 2, max(10, y - 2)), FONT, 0.3, DIM, 1, cv2.LINE_AA)


def waveform_native(rgbf: np.ndarray, w: int, h: int, transfer: str = "pq",
                    primaries: str = "2020") -> np.ndarray:
    w, h = max(120, int(w)), max(80, int(h))
    luma = np.clip(_luma_signal(rgbf, primaries), 0.0, 1.0) * 255.0
    base = np.empty((h, w, 3), np.float32)
    base[:] = BG
    base += _trace(_accum(luma.astype(np.float32), w), w, h, (110, 255, 150))
    img = np.clip(base, 0, 255).astype(np.uint8)
    _graticule_native(img, 0, w - 1, transfer)
    cv2.putText(img, "Waveform NATIVE %s" % _tag(transfer, primaries),
                (6, 14), FONT, 0.4, GOLD, 1, cv2.LINE_AA)
    return img


def rgb_parade_native(rgbf: np.ndarray, w: int, h: int, transfer: str = "pq",
                      primaries: str = "2020") -> np.ndarray:
    w, h = max(150, int(w)), max(80, int(h))
    gap = 6
    cw = (w - 2 * gap) // 3
    base = np.empty((h, w, 3), np.float32)
    base[:] = BG
    tints = ((255, 60, 60), (60, 255, 60), (90, 90, 255))   # R, G, B
    for i in range(3):
        plane = np.clip(rgbf[..., i], 0.0, 1.0) * 255.0
        x0 = i * (cw + gap)
        base[:, x0:x0 + cw, :] += _trace(_accum(plane.astype(np.float32), cw), cw, h, tints[i])
    img = np.clip(base, 0, 255).astype(np.uint8)
    for i in range(3):
        x0 = i * (cw + gap)
        _graticule_native(img, x0, x0 + cw - 1, transfer)
    cv2.putText(img, "RGB parade NATIVE %s" % _tag(transfer, primaries),
                (6, 14), FONT, 0.4, GOLD, 1, cv2.LINE_AA)
    return img


# --- Histogram (native) ---------------------------------------------------------

def histogram_native(rgbf: np.ndarray, w: int, h: int, transfer: str = "pq",
                     primaries: str = "2020") -> np.ndarray:
    w, h = max(120, int(w)), max(120, int(h))
    luma = np.clip(_luma_signal(rgbf, primaries), 0.0, 1.0)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    left, right, top, bottom = 40, 12, 26, 22
    pw = max(1, w - left - right)
    ph = max(1, h - top - bottom)
    baseline = h - bottom
    bins = 128
    counts, edges = np.histogram(luma, bins=bins, range=(0.0, 1.0))
    peak = counts.max()
    norm = counts / peak if peak > 0 else counts
    low = np.array([51, 26, 102], np.float32)
    high = np.array([255, 217, 38], np.float32)
    barw = pw / bins
    for i in range(bins):
        t = (edges[i] + edges[i + 1]) / 2.0
        col = (low + (high - low) * t).astype(np.uint8)
        x0 = int(left + i * barw)
        x1 = max(x0 + 1, int(left + (i + 1) * barw))
        bh = int(norm[i] * ph)
        if bh > 0:
            img[baseline - bh:baseline, x0:x1] = col
    cv2.line(img, (left, baseline), (w - right, baseline), (68, 68, 68), 1, cv2.LINE_AA)
    if transfer == "pq":
        ticks = [(0.0, "0"), (_nits_to_pq(100), "100"), (_nits_to_pq(203), "203"),
                 (_nits_to_pq(1000), "1k"), (_nits_to_pq(4000), "4k"), (1.0, "10k")]
    else:
        ticks = [(v / 100.0, str(v)) for v in (0, 25, 50, 75, 100)]
    for f, lab in ticks:
        x = int(left + f * pw)
        cv2.line(img, (x, baseline), (x, baseline + 3), DIM, 1, cv2.LINE_AA)
        cv2.putText(img, lab, (x - 8, baseline + 15), FONT, 0.3, DIM, 1, cv2.LINE_AA)
    # stats: true luminance (linear Y) in nits for PQ, signal % otherwise
    lin = _linearize(rgbf, transfer)
    yrow = _RGB2XYZ.get(primaries, _RGB2XYZ["709"])[1].astype(np.float32)
    ylin = lin @ yrow
    if transfer == "pq":
        nits = ylin * 10000.0
        mean_n, p99_n = float(nits.mean()), float(np.percentile(nits, 99))
        mx = int(left + np.clip(_nits_to_pq(max(mean_n, 0.0)), 0, 1) * pw)
        px = int(left + np.clip(_nits_to_pq(max(p99_n, 0.0)), 0, 1) * pw)
        s_mean = ("mean %d nit" % round(mean_n)) if mean_n >= 1 else "mean %.2f nit" % mean_n
        s_p99 = ("99%% %d nit" % round(p99_n)) if p99_n >= 1 else "99%% %.2f nit" % p99_n
    else:
        mean_v, p99_v = float(luma.mean()), float(np.percentile(luma, 99))
        mx = int(left + np.clip(mean_v, 0, 1) * pw)
        px = int(left + np.clip(p99_v, 0, 1) * pw)
        s_mean = "mean %.0f%%" % (mean_v * 100.0)
        s_p99 = "99%% %.0f%%" % (p99_v * 100.0)
    ys = np.arange(top, baseline)
    dash = (ys // 3) % 2 == 0
    if 0 <= mx < w:
        img[ys[dash], mx] = GOLD
    if 0 <= px < w:
        img[ys[dash], px] = RED
    cv2.putText(img, "Luma NATIVE %s" % _tag(transfer, primaries),
                (left, 16), FONT, 0.4, GOLD, 1, cv2.LINE_AA)
    cv2.putText(img, s_mean, (max(left, w - right - 170), 16), FONT, 0.35, GOLD, 1, cv2.LINE_AA)
    cv2.putText(img, s_p99, (max(left, w - right - 76), 16), FONT, 0.35, RED, 1, cv2.LINE_AA)
    return img


# --- Vectorscope (native Y'CbCr) -------------------------------------------------

def _ycbcr(rgbf: np.ndarray, primaries: str):
    cf, dcb, dcr = _YCC.get(primaries, _YCC["709"])
    y = cf[0] * rgbf[..., 0] + cf[1] * rgbf[..., 1] + cf[2] * rgbf[..., 2]
    cb = (rgbf[..., 2] - y) / dcb
    cr = (rgbf[..., 0] - y) / dcr
    return y, cb, cr


@lru_cache(maxsize=8)
def _vscope_bg_native(size: int, primaries: str) -> bytes:
    """CbCr graticule: rings + 75%-bar target crosses computed from the SAME
    matrix used to plot, so target positions are colourimetrically exact."""
    img = np.empty((size, size, 3), np.uint8)
    img[:] = BG
    cx = cy = size // 2
    radius = max(4, size // 2 - 26)
    for frac in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(img, (cx, cy), int(radius * frac), GRID, 1, cv2.LINE_AA)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), SPOKE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), SPOKE, 1, cv2.LINE_AA)
    scale = radius / 0.5
    bars = {"R": (0.75, 0.0, 0.0), "Yl": (0.75, 0.75, 0.0), "G": (0.0, 0.75, 0.0),
            "Cy": (0.0, 0.75, 0.75), "B": (0.0, 0.0, 0.75), "Mg": (0.75, 0.0, 0.75)}
    for name, rgb in bars.items():
        v = np.array(rgb, np.float32).reshape(1, 1, 3)
        _, cb, cr = _ycbcr(v, primaries)
        x = int(round(cx + cb.item() * scale))   # .item(): numpy>=2 rejects
        y = int(round(cy - cr.item() * scale))   # float() on (1,1) arrays
        cv2.drawMarker(img, (x, y), DIM, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cv2.putText(img, name, (x + 5, y - 4), FONT, 0.34, TEXT, 1, cv2.LINE_AA)
    return img.tobytes() + bytes([size & 0xFF, (size >> 8) & 0xFF])


def vectorscope_native(rgbf: np.ndarray, size: int, transfer: str = "pq",
                       primaries: str = "2020", samples: int = 6000) -> np.ndarray:
    size = max(120, int(size))
    raw = _vscope_bg_native(size, primaries)
    img = np.frombuffer(raw[:-2], np.uint8).reshape(size, size, 3).copy()
    cx = cy = size // 2
    radius = max(4, size // 2 - 26)
    flat = rgbf.reshape(-1, 3)
    if flat.shape[0] == 0:
        return img
    n = min(samples, flat.shape[0])
    rng = np.random.default_rng(0)          # fixed seed: deterministic renders
    idx = rng.integers(0, flat.shape[0], size=n)
    s = flat[idx].astype(np.float32)
    _, cb, cr = _ycbcr(s, primaries)
    scale = radius / 0.5
    xs = np.clip(cx + cb * scale, 0, size - 2).astype(np.int32)
    ys = np.clip(cy - cr * scale, 0, size - 2).astype(np.int32)
    # cosmetic dot colour: linearise, rotate into 709, re-encode for display
    lin = _linearize(s, transfer)
    if transfer == "pq":
        lin = lin * (10000.0 / 203.0)        # 203 nit = reference white
    elif transfer == "hlg":
        lin = lin / max(float(_hlg_inv_oetf(np.float64(0.75))), 1e-6)
    rgb709 = np.clip(lin @ _to709_matrix(primaries).T, 0.0, 1.0)
    dots = (np.power(rgb709, 1.0 / 2.2) * 255.0).astype(np.uint8)
    for ox in (0, 1):
        for oy in (0, 1):
            img[ys + oy, xs + ox] = dots
    cv2.putText(img, "Vectorscope NATIVE Y'CbCr %s" % _PRIM_LABEL.get(primaries, primaries),
                (6, 14), FONT, 0.34, GOLD, 1, cv2.LINE_AA)
    return img


# --- False colour (native, nits-banded for PQ) -----------------------------------

_PQ_BANDS = [(0.05, (40, 30, 90), "<0.05"), (1.0, (60, 70, 200), "1"),
             (10.0, (50, 160, 200), "10"), (100.0, (150, 150, 150), "100"),
             (203.0, (70, 200, 70), "203"), (600.0, (210, 200, 70), "600"),
             (1000.0, (235, 150, 40), "1k"), (4000.0, (240, 70, 40), "4k"),
             (10000.0, (255, 60, 255), "10k")]


@lru_cache(maxsize=1)
def _false_lut_pq() -> np.ndarray:
    """1024-entry LUT over PQ signal 0..1 -> flat nits-band colours."""
    lut = np.zeros((1024, 3), np.uint8)
    sig = _pq_eotf(np.linspace(0.0, 1.0, 1024)) * 10000.0
    prev = 0.0
    for hi, col, _lab in _PQ_BANDS:
        m = (sig >= prev) & (sig < hi)
        lut[m] = col
        prev = hi
    lut[sig >= prev] = _PQ_BANDS[-1][1]
    return lut


def false_color_native(rgbf: np.ndarray, w: int, h: int, transfer: str = "pq",
                       primaries: str = "2020") -> np.ndarray:
    w, h = max(120, int(w)), max(80, int(h))
    legend_h = 16
    luma = np.clip(_luma_signal(rgbf, primaries), 0.0, 1.0)
    if transfer == "pq":
        colored = _false_lut_pq()[np.clip(luma * 1023.0, 0, 1023).astype(np.int32)]
    else:
        colored = _false_lut()[np.clip(luma * 255.0, 0, 255).astype(np.uint8)]
    sh, sw = colored.shape[:2]
    avail_h = max(40, h - legend_h)
    scale = min(w / sw, avail_h / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(colored, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.empty((h, w, 3), np.uint8)
    canvas[:] = BG
    y0, x0 = (avail_h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    if transfer == "pq":
        seg = w // len(_PQ_BANDS)
        yl = h - legend_h
        for i, (_hi, col, lab) in enumerate(_PQ_BANDS):
            xa = i * seg
            canvas[yl + 3:yl + 9, xa + 2:xa + seg - 2] = col
            cv2.putText(canvas, lab, (xa + 2, h - 1), FONT, 0.28, TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "False colour NATIVE (nits)", (6, 14), FONT, 0.36,
                    GOLD, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "False colour NATIVE (%s signal)" % transfer.upper(),
                    (6, 14), FONT, 0.36, GOLD, 1, cv2.LINE_AA)
    return canvas


def mark_display_referred(img: np.ndarray) -> np.ndarray:
    """Stamp scopes rendered from the TONEMAPPED preview of an HDR/WCG file
    (no native tap available - e.g. ffmpeg missing). Honest labelling: these
    values are display-referred, not the file's own signal."""
    h = img.shape[0]
    y = h - 20 if h > 60 else max(12, h - 4)
    cv2.putText(img, "DISPLAY-REFERRED (tonemapped preview)", (6, y),
                FONT, 0.32, (240, 160, 60), 1, cv2.LINE_AA)
    return img
