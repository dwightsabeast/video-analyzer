#!/usr/bin/env python3
"""
va_perceptual - model the human eye, not just pixels.

  * banding_map  - detect contouring/banding (the #1 streaming/HDR complaint that
                   PSNR misses): contour edges sitting inside otherwise-smooth regions.
  * jnd_map      - artifact-VISIBILITY map: per-pixel error divided by a local
                   just-noticeable-difference threshold (texture + luminance masking),
                   so only errors a human could actually see light up.
  * saliency_map - spectral-residual saliency (Hou & Zhang) via numpy FFT.
  * visually_lossless - saliency-weighted verdict: can anyone even tell A from B?

Pure cv2 + numpy.
"""

from __future__ import annotations

import cv2
import numpy as np


def _luma(bgr):
    return (0.2126 * bgr[:, :, 2] + 0.7152 * bgr[:, :, 1] + 0.0722 * bgr[:, :, 0]).astype(np.float32)


def _local_std(x, k=5):
    mean = cv2.blur(x, (k, k))
    sq = cv2.blur(x * x, (k, k))
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def saliency_map(bgr, size=64):
    """Spectral-residual saliency, normalised 0..1, at the frame's resolution."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h0, w0 = g.shape
    small = cv2.resize(g, (size, size))
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    recon = np.fft.ifft2(np.exp(residual + 1j * phase))
    sal = np.abs(recon) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), 2.0)
    sal = cv2.resize(sal, (w0, h0))
    sal -= sal.min()
    m = sal.max()
    return sal / m if m > 0 else sal


def _line_kernels(length=11, n_angles=8):
    """Normalised 1-px-wide line-mean kernels at n_angles orientations."""
    ks = []
    half = length // 2
    for i in range(n_angles):
        a = np.pi * i / n_angles
        k = np.zeros((length, length), np.float32)
        for t in range(-half, half + 1):
            y = half + int(round(t * np.sin(a)))
            x = half + int(round(t * np.cos(a)))
            k[y, x] = 1.0
        ks.append(k / k.sum())
    return ks


_LINE_KERNELS = _line_kernels()


def banding_map(bgr):
    """Return (heatmap 0..1, banding_percent).

    Banding/contouring = quantization staircase edges in otherwise-smooth regions
    (gradients, skies). Method: the SIGNED residual against a Gaussian-debanded
    reference is averaged along 8 orientations (11-px line means); a contour
    keeps its sign along the iso-luma curve so the directional mean survives,
    while dither/noise (spatially unstructured) averages toward zero. Candidate
    pixels (directional coherence above an adaptive, noise-scaled threshold,
    inside low-texture areas) are then filtered by connected-component geometry:
    only long, thin, curve-like ridges count as banding — speckle is discarded.
    banding_percent = share of frame pixels on/near such coherent contours
    (luma only; a properly dithered gradient scores ~0)."""
    luma = _luma(bgr)
    ref = cv2.GaussianBlur(luma, (0, 0), 2.0)
    resid = luma - ref                                   # signed residual
    tex = _local_std(luma, 9)
    smooth = tex < 3.0

    # directional coherence: mean signed residual along 8 orientations
    coh = np.abs(cv2.filter2D(resid, -1, _LINE_KERNELS[0]))
    for k in _LINE_KERNELS[1:]:
        np.maximum(coh, np.abs(cv2.filter2D(resid, -1, k)), out=coh)

    # adaptive threshold: floor for clean content, scaled up by measured noise
    noise = float(np.median(np.abs(resid)[smooth])) if smooth.any() else 0.0
    thr = max(0.22, 1.2 * noise)
    cand = ((coh > thr) & smooth).astype(np.uint8)

    # structure filter: keep only long, thin connected ridges (contours)
    h, w = luma.shape
    min_len = max(16, min(h, w) // 16)
    band = np.zeros((h, w), bool)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n_lab > 1:
        bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        long_enough = np.maximum(bw, bh) >= min_len
        fill = area / np.maximum(bw * bh, 1.0)
        aspect = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1.0)
        thin = (fill <= 0.35) | (aspect >= 4.0)
        keep = np.where(long_enough & thin)[0] + 1
        if keep.size:
            band = np.isin(labels, keep)
    band = cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    heat = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.5)
    m = heat.max()
    if m > 0:
        heat = heat / m
    return heat, round(float(band.mean()) * 100.0, 3)


def jnd_map(ref_bgr, dist_bgr, k_tex=0.6, k_lum=0.04):
    """Return (visibility heatmap 0..1, percent_visible). Error normalised by a local
    JND threshold that rises with texture (masking) and toward black/white."""
    if dist_bgr.shape != ref_bgr.shape:
        dist_bgr = cv2.resize(dist_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
    a = _luma(ref_bgr)
    b = _luma(dist_bgr)
    err = np.abs(a - b)
    std = _local_std(a, 5)
    lum_mask = 1.0 + k_lum * np.abs(a - 128.0)
    threshold = (1.0 + k_tex * std) * lum_mask
    visibility = err / threshold
    pct = float((visibility > 1.0).mean()) * 100.0
    return np.clip(visibility / 4.0, 0, 1), round(pct, 3)


def visually_lossless(ref_bgr, dist_bgr, thresh_pct=0.5):
    """Saliency-weighted 'can anyone tell?' verdict for two frames."""
    vmap, pct = jnd_map(ref_bgr, dist_bgr)
    sal = saliency_map(ref_bgr)
    weighted = float((vmap * sal).sum() / (sal.sum() + 1e-6)) * 100.0
    if pct < thresh_pct:
        verdict = "visually lossless"
    elif pct < 2.0:
        verdict = "near-transparent"
    else:
        verdict = "visible artifacts"
    return {"visible_pct": round(pct, 3), "saliency_weighted_pct": round(weighted, 3),
            "verdict": verdict}


def colorize(map01, cmap=cv2.COLORMAP_JET):
    """0..1 float map -> RGB heatmap image."""
    m = np.clip(map01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(m, cmap), cv2.COLOR_BGR2RGB)
