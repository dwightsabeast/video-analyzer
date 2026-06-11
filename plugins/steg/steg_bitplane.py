"""Bit-plane / LSB steganalysis for frames and audio.

Two classical detectors, fused with RS as the reliable estimator:

  * RS analysis (Fridrich, Goljan, Du) - flips LSBs under a mask and counts
    Regular vs Singular groups; their divergence yields an estimate of the
    embedded message length as a fraction of the LSB plane. This is the
    primary, robust detector;
  * chi-square attack (Westfeld & Pfitzmann) - LSB embedding equalises the
    counts of each value pair (2i, 2i+1). Reported as CORROBORATING context
    only: it over-flags smooth histograms, so it never triggers a finding on
    its own (this is why RS/Sample-Pairs superseded it for spatial embedding).

Plus an LSB-plane image for eyeballing (real payloads look like structured
noise; natural content looks random).

IMPORTANT: pixel-LSB stego does NOT survive lossy video coding, so frame
analysis only means anything on a lossless/intra source - the plugin gates on
that. Natural images have noisy LSBs, so a low RS rate is normal; this is an
indicator, not proof. numpy only; pure functions, unit-tested directly."""

from __future__ import annotations

import math

import numpy as np


# --- regularized upper incomplete gamma (chi-square survival, no scipy) --------

def _gammq(a: float, x: float) -> float:
    if x <= 0:
        return 1.0
    if a <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:                              # series for P, return 1-P
        ap, s, term = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b, c = x + 1.0 - a, 1e300                    # continued fraction for Q
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h * math.exp(-x + a * math.log(x) - gln)


def chi_square(values: np.ndarray) -> dict:
    """Westfeld chi-square LSB test on an integer channel (low byte). Returns
    {p_embed, chi2, df}. Context only - see module docstring."""
    v = np.asarray(values).ravel().astype(np.int64) & 0xFF
    hist = np.bincount(v, minlength=256).astype(np.float64)
    even = hist[0::2]                            # n_{2i}
    odd = hist[1::2]                             # n_{2i+1}
    expected = (even + odd) / 2.0
    mask = expected > 0
    if mask.sum() < 2:
        return {"p_embed": 0.0, "chi2": 0.0, "df": 0}
    chi2 = float(np.sum((even[mask] - expected[mask]) ** 2 / expected[mask]))
    df = int(mask.sum() - 1)
    return {"p_embed": float(_gammq(df / 2.0, chi2 / 2.0)), "chi2": chi2, "df": df}


# --- RS analysis ---------------------------------------------------------------

def _flip_f1(x):                                # LSB flip: 0<->1, 2<->3, ...
    return x ^ 1


def _flip_fm1(x):                               # shifted flip: 1<->2, 3<->4, ...
    return ((x + 1) ^ 1) - 1


def _rs_counts(groups, mask, flip, vrange):
    g = groups.astype(np.int64)
    f = np.abs(np.diff(g, axis=1)).sum(axis=1)             # discrimination f(G)
    fl = g.copy()
    cols = np.flatnonzero(mask)
    fl[:, cols] = np.clip(flip(fl[:, cols]), vrange[0], vrange[1])
    ff = np.abs(np.diff(fl, axis=1)).sum(axis=1)           # f(F(G))
    R = float(np.mean(ff > f))
    S = float(np.mean(ff < f))
    return R, S


def rs_analysis(values: np.ndarray, group=4, vrange=(0, 255)) -> dict:
    """RS steganalysis. Returns {rate, R_m, S_m, R_nm, S_nm} where rate is the
    estimated embedded-message length as a fraction of the LSB plane (0..~1).
    vrange bounds the post-flip clip (8-bit pixels vs 16-bit audio)."""
    v = np.asarray(values).ravel().astype(np.int64)
    n = (v.size // group) * group
    if n < group * 64:
        return {"rate": 0.0, "R_m": 0, "S_m": 0, "R_nm": 0, "S_nm": 0}
    groups = v[:n].reshape(-1, group)
    mask = np.array([1, 0, 0, 1][:group] + [0] * max(0, group - 4))
    # positive mask uses F1, negative mask uses F-1; "1" = after flipping ALL
    # LSBs (Fridrich's second measurement point).
    dpos0, spos0 = _rs_counts(groups, mask, _flip_f1, vrange)
    dneg0, sneg0 = _rs_counts(groups, mask, _flip_fm1, vrange)
    allflip = (v[:n] ^ 1).reshape(-1, group)
    dpos1, spos1 = _rs_counts(allflip, mask, _flip_f1, vrange)
    dneg1, sneg1 = _rs_counts(allflip, mask, _flip_fm1, vrange)
    Rm, Sm, Rnm, Snm = dpos0, spos0, dneg0, sneg0
    # Fridrich RS quadratic on d = R - S for each (mask, point):
    #   2(d1+d0) z^2 + (dn0 - dn1 - d1 - 3 d0) z + (d0 - dn0) = 0,  p = z/(z-1/2)
    d0, dn0 = dpos0 - spos0, dneg0 - sneg0
    d1, dn1 = dpos1 - spos1, dneg1 - sneg1
    a = 2.0 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3.0 * d0
    c = d0 - dn0
    rate = 0.0
    z = None
    if abs(a) > 1e-12:
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            rt = math.sqrt(disc)
            z1, z2 = (-b + rt) / (2 * a), (-b - rt) / (2 * a)
            z = z1 if abs(z1) <= abs(z2) else z2
        else:
            z = -b / (2 * a)
    elif abs(b) > 1e-12:
        z = -c / b
    if z is not None and abs(z - 0.5) > 1e-9:
        rate = abs(z / (z - 0.5))
    return {"rate": float(min(max(rate, 0.0), 1.5)), "R_m": round(Rm, 4),
            "S_m": round(Sm, 4), "R_nm": round(Rnm, 4), "S_nm": round(Snm, 4)}


def lsb_plane(gray) -> np.ndarray:
    """LSB plane of a luma frame as a 0/255 uint8 image."""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = g.mean(axis=2)
    return ((g.astype(np.uint8) & 1) * 255).astype(np.uint8)


RS_WARN = 0.10        # RS embedded-rate above this = suspicious (clean ~<0.05)
RS_WEAK = 0.05


def analyze_channel(values, group=4, vrange=(0, 255)) -> dict:
    """Fuse RS (primary, robust) + chi-square (corroborating context only).
    suspicious is RS-driven; a high chi p on top raises confidence."""
    cs = chi_square(values)
    rs = rs_analysis(values, group=group, vrange=vrange)
    rate = rs["rate"]
    suspicious = rate > RS_WARN
    if not suspicious:
        conf = "weak" if rate > RS_WEAK else "clean"
    else:
        conf = "high" if cs["p_embed"] > 0.95 else "medium"
    return {"chi": cs, "rs": rs, "suspicious": bool(suspicious),
            "confidence": conf}

# NOTE on audio LSB: deliberately NOT implemented. Spatial RS assumes tiny
# neighbour differences (smooth image regions), which a high-amplitude 1-D
# audio signal violates - the estimator saturates. And clean PCM LSBs are
# frequently already random (dither / recording noise floor), so naive LSB
# tests over-flag. Reliable audio LSB steganalysis needs a dedicated method;
# the container layer still catches data appended to / wrapping the audio.
