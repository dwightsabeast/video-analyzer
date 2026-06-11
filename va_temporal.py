#!/usr/bin/env python3
"""
va_temporal - integrity of the time axis.

  * cadence()     - duplicate-frame ratio and cadence classification (e.g. 2:3
                    pulldown / telecine, heavy duplication) from frame-to-frame diffs.
  * pse_flashes() - Harding-style photosensitive-epilepsy screening: luminance and
                    saturated-red flashes per second (risk if > 3/s in any window).

Simplified relative to a certified Harding analyser, but a real safety indicator.
Pure cv2 + numpy; takes a va_ffmpeg.VideoSource.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def cadence(source, max_frames=1500) -> dict:
    """Duplicate ratio + cadence/interlace classification. Combing is measured at
    FULL vertical resolution (it is a line-pair artifact and vanishes on downscale);
    duplicate diffs use a downscale for speed."""
    fps = source.fps or 25.0
    source.start(0)
    prev_small = None
    diffs, combr = [], []
    n = 0
    while n < max_frames:
        fr = source.read()
        if fr is None:
            break
        gf = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        up = gf[:-2, :] - gf[1:-1, :]
        dn = gf[2:, :] - gf[1:-1, :]
        combr.append(float(np.mean(np.sqrt(np.clip(up * dn, 0, None)))))
        small = cv2.resize(gf, (160, max(1, int(160 * gf.shape[0] / gf.shape[1]))))
        if prev_small is not None:
            diffs.append(float(np.mean(np.abs(small - prev_small))))
        prev_small = small
        n += 1
    source.close()
    if n < 3:
        return {"dup_pct": 0.0, "dup_frames": [], "cadence": "too short",
                "comb_pct": 0.0, "fps": fps}
    diffs = np.array(diffs)
    eps = max(0.3, float(np.median(diffs)) * 0.12)
    dup = diffs < eps
    dup_idx = [int(i + 1) for i in np.where(dup)[0]]
    dup_pct = round(float(dup.mean()) * 100.0, 2)
    cr = np.array(combr)
    comb_pct = round(float((cr > 1.5).mean()) * 100.0, 2)

    cad = "progressive (no significant duplication)"
    if dup_idx and len(dup_idx) > 1:
        med = float(np.median(np.diff(dup_idx)))
        if 16 <= dup_pct <= 24 and 4 <= med <= 6:
            cad = "2:3 pulldown / telecine (duplicate cadence)"
        elif dup_pct >= 40:
            cad = "heavy duplication - true rate ~%.1f fps" % (fps * (1 - dup_pct / 100.0))
        elif dup_pct >= 3:
            cad = "%.0f%% duplicate frames" % dup_pct
    elif dup_pct >= 3:
        cad = "%.0f%% duplicate frames" % dup_pct
    if comb_pct > 15 and cad.startswith("progressive"):
        cad = "interlaced / telecine combing (%.0f%% combed frames)" % comb_pct
    return {"dup_pct": dup_pct, "dup_frames": dup_idx[:200], "cadence": cad,
            "comb_pct": comb_pct, "fps": round(fps, 3)}


def _flashes_per_sec(signal, swing, fps):
    if len(signal) < 2:
        return 0.0
    trans = (np.abs(np.diff(signal)) >= swing).astype(np.float32)
    win = max(1, int(round(fps)))
    if len(trans) >= win:
        counts = np.convolve(trans, np.ones(win), "valid")
        peak = float(counts.max())
    else:
        peak = float(trans.sum())
    return round(peak / 2.0, 2)   # a flash = an opposing transition pair


def pse_flashes(source, max_frames=4000) -> dict:
    fps = source.fps or 25.0
    source.start(0)
    luma, red = [], []
    n = 0
    while n < max_frames:
        fr = source.read()
        if fr is None:
            break
        s = cv2.resize(fr, (128, 72))
        b = s[:, :, 0].astype(np.float32)
        g = s[:, :, 1].astype(np.float32)
        r = s[:, :, 2].astype(np.float32)
        luma.append(float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean()))
        red.append(float(((r > 204) & (g < 76) & (b < 76)).mean()) * 255.0)
        n += 1
    source.close()
    lf = _flashes_per_sec(np.array(luma), 25.0, fps)        # ~10% of 255
    rf = _flashes_per_sec(np.array(red), 0.25 * 255, fps)   # 25% red-area swing
    return {"luminance_flashes_per_sec": lf, "red_flashes_per_sec": rf,
            "risk": bool(lf > 3 or rf > 3), "frames": n}
