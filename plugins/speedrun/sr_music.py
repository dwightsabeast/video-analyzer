"""Music-bed continuity screening (splice hunting in the soundtrack).

Game music is the most continuous thing in a capture: decoded PCM keeps its
spectrum evolving smoothly even while the picture cuts between gameplay and
menus. A video splice therefore tends to leave a one-hop spectral jump in
the music bed - exactly the artifact that betrayed the music skip in the
debunked Diablo 3:12 run. This module computes a spectral-flux novelty
curve, normalises it against local musical activity, and flags outlier
discontinuities; each flag is checked for a context change (different
spectrum before vs after) and for "restart" jumps (the audio after the cut
matching a much earlier point of the soundtrack, i.e. a re-used segment).

Compute only - no Tk. analyze_samples() is pure numpy on a PCM array so the
selftest can feed it synthetic audio; analyze() decodes via va_audio."""

from __future__ import annotations

import numpy as np

import va_audio

SR = 11025
FRAME = 2048
HOP = 512
SCORE_THR = 7.0          # local-sigma outlier threshold
CTX_COS = 0.993          # context cosine below this = spectrum changed
RESTART_COS = 0.996      # fingerprint match above this = re-used audio
RESTART_MARGIN = 0.003   # ...and it must beat plain continuation by this
SILENCE_DB = -55.0
MAX_FINDINGS = 12


def _band_edges(nbins: int, nbands: int = 24) -> np.ndarray:
    return np.linspace(0, nbins, nbands + 1).astype(int)[:-1]


def analyze_samples(x: np.ndarray, sr: int = SR) -> dict:
    """Novelty scan of mono float PCM. Returns findings + a plot curve."""
    x = np.asarray(x, np.float32).ravel()
    frame, hop = FRAME, HOP
    if x.size < frame * 4:
        return {"ok": False, "error": "audio too short"}
    nhop = (x.size - frame) // hop + 1
    if nhop > 200_000:                     # very long file: halve resolution
        hop *= 2
        nhop = (x.size - frame) // hop + 1
    hop_s = hop / float(sr)
    win = np.hanning(frame).astype(np.float32)
    edges = _band_edges(frame // 2 + 1)
    flux = np.zeros(nhop, np.float32)
    rms = np.zeros(nhop, np.float32)
    fp = np.zeros((nhop, edges.size), np.float32)
    prev = None
    chunk = 4096
    for c0 in range(0, nhop, chunk):
        c1 = min(nhop, c0 + chunk)
        idx = (np.arange(c0, c1)[:, None] * hop + np.arange(frame)[None, :])
        w = x[idx]
        rms[c0:c1] = np.sqrt(np.mean(w * w, axis=1) + 1e-12)
        logm = np.log1p(np.abs(np.fft.rfft(w * win, axis=1)).astype(np.float32))
        fp[c0:c1] = np.add.reduceat(logm, edges, axis=1) / np.diff(
            np.append(edges, logm.shape[1]))[None, :]
        block = logm if prev is None else np.vstack((prev, logm))
        d = np.clip(np.diff(block, axis=0), 0.0, None).mean(axis=1)
        flux[c0 + (1 if prev is None else 0):c1] = d
        prev = logm[-1:]
    silent = 20.0 * np.log10(rms + 1e-12) < SILENCE_DB
    flux[silent] = 0.0
    flux[np.roll(silent, 1)] = 0.0
    # local normalisation: how unusual is this hop vs +-1.5 s of context
    k = max(9, int(round(3.0 / hop_s)) | 1)
    ker = np.ones(k, np.float32) / k
    loc_mean = np.convolve(flux, ker, mode="same")
    loc_mad = np.convolve(np.abs(flux - loc_mean), ker, mode="same")
    score = (flux - loc_mean) / (loc_mad + 1e-6)
    act = flux[~silent]
    floor = float(np.percentile(act, 98)) if act.size else 0.0
    cand = np.flatnonzero((score > SCORE_THR) & (flux > floor))
    # non-max suppression within 0.35 s
    cand = cand[np.argsort(score[cand])[::-1]]
    keep, taken = [], np.zeros(nhop, bool)
    r = max(1, int(round(0.35 / hop_s)))
    for i in cand:
        if not taken[max(0, i - r):i + r].any():
            keep.append(int(i))
            taken[i] = True
        if len(keep) >= MAX_FINDINGS:
            break
    keep.sort()
    # per-second fingerprints for restart matching
    per = max(1, int(round(1.0 / hop_s)))
    nsec = nhop // per
    fpsec = fp[:nsec * per].reshape(nsec, per, -1).mean(axis=1) if nsec else fp[:0]

    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    findings = []
    ctxn = max(2, int(round(1.0 / hop_s)))
    for i in keep:
        t = i * hop_s
        a = fp[max(0, i - ctxn - 2):max(1, i - 2)].mean(axis=0)
        b = fp[i + 2:i + 2 + ctxn].mean(axis=0) if i + 4 < nhop else a
        changed = _cos(a, b) < CTX_COS
        matched_at = None
        s = int(t)
        if nsec and s + 4 <= nsec and s > 6:
            post = fpsec[s + 1:s + 4].ravel()
            best, best_s = 0.0, None
            for e0 in range(0, s - 5):
                cv = _cos(post, fpsec[e0:e0 + 3].ravel())
                if cv > best:
                    best, best_s = cv, e0
            cont = _cos(post, fpsec[max(0, s - 3):s].ravel()) if s >= 3 else 0.0
            if best > RESTART_COS and best - cont > RESTART_MARGIN:
                matched_at = best_s
        sev = ("warn" if (changed or matched_at is not None)
               and score[i] > SCORE_THR + 2.0 else "info")
        txt = "music bed jumps (x%.1f local dev)" % float(score[i])
        if changed:
            txt += ", spectrum differs before/after"
        if matched_at is not None:
            txt += ", audio after the jump matches ~%d:%02d (re-used segment?)" \
                   % (matched_at // 60, matched_at % 60)
        findings.append({"t": float(t), "score": float(score[i]),
                         "severity": sev, "text": txt})
    # compact curve for plotting
    step = max(1, nhop // 2400)
    n2 = (nhop // step) * step
    cv = score[:n2].reshape(-1, step).max(axis=1)
    return {"ok": True, "sr": int(sr), "hop_s": float(hop_s),
            "duration_s": float(x.size / sr), "thr": float(SCORE_THR),
            "findings": findings,
            "curve": {"t": (np.arange(cv.size) * step * hop_s).tolist(),
                      "v": np.clip(cv, 0, 25).tolist()}}


def analyze(path, on_progress=None, cancel=None) -> dict:
    if on_progress is not None:
        on_progress("decoding audio")
    x = va_audio.decode_pcm(path, sr=SR, mono=True, cancel=cancel)
    if x is None:
        err = None
        try:
            err = va_audio.last_error()
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "error": err or "could not decode audio"}
    if on_progress is not None:
        on_progress("scanning spectrum")
    return analyze_samples(x, SR)


def marks(res: dict) -> list:
    if not res or not res.get("ok"):
        return []
    return [(f["t"], "splice?" if f["severity"] == "warn"
             else "jump") for f in res["findings"]]


def _mmss(t: float) -> str:
    return "%d:%05.2f" % (int(t) // 60, t - int(t) // 60 * 60)


def render_report(res: dict) -> str:
    if not res.get("ok"):
        return "Music continuity scan failed: %s" % res.get("error", "?")
    lines = ["MUSIC-BED CONTINUITY SCAN  (%.1f s audio, hop %.0f ms, "
             "threshold x%.0f local dev)"
             % (res["duration_s"], res["hop_s"] * 1000, res["thr"]), ""]
    if not res["findings"]:
        lines.append("No suspicious discontinuities in the music bed.")
    for f in res["findings"]:
        lines.append("[%s] %-8s %s" % (f["severity"].upper(), _mmss(f["t"]),
                                       f["text"]))
    lines += ["",
              "A splice usually cuts game music mid-phrase: one-hop spectral",
              "jumps that local musical activity cannot explain. Loud SFX and",
              "legitimate track changes also trigger - step through the video",
              "at each mark before drawing conclusions."]
    return "\n".join(lines)


def plot(res: dict, w: int = 1100, h: int = 240):
    """Score curve as an RGB image in the house dark style (cv2 optional)."""
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return None
    img = np.full((h, w, 3), (16, 18, 20), np.uint8)
    if not res.get("ok"):
        return img
    t = np.asarray(res["curve"]["t"], np.float32)
    v = np.asarray(res["curve"]["v"], np.float32)
    dur = max(res["duration_s"], 1e-6)
    top, bot, lf, rt = 18, h - 22, 36, w - 8
    vmax = 25.0
    grid = 60 if dur > 300 else 30 if dur > 90 else 10
    for gs in range(0, int(dur) + 1, grid):
        x0 = int(lf + gs / dur * (rt - lf))
        cv2.line(img, (x0, top), (x0, bot), (34, 38, 42), 1)
        cv2.putText(img, "%d:%02d" % (gs // 60, gs % 60), (x0 + 2, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 116, 122), 1, cv2.LINE_AA)
    ty = int(bot - res["thr"] / vmax * (bot - top))
    for x0 in range(lf, rt, 9):
        cv2.line(img, (x0, ty), (x0 + 4, ty), (240, 192, 64), 1)
    if t.size > 1:
        xs = (lf + t / dur * (rt - lf)).astype(int)
        ys = (bot - np.clip(v, 0, vmax) / vmax * (bot - top)).astype(int)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (78, 201, 176), 1, cv2.LINE_AA)
    for f in res["findings"]:
        x0 = int(lf + f["t"] / dur * (rt - lf))
        col = (235, 100, 100) if f["severity"] == "warn" else (240, 192, 64)
        cv2.line(img, (x0, top), (x0, bot), col, 1)
    cv2.putText(img, "music-bed novelty (local-dev x) - amber dash = flag "
                "threshold, red = suspected splice", (lf, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 176, 182), 1, cv2.LINE_AA)
    return img
