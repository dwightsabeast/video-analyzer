"""Burned-in timer / timecode OCR (calibration-based, dependency-free).

Reads a fixed-position on-screen counter - LiveSplit overlay, in-game timer,
SMPTE burn-in, CCTV timestamp - on every frame and audits it: backward jumps
(splice tell), forward skips, freezes, and CLOCK DRIFT: a least-squares slope
of timer-time vs video-time per continuous segment. A timer ticking at 95% of
video speed is the cleanest possible evidence of slowed-down footage.

No OCR engine needed: overlays use one fixed font, so the moderator
calibrates once - box the region, the glyphs are segmented automatically,
they type what the timer currently reads, and every glyph becomes a template.
Calibrate on a couple of frames until all ten digits have been seen.

Compute only - no Tk. Segmentation, the glyph bank, timer parsing and the
reading analysis are pure (selftest drives them with rendered frames)."""

from __future__ import annotations

import re

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

from va_ffmpeg import VideoSource

GH, GW = 18, 12          # canonical glyph bitmap size

OPTS = {
    "match_dist": 0.22,      # mean |a-b| (0..1) above this = unrecognised
    "min_ink": 4,            # pixels of ink for a column-run to be a glyph
    "back_jump_s": 0.25,     # timer decrease beyond this = backward jump
    "skip_factor": 4.0,      # forward skip if dTimer > factor*dVideo (+0.75s)
    "freeze_s": 3.0,         # timer static this long while video runs = note
    "drift_tol": 0.01,       # |slope-1| beyond this = clock drift
    "min_seg_s": 4.0,        # segments shorter than this don't get a slope
}


# --- image side -----------------------------------------------------------------

def _gray(frame):
    f = np.asarray(frame)
    return f.mean(axis=2).astype(np.float32) if f.ndim == 3 else f.astype(np.float32)


def binarize(gray) -> np.ndarray:
    """Ink mask for a high-contrast overlay region; auto polarity (glyphs are
    the minority class)."""
    g = np.asarray(gray, np.float32)
    lo, hi = np.percentile(g, 8), np.percentile(g, 92)
    thr = 0.5 * (lo + hi)
    ink = g > thr
    if ink.mean() > 0.5:
        ink = ~ink
    return ink


def _norm_glyph(ink) -> "np.ndarray | None":
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    crop = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype(np.float32)
    if cv2 is not None:
        return cv2.resize(crop, (GW, GH), interpolation=cv2.INTER_AREA)
    ys = np.linspace(0, crop.shape[0] - 1, GH).astype(int)
    xs = np.linspace(0, crop.shape[1] - 1, GW).astype(int)
    return crop[np.ix_(ys, xs)]


def segment_glyphs(region_gray, min_ink=OPTS["min_ink"]) -> list:
    """Left-to-right glyphs in the region: [{'x0','x1','bmp'}]."""
    ink = binarize(region_gray)
    colink = ink.sum(axis=0)
    on = colink > 0
    out = []
    x = 0
    W = on.size
    while x < W:
        if not on[x]:
            x += 1
            continue
        x0 = x
        while x < W and on[x]:
            x += 1
        if ink[:, x0:x].sum() >= min_ink:
            bmp = _norm_glyph(ink[:, x0:x])
            if bmp is not None:
                out.append({"x0": int(x0), "x1": int(x), "bmp": bmp})
    return out


class GlyphBank:
    """Calibrated glyph templates -> characters."""

    def __init__(self):
        self.chars: list = []
        self.bmps: list = []

    def add(self, bmp, ch):
        self.chars.append(str(ch))
        self.bmps.append(np.asarray(bmp, np.float32))

    def coverage(self) -> str:
        return "".join(sorted(set(self.chars)))

    def match(self, bmp, max_dist=OPTS["match_dist"]):
        if not self.bmps:
            return "?", 1.0
        b = np.asarray(bmp, np.float32)
        dists = [float(np.abs(b - t).mean()) for t in self.bmps]
        i = int(np.argmin(dists))
        return (self.chars[i], dists[i]) if dists[i] <= max_dist else ("?", dists[i])

    def calibrate(self, region_gray, text):
        """Map the region's glyphs (left to right) onto `text`. Raises
        ValueError with the counts when they disagree."""
        segs = segment_glyphs(region_gray)
        text = "".join(str(text).split())
        if len(segs) != len(text):
            raise ValueError("region shows %d glyph(s) but %d character(s) "
                             "were typed" % (len(segs), len(text)))
        for s, ch in zip(segs, text):
            self.add(s["bmp"], ch)
        return len(segs)

    def read(self, region_gray) -> "tuple[str, float]":
        """(string, mean_distance) for the region."""
        segs = segment_glyphs(region_gray)
        if not segs:
            return "", 1.0
        out, dists = [], []
        for s in segs:
            ch, d = self.match(s["bmp"])
            out.append(ch)
            dists.append(d)
        return "".join(out), float(np.mean(dists))


# --- timer parsing ----------------------------------------------------------------

_PATTERNS = (
    ("smpte", re.compile(r"^(\d+):([0-5]\d):([0-5]\d)[:;](\d\d)$")),
    ("hms", re.compile(r"^(\d+):([0-5]\d):([0-5]\d)(?:[.,](\d{1,3}))?$")),
    ("ms", re.compile(r"^(\d+):([0-5]\d)(?:[.,](\d{1,3}))?$")),
    ("s", re.compile(r"^(\d+)[.,](\d{1,3})$")),
    ("count", re.compile(r"^(\d{2,})$")),
)


def _frac(s):
    return int(s) / (10.0 ** len(s)) if s else 0.0


def parse_timer(s, fps=None) -> "tuple[str, float] | None":
    """(kind, seconds) or None. SMPTE frames and bare counters need fps."""
    s = (s or "").strip()
    if "?" in s or not s:
        return None
    for kind, rx in _PATTERNS:
        m = rx.match(s)
        if not m:
            continue
        g = m.groups()
        if kind == "smpte":
            if not fps:
                return None
            return kind, (int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])
                          + int(g[3]) / float(fps))
        if kind == "hms":
            return kind, (int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])
                          + _frac(g[3]))
        if kind == "ms":
            return kind, int(g[0]) * 60 + int(g[1]) + _frac(g[2])
        if kind == "s":
            return kind, int(g[0]) + _frac(g[1])
        if kind == "count":
            return (kind, int(g[0]) / float(fps)) if fps else None
    return None


# --- scan + analysis ---------------------------------------------------------------

def scan(path, rect, bank, on_progress=None, cancel=None) -> dict:
    """Read the boxed region on every frame. rect = (x0, y0, x1, y1) in
    decoded-frame coordinates (the GUI preview geometry)."""
    x0, y0, x1, y1 = [int(v) for v in rect]
    try:
        vs = VideoSource(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "cannot open: %r" % (exc,)}
    try:
        fps = vs.fps or 30.0
        tv, raw, secs = [], [], []
        vs.start(0)
        i = 0
        while True:
            if cancel is not None and cancel():
                return {"ok": False, "error": "cancelled"}
            fr = vs.read()
            if fr is None:
                break
            g = _gray(fr)[y0:y1, x0:x1]
            s, _d = bank.read(g)
            p = parse_timer(s, fps)
            tv.append(i / fps)
            raw.append(s)
            secs.append(p[1] if p else None)
            i += 1
            if on_progress is not None and i % 300 == 0:
                m, sec = divmod(int(i / fps), 60)
                on_progress("read %d:%02d" % (m, sec))
        if not i:
            return {"ok": False, "error": "no frames decoded"}
        res = analyze_readings(tv, secs, fps)
        res.update({"ok": True, "fps": fps, "frames": i,
                    "raw_preview": raw[:5] + ["..."] + raw[-3:] if i > 8 else raw})
        return res
    finally:
        vs.close()


def analyze_readings(tv, secs, fps, opts=None) -> dict:
    """Audit (video_t, timer_t|None) pairs: jumps, skips, freezes, drift."""
    o = dict(OPTS)
    o.update(opts or {})
    tv = np.asarray(tv, np.float64)
    n = tv.size
    valid = [(tv[i], secs[i]) for i in range(n) if secs[i] is not None]
    events = []
    read_rate = len(valid) / max(1, n)
    segments = []        # runs of consistent forward motion for drift fitting
    cur = [valid[0]] if valid else []
    last_change = valid[0] if valid else None
    for (va, ta), (vb, tb) in zip(valid, valid[1:]):
        dvv, dtt = vb - va, tb - ta
        if dtt < -o["back_jump_s"]:
            events.append({"t": float(vb), "severity": "warn",
                           "text": "timer jumps BACKWARD %.2fs (%.2f -> %.2f) - "
                                   "splice or reset" % (-dtt, ta, tb)})
            segments.append(cur)
            cur = [(vb, tb)]
        elif dtt > o["skip_factor"] * max(dvv, 1.0 / fps) + 0.75:
            events.append({"t": float(vb), "severity": "warn",
                           "text": "timer skips ahead %.2fs in %.2fs of video - "
                                   "removed gameplay or timer catch-up"
                                   % (dtt, dvv)})
            segments.append(cur)
            cur = [(vb, tb)]
        else:
            cur.append((vb, tb))
        if tb != ta:
            if (last_change is not None
                    and vb - last_change[0] > o["freeze_s"]
                    and tb == last_change[1]):
                pass
            last_change = (vb, tb)
        elif last_change is not None and vb - last_change[0] > o["freeze_s"]:
            events.append({"t": float(last_change[0]), "severity": "info",
                           "text": "timer frozen for %.1fs while video runs "
                                   "(pause/load, or a paused timer overlay)"
                                   % (vb - last_change[0])})
            last_change = (vb, tb)
    segments.append(cur)

    def _split(seg, depth=0):
        """Piecewise-linear changepoint split: a timer that changes speed
        mid-segment (no jump) still gets per-rate spans."""
        n2 = len(seg)
        if depth >= 3 or n2 < 24:
            return [seg]
        a = np.asarray(seg)

        def sse(part):
            if len(part) < 4:
                return 0.0
            x = part[:, 0] - part[:, 0].mean()
            y = part[:, 1] - part[:, 1].mean()
            d = float((x * x).sum())
            if d <= 0:
                return 0.0
            sl = float((x * y).sum() / d)
            r = y - sl * x
            return float((r * r).sum())

        whole = sse(a)
        if whole < 1e-3 * n2:
            return [seg]
        best, best_i = whole, None
        step = max(8, n2 // 40)
        for i in range(12, n2 - 12, step):
            s2 = sse(a[:i]) + sse(a[i:])
            if s2 < best:
                best, best_i = s2, i
        if best_i is None or best > whole / 4.0:
            return [seg]
        return _split(seg[:best_i], depth + 1) + _split(seg[best_i:], depth + 1)

    segments = [p for seg in segments for p in _split(seg)]
    slopes = []
    for seg in segments:
        if len(seg) < 8:
            continue
        a = np.asarray(seg)
        if a[-1, 0] - a[0, 0] < o["min_seg_s"]:
            continue
        x = a[:, 0] - a[:, 0].mean()
        y = a[:, 1] - a[:, 1].mean()
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        slope = float((x * y).sum() / denom)
        span = float(a[-1, 0] - a[0, 0])
        slopes.append((slope, span, float(a[0, 0]), float(a[-1, 0])))
        if abs(slope - 1.0) > o["drift_tol"]:
            events.append({"t": float(a[0, 0]), "severity": "warn",
                           "text": "timer runs at %.2f%% of video speed over "
                                   "%d:%02d-%d:%02d - footage speed change or "
                                   "VFR mismatch"
                                   % (100 * slope, a[0, 0] // 60, a[0, 0] % 60,
                                      a[-1, 0] // 60, a[-1, 0] % 60)})
    overall = (sum(s * w for s, w, *_ in slopes)
               / max(1e-9, sum(w for _s, w, *_ in slopes))) if slopes else None
    events.sort(key=lambda e: e["t"])
    # drift residual curve: (timer elapsed) - (video elapsed); flat = honest,
    # sloped = speed change, steps = jumps. Feeds the timeline as a series.
    series = {"t": [], "v": []}
    if valid:
        v0, t0 = valid[0]
        step = max(1, len(valid) // 6000)
        for va, ta in valid[::step]:
            series["t"].append(float(va))
            series["v"].append(float((ta - t0) - (va - v0)))
    return {"read_rate": round(read_rate, 3), "events": events,
            "series": series,
            "segments": [{"slope": round(s, 4), "span_s": round(w, 2),
                          "t0": round(t0, 2), "t1": round(t1, 2)}
                         for s, w, t0, t1 in slopes],
            "overall_slope": (round(overall, 4) if overall is not None else None)}


def marks(res: dict) -> list:
    if not res or not res.get("ok"):
        return []
    return [(e["t"], e["text"][:40]) for e in res["events"]][:200]


def render_report(res: dict) -> str:
    if not res.get("ok"):
        return "Timer OCR scan failed: %s" % res.get("error", "?")
    lines = ["BURNED-IN TIMER AUDIT  (%d frames @ %.3f fps, %.0f%% readable)"
             % (res["frames"], res["fps"], res["read_rate"] * 100), ""]
    if res.get("overall_slope") is not None:
        lines.append("timer speed vs video : %.2f%%  (100%% = honest clock)"
                     % (100 * res["overall_slope"]))
    for s in res.get("segments", []):
        lines.append("  segment %7.2fs-%7.2fs  slope %.4f"
                     % (s["t0"], s["t1"], s["slope"]))
    lines.append("")
    if not res["events"]:
        lines.append("Timer is monotonic, continuous, and tracks video time.")
    for e in res["events"][:60]:
        m, s = divmod(e["t"], 60.0)
        lines.append("[%s] %d:%05.2f  %s" % (e["severity"].upper(),
                                             int(m), s, e["text"]))
    lines += ["", "Backward jumps and skips are splice-class evidence; a "
                  "consistent off-100% slope is the cleanest slowdown tell.",
              "Freezes are normal where the overlay pauses on loads."]
    return "\n".join(lines)
