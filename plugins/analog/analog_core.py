"""Analog / tape artifact detection (VHS, capture-card, time-base errors).

Digitised analog video carries signatures no digital pipeline produces:
dropout streaks (bright 1-2 line horizontal flashes where the tape lost
oxide), head-switching noise (a torn band in the last lines of every field,
below the visible area on a CRT but glaring on a flat panel), time-base
jitter (per-line horizontal wobble a TBC failed to remove), and AGC/film
flicker (slow global luma pumping). This module scans the whole file at
decode resolution and scores each family per frame, merging per-frame hits
into timeline events.

Compute only - no Tk. The detectors (dropout_rows, head_switch_ratio,
jitter_metric, flicker_score) are pure numpy on frame arrays so the selftest
drives them with synthetic artifacts."""

from __future__ import annotations

import numpy as np

from va_ffmpeg import VideoSource

OPTS = {
    "dropout_z": 7.0,        # row deviation (MAD units) to call a dropout row
    "dropout_min_cols": 0.45,  # fraction of row width the streak must span
    "head_ratio": 2.6,       # bottom-band HF energy vs body to flag head-switch
    "head_rows": 6,          # rows in the bottom band
    "jitter_px": 0.55,       # mean |row shift| (px) to flag time-base wobble
    "flicker_hz": (0.2, 4.0),  # AGC/film flicker search band
    "flicker_ratio": 30.0,   # in-band peak vs median AC power (tone-ness)
    "min_event_s": 0.0,      # dropouts are single-frame events; keep all
}


def _gray(frame):
    f = np.asarray(frame)
    return f.mean(axis=2).astype(np.float32) if f.ndim == 3 else f.astype(np.float32)


def dropout_rows(gray, z=OPTS["dropout_z"], min_cols=OPTS["dropout_min_cols"],
                 skip_bottom=10):
    """Rows where luma rides far above the local vertical neighbourhood across
    most of the width - the classic bright dropout streak. The bottom band is
    excluded (head-switching noise lives there and has its own detector).
    Returns row indices."""
    g = gray
    if g.shape[0] < 9 + skip_bottom:
        return []
    up = np.roll(g, 2, axis=0)
    dn = np.roll(g, -2, axis=0)
    local = np.minimum(up, dn)                 # streaks are 1-2 lines: 2 away is clean
    resid = g - local
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) + 1e-3
    hot = (resid - med) / (1.4826 * mad) > z
    frac = hot[2:-skip_bottom].mean(axis=1)
    rows = np.flatnonzero(frac > min_cols) + 2
    # merge adjacent rows (a streak is one event even if 2 lines tall)
    out, last = [], -10
    for r in rows:
        if r - last > 2:
            out.append(int(r))
        last = r
    return out


def head_switch_ratio(gray, band=OPTS["head_rows"]):
    """Horizontal high-frequency energy in the bottom band vs the body.
    VHS head switching lives in the last ~5 lines of every frame."""
    g = gray
    if g.shape[0] < band * 4:
        return 0.0
    hf = np.abs(np.diff(g, axis=1))
    bottom = float(hf[-band:].mean())
    body = float(hf[g.shape[0] // 4: -band * 2].mean()) + 1e-3
    return bottom / body


def jitter_metric(gray, prev_gray):
    """Mean |per-row horizontal shift| between consecutive frames, estimated
    from row-wise gradient correlation at offsets -2..2 px with parabolic
    refinement. Global pans move every row together and are subtracted; what
    remains is line wobble = time-base error."""
    if prev_gray is None or gray.shape != prev_gray.shape:
        return 0.0
    a = np.diff(gray, axis=1)
    b = np.diff(prev_gray, axis=1)
    h = a.shape[0]
    rows = np.arange(8, h - 8, max(1, h // 48))   # sample rows
    shifts = []
    for r in rows:
        va, vb = a[r], b[r]
        if va.std() < 1.0 or vb.std() < 1.0:
            continue                                # flat row: no signal
        cors = [float(np.dot(va[4:-4], np.roll(vb, s)[4:-4])) for s in (-2, -1, 0, 1, 2)]
        i = int(np.argmax(cors))
        s = i - 2
        if 0 < i < 4:                               # parabolic sub-pixel peak
            c0, c1, c2 = cors[i - 1], cors[i], cors[i + 1]
            den = c0 - 2 * c1 + c2
            if den < 0:
                s += 0.5 * (c0 - c2) / den
        shifts.append(s)
    if len(shifts) < 6:
        return 0.0
    sh = np.array(shifts, np.float32)
    return float(np.mean(np.abs(sh - np.median(sh))))   # remove global pan


def flicker_score(means, fps, band=OPTS["flicker_hz"]):
    """Tonal flicker in the frame-mean-luma series: the in-band spectral PEAK
    against the median AC power (a flat noise floor scores ~5-10; AGC pumping
    or film flicker concentrates power in one tone and scores hundreds).
    Returns (peak_ratio, peak_hz)."""
    x = np.asarray(means, np.float32)
    if x.size < int(4 * fps):
        return 0.0, 0.0
    x = x - x.mean()
    win = np.hanning(x.size).astype(np.float32)
    mag = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fps)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return 0.0, 0.0
    med = float(np.median(mag[1:])) + 1e-9
    i = int(np.argmax(mag * m))
    return float(mag[i] / med), float(freqs[i])


def scan(path, opts=None, on_progress=None, cancel=None) -> dict:
    """Whole-file analog-artifact sweep. Returns events + per-family stats."""
    o = dict(OPTS)
    o.update(opts or {})
    try:
        vs = VideoSource(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "cannot open: %r" % (exc,)}
    try:
        fps = vs.fps or 30.0
        events = []
        means = []
        jit_series = []
        head_hits = jit_hits = 0
        drop_frames = 0
        prev = None
        vs.start(0)
        i = 0
        while True:
            if cancel is not None and cancel():
                return {"ok": False, "error": "cancelled"}
            fr = vs.read()
            if fr is None:
                break
            g = _gray(fr)
            means.append(float(g.mean()))
            rows = dropout_rows(g, o["dropout_z"], o["dropout_min_cols"])
            if rows:
                drop_frames += 1
                events.append({"t": i / fps, "kind": "dropout", "severity": "warn",
                               "text": "dropout streak%s at row%s %s"
                                       % ("s" if len(rows) > 1 else "",
                                          "s" if len(rows) > 1 else "",
                                          ",".join(str(r) for r in rows[:4]))})
            hr = head_switch_ratio(g, o["head_rows"])
            if hr > o["head_ratio"]:
                head_hits += 1
            jm = jitter_metric(g, prev)
            jit_series.append(float(jm))
            if jm > o["jitter_px"]:
                jit_hits += 1
                if jit_hits in (1, 5) or jit_hits % 60 == 0:
                    events.append({"t": i / fps, "kind": "jitter",
                                   "severity": "info",
                                   "text": "line wobble %.2f px (time-base error)" % jm})
            prev = g
            i += 1
            if on_progress is not None and i % 300 == 0:
                m, s = divmod(int(i / fps), 60)
                on_progress("scanned %d:%02d" % (m, s))
        if not i:
            return {"ok": False, "error": "no frames decoded"}
        head_frac = head_hits / i
        jit_frac = jit_hits / i
        fl_ratio, fl_hz = flicker_score(means, fps, o["flicker_hz"])
        if head_frac > 0.5:
            events.insert(0, {"t": 0.0, "kind": "head_switch", "severity": "warn",
                              "text": "head-switching noise in the bottom %d lines "
                                      "on %d%% of frames (VHS/tape source; crop or "
                                      "mask before publishing)"
                                      % (o["head_rows"], round(head_frac * 100))})
        if jit_frac > 0.25:
            events.insert(0, {"t": 0.0, "kind": "jitter", "severity": "warn",
                              "text": "horizontal line wobble on %d%% of frames - "
                                      "time-base error (no/weak TBC in the capture "
                                      "chain)" % round(jit_frac * 100)})
        if fl_ratio > o["flicker_ratio"]:
            events.insert(0, {"t": 0.0, "kind": "flicker", "severity": "info",
                              "text": "global luma flicker tone at ~%.1f Hz (%.0fx "
                                      "the noise floor) - AGC pumping or "
                                      "film-transfer flicker" % (fl_hz, fl_ratio)})
        step = max(1, len(jit_series) // 6000)
        sub = jit_series[::step]
        return {"ok": True, "fps": fps, "frames": i,
                "series": {"t": [j * step / fps for j in range(len(sub))],
                           "v": sub},
                "events": events,
                "stats": {"dropout_frames": drop_frames,
                          "dropout_pct": round(100.0 * drop_frames / i, 2),
                          "head_switch_pct": round(100.0 * head_frac, 1),
                          "jitter_pct": round(100.0 * jit_frac, 1),
                          "flicker_peak_ratio": round(fl_ratio, 1),
                          "flicker_hz": round(fl_hz, 2)}}
    finally:
        vs.close()


def marks(res: dict) -> list:
    if not res or not res.get("ok"):
        return []
    return [(e["t"], "%s" % e["kind"]) for e in res["events"] if e["t"] > 0.0][:200]


def render_report(res: dict) -> str:
    if not res.get("ok"):
        return "Analog artifact scan failed: %s" % res.get("error", "?")
    st = res["stats"]
    lines = ["ANALOG / TAPE ARTIFACT SCAN  (%d frames @ %.3f fps)"
             % (res["frames"], res["fps"]), "",
             "dropout streaks : %d frame(s)  (%.2f%%)"
             % (st["dropout_frames"], st["dropout_pct"]),
             "head-switching  : %.1f%% of frames" % st["head_switch_pct"],
             "line jitter     : %.1f%% of frames" % st["jitter_pct"],
             "luma flicker    : peak %.0fx noise floor at %.2f Hz"
             % (st["flicker_peak_ratio"], st["flicker_hz"]), ""]
    if not res["events"]:
        lines.append("No analog artifacts detected - consistent with a digital "
                     "source or a well-restored transfer.")
    for e in res["events"][:60]:
        m, s = divmod(e["t"], 60.0)
        lines.append("[%s] %d:%05.2f  %-12s %s"
                     % (e["severity"].upper(), int(m), s, e["kind"], e["text"]))
    if len(res["events"]) > 60:
        lines.append("... %d more events (all marked on the timeline)"
                     % (len(res["events"]) - 60))
    lines += ["", "Field-order / interlace / telecine analysis lives in the core",
              "(Analyze pass + cadence); this scan covers tape-era artifacts."]
    return "\n".join(lines)


# --- QC checks (profile "analog", registered by plugin.py) --------------------

def _rep(ctx):
    """Scan results: own ctx key (CLI), the GUI's cached-results bridge,
    or the legacy shared key."""
    return (ctx.get("analog") or (ctx.get("adv") or {}).get("analog")
            or ctx.get("verify") or {})


def chk_dropouts(warn_pct=0.5):
    def f(ctx):
        st = _rep(ctx).get("stats")
        if not st:
            return "pass", "not measured (run the analog scan)"
        if st["dropout_pct"] > warn_pct:
            return "warn", "dropout streaks on %.2f%% of frames" % st["dropout_pct"]
        return "pass", "%d dropout frame(s)" % st["dropout_frames"]
    return ("dropouts", "Tape dropouts", f)


def chk_head_switch(warn_pct=50.0):
    def f(ctx):
        st = _rep(ctx).get("stats")
        if not st:
            return "pass", "not measured (run the analog scan)"
        if st["head_switch_pct"] > warn_pct:
            return "warn", ("head-switching noise on %.0f%% of frames - "
                            "crop/mask bottom lines" % st["head_switch_pct"])
        return "pass", "bottom-band energy normal"
    return ("head_switch", "Head-switching band", f)


def chk_jitter(warn_pct=25.0):
    def f(ctx):
        st = _rep(ctx).get("stats")
        if not st:
            return "pass", "not measured (run the analog scan)"
        if st["jitter_pct"] > warn_pct:
            return "warn", "line wobble on %.0f%% of frames (TBC needed)" % st["jitter_pct"]
        return "pass", "line timing stable"
    return ("jitter", "Time-base stability", f)


def chk_flicker(warn_ratio=30.0):
    def f(ctx):
        st = _rep(ctx).get("stats")
        if not st:
            return "pass", "not measured (run the analog scan)"
        if st["flicker_peak_ratio"] > warn_ratio:
            return "warn", ("luma flicker tone at %.2f Hz (%.0fx noise floor)"
                            % (st["flicker_hz"], st["flicker_peak_ratio"]))
        return "pass", "no AGC/film flicker"
    return ("flicker", "Luma flicker", f)


def qc_checks():
    return [chk_dropouts(), chk_head_switch(), chk_jitter(), chk_flicker()]
