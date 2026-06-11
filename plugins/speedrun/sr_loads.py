"""Load-screen detection and load-removed time (LRT).

Hardware speed leaks into load screens, so many leaderboards time runs with
loads removed - live that is LiveSplit load-removers / AutoSplit territory.
This module does it after the fact, from the video alone: it sweeps the file
at reduced resolution and flags near-black frames, frozen frames, and frames
matching moderator-captured reference load screens (AutoSplit-style image
compare). Segments land on the timeline as marks and the retimer subtracts
them from a marked range to quote LRT next to RTA.

Compute only - no Tk in here. The detection core (classify / segments) is
pure numpy so the selftest can drive it with synthetic arrays."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001 - degraded fallback below
    cv2 = None

from va_ffmpeg import VideoSource

SIG_W, SIG_H = 48, 27        # tiny luma signature: cheap, codec-noise tolerant

OPTS = {
    "black_thr": 18.0,       # mean luma (0-255) below this = black frame
    "static_thr": 0.9,       # mean |Δsig| below this = frozen frame
    "static_min_s": 0.50,    # frozen stretch must last this long
    "ref_thr": 14.0,         # mean |sig - ref| below this = load-screen match
    "min_s": 0.30,           # drop blips shorter than this
    "merge_gap_s": 0.15,     # bridge same-kind gaps up to this long
}
PRESETS = {
    "strict": {"black_thr": 12.0, "static_thr": 0.55, "ref_thr": 10.0,
               "static_min_s": 0.80},
    "normal": {},
    "loose":  {"black_thr": 26.0, "static_thr": 1.60, "ref_thr": 20.0,
               "static_min_s": 0.35},
}
KIND = {1: "black", 2: "match", 3: "static"}


def sig_of(frame) -> "np.ndarray | None":
    """Reduce a frame (HxW or HxWx3, any size) to the tiny luma signature."""
    if frame is None:
        return None
    f = np.asarray(frame)
    if f.ndim == 3:
        f = f.mean(axis=2)
    f = f.astype(np.float32)
    if cv2 is not None:
        return cv2.resize(f, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
    ys = np.linspace(0, f.shape[0] - 1, SIG_H).astype(int)
    xs = np.linspace(0, f.shape[1] - 1, SIG_W).astype(int)
    return f[np.ix_(ys, xs)]


make_ref = sig_of    # captured reference load screens use the same signature


def _runs(mask: np.ndarray):
    """[(start, end_inclusive)] for each True run."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(idx[0::2], idx[1::2] - 1))


def classify(lumas, diffs, refd, fps, opts) -> np.ndarray:
    """Per-frame load kind: 0 none, 1 black, 2 ref-match, 3 static.
    Priority match > black > static (a captured ref usually IS black/static)."""
    o = dict(OPTS)
    o.update(opts or {})
    lumas = np.asarray(lumas, np.float32)
    diffs = np.asarray(diffs, np.float32)
    refd = np.asarray(refd, np.float32)
    kinds = np.zeros(lumas.shape[0], np.int8)
    need = max(1, int(round(float(o["static_min_s"]) * (fps or 30.0))))
    frozen = diffs < float(o["static_thr"])
    for s, e in _runs(frozen):
        if e - s + 1 >= need:
            kinds[s:e + 1] = 3
    kinds[lumas < float(o["black_thr"])] = 1
    kinds[refd < float(o["ref_thr"])] = 2
    return kinds


def segments(kinds: np.ndarray, fps: float, opts=None) -> list:
    """Merge per-frame kinds into [{start,end,frames,kind,t0,t1}] segments,
    bridging small gaps and dropping blips."""
    o = dict(OPTS)
    o.update(opts or {})
    fps = fps or 30.0
    k = np.asarray(kinds, np.int8).copy()
    gap = int(round(float(o["merge_gap_s"]) * fps))
    if gap > 0:
        for s, e in _runs(k == 0):
            if (e - s + 1 <= gap and s > 0 and e < k.shape[0] - 1
                    and k[s - 1] == k[e + 1]):
                k[s:e + 1] = k[s - 1]
    need = max(1, int(round(float(o["min_s"]) * fps)))
    out = []
    for s, e in _runs(k != 0):
        if e - s + 1 < need:
            continue
        vals, counts = np.unique(k[s:e + 1], return_counts=True)
        kind = KIND.get(int(vals[np.argmax(counts)]), "load")
        out.append({"start": int(s), "end": int(e), "frames": int(e - s + 1),
                    "kind": kind, "t0": float(s / fps), "t1": float((e + 1) / fps)})
    return out


def scan(path, refs=None, preset="normal", opts=None,
         on_progress=None, cancel=None) -> dict:
    """Sweep the whole file at reduced resolution and detect load screens.
    ``refs`` are signatures from make_ref() (or raw frames - both accepted)."""
    o = dict(OPTS)
    o.update(PRESETS.get(preset or "normal", {}))
    o.update(opts or {})
    rs = []
    for r in refs or []:
        r = np.asarray(r)
        rs.append(r.astype(np.float32) if r.shape == (SIG_H, SIG_W)
                  else sig_of(r))
    try:
        vs = VideoSource(path, decode_max=256)
    except Exception as exc:  # noqa: BLE001 - decode-layer surprises
        return {"ok": False, "error": "cannot open: %r" % (exc,)}
    try:
        fps = vs.fps or 30.0
        lumas, diffs, refd = [], [], []
        prev = None
        vs.start(0)
        i = 0
        while True:
            if cancel is not None and cancel():
                return {"ok": False, "error": "cancelled"}
            fr = vs.read()
            if fr is None:
                break
            s = sig_of(fr)
            lumas.append(float(s.mean()))
            diffs.append(float(np.abs(s - prev).mean()) if prev is not None
                         else 1e9)
            refd.append(min((float(np.abs(s - r).mean()) for r in rs),
                            default=1e9))
            prev = s
            i += 1
            if on_progress is not None and i % 600 == 0:
                m, sec = divmod(int(i / fps), 60)
                on_progress("scanned %d:%02d" % (m, sec))
        if not i:
            return {"ok": False, "error": "no frames decoded"}
        kinds = classify(lumas, diffs, refd, fps, o)
        segs = segments(kinds, fps, o)
        lf = sum(g["frames"] for g in segs)
        return {"ok": True, "fps": fps, "frames_scanned": i,
                "segments": segs, "load_frames": lf, "load_s": lf / fps,
                "refs_used": len(rs), "preset": preset, "opts": o}
    finally:
        vs.close()


def frames_in_range(res: dict, f0: int, f1: int) -> int:
    """Detected load frames inside [f0, f1] (inclusive), for LRT maths."""
    if not res or not res.get("ok"):
        return 0
    lo, hi = min(f0, f1), max(f0, f1)
    return sum(max(0, min(hi, g["end"]) - max(lo, g["start"]) + 1)
               for g in res["segments"])


def marks(res: dict) -> list:
    """[(t_seconds, label)] timeline marks, one per segment."""
    if not res or not res.get("ok"):
        return []
    return [(g["t0"], "%s %.2fs" % (g["kind"], g["frames"] / res["fps"]))
            for g in res["segments"]]


def _hms(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    return "%d:%02d:%02d.%03d" % (s // 3600, s % 3600 // 60, s % 60, ms)


def render_report(res: dict) -> str:
    if not res.get("ok"):
        return "Load scan failed: %s" % res.get("error", "?")
    fps = res["fps"]
    lines = ["LOAD-SCREEN SCAN  (%d frames @ %.3f fps, preset %s, %d reference%s)"
             % (res["frames_scanned"], fps, res.get("preset", "?"),
                res["refs_used"], "" if res["refs_used"] == 1 else "s"), ""]
    if not res["segments"]:
        lines.append("No load screens detected. Loosen the preset or capture a")
        lines.append("reference frame of this game's load screen and rescan.")
        return "\n".join(lines)
    lines.append("%-4s %-13s %-13s %8s %8s  %s"
                 % ("#", "from", "to", "frames", "secs", "kind"))
    for n, g in enumerate(res["segments"], 1):
        lines.append("%-4d %-13s %-13s %8d %8.2f  %s"
                     % (n, _hms(g["t0"]), _hms(g["t1"]), g["frames"],
                        g["frames"] / fps, g["kind"]))
    lines += ["", "Total detected loads: %d frames = %s"
              % (res["load_frames"], _hms(res["load_s"])),
              "",
              "kinds: black = near-black frame, match = looks like a captured",
              "reference load screen, static = frozen picture (pause menus and",
              "cutscene holds also read as static - eyeball before quoting LRT)."]
    return "\n".join(lines)
