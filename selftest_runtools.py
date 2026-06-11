"""Headless self-test for the speedrun plugin's workbench modules (sr_luck,
sr_loads, sr_music). Companion to selftest.py; run directly:

    python selftest_runtools.py

Needs numpy; the va_loads end-to-end leg also needs ffmpeg (it synthesises a
small test video) and is skipped when ffmpeg is missing."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import va_plugins

va_luck = va_plugins.load_module("speedrun", "sr_luck")
va_loads = va_plugins.load_module("speedrun", "sr_loads")
va_music = va_plugins.load_module("speedrun", "sr_music")
if not all((va_luck, va_loads, va_music)):
    print("speedrun plugin not installed - nothing to test")
    sys.exit(0)

FAILS = []


def check(name, cond):
    print("  [%s] %s" % ("ok" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


def t_luck():
    print("va_luck:")
    from fractions import Fraction

    def brute(n, k, p):
        fp = Fraction(p).limit_denominator(10 ** 9)
        return float(sum(math.comb(n, i) * fp ** i * (1 - fp) ** (n - i)
                         for i in range(k, n + 1)))

    for n, k, p in [(10, 3, 0.5), (20, 20, 0.5), (262, 42, 0.0473),
                    (1000, 600, 0.5), (100, 1, 0.01)]:
        got = 10 ** va_luck.binom_tail_log10(n, k, p)
        exp = brute(n, k, p)
        check("tail(%d,%d,%g) exact" % (n, k, p),
              abs(got - exp) <= 1e-9 * max(exp, 1e-300))
    check("k=0 is certainty", va_luck.binom_tail_log10(10, 0, 0.5) == 0.0)
    check("k>n impossible", va_luck.binom_tail_log10(10, 11, 0.5) == -math.inf)
    check("huge-n approx sane",
          -4.3e6 < va_luck.binom_tail_log10(10 ** 8, 6 * 10 ** 7, 0.5) < -8e5)
    res = va_luck.evaluate([{"label": "pearls", "n": 262, "k": 42,
                             "p": 0.0473, "select": 1000}])
    check("verdict bands", res["verdict"] in ("suspicious", "implausible"))
    check("report renders", "Caveats" in va_luck.render_report(res))


def t_loads_core():
    print("va_loads core:")
    fps, n = 30.0, 300
    lumas = np.full(n, 120.0)
    diffs = np.full(n, 5.0)
    refd = np.full(n, 1e9)
    lumas[60:90] = 5.0
    diffs[150:210] = 0.2
    diffs[250:253] = 0.2          # too short to count
    segs = va_loads.segments(
        va_loads.classify(lumas, diffs, refd, fps, {}), fps, {})
    check("kinds", [g["kind"] for g in segs] == ["black", "static"])
    check("bounds", segs and segs[0]["start"] == 60 and segs[0]["end"] == 89
          and segs[1]["start"] == 150 and segs[1]["end"] == 209)
    fake = {"ok": True, "segments": segs}
    check("range clip", va_loads.frames_in_range(fake, 80, 160) == 21)
    check("json-safe", all(isinstance(g["t0"], float) for g in segs))


def t_loads_e2e():
    print("va_loads end-to-end:")
    ff = shutil.which("ffmpeg")
    if not ff:
        print("  [skip] ffmpeg not on PATH")
        return
    tmp = tempfile.mkdtemp(prefix="va_st_loads_")
    vid = os.path.join(tmp, "loadtest.mp4")
    srcs = ["testsrc2=duration=2:size=320x240:rate=30",
            "color=black:duration=1:size=320x240:rate=30",
            "testsrc2=duration=2:size=320x240:rate=30",
            "smptebars=duration=1.5:size=320x240:rate=30",
            "testsrc2=duration=1:size=320x240:rate=30"]
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y"]
    for s in srcs:
        cmd += ["-f", "lavfi", "-i", s]
    cmd += ["-filter_complex", "[0][1][2][3][4]concat=n=5:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", vid]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        res = va_loads.scan(vid)
        check("scan ok", bool(res.get("ok")))
        blk = [g for g in res["segments"] if g["kind"] == "black"]
        sta = [g for g in res["segments"] if g["kind"] == "static"]
        check("black @2-3s", len(blk) == 1 and abs(blk[0]["t0"] - 2.0) < 0.2
              and abs(blk[0]["t1"] - 3.0) < 0.2)
        check("static @5-6.5s",
              any(abs(g["t0"] - 5.0) < 0.3 and abs(g["t1"] - 6.5) < 0.3
                  for g in sta))
        from va_ffmpeg import VideoSource
        vs = VideoSource(vid, decode_max=256)
        ref = vs.frame_at(int(5.7 * vs.fps))
        vs.close()
        res2 = va_loads.scan(vid, refs=[va_loads.make_ref(ref)])
        check("ref match", any(g["kind"] == "match" and abs(g["t0"] - 5.0) < 0.3
                               for g in res2["segments"]))
        check("report renders", "Total detected loads" in
              va_loads.render_report(res2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _chord(rng, freqs, dur, sr, amp=0.25):
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * amp / len(freqs)
            for f in freqs)
    x *= 1.0 + 0.15 * np.sin(2 * np.pi * 0.7 * t)
    return (x + rng.normal(0, 0.012, x.size)).astype(np.float32)


def _xfade(a, b, n=4096):
    w = np.linspace(0, 1, n).astype(np.float32)
    return np.concatenate([a[:-n], a[-n:] * (1 - w) + b[:n] * w, b[n:]])


def t_music():
    print("va_music:")
    sr = va_music.SR
    rng = np.random.default_rng(7)
    CH1, CH2 = [220.0, 277.2, 329.6], [246.9, 311.1, 370.0]
    CH3, CH4 = [196.0, 246.9, 293.7], [261.6, 329.6, 392.0]
    ctrl = _chord(rng, CH1, 12, sr)
    for ch in (CH2, CH3, CH1, CH2):
        ctrl = _xfade(ctrl, _chord(rng, ch, 12, sr))
    res_c = va_music.analyze_samples(ctrl, sr)
    check("control clean", res_c["ok"] and not
          [f for f in res_c["findings"] if f["severity"] == "warn"])
    spl = np.concatenate([_chord(rng, CH1, 30, sr), _chord(rng, CH3, 20, sr)])
    res_s = va_music.analyze_samples(spl, sr)
    check("hard splice flagged",
          any(abs(f["t"] - 30.0) < 0.5 for f in res_s["findings"]))
    mid = _xfade(_xfade(_chord(rng, CH2, 5, sr), _chord(rng, CH4, 5, sr)),
                 _chord(rng, CH3, 5, sr))
    rst = np.concatenate([_chord(rng, CH1, 10, sr), mid, mid])
    res_r = va_music.analyze_samples(rst, sr)
    check("verbatim repeat matched",
          any("re-used" in f["text"] for f in res_r["findings"]))
    img = va_music.plot(res_s)
    check("plot renders", img is None or img.shape[2] == 3)
    check("report renders",
          "MUSIC-BED" in va_music.render_report(res_s))


if __name__ == "__main__":
    t_luck()
    t_loads_core()
    t_loads_e2e()
    t_music()
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        sys.exit(1)
    print("all speedrun-workbench checks passed")
