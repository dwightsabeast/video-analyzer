"""Headless self-test for the domain-pack plugins: analog artifact detector,
captions QC, and timer OCR. Skips any pack that isn't installed and any
media-fixture leg when ffmpeg is missing.

    python selftest_packs.py [--only=analog,captions,ocr]"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

import va_plugins

FAILS = []
ONLY = None
for a in sys.argv[1:]:
    if a.startswith("--only="):
        ONLY = set(a.split("=", 1)[1].split(","))


def want(name):
    return ONLY is None or name in ONLY


def check(name, cond):
    print("  [%s] %s" % ("ok" if cond else "FAIL", name))
    if not cond:
        FAILS.append(name)


FFMPEG = shutil.which("ffmpeg")


def write_gray_video(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-s", "%dx%d" % (w, h),
                          "-r", str(fps), "-i", "-", "-c:v", "libx264",
                          "-preset", "ultrafast", "-qp", "0", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0


# === analog =====================================================================

def t_analog(tmp):
    A = va_plugins.load_module("analog", "analog_core")
    if A is None:
        print("analog: [skip] plugin not installed")
        return
    print("analog:")
    rng = np.random.default_rng(3)
    base = np.tile(np.linspace(40, 180, 360)[:, None], (1, 480)).astype(np.float32)
    base += rng.normal(0, 2, base.shape)
    hit = base.copy()
    hit[120, 40:440] += 95.0
    check("clean has no dropouts", A.dropout_rows(base) == [])
    check("streak found", any(abs(r - 120) <= 1 for r in A.dropout_rows(hit)))
    nb = base.copy()
    nb[-6:] = rng.uniform(0, 255, (6, 480))
    check("head band not a dropout", A.dropout_rows(nb) == [])
    check("head-switch ratio", A.head_switch_ratio(base) < 1.6
          and A.head_switch_ratio(nb) > 2.6)
    stripes = (np.indices((360, 480))[1] // 6 % 2 * 120 + 60).astype(np.float32)
    stripes += rng.normal(0, 2, stripes.shape)
    wob = stripes.copy()
    for r in range(360):
        wob[r] = np.roll(stripes[r], int(rng.integers(-1, 2)))
    check("jitter metric", A.jitter_metric(stripes, stripes.copy()) < 0.15
          and A.jitter_metric(wob, stripes) > 0.5)
    t = np.arange(300) / 30.0
    fl, hz = A.flicker_score(120 + 6 * np.sin(2 * np.pi * t)
                             + rng.normal(0, .3, 300), 30.0)
    fc, _ = A.flicker_score(120 + rng.normal(0, .3, 300), 30.0)
    check("flicker tone vs noise", fl > 100 and abs(hz - 1.0) < 0.2 and fc < 25)
    if not FFMPEG:
        print("  [skip] e2e (no ffmpeg)")
        return
    tex = (np.indices((360, 480))[1] // 6 % 2 * 100 + 70).astype(np.float32)
    tex += np.tile(np.linspace(-20, 20, 360)[:, None], (1, 480))
    drop_at = set(range(20, 260, 20))
    vhs, ctl = [], []
    for i in range(300):
        f = tex + rng.normal(0, 2.5, tex.shape)
        ctl.append(f.copy())
        g = f.copy()
        for r in range(360):
            g[r] = np.roll(g[r], int(rng.integers(-1, 2)))
        g[-5:] = rng.uniform(0, 255, (5, 480))
        if i in drop_at:
            row = int(rng.integers(40, 320))
            g[row, 20:460] = np.minimum(g[row, 20:460] + 100, 255)
        vhs.append(g)
    pv, pc = os.path.join(tmp, "vhs.mp4"), os.path.join(tmp, "ctl.mp4")
    write_gray_video(pv, vhs)
    write_gray_video(pc, ctl)
    res = A.scan(pv)
    st = res.get("stats") or {}
    kinds = {e["kind"] for e in res.get("events", [])}
    nd = sum(1 for e in res.get("events", []) if e["kind"] == "dropout")
    check("vhs head-switch flagged", res["ok"] and st.get("head_switch_pct", 0) > 50
          and "head_switch" in kinds)
    check("vhs jitter flagged", st.get("jitter_pct", 0) > 25 and "jitter" in kinds)
    check("dropout count ~12", 8 <= nd <= 18)
    res_c = A.scan(pc)
    check("control clean", res_c["ok"] and not
          [e for e in res_c["events"] if e["severity"] == "warn"])
    sv, sc = res.get("series", {}).get("v", []), res_c.get("series", {}).get("v", [])
    check("jitter series separates", sv and sc and
          float(np.mean(sv)) > 0.5 and float(np.mean(sc)) < 0.2)
    check("report renders", "ANALOG" in A.render_report(res))


# === captions ===================================================================

def t_captions(tmp):
    C = va_plugins.load_module("captions", "cap_core")
    if C is None:
        print("captions: [skip] plugin not installed")
        return
    print("captions:")
    speech = [(0, 2), (3, 5), (6, 8), (9, 11)]
    cues = [(0.4, 1.9, "hello there"), (3.4, 4.9, "x" * 80),
            (6.4, 7.9, "third cue line"), (6.9, 8.4, "overlapper")]
    q = C.qc_cues(cues, speech, 12.0)
    check("sync median", abs(q["sync_median_s"] - 0.4) < 0.12)
    check("cps + overlap", q["cps_over"] == 1 and q["overlaps"] == 1)
    cs = C.parse_srt("1\n00:00:01,000 --> 00:00:02,500\n<i>Tagged</i> line\n\n"
                     "2\n00:00:03,000 --> 00:00:04,000\ntwo\nlines\n")
    check("srt parse", len(cs) == 2 and cs[0][2] == "Tagged line")
    if not FFMPEG:
        print("  [skip] e2e (no ffmpeg)")
        return
    sr = 16000
    on = [(i * 3.0, i * 3.0 + 2.0) for i in range(8)]
    x = np.zeros(24 * sr, np.float32)
    rng = np.random.default_rng(1)
    for a, b in on:
        ts_ = np.arange(int(a * sr), int(b * sr))
        x[ts_] = 0.4 * np.sin(2 * np.pi * 300 * ts_ / sr) \
            + rng.normal(0, 0.05, ts_.size)
    wavp = os.path.join(tmp, "cap.wav")
    with wave.open(wavp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype("<i2").tobytes())

    def ts(t):
        ms = int(round((t - int(t)) * 1000))
        s = int(t)
        return "%02d:%02d:%02d,%03d" % (s // 3600, s % 3600 // 60, s % 60, ms)

    blocks = []
    for i, (a, _b) in enumerate(on):
        t0 = a + 0.4
        if i == 3:
            t1, txt = t0 + 1.0, "w" * 60
        elif i == 5:
            t1, txt = t0 + 3.2, "this cue deliberately overlaps the next one"
        else:
            t1, txt = t0 + 1.5, "ordinary caption line %d" % i
        blocks.append("%d\n%s --> %s\n%s\n" % (i + 1, ts(t0), ts(t1), txt))
    srtp = os.path.join(tmp, "cap.srt")
    open(srtp, "w").write("\n".join(blocks))
    mkv = os.path.join(tmp, "capfix.mkv")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=size=320x180:rate=15:duration=24",
                    "-i", wavp, "-i", srtp, "-map", "0:v", "-map", "1:a",
                    "-map", "2:s", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-b:a", "96k", "-c:s", "srt", mkv],
                   check=True)
    rep = C.analyze(mkv)
    q = rep.get("qc") or {}
    check("e2e cues read", rep["ok"] and q.get("cues") == 8)
    check("e2e sync detected", 0.25 < q.get("sync_median_s", 0) < 0.55)
    check("e2e cps + overlap", q.get("cps_over", 0) >= 1
          and q.get("overlaps", 0) == 1)
    import va_qc
    va_qc.register_profile("captions", C.qc_checks())
    ids = {c["id"]: c["status"]
           for c in va_qc.evaluate({"captions": rep}, "captions")["checks"]}
    check("profile statuses (ctx key)", ids.get("cap_sync") == "warn"
          and ids.get("cap_speed") == "warn"
          and ids.get("cap_presence") == "pass")
    ids2 = {c["id"]: c["status"]
            for c in va_qc.evaluate({"verify": rep}, "captions")["checks"]}
    check("legacy ctx key honoured", ids2.get("cap_sync") == "warn")
    ids3 = {c["id"]: c["status"]
            for c in va_qc.evaluate({"adv": {"captions": rep}},
                                    "captions")["checks"]}
    check("GUI adv bridge honoured", ids3.get("cap_sync") == "warn")
    check("report renders", "CAPTIONS" in C.render_report(rep))


# === ocr =======================================================================

def t_ocr(tmp):
    O = va_plugins.load_module("ocr", "ocr_core")
    if O is None:
        print("ocr: [skip] plugin not installed")
        return
    print("ocr:")
    check("parse ms", O.parse_timer("1:23.45") == ("ms", 83.45))
    check("parse hms", O.parse_timer("01:02:03.500") == ("hms", 3723.5))
    check("parse smpte", O.parse_timer("00:01:02:15", fps=30) == ("smpte", 62.5))
    check("parse rejects", O.parse_timer("1:?3.45") is None
          and O.parse_timer("07:62") is None)
    tv = [i / 30 for i in range(300)]
    secs = [60 + min(t, 4.0) if t < 9.0 else 60 + 4.0 + (t - 9.0) for t in tv]
    r = O.analyze_readings(tv, secs, 30.0)
    check("freeze flagged", any("frozen" in e["text"] for e in r["events"]))
    try:
        import cv2
    except Exception:  # noqa: BLE001
        print("  [skip] glyph + e2e (no cv2)")
        return
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def region_for(s):
        img = np.full((120, 640), 20, np.uint8)
        x = 10
        for ch in s:
            cv2.putText(img, ch, (x, 85), FONT, 1.2, 255, 2, cv2.LINE_AA)
            x += cv2.getTextSize(ch, FONT, 1.2, 2)[0][0] + 5
        return img

    bank = O.GlyphBank()
    n = bank.calibrate(region_for("0123456789:.").astype(np.float32),
                       "0123456789:.")
    check("calibration maps 12 glyphs", n == 12)
    s, _d = bank.read(region_for("3:07.62").astype(np.float32))
    check("readback exact", s == "3:07.62")
    if not FFMPEG:
        print("  [skip] e2e (no ffmpeg)")
        return

    def timer_at(t):
        if t < 10.0:
            tt = 60.0 + t
        elif t < 15.0:
            tt = 60.0 + t - 3.0
        else:
            tt = 60.0 + 12.0 + (t - 15.0) * 0.93
        m = int(tt // 60)
        return "%d:%05.2f" % (m, tt - m * 60)

    vid = os.path.join(tmp, "timer.mp4")
    write_gray_video(vid, [region_for(timer_at(i / 30.0)) for i in range(600)])
    res = O.scan(vid, (0, 30, 320, 115), bank)
    check("scan reads >95%", res["ok"] and res["read_rate"] > 0.95)
    back = [e for e in res["events"] if "BACKWARD" in e["text"]]
    check("backward jump at 10s", bool(back) and abs(back[0]["t"] - 10.0) < 0.2)
    drift = [g for g in res["segments"] if abs(g["slope"] - 0.93) < 0.012]
    check("drift segment isolated", bool(drift)
          and abs(drift[0]["t0"] - 15.0) < 1.2)
    check("honest segment at 1.0",
          any(g["slope"] > 0.995 and g["span_s"] > 4 for g in res["segments"]))
    ser = res.get("series") or {}
    check("drift series present", len(ser.get("v", [])) > 100)
    if ser.get("v"):
        check("drift series shows jump+slowdown", ser["v"][-1] < -3.2
              and abs(ser["v"][len(ser["v"]) // 4]) < 0.1)
    check("report renders", "TIMER AUDIT" in O.render_report(res))


def t_steg(tmp):
    C = va_plugins.load_module("steg", "steg_container")
    B = va_plugins.load_module("steg", "steg_bitplane")
    K = va_plugins.load_module("steg", "steg_codec")
    S = va_plugins.load_module("steg", "steg_scan")
    if not all((C, B, K, S)):
        print("steg: [skip] plugin not installed")
        return
    print("steg:")
    # container entropy units
    check("entropy bounds", abs(C.shannon(bytes(range(256))) - 8.0) < 1e-9
          and C.shannon(b"\x00" * 999) == 0.0)
    # bitplane: 1/f JPEG cover clean vs LSB-embedded
    try:
        import cv2
        have_cv2 = True
    except Exception:  # noqa: BLE001
        have_cv2 = False
    if have_cv2:
        def natimg(seed=0, beta=1.5, h=384, w=384):
            r = np.random.default_rng(seed)
            fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
            f = np.sqrt(fy**2 + fx**2); f[0, 0] = 1e-6
            im = np.fft.ifft2((r.normal(size=(h, w)) + 1j*r.normal(size=(h, w))) / f**beta).real
            rng = float(np.max(im) - np.min(im)) + 1e-9
            base = ((im - im.min()) / rng * 235 + 10).astype(np.uint8)
            _ok, enc = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return cv2.imdecode(enc, 0)

        def embed(img, frac, seed=1):
            r = np.random.default_rng(seed); flat = img.copy().ravel()
            k = int(len(flat) * frac); flat[:k] = (flat[:k] & 0xFE) | r.integers(0, 2, k)
            return flat.reshape(img.shape)
        clean = natimg(0)
        a0 = B.analyze_channel(clean)
        a100 = B.analyze_channel(embed(clean, 1.0))
        check("LSB clean reads clean", not a0["suspicious"] and a0["rs"]["rate"] < 0.1)
        check("LSB full-embed flagged", a100["suspicious"] and a100["rs"]["rate"] > 0.5)
    # codec: SEI parse + size residual units
    uuid = bytes(range(16)); payload = uuid + b"X" * 40
    rbsp = bytes([5, len(payload)]) + payload + b"\x80"
    stream = b"\x00\x00\x00\x01" + bytes([0x06]) + rbsp
    seis = K.parse_sei(stream, codec="h264")
    check("SEI user-data parsed", any(s["type"] == 5 and s["size"] == len(payload)
                                      for s in seis))
    rng = np.random.default_rng(0); cx = rng.uniform(10, 100, 200)
    sizes = cx * 50 + rng.normal(0, 30, 200); sizes[57] += 4000
    check("size residual flags fat frame",
          57 in K.analyze_sizes(sizes, cx, z=6.0)["anomaly_frames"])
    if not FFMPEG:
        print("  [skip] e2e (no ffmpeg)")
        return
    import zipfile
    base = os.path.join(tmp, "clean.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=256x192:rate=10:duration=2", "-c:v", "libx264",
                    "-preset", "ultrafast", base], check=True)
    r_clean = S.scan(base)
    check("clean lossy: no warnings", r_clean["ok"] and r_clean["warn"] == 0)
    check("LSB gated on lossy", "lossy" in r_clean["gated"].get("lsb", ""))
    poly = os.path.join(tmp, "poly.mp4")
    zp = os.path.join(tmp, "s.zip")
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("x.txt", "secret\n" * 80)
    with open(poly, "wb") as o:
        o.write(open(base, "rb").read()); o.write(open(zp, "rb").read())
    r_poly = S.scan(poly)
    kinds = {f["kind"] for f in r_poly["findings"] if f["severity"] == "warn"}
    check("polyglot: appended data + embedded zip",
          "appended_data" in kinds and any(
              f["kind"] == "embedded_file" and "ZIP" in f["text"]
              for f in r_poly["findings"]))
    enc = os.path.join(tmp, "enc.mp4")
    with open(enc, "wb") as o:
        o.write(open(base, "rb").read()); o.write(os.urandom(30000))
    ap = [f for f in S.scan(enc)["findings"] if f["kind"] == "appended_data"]
    check("encrypted append flagged hi-entropy",
          bool(ap) and "compressed/encrypted" in ap[0]["text"])
    # lossless LSB end to end
    ll = os.path.join(tmp, "ll.mkv")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=256x256:rate=8:duration=2", "-c:v", "ffv1", ll],
                   check=True)
    rc = S.scan(ll)
    check("lossless clean: LSB ran, clean", rc.get("lsb") is not None
          and rc["lsb"]["suspicious_frames"] == 0)
    import va_ffmpeg
    vs = va_ffmpeg.VideoSource(ll, decode_max=10000); fr = []
    vs.start(0)
    while True:
        f0 = vs.read()
        if f0 is None:
            break
        fr.append(np.asarray(f0).copy())
    vs.close()
    rng = np.random.default_rng(1)
    lle = os.path.join(tmp, "lle.mkv")
    p = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-s", "256x256", "-r", "8", "-i", "-",
                          "-c:v", "ffv1", lle], stdin=subprocess.PIPE)
    for f0 in fr:
        g = f0.astype(np.uint8).reshape(-1)
        g[:] = (g & 0xFE) | rng.integers(0, 2, g.size).astype(np.uint8)
        p.stdin.write(g.tobytes())
    p.stdin.close(); p.wait()
    re = S.scan(lle)
    check("lossless embedded: LSB payload flagged",
          any(f["kind"] == "lsb" for f in re["findings"])
          and re["lsb"]["suspicious_frames"] > re["lsb"]["frames"] // 2)
    check("report renders", "HIDDEN-DATA" in S.render_report(re))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="va_st_packs_") as tmp:
        if want("analog"):
            t_analog(tmp)
        if want("captions"):
            t_captions(tmp)
        if want("ocr"):
            t_ocr(tmp)
        if want("steg"):
            t_steg(tmp)
    print()
    if FAILS:
        print("FAILED: %d check(s): %s" % (len(FAILS), ", ".join(FAILS)))
        sys.exit(1)
    print("all domain-pack checks passed")
