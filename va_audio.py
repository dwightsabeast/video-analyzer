#!/usr/bin/env python3
"""
va_audio - audio analysis: EBU R128 loudness (summary + over time), per-channel
statistics (peak/RMS/DC/dynamic range), silence detection, stereo phase
correlation, spectrogram / waveform images, a PCM decode helper, a waveform
overview for timeline scrubbing, and an audio FORENSIC battery (clipping,
dropouts/clicks, splice candidates, channel layout sanity, bandwidth history,
loudness steps). All via ffmpeg + numpy; pure compute, no Tk.

Every analysis pass decodes ONLY the audio stream (-vn or -map 0:a:0) - a UHD
video stream sharing the container must never be decoded along the way.
"""

from __future__ import annotations

import re
import os
import subprocess
import time

import numpy as np

from va_ffmpeg import find_ffmpeg, ffprobe_json, CREATIONFLAGS
import va_metrics


def has_audio(path) -> bool:
    data = ffprobe_json(path)
    if not data:
        return False
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def audio_info(path) -> "dict | None":
    data = ffprobe_json(path)
    if not data:
        return None
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not a:
        return None
    return {
        "codec": a.get("codec_name", ""),
        "profile": a.get("profile", ""),
        "channels": a.get("channels", 0),
        "channel_layout": a.get("channel_layout", ""),
        "sample_rate": a.get("sample_rate", ""),
        "bit_rate": a.get("bit_rate", ""),
    }


_LAST_ERROR = None   # why the most recent call came back empty (GUI hint)


def last_error() -> "str | None":
    """Reason the most recent va_audio call returned None/empty: 'no ffmpeg',
    'no audio stream', 'timeout', 'cancelled', 'ffmpeg failed' or None (all good).
    Lets callers stop reporting a timed-out pass as "no audio track"."""
    return _LAST_ERROR


def _fail(reason):
    global _LAST_ERROR
    _LAST_ERROR = reason


def _timeout(path) -> float:
    """Audio passes run far faster than realtime; scale the kill-switch anyway."""
    return max(120.0, 1.5 * va_metrics.media_duration(path))


def _run(args, timeout=120, cancel=None):
    """Killable run via the shared va_metrics child registry (kill_all-aware).
    Flags 'timeout'/'cancelled' in last_error() when the child was put down."""
    r = va_metrics._run(args, timeout=timeout, cancel=cancel)
    if r is None:
        _fail("ffmpeg failed")
    elif getattr(r, "killed", None):
        _fail("timeout" if r.killed == "timeout" else "cancelled")
    return r


def kill_all():
    """Kill any live engine children (same registry as va_metrics.kill_all)."""
    va_metrics.kill_all()


# --- PCM decode (shared by the forensic battery and the waveform strip) -------

def decode_pcm(path, sr=16000, mono=True, t0=None, dur=None,
               max_samples=480_000_000, cancel=None) -> "np.ndarray | None":
    """Decode the FIRST audio stream to float32 PCM via an ffmpeg pipe.
    Returns shape (n,) when mono else (n, channels). None on failure."""
    _fail(None)
    exe = find_ffmpeg()
    if not exe:
        _fail("no ffmpeg")
        return None
    info = audio_info(path)
    if not info:
        _fail("no audio stream")
        return None
    ch = 1 if mono else max(1, int(info.get("channels") or 1))
    args = [exe, "-v", "error", "-nostdin"]
    if t0 is not None:
        args += ["-ss", "%.3f" % max(0.0, float(t0))]
    args += ["-i", path]
    if dur is not None:
        args += ["-t", "%.3f" % max(0.01, float(dur))]
    args += ["-map", "0:a:0", "-ac", str(ch), "-ar", str(int(sr)),
             "-f", "f32le", "-"]
    deadline = time.monotonic() + _timeout(path)
    chunks, total = [], 0
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=CREATIONFLAGS)
    except OSError:
        _fail("ffmpeg failed")
        return None
    va_metrics._track(proc)
    try:
        while True:
            if cancel is not None and cancel():
                proc.kill()
                _fail("cancelled")
                return None
            if time.monotonic() > deadline:
                proc.kill()
                _fail("timeout")
                return None
            buf = proc.stdout.read(1 << 20)
            if not buf:
                break
            chunks.append(buf)
            total += len(buf)
            if total // 4 >= max_samples:
                proc.kill()
                break
        proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        _fail("ffmpeg failed")
        return None
    finally:
        va_metrics._untrack(proc)
    if not chunks:
        _fail("ffmpeg failed")
        return None
    raw = b"".join(chunks)
    raw = raw[: (len(raw) // (4 * ch)) * 4 * ch]
    x = np.frombuffer(raw, dtype="<f4")
    if not mono and ch > 1:
        x = x.reshape(-1, ch)
    return x


def waveform_overview(path, buckets=2400, sr=8000, cancel=None) -> "dict | None":
    """Min/max/RMS envelope of the (mono-folded) track for timeline drawing:
    {duration_s, buckets, vmin[], vmax[], rms[]} - a few KB regardless of length."""
    x = decode_pcm(path, sr=sr, mono=True, cancel=cancel)
    if x is None or x.size < 2:
        return None
    n = int(x.size)
    buckets = max(64, min(int(buckets), n))
    edge = (n // buckets) * buckets
    seg = x[:edge].reshape(buckets, -1)
    rms = np.sqrt(np.mean(seg * seg, axis=1))
    return {"duration_s": n / float(sr), "buckets": buckets,
            "vmin": seg.min(axis=1).tolist(), "vmax": seg.max(axis=1).tolist(),
            "rms": rms.tolist()}

# --- Classic passes (ffmpeg filters; -vn keeps UHD video streams cold) --------

def loudness(path, cancel=None) -> "dict | None":
    """One ebur128 pass -> {summary:{integrated,lra,true_peak,threshold}, t,M,S}.
    Digital silence measures true_peak_dbfs = -inf (a value, not a missing peak).
    None means no usable input - see last_error() for which kind."""
    _fail(None)
    exe = find_ffmpeg()
    if not exe:
        _fail("no ffmpeg")
        return None
    if not has_audio(path):
        _fail("no audio stream")
        return None
    args = [exe, "-hide_banner", "-nostats", "-i", path, "-vn", "-sn", "-dn",
            "-af", "ebur128=peak=true:metadata=1,ametadata=print:file=-",
            "-f", "null", "-"]
    r = _run(args, timeout=_timeout(path), cancel=cancel)
    if r is None:
        return None
    t, M, S = [], [], []
    cur_t = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([-\d.]+)", line)
            cur_t = float(m.group(1)) if m else (t[-1] if t else 0.0)
        elif line.startswith("lavfi.r128.M="):
            try:
                M.append(float(line.split("=", 1)[1])); t.append(cur_t if cur_t is not None else len(M) * 0.1)
            except ValueError:
                pass
        elif line.startswith("lavfi.r128.S="):
            try:
                S.append(float(line.split("=", 1)[1]))
            except ValueError:
                pass
    txt = r.stderr
    summary = None
    if "Summary:" in txt:
        s = txt[txt.rfind("Summary:"):]

        def grab(p):
            m = re.search(p, s)
            return float(m.group(1)) if m else None
        summary = {
            "integrated_lufs": grab(r"I:\s*(-?inf|[-\d.]+)\s*LUFS"),
            "lra_lu": grab(r"LRA:\s*(-?inf|[-\d.]+)\s*LU"),
            "threshold_lufs": grab(r"Threshold:\s*(-?inf|[-\d.]+)\s*LUFS"),
            "true_peak_dbfs": grab(r"Peak:\s*(-?inf|[-\d.]+)\s*dBFS"),
        }
    return {"summary": summary, "t": t, "M": M, "S": S}


def astats(path, cancel=None) -> "dict | None":
    """Per-channel and overall: peak dB, RMS dB, DC offset, dynamic range."""
    _fail(None)
    exe = find_ffmpeg()
    if not exe:
        _fail("no ffmpeg")
        return None
    if not has_audio(path):
        _fail("no audio stream")
        return None
    r = _run([exe, "-hide_banner", "-i", path, "-vn", "-sn", "-dn",
              "-af", "astats=metadata=0", "-f", "null", "-"],
             timeout=_timeout(path), cancel=cancel)
    if r is None:
        return None
    channels, overall, cur = [], {}, None

    def num(line):
        m = re.search(r":\s*([-\d.]+|inf|-inf|nan)", line)
        return m.group(1) if m else None

    for raw in r.stderr.splitlines():
        line = raw.split("] ", 1)[-1].strip()
        if line.startswith("Channel:"):
            cur = {"channel": line.split(":", 1)[1].strip()}
            channels.append(cur)
        elif line.startswith("Overall"):
            cur = overall
        elif cur is not None:
            for key, tag in (("Peak level dB", "peak_db"), ("RMS level dB", "rms_db"),
                             ("DC offset", "dc_offset"), ("Dynamic range", "dynamic_range")):
                if line.startswith(key):
                    cur[tag] = num(line)
    return {"channels": channels, "overall": overall}


def silence_segments(path, noise_db=-50, min_dur=0.5, cancel=None) -> list:
    """[(start_s, end_s), ...] of silence via ffmpeg silencedetect. Silence that
    runs to EOF emits no silence_end - the stream duration fills in."""
    _fail(None)
    exe = find_ffmpeg()
    if not exe:
        _fail("no ffmpeg")
        return []
    if not has_audio(path):
        _fail("no audio stream")
        return []
    r = _run([exe, "-hide_banner", "-i", path, "-vn", "-sn", "-dn", "-af",
              "silencedetect=n=%ddB:d=%g" % (noise_db, min_dur), "-f", "null", "-"],
             timeout=_timeout(path), cancel=cancel)
    if r is None:
        return []
    starts, ends = [], []
    for line in r.stderr.splitlines():
        va_metrics._grow(starts, line, "silence_start")
        va_metrics._grow(ends, line, "silence_end")
    dur = va_metrics.media_duration(path) if len(starts) > len(ends) else 0.0
    segs = []
    for i, s in enumerate(starts):
        segs.append((s, ends[i] if i < len(ends) else max(dur, s)))
    return va_metrics._clean_segments(segs)


def correlation(path, cancel=None) -> "float | None":
    """Average stereo phase correlation (-1..+1) via aphasemeter, or None if not stereo."""
    _fail(None)
    info = audio_info(path)
    if not info or info.get("channels") != 2:
        _fail("no audio stream" if not info else "not stereo")
        return None
    exe = find_ffmpeg()
    if not exe:
        _fail("no ffmpeg")
        return None
    r = _run([exe, "-hide_banner", "-nostats", "-i", path, "-vn", "-sn", "-dn",
              "-af", "aphasemeter=video=0,ametadata=print:file=-", "-f", "null", "-"],
             timeout=_timeout(path), cancel=cancel)
    if r is None:
        return None
    vals = [float(m) for m in re.findall(r"lavfi\.aphasemeter\.phase=([-\d.]+)", r.stdout)]
    return round(sum(vals) / len(vals), 3) if vals else None


def spectrogram_png(path, out_path, w=640, h=320, cancel=None) -> "str | None":
    _fail(None)
    exe = find_ffmpeg()
    if not exe or not has_audio(path):
        _fail("no ffmpeg" if not exe else "no audio stream")
        return None
    r = _run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
              "-lavfi", "showspectrumpic=s=%dx%d:legend=1" % (w, h), "-frames:v", "1", out_path],
             timeout=_timeout(path), cancel=cancel)
    return out_path if (r and r.returncode == 0 and os.path.isfile(out_path)) else None


def waveform_png(path, out_path, w=640, h=160, cancel=None) -> "str | None":
    _fail(None)
    exe = find_ffmpeg()
    if not exe or not has_audio(path):
        _fail("no ffmpeg" if not exe else "no audio stream")
        return None
    r = _run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
              "-lavfi", "showwavespic=s=%dx%d:split_channels=1" % (w, h), "-frames:v", "1", out_path],
             timeout=_timeout(path), cancel=cancel)
    return out_path if (r and r.returncode == 0 and os.path.isfile(out_path)) else None

# --- Forensic battery ----------------------------------------------------------
#
# Indicators, not proof: each scan reports WHERE the signal behaves like an
# edit/defect, in the same {severity, area, t, text} shape va_forensics uses,
# so audio findings ride the same report/marks/QC/export plumbing as video.

_CEIL = 0.985            # |sample| at/above this counts as clipped (FS = 1.0)


def clipping_scan(x, sr) -> dict:
    """Runs of consecutive near-full-scale samples = hard clipping.
    {events: [{t, dur, peak_pct}], count, clipped_pct}"""
    if x is None or x.size == 0:
        return {"events": [], "count": 0, "clipped_pct": 0.0}
    hot = np.abs(x) >= _CEIL
    clipped_pct = 100.0 * float(hot.sum()) / float(x.size)
    d = np.diff(hot.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if hot.size and hot[0]:
        starts = np.r_[0, starts]
    if hot.size and hot[-1]:
        ends = np.r_[ends, hot.size]
    events = []
    run_min = max(3, int(0.0008 * sr))          # >= ~0.8 ms at the analysis rate
    for s, e in zip(starts, ends):
        if e - s >= run_min:
            events.append({"t": float(s) / sr, "dur": float(e - s) / sr,
                           "peak_pct": round(100.0 * float(np.abs(x[s:e]).max()), 1)})
    merged = []
    for ev in events:                            # merge events < 50 ms apart
        if merged and ev["t"] - (merged[-1]["t"] + merged[-1]["dur"]) < 0.05:
            merged[-1]["dur"] = ev["t"] + ev["dur"] - merged[-1]["t"]
        else:
            merged.append(dict(ev))
    return {"events": merged[:200], "count": len(merged),
            "clipped_pct": round(clipped_pct, 4)}


def dropout_scan(x, sr) -> dict:
    """Digital dropouts (runs of EXACT zeros mid-stream) and clicks/pops
    (isolated sample-delta outliers). {dropouts: [{t, dur}], clicks: [{t, mag}]}"""
    out = {"dropouts": [], "clicks": []}
    if x is None or x.size < sr // 2:
        return out
    n = x.size
    guard = int(0.1 * sr)                        # head/tail priming is normal
    z = (x == 0.0)
    d = np.diff(z.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if z[0]:
        starts = np.r_[0, starts]
    if z[-1]:
        ends = np.r_[ends, n]
    min_run = max(2, int(0.02 * sr))             # >= 20 ms of dead zeros
    for s, e in zip(starts, ends):
        if e - s >= min_run and s > guard and e < n - guard:
            out["dropouts"].append({"t": round(float(s) / sr, 3),
                                    "dur": round(float(e - s) / sr, 4)})
    # clicks: |first difference| vs a robust (median+MAD) yardstick
    dx = np.abs(np.diff(x))
    med = float(np.median(dx))
    mad = float(np.median(np.abs(dx - med))) + 1e-9
    thr = med + 16.0 * 1.4826 * mad
    floor = 0.1                                  # ignore micro-deltas in quiet content
    idx = np.where(dx > max(thr, floor))[0]
    if idx.size:
        groups = [[idx[0]]]
        for i in idx[1:]:
            if i - groups[-1][-1] <= int(0.03 * sr):
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups[:400]:
            peak_i = max(g, key=lambda i: dx[i])
            if guard < peak_i < n - guard:
                out["clicks"].append({"t": round(float(peak_i) / sr, 3),
                                      "mag": round(float(dx[peak_i]), 3)})
    out["dropouts"] = out["dropouts"][:100]
    out["clicks"] = out["clicks"][:100]
    return out


def splice_scan_audio(x, sr, hop_s=0.1) -> dict:
    """Audio edit-point screen: step changes in the BACKGROUND NOISE FLOOR and
    in the spectral rolloff that persist (recording/room change), away from
    silence. {candidates: [{t, floor_jump_db, rolloff_ratio, rms_jump_db, score}]}"""
    out = {"candidates": [], "hop_s": hop_s}
    if x is None or x.size < sr * 2:
        return out
    hop = max(64, int(hop_s * sr))
    nfr = x.size // hop
    if nfr < 12:
        return out
    fr = x[:nfr * hop].reshape(nfr, hop)
    ax = np.abs(fr)
    rms = 20.0 * np.log10(np.sqrt(np.mean(fr * fr, axis=1)) + 1e-9)
    floor = 20.0 * np.log10(np.quantile(ax, 0.1, axis=1) + 1e-9)
    win = np.hanning(hop).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr * win, axis=1))
    cum = np.cumsum(spec * spec, axis=1)
    tot = cum[:, -1:] + 1e-12
    roll_bin = np.argmax(cum >= 0.95 * tot, axis=1)
    freqs = np.fft.rfftfreq(hop, 1.0 / sr)
    roll = freqs[np.clip(roll_bin, 0, freqs.size - 1)] + 1.0
    K = 5
    cands = []
    for i in range(K, nfr - K):
        pf = float(np.median(floor[i - K:i])); nf = float(np.median(floor[i:i + K]))
        pr = float(np.median(rms[i - K:i])); nr = float(np.median(rms[i:i + K]))
        if pr < -65.0 or nr < -65.0:        # silence boundaries are normal edits
            continue
        if min(pf, nf) < -75.0:             # rising out of (near-)digital silence
            continue                        # = content start/stop, not a splice
        fj = abs(nf - pf)
        rj = abs(nr - pr)
        po = float(np.median(roll[i - K:i])); no = float(np.median(roll[i:i + K]))
        rr = max(po, no) / max(1.0, min(po, no))
        score = int(fj >= 8.0) + int(fj >= 14.0) + int(rr >= 1.6) + int(rj >= 10.0)
        if score >= 1 and fj >= 8.0:
            cands.append({"t": round(i * hop / float(sr), 2),
                          "floor_jump_db": round(fj, 1),
                          "rolloff_ratio": round(rr, 2),
                          "rms_jump_db": round(rj, 1), "score": score})
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:                          # de-dup within 0.5 s, best first
        if all(abs(c["t"] - p["t"]) > 0.5 for p in picked):
            picked.append(c)
        if len(picked) >= 20:
            break
    picked.sort(key=lambda c: c["t"])
    out["candidates"] = picked
    return out


def loudness_steps(series, min_lu=6.0, sustain_s=3.0) -> list:
    """Sustained jumps in momentary loudness (level rides / gain edits):
    [{t, delta_lu}] from a loudness() {t, M} series. Robust to short peaks."""
    t = (series or {}).get("t") or []
    M = (series or {}).get("M") or []
    if len(M) < 30 or len(t) != len(M):
        return []
    tv = np.array(t, dtype=np.float64)
    mv = np.array(M, dtype=np.float64)
    ok = mv > -120.0
    tv, mv = tv[ok], mv[ok]
    if mv.size < 30:
        return []
    dt = float(np.median(np.diff(tv))) if tv.size > 1 else 0.1
    k = max(3, int(round(sustain_s / max(dt, 1e-3))))
    if mv.size < 2 * k + 2:
        return []
    out = []
    i = k
    while i < mv.size - k:
        a = float(np.median(mv[i - k:i])); b = float(np.median(mv[i:i + k]))
        if a > -90 and b > -90 and abs(b - a) >= min_lu:
            # settle: the full step size is prev-median vs the median AFTER
            # the transition has cleared the window
            j = min(mv.size, i + 2 * k)
            settled = float(np.median(mv[j - k:j]))
            tt = float(tv[i])
            if not out or tt - out[-1]["t"] > sustain_s:
                out.append({"t": round(tt, 2), "delta_lu": round(settled - a, 1)})
            i += k
        i += 1
    return out[:20]

def channel_check(path, cancel=None) -> "dict | None":
    """Per-channel level sanity on a mid-file window: silent channels, fake
    stereo (decorrelated-free mono fold), polarity inversion."""
    info = audio_info(path)
    if not info:
        return None
    nch = max(1, int(info.get("channels") or 1))
    dur = va_metrics.media_duration(path)
    t0 = max(0.0, min(dur * 0.25, max(0.0, dur - 30.0)))
    x = decode_pcm(path, sr=8000, mono=False, t0=t0,
                   dur=min(30.0, dur or 30.0), cancel=cancel)
    if x is None or x.size == 0:
        return None
    if x.ndim == 1:
        x = x[:, None]
    nch = x.shape[1]
    chans = []
    for c in range(nch):
        v = x[:, c]
        rms = float(np.sqrt(np.mean(v * v)))
        chans.append({"idx": c, "rms_db": round(20.0 * np.log10(rms + 1e-9), 1),
                      "active": bool(rms > 1e-4)})
    res = {"layout": info.get("channel_layout", ""), "count": nch,
           "channels": chans, "stereo": None}
    if nch == 2:
        L, R = x[:, 0], x[:, 1]
        sl = float(np.std(L)); sr_ = float(np.std(R))
        if sl > 1e-6 and sr_ > 1e-6:
            corr = float(np.corrcoef(L, R)[0, 1])
            res["stereo"] = {
                "correlation": round(corr, 3),
                "fake_stereo": bool(corr > 0.995 and abs(sl - sr_) / max(sl, sr_) < 0.02),
                "phase_inverted": bool(corr < -0.9),
            }
    return res


def _audio_spectrum(path, sr=44100, max_s=150.0, win=4096, hop=2048, cancel=None):
    """Mean magnitude spectrum of (up to) the first max_s seconds of the first
    audio stream, decoded mono at sr. Returns (freqs_hz, mean_mag) or None."""
    exe = find_ffmpeg()
    if not exe:
        return None
    args = [exe, "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0",
            "-ac", "1", "-ar", str(sr), "-t", "%.3f" % max_s, "-f", "f32le", "-"]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=CREATIONFLAGS)
    except OSError:
        return None
    va_metrics._track(proc)
    hann = np.hanning(win).astype(np.float32)
    acc = None
    nwin = 0
    buf = b""
    try:
        while True:
            if cancel is not None and cancel():
                break
            chunk = proc.stdout.read(1 << 16)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= win * 4:
                x = np.frombuffer(buf[:win * 4], np.float32)
                buf = buf[hop * 4:]
                if not np.all(np.isfinite(x)):
                    x = np.nan_to_num(x)
                mag = np.abs(np.fft.rfft(x * hann))
                acc = mag if acc is None else acc + mag
                nwin += 1
    finally:
        try:
            proc.stdout.close()
            proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
        va_metrics._untrack(proc)
    if acc is None or nwin < 4:
        return None
    return np.fft.rfftfreq(win, 1.0 / sr), acc / nwin


def audio_cutoff(path, cancel=None) -> "dict | None":
    """Where does the audio band actually end? Lossy encoders low-pass near the
    top of the band; a RESAMPLE-based slowdown drags that cutoff down by the
    speed factor, which is the tell. Returns {cutoff_hz, nyquist_hz, ratio,
    sample_rate, codec, profile, bit_rate} or None (no usable audio)."""
    spec = _audio_spectrum(path, cancel=cancel)
    if spec is None:
        return None
    freqs, mag = spec
    db = 20.0 * np.log10(mag + 1e-12)
    # ~0.5 kHz smoothing kills harmonic comb structure before edge-finding
    k = max(3, int(round(500.0 / max(1e-9, freqs[1] - freqs[0]))) | 1)
    sm = np.convolve(db, np.ones(k) / k, mode="same")
    body = sm[(freqs >= 200) & (freqs <= 4000)]
    if body.size == 0 or not np.isfinite(body).any():
        return None
    rel = sm - float(np.max(body))
    idx = np.where(rel > -55.0)[0]          # encoder stopbands sit 70-90 dB down
    cutoff = float(freqs[idx[-1]]) if idx.size else 0.0
    nyq = float(freqs[-1])
    pd = ffprobe_json(path) or {}
    ast = next((s for s in pd.get("streams", [])
                if s.get("codec_type") == "audio"), None) or {}
    try:
        br = int(ast.get("bit_rate") or 0)
    except (TypeError, ValueError):
        br = 0
    try:
        srate = int(ast.get("sample_rate") or 0)
    except (TypeError, ValueError):
        srate = 0
    return {"cutoff_hz": round(cutoff, 1), "nyquist_hz": nyq,
            "ratio": round(cutoff / nyq, 3) if nyq else 0.0,
            "sample_rate": srate, "codec": str(ast.get("codec_name") or ""),
            "profile": str(ast.get("profile") or ""), "bit_rate": br}


def bandwidth_analysis(path, cancel=None) -> "dict | None":
    """Spectral cutoff vs the channel's Nyquist: a band that ends far below
    Nyquist on a high-rate stream betrays an earlier lossy generation or a
    resample-based speed change. Wraps audio_cutoff (one decoder)."""
    cut = audio_cutoff(path, cancel=cancel)
    if not cut:
        return None
    ratio = cut.get("ratio") or 0.0
    codec = (cut.get("codec") or "").lower()
    note = "full-band (cutoff %.1f kHz of %.1f kHz Nyquist)" % (
        cut.get("cutoff_hz", 0) / 1000.0, cut.get("nyquist_hz", 0) / 1000.0)
    suspicious = False
    if "he" in (cut.get("profile") or "").lower():
        note = "HE-AAC SBR band - cutoff measurement not meaningful"
    elif ratio and ratio < 0.70:
        suspicious = True
        note = ("band ends at %.1f kHz (%.0f%% of Nyquist) - earlier lossy "
                "generation, low-bitrate ancestor, or resampled speed change" % (
                    cut.get("cutoff_hz", 0) / 1000.0, 100.0 * ratio))
    elif ratio and ratio < 0.85:
        note = ("mild band limit at %.1f kHz (%.0f%% of Nyquist) - normal for "
                "efficient lossy encodes (%s)" % (
                    cut.get("cutoff_hz", 0) / 1000.0, 100.0 * ratio, codec or "?"))
    cut = dict(cut)
    cut["suspicious"] = suspicious
    cut["note"] = note
    return cut


def forensic_battery(path, loudness_series=None, deep=True,
                     cancel=None, on_progress=None) -> dict:
    """The audio integrity battery. One 16 kHz mono decode feeds the sample
    scans; channel/bandwidth run their own short decodes. Pass loudness_series=
    a loudness() result to reuse the Analyze pass instead of re-measuring.
    Returns {present, info, duration_s, clipping, dropouts, splices, channels,
    bandwidth, loudness_steps, findings, summary}."""
    def _prog(msg):
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:   # noqa: BLE001
                pass

    rep = {"present": False, "info": None, "duration_s": None,
           "clipping": None, "dropouts": None, "splices": None,
           "channels": None, "bandwidth": None, "loudness_steps": None,
           "findings": [], "summary": "no audio track"}
    info = audio_info(path)
    if not info:
        return rep
    rep["present"] = True
    rep["info"] = info

    _prog("audio: decoding PCM")
    sr = 16000
    x = decode_pcm(path, sr=sr, mono=True, cancel=cancel)
    if x is not None and x.size:
        rep["duration_s"] = round(x.size / float(sr), 2)
        _prog("audio: clipping scan")
        rep["clipping"] = clipping_scan(x, sr)
        _prog("audio: dropout / click scan")
        rep["dropouts"] = dropout_scan(x, sr)
        _prog("audio: splice scan")
        rep["splices"] = splice_scan_audio(x, sr)
    if deep:
        _prog("audio: channel sanity")
        rep["channels"] = channel_check(path, cancel=cancel)
        _prog("audio: bandwidth history")
        rep["bandwidth"] = bandwidth_analysis(path, cancel=cancel)
    if loudness_series is None and deep:
        _prog("audio: loudness steps")
        loudness_series = loudness(path, cancel=cancel)
    rep["loudness_steps"] = loudness_steps(loudness_series) if loudness_series else []

    finds = []

    def add(sev, t, text):
        finds.append({"severity": sev, "area": "audio",
                      "t": (round(float(t), 2) if t is not None else None),
                      "text": text})

    cl = rep.get("clipping") or {}
    if cl.get("count"):
        worst = sorted(cl["events"], key=lambda e: -e["dur"])[:3]
        sev = "warn" if (cl["clipped_pct"] > 0.01 or cl["count"] >= 3) else "info"
        for e in worst:
            add(sev, e["t"], "hard clipping @ %.2fs (%.0f ms run)" % (
                e["t"], e["dur"] * 1000.0))
        if cl["count"] > 3:
            add("info", None, "clipping total: %d event(s), %.3f%% of samples "
                "at full scale" % (cl["count"], cl["clipped_pct"]))
    dr = rep.get("dropouts") or {}
    for ev in (dr.get("dropouts") or [])[:6]:
        add("warn", ev["t"], "digital dropout @ %.2fs (%.0f ms of exact zeros "
            "mid-stream)" % (ev["t"], ev["dur"] * 1000.0))
    clicks = dr.get("clicks") or []
    if rep.get("duration_s") and len(clicks) / max(rep["duration_s"], 1.0) * 60.0 > 20:
        worst = sorted(clicks, key=lambda c: -c["mag"])[:3]
        add("info", worst[0]["t"], "%d click/pop transients (%.0f/min) - damaged "
            "source, vinyl/tape artefacts, or buffer glitches" % (
                len(clicks), len(clicks) / max(rep["duration_s"], 1.0) * 60.0))
    sp = rep.get("splices") or {}
    for c in (sp.get("candidates") or [])[:10]:
        if c["score"] >= 3 or (c["score"] >= 2 and c["floor_jump_db"] >= 14.0):
            add("warn", c["t"], "audio splice signature @ %.2fs (noise floor "
                "steps %.1f dB, rolloff ratio %.2f)" % (
                    c["t"], c["floor_jump_db"], c["rolloff_ratio"]))
        else:
            add("info", c["t"], "background noise floor steps %.1f dB @ %.2fs - "
                "weak edit signal" % (c["floor_jump_db"], c["t"]))
    ch = rep.get("channels") or {}
    for c in (ch.get("channels") or []):
        if not c["active"]:
            add("info", None, "channel %d (%s) is silent" % (
                c["idx"], ch.get("layout") or "?"))
    st = ch.get("stereo") or {}
    if st.get("phase_inverted"):
        add("warn", None, "stereo channels are phase-INVERTED (correlation %.2f) - "
            "mono playback will cancel" % st.get("correlation", -1.0))
    elif st.get("fake_stereo"):
        add("info", None, "stereo channels are identical (correlation %.3f) - "
            "mono source folded to stereo" % st.get("correlation", 1.0))
    bw = rep.get("bandwidth") or {}
    if bw.get("suspicious"):
        add("info", None, bw.get("note", "band-limited audio"))
    for s in (rep.get("loudness_steps") or [])[:6]:
        add("info", s["t"], "loudness steps %+.1f LU @ %.1fs (sustained) - "
            "level edit or programme change" % (s["delta_lu"], s["t"]))

    sev_rank = {"warn": 0, "info": 1}
    finds.sort(key=lambda f: (sev_rank.get(f["severity"], 2),
                              f["t"] if f["t"] is not None else 1e12))
    rep["findings"] = finds
    warns = sum(1 for f in finds if f["severity"] == "warn")
    rep["summary"] = ("%d audio concern(s), %d note(s) - indicators, not proof"
                      % (warns, len(finds) - warns)) if finds else \
        "no audio integrity concerns surfaced"
    return rep


def render_battery(rep) -> str:
    """Readable text rendering of a forensic_battery() dict."""
    if not rep or not rep.get("present"):
        return "AUDIO FORENSICS - no audio track"
    info = rep.get("info") or {}
    L = ["AUDIO FORENSICS",
         "=" * 64,
         "  %-18s %s %s, %s ch (%s), %s Hz, %s b/s" % (
             "stream", info.get("codec", "?"), info.get("profile") or "",
             info.get("channels", "?"), info.get("channel_layout") or "?",
             info.get("sample_rate", "?"), info.get("bit_rate") or "?"),
         "",
         "VERDICT: %s" % rep.get("summary", ""), ""]
    finds = rep.get("findings") or []
    if finds:
        L.append("FINDINGS (%d)" % len(finds))
        for f in finds:
            ts = ("@ %7.2fs " % f["t"]) if f.get("t") is not None else "          "
            L.append("  [%-4s] %s%s" % (f["severity"].upper(), ts, f["text"]))
        L.append("")
    cl = rep.get("clipping") or {}
    L.append("  %-18s %s event(s), %.4f%% samples at ceiling" % (
        "clipping", cl.get("count", 0), cl.get("clipped_pct", 0.0)))
    dr = rep.get("dropouts") or {}
    L.append("  %-18s %d dropout(s), %d click(s)" % (
        "dropouts/clicks", len(dr.get("dropouts") or []), len(dr.get("clicks") or [])))
    sp = rep.get("splices") or {}
    L.append("  %-18s %d candidate(s)" % ("splice scan", len(sp.get("candidates") or [])))
    bw = rep.get("bandwidth") or {}
    if bw:
        L.append("  %-18s %s" % ("bandwidth", bw.get("note", "")))
    ch = rep.get("channels") or {}
    if ch:
        act = sum(1 for c in ch.get("channels") or [] if c["active"])
        L.append("  %-18s %d/%d active (%s)" % (
            "channels", act, ch.get("count", 0), ch.get("layout") or "?"))
    ls = rep.get("loudness_steps") or []
    L.append("  %-18s %d sustained step(s)" % ("loudness", len(ls)))
    return "\n".join(L)

# --- Playback follower (bundled ffplay; video keeps the clock) -----------------

class AudioFollower:
    """Plays the file's audio in step with the GUI's wall-clock-paced video via
    the bundled ffplay (no display, audio only). The video loop owns the clock;
    seeks/pauses just restart or kill the child. Zero extra dependencies."""

    def __init__(self, path):
        from va_ffmpeg import find_tool
        self.path = path
        self.exe = find_tool("ffplay")
        self.proc = None
        self._t0 = 0.0
        self._wall0 = 0.0

    @property
    def available(self) -> bool:
        return bool(self.exe)

    def start(self, t: float, volume: int = 100):
        """(Re)start playback at media time t seconds."""
        self.stop()
        if not self.exe:
            return
        args = [self.exe, "-nodisp", "-autoexit", "-loglevel", "quiet", "-vn",
                "-volume", str(int(volume)), "-ss", "%.3f" % max(0.0, t),
                "-i", self.path]
        try:
            self.proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, creationflags=CREATIONFLAGS)
            va_metrics._track(self.proc)
            self._t0 = max(0.0, t)
            self._wall0 = time.monotonic()
        except OSError:
            self.proc = None

    def expected_t(self) -> "float | None":
        """Media time the audio SHOULD be at now (wall-clock model), or None."""
        if self.proc is None or self.proc.poll() is not None:
            return None
        return self._t0 + (time.monotonic() - self._wall0)

    def playing(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None:
            try:
                p.kill()
                p.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            va_metrics._untrack(p)


def audition_wav(path, t, dur=0.35, out_path=None) -> "str | None":
    """Extract a tiny WAV snippet around t for paused-scrub audition (the GUI
    feeds it to winsound on Windows). Returns the wav path or None."""
    exe = find_ffmpeg()
    if not exe:
        return None
    import tempfile
    out_path = out_path or os.path.join(tempfile.gettempdir(), "va_audition.wav")
    args = [exe, "-y", "-v", "error", "-nostdin", "-ss", "%.3f" % max(0.0, t),
            "-i", path, "-t", "%.3f" % dur, "-map", "0:a:0", "-ac", "2",
            "-ar", "48000", "-c:a", "pcm_s16le", out_path]
    r = va_metrics._run(args, timeout=10)
    return out_path if (r and r.returncode == 0 and os.path.isfile(out_path)) else None


def render_loudness(series, w, h, target=-23.0):
    """RGB chart of momentary/short-term loudness over time (LUFS), with target line."""
    import cv2
    w, h = max(160, int(w)), max(80, int(h))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (24, 24, 24)
    lo, hi = -60.0, 0.0
    left, right, top, bot = 36, 8, 8, 18
    pw, ph = max(1, w - left - right), max(1, h - top - bot)

    def y_of(v):
        v = max(lo, min(hi, v))
        return int(top + (1 - (v - lo) / (hi - lo)) * ph)

    for db in (0, -23, -40, -60):
        y = y_of(db)
        col = (90, 90, 90) if db != int(target) else (78, 201, 176)
        cv2.line(img, (left, y), (w - right, y), col, 1, cv2.LINE_AA)
        cv2.putText(img, str(db), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    def plot(vals, color):
        vals = [v for v in vals if v is not None and v > -120]
        n = len(vals)
        if n < 2:
            return
        pts = []
        for i, v in enumerate(vals):
            x = int(left + i / (n - 1) * pw)
            pts.append((x, y_of(v)))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    plot(series.get("M", []), (90, 120, 230))     # momentary
    plot(series.get("S", []), (240, 192, 64))      # short-term
    cv2.putText(img, "Loudness LUFS  M=blue S=gold", (left, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)
    return img
