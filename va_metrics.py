#!/usr/bin/env python3
"""
va_metrics - per-frame signal metrics, events, loudness and codec internals.

Treats a video as a dataset. Primary path feeds the file straight to ffmpeg as a
normal input (`-i FILE -vf signalstats,metadata=print`) which handles Windows
paths natively; a ffprobe `movie=` path is kept as a fallback for setups that
only ship ffprobe. Pure compute - no Tkinter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

from va_ffmpeg import find_ffmpeg, find_ffprobe
import va_hwaccel

# Hide child consoles on Windows - a windowed .exe has no console for ffmpeg
# to inherit, so every spawn would otherwise flash up its own cmd window.
CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


# Ordered signalstats tags pulled per frame.
SIGNALSTATS_TAGS = [
    "YMIN", "YAVG", "YMAX", "UAVG", "VAVG",
    "SATAVG", "SATMAX", "HUEMED", "HUEAVG",
    "YDIF", "TOUT", "VREP", "BRNG",
]

# ffprobe frame time fields: 4.x emits pkt_pts_time + best_effort_timestamp_time,
# 5.0+ renamed pkt_pts_time to pts_time. Asking for both keeps every version
# happy (unknown names are silently dropped, never padded).
_TIME_FIELDS = "pts_time,best_effort_timestamp_time"


def _lavfi_escape(path: str) -> str:
    """Escape a path for a lavfi 'movie=' source. Forward slashes are accepted on
    Windows and sidestep backslash-escaping; only the drive colon needs escaping."""
    s = path.replace("\\", "/")
    return s.replace(":", "\\:").replace("'", "\\'")


def _kill(proc):
    try:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=2)
    except (OSError, subprocess.SubprocessError):
        pass


# --- Child-process control -----------------------------------------------------

_PROCS = set()                  # live engine children, so the GUI can kill_all()
_PROCS_LOCK = threading.Lock()


def _track(proc):
    with _PROCS_LOCK:
        _PROCS.add(proc)


def _untrack(proc):
    with _PROCS_LOCK:
        _PROCS.discard(proc)


def kill_all():
    """Kill every ffmpeg/ffprobe child the engine still has running (call on
    GUI close/cancel; va_audio shares this registry)."""
    with _PROCS_LOCK:
        procs = list(_PROCS)
    for p in procs:
        try:
            p.kill()
        except OSError:
            pass


def _run(args, timeout=None, cancel=None) -> "subprocess.CompletedProcess | None":
    """subprocess.run stand-in that stays killable: the child is registered for
    kill_all(), cancel() is polled every 0.25 s and timeout is enforced. Returns
    a CompletedProcess holding whatever output was produced (`.killed` is set to
    'cancel'/'timeout' when the child was put down), or None if the tool would
    not start."""
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                creationflags=CREATIONFLAGS)
    except OSError:
        return None
    _track(proc)
    out, err, killed = "", "", None
    deadline = (time.monotonic() + timeout) if timeout else None
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel():
                    killed = "cancel"
                elif deadline is not None and time.monotonic() >= deadline:
                    killed = "timeout"
                else:
                    continue
                proc.kill()
                try:
                    out, err = proc.communicate(timeout=2)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                break
    finally:
        _untrack(proc)
    rc = proc.returncode if proc.returncode is not None else -1
    r = subprocess.CompletedProcess(args, rc, out or "", err or "")
    r.killed = killed
    return r


def media_duration(path) -> float:
    """Container duration in seconds via ffprobe (0.0 when unknown)."""
    exe = find_ffprobe()
    if not exe:
        return 0.0
    r = _run([exe, "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", path], timeout=15)
    try:
        return max(0.0, float(r.stdout.strip())) if r else 0.0
    except ValueError:
        return 0.0


def _pass_timeout(path, mult=5.0, floor=300.0) -> float:
    """Kill-switch for whole-file decode passes: a generous multiple of runtime."""
    d = media_duration(path)
    return max(floor, mult * d) if d else 900.0


def _clean_segments(segs) -> list:
    """Segment hygiene shared by the event/silence parsers: swap inverted pairs,
    drop zero-length segments, drop exact repeats (echoed event lines)."""
    out = []
    for s, e in segs:
        if e < s:
            s, e = e, s
        if e <= s or (s, e) in out:
            continue
        out.append((s, e))
    return out


def _grow(acc, line, key):
    """Append `key=<float>` from an event line, skipping consecutive repeats."""
    m = re.search(key + r"[:=]\s*([-\d.]+)", line)
    if not m:
        return
    try:
        v = float(m.group(1))
    except ValueError:
        return
    if not acc or acc[-1] != v:
        acc.append(v)


class MetricsTable:
    """Per-frame metrics: a time column plus one column per signalstats tag."""

    def __init__(self):
        self.t: list = []
        self.cols: dict = {tag: [] for tag in SIGNALSTATS_TAGS}

    def __len__(self) -> int:
        return len(self.t)

    def arrays(self) -> dict:
        out = {"t": np.asarray(self.t, dtype=float)}
        for k, v in self.cols.items():
            out[k] = np.asarray(v, dtype=float)
        return out

    def summary(self) -> dict:
        a = self.arrays()
        rep = {}
        for k, v in a.items():
            if k == "t" or v.size == 0:
                continue
            good = v[~np.isnan(v)]
            if good.size == 0:
                continue
            rep[k] = {
                "min": float(good.min()), "mean": float(good.mean()),
                "max": float(good.max()), "p1": float(np.percentile(good, 1)),
                "p99": float(np.percentile(good, 99)),
            }
        return rep


def _append_row(table: MetricsTable, t: float, tags: dict):
    table.t.append(t)
    for tag in SIGNALSTATS_TAGS:
        table.cols[tag].append(tags.get(tag, float("nan")))


def signalstats(path, on_progress=None, cancel=None, timeout=None) -> MetricsTable:
    """Per-frame signalstats. Uses ffmpeg (native input) when available, else
    ffprobe (movie= source). on_progress(n) fires periodically; cancel()->True stops."""
    if find_ffmpeg():
        return _signalstats_ffmpeg(path, on_progress, cancel)
    if find_ffprobe():
        return _signalstats_ffprobe(path, on_progress, cancel)
    return MetricsTable()


def _signalstats_ffmpeg(path, on_progress, cancel) -> MetricsTable:
    exe = find_ffmpeg()
    table = MetricsTable()
    args = [exe, "-hide_banner", "-nostats"] + va_hwaccel.decode_args(path) + ["-i", path, "-vf",
            "signalstats=stat=tout+vrep+brng,metadata=print:file=-",
            "-an", "-f", "null", os.devnull]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            creationflags=CREATIONFLAGS)
    _track(proc)
    pend_t, pend = None, {}
    try:
        for line in proc.stdout:
            if cancel is not None and cancel():
                break
            line = line.strip()
            if line.startswith("frame:"):
                if pend_t is not None:
                    _append_row(table, pend_t, pend)
                    if on_progress is not None and len(table) % 30 == 0:
                        on_progress(len(table))
                m = re.search(r"pts_time:([-\d.]+)", line)
                pend_t = float(m.group(1)) if m else float(len(table))
                pend = {}
            elif line.startswith("lavfi.signalstats."):
                key, _, val = line.partition("=")
                tag = key.rsplit(".", 1)[-1]
                if tag in table.cols:
                    try:
                        pend[tag] = float(val)
                    except ValueError:
                        pass
        if pend_t is not None:
            _append_row(table, pend_t, pend)
    finally:
        _kill(proc)
        _untrack(proc)
    if on_progress is not None:
        on_progress(len(table))
    return table


def _signalstats_ffprobe(path, on_progress, cancel) -> MetricsTable:
    exe = find_ffprobe()
    table = MetricsTable()
    src = "movie=" + _lavfi_escape(path) + ",signalstats=stat=tout+vrep+brng"
    entries = "frame=" + _TIME_FIELDS + ":frame_tags=" + ",".join(
        "lavfi.signalstats." + t for t in SIGNALSTATS_TAGS)
    args = [exe, "-v", "error", "-f", "lavfi", "-i", src,
            "-show_entries", entries, "-of", "csv=p=0"]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            creationflags=CREATIONFLAGS)
    _track(proc)
    ncols = len(SIGNALSTATS_TAGS)
    try:
        for line in proc.stdout:
            if cancel is not None and cancel():
                break
            parts = line.rstrip("\n").split(",")
            if len(parts) < 1 + ncols:
                continue
            t = None
            for p in parts[:len(parts) - ncols]:   # 1 or 2 time columns by version
                try:
                    t = float(p)
                    break
                except ValueError:
                    pass
            table.t.append(t if t is not None else float(len(table.t)))
            for i, tag in enumerate(SIGNALSTATS_TAGS):
                try:
                    table.cols[tag].append(float(parts[len(parts) - ncols + i]))
                except ValueError:
                    table.cols[tag].append(float("nan"))
            if on_progress is not None and len(table) % 30 == 0:
                on_progress(len(table))
    finally:
        _kill(proc)
        _untrack(proc)
    if on_progress is not None:
        on_progress(len(table))
    return table


def flash_risk(table: MetricsTable, jump=40.0, bit_depth=None) -> list:
    """Frame indices where mean luma jumps hard enough to flag a PSE flash risk.
    `jump` is in 8-bit steps; signalstats reports native code values, so the
    threshold is rescaled by bit depth (pass the probe's, else inferred from YMAX)."""
    a = table.arrays()
    y = a.get("YAVG")
    if y is None or y.size < 2:
        return []
    if bit_depth is None:
        ref = a.get("YMAX")
        ref = ref if ref is not None and ref.size else y
        good = ref[~np.isnan(ref)]
        peak = float(good.max()) if good.size else 0.0
        bit_depth = 8 if peak <= 255 else 10 if peak <= 1023 else 12 if peak <= 4095 else 16
    thr = jump * ((2 ** int(bit_depth) - 1) / 255.0)
    d = np.abs(np.diff(y))
    return [int(i + 1) for i in np.where(d >= thr)[0]]


# --- Events ------------------------------------------------------------------

def black_segments(path, cancel=None) -> list:
    """[(start_s, end_s), ...] of black passages via ffmpeg blackdetect."""
    exe = find_ffmpeg()
    if not exe:
        return []
    args = [exe, "-hide_banner"] + va_hwaccel.decode_args(path) + ["-i", path, "-vf",
            "blackdetect=d=0.05:pic_th=0.98", "-an", "-f", "null", os.devnull]
    r = _run(args, timeout=_pass_timeout(path), cancel=cancel)
    segs = []
    for line in (r.stderr.splitlines() if r else []):
        if "black_start" not in line:
            continue
        d = {}
        for tok in line.split():
            if tok.startswith(("black_start:", "black_end:", "black_duration:")):
                k, v = tok.split(":", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
        if "black_start" in d:
            end = d.get("black_end", d["black_start"] + d.get("black_duration", 0.0))
            segs.append((d["black_start"], end))
    return _clean_segments(segs)


def freeze_segments(path, cancel=None) -> list:
    """[(start_s, end_s), ...] of frozen passages via ffmpeg freezedetect.
    A freeze that runs to EOF has no freeze_end - the stream duration fills in."""
    exe = find_ffmpeg()
    if not exe:
        return []
    args = [exe, "-hide_banner"] + va_hwaccel.decode_args(path) + ["-i", path, "-vf",
            "freezedetect=n=-60dB:d=0.5,metadata=mode=print:file=-",
            "-an", "-f", "null", os.devnull]
    r = _run(args, timeout=_pass_timeout(path), cancel=cancel)
    if r is None:
        return []
    starts, ends = [], []
    for line in r.stdout.splitlines():     # metadata print (clean, once per event)
        _grow(starts, line, "freeze_start")
        _grow(ends, line, "freeze_end")
    if not starts:                          # ffprobe-less fallback: the stderr log
        for line in r.stderr.splitlines():  # (parsing both double-counted EOF
            _grow(starts, line, "freeze_start")  # freezes into a phantom segment)
            _grow(ends, line, "freeze_end")
    dur = media_duration(path) if len(starts) > len(ends) else 0.0
    segs = []
    for i, s in enumerate(starts):
        segs.append((s, ends[i] if i < len(ends) else max(dur, s)))
    return _clean_segments(segs)


def scene_cuts(path, threshold=0.3, cancel=None) -> list:
    """Times (s) of detected scene cuts. ffmpeg native input preferred."""
    exe = find_ffmpeg()
    if exe:
        args = [exe, "-hide_banner", "-nostats"] + va_hwaccel.decode_args(path) + ["-i", path, "-vf",
                "select='gt(scene\\," + ("%g" % threshold) + ")',metadata=print:file=-",
                "-an", "-f", "null", os.devnull]
        r = _run(args, timeout=_pass_timeout(path), cancel=cancel)
        cuts = []
        for line in (r.stdout.splitlines() if r else []):
            if line.startswith("frame:"):
                _grow(cuts, line, "pts_time")
        return cuts
    exe = find_ffprobe()
    if not exe:
        return []
    src = ("movie=" + _lavfi_escape(path) +
           ",select='gt(scene," + ("%g" % threshold) + ")'")
    args = [exe, "-v", "error", "-f", "lavfi", "-i", src,
            "-show_entries", "frame=" + _TIME_FIELDS, "-of", "csv=p=0"]
    r = _run(args, timeout=_pass_timeout(path), cancel=cancel)
    cuts = []
    for line in (r.stdout.splitlines() if r else []):
        for part in line.strip().split(","):
            try:
                v = float(part)
            except ValueError:
                continue
            if not cuts or cuts[-1] != v:
                cuts.append(v)
            break
    return cuts


# --- Audio loudness (EBU R128) -----------------------------------------------

def loudness(path, cancel=None) -> "dict | None":
    """Integrated loudness / LRA / true peak via ffmpeg ebur128. None if no audio.
    A digitally silent track measures true_peak_dbfs = -inf (present, not missing)."""
    exe = find_ffmpeg()
    if not exe:
        return None
    args = [exe, "-hide_banner", "-i", path, "-vn", "-sn", "-dn",
            "-af", "ebur128=peak=true", "-f", "null", os.devnull]
    r = _run(args, timeout=max(120.0, 1.5 * media_duration(path)), cancel=cancel)
    if r is None:
        return None
    txt = r.stderr
    if "Summary:" not in txt:
        return None
    txt = txt[txt.rfind("Summary:"):]

    def grab(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None

    out = {
        "integrated_lufs": grab(r"I:\s*(-?inf|[-\d.]+)\s*LUFS"),
        "lra_lu": grab(r"LRA:\s*(-?inf|[-\d.]+)\s*LU"),
        "threshold_lufs": grab(r"Threshold:\s*(-?inf|[-\d.]+)\s*LUFS"),
        "true_peak_dbfs": grab(r"Peak:\s*(-?inf|[-\d.]+)\s*dBFS"),
    }
    if all(v is None for v in out.values()):
        return None
    return out


# --- Codec internals ---------------------------------------------------------

def frame_sizes(path, cancel=None) -> dict:
    """Per-frame {t, size_bytes, pict_type} for a bitrate-over-time / GOP view."""
    exe = find_ffprobe()
    out = {"t": [], "size": [], "type": [], "key": []}
    if not exe:
        return out
    # key=value rows: the field set varies by ffprobe version, so no positions.
    args = [exe, "-v", "error", "-select_streams", "v", "-show_entries",
            "frame=" + _TIME_FIELDS + ",pkt_size,pict_type,key_frame",
            "-of", "compact=p=0:nk=0", path]
    r = _run(args, timeout=_pass_timeout(path), cancel=cancel)
    for line in (r.stdout.splitlines() if r else []):
        d = dict(p.split("=", 1) for p in line.strip().split("|") if "=" in p)
        if not d:
            continue
        try:
            out["t"].append(float(d.get("pts_time") or d.get("best_effort_timestamp_time") or ""))
        except ValueError:
            out["t"].append(float(len(out["t"])))
        try:
            out["size"].append(int(d.get("pkt_size") or ""))
        except ValueError:
            out["size"].append(0)
        out["type"].append(d.get("pict_type") or "?")
        out["key"].append(1 if d.get("key_frame") == "1" else 0)
    return out


def motion_series(source, step=1, max_side=240) -> dict:
    """Mean optical-flow magnitude per frame (camera/subject motion over time).

    ``source`` is a va_ffmpeg.VideoSource. Farneback flow on downscaled grayscale
    frames. Returns {t, motion}."""
    try:
        import cv2
    except ImportError:
        return {"t": [], "motion": []}
    out = {"t": [], "motion": []}
    source.start(0)
    prev = None
    idx = 0
    while True:
        frame = source.read()
        if frame is None:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            s = max_side / max(h, w) if max(h, w) > max_side else 1.0
            small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                out["t"].append(idx / (source.fps or 30.0))
                out["motion"].append(float(mag.mean()))
            prev = gray
        idx += 1
    source.close()
    return out

# --- Combined single-decode analysis pass --------------------------------------
#
# The passes above each decode the whole file once. analyze_pass() computes all
# of them in ONE ffmpeg process / ONE decode via -filter_complex: the decoded
# frames fan out to signalstats, blackdetect, freezedetect and scene-cut select,
# and the audio (when present) runs ebur128 + silencedetect in the same process.
# Decoding dominates the cost of every one of these measurements, so this is the
# single biggest power-efficiency lever in the engine: the same numbers for
# roughly one-quarter of the work. Filters do not modify pixels, so results are
# bit-identical to the separate passes (selftest asserts parity). Hardware
# decode (va_hwaccel) moves the remaining decode onto fixed-function silicon.

def _black_from_lines(lines) -> list:
    """Parse blackdetect log lines into [(start, end), ...]."""
    segs = []
    for line in lines:
        if "black_start" not in line:
            continue
        d = {}
        for tok in line.split():
            if tok.startswith(("black_start:", "black_end:", "black_duration:")):
                k, v = tok.split(":", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
        if "black_start" in d:
            end = d.get("black_end", d["black_start"] + d.get("black_duration", 0.0))
            segs.append((d["black_start"], end))
    return _clean_segments(segs)


def _pair_to_eof(starts, ends, path) -> list:
    """Pair start/end event lists; an event running to EOF gets the duration."""
    dur = media_duration(path) if len(starts) > len(ends) else 0.0
    segs = []
    for i, s in enumerate(starts):
        segs.append((s, ends[i] if i < len(ends) else max(dur, s)))
    return _clean_segments(segs)


def _parse_r128_meta(text) -> tuple:
    """(t, M, S) momentary/short-term loudness series from ametadata=print output."""
    t, M, S = [], [], []
    cur_t = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([-\d.]+)", line)
            cur_t = float(m.group(1)) if m else (t[-1] if t else 0.0)
        elif line.startswith("lavfi.r128.M="):
            try:
                M.append(float(line.split("=", 1)[1]))
                t.append(cur_t if cur_t is not None else len(M) * 0.1)
            except ValueError:
                pass
        elif line.startswith("lavfi.r128.S="):
            try:
                S.append(float(line.split("=", 1)[1]))
            except ValueError:
                pass
    return t, M, S


def _parse_r128_summary(text) -> "dict | None":
    if "Summary:" not in text:
        return None
    s = text[text.rfind("Summary:"):]

    def grab(p):
        m = re.search(p, s)
        return float(m.group(1)) if m else None

    out = {"integrated_lufs": grab(r"I:\s*(-?inf|[-\d.]+)\s*LUFS"),
           "lra_lu": grab(r"LRA:\s*(-?inf|[-\d.]+)\s*LU"),
           "threshold_lufs": grab(r"Threshold:\s*(-?inf|[-\d.]+)\s*LUFS"),
           "true_peak_dbfs": grab(r"Peak:\s*(-?inf|[-\d.]+)\s*dBFS")}
    return None if all(v is None for v in out.values()) else out


def analyze_pass(path, on_progress=None, cancel=None, scene_threshold=0.3,
                 want_audio=True, silence_db=-50, silence_d=0.5):
    """Whole-file analysis in a single decode. Returns
    {table, events:{black,freeze,scene_cuts}, audio:{summary,t,M,S}|None,
     silence:[(s,e)], cancelled, elapsed, decode, passes_merged}
    or None when the combined graph could not run - callers then fall back to
    the individual passes. on_progress(n_frames) streams like signalstats()."""
    exe = find_ffmpeg()
    if not exe:
        return None
    try:
        import va_perf
        thread_args = list(va_perf.ffmpeg_thread_args())
    except Exception:
        thread_args = []
    has_aud = False
    if want_audio:
        try:
            import va_audio
            has_aud = bool(va_audio.has_audio(path))
        except Exception:
            has_aud = False
    graph = ("[0:v]signalstats=stat=tout+vrep+brng,metadata=print:file=-,"
             "blackdetect=d=0.05:pic_th=0.98,"
             "freezedetect=n=-60dB:d=0.5,"
             "metadata=mode=print:key=lavfi.freezedetect.freeze_start:file=fs.txt,"
             "metadata=mode=print:key=lavfi.freezedetect.freeze_end:file=fe.txt,"
             "split=2[vmain][vsc];"
             "[vsc]select='gt(scene\\,%g)',metadata=print:file=sc.txt[vcut]"
             % scene_threshold)
    if has_aud:
        graph += (";[0:a]ebur128=peak=true:metadata=1,ametadata=print:file=au.txt,"
                  "silencedetect=n=%ddB:d=%g[amain]" % (silence_db, silence_d))
    tmpdir = tempfile.mkdtemp(prefix="va_pass_")
    args = ([exe, "-hide_banner", "-nostats"] + thread_args +
            va_hwaccel.decode_args(path) +
            ["-i", path, "-filter_complex", graph, "-map", "[vmain]"])
    if has_aud:
        args += ["-map", "[amain]"]
    args += ["-f", "null", os.devnull, "-map", "[vcut]", "-f", "null", os.devnull]
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                cwd=tmpdir, creationflags=CREATIONFLAGS)
    except OSError:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None
    _track(proc)
    err_lines: list = []

    def _drain():
        try:
            for ln in proc.stderr:
                if len(err_lines) < 500000:
                    err_lines.append(ln)
        except (OSError, ValueError):
            pass

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    timed_out = []
    dog = threading.Timer(_pass_timeout(path), lambda: (timed_out.append(1), proc.kill()))
    dog.daemon = True
    dog.start()
    table = MetricsTable()
    cancelled = False
    pend_t, pend = None, {}
    try:
        for line in proc.stdout:
            if cancel is not None and cancel():
                cancelled = True
                proc.kill()
                break
            line = line.strip()
            if line.startswith("frame:"):
                if pend_t is not None:
                    _append_row(table, pend_t, pend)
                    if on_progress is not None and len(table) % 30 == 0:
                        on_progress(len(table))
                m = re.search(r"pts_time:([-\d.]+)", line)
                pend_t = float(m.group(1)) if m else float(len(table))
                pend = {}
            elif line.startswith("lavfi.signalstats."):
                key, _, val = line.partition("=")
                tag = key.rsplit(".", 1)[-1]
                if tag in table.cols:
                    try:
                        pend[tag] = float(val)
                    except ValueError:
                        pass
        if pend_t is not None:
            _append_row(table, pend_t, pend)
    finally:
        dog.cancel()
        try:
            proc.wait(timeout=15)
        except (OSError, subprocess.SubprocessError):
            proc.kill()
        drain.join(timeout=5)
        _kill(proc)
        _untrack(proc)
    cancelled = cancelled or bool(timed_out)

    def _file(name):
        try:
            with open(os.path.join(tmpdir, name), encoding="utf-8",
                      errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    fs_txt, fe_txt, sc_txt = _file("fs.txt"), _file("fe.txt"), _file("sc.txt")
    au_txt = _file("au.txt") if has_aud else ""
    shutil.rmtree(tmpdir, ignore_errors=True)
    err = "".join(err_lines)
    if len(table) == 0 and proc.returncode not in (0, None) and not cancelled:
        return None                       # graph refused to run -> fall back
    if on_progress is not None:
        on_progress(len(table))

    f_starts, f_ends = [], []
    for line in fs_txt.splitlines():
        _grow(f_starts, line, "freeze_start")
    for line in fe_txt.splitlines():
        _grow(f_ends, line, "freeze_end")
    cuts: list = []
    for line in sc_txt.splitlines():
        if line.startswith("frame:"):
            _grow(cuts, line, "pts_time")
    s_starts, s_ends = [], []
    for line in err.splitlines():
        _grow(s_starts, line, "silence_start")
        _grow(s_ends, line, "silence_end")
    audio = None
    if has_aud:
        t, M, S = _parse_r128_meta(au_txt)
        audio = {"summary": _parse_r128_summary(err), "t": t, "M": M, "S": S}
    events = {"black": _black_from_lines(err.splitlines()),
              "freeze": _pair_to_eof(f_starts, f_ends, path),
              "scene_cuts": cuts}
    return {"table": table, "events": events, "audio": audio,
            "silence": _pair_to_eof(s_starts, s_ends, path) if has_aud else [],
            "cancelled": cancelled, "elapsed": time.monotonic() - t0,
            "decode": (va_hwaccel.decode_method(path) or "cpu"),
            "passes_merged": 4 + (2 if has_aud else 0)}

