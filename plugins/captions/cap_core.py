"""Caption / subtitle QC.

Extracts embedded text subtitle tracks (SubRip/ASS/mov_text/WebVTT via
ffmpeg's SRT converter) or a same-stem .srt sidecar, builds a speech map
from an audio silence scan, and QCs the cues the way a compliance reviewer
would: presence, speech coverage, systematic sync offset (median cue-start
to nearest speech onset), reading speed (chars/sec), minimum duration,
overlaps, and line-length limits. Bitmap tracks (PGS/DVB/VobSub) and EIA-608
closed captions are reported present-but-not-decoded.

Compute only - no Tk. The QC core (qc_cues) is pure so the selftest feeds it
synthetic cue/speech timelines."""

from __future__ import annotations

import os
import re
import subprocess

from va_ffmpeg import find_ffmpeg, ffprobe_json

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
BITMAP_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub",
                 "dvb_teletext"}

OPTS = {
    "cps_warn": 21.0,        # Netflix-style adult reading speed ceiling
    "min_dur_s": 0.833,      # 5/6 s minimum display time
    "max_line_chars": 42,
    "max_lines": 2,
    "sync_warn_s": 0.30,     # systematic offset beyond this = out of sync
    "sync_window_s": 2.5,    # how far to look for the matching speech onset
    "coverage_warn_pct": 70.0,
    "silence_db": -35.0,
    "silence_min_s": 0.40,
}


# --- discovery / extraction ----------------------------------------------------

def probe_subs(path) -> dict:
    """{"text": [stream...], "bitmap": [...], "cc": bool, "duration": s}"""
    pd = ffprobe_json(path) or {}
    text, bitmap = [], []
    cc = False
    sidx = 0
    for s in pd.get("streams", []):
        if s.get("codec_type") == "video" and str(s.get("closed_captions", 0)) not in ("0", "", "None"):
            cc = True
        if s.get("codec_type") != "subtitle":
            continue
        info = {"sidx": sidx, "codec": s.get("codec_name", "?"),
                "lang": (s.get("tags") or {}).get("language", ""),
                "title": (s.get("tags") or {}).get("title", "")}
        (text if info["codec"] in TEXT_CODECS else bitmap).append(info)
        sidx += 1
    try:
        dur = float((pd.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    return {"text": text, "bitmap": bitmap, "cc": cc, "duration": dur}


def find_sidecar(path) -> "str | None":
    for ext in (".srt", ".en.srt"):
        p = os.path.splitext(path)[0] + ext
        if os.path.isfile(p):
            return p
    return None


_TS = r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})"


def parse_srt(text: str) -> list:
    """[(start_s, end_s, text)] - tolerant block parser, tags stripped."""
    cues = []
    for m in re.finditer(_TS + r"\s*-->\s*" + _TS + r"[^\n]*\n(.*?)(?:\n\s*\n|\Z)",
                         text.replace("\r\n", "\n").replace("\r", "\n"),
                         re.DOTALL):
        g = [int(x) for x in m.groups()[:8]]
        t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = re.sub(r"<[^>]+>|\{\\[^}]*\}", "", m.group(9)).strip()
        if body:
            cues.append((t0, t1, body))
    cues.sort(key=lambda c: c[0])
    return cues


def extract_cues(path, sidx, timeout=300) -> "list | None":
    """Convert one embedded text track to SRT via ffmpeg and parse it."""
    exe = find_ffmpeg()
    if not exe:
        return None
    args = [exe, "-v", "error", "-nostdin", "-i", path,
            "-map", "0:s:%d" % sidx, "-f", "srt", "-"]
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout,
                           creationflags=CREATIONFLAGS)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    return parse_srt(r.stdout.decode("utf-8", "replace"))


def silence_map(path, db=OPTS["silence_db"], min_s=OPTS["silence_min_s"],
                timeout=600) -> "list | None":
    """[(start, end)] silent stretches from one audio-only silencedetect pass."""
    exe = find_ffmpeg()
    if not exe:
        return None
    args = [exe, "-v", "info", "-nostdin", "-i", path, "-vn", "-sn", "-dn",
            "-af", "silencedetect=noise=%gdB:d=%g" % (db, min_s),
            "-f", "null", "-"]
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout,
                           creationflags=CREATIONFLAGS)
    except (OSError, subprocess.SubprocessError):
        return None
    txt = r.stderr.decode("utf-8", "replace")
    if "Stream map" in txt and "matches no streams" in txt:
        return None
    starts = [float(m.group(1)) for m in
              re.finditer(r"silence_start:\s*([\d.]+)", txt)]
    ends = [float(m.group(1)) for m in
            re.finditer(r"silence_end:\s*([\d.]+)", txt)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        out.append((s, e if e is not None else 1e9))
    return out


def speech_spans(silence, duration) -> list:
    """Complement of the silence map inside [0, duration]."""
    spans, t = [], 0.0
    for s, e in sorted(silence or []):
        if s > t:
            spans.append((t, min(s, duration)))
        t = max(t, e)
    if t < duration:
        spans.append((t, duration))
    return [(a, b) for a, b in spans if b - a > 0.05]


# --- QC core (pure) -------------------------------------------------------------

def qc_cues(cues, speech, duration, opts=None) -> dict:
    o = dict(OPTS)
    o.update(opts or {})
    events, viol = [], []
    n = len(cues)
    # coverage of speech by cues
    cov = 0.0
    sp_total = sum(b - a for a, b in speech) or 1e-9
    for a, b in speech:
        for t0, t1, _ in cues:
            lo, hi = max(a, t0), min(b, t1)
            if hi > lo:
                cov += hi - lo
    cov_pct = 100.0 * cov / sp_total
    # systematic sync: cue start vs nearest speech onset
    onsets = [a for a, _b in speech]
    deltas = []
    for t0, _t1, _ in cues:
        near = min(onsets, key=lambda x: abs(t0 - x)) if onsets else None
        if near is not None and abs(t0 - near) <= o["sync_window_s"]:
            deltas.append(t0 - near)
    deltas.sort()
    med = deltas[len(deltas) // 2] if deltas else 0.0
    mad = (sorted(abs(d - med) for d in deltas)[len(deltas) // 2]
           if deltas else 0.0)
    # per-cue checks
    cps_over = overlaps = short = longlines = 0
    for i, (t0, t1, txt) in enumerate(cues):
        dur = max(t1 - t0, 1e-3)
        cps = len(txt.replace("\n", " ")) / dur
        if cps > o["cps_warn"]:
            cps_over += 1
            events.append((t0, "warn", "reading speed %.0f cps (limit %.0f): %r"
                           % (cps, o["cps_warn"], txt[:40])))
        if dur < o["min_dur_s"]:
            short += 1
            events.append((t0, "info", "cue shown only %dms" % round(dur * 1000)))
        lines = txt.split("\n")
        if len(lines) > o["max_lines"] or any(len(l) > o["max_line_chars"]
                                              for l in lines):
            longlines += 1
            events.append((t0, "info", "line layout: %d line(s), longest %d chars"
                           % (len(lines), max(len(l) for l in lines))))
        if i + 1 < n and cues[i + 1][0] < t1 - 1e-3:
            overlaps += 1
            events.append((t0, "warn", "overlaps next cue (%0.2fs overlap)"
                           % (t1 - cues[i + 1][0])))
        if t1 > duration + 1.0:
            events.append((t0, "warn", "cue runs past end of video"))
    if deltas and abs(med) > o["sync_warn_s"] and mad < 0.45:
        events.insert(0, (0.0, "warn",
                          "systematic sync offset: cues start %+.2fs vs speech "
                          "onsets (median of %d, spread %.2fs) - shift the track"
                          % (med, len(deltas), mad)))
    if cov_pct < o["coverage_warn_pct"]:
        events.insert(0, (0.0, "warn",
                          "only %.0f%% of speech time is captioned" % cov_pct))
    events.sort(key=lambda e: e[0])
    return {"cues": n, "coverage_pct": round(cov_pct, 1),
            "sync_median_s": round(med, 3), "sync_mad_s": round(mad, 3),
            "sync_n": len(deltas), "cps_over": cps_over, "overlaps": overlaps,
            "short": short, "longlines": longlines, "events": events}


# --- orchestrator ----------------------------------------------------------------

def analyze(path, sidx=None, on_progress=None) -> dict:
    """Full captions QC for one file. sidx=None -> first embedded text track,
    falling back to a .srt sidecar."""
    say = on_progress or (lambda m: None)
    say("probing subtitle tracks")
    info = probe_subs(path)
    cues, source = None, None
    if sidx is not None:
        say("extracting track s:%d" % sidx)
        cues = extract_cues(path, sidx)
        source = "embedded s:%d" % sidx
    elif info["text"]:
        s = info["text"][0]
        say("extracting track s:%d" % s["sidx"])
        cues = extract_cues(path, s["sidx"])
        source = "embedded s:%d (%s%s)" % (s["sidx"], s["codec"],
                                           " " + s["lang"] if s["lang"] else "")
    if cues is None:
        side = find_sidecar(path)
        if side:
            say("reading sidecar")
            try:
                cues = parse_srt(open(side, "r", encoding="utf-8",
                                      errors="replace").read())
                source = "sidecar %s" % os.path.basename(side)
            except OSError:
                cues = None
    rep = {"ok": True, "streams": info, "source": source,
           "duration": info["duration"]}
    if cues is None:
        rep["qc"] = None
        return rep
    say("scanning audio for the speech map")
    sil = silence_map(path)
    speech = speech_spans(sil, info["duration"]) if sil is not None else []
    rep["speech_spans"] = len(speech)
    rep["qc"] = qc_cues(cues, speech, info["duration"])
    rep["cue_preview"] = [(round(a, 2), round(b, 2), t[:60])
                          for a, b, t in cues[:8]]
    return rep


def marks(rep: dict) -> list:
    q = (rep or {}).get("qc")
    if not q:
        return []
    return [(t, txt[:46]) for t, _sev, txt in q["events"] if t > 0.0][:200]


def render_report(rep: dict) -> str:
    if not rep.get("ok"):
        return "Captions QC failed: %s" % rep.get("error", "?")
    st = rep["streams"]
    lines = ["CAPTIONS / SUBTITLES QC", "",
             "embedded text tracks : %d  %s"
             % (len(st["text"]), ["%s s:%d %s" % (t["codec"], t["sidx"],
                                                  t["lang"]) for t in st["text"]]
                if st["text"] else ""),
             "bitmap tracks        : %d  (PGS/DVB/VobSub - presence only, "
             "not decoded)" % len(st["bitmap"]),
             "EIA-608/708 CC flag  : %s" % ("present (not decoded)"
                                            if st["cc"] else "none")]
    if rep.get("source"):
        lines.append("QC source            : %s" % rep["source"])
    q = rep.get("qc")
    if q is None:
        lines += ["", "No text track or .srt sidecar to QC. Bitmap/CC tracks "
                      "need an OCR pass (not implemented)."]
        return "\n".join(lines)
    lines += ["",
              "cues %d · speech coverage %.0f%% · sync median %+.2fs "
              "(n=%d, spread %.2fs)"
              % (q["cues"], q["coverage_pct"], q["sync_median_s"],
                 q["sync_n"], q["sync_mad_s"]),
              "reading-speed violations %d · overlaps %d · too-short %d · "
              "line-layout %d" % (q["cps_over"], q["overlaps"], q["short"],
                                  q["longlines"]), ""]
    for t, sev, txt in q["events"][:50]:
        m, s = divmod(t, 60.0)
        lines.append("[%s] %d:%05.2f  %s" % (sev.upper(), int(m), s, txt))
    if len(q["events"]) > 50:
        lines.append("... %d more (timeline-marked)" % (len(q["events"]) - 50))
    return "\n".join(lines)


# --- QC checks (profile "captions") ---------------------------------------------

def _r(ctx):
    """Captions report: own ctx key (CLI), the GUI's cached-results bridge,
    or the legacy shared key."""
    return (ctx.get("captions") or (ctx.get("adv") or {}).get("captions")
            or ctx.get("verify") or {})


def _q(ctx):
    return (_r(ctx).get("qc")) or None


def chk_presence():
    def f(ctx):
        rep = _r(ctx)
        st = rep.get("streams") or {}
        if rep.get("qc") is not None:
            return "pass", "QC'd %s" % (rep.get("source") or "?")
        if st.get("bitmap") or st.get("cc"):
            return "warn", "only bitmap/CC captions (not decoded here)"
        if not rep:
            return "pass", "not measured (run the captions QC)"
        return "fail", "no caption track or sidecar found"
    return ("cap_presence", "Captions present", f)


def chk_coverage(warn_pct=OPTS["coverage_warn_pct"]):
    def f(ctx):
        q = _q(ctx)
        if q is None:
            return "pass", "not measured"
        if q["coverage_pct"] < warn_pct:
            return "warn", "%.0f%% of speech captioned (target %.0f%%)" % (
                q["coverage_pct"], warn_pct)
        return "pass", "%.0f%% of speech captioned" % q["coverage_pct"]
    return ("cap_coverage", "Speech coverage", f)


def chk_sync(warn_s=OPTS["sync_warn_s"]):
    def f(ctx):
        q = _q(ctx)
        if q is None:
            return "pass", "not measured"
        if abs(q["sync_median_s"]) > warn_s and q["sync_mad_s"] < 0.45:
            return "warn", "systematic offset %+.2fs" % q["sync_median_s"]
        return "pass", "median offset %+.2fs" % q["sync_median_s"]
    return ("cap_sync", "Caption sync", f)


def chk_speed():
    def f(ctx):
        q = _q(ctx)
        if q is None:
            return "pass", "not measured"
        if q["cps_over"]:
            return "warn", "%d cue(s) above reading-speed limit" % q["cps_over"]
        return "pass", "reading speed within limits"
    return ("cap_speed", "Reading speed", f)


def chk_layout():
    def f(ctx):
        q = _q(ctx)
        if q is None:
            return "pass", "not measured"
        bad = q["overlaps"] + q["short"]
        if q["overlaps"]:
            return "warn", "%d overlapping cue(s)" % q["overlaps"]
        if bad:
            return "pass", "%d short cue(s), %d layout note(s)" % (
                q["short"], q["longlines"])
        return "pass", "timing/layout clean"
    return ("cap_layout", "Cue timing & layout", f)


def qc_checks():
    return [chk_presence(), chk_coverage(), chk_sync(), chk_speed(),
            chk_layout()]
