#!/usr/bin/env python3
"""
sr_verify (speedrun plugin) - submission / run verification: "is this one continuous,
unmanipulated capture?" Built for moderation workflows (speedrun verification,
contest entries, evidence intake) on top of the va_forensics battery, adding
the context that battery lacks:

  * platform_screen() - detects platform / pipeline re-encodes (YouTube, HLS
        stream captures, ffmpeg pipelines). A platform re-encode rewrites
        keyframe structure, encoder traces and noise statistics, so several
        forensic signals describe the PLATFORM's encoder rather than the
        submitted capture. The screen reports which checks lost power instead
        of letting them mislead.
  * tempo_check() - speed-manipulation screen: regular duplicate-frame
        insertion (unique-content rate below the container rate), frame-rate
        bookkeeping mismatches, and an audio spectral-cutoff probe
        (resample-based slowdowns drag the audio band down with them).
  * verify_run() - one call: forensics battery + both screens + plain-language
        verdicts with confidence ratings. render_verify() formats them for a
        moderator, with the engineer-grade forensics render appended.

Every signal is an INDICATOR to corroborate, never proof. A clean screen does
not certify a run; a flagged one means "watch these spots / request the local
recording", not "ban". Pure engine module: no Tk; ffmpeg via va_ffmpeg.
"""

from __future__ import annotations

import os
import textwrap
import subprocess

import numpy as np

from va_ffmpeg import find_ffmpeg, ffprobe_json, VideoSource
import va_forensics
import va_temporal
import va_audio
import va_metrics

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

# verdict/QC check ids whose evidence a platform re-encode rewrites
_WEAKENED_BY_REENCODE = ("splices", "recompress", "fingerprint", "noise_floor")


def _rate(txt) -> float:
    """'60000/1001' -> 59.94; tolerant of junk/None."""
    try:
        if isinstance(txt, (int, float)):
            return float(txt)
        num, _, den = str(txt or "").partition("/")
        n = float(num)
        d = float(den) if den else 1.0
        return n / d if d else 0.0
    except (TypeError, ValueError):
        return 0.0


# === Source / platform screen =================================================

def platform_screen(path, fingerprint=None) -> dict:
    """Whose encoder produced the bytes being judged? Returns
    {platform, reencoded, evidence[], weakened[], advice}. `weakened` lists
    check ids whose evidence no longer describes the original capture.
    reencoded: True (platform re-encode) / False (remux, original bitstream
    survives) / None (unknown). Pass fingerprint= a precomputed
    va_forensics.encoder_fingerprint() to avoid re-scanning."""
    out = {"platform": None, "reencoded": None, "evidence": [],
           "weakened": [], "advice": None}
    fp = fingerprint if fingerprint is not None else va_forensics.encoder_fingerprint(path)
    tags = fp.get("tags") or {}
    handlers = [str(h) for h in (tags.get("handlers") or [])]
    markers = fp.get("markers") or []
    encs = [str(tags.get("format_encoder") or "")] + \
           [str(s) for s in (tags.get("stream_encoders") or [])]

    if any("Google" in h for h in handlers):
        out["platform"] = "youtube"
        out["reencoded"] = True
        out["evidence"].append(
            "stream handler 'ISO Media file produced by Google Inc.' - this is "
            "YouTube's re-encode, not the file that was uploaded")
    pd = ffprobe_json(path) or {}
    fmt_name = str((pd.get("format") or {}).get("format_name") or "")
    if "mpegts" in fmt_name:
        out["platform"] = out["platform"] or "stream-capture"
        out["evidence"].append(
            "MPEG-TS container - a live-stream / HLS segment capture; "
            "disconnect-resume artifacts can mimic splices")
    lavf = any("Lavf" in e for e in encs) or ("ffmpeg-mux (Lavf)" in markers)
    if lavf:
        out["evidence"].append(
            "ffmpeg (Lavf) muxer traces - re-muxed or re-encoded by an "
            "ffmpeg-based tool (OBS, yt-dlp and most platforms alike)")
        out["platform"] = out["platform"] or "ffmpeg-pipeline"
    if fp.get("settings") and out["platform"] != "youtube":
        # the original encoder's SEI survived: the video stream was COPIED by
        # the last tool, so bitstream evidence still belongs to the capture
        out["evidence"].append(
            "encoder settings SEI present in the bitstream - this is an "
            "original encoder output (a platform re-encode would have "
            "replaced it)")
        if out["reencoded"] is None and out["platform"] in ("ffmpeg-pipeline",
                                                            "stream-capture"):
            out["reencoded"] = False
    if out["reencoded"]:
        out["weakened"] = list(_WEAKENED_BY_REENCODE)
        out["advice"] = (
            "Platform re-encode: keyframe cadence, encoder fingerprint, "
            "recompression and noise statistics now describe the platform's "
            "encoder, not the submitted capture. Treat those checks as weak "
            "here and request the runner's LOCAL RECORDING for anything "
            "contentious.")
    return out


# === Tempo / speed-manipulation screen =======================================

_TEMPO_CAVEATS = [
    "30 fps captures upsampled to 60 fps, emulator frame pacing and encoder "
    "frame drops all produce duplicate frames legitimately - check what the "
    "runner claims to have captured with before reading this as manipulation",
    "speed-up by frame DROPPING leaves no duplicate trail, and small (<25%) "
    "tempo changes hide inside normal encoder variation - corroborate with "
    "audio pitch and in-game timers",
]


def tempo_check(path, cadence=None, cancel=None) -> dict:
    """Speed-manipulation screen fusing three independent signals: regular
    duplicate-frame insertion (video), container-vs-measured frame-rate
    bookkeeping, and the audio spectral cutoff. Pass cadence= a precomputed
    va_temporal.cadence() to skip the decode pass. Returns
    {score 0..1, label, findings[{severity,text}], video{}, audio{}, caveats[]}."""
    findings = []
    score = 0.0
    vid = {}
    pd = ffprobe_json(path) or {}
    vst = next((s for s in pd.get("streams", [])
                if s.get("codec_type") == "video"), None) or {}
    r_fps = _rate(vst.get("r_frame_rate"))
    a_fps = _rate(vst.get("avg_frame_rate"))
    vid["container_fps"] = round(r_fps, 3)
    vid["average_fps"] = round(a_fps, 3)
    if r_fps > 0 and a_fps > 0 and abs(r_fps - a_fps) / max(r_fps, a_fps) > 0.02:
        findings.append({"severity": "info", "text":
                         "frame-rate bookkeeping disagrees: container ticks at %.3f fps "
                         "but frames average %.3f fps - variable timing or a re-timed "
                         "edit" % (r_fps, a_fps)})
        score += 0.15

    cad = cadence
    if cad is None:
        try:
            cad = va_temporal.cadence(VideoSource(path))
        except Exception:
            cad = None
    if cad:
        dup_idx = cad.get("dup_frames") or []
        dup_pct = float(cad.get("dup_pct") or 0.0)
        fps = float(cad.get("fps") or 0.0) or (a_fps or 60.0)
        vid.update({"dup_pct": dup_pct, "cadence": cad.get("cadence"),
                    "fps": round(fps, 3)})
        regular, med_iv = False, None
        if len(dup_idx) >= 6:
            iv = np.diff(np.asarray(dup_idx, dtype=float))
            iv = iv[iv > 0]
            if iv.size >= 5:
                med_iv = float(np.median(iv))
                mad = float(np.median(np.abs(iv - med_iv)))
                regular = med_iv >= 2.0 and (mad / med_iv) < 0.25
        content_fps = fps * (1.0 - dup_pct / 100.0)
        vid["regular_cadence"] = bool(regular)
        vid["content_fps"] = round(content_fps, 2)
        telecine = "telecine" in (cad.get("cadence") or "")
        if regular and dup_pct >= 12.0 and not telecine:
            findings.append({"severity": "warn", "text":
                             "duplicate frames inserted on a regular cadence (every ~%.0f "
                             "frames, %.0f%% of frames): unique content runs at ~%.1f fps "
                             "inside a %.1f fps container - the signature of footage "
                             "captured or slowed at a lower rate and re-timed to the "
                             "container rate" % (med_iv, dup_pct, content_fps, fps)})
            score += 0.45
        elif dup_pct >= 40.0 and not telecine:
            findings.append({"severity": "warn", "text":
                             "%.0f%% duplicate frames - the true content rate is ~%.1f fps, "
                             "not the container's %.1f fps" % (dup_pct, content_fps, fps)})
            score += 0.3
        elif dup_pct >= 12.0 and not telecine:
            findings.append({"severity": "info", "text":
                             "%.0f%% duplicate frames on an irregular pattern - looks like "
                             "lag or encoder drops rather than re-timing" % dup_pct})
            score += 0.1

    cut = None
    try:
        if va_audio.has_audio(path):
            cut = va_audio.audio_cutoff(path, cancel=cancel)
    except Exception:
        cut = None
    if cut:
        lossy = cut["codec"] in ("aac", "mp3", "vorbis", "opus", "ac3", "eac3", "wmav2")
        he = "HE" in cut["profile"].upper()       # HE-AAC SBR legitimately halves the band
        if cut["cutoff_hz"] > 0 and cut["sample_rate"] >= 32000 and not he:
            if cut["ratio"] < 0.40 and (cut["bit_rate"] == 0 or
                                        cut["bit_rate"] >= 96000 or not lossy):
                findings.append({"severity": "warn", "text":
                                 "audio band ends at %.1f kHz in a stream that should reach "
                                 "%.1f kHz - the audio passed through a much lower sample "
                                 "rate or was slowed by resampling at some point" % (
                                     cut["cutoff_hz"] / 1000.0,
                                     cut["sample_rate"] / 2000.0)})
                score += 0.3
            elif cut["ratio"] < 0.55 and cut["bit_rate"] >= 128000 and lossy:
                findings.append({"severity": "info", "text":
                                 "audio cutoff %.1f kHz is low for %s at %d kb/s - worth an "
                                 "ear for pitch shift" % (cut["cutoff_hz"] / 1000.0,
                                                          cut["codec"],
                                                          cut["bit_rate"] // 1000)})
                score += 0.1

    score = min(1.0, round(score, 2))
    label = ("tempo-manipulation indicators present" if score >= 0.6 else
             "weak tempo anomalies - corroborate before acting" if score >= 0.3 else
             "no tempo-manipulation indicators")
    return {"score": score, "label": label, "findings": findings,
            "video": vid, "audio": cut, "caveats": list(_TEMPO_CAVEATS)}


# === Verdict layer ============================================================

def _verdict(vid, verdict, confidence, plain, caveat=None):
    return {"id": vid, "verdict": verdict, "confidence": confidence,
            "plain": plain, "caveat": caveat}


def build_verdicts(rep, platform, tempo) -> list:
    """Translate battery + screen output into moderator language. Each entry:
    {id, verdict: clear|note|review, confidence: low|medium|high, plain,
    caveat}. 'review' = look at it with eyes; never an accusation."""
    out = []
    plat = platform or {}
    weak = set(plat.get("weakened") or [])
    reenc = bool(plat.get("reencoded"))

    if reenc:
        out.append(_verdict("source", "note", "high",
            "This file is a %s re-encode, not the original capture. Splice, "
            "encoder, recompression and noise evidence below describes the "
            "platform's encoder." % (plat.get("platform") or "platform"),
            "Request the runner's local recording before acting on weakened "
            "signals."))
    elif plat.get("platform"):
        out.append(_verdict("source", "clear", "medium",
            "Source pipeline: %s. %s" % (plat.get("platform"),
                                         "; ".join(plat.get("evidence") or [])),
            "ffmpeg traces are normal for OBS captures and downloads alike - "
            "not suspicious by themselves."))
    else:
        out.append(_verdict("source", "clear", "medium",
            "No platform re-encode markers - consistent with a direct capture."))

    sp = rep.get("splices") or {}
    cands = sp.get("candidates") or []
    strong = [c for c in cands if c.get("score", 0) >= 3]
    weakd = "splices" in weak
    if strong:
        times = ", ".join("%.1fs" % c["t"] for c in strong[:6])
        if weakd:
            out.append(_verdict("splices", "review", "low",
                "%d corroborated cut point(s) at %s - BUT this is a platform "
                "re-encode, so keyframe and frame-size structure belongs to the "
                "platform. Only the audio-seam part of the signal still speaks "
                "for the capture." % (len(strong), times),
                "Step through these timestamps frame-by-frame and request the "
                "local file."))
        else:
            out.append(_verdict("splices", "review",
                "high" if len(strong) > 1 else "medium",
                "%d corroborated cut point(s) at %s. Two recordings may have "
                "been joined there - step through frame-by-frame (n/p in the "
                "GUI) and watch for position, timer, score or inventory jumps." %
                (len(strong), times),
                "Scene cuts, capture hiccups and stream resumes fire this too; "
                "a cut point is where to look, not a conviction."))
    else:
        nweak = len([c for c in cands if len(c.get("signals") or []) >= 2])
        extra = (" (platform re-encode limits what this can see)" if weakd else
                 ("; %d weak candidate(s) did not corroborate" % nweak if nweak else ""))
        out.append(_verdict("splices", "clear", "low" if weakd else "medium",
                            "No corroborated cut points%s." % extra))

    lp = rep.get("loops") or {}
    loops = lp.get("loops") or []
    periodic = lp.get("periodic") or []
    if loops:
        L = loops[0]
        out.append(_verdict("loops", "review", "high",
            "Footage repeats: %.1fs-%.1fs replays %.1fs-%.1fs (%.1fs long, %d "
            "repeated sequence(s) in total). The same seconds are shown twice - "
            "the classic cover for a removed segment." % (
                L["repeat_t"], L["repeat_t"] + L["duration_s"], L["src_t"],
                L["src_t"] + L["duration_s"], L["duration_s"], len(loops)),
            "Compare both occurrences side by side; identical noise and HUD "
            "ticks mean identical frames, not similar play."))
    elif periodic:
        p = periodic[0]
        out.append(_verdict("loops", "note", "low",
            "The timeline repeats every %.1fs over %.0f%% of the file. Idle "
            "screens, menus and attract loops do this legitimately." % (
                p["period_s"], p["coverage_pct"]),
            "Only suspicious if that span is claimed as live progress."))
    else:
        out.append(_verdict("loops", "clear", "high", "No repeated footage found."))

    tm = tempo or {}
    ts = float(tm.get("score") or 0.0)
    tplain = "; ".join(f["text"] for f in (tm.get("findings") or [])) or \
        "Frame timing and the audio band look consistent with the container's frame rate."
    out.append(_verdict(
        "tempo",
        "review" if ts >= 0.6 else ("note" if ts >= 0.3 else "clear"),
        "medium" if ts >= 0.6 else "low",
        tplain,
        (tm.get("caveats") or [None])[0] if ts >= 0.3 else None))

    fp = rep.get("fingerprint") or {}
    fwarn = [f["text"] for f in (fp.get("findings") or []) if f.get("severity") == "warn"]
    if fwarn:
        out.append(_verdict("fingerprint", "review",
            "low" if "fingerprint" in weak else "medium",
            "Encoder bookkeeping does not add up: %s. Someone edited metadata "
            "or passed the file off as something it is not." % fwarn[0]))

    md = rep.get("metadata") or {}
    mwarn = [f["text"] for f in (md.get("findings") or []) if f.get("severity") == "warn"]
    if mwarn:
        out.append(_verdict("metadata", "note", "medium",
            "Metadata inconsistency: %s." % mwarn[0],
            "Editors and remuxers cause this innocently; it marks processing, "
            "not guilt."))

    co = rep.get("container") or {}
    cwarn = [f["text"] for f in (co.get("findings") or []) if f.get("severity") == "warn"]
    if cwarn:
        out.append(_verdict("container", "note", "medium",
            "The container shows post-capture processing: %s." % cwarn[0]))

    nz = rep.get("noise") or {}
    outliers = nz.get("outliers") or []
    if outliers and "noise_floor" not in weak:
        med = float(nz.get("median_sigma") or 0.0)
        out.append(_verdict("noise", "note", "medium" if med >= 0.3 else "low",
            "Sensor-noise texture changes at %s - content from another source "
            "may be spliced in." % ", ".join("%.1fs" % o["t"] for o in outliers[:5]),
            "Rendered game frames carry almost no sensor noise, so this check "
            "is weak on direct captures; it bites on camera or CRT footage."))

    enf = rep.get("enf") or {}
    if enf.get("present"):
        jumps = enf.get("jumps") or []
        if jumps:
            out.append(_verdict("enf", "note", "medium",
                "Mains hum (%s Hz) breaks at %s - corroborates an audio edit "
                "there." % (enf.get("base_hz"),
                            ", ".join("%.1fs" % t for t in jumps[:5]))))
        else:
            out.append(_verdict("enf", "clear", "medium",
                "Continuous %s Hz mains hum across the whole file - supports an "
                "unbroken recording." % enf.get("base_hz")))

    c2 = rep.get("c2pa") or {}
    if c2.get("present"):
        if c2.get("validation") == "INVALID":
            out.append(_verdict("provenance", "review", "high",
                "Content Credentials are INVALID - the file was modified after "
                "it was signed."))
        else:
            out.append(_verdict("provenance", "clear", "high",
                "Content Credentials present (%s)." %
                (c2.get("validation") or "unvalidated")))

    rc = rep.get("recompression") or {}
    dq = rc.get("dct_double_quant")
    if isinstance(dq, (int, float)) and dq > 0.35 and not reenc:
        out.append(_verdict("recompress", "note", "low",
            "Re-encoded at least once (double-quantization %.2f). Normal for "
            "uploads and downloads; only meaningful if the file is claimed to "
            "be the untouched capture." % dq))
    return out


def verify_run(path, deep=True, forensics=None, cadence=None, cancel=None,
               on_progress=None) -> dict:
    """Forensics battery + platform screen + tempo screen + verdict layer.
    Pass forensics= a precomputed forensics_report() (and cadence= a
    va_temporal.cadence()) to skip re-running those passes. Returns
    {file, overall: CLEAR|NOTE|REVIEW, summary, platform, tempo, verdicts,
    forensics}."""
    def prog(msg):
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:
                pass
    rep = forensics if forensics is not None else va_forensics.forensics_report(
        path, deep=deep, cancel=cancel, on_progress=on_progress)
    prog("platform screen")
    platform = platform_screen(path, fingerprint=rep.get("fingerprint"))
    prog("tempo screen")
    tempo = tempo_check(path, cadence=cadence, cancel=cancel)
    verdicts = build_verdicts(rep, platform, tempo)
    n_rev = sum(1 for v in verdicts if v["verdict"] == "review")
    n_note = sum(1 for v in verdicts if v["verdict"] == "note")
    overall = "REVIEW" if n_rev else ("NOTE" if n_note else "CLEAR")
    summary = {"REVIEW": "%d area(s) need eyes-on review" % n_rev,
               "NOTE": "%d observation(s), nothing corroborated" % n_note,
               "CLEAR": "no manipulation indicators surfaced"}[overall]
    return {"file": os.path.basename(path), "overall": overall, "summary": summary,
            "platform": platform, "tempo": tempo, "verdicts": verdicts,
            "forensics": rep}


def verify_marks(v) -> list:
    """[(t_seconds, label)] timeline marks for the GUI issue navigator."""
    return va_forensics.report_marks((v or {}).get("forensics") or {})


def _wrap(text, indent="    ") -> str:
    return "\n".join(textwrap.wrap(str(text), width=86, initial_indent=indent,
                                   subsequent_indent=indent)) or (indent + str(text))


def render_verify(v) -> str:
    """Moderator-facing text report: verdicts first, full battery appended."""
    v = v or {}
    L = ["RUN VERIFICATION - %s" % v.get("file", "?"), "=" * 64, "",
         "OVERALL: %s - %s" % (v.get("overall", "?"), v.get("summary", ""))]
    plat = v.get("platform") or {}
    if plat.get("advice"):
        L += ["", _wrap("SOURCE ADVICE: " + plat["advice"], "")]
    L.append("")
    order = {"review": 0, "note": 1, "clear": 2}
    for vd in sorted(v.get("verdicts") or [], key=lambda x: order.get(x["verdict"], 3)):
        L.append("[%-6s] %-12s (confidence %s)" % (vd["verdict"].upper(),
                                                   vd["id"], vd["confidence"]))
        L.append(_wrap(vd["plain"]))
        if vd.get("caveat"):
            L.append(_wrap("! " + vd["caveat"], "    "))
        L.append("")
    L += ["Indicators, not proof: a clean screen does not certify a run, and a",
          "flagged one means 'watch these spots', not 'ban'.",
          "", "-" * 64, "FULL BATTERY OUTPUT", ""]
    L.append(va_forensics.render_report(v.get("forensics") or {}))
    return "\n".join(L)


# --- QC checks for the speedrun profile (registered by plugin.py) -----------
# ctx["verify"] = {"platform": platform_screen(), "tempo": tempo_check()}

def chk_platform_source():
    def f(ctx):
        v = ctx.get("verify") or {}
        if not v:
            return "pass", "not measured (run the speedrun/verify pass)"
        p = v.get("platform") or {}
        if p.get("reencoded"):
            return "warn", "platform re-encode (%s) - splice/encoder/noise evidence is " \
                           "weakened; request the local recording" % (p.get("platform") or "?")
        if p.get("platform"):
            return "pass", "%s traces (normal for OBS captures and downloads alike)" % p["platform"]
        return "pass", "no platform re-encode markers"
    return ("platform", "Source pipeline", f)


def chk_tempo(warn_at=0.45):
    def f(ctx):
        t = (ctx.get("verify") or {}).get("tempo") or {}
        if not t:
            return "pass", "not measured (run the speedrun/verify pass)"
        s = float(t.get("score") or 0.0)
        msg = "; ".join(x["text"] for x in (t.get("findings") or [])[:2]) or t.get("label", "")
        if s >= warn_at:
            return "warn", msg
        if s >= 0.2:
            return "pass", "weak anomalies (score %.2f): %s" % (s, msg)
        return "pass", t.get("label", "ok")
    return ("tempo", "Tempo / speed integrity", f)
