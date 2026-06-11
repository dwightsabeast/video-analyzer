#!/usr/bin/env python3
"""
va_qc - configurable quality-control profiles.

Evaluates an analysis context (probe + metric summary + loudness + events +
gamut + silence) against a named profile and returns pass / warn / fail checks
plus an overall verdict. Pure data; no ffmpeg, no Tk.

Context schema (all optional):
    {"probe": {...}, "summary": {METRIC: {min,mean,max,p1,p99}},
     "loudness": {integrated_lufs, lra_lu, true_peak_dbfs}, "silence": [(s,e)],
     "events": {"black": [(s,e)], "freeze": [(s,e)], "scene_cuts": [t]},
     "gamut": {coverage_2020_pct, outside_709_pct},
     "forensics": va_forensics.forensics_report(),
     "verify": {"platform": ..., "tempo": ...}}  (a run-verification plugin
supplies the verify screens; see plugins/speedrun)
"""

from __future__ import annotations


def _loud(ctx):
    return ctx.get("loudness") or {}


def chk_loudness(lo, hi):
    def f(ctx):
        i = _loud(ctx).get("integrated_lufs")
        silent = _loud(ctx).get("true_peak_dbfs") == float("-inf")
        if i is None:
            return ("warn", "digital silence (no programme loudness)") if silent \
                else ("warn", "no loudness measured (no audio?)")
        if silent and i <= -69:   # ebur128 pins silence at the -70 gate floor
            return "fail", "digital silence (%.1f LUFS, target %g..%g)" % (i, lo, hi)
        if lo <= i <= hi:
            return "pass", "%.1f LUFS (target %g..%g)" % (i, lo, hi)
        if lo - 2 <= i <= hi + 2:
            return "warn", "%.1f LUFS (just outside %g..%g)" % (i, lo, hi)
        return "fail", "%.1f LUFS (target %g..%g)" % (i, lo, hi)
    return ("loudness", "Integrated loudness", f)


def chk_true_peak(max_tp):
    def f(ctx):
        tp = _loud(ctx).get("true_peak_dbfs")
        if tp is None:
            return "warn", "no true-peak measured"
        if tp == float("-inf"):
            return "pass", "digital silence (-inf dBTP, limit %g)" % max_tp
        if tp <= max_tp:
            return "pass", "%.1f dBTP (limit %g)" % (tp, max_tp)
        if tp <= max_tp + 1:
            return "warn", "%.1f dBTP (limit %g)" % (tp, max_tp)
        return "fail", "%.1f dBTP exceeds %g" % (tp, max_tp)
    return ("true_peak", "True peak", f)


def chk_no_black(severity="fail"):
    def f(ctx):
        segs = (ctx.get("events") or {}).get("black") or []
        if not segs:
            return "pass", "none"
        return severity, "%d black segment(s): %s" % (
            len(segs), ", ".join("%.1f-%.1fs" % (s, e) for s, e in segs[:4]))
    return ("black", "Black frames", f)


def chk_no_freeze(severity="warn"):
    def f(ctx):
        segs = (ctx.get("events") or {}).get("freeze") or []
        if not segs:
            return "pass", "none"
        return severity, "%d freeze segment(s)" % len(segs)
    return ("freeze", "Frozen frames", f)


def chk_broadcast_range(max_frac=0.02):
    def f(ctx):
        brng = (ctx.get("summary") or {}).get("BRNG")
        if not brng:
            return "warn", "not measured"
        v = brng.get("p99", brng.get("mean", 0))
        p = ctx.get("probe") or {}
        if p.get("is_pq") or p.get("is_hlg") or p.get("is_hdr"):
            # PQ/HLG legitimately uses the full code range - don't warn on HDR
            return "pass", "p99 %.3f - SDR-oriented check, informational for HDR" % v
        if v <= max_frac:
            return "pass", "p99 %.3f (limit %g)" % (v, max_frac)
        return "warn", "p99 %.3f out-of-range pixels (limit %g)" % (v, max_frac)
    return ("brng", "Broadcast-range levels", f)


def chk_has_audio(severity="warn"):
    def f(ctx):
        return ("pass", "present") if _loud(ctx).get("integrated_lufs") is not None \
            else (severity, "no audio track")
    return ("audio", "Audio present", f)


def chk_silence(max_total=5.0):
    def f(ctx):
        segs = ctx.get("silence") or []
        tot = sum(e - s for s, e in segs)
        if tot <= max_total:
            return "pass", "%.1fs total" % tot
        return "warn", "%.1fs of silence" % tot
    return ("silence", "Excessive silence", f)


def chk_hdr_metadata():
    def f(ctx):
        p = ctx.get("probe") or {}
        if not p.get("is_hdr"):
            return "pass", "SDR"
        ok = bool(p.get("is_pq") or p.get("is_hlg")) and p.get("is_wide_gamut")
        return ("pass", "HDR tagged (%s)" % p.get("transfer")) if ok \
            else ("warn", "HDR but incomplete color tags")
    return ("hdr", "HDR signalling", f)


def chk_banding(max_pct=8.0):
    def f(ctx):
        b = ctx.get("banding")
        if b is None:
            return "pass", "not measured"
        return ("pass", "%.1f%% banding" % b) if b <= max_pct else ("warn", "%.1f%% banding (limit %g)" % (b, max_pct))
    return ("banding", "Banding / contouring", f)


def chk_pse():
    def f(ctx):
        p = ctx.get("pse")
        if not p:
            return "pass", "not measured"
        if p.get("risk"):
            return "fail", "PSE risk: %.1f luma / %.1f red flashes per sec (>3)" % (
                p.get("luminance_flashes_per_sec", 0), p.get("red_flashes_per_sec", 0))
        return "pass", "%.1f flashes/s" % max(p.get("luminance_flashes_per_sec", 0),
                                              p.get("red_flashes_per_sec", 0))
    return ("pse", "Flash safety (PSE)", f)


def chk_cadence():
    def f(ctx):
        c = ctx.get("cadence")
        if not c:
            return "pass", "not measured"
        cad = c.get("cadence", "")
        bad = any(k in cad for k in ("interlac", "telecine", "heavy", "duplicate"))
        return ("warn", cad) if bad else ("pass", cad)
    return ("cadence", "Cadence / interlacing", f)


def chk_maxcll():
    def f(ctx):
        h = ctx.get("hdr_cll")
        if not h:
            return "pass", "n/a (SDR or not measured)"
        notes = h.get("notes") or []
        return ("warn", "; ".join(notes)) if any("exceeds" in n for n in notes) else ("pass", "; ".join(notes) or "ok")
    return ("maxcll", "HDR light-level metadata", f)


def chk_dynhdr():
    def f(ctx):
        m = ctx.get("hdr_meta")
        if not m:
            return "pass", "n/a"
        flags = m.get("flags") or []
        if flags:
            return "warn", "; ".join(fl[1] for fl in flags)
        dv = m.get("dolby_vision")
        if dv:
            return "pass", "Dolby Vision %s" % dv.get("profile_desc", "")
        return "pass", "HDR10+ present" if m.get("hdr10plus") else "ok"
    return ("dynhdr", "Dolby Vision / HDR10+", f)


# --- Forensics / integrity checks (ctx["forensics"] = va_forensics report) ----

def _forn(ctx):
    return ctx.get("forensics") or {}


def chk_provenance():
    def f(ctx):
        c2 = _forn(ctx).get("c2pa") or {}
        if not _forn(ctx):
            return "pass", "not measured (run with forensics)"
        if not c2.get("present"):
            return "pass", "no Content Credentials embedded"
        v = c2.get("validation")
        if v == "INVALID":
            return "fail", "C2PA present but INVALID - modified after signing or untrusted"
        if v == "VALID":
            return "pass", "C2PA valid (%s)" % (c2.get("issuer") or "issuer unknown")
        return "warn", "C2PA present but not validated (install c2patool)"
    return ("provenance", "Content Credentials", f)


def chk_container_integrity():
    def f(ctx):
        co = _forn(ctx).get("container") or {}
        if not _forn(ctx):
            return "pass", "not measured (run with forensics)"
        warns = [x["text"] for x in (co.get("findings") or []) if x.get("severity") == "warn"]
        infos = [x for x in (co.get("findings") or []) if x.get("severity") == "info"]
        if warns:
            return "warn", "; ".join(warns[:2])
        return "pass", "%d informational note(s)" % len(infos) if infos else "clean"
    return ("container", "Container integrity", f)


def chk_splices():
    def f(ctx):
        sp = _forn(ctx).get("splices") or {}
        if not _forn(ctx):
            return "pass", "not measured (run with forensics)"
        strong = [c for c in (sp.get("candidates") or []) if c.get("score", 0) >= 3]
        if strong:
            return "warn", "%d corroborated splice candidate(s): %s" % (
                len(strong), ", ".join("%.1fs" % c["t"] for c in strong[:5]))
        weak = [c for c in (sp.get("candidates") or []) if len(c.get("signals") or []) >= 2]
        if weak:
            return "pass", "%d weak (multi-signal) candidate(s), none corroborated" % len(weak)
        return "pass", "none corroborated"
    return ("splices", "Splice indicators", f)


def chk_loops():
    def f(ctx):
        lp = _forn(ctx).get("loops") or {}
        if not _forn(ctx):
            return "pass", "not measured (run with forensics)"
        loops = lp.get("loops") or []
        periodic = lp.get("periodic") or []
        if loops:
            L = loops[0]
            return "warn", "%d repeated sequence(s), longest %.1fs (%.1fs replays %.1fs)" % (
                len(loops), L["duration_s"], L["repeat_t"], L["src_t"])
        if periodic:
            p = periodic[0]
            return "warn", "timeline hash-periodic every %.1fs (%.0f%% coverage)" % (
                p["period_s"], p["coverage_pct"])
        return "pass", "none"
    return ("loops", "Repeated / looped sequences", f)


def chk_noise_floor():
    def f(ctx):
        nz = _forn(ctx).get("noise") or {}
        if not _forn(ctx) or not nz:
            return "pass", "not measured (run with forensics)"
        outs = nz.get("outliers") or []
        if outs:
            return "warn", "noise floor inconsistent at %s" % ", ".join(
                "%.1fs" % o["t"] for o in outs[:5])
        if (nz.get("median_sigma") or 0) < 0.05:
            return "pass", "noise floor near zero - not informative"
        return "pass", "consistent"
    return ("noise_floor", "Sensor-noise consistency", f)


def chk_enf():
    def f(ctx):
        enf = _forn(ctx).get("enf") or {}
        if not _forn(ctx) or not enf:
            return "pass", "not measured (run with forensics)"
        if not enf.get("present"):
            return "pass", "no mains hum (common; proves nothing)"
        jumps = enf.get("jumps") or []
        if jumps:
            return "warn", "mains hum at %s Hz with %d discontinuity(ies): %s" % (
                enf.get("base_hz"), len(jumps), ", ".join("%.1fs" % t for t in jumps[:5]))
        return "pass", "continuous %s Hz hum (supports continuity)" % enf.get("base_hz")
    return ("enf", "Mains-hum (ENF) continuity", f)


def chk_recompress(max_dq=0.35):
    def f(ctx):
        rc = _forn(ctx).get("recompression") or {}
        dq = rc.get("dct_double_quant")
        if dq is None:
            return "pass", "not measured (run with forensics)"
        if ((( ctx.get("verify") or {}).get("platform")) or {}).get("reencoded"):
            return "pass", "double-quantization %.2f - expected for a platform " \
                           "re-encode (not informative here)" % dq
        if dq > max_dq:
            return "warn", "double-quantization structure %.2f (limit %g) - re-encoded" % (dq, max_dq)
        return "pass", "double-quantization %.2f (limit %g)" % (dq, max_dq)
    return ("recompress", "Recompression / generation loss", f)


def chk_audio_integrity():
    def f(ctx):
        au = (ctx.get("forensics") or {}).get("audio") or {}
        if not au.get("present"):
            return "pass", "no audio track (nothing to check)"
        merged = [x for x in (ctx.get("forensics") or {}).get("findings", [])
                  if x.get("area") == "audio"]
        src = merged if merged else au.get("findings", [])
        warns = [x for x in src if x.get("severity") == "warn"]
        if warns:
            return "warn", "%d audio integrity concern(s): %s" % (
                len(warns), "; ".join(w["text"][:70] for w in warns[:3]))
        notes = [x for x in src if x.get("severity") == "info"]
        if notes:
            return "pass", "only informational audio notes (%d)" % len(notes)
        return "pass", "clipping/dropout/splice/channel scans clean"
    return ("audio_integrity", "Audio integrity (forensic battery)", f)


PROFILES = {
    "broadcast-r128": [chk_loudness(-24, -22), chk_true_peak(-1.0), chk_no_black("fail"),
                       chk_no_freeze("fail"), chk_broadcast_range(0.02), chk_has_audio("fail"),
                       chk_hdr_metadata(), chk_pse(), chk_cadence(), chk_maxcll(), chk_dynhdr()],
    "web-streaming": [chk_loudness(-15, -13), chk_true_peak(-1.0), chk_no_black("warn"),
                      chk_no_freeze("warn"), chk_has_audio("warn"), chk_hdr_metadata(),
                      chk_banding(), chk_pse(), chk_maxcll(), chk_dynhdr()],
    "general": [chk_no_black("warn"), chk_no_freeze("warn"), chk_broadcast_range(0.05),
                chk_silence(10.0), chk_has_audio("warn"), chk_hdr_metadata(),
                chk_banding(), chk_pse(), chk_cadence(), chk_dynhdr()],
    "integrity": [chk_provenance(), chk_container_integrity(), chk_splices(),
                  chk_audio_integrity(),
                  chk_loops(), chk_noise_floor(), chk_enf(), chk_recompress(),
                  chk_no_black("warn"), chk_no_freeze("warn")],
    # domain profiles (e.g. "speedrun") are contributed by plugins via
    # register_profile() - see va_plugins / plugins/<name>/plugin.py
}



def profile_names() -> list:
    return list(PROFILES.keys())


def register_profile(name: str, checks: list):
    """Plugins contribute QC profiles (e.g. the speedrun plugin's
    run-verification profile). Last registration of a name wins."""
    PROFILES[str(name)] = list(checks)


_EXTENSIONS: dict = {}     # profile -> {owner -> [checks]}


def extend_profile(profile: str, checks: list, owner: str = "plugin"):
    """Append checks to an EXISTING profile without owning it. Keyed by
    owner: re-registration replaces (reload-safe), retract_extensions(owner)
    removes everything an owner added."""
    _EXTENSIONS.setdefault(str(profile), {})[str(owner)] = list(checks)


def retract_extensions(owner: str):
    for ext in _EXTENSIONS.values():
        ext.pop(str(owner), None)


def evaluate(ctx, profile="general") -> dict:
    checks = list(PROFILES.get(profile, PROFILES["general"]))
    for owned in _EXTENSIONS.get(profile, {}).values():
        checks = checks + list(owned)
    results = []
    verdict = "pass"
    for cid, label, fn in checks:
        try:
            status, detail = fn(ctx)
        except Exception as exc:  # a broken check must not abort the report
            status, detail = "warn", "check error: %s" % exc
        results.append({"id": cid, "label": label, "status": status, "detail": detail})
        if status == "fail":
            verdict = "fail"
        elif status == "warn" and verdict != "fail":
            verdict = "warn"
    return {"profile": profile, "verdict": verdict, "checks": results}
