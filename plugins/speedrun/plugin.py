"""Speedrun forensics plugin.

Adds the run-verification workbench to the core video-forensics app: the
Speedrun ▾ toolbar menu (Verify run, speedrun QC profile, retimer, load
remover, music continuity scan, luck calculator) and - headless - the
`speedrun` QC profile plus the analyze.py deep pass that writes
<name>.verify.txt.

GUI entry: register(api). Headless entry: register_headless(api)."""

from __future__ import annotations

import os

import va_qc
import va_forensics
import sr_verify
import sr_music

STATE = {"loads_res": None, "load_refs": [],
         "retime": {"start": None, "end": None}}


def _qc_checks():
    return [sr_verify.chk_platform_source(), sr_verify.chk_tempo(),
            va_qc.chk_splices(), va_qc.chk_loops(), va_qc.chk_noise_floor(),
            va_qc.chk_enf(), va_qc.chk_container_integrity(),
            va_qc.chk_provenance(), va_qc.chk_recompress(),
            va_qc.chk_no_freeze("warn")]


# --- GUI ----------------------------------------------------------------------

def register(api):
    import sr_dialogs
    va_qc.register_profile("speedrun", _qc_checks())
    m = api.add_toolbar_menu("Speedrun ▾")
    api.add_command(m, "Verify run (splice / tempo / source)...",
                    lambda: _verify(api))
    api.add_command(m, "QC check (speedrun profile)...",
                    lambda: api.run_qc("speedrun"))
    api.add_separator(m)
    api.add_command(m, "Retime run (frame-accurate)...",
                    lambda: sr_dialogs.open_retimer(api, STATE))
    api.add_command(m, "Load remover / LRT...",
                    lambda: sr_dialogs.open_load_scan(api, STATE))
    api.add_command(m, "Music continuity scan", lambda: _music_scan(api))
    api.add_command(m, "Luck calculator (RNG odds)...",
                    lambda: sr_dialogs.open_luck(api))
    api.on_open(_on_open)


def _on_open(_path):
    STATE["loads_res"] = None
    STATE["retime"] = {"start": None, "end": None}
    # load_refs survive a file change: same game, next run


def _verify(api):
    if not api.path:
        return
    api.status("Verify run: forensics battery + platform + tempo screens...",
               "warn")
    path = api.path

    def worker():
        v = sr_verify.verify_run(
            path, on_progress=lambda m: api.post(
                lambda m=m: api.status("Verify: %s..." % m, "warn")))
        api.post(_verify_done, api, path, v)

    api.run_bg(worker)


def _verify_done(api, path, v):
    if path != api.path or api.closing():
        return
    api.add_marks("verify", sr_verify.verify_marks(v))
    bad = v.get("overall") != "CLEAR"
    api.status("Verify: %s - %s" % (v.get("overall", "?"),
                                    v.get("summary", "")),
               "warn" if bad else "ok")
    vfrep = v.get("forensics") or {}
    if vfrep:
        api.adv_store("forensics", "Forensics / integrity report",
                      text=va_forensics.render_report(vfrep), data=vfrep)
    api.adv_store("verify", "Verify run - %s (%s)"
                  % (v.get("overall", "?"), v.get("summary", "")),
                  text=sr_verify.render_verify(v),
                  data={k: v.get(k) for k in ("overall", "summary",
                                              "platform", "tempo",
                                              "verdicts")})
    api.text_popup("Verify run - %s" % os.path.basename(path),
                   sr_verify.render_verify(v))


def _music_scan(api):
    if not api.path:
        return
    api.status("Music continuity: decoding audio...", "warn")
    path = api.path

    def worker():
        res = sr_music.analyze(
            path, on_progress=lambda m: api.post(
                lambda m=m: api.status("Music continuity: %s..." % m, "warn")))
        api.post(_music_done, api, path, res)

    api.run_bg(worker)


def _music_done(api, path, res):
    if path != api.path or api.closing():
        return
    if not res.get("ok"):
        api.status("Music scan: %s" % res.get("error", "failed"), "err")
        return
    api.add_marks("music", sr_music.marks(res))
    img = sr_music.plot(res)
    warns = sum(1 for f in res["findings"] if f["severity"] == "warn")
    title = ("Music continuity - %d discontinuit%s (%d suspect)"
             % (len(res["findings"]),
                "y" if len(res["findings"]) == 1 else "ies", warns))
    api.adv_store("music", title, text=sr_music.render_report(res),
                  data={"findings": res["findings"],
                        "duration_s": res["duration_s"]}, img=img)
    api.status("Music continuity: %d mark(s)%s"
               % (len(res["findings"]),
                  " - %d suspect splice(s)" % warns if warns else ""),
               "warn" if warns else "ok")
    if img is not None:
        api.image_popup(title + "  (n/p jumps between marks)", img)


# --- Headless (analyze.py, selftests) ------------------------------------------

def register_headless(api):
    api.register_qc_profile("speedrun", _qc_checks())
    api.register_profile_pass(
        "speedrun",
        run=lambda path, forensics=None, cadence=None: sr_verify.verify_run(
            path, forensics=forensics, cadence=cadence),
        echo="verify screens (platform / tempo)",
        ctx=lambda v: {"platform": v["platform"], "tempo": v["tempo"]},
        json_extract=lambda v: {k: v.get(k) for k in (
            "overall", "summary", "platform", "tempo", "verdicts")},
        render=sr_verify.render_verify,
        suffix=".verify.txt",
        needs_forensics=True)
