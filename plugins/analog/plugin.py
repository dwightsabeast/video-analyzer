"""Analog artifact detector plugin.

Adds 'Analog ▾' to the toolbar (whole-file artifact scan with timeline
marks + report) and - headless - the 'analog' QC profile plus the analyze.py
pass that writes <name>.analog.txt. Detection lives in analog_core."""

from __future__ import annotations

import os

import analog_core


def register(api):
    import va_qc
    va_qc.register_profile("analog", analog_core.qc_checks())
    m = api.add_toolbar_menu("Analog ▾")
    api.add_command(m, "Scan for tape/analog artifacts", lambda: _scan(api))
    api.add_command(m, "QC check (analog profile)...",
                    lambda: api.run_qc("analog"))


def _scan(api):
    if not api.path:
        return
    api.status("Analog scan: dropouts / head-switching / jitter / flicker...",
               "warn")
    path = api.path

    def worker():
        res = analog_core.scan(
            path, on_progress=lambda msg: api.post(
                lambda msg=msg: api.status("Analog scan: %s..." % msg, "warn")),
            cancel=api.closing)
        api.post(_done, api, path, res)

    api.run_bg(worker)


def _done(api, path, res):
    if path != api.path or api.closing():
        return
    if not res.get("ok"):
        api.status("Analog scan: %s" % res.get("error", "failed"), "err")
        return
    api.add_marks("analog", analog_core.marks(res))
    ser = res.get("series") or {}
    if ser.get("v"):
        api.add_series("Analog: line jitter (px)", ser["t"], ser["v"], vmin=0.0)
    warns = sum(1 for e in res["events"] if e["severity"] == "warn")
    title = "Analog artifacts - %d event(s), %d warn" % (len(res["events"]),
                                                         warns)
    api.adv_store("analog", title, text=analog_core.render_report(res),
                  data={"stats": res["stats"],
                        "events": res["events"][:400]})
    api.status("Analog scan: %d event(s)%s"
               % (len(res["events"]),
                  " - %d warn (see report popup)" % warns if warns else ""),
               "warn" if warns else "ok")
    api.text_popup("Analog artifacts - %s" % os.path.basename(path),
                   analog_core.render_report(res))


def register_headless(api):
    api.register_qc_profile("analog", analog_core.qc_checks())
    api.register_profile_pass(
        "analog",
        run=lambda path, forensics=None, cadence=None: analog_core.scan(path),
        echo="analog artifact scan (dropouts / head-switch / jitter / flicker)",
        ctx=lambda rep: rep,
        json_extract=lambda rep: {"stats": rep.get("stats"),
                                  "events": (rep.get("events") or [])[:400]},
        render=analog_core.render_report,
        suffix=".analog.txt",
        needs_forensics=False,
        ctx_key="analog")
