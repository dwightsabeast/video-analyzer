"""Captions QC plugin.

Adds 'Captions ▾' to the toolbar: pick a text track (or the .srt sidecar),
QC it against the audio speech map, mark violations on the timeline.
Headless: the 'captions' QC profile and the analyze.py pass that writes
<name>.captions.txt. The engine lives in cap_core."""

from __future__ import annotations

import os

import cap_core


def register(api):
    import va_qc
    va_qc.register_profile("captions", cap_core.qc_checks())
    m = api.add_toolbar_menu("Captions ▾")
    api.add_command(m, "Captions QC report...", lambda: _open_picker(api))
    api.add_command(m, "QC check (captions profile)...",
                    lambda: api.run_qc("captions"))


_DLG = {}


def _open_picker(api):
    if not api.path:
        return
    import tkinter as tk
    from tkinter import ttk
    import va_theme
    dlg = _DLG.get("w")
    if dlg is not None and dlg.winfo_exists():
        dlg.lift()
        return
    win = tk.Toplevel(api.root)
    _DLG["w"] = win
    win.title("Captions QC")
    win.geometry("560x180")
    win.configure(bg=va_theme.C["bg"])
    info = cap_core.probe_subs(api.path)
    side = cap_core.find_sidecar(api.path)
    choices, vals = [], []
    for t in info["text"]:
        choices.append("embedded s:%d  %s %s %s" % (t["sidx"], t["codec"],
                                                    t["lang"], t["title"]))
        vals.append(t["sidx"])
    if side:
        choices.append("sidecar  %s" % os.path.basename(side))
        vals.append(None)
    ttk.Label(win, text="%d text track(s), %d bitmap, CC flag: %s"
              % (len(info["text"]), len(info["bitmap"]),
                 "yes" if info["cc"] else "no")).pack(anchor="w", padx=12,
                                                      pady=(12, 4))
    var = tk.StringVar(value=choices[0] if choices else "")
    cb = ttk.Combobox(win, textvariable=var, state="readonly", width=58,
                      values=choices)
    cb.pack(padx=12, pady=4, anchor="w")
    if not choices:
        ttk.Label(win, text="No text track or .srt sidecar - the report will "
                            "cover presence only.",
                  foreground=va_theme.C["muted"]).pack(anchor="w", padx=12)

    def run():
        sidx = vals[cb.current()] if choices else None
        win.destroy()
        _run_qc_scan(api, sidx)

    ttk.Button(win, text="Run captions QC", command=run).pack(pady=10)


def _run_qc_scan(api, sidx):
    path = api.path
    api.status("Captions QC: extracting cues + speech map...", "warn")

    def worker():
        rep = cap_core.analyze(
            path, sidx=sidx,
            on_progress=lambda m: api.post(
                lambda m=m: api.status("Captions QC: %s..." % m, "warn")))
        api.post(_done, api, path, rep)

    api.run_bg(worker)


def _done(api, path, rep):
    if path != api.path or api.closing():
        return
    if not rep.get("ok"):
        api.status("Captions QC: %s" % rep.get("error", "failed"), "err")
        return
    api.add_marks("cap", cap_core.marks(rep))
    q = rep.get("qc")
    if q:
        warns = sum(1 for _t, s, _x in q["events"] if s == "warn")
        title = ("Captions QC - %d cues, %.0f%% coverage, %d warn"
                 % (q["cues"], q["coverage_pct"], warns))
        ok = warns == 0
    else:
        title = "Captions QC - no text track to QC"
        ok = False
    api.adv_store("captions", title, text=cap_core.render_report(rep),
                  data={k: rep.get(k) for k in ("streams", "source",
                                                "duration", "qc")})
    api.status(title, "ok" if ok else "warn")
    api.text_popup("Captions QC - %s" % os.path.basename(path),
                   cap_core.render_report(rep))


def register_headless(api):
    api.register_qc_profile("captions", cap_core.qc_checks())
    api.register_profile_pass(
        "captions",
        run=lambda path, forensics=None, cadence=None: cap_core.analyze(path),
        echo="captions QC (tracks / coverage / sync / reading speed)",
        ctx=lambda rep: rep,
        json_extract=lambda rep: {k: rep.get(k) for k in (
            "streams", "source", "duration", "qc")},
        render=cap_core.render_report,
        suffix=".captions.txt",
        needs_forensics=False,
        ctx_key="captions")
