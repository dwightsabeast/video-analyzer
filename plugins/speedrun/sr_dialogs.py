"""Tk dialogs for the speedrun plugin: frame-accurate retimer, load-remover
scan, RNG luck calculator. Pure UI - the maths lives in sr_loads / sr_luck.
Everything reaches the app through the plugin api (va_plugins.AppApi); shared
plugin state (captured load refs, last scan, retime marks) lives in the
STATE dict owned by plugin.py so it survives reopening the dialogs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, scrolledtext

import va_theme
import sr_loads
import sr_luck


def clr(key):
    return va_theme.C[key]


def hms(t: float) -> str:
    neg = t < 0
    t = abs(t)
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    return "%s%d:%02d:%02d.%03d" % ("-" if neg else "",
                                    s // 3600, s % 3600 // 60, s % 60, ms)


_OPEN = {}


def _show(key, maker):
    dlg = _OPEN.get(key)
    if dlg is not None and dlg.win.winfo_exists():
        dlg.win.lift()
        return
    _OPEN[key] = maker()


def open_retimer(api, state):
    if api.path:
        _show("retime", lambda: RetimeDialog(api, state))


def open_load_scan(api, state):
    if api.path:
        _show("loads", lambda: LoadScanDialog(api, state))


def open_luck(api):
    _show("luck", lambda: LuckDialog(api))


def _win(api, title, geom):
    w = tk.Toplevel(api.root)
    w.title(title)
    w.geometry(geom)
    w.configure(bg=clr("bg"))
    return w


def _skin(text_widget):
    text_widget.configure(bg=clr("editor_bg"), fg=clr("editor_fg"),
                          insertbackground=clr("editor_fg"), relief="flat")


class RetimeDialog:
    """Frame-accurate retimer: mark start/end on the playhead, get RTA (and
    LRT when a load scan exists) plus a paste-ready mod note."""

    def __init__(self, api, state):
        self.api = api
        self.state = state
        self.win = _win(api, "Retime run", "640x360")
        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        self.lbl_cur = ttk.Label(top, text="playhead: -")
        self.lbl_cur.pack(side=tk.LEFT)
        ttk.Label(top, text="video fps:").pack(side=tk.LEFT, padx=(18, 4))
        self.v_fps = tk.StringVar(value="%.6g" % (api.fps or 30.0))
        ttk.Entry(top, textvariable=self.v_fps, width=9).pack(side=tk.LEFT)
        ttk.Label(top, text="game fps (opt):").pack(side=tk.LEFT, padx=(12, 4))
        self.g_fps = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.g_fps, width=7).pack(side=tk.LEFT)

        row = ttk.Frame(self.win)
        row.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(row, text="Mark start = playhead",
                   command=lambda: self._mark("start")).pack(side=tk.LEFT)
        ttk.Button(row, text="Mark end = playhead",
                   command=lambda: self._mark("end")).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Go to start",
                   command=lambda: self._goto("start")).pack(side=tk.LEFT,
                                                             padx=(18, 0))
        ttk.Button(row, text="Go to end",
                   command=lambda: self._goto("end")).pack(side=tk.LEFT, padx=6)

        self.lbl_calc = tk.Label(self.win, anchor="w", justify=tk.LEFT,
                                 bg=clr("bg"), fg=clr("text"),
                                 font=api.mono_font)
        self.lbl_calc.pack(fill=tk.X, padx=10, pady=(8, 4))

        ttk.Label(self.win, text="Mod note (paste into the submission):").pack(
            anchor="w", padx=10, pady=(6, 0))
        self.note = tk.Text(self.win, height=5, wrap=tk.WORD,
                            font=api.mono_font)
        _skin(self.note)
        self.note.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 4))
        bot = ttk.Frame(self.win)
        bot.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(bot, text="Copy mod note", command=self._copy).pack(side=tk.LEFT)
        self.lbl_hint = ttk.Label(bot, text="timing convention: duration = "
                                            "(end - start) / fps",
                                  foreground=clr("muted"))
        self.lbl_hint.pack(side=tk.RIGHT)
        self._job = None
        self.win.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    def _mark(self, which):
        self.state["retime"][which] = int(self.api.cur_index)

    def _goto(self, which):
        idx = self.state["retime"].get(which)
        if idx is not None:
            self.api.goto_frame(idx)

    def _fps(self, var, fallback):
        try:
            v = float(var.get())
            return v if v > 0 else fallback
        except (TypeError, ValueError):
            return fallback

    def _close(self):
        if self._job is not None:
            try:
                self.win.after_cancel(self._job)
            except Exception:  # noqa: BLE001
                pass
        self.win.destroy()

    def _tick(self):
        if not self.win.winfo_exists():
            return
        api = self.api
        self.lbl_cur.config(text="playhead: frame %d" % api.cur_index)
        vfps = self._fps(self.v_fps, api.fps or 30.0)
        s = self.state["retime"].get("start")
        e = self.state["retime"].get("end")
        lines = ["start: %s    end: %s"
                 % ("-" if s is None else s, "-" if e is None else e)]
        note = ""
        if s is not None and e is not None and e > s:
            n = e - s
            rta = n / vfps
            lines.append("RTA  %s   (%d frames @ %.6g fps)" % (hms(rta), n, vfps))
            gfps = self._fps(self.g_fps, 0.0)
            if gfps:
                lines.append("     = %d game frames @ %.6g fps"
                             % (round(rta * gfps), gfps))
            note = ("Mod note: retimed at %.6g fps - start frame %d, end frame "
                    "%d (%d frames) -> RTA %s." % (vfps, s, e, n, hms(rta)))
            res = self.state.get("loads_res")
            if res and res.get("ok"):
                lf = sr_loads.frames_in_range(res, s, e - 1)
                lrt = max(0, n - lf) / vfps
                lines.append("LRT  %s   (%d load frames removed)" % (hms(lrt), lf))
                note += (" Loads detected by scan: %d frames -> LRT %s."
                         % (lf, hms(lrt)))
            note += " (Video Analyzer retimer)"
        elif s is not None and e is not None:
            lines.append("end must be after start")
        self.lbl_calc.config(text="\n".join(lines))
        cur = self.note.get("1.0", "end-1c")
        if note != cur:
            self.note.delete("1.0", tk.END)
            self.note.insert("1.0", note)
        self._job = self.win.after(180, self._tick)

    def _copy(self):
        txt = self.note.get("1.0", "end-1c").strip()
        if txt:
            self.win.clipboard_clear()
            self.win.clipboard_append(txt)
            self.lbl_hint.config(text="copied")


class LoadScanDialog:
    """Reference capture + whole-file load-screen scan (feeds the retimer's
    LRT line, the timeline marks and the exports)."""

    def __init__(self, api, state):
        self.api = api
        self.state = state
        self.win = _win(api, "Load remover / LRT", "760x480")
        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Button(top, text="Capture playhead frame as load reference",
                   command=self._capture).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear references",
                   command=self._clear).pack(side=tk.LEFT, padx=6)
        self.lbl_refs = ttk.Label(top, text="")
        self.lbl_refs.pack(side=tk.LEFT, padx=10)
        ttk.Label(top, text="preset:").pack(side=tk.LEFT, padx=(18, 4))
        self.preset = tk.StringVar(value="normal")
        cb = ttk.Combobox(top, textvariable=self.preset, state="readonly",
                          width=8, values=["strict", "normal", "loose"])
        cb.pack(side=tk.LEFT)
        self.btn_scan = ttk.Button(top, text="Scan file", command=self._scan)
        self.btn_scan.pack(side=tk.LEFT, padx=12)

        self.out = scrolledtext.ScrolledText(self.win, wrap=tk.NONE,
                                             font=api.mono_font)
        _skin(self.out)
        self.out.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 6))
        self.out.insert("1.0",
                        "Park the playhead on a load screen and capture it as a\n"
                        "reference (repeat for each distinct load screen), then Scan.\n"
                        "Without references the scan still finds black and frozen\n"
                        "stretches. Results mark the timeline (n/p to jump) and feed\n"
                        "the retimer's LRT line and the JSON/HTML exports.")
        ttk.Label(self.win, foreground=clr("muted"),
                  text="pause menus and held cutscene frames also read as "
                       "'static' - review marks before quoting LRT").pack(
            anchor="w", padx=10, pady=(0, 8))
        self._refresh()

    def _refresh(self):
        n = len(self.state["load_refs"])
        self.lbl_refs.config(text="%d reference%s" % (n, "" if n == 1 else "s"))

    def _capture(self):
        _idx, bgr = self.api.current_frame()
        if bgr is None:
            self.lbl_refs.config(text="no frame - open a file first")
            return
        self.state["load_refs"].append(sr_loads.make_ref(bgr))
        self._refresh()

    def _clear(self):
        self.state["load_refs"] = []
        self._refresh()

    def _scan(self):
        api = self.api
        if not api.path:
            return
        self.btn_scan.config(state=tk.DISABLED)
        api.status("Load scan: decoding whole file (reduced resolution)...",
                   "warn")
        path, refs = api.path, list(self.state["load_refs"])
        preset = self.preset.get()

        def worker():
            res = sr_loads.scan(
                path, refs=refs, preset=preset,
                on_progress=lambda m: api.post(
                    lambda m=m: api.status("Load scan: %s..." % m, "warn")),
                cancel=api.closing)
            api.post(self._done, path, res)

        api.run_bg(worker)

    def _done(self, path, res):
        api = self.api
        if self.win.winfo_exists():
            self.btn_scan.config(state=tk.NORMAL)
            self.out.delete("1.0", tk.END)
            self.out.insert("1.0", sr_loads.render_report(res))
        if path != api.path or api.closing():
            return
        if not res.get("ok"):
            api.status("Load scan: %s" % res.get("error", "failed"), "err")
            return
        self.state["loads_res"] = res
        api.add_marks("load", sr_loads.marks(res))
        api.adv_store("loads", "Load screens - %d segments, %.2fs"
                      % (len(res["segments"]), res["load_s"]),
                      text=sr_loads.render_report(res),
                      data={k: res[k] for k in ("fps", "frames_scanned",
                                                "segments", "load_frames",
                                                "load_s", "refs_used",
                                                "preset")})
        api.status("Load scan: %d segment(s), %.2fs of loads - timeline "
                   "marked, retimer now quotes LRT"
                   % (len(res["segments"]), res["load_s"]), "ok")


class LuckDialog:
    """Dream-report style binomial odds for visible RNG events."""

    COLS = (("event label", 18), ("trials", 7), ("successes", 9),
            ("p(success)", 10), ("pick-of", 7))

    def __init__(self, api):
        self.api = api
        self.win = _win(api, "Luck calculator (RNG odds)", "780x520")
        ttk.Label(self.win, text="One row per independent event class seen in "
                                 "the video. p comes from verified game data; "
                                 "'pick-of' = how many comparable runs/runners "
                                 "this one was effectively selected from.",
                  wraplength=740).pack(anchor="w", padx=10, pady=(10, 4))
        self.grid_fr = ttk.Frame(self.win)
        self.grid_fr.pack(fill=tk.X, padx=10)
        for c, (name, _w) in enumerate(self.COLS):
            ttk.Label(self.grid_fr, text=name,
                      foreground=clr("muted")).grid(row=0, column=c,
                                                    padx=3, sticky="w")
        self.rows = []
        for _ in range(4):
            self._add_row()
        row = ttk.Frame(self.win)
        row.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(row, text="Add row", command=self._add_row).pack(side=tk.LEFT)
        ttk.Button(row, text="Compute odds",
                   command=self._compute).pack(side=tk.LEFT, padx=8)
        self.lbl_err = ttk.Label(row, text="", foreground=clr("err"))
        self.lbl_err.pack(side=tk.LEFT, padx=10)
        self.out = scrolledtext.ScrolledText(self.win, wrap=tk.NONE,
                                             font=api.mono_font)
        _skin(self.out)
        self.out.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))

    def _add_row(self):
        r = len(self.rows) + 1
        ents = []
        for c, (_n, width) in enumerate(self.COLS):
            e = ttk.Entry(self.grid_fr, width=width)
            e.grid(row=r, column=c, padx=3, pady=2, sticky="w")
            ents.append(e)
        self.rows.append(ents)

    def _compute(self):
        events = []
        self.lbl_err.config(text="")
        for ents in self.rows:
            vals = [e.get().strip() for e in ents]
            if not any(vals[1:4]):
                continue
            try:
                events.append({"label": vals[0] or "event %d" % (len(events) + 1),
                               "n": int(vals[1]), "k": int(vals[2]),
                               "p": float(vals[3]),
                               "select": int(vals[4]) if vals[4] else 1})
            except ValueError:
                self.lbl_err.config(text="bad row: '%s' - need integers for "
                                         "trials/successes, float for p"
                                         % (vals[0] or "?"))
                return
        if not events:
            self.lbl_err.config(text="enter at least one event row")
            return
        res = sr_luck.evaluate(events)
        txt = sr_luck.render_report(res)
        self.out.delete("1.0", tk.END)
        self.out.insert("1.0", txt)
        self.api.adv_store("luck", "RNG luck analysis - %s"
                           % res["verdict"].upper(), text=txt, data=res)
