"""Region-selection + calibration dialog for the timer OCR plugin. Pure UI;
the engine lives in ocr_core. The dialog is non-modal: step the main window
to a frame where the timer is visible, box it, type what it reads, calibrate
- repeat on another frame until all ten digits are trained."""

from __future__ import annotations

import os
import tempfile
import uuid

import numpy as np
import tkinter as tk
from tkinter import ttk

import va_theme
import ocr_core


def clr(key):
    return va_theme.C[key]


def _ppm_bytes(gray) -> bytes:
    g = np.clip(np.asarray(gray), 0, 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=2)
    return b"P6\n%d %d\n255\n" % (g.shape[1], g.shape[0]) + rgb.tobytes()


class RegionDialog:
    MAXW, MAXH = 880, 440

    def __init__(self, api, state):
        self.api = api
        self.state = state
        self.win = tk.Toplevel(api.root)
        self.win.title("Timer OCR - region & calibration")
        self.win.configure(bg=clr("bg"))
        self.scale = 1.0
        self.frame = None
        self._photo = None
        self._rect_id = None
        self._drag = None
        self._tmp = os.path.join(tempfile.gettempdir(),
                                 "va_ocr_%s.ppm" % uuid.uuid4().hex[:8])

        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Button(top, text="Grab current frame",
                   command=self._grab).pack(side=tk.LEFT)
        self.lbl_info = ttk.Label(top, text="step the main window to a frame "
                                            "where the timer shows, then grab")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.win, width=self.MAXW, height=self.MAXH,
                                bg=va_theme.WELL["bg"], highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(padx=10, pady=4)
        self.canvas.bind("<Button-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._up)

        cal = ttk.Frame(self.win)
        cal.pack(fill=tk.X, padx=10, pady=4)
        self.lbl_glyphs = ttk.Label(cal, text="glyphs in box: -")
        self.lbl_glyphs.pack(side=tk.LEFT)
        ttk.Label(cal, text="timer reads:").pack(side=tk.LEFT, padx=(16, 4))
        self.ent = ttk.Entry(cal, width=16)
        self.ent.pack(side=tk.LEFT)
        ttk.Button(cal, text="Calibrate from this frame",
                   command=self._calibrate).pack(side=tk.LEFT, padx=8)
        ttk.Button(cal, text="Read box now",
                   command=self._read_now).pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(self.win, text=self._trained(),
                                    foreground=clr("muted"))
        self.lbl_status.pack(anchor="w", padx=10, pady=(2, 10))
        self._grab()

    # --- frame handling ---------------------------------------------------
    def _grab(self):
        idx, bgr = self.api.current_frame()
        if bgr is None:
            self.lbl_info.config(text="no frame - open a video first")
            return
        self.frame = np.asarray(bgr).mean(axis=2).astype(np.float32)
        h, w = self.frame.shape
        self.scale = min(1.0, self.MAXW / w, self.MAXH / h)
        try:
            import cv2
            disp = cv2.resize(self.frame, (int(w * self.scale),
                                           int(h * self.scale)))
        except Exception:  # noqa: BLE001
            disp = self.frame
            self.scale = 1.0
        with open(self._tmp, "wb") as fh:
            fh.write(_ppm_bytes(disp))
        self._photo = tk.PhotoImage(file=self._tmp)
        self.canvas.delete("all")
        self.canvas.config(width=self._photo.width(),
                           height=self._photo.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.lbl_info.config(text="frame %d - drag a box around the timer" % idx)
        self._rect_id = None
        if self.state.get("rect"):
            x0, y0, x1, y1 = [v * self.scale for v in self.state["rect"]]
            self._rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=clr("accent"), width=2)
            self._update_glyphs()

    # --- rectangle drag ----------------------------------------------------
    def _down(self, ev):
        self._drag = (ev.x, ev.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            ev.x, ev.y, ev.x, ev.y, outline=clr("accent"), width=2)

    def _move(self, ev):
        if self._drag and self._rect_id:
            self.canvas.coords(self._rect_id, self._drag[0], self._drag[1],
                               ev.x, ev.y)

    def _up(self, ev):
        if not self._drag:
            return
        x0, y0 = self._drag
        self._drag = None
        x1, y1 = ev.x, ev.y
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return
        s = self.scale or 1.0
        rect = (int(min(x0, x1) / s), int(min(y0, y1) / s),
                int(max(x0, x1) / s), int(max(y0, y1) / s))
        self.state["rect"] = rect
        self._update_glyphs()

    def _region(self):
        r = self.state.get("rect")
        if r is None or self.frame is None:
            return None
        x0, y0, x1, y1 = r
        return self.frame[y0:y1, x0:x1]

    def _update_glyphs(self):
        reg = self._region()
        if reg is None or reg.size == 0:
            return
        n = len(ocr_core.segment_glyphs(reg))
        self.lbl_glyphs.config(text="glyphs in box: %d" % n)

    # --- calibration --------------------------------------------------------
    def _trained(self):
        bank = self.state.get("bank")
        cov = bank.coverage() if bank else ""
        return ("trained glyphs: %r - calibrate on more frames until all ten "
                "digits appear" % cov) if cov else \
            "no calibration yet - box the timer, type its reading, calibrate"

    def _calibrate(self):
        reg = self._region()
        txt = self.ent.get().strip()
        if reg is None or not txt:
            self.lbl_status.config(text="need a box and the timer text",
                                   foreground=clr("err"))
            return
        bank = self.state.get("bank") or ocr_core.GlyphBank()
        try:
            n = bank.calibrate(reg, txt)
        except ValueError as exc:
            self.lbl_status.config(text=str(exc), foreground=clr("err"))
            return
        self.state["bank"] = bank
        self.lbl_status.config(text="mapped %d glyph(s). %s"
                               % (n, self._trained()), foreground=clr("ok"))

    def _read_now(self):
        reg = self._region()
        bank = self.state.get("bank")
        if reg is None or bank is None:
            self.lbl_status.config(text="calibrate first", foreground=clr("err"))
            return
        s, d = bank.read(reg)
        p = ocr_core.parse_timer(s, self.api.fps)
        self.lbl_status.config(
            text="box reads %r (mean dist %.3f) -> %s" %
                 (s, d, "%.3fs [%s]" % (p[1], p[0]) if p else "not a timer"),
            foreground=clr("ok") if p else clr("warn"))
