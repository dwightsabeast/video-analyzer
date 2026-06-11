#!/usr/bin/env python3
"""
Video Analyzer - a one-stop video data-science workbench
========================================================
Plays a video and analyses it as a dataset: HDR/colour metadata, a full set of
live scopes (vectorscope, waveform, RGB parade, false colour, CIE gamut, luma
histogram), a per-frame metrics timeline (signalstats), automatic event
detection (black / freeze / scene cuts), EBU R128 loudness, reference-quality
comparison (PSNR / SSIM / VMAF), and CSV / JSON / HTML export.

Decoding goes through ffmpeg (reliable HEVC / 10-bit / HDR, with HDR->SDR
tonemapping), falling back to OpenCV when ffmpeg is absent. Heavy lifting is
done by the bundled ffmpeg/ffprobe binary, so the only Python dependencies are
opencv-python and numpy.

Engine modules (all headless-testable): va_ffmpeg, va_probe, va_metrics,
va_scopes, va_quality, va_export.

Usage:
    python video-analyzer.py [path/to/video]
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
import tempfile
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk, scrolledtext, messagebox

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAVE_DND = True
except Exception:  # optional - DnD is a nicety, never a hard dependency
    TkinterDnD = None
    DND_FILES = None
    HAVE_DND = False

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "\nMissing dependency: %s\nInstall with:\n"
        "    pip install opencv-python numpy\n\n" % exc)
    raise

from va_ffmpeg import VideoSource, NativeTap, find_ffmpeg, find_ffprobe
from va_probe import (classify_hdr, format_ffprobe, build_summary,
                      extract_with_opencv, format_opencv)
from va_ffmpeg import ffprobe_json
import va_scopes as scopes
import va_metrics as metrics
import va_quality as quality
import va_export as export
import va_audio
import va_compare
import va_forensics
import va_perceptual
import va_temporal
import va_hdr
import va_qc
import va_dynhdr
import va_theme
import va_perf
import va_plugins
import va_plugins_ui


def clr(key: str) -> str:
    """Color from the active theme palette (live - follows theme switches)."""
    return va_theme.C[key]


PANEL_BG = (16, 16, 18)   # matches va_theme.WELL["bg"] - instrument wells stay dark

# (label, key) for the scope tabs and the timeline metric selector
SCOPES = [("Vectorscope", "vectorscope"), ("Waveform", "waveform"),
          ("RGB parade", "parade"), ("False colour", "false"),
          ("CIE gamut", "cie"), ("Histogram", "hist")]
TIMELINE_METRICS = [("Luma average", "YAVG"), ("Luma max", "YMAX"),
                    ("Saturation average", "SATAVG"), ("Saturation max", "SATMAX"),
                    ("Hue average", "HUEAVG"), ("Frame difference", "YDIF"),
                    ("Temporal outliers", "TOUT"), ("Broadcast range", "BRNG")]


def encode_ppm_bytes(rgb: np.ndarray) -> bytes:
    """Serialise an RGB uint8 array as binary PPM (P6) for tk.PhotoImage(file=)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()


# --- pure helpers (unit-testable without Tk) ---------------------------------

def frame_to_x(frame: int, width: int, nb_frames: int) -> float:
    if nb_frames <= 1:
        return 0.0
    return max(0.0, min(1.0, frame / (nb_frames - 1))) * (width - 1)


def x_to_frame(x: float, width: int, nb_frames: int) -> int:
    if width <= 1 or nb_frames <= 1:
        return 0
    return int(round(max(0.0, min(1.0, x / (width - 1))) * (nb_frames - 1)))


def metric_polyline(values, width: int, height: int, pad: int = 4,
                    vmin=None, vmax=None):
    """Map a metric series to canvas (x, y) points. Returns [] if no data."""
    vals = [v for v in values if v is not None and v == v]
    if not vals:
        return []
    lo = min(vals) if vmin is None else vmin
    hi = max(vals) if vmax is None else vmax
    rng = (hi - lo) or 1.0
    n = len(values)
    stride = max(1, n // max(1, width))
    pts = []
    usable_h = max(1, height - 2 * pad)
    for i in range(0, n, stride):
        v = values[i]
        if v is None or v != v:
            continue
        x = frame_to_x(i, width, n)
        y = pad + (1.0 - (v - lo) / rng) * usable_h
        pts.append((x, y))
    return pts


def parse_dnd_paths(data: str) -> list:
    """Parse a tkinterdnd2 <<Drop>> data string into file paths.

    Paths containing spaces arrive brace-wrapped, e.g. "{C:/My Clips/a.mp4} D:/b.mkv"."""
    s = (data or "").strip()
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            j = s.find("}", i)
            if j == -1:
                out.append(s[i + 1:]); break
            out.append(s[i + 1:j]); i = j + 1
        elif ch == " ":
            i += 1
        else:
            j = s.find(" ", i)
            if j == -1:
                out.append(s[i:]); break
            out.append(s[i:j]); i = j
    return [x.strip() for x in out if x.strip()]


def issue_frames(events: dict, fps: float, extra=None) -> list:
    """Flatten detected events into a sorted [(frame, label)] list of issues.
    ``extra`` = [(t_seconds, label)] appends forensic / custom marks."""
    fps = fps or 25.0
    items = {}
    for seg in (events or {}).get("black", []):
        items.setdefault(int(round(seg[0] * fps)), "black")
    for seg in (events or {}).get("freeze", []):
        items.setdefault(int(round(seg[0] * fps)), "freeze")
    for c in (events or {}).get("scene_cuts", []):
        items.setdefault(int(round(c * fps)), "scene cut")
    for t, label in (extra or []):
        items.setdefault(int(round(float(t) * fps)), str(label))
    return [(f, items[f]) for f in sorted(items)]


class AudioStrip(tk.Canvas):
    """Audio waveform timeline: min/max envelope + RMS body, playhead synced
    with the video, click/drag to scrub, mouse wheel to zoom (double-click
    resets), silence shading and forensic-finding ticks."""

    SPAN_MIN = 0.005     # max zoom = 0.5% of the file

    def __init__(self, parent, on_seek, height=86):
        super().__init__(parent, height=height, bg=va_theme.WELL["panel"],
                         highlightthickness=1,
                         highlightbackground=va_theme.WELL["faint"])
        self.on_seek = on_seek
        self.ov = None
        self.dur = 0.0
        self.marks = []          # [(t_seconds, severity)]
        self.silence = []        # [(t0, t1)]
        self.playhead_t = 0.0
        self.z0, self.z1 = 0.0, 1.0
        self._ph = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._click)
        self.bind("<Double-Button-1>", self._reset_zoom)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._wheel(e, 120))
        self.bind("<Button-5>", lambda e: self._wheel(e, -120))

    # -- data ------------------------------------------------------------
    def set_data(self, ov):
        self.ov = ov if (ov and ov.get("buckets")) else None
        self.dur = float((ov or {}).get("duration_s") or 0.0)
        self.z0, self.z1 = 0.0, 1.0
        self.redraw()

    def set_marks(self, marks):
        self.marks = list(marks or [])
        self.redraw()

    def set_silence(self, segs):
        self.silence = list(segs or [])
        self.redraw()

    def set_playhead(self, t):
        self.playhead_t = float(t)
        if self._ph is None:
            return
        x = self._t_to_x(self.playhead_t)
        h = self.winfo_height()
        try:
            if x is None:
                self.coords(self._ph, -10, 0, -10, h)
            else:
                self.coords(self._ph, x, 0, x, h)
        except tk.TclError:
            pass

    # -- geometry ----------------------------------------------------------
    def _t_to_x(self, t):
        if self.dur <= 0:
            return None
        f = t / self.dur
        span = max(self.z1 - self.z0, 1e-9)
        if f < self.z0 - 0.001 or f > self.z1 + 0.001:
            return None
        return int((f - self.z0) / span * max(self.winfo_width(), 1))

    def _x_to_frac(self, x):
        w = max(self.winfo_width(), 1)
        return self.z0 + (max(0, min(x, w)) / w) * (self.z1 - self.z0)

    # -- interaction -------------------------------------------------------
    def _click(self, event):
        if self.dur > 0:
            self.on_seek(max(0.0, min(1.0, self._x_to_frac(event.x))))

    def _reset_zoom(self, _event=None):
        self.z0, self.z1 = 0.0, 1.0
        self.redraw()

    def _wheel(self, event, delta=None):
        if self.ov is None:
            return
        d = delta if delta is not None else event.delta
        factor = 0.78 if d > 0 else 1.28
        span = (self.z1 - self.z0) * factor
        span = max(self.SPAN_MIN, min(1.0, span))
        c = self._x_to_frac(event.x)
        rel = (event.x / max(self.winfo_width(), 1))
        z0 = c - rel * span
        self.z0 = max(0.0, min(z0, 1.0 - span))
        self.z1 = self.z0 + span
        self.redraw()

    # -- drawing -----------------------------------------------------------
    def redraw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        w, h = self.winfo_width(), self.winfo_height()
        self._ph = None
        if w < 20 or h < 20:
            return
        if self.ov is None:
            self.create_text(w // 2, h // 2, text="audio waveform appears here",
                             fill=va_theme.WELL["faint"], font=("TkDefaultFont", 8))
            return
        vmin = self.ov["vmin"]; vmax = self.ov["vmax"]; rms = self.ov["rms"]
        B = len(vmin)
        mid = h * 0.56
        amp = (h - 22) * 0.5
        span = max(self.z1 - self.z0, 1e-9)
        # silence shading first (behind the waveform)
        for t0, t1 in self.silence:
            if self.dur <= 0:
                break
            x0 = (t0 / self.dur - self.z0) / span * w
            x1 = (t1 / self.dur - self.z0) / span * w
            if x1 < 0 or x0 > w:
                continue
            self.create_rectangle(max(0, x0), 12, min(w, x1), h - 2,
                                  fill="#16161a", outline="")
        env = "#2e5a50"
        body = va_theme.WELL["line"]
        for x in range(w):
            b0 = int((self.z0 + (x / w) * span) * B)
            b1 = max(b0 + 1, int((self.z0 + ((x + 1) / w) * span) * B))
            b0 = max(0, min(b0, B - 1)); b1 = max(1, min(b1, B))
            lo = min(vmin[b0:b1]); hi = max(vmax[b0:b1])
            rm = max(rms[b0:b1])
            y0 = mid - hi * amp; y1 = mid - lo * amp
            self.create_line(x, y0, x, y1, fill=env)
            self.create_line(x, mid - rm * amp, x, mid + rm * amp, fill=body)
        # forensic ticks on top
        for t, sev in self.marks:
            x = self._t_to_x(float(t))
            if x is not None:
                col = "#ff9f40" if str(sev).startswith("warn") else "#4fd1c5"
                self.create_line(x, 2, x, 14, fill=col, width=2)
                self.create_line(x, 2, x, h - 2, fill=col, dash=(2, 5))
        if self.z0 > 0.0 or self.z1 < 1.0:
            self.create_text(4, h - 8, anchor="w", fill=va_theme.WELL["faint"],
                             font=("TkDefaultFont", 7),
                             text="zoom %.0f%%-%.0f%% (double-click resets)"
                                  % (self.z0 * 100, self.z1 * 100))
        self._ph = self.create_line(-10, 0, -10, h, fill=va_theme.WELL["play"], width=2)
        self.set_playhead(self.playhead_t)


class VideoAnalyzerApp:
    CHART_UPDATE_MS = 80
    ANALYSIS_MAX = 480

    def __init__(self, root: tk.Tk, initial_path: "str | None" = None):
        self.root = root
        self.root.title("Video Analyzer - data-science workbench")
        self.root.geometry("1600x950")
        self.root.minsize(1180, 700)
        self.theme_mode = va_theme.load_pref()
        va_theme.apply(self.root, self.theme_mode)
        self.perf_mode = va_theme.load_ui_key("perf", "max")
        if self.perf_mode not in va_perf.MODES:
            self.perf_mode = "max"
        va_perf.set_mode(self.perf_mode)
        self.frame_cache = va_perf.FrameCache()   # RAM-backed scrub/step replay
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.source: "VideoSource | None" = None
        self.source_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        self.playing = False
        self.fps = 30.0
        self.total_frames = 0
        self.cur_index = 0
        self._seek_request: "int | None" = None
        self.current_bgr: "np.ndarray | None" = None

        self.is_hdr = False
        self.is_pq = False
        self.is_wide_gamut = False
        self.native_tap = None       # second decode: the file's NATIVE signal
        self.native_meta = None      # {'transfer','primaries','bit_depth'}
        self.current_native = None   # (float32 RGB frame, frame index)
        self._tap_epoch = 0
        self.hdr_label = "SDR"
        self.path: "str | None" = None
        self._closing = False
        self._open_epoch = 0
        self._play_epoch = 0
        self._scrub_job: "str | None" = None
        self._scrub_target: "int | None" = None
        self._uid = "%d_%s" % (os.getpid(), uuid.uuid4().hex[:6])

        self.table: "metrics.MetricsTable | None" = None
        self.events: dict = {}
        self.loudness = None
        self.gamut_cov = None
        self._analyze_thread: "threading.Thread | None" = None
        self._analyze_cancel = False
        self.issues: list = []
        self.forensic_marks: list = []
        self.adv_results: dict = {}
        self.audio = None
        self.audio_wave = None            # waveform_overview dict for the strip
        self.audio_marks: list = []       # (t, label) from the audio battery
        self.scan_marks: list = []        # (t, label) from plugin scans
        self.plugin_buttons: list = []    # plugin menubuttons needing a file
        self.plugin_open_hooks: list = [] # plugin callbacks on file open
        self.plugin_analyze_hooks: list = []  # plugin callbacks post-Analyze
        self.plugin_series: dict = {}     # label -> {t, v, vmin, vmax} curves
        self.plugin_overlays: dict = {}   # name -> fn(frame_bgr, idx)
        self.plugin_keys: dict = {}       # tk sequence -> plugin callback
        self.follower = None              # ffplay-backed audio playback
        self.audio_follow = va_theme.load_ui_key("audio_follow", "on") == "on"
        self._audition_job: "str | None" = None

        self.active_scope = "vectorscope"
        self._pending: "dict | None" = None
        self._present_pending = False
        self._resize_job: "str | None" = None
        self._img_error_shown = False
        self._tmpdir = tempfile.gettempdir()
        self._sizes = {"video": (640, 360), "scope": (380, 360)}
        self._photo_video = None
        self._photo_scope = None
        self._photo_spectro = None
        self._photo_loud = None

        self._build_ui()
        self._bind_shortcuts()
        if initial_path and os.path.isfile(initial_path):
            self.root.after(200, lambda: self._load_video(initial_path))

    def _post(self, fn, *args):
        """Marshal a callback onto the Tk main thread; silently dropped once the
        window is closing (workers must never touch dead widgets)."""
        if self._closing:
            return
        try:
            self.root.after(0, fn, *args)
        except (RuntimeError, tk.TclError):
            pass

    def _pick_font(self, preferred, size, fallback):
        try:
            fams = set(tkfont.families(self.root))
        except tk.TclError:
            return (fallback, size)
        for f in preferred:
            if f in fams:
                return (f, size)
        return (fallback, size)

    def _build_ui(self):
        self.mono_font = self._pick_font(
            ["Consolas", "Menlo", "DejaVu Sans Mono", "Courier New"], 9, "TkFixedFont")

        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=(6, 2))
        self.toolbar = bar                # plugins anchor their menus here
        ttk.Button(bar, text="Open Video...", command=self._open_file).pack(side=tk.LEFT)
        # transport cluster bracketed by the issue-jump buttons:
        # ⏮ issue · ◀| step · ▶ play · step |▶ · issue ⏭
        self.btn_prev_issue = ttk.Button(bar, text="⏮ Issue", command=lambda: self._jump_issue(-1), state=tk.DISABLED)
        self.btn_prev_issue.pack(side=tk.LEFT, padx=(10, 4))
        self.btn_back = ttk.Button(bar, text="◀|", width=3,
                                   command=lambda: self._step_frame(-1), state=tk.DISABLED)
        self.btn_back.pack(side=tk.LEFT)
        self.btn_play = ttk.Button(bar, text="▶  Play", command=self._toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        self.btn_fwd = ttk.Button(bar, text="|▶", width=3,
                                  command=lambda: self._step_frame(1), state=tk.DISABLED)
        self.btn_fwd.pack(side=tk.LEFT)
        self.btn_next_issue = ttk.Button(bar, text="Issue ⏭", command=lambda: self._jump_issue(1), state=tk.DISABLED)
        self.btn_next_issue.pack(side=tk.LEFT, padx=(4, 0))
        self.btn_stop = ttk.Button(bar, text="⏹  Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(10, 4))
        self.btn_sound = ttk.Button(bar, text="🔊" if self.audio_follow else "🔇",
                                    width=3, command=self._toggle_sound, state=tk.DISABLED)
        self.btn_sound.pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=3)
        self.btn_analyze = ttk.Button(bar, text="Analyze", style="Accent.TButton",
                                      command=self._start_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT)

        self.btn_compare = ttk.Menubutton(bar, text="Compare ▾", state=tk.DISABLED)
        cmenu = tk.Menu(self.btn_compare, tearoff=0)
        cmenu.add_command(label="Quality metrics (PSNR/SSIM/VMAF)...", command=self._compare)
        cmenu.add_command(label="A/B visual viewer...", command=self._open_ab_viewer)
        cmenu.add_command(label="Encode ladder - H.264 (find optimal bitrate)...",
                          command=lambda: self._encode_ladder("h264"))
        cmenu.add_command(label="Encode ladder - HEVC (libx265)...",
                          command=lambda: self._encode_ladder("hevc"))
        cmenu.add_command(label="Encode ladder - AV1 (SVT-AV1)...",
                          command=lambda: self._encode_ladder("av1"))
        self.btn_compare["menu"] = cmenu
        self.btn_compare.pack(side=tk.LEFT, padx=4)
        self.btn_advanced = ttk.Menubutton(bar, text="Advanced ▾", state=tk.DISABLED)
        amenu = tk.Menu(self.btn_advanced, tearoff=0)
        # one submenu per tool family (speedrun verification lives in its own
        # top-level Speedrun menu, below)
        m_qc = tk.Menu(amenu, tearoff=0)
        m_qc.add_command(label="QC check (general profile)...", command=self._run_qc)
        m_qc.add_command(label="Banding map (current frame)", command=self._show_banding)
        amenu.add_cascade(label="QC & picture", menu=m_qc)
        m_hdr = tk.Menu(amenu, tearoff=0)
        m_hdr.add_command(label="HDR multi-display preview", command=self._show_hdr_multi)
        m_hdr.add_command(label="HDR metadata report (DV / HDR10+)...", command=self._hdr_meta_report)
        m_hdr.add_command(label="Dynamic metadata vs content...", command=self._dynhdr_verify)
        m_hdr.add_command(label="Dynamic vs static tonemap A/B...", command=self._dynamic_vs_static)
        m_hdr.add_command(label="Plot DV L1 brightness (dovi_tool)...", command=self._plot_dovi)
        m_hdr.add_command(label="Plot HDR10+ brightness (hdr10plus_tool)...", command=self._plot_h10p)
        amenu.add_cascade(label="HDR & dynamic metadata", menu=m_hdr)
        m_cont = tk.Menu(amenu, tearoff=0)
        m_cont.add_command(label="MediaInfo report...", command=self._mediainfo_report)
        m_cont.add_command(label="MP4 boxes / DV signaling (mp4dump)...", command=self._mp4_boxes)
        m_cont.add_command(label="MKV structure / DV signaling (mkvinfo)...", command=self._mkv_boxes)
        amenu.add_cascade(label="Container & structure", menu=m_cont)
        m_for = tk.Menu(amenu, tearoff=0)
        m_for.add_command(label="Forensics report (integrity scan)...",
                          command=self._forensics_report)
        m_for.add_command(label="Content Credentials (C2PA)...", command=self._c2pa_report)
        m_for.add_command(label="ELA residual map (current frame)", command=self._show_ela)
        m_for.add_command(label="Noise consistency map (current frame)",
                          command=self._show_noise)
        m_for.add_command(label="ENF mains-hum trace...", command=self._show_enf)
        amenu.add_cascade(label="Forensics & integrity", menu=m_for)
        self.btn_advanced["menu"] = amenu
        self.btn_advanced.pack(side=tk.LEFT, padx=4)
        self.export_mb = ttk.Menubutton(bar, text="Export ▾", state=tk.DISABLED)
        menu = tk.Menu(self.export_mb, tearoff=0)
        menu.add_command(label="Per-frame CSV...", command=lambda: self._export("csv"))
        menu.add_command(label="Analysis JSON...", command=lambda: self._export("json"))
        menu.add_command(label="HTML report...", command=lambda: self._export("html"))
        self.export_mb["menu"] = menu
        self.export_mb.pack(side=tk.LEFT, padx=4)
        self.btn_plugins = ttk.Menubutton(bar, text="Plugins ▾")
        pmenu = tk.Menu(self.btn_plugins, tearoff=0)
        pmenu.add_command(label="Manage plugins...", command=self._open_plugins)
        pmenu.add_separator()
        # enabled plugins append their own cascades here via the AppApi
        self.btn_plugins["menu"] = pmenu
        self.plugins_menu = pmenu
        self._menus = [menu, cmenu, amenu, m_qc, m_hdr, m_cont, m_for, pmenu]
        for m in self._menus:
            va_theme.style_menu(m)
        ttk.Button(bar, text="Tools", command=self._open_tools).pack(side=tk.LEFT, padx=4)
        self.btn_plugins.pack(side=tk.LEFT, padx=4)

        self.lbl_hdr = ttk.Label(bar, text="", style="SDR.TLabel")
        self.lbl_hdr.pack(side=tk.LEFT, padx=(16, 0))
        self.lbl_status = ttk.Label(bar, text="No file loaded  ·  Space play/pause, ←/→ seek, n next issue, O open",
                                    foreground=clr("muted"))
        self.lbl_status.pack(side=tk.RIGHT)
        self.btn_theme = ttk.Button(bar, text="◐", width=3, command=self._toggle_theme)
        self.btn_theme.pack(side=tk.RIGHT, padx=(8, 10))

        seek = ttk.Frame(self.root)
        seek.pack(fill=tk.X, padx=6, pady=2)
        self.seek_var = tk.IntVar(value=0)
        self.seeking = False
        self.seek_bar = ttk.Scale(seek, from_=0, to=100, orient=tk.HORIZONTAL,
                                  variable=self.seek_var, command=self._on_seek)
        self.seek_bar.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.lbl_time = ttk.Label(seek, text="00:00:00 / 00:00:00", width=24, anchor=tk.CENTER)
        self.lbl_time.pack(side=tk.RIGHT)

        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 4))

        left = ttk.PanedWindow(pane, orient=tk.VERTICAL)
        pane.add(left, weight=3)
        vf = ttk.LabelFrame(left, text=" VIDEO ")
        left.add(vf, weight=3)
        self.video_label = tk.Label(vf, bg=va_theme.WELL["bg"], fg=va_theme.WELL["faint"],
                                    text="\n\n  Open a video to begin  \n\n")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        info = ttk.LabelFrame(left, text=" STREAM & HDR INFORMATION ")
        left.add(info, weight=2)
        self.info_text = scrolledtext.ScrolledText(
            info, wrap=tk.WORD, font=self.mono_font, state=tk.DISABLED, relief=tk.FLAT)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._skin_info_text()

        right = ttk.Frame(pane)
        pane.add(right, weight=2)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.scope_labels = {}
        for label, key in SCOPES:
            fr = ttk.Frame(self.notebook, width=420, height=380)
            fr.pack_propagate(False)   # a large scope image must not resize the layout
            self.notebook.add(fr, text=label)
            lab = tk.Label(fr, bg=va_theme.WELL["bg"])
            lab.pack(fill=tk.BOTH, expand=True)
            self.scope_labels[key] = lab
        af = ttk.Frame(self.notebook, width=600, height=380)
        af.pack_propagate(False)
        self.notebook.add(af, text="Audio")
        atb = ttk.Frame(af)
        atb.pack(fill=tk.X, padx=4, pady=(4, 0))
        self.btn_audio_forensics = ttk.Button(atb, text="Audio forensics...",
                                              command=self._audio_forensics,
                                              state=tk.DISABLED)
        self.btn_audio_forensics.pack(side=tk.LEFT)
        ttk.Label(atb, text="drag = scrub · wheel = zoom · double-click = fit",
                  foreground=clr("muted")).pack(side=tk.RIGHT)
        self.audio_strip = AudioStrip(af, self._strip_seek, height=86)
        self.audio_strip.pack(fill=tk.X, padx=4, pady=(3, 2))
        self.audio_stats = tk.Label(af, bg=va_theme.WELL["panel"], fg=va_theme.WELL["text"],
                                    anchor="w", justify=tk.LEFT,
                                    font=self.mono_font, text="Audio appears after Analyze.")
        self.audio_stats.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.audio_spectro = tk.Label(af, bg=va_theme.WELL["panel"])
        self.audio_spectro.pack(pady=2)
        self.audio_loud = tk.Label(af, bg=va_theme.WELL["panel"])
        self.audio_loud.pack(pady=2)
        self._build_tools_window()
        # plugins last: the full api surface (menus, status, marks) is live now
        self.plugin_api = va_plugins.AppApi(self)
        self.plugin_report = va_plugins.load_gui(self.plugin_api)
        errs = ["%s (%s)" % (r["name"], r["error"])
                for r in self.plugin_report if r["ok"] is False]
        if errs:
            self.lbl_status.config(text="plugin errors: " + "; ".join(errs),
                                   foreground=clr("err"))
        self.root.after(400, self._refresh_setup)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_scope_change)
        self.readout = ttk.Label(right, foreground=clr("muted"), anchor=tk.W, justify=tk.LEFT,
                                 text="Loudness / gamut / quality appear here after Analyze.")
        self.readout.pack(fill=tk.X, padx=2, pady=(3, 0))

        tl = ttk.LabelFrame(self.root, text=" METRICS TIMELINE ")
        tl.pack(fill=tk.X, padx=6, pady=(0, 6))
        head = ttk.Frame(tl)
        head.pack(fill=tk.X)
        ttk.Label(head, text="Metric:").pack(side=tk.LEFT, padx=(4, 4))
        self.metric_var = tk.StringVar(value=TIMELINE_METRICS[0][0])
        self.metric_combo = ttk.Combobox(head, textvariable=self.metric_var, state="readonly",
                                         width=22, values=[m[0] for m in TIMELINE_METRICS])
        self.metric_combo.pack(side=tk.LEFT)
        self.metric_combo.bind("<<ComboboxSelected>>", lambda e: self._draw_timeline())
        self.lbl_analysis = ttk.Label(head, text="", foreground=clr("muted"))
        self.lbl_analysis.pack(side=tk.RIGHT, padx=6)
        self.timeline = tk.Canvas(tl, height=96, bg=va_theme.WELL["bg"], highlightthickness=0)
        self.timeline.pack(fill=tk.X, padx=2, pady=2)
        self.timeline.bind("<Button-1>", self._on_timeline_click)
        self.timeline.bind("<B1-Motion>", self._on_timeline_click)
        self.timeline.bind("<Configure>", lambda e: self._draw_timeline())

        if HAVE_DND:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)
                self.video_label.drop_target_register(DND_FILES)
                self.video_label.dnd_bind("<<Drop>>", self._on_drop)
                self.video_label.config(text="  Open or drag a video here  ")
            except Exception:
                pass
        self.root.bind("<Configure>", self._on_configure)
        self.root.after(300, self._update_sizes)

    def _hotkeys_ok(self) -> bool:
        """Suppress app hotkeys while an interactive widget has focus, so Space
        does not both press a focused button AND toggle playback, and arrows
        do not double-seek when the seek bar or a combobox has focus."""
        try:
            w = self.root.focus_get()
        except (KeyError, tk.TclError):
            return True
        return not isinstance(w, (ttk.Button, ttk.Combobox, ttk.Scale,
                                  ttk.Entry, ttk.Spinbox, tk.Entry, tk.Text))

    def _hk(self, fn):
        return lambda e: fn() if self._hotkeys_ok() else None

    RESERVED_KEYS = frozenset((
        "<space>", "<Left>", "<Right>", "<comma>", "<period>", "<Home>",
        "<o>", "<O>", "<n>", "<N>", "<p>", "<P>"))

    def _bind_shortcuts(self):
        self.root.bind("<space>", self._hk(self._toggle_play))
        self.root.bind("<Left>", self._hk(lambda: self._nudge(-self.fps)))
        self.root.bind("<Right>", self._hk(lambda: self._nudge(self.fps)))
        self.root.bind("<comma>", self._hk(lambda: self._step_frame(-1)))
        self.root.bind("<period>", self._hk(lambda: self._step_frame(1)))
        self.root.bind("<Home>", self._hk(self._stop))
        for key in ("<o>", "<O>"):
            self.root.bind(key, self._hk(self._open_file))
        for key in ("<n>", "<N>"):
            self.root.bind(key, self._hk(lambda: self._jump_issue(1)))
        for key in ("<p>", "<P>"):
            self.root.bind(key, self._hk(lambda: self._jump_issue(-1)))

    # --- file loading & metadata --------------------------------------------

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v "
                                       "*.ts *.mpg *.mpeg *.3gp *.hevc *.265"),
                       ("All files", "*.*")])
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        """Kick off a threaded open: VideoSource construction probes decode
        pipelines (seconds, sometimes much longer on broken GPU drivers), so it
        must not run on the UI thread."""
        if self._closing:
            return
        self._open_epoch += 1
        epoch = self._open_epoch
        self.playing = False
        self.btn_play.config(text="▶  Play")
        self._seek_request = None
        self._analyze_cancel = True
        try:
            metrics.kill_all()           # abort any in-flight analysis decodes
        except Exception:                # noqa: BLE001
            pass
        self._stop_tap()
        if self.follower:
            self.follower.stop()
        self.lbl_status.config(text="Opening %s ..." % os.path.basename(path),
                               foreground=clr("warn"))
        for b in (self.btn_play, self.btn_stop, self.btn_back, self.btn_fwd,
                  self.btn_analyze, self.btn_compare, self.btn_advanced,
                  self.btn_sound, self.btn_audio_forensics,
                  *self.plugin_buttons):
            b.config(state=tk.DISABLED)
        threading.Thread(target=self._open_worker, args=(path, epoch), daemon=True).start()

    def _open_worker(self, path: str, epoch: int):
        try:
            src = VideoSource(path)
        except Exception as exc:  # noqa: BLE001 - any decode-layer surprise
            self._post(self._open_failed, path, epoch, repr(exc))
            return
        if not (bool(src.info.get("ok")) and src.width > 0):
            src.close()
            self._post(self._open_failed, path, epoch, None)
            return
        self._post(self._open_done, path, epoch, src)

    def _open_failed(self, path: str, epoch: int, err: "str | None"):
        if epoch != self._open_epoch:
            return
        detail = ("\n\n" + err) if err else ""
        messagebox.showerror("Video Analyzer", "Could not open video file:\n\n%s%s" % (path, detail))
        self.lbl_status.config(text="Error: cannot open file", foreground=clr("err"))

    def _open_done(self, path: str, epoch: int, src):
        if epoch != self._open_epoch or self._closing:
            src.close()                  # a newer open superseded this one
            return
        with self.source_lock:
            if self.source:
                self.source.close()
            self.source = src
        self.fps = src.fps or 30.0
        self.total_frames = src.nb_frames or 0
        self.is_hdr = bool(src.info["is_hdr"])
        self.is_pq = bool(src.info["is_pq"])
        self.is_wide_gamut = bool(src.info["is_wide_gamut"])
        # scopes must read the FILE's colour, not the tonemapped preview:
        # spin up the native-signal tap (second decode, no tonemap)
        self._stop_tap()
        if NativeTap.needed(src.info):
            threading.Thread(target=self._tap_start,
                             args=(path, dict(src.info), self._tap_epoch),
                             daemon=True).start()

        self.path = path
        self.cur_index = 0
        self.table = None
        self.events = {}
        self.loudness = None
        self.gamut_cov = None
        self.forensic_marks = []
        self.adv_results = {}
        self.audio = None
        self.audio_wave = None
        self.audio_marks = []
        self.scan_marks = []
        self.plugin_series = {}           # per-file curves die with the file
        self._refresh_metric_choices()
        for cb in list(self.plugin_open_hooks):
            try:
                cb(path)
            except Exception:  # noqa: BLE001 - a plugin hook must not break open
                pass
        if self.follower:
            self.follower.stop()
        self.follower = va_audio.AudioFollower(path)
        self.audio_strip.set_marks([])
        self.audio_strip.set_silence([])
        self.audio_strip.set_data(None)
        threading.Thread(target=self._wave_worker, args=(path, epoch),
                         daemon=True).start()
        self.frame_cache.clear()                  # new file, new geometry
        self.seek_bar.config(to=max(self.total_frames - 1, 1))
        for b in (self.btn_play, self.btn_stop, self.btn_back, self.btn_fwd,
                  self.btn_analyze, self.btn_compare, self.btn_advanced,
                  *self.plugin_buttons):
            b.config(state=tk.NORMAL)
        self.export_mb.config(state=tk.DISABLED)
        if self.is_hdr and self.source.backend == "opencv":
            self.lbl_status.config(
                text="%s  ·  opencv fallback - HDR preview NOT tonemapped (install ffmpeg)"
                     % os.path.basename(path), foreground=clr("warn"))
        elif "ipt-np" in self.source.backend:
            self.lbl_status.config(
                text="%s  ·  %s  ·  DV P5: software IPT decode (colour approximate - "
                     "an ffmpeg build with libplacebo renders the RPU exactly)"
                     % (os.path.basename(path), self.source.backend),
                foreground=clr("warn"))
        elif "libplacebo-dovi" in self.source.backend:
            self.lbl_status.config(
                text="%s  ·  %s  ·  DV P5 rendered via Dolby Vision RPU"
                     % (os.path.basename(path), self.source.backend),
                foreground=clr("ok"))
        else:
            self.lbl_status.config(text="%s  ·  %s" % (os.path.basename(path), self.source.backend),
                                   foreground=clr("ok"))
        self.lbl_analysis.config(text="not analysed")
        self.readout.config(text="Analyzing in the background...")
        self._draw_timeline()

        threading.Thread(target=self._extract_info, args=(path,), daemon=True).start()
        if not self._render_frame_at(0):
            self.btn_play.config(state=tk.DISABLED)
            self.lbl_status.config(text="Opened, but no frame could be decoded", foreground=clr("warn"))
            messagebox.showwarning("Video Analyzer",
                                   "Metadata was read, but no video frame could be decoded.")
            return
        self.root.after(350, self._start_analysis)

    def _extract_info(self, path: str):
        probe = ffprobe_json(path, frames=True)
        if probe:
            detail = format_ffprobe(probe)
            hdr_label, is_wide = "SDR", False
            for s in probe.get("streams", []):
                if s.get("codec_type") == "video":
                    frame_sd = []
                    for f in probe.get("frames", []):
                        if f.get("stream_index", 0) == s.get("index", 0):
                            frame_sd.extend(f.get("side_data_list", []))
                    all_sd = s.get("side_data_list", []) + frame_sd
                    info = classify_hdr(s, all_sd or None)
                    hdr_label = info.get("HDR Format", "SDR")
                    is_wide = "Wide Gamut" in info
                    break
            summary = [("HDR Format", hdr_label)] + build_summary(probe)
        else:
            ocv = extract_with_opencv(path)
            detail = format_opencv(ocv.get("opencv", ocv))
            hdr_label = "SDR (metadata limited)" if find_ffprobe() else "SDR (no ffprobe)"
            is_wide = False
            summary = [("HDR Format", hdr_label), ("Metadata", "OpenCV fallback")]
        self.hdr_label = hdr_label
        self.is_wide_gamut = is_wide
        # NOTE: is_hdr/is_pq stay as probed from the stream (set in _open_done);
        # the label is display-only. The old substring test ("HDR" in label)
        # wrongly enabled HDR tools for "Wide Color Gamut (possibly HDR)" files.
        self._post(self._set_info, summary, detail)
        self._post(self._update_hdr_badge)

    def _set_info(self, summary, detail):
        it = self.info_text
        it.config(state=tk.NORMAL)
        it.delete("1.0", tk.END)
        it.insert(tk.END, "  ▌ SUMMARY\n\n", ("summary_title",))
        for label, val in summary:
            it.insert(tk.END, "  %-16s" % label, ("key",))
            it.insert(tk.END, "%s\n" % val, ("value",))
        it.insert(tk.END, "\n  " + "─" * 56 + "\n\n", ("dim",))
        for line in detail.split("\n"):
            self._insert_detail_line(line)
        it.config(state=tk.DISABLED)

    def _insert_detail_line(self, line: str):
        it = self.info_text
        stripped = line.strip()
        if any(c in line for c in "═┌┐└┘│"):
            it.insert(tk.END, line + "\n", ("header",))
        elif stripped and stripped == stripped.upper() and ":" not in stripped \
                and any(ch.isalpha() for ch in stripped):
            it.insert(tk.END, line + "\n", ("header",))
        elif " : " in line:
            k, v = line.split(" : ", 1)
            it.insert(tk.END, k + " : ", ("key",))
            it.insert(tk.END, v + "\n", ("value",))
        else:
            it.insert(tk.END, line + "\n")

    def _update_hdr_badge(self):
        if self.is_hdr:
            self.lbl_hdr.config(text="⬤  %s" % self.hdr_label, style="HDR.TLabel")
        else:
            self.lbl_hdr.config(text="○  %s" % self.hdr_label, style="SDR.TLabel")

    # --- background analysis -------------------------------------------------

    def _start_analysis(self):
        if not self.path or self._closing:
            return
        if self._analyze_thread and self._analyze_thread.is_alive():
            # cancel the previous run (kills its ffmpeg children) and retry
            # shortly - the old behavior was a silent no-op that left the
            # Analyze button apparently broken.
            self._analyze_cancel = True
            try:
                metrics.kill_all()
            except Exception:  # noqa: BLE001
                pass
            self.lbl_analysis.config(text="restarting analysis...")
            self.root.after(150, self._start_analysis)
            return
        self._analyze_cancel = False
        self.btn_analyze.config(state=tk.DISABLED)
        self.lbl_analysis.config(text="analysing...")
        self._analyze_thread = threading.Thread(
            target=self._analysis_worker, args=(self.path,), daemon=True)
        self._analyze_thread.start()

    def _analysis_worker(self, path: str):
        try:
            cancel = lambda: self._analyze_cancel  # noqa: E731
            # all whole-file video measurements in ONE decode (signalstats +
            # black/freeze/scene events + loudness/silence when audio exists);
            # va_metrics.analyze_pass falls back to None if the combined graph
            # cannot run, in which case the per-pass path below still works
            combined = metrics.analyze_pass(path, on_progress=self._progress_cb,
                                            cancel=cancel)
            if self._analyze_cancel or path != self.path:
                return
            if combined is not None:
                table, ev = combined["table"], combined["events"]
            else:
                table = metrics.signalstats(path, on_progress=self._progress_cb,
                                            cancel=cancel)
                if self._analyze_cancel or path != self.path:
                    return
                ev = {"black": metrics.black_segments(path, cancel=cancel),
                      "freeze": metrics.freeze_segments(path, cancel=cancel),
                      "scene_cuts": metrics.scene_cuts(path, cancel=cancel)}
            cov = None
            with self.frame_lock:
                nat = self.current_native
                bgr = None if self.current_bgr is None else self.current_bgr.copy()
            nmeta = self.native_meta
            try:
                if nat is not None and nmeta:
                    _, cov = scopes.cie_gamut_native(
                        nat[0], 320, nmeta["transfer"], nmeta["primaries"])
                elif bgr is not None:
                    _, cov = scopes.cie_gamut(bgr, 320)
            except Exception:
                cov = None
            audio = None
            loud = None
            if va_audio.has_audio(path):
                spectro = os.path.join(self._tmpdir, "va_spectro_%s.png" % self._uid)
                va_audio.spectrogram_png(path, spectro, 560, 250, cancel=cancel)
                from_pass = (combined or {}).get("audio")
                audio = {"loudness": (from_pass if from_pass and from_pass.get("summary")
                                      else va_audio.loudness(path, cancel=cancel)),
                         "astats": va_audio.astats(path, cancel=cancel),
                         "silence": (combined["silence"] if combined is not None
                                     else va_audio.silence_segments(path, cancel=cancel)),
                         "correlation": va_audio.correlation(path, cancel=cancel),
                         "spectro": spectro}
                # loudness + silence already rode the combined video pass
                # (zero extra decodes); astats/phase/spectrogram stay audio-only
                loud = (audio.get("loudness") or {}).get("summary")
            if self._analyze_cancel or path != self.path:
                return
            perf_note = None
            if combined is not None:
                perf_note = "%d analyses · 1 decode (%s) · %.0fs" % (
                    combined["passes_merged"], combined["decode"], combined["elapsed"])
            self._post(self._analysis_done, table, ev, loud, cov, audio, perf_note)
        except Exception as exc:  # surface failures instead of dying silently
            self._post(self._analysis_failed, repr(exc))

    def _progress_cb(self, n: int):
        self._post(lambda: self.lbl_analysis.config(text="analysing... %d frames" % n))

    def _analysis_done(self, table, ev, loud, cov, audio=None, perf_note=None):
        self.table = table
        self.events = ev
        self.loudness = loud
        self.gamut_cov = cov
        self.audio = audio
        self.btn_analyze.config(state=tk.NORMAL)
        self.export_mb.config(state=tk.NORMAL)
        self.issues = issue_frames(ev, self.fps,
                                   extra=self.forensic_marks + self.audio_marks + self.scan_marks)
        st = tk.NORMAL if self.issues else tk.DISABLED
        self.btn_prev_issue.config(state=st)
        self.btn_next_issue.config(state=st)
        if len(table) == 0:
            why = ("no ffmpeg/ffprobe found — put ffmpeg(.exe) beside the script"
                   if not (find_ffmpeg() or find_ffprobe())
                   else "no data returned (ffmpeg could not read this file)")
            self.lbl_analysis.config(text="0 frames — " + why)
            self.lbl_status.config(text="Analyze: " + why, foreground=clr("err"))
        else:
            note = "" if find_ffmpeg() else "  (add ffmpeg for events/loudness)"
            extra = (" · " + perf_note) if perf_note else ""
            self.lbl_analysis.config(
                text="%d frames · %d issues%s%s" % (len(table), len(self.issues),
                                                    note, extra))
        self._update_readout()
        self._set_audio()
        self.audio_strip.set_silence((audio or {}).get("silence") or [])
        for cb in list(self.plugin_analyze_hooks):
            try:
                cb(self.path)
            except Exception:  # noqa: BLE001 - a plugin hook must not break Analyze
                pass
        self._draw_timeline()

    @staticmethod
    def _num(v, fmt="%.1f"):
        if v is None or v != v:
            return "n/a"
        if v == float("-inf"):
            return "-inf"
        try:
            return fmt % v
        except (TypeError, ValueError):
            return str(v)

    def _update_readout(self):
        parts = []
        if self.loudness:
            parts.append("Loudness  I %s LUFS · LRA %s LU · TP %s dBFS" % (
                self._num(self.loudness.get("integrated_lufs")),
                self._num(self.loudness.get("lra_lu")),
                self._num(self.loudness.get("true_peak_dbfs"))))
        else:
            reason = va_audio.last_error()
            parts.append("Loudness  " + {
                "timeout": "audio analysis timed out",
                "cancelled": "audio analysis cancelled",
                "ffmpeg failed": "audio analysis failed",
                "no ffmpeg": "needs ffmpeg",
            }.get(reason or "", "no audio track"))
        if self.gamut_cov:
            parts.append("Gamut  %.0f%% of Rec.2020 used · %.0f%% outside Rec.709" % (
                self.gamut_cov.get("coverage_2020_pct", 0), self.gamut_cov.get("outside_709_pct", 0)))
        self.readout.config(text="     ".join(parts), foreground=clr("soft"))

    # --- playback ------------------------------------------------------------

    def _toggle_play(self):
        if not self.source:
            return
        if self.playing:
            self.playing = False
            self.btn_play.config(text="▶  Play")
        else:
            if self.total_frames and self.cur_index >= self.total_frames - 1:
                self.cur_index = 0       # play at EOF restarts from the top
            self.playing = True
            self._play_epoch += 1
            self.btn_play.config(text="⏸  Pause")
            threading.Thread(target=self._play_loop, args=(self._play_epoch,),
                             daemon=True).start()

    def _stop(self):
        self.playing = False
        self.btn_play.config(text="▶  Play")
        self._seek_request = None
        if self.follower:
            self.follower.stop()
        if self.source:
            self._render_frame_at(0)

    def _on_playback_end(self):
        self.playing = False
        self.btn_play.config(text="▶  Play")

    def _play_loop(self, epoch: int):
        src = self.source
        if src is None:
            return
        with self.source_lock:
            src.start(self.cur_index)
            base_idx = src.index
        base_time = time.perf_counter()
        last_chart = 0.0
        if (self.audio_follow and self.follower and self.follower.available
                and self.audio_wave):
            self.follower.start(base_idx / self.fps if self.fps else 0.0)
        while self.playing and epoch == self._play_epoch:
            fol = self.follower if (self.audio_follow and self.follower
                                    and self.follower.available
                                    and self.audio_wave) else None
            seek = self._seek_request
            if seek is not None:
                self._seek_request = None
                with self.source_lock:
                    src.seek(seek)
                base_idx = seek
                base_time = time.perf_counter()
                if fol:
                    fol.start(seek / self.fps if self.fps else 0.0)
            now = time.perf_counter()
            target = base_idx + int((now - base_time) * self.fps)
            with self.source_lock:
                cur = src.index
                skip = target - cur
                if skip > 1:
                    src.grab(min(skip - 1, 120))
                frame = src.read()
                pos = src.index
            if frame is None:
                if epoch == self._play_epoch:
                    self.playing = False
                    self._post(self._on_playback_end)
                break
            self.cur_index = pos
            with self.frame_lock:
                self.current_bgr = frame
            self.frame_cache.put(pos, frame)   # replay/scrub-back is then free
            want_charts = (now - last_chart) * 1000.0 >= self.CHART_UPDATE_MS
            if want_charts:
                last_chart = now
                if fol and self.fps:
                    exp = fol.expected_t()
                    vt = pos / self.fps
                    near_end = (self.total_frames
                                and pos >= self.total_frames - int(2 * self.fps))
                    if (exp is None and not near_end) or \
                            (exp is not None and abs(exp - vt) > 0.4):
                        fol.start(vt)   # (re)sync audio to the video clock
            payload = self._render_all(frame, want_charts)
            payload["pos"] = pos
            with self.frame_lock:
                self._pending = payload
            self._schedule_present()
            nxt = base_time + (pos - base_idx) / self.fps
            sleep_t = nxt - time.perf_counter()
            if sleep_t > 0:
                time.sleep(min(sleep_t, 0.25))
        if self.follower:
            self.follower.stop()        # paused/EOF/superseded - audio off

    def _render_frame_at(self, idx: int) -> bool:
        frame = self.frame_cache.get(idx)      # replay from RAM when we can -
        if frame is None:                      # zero decode work on a cache hit
            with self.source_lock:
                if self.source is None:
                    return False
                self.source.seek(idx)
                frame = self.source.read()
            if frame is None:
                return False
            self.frame_cache.put(idx, frame)
        self.cur_index = idx
        with self.frame_lock:
            self.current_bgr = frame
        self._update_sizes()
        payload = self._render_all(frame, True)
        payload["pos"] = idx
        self._apply_payload(payload)
        return True

    # --- present (main thread only) -----------------------------------------

    def _schedule_present(self):
        if self._present_pending:
            return
        self._present_pending = True
        self.root.after(0, self._present)

    def _present(self):
        self._present_pending = False
        with self.frame_lock:
            payload = self._pending
            self._pending = None
        if payload:
            self._apply_payload(payload)

    def _apply_payload(self, payload: dict):
        if payload.get("video"):
            img = self._photo_from_ppm(payload["video"], "video")
            if img is not None:
                self.video_label.config(image=img, text="")
                self._photo_video = img
        if payload.get("scope"):
            img = self._photo_from_ppm(payload["scope"], "scope")
            if img is not None:
                self.scope_labels[self.active_scope].config(image=img)
                self._photo_scope = img
        pos = payload.get("pos")
        if pos is not None and not self.seeking:
            self.seeking = True
            self.seek_var.set(pos)
            self._update_time_label(pos)
            self.seeking = False
            self._move_playhead(pos)

    def _photo_from_ppm(self, ppm: bytes, kind: str):
        try:
            path = os.path.join(self._tmpdir, "video_analyzer_%s_%s.ppm" % (self._uid, kind))
            with open(path, "wb") as fh:
                fh.write(ppm)
            return tk.PhotoImage(file=path)
        except Exception as exc:  # noqa: BLE001
            if not self._img_error_shown:
                self._img_error_shown = True
                self.lbl_status.config(text="Cannot display frames: %s" % exc, foreground=clr("err"))
            return None

    # --- seeking -------------------------------------------------------------

    def _on_seek(self, val):
        if self.seeking:
            return
        frame_no = int(float(val))
        if self.playing:
            self._seek_request = frame_no
        else:
            # debounce: dragging fires per pixel, and every render is a fresh
            # ffmpeg seek+decode - coalesce to one render per ~60 ms
            self._scrub_target = frame_no
            if self._scrub_job is None:
                self._scrub_job = self.root.after(60, self._scrub_render)
        self._update_time_label(frame_no)

    def _scrub_render(self):
        self._scrub_job = None
        tgt, self._scrub_target = self._scrub_target, None
        if tgt is None or self._closing:
            return
        self._render_frame_at(tgt)
        if (not self.playing and self.audio_follow and self.audio_wave
                and os.name == "nt" and self.path):
            if self._audition_job is not None:
                try:
                    self.root.after_cancel(self._audition_job)
                except (ValueError, tk.TclError):
                    pass
            self._audition_job = self.root.after(150, self._audition_now)
        if self._scrub_target is not None and self._scrub_job is None:
            self._scrub_job = self.root.after(30, self._scrub_render)  # chase the drag

    def _nudge(self, delta):
        if not self.source:
            return
        cur = self.seek_var.get()
        tgt = int(max(0, min(self.total_frames - 1, cur + delta)))
        self.seeking = True
        self.seek_var.set(tgt)
        self.seeking = False
        if self.playing:
            self._seek_request = tgt
        else:
            self._render_frame_at(tgt)
        self._update_time_label(tgt)

    def _step_frame(self, delta):
        """Advance/reverse exactly one frame. Stepping implies frame-accurate
        inspection, so playback pauses first (matches NLE jog behavior)."""
        if not self.source:
            return
        if self.playing:
            self.playing = False
            self.btn_play.config(text="▶  Play")
        self._nudge(delta)

    def _toggle_sound(self):
        """Speaker button: audio follows playback (and scrubs) when on."""
        self.audio_follow = not self.audio_follow
        va_theme.save_ui_key("audio_follow", "on" if self.audio_follow else "off")
        self.btn_sound.config(text="🔊" if self.audio_follow else "🔇")
        if not self.audio_follow:
            if self.follower:
                self.follower.stop()
            self.lbl_status.config(text="audio muted", foreground=clr("muted"))
        else:
            note = "audio follow ON"
            if self.follower and not self.follower.available:
                note += " - ffplay not found (bundle it beside the script)"
            elif self.playing and self.follower and self.fps:
                self.follower.start(self.cur_index / self.fps)
            self.lbl_status.config(text=note, foreground=clr("ok"))

    def _strip_seek(self, frac):
        """Click/drag on the audio waveform strip = scrub the video."""
        if not self.source or not self.total_frames:
            return
        tgt = int(max(0, min(self.total_frames - 1,
                             round(frac * (self.total_frames - 1)))))
        self.seeking = True
        self.seek_var.set(tgt)
        self.seeking = False
        if self.playing:
            self._seek_request = tgt
        else:
            self._scrub_target = tgt
            if self._scrub_job is None:
                self._scrub_job = self.root.after(30, self._scrub_render)
        self._update_time_label(tgt)
        self._move_playhead(tgt)

    def _audition_now(self):
        """Paused-scrub audition: play ~1/3 s of audio at the playhead
        (Windows winsound; snippet extraction runs off-thread)."""
        self._audition_job = None
        if self.playing or not self.path or not self.fps:
            return
        t = self.cur_index / self.fps
        path = self.path
        wav = os.path.join(self._tmpdir, "va_audition_%s.wav" % self._uid)

        def worker():
            out = va_audio.audition_wav(path, t, out_path=wav)
            if out and os.name == "nt":
                try:
                    import winsound
                    winsound.PlaySound(out, winsound.SND_FILENAME |
                                       winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                except Exception:   # noqa: BLE001 - audition is best-effort
                    pass
        threading.Thread(target=worker, daemon=True).start()

    def _wave_worker(self, path, epoch):
        ov = va_audio.waveform_overview(path)
        self._post(self._wave_done, path, epoch, ov)

    def _wave_done(self, path, epoch, ov):
        if epoch != self._open_epoch or self._closing or path != self.path:
            return
        self.audio_wave = ov
        self.audio_strip.set_data(ov)
        has = bool(ov)
        self.btn_sound.config(state=tk.NORMAL if has else tk.DISABLED)
        self.btn_audio_forensics.config(state=tk.NORMAL if has else tk.DISABLED)

    def _audio_forensics(self):
        if not self.path:
            return
        self.lbl_status.config(text="Audio forensics battery (clipping / dropouts / "
                                    "splices / channels / bandwidth)...",
                               foreground=clr("warn"))
        series = None
        if isinstance(self.audio, dict) and isinstance(self.audio.get("loudness"), dict):
            series = self.audio["loudness"]
        self._run_bg(self._audio_forensics_worker, self.path, series)

    def _audio_forensics_worker(self, path, series):
        rep = va_audio.forensic_battery(
            path, loudness_series=series,
            on_progress=lambda m: self._post(
                lambda m=m: self.lbl_status.config(text="%s..." % m,
                                                   foreground=clr("warn"))))
        self._post(self._audio_forensics_done, path, rep)

    def _audio_forensics_done(self, path, rep):
        if path != self.path or self._closing:
            return
        finds = [f for f in rep.get("findings", []) if f.get("t") is not None]
        self.audio_marks = [(float(f["t"]), "audio (%s)" % f.get("severity", ""))
                            for f in finds]
        self.audio_strip.set_marks(
            [(float(f["t"]), f.get("severity", "info")) for f in finds])
        self.issues = issue_frames(self.events, self.fps,
                                   extra=self.forensic_marks + self.audio_marks + self.scan_marks)
        st = tk.NORMAL if self.issues else tk.DISABLED
        self.btn_prev_issue.config(state=st)
        self.btn_next_issue.config(state=st)
        warns = sum(1 for f in rep.get("findings", []) if f.get("severity") == "warn")
        self.lbl_status.config(text="Audio forensics: %s" % rep.get("summary", "done"),
                               foreground=clr("warn") if warns else clr("ok"))
        self._adv_store("audio_forensics", "Audio forensics battery",
                        text=va_audio.render_battery(rep), data=rep)
        self._text_popup("Audio forensics - %s" % os.path.basename(path),
                         va_audio.render_battery(rep))

    def _update_time_label(self, frame_no):
        cur = frame_no / self.fps if self.fps else 0
        tot = self.total_frames / self.fps if self.fps else 0
        self.lbl_time.config(text="%s / %s" % (
            time.strftime("%H:%M:%S", time.gmtime(cur)),
            time.strftime("%H:%M:%S", time.gmtime(tot))))

    # --- rendering (pure compute, safe off-thread) --------------------------

    def _render_all(self, bgr, want_charts: bool) -> dict:
        out = {}
        out["video"] = encode_ppm_bytes(self._render_video(bgr, self._sizes["video"]))
        if want_charts:
            out["scope"] = encode_ppm_bytes(self._render_active_scope(bgr))
        return out

    def _analysis_frame(self, bgr):
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m <= self.ANALYSIS_MAX:
            return bgr
        s = self.ANALYSIS_MAX / m
        return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

    def _render_active_scope(self, bgr):
        # Native first: scopes read the file's own signal (PQ/HLG/wide gamut),
        # decoded without tonemap by the NativeTap - NOT the preview frame.
        with self.frame_lock:
            nat = self.current_native
        meta = self.native_meta
        w, h = self._sizes["scope"]
        key = self.active_scope
        if nat is not None and meta:
            frame, t, p = nat[0], meta["transfer"], meta["primaries"]
            if key == "vectorscope":
                return scopes.vectorscope_native(frame, max(120, min(w, h)), t, p)
            if key == "waveform":
                return scopes.waveform_native(frame, w, h, t, p)
            if key == "parade":
                return scopes.rgb_parade_native(frame, w, h, t, p)
            if key == "false":
                return scopes.false_color_native(frame, w, h, t, p)
            if key == "cie":
                img, cov = scopes.cie_gamut_native(frame, max(160, min(w, h)), t, p)
                self.gamut_cov = cov
                return img
            return scopes.histogram_native(frame, w, h, t, p)
        small = self._analysis_frame(bgr)
        if key == "vectorscope":
            img = scopes.vectorscope(small, max(120, min(w, h)))
        elif key == "waveform":
            img = scopes.waveform(small, w, h)
        elif key == "parade":
            img = scopes.rgb_parade(small, w, h)
        elif key == "false":
            img = scopes.false_color(small, w, h)
        elif key == "cie":
            img, cov = scopes.cie_gamut(small, max(160, min(w, h)))
            self.gamut_cov = cov
        else:
            img = scopes.histogram(small, w, h, wide=self.is_wide_gamut,
                                   is_hdr=self.is_hdr)
        if self.is_hdr or self.is_wide_gamut:
            # HDR/WCG file but no native tap (yet): be honest about the source
            scopes.mark_display_referred(img)
        return img

    # --- native scope tap -----------------------------------------------------

    def _stop_tap(self):
        self._tap_epoch += 1
        tap, self.native_tap = self.native_tap, None
        self.native_meta = None
        with self.frame_lock:
            self.current_native = None
        if tap is not None:
            try:
                tap.close()
            except Exception:  # noqa: BLE001
                pass

    def _tap_start(self, path, info, epoch):
        """Build the tap off the UI thread (probe + ffmpeg spawn can be slow)."""
        try:
            tap = NativeTap(path, info)
        except Exception:  # noqa: BLE001 - scopes fall back to display frames
            return
        if epoch != self._tap_epoch or self._closing:
            tap.close()
            return
        self.native_tap = tap
        self.native_meta = tap.meta
        threading.Thread(target=self._tap_loop, args=(tap, epoch), daemon=True).start()

    def _tap_loop(self, tap, epoch):
        """Chase the playhead on the native-signal decode. During play the
        latest decoded frame wins (scopes lag a frame or two at worst, stay
        native); when paused it converges on the exact current frame, so
        stepped/paused scopes are frame-accurate."""
        last_shown = -1
        while epoch == self._tap_epoch and not self._closing:
            tgt = self.cur_index
            pos = tap.index
            if self.playing:
                if pos > tgt + int(tap.fps) or tgt - pos > int(tap.fps) * 4:
                    tap.seek(tgt)            # restarted/far behind: jump
                elif tgt > pos:
                    tap.grab(tgt - pos)      # slightly behind: skip-decode
            else:
                if last_shown == tgt:
                    time.sleep(0.05)
                    continue
                if pos != tgt:
                    tap.seek(tgt)
            frame = tap.read()
            if epoch != self._tap_epoch:
                break
            if frame is None:                # EOF / decode hiccup: one retry
                tap.seek(min(self.cur_index, max(0, (self.total_frames or 1) - 1)))
                frame = tap.read()
            if frame is None:
                last_shown = self.cur_index  # stop retrying until playhead moves
                time.sleep(0.3)
                continue
            shown = tap.index - 1
            with self.frame_lock:
                if epoch != self._tap_epoch:
                    break                    # a new file superseded this tap
                self.current_native = (frame, shown)
            last_shown = shown
            if not self.playing:
                self._post(self._refresh_scope_native, epoch)
            else:
                time.sleep(max(0.0, self.CHART_UPDATE_MS / 2000.0))
        tap.close()

    def _refresh_scope_native(self, epoch):
        """Repaint the active scope once the tap has the paused frame."""
        if (epoch != self._tap_epoch or self._closing or self.playing
                or not self.source):
            return
        with self.frame_lock:
            bgr = self.current_bgr
        if bgr is None:
            return
        self._update_sizes()
        self._apply_payload({"scope": encode_ppm_bytes(self._render_active_scope(bgr))})

    def _update_sizes(self):
        w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if w > 1 and h > 1:
            self._sizes["video"] = (w, h)
        lab = self.scope_labels.get(self.active_scope)
        if lab is not None:
            w, h = lab.winfo_width(), lab.winfo_height()
            if w > 1 and h > 1:
                self._sizes["scope"] = (w, h)

    def _refresh_metric_choices(self):
        """Metric selector = core metrics + any plugin timeline series."""
        vals = [m[0] for m in TIMELINE_METRICS] + sorted(self.plugin_series)
        self.metric_combo.config(values=vals)
        if self.metric_var.get() not in vals:
            self.metric_var.set(TIMELINE_METRICS[0][0])

    def _refresh_preview(self):
        """Repaint the paused frame (after a plugin overlay add/remove)."""
        if self.source and not self.playing:
            try:
                self._render_frame_at(self.cur_index)
            except Exception:  # noqa: BLE001
                pass

    def _apply_overlays(self, bgr):
        """Run plugin preview overlays on a copy of the display frame."""
        out = bgr.copy()
        idx = self.cur_index
        for fn in list(self.plugin_overlays.values()):
            try:
                r = fn(out, idx)
                if r is not None:
                    out = r
            except Exception:  # noqa: BLE001 - a broken overlay must not kill render
                pass
        return out

    def _render_video(self, bgr, box):
        if self.plugin_overlays:
            bgr = self._apply_overlays(bgr)
        bw, bh = box
        sh, sw = bgr.shape[:2]
        scale = min(bw / sw, bh / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)
        canvas = np.empty((bh, bw, 3), np.uint8)
        canvas[:] = PANEL_BG
        y0, x0 = (bh - nh) // 2, (bw - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = rgb
        return canvas

    def _on_scope_change(self, _event=None):
        try:
            idx = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        if idx >= len(SCOPES):
            return
        self.active_scope = SCOPES[idx][1]
        self._update_sizes()
        with self.frame_lock:
            bgr = None if self.current_bgr is None else self.current_bgr.copy()
        if bgr is not None:
            self._apply_payload({"scope": encode_ppm_bytes(self._render_active_scope(bgr))})

    # --- timeline ------------------------------------------------------------

    def _draw_timeline(self):
        c = self.timeline
        c.delete("all")
        self._playhead = None
        w = c.winfo_width()
        h = int(c["height"])
        if w <= 1:
            return
        nb = max(self.total_frames, len(self.table) if self.table else 0, 1)
        psel = self.plugin_series.get(self.metric_var.get())
        if (not self.table or len(self.table) == 0) and not psel:
            c.create_text(10, h // 2, anchor="w", fill=va_theme.WELL["faint"],
                          text="Run Analyze to plot per-frame metrics across the whole file.")
            self._playhead = c.create_line(0, 0, 0, h, fill=va_theme.WELL["play"])
            self._move_playhead(self.cur_index)
            return
        fps = self.fps or 25.0
        for s, e in self.events.get("black", []):
            c.create_rectangle(frame_to_x(int(s * fps), w, nb), 0,
                               frame_to_x(int(e * fps), w, nb), h,
                               fill=va_theme.WELL["ev_black"], outline="", stipple="gray50")
        for s, e in self.events.get("freeze", []):
            c.create_rectangle(frame_to_x(int(s * fps), w, nb), 0,
                               frame_to_x(int(e * fps), w, nb), h,
                               fill=va_theme.WELL["ev_freeze"], outline="", stipple="gray25")
        if psel:
            # plugin series carry explicit timestamps: place each point at
            # its container-frame x; vertical scale from the series itself
            key = self.metric_var.get()
            series = list(psel["v"])
            tt = psel["t"]
            vals = [v for v in series if v is not None and v == v]
            pts = []
            if vals and len(tt) == len(series):
                lo = psel.get("vmin")
                hi = psel.get("vmax")
                lo = min(vals) if lo is None else lo
                hi = max(vals) if hi is None else hi
                rng = (hi - lo) or 1.0
                stride = max(1, len(series) // max(1, w))
                for i in range(0, len(series), stride):
                    v = series[i]
                    if v is None or v != v:
                        continue
                    x = frame_to_x(int(round(float(tt[i]) * fps)), w, nb)
                    y = int(4 + (h - 8) * (1.0 - (min(max(v, lo), hi) - lo) / rng))
                    pts.append((x, y))
        else:
            key = dict(TIMELINE_METRICS).get(self.metric_var.get(), "YAVG")
            series = list(self.table.arrays().get(key, []))
            peak = (1 << (self.source.info.get("bit_depth", 8) if self.source else 8)) - 1
            vmin = 0 if key in ("YAVG", "YMAX", "SATAVG", "SATMAX") else None
            vmax = float(peak) if key in ("YAVG", "YMAX") else None
            # x positions: table rows are decoded frames; the canvas (events,
            # clicks, playhead) lives in container-frame domain - map between them.
            n_tbl = len(series)
            pts = metric_polyline(series, w, h, vmin=vmin, vmax=vmax)
            if pts and n_tbl > 1 and nb > 1 and n_tbl != nb:
                sx = (nb - 1) / (n_tbl - 1)
                pts = [(frame_to_x(int(round(x_to_frame(x, w, n_tbl) * sx)), w, nb), y)
                       for x, y in pts]
        if len(pts) >= 2:
            flat = [coord for p in pts for coord in p]
            c.create_line(*flat, fill=va_theme.WELL["line"], width=1)
        for cut in self.events.get("scene_cuts", []):
            x = frame_to_x(int(cut * fps), w, nb)
            c.create_line(x, 0, x, h, fill=va_theme.WELL["cut"])
        for t, _lab in self.forensic_marks + self.scan_marks:
            x = frame_to_x(int(t * fps), w, nb)
            c.create_line(x, 0, x, h, dash=(3, 2),
                          fill=va_theme.WELL.get("forensic", "#b07fd8"))
        vals = [v for v in series if v is not None and v == v]
        if vals:
            c.create_text(6, 8, anchor="nw", fill=va_theme.WELL["text"],
                          text="%s  max %.1f" % (key, max(vals)))
            c.create_text(6, h - 8, anchor="sw", fill=va_theme.WELL["text"],
                          text="min %.1f" % min(vals))
        self._playhead = c.create_line(0, 0, 0, h, fill=va_theme.WELL["play"])
        self._move_playhead(self.cur_index)

    def _move_playhead(self, pos):
        if getattr(self, "_playhead", None) is None:
            return
        c = self.timeline
        w = c.winfo_width()
        nb = max(self.total_frames, len(self.table) if self.table else 0, 1)
        x = frame_to_x(pos, w, nb)
        try:
            c.coords(self._playhead, x, 0, x, int(c["height"]))
        except tk.TclError:
            pass
        if self.fps:
            self.audio_strip.set_playhead(pos / self.fps)

    def _on_timeline_click(self, event):
        if not self.source:
            return
        w = self.timeline.winfo_width()
        nb = max(self.total_frames, len(self.table) if self.table else 0, 1)
        frame = x_to_frame(event.x, w, nb)
        self.seeking = True
        self.seek_var.set(frame)
        self.seeking = False
        if self.playing:
            self._seek_request = frame
        else:
            self._render_frame_at(frame)
        self._update_time_label(frame)
        self._move_playhead(frame)

    # --- export & compare ----------------------------------------------------

    def _export(self, fmt: str):
        if not self.table:
            messagebox.showinfo("Export", "Analyze the file first.")
            return
        base = os.path.splitext(os.path.basename(self.path or "analysis"))[0]
        ext = {"csv": ".csv", "json": ".json", "html": ".html"}[fmt]
        path = filedialog.asksaveasfilename(defaultextension=ext, initialfile=base + ext,
                                            filetypes=[(fmt.upper(), "*" + ext)])
        if not path:
            return
        try:
            src = dict(self.source.info) if self.source else {}
            if fmt == "csv":
                export.write_csv(self.table, path)
            elif fmt == "json":
                export.write_json(self.table, path, source=src,
                                  events=self.events, loudness=self.loudness,
                                  advanced=self.adv_results)
            else:
                export.write_html_report(self.table, path, source=src, events=self.events,
                                         loudness=self.loudness,
                                         quality=getattr(self, "_last_quality", None),
                                         gamut=self.gamut_cov,
                                         forensics=(self.adv_results.get("forensics") or {}).get("data"),
                                         advanced=self.adv_results)
            n_adv = len(self.adv_results)
            self.lbl_status.config(
                text="Saved %s%s" % (os.path.basename(path),
                                     " (incl. %d advanced result%s)" % (n_adv, "s" if n_adv != 1 else "")
                                     if n_adv and fmt != "csv" else ""),
                foreground=clr("ok"))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))

    def _compare(self):
        if not self.path:
            return
        ref = filedialog.askopenfilename(title="Select the reference / master video")
        if not ref:
            return
        self.btn_compare.config(state=tk.DISABLED)
        self.lbl_status.config(text="Comparing (decodes both files)...", foreground=clr("warn"))
        self._run_bg(self._compare_worker, self.path, ref)

    def _compare_worker(self, distorted, reference):
        res = quality.compare(distorted, reference, vmaf=quality.vmaf_available())
        self._post(self._compare_done, res)

    def _compare_done(self, res):
        self._last_quality = res
        self.btn_compare.config(state=tk.NORMAL)
        self.lbl_status.config(text="Comparison done", foreground=clr("ok"))
        lines = ["%s  vs  %s" % (res["distorted"], res["reference"]), ""]
        for m in ("psnr", "ssim", "xpsnr", "vmaf"):
            d = res.get(m)
            if d and d.get("average") is not None:
                lines.append("%-5s  %.3f" % (m.upper(), d["average"]))
        vm = res.get("vmaf") or {}
        if vm.get("model"):
            lines.append("VMAF model: %s" % vm["model"])
        for m in ("psnr", "ssim", "xpsnr", "vmaf"):
            d = res.get(m)
            if d and d.get("error"):
                lines.append("%-5s  could not measure: %s" % (m.upper(), d["error"]))
        notes = list(res.get("notes") or [])
        for m in ("psnr", "ssim", "xpsnr", "vmaf"):
            d = res.get(m)
            if d and d.get("note") and d["note"] not in notes:
                notes.append(d["note"])
        for n in notes:
            lines.append("note: %s" % n)
        if not res.get("vmaf_available"):
            lines.append("(VMAF needs an ffmpeg built with libvmaf)")
        if not res.get("xpsnr_available"):
            lines.append("(XPSNR needs ffmpeg 7.1 or newer)")
        messagebox.showinfo("Reference quality", "\n".join(lines))
        self.readout.config(text=self.readout.cget("text") + "     " +
                            "  ".join("%s %.2f" % (m.upper(), res[m]["average"])
                                      for m in ("psnr", "ssim", "xpsnr", "vmaf")
                                      if res.get(m) and res[m].get("average") is not None))

    # --- resize & cleanup ----------------------------------------------------

    def _on_configure(self, _event):
        self._update_sizes()
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self._rerender_static)

    def _rerender_static(self):
        self._resize_job = None
        self._draw_timeline()
        if self.playing:
            return
        with self.frame_lock:
            bgr = self.current_bgr
        if bgr is None:
            return
        self._update_sizes()
        self._apply_payload(self._render_all(bgr, True))

    def _on_close(self):
        self._stop_tap()
        self._closing = True
        self.playing = False
        if self.follower:
            self.follower.stop()
        self._analyze_cancel = True
        try:
            metrics.kill_all()           # no orphaned ffmpeg passes after close
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
        with self.source_lock:
            if self.source:
                self.source.close()
        for pat in ("video_analyzer_%s_video.ppm" % self._uid,
                    "video_analyzer_%s_scope.ppm" % self._uid,
                    "video_analyzer_%s_spectro.ppm" % self._uid,
                    "video_analyzer_%s_loud.ppm" % self._uid,
                    "va_spectro_%s.png" % self._uid,
                    "va_popup_%s.ppm" % self._uid,
                    "va_audition_%s.wav" % self._uid):
            try:
                os.remove(os.path.join(self._tmpdir, pat))
            except OSError:
                pass
        self.root.destroy()

    def _on_drop(self, event):
        for path in parse_dnd_paths(getattr(event, "data", "")):
            if os.path.isfile(path):
                self._load_video(path)
                break

    def _jump_issue(self, direction):
        if not self.issues:
            return
        frames = [f for f, _ in self.issues]
        cur = self.cur_index
        if direction > 0:
            target = next((f for f in frames if f > cur), frames[0])
        else:
            earlier = [f for f in frames if f < cur]
            target = earlier[-1] if earlier else frames[-1]
        i = frames.index(target)
        label = dict(self.issues).get(target, "issue")
        self.seeking = True
        self.seek_var.set(target)
        self.seeking = False
        if self.playing:
            self._seek_request = target
        else:
            self._render_frame_at(target)
        self._update_time_label(target)
        self._move_playhead(target)
        secs = target / (self.fps or 25.0)
        self.lbl_status.config(
            text="Issue %d/%d: %s @ %s" % (i + 1, len(frames), label,
                                           time.strftime("%H:%M:%S", time.gmtime(secs))),
            foreground=clr("warn"))


    def _analysis_failed(self, msg):
        self.btn_analyze.config(state=tk.NORMAL)
        self.lbl_analysis.config(text="analysis failed")
        self.lbl_status.config(text="Analyze error: " + msg, foreground=clr("err"))


    def _set_audio(self):
        a = self.audio
        if not a:
            reason = va_audio.last_error()
            self.audio_stats.config(text={
                "timeout": "Audio analysis timed out - try again or use a shorter clip.",
                "cancelled": "Audio analysis was cancelled.",
                "ffmpeg failed": "Audio analysis failed (ffmpeg error).",
                "no ffmpeg": "Audio analysis needs ffmpeg.",
            }.get(reason or "", "No audio track in this file."))
            return
        ld = a.get("loudness")
        summ = ld.get("summary") if ld else None
        lines = []
        if summ:
            lines.append("Integrated %s LUFS    LRA %s LU    True peak %s dBFS" % (
                summ.get("integrated_lufs"), summ.get("lra_lu"), summ.get("true_peak_dbfs")))
        st = a.get("astats")
        if st and st.get("channels"):
            lines.append("   ".join("ch%s  peak %s / rms %s dB" % (
                c.get("channel"), c.get("peak_db"), c.get("rms_db")) for c in st["channels"]))
        lines.append("Silence: %d segment(s)     Stereo correlation: %s" % (
            len(a.get("silence") or []), a.get("correlation")))
        self.audio_stats.config(text="\n".join(lines))
        spec = a.get("spectro")
        if spec and os.path.isfile(spec):
            bgr = cv2.imread(spec)
            if bgr is not None:
                img = self._photo_from_ppm(encode_ppm_bytes(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), "spectro")
                if img is not None:
                    self.audio_spectro.config(image=img)
                    self._photo_spectro = img
        if ld:
            chart = va_audio.render_loudness(ld, 560, 120)
            img = self._photo_from_ppm(encode_ppm_bytes(chart), "loud")
            if img is not None:
                self.audio_loud.config(image=img)
                self._photo_loud = img



    def _run_bg(self, target, *args):
        """Run a worker in a thread; surface any exception to the status bar instead
        of letting the daemon thread die silently."""
        def runner():
            try:
                target(*args)
            except Exception as exc:  # noqa: BLE001
                self._post(lambda e=exc: self.lbl_status.config(
                    text="Error: %s" % e, foreground=clr("err")))
        threading.Thread(target=runner, daemon=True).start()

    def _open_ab_viewer(self):
        if not self.path:
            return
        ref = filedialog.askopenfilename(title="Select the other video (B) to compare")
        if not ref:
            return
        try:
            ABViewer(self.root, self.path, ref)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("A/B viewer", str(exc))

    LADDER_CODECS = {"h264": "H.264 (libx264)", "hevc": "HEVC (libx265)",
                     "av1": "AV1 (SVT-AV1)"}

    def _encode_ladder(self, codec="h264"):
        if not self.path:
            return
        if not va_compare.encoder_available(codec):
            messagebox.showinfo("Encode ladder",
                                "%s is not available in this ffmpeg build."
                                % self.LADDER_CODECS.get(codec, codec))
            return
        self.btn_compare.config(state=tk.DISABLED)
        self.lbl_status.config(text="Encoding ladder - %s (transcodes several times)..."
                               % self.LADDER_CODECS.get(codec, codec), foreground=clr("warn"))
        self._run_bg(self._ladder_worker, self.path, codec)

    def _ladder_worker(self, path, codec="h264"):
        res = va_compare.encode_ladder(path, [250, 500, 1000, 2000, 4000, 8000], codec=codec)
        self._post(self._ladder_done, res, va_compare.optimal_bitrate(res), codec)

    def _ladder_done(self, res, opt, codec="h264"):
        self.btn_compare.config(state=tk.NORMAL)
        label = self.LADDER_CODECS.get(codec, codec)
        if not res:
            why = va_compare.last_error() or "no rungs could be encoded"
            self.lbl_status.config(text="Encode ladder: %s" % why, foreground=clr("warn"))
            messagebox.showwarning("Encode ladder - %s" % label, why)
            return
        self.lbl_status.config(text="Encode ladder done (%s)" % label, foreground=clr("ok"))
        lines = ["%-9s %-7s %-7s %-6s" % ("bitrate", "PSNR", "SSIM", "VMAF")]
        for r in res:
            lines.append("%-9s %-7s %-7s %-6s" % (
                "%dk" % r["bitrate_kbps"],
                ("%.2f" % r["psnr"]) if r["psnr"] not in (None, float("inf")) else "-",
                ("%.4f" % r["ssim"]) if r["ssim"] is not None else "-",
                ("%.1f" % r["vmaf"]) if r["vmaf"] is not None else "-"))
        if opt:
            lines += ["", "Suggested bitrate: %dk  (%s %.3f)" % (
                opt["bitrate_kbps"], opt["metric"], opt["value"])]
        messagebox.showinfo("Encode ladder - %s" % label, "\n".join(lines))


    def _adv_store(self, key, title, text=None, data=None, img=None):
        """Cache an Advanced-tool result for export. JSON gets text+data; the
        HTML report additionally embeds img (RGB array) as a PNG."""
        png = None
        if img is not None:
            try:
                small = img
                h, w = small.shape[:2]
                if w > 1400:
                    sc = 1400.0 / w
                    small = cv2.resize(small, (1400, max(1, int(h * sc))),
                                       interpolation=cv2.INTER_AREA)
                okf, buf = cv2.imencode(".png", small[:, :, ::-1])
                png = buf.tobytes() if okf else None
            except Exception:  # noqa: BLE001 - cache must never break the tool
                png = None
        self.adv_results[key] = {"title": title,
                                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "text": text, "data": data, "png": png}

    def _image_popup(self, title, rgb):
        if rgb is None:
            messagebox.showinfo(title, "Nothing to show (no frame, or not applicable).")
            return
        h, w = rgb.shape[:2]
        if w > 1500 or h > 900:                # keep popups on-screen
            s = min(1500.0 / w, 900.0 / h)
            rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=va_theme.WELL["panel"])
        lab = tk.Label(win, bg=va_theme.WELL["bg"])
        lab.pack(fill=tk.BOTH, expand=True)
        try:
            path = os.path.join(self._tmpdir, "va_popup_%s.ppm" % self._uid)
            with open(path, "wb") as fh:
                fh.write(encode_ppm_bytes(rgb))
            img = tk.PhotoImage(file=path)
            lab.config(image=img)
            lab._img = img
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(title, str(exc))

    def _show_banding(self):
        with self.frame_lock:
            bgr = None if self.current_bgr is None else self.current_bgr.copy()
        if bgr is None:
            return
        heat, pct = va_perceptual.banding_map(bgr)
        vis = va_perceptual.colorize(heat)
        self._adv_store("banding_map", "Banding map (frame %d) - %.1f%% banding"
                        % (self.cur_index, pct),
                        data={"frame": self.cur_index,
                              "banding_pct": round(float(pct), 2)}, img=vis)
        self._image_popup("Banding map - %.1f%% banding" % pct, vis)

    def _show_hdr_multi(self):
        if not self.path or not self.is_pq:
            msg = ("This preview applies to PQ (HDR10 / Dolby Vision) files."
                   + ("\n\nThis file is HLG - PQ light-level math does not apply."
                      if self.is_hdr else ""))
            messagebox.showinfo("HDR multi-display", msg)
            return
        self.lbl_status.config(text="Rendering HDR multi-display...", foreground=clr("warn"))
        self._run_bg(self._hdr_multi_worker, self.path, self.cur_index)

    def _hdr_multi_worker(self, path, idx):
        try:
            mont, stats = va_hdr.multi_display(path, idx)
        except Exception:  # noqa: BLE001
            mont, stats = None, None

        def show():
            self.lbl_status.config(text="HDR multi-display", foreground=clr("ok"))
            if mont is not None:
                self._adv_store("hdr_multi_display",
                                "HDR multi-display preview (frame %d)" % idx,
                                data=stats, img=mont)
            self._image_popup("HDR multi-display (SDR / 400 / 1000 / 4000 nits)", mont)
        self._post(show)

    def _run_qc(self, profile="general"):
        if not self.path:
            return
        self.lbl_status.config(text="Running QC checks (%s profile)..." % profile,
                               foreground=clr("warn"))
        self._run_bg(self._qc_worker, self.path, profile)

    def _qc_worker(self, path, profile="general"):
        with self.frame_lock:
            bgr = None if self.current_bgr is None else self.current_bgr.copy()
        banding = va_perceptual.banding_map(bgr)[1] if bgr is not None else None
        loud = None
        if self.audio and self.audio.get("loudness"):
            loud = self.audio["loudness"].get("summary")
        loud = loud or self.loudness
        vs_cad, vs_pse = VideoSource(path), VideoSource(path)
        try:
            cad = va_temporal.cadence(vs_cad)
            pse = va_temporal.pse_flashes(vs_pse)
        finally:
            vs_cad.close()
            vs_pse.close()
        ctx = {
            "probe": dict(self.source.info) if self.source else {},
            "summary": self.table.summary() if self.table else {},
            "loudness": loud, "events": self.events or {}, "gamut": self.gamut_cov,
            "silence": (self.audio or {}).get("silence", []) if self.audio else [],
            "banding": banding, "cadence": cad,
            "pse": pse,
            "hdr_cll": va_hdr.maxcll_maxfall(path) if self.is_pq else None,
            # cached forensics / verify-run results feed the integrity and
            # speedrun profiles; missing passes degrade to "not measured"
            "forensics": (self.adv_results.get("forensics") or {}).get("data"),
            "verify": (self.adv_results.get("verify") or {}).get("data"),
            # every cached tool result, for plugin QC checks (ctx["adv"][key])
            "adv": {k: (v or {}).get("data")
                    for k, v in self.adv_results.items()},
        }
        self._post(self._qc_done, va_qc.evaluate(ctx, profile))

    def _qc_done(self, res):
        self.lbl_status.config(text="QC verdict: %s" % res["verdict"].upper(),
                               foreground=clr("ok") if res["verdict"] == "pass" else clr("warn"))
        lines = ["Profile: %s    Verdict: %s" % (res["profile"], res["verdict"].upper()), ""]
        for c in res["checks"]:
            lines.append("[%s] %-22s %s" % (c["status"].upper(), c["label"], c["detail"]))
        self._adv_store("qc", "QC check (%s profile) - %s" % (res["profile"], res["verdict"].upper()),
                        text="\n".join(lines), data=res)
        messagebox.showinfo("QC report", "\n".join(lines))


    def _hdr_meta_report(self):
        if not self.path:
            return
        meta = va_dynhdr.inspect(self.path)
        lines = va_dynhdr.summary_lines(meta)
        flags = meta.get("flags") or []
        if flags:
            lines += [""] + ["[%s] %s" % (sv.upper(), m) for sv, m in flags]
        self._adv_store("hdr_metadata", "HDR metadata report (DV / HDR10+)",
                        text="\n".join(lines))
        messagebox.showinfo("HDR metadata", "\n".join(lines))

    def _dynhdr_verify(self):
        if not self.path:
            return
        self.lbl_status.config(text="Checking dynamic metadata vs content...", foreground=clr("warn"))
        self._run_bg(self._dynhdr_worker, self.path)

    def _dynhdr_worker(self, path):
        self._post(self._dynhdr_done, va_dynhdr.verify_metadata(path))

    def _dynhdr_done(self, res):
        self.lbl_status.config(text="Dynamic metadata check done", foreground=clr("ok"))
        timeline = va_dynhdr.render_dynamic_timeline(res)
        self._image_popup("Peak nits per scene - measured vs declared", timeline)
        lines = ["Dynamic metadata present: %s (%s)" % (res["has_dynamic"], res.get("source"))]
        for r in res["scenes"][:12]:
            lines.append("scene %s @%.1fs  measured max %s / avg %s  declared %s  - %s" % (
                r["scene"], r["t"], r["measured_max"], r["measured_avg"], r["declared_max"], r["status"]))
        if res.get("static") and res["static"].get("notes"):
            lines += [""] + res["static"]["notes"]
        self._adv_store("dynamic_vs_content", "Dynamic metadata vs content",
                        text="\n".join(lines), data=res, img=timeline)
        messagebox.showinfo("Dynamic metadata vs content", "\n".join(lines))

    def _dynamic_vs_static(self):
        if not self.path:
            return
        self.lbl_status.config(text="Rendering dynamic vs static (libplacebo)...", foreground=clr("warn"))
        self._run_bg(self._dvs_worker, self.path, self.cur_index)

    def _dvs_worker(self, path, idx):
        img, msg = va_dynhdr.dynamic_vs_static(path, idx)

        def show():
            self.lbl_status.config(text=msg, foreground=clr("ok") if img is not None else clr("warn"))
            if img is not None:
                self._adv_store("dynamic_vs_static", "Dynamic vs static tonemap A/B",
                                data={"note": msg}, img=img)
                self._image_popup("Dynamic vs static tonemap", img)
            else:
                messagebox.showinfo("Dynamic vs static", msg)
        self._post(show)


    def _build_tools_window(self):
        """The Tools dialog (ffmpeg capabilities + DV/HDR10+ helper downloads).

        Built once and hidden on close (never destroyed), so download progress
        callbacks always have live widgets to write to."""
        win = tk.Toplevel(self.root)
        win.title("Tools")
        win.transient(self.root)
        win.resizable(True, False)
        win.minsize(680, 1)
        win.withdraw()
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.configure(bg=clr("bg"))            # Toplevels do not follow ttk styles
        self.setup_caps = tk.Label(win, justify=tk.LEFT, anchor="w", bg=clr("editor_bg"),
                                   fg=clr("editor_fg"), font=self.mono_font,
                                   padx=10, pady=8)
        self.setup_caps.pack(fill=tk.X, padx=12, pady=(12, 10))
        prow = ttk.Frame(win)
        prow.pack(fill=tk.X, padx=12, pady=(0, 4))
        ttk.Label(prow, text="PERFORMANCE  ·  how hard the engine drives this machine",
                  style="Caption.TLabel").pack(side=tk.LEFT)
        self.perf_var = tk.StringVar(value=self.perf_mode)
        pcb = ttk.Combobox(prow, textvariable=self.perf_var, state="readonly",
                           width=9, values=list(va_perf.MODES))
        pcb.pack(side=tk.RIGHT)
        pcb.bind("<<ComboboxSelected>>", self._set_perf_mode)
        ttk.Label(prow, text="mode:").pack(side=tk.RIGHT, padx=(0, 6))
        self.setup_perf = tk.Label(win, justify=tk.LEFT, anchor="w", bg=clr("editor_bg"),
                                   fg=clr("editor_fg"), font=self.mono_font,
                                   padx=10, pady=8)
        self.setup_perf.pack(fill=tk.X, padx=12, pady=(0, 10))
        ttk.Label(win, text="OPTIONAL HELPER TOOLS  ·  downloaded on demand, installed beside the app",
                  style="Caption.TLabel").pack(anchor=tk.W, padx=12, pady=(2, 4))
        self._tool_rows = {}
        self._tool_descs = []
        for i, (tname, desc) in enumerate((
                ("ffmpeg", "decode + scopes engine; full GPL build with libvmaf & libplacebo (BtbN, ~150 MB)"),
                ("dovi_tool", "Dolby Vision RPU extract / plot (quietvoid)"),
                ("hdr10plus_tool", "HDR10+ metadata extract / plot (quietvoid)"),
                ("mediainfo", "deep container & stream metadata report (MediaArea)"),
                ("mkvextract", "MKV track extraction + mkvinfo/mkvmerge container reports (MKVToolNix, Windows)"),
                ("c2patool", "Content Credentials (C2PA) manifest validation (contentauth)"),
                ("mp4dump", "MP4 box inspector - verifies DV dvcC/dvvC signaling (Bento4)"))):
            if i:
                ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=(7, 0))
            row = ttk.Frame(win)
            row.pack(fill=tk.X, padx=12, pady=(7, 0))
            ttk.Label(row, text=tname, width=16, font=self.mono_font).pack(side=tk.LEFT)
            st = ttk.Label(row, text="", foreground=clr("muted"))
            st.pack(side=tk.LEFT, padx=8)
            btn = ttk.Button(row, text="Download", width=13,
                             command=lambda n=tname: self._download_tool(n))
            btn.pack(side=tk.RIGHT)
            self._tool_rows[tname] = (st, btn)
            d = ttk.Label(win, text=desc, foreground=clr("faint"))
            d.pack(anchor=tk.W, padx=(12, 12), pady=(1, 0))
            self._tool_descs.append(d)
        ttk.Button(win, text="Tidy into tools/", command=self._tidy_tools).pack(
            side=tk.LEFT, padx=(8, 0))
        ttk.Button(win, text="Refresh", command=self._refresh_setup).pack(
            anchor=tk.W, padx=12, pady=(14, 12))
        self.tools_win = win

    def _tidy_tools(self):
        """Move loose tool executables from the script folder into tools/."""
        import va_tools
        moved = va_tools.tidy_tool_folder()
        self.lbl_status.config(
            text=("moved into tools/: " + ", ".join(moved)) if moved
            else "tool folder already tidy", foreground=clr("ok"))
        self._refresh_setup()

    def _open_tools(self):
        self._refresh_setup()
        self.tools_win.deiconify()
        self.tools_win.lift()
        self.tools_win.focus_set()

    def _skin_info_text(self):
        self.info_text.config(bg=clr("editor_bg"), fg=clr("editor_fg"),
                              insertbackground=clr("editor_fg"))
        for tag, key in (("summary_title", "accent"), ("header", "accent"),
                         ("key", "info"), ("value", "ok"), ("dim", "muted")):
            self.info_text.tag_config(tag, foreground=clr(key))

    def _toggle_theme(self):
        self._apply_theme("light" if self.theme_mode == "dark" else "dark")

    def _apply_theme(self, mode):
        """Re-theme the live UI. Instrument wells (video/scopes/timeline) stay dark."""
        self.theme_mode = mode
        va_theme.apply(self.root, mode)
        va_theme.save_pref(mode)
        for m in getattr(self, "_menus", ()):
            va_theme.style_menu(m)
        self._skin_info_text()
        self.tools_win.configure(bg=clr("bg"))
        self.setup_caps.config(bg=clr("editor_bg"), fg=clr("editor_fg"))
        self.setup_perf.config(bg=clr("editor_bg"), fg=clr("editor_fg"))
        for d in getattr(self, "_tool_descs", ()):
            d.config(foreground=clr("faint"))
        self.readout.config(foreground=clr("muted"))
        self.lbl_analysis.config(foreground=clr("muted"))
        self.lbl_status.config(foreground=clr("muted"))
        self._update_hdr_badge()
        self._refresh_setup()
        self._draw_timeline()

    def _text_popup(self, title, text):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("860x620")
        win.configure(bg=clr("editor_bg"))
        st = scrolledtext.ScrolledText(win, wrap=tk.NONE, font=self.mono_font,
                                       bg=clr("editor_bg"), fg=clr("editor_fg"),
                                       insertbackground=clr("editor_fg"))
        st.pack(fill=tk.BOTH, expand=True)
        st.insert("1.0", text)
        st.config(state=tk.DISABLED)

    def _tool_feature(self, worker, busy_msg):
        if not self.path:
            return
        self.lbl_status.config(text=busy_msg, foreground=clr("warn"))
        self._run_bg(worker, self.path)

    def _plot_dovi(self):
        self._tool_feature(self._plot_dovi_worker, "Extracting DV RPU and plotting (dovi_tool)...")

    def _plot_dovi_worker(self, path):
        img, msg = va_dynhdr.plot_dovi_l1(path)
        self._post(lambda: self._tool_feature_done(
            "DV L1 brightness plot (dovi_tool)", img, None, msg,
            key="dovi_l1_plot", data={"note": msg}))

    def _plot_h10p(self):
        self._tool_feature(self._plot_h10p_worker, "Extracting HDR10+ metadata and plotting...")

    def _plot_h10p_worker(self, path):
        img, msg = va_dynhdr.plot_hdr10plus(path)
        self._post(lambda: self._tool_feature_done(
            "HDR10+ brightness plot (hdr10plus_tool)", img, None, msg,
            key="hdr10plus_plot", data={"note": msg}))

    def _mediainfo_report(self):
        self._tool_feature(self._mediainfo_worker, "Running MediaInfo...")

    def _mediainfo_worker(self, path):
        text, msg = va_dynhdr.mediainfo_report(path)
        self._post(lambda: self._tool_feature_done(
            "MediaInfo - %s" % os.path.basename(path), None, text, msg,
            key="mediainfo"))

    def _mkv_boxes(self):
        self._tool_feature(self._mkv_boxes_worker, "Reading MKV structure (mkvinfo)...")

    def _mkv_boxes_worker(self, path):
        text, msg = va_dynhdr.mkv_report(path)
        self._post(lambda: self._tool_feature_done(
            "MKV structure / DV signaling - %s" % os.path.basename(path), None, text, msg,
            key="mkv_structure"))

    def _c2pa_report(self):
        self._tool_feature(self._c2pa_worker, "Validating Content Credentials (c2patool)...")

    def _c2pa_worker(self, path):
        text, msg = va_forensics.content_credentials_report(path)
        self._post(lambda: self._tool_feature_done(
            "Content Credentials (C2PA) - %s" % os.path.basename(path), None, text, msg,
            key="content_credentials"))

    def _forensics_report(self):
        if not self.path:
            return
        self.lbl_status.config(text="Forensics battery (hash / splice / container / "
                                    "noise / loops / ENF)...", foreground=clr("warn"))
        self._run_bg(self._forensics_worker, self.path)

    def _forensics_worker(self, path):
        rep = va_forensics.forensics_report(
            path, on_progress=lambda m: self._post(
                lambda m=m: self.lbl_status.config(text="Forensics: %s..." % m,
                                                   foreground=clr("warn"))))
        self._post(self._forensics_done, path, rep)

    def _forensics_done(self, path, rep):
        if path != self.path or self._closing:
            return
        self.forensic_marks = va_forensics.report_marks(rep)
        self.issues = issue_frames(self.events, self.fps,
                                   extra=self.forensic_marks + self.audio_marks + self.scan_marks)
        st = tk.NORMAL if self.issues else tk.DISABLED
        self.btn_prev_issue.config(state=st)
        self.btn_next_issue.config(state=st)
        self._draw_timeline()
        warns = sum(1 for f in rep.get("findings", []) if f.get("severity") == "warn")
        self.lbl_status.config(
            text="Forensics: %s" % rep.get("summary", "done"),
            foreground=clr("warn") if warns else clr("ok"))
        self._adv_store("forensics", "Forensics / integrity report",
                        text=va_forensics.render_report(rep), data=rep)
        au = rep.get("audio") or {}
        if au.get("present"):
            self._adv_store("audio_forensics", "Audio forensics battery",
                            text=va_audio.render_battery(au), data=au)
            self.audio_strip.set_marks(
                [(float(f["t"]), f.get("severity", "info"))
                 for f in rep.get("findings", [])
                 if f.get("area") == "audio" and f.get("t") is not None])
        self._text_popup("Forensics / integrity - %s" % os.path.basename(path),
                         va_forensics.render_report(rep))

    # --- Plugin support: playhead jumps, dialog tracking, scan marks ------

    def _goto_frame(self, idx):
        """Jump the playhead to an exact frame (used by the retimer)."""
        if not self.source or not self.total_frames:
            return
        tgt = int(max(0, min(self.total_frames - 1, idx)))
        if self.playing:
            self._seek_request = tgt
            return
        self.seeking = True
        self.seek_var.set(tgt)
        self.seeking = False
        self._scrub_target = tgt
        if self._scrub_job is None:
            self._scrub_job = self.root.after(15, self._scrub_render)
        self._update_time_label(tgt)
        self._move_playhead(tgt)

    def _lift_dialog(self, attr, maker):
        dlg = getattr(self, attr, None)
        if dlg is not None and dlg.win.winfo_exists():
            dlg.win.lift()
            return
        setattr(self, attr, maker(self))

    def _open_plugins(self):
        self._lift_dialog("_plugins_dlg", va_plugins_ui.PluginManagerDialog)

    def _refresh_scan_issues(self):
        """Re-fold scan marks into the n/p issue list and the timeline."""
        self.issues = issue_frames(self.events, self.fps,
                                   extra=self.forensic_marks
                                   + self.audio_marks + self.scan_marks)
        st = tk.NORMAL if self.issues else tk.DISABLED
        self.btn_prev_issue.config(state=st)
        self.btn_next_issue.config(state=st)
        self._draw_timeline()

    def _show_ela(self):
        with self.frame_lock:
            bgr = None if self.current_bgr is None else self.current_bgr.copy()
        if bgr is None:
            return
        heat, score = va_forensics.ela_map(bgr)
        if heat is None:
            messagebox.showinfo("ELA residual map", "Could not compute an ELA map for this frame.")
            return
        vis = va_perceptual.colorize(heat)
        self._adv_store("ela_map", "ELA residual map (frame %d) - mean %.1f"
                        % (self.cur_index, score),
                        data={"frame": self.cur_index, "ela_mean": round(float(score), 2)},
                        img=vis)
        self._image_popup("ELA residual - mean %.1f  (uniform = one compression "
                          "history; hot patches differ)" % score, vis)

    def _show_noise(self):
        with self.frame_lock:
            bgr = None if self.current_bgr is None else self.current_bgr.copy()
        if bgr is None:
            return
        nmap, stats = va_forensics.noise_map(bgr)
        if nmap is None:
            messagebox.showinfo("Noise map", "Frame too small for block noise analysis.")
            return
        extra = "  -  " + stats["note"] if stats.get("note") else ""
        vis = va_perceptual.colorize(nmap)
        self._adv_store("noise_map", "Noise consistency map (frame %d)" % self.cur_index,
                        data=dict(stats, frame=self.cur_index), img=vis)
        self._image_popup("Noise consistency - sigma %.2f, p95 deviation %.2f%s  "
                          "(hot = noise unlike the rest of the frame)" % (
                              stats.get("sigma_median", 0), stats.get("deviation_p95", 0), extra),
                          vis)

    def _show_enf(self):
        self._tool_feature(self._enf_worker, "Tracing mains hum (ENF)...")

    def _enf_worker(self, path):
        res = va_forensics.enf_trace(path)
        img = va_forensics.render_enf(res) if res.get("present") else None
        self._post(lambda: self._tool_feature_done(
            "ENF mains-hum trace - %s" % os.path.basename(path), img, None,
            res.get("note", "no analysable audio"), key="enf",
            data={k: res.get(k) for k in ("present", "base_hz", "jumps", "note")}))

    def _mp4_boxes(self):
        self._tool_feature(self._mp4_boxes_worker, "Dumping MP4 boxes (mp4dump)...")

    def _mp4_boxes_worker(self, path):
        text, msg = va_dynhdr.mp4_report(path)
        self._post(lambda: self._tool_feature_done(
            "MP4 boxes / DV signaling - %s" % os.path.basename(path), None, text, msg,
            key="mp4_boxes"))

    def _tool_feature_done(self, title, img, text, msg, key=None, data=None):
        ok = img is not None or text is not None
        if ok and key:
            self._adv_store(key, title, text=text, data=data, img=img)
        self.lbl_status.config(text=title if ok else msg,
                               foreground=clr("ok") if ok else clr("warn"))
        if img is not None:
            self._image_popup(title, img)
        elif text is not None:
            self._text_popup(title, text)
        else:
            messagebox.showinfo(title, msg)

    def _refresh_setup(self):
        import va_ffmpeg
        try:
            va_ffmpeg.reset_tool_caches()      # tool paths + filter capabilities
        except AttributeError:
            va_ffmpeg.find_tool.cache_clear()
        try:
            import va_hwaccel
            va_hwaccel.hwaccels.cache_clear()  # a new ffmpeg build changes these
            va_hwaccel.reset_cache()           # forget verified decode recipes
            hw = ", ".join(sorted(va_hwaccel.hwaccels())) or "none"
        except Exception:  # noqa: BLE001
            hw = "?"
        ff = va_ffmpeg.find_ffmpeg()
        fp = va_ffmpeg.find_ffprobe()
        self.setup_caps.config(text=chr(10).join([
            "ffmpeg : %s" % (ff or "NOT FOUND  (put ffmpeg(.exe) beside the app or on PATH)"),
            "ffprobe: %s" % (fp or "NOT FOUND"),
            "libvmaf: %s     libplacebo: %s" % (
                "yes" if va_ffmpeg.has_filter("libvmaf") else "no",
                "yes" if va_ffmpeg.has_filter("libplacebo") else "no"),
            "hw decode (build): %s   - run hwinfo.py for a real device probe" % hw,
        ]))
        for name, (st, btn) in self._tool_rows.items():
            found = bool(va_ffmpeg.find_tool(name))
            st.config(text="installed" if found else "not installed",
                      foreground=clr("ok") if found else clr("muted"))
            btn.config(text="Re-download" if found else "Download")
        self._refresh_perf_panel()

    def _set_perf_mode(self, _event=None):
        m = self.perf_var.get()
        self.perf_mode = va_perf.set_mode(m)
        va_theme.save_ui_key("perf", self.perf_mode)
        self.frame_cache = va_perf.FrameCache()    # budget follows the mode
        self.lbl_status.config(
            text="Performance mode: %s - applies to the next analysis/compare run"
                 % self.perf_mode, foreground=clr("ok"))
        self._refresh_perf_panel()

    def _refresh_perf_panel(self):
        """Fill the PERFORMANCE readout off the main thread (the GPU probe can
        take a second on first run)."""
        def work():
            try:
                txt = va_perf.summary()
                cs = self.frame_cache.stats()
                txt += "\nframe cache: %d frames · %d / %d MB · %d hits this session" % (
                    cs["frames"], cs["bytes"] >> 20, cs["budget"] >> 20, cs["hits"])
            except Exception as exc:   # noqa: BLE001
                txt = "performance probe failed: %r" % (exc,)
            self._post(lambda: self.setup_perf.config(text=txt))
        threading.Thread(target=work, daemon=True).start()

    def _download_tool(self, name):
        st, btn = self._tool_rows[name]
        btn.config(state=tk.DISABLED)
        st.config(text="downloading...", foreground=clr("warn"))
        self._run_bg(self._download_tool_worker, name)

    def _download_tool_worker(self, name):
        import va_ffmpeg
        import va_tools
        dest = va_tools.tools_dir()
        try:
            path = va_tools.install_tool(
                name, dest, on_progress=lambda m, n=name: self._post(
                    lambda: self._tool_progress(n, m)))
            try:
                va_ffmpeg.reset_tool_caches()  # tool paths AND filter capabilities
            except AttributeError:
                va_ffmpeg.find_tool.cache_clear()
            msg, color = "installed: " + os.path.basename(path), clr("ok")
        except Exception as exc:  # noqa: BLE001
            msg, color = "failed: %s" % exc, clr("err")
        self._post(lambda: self._tool_done(name, msg, color))

    def _tool_progress(self, name, m):
        row = self._tool_rows.get(name)
        if row:
            row[0].config(text=m, foreground=clr("warn"))

    def _tool_done(self, name, msg, color):
        row = self._tool_rows.get(name)
        if row:
            import va_ffmpeg
            msg = " ".join(msg.split())            # tool stderr may contain newlines
            if len(msg) > 72:                      # keep long errors from stretching the row
                msg = msg[:69] + "..."
            row[0].config(text=msg, foreground=color)
            row[1].config(state=tk.NORMAL,
                          text="Re-download" if va_ffmpeg.find_tool(name) else "Download")


class ABViewer:
    """A small Toplevel for synced A/B comparison of two videos."""
    MODES = [("A", "a"), ("B", "b"), ("Difference", "diff"), ("Heatmap", "heatmap"),
             ("Split", "split"), ("Blend", "blend")]

    def __init__(self, parent, path_a, path_b):
        self.A = VideoSource(path_a)
        self.B = VideoSource(path_b)
        self.n = max(1, min(self.A.nb_frames or 1, self.B.nb_frames or 1))
        self.win = tk.Toplevel(parent)
        self.win.title("A/B  -  %s  vs  %s" % (os.path.basename(path_a), os.path.basename(path_b)))
        self.win.geometry("900x640")
        self.win.configure(bg=va_theme.WELL["panel"])
        self.win.protocol("WM_DELETE_WINDOW", self._close)
        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(top, text="A = %s   B = %s   |   Mode:" % (
            os.path.basename(path_a), os.path.basename(path_b))).pack(side=tk.LEFT)
        self.mode = tk.StringVar(value="Heatmap")
        cb = ttk.Combobox(top, textvariable=self.mode, state="readonly", width=12,
                          values=[m[0] for m in self.MODES])
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._render())
        self.pos = tk.IntVar(value=0)
        ttk.Scale(self.win, from_=0, to=self.n - 1, orient=tk.HORIZONTAL,
                  variable=self.pos, command=lambda v: self._render()).pack(fill=tk.X, padx=8)
        self.label = tk.Label(self.win, bg=va_theme.WELL["bg"])
        self.label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._photo = None
        self._tmp = os.path.join(tempfile.gettempdir(),
                                 "va_ab_%s.ppm" % uuid.uuid4().hex[:8])
        self.win.after(120, self._render)

    def _render(self, *_):
        idx = int(float(self.pos.get()))
        fa = self.A.frame_at(idx)
        fb = self.B.frame_at(idx)
        key = dict(self.MODES).get(self.mode.get(), "heatmap")
        rgb = va_compare.difference(fa, fb, key)
        if rgb is None:
            return
        lw = max(320, self.label.winfo_width())
        lh = max(240, self.label.winfo_height())
        h, w = rgb.shape[:2]
        sc = min(lw / w, lh / h)
        rgb = cv2.resize(rgb, (max(1, int(w * sc)), max(1, int(h * sc))))
        try:
            with open(self._tmp, "wb") as fh:
                fh.write(encode_ppm_bytes(rgb))
            self._photo = tk.PhotoImage(file=self._tmp)
            self.label.config(image=self._photo)
        except Exception:  # noqa: BLE001
            pass

    def _close(self):
        self.A.close()
        self.B.close()
        self.win.destroy()


_CLI_MODES = {"analyze": "analyze",          # headless batch QC
              "hwinfo": "hwinfo",            # hardware decode/tonemap probe
              "tools": "va_tools",           # download helper binaries
              "pack-plugins": "pack_plugins"}  # zip plugins for sharing


def main():
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_MODES:
        # `video-analyzer(.exe) analyze clip.mkv` - the headless CLIs ride in
        # the same single exe, so a frozen install needs no Python at all.
        import importlib
        import va_paths
        va_paths.attach_console()              # windowed exe -> parent console
        mode = sys.argv.pop(1)
        sys.argv[0] = "%s %s" % (os.path.basename(sys.argv[0]), mode)
        sys.exit(importlib.import_module(_CLI_MODES[mode]).main() or 0)

    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    import va_paths
    if va_paths.is_frozen():
        # No console to print tracebacks to - surface crashes in a dialog.
        def _show_error(exc_type, exc, tb):
            import traceback
            txt = "".join(traceback.format_exception(exc_type, exc, tb))
            try:
                messagebox.showerror("Video Analyzer - unexpected error", txt[-1800:])
            except Exception:  # noqa: BLE001
                pass
        sys.excepthook = _show_error
        root.report_callback_exception = _show_error
    path = sys.argv[1] if len(sys.argv) > 1 else None
    VideoAnalyzerApp(root, initial_path=path)
    root.mainloop()


if __name__ == "__main__":
    main()
