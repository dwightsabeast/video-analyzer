#!/usr/bin/env python3
"""
va_theme - the app's visual identity: a flat "broadcast rack" control surface.

Two palettes - graphite dark and warm-paper light - sharing one amber signal
accent. Video, scopes and the timeline render inside true-dark instrument
"wells" in BOTH themes (like the CRT face of a vectorscope), so measurement
areas keep their reference look while the chrome around them adapts.

    import va_theme
    va_theme.apply(root, "dark")     # builds every ttk style, returns palette
    va_theme.C["ok"]                 # live palette - always the active theme
    va_theme.WELL["play"]            # instrument-well colors (theme-invariant)

Standard library only. The palette dicts are plain data and unit-testable
without Tk; apply() is the only function that needs a real root.
"""

from __future__ import annotations

import os
import json

DARK = {
    "bg":        "#161618",   # window
    "surface":   "#1d1d20",   # panel interiors
    "raised":    "#26262b",   # buttons, toolbar controls
    "hover":     "#313137",
    "hairline":  "#303036",   # 1px borders
    "text":      "#e6e4df",   # warm off-white
    "soft":      "#b8b6b0",
    "muted":     "#8a8a92",
    "faint":     "#5c5c64",
    "accent":    "#e3a455",   # signal amber
    "accent_hi": "#f0b96b",
    "accent_fg": "#1a1208",   # text on amber
    "ok":        "#4ec9a8",
    "warn":      "#e3c555",
    "err":       "#e06c5f",
    "info":      "#7fb4d8",
    "editor_bg": "#1a1a1d",   # text/report surfaces (theme-following)
    "editor_fg": "#d6d4cf",
    "sel_bg":    "#3a3a42",
    "sel_fg":    "#f0eee9",
}

LIGHT = {
    "bg":        "#f2f0ea",   # warm paper, deliberately not white
    "surface":   "#faf9f5",
    "raised":    "#e7e4dc",
    "hover":     "#dcd8ce",
    "hairline":  "#d5d1c7",
    "text":      "#2b2b2e",
    "soft":      "#4e4e54",
    "muted":     "#6e6e76",
    "faint":     "#9b9ba2",
    "accent":    "#b9742a",   # deeper amber for paper contrast
    "accent_hi": "#a06322",
    "accent_fg": "#fdf8f0",
    "ok":        "#1f8a6d",
    "warn":      "#9a7414",
    "err":       "#c2473a",
    "info":      "#2b6cb0",
    "editor_bg": "#faf9f5",
    "editor_fg": "#2e2e32",
    "sel_bg":    "#cfc9bb",
    "sel_fg":    "#1d1d20",
}

# Instrument wells: video / scopes / timeline / A-B viewer. Theme-invariant,
# like the face of a reference monitor - graphics tuned for this dark ground.
WELL = {
    "bg":        "#101012",
    "panel":     "#0c0c0e",
    "text":      "#8a8a92",
    "faint":     "#5a5a62",
    "line":      "#4ec9a8",   # metric trace
    "cut":       "#e06c5f",   # scene-cut markers
    "play":      "#e3a455",   # playhead
    "ev_black":  "#1c2426",   # detected black segments
    "ev_freeze": "#262419",   # detected freezes
    "forensic":  "#b07fd8",   # forensic finding markers
}

C = dict(DARK)          # live palette - mutated in place by apply()
def _pref_path() -> str:
    """va_ui.json beside the .exe when frozen, beside the scripts otherwise -
    keeps settings portable and out of PyInstaller's throwaway temp dir."""
    try:
        import va_paths
        return os.path.join(va_paths.app_dir(), "va_ui.json")
    except ImportError:  # pragma: no cover
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "va_ui.json")


_PREF = _pref_path()


def _load_all() -> dict:
    try:
        d = json.load(open(_PREF))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def load_ui_key(key, default=None):
    """Read one persisted UI preference (theme, perf mode, ...)."""
    return _load_all().get(key, default)


def save_ui_key(key, value):
    """Persist one UI preference without clobbering the others."""
    d = _load_all()
    d[key] = value
    try:
        json.dump(d, open(_PREF, "w"))
    except OSError:
        pass


def load_pref(default="dark"):
    mode = load_ui_key("theme", default)
    return mode if mode in ("dark", "light") else default


def save_pref(mode):
    save_ui_key("theme", mode)



def palette(mode):
    return LIGHT if mode == "light" else DARK


def style_menu(menu):
    """Recolor a tk.Menu (menus do not follow ttk styles)."""
    try:
        menu.configure(bg=C["surface"], fg=C["text"], bd=0, relief="flat",
                       activebackground=C["sel_bg"], activeforeground=C["sel_fg"],
                       activeborderwidth=0, disabledforeground=C["faint"])
    except Exception:  # noqa: BLE001 - some platforms reject options
        pass


def apply(root, mode="dark"):
    """Build the full ttk style set for `mode`. Safe to call again to re-theme."""
    from tkinter import ttk

    C.clear()
    C.update(palette(mode))
    root.configure(bg=C["bg"])

    st = ttk.Style(root)
    try:
        st.theme_use("clam")            # the most styleable built-in base
    except Exception:  # noqa: BLE001
        pass

    body = ("Segoe UI", 9)
    cap = ("Segoe UI", 8, "bold")

    st.configure(".", background=C["bg"], foreground=C["text"], font=body,
                 bordercolor=C["hairline"], lightcolor=C["bg"], darkcolor=C["bg"],
                 troughcolor=C["raised"], focuscolor=C["accent"],
                 selectbackground=C["sel_bg"], selectforeground=C["sel_fg"],
                 insertcolor=C["text"], fieldbackground=C["editor_bg"])
    st.configure("TFrame", background=C["bg"])
    st.configure("TLabel", background=C["bg"], foreground=C["text"])
    st.configure("Caption.TLabel", font=cap, foreground=C["muted"])

    st.configure("TButton", background=C["raised"], foreground=C["text"],
                 bordercolor=C["hairline"], focuscolor=C["raised"],
                 lightcolor=C["raised"], darkcolor=C["raised"],
                 padding=(11, 5), relief="flat")
    st.map("TButton",
           background=[("disabled", C["bg"]), ("pressed", C["hover"]), ("active", C["hover"])],
           foreground=[("disabled", C["faint"])],
           bordercolor=[("active", C["muted"])])
    st.configure("Accent.TButton", background=C["accent"], foreground=C["accent_fg"],
                 bordercolor=C["accent"], lightcolor=C["accent"], darkcolor=C["accent"],
                 font=("Segoe UI", 9, "bold"))
    st.map("Accent.TButton",
           background=[("disabled", C["raised"]), ("pressed", C["accent_hi"]),
                       ("active", C["accent_hi"])],
           foreground=[("disabled", C["faint"])],
           bordercolor=[("disabled", C["hairline"])])

    st.configure("TMenubutton", background=C["raised"], foreground=C["text"],
                 bordercolor=C["hairline"], padding=(11, 5), relief="flat",
                 arrowcolor=C["muted"])
    st.map("TMenubutton",
           background=[("disabled", C["bg"]), ("active", C["hover"])],
           foreground=[("disabled", C["faint"])])

    st.configure("TNotebook", background=C["bg"], bordercolor=C["hairline"],
                 tabmargins=(0, 4, 0, 0))
    st.configure("TNotebook.Tab", background=C["bg"], foreground=C["muted"],
                 bordercolor=C["hairline"], padding=(13, 5), font=cap)
    st.map("TNotebook.Tab",
           background=[("selected", C["surface"])],
           foreground=[("selected", C["accent"]), ("active", C["soft"])],
           expand=[("selected", (0, 1, 0, 0))])

    st.configure("TLabelframe", background=C["bg"], bordercolor=C["hairline"],
                 lightcolor=C["bg"], darkcolor=C["bg"], relief="solid", borderwidth=1)
    st.configure("TLabelframe.Label", background=C["bg"], foreground=C["muted"], font=cap)

    st.configure("Horizontal.TScale", background=C["bg"], troughcolor=C["raised"],
                 bordercolor=C["hairline"], lightcolor=C["accent"], darkcolor=C["accent"],
                 gripcount=0)
    st.map("Horizontal.TScale", background=[("active", C["bg"])])

    for sb in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        st.configure(sb, background=C["raised"], troughcolor=C["bg"],
                     bordercolor=C["bg"], arrowcolor=C["muted"], relief="flat")
        st.map(sb, background=[("active", C["hover"])])

    st.configure("TCombobox", fieldbackground=C["editor_bg"], background=C["raised"],
                 foreground=C["editor_fg"], bordercolor=C["hairline"],
                 arrowcolor=C["muted"], padding=(6, 3))
    st.map("TCombobox",
           fieldbackground=[("readonly", C["raised"])],
           foreground=[("readonly", C["text"])],
           selectbackground=[("readonly", C["raised"])],
           selectforeground=[("readonly", C["text"])])
    root.option_add("*TCombobox*Listbox.background", C["surface"])
    root.option_add("*TCombobox*Listbox.foreground", C["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["sel_bg"])
    root.option_add("*TCombobox*Listbox.selectForeground", C["sel_fg"])

    st.configure("Treeview", background=C["surface"],
                 fieldbackground=C["surface"], foreground=C["text"],
                 bordercolor=C["hairline"], lightcolor=C["surface"],
                 darkcolor=C["surface"], rowheight=24, relief="flat")
    st.map("Treeview",
           background=[("selected", C["sel_bg"])],
           foreground=[("disabled", C["faint"]), ("selected", C["sel_fg"])])
    st.configure("Treeview.Heading", background=C["raised"],
                 foreground=C["muted"], bordercolor=C["hairline"],
                 relief="flat", font=cap, padding=(6, 4))
    st.map("Treeview.Heading", background=[("active", C["hover"])])

    st.configure("TPanedwindow", background=C["bg"])
    st.configure("Sash", sashthickness=6, gripcount=0, background=C["bg"])
    st.configure("TSeparator", background=C["hairline"])

    st.configure("HDR.TLabel", foreground=C["accent"], font=("Segoe UI", 10, "bold"))
    st.configure("SDR.TLabel", foreground=C["muted"], font=("Segoe UI", 10))
    return C
