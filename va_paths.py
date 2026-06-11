#!/usr/bin/env python3
"""
va_paths - one answer to "where is the app?" for source AND frozen (.exe) runs.

Running from source, the app root is the folder holding these scripts. Frozen
with PyInstaller --onefile, __file__ points into a throwaway extraction dir
under %TEMP%, so anything that must sit BESIDE the program - tools/, plugins/,
va_ui.json - would land in the wrong place and vanish on exit. app_dir()
returns the folder of the .exe instead, keeping the portable-app layout:

    video-analyzer.exe      <- sys.executable
    tools/ffmpeg.exe        <- found via app_dir()
    plugins/<name>/         <- found via app_dir()
    va_ui.json              <- written via app_dir()

attach_console() lets the windowed exe print into the cmd/PowerShell window
that launched it, so `video-analyzer.exe analyze clip.mkv` behaves like a
normal console tool while plain double-click stays console-free.
"""

from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    """True when running as a PyInstaller-style bundle."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> str:
    """The folder external companions live in: beside the exe when frozen,
    beside these scripts when run from source."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def bundle_dir() -> str:
    """Where bundled read-only data was unpacked (PyInstaller _MEIPASS).
    Equals app_dir() when running from source."""
    return getattr(sys, "_MEIPASS", app_dir())


def attach_console() -> None:
    """Windows, frozen-windowed only: re-attach to the parent console so CLI
    subcommands can print. Falls back to devnull streams so argparse/print
    never crash when double-clicked without a console."""
    if os.name == "nt" and is_frozen():
        try:
            import ctypes
            if ctypes.windll.kernel32.AttachConsole(-1):   # parent process
                for name in ("stdout", "stderr"):
                    try:
                        setattr(sys, name, open("CONOUT$", "w", buffering=1,
                                                encoding="utf-8", errors="replace"))
                    except OSError:
                        pass
                print()                    # step below the shell prompt line
        except Exception:  # noqa: BLE001 - console is a nicety, never fatal
            pass
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
