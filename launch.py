#!/usr/bin/env python3
"""
launch.py - preflight launcher for Video Analyzer.

Checks that ffmpeg + ffprobe are reachable (bundled beside the scripts or on
PATH). If they are, it starts the app. If not, it explains what to download and
where to put it, then offers to start anyway with reduced (OpenCV-only)
functionality. Works with a GUI dialog when Tk is present, else on the console.

    python launch.py [video]
"""

from __future__ import annotations

import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import va_ffmpeg as F

OFFICIAL_URL = "https://ffmpeg.org/download.html"
WINDOWS_BUILDS_URL = "https://www.gyan.dev/ffmpeg/builds/"


def status():
    return F.find_ffmpeg(), F.find_ffprobe(), F.has_filter("libvmaf")


def message(ffmpeg, ffprobe, libvmaf) -> str:
    lines = [
        "Video Analyzer works best with ffmpeg + ffprobe.",
        "",
        "  ffmpeg   : %s" % (ffmpeg or "NOT FOUND"),
        "  ffplay   : %s" % (F.find_tool("ffplay") or "not found - audio playback disabled"),
        "  ffprobe  : %s" % (ffprobe or "NOT FOUND"),
        "  libvmaf  : %s" % ("yes" if libvmaf else "no (VMAF disabled)"),
        "",
        "Without ffmpeg, decoding falls back to OpenCV (no HDR tonemapping and",
        "some HEVC/10-bit files will not open) and per-frame analysis is disabled.",
        "",
        "Download a build, then drop ffmpeg(.exe) and ffprobe(.exe) beside this",
        "script (next to launch.py) or anywhere on your PATH:",
        "  Official builds : %s" % OFFICIAL_URL,
        "  Windows (incl. VMAF): %s" % WINDOWS_BUILDS_URL,
    ]
    return "\n".join(lines)


def launch_app(argv):
    app_path = os.path.join(HERE, "video-analyzer.py")
    if not os.path.isfile(app_path):
        print("Cannot find video-analyzer.py next to launch.py."); return 2
    sys.argv = [app_path] + list(argv)
    spec = importlib.util.spec_from_file_location("va_app_main", app_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SyntaxError as exc:
        print("video-analyzer.py is damaged or incomplete (syntax error at "
              "line %s). Restore it from backup/git before launching." % exc.lineno)
        return 2
    except ImportError as exc:
        print("A required package is missing: %s" % exc)
        print("Install the dependencies with:  pip install opencv-python numpy")
        return 2
    mod.main()
    return 0


def main():
    ffmpeg, ffprobe, libvmaf = status()
    forward = sys.argv[1:]
    if ffmpeg and ffprobe:
        try:
            import va_hwaccel
            hw = ", ".join(sorted(va_hwaccel.hwaccels())) or "none"
        except Exception:
            hw = "?"
        print("Video Analyzer - capabilities:")
        print("  ffmpeg/ffprobe : found")
        print("  libvmaf        : %s" % ("yes" if libvmaf else "no (VMAF disabled)"))
        print("  libplacebo     : %s" % ("yes" if F.has_filter("libplacebo") else "no (GPU HDR tonemap disabled)"))
        print("  hw decode      : %s   (build list; run hwinfo.py for a real device probe)" % hw)
        print("  dovi_tool      : %s" % ("found" if F.find_tool("dovi_tool") else "not found - get it: python va_tools.py"))
        print("  hdr10plus_tool : %s" % ("found" if F.find_tool("hdr10plus_tool") else "not found - get it: python va_tools.py"))
        try:
            import va_perf
            for ln in va_perf.summary().splitlines():
                print("  " + ln)
        except Exception:   # noqa: BLE001 - perf readout must never block launch
            pass
        return launch_app(forward)

    msg = message(ffmpeg, ffprobe, libvmaf)
    proceed = False
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        proceed = messagebox.askyesno(
            "ffmpeg not found",
            msg + "\n\nStart anyway with reduced functionality?")
        r.destroy()
    except Exception:
        print(msg)
        try:
            proceed = input("\nStart anyway with reduced functionality? [y/N] ").strip().lower().startswith("y")
        except EOFError:
            proceed = False
    if proceed:
        return launch_app(forward)
    print("Exiting. Install ffmpeg/ffprobe and run again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
