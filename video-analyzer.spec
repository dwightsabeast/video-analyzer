# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec - builds ONE self-contained video-analyzer.exe.
#
#   build.bat            (or: py -3 -m PyInstaller --clean --noconfirm video-analyzer.spec)
#
# Layout after build (portable - zip the folder to share):
#   video-analyzer.exe   all Python code, numpy/cv2, tkinter - no Python needed
#   tools\               ffmpeg.exe etc. stay EXTERNAL (844 MB; found beside the exe,
#                        or downloadable in-app via Setup -> Download)
#   plugins\             user-editable plugins, loaded from beside the exe
#   va_ui.json           settings, written beside the exe
#
# The same exe is also the CLI:  video-analyzer.exe analyze | hwinfo | tools | pack-plugins

import os

hidden = [
    # engine modules imported lazily / by plugins - PyInstaller cannot see them all
    "va_audio", "va_compare", "va_dynhdr", "va_export", "va_ffmpeg",
    "va_forensics", "va_hdr", "va_hwaccel", "va_ipt", "va_metrics", "va_paths",
    "va_perceptual", "va_perf", "va_plugins", "va_plugins_ui", "va_probe",
    "va_qc", "va_quality", "va_rpu", "va_scopes", "va_temporal", "va_theme",
    "va_tools",
    # CLI subcommands dispatched via importlib (video-analyzer.exe analyze ...)
    "analyze", "hwinfo", "pack_plugins",
    # stdlib that external plugins commonly import at runtime
    "csv", "wave", "struct", "zipfile", "tempfile", "textwrap", "uuid",
]

datas = []
try:                       # optional drag-and-drop support (ships tkdnd binaries)
    import tkinterdnd2     # noqa: F401
    from PyInstaller.utils.hooks import collect_data_files
    datas += collect_data_files("tkinterdnd2")
    hidden += ["tkinterdnd2"]
except ImportError:
    pass

a = Analysis(
    ["video-analyzer.py"],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    excludes=[
        # never used - keep the exe small and the AV scanners calm
        "matplotlib", "scipy", "pandas", "PIL", "IPython", "jedi",
        "PyQt5", "PySide2", "PyQt6", "PySide6", "setuptools", "pytest",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="video-analyzer",
    debug=False,
    strip=False,
    upx=False,                 # UPX-packed exes trip antivirus heuristics
    console=False,             # windowed; CLI modes re-attach via va_paths.attach_console
    disable_windowed_traceback=False,
)
