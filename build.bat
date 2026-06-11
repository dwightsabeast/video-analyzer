@echo off
rem ============================================================================
rem  build.bat - package Video Analyzer into a single video-analyzer.exe
rem
rem  Run this ONCE on a machine with Python 3.9+ installed. The resulting exe
rem  needs NO Python at all - share the app folder (exe + tools\ + plugins\).
rem
rem  Output: video-analyzer.exe in THIS folder (next to tools\ and plugins\),
rem  so the build result is immediately runnable in place.
rem ============================================================================
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% --version >nul 2>nul || (
    echo Python not found. Install it from https://www.python.org/downloads/
    pause & exit /b 1
)

echo [1/3] Installing build dependencies (pyinstaller, numpy, opencv) ...
%PY% -m pip install --upgrade --quiet pyinstaller numpy opencv-python
%PY% -m pip install --quiet tkinterdnd2 2>nul

echo [2/3] Building video-analyzer.exe (takes a few minutes) ...
%PY% -m PyInstaller --clean --noconfirm video-analyzer.spec || (
    echo BUILD FAILED - scroll up for the PyInstaller error.
    pause & exit /b 1
)

echo [3/3] Placing the exe beside tools\ and plugins\ ...
move /y "dist\video-analyzer.exe" "video-analyzer.exe" >nul
rmdir /s /q build dist 2>nul

echo.
echo Done: video-analyzer.exe
echo   - double-click it to launch the GUI
echo   - CLI: video-analyzer.exe analyze ^<file-or-folder^>   (batch QC reports)
echo          video-analyzer.exe hwinfo ^<hdr-clip^>          (GPU decode probe)
echo          video-analyzer.exe tools                       (download ffmpeg etc.)
echo   - to share: zip this folder (exe + tools\ + plugins\); Python not required
echo   - first launch may take ~10 s while the exe unpacks itself
pause
