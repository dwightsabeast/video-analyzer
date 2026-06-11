@echo off
REM Launch Video Analyzer (preflight-checks ffmpeg, then starts the GUI)
REM Prefers the py launcher: a bare "python" can hit the Microsoft Store stub.
where py >nul 2>nul
if %errorlevel%==0 goto haspy
python "%~dp0launch.py" %*
goto done
:haspy
py -3 "%~dp0launch.py" %*
:done
if errorlevel 1 pause
