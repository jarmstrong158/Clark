@echo off
REM ============================================================
REM  Clark Operations Dashboard
REM  Double-click this file to launch.
REM
REM  Opens a web UI in your default browser with forms for
REM  plan / what-if / compare / sweep / calendar / morning
REM  briefing - direct over `clark serve`, no language model in
REM  the loop.
REM
REM  The black window that opens is the dashboard server. Closing
REM  it stops the dashboard. If we have to auto-launch `clark
REM  serve` it opens in its OWN window - close that to stop serve.
REM ============================================================
title Clark Operations Dashboard
cd /d "%~dp0"

REM --- Kill any prior dashboard on :8092 so re-launch picks up edits.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8092 .*LISTENING"') do (
    echo Stopping prior ops dashboard PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

REM --- Is clark serve already up on :8000? Every form posts to it.
curl -sf -o NUL --max-time 2 http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto launch_dash

REM --- serve is down: auto-launch it if we can find the checkpoint+configs.
if not exist "clark\data\checkpoints\clark_foundation.pt" goto no_serve
if not exist "clark\data\configs" goto no_serve

echo.
echo clark serve not running on :8000 - auto-launching.
echo (separate window; close that window to stop serve.)
start "Clark Serve port 8000" cmd /k python -m cli.main serve --model clark\data\checkpoints\clark_foundation.pt --facilities-dir clark\data\configs --port 8000

echo Waiting for clark serve to become ready (up to 40s)...
set /a CLARK_TRIES=0
:waitloop
set /a CLARK_TRIES+=1
ping -n 2 127.0.0.1 >nul
curl -sf -o NUL --max-time 1 http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto serve_ready
if %CLARK_TRIES% geq 40 goto serve_timeout
goto waitloop

:serve_ready
echo clark serve is ready.
goto launch_dash

:serve_timeout
echo.
echo  WARN: clark serve didn't come up in 40s. Dashboard will launch
echo  anyway, but forms will show errors until serve is reachable on
echo  :8000. Check the "Clark Serve" window for errors.
echo.
goto launch_dash

:no_serve
echo.
echo ------------------------------------------------------------
echo  NOTE: clark serve is not running and I couldn't auto-launch
echo  it - expected to find:
echo     clark\data\checkpoints\clark_foundation.pt
echo     clark\data\configs
echo  in this folder. Start clark serve manually:
echo     python -m cli.main serve --model clark\data\checkpoints\clark_foundation.pt --facilities-dir clark\data\configs
echo.
echo  Dashboard will load but every form will show errors.
echo ------------------------------------------------------------
echo.
goto launch_dash

:launch_dash
echo Launching ops dashboard on http://127.0.0.1:8092 ...
echo Press Ctrl-C in this window to stop.
echo.

REM Open the browser tab explicitly via Windows `start` rather than
REM relying on Python's webbrowser.open - that one silently no-ops
REM when launched from a bat-spawned cmd on some Windows setups.
REM Run the dashboard with --no-browser to avoid a double-open.
start "" "http://127.0.0.1:8092/"

python -m cli.main ops --port 8092 --no-browser

echo.
echo ============================================================
echo  The dashboard server has stopped. If it never started, the
echo  most likely cause is a missing Python or missing deps:
echo     install Python 3.10+ from https://python.org, then run
echo     pip install -e .   from this folder, and re-launch.
echo ============================================================
pause
