@echo off
REM ============================================================
REM  Clark Operations Dashboard
REM  Double-click this file to launch.
REM
REM  Opens a web UI in your default browser with forms for
REM  plan / what-if / compare / sweep / calendar / morning
REM  briefing — direct over `clark serve`, no language model in
REM  the loop.
REM
REM  The black window that opens is the dashboard server. Closing
REM  it stops the dashboard. If we have to auto-launch `clark
REM  serve` it opens in its OWN window — close that to stop serve.
REM ============================================================
setlocal enabledelayedexpansion
title Clark Operations Dashboard
cd /d "%~dp0"

REM --- Kill any prior dashboard on :8092 so re-launch picks up edits.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8092 .*LISTENING"') do (
    echo Stopping prior ops dashboard PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

REM --- Ensure clark serve is up on :8000 (every form posts to it).
REM Same approach as Run Clark Chat.bat: probe /health; if down,
REM launch serve in a separate window so the user sees its logs and
REM can close it independently.
curl -sf -o NUL --max-time 2 http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
    set CLARK_CKPT=clark\data\checkpoints\clark_foundation.pt
    set CLARK_CFGS=clark\data\configs
    if exist "%CLARK_CKPT%" if exist "%CLARK_CFGS%" (
        echo.
        echo clark serve not running on :8000 — auto-launching.
        echo (separate window; close that window to stop serve.)

        REM Preemptive check: if port 8000 has any lingering state
        REM (TIME_WAIT from a prior serve close), wait for it to clear.
        netstat -ano | findstr /R /C:":8000 " >nul 2>&1
        if not errorlevel 1 (
            echo Port 8000 has lingering connection state — waiting 10s to release...
            ping -n 11 127.0.0.1 >nul
        )

        start "Clark Serve (port 8000)" cmd /k "cd /d %~dp0 ^&^& python -m cli.main serve --model clark\data\checkpoints\clark_foundation.pt --facilities-dir clark\data\configs --port 8000"

        echo Waiting for clark serve to become ready (up to 30s)...
        set CLARK_READY=0
        for /l %%i in (1,1,30) do (
            if "!CLARK_READY!"=="0" (
                ping -n 2 127.0.0.1 >nul
                curl -sf -o NUL --max-time 1 http://127.0.0.1:8000/health >nul 2>&1
                if not errorlevel 1 set CLARK_READY=1
            )
        )

        if "!CLARK_READY!"=="0" (
            echo.
            echo  WARN: clark serve didn't come up in 30s. Dashboard
            echo  will launch anyway, but every form will show errors
            echo  until serve is reachable on :8000. Check the
            echo  "Clark Serve" window for errors.
            echo.
        ) else (
            echo clark serve is ready.
            echo.
        )
    ) else (
        echo.
        echo ------------------------------------------------------------
        echo  NOTE: clark serve is not running and I couldn't auto-launch
        echo  it — expected to find:
        echo     %CLARK_CKPT%
        echo     %CLARK_CFGS%
        echo  in the current directory. Start clark serve manually:
        echo     python -m cli.main serve --model %CLARK_CKPT% --facilities-dir %CLARK_CFGS%
        echo.
        echo  Dashboard will load but every form will show errors.
        echo ------------------------------------------------------------
        echo.
    )
)

echo Launching ops dashboard on http://127.0.0.1:8092 ...
echo Press Ctrl-C in this window to stop.
echo.

REM Open the browser tab explicitly via Windows `start` rather than
REM relying on Python's webbrowser.open — that one silently no-ops
REM when launched from a bat-spawned cmd on some Windows setups,
REM which is why users saw the server come up but no tab open.
REM Run the dashboard with --no-browser to avoid a double-open if
REM webbrowser DID happen to fire.
start "" "http://127.0.0.1:8092/"

python -m cli.main ops --port 8092 --no-browser

if errorlevel 1 (
    echo.
    echo ============================================================
    echo  The dashboard couldn't start. The most likely cause is
    echo  that Python isn't installed, or you're missing deps.
    echo.
    echo  Install Python 3.10+ from https://python.org, then run:
    echo     pip install -e .
    echo  from this folder before re-launching.
    echo ============================================================
    pause
)
