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
REM  it stops the dashboard. (Your `clark serve` instance, if any,
REM  is independent and keeps running.)
REM
REM  This dashboard talks to `clark serve` on :8000. If it's not
REM  running, the header will show "clark serve unreachable" —
REM  Run Clark Chat.bat (in the clark-mcp repo) auto-starts serve.
REM ============================================================
setlocal
title Clark Operations Dashboard
cd /d "%~dp0"

REM Kill any prior ops dashboard on :8092 so a re-launch always
REM picks up the latest HTML edits.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8092 .*LISTENING"') do (
    echo Stopping prior ops server PID %%a so the new launch picks up the latest code...
    taskkill /F /PID %%a >nul 2>&1
)

python -m cli.main ops --port 8092

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
