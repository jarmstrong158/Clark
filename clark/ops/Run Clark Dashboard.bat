@echo off
REM ============================================================
REM  Clark Operations Dashboard
REM  Double-click to launch. Opens a local web UI for plan /
REM  what-if / compare / sweep / calendar / morning briefing.
REM
REM  This is FORMS over clark serve — no language model in the
REM  loop. Use this for the staffing questions that actually
REM  matter; use the chat (Run Clark Chat.bat) for prompts that
REM  resist forms.
REM
REM  Talks to clark serve on :8000. If serve isn't running, the
REM  page will show "clark serve unreachable" in the header —
REM  start it (or use Run Clark Chat.bat which auto-starts it).
REM ============================================================
setlocal
title Clark Operations Dashboard
cd /d "%~dp0\..\..\..\"

REM Kill any prior ops dashboard on :8092 so a re-launch always
REM picks up the latest HTML edits.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8092 .*LISTENING"') do (
    echo Stopping prior ops server PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

echo Launching ops dashboard on http://127.0.0.1:8092 ...
echo (Press Ctrl-C here to stop. Browser will open automatically.)
echo.

python -m cli.main ops --port 8092

if errorlevel 1 (
    echo.
    echo ============================================================
    echo  Failed to launch. Make sure you've run `pip install -e .`
    echo  in the clark repo at least once.
    echo ============================================================
    pause
)
