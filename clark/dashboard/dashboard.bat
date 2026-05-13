@echo off
REM Launches the Clark training dashboard server and opens it in your browser.
REM Just double-click this file. Server runs in this window — close the window
REM to stop it. Training is not affected.

REM Hop up to the repo root regardless of where this .bat was launched from.
REM .bat lives at <repo>\clark\dashboard\ — two ".." gets us to <repo>.
cd /d "%~dp0\..\.."

REM Open browser after a brief delay so the server has time to bind.
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080/"

REM Run the server in this window (foreground) so logs are visible.
python -m cli.main dashboard
