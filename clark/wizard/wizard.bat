@echo off
REM Launches the Clark facility setup wizard — a web UI that walks
REM through configuring your warehouse for fine-tuning.
REM
REM Just double-click this file. The wizard opens in your default
REM browser. The server runs in this window — close the window when
REM you're done. Your saved sessions persist on disk.

REM Hop up to the repo root regardless of where this .bat was launched from.
REM .bat lives at <repo>\clark\wizard\ — two ".." gets us to <repo>.
cd /d "%~dp0\..\.."

REM Run the wizard server (it opens the browser itself after a short delay).
python -m cli.main wizard
