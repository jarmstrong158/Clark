@echo off
REM ============================================================
REM  Clark Facility Setup Wizard
REM  Double-click this file to launch.
REM
REM  The wizard opens in your default browser. Walk through the
REM  steps to configure Clark for your warehouse.
REM
REM  The black window that opens is the wizard server. Closing
REM  it stops the wizard. (Your saved progress is on disk and
REM  will still be there next time you run it.)
REM ============================================================
title Clark Facility Setup Wizard
cd /d "%~dp0"

python -m cli.main wizard

REM If python exits with an error (e.g. python not installed), keep
REM the window open so the user can read the message.
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  The wizard couldn't start. The most likely cause is that
    echo  Python isn't installed, or you're missing dependencies.
    echo.
    echo  Install Python 3.10+ from https://python.org, then run:
    echo     pip install -e .
    echo  from this folder before re-launching.
    echo ============================================================
    pause
)
