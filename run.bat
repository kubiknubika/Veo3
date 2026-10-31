@echo off
setlocal
rem =====================================================================
rem  Veo3 Pro Generator - one-click launcher for Windows
rem  Double-click this file to start the GUI. The first run creates a
rem  virtual environment and installs all dependencies automatically.
rem =====================================================================

rem Work from the folder that contains this script (handles "Run from any CWD")
cd /d "%~dp0"

title Veo3 Pro Generator

rem ---------- Find Python 3.10+ ----------
set "PYTHON="
where py >nul 2>nul && set "PYLAUNCHER=1"

if defined PYLAUNCHER (
    rem Try the newest interpreters first, require 3.10+
    for %%V in (3.13 3.12 3.11 3.10) do (
        if not defined PYTHON (
            py -%%V -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "PYTHON=py -%%V"
        )
    )
)

if not defined PYTHON (
    where python >nul 2>nul && (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "PYTHON=python"
    )
)

if not defined PYTHON (
    echo [ERROR] Python 3.10 or newer was not found on this system.
    echo.
    echo Install it from https://www.python.org/downloads/
    echo IMPORTANT: tick the "Add python.exe to PATH" option during install.
    echo.
    pause
    exit /b 1
)

rem ---------- Create virtual environment on first run ----------
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "VPY=.venv\Scripts\python.exe"

rem ---------- Speed up first-time installs with uv (optional) ----------
rem uv is a drop-in pip replacement that installs the same packages ~5-10x
rem faster. It is installed inside the .venv only - nothing touches the system.
set "PIP=%VPY% -m pip"
"%VPY%" -m pip install uv >nul 2>nul
"%VPY%" -m uv --help >nul 2>nul
if not errorlevel 1 set "PIP=%VPY% -m uv pip"

rem ---------- Install dependencies when missing or requirements.txt changed ----------
set "NEED_INSTALL="
for /f "usebackq delims=" %%H in (`"%VPY%" -c "import hashlib;print(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest()[:16])" 2^>nul`) do set "REQHASH=%%H"
if defined REQHASH (
    if not exist ".venv\.deps_stamp" set "NEED_INSTALL=1"
    findstr /x /c:"%REQHASH%" ".venv\.deps_stamp" >nul 2>nul || set "NEED_INSTALL=1"
) else (
    rem Fallback for unusual setups where the probe cannot run: import check only
    "%VPY%" -c "import google.genai, PIL, dotenv, flet" >nul 2>nul || set "NEED_INSTALL=1"
)
if defined NEED_INSTALL (
    echo [SETUP] Installing dependencies - this may take a few minutes...
    "%PIP%" install --upgrade pip >nul 2>nul
    "%PIP%" install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    if defined REQHASH (echo %REQHASH%> ".venv\.deps_stamp")
)

rem ---------- Warn if API key is missing ----------
if not exist ".env" (
    if not defined GEMINI_API_KEY (
        echo [WARNING] No .env file with GEMINI_API_KEY found.
        echo Create ".env" in this folder with the line:
        echo     GEMINI_API_KEY=your_google_api_key_here
        echo Get a key at https://aistudio.google.com/apikey
        echo.
    )
)

rem ---------- Check ffmpeg (needed for frames/stitching; app runs without it) ----------
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARNING] ffmpeg was not found on PATH. Video frame extraction and
    echo stitching will fail until you install it, e.g.:  winget install Gyan.FFmpeg
    echo ^(then reopen the console window^). The rest of the app still works.
    echo.
)

rem ---------- Launch ----------
echo [RUN] Starting Veo3 Pro Generator...
"%VPY%" veo_gui.py
if not errorlevel 1 goto :launch_ok
echo.
echo [ERROR] The application exited with an error (code above).
pause
:launch_ok
endlocal
