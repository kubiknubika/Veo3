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
rem NOTE: never wrap "%PIP%" in quotes at execution time - cmd would treat the
rem whole string as a single file path and fail with "'...' is not recognized".
set "PIP="%VPY%" -m pip"
call :try_uv

rem ---------- Install dependencies when missing or requirements.txt changed ----------
set "NEED_INSTALL="
set "REQHASH="
for /f "usebackq delims=" %%H in (`"%VPY%" -c "import hashlib;print(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest()[:16])" 2^>nul`) do set "REQHASH=%%H"
if not defined REQHASH set "NEED_INSTALL=1"
if defined REQHASH (
    if not exist ".venv\.deps_stamp" set "NEED_INSTALL=1"
    findstr /x /c:"%REQHASH%" ".venv\.deps_stamp" >nul 2>nul || set "NEED_INSTALL=1"
)
if defined NEED_INSTALL (
    echo [SETUP] Installing dependencies - this may take a few minutes...
    call :install_deps
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
exit /b 0

rem =====================================================================
rem  Subroutines
rem =====================================================================

:try_uv
rem Probe whether the "uv" accelerator is usable inside the venv.
rem The old code ran the installer as "<python.exe> -m uv pip install ..."
rem and then executed it with extra quotes around the whole command line,
rem so cmd tried to open a single file literally named
rem '".venv\Scripts\python.exe -m uv pip"' -> "is not recognized".
rem uv's real entry point is .venv\Scripts\uv.exe ("uv pip ..."); if anything
rem goes wrong we silently keep plain pip.
"%VPY%" -m pip install uv >nul 2>nul
set "UVEXE="
for %%P in ("%VPY%") do set "UVEXE=%%~dpPuv.exe"
if not defined UVEXE exit /b 0
if not exist "%UVEXE%" exit /b 0
set "VIRTUAL_ENV="
for %%P in ("%VPY%") do set "VIRTUAL_ENV=%%~dpP.."
"%UVEXE%" pip --help >nul 2>nul || exit /b 0
set "PIP="%UVEXE%" pip"
exit /b 0

:install_deps
rem Install requirements.txt. %PIP% already contains quoted paths, so it must
rem NOT be wrapped in extra quotes here - cmd would then search for one single
rem file whose name is the whole command line and print "not recognized".
%PIP% install --upgrade pip >nul 2>nul
if errorlevel 1 "%VPY%" -m pip install --upgrade pip >nul 2>nul
%PIP% install -r requirements.txt
if not errorlevel 1 goto :deps_ok
echo [WARNING] Primary installer failed - retrying with plain pip...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
:deps_ok
if defined REQHASH (echo %REQHASH%> ".venv\.deps_stamp")
exit /b 0
