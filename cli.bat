@echo off
setlocal
rem =====================================================================
rem  Veo3 Pro Generator - CLI launcher for Windows
rem  Usage examples:
rem     cli.bat                       interactive menu
rem     cli.bat video "a cat surfing"  generate a video
rem     cli.bat tasks                  show task journal
rem =====================================================================

cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"

rem Reuse the environment created by run.bat; build it on demand otherwise
if exist "%VPY%" goto :venv_ready
echo [SETUP] Virtual environment not found - creating it now...
set "BOOTSTRAP="
where py >nul 2>nul && set "BOOTSTRAP=py -3"
if not defined BOOTSTRAP (
    where python >nul 2>nul && set "BOOTSTRAP=python"
)
if not defined BOOTSTRAP (
    echo [ERROR] Python 3.10+ not found. Install it from python.org,
    echo then run "run.bat" once to finish the setup.
    pause
    exit /b 1
)
%BOOTSTRAP% -m venv .venv
if not exist "%VPY%" (
    echo [ERROR] Failed to create the virtual environment.
    pause
    exit /b 1
)
:venv_ready

rem Optional accelerator: uv inside the venv (drop-in pip replacement, ~5-10x faster)
set "PIP=%VPY% -m pip"
"%VPY%" -m pip install uv >nul 2>nul
"%VPY%" -m uv --help >nul 2>nul
if not errorlevel 1 set "PIP=%VPY% -m uv pip"

"%VPY%" -c "import google.genai, PIL, dotenv" >nul 2>nul
if not errorlevel 1 goto :deps_ready
echo [SETUP] Installing dependencies...
"%PIP%" install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
:deps_ready

"%VPY%" veo_generator.py %*
exit /b %errorlevel%
