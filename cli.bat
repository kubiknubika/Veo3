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
rem NOTE: "%PIP%" must never be wrapped in quotes at run time - cmd would then
rem search for one single file whose name is the whole command line.
set "PIP="%VPY%" -m pip"
call :try_uv

"%VPY%" -c "import google.genai, PIL, dotenv" >nul 2>nul
if not errorlevel 1 goto :run_main
echo [SETUP] Installing dependencies...
call :install_deps

:run_main
"%VPY%" veo_generator.py %*
exit /b %errorlevel%

rem =====================================================================
rem  Subroutines
rem =====================================================================

:try_uv
rem The old code built the command "<python.exe> -m uv pip install ..." and
rem executed it with extra quotes around the whole string, so cmd searched for
rem a single file literally named '".venv\Scripts\python.exe -m uv pip"' and
rem printed "is not internal or external command...". The working entry point
rem is .venv\Scripts\uv.exe ("uv pip ..."); fall back to plain pip otherwise.
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
%PIP% install -r requirements.txt
if not errorlevel 1 exit /b 0
echo [WARNING] Primary installer failed - retrying with plain pip...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
exit /b 0
