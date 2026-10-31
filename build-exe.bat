@echo off
setlocal
rem =====================================================================
rem  Veo3 Pro Generator - one-click PyInstaller build for Windows
rem  Double-click to produce dist\VeoGenerator\VeoGenerator.exe (a folder
rem  that runs WITHOUT Python on any Windows PC). Requires internet the
rem  first time (installs .venv + pyinstaller into this repo only).
rem =====================================================================

cd /d "%~dp0"
title Build VeoGenerator.exe

set "VPY=.venv\Scripts\python.exe"

rem ---------- Reuse the environment from run.bat; create it on demand ----------
if exist "%VPY%" goto :venv_ready
echo [SETUP] Virtual environment not found - creating it...
set "BOOTSTRAP="
where py >nul 2>nul && set "BOOTSTRAP=py -3"
if not defined BOOTSTRAP (
    where python >nul 2>nul && set "BOOTSTRAP=python"
)
if not defined BOOTSTRAP (
    echo [ERROR] Python 3.10+ not found. Install it from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH"^), then run run.bat once, then retry.
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

rem ---------- App dependencies (needed so collect_all can find them) ----------
"%VPY%" -c "import google.genai, PIL, dotenv, flet" >nul 2>nul
if not errorlevel 1 goto :deps_ready
echo [SETUP] Installing app dependencies...
"%VPY%" -m pip install -r requirements.txt >nul 2>nul
if errorlevel 1 "%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
:deps_ready

rem ---------- PyInstaller ----------
"%VPY%" -c "import PyInstaller" >nul 2>nul
if not errorlevel 1 goto :pi_ready
echo [SETUP] Installing PyInstaller into .venv...
"%VPY%" -m pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] Could not install pyinstaller. Check your internet connection.
    pause
    exit /b 1
)
:pi_ready

rem ---------- Build ----------
echo [BUILD] Packaging with PyInstaller - takes a few minutes, please wait...
"%VPY%" -m PyInstaller veo.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. The message above usually names the missing piece.
    pause
    exit /b 1
)

echo.
echo [OK] Built successfully:
echo      dist\VeoGenerator\VeoGenerator.exe
echo.
echo To share: copy the whole dist\VeoGenerator folder to the target PC,
echo put a ".env" file with GEMINI_API_KEY next to the exe (.env.example is
echo inside the folder as a template) and double-click VeoGenerator.exe.
echo ffmpeg must be on PATH there too: winget install Gyan.FFmpeg
echo.
explorer "dist\VeoGenerator"
pause
endlocal
