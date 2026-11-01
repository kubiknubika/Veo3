@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =====================================================================
rem  Veo3 Pro Generator - uninstaller
rem  Removes the virtual environment and cached files created by run.bat.
rem  Your source code, .env (API key) and generated videos are preserved.
rem =====================================================================

cd /d "%~dp0"

echo [UNINSTALL] Veo3 Pro Generator cleanup
echo.

if exist ".venv" (
    echo [UNINSTALL] Removing virtual environment: .venv
    rmdir /s /q ".venv"
    if errorlevel 1 (
        echo [ERROR] Could not delete .venv. Close the app/terminal and try again.
        pause
        exit /b 1
    )
) else (
    echo [INFO] No .venv folder found - nothing to remove.
)

if exist "__pycache__" (
    echo [UNINSTALL] Removing __pycache__
    rmdir /s /q "__pycache__"
)

if exist ".pytest_cache" (
    echo [UNINSTALL] Removing .pytest_cache
    rmdir /s /q ".pytest_cache"
)

echo.
echo [DONE] Cleanup finished. Source files, .env and videos were NOT deleted.
pause
endlocal
