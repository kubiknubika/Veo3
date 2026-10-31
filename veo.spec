# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the Veo3 Pro Generator GUI.

Build from a Windows machine inside the prepared venv:

    .venv\\Scripts\\pip install pyinstaller
    .venv\\Scripts\\pyinstaller veo.spec --clean --noconfirm

Result: dist\\VeoGenerator\\VeoGenerator.exe  (one-file, console hidden).
Ship that folder next to a .env file containing GEMINI_API_KEY; ffmpeg must
be on PATH (winget install Gyan.FFmpeg) or bundled as ffmpeg.exe beside it.

Notes:
  * google.genai pulls in many lazily-imported submodules; `collect_all`
    keeps its data files and hidden imports intact.
  * flet ships web/runner assets that must be collected as package data.
  * The app code already handles `sys.frozen` (BASE_DIR -> executable dir),
    so output/, prompt_templates.json and .env resolve next to the exe.
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

for pkg in ("flet", "google.genai"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["veo_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["pillow", "PIL.Image", "dotenv", "task_journal"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy", "scipy"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VeoGenerator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI app: no black console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VeoGenerator",
)
