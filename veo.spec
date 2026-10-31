# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the Veo3 Pro Generator GUI.

Build on Windows by double-clicking build-exe.bat (it prepares .venv,
installs pyinstaller and runs this spec). Manual equivalent:

    .venv\\Scripts\\pip install pyinstaller
    .venv\\Scripts\\pyinstaller veo.spec --clean --noconfirm

Result: dist\\VeoGenerator\\VeoGenerator.exe  (folder build, console hidden -
a folder build starts faster than one-file because nothing is unpacked to a
temp dir on every launch).

Ship the whole VeoGenerator folder + a .env file with GEMINI_API_KEY
(.env.example is bundled next to the exe); ffmpeg must be on PATH on the
target PC (winget install Gyan.FFmpeg) or drop ffmpeg.exe next to the app.

Notes:
  * google.genai pulls in many lazily-imported submodules; collect_all
    keeps its data files and hidden imports intact.
  * flet ships web/runner assets that must be collected as package data.
  * The app code already handles sys.frozen (BASE_DIR -> executable dir),
    so output/, prompt_templates.json and .env resolve next to the exe.
  * prompt_templates.json is optional at runtime (built-in defaults exist);
    if it is present in the repo when you build, it gets bundled next to the
    exe. A user-edited copy sitting there will ship too - delete it first if
    you want the vanilla defaults.
"""

import os

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

for pkg in ("flet", "google.genai"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Ship the template file next to the exe when it exists (optional at runtime).
if os.path.exists("prompt_templates.json"):
    datas.append(("prompt_templates.json", "."))

# Ship .env.example so the frozen app user knows where to put GEMINI_API_KEY.
if os.path.exists(".env.example"):
    datas.append((".env.example", "."))

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
