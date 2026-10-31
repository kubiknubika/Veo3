' Veo3 Pro Generator - silent GUI launcher (no console window).
' Double-click this file to start the app. On the very first run it opens
' the visible setup console (run.bat); afterwards the GUI starts silently
' via pythonw.exe. To create a taskbar/desktop shortcut: right-click this
' file -> Send to -> Desktop (create shortcut).
Option Explicit

Dim sh, fso, appDir, venvPy
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = appDir
venvPy = appDir & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(venvPy) Then
    ' First run: show the console so the user can watch/fix the setup
    sh.Run Chr(34) & appDir & "\run.bat" & Chr(34), 1, True
Else
    sh.Run Chr(34) & venvPy & Chr(34) & " veo_gui.py", 0, False
End If
