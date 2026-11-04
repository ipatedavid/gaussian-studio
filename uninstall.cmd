@echo off
rem Double-click to remove Gaussian Studio from Windows (runs uninstall.ps1).
rem It all runs on one line from outside this folder, since the folder is deleted meanwhile.
cd /d "%TEMP%" & powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %* & pause & exit /b
