@echo off
rem Double-click to set up Gaussian Studio on Windows (runs install.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
