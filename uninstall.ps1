# Removes Gaussian Studio from Windows, asking before touching your results or your SSH key.
# Double-click uninstall.cmd, or run: powershell -ExecutionPolicy Bypass -File uninstall.ps1
# For scripts: uninstall.ps1 --yes [--delete-results] [--delete-key] [--ignore-pods]
# This only finds Python; desktop\installer.py does the rest.
$ErrorActionPreference = 'Stop'

# A Python outside the app's folder, so the whole folder can be deleted; the app's own as a fallback.
$python = @(Join-Path $PSScriptRoot 'desktop\.venv\Scripts\python.exe')
if (Get-Command py -ErrorAction SilentlyContinue) { $python = @('py', '-3') }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $python = @('python') }
$pyArgs = @($python | Select-Object -Skip 1)
& $python[0] @pyArgs -X utf8 (Join-Path $PSScriptRoot 'desktop\installer.py') uninstall @args
exit $LASTEXITCODE
