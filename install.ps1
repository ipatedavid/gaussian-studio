# Sets up Gaussian Studio on Windows and adds its Start menu and desktop shortcuts.
# Double-click install.cmd, or run: powershell -ExecutionPolicy Bypass -File install.ps1
# Run it again after updating; your keys and settings are kept.
# This only finds Python (offering to install it); desktop\installer.py does the rest.
$ErrorActionPreference = 'Stop'

# Whether a command is a working Python 3.10 or newer. The Microsoft Store's "python"
# placeholder opens the Store instead, so this rules it out too.
function Test-Python($command, $arguments) {
    try {
        & $command @arguments -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}
function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in '3.13', '3.12', '3.11', '3.10') {
            if (Test-Python 'py' @("-$version")) { return @('py', "-$version") }
        }
    }
    if ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-Python 'python' @())) {
        return @('python')
    }
    return $null
}

Write-Host 'Looking for Python 3.10 or newer...'
$python = @(Find-Python)
if ($python.Count -eq 0 -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    $answer = Read-Host 'Python is missing. Install Python 3.12 now with winget? [Y/n]'
    if ($answer -notmatch '^[Nn]') {
        winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                    [Environment]::GetEnvironmentVariable('Path', 'User')
        $python = @(Find-Python)
    }
}
if ($python.Count -eq 0) {
    Write-Host 'Python 3.10+ is missing. Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run this again.' -ForegroundColor Red
    exit 1
}
$pyArgs = @($python | Select-Object -Skip 1)
Write-Host ("Using " + (& $python[0] @pyArgs --version))
& $python[0] @pyArgs -X utf8 (Join-Path $PSScriptRoot 'desktop\installer.py') install
exit $LASTEXITCODE
