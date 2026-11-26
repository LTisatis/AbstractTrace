@echo off
setlocal
cd /d "%~dp0" || goto :failed

set "ABSTRACTTRACE_UV=.review-data\tools\uv-0.9.18\uv.exe"
where uv.exe >nul 2>&1
if not errorlevel 1 (
    set "ABSTRACTTRACE_UV=uv.exe"
    goto :start
)
if exist "%ABSTRACTTRACE_UV%" goto :start

echo Downloading uv 0.9.18 from Astral's GitHub releases...
set "ABSTRACTTRACE_ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ABSTRACTTRACE_ARCH=%PROCESSOR_ARCHITEW6432%"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -Command ^
    "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';" ^
    "switch ($env:ABSTRACTTRACE_ARCH) {" ^
    " 'AMD64' { $target = 'x86_64'; $checksum = '28cbe5d30907a774bfe27a517a39b494ec6f7d3816bda8bbf6f9645490449182' }" ^
    " 'ARM64' { $target = 'aarch64'; $checksum = 'fadb43ba13091f44e1786fc3967e65c7786d86192aa205d718307c649927cfc2' }" ^
    " default { throw 'This launcher requires Windows x64 or ARM64.' } }" ^
    "$uvPath = [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $env:ABSTRACTTRACE_UV));" ^
    "$tools = Split-Path -Parent $uvPath; New-Item -ItemType Directory -Path $tools -Force | Out-Null;" ^
    "$suffix = [Guid]::NewGuid().ToString('N'); $archive = Join-Path $tools ($suffix + '.zip'); $stagedExe = Join-Path $tools ($suffix + '.tmp');" ^
    "try {" ^
    " $assetUrl = 'https://github.com/astral-sh/uv/releases/download/0.9.18/uv-' + $target + '-pc-windows-msvc.zip';" ^
    " Invoke-WebRequest -UseBasicParsing -Uri $assetUrl -OutFile $archive -TimeoutSec 180;" ^
    " $hash = [Security.Cryptography.SHA256]::Create(); $stream = [IO.File]::OpenRead($archive);" ^
    " try { $actual = [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '') } finally { $stream.Dispose(); $hash.Dispose() }" ^
    " if ($actual -ne $checksum) { throw 'The downloaded uv archive failed SHA-256 verification.' }" ^
    " Add-Type -AssemblyName System.IO.Compression.FileSystem;" ^
    " $zip = [IO.Compression.ZipFile]::OpenRead($archive);" ^
    " try { $entry = $zip.Entries | Where-Object { $_.Name -eq 'uv.exe' } | Select-Object -First 1;" ^
    "       if ($null -eq $entry) { throw 'uv.exe was not found in the verified archive.' }" ^
    "       [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $stagedExe) } finally { $zip.Dispose() }" ^
    " Move-Item -LiteralPath $stagedExe -Destination $uvPath -Force;" ^
    "} finally { [IO.File]::Delete($archive); [IO.File]::Delete($stagedExe) }" ^
    "exit 0"
if errorlevel 1 goto :failed

:start
echo Starting AbstractTrace. Keep this window open while using the web app.
"%ABSTRACTTRACE_UV%" run --locked --no-dev python -m abstract_review serve --open --port 0
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo AbstractTrace could not start. Check the error above and the Quick Start in README.md.
pause
exit /b 1
