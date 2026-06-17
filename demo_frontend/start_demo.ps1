$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$server = Join-Path $PSScriptRoot "server.py"

if (!(Test-Path $python)) {
    throw "Project virtual environment was not found: $python"
}

Write-Host "Starting spinal anomaly demo..."
Write-Host "URL: http://127.0.0.1:8008"
& $python $server
