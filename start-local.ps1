# Starts the whole project locally: the FastAPI backend (port 7860) and the website (port 3000).
# Each runs in its own window; the backend restarts automatically if it ever stops unexpectedly.
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File start-local.ps1   or double-click start-local.bat
# The website runs in production mode (every page pre-built, instant navigation). It is rebuilt automatically when
# its code has changed since the last build. Use -Dev for live-reload while editing the website code.
param([switch]$NoBrowser, [switch]$Dev)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Api = "http://127.0.0.1:7860/health"
$Web = "http://localhost:3000"

function Test-Url($url, $timeout = 3) {
    try { return (Invoke-WebRequest -UseBasicParsing $url -TimeoutSec $timeout).StatusCode -eq 200 } catch { return $false }
}

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { throw "Python was not found on PATH. Install Python 3.13 or activate your environment first." }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm was not found on PATH. Install Node.js first." }

# ---- backend ---------------------------------------------------------------------------------
if (Test-Url $Api) {
    Write-Host "Backend already running at $Api"
} else {
    Write-Host "Starting backend (FastAPI, port 7860) in a new window..."
    $loop = @"
`$host.UI.RawUI.WindowTitle = 'GlucoCast backend (port 7860)'
Set-Location '$Root'
# Real OhioT1DM files take 20-60 s in mode=all (the research XML parser is slow); the default 30 s limit would
# abandon them half-way and later uploads would compete with the abandoned work.
`$env:PREDICTION_TIMEOUT_SECONDS = '180'
while (`$true) {
    & '$python' -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860
    Write-Host 'Backend stopped (exit code' `$LASTEXITCODE '). Restarting in 3 s - close this window to stop it for good.' -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}
"@
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $loop | Out-Null
}

# ---- website ---------------------------------------------------------------------------------
if (Test-Url $Web) {
    Write-Host "Website already running at $Web"
} else {
    $front = Join-Path $Root "frontend"
    if (-not (Test-Path (Join-Path $front "node_modules"))) {
        Write-Host "Installing website dependencies (first run only)..."
        Push-Location $front; npm ci --no-audit --no-fund; Pop-Location
    }
    if ($Dev) {
        Write-Host "Starting website in DEVELOPMENT mode (pages compile on first visit) in a new window..."
        $webCmd = "`$host.UI.RawUI.WindowTitle = 'GlucoCast website DEV (port 3000)'; Set-Location '$front'; npm run dev"
    } else {
        $buildId = Join-Path $front ".next\BUILD_ID"
        $srcItems = @("src", "public") | ForEach-Object { Get-ChildItem (Join-Path $front $_) -Recurse -File }
        $srcItems += @("package.json", "next.config.mjs", "tailwind.config.ts") | ForEach-Object { Get-Item (Join-Path $front $_) }
        $newest = ($srcItems | Measure-Object -Property LastWriteTime -Maximum).Maximum
        if (-not (Test-Path $buildId) -or (Get-Item $buildId).LastWriteTime -lt $newest) {
            Write-Host "Building the website (only needed after code changes, about 1 minute)..."
            Push-Location $front
            npm run build
            $buildOk = $LASTEXITCODE -eq 0
            Pop-Location
            if (-not $buildOk) { throw "Website build failed - see the messages above. You can still run it with -Dev." }
        } else {
            Write-Host "Website build is up to date."
        }
        Write-Host "Starting website (Next.js production server, port 3000) in a new window..."
        $webCmd = "`$host.UI.RawUI.WindowTitle = 'GlucoCast website (port 3000)'; Set-Location '$front'; npm run start -- -p 3000"
    }
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $webCmd | Out-Null
}

# ---- wait until both answer ------------------------------------------------------------------
Write-Host "Waiting for both servers..."
$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline -and -not (Test-Url $Api)) { Start-Sleep -Seconds 2 }
$webOk = $false
while ((Get-Date) -lt $deadline -and -not $webOk) { $webOk = Test-Url $Web 60; if (-not $webOk) { Start-Sleep -Seconds 2 } }
$apiOk = Test-Url $Api
Write-Host ("Backend : " + ($(if ($apiOk) { "ready  $Api" } else { "NOT ready - check its window for errors" })))
Write-Host ("Website : " + ($(if ($webOk) { "ready  $Web" } else { "NOT ready - check its window for errors" })))
if ($apiOk -and $webOk -and -not $NoBrowser) { Start-Process $Web }
