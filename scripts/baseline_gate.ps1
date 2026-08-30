# baseline_gate.ps1 — Cordi v2 baseline validation gate
# Re-runs deterministic + live integration suites and asserts stable counts.
# Usage: powershell -File scripts/baseline_gate.ps1
param(
    [switch]$SkipLive
)

$ErrorActionPreference = 'Stop'
$basetemp = "C:\tmp\pytest_cordiiv2"
$logPath = Join-Path $PSScriptRoot "..\logs\baseline_gate.log"

function Ensure-LogDir {
    $logDir = Split-Path -Parent $logPath
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
}

function Write-Log([string]$line) {
    Ensure-LogDir
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts | $line" | Out-File -Append -FilePath $logPath
}

function Invoke-Pytest([string[]]$ExtraArgs) {
    $argsList = @("--basetemp", $basetemp) + $ExtraArgs + @("-q")
    $cmd = "pytest " + ($argsList -join ' ')
    Write-Host ">>> $cmd"
    $output = & pytest @argsList | Out-String
    Write-Host $output
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed with exit code $LASTEXITCODE"
    }
    return $output
}

Write-Host "==> Deterministic gate"
$det = Invoke-Pytest @()
if ($det -notmatch "(\d+) passed.*?(\d+) skipped") {
    throw "Could not parse passed/skipped counts from deterministic output`n$det"
}
$passed = [long]$Matches[1]
$skipped = [long]$Matches[2]
Write-Host "Passed: $passed, Skipped: $skipped"
if ($passed -lt 276) {
    $msg = "BASELINE FAIL: $passed passed (expected >=276)"
    Write-Host $msg
    Write-Log "deterministic=FAIL passed=$passed skipped=$skipped"
    throw $msg
}
if ($skipped -gt 7) {
    $msg = "BASELINE FAIL: $skipped skipped (expected <=7)"
    Write-Host $msg
    Write-Log "deterministic=FAIL passed=$passed skipped=$skipped"
    throw $msg
}
Write-Log "deterministic=OK passed=$passed skipped=$skipped"

$livePassed = $null
$liveSkipped = $null
if (-not $SkipLive) {
    Write-Host "`n==> Live integration gate"
    $live = Invoke-Pytest @("--live", "-k", "integration")
    if ($live -notmatch "(\d+) passed.*?(\d+) skipped") {
        throw "Could not parse passed/skipped counts from live output`n$live"
    }
    $livePassed = [long]$Matches[1]
    $liveSkipped = [long]$Matches[2]
    Write-Host "Live passed: $livePassed, skipped: $liveSkipped"
    if ($livePassed -lt 4) {
        $msg = "LIVE FAIL: $livePassed/4 integration tests passed (expected >=4)"
        Write-Host $msg
        Write-Log "live=FAIL passed=$livePassed skipped=$liveSkipped"
        throw $msg
    }
    Write-Log "live=OK passed=$livePassed skipped=$liveSkipped"
} else {
    Write-Host "`n==> Skipping live gate (--SkipLive)"
    Write-Log "deterministic=OK passed=$passed skipped=$skipped live=SKIPPED"
}

Write-Host "`n==> Baseline gate PASSED"

