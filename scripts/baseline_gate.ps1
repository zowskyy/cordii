# baseline_gate.ps1 — Cordi v2 baseline validation gate
# Re-runs deterministic + live integration suites and asserts stable counts.
# Usage: powershell -File scripts/baseline_gate.ps1
param(
    [switch]$SkipLive
)

$ErrorActionPreference = 'Stop'
$basetemp = "C:\tmp\pytest_cordiiv2"
$logPath = Join-Path $PSScriptRoot "..\logs\baseline_gate.log"

# Thresholds (update when the baseline contract changes)
$MIN_PASSED = 288
$MAX_SKIPPED = 8
$MIN_LIVE_PASSED = 4


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
    $argsList = @("--basetemp", $basetemp, "--tb=no") + $ExtraArgs + @("-q")
    $cmd = "pytest " + ($argsList -join ' ')
    Write-Host ">>> $cmd"
    $output = & pytest @argsList 2>&1 | Out-String
    Write-Host $output
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed with exit code $LASTEXITCODE"
    }
    return $output
}


# Calibration validation (runs before tests)
Write-Host "`n==> Calibration validation"
$calibOutput = & python scripts/capacity_calculator.py --model 1.5b 2>&1 | Out-String
Write-Host $calibOutput
if ($LASTEXITCODE -ne 0) {
    $msg = "CALIBRATION FAIL: capacity_calculator.py --model 1.5b failed"
    Write-Host $msg -ForegroundColor Red
    Write-Log "calibration=FAIL"
    throw $msg
}
Write-Log "calibration=OK"


Write-Host "`n==> Deterministic gate"
$det = Invoke-Pytest @()
# Robust parsing: look for "X passed" and "Y skipped" anywhere in output
if (-not ($det -match "(\d+) passed")) {
    throw "Could not parse 'passed' count from deterministic output`n$det"
}
$passed = [long]$Matches[1]

if (-not ($det -match "(\d+) skipped")) {
    throw "Could not parse 'skipped' count from deterministic output`n$det"
}
$skipped = [long]$Matches[1]

Write-Host "Passed: $passed, Skipped: $skipped"
if ($passed -lt $MIN_PASSED) {
    $msg = "BASELINE FAIL: $passed passed (expected >=$MIN_PASSED)"
    Write-Host $msg -ForegroundColor Red
    Write-Log "deterministic=FAIL passed=$passed skipped=$skipped"
    throw $msg
}
if ($skipped -gt $MAX_SKIPPED) {
    $msg = "BASELINE FAIL: $skipped skipped (expected <=$MAX_SKIPPED)"
    Write-Host $msg -ForegroundColor Red
    Write-Log "deterministic=FAIL passed=$passed skipped=$skipped"
    throw $msg
}
Write-Log "deterministic=OK passed=$passed skipped=$skipped"


$livePassed = $null
$liveSkipped = $null
if (-not $SkipLive) {
    Write-Host "`n==> Live integration gate"
    $live = Invoke-Pytest @("--live", "-k", "integration")
    if (-not ($live -match "(\d+) passed")) {
        throw "Could not parse 'passed' count from live output`n$live"
    }
    $livePassed = [long]$Matches[1]

    if (-not ($live -match "(\d+) skipped")) {
        throw "Could not parse 'skipped' count from live output`n$live"
    }
    $liveSkipped = [long]$Matches[1]

    Write-Host "Live passed: $livePassed, skipped: $liveSkipped"
    if ($livePassed -lt $MIN_LIVE_PASSED) {
        $msg = "LIVE FAIL: $livePassed/$MIN_LIVE_PASSED integration tests passed (expected >=$MIN_LIVE_PASSED)"
        Write-Host $msg -ForegroundColor Red
        Write-Log "live=FAIL passed=$livePassed skipped=$liveSkipped"
        throw $msg
    }
    Write-Log "live=OK passed=$livePassed skipped=$liveSkipped"
} else {
    Write-Host "`n==> Skipping live gate (--SkipLive)"
    Write-Log "deterministic=OK passed=$passed skipped=$skipped live=SKIPPED"
}


$summary = "baseline=OK passed=$passed skipped=$skipped live=$(if ($null -ne $livePassed) { $livePassed } else { 'SKIPPED' })"
Write-Host "`n==> $summary" -ForegroundColor Green
Write-Log $summary
