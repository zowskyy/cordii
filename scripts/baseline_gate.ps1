# baseline_gate.ps1 — Cordi v2 baseline validation gate
# Re-runs deterministic + live integration suites and asserts stable counts.
# Usage: powershell -File scripts/baseline_gate.ps1
param(
    [switch]$SkipLive
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$basetemp = Join-Path $repoRoot "tmp\pytest_cordiiv2"

function Invoke-Pytest([string[]]$ExtraArgs) {
    $argsList = @("--basetemp", $basetemp) + $ExtraArgs + @("-q")
    $cmd = "pytest " + ($argsList -join ' ')
    Write-Host ">>> $cmd"
    $output = & pytest @argsList
    Write-Host $output
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed with exit code $LASTEXITCODE"
    }
    return $output
}

Write-Host "==> Deterministic gate"
$det = Invoke-Pytest @()
if ($det -notmatch "(\d+) passed") {
    throw "Could not parse passed count from deterministic output"
}
$passed = [long]$Matches[1]
Write-Host "Passed: $passed"
if ($passed -lt 276) {
    throw "Baseline regression: expected >=276 passed, got $passed"
}

if (-not $SkipLive) {
    Write-Host "`n==> Live integration gate"
    $live = Invoke-Pytest @("--live", "-k", "integration")
    if ($live -notmatch "(\d+) passed") {
        throw "Could not parse passed count from live output"
    }
    $livePassed = [long]$Matches[1]
    Write-Host "Live passed: $livePassed"
    if ($livePassed -lt 4) {
        throw "Baseline regression: expected >=4 live passed, got $livePassed"
    }
} else {
    Write-Host "`n==> Skipping live gate (--SkipLive)"
}

Write-Host "`n==> Baseline gate PASSED"
