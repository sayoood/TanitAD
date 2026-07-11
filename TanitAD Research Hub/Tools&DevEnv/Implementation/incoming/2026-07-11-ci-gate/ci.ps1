<#
.SYNOPSIS
    TanitAD commit gate (Tools&DevEnv backlog #3) - one command an agent or a
    pre-commit hook runs before committing to `stack/`.

.DESCRIPTION
    Two gates, fail-fast, measured:

      1. I2 tripwire  (~2 s) - encoder batch-1 == batch-B invariant on the real
         WorldModel. Deployment is batch-1 streaming on Orin; a batch-statistic
         layer is silent in training and breaks the exported engine. Runs FIRST
         so this whole class of bug fails before the slow suite.
      2. Suite + timing guard - the full pytest suite (gate G-E) run THROUGH
         `profile_testsuite.py check`, which fails the commit if any test fails,
         if warm import/collection overhead exceeds -MaxWarmOverhead, or if any
         single `call` test exceeds -MaxTest (catches a newly-added slow fixture).
         If the profiler is not found, falls back to plain `pytest -q` (timing
         guard skipped, with a warning).

    Exit code is nonzero if EITHER gate fails. Total wall-clock is reported.

.PARAMETER StackDir
    Path to the repo `stack/` dir. Auto-detected by walking up from this script
    if omitted.
.PARAMETER Python
    Python interpreter. Defaults to the off-Drive `tanitad` venv if present, else
    `python` on PATH.
.PARAMETER MaxWarmOverhead
    Budget (s) for warm wall-minus-reported-test time. Default 4.0.
.PARAMETER MaxTest
    Budget (s) for any single `call` test. Default 6.0.
.PARAMETER SkipTiming
    Run plain `pytest -q` instead of the profiler timing guard.

.EXAMPLE
    pwsh -File ci.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ci.ps1 -MaxTest 4.0
#>
[CmdletBinding()]
param(
    [string]$StackDir,
    [string]$Python,
    [double]$MaxWarmOverhead = 4.0,
    [double]$MaxTest = 6.0,
    [switch]$SkipTiming
)

$ErrorActionPreference = 'Stop'
$scriptDir = $PSScriptRoot

function Find-StackDir([string]$start) {
    $d = Get-Item -LiteralPath $start
    while ($null -ne $d) {
        $cand = Join-Path $d.FullName 'stack'
        if (Test-Path (Join-Path $cand 'tanitad')) { return (Resolve-Path $cand).Path }
        if (Test-Path (Join-Path $d.FullName 'tanitad')) { return $d.FullName }
        $d = $d.Parent
    }
    return $null
}

function Resolve-Python([string]$explicit) {
    if ($explicit) { return $explicit }
    $venv = 'C:\Users\Admin\venvs\tanitad\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }
    return 'python'
}

# --- resolve environment ---------------------------------------------------------
if (-not $StackDir) { $StackDir = Find-StackDir $scriptDir }
if (-not $StackDir -or -not (Test-Path (Join-Path $StackDir 'tanitad'))) {
    Write-Error "Could not locate stack/ (with a tanitad package). Pass -StackDir."
    exit 3
}
$py = Resolve-Python $Python
$tripwire = Join-Path $scriptDir 'ci_i2_tripwire.py'

# profile_testsuite.py: prefer the sibling (integrated at stack/scripts/), else the
# pending intake location, else fall back to plain pytest.
$profiler = $null
foreach ($cand in @(
        (Join-Path $scriptDir 'profile_testsuite.py'),
        (Join-Path $StackDir 'scripts\profile_testsuite.py'),
        (Join-Path (Split-Path $scriptDir -Parent) '2026-07-09-testsuite-io-profiling\profile_testsuite.py')
    )) {
    if (Test-Path $cand) { $profiler = $cand; break }
}

Write-Host "=== TanitAD CI gate ===" -ForegroundColor Cyan
Write-Host "  python : $py"
Write-Host "  stack  : $StackDir"
Write-Host "  profiler: $(if ($profiler) { $profiler } else { '<not found -> plain pytest, timing guard SKIPPED>' })"
Write-Host ""

$overall = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- gate 1: I2 tripwire ---------------------------------------------------------
Write-Host "[1/2] I2 batch-consistency tripwire ..." -ForegroundColor Yellow
$t1 = [System.Diagnostics.Stopwatch]::StartNew()
& $py $tripwire --stack-dir $StackDir
$rc1 = $LASTEXITCODE
$t1.Stop()
Write-Host ("      -> exit {0} in {1:N1}s" -f $rc1, $t1.Elapsed.TotalSeconds)
if ($rc1 -ne 0) { $overall = 1 }

# --- gate 2: suite + timing guard ------------------------------------------------
Write-Host "[2/2] pytest suite + timing guard ..." -ForegroundColor Yellow
$t2 = [System.Diagnostics.Stopwatch]::StartNew()
if ($profiler -and -not $SkipTiming) {
    & $py $profiler check --stack-dir $StackDir `
        --max-warm-overhead $MaxWarmOverhead --max-test $MaxTest
    $rc2 = $LASTEXITCODE
} else {
    if ($SkipTiming) { Write-Host "      (timing guard skipped by -SkipTiming)" }
    Push-Location $StackDir
    & $py -m pytest -q
    $rc2 = $LASTEXITCODE
    Pop-Location
}
$t2.Stop()
Write-Host ("      -> exit {0} in {1:N1}s" -f $rc2, $t2.Elapsed.TotalSeconds)
if ($rc2 -ne 0) { $overall = 1 }

$sw.Stop()
Write-Host ""
$verdict = if ($overall -eq 0) { 'PASS' } else { 'FAIL' }
$color = if ($overall -eq 0) { 'Green' } else { 'Red' }
Write-Host ("=== CI gate {0} - total {1:N1}s (I2 {2:N1}s + suite {3:N1}s) ===" -f `
        $verdict, $sw.Elapsed.TotalSeconds, $t1.Elapsed.TotalSeconds, $t2.Elapsed.TotalSeconds) `
    -ForegroundColor $color
exit $overall
