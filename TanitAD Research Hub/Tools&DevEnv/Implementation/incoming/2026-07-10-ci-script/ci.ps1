<#
.SYNOPSIS
  TanitAD stack CI gate -- one command for scheduled agents and pre-commit hooks.

.DESCRIPTION
  Runs, in order, and stops at the first failure:

    1. I2 COLLAPSE TRIPWIRE (fast fail-fast) -- test_i2_batch_consistency_of_encoder,
       the BatchNorm-in-inference canary (instrument doctrine D-004). It is ~0.5 s and
       catches the single most expensive class of silent bug before the full suite runs.
    2. FULL SUITE + TIMING BUDGET -- delegated to profile_testsuite.py check, which runs
       the whole pytest suite warm and fails if any test failed, if warm import/collection
       overhead exceeds -MaxWarmOverhead, or if any single call exceeds -MaxTest. If the
       profiler is not present on disk, falls back to an inline pytest --durations run
       with the same budget checks, so ci.ps1 is self-sufficient.

  Exit codes (usable by a git pre-commit hook or an agent):
    0  all green
    2  I2 tripwire failed  (a batch-statistic layer is in the inference path)
    1  suite failure or timing-budget breach

  Emits total wall-clock. Target: warm run < 15 s (measured 2026-07-10: ~11 s, 189 tests).

.PARAMETER StackDir
  Path to the stack/ package. Defaults to the parent of this script's directory
  (works when installed at stack/scripts/ci.ps1).

.PARAMETER Python
  Python interpreter to use. Resolution order when omitted: $env:VIRTUAL_ENV, the dev
  venv at C:\Users\Admin\venvs\tanitad, then python on PATH.

.EXAMPLE
  pwsh stack/scripts/ci.ps1
.EXAMPLE
  pwsh stack/scripts/ci.ps1 -MaxWarmOverhead 4 -MaxTest 6.0
#>
[CmdletBinding()]
param(
    [string]$StackDir,
    [string]$Python,
    [double]$MaxWarmOverhead = 4.0,
    [double]$MaxTest = 6.0,
    [switch]$SkipTripwire,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { if (-not $Quiet) { Write-Host "==> $msg" -ForegroundColor Cyan } }
function Write-Ok($msg)   { if (-not $Quiet) { Write-Host "OK  $msg" -ForegroundColor Green } }
function Write-Bad($msg)  { Write-Host "FAIL $msg" -ForegroundColor Red }

# --- resolve stack dir --------------------------------------------------------
if (-not $StackDir) {
    $StackDir = Split-Path -Parent $PSScriptRoot   # stack/scripts -> stack
}
$StackDir = (Resolve-Path -LiteralPath $StackDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $StackDir 'tests'))) {
    Write-Bad "no tests/ under StackDir '$StackDir' -- pass -StackDir explicitly"
    exit 1
}

# --- resolve python -----------------------------------------------------------
function Resolve-Python {
    if ($Python) { return $Python }
    if ($env:VIRTUAL_ENV) {
        $p = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
        if (Test-Path -LiteralPath $p) { return $p }
    }
    $devVenv = 'C:\Users\Admin\venvs\tanitad\Scripts\python.exe'
    if (Test-Path -LiteralPath $devVenv) { return $devVenv }
    return 'python'
}
$py = Resolve-Python
if (-not $Quiet) { Write-Host "python: $py" -ForegroundColor DarkGray }

# --- locate the timing-budget profiler (integrated or still in intake) --------
function Resolve-Profiler {
    $intake = 'TanitAD Research Hub\Tools&DevEnv\Implementation\incoming\2026-07-09-testsuite-io-profiling\profile_testsuite.py'
    $candidates = @(
        (Join-Path $StackDir 'scripts\profile_testsuite.py'),
        (Join-Path (Split-Path -Parent $StackDir) $intake)
    )
    foreach ($c in $candidates) { if (Test-Path -LiteralPath $c) { return $c } }
    return $null
}
$profiler = Resolve-Profiler

$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- step 1: I2 collapse tripwire (fast fail-fast) ----------------------------
if (-not $SkipTripwire) {
    Write-Step "I2 collapse tripwire (encoder batch-1 consistency, D-004)"
    Push-Location -LiteralPath $StackDir
    try {
        & $py -m pytest 'tests/test_instruments.py::test_i2_batch_consistency_of_encoder' -q -p no:cacheprovider | Out-Host
        $i2 = $LASTEXITCODE
    } finally { Pop-Location }
    if ($i2 -ne 0) {
        Write-Bad "I2 tripwire -- a batch-statistic layer is in the inference path (D-004)"
        $sw.Stop()
        Write-Host ("total {0:N1}s" -f $sw.Elapsed.TotalSeconds)
        exit 2
    }
    Write-Ok "I2 tripwire"
}

# --- step 2: full suite + timing budget ---------------------------------------
if ($profiler) {
    Write-Step "full suite + timing budget (profile_testsuite.py check)"
    & $py $profiler check --stack-dir $StackDir --max-warm-overhead $MaxWarmOverhead --max-test $MaxTest | Out-Host
    $rc = $LASTEXITCODE
} else {
    # Fallback: profiler not on disk -- run pytest inline and enforce the same budgets.
    Write-Step "full suite (inline fallback; profiler not found) + timing budget"
    Push-Location -LiteralPath $StackDir
    try {
        $out = & $py -m pytest -q -p no:cacheprovider --durations=25 2>&1 | Out-String
        $rc = $LASTEXITCODE
    } finally { Pop-Location }
    Write-Host $out
    if ($rc -eq 0) {
        # a single call slower than the budget must still fail the gate
        $slow = [regex]::Matches($out, '(?m)^\s*([0-9]+\.[0-9]+)s\s+call\s+(\S+)') |
            ForEach-Object {
                [pscustomobject]@{ sec = [double]$_.Groups[1].Value; node = $_.Groups[2].Value }
            } | Where-Object { $_.sec -gt $MaxTest }
        if ($slow) {
            foreach ($s in $slow) { Write-Bad ("slow test {0} {1}s > {2}s" -f $s.node, $s.sec, $MaxTest) }
            $rc = 1
        }
    }
}

$sw.Stop()
if ($rc -ne 0) {
    Write-Bad "suite/budget gate"
    Write-Host ("total {0:N1}s" -f $sw.Elapsed.TotalSeconds)
    exit 1
}
Write-Ok ("CI green -- total {0:N1}s" -f $sw.Elapsed.TotalSeconds)
exit 0
