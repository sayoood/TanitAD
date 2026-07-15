<#
.SYNOPSIS
  TanitAD one-command commit gate (Tools&DevEnv backlog P0 #1).

.DESCRIPTION
  Activates the tanitad venv and runs scripts/ci_check.py: the I2 tripwire
  (fail-fast), then the (quick or full) test suite with a per-test latency
  budget and an optional warm-wall budget. Exit 0 = safe to commit.

  All real logic lives in ci_check.py (unit-tested in tests/test_ci.py); this
  wrapper only resolves the interpreter and forwards flags, so Windows agents
  and pre-commit hooks have a single entry point.

.EXAMPLE
  pwsh stack/scripts/ci.ps1            # full suite gate
  pwsh stack/scripts/ci.ps1 -Quick     # curated pre-commit safety subset
  pwsh stack/scripts/ci.ps1 -WarmBudgetS 30   # also fail if warm wall > 30 s
#>
[CmdletBinding()]
param(
    [switch]$Quick,                 # curated safety subset (fast pre-commit gate)
    [double]$SlowTestS = 6.0,       # per-test call-duration budget (seconds)
    [double]$WarmBudgetS = 0.0,     # 0 = no wall budget; >0 fails if warm wall exceeds it
    [string]$Python = ""            # override interpreter; default = venv, else PATH python
)

$ErrorActionPreference = "Stop"

# stack/ is the parent of scripts/ ; pytest testpaths are relative to it.
$stackDir = Split-Path -Parent $PSScriptRoot
$ciCheck  = Join-Path $PSScriptRoot "ci_check.py"

# Resolve the interpreter: explicit override > active venv > known tanitad venv > PATH.
if (-not $Python) {
    if ($env:VIRTUAL_ENV) {
        $Python = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    } elseif (Test-Path "C:\Users\Admin\venvs\tanitad\Scripts\python.exe") {
        $Python = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
    } else {
        $Python = "python"
    }
}

$ciArgs = @("--slow-test-s", $SlowTestS, "--warm-budget-s", $WarmBudgetS)
if ($Quick) { $ciArgs += "--quick" }

Push-Location $stackDir
try {
    & $Python $ciCheck @ciArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
