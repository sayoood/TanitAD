<#
.SYNOPSIS
  TanitAD one-command test gate (Windows). Thin wrapper over scripts/ci_gate.py.

.DESCRIPTION
  Activates the off-Drive venv if present, then runs the pytest suite through
  ci_gate.py, which fails on any test failure, COLLECTION ERROR, per-test
  slowness (>6 s), total-wall blow-out (>90 s), or a missing/failing I2 tripwire.

  Run from the stack/ dir before every commit/push:
      .\scripts\ci.ps1
      .\scripts\ci.ps1 -MaxTestSeconds 4 -- -k comma2k19   # tighter + pytest passthrough

.NOTES
  On the pod (Linux) call the core directly:  python scripts/ci_gate.py
#>
[CmdletBinding()]
param(
    [double]$MaxTestSeconds = 15.0,
    [double]$MaxWallSeconds = 90.0,
    [string]$Venv = "$env:USERPROFILE\venvs\tanitad\Scripts\Activate.ps1",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$gate = Join-Path $here "ci_gate.py"
$stackRoot = Split-Path -Parent $here          # scripts/ -> stack/

if (Test-Path $Venv) { . $Venv }

$argList = @($gate,
    "--rootdir", $stackRoot,
    "--max-test-seconds", $MaxTestSeconds,
    "--max-wall-seconds", $MaxWallSeconds)
if ($PytestArgs) { $argList += $PytestArgs }

& python @argList
exit $LASTEXITCODE
