<#
.SYNOPSIS
  TanitAD one-command test gate (Windows). Thin wrapper over tools/ci_gate.py.

.DESCRIPTION
  Activates the off-Drive venv if present, then runs the stack/ pytest suite
  through ci_gate.py, which fails on any test failure, COLLECTION ERROR,
  per-test slowness, total-wall blow-out, a missing/failing I2 tripwire, a
  required SUITE below its manifest floor, a total-collected count under
  --min-total, or (with -GpuSmoke require) a CUDA device-parity failure.

  Run from the repo root before every commit/push:
      .\tools\ci.ps1
      .\tools\ci.ps1 -GpuSmoke require            # also gate the CUDA path
      .\tools\ci.ps1 -MaxTestSeconds 4 -- -k comma2k19

.NOTES
  On a pod (Linux) call the core directly:  python tools/ci_gate.py --rootdir stack
#>
[CmdletBinding()]
param(
    [double]$MaxTestSeconds = 15.0,
    [double]$MaxWallSeconds = 150.0,
    [ValidateSet("off", "warn", "require")]
    [string]$GpuSmoke = "warn",
    [string]$Json = "",
    [string]$Venv = "$env:USERPROFILE\venvs\tanitad\Scripts\Activate.ps1",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$gate = Join-Path $here "ci_gate.py"
$repoRoot = Split-Path -Parent $here            # tools/ -> repo root
$stackRoot = Join-Path $repoRoot "stack"

if (Test-Path $Venv) { . $Venv }

$argList = @($gate,
    "--rootdir", $stackRoot,
    "--max-test-seconds", $MaxTestSeconds,
    "--max-wall-seconds", $MaxWallSeconds,
    "--gpu-smoke", $GpuSmoke)
if ($Json) { $argList += @("--json", $Json) }
if ($PytestArgs) { $argList += $PytestArgs }

& python @argList
exit $LASTEXITCODE
