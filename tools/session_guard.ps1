<#
.SYNOPSIS
  TanitAD D-026 session-end guard (Windows). Thin wrapper over tools/session_guard.py.

.DESCRIPTION
  Run at the END of every agent session, from the repo/worktree root, before you
  declare the session done. It BLOCKS (non-zero exit) on uncommitted deliverable
  files in the hub areas, and WARNs on unmerged agent/* branches vs the tip and on
  INTAKE packages whose orchestrator verdict is still unfilled and older than the
  age budget. Use -Strict to make the warnings block too.

      .\tools\session_guard.ps1                 # gate this worktree
      .\tools\session_guard.ps1 -Strict          # branches + stale INTAKEs also block
      .\tools\session_guard.ps1 -Base origin/main
      .\tools\session_guard.ps1 -Json            # machine-readable

.NOTES
  On the pod (Linux) call the core directly:  python tools/session_guard.py
#>
[CmdletBinding()]
param(
    [string]$Base = "HEAD",
    [int]$MaxIntakeAgeDays = 3,
    [switch]$Strict,
    [switch]$Json,
    [string]$Venv = "$env:USERPROFILE\venvs\tanitad\Scripts\Activate.ps1"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$guard = Join-Path $here "session_guard.py"

if (Test-Path $Venv) { . $Venv }

$argList = @($guard, "--base", $Base, "--max-intake-age-days", $MaxIntakeAgeDays)
if ($Strict) { $argList += "--strict" }
if ($Json) { $argList += "--json" }

& python @argList
exit $LASTEXITCODE
