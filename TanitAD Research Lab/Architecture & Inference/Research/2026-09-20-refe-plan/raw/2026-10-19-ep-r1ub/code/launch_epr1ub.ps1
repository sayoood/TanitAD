# Launch EP-R1ub (or its C2 shuffled-label control) DETACHED, so it survives the launching session.  CPU only: it never touches the GPU lock.
# Same pattern as launch_epr1.ps1: WMI Win32_Process.Create, CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, stdin from NUL.
# Script execution is disabled on this box and -ExecutionPolicy Bypass is never used: paste the body into a PowerShell prompt (or run it as a scriptblock) instead of running the file.
#   .\launch_epr1ub.ps1 -Fraction 10          # EP-R1ub at f = 10 % (writes run\f10\epr1ub_result.json ONCE [+ epr1_result.json and the rows when a config is feasible]; the python refuses to overwrite)
#   .\launch_epr1ub.ps1 -Fraction 10 -C2      # the C2 shuffled-label negative control at c* (needs a FEASIBLE epr1_result.json; its own token)
#   add -DryRun to run every refusal AND the launch-gate verification, print the exact command line and stop before the WMI call.
#
# NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS (PI, 2026-09-26; PREREG_EPR1UB.md s6).  This launcher REFUSES unless epr1ub_gate.py VERIFIES a token for the EXACT command it is about to run:
# argv sha256, the bound code files' sha256 (+ the landed parents' blobs at the token's commit), the S0 bank manifest, the ranked logs, the P2 checkpoint, the S0 marker, the f10 identity reference.
# It launches THE ARGV THE GATE PRINTED (the thing verified is the thing run).  -Epochs 11, a different fraction, an edited epr1ub.py or a changed manifest therefore refuse.
# Mint the token first (landed parents deployed + the staged files in $CodeDir):
#   C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -I C:\Users\Admin\epr1ub\code\epr1ub_gate.py --fraction 10 --commit <40-hex landed commit of the parents>   [--stage c2]
param(
    [Parameter(Mandatory = $true)][ValidateSet(10)][int]$Fraction,
    [switch]$C2,
    [int]$Threads = 8,
    [int]$Epochs = 10,
    [int]$BatchSets = 256,
    [string]$CodeDir = "C:\Users\Admin\epr1ub\code",
    [string]$S0Run = "C:\Users\Admin\srttc_s0\run",
    [string]$Bank = "C:\Users\Admin\srttc_s0\work\bank",
    [string]$OutRoot = "C:\Users\Admin\epr1ub\run",
    [string]$GateRoot = "C:\Users\Admin\epr1ub\gate",
    [string]$GitDir = "C:/Users/Admin/tanitad-push/.git",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$gate = "$CodeDir\epr1ub_gate.py"
$out = "$OutRoot\f$Fraction"
$stage = if ($C2) { "c2" } else { "run" }
$token = "$GateRoot\f${Fraction}_$stage\epr1ub_gate_token.json"

if (-not (Test-Path "$S0Run\S0_DONE")) { throw "REFUSED: $S0Run\S0_DONE does not exist. S0 is not done." }
if (-not (Test-Path "$Bank\MANIFEST.sha256.json")) { throw "REFUSED: $Bank\MANIFEST.sha256.json does not exist." }
if (-not (Test-Path "$CodeDir\epr1ub.py")) { throw "REFUSED: $CodeDir\epr1ub.py is not deployed." }
if (-not (Test-Path $gate)) { throw "REFUSED: $gate is not deployed (no training launch without a launch-gate pass)." }
if ($C2) {
    if (-not (Test-Path "$out\epr1_result.json")) { throw "REFUSED: $out\epr1_result.json does not exist: no feasible c* to run C2 on." }
    if (Test-Path "$out\epr1_c2_result.json") { throw "REFUSED: $out\epr1_c2_result.json already exists (this read is written once)." }
    $log = "$out\epr1ub_c2.log"
} else {
    foreach ($f in @("epr1ub_result.json", "epr1_result.json")) {
        if (Test-Path "$out\$f") { throw "REFUSED: $out\$f already exists. c* is chosen ONCE and never re-drawn; move the directory away to rerun." }
    }
    $log = "$out\epr1ub.log"
}

# --- the launch gate: verify the signed token against the exact command; the gate prints ONE json line {ok, reasons, argv, ...} (exit 0 = verified, 3 = refused)
$gargs = @("-I", $gate, "verify", "--stage", $stage, "--fraction", "$Fraction", "--python", $py, "--code-dir", $CodeDir, "--out-dir", $out,
           "--threads", "$Threads", "--epochs", "$Epochs", "--batch-sets", "$BatchSets", "--token", $token, "--git-dir", $GitDir)
$raw = & $py @gargs
$rc = $LASTEXITCODE
$line = @($raw | Where-Object { ($_ -is [string]) -and $_.TrimStart().StartsWith("{") }) | Select-Object -Last 1
if (-not $line) { throw "REFUSED: the launch gate printed no verdict (exit code $rc). No training launch without a launch-gate pass." }
$verdict = $line | ConvertFrom-Json
if (($rc -ne 0) -or (-not $verdict.ok)) { throw ("REFUSED by the launch gate (exit code $rc): " + (@($verdict.reasons) -join " | ")) }
$argv = @($verdict.argv)
if ($argv.Count -lt 5) { throw "REFUSED: the launch gate returned an implausible argv ($($argv.Count) elements)." }

function Quote-Arg([string]$s) { if ($s -match '[\s\\/]') { '"' + $s + '"' } else { $s } }
$cmdline = 'cmd.exe /c "' + ((@($argv) | ForEach-Object { Quote-Arg $_ }) -join ' ') + ' < NUL >> "' + $log + '" 2>&1"'
if ($DryRun) {
    "DRY RUN (gate verified; not launched): $cmdline"
} else {
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $CodeDir; ProcessStartupInformation = $si }
    if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
    "launched cmd.exe pid $($r.ProcessId) ($stage, f = $Fraction %); log $log"
}
