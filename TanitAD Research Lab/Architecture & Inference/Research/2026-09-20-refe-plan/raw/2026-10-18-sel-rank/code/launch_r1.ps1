# Launch SR-TTC R1 (or the A2.4 learning-curve read) DETACHED, so it survives the launching session.  CPU only: it never touches the GPU lock.
# Same pattern as launch_s0.ps1: WMI Win32_Process.Create, CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, stdin from NUL (a job started from a session dies with it otherwise).
# Script execution is disabled on this box and -ExecutionPolicy Bypass is never used: paste the body into a PowerShell prompt instead of running the file.
#   .\launch_r1.ps1 -Fraction 10             # R1 at the nested size f = 10 %  (writes run\f10\r1_result.json ONCE; the python refuses to overwrite it)
#   .\launch_r1.ps1 -Fraction 10 -Lc         # the A2.4 learning-curve read at c* (needs r1_result.json of the same fraction and the same bank manifest)
#   add -DryRun to run every refusal AND the launch-gate verification, print the exact command line and stop before the WMI call.
#
# NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS (PI, 2026-09-26; MM_DECISIONS_R1.md row 11).  This launcher REFUSES unless srttc_r1_gate.py VERIFIES a token for the EXACT command it is about to run:
#   1. it asks the gate to rebuild the R1 command from -Fraction/-Threads/-Epochs/-BatchSets (+ the registered seeds / criterion / both-rule, all NAMED in the argv) and to verify the signed token
#      r1_gate_token.json against it: argv sha256, the code files' sha256 and their git blobs at the token's commit, the S0 bank manifest, the ranked logs, the P2 checkpoint, the S0 marker;
#   2. it launches THE ARGV THE GATE PRINTED (the thing verified is the thing run).  -Epochs 11, a different fraction, an edited srttc_r1.py or a changed manifest therefore refuse.
# Mint the token first (after S0_DONE, with the landed files deployed to $CodeDir):
#   C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -I C:\Users\Admin\srttc_r1\code\srttc_r1_gate.py --fraction 10 --commit <40-hex landed commit>
# It also still REFUSES to start unless S0 is done: C:\Users\Admin\srttc_s0\run\S0_DONE and the bank manifest (written by the S0 post-watcher after S0_DONE) must both exist.
# The python code is the landed copy under C:\Users\Admin\srttc_r1\code (srttc_r1.py); the python itself re-checks the marker, every bank unit's sha256, and the dev-gate / navtest path guard.
# The -CodeDir/-S0Run/-Bank/-OutRoot/-GateRoot/-GitDir parameters exist so the refusals can be tested against a scratch tree; the gate re-validates every path against the registered one.
param(
    [Parameter(Mandatory = $true)][ValidateSet(10, 25, 50, 100)][int]$Fraction,
    [switch]$Lc,
    [int]$Threads = 8,
    [int]$Epochs = 10,
    [int]$BatchSets = 256,
    [string]$CodeDir = "C:\Users\Admin\srttc_r1\code",
    [string]$S0Run = "C:\Users\Admin\srttc_s0\run",
    [string]$Bank = "C:\Users\Admin\srttc_s0\work\bank",
    [string]$OutRoot = "C:\Users\Admin\srttc_r1\run",
    [string]$GateRoot = "C:\Users\Admin\srttc_r1\gate",
    [string]$GitDir = "C:/Users/Admin/tanitad-push/.git",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$gate = "$CodeDir\srttc_r1_gate.py"
$out = "$OutRoot\f$Fraction"
$stage = if ($Lc) { "lc" } else { "run" }
$token = "$GateRoot\f${Fraction}_$stage\r1_gate_token.json"

if (-not (Test-Path "$S0Run\S0_DONE")) { throw "REFUSED: $S0Run\S0_DONE does not exist. S0 is not done; R1 does not start (a S0_FAILED marker or a live driver means wait or fix S0 first)." }
if (-not (Test-Path "$Bank\MANIFEST.sha256.json")) { throw "REFUSED: $Bank\MANIFEST.sha256.json does not exist yet. The S0 post-watcher writes it after S0_DONE; wait for FINALIZED.txt in $S0Run." }
if (-not (Test-Path "$CodeDir\srttc_r1.py")) { throw "REFUSED: $CodeDir\srttc_r1.py is not deployed. Copy the landed srttc_r1.py there first." }
if (-not (Test-Path $gate)) { throw "REFUSED: $gate is not deployed. Copy the landed srttc_r1_gate.py there first (no training launch without a launch-gate pass)." }
if ($Lc) {
    if (-not (Test-Path "$out\r1_result.json")) { throw "REFUSED: $out\r1_result.json does not exist; run R1 at this fraction first." }
    if (Test-Path "$out\lc_result.json") { throw "REFUSED: $out\lc_result.json already exists (the LC read is written once)." }
    $sub = "lc"; $log = "$out\lc.log"
} else {
    if (Test-Path "$out\r1_result.json") { throw "REFUSED: $out\r1_result.json already exists. c* is chosen ONCE and never re-drawn; move the directory away to rerun." }
    $sub = "run"; $log = "$out\r1.log"
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
    "launched cmd.exe pid $($r.ProcessId) ($sub, f = $Fraction %); log $log"
}
