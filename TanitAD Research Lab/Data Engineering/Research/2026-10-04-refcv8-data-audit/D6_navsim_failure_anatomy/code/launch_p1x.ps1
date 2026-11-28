# D6 P1' launcher (SPEC_P1P2_A2_P1PRIME.md) -- DETACHED (Start-Process -WindowStyle Hidden).
#
# A2 is REGISTERED (sha256 0ccd6c5aa4de3d56abd0c30907da9fbb509dcd347d209259aece891fe7f64705, raw/SPEC_SHA256.txt; marker raw/SPEC_A2_REGISTERED.txt).
# Queue order ruling (Master Mind): P2 chain (holds the lock now) -> the battery's step-50,400 four-family chain -> P1'. To keep that order the GATE waits for TWO markers:
#   raw/SPEC_A2_REGISTERED.txt (containing the A2 sha256)   AND   raw/P1X_GPU_GO.txt (created when the battery's chain has finished with the lock).
# Starts exactly two hidden processes:
#   1. the GATE (run_p1p2_gate.py, env-configured): when both markers exist it starts battery/code/with_gpu_lock.py --job refcv8-d6-p1p2 -- python run_p1p2_chain.py
#      with D6_STAGES=P1X (export of ALL 5,912 navhard scenes on ckpt_50400, md5 b418d0fc...).  with_gpu_lock WAITS for the lock and the smi gate; never touches a held lock.
#   2. run_p1x_cpu.py (CPU only): waits for raw/p1x_gpu_done.json and for the OFFICIAL 50,400 navhard scoring (counts.json PASS), builds the token sets by rule, writes
#      raw/p1x_sets_ready.json and WAITS for raw/P1X_SETS_ACK.txt before scoring any fan; then scores (<= 2 processes, free-commit gate 6 GB).
# Logs under raw/: gate_p1x.log, gpu_wait_p1x.log, gpu_rec_p1x.json, chain.log, chain_P1X.log, cpu_chain.log.  Kill by explicit PID only (raw/launch_pids_p1x.json).

$ErrorActionPreference = 'Stop'
$py   = 'C:\Users\Admin\venvs\tanitad\Scripts\python.exe'
$d6   = 'D:\Projects\TanitAD\TanitAD Research Lab\Data Engineering\Research\2026-10-04-refcv8-data-audit\D6_navsim_failure_anatomy'
$code = Join-Path $d6 'code'
$raw  = Join-Path $d6 'raw'
foreach ($p in @($py, (Join-Path $code 'run_p1p2_gate.py'), (Join-Path $code 'run_p1p2_chain.py'), (Join-Path $code 'run_p1x_cpu.py'), (Join-Path $code 'bridge_fix\run_bridge7.py'),
                 (Join-Path $raw 'SPEC_A2_REGISTERED.txt'), 'D:\refcv7_eval_kit\ckpt\ckpt_50400.pt')) {
    if (-not (Test-Path $p)) { throw "missing $p" }
}
function Q([string]$s) { '"' + $s + '"' }

$env:D6_GATE_MARK    = 'SPEC_A2_REGISTERED.txt'
$env:D6_GATE_SHA     = '0ccd6c5aa4de3d56abd0c30907da9fbb509dcd347d209259aece891fe7f64705'
$env:D6_GATE_GO      = 'P1X_GPU_GO.txt'
$env:D6_GATE_WAITLOG = 'gpu_wait_p1x.log'
$env:D6_GATE_REC     = 'gpu_rec_p1x.json'
$env:D6_GATE_LOG     = 'gate_p1x.log'
$env:D6_STAGES       = 'P1X'
$env:D6_DONE         = 'p1x_gpu_done.json'
$gate = Start-Process -FilePath $py -ArgumentList @((Q (Join-Path $code 'run_p1p2_gate.py'))) -WindowStyle Hidden -PassThru
Remove-Item Env:D6_GATE_MARK, Env:D6_GATE_SHA, Env:D6_GATE_GO, Env:D6_GATE_WAITLOG, Env:D6_GATE_REC, Env:D6_GATE_LOG, Env:D6_STAGES, Env:D6_DONE

$env:D6_GATE_GB = '6.0'
$env:D6_MAX_PAR = '2'
$cpu = Start-Process -FilePath $py -ArgumentList @((Q (Join-Path $code 'run_p1x_cpu.py'))) -WindowStyle Hidden -PassThru
Remove-Item Env:D6_GATE_GB, Env:D6_MAX_PAR

@{ gate_pid = $gate.Id; cpu_waiter_pid = $cpu.Id; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -Path (Join-Path $raw 'launch_pids_p1x.json') -Encoding ascii
"queued: gate_pid=$($gate.Id) cpu_waiter_pid=$($cpu.Id)"
