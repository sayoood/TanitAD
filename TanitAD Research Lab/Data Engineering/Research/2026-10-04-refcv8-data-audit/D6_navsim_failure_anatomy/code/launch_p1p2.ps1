# D6 P1 + P2 launcher -- DETACHED (Start-Process -WindowStyle Hidden), so it survives the console that runs it.
#
# SPEC_P1P2.md was registered by the Master Mind (2026-10-04T12:00:42Z). Amendment SPEC_P1P2_A1.md (universe = FAN117) is held back by the GATE below.
#
# Starts exactly two hidden processes:
#   1. the GATE:  run_p1p2_gate.py -- polls for raw/SPEC_A1_REGISTERED.txt (must CONTAIN the A1 sha256). Only then does it start
#      battery/code/with_gpu_lock.py --job refcv8-d6-p1p2 ... -- python run_p1p2_chain.py
#      with_gpu_lock WAITS for the dev-box GPU lock (held by the battery's step-50,400 chain, job refcv7-milestone-step50400) and never touches a held lock;
#      the chain then runs bridge_fix/run_bridge7.py (P1 export, P2 arms, K7) while the wrapper holds the lock, and the wrapper releases it.
#      Until the marker exists NOTHING touches the lock or the GPU.
#   2. the CPU stage:  run_p1p2_cpu.py -- polls for raw/p1p2_gpu_done.json, then scores on the CPU (<= 3 processes, RAM-gated). Never touches the GPU.
# Logs under raw/: gate.log, gpu_wait.log (the lock-wait lines), gpu_rec.json, chain.log, cpu_chain.log.  Kill by explicit PID only (raw/launch_pids.json).

$ErrorActionPreference = 'Stop'
$py   = 'C:\Users\Admin\venvs\tanitad\Scripts\python.exe'
$d6   = 'D:\Projects\TanitAD\TanitAD Research Lab\Data Engineering\Research\2026-10-04-refcv8-data-audit\D6_navsim_failure_anatomy'
$code = Join-Path $d6 'code'
$raw  = Join-Path $d6 'raw'
$wr   = 'D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-09-28-refcv7-standard-tests\battery\code\with_gpu_lock.py'
foreach ($p in @($py, $wr, (Join-Path $code 'run_p1p2_gate.py'), (Join-Path $code 'run_p1p2_chain.py'), (Join-Path $code 'run_p1p2_cpu.py'), (Join-Path $code 'bridge_fix\run_bridge7.py'))) {
    if (-not (Test-Path $p)) { throw "missing $p" }
}
function Q([string]$s) { '"' + $s + '"' }

$gate = Start-Process -FilePath $py -ArgumentList @((Q (Join-Path $code 'run_p1p2_gate.py'))) -WindowStyle Hidden -PassThru
$cpu  = Start-Process -FilePath $py -ArgumentList @((Q (Join-Path $code 'run_p1p2_cpu.py'))) -WindowStyle Hidden -PassThru
@{ gate_pid = $gate.Id; cpu_waiter_pid = $cpu.Id; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -Path (Join-Path $raw 'launch_pids.json') -Encoding ascii
"queued: gate_pid=$($gate.Id) cpu_waiter_pid=$($cpu.Id)"
