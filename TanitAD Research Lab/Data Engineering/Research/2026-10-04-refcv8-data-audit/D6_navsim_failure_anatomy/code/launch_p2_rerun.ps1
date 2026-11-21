# D6 P2 + K7 RE-RUN launcher -- DETACHED (Start-Process -WindowStyle Hidden).
#
# Why: the first P2/K7 attempt died after ~2 s with "other python compute app(s) on the card" (two foreign Python313 processes appeared under our lock at 15:12;
# the bridge's own guard refused to share the GPU). No P2/K7 row was written, no number exists. The chain now waits and retries on exactly that message
# (code/run_p1p2_chain.py, disclosed in RESULT s13). P1 is complete and is NOT re-run (D6_STAGES=P2,K7).
#
# Starts exactly two hidden processes:
#   1. battery/code/with_gpu_lock.py --job refcv8-d6-p1p2 -- python run_p1p2_chain.py   (WAITS for the lock and the battery's smi gate; never touches a held lock)
#   2. run_p1p2_cpu.py (CPU only): waits for raw/p2_gpu_done.json, then re-scores the P2/K7 plans on the CPU -> raw/p2_cpu_done.json

$ErrorActionPreference = 'Stop'
$py   = 'C:\Users\Admin\venvs\tanitad\Scripts\python.exe'
$d6   = 'D:\Projects\TanitAD\TanitAD Research Lab\Data Engineering\Research\2026-10-04-refcv8-data-audit\D6_navsim_failure_anatomy'
$code = Join-Path $d6 'code'
$raw  = Join-Path $d6 'raw'
$wr   = 'D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-09-28-refcv7-standard-tests\battery\code\with_gpu_lock.py'
foreach ($p in @($py, $wr, (Join-Path $code 'run_p1p2_chain.py'), (Join-Path $code 'run_p1p2_cpu.py'), (Join-Path $code 'bridge_fix\run_bridge7.py'))) {
    if (-not (Test-Path $p)) { throw "missing $p" }
}
function Q([string]$s) { '"' + $s + '"' }

$env:D6_STAGES = 'P2,K7'
$env:D6_DONE   = 'p2_gpu_done.json'
$gpuArgs = @((Q $wr), '--job', 'refcv8-d6-p1p2', '--log', (Q (Join-Path $raw 'gpu_wait_p2.log')), '--rec', (Q (Join-Path $raw 'gpu_rec_p2.json')),
             '--max-wait-s', '43200', '--child-timeout-s', '14400', '--', (Q $py), (Q (Join-Path $code 'run_p1p2_chain.py')))
$gpu = Start-Process -FilePath $py -ArgumentList $gpuArgs -WindowStyle Hidden -PassThru
Remove-Item Env:D6_STAGES, Env:D6_DONE

$env:D6_CPU_STAGES = 'P2'
$env:D6_WAIT       = 'p2_gpu_done.json'
$env:D6_CPU_DONE   = 'p2_cpu_done.json'
$cpu = Start-Process -FilePath $py -ArgumentList @((Q (Join-Path $code 'run_p1p2_cpu.py'))) -WindowStyle Hidden -PassThru
Remove-Item Env:D6_CPU_STAGES, Env:D6_WAIT, Env:D6_CPU_DONE

@{ gpu_queue_pid = $gpu.Id; cpu_waiter_pid = $cpu.Id; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -Path (Join-Path $raw 'launch_pids_p2.json') -Encoding ascii
"queued: gpu_queue_pid=$($gpu.Id) cpu_waiter_pid=$($cpu.Id)"
