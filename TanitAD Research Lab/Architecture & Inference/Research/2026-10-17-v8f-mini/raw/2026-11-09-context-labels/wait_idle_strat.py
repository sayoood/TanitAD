# Local copy of _rlfan_work/wait_idle.py (2026-11-08, Master Mind). Only change: the CPU-only test suite run_regr.py names test_*train.py files in its argv, which matched the PAT "train.py" as a false positive. Every other guard is unchanged, and the shared lock is still taken.
"""Block until the shared dev-box GPU is genuinely idle (no other GPU job process, < 700 MiB used, 4 polls in a row), then exit 0.
Another agent's `ladder_eval_diag.py` runs here; its memory use dips to ~80 MiB between rows, so a single reading proves nothing."""
import subprocess
import sys
import time

# 2026-10-12: 700 MiB never cleared -- the PI's desktop apps (Chrome, VLC, ChatGPT, Snipping Tool) hold ~750 MiB with no GPU job running.
# The job-process check is the real guard; the memory bar only has to sit well below a job's ~5 GB.
# 2026-10-15 23:00 (Master Mind): 2000 never cleared either -- the desktop now holds ~2.5-2.6 GB with NO GPU job (P2 table done, probe waiting).
# Raised to 3500 = aux_navtest_forward.GPU_MAX_USED_MIB, the bar acquire_gpu_lock enforces AFTER taking the lock.  The PAT process check is unchanged.
MAX_MIB = 3500
PAT = ("ladder_eval_diag", "scorer_finetune", "rl_fan_dump", "onpolicy_dump", "train.py", "eval_checkpoint", "lgdiag")


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    lines = out.strip().splitlines()
    return int(lines[0]) if lines and lines[0].strip().isdigit() else 10 ** 9   # unreadable = NOT idle (2026-10-12: an empty read crashed the waiter and the wrapper ran anyway)


def others():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | ForEach-Object { $_.CommandLine }")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    me = " ".join(sys.argv)
    return [l for l in out.splitlines() if any(p in l for p in PAT) and "wait_idle" not in l and "run_regr.py" not in l and l.strip() != me]


idle = 0
t0 = time.time()
while True:
    u, o = used_mib(), others()
    idle = idle + 1 if (u < MAX_MIB and not o) else 0
    if idle >= 4:
        print(f"idle after {time.time() - t0:.0f}s (used {u} MiB)", flush=True)
        sys.exit(0)
    time.sleep(15)
