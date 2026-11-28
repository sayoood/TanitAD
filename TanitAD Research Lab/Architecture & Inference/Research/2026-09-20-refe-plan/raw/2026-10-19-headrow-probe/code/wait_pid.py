"""Block until the given PID exits (poll every 5 s) or the timeout elapses.  usage: python wait_pid.py <pid> [timeout_s]"""
import sys
import time

import psutil

pid = int(sys.argv[1])
to = float(sys.argv[2]) if len(sys.argv) > 2 else 6 * 3600
t0 = time.time()
while psutil.pid_exists(pid) and time.time() - t0 < to:
    try:
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
    except psutil.NoSuchProcess:
        break
    time.sleep(5)
print("exited" if not psutil.pid_exists(pid) else "timeout")
