"""lowrun.py <log> <cwd> <cmd...> : run cmd at BELOW_NORMAL priority, stdout+stderr -> log, write <log>.rc and <log>.t when done.  Nothing else."""
import os, subprocess, sys, time
log, cwd, *cmd = sys.argv[1:]
t0 = time.time()
with open(log, "wb") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, creationflags=0x00004000)
    rc = p.wait()
open(log + ".t", "w").write(str(int(time.time() - t0)))
open(log + ".rc", "w").write(str(rc))
