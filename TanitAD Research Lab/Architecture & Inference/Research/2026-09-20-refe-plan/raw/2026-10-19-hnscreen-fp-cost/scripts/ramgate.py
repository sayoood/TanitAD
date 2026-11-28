"""wait until free RAM >= N GB (default 4) AND free disk on C: >= M GB (default 0 = not checked), polling every 15 s, at most 20 min; exit 0 when both hold, 1 when not.   usage: ramgate.py [RAM_GB] [DISK_GB]
(2026-10-19: C: fell to 0.1-0.6 GB free from other writers; a test that dies of ENOSPC reads as a failure - or, for a mutant, as a false RED - so the disk is gated too.)"""
import sys
import time

import psutil

need = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
disk = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t0 = time.time()


def ok():
    return psutil.virtual_memory().available >= need * 2**30 and (disk <= 0 or psutil.disk_usage("C:\\").free >= disk * 2**30)


while not ok():
    if time.time() - t0 > 1200:
        print(f"RAMGATE_TIMEOUT free RAM {psutil.virtual_memory().available / 2**30:.2f} GB (need {need}), free C: {psutil.disk_usage('C:' + chr(92)).free / 2**30:.2f} GB (need {disk})")
        sys.exit(1)
    time.sleep(15)
print(f"RAMGATE_OK free RAM {psutil.virtual_memory().available / 2**30:.2f} GB, free C: {psutil.disk_usage('C:' + chr(92)).free / 2**30:.2f} GB after {time.time() - t0:.0f} s")
