"""wait_commit.py -- exit 0 once >= GB of commit is free on the dev box (checked every 60 s, up to MAX_MIN minutes), else exit 1.
Why: the first STRAT1 read's dumps died of commit exhaustion (ArrayMemoryError, 2026-11-16 18:02Z) while other CPU jobs ran (memory devbox-commit-limit-oom).
Usage: python wait_commit.py [GB=5.5] [MAX_MIN=300]"""
import ctypes
import sys
import time


def commit_free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


if __name__ == "__main__":
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else 5.5
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    for i in range(mx):
        cf = commit_free_gb()
        if cf >= gb:
            print(f"commit free {cf:.2f} GB >= {gb} after {i} min", flush=True)
            sys.exit(0)
        time.sleep(60)
    print(f"commit stayed under {gb} GB for {mx} min (last {cf:.2f})", flush=True)
    sys.exit(1)
