"""Memory gate (Master Mind, 2026-11-07): free physical memory at job start >= 6 GB -> go; below 6 GB -> wait until ../voe1b/VOE1B_RUN.done exists (poll 30 s, 8 h ceiling).
Exit 0 = go, 2 = ceiling hit (nothing runs). free_gb() is also used by run_imag1b.py before the TRAIN pass."""
import ctypes, os, subprocess, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DONE = os.path.normpath(os.path.join(HERE, "..", "voe1b", "VOE1B_RUN.done"))
MIN_GB, POLL_S, CEIL_S = 6.0, 30, 8 * 3600


def free_gb():
    """Free physical memory in GB: Win32_OperatingSystem.FreePhysicalMemory (what the Master Mind quoted), else GlobalMemoryStatusEx."""
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True, timeout=60).stdout.strip()
        return float(o) / 1048576.0
    except Exception:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30


if __name__ == "__main__":
    f = free_gb()
    print(f"[gate] free physical memory {f:.2f} GB at {time.strftime('%H:%M:%S')}; need >= {MIN_GB} GB, else wait for {DONE}", flush=True)
    if f >= MIN_GB:
        print("[gate] go", flush=True)
        sys.exit(0)
    t0 = time.time()
    while not os.path.exists(DONE):
        if time.time() - t0 > CEIL_S:
            print("[gate] ceiling hit, nothing runs", flush=True)
            sys.exit(2)
        time.sleep(POLL_S)
    print(f"[gate] VOE1B_RUN.done exists after {time.time() - t0:.0f} s; free now {free_gb():.2f} GB; go", flush=True)
    sys.exit(0)
