"""Memory gate (card: start only with >= 5,000 MB free RAM; the box is shared). Polls free physical memory every 30 s until it is >= 5,000 MB; 8 h ceiling (exit 2, nothing runs)."""
import ctypes, subprocess, sys, time
sys.dont_write_bytecode = True
MIN_MB, POLL_S, CEIL_S = 5000.0, 30, 8 * 3600


def free_mb():
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True, timeout=60).stdout.strip()
        return float(o) / 1024.0
    except Exception:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 20


if __name__ == "__main__":
    t0 = time.time()
    f = free_mb()
    print(f"[gate] free physical memory {f:.0f} MB at {time.strftime('%H:%M:%S')}; need >= {MIN_MB:.0f} MB", flush=True)
    while f < MIN_MB:
        if time.time() - t0 > CEIL_S:
            print("[gate] ceiling hit, nothing runs", flush=True)
            sys.exit(2)
        time.sleep(POLL_S)
        f = free_mb()
    print(f"[gate] go after {time.time() - t0:.0f} s, free {f:.0f} MB", flush=True)
