"""pk_peak.py -- print this process's PEAK working set at exit (Master Mind constraint 2026-10-21 21:45: every dev-box process < 4 GB while C2 is live).
MEASURED via psapi.GetProcessMemoryInfo; a report, never a failure."""
import atexit
import sys


def report(tag: str = "") -> None:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            print(f"ZZPKPEAKRSSZZ {tag} peak_working_set_GB={c.PeakWorkingSetSize / 2**30:.3f} peak_commit_GB={c.PeakPagefileUsage / 2**30:.3f}",
                  file=sys.stderr, flush=True)
        else:
            print(f"ZZPKPEAKRSSZZ {tag} unavailable GetProcessMemoryInfo=0", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ZZPKPEAKRSSZZ {tag} unavailable {e!r}", file=sys.stderr, flush=True)


def arm(tag: str) -> None:
    atexit.register(report, tag)
