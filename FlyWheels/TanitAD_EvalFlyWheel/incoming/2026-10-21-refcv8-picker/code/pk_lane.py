"""pk_lane.py -- PREREG R8-PICKER section 3: run `pf_lane.py` (UNCHANGED, imported) with the battery ARCHIVE re-pointed to the picker archive.

The re-point is a PATH and is exactly the established pattern of `retime_trainfit_r8._patch_archive` (CP.ARCHIVE / CR.ARCHIVE), extended to the
other modules that hold the same constant: tac_retime_r8.ARCHIVE, nr_matched_instrument.ARCHIVE, prior_parity_r8.ARCHIVE (also its env R8_ARCHIVE,
read at import).  No function body is replaced; pf_lane.mods() is wrapped only to set those attributes after its own imports.

    PK_ARCHIVE=C:/Users/Admin/refcv8_pf/picker/archive python pk_lane.py <stage> --pkg <pfpkg> [--tree T] [--table F] --registered R
"""
import os
import sys
from pathlib import Path

ARCH = os.environ.get("PK_ARCHIVE", "C:/Users/Admin/refcv8_pf/picker/archive").replace("\\", "/")
os.environ["R8_ARCHIVE"] = ARCH                     # prior_parity_r8 reads it at import
HOLDW = Path("C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code")
sys.path.insert(0, str(HOLDW))
import pf_lane as PL  # noqa: E402

_orig_mods = PL.mods


def mods():
    M = _orig_mods()
    M["CP"].ARCHIVE = ARCH
    M["CR"].ARCHIVE = ARCH
    M["TAC"].ARCHIVE = ARCH
    M["PP"].ARCHIVE = Path(ARCH)
    import nr_matched_instrument as NR
    NR.ARCHIVE = ARCH
    print(f"[pk_lane] ARCHIVE re-pointed -> {ARCH} (CP, CR, TAC, PP, NR)", flush=True)
    return M


PL.mods = mods


def _peak_rss_report():
    """Master Mind constraint 2026-10-21 21:45: every dev-box process < 4 GB while C2 is live -> print this stage's PEAK working set (MEASURED, not estimated)."""
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
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            print(f"ZZPKPEAKRSSZZ unavailable GetProcessMemoryInfo=0 err={ctypes.get_last_error()}", file=sys.stderr, flush=True)
        else:
            print(f"ZZPKPEAKRSSZZ stage={sys.argv[1] if len(sys.argv) > 1 else '?'} peak_working_set_GB={c.PeakWorkingSetSize / 2**30:.3f} "
                  f"peak_commit_GB={c.PeakPagefileUsage / 2**30:.3f}", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001 -- a report, never a failure
        print(f"ZZPKPEAKRSSZZ unavailable {e!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    import atexit
    atexit.register(_peak_rss_report)
    sys.exit(PL.main())
