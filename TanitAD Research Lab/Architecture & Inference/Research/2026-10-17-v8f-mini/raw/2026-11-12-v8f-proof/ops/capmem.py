"""capmem.py : run one Python job under a hard per-process commit limit (Windows job object), one such job at a time.

usage:  python capmem.py <limit_mb> <script.py> [args...]
        python capmem.py <limit_mb> -m <module> [args...]

Order of steps, each fail-closed (the job is not run if a step fails):
  1. a named mutex (Local\\v8proof_pycap): only one capped job of this fork runs at a time; wait at most CAPMEM_MUTEX_S (default 14400 s).
  2. a gate on free commit memory: at least limit + CAPMEM_RESERVE_MB (default 1500) must be free; polled every 30 s for at most
     CAPMEM_WAIT_S (default 5400 s). On timeout: exit 95, stderr line "ZZ CAPMEM STOP ...".
  3. the limit: JOB_OBJECT_LIMIT_PROCESS_MEMORY on this process. An allocation above it raises MemoryError here, in this process only.
  4. the job runs in this process (runpy), with sys.argv and sys.path[0] as if it had been started directly.
At exit one line is appended to CAPMEM_LOG (default F:/TanitAD-scratch/v8proof/ops/capmem_peaks.log):
  <utc> | <job> | peak_mb <n> | limit_mb <m> | free_mb_at_start <f>
Nothing is printed to stdout or stderr on success, so pipelines that read the last line of a tool are not disturbed.
Exit codes of this launcher itself: 94 usage, 95 gate timeout, 96 mutex timeout, 97 a Windows call failed.
"""
import atexit
import ctypes
import os
import runpy
import sys
import time
from ctypes import wintypes


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]


class _BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _EXT(ctypes.Structure):
    _fields_ = [("Basic", _BASIC), ("Io", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _die(code, msg):
    sys.stderr.write("ZZ CAPMEM STOP: %s\n" % msg)
    sys.stderr.flush()
    os._exit(code)


def main():
    if len(sys.argv) < 3:
        _die(94, "usage: capmem.py <limit_mb> <script.py | -m module> [args...]")
    lim_mb = int(sys.argv[1])
    rest = sys.argv[2:]
    is_mod = rest[0] == "-m"
    if is_mod and len(rest) < 2:
        _die(94, "-m needs a module name")
    reserve_mb = int(os.environ.get("CAPMEM_RESERVE_MB", "1500"))
    floor_mb = int(os.environ.get("CAPMEM_FLOOR_MB", "5500"))          # shared dev-box rule (2026-11-19): no capped job starts under 5.5 GB of free commit memory
    wait_s = int(os.environ.get("CAPMEM_WAIT_S", "5400"))
    mutex_s = int(os.environ.get("CAPMEM_MUTEX_S", "14400"))
    log_path = os.environ.get("CAPMEM_LOG", "F:/TanitAD-scratch/v8proof/ops/capmem_peaks.log")
    assert ctypes.sizeof(_EXT) == 144, ctypes.sizeof(_EXT)

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    k32.WaitForSingleObject.restype = wintypes.DWORD
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.QueryInformationJobObject.restype = wintypes.BOOL
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    k32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]

    # 1. one capped job at a time
    hm = k32.CreateMutexW(None, False, "Local\\v8proof_pycap")
    if not hm:
        _die(97, "CreateMutexW failed (%d)" % ctypes.get_last_error())
    k32.ReleaseMutex.restype = wintypes.BOOL
    k32.ReleaseMutex.argtypes = [wintypes.HANDLE]

    # 2. gate on free commit memory. The gate is waited for WITHOUT the mutex (a job that waits for memory must not hold the
    #    other capped jobs back), then the mutex is taken and the gate is read again; if the gate does not hold then, the mutex is released.
    def free_mb():
        m = _MEMSTAT()
        m.dwLength = ctypes.sizeof(_MEMSTAT)
        if not k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            _die(97, "GlobalMemoryStatusEx failed (%d)" % ctypes.get_last_error())
        return m.ullAvailPageFile // (1024 * 1024)

    t0 = time.time()
    while True:
        f0 = free_mb()
        while f0 < max(lim_mb + reserve_mb, floor_mb):
            if time.time() - t0 > wait_s:
                _die(95, "free commit memory stayed under %d MB (limit %d + reserve %d) for %d s; last read %d MB"
                     % (max(lim_mb + reserve_mb, floor_mb), lim_mb, reserve_mb, wait_s, f0))
            time.sleep(30)
            f0 = free_mb()
        r = k32.WaitForSingleObject(hm, mutex_s * 1000)
        if r not in (0x0, 0x80):          # WAIT_OBJECT_0, WAIT_ABANDONED
            _die(96, "another capped job held the mutex for %d s" % mutex_s)
        f0 = free_mb()
        if f0 >= max(lim_mb + reserve_mb, floor_mb):
            break
        k32.ReleaseMutex(hm)
        time.sleep(30)

    # 3. the limit
    hj = k32.CreateJobObjectW(None, None)
    if not hj:
        _die(97, "CreateJobObjectW failed (%d)" % ctypes.get_last_error())
    info = _EXT()
    info.Basic.LimitFlags = 0x00000100            # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = lim_mb * 1024 * 1024
    if not k32.SetInformationJobObject(hj, 9, ctypes.byref(info), ctypes.sizeof(info)):
        _die(97, "SetInformationJobObject failed (%d)" % ctypes.get_last_error())
    if not k32.AssignProcessToJobObject(hj, k32.GetCurrentProcess()):
        _die(97, "AssignProcessToJobObject failed (%d)" % ctypes.get_last_error())
    chk = _EXT()
    if not k32.QueryInformationJobObject(hj, 9, ctypes.byref(chk), ctypes.sizeof(chk), None) \
            or chk.ProcessMemoryLimit != lim_mb * 1024 * 1024 or not (chk.Basic.LimitFlags & 0x100):
        _die(97, "the limit did not read back")

    job_name = " ".join(rest[:2] if is_mod else rest[:1])

    def _log():
        q = _EXT()
        peak = -1
        if k32.QueryInformationJobObject(hj, 9, ctypes.byref(q), ctypes.sizeof(q), None):
            peak = q.PeakProcessMemoryUsed // (1024 * 1024)
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("%s | %s | peak_mb %d | limit_mb %d | free_mb_at_start %d\n"
                         % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), job_name, peak, lim_mb, f0))
        except OSError:
            pass

    atexit.register(_log)

    # 4. the job
    if is_mod:
        sys.argv = [rest[1]] + rest[2:]
        sys.path[0] = os.getcwd()
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
    else:
        script = rest[0]
        sys.argv = [script] + rest[1:]
        sys.path[0] = os.path.dirname(os.path.abspath(script))
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
