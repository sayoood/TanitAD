"""run_capped.py -- run a command inside a Windows Job Object with a hard per-process COMMIT cap, and report its PEAK working set.

Why: the dev box shares its RAM with another session's GPU job; the brief caps every dev-box process of this task at 4 GB.
A job-object ProcessMemoryLimit makes an allocation past the cap FAIL (MemoryError inside the child) instead of overshooting,
so "the CPU forward does not fit" is a MEASURED, harmless outcome and never an overshoot.

    python run_capped.py --limit-gb 4 --tag smoke_wrapped -- <python.exe> script.py args...

Output (stderr, last line): ZZCAPPEDZZ tag=<tag> rc=<rc> peak_ws_gb=<x> peak_process_commit_gb=<y> limit_gb=<L> wall_s=<s>
The child's own stdout/stderr pass through. Exit code = the child's. Windows only (ctypes, kernel32 / psapi).
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes

JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                  "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class PIDLIST(ctypes.Structure):
    _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD), ("NumberOfProcessIdsInList", wintypes.DWORD),
                ("ProcessIdList", ctypes.c_size_t * 64)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-gb", type=float, default=4.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--ws-kill-gb", type=float, default=0.0, help="kill the job when any process's working set exceeds this (0 = off)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    k32, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.ProcessMemoryLimit = int(a.limit_gb * 2 ** 30)
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        print("ZZCAPPEDZZ tag=%s SETLIMIT_FAILED err=%d" % (a.tag, ctypes.get_last_error()), file=sys.stderr, flush=True)
        return 97
    t0 = time.time()
    p = subprocess.Popen(cmd)
    ok = k32.AssignProcessToJobObject(job, int(p._handle))
    if not ok:
        print("ZZCAPPEDZZ tag=%s ASSIGN_FAILED err=%d (the child is NOT capped; refusing)" % (a.tag, ctypes.get_last_error()),
              file=sys.stderr, flush=True)
        p.kill()
        return 98
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32.OpenProcess.restype = wintypes.HANDLE
    peak_ws = 0

    def sample_ws() -> int:
        """Max over the job's live processes of their PEAK working set (the venv python.exe is a launcher; the real
        interpreter is its child, so the handle returned by Popen is NOT the process to measure)."""
        pl = PIDLIST()
        k32.QueryInformationJobObject(job, 3, ctypes.byref(pl), ctypes.sizeof(pl), None)   # JobObjectBasicProcessIdList
        best = 0
        for i in range(min(int(pl.NumberOfProcessIdsInList), 64)):
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pl.ProcessIdList[i]))
            if h:
                c = PMC()
                c.cb = ctypes.sizeof(PMC)
                if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                    best = max(best, int(c.PeakWorkingSetSize))
                k32.CloseHandle(h)
        return best

    killed = False
    while p.poll() is None:
        peak_ws = max(peak_ws, sample_ws())
        if a.ws_kill_gb > 0 and peak_ws > a.ws_kill_gb * 2 ** 30:
            killed = True
            k32.TerminateJobObject(job, 99)
            print("ZZCAPPEDZZ tag=%s WS_KILL peak_ws_gb=%.3f > %.2f" % (a.tag, peak_ws / 2 ** 30, a.ws_kill_gb), file=sys.stderr, flush=True)
            break
        time.sleep(0.25)
    if killed:
        try:
            p.wait(timeout=10)
        except Exception:       # noqa: BLE001
            pass
    peak_ws = max(peak_ws, sample_ws())
    ei = EXT()
    k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(ei), ctypes.sizeof(ei), None)
    peak_commit = int(ei.PeakProcessMemoryUsed)
    print("ZZCAPPEDZZ tag=%s killed=%s rc=%s peak_ws_gb=%.3f peak_process_commit_gb=%.3f limit_gb=%.2f wall_s=%.1f" % (
        a.tag, killed, p.returncode, peak_ws / 2 ** 30, peak_commit / 2 ** 30, a.limit_gb, time.time() - t0), file=sys.stderr, flush=True)
    return int(p.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
