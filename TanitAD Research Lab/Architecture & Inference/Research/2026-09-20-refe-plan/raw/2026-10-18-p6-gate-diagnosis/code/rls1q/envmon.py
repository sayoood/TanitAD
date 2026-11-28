"""Every 10 s: C: free MB, free virtual (commit headroom) MB, free physical MB -> envmon.tsv.  Usage: envmon.py <seconds>"""
import ctypes, shutil, sys, time
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong)]
end = time.time() + float(sys.argv[1])
with open("C:/Users/Admin/p6gate/rls1q_diag/envmon.tsv", "a") as f:
    while time.time() < end:
        m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        f.write(f"{time.strftime('%H:%M:%S')}\tCfree_MB={shutil.disk_usage('C:/').free // 2**20}\tcommit_avail_MB={m.ullAvailPageFile // 2**20}\tcommit_total_MB={m.ullTotalPageFile // 2**20}\tphys_avail_MB={m.ullAvailPhys // 2**20}\n"); f.flush()
        time.sleep(10)
