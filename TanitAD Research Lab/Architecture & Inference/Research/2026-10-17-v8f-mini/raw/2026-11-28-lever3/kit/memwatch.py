import ctypes, sys, time
class MS(ctypes.Structure):
    _fields_=[("l",ctypes.c_ulong),("load",ctypes.c_ulong),("tp",ctypes.c_ulonglong),("ap",ctypes.c_ulonglong),("tpf",ctypes.c_ulonglong),("apf",ctypes.c_ulonglong),("tv",ctypes.c_ulonglong),("av",ctypes.c_ulonglong),("ae",ctypes.c_ulonglong)]
out=sys.argv[1]; end=time.time()+float(sys.argv[2]); lo=1e18; n=0
m=MS(); m.l=ctypes.sizeof(MS)
with open(out,"w") as f:
    while time.time()<end:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        v=m.apf/2**30; lo=min(lo,v); f.write(f"{time.time():.0f} free_commit_gib {v:.2f} avail_phys_gib {m.ap/2**30:.2f}\n"); f.flush(); time.sleep(2)
    f.write(f"MIN_FREE_COMMIT_GIB {lo:.2f}\n")
