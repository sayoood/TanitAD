import ctypes as c
class M(c.Structure):
    _fields_=[("l",c.c_ulong),("load",c.c_ulong),("tp",c.c_ulonglong),("ap",c.c_ulonglong),("tf",c.c_ulonglong),("af",c.c_ulonglong),("tv",c.c_ulonglong),("av",c.c_ulonglong),("ae",c.c_ulonglong)]
class P(c.Structure):
    _fields_=[("cb",c.c_ulong),("pfc",c.c_ulong),("pws",c.c_size_t),("ws",c.c_size_t),("qpp",c.c_size_t),("qp",c.c_size_t),("qnpp",c.c_size_t),("qnp",c.c_size_t),("pfu",c.c_size_t),("ppfu",c.c_size_t),("priv",c.c_size_t)]
def status():
    m=M(); m.l=c.sizeof(M); c.windll.kernel32.GlobalMemoryStatusEx(c.byref(m))
    return {"phys_avail_GB": m.ap/2**30, "commit_limit_GB": m.tf/2**30, "commit_avail_GB": m.af/2**30}
def own_commit_gb():
    """This process's own commit charge (PROCESS_MEMORY_COUNTERS_EX.PrivateUsage), in GB (2**30)."""
    p=P(); p.cb=c.sizeof(P)
    h=c.windll.kernel32.GetCurrentProcess()
    ok=c.windll.psapi.GetProcessMemoryInfo(c.c_void_p(h), c.byref(p), p.cb)
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return p.priv/2**30
if __name__ == "__main__":
    d={k: round(v, 2) for k, v in status().items()}
    d["own_commit_GB"]=round(own_commit_gb(), 3)
    print(d)
