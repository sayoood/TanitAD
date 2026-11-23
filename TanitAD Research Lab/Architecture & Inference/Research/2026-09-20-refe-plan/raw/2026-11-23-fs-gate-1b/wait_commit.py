"""Block until Windows free commit (ullAvailPageFile) >= GB (default 5.5), polling every 60 s.  usage: wait_commit.py [GB]"""
import ctypes, sys, time


def free_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPageFile / float(1 << 30)


need = float(sys.argv[1]) if len(sys.argv) > 1 else 5.5
while True:
    g = free_gb()
    if g >= need:
        print('commit_free_gb %.2f' % g)
        break
    time.sleep(60)
