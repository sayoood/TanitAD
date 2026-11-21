#!/usr/bin/env python3
"""Free commit charge in whole GB (INTEGER, floor), via GlobalMemoryStatusEx ullAvailPageFile.
No PowerShell, no locale: a decimal-comma locale cannot break the parse.
  commit_guard.py            -> prints e.g. 11
  commit_guard.py --min 6    -> prints the integer, exit 0 if >= 6 else exit 3
"""
import ctypes
import sys


def free_commit_gb() -> int:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise SystemExit("GlobalMemoryStatusEx failed")
    return int(s.ullAvailPageFile // (1 << 30))


if __name__ == "__main__":
    g = free_commit_gb()
    print(g)
    if len(sys.argv) > 2 and sys.argv[1] == "--min":
        sys.exit(0 if g >= int(sys.argv[2]) else 3)
