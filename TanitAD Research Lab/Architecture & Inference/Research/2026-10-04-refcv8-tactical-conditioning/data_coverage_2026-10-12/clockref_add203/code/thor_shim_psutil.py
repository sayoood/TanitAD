"""Minimal psutil shim (Thor's tanitad-train venv has no psutil; we do not install into it).
Only what _common.ram_guard needs: virtual_memory().available from /proc/meminfo MemAvailable."""
import collections
_VM = collections.namedtuple("svmem", "total available")
def virtual_memory():
    d = {}
    with open("/proc/meminfo") as fh:
        for ln in fh:
            k, v = ln.split(":", 1)
            d[k] = int(v.split()[0]) * 1024
    return _VM(d["MemTotal"], d["MemAvailable"])
