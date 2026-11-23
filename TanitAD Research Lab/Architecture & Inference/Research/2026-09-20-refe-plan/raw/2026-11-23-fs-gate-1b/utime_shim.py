import os, runpy, sys
_u = os.utime
def utime(path, times=None, *, ns=None, **kw):
    if ns is not None:
        ns = tuple(max(int(x), 315532800 * 10**9) for x in ns)      # >= 1980-01-01 (exFAT floor)
    return _u(path, times, ns=ns, **kw) if ns is not None else _u(path, times, **kw)
os.utime = utime
sys.argv = sys.argv[1:]
sys.path.insert(0, os.getcwd())
runpy.run_path(sys.argv[0], run_name="__main__")
