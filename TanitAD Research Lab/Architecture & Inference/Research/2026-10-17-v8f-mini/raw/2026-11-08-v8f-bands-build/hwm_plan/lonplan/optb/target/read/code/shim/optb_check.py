"""Shim: readchain_tools.load_checker() does `import optb_check as C` from OPTB_PKG. For the OPTB3 read, OPTB_PKG points here and C is the OPTB3 checker of record (TABLE with FTT tags, RUN_ORDER, INIT_SHA, C2/C3/C5 functions)."""
import sys
sys.path.insert(0, "C:/Users/Admin/optb3_pkg")
from optb3_check import *  # noqa: F401,F403
import optb3_check as _o
for _n in dir(_o):
    if not _n.startswith("__"):
        globals()[_n] = getattr(_o, _n)
