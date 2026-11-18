"""AMEND_HWMPLAN_7: points AMEND_5's study-local modules at the amend7 folders and at RA7, BEFORE they are used. Import this first.
amend5_stage reads AMEND5_ROOT / AMEND5_READOUT_DIR / AMEND5_BASE_SOLVER / AMEND4_ROOT from the environment at import; RA_SHA, SPEC, SPEC_SHA, READOUT_FALLBACKS and readout_path are patched here, and ra_l1.load_readout /
ra_l1.plan_arrays are replaced by the ra7_l1 versions (the originals stay reachable as ra_l1._orig_load_readout / _orig_plan_arrays for the equivalence tests).
A chassis whose RA7_FIT_<k>.json does not exist gets the sha '0'*64, so every use of its readout refuses; AMEND_5's and AMEND_6's readouts are never a fallback.
AMEND7_ROOT (environment) moves the folder for the build's smokes; the default is the folder above build/."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AMEND7_ROOT") or os.path.dirname(HERE)
A5 = "F:/TanitAD-scratch/actrsp/lonplan/amend5"
A6 = "F:/TanitAD-scratch/actrsp/lonplan/amend6"
SPEC_FILE = os.environ.get("AMEND7_SPEC") or "F:/TanitAD-scratch/actrsp/lonplan/amend7/AMEND_HWMPLAN_7.md"
os.environ["AMEND5_ROOT"] = ROOT
os.environ["AMEND5_READOUT_DIR"] = ROOT + "/readout_in"
os.environ["AMEND5_BASE_SOLVER"] = A5 + "/BASE_SOLVER_AMEND5.md"
os.environ["AMEND4_ROOT"] = ROOT + "/_no_amend4"           # P / P23 files are not used
for p in (A5 + "/build", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
if A6 + "/build" not in sys.path:
    sys.path.append(A6 + "/build")                          # fit_ra6 (AMEND_6's pairs / ridge / fold map) is imported, never edited; its amend6_env is never imported here
import amend5_stage as ST  # noqa: E402

ST.READOUT_FALLBACKS = []
ST.SPEC = SPEC_FILE
ST.SPEC_SHA = open(SPEC_FILE[:-3] + ".sha256", encoding="utf-8").read().split()[0]
READOUT_NAME = "readout7_MLP_R35_{k}{b}.pt"


def readout_path7(k, b=""):
    return f"{ST.READOUT_DIR}/{READOUT_NAME.format(k=k, b=b)}"


ST.readout_path = lambda k: readout_path7(k)
for _k in ("s0", "s1"):
    _p = f"{ROOT}/RA7_FIT_{_k}.json"
    ST.RA_SHA[_k] = json.load(open(_p, encoding="utf-8")).get("sha256", "0" * 64) if os.path.exists(_p) else "0" * 64

import ra_l1 as R  # noqa: E402
import ra7_l1  # noqa: E402

if not hasattr(R, "_orig_plan_arrays"):
    R._orig_plan_arrays, R._orig_load_readout = R.plan_arrays, R.load_readout
R.load_readout = ra7_l1.load_readout7
R.plan_arrays = ra7_l1.plan_arrays7


def sha_B(k):
    """Seed set B readout sha256 from RA7_FIT_<k>.json ('0'*64 when absent)."""
    p = f"{ROOT}/RA7_FIT_{k}.json"
    return json.load(open(p, encoding="utf-8")).get("sha256_B", "0" * 64) if os.path.exists(p) else "0" * 64
