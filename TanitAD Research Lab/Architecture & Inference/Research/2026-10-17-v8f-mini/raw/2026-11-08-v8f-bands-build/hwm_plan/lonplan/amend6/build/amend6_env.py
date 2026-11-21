"""AMEND_HWMPLAN_6: points AMEND_5's study-local modules at the amend6 folders and at RA6, BEFORE they are used. Import this first.
amend5_stage reads AMEND5_ROOT / AMEND5_READOUT_DIR / AMEND5_BASE_SOLVER / AMEND4_ROOT from the environment at import; RA_SHA, SPEC, SPEC_SHA and READOUT_FALLBACKS are patched here.
A chassis whose RA6_FIT_<k>.json does not exist gets the sha '0'*64, so every use of its readout refuses; AMEND_5's readout is never a fallback."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A5 = "F:/TanitAD-scratch/actrsp/lonplan/amend5"
os.environ["AMEND5_ROOT"] = ROOT
os.environ["AMEND5_READOUT_DIR"] = ROOT + "/readout_in"
os.environ["AMEND5_BASE_SOLVER"] = A5 + "/BASE_SOLVER_AMEND5.md"
os.environ["AMEND4_ROOT"] = ROOT + "/_no_amend4"           # P / P23 files are not used: the beside RA-L2 row is not rerun
for p in (A5 + "/build", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import amend5_stage as ST  # noqa: E402

ST.READOUT_FALLBACKS = []
ST.SPEC = ROOT + "/AMEND_HWMPLAN_6.md"
ST.SPEC_SHA = open(ROOT + "/AMEND_HWMPLAN_6.sha256", encoding="utf-8").read().split()[0]
for _k in ("s0", "s1"):
    _p = f"{ROOT}/RA6_FIT_{_k}.json"
    ST.RA_SHA[_k] = json.load(open(_p, encoding="utf-8"))["sha256"] if os.path.exists(_p) else "0" * 64
