"""score_navtest_refe.py, UNCHANGED, with only its two path constants re-pointed at the E: drive and W3's harness taken
from a drive-repointed scratch copy (w3e/code: D:/ -> E:/ by sed, nothing else). Usage = score_navtest_refe's own args."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# P7-READ-ROUTE (2026-11-12): PKG = the package this file sits in (<PKG>/eval/harness/score_e.py), $TANITAD_REFE_PKG overrides; was a hard-coded E: path
PKG = (os.environ.get("TANITAD_REFE_PKG") or os.path.dirname(os.path.dirname(HERE))).replace("\\", "/")
spec = importlib.util.spec_from_file_location("score_navtest_refe", f"{PKG}/eval/score_navtest_refe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.W3_CODE = os.path.join(HERE, "w3e", "code").replace("\\", "/")
def _navsim_root():
    # P6 layout fix 2026-11-06 (path roots only): $TANITAD_NAVSIM_ROOT if set, else the FIRST EXISTING of E:/ F:/ D:/ Archive/devbox-C/navsim; REFUSES if none exists
    r = os.environ.get("TANITAD_NAVSIM_ROOT")
    cands = [r.replace("\\", "/").rstrip("/")] if r else [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit("REFUSING: navsim archive root not found (tried: " + ", ".join(cands) + "); set TANITAD_NAVSIM_ROOT. No silent default.")


m.W3_EXPORT = _navsim_root() + "/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"       # was m.W3_EXPORT.replace("D:/", "E:/")
m.PKG = PKG
sys.exit(m.main(sys.argv[1:]))
