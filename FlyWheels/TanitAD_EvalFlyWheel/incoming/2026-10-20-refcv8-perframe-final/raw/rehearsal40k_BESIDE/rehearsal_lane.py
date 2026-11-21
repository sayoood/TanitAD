"""rehearsal_lane.py -- 40k REHEARSAL, BESIDE, not binding.  Runs pf_lane.main() UNCHANGED, with ONE deviation from the chain:
the refcv8 archive root (CP/CR/TAC/PP .ARCHIVE) is re-pointed to REH_ARCHIVE (default the original read-only 40k pf battery root),
because clamp/compliant/tac REQUIRE <archive>/<tree>/PAIRED_READ_<tree>.json (clamp_posthoc_r8.py:386, compliant_reread_r8.py:456-457,
tac_retime_r8.py:190) and the brief forbids placing a PAIRED_READ_* file in the real archive.  Writes nothing outside --pkg / the clamp scratch."""
import os, sys
from pathlib import Path
ROOT = os.environ.get("REH_ARCHIVE", "C:/Users/Admin/refcv8_pf/battery40k_pf")
os.environ["R8_ARCHIVE"] = ROOT
sys.path.insert(0, os.environ["PF_LANE_DIR"])
import pf_lane as L
_orig = L.configure
def configure(M, pkg, reg):
    out = _orig(M, pkg, reg)
    for k in ("CP", "CR", "TAC"):
        M[k].ARCHIVE = ROOT
    M["PP"].ARCHIVE = Path(ROOT)
    print(f"[rehearsal_lane] archive root re-pointed to {ROOT}", flush=True)
    return out
L.configure = configure
sys.exit(L.main())
