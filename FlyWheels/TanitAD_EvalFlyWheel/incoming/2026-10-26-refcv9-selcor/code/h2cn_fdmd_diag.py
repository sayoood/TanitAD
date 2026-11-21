"""h2cn_fdmd_diag.py -- DESCRIPTIVE (no bar): false-/missed-decel of the shipped H2CN_DEPLOY8 stage vs RETIME_DEPLOY8 on the SAME battery draw,
from the banked DEPLOY8_POS positions (already staged, S6). Tells whether the SELCOR lever (FD) still has room once H2cn is the stage."""
import sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_common as C  # noqa: E402
A = "C:/Users/Admin/refcv8_pf/ship_r8_sub/out/arms/h2cn/{}/DEPLOY8_POS_step48852ema_pf_s{}.npz"
PR = C.stack()["PR"]
for sd in (0, 1):
    E = C.load_eval_pf(sd)
    comp = np.asarray(E["compliant"], bool)
    out = {}
    for arm in ("RETIME_DEPLOY8", "H2CN_DEPLOY8", "H2CN_NEAR"):
        z = np.load(A.format(arm, sd))
        assert np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S6_clip"], E["clip"]), "row alignment"
        fd, md, gt, pc = C.lon_indicators(z["S6_pos"], E, comp)
        out[arm] = (fd, md)
    eid = list(E["eid"])
    for arm in ("H2CN_DEPLOY8", "H2CN_NEAR"):
        for k, j in (("FD", 0), ("MD", 1)):
            c = PR.paired_cell(out[arm][j], out["RETIME_DEPLOY8"][j], eid)
            print(f"s{sd} {arm:13s} {k}: {np.nanmean(out[arm][j]):.4f} vs RETIME {np.nanmean(out['RETIME_DEPLOY8'][j]):.4f}  "
                  f"delta {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] n={c['n_windows']}")
