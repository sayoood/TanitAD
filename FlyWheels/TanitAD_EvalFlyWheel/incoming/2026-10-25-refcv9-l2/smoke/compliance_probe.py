import sys
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C
E = C.load_eval_pf(0)
ex = np.load(C.ARCH / C.EVAL_TREE / "dump_s0" / "refcv6_extras.npz")
rows = E["rows"]
vm, vv, gm = ex["v_max_ms"][rows], ex["v_max_valid"][rows], ex["gt_max_speed_6s"][rows]
print("valid frac", vv.mean(), "v_max range", np.nanmin(vm), np.nanmax(vm), "gm range", np.nanmin(gm), np.nanmax(gm))
for tol in (0.0, 0.05, 0.1, 0.5, 1.0):
    m = (gm <= vm + tol) | (vv < 0.5)
    print(tol, m.sum(), "agree w/ banked", (m == E["compliant"]).mean(), " mism", int((m != E["compliant"]).sum()))
m2 = (gm <= vm) & (vv > 0.5)
print("strict valid:", m2.sum(), int((m2 != E["compliant"]).sum()))
