import sys, math
import numpy as np
sys.path.insert(0, ".")
import labeller as L
import test_labeller as TL
panel = TL._panel()
np.set_printoptions(precision=2, suppress=True, linewidth=200)
n = 0
for gg, vv, r, own, v_own in panel:
    res = L.recovery_label(gg, vv, r, own, v_own, check=False, ladder=(30,))
    if res["label_own"] is None: continue
    Lo = res["label_own"]
    pts = np.concatenate([[[-v_own*L.DT, 0.0]], [[0.0, 0.0]], Lo])
    sp, ac, lat, kap = L.controls(pts)
    v = L.violations(Lo, v_own, 31)
    if v["accel"] > 0:
        print("v_own %.1f  v_h %.1f  off(e,dpsi)=?  max|a| %.2f at k=%d" % (v_own, vv[r], v["max_abs_accel"], int(np.argmax(np.abs(ac)))))
        print("   speed:", sp[:10]); print("   accel:", ac[:10])
        n += 1
        if n >= 4: break
