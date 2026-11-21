"""Read PT3M's END ceilings ahead of the formal report; the same code path as report_pt3m.compare, ceilings only. Decides nothing."""
import os, sys
sys.dont_write_bytecode = True
PT3 = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest3"
sys.path.insert(0, PT3)
import numpy as np
import plantest3_real as R3
import plantest_lib as P

RUNS = f"{PT3}/mlp4/runs"
for t in ("M2b_s0", "M2b_s1"):
    c = R3.prepare3(t)
    sc = c.score_idx
    _, Wt, pos = P.boot_setup(c.eid[sc])
    bm = lambda x: P.boot_mean(x[np.isfinite(x)], pos[np.isfinite(x)], Wt)
    def bd(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return P.boot_mean_diff(a[m], b[m], pos[m], Wt)
    z = np.load(f"{RUNS}/ceil_{t}.npz")
    k = np.load(f"{RUNS}/kin_{t}_true.npz")["plan_raw"]
    mets = lambda p: P.window_metrics(p, c.A[sc], c.v0[sc])
    h, o, r, ki = mets(P.hold_plan(c.A[sc])), mets(z["oracle"]), mets(z["read"]), mets(k)
    for q in ("acc_mae", "yaw_mae"):
        do, dr, dk = bd(o[q], h[q]), bd(r[q], h[q]), bd(ki[q], h[q])
        print(t, q, "HOLD %.4f" % bm(h[q])[0], "ORACLE %.4f" % bm(o[q])[0], "READ %.4f" % bm(r[q])[0], "KIN %.4f" % bm(ki[q])[0],
              "| O-H %.4f [%.4f,%.4f] %s" % (*do, P.sep_label(*do)), "| R-H %.4f [%.4f,%.4f] %s" % (*dr, P.sep_label(*dr)), "| K-H %.4f %s" % (dk[0], P.sep_label(*dk)), "n", int(np.isfinite(o[q]).sum()), int(np.isfinite(r[q]).sum()))
    for nm, pl in (("hold", h), ("oracle", o), ("read", r)):
        print(t, nm, "ADE %.4f" % bm(pl["ade"])[0], "speed_err %.4f" % bm(pl["speed_err"])[0], "heading %.4f" % bm(pl["heading_err"])[0])
