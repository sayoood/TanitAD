#!/usr/bin/env python3
"""L4 AUDIT, zero GPU: does refav1's decode produce INFEASIBLE paths at all?

The brief's L4 says to reuse the sibling stream's feasibility-aware decode. Before
building an arm on it, measure whether refav1 has anything for it to fix: the module
(`tanitad.refs.feasible_decode`) re-derives the SCORER's own envelope flags from a
path, so `assert_feasible` on the banked `cl` paths answers it directly.

⛔ THE CONTROL THAT MUST READ A KNOWN VALUE: the GROUND-TRUTH path `g` is a real
vehicle's recorded motion and MUST be feasible. If `g` reads a high violation rate,
the dt / origin handling here is wrong and NOTHING below is about the planner.
ASCII only.
"""
import glob, sys
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack")
from tanitad.refs import feasible_decode as FD          # noqa: E402

DT = 0.2      # the refav1 dump grid (K=10 over 2.0 s), NOT fan_safety's 0.5 s
import os
VMIN = float(os.environ.get("FEAS_VMIN", "0"))  # near-stationary exclusion


def load(dumpdir, key, vmin=0.0):
    out = []
    for f in sorted(glob.glob(dumpdir + "/ep*.npz")):
        z = np.load(f, allow_pickle=True)
        if key in z.files:
            a = z[key]
            if vmin > 0.0 and "v0" in z.files:
                a = a[z["v0"] >= vmin]
            out.append(a)
    out = [a for a in out if len(a)]
    return np.concatenate(out, 0) if out else None


def audit(dumpdir, label, vmin=0.0):
    print("==== %s   (v0 >= %.1f m/s)" % (label, vmin))
    for key in ("g", "cl", "ha0_ext", "ha", "ol"):
        p = load(dumpdir, key, vmin)
        if p is None:
            print("   %-8s ABSENT" % key)
            continue
        t = torch.from_numpy(p).to(torch.float64)
        # prepend the ego origin so the FIRST step's speed is recoverable
        t = torch.cat([torch.zeros(t.shape[0], 1, 2, dtype=t.dtype), t], dim=1)
        r = FD.assert_feasible(t, dt=DT)
        tag = " <- CONTROL, must be ~0" if key == "g" else ""
        print("   %-8s n=%d envelope_rate %.4f  kamm_over %.4f  max|a| %.3f  "
              "max|kappa| %.4f  peak_g mean %.3f max %.3f%s" % (
                  key, t.shape[0], r["envelope_rate"], r.get("kamm_over_rate", float("nan")),
                  r["max_abs_accel"], r["max_abs_kappa"], r["peak_g_mean"],
                  r["peak_g_max"], tag))


if __name__ == "__main__":
    print("constants: A_MAX %.3f  KAPPA_MAX %.3f  MU_KAMM %.3f  dt used %.2f s"
          % (FD.A_MAX_MPS2, FD.KAPPA_MAX_1PM, FD.MU_KAMM, DT))
    print()
    for i in range(0, len(sys.argv) - 1, 2):
        audit(sys.argv[1 + i], sys.argv[2 + i], VMIN)
        print()
