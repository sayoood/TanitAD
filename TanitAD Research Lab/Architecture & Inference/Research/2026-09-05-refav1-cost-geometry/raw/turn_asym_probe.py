"""WHY IS A SIGN-SYMMETRIC PENALTY ASYMMETRIC IN OUTCOME?

`W_KAPPA` penalises `k^2`, which cannot distinguish left from right. Yet MEASURED:
under a decoded TURN_L the realised mean `k^2` falls 98.0 % while under TURN_R it falls
27.9 %. A sign-symmetric cost cannot produce that alone, so the asymmetry must live in
something the cost INTERACTS with. The cheapest candidate is the DEMAND: if the corpus's
left turns need more curvature than its right turns, a quadratic penalty pulls the left
ones proportionally further from their target.

This measures the demand from the GROUND-TRUTH paths, binned by the HARNESS'S OWN true
class (`lat_label` in the decisions dump) -- so it is a statement about the corpus, and
it does not re-derive a labeller.

⛔ TWO CONTROLS, BOTH OF WHICH KILLED THE FIRST VERSION OF THIS PROBE:
  1. `kappa = yawrate / speed` has a LOW-SPEED SINGULARITY. Version 1 ran at every
     window and read `max|k|GT` = **1315.36** against `kappa_max` 0.200, with a left-bin
     p90 of 394.7 -- nonsense, and the control caught it. Fixed by the same `v0 >= 2 m/s`
     cut `feas_audit.py` uses for exactly this reason, PLUS a per-step speed floor.
     The control must now read `max|k|GT` <= KAPPA_MAX or the block is INADMISSIBLE.
  2. Version 1 classified turns by the sign of the net GT yaw and got left 8 / right 10,
     against the harness's reported n_true of 11 / 8. A DIFFERENT LABELLER IS A
     DIFFERENT EXPERIMENT. Fixed by reading `lat_label`; the counts must now match the
     record's `per_class.n_true` exactly, and the probe says so.
ASCII only.
"""
import glob
import json
import os
import sys

import numpy as np

P4 = "C:/Users/Admin/refav1_margin/p4out"
KAPPA_MAX = 0.200
VMIN = 2.0
SPEED_FLOOR = 2.0          # per-step, for the kappa denominator


def signed_kappa(xy, dt=0.2, floor=SPEED_FLOOR):
    p = np.concatenate([np.zeros((xy.shape[0], 1, 2)), xy], axis=1)
    v = np.diff(p, axis=1) / dt
    sp = np.linalg.norm(v, axis=-1)
    th = np.unwrap(np.arctan2(v[..., 1], v[..., 0]), axis=-1)
    dth = np.diff(th, axis=-1) / dt
    spd = sp[:, 1:]
    k = np.where(spd >= floor, dth / np.maximum(spd, 1e-6), np.nan)
    return k, sp, th


def main(tag="ccos_argmax"):
    rec = json.load(open("%s/rec_%s.json" % (P4, tag), encoding="utf-8"))
    ld = rec["arms"]["cl"]["four_families"]["tactical"]["lateral_decision"]
    order = ld["class_order"]
    want = {c: ld["per_class"][c]["n_true"] for c in order}
    print("arm: %s" % rec["arm"])
    print("class_order            : %s" % order)
    print("reported per-class n_true: %s" % want)

    G, V0, LAB = [], [], []
    for f in sorted(glob.glob("%s/dump_%s/ep*.npz" % (P4, tag))):
        z = np.load(f, allow_pickle=True)
        dpath = os.path.join(os.path.dirname(f), "decisions",
                             os.path.basename(f))
        if not os.path.exists(dpath):
            print("MISSING decisions for %s - block INADMISSIBLE" % os.path.basename(f))
            return
        zd = np.load(dpath, allow_pickle=True)
        G.append(z["g"]); V0.append(z["v0"]); LAB.append(zd["lat_label"])
    g = np.concatenate(G, 0).astype(np.float64)
    v0 = np.concatenate(V0, 0).astype(np.float64)
    lab = np.concatenate(LAB, 0)

    got = {c: int((lab == i).sum()) for i, c in enumerate(order)}
    print("probe    per-class n_true: %s" % got)
    if got != want:
        print("\n\u26d4 CONTROL 2 FAILED: the probe's class counts do not match the record's."
              "\n   A different labeller is a different experiment - block INADMISSIBLE.")
        return
    print("   CONTROL 2 PASSED (same labeller)")

    kg, _, _ = signed_kappa(g)
    mx = np.nanmax(np.abs(kg))
    print("\nCONTROL 1 - GT feasibility: max|k|GT = %.4f  (KAPPA_MAX %.3f)  %s"
          % (mx, KAPPA_MAX, "PASSED" if mx <= KAPPA_MAX else
             "FAILED - block INADMISSIBLE"))
    if mx > KAPPA_MAX:
        return

    keep = v0 >= VMIN
    print("near-stationary cut v0 >= %.1f m/s keeps %d of %d windows"
          % (VMIN, keep.sum(), len(g)))
    peak = np.zeros(len(kg))
    for i in range(len(kg)):
        row = kg[i][~np.isnan(kg[i])]
        peak[i] = row[np.abs(row).argmax()] if len(row) else np.nan

    print("\n%-11s %5s %12s %12s %12s" % ("true class", "n", "med|k|GT",
                                          "p90|k|GT", "med v0"))
    med = {}
    for i, c in enumerate(order):
        m = (lab == i) & keep & ~np.isnan(peak)
        if m.sum() == 0:
            print("%-11s %5d  (empty after the cut)" % (c, 0)); continue
        ak = np.abs(peak[m])
        med[c] = float(np.median(ak))
        print("%-11s %5d %12.5f %12.5f %12.2f"
              % (c, m.sum(), med[c], np.quantile(ak, 0.9), np.median(v0[m])))
    if "turn_left" in med and "turn_right" in med and med["turn_right"] > 0:
        r = med["turn_left"] / med["turn_right"]
        print("\nDEMAND RATIO median|k|GT  left / right = %.3f" % r)
        print("(the measured SUPPRESSION ratio of W_KAPPA was 98.0 %% / 27.9 %% "
              "= 3.5x in suppressed FRACTION)")


if __name__ == "__main__":
    main(*sys.argv[1:])
