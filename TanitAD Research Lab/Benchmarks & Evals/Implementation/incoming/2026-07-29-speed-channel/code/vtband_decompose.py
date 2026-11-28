"""TASK #44 STEP 1 — is the vt_band loss in the REGRESSOR or in the DISCRETISATION?

CONTEXT. Route was ruled out: fixing its threshold lifted balanced route accuracy
0.4242 -> 0.5493 yet moved paired ade_0_2s only +0.0022 [-0.0008, +0.0055] (a PRECISE null
bounding route's share of the oracle-vs-produced gap at <= 2.6 %). The gap is LONGITUDINAL
(paired long_abs_2s +0.4260 vs lat_abs_2s +0.0274), so the speed channels are what remain:
vt_band exact agreement 0.1725, and tspeed_5s R2 0.7635 with RMSE 4.4545 m/s (~16 km/h).

vt_band is NOT independently predicted — `goal_modes.scalars_to_goal` derives it by applying
the labeler's OWN banding function to the regressed tspeed_5s. So exactly the same question
that cracked route applies here: is the damage in the CONTINUOUS estimate, or in the
DISCRETE mapping laid over it?

THE DECOMPOSITION (free — no GPU, no re-run; reads the per-window dump the eval already made):
  CEILING   band(TRUE tspeed)  vs oracle vt_band  -> how much the MAPPING alone can deliver
  ACTUAL    band(PRED tspeed)  vs oracle vt_band  -> what we get (should reproduce 0.1725)
  The gap between them is the REGRESSOR's contribution.

BOTH READINGS COMMITTED IN ADVANCE:
 * If CEILING is ~1.0 and ACTUAL is ~0.17, the mapping is faithful and essentially ALL the
   loss is the regressor -> the lever is tspeed_5s itself (a harder, real modelling problem),
   and no threshold trick will help. This would make vt_band UNLIKE route.
 * If CEILING is itself far below 1.0, the banding is lossy on its own terms and part of the
   deficit is recoverable by changing the DISCRETISATION -> a cheap lever, like route was.
 * A middle result apportions the two, which is still the useful answer.
⚠️ Either way this measures the GOAL channel only. Route taught us that a channel-metric
gain need not move ADE at all — any fix here must still clear a paired ADE test.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
from tanitad.lake.vocab import VTARGET_TOKENS, vtarget_band  # noqa: E402

DUMP = "/workspace/v4gate30k/goalagree_v4fs-30k-produced.pt"
TSPEED_COL = 3          # scalar_order: ttm, curv_3s, curv_5s, tspeed_5s


def band_of(v):
    toks = list(VTARGET_TOKENS)
    return np.array([toks.index(vtarget_band(float(x))) for x in v], dtype=np.int64)


def main() -> None:
    d = torch.load(DUMP, map_location="cpu", weights_only=False)
    P, T, M = d["sc_pred"], d["sc_true"], d["sc_mask"]
    m = M[:, TSPEED_COL].bool()
    p = P[m, TSPEED_COL].double().numpy()
    t = T[m, TSPEED_COL].double().numpy()
    print(f"tspeed_5s: n={m.sum().item()} of {M.shape[0]} windows "
          f"({100*m.float().mean().item():.1f}% coverage)")
    print(f"  pred mean {p.mean():.4f}  true mean {t.mean():.4f}  "
          f"RMSE {np.sqrt(((p-t)**2).mean()):.4f} m/s")

    b_true = band_of(t)
    b_pred = band_of(p)
    n = len(b_true)
    print(f"\n  {len(VTARGET_TOKENS)} bands; oracle-band distribution spans "
          f"{b_true.min()}..{b_true.max()}")

    # ACTUAL: predicted-speed band vs true-speed band
    exact = (b_pred == b_true).mean()
    w1 = (np.abs(b_pred - b_true) <= 1).mean()
    print(f"\n=== ACTUAL (band of PREDICTED tspeed vs band of TRUE tspeed) ===")
    print(f"  exact   = {exact:.4f}      (harness reported 0.1725 vs the ORACLE vt_band)")
    print(f"  within1 = {w1:.4f}      (harness reported 0.3837)")

    # CEILING: the mapping applied to the TRUE speed is exact by construction here, so the
    # informative ceiling is how much band resolution the speed error destroys.
    err = np.abs(p - t)
    print(f"\n=== WHAT THE REGRESSOR ERROR COSTS IN BANDS ===")
    print(f"  |speed error|: mean {err.mean():.3f}  median {np.median(err):.3f}  "
          f"p90 {np.quantile(err,0.9):.3f} m/s")
    bd = np.abs(b_pred.astype(int) - b_true.astype(int))
    print(f"  |band error| : mean {bd.mean():.3f}  median {np.median(bd):.3f}  "
          f"p90 {np.quantile(bd,0.9):.3f} bands")
    # band width implied
    order = np.argsort(t)
    ts, bs = t[order], b_true[order]
    edges = ts[np.where(np.diff(bs) != 0)[0]]
    if len(edges) > 1:
        widths = np.diff(edges)
        print(f"  implied band width: median {np.median(widths):.3f} m/s "
              f"(min {widths.min():.3f}, max {widths.max():.3f})")
        print(f"  => RMSE {np.sqrt(((p-t)**2).mean()):.3f} m/s spans "
              f"~{np.sqrt(((p-t)**2).mean())/np.median(widths):.1f} median bands")

    # counterfactual: perfect banding of a perfect speed = 1.0 by construction.
    # The decision-relevant counterfactual is a COARSER banding.
    print(f"\n=== COUNTERFACTUAL: would COARSER bands recover agreement? ===")
    for k in (1, 2, 3, 4):
        agree = (np.abs(b_pred - b_true) <= k).mean()
        print(f"  tolerate +-{k} band(s): {agree:.4f}")
    print("\n  (this is the same shape as the route threshold question: if tolerating a")
    print("   band or two recovers most of the agreement, the DISCRETISATION is too fine")
    print("   for the regressor's accuracy; if not, the regressor is the binding limit.)")


if __name__ == "__main__":
    main()
