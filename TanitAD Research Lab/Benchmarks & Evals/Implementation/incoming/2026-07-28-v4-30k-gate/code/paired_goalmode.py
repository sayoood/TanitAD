"""PAIRED episode-cluster bootstrap: goal ORACLE vs goal PRODUCED, same arm, same windows.

WHY THIS EXISTS. The 30k gate reported the two goal modes as TWO INDEPENDENT CIs
(oracle 0.6423 [0.5348, 0.7586], produced 0.8563 [0.7282, 1.0035]) and I said explicitly
that the +0.2140 difference "carries no interval". Two independent intervals cannot be
combined in quadrature here — the arms are scored on the SAME windows, so their estimates
are not independent, and quadrature is invalid as well as weaker. The paired test cancels
the shared per-window difficulty inside each draw.

This is the cheapest possible decision-grade measurement: NO forward pass, no GPU. It
reads the windows both gate runs already persisted.

DISCIPLINE
  * refuses outright if the two window sets are not identical (eid sequence AND ground
    truth), because a paired test on misaligned windows is meaningless, not merely noisy;
  * `overlapping_holdout_se` is not used anywhere;
  * the lateral/longitudinal split is SELF-CHECKED against the numbers taniteval.driving
    already printed. If my ego-frame convention disagrees with the harness, the
    decomposition is NOT quoted.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/tev/taniteval")
from taniteval import ci, rollout  # noqa: E402

OUT = "/workspace/v4gate30k"
A_KEY, B_KEY = "v4fs-30k-produced", "v4fs-30k-oracle"   # delta = produced - oracle

# what taniteval.driving printed for these two arms, for the self-check below
DRIVING_REF = {
    "v4fs-30k-oracle":   {"ade": 0.6423, "along": 1.0389, "cross": 0.5242},
    "v4fs-30k-produced": {"ade": 0.8563, "along": 1.4649, "cross": 0.5515},
}

wa = rollout.load_windows(f"{OUT}/windows_{A_KEY}.pt")
wb = rollout.load_windows(f"{OUT}/windows_{B_KEY}.pt")

same_eid = list(wa["eid"]) == list(wb["eid"])
print(f"[align] n_a={len(wa['eid'])} n_b={len(wb['eid'])} eid_identical={same_eid}")
assert same_eid, "REFUSING a paired test on non-identical window sets"
assert torch.allclose(wa["gt"], wb["gt"], atol=1e-5), "GT differs -> not the same windows"
eid = wa["eid"]
print(f"[align] GT identical, n_windows={len(eid)}, n_episodes={len(set(eid))}")


def ade_per_window(w):
    return (w["pred"] - w["gt"]).norm(dim=-1).mean(dim=1).numpy()


def axis_per_window(w, axis):
    """|component| of the error, ego frame: axis 0 = along-track, 1 = cross-track."""
    return (w["pred"] - w["gt"])[..., axis].abs().mean(dim=1).numpy()


# ---- SELF-CHECK: does my ego-frame convention reproduce driving.py? ----------
print("\n=== self-check vs taniteval.driving (my split is quoted ONLY if this passes) ===")
ok = True
for key, w in ((A_KEY, wa), (B_KEY, wb)):
    ref = DRIVING_REF[key]
    got = {"ade": float(ade_per_window(w).mean()),
           "along": float(axis_per_window(w, 0).mean()),
           "cross": float(axis_per_window(w, 1).mean())}
    for k in ("ade", "along", "cross"):
        d = abs(got[k] - ref[k])
        flag = "OK" if d <= 0.01 else "MISMATCH"
        if d > 0.01:
            ok = False
        print(f"  {key:20s} {k:6s} mine={got[k]:.4f} driving={ref[k]:.4f} d={d:.4f} {flag}")
DECOMP_QUOTABLE = ok
print(f"  => decomposition quotable: {DECOMP_QUOTABLE}")

# ---- the paired tests -------------------------------------------------------
print("\n=== PAIRED episode-cluster bootstrap, B=2000 (delta = PRODUCED - ORACLE) ===")
print(f"{'metric':12s} {'produced':>10s} {'oracle':>9s} {'delta':>10s} {'CI95':>24s}  sep")

metrics = [("ade_0_2s", ade_per_window(wa), ade_per_window(wb))]
if DECOMP_QUOTABLE:
    metrics += [("along", axis_per_window(wa, 0), axis_per_window(wb, 0)),
                ("cross", axis_per_window(wa, 1), axis_per_window(wb, 1))]
for i, lab in enumerate(("de@0.5s", "de@1s", "de@1.5s", "fde@2s")):
    metrics.append((lab,
                    (wa["pred"][:, i] - wa["gt"][:, i]).norm(dim=-1).numpy(),
                    (wb["pred"][:, i] - wb["gt"][:, i]).norm(dim=-1).numpy()))

rows = {}
for name, a, b in metrics:
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0)
    sep = "SEPARATED" if r["separated"] else "overlaps 0"
    print(f"{name:12s} {a.mean():10.4f} {b.mean():9.4f} {r['delta']:+10.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {sep}")
    rows[name] = {"produced": float(a.mean()), "oracle": float(b.mean()),
                  "delta": float(r["delta"]), "lo": float(r["lo"]),
                  "hi": float(r["hi"]), "separated": bool(r["separated"])}

# ---- each mode against the constant-velocity floor, paired ------------------
print("\n=== each goal mode vs the CV floor (paired, same windows) ===")
cv = (wa["cv"] - wa["gt"]).norm(dim=-1).mean(dim=1).numpy()
for key, w in ((A_KEY, wa), (B_KEY, wb)):
    a = ade_per_window(w)
    r = ci.paired_episode_cluster_bootstrap(a, cv, eid, n_boot=2000, seed=0)
    sep = "SEPARATED" if r["separated"] else "overlaps 0"
    verdict = "WORSE than CV" if r["delta"] > 0 else "better than CV"
    print(f"{key:20s} arm={a.mean():.4f} cv={cv.mean():.4f} "
          f"delta={r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {sep} -> {verdict}")
    rows[f"{key}_vs_cv"] = {"arm": float(a.mean()), "cv": float(cv.mean()),
                            "delta": float(r["delta"]), "lo": float(r["lo"]),
                            "hi": float(r["hi"]), "separated": bool(r["separated"])}

out = {"estimator": "paired episode-cluster bootstrap (taniteval.ci), B=2000, seed=0",
       "n_windows": len(eid), "n_episodes": len(set(eid)),
       "delta_sign": "produced - oracle",
       "decomposition_self_check_passed": DECOMP_QUOTABLE,
       "rows": rows}
with open(f"{OUT}/paired_goalmode.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\n-> {OUT}/paired_goalmode.json")
