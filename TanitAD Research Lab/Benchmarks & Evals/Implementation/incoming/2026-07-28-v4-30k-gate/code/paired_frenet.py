"""PAIRED along/cross (Frenet) decomposition: goal PRODUCED vs ORACLE, same 881 windows.

WHY THIS EXISTS. The gate's §1.1 decomposition is `taniteval.driving`'s own output — correct,
but UNPAIRED: two independent single-arm means with no interval on their difference. My first
attempt at a paired version was refused by its own self-check because I got the quantity wrong
in two ways at once: the ego frame instead of the **GT-tangent Frenet** frame, and a window-mean
instead of the value **at 2 s**.

This uses `driving.frenet()` itself rather than re-deriving it, and reproduces
`long_abs_2s_m = al[:, -1].abs()` / `lat_abs_2s_m = cr[:, -1].abs()` exactly as driving.py builds
them (driving.py:327, :335). The self-check below is the gate on quoting anything: if these do not
land on the numbers driving.py already printed, nothing is reported.

Estimator: paired episode-cluster bootstrap (taniteval.ci, B=2000, seed 0).
`overlapping_holdout_se` is not used. No GPU, no forward pass.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/workspace/tev/taniteval")
from taniteval import ci, rollout  # noqa: E402
from taniteval.driving import frenet  # noqa: E402

OUT = "/workspace/v4gate30k"
A_KEY, B_KEY = "v4fs-30k-produced", "v4fs-30k-oracle"   # delta = produced - oracle

# what taniteval.driving already printed for these arms — the bar this must clear
DRIVING_REF = {
    "v4fs-30k-oracle":   {"along": 1.0389, "cross": 0.5242},
    "v4fs-30k-produced": {"along": 1.4649, "cross": 0.5515},
}
TOL = 0.001

wa = rollout.load_windows(f"{OUT}/windows_{A_KEY}.pt")
wb = rollout.load_windows(f"{OUT}/windows_{B_KEY}.pt")

assert list(wa["eid"]) == list(wb["eid"]), "REFUSING a paired test on non-identical window sets"
assert torch.allclose(wa["gt"], wb["gt"], atol=1e-5), "GT differs -> not the same windows"
eid = wa["eid"]
print(f"[align] n_windows={len(eid)} n_episodes={len(set(eid))} eid+GT identical")


def frenet_2s(w):
    """Reproduce driving.py:327/:335 — along/cross AT the 2 s waypoint, absolute."""
    al, cr = frenet(w["pred"], w["gt"])          # each [N, 4], signed
    return al[:, -1].abs().numpy(), cr[:, -1].abs().numpy(), al[:, -1].numpy(), cr[:, -1].numpy()


a_al, a_cr, a_al_s, a_cr_s = frenet_2s(wa)
b_al, b_cr, b_al_s, b_cr_s = frenet_2s(wb)

print("\n=== self-check vs taniteval.driving (nothing is quoted unless this passes) ===")
ok = True
for key, al, cr in ((A_KEY, a_al, a_cr), (B_KEY, b_al, b_cr)):
    ref = DRIVING_REF[key]
    for lab, got, exp in (("along", float(al.mean()), ref["along"]),
                          ("cross", float(cr.mean()), ref["cross"])):
        d = abs(got - exp)
        if d > TOL:
            ok = False
        print(f"  {key:20s} {lab:5s} mine={got:.4f} driving={exp:.4f} d={d:.4f} "
              f"{'OK' if d <= TOL else 'MISMATCH'}")
if not ok:
    print("\n  => SELF-CHECK FAILED. Refusing to report a paired decomposition.")
    raise SystemExit(1)
print("  => self-check PASSED; the quantity matches driving.py exactly.")

print("\n=== PAIRED episode-cluster bootstrap, B=2000 (delta = PRODUCED - ORACLE) ===")
print(f"{'metric':20s} {'produced':>10s} {'oracle':>9s} {'delta':>10s} {'CI95':>24s}  sep")
rows = {}
tests = [
    ("long_abs_2s_m", a_al, b_al),
    ("lat_abs_2s_m",  a_cr, b_cr),
    ("long_signed_2s_m", a_al_s, b_al_s),   # sign = bias direction (+ = AHEAD of GT)
    ("lat_signed_2s_m",  a_cr_s, b_cr_s),   # + = LEFT of GT
]
for name, a, b in tests:
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0)
    sep = "SEPARATED" if r["separated"] else "overlaps 0"
    print(f"{name:20s} {a.mean():10.4f} {b.mean():9.4f} {r['delta']:+10.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {sep}")
    rows[name] = {"produced": float(a.mean()), "oracle": float(b.mean()),
                  "delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
                  "separated": bool(r["separated"])}

out = {"estimator": "paired episode-cluster bootstrap (taniteval.ci), B=2000, seed=0",
       "quantity": "driving.frenet() at the 2 s waypoint; reproduces driving.py:327/:335",
       "self_check_vs_driving": "PASSED (tol 0.001 on both arms, both axes)",
       "n_windows": len(eid), "n_episodes": len(set(eid)),
       "delta_sign": "produced - oracle", "rows": rows}
with open(f"{OUT}/paired_frenet.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\n-> {OUT}/paired_frenet.json")
