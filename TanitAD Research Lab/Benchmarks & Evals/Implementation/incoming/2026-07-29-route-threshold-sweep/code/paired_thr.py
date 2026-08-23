"""PRE-REGISTERED PRIMARY for the route-threshold arm: paired Delta ade_0_2s.

Arm B (--route-thr 0.35) vs Arm A (default tanh(1.0)), same checkpoint, same parity val,
same 881 windows / 40 episodes. Estimator: paired episode-cluster bootstrap (B=2000,
seed 0). overlapping_holdout_se not used. Refuses outright unless the two window sets are
identical in eid sequence AND ground truth.

Outcomes were committed in PREREG_thr035_ade.md before the arm was run.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/workspace/tev/taniteval")
from taniteval import ci, rollout            # noqa: E402
from taniteval.driving import frenet         # noqa: E402

OUT = "/workspace/v4gate30k"
A_KEY, B_KEY = "v4fs-30k-produced-thr035", "v4fs-30k-produced"   # delta = thr035 - default

wa = rollout.load_windows(f"{OUT}/windows_{A_KEY}.pt")
wb = rollout.load_windows(f"{OUT}/windows_{B_KEY}.pt")
assert list(wa["eid"]) == list(wb["eid"]), "REFUSING a paired test on non-identical windows"
assert torch.allclose(wa["gt"], wb["gt"], atol=1e-5), "GT differs -> not the same windows"
eid = wa["eid"]
print(f"[align] n={len(eid)} episodes={len(set(eid))} eid+GT identical")


def ade(w):
    return (w["pred"] - w["gt"]).norm(dim=-1).mean(dim=1).numpy()


def fde(w):
    return (w["pred"][:, -1] - w["gt"][:, -1]).norm(dim=-1).numpy()


def fr(w, axis):
    al, cr = frenet(w["pred"], w["gt"])
    return (al if axis == 0 else cr)[:, -1].abs().numpy()


tests = [("ade_0_2s", ade(wa), ade(wb)),
         ("fde@2s", fde(wa), fde(wb)),
         ("long_abs_2s", fr(wa, 0), fr(wb, 0)),
         ("lat_abs_2s", fr(wa, 1), fr(wb, 1))]

print(f"{'metric':12s} {'thr035':>9s} {'default':>9s} {'delta':>10s} {'CI95':>22s}  sep")
rows = {}
for name, a, b in tests:
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0)
    sep = "SEPARATED" if r["separated"] else "overlaps 0"
    print(f"{name:12s} {a.mean():9.4f} {b.mean():9.4f} {r['delta']:+10.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {sep}")
    rows[name] = {"thr035": float(a.mean()), "default": float(b.mean()),
                  "delta": float(r["delta"]), "lo": float(r["lo"]),
                  "hi": float(r["hi"]), "separated": bool(r["separated"])}

prim = rows["ade_0_2s"]
verdict = ("OUTCOME A — separated" if prim["separated"] else
           "OUTCOME B — NOT separated: the route fix does not transfer to trajectory error")
print(f"\nPRE-REGISTERED PRIMARY (ade_0_2s): {verdict}")
json.dump({"estimator": "paired episode-cluster bootstrap (taniteval.ci), B=2000, seed=0",
           "delta_sign": "thr035 - default", "n_windows": len(eid),
           "n_episodes": len(set(eid)), "verdict": verdict, "rows": rows},
          open(f"{OUT}/paired_thr035.json", "w"), indent=1)
print(f"-> {OUT}/paired_thr035.json")
