#!/usr/bin/env python3
"""Does `synth_lead_collision_rate` measure a COLLISION, or a longitudinal overlap
with a car that may be metres to the side?  cl_metrics computes it from x alone:

    "synth_collision": float(e["headway_m"] <= 0.0)      # headway = x - 5.242

with NO lateral gate, while `synth_inlane` (|y| <= 1.8) is computed separately and
never intersected with it.  A real bumper-to-bumper contact needs BOTH.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/cutin/rollouts")
sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

HALF_W_EGO = 0.9   # half the ego track width; |y| below this = genuine overlap
LANE = 1.8         # the panel's own in-lane band


def rows(arm, cond, ref):
    d = json.loads((ROOT / f"rollouts_{arm}_{cond}.json").read_text())
    out = {}
    for r in d["rollouts"]:
        for k, s in enumerate(r["steps"]):
            out[(r["start_frame"], k)] = s["lead"][ref]
    return out


print("=" * 96)
print("HOW MANY OF THE COUNTED 'COLLISIONS' ARE LATERALLY OVERLAPPING AT ALL?")
print("=" * 96)
for arm in ("flagship-v1", "refc-base"):
    for cond in ("lead25", "lead15", "lead8", "cutin"):
        for src, lbl in ((cond, "visible   "), ("empty", "counterfact")):
            R = rows(arm, src, cond)
            hits = [e for e in R.values() if e["headway_m"] <= 0]
            if not hits:
                print(f"{arm:12s} {cond:7s} {lbl} n_coll=0")
                continue
            ay = np.abs([e["y"] for e in hits])
            print(f"{arm:12s} {cond:7s} {lbl} n_coll={len(hits):4d} "
                  f"|y| median={np.median(ay):6.2f} "
                  f"frac|y|<={LANE}: {float((ay <= LANE).mean()):.3f}  "
                  f"frac|y|<={HALF_W_EGO}: {float((ay <= HALF_W_EGO).mean()):.3f}")
    print()

print("=" * 96)
print("HEADLINE RE-RUN WITH A LATERAL GATE: collision = (headway<=0) AND (|y|<=gate)")
print("=" * 96)
for gate, name in ((None, "NO GATE (as published)"), (LANE, "|y|<=1.8 (panel in-lane)"),
                   (HALF_W_EGO, "|y|<=0.9 (true overlap)")):
    print(f"\n--- {name} ---")
    for arm in ("flagship-v1", "refc-base"):
        for cond in ("lead25", "lead15", "lead8", "cutin"):
            A = rows(arm, cond, cond)
            B = rows(arm, "empty", cond)
            common = sorted(set(A) & set(B))
            f = (lambda e: float(e["headway_m"] <= 0)) if gate is None else \
                (lambda e: float(e["headway_m"] <= 0 and abs(e["y"]) <= gate))
            a = [f(A[c]) for c in common]
            b = [f(B[c]) for c in common]
            r = paired_episode_cluster_bootstrap(a, b, [c[0] for c in common])
            print(f"  {arm:12s} {cond:7s} vis={np.mean(a):.4f} cf={np.mean(b):.4f} "
                  f"delta={r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] sep={r['separated']}")

print()
print("=" * 96)
print("ARM CONTRAST at lead8 / cutin under the same gates")
print("=" * 96)
for gate, name in ((None, "NO GATE"), (LANE, "|y|<=1.8"), (HALF_W_EGO, "|y|<=0.9")):
    for cond in ("lead8", "cutin"):
        A = rows("flagship-v1", cond, cond)
        B = rows("refc-base", cond, cond)
        common = sorted(set(A) & set(B))
        f = (lambda e: float(e["headway_m"] <= 0)) if gate is None else \
            (lambda e: float(e["headway_m"] <= 0 and abs(e["y"]) <= gate))
        a = [f(A[c]) for c in common]
        b = [f(B[c]) for c in common]
        r = paired_episode_cluster_bootstrap(a, b, [c[0] for c in common])
        print(f"  {name:10s} {cond:7s} flag={np.mean(a):.4f} refc={np.mean(b):.4f} "
              f"delta={r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] sep={r['separated']}")

print()
print("=" * 96)
print("IN-LANE RATE per cell (the report says 40-66 %)")
print("=" * 96)
for arm in ("flagship-v1", "refc-base"):
    for cond in ("lead25", "lead15", "lead8", "cutin"):
        R = rows(arm, cond, cond)
        il = np.mean([abs(e["y"]) <= LANE for e in R.values()])
        infov = np.mean([abs(np.degrees(np.arctan2(e["y"], e["x"]))) <= 60 and e["x"] > 0
                         for e in R.values()])
        print(f"  {arm:12s} {cond:7s} in_lane={il:.4f} in_fov(approx)={infov:.4f}")
