#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the cutin-scene stream's headline numbers.

Reads ONLY the banked raw rollout dumps. Does NOT import cl_metrics or cutin_report.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/cutin/rollouts")
sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap, episode_cluster_bootstrap  # noqa: E402

ARMS = ("flagship-v1", "refc-base")
CONDS = ("empty", "behind", "lead25", "lead15", "lead8", "cutin")
MAN = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")


def load(arm, cond):
    return json.loads((ROOT / f"rollouts_{arm}_{cond}.json").read_text())


print("=" * 78)
print("A. SHAPE / COUNTS / PAIRING KEYS")
print("=" * 78)
shapes = {}
for arm in ARMS:
    for cond in CONDS:
        d = load(arm, cond)
        nr = len(d["rollouts"])
        starts = [r["start_frame"] for r in d["rollouts"]]
        ns = [len(r["steps"]) for r in d["rollouts"]]
        igt = sorted({s["i_gt"] for r in d["rollouts"] for s in r["steps"]})
        shapes[(arm, cond)] = (nr, starts, sum(ns), len(d["gt"]))
        print(f"{arm:12s} {cond:7s} rollouts={nr} starts={starts} steps={sum(ns)} "
              f"gt_len={len(d['gt'])} i_gt[min,max]=[{igt[0]},{igt[-1]}] ckpt={d['ckpt']}")

print()
print("=" * 78)
print("B. TRUNCATION: how many windows would drop_truncated=True remove?")
print("=" * 78)
for arm in ARMS:
    for cond in ("empty", "lead8"):
        d = load(arm, cond)
        N = len(d["gt"])
        tr = sum(1 for r in d["rollouts"] for s in r["steps"] if (s["i_gt"] + 20) >= N)
        tot = sum(len(r["steps"]) for r in d["rollouts"])
        print(f"{arm:12s} {cond:7s} truncated={tr}/{tot}  -> rows kept={tot - tr}")

print()
print("=" * 78)
print("C. HALF-LENGTH CONSISTENCY (the control-caught bug: was the FIX banked?)")
print("   For every condition, x - headway_m must be the SAME constant")
print("   (= EGO_FRONT_OVERHANG_M 3.7 + half_len 1.542 = 5.242)")
print("=" * 78)
for arm in ARMS:
    for cond in CONDS:
        d = load(arm, cond)
        offs = set()
        for r in d["rollouts"]:
            for s in r["steps"]:
                for ref, e in (s.get("lead") or {}).items():
                    if e.get("headway_m") is not None and e.get("x") is not None:
                        offs.add(round(e["x"] - e["headway_m"], 3))
        print(f"{arm:12s} {cond:7s} distinct (x - headway) offsets: {sorted(offs)}")

print()
print("=" * 78)
print("D. NOISE FLOOR: is `behind` bit-identical to `empty`?")
print("=" * 78)
for arm in ARMS:
    a, b = load(arm, "empty"), load(arm, "behind")
    fields = ("x", "y", "yaw", "v", "steer", "accel")
    diffs = []
    n = 0
    for ra, rb in zip(a["rollouts"], b["rollouts"]):
        for sa, sb in zip(ra["steps"], rb["steps"]):
            va = [sa["ego"][0], sa["ego"][1], sa["ego"][3], sa["v"], sa["steer"], sa["accel"]]
            vb = [sb["ego"][0], sb["ego"][1], sb["ego"][3], sb["v"], sb["steer"], sb["accel"]]
            diffs.append(np.abs(np.array(va) - np.array(vb)))
            n += len(va)
    D = np.array(diffs)
    # also compare the maneuver logits, which the trajectory check does NOT cover
    la, lb = [], []
    for ra, rb in zip(a["rollouts"], b["rollouts"]):
        for sa, sb in zip(ra["steps"], rb["steps"]):
            la.append(sa["extra"]["maneuver_logits"])
            lb.append(sb["extra"]["maneuver_logits"])
    L = np.abs(np.array(la) - np.array(lb))
    print(f"{arm:12s} traj elements={n} max|diff|={D.max():.3e}  "
          f"maneuver_logits max|diff|={L.max():.3e}")

print()
print("=" * 78)
print("E. COLLISION RATE (headway<=0) — visible vs matched counterfactual, lead8")
print("=" * 78)


def rows_for(arm, cond, lead_ref, drop_trunc=True):
    """Per-tick rows with (eid, k) keys and the lead geometry under `lead_ref`."""
    d = load(arm, cond)
    N = len(d["gt"])
    out = {}
    for r in d["rollouts"]:
        for k, s in enumerate(r["steps"]):
            if drop_trunc and (s["i_gt"] + 20) >= N:
                continue
            L = (s.get("lead") or {}).get(lead_ref) if lead_ref else None
            out[(r["start_frame"], k)] = {"s": s, "L": L, "eid": r["start_frame"]}
    return out


for arm in ARMS:
    for cond in ("lead25", "lead15", "lead8", "cutin", "behind"):
        vis = rows_for(arm, cond, cond)
        cf = rows_for(arm, "empty", cond)
        common = sorted(set(vis) & set(cf))
        a = [float(vis[c]["L"]["headway_m"] <= 0.0) for c in common]
        b = [float(cf[c]["L"]["headway_m"] <= 0.0) for c in common]
        eid = [c[0] for c in common]
        r = paired_episode_cluster_bootstrap(a, b, eid)
        hw_a = np.mean([vis[c]["L"]["headway_m"] for c in common])
        tg = [vis[c]["L"]["time_gap_s"] for c in common if vis[c]["L"]["time_gap_s"] is not None]
        print(f"{arm:12s} {cond:7s} n={len(common)} coll_vis={np.mean(a):.4f} "
              f"coll_cf={np.mean(b):.4f} delta={r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] "
              f"sep={r['separated']}  mean_hw={hw_a:.3f} mean_tg={np.mean(tg):.4f}")

print()
print("=" * 78)
print("F. HEADWAY delta (visible - counterfactual)")
print("=" * 78)
for arm in ARMS:
    for cond in ("lead25", "lead15", "lead8", "cutin", "behind"):
        vis = rows_for(arm, cond, cond)
        cf = rows_for(arm, "empty", cond)
        common = sorted(set(vis) & set(cf))
        a = [vis[c]["L"]["headway_m"] for c in common]
        b = [cf[c]["L"]["headway_m"] for c in common]
        r = paired_episode_cluster_bootstrap(a, b, [c[0] for c in common])
        print(f"{arm:12s} {cond:7s} headway delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] sep={r['separated']}")

print()
print("=" * 78)
print("G. MANOEUVRE HEAD class share (argmax of maneuver_logits), all cells")
print("=" * 78)
for arm in ARMS:
    for cond in CONDS:
        d = load(arm, cond)
        N = len(d["gt"])
        cls = []
        for r in d["rollouts"]:
            for s in r["steps"]:
                if (s["i_gt"] + 20) >= N:
                    continue
                cls.append(int(np.argmax(s["extra"]["maneuver_logits"])))
        cls = np.array(cls)
        share = {MAN[c]: round(float((cls == c).mean()), 4) for c in range(5)}
        print(f"{arm:12s} {cond:7s} n={cls.size} {share}")

print()
print("=" * 78)
print("H. ARM CONTRAST at lead8 (flagship - refc), collision / headway / time-gap")
print("=" * 78)
for cond in ("lead8", "cutin"):
    A = rows_for("flagship-v1", cond, cond)
    B = rows_for("refc-base", cond, cond)
    common = sorted(set(A) & set(B))
    for name, fn in (("collision", lambda r: float(r["L"]["headway_m"] <= 0.0)),
                     ("headway_m", lambda r: r["L"]["headway_m"]),
                     ("time_gap_s", lambda r: r["L"]["time_gap_s"] if r["L"]["time_gap_s"] is not None else np.nan)):
        a = [fn(A[c]) for c in common]
        b = [fn(B[c]) for c in common]
        a = np.array(a, float); b = np.array(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        eid = [c[0] for c, o in zip(common, ok) if o]
        r = paired_episode_cluster_bootstrap(a[ok], b[ok], eid)
        print(f"{cond:7s} {name:11s} n={ok.sum()} delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] sep={r['separated']}")

print()
print("=" * 78)
print("I. IS ade_0_2s == dist_to_gt == cross_track_abs?  (duplicate-metric check)")
print("=" * 78)
print("  (checked against the panel in the next script)")

print()
print("=" * 78)
print("J. accel < -0.5 m/s2 fraction, and mean commanded accel")
print("=" * 78)
for arm in ARMS:
    for cond in CONDS:
        d = load(arm, cond)
        N = len(d["gt"])
        acc = [s["accel"] for r in d["rollouts"] for s in r["steps"] if (s["i_gt"] + 20) < N]
        vt = [s["v_target"] for r in d["rollouts"] for s in r["steps"] if (s["i_gt"] + 20) < N]
        acc = np.array(acc)
        print(f"{arm:12s} {cond:7s} mean_accel={acc.mean():+.4f} "
              f"frac(<-0.5)={float((acc < -0.5).mean()):.4f} "
              f"min={acc.min():+.4f} max={acc.max():+.4f} mean_v_target={np.mean(vt):.4f}")
