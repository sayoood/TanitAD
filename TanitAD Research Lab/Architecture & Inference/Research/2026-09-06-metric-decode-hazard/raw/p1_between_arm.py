"""P1e - the banked T1 dumps hold DECODED WAYPOINTS ONLY (g / cl / ha / v0 / ws),
so a readout FITTED AT EVAL TIME cannot be applied to them: the readout's INPUT
(the per-step latent transition) was never dumped.

What they CAN still settle, at zero GPU: the three T1-read arms carry a
BIT-IDENTICAL `step_readout_op`. How much of their decoded trajectory is the
ARM, and how much is the shared readout?

CONTROLS
  * reproduce each arm's published ADE from the dump (must match t1.json).
  * paired on the SAME (episode, window) grid, or the comparison is invalid.
"""
import glob
import json
import os

import numpy as np

ROOT = "/home/nvidia/t1dumps"
ARMS = ["emao14_30k", "emao14_30k_tauramp", "o14fut30k"]


def load(arm):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, arm, "dumps", "ep*.npz"))):
        d = np.load(f, allow_pickle=True)
        eid = int(d["eid"][0])
        out[eid] = {k: d[k] for k in ("g", "cl", "ha", "v0", "ws")}
    return out


A = {a: load(a) for a in ARMS}
eids = sorted(set.intersection(*[set(A[a]) for a in ARMS]))
print(f"[grid] episodes common to all arms: {len(eids)}")


def ade(p, q):
    return float(np.linalg.norm(p - q, axis=-1).mean())


def paired(a, b, key_a, key_b):
    num = tot = 0.0
    n = 0
    for e in eids:
        wa, wb = A[a][e]["ws"], A[b][e]["ws"]
        if wa.shape != wb.shape or not np.array_equal(wa, wb):
            m = np.intersect1d(wa, wb)
            ia = np.searchsorted(wa, m)
            ib = np.searchsorted(wb, m)
        else:
            ia = ib = np.arange(len(wa))
        pa, pb = A[a][e][key_a][ia], A[b][e][key_b][ib]
        d = np.linalg.norm(pa - pb, axis=-1)
        num += d.sum()
        tot += d.size
        n += len(ia)
    return num / tot, n


res = {"n_episodes": len(eids), "note": "dumps hold decoded waypoints only"}
print("\n=== CONTROL: each arm's own ADE from the dump (vs its t1.json) ===")
res["arm_vs_gt"] = {}
for a in ARMS:
    cl, n = paired(a, a, "cl", "g")
    ha, _ = paired(a, a, "ha", "g")
    res["arm_vs_gt"][a] = {"ade_cl": cl, "ade_ha": ha, "n_windows": n}
    print(f"  {a:22s} ade(cl,g)={cl:8.4f}  ade(ha,g)={ha:8.4f}  n={n}")

print("\n=== BETWEEN ARMS: how far apart are the three decoded rollouts? ===")
res["between_arms"] = {}
for i, a in enumerate(ARMS):
    for b in ARMS[i + 1:]:
        d_cl, n = paired(a, b, "cl", "cl")
        d_ha, _ = paired(a, b, "ha", "ha")
        res["between_arms"][f"{a}__vs__{b}"] = {"cl": d_cl, "ha": d_ha, "n": n}
        print(f"  {a:22s} vs {b:22s}  cl={d_cl:8.4f}  ha={d_ha:8.4f}")

print("\n=== GT-vs-GT control (must be EXACTLY 0: same corpus, same grid) ===")
res["gt_control"] = {}
for i, a in enumerate(ARMS):
    for b in ARMS[i + 1:]:
        d, _ = paired(a, b, "g", "g")
        res["gt_control"][f"{a}__vs__{b}"] = d
        print(f"  {a:22s} vs {b:22s}  ade(g,g)={d:.10f}")

print("\n=== the ratio that matters ===")
base = np.mean([res["arm_vs_gt"][a]["ade_cl"] for a in ARMS])
btw = np.mean([v["cl"] for v in res["between_arms"].values()])
res["mean_arm_vs_gt_cl"] = float(base)
res["mean_between_arm_cl"] = float(btw)
res["between_over_error"] = float(btw / base)
print(f"  mean ade(cl, GT)            = {base:.4f} m")
print(f"  mean ade(cl_A, cl_B)        = {btw:.4f} m")
print(f"  between-arm / arm-vs-GT     = {btw/base:.4f}")

print("\n=== the published deltas these have to be read against ===")
for a in ARMS:
    r = res["arm_vs_gt"][a]
    print(f"  {a:22s} cl-ha = {r['ade_cl']-r['ade_ha']:+8.4f} m")
print(json.dumps(res, indent=1)[:80] + " ...")
with open("/tmp/p1_between_arm.json", "w") as fh:
    json.dump(res, fh, indent=1)
print("[wrote] /tmp/p1_between_arm.json")
