#!/usr/bin/env python3
"""Does the in-domain probe head (V) read well partly because its held-out fold sits on STREETS its training folds also drove?  CPU, read-only.
For each of the 1,058 C3 tokens: NN distance (x, y, 20cos h, 20sin h; same map) to the tokens of the OTHER four folds (= V's training tokens for that fold; fold = sha256(log) % 5, folds.json).
Then V's and H1's IoU by that distance, and the fraction of the V-minus-H1 gain that sits at far-from-training-folds frames."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C
import shf_arrays as SA
import d1_place_novelty as D1
import importlib
ARMS = ("H1-s0",)
Dn = SA.domain("navtest", ARMS)
P_nt = D1.poses(f"{C.AUXDIR}/label_store_navtest")
folds = json.load(open(os.path.join(C.PK, "raw", "2026-10-15-scene-tap-probe", "folds.json"), encoding="utf-8"))
fold_of = {r[0]: r[2] for r in folds["rows"]}
fo = np.array([fold_of[t] for t in Dn["tokens"]])
feats = [D1.feat(P_nt[k]) for k in Dn["keys"]]
d = np.full(Dn["n"], np.nan)
for f in range(5):
    for m in ("LV", "BOS", "PIT", "SG"):
        ref = np.array([feats[j][1] for j in range(Dn["n"]) if fo[j] != f and feats[j][0] == m])
        q = [j for j in range(Dn["n"]) if fo[j] == f and feats[j][0] == m]
        if len(q) == 0:
            continue
        if len(ref) == 0:
            d[q] = 1e9
            continue
        tr = cKDTree(ref)
        for j in q:
            d[j] = tr.query(feats[j][1])[0]
assert not np.isnan(d).any()
pl = np.load(os.path.join(C.SCRATCH, "place_dist.npz"))
dn_navtrain = pl["dn"]
exec(open(os.path.join(C.HERE, "d2_indomain_vs_transfer.py"), encoding="utf-8").read().split("units = {")[0].split("Dn = SA.domain")[0])  # imports only
tab = np.load(os.path.join(C.PK, "raw", "2026-10-11-c3-selection-headroom", "table.npz"), allow_pickle=False)
tab_tokens = [str(t) for t in tab["token"]]
pos = {t: i for i, t in enumerate(Dn["tokens"])}
b = SA.band()[None]
def load_V(dirname, seed):
    PM = np.zeros_like(Dn["TM"])
    for f in range(5):
        z = np.load(f"{C.ART}/{dirname}/units/V_s{seed}_f{f}.npz", allow_pickle=True)
        for r, hi in enumerate(z["held_idx"]):
            PM[pos[tab_tokens[int(hi)]]] = z["logit"][r] > 0
    return PM
def parts(PM):
    return (Dn["TM"] & PM & b).sum((1, 2)).astype(float), ((Dn["TM"] | PM) & b).sum((1, 2)).astype(float)
U = {"V2k_s0": parts(load_V("scene_tap_probe", 0)), "V10k_s0": parts(load_V("scene_tap_probe_a2", 0)), "H1-s0": parts(Dn["PM"]["H1-s0"])}
edges = [0, 2, 10, 1e9]
res = {"dist_to_other_folds_summary": {"median": float(np.median(d)), "share_le_2m": float((d <= 2).mean()), "share_le_10m": float((d <= 10).mean()), "share_gt_10m": float((d > 10).mean())},
       "dist_to_navtrain_training_summary": {"median": float(np.median(dn_navtrain)), "share_le_2m": float((dn_navtrain <= 2).mean()), "share_gt_10m": float((dn_navtrain > 10).mean())}}
bins = np.digitize(d, edges[1:-1])
res["by_dist_to_other_folds"] = {}
for k, nm in enumerate(["<2", "2-10", ">=10"]):
    m = bins == k
    res["by_dist_to_other_folds"][nm] = dict(n=int(m.sum()), **{u: float(U[u][0][m].sum() / U[u][1][m].sum()) for u in U})
res["V_minus_H1_gain_by_dist_to_other_folds"] = {nm: {"V2k": v["V2k_s0"] - v["H1-s0"], "V10k": v["V10k_s0"] - v["H1-s0"]} for nm, v in res["by_dist_to_other_folds"].items()}
res["spearman_V10k_frame_iou_vs_dist_other_folds"] = float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(d, U["V10k_s0"][0] / np.maximum(U["V10k_s0"][1], 1))[0])
res["spearman_H1_frame_iou_vs_dist_navtrain"] = float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(dn_navtrain, U["H1-s0"][0] / np.maximum(U["H1-s0"][1], 1))[0])
C.jdump(res, os.path.join(C.HERE, "d3_probe_place_leak.json"))
print(json.dumps(res, indent=1, sort_keys=True))
