"""SH5 C-REACH diagnosis, probe 1 (CPU, read-only): what can a reader of the PLANTED channel achieve at best?
Oracle within-set DAC AUC of the planted signed key clearances, plant-entry counts per token, and plant ambiguity (same (x, y), different clearance) inside a token."""
import json, sys, numpy as np
sys.path.insert(0, "C:/Users/Admin/par18/sh5/code")
import sh5_lib as L, sh5_train_eval as TE
DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, row_of = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
FL, _ = L.refs()
lab = D["lab"][..., L.DAC_COL]; sc = D["sc_key"].astype(np.float64); code = D["code"]
print("N", D["N"], "T", D["T"], "logs", D["n_logs"], "fold sizes", np.bincount(D["fold"]).tolist())
print("label values", np.unique(lab, return_counts=True))
def aucs(z, m=None):
    m = np.ones(len(z), bool) if m is None else m
    st = FL.pair_stats(z[m], lab[m] == 0.0, lab[m] == 1.0, code[m], names=("ALLQ", "Q4", "Q123"))
    return {k: round(float(n.sum() / d.sum()), 4) if d.sum() > 0 else None for k, (n, d) in st.items()}
f0 = D["fold"] == 0
R = {}
R["P2_row"] = {"all": aucs(D["z6"][..., 1]), "fold0": aucs(D["z6"][..., 1], f0)}
mn = sc.min(-1)
R["min_k_signed_clear"] = {"all": aucs(mn), "fold0": aucs(mn, f0)}
R["min_k_abs_clear(tgt)"] = {"all": aucs(D["tgt"].min(-1).astype(np.float64)), "fold0": aucs(D["tgt"].min(-1).astype(np.float64), f0)}
for k in range(9):
    R[f"sc_key_t{k}"] = {"fold0": aucs(sc[..., k], f0)}
# capped as planted: clip to +-10
R["min_k_signed_clear_clip10"] = {"fold0": aucs(np.clip(sc, -10, 10).min(-1), f0)}
# label vs plant consistency
uns = lab == 0.0
R["P(unsafe | min_k sc<0)"] = float(uns[mn < 0].mean()); R["P(min_k sc<0 | unsafe)"] = float((mn[uns] < 0).mean())
# t=0 clearance: constant within set? within token?
r0 = sc[..., 0]
R["t0_within_set_range_max"] = float((r0.max(1) - r0.min(1)).max())
tok = D["token_idx"]
dif = []
for t in range(D["T"]):
    s = D["tok_sets"][t]
    if len(s) > 1:
        dif.append(float(r0[s, 0].max() - r0[s, 0].min()))
R["t0_across_sets_same_token_range"] = {"n_tokens_multi": len(dif), "max": max(dif) if dif else None, "median": float(np.median(dif)) if dif else None, "frac>0.05": float(np.mean(np.array(dif) > 0.05)) if dif else None}
ns = np.array([len(s) for s in D["tok_sets"]])
R["sets_per_token"] = {"min": int(ns.min()), "median": float(np.median(ns)), "max": int(ns.max()), "entries_per_token_max": int(ns.max() * 64 * 9), "entries_mean": float(ns.mean() * 576)}
# plant ambiguity inside a token: pairs of entries within 0.1 m with |dsc| > 0.5 m (sample tokens)
rng = np.random.default_rng(0)
amb = {"pairs_close": 0, "pairs_close_conflict": 0, "pairs_close_cross_set": 0, "pairs_close_cross_set_conflict": 0}
from scipy.spatial import cKDTree
for t in rng.choice(D["T"], 200, replace=False):
    s = D["tok_sets"][t]
    xy = D["pts"][s][..., :2].reshape(-1, 2).astype(np.float64); c = sc[s].reshape(-1)
    sid = np.repeat(np.arange(len(s)), 576)
    kt = np.tile(np.arange(9), len(s) * 64)
    m = kt > 0
    tr = cKDTree(xy[m]); pr = tr.query_pairs(0.1, output_type="ndarray")
    if len(pr) == 0: continue
    cc, ss = c[m], sid[m]
    d = np.abs(cc[pr[:, 0]] - cc[pr[:, 1]]); x = ss[pr[:, 0]] != ss[pr[:, 1]]
    amb["pairs_close"] += len(pr); amb["pairs_close_conflict"] += int((d > 0.5).sum())
    amb["pairs_close_cross_set"] += int(x.sum()); amb["pairs_close_cross_set_conflict"] += int((x & (d > 0.5)).sum())
R["plant_ambiguity_200tok_t>0_within0.1m"] = amb
# key-point spread across the 64 proposals of a set, per key time (how sharp must the lookup be?)
pts = D["pts"]
R["xy_spread_per_keytime_median_m"] = [round(float(np.median(np.linalg.norm(pts[:, :, k, :2] - pts[:, :, k, :2].mean(1, keepdims=True), axis=-1))), 3) for k in range(9)]
# nearest OTHER proposal distance per key time (median)
nn = []
for k in range(9):
    v = []
    for i in rng.choice(D["N"], 200, replace=False):
        p = pts[i, :, k, :2].astype(np.float64); dm = np.linalg.norm(p[:, None] - p[None], axis=-1); np.fill_diagonal(dm, np.inf); v.append(np.median(dm.min(1)))
    nn.append(round(float(np.median(v)), 3))
R["nearest_other_proposal_m_median_per_keytime"] = nn
R["tgt_stats"] = {"mean": float(D["tgt"].mean()), "frac_capped": float((D["tgt"] >= 10).mean())}
print(json.dumps(R, indent=1))
json.dump(R, open("C:/Users/Admin/par18/sh5diag/raw/probe1_oracle.json", "w"), indent=1)
