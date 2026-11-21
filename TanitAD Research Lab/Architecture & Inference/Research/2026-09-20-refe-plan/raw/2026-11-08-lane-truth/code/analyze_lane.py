#!/usr/bin/env python3
"""PREREG_LANE_SPEED analysis (CPU). Reads: C3 table, ego-probe results (via sel_common, unchanged), lane-truth side files from the relabel chain."""
import json, sys, os, lzma, hashlib
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107")
import sel_common as sc
sc.C3 = "F:/scratch/lane_truth_work/run/c3tab"
W = "F:/scratch/lane_truth_work/run"
OUT = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/lane_truth"
LEAF = "center_line.CenterLine.info"
D0, D1 = 0.5, 2.0
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res = {"pins": {"table.npz": sha(W + "/c3tab/table.npz"), "results.npz": sha(sc.EGO + "/full/results.npz"), "tokens_cruise.json": sha(sc.EGO + "/tokens_cruise.json"),
                "PREREG": sha(OUT + "/PREREG_LANE_SPEED.md")}}


def soft(d):
    return np.clip(1.0 - (np.asarray(d, np.float64) - D0) / (D1 - D0), 0.0, 1.0)


def dvec(d):
    return np.array([np.nan if p.get(LEAF) is None else float(p[LEAF]) for p in d["props"]])


def read_labels(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        out[(d["key"][0], d["key"][1])] = dvec(d)
    return out


# ---------------- gates G-L1, G-L2 (control) ----------------
banked = {}
BP = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
for ln in lzma.open(BP, "rt"):
    d = json.loads(ln)
    banked[(tuple(d["key"]), d["ckpt_step"])] = dvec(d)  # AMENDMENT_2: a banked token has rows for ranks 0 and 1; the full key is the row's identity
assert len(banked) == 3137, "banked rows collapsed by the join key"


def read_full(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        out[(tuple(d["key"]), d["ckpt_step"])] = dvec(d)
    return out


ctrl = read_full(W + "/out_ctrl2_ph/teacher_lane_r0.jsonl")
real = read_full(W + "/out_ctrl2_real/teacher_lane_r0.jsonl")
dd, dr, unequal = [], [], []
inter = [k for k in ctrl if k in real]  # 2026-11-08: intersection of both arms (6 of 40 sets failed scenario build in both)
ctrl = {k: ctrl[k] for k in inter}
for k, v in ctrl.items():
    b = banked[k]
    dd.append(np.abs(v - b))
    dr.append(np.abs(v - real[k]))
    for j in np.where(~(np.abs(v - b) <= 1e-3))[0]:
        unequal.append([k[0][0][:22], k[0][1], int(j), float(v[j]), float(b[j])])
dd, dr = np.concatenate(dd), np.concatenate(dr)
ds = np.abs(soft(np.concatenate(list(ctrl.values()))) - soft(np.concatenate([banked[k] for k in ctrl])))
res["G_L1"] = {"n_sets": len(ctrl), "n_props": int(dd.size), "frac_abs_d_le_1e-3": float((dd <= 1e-3).mean()), "frac_soft_le_0.01": float((ds <= 0.01).mean()),
               "max_abs_d": float(np.nanmax(dd)), "unequal_props": unequal[:20], "n_unequal": len(unequal),
               "PASS": bool((dd <= 1e-3).mean() >= 0.99 and (ds <= 0.01).mean() >= 0.99)}
res["G_L2"] = {"frac_abs_d_le_1e-6_vs_real_teacher": float((dr <= 1e-6).mean()), "max_abs_d": float(np.nanmax(dr)), "PASS": bool((dr <= 1e-6).all())}

# ---------------- main table ----------------
D = sc.load()
N = len(D["token"])
r = np.arange(N)
v0 = np.maximum(D["v0"], 1e-6)
ratio = D["v4_all"] / v0[:, None]
L = D["logits"].astype(np.float64)
P = 1.0 / (1.0 + np.exp(-L))
val, cru, pdms, pick = D["valid"], D["cruise"], D["pdms"], D["pick"]
lab = {}
for f in (W + "/out_cruise/teacher_lane_r0.jsonl", W + "/out_other/teacher_lane_r0.jsonl"):
    if os.path.exists(f):
        lab.update(read_labels(f))
dist = np.full((N, 64), np.nan)
has = np.zeros(N, bool)
for i in range(N):
    k = (str(D["log"][i]), str(D["token"][i]))
    if k in lab:
        dist[i] = lab[k]
        has[i] = True
T = soft(dist)
Tneg = -dist
res["G_L3"] = {"n_cruise": int(cru.sum()), "n_cruise_labelled_all64_finite": int((cru & has & np.isfinite(dist).all(1)).sum()), "n_labelled_total": int(has.sum())}
res["G_L3"]["PASS"] = res["G_L3"]["n_cruise_labelled_all64_finite"] >= 150


def spear(X, Y, M, method):
    out = np.full(N, np.nan)
    for i in np.where(M.any(1))[0]:
        m = M[i]
        if m.sum() < 5:
            continue
        if method == "legacy":
            rx = np.argsort(np.argsort(X[i, m]))
            ry = np.argsort(np.argsort(Y[i, m]))
        else:
            rx = rankdata(X[i, m])
            ry = rankdata(Y[i, m])
        if rx.std() == 0 or ry.std() == 0:
            continue
        out[i] = np.corrcoef(rx, ry)[0, 1]
    return out


ul, li, draws = sc.log_boot_idx(D["log"], 10000, 20260927)
nl = len(ul)


def B(v, m):
    return dict(zip(("point", "lo", "hi"), sc.boot_mean(v, m, li, nl, draws)))


cm = cru[:, None] & val & np.isfinite(dist)
cm_pred = cru[:, None] & val
blk = {}
for method in ("tie", "legacy"):
    rp = spear(L[..., 6], ratio, cm_pred, method)
    rpl = spear(L[..., 6], ratio, cm, method)
    rt = spear(T, ratio, cm, method)
    rn = spear(Tneg, ratio, cm, method)
    ok = np.isfinite(rpl) & np.isfinite(rt)
    gap = np.where(ok, rpl - rt, np.nan)
    ok2 = np.isfinite(rpl) & np.isfinite(rn)
    gap2 = np.where(ok2, rpl - rn, np.nan)
    blk[method] = {"n_cruise": int(cru.sum()), "n_pred_defined_all_valid": int(np.isfinite(rp[cru]).sum()), "pred_mean_all_defined": float(np.nanmean(rp[cru])),
                   "n_entered": int(ok.sum()), "n_true_varies": int(np.isfinite(rt[cru]).sum()),
                   "pred_entered": B(np.where(ok, rpl, np.nan), cru), "true_entered": B(np.where(ok, rt, np.nan), cru), "gap_pred_minus_true": B(gap, cru),
                   "sens_negdist": {"n_entered": int(ok2.sum()), "true": B(np.where(ok2, rn, np.nan), cru), "gap": B(gap2, cru)}}
    if method == "tie":
        np.save(OUT + "/per_token_gap.npy", np.stack([gap, rpl, rt]))
res["spearman"] = blk
b = json.load(open("E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107/p1_diag_all.json"))["Q4_within_token_spearman_vs_speed"]["lane"]
res["G_L4_pred_repro"] = {"banked_lane_legacy": b, "mine_legacy": blk["legacy"]["pred_mean_all_defined"], "n_mine": blk["legacy"]["n_pred_defined_all_valid"]}

# ---------------- read label per AMENDMENT_2 section 4 ----------------
gl1b = json.load(open(OUT + "/gl1b.json"))
res["G_L1b"] = {k: v for k, v in gl1b.items() if k != "per_set"}
if res["G_L1"]["PASS"]:
    LABEL = "AS WRITTEN"
elif gl1b["PASS"]:
    LABEL = "CONDITIONAL: G-L1 FAILED as written, G-L1b passed"
else:
    LABEL = "INVALID (G-L1 and G-L1b both failed): correlation is unlabelled diagnostic only"
# ---------------- verdict per PREREG section 6 ----------------
g = blk["tie"]["gap_pred_minus_true"]
ne = blk["tie"]["n_entered"]
if ne < 30:
    verdict = "O4"
elif g["point"] < -0.15 and g["hi"] < 0:
    verdict = "O1"
elif abs(g["point"]) <= 0.15:
    verdict = "O2"
elif g["point"] > 0.15 and g["lo"] > 0:
    verdict = "O3"
else:
    verdict = "O4"
near = min(abs(g["point"] - 0.15), abs(g["point"] + 0.15))
res["near_threshold_within_0.03"] = bool(near <= 0.03)
if LABEL.startswith("CONDITIONAL") and near <= 0.03 and verdict != "O4":
    verdict = "O4 (CONDITIONAL and G within 0.03 of a threshold)"
res["VERDICT"] = {"label": LABEL, "outcome": verdict, "gap": g, "n_entered": ne, "distance_to_nearest_threshold": near}

# ---------------- margin block (p1_diag.attrib, copied) ----------------
agg = sc.agg_np(L, np.float64)
rp_ = ratio[r, pick]
slow = rp_ < 0.85
pop = cru & slow
F = val & (ratio >= 0.9)
has_fast = F.any(1)
popf = pop & has_fast
NEG = -1e9
f_or = np.argmax(np.where(F, pdms + 1e-9 * agg, NEG), 1)
f_sc = np.argmax(np.where(F, agg, NEG), 1)
TERMS = ("NC", "DAC", "EP", "TTC", "C", "lane")


def attrib(f):
    lp, lf = np.log(np.maximum(P[r, pick], 1e-300)), np.log(np.maximum(P[r, f], 1e-300))
    w = np.array(sc.V1_W)

    def avg(k):
        return (P[r, k, 2] * w[0] + P[r, k, 3] * w[1] + P[r, k, 4] * w[2]) / w.sum()
    ap_, af = avg(pick), avg(f)
    d_avg_log = np.log(ap_) - np.log(af)
    lin = {"EP": w[0] * (P[r, pick, 2] - P[r, f, 2]) / w.sum(), "TTC": w[1] * (P[r, pick, 3] - P[r, f, 3]) / w.sum(), "C": w[2] * (P[r, pick, 4] - P[r, f, 4]) / w.sum()}
    dd_ = ap_ - af
    safe = np.where(np.abs(dd_) > 1e-12, dd_, 1.0)
    terms = {"NC": lp[:, 0] - lf[:, 0], "DAC": lp[:, 1] - lf[:, 1], "lane": lp[:, 6] - lf[:, 6]}
    for k in ("EP", "TTC", "C"):
        terms[k] = np.where(np.abs(dd_) > 1e-12, d_avg_log * lin[k] / safe, 0.0)
    tot = np.log(agg[r, pick]) - np.log(agg[r, f])
    return terms, tot


mb = {"n_pop_slow_cruise": int(pop.sum()), "n_popf": int(popf.sum())}
for name, f in (("f_sc", f_sc), ("f_or", f_or)):
    terms, tot = attrib(f)
    Tm = np.stack([terms[k] for k in TERMS], 1)
    dom = np.argmax(Tm, 1)
    idx = np.where(popf)[0]
    o = {"margin_dlog_agg": B(tot, popf), "terms": {k: B(terms[k], popf) for k in TERMS},
         "lane_share_of_margin_sum_ratio": float(terms["lane"][idx].sum() / tot[idx].sum()),
         "identity_max_abs_sum_minus_margin": float(np.abs(Tm.sum(1)[idx] - tot[idx]).max()),
         "dominant_counts": {k: int(((dom == i) & popf).sum()) for i, k in enumerate(TERMS)},
         "lane_dominant_of_popf": int(((dom == 5) & popf).sum())}
    lt = has & popf & np.isfinite(T).all(1)
    if lt.any():
        tp, tf = T[r, pick], T[r, f]
        eq = lt & (np.abs(tf - tp) <= 1e-12)
        o["lane_truth"] = {"n": int(lt.sum()), "true_target_pick": float(tp[lt].mean()), "true_target_f": float(tf[lt].mean()),
                           "pred_lane_pick": float(P[r, pick, 6][lt].mean()), "pred_lane_f": float(P[r, f, 6][lt].mean()),
                           "n_pred_prefers_pick_but_truth_f_ge_pick": int(((L[r, pick, 6] > L[r, f, 6]) & (tf >= tp - 1e-12))[lt].sum()),
                           "n_pred_prefers_pick_and_truth_f_gt_pick": int(((L[r, pick, 6] > L[r, f, 6]) & (tf > tp + 1e-12))[lt].sum()),
                           "n_truth_f_gt_pick": int((tf > tp + 1e-12)[lt].sum()),
                           "n_truth_equal": int(eq.sum()),
                           "mean_lane_term_where_truth_equal": float(terms["lane"][eq].mean()) if eq.any() else None}
    mb[name] = o
res["margin"] = mb
res["G_L4_margin"] = {"lane_dominant_f_sc": mb["f_sc"]["lane_dominant_of_popf"], "n_popf": mb["n_popf"], "n_pop": mb["n_pop_slow_cruise"], "expected": "31 of 55"}
json.dump(res, open(OUT + "/lane_speed_result.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("G_L1", "G_L2", "G_L3", "G_L4_pred_repro", "G_L4_margin", "VERDICT")}, indent=1, default=float))
