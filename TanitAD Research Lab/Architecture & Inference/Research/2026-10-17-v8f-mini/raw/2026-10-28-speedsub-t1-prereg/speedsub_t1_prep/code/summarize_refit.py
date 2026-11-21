#!/usr/bin/env python3
"""Tabulates refit_all.json + refit_seeds.json beside fit.json and READ.json (numbers only; reads no eval window). Writes REFIT_TABLE.md and refit_summary.json."""
import json
import sys

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/speedsub_t1_prep")
OUT = "F:/TanitAD-scratch/speedsub_t1_prep"
CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
R = json.load(open(f"{OUT}/refit_all.json"))
S = json.load(open(f"{OUT}/refit_seeds.json"))
fit = json.load(open("F:/TanitAD-scratch/speedread/fit/fit.json"))
rd = json.load(open("F:/TanitAD-scratch/speedread/read/READ.json"))
sd = np.array(R["train_sd"])
n_ho = R["holdout_n_per_cell"]
const_ho = np.array(R["const_holdout_rmse_ms"])

# persistence baseline: the measured speed v (ego col 0, m/s) as the prediction of every cell, scored on the same fold-0 holdout rows
z = np.load("F:/TanitAD-scratch/speedread/train/train_arrays.npz")
import hashlib  # noqa: E402
folds = np.array([int(hashlib.sha256(str(c).encode()).hexdigest()[:8], 16) % 5 for c in z["clip"]])
ho = np.flatnonzero(folds == 0)
persist = []
for c in range(5):
    m = z["m"][ho, c]
    persist.append(float(np.sqrt(np.mean((z["ego"][ho, 0][m] - z["y"][ho, c][m]) ** 2))))
persist = np.array(persist)

rows = {}
fj = fit["mlp_ego"]["holdout_rmse_sd"]
rows["mlp_ego (fit.json, banked)"] = {"sd": fj}
rows["lin_ego (fit.json, banked)"] = {"sd": fit["lin_ego"]["holdout_rmse_sd"]}
rows["lin (scene+ego ridge, fit.json, SPEEDREAD primary)"] = {"sd": fit["lin"]["holdout_rmse_sd"]}
rows["mlp (scene+ego MLP, fit.json)"] = {"sd": fit["mlp"]["holdout_rmse_sd"]}
names = {"full": "REFIT full = mlp_ego (control)", "a_p10f_fok": "(a) p10f + frame_ok", "b_nonav": "(b) p10f + limit + limit_ok + frame_ok (no nav)", "c_nolimit": "(c, extra) p10f + nav + frame_ok (no limit)"}
for k, nm in names.items():
    rows[f"MLP {nm}"] = {"sd": R["designs"][k]["mlp"]["holdout_rmse_sd"], "ep": R["designs"][k]["mlp"]["best_epoch"], "dim": R["designs"][k]["d"]}
for k, nm in names.items():
    rows[f"RIDGE {nm}"] = {"sd": R["designs"][k]["ridge"]["holdout_rmse_sd"], "dim": R["designs"][k]["d"], "lam": R["designs"][k]["ridge"]["lam"]}
rows["persistence: cell := measured speed v (no fit)"] = {"sd": list(persist / sd)}
rows["CONST (train mean)"] = {"sd": list(const_ho / sd)}
out = {"cells": CELLS, "holdout_n_per_cell": n_ho, "rows": {}}
for nm, r in rows.items():
    s_ = np.array(r["sd"])
    ms = s_ * sd
    sk = 1.0 - ms ** 2 / const_ho ** 2
    out["rows"][nm] = {"holdout_rmse_sd": [float(v) for v in s_], "mean_sd": float(s_.mean()), "holdout_rmse_ms": [float(v) for v in ms], "holdout_skill": [float(v) for v in sk], "mean_skill": float(sk.mean()),
                       **{k: v for k, v in r.items() if k != "sd"}}
# seed noise floor (MLP)
sn = {}
for k in names:
    vals = [R["designs"][k]["mlp"]["holdout_rmse_sd"]] + [S["seeds"][s][k]["holdout_rmse_sd"] for s in ("1", "2")]
    V = np.array(vals)
    sn[k] = {"mean_sd_per_seed": [float(v.mean()) for v in V], "mean_of_means": float(V.mean()), "range_of_means": float(V.mean(1).max() - V.mean(1).min()), "per_cell_range": [float(v) for v in (V.max(0) - V.min(0))],
             "best_epochs": [R["designs"][k]["mlp"]["best_epoch"]] + [S["seeds"][s][k]["best_epoch"] for s in ("1", "2")]}
out["mlp_seed_floor"] = sn
# deltas (seed-0 numbers, and 3-seed means)
d = {}
m3 = {k: np.array([R["designs"][k]["mlp"]["holdout_rmse_sd"]] + [S["seeds"][s][k]["holdout_rmse_sd"] for s in ("1", "2")]).mean(0) for k in names}
d["nav_effect_b_to_full_3seed_mean_sd"] = float(m3["full"].mean() - m3["b_nonav"].mean())
d["nav_effect_a_to_c_3seed_mean_sd"] = float(m3["c_nolimit"].mean() - m3["a_p10f_fok"].mean())
d["limit_effect_a_to_b_3seed_mean_sd"] = float(m3["b_nonav"].mean() - m3["a_p10f_fok"].mean())
d["limit_effect_c_to_full_3seed_mean_sd"] = float(m3["full"].mean() - m3["c_nolimit"].mean())
d["per_cell_3seed_mean_sd"] = {k: [float(v) for v in m3[k]] for k in names}
d["share_of_a_to_full_gap_removed_by_adding_limit_first(b)"] = float((m3["a_p10f_fok"].mean() - m3["b_nonav"].mean()) / (m3["a_p10f_fok"].mean() - m3["full"].mean()))
out["deltas"] = d
# ESTIMATED eval-141 RMSE of each MLP refit: banked mlp_ego eval RMSE x (refit fold-0 RMSE / mlp_ego fold-0 RMSE) per cell. Assumes the fold-0 -> eval transfer ratio is design-independent. NOT an eval.
est = {}
for s in ("0", "1"):
    me = np.array(rd["cells"][s]["mlp_ego"]["rmse"])
    town = np.array(rd["cells"][s]["T_own"]["rmse"])
    est[s] = {"T_own_eval141_rmse_ms": [float(v) for v in town], "mlp_ego_eval141_rmse_ms": [float(v) for v in me]}
    for k in ("b_nonav", "a_p10f_fok", "c_nolimit"):
        ratio = np.array(R["designs"][k]["mlp"]["holdout_rmse_sd"]) / np.array(fj)
        e = me * ratio
        est[s][k] = {"ESTIMATED_eval141_rmse_ms": [float(v) for v in e], "ratio_to_mlp_ego_fold0": [float(v) for v in ratio], "below_T_own_cells": [bool(v) for v in e < town], "n_cells_below_T_own": int((e < town).sum())}
    est[s]["mlp_ego"] = {"n_cells_below_T_own": int((me < town).sum())}
out["ESTIMATED_eval141_vs_T_own"] = est
json.dump(out, open(f"{OUT}/refit_summary.json", "w", encoding="utf-8", newline="\n"), indent=1)

# ---- markdown
L = []
L.append("| read-out (TRAIN fold-0 holdout, " + ", ".join(f"n={v}" for v in n_ho) + ") | dim | " + " | ".join(CELLS) + " | mean (SD units) | mean skill |")
L.append("|---|---|" + "---|" * (len(CELLS) + 2))
for nm, r in out["rows"].items():
    L.append(f"| {nm} | {r.get('dim', '')} | " + " | ".join(f"{v:.4f}" for v in r["holdout_rmse_sd"]) + f" | **{r['mean_sd']:.5f}** | {r['mean_skill']:.4f} |")
L.append("")
L.append("RMSE in m/s on the same rows, and skill = 1 - MSE / MSE(train-mean CONST) per cell:")
L.append("")
L.append("| read-out | " + " | ".join(f"{c} RMSE m/s" for c in CELLS) + " | " + " | ".join(f"{c} skill" for c in CELLS) + " |")
L.append("|---|" + "---|" * (2 * len(CELLS)))
for nm, r in out["rows"].items():
    L.append(f"| {nm} | " + " | ".join(f"{v:.3f}" for v in r["holdout_rmse_ms"]) + " | " + " | ".join(f"{v:.4f}" for v in r["holdout_skill"]) + " |")
L.append("")
L.append("MLP seed floor (same design, MLP init/order seeds 0, 1, 2; mean over cells, SD units):")
L.append("")
L.append("| design | seed 0 | seed 1 | seed 2 | range of the three means | best epochs |")
L.append("|---|---|---|---|---|---|")
for k, v in sn.items():
    L.append(f"| {names[k]} | " + " | ".join(f"{x:.5f}" for x in v["mean_sd_per_seed"]) + f" | {v['range_of_means']:.5f} | {v['best_epochs']} |")
L.append("")
L.append("T_own (eval-141, m/s, from READ.json; a different population) beside mlp_ego (eval-141) and the ESTIMATED eval-141 RMSE of the refits (mlp_ego eval RMSE x fold-0 ratio):")
L.append("")
for s in ("0", "1"):
    e = est[s]
    L.append(f"seed {s}: T_own {np.round(e['T_own_eval141_rmse_ms'], 3).tolist()} | mlp_ego {np.round(e['mlp_ego_eval141_rmse_ms'], 3).tolist()} (below T_own in {e['mlp_ego']['n_cells_below_T_own']}/5)")
    for k in ("b_nonav", "c_nolimit", "a_p10f_fok"):
        L.append(f"  - {names[k]}: ESTIMATED {np.round(e[k]['ESTIMATED_eval141_rmse_ms'], 3).tolist()} (below T_own in {e[k]['n_cells_below_T_own']}/5 cells: {e[k]['below_T_own_cells']})")
open(f"{OUT}/REFIT_TABLE.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
print(json.dumps(out["deltas"], indent=1))

# ---- direct (population-mismatched) read: fold-0 RMSE (m/s) of each refit against T_own's eval-141 RMSE, and mlp_ego's fold-0 vs eval-141 agreement
dd = {"mlp_ego_fold0_vs_eval141_relative_diff_per_cell": {}, "direct_fold0_vs_T_own": {}}
f0 = {k: np.array(R["designs"][k]["mlp"]["holdout_rmse_sd"]) * sd for k in names}
for s in ("0", "1"):
    me = np.array(rd["cells"][s]["mlp_ego"]["rmse"]); town = np.array(rd["cells"][s]["T_own"]["rmse"])
    dd["mlp_ego_fold0_vs_eval141_relative_diff_per_cell"][s] = [float(v) for v in (f0["full"] / me - 1.0)]
    dd["direct_fold0_vs_T_own"][s] = {k: {"fold0_rmse_ms": [float(v) for v in f0[k]], "below_T_own": [bool(v) for v in f0[k] < town], "n_below": int((f0[k] < town).sum())} for k in names}
    dd["T_own_eval141_skill_mean"] = dd.get("T_own_eval141_skill_mean", {}); dd["T_own_eval141_skill_mean"][s] = float(np.mean(rd["cells"][s]["T_own"]["skill"]))
    dd.setdefault("mlp_ego_eval141_skill_mean", {})[s] = float(np.mean(rd["cells"][s]["mlp_ego"]["skill"]))
out["direct"] = dd
json.dump(out, open(f"{OUT}/refit_summary.json", "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(dd, indent=1)[:3500])
