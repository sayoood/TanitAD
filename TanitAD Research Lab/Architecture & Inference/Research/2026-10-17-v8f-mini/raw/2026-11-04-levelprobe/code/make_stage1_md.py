#!/usr/bin/env python3
"""make_stage1_md.py -- STAGE1.md from raw/STAGE1.json (every number MEASURED, with its artifact path).  python -I code/make_stage1_md.py"""
import json
import sys

PK = "E:/Projects/TanitAD-artifacts/dagland/levelprobe"
r = json.load(open(f"{PK}/raw/STAGE1.json", encoding="utf-8"))
m, c = r["main"], r["controls"]


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def row(name, e):
    return f"| {name} | {e['n']} | {f(e.get('rmse'))} | {f(e.get('mae'))} | {f(e.get('r2'))} | {f(e.get('y_mean'), 2)} / {f(e.get('y_std'), 2)} |" if e.get("n") else f"| {name} | 0 | | | | |"


L = []
L.append("# STAGE1: how well does the frozen image grid of the t0 frame predict the logged speed at t0? (H-LEVELPROBE-1 Stage 1, AS WRITTEN)\n")
L.append(f"Card `PREREG_LEVELPROBE.md` sha256 `{r['prereg_sha256']}` (frozen before this fit; `FROZEN_SHA.txt`, `ERRATA_1.md`). Tier **T0-class probe** (frozen features of logged frames; NOT driving performance). Evidence class **MEASURED (ours)**; "
         f"artifact `raw/STAGE1.json` (+ `raw/stage1_pred_eval608.npz`); code `code/stage1.py`; run clock {r['run_clock']}.\n")
L.append("**Coverage (F1, `raw/F1_coverage.json`): every number below is on the 608 eval windows with v0 < 6 (of the 2,399 the T1 rig rolls out) and was fitted on the 24,000 TRAIN windows of P-STOP + P-MOVE (labels in [0, 6)). It says nothing about the 1,791 windows with v0 >= 6, which hold 85.1 % of the M1 windows.**\n")
L.append(f"Features: probe 3's G8 grid (DINOv3-L/16 pool8x20, current frame) mean-pooled to G2x4 = {8192} features; vision only (`assertions.ego_nav_columns_loaded_into_fit` = {r['assertions']['ego_nav_columns_loaded_into_fit']}). "
         f"Ridge, alpha = c x trace/D, c chosen by episode-grouped 5-fold CV on the TRAIN windows (folds = `int(sha12,16) % 5`): chosen **c = {m['chosen_c']}** (grid edge: {m['chosen_c_at_grid_edge']}); CV RMSE by c: "
         + ", ".join(f"{k}: {v:.3f}" for k, v in m["cv_rmse_by_c"].items()) + ". Fit and score clip sets disjoint (asserted): "
         f"{r['assertions']['n_train_clips']} train clips, {r['assertions']['n_eval_clips']} eval clips.\n")
L.append("## 1. The scores on the eval windows (m/s)\n")
L.append("| subset | n | RMSE | MAE | R^2 (vs eval mean) | eval v0 mean / std |\n|---|---|---|---|---|---|")
for k, e in m["eval"].items():
    L.append(row(k, e))
b = c["K3_train_mean_baseline"]["eval"]
L.append("\nK3, the train-mean predictor (label mean %s m/s) on the same subsets:\n" % f(m["train_label_mean"], 3))
L.append("| subset | n | RMSE | MAE | R^2 (vs eval mean) |\n|---|---|---|---|---|")
for k, e in b.items():
    L.append(f"| {k} | {e['n']} | {f(e.get('rmse'))} | {f(e.get('mae'))} | {f(e.get('r2'))} |" if e.get("n") else f"| {k} | 0 | | | |")
L.append("\nRMSE skill versus the train-mean predictor (1 - RMSE_ridge / RMSE_mean): " + ", ".join(f"{k} {f(v)}" for k, v in m["skill_vs_train_mean_rmse"].items() if v is not None) + ".\n")
L.append("## 2. Controls\n")
L.append("| control | threshold | reading | PASS |\n|---|---|---|---|")
L.append(f"| POOL equals an independent loop | max abs dev <= 1e-4 | {c['POOL_equals_independent_loop']['max_abs_dev']:.2e} | {c['POOL_equals_independent_loop']['PASS']} |")
k1, k2 = c["K1_shuffled_train_labels"], c["K2_v_logged_appended"]
L.append(f"| K1 labels shuffled within TRAIN (seed {k1['seed']}, same pipeline) | eval R^2 <= 0.02 | {f(k1['eval_all']['r2'], 4)} (RMSE {f(k1['eval_all']['rmse'])}, chosen c {k1['chosen_c']}) | {k1['PASS']} |")
L.append(f"| K2 v_logged appended (column scale {k2['column_scale']:.3g}) | eval R^2 >= 0.99 | {f(k2['eval_all']['r2'], 4)} (RMSE {f(k2['eval_all']['rmse'])}, chosen c {k2['chosen_c']}) | {k2['PASS']} |")
L.append(f"| K3 train-mean RMSE (reported) | baseline | {f(b['all']['rmse'])} m/s on all 608 | reported |")
L.append(f"\nAll controls pass: **{r['controls_all_pass']}**.\n")
L.append("## 3. What this can and cannot say\n"
         "* It is a ridge on a pooled (G2x4) readout of ONE frozen encoder at ONE frame; a low R^2 is not a statement about learnability (a trained in-model head could do better), and a high one is not driving performance.\n"
         "* The train mixture is 50 % stopped windows (P-STOP, v0 < 0.5) by construction; the eval windows are 101 stopped and 507 in [0.5, 6): the mixtures differ, so R^2 and RMSE are not comparable to a probe fitted on the natural mix.\n"
         "* The estimator is one fit on one train set, scored on 141 eval episodes (608 windows with v0 < 6): there is no interval here; the eval windows are correlated within episodes and per-class cells are small.\n"
         "* K3's RMSE skill is inflated by the mixture shift: the train-mean predictor sits at 1.87 m/s while the eval mean is 3.07 m/s. The mixture-free reference is the eval-mean predictor, whose RMSE is the eval std (1.98 m/s overall): the ridge's 1.42 m/s is 28 % below that (R^2 0.489); on the M1 windows 1.59 m/s versus an std of 1.69 m/s (R^2 0.117).\n"
         "* R^2 inside the S and O classes (n = 32 and 8, eval std 0.17 and 0.12 m/s) is not interpretable (the denominator is near zero); read RMSE / MAE there.\n"
         "* Stage 2's HLVL-PRED cannot use this fit (F1): it needs levels for all 2,399 windows. See `AMEND_A1_PROPOSED.md`.\n")
open(f"{PK}/STAGE1.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote STAGE1.md")
