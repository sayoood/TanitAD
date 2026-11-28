#!/usr/bin/env python3
"""p1p_probe_refit.py -- SPEC_P1P.md section 1 + control R1.  TRAIN ONLY: no eval-141 window is read here.
Refits the trigger probe's P-MOVE V+E linear probe (label GO, and the secondary GO24) with the probe's own committed code (p1p_tp_lib.py == trigger_probe/tp_lib.py, byte for byte):
five fold models (PCA-64 + fit_probe on outer != f) and one all-train model, saves their parameters, and checks the reproduction of the banked artifacts (R1).
Writes P1P_PROBE_MODELS.npz and P1P_REFIT_R1.json.   CUDA_VISIBLE_DEVICES=-1  python -I p1p_probe_refit.py"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TP = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe"
RAW = f"{TP}/raw"
SCR = "F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch"
sys.path.insert(0, HERE)
import p1p_tp_lib as L  # noqa: E402

K_PCA = 64


def closure_parts(predict):
    """The scaler and the fitted LogisticRegression captured by tp_lib.fit_probe's predict closure."""
    fv = dict(zip(predict.__code__.co_freevars, (c.cell_contents for c in predict.__closure__)))
    return fv["sc"], fv["m"]


def main():
    t0 = time.time()
    A = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    idx = np.load(f"{RAW}/sample_pmove.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    outer, inner = L.outer_fold(sha), L.inner_fold(sha)
    y, y24 = A["go"][idx].astype(int), A["go24"][idx].astype(int)
    E = A["E"][idx].astype(np.float64)
    n = len(idx)
    X = np.load(f"{SCR}/V_pmove.npy", mmap_mode="r")
    assert X.shape == (n, 40 * 1024), X.shape
    read = lambda r: np.asarray(X[r], np.float32)
    bank_S = np.load(f"{RAW}/pca_V_pmove.npz")["S"]
    bank_oof = np.load(f"{RAW}/oof_pmove_V_linear.npz")
    out = {"n": int(n), "n_go": int(y.sum()), "n_clips": int(len(np.unique(clip))), "folds": {}}
    mu_all, comp_all = [], []
    M = {k: [] for k in ("go_mean", "go_scale", "go_coef", "go_icpt", "go_C", "go24_mean", "go24_scale", "go24_coef", "go24_icpt", "go24_C")}
    for f in range(6):
        rows = np.flatnonzero(outer != f) if f < 5 else np.arange(n)
        p = L.StreamPCA(k=K_PCA, seed=0).fit(read, rows, X.shape[1])
        S = p.transform(read, np.arange(n))
        d_S = float(np.abs(S - bank_S[f]).max())
        scale_S = float(np.abs(bank_S[f]).max())
        mu_all.append(p.mu)
        comp_all.append(p.comp)
        rec = {"pca_rows": int(len(rows)), "pca_max_abs_diff_vs_banked_S": d_S, "banked_S_max_abs": scale_S}
        Xf = np.concatenate([S.astype(np.float64), E], 1)
        Xb = np.concatenate([bank_S[f].astype(np.float64), E], 1)               # exactly what run_probe.arm_feats fed the banked fit
        for tag, lab, bkey in (("go", y, "V+E"), ("go24", y24, "GO24__V+E")):
            if f == 5 and tag == "go24":
                continue
            tr = (outer != f) if f < 5 else np.ones(n, bool)
            pred, hp, tab = L.fit_probe(Xf[tr], lab[tr], inner[tr], "linear")
            sc, m = closure_parts(pred)
            for k_, v_ in (("mean", sc.mean_), ("scale", sc.scale_), ("coef", m.coef_[0]), ("icpt", np.array(m.intercept_[0])), ("C", np.array(hp))):
                M[f"{tag}_{k_}"].append(np.asarray(v_, np.float64))
            chk = ((Xf[:50] - sc.mean_) / sc.scale_) @ m.coef_[0] + m.intercept_[0]
            rec[f"{tag}_closure_vs_predict_max_abs"] = float(np.abs(chk - pred(Xf[:50])).max())
            rec[f"{tag}_C"] = float(hp)
            rec[f"{tag}_inner_cv_auc"] = {str(k): v for k, v in tab.items()}
            if f < 5:
                te = outer == f
                held = pred(Xf[te])
                bk = bank_oof[bkey][te]
                rec[f"{tag}_heldout_vs_banked_oof_max_abs"] = float(np.abs(held - bk).max())
                # the same fit applied to the banked PCA scores: separates "PCA not bit-reproducible" from "logistic fit not reproducible"
                pred_b, hp_b, _ = L.fit_probe(Xb[tr], lab[tr], inner[tr], "linear")
                rec[f"{tag}_banked_S_refit_heldout_vs_banked_oof_max_abs"] = float(np.abs(pred_b(Xb[te]) - bk).max())
                rec[f"{tag}_banked_S_refit_C"] = float(hp_b)
                from sklearn.metrics import roc_auc_score
                rec[f"{tag}_heldout_auc"] = float(roc_auc_score(lab[te], held))
                rec[f"{tag}_banked_heldout_auc"] = float(roc_auc_score(lab[te], bk))
        out["folds"][str(f)] = rec
        print(f"fold {f} {time.time() - t0:.0f}s", json.dumps({k: v for k, v in rec.items() if not k.endswith("inner_cv_auc")}), flush=True)
    np.savez(f"{HERE}/P1P_PROBE_MODELS.npz", pca_mu=np.stack(mu_all), pca_comp=np.stack(comp_all), **{k: np.stack(v) for k, v in M.items()})
    # pooled OOF AUC of the refit vs the banked value (summary, same pooled scores the banked table used)
    out["seconds"] = time.time() - t0
    json.dump(out, open(f"{HERE}/P1P_REFIT_R1.json", "w"), indent=1)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
