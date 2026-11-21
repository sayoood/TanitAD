"""d2_fit.py -- PREREG_LHA0_D2 stage 1, FIT SIDE ONLY: fit the D2 head on the FRESH split (LH's 292-clip FIT rows, LH's 5 folds), choose lambda by out-of-fold CE on the
fresh folds, refit on all FIT rows, freeze.  Reads NO eval139 file (the eval paths are not even constructed here).

inputs   F:/c_overflow_2026-10-26/tmp/lha0/universe.npz   (the LHA0 diagnostic's own table: z_tac / v0 / a0 / psi_dot0 for the fresh U rows, the fit-row index d_u / d_y / d_eid,
                                                           the score masks; a0 / psi_dot0 come from refcv9_lha0/result/lha0_features.npz via that package's prep -- not re-derived)
         refcv9_lha0/result/oof_logits/A_s0_l100.npz       (LH's own out-of-fold 6-logits on the same fresh rows: A == LH's feature set; K_REPRO_A reproduced its CE to 0.0)
heads    D2        PRIMARY. binary logistic on ALL 8,178 FIT rows (the literal brief: "the same 292-clip FIT rows"), y = 1[class in decel set], every other class incl. ACCELERATE = 0
         D2NOACC   beside: the K-HEAD-renormalisation reading -- ACCELERATE rows EXCLUDED from the fit (5-class row set), complement = {FOLLOW, CRUISE}
         y = 1 iff class in G_DECEL = (2, 3, 4) = (BRAKE_TO, CREEP, HOLD)
CV       5 folds over fresh EPISODES = lh_fit.folds_of(d_eid, 5, 0) (numpy default_rng(0)); standardisation stats fitted on the training rows of each fold only;
         lambda in {1e-3, 1e-2, 1e-1, 1, 10, 100}; criterion = mean OOF binary CE over the FIT rows of the head's row set; ties -> smaller lambda.
descriptive (never gating): fresh OOF decel AUC on the diag's population P for D2 / D2NOACC / LH-OOF / raw -a0, with the paired episode-cluster bootstrap (B 2000, seed 0).
Writes result/D2_FIT_INFO.json, D2_FROZEN.json, D2NOACC_FROZEN.json, result/d2_fresh_oof.npz (arrays only, no ids).  CPU only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_common as D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = D.ROOT


def main() -> int:
    t0 = time.time()
    import l2_common as C
    import lh_fit as LF
    import k_head as KH
    assert tuple(C.G_DECEL) == D.G_DECEL and tuple(C.G_CRUISE) == D.G_CRUISE and tuple(C.G_ACCEL) == D.G_ACCEL, (C.G_DECEL, C.G_CRUISE, C.G_ACCEL)
    assert tuple(LF.LON6_V7) == (0, 1, 3, 4, 5, 7) and tuple(LF.LAMBDAS) == D.LAMBDAS, (LF.LON6_V7, LF.LAMBDAS)
    z = np.load(D.UNIVERSE)
    U = {k: z[k] for k in z.files}
    d_u, d_y, d_eid = U["d_u"], U["d_y"], U["d_eid"]
    assert len(d_u) == 8178 and len(np.unique(d_eid)) == 290 and np.array_equal(np.bincount(d_y), [1796, 2395, 1876, 103, 130, 1878]), np.bincount(d_y)
    assert np.array_equal(U["eid_u"][d_u], d_eid) and U["is_fit"][d_u].all()
    fin = np.isfinite(U["f_a0"]) & np.isfinite(U["f_psi"]) & np.isfinite(U["v0u"])
    assert fin.all(), int((~fin).sum())
    Xraw_u = D.design(U["f_a0"], U["v0u"], U["f_psi"])                       # [10107, 3] raw features for every universe row
    # ---- folds: LH's, via the same function call the diagnostic used ----
    fold_d = LF.folds_of(d_eid, 5, 0)
    e2f = {int(e): int(f) for e, f in zip(d_eid, fold_d)}
    assert all(len({int(f) for f in fold_d[d_eid == e]}) == 1 for e in np.unique(d_eid)), "an episode sits in two folds"
    fold_u = np.array([e2f[int(e)] for e in U["eid_u"]])
    fold_sizes = {"rows": np.bincount(fold_d, minlength=5).tolist(), "episodes": [int(len(np.unique(d_eid[fold_d == k]))) for k in range(5)]}
    # ---- LH's own OOF logits (K-REPRO of the diagnostic's own gate, re-read here) ----
    lg_lh = np.load(D.LHA0 / "result" / "oof_logits" / "A_s0_l100.npz")["logits"]
    ce_lh = float(LF.ce_rows(LF.log_softmax(lg_lh[d_u]), d_y).mean())
    assert abs(ce_lh - 1.0917272119292791) < 1e-9, ce_lh

    heads = {}
    for name, neg in (("D2", D.G_CRUISE + D.G_ACCEL), ("D2NOACC", D.G_CRUISE)):
        in_set = np.isin(d_y, D.G_DECEL + neg)                               # fit rows that carry this head's binary target
        yb = np.isin(d_y, D.G_DECEL).astype(np.int64)
        n_set, n_pos = int(in_set.sum()), int(yb[in_set].sum())
        oof_u = {lam: np.full(len(fold_u), np.nan) for lam in D.LAMBDAS}
        fold_ce = {lam: [] for lam in D.LAMBDAS}
        n_iter = {lam: [] for lam in D.LAMBDAS}
        for k in range(5):
            tr = in_set & (fold_d != k)                                      # training rows: binary set, outside fold k
            te_fit = in_set & (fold_d == k)
            mu, sd = D.std_stats(Xraw_u[d_u[tr]])
            Xtr = (Xraw_u[d_u[tr]] - mu) / sd
            te_u = np.flatnonzero(fold_u == k)                               # every universe row of fold-k episodes is scored by the fold-k model
            Xte_u = (Xraw_u[te_u] - mu) / sd
            for lam in D.LAMBDAS:
                m = D.fit_binary(Xtr, yb[tr], lam)
                lo = D.logit_of(m, Xte_u)
                oof_u[lam][te_u] = lo
                # CE on the fold's FIT rows of the head's row set
                lo_fit = D.logit_of(m, (Xraw_u[d_u[te_fit]] - mu) / sd)
                fold_ce[lam].append(float(D.bce(lo_fit, yb[te_fit]).mean()))
                n_iter[lam].append(m["n_iter"])
        table = {}
        for lam in D.LAMBDAS:
            lo_all = oof_u[lam][d_u]
            assert np.isfinite(lo_all).all()
            table[f"{lam:g}"] = {"lambda": lam, "mean_ce_oof": float(D.bce(lo_all[in_set], yb[in_set]).mean()), "fold_ce": fold_ce[lam], "n_iter": n_iter[lam]}
        chosen = min(D.LAMBDAS, key=lambda l: (table[f"{l:g}"]["mean_ce_oof"], l))
        p_prior = n_pos / n_set
        const_ce = float(-(p_prior * np.log(p_prior) + (1 - p_prior) * np.log(1 - p_prior)))   # in-sample prior entropy (descriptive floor)
        # ---- refit on ALL fit rows of the set, freeze ----
        mu, sd = D.std_stats(Xraw_u[d_u[in_set]])
        fit = D.fit_binary((Xraw_u[d_u[in_set]] - mu) / sd, yb[in_set], chosen)
        path = ROOT / f"{name}_FROZEN.json"
        meta = {"head": name, "n_fit_rows": n_set, "n_fit_pos": n_pos, "n_fit_episodes": int(len(np.unique(d_eid[in_set]))), "d": 3,
                "accel_is_negative": name == "D2", "lambda_selection": "min mean OOF binary CE over the fit rows of the row set, LH's 5 folds, grid " + str(list(D.LAMBDAS)),
                "fit_source": "fresh split only, 292 lead clips (lh_fit --lead-clips-only rows), no eval139 file read"}
        sha = D.freeze(path, fit, mu, sd, chosen, meta)
        blob = D.load_frozen(path, sha)
        reload_ok = bool(np.array_equal(D.head_logit(blob, U["f_a0"][d_u], U["v0u"][d_u], U["f_psi"][d_u]), D.logit_of(fit, (Xraw_u[d_u] - mu) / sd)))
        heads[name] = {"head": name, "row_set": "classes " + str([0, 1, 2, 3, 4, 5] if name == "D2" else [0, 1, 2, 3, 4]), "n_fit_rows": n_set, "n_fit_pos": n_pos,
                       "pos_rate": round(p_prior, 4), "constant_prior_ce_in_sample": const_ce, "table": table, "chosen_lambda": chosen, "chosen_oof_ce": table[f"{chosen:g}"]["mean_ce_oof"],
                       "frozen_file": str(path), "frozen_sha256": sha, "reload_logits_bit_equal": reload_ok, "w": [float(x) for x in fit["w"]], "b": float(fit["b"]),
                       "mu": [float(x) for x in mu], "sd": [float(x) for x in sd], "_oof_u": oof_u[chosen]}
        assert reload_ok
    # ---- descriptive fresh-OOF AUCs on the diagnostic's population P (never gating) ----
    P = np.flatnonzero(U["Pmask"])
    yP, eP = U["is_dec"][P], U["eid_u"][P]
    sc = {"D2": heads["D2"]["_oof_u"][P], "D2NOACC": heads["D2NOACC"]["_oof_u"][P], "LH_OOF": D.lh_logit(lg_lh)[P], "neg_a0": -U["f_a0"][P]}
    R = D.paired_boot(yP, eP, sc)
    desc = {"population": {"n_rows": int(len(P)), "n_decel": int(yP.sum()), "n_episodes": R["n_episodes"]}, "auc": {k: D.pr(v) for k, v in R["pt"].items()},
            "D2_minus_LH_OOF": D.diff(R, "D2", "LH_OOF"), "D2_minus_neg_a0": D.diff(R, "D2", "neg_a0"), "D2NOACC_minus_D2": D.diff(R, "D2NOACC", "D2"),
            "unit_weight_wauc_ok": R["unit_ok"], "note": "fresh OOF; lambda was selected on the same folds' CE (3 features, 6 lambdas: optimism immaterial but disclosed); DESCRIPTIVE"}
    np.savez(ROOT / "result" / "d2_fresh_oof.npz", P=P, y=yP, eid_u=U["eid_u"], fold_u=fold_u, d2_logit_oof=heads["D2"]["_oof_u"], d2noacc_logit_oof=heads["D2NOACC"]["_oof_u"])
    out = {"tool": "d2_fit.py", "prereg": "PREREG_LHA0_D2.md s2", "script_sha256": D.sha256_file(__file__), "d2_common_sha256": D.sha256_file(D.HERE / "d2_common.py"),
           "inputs_sha256": {"universe.npz": D.sha256_file(D.UNIVERSE), "oof_logits/A_s0_l100.npz": D.sha256_file(D.LHA0 / "result" / "oof_logits" / "A_s0_l100.npz"),
                             "lha0_features.npz": D.sha256_file(D.LHA0 / "result" / "lha0_features.npz")},
           "folds": {"k": 5, "seed": 0, "by": "fresh episode", "fn": "lh_fit.folds_of(d_eid, 5, 0)", **fold_sizes},
           "gates": {"G_sets_equal_l2_common": True, "fit_rows": 8178, "fit_episodes": 290, "class_counts_equal_LH_FIT_INFO": True, "lh_oof_ce_reproduced": ce_lh,
                     "features_all_finite": True},
           "heads": {k: {kk: vv for kk, vv in v.items() if kk != "_oof_u"} for k, v in heads.items()}, "fresh_oof_descriptive": desc, "wall_s": round(time.time() - t0, 1)}
    (ROOT / "result" / "D2_FIT_INFO.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    h = out["heads"]
    print(f"ZZFITZZ D2 lambda {h['D2']['chosen_lambda']:g} ce {h['D2']['chosen_oof_ce']:.5f} sha {h['D2']['frozen_sha256'][:12]} | NOACC lambda {h['D2NOACC']['chosen_lambda']:g} sha {h['D2NOACC']['frozen_sha256'][:12]}"
          f" | fresh OOF AUC {json.dumps(desc['auc'])} | D2-LH {desc['D2_minus_LH_OOF']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
