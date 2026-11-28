"""hsgate_fdcap_frontier.py -- FIT-SIDE ONLY, DESCRIPTIVE (no bar, no eval139 read, nothing frozen, nothing chosen for any scored arm).  NEW file, refcv9-hsgate agent.

Follows hsgate_fdcap_feasibility.py (quantile grid: 0 feasible thresholds under the PREREG_HSGATE s5 cap).  Two questions for the Master Mind's next pre-registration:
 A. EXACT frontier (every distinct logit threshold, per-row prefix scan, not a grid): does ANY threshold of the registered gate family satisfy
    FD_FIT(gate) <= FD_FIT(H2cn-OOF), and what MD does it buy there?  Read on HELD-OUT logits (5-fold episode CV, fold seed 1, the registered lambda)
    and on the frozen gate in-sample (beside).
 B. FD-AWARE LABELS (labels may use privileged GT -- CLAUDE.md 'LABELS MAY USE EGO'; inference features unchanged, the same 15): the same gate family
    re-fit under two alternative labels, held-out CV with the same folds and the registered lambda (lambda NOT re-selected -- descriptive):
      LAB_NOFD  = 1 iff staged-ADE_S6(P_F) < ADE(P_H-OOF)  AND gating would not ADD a false decel (not [GT-cruise and F decel and H not])
      LAB_MDFIX = 1 iff GT-decel and F decels and H-OOF does not   (gating fixes a missed decel)
    Each: held-out tau0 row + exact frontier.  Plus the ORACLE counts (rows where gating fixes an MD / adds an FD / removes an FD).
FD / MD exactly as l2_common.lon_indicators (plan_lon_class, dt = the fit-side S6 grid dt).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hsgate_common as H  # noqa: E402

import numpy as np  # noqa: E402


def heldout_logits(X, y, fold, lam):
    z = np.full(len(y), np.nan)
    for j in np.unique(fold):
        tr, te = fold != j, fold == j
        mu, sd = H.standardise_fit(X[tr])
        m = H.lr_fit((X[tr] - mu) / sd, y[tr], lam)
        z[te] = H.gate_logit({"mu": mu, "sd": sd, "w": m["w"], "b": m["b"]}, X[te])
    return z


def main() -> int:
    t0 = time.time()
    H.env_check()
    prereg = H.check_prereg()
    L = H.lane()
    import hsgate_fit as HF
    tr6 = HF.load_fit_side(L)["trs"]["S6"]
    R = np.load(H.SCRATCH / "FIT_ROWS.npz")
    assert np.array_equal(R["clip_index"], tr6["clip"]) and np.array_equal(R["ws"], tr6["ws"]), "FIT_ROWS row order != load_fresh S6 rows"
    gate = json.loads((H.WORK / "HSGATE_FROZEN.json").read_text(encoding="utf-8"))
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_common as L2
    E = {"g": np.asarray(tr6["G"], np.float64), "v0": np.asarray(tr6["v0"], np.float64), "dt": float(tr6["dt"])}
    allm = np.ones(len(R["y"]), bool)
    fdF, mdF, gt, pcF = L2.lon_indicators(R["PF6"], E, allm)
    fdH, mdH, _, pcH = L2.lon_indicators(R["PH6"], E, allm)
    X, y, aF, aH, fold = R["X"], R["y"], R["adeF"], R["adeH_oof"], R["cv_fold"]
    n = len(y)
    cru, dec = np.isfinite(fdH), np.isfinite(mdH)
    n_cru, n_dec = int(cru.sum()), int(dec.sum())
    dFD = np.where(cru, np.nan_to_num(fdF) - np.nan_to_num(fdH), 0.0)          # +1: gating ADDS a false decel; -1: removes one
    dMD = np.where(dec, np.nan_to_num(mdF) - np.nan_to_num(mdH), 0.0)          # -1: gating FIXES a missed decel; +1: creates one
    FD0, MD0, ADE0 = float(np.nansum(fdH)) / n_cru, float(np.nansum(mdH)) / n_dec, float(aH.mean())
    oracle = {"n_rows": n, "n_GT_cruise": n_cru, "n_GT_decel": n_dec, "rows_gating_fixes_MD": int((dMD < 0).sum()), "rows_gating_adds_MD": int((dMD > 0).sum()),
              "rows_gating_adds_FD": int((dFD > 0).sum()), "rows_gating_removes_FD": int((dFD < 0).sum()),
              "MD_if_gate_exactly_the_MD_fix_rows": float((np.nansum(mdH) - (dMD < 0).sum()) / n_dec),
              "FD_H2cn_OOF": FD0, "MD_H2cn_OOF": MD0, "ADE_H2cn_OOF": ADE0}

    def frontier(z):
        o = np.argsort(-z, kind="stable")
        zs = z[o]
        cFD, cMD, cADE = np.cumsum(dFD[o]), np.cumsum(dMD[o]), np.cumsum((aF - aH)[o])
        cut = np.r_[np.where(zs[:-1] > zs[1:])[0], n - 1]                     # prefix ends at distinct thresholds only
        k = cut + 1
        FD = FD0 + cFD[cut] / n_cru
        MD = MD0 + cMD[cut] / n_dec
        ADE = ADE0 + cADE[cut] / n
        feas = FD <= FD0 + 1e-12
        out = {"n_thresholds": int(len(cut)), "n_feasible_under_cap": int(feas.sum())}
        if feas.any():
            i_md = int(np.argmin(np.where(feas, MD, np.inf)))
            i_k = int(np.max(np.where(feas)[0]))
            out["best_MD_under_cap"] = {"F_rate": float(k[i_md] / n), "k": int(k[i_md]), "FD": float(FD[i_md]), "MD": float(MD[i_md]), "ADE_S6": float(ADE[i_md]),
                                        "tau_logit": float(zs[cut[i_md]])}
            out["largest_F_rate_under_cap"] = {"F_rate": float(k[i_k] / n), "k": int(k[i_k]), "FD": float(FD[i_k]), "MD": float(MD[i_k]), "ADE_S6": float(ADE[i_k]),
                                               "tau_logit": float(zs[cut[i_k]])}
        for slack in (0.002, 0.005):                                            # descriptive: MD bought at a small FD price
            f2 = FD <= FD0 + slack
            if f2.any():
                i = int(np.argmin(np.where(f2, MD, np.inf)))
                out[f"best_MD_at_FD_cost_le_{slack}"] = {"F_rate": float(k[i] / n), "FD": float(FD[i]), "MD": float(MD[i]), "ADE_S6": float(ADE[i]),
                                                          "tau_logit": float(zs[cut[i]])}
        g0 = z >= 0
        out["tau0"] = {"F_rate": float(g0.mean()), "FD": float(FD0 + dFD[g0].sum() / n_cru), "MD": float(MD0 + dMD[g0].sum() / n_dec),
                       "ADE_S6": float(np.where(g0, aF, aH).mean())}
        return out

    lam = float(gate["lambda"])
    z_ho = heldout_logits(X, y, fold, lam)
    z_in = H.gate_logit(gate, X)
    labs = {"LAB_ADE_registered": y, "LAB_NOFD": (y.astype(bool) & ~(dFD > 0)).astype(np.float64), "LAB_MDFIX": (dMD < 0).astype(np.float64)}
    res = {"registered_gate_heldout": frontier(z_ho), "registered_gate_insample": frontier(z_in)}
    for nm in ("LAB_NOFD", "LAB_MDFIX"):
        zz = heldout_logits(X, labs[nm], fold, lam)
        r = frontier(zz)
        r["label_base_rate"] = float(labs[nm].mean())
        res[nm + "_heldout"] = r
    controls = {"K-REPRO_tau0_heldout_equals_feasibility_run": None,
                "K-FROZEN_insample_decisions_equal_g_final": bool(np.array_equal(z_in >= 0, R["g_final"])),
                "K-PREFIX_EMPTY_equals_H2cn": bool(abs(FD0 - float(np.nanmean(fdH))) < 1e-12 and abs(MD0 - float(np.nanmean(mdH))) < 1e-12),
                "K-PREFIX_FULL_equals_FACTORED": bool(abs(FD0 + dFD.sum() / n_cru - float(np.nanmean(fdF))) < 1e-12 and abs(MD0 + dMD.sum() / n_dec - float(np.nanmean(mdF))) < 1e-12)}
    fz = H.WORK / "next" / "FDCAP_FEASIBILITY_FIT.json"
    if fz.exists():
        prev = json.loads(fz.read_text(encoding="utf-8"))["registered_tau0_heldout"]
        t0r = res["registered_gate_heldout"]["tau0"]
        controls["K-REPRO_tau0_heldout_equals_feasibility_run"] = bool(abs(prev["FD"] - t0r["FD"]) < 1e-12 and abs(prev["MD"] - t0r["MD"]) < 1e-12
                                                                        and abs(prev["F_rate"] - t0r["F_rate"]) < 1e-12)
    controls["ALL_PASS"] = bool(all(v is True for v in controls.values()))
    out = {"tool": "hsgate_fdcap_frontier.py", "role": "FIT-SIDE DESCRIPTIVE ONLY (no bar, no eval139 read, nothing frozen, no threshold chosen for any scored arm)",
           "prereg": prereg, "gate_sha256": H.sha256_file(H.WORK / "HSGATE_FROZEN.json"), "lambda": lam, "lambda_note": "registered lambda reused for the alt labels, NOT re-selected",
           "cv": "5-fold episode CV, fold seed 1 (the registered lambda-CV folds, FIT_ROWS.cv_fold)", "oracle": oracle, "controls": controls, "results": res,
           "fit_rows_npz_sha256": H.sha256_file(H.SCRATCH / "FIT_ROWS.npz"), "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    H.jdump(out, H.WORK / "next" / "FDCAP_FRONTIER_FIT.json")
    print(f"[frontier] controls {controls}", flush=True)
    print(f"[frontier] oracle {oracle}", flush=True)
    for k_, v in res.items():
        print(f"[frontier] {k_}: {json.dumps(v)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
