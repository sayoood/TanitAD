"""hsgate_fdcap_feasibility.py -- FIT-SIDE ONLY, DESCRIPTIVE (no bar, no eval139 read): is the committed G2-FAIL lever of PREREG_HSGATE s5 feasible?
"next = the same gate with its threshold chosen on FIT rows under an FD cap (FD_FIT(gate) <= FD_FIT(H2cn-OOF)), pre-registered first".
This script does NOT choose a threshold for any scored arm; it prints, on the FRESH FIT rows only, the FD / MD / staged-ADE_S6 / F-rate of the registered gate
family over a threshold grid, so the Master Mind can seal that pre-registration knowing whether any threshold satisfies the cap.  NEW file, refcv9-hsgate agent.

Inputs: the registered gate (HSGATE_FROZEN.json: lambda, features), FIT_ROWS.npz (X, y, ADEs, S6 positions of P_F / P_H(OOF), CV fold of every row) written by
hsgate_fit.py, and the S6 GT / v0 / dt of the same rows re-loaded through hsgate_fit.load_fit_side (asserted row-aligned).  FD / MD exactly as l2_common.lon_indicators.
Two readings: (a) HELD-OUT logits (5-fold episode CV, fold seed 1 == the registered lambda CV folds, the registered lambda) -- the honest one; (b) the frozen
gate in-sample (beside).
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


def main() -> int:
    t0 = time.time()
    H.env_check()
    prereg = H.check_prereg()
    L = H.lane()
    import hsgate_fit as HF
    D = HF.load_fit_side(L)
    tr6 = D["trs"]["S6"]
    R = np.load(H.SCRATCH / "FIT_ROWS.npz")
    assert np.array_equal(R["clip_index"], tr6["clip"]) and np.array_equal(R["ws"], tr6["ws"]), "FIT_ROWS row order != load_fresh S6 rows"
    gate = json.loads((H.WORK / "HSGATE_FROZEN.json").read_text(encoding="utf-8"))
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_common as L2
    E = {"g": np.asarray(tr6["G"], np.float64), "v0": np.asarray(tr6["v0"], np.float64), "dt": float(tr6["dt"])}
    allm = np.ones(len(R["y"]), bool)
    fdF, mdF, gt, _ = L2.lon_indicators(R["PF6"], E, allm)
    fdH, mdH, _, _ = L2.lon_indicators(R["PH6"], E, allm)
    X, y, aF, aH, fold = R["X"], R["y"], R["adeF"], R["adeH_oof"], R["cv_fold"]
    z = np.full(len(y), np.nan)
    for j in range(5):
        tr, te = fold != j, fold == j
        mu, sd = H.standardise_fit(X[tr])
        m = H.lr_fit((X[tr] - mu) / sd, y[tr], gate["lambda"])
        z[te] = H.gate_logit({"mu": mu, "sd": sd, "w": m["w"], "b": m["b"]}, X[te])
    z_in = H.gate_logit(gate, X)
    ref = {"FD_FIT_H2cn_OOF": float(np.nanmean(fdH)), "MD_FIT_H2cn_OOF": float(np.nanmean(mdH)), "FD_FIT_FACTORED": float(np.nanmean(fdF)),
           "MD_FIT_FACTORED": float(np.nanmean(mdF)), "n_GT_cruise": int(np.isfinite(fdH).sum()), "n_GT_decel": int(np.isfinite(mdH).sum()),
           "ADE_H2cn_OOF": float(aH.mean()), "ADE_FACTORED": float(aF.mean()), "n_rows": int(len(y))}

    def row(zz, t):
        g = zz >= t
        return {"tau_logit": float(t), "F_rate": float(g.mean()), "FD": float(np.nanmean(np.where(g, fdF, fdH))), "MD": float(np.nanmean(np.where(g, mdF, mdH))),
                "ADE_S6": float(np.where(g, aF, aH).mean())}
    qs = np.unique(np.concatenate([[0.0], np.quantile(z, np.linspace(0.5, 0.999, 60))]))
    table_ho = [row(z, t) for t in sorted(qs)]
    table_in = [row(z_in, t) for t in sorted(np.unique(np.concatenate([[0.0], np.quantile(z_in, np.linspace(0.5, 0.999, 60))])))]
    cap = ref["FD_FIT_H2cn_OOF"]
    lo, hi = row(z, np.inf), row(z, -np.inf)                       # known-value controls: no gating == H2cn-OOF, always-F == FACTORED
    controls = {"K-NONE_equals_H2cn_OOF": bool(lo["F_rate"] == 0 and lo["FD"] == cap and lo["MD"] == ref["MD_FIT_H2cn_OOF"] and lo["ADE_S6"] == ref["ADE_H2cn_OOF"]),
                "K-ALL_equals_FACTORED": bool(hi["F_rate"] == 1 and hi["FD"] == ref["FD_FIT_FACTORED"] and hi["MD"] == ref["MD_FIT_FACTORED"] and hi["ADE_S6"] == ref["ADE_FACTORED"]),
                "K-FROZEN_insample_decisions_equal_FIT_ROWS_g_final": bool(np.array_equal(z_in >= 0, R["g_final"])),
                "heldout_logits_all_finite": bool(np.isfinite(z).all()), "dt": float(tr6["dt"]),
                "v0_range": [float(np.min(E["v0"])), float(np.max(E["v0"]))]}
    controls["ALL_PASS"] = bool(controls["K-NONE_equals_H2cn_OOF"] and controls["K-ALL_equals_FACTORED"] and controls["K-FROZEN_insample_decisions_equal_FIT_ROWS_g_final"] and controls["heldout_logits_all_finite"])
    feas = [r for r in table_ho if r["FD"] <= cap and r["F_rate"] > 0]
    out = {"tool": "hsgate_fdcap_feasibility.py", "role": "FIT-SIDE DESCRIPTIVE ONLY (no bar, no eval139 read, no threshold chosen for any scored arm)",
           "prereg": prereg, "gate_sha256": H.sha256_file(H.WORK / "HSGATE_FROZEN.json"), "lambda": gate["lambda"], "reference": ref, "cap_FD": cap, "controls": controls,
           "registered_tau0_heldout": row(z, 0.0), "registered_tau0_insample": row(z_in, 0.0),
           "n_feasible_heldout_thresholds_with_F_rate_gt_0": len(feas),
           "best_feasible_heldout_by_MD": (min(feas, key=lambda r: (r["MD"], -r["tau_logit"])) if feas else None),
           "best_feasible_heldout_by_ADE": (min(feas, key=lambda r: (r["ADE_S6"], -r["tau_logit"])) if feas else None),
           "table_heldout": table_ho, "table_insample": table_in, "fit_rows_npz_sha256": H.sha256_file(H.SCRATCH / "FIT_ROWS.npz"),
           "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    od = H.WORK / "next"
    H.jdump(out, od / "FDCAP_FEASIBILITY_FIT.json")
    print(f"[fdcap] ref FD H2cn-OOF {cap:.4f} MD {ref['MD_FIT_H2cn_OOF']:.4f} | FACTORED FD {ref['FD_FIT_FACTORED']:.4f} MD {ref['MD_FIT_FACTORED']:.4f}", flush=True)
    print(f"[fdcap] controls {controls}", flush=True)
    print(f"[fdcap] tau0 held-out: {out['registered_tau0_heldout']}", flush=True)
    print(f"[fdcap] feasible held-out thresholds (F>0): {len(feas)}; best by MD {out['best_feasible_heldout_by_MD']}; best by ADE {out['best_feasible_heldout_by_ADE']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
