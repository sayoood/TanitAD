#!/usr/bin/env python3
"""diag_vhi.py -- POST-HOC DIAGNOSTIC, run AFTER the HPROJ read was fixed. NOT a PREREG row, carries NO bar, changes NO outcome.

Why it exists. The HPROJ read came back CONSISTENCY NOT RESTORED. The PREREG's explanation for that outcome ("H's violations are not of the posted
limit: they are stops, leads or turns") is a HYPOTHESIS about the rule mix. The per-rule breakdown shows SPEED carries nearly all of the excess, and the
eval's SPEED rule caps at min(posted limit, v_hi_ms) -- v_hi_ms is a GT-derived speed-band edge (SPEED_BAND), not the posted-limit INPUT the projection
reads. This script measures how much of the remaining SPEED cost is the v_hi term, so the committed constraint-aware PREREG is written against the
measured mix rather than the guessed one.

   HPv_or   cap = min(posted limit, GT v_hi_ms)            T0 ORACLE (reads a GT label at inference). A ceiling on what a v_hi term could buy. NEVER a result.
   HPv_pr   cap = min(posted limit, T's predicted v_hi_ms) T1-class (T's own `cons_pred` head; vision-only at inference). Nothing fitted, nothing tuned.
   Bv_pr    B's reconstructed traj60 under the same predicted cap (so the projection's own cost on B is separable from H's).
No weight is changed. Same estimator (paired episode-cluster bootstrap, B = 2000) and the same eval-141 windows as the PREREG rows.
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

import v8f_hproj as HP  # noqa: E402

H = HP.H


def main():
    ev = H.load_eval_module()
    from taniteval import v8f_beta_eval as BE
    from tanitad.data import v8f_mini_data as D
    res = {"label": "POST-HOC DIAGNOSTIC (not a PREREG row, no bar, no outcome change)", "seeds": {}}
    for seed in H.SEEDS:
        ctx = HP.load_ctx(seed)
        X, z, gt, eid, tau = ctx["X"], ctx["z"], ctx["gt"], ctx["eid"], ctx["tau"]
        S = np.load(f"{HP.OUT}/score_{seed}_perwindow.npz")
        cap_lim, H60 = S["cap"].astype(np.float64), S["H60"]
        tau60, _, _ = HP.recon_traj60(tau, ctx["P"], HP.basis_np(ctx["cfg"].nb))
        lim = np.asarray(X["limit_ms"], np.float64)
        vhi_gt = np.where(np.asarray(X["has"], bool), np.asarray(X["cons"][:, D.CONS_IDX["v_hi_ms"]], np.float64), np.nan)
        vhi_pr = z["cons_pred"][:, D.CONS_IDX["v_hi_ms"]].astype(np.float64)
        info = {"n_windows": int(len(eid)), "pred_v_hi_ms": {"min": float(vhi_pr.min()), "p05": float(np.percentile(vhi_pr, 5)), "median": float(np.median(vhi_pr)),
                                                           "max": float(vhi_pr.max()), "n_nonpositive": int((vhi_pr <= 0).sum()), "n_nonfinite": int((~np.isfinite(vhi_pr)).sum())},
                "gt_v_hi_ms": {"n_finite": int(np.isfinite(vhi_gt).sum()), "median": float(np.nanmedian(vhi_gt))},
                "posted_limit_ms_median": float(np.median(lim)),
                "pred_v_hi_vs_gt_v_hi_mae_ms": float(np.nanmean(np.abs(vhi_pr - vhi_gt))),
                "n_windows_gt_v_hi_below_limit": int((np.isfinite(vhi_gt) & (vhi_gt < lim)).sum()),
                "n_windows_pred_v_hi_below_limit": int((vhi_pr < lim).sum())}
        # ---- the GT plan against the limit ALONE vs against the rule's own cap (is the posted limit even a constraint the human respects?)
        seg = np.diff(np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1), axis=1)
        v = np.linalg.norm(seg, axis=-1) / HP.DT
        vmax = v[:, 10:].max(1)
        rule_cap = np.fmin(lim, vhi_gt)
        info["GT_plan_vs_posted_limit_alone"] = {"violation_rate": float((vmax > lim + 0.5).mean()), "n": int((vmax > lim + 0.5).sum())}
        info["GT_plan_vs_rule_cap_min_limit_vhi"] = {"violation_rate": float((np.isfinite(rule_cap) & (vmax > np.where(np.isfinite(rule_cap), rule_cap, 0) + 0.5)).mean())}
        caps = {"HPv_or": np.where(np.isfinite(vhi_gt), np.fmin(lim, vhi_gt), lim), "HPv_pr": np.fmin(lim, np.maximum(vhi_pr, 0.0))}
        arms = {"HPv_or": HP.project_f32(H60, caps["HPv_or"].astype(np.float32))[:, 1::2],
                "HPv_pr": HP.project_f32(H60, caps["HPv_pr"].astype(np.float32))[:, 1::2],
                "Bv_pr": HP.project_f32(tau60, caps["HPv_pr"].astype(np.float32))[:, 1::2]}
        ex = H.extras_of(z, D)
        PWa, V = {}, {}
        for n, p in arms.items():
            r, pw = ev.evaluate(p, X, ex, HP.B_BOOT, with_families=False)
            PWa[n] = H.arm_pw(p, gt, BE)
            V[n] = {k: pw[k].astype(float) for k in HP.CK}
            info[f"{n}_rate"] = {k: float(V[n][k].mean()) for k in HP.CK}
        # references from the PREREG run
        sc = json.load(open(f"{HP.OUT}/score_{seed}.json"))
        info["reference_rates_from_score_json"] = {n: sc["consistency"]["rate"][n] for n in ("B", "H", "HP", "Bp")}
        pwB = H.arm_pw(tau, gt, BE)
        vB = {k: z[k].astype(float) for k in HP.CK}
        info["paired_vs_B"] = {n: {"ade2": BE.paired_row(PWa[n]["ade2"], pwB["ade2"], eid, HP.B_BOOT), "ade6": BE.paired_row(PWa[n]["ade6"], pwB["ade6"], eid, HP.B_BOOT),
                                   "cons": {k: BE.paired_row(V[n][k], vB[k], eid, HP.B_BOOT) for k in HP.CK}} for n in arms}
        res["seeds"][seed] = info
        pv = info["paired_vs_B"]
        print(seed, {n: {"ade2": round(pv[n]["ade2"]["mean"], 4), "ade6": round(pv[n]["ade6"]["mean"], 4), "cons_any": f"{pv[n]['cons']['viol_any']['mean']:+.4f} hi {pv[n]['cons']['viol_any']['hi']:+.4f}"} for n in arms}, flush=True)
    H.jdump(res, f"{HP.OUT}/diag_vhi.json")


if __name__ == "__main__":
    main()
