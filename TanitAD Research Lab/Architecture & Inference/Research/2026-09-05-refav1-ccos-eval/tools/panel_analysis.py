"""D-REFAV1-CCOS-EVAL step 1(a) + step 2 on the FULL 282-window grid, from the REAL
implementation's goal term (`box_panel_282.json`, Thor, `ccos_box_panel.py`: one `plan()`
capture PER METRIC, so each metric's closure is the one the planner would use; `goal_<m>` is
`refa_v1._goal_term`'s float32 output, `cost_<m>` the closure's TOTAL cost at shipped weights),
cross-checked against the second, independently written probe (`ccos_panel_probe.json`, same
box core, one capture) and the float64 re-implementation banked inside both.

Zero GPU. n = 282 windows / 141 episode clusters. Intervals: episode-cluster bootstrap
(`taniteval.ci`) where a per-window statistic is reported with one.

OUTPUTS (all per metric in COST_METRICS = cos / chord / ccos):
  * goal(cv): the do-nothing candidate's goal term (its TOTAL cost, since its penalties are 0)
  * the a-priori ITERATION-0 EXCLUSION: the fraction of a REAL 300-sample iCEM iteration-0
    population whose penalty alone exceeds goal(cv) - min_n goal(n) <= goal(cv), i.e. that
    cannot beat cv whatever the world model says (goal term non-negative). Reported at the
    median window and at the most favourable window, at SHIPPED and at COMPENSATED weights.
  * the L/R decision as a fraction of the term (|g(kap+0.1) - g(kap-0.1)| / mean), box spread
  * weight-neutrality factors vs cos: value (median cell ratio), decision (median box-spread
    ratio), L/R-gap ratio — on n = 282 (the dev-box n = 40 numbers are the pilot)
  * CONTROLS, each with its KNOWN value: zero-model ptp (must be 0.0 exactly, all metrics),
    identity (chord exactly 0; cos/ccos <= float32 cosine error), ccos(cv) (1.0 exactly only
    when the cv field is bit-identical to z_ref — measured, not assumed), float64 cross-check
    max|ccos32 - ccos64|, and the two probes' agreement on cos/chord/ccos for the shared rows.
  * the DELIBERATE-REGRESSION reading: chord (uncentred) must remain 100 % excluded.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

W_JERK, W_KAPPA, W_VEND = 0.02, 0.05, 0.10


def icem_iter0(stack):
    sys.path.insert(0, stack)
    import torch
    from tanitad.refs.refa_v1_plan import colored_noise, PlanConfig, _clip
    pc = PlanConfig(horizon=10, dt=0.2, seed=0)
    gen = torch.Generator().manual_seed(0)
    S = _clip(torch.zeros(pc.horizon, 2) + colored_noise((pc.n_samples, pc.horizon, 2), pc.beta,
                                                         generator=gen), pc)
    jerk = ((S[:, 1:, 0] - S[:, :-1, 0]) / pc.dt).pow(2).mean(-1).numpy()
    kap = S[..., 1].pow(2).mean(-1).numpy()
    return jerk, kap


def ci_median(x, eid, n_boot=2000, seed=0):
    """episode-cluster bootstrap of the MEDIAN of a per-window statistic."""
    rng = np.random.default_rng(seed)
    eid = np.asarray(eid); x = np.asarray(x, float)
    ue = np.unique(eid); idx = {e: np.where(eid == e)[0] for e in ue}
    meds = []
    for _ in range(n_boot):
        pick = rng.choice(ue, size=len(ue), replace=True)
        meds.append(np.median(np.concatenate([x[idx[e]] for e in pick])))
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="box_panel_282.json (primary)")
    ap.add_argument("--probe", default=None, help="ccos_panel_probe.json (cross-check)")
    ap.add_argument("--stack", required=True)
    ap.add_argument("--factor-json", default=None, help="ccos_scale_factor_devbox40.json (pilot)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    J = json.load(open(a.panel, encoding="utf-8"))
    BN, rows = J["box_names"], J["rows"]
    N = len(rows); eid = np.array([r["ep"] for r in rows])
    nd = 22 if "rampR" in BN else len(BN)     # designed rows; seeds/proposal follow
    i_cv = BN.index("cv"); iL, iR = BN.index("kap+0.1"), BN.index("kap-0.1")
    kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]
    jerk, kap = icem_iter0(a.stack)
    pen_ship = W_JERK * jerk + W_KAPPA * kap
    G = {m: np.array([r[f"goal_{m}"][:nd] for r in rows], float) for m in ("cos", "chord", "ccos")}
    C = {m: np.array([r[f"cost_{m}"][:nd] for r in rows], float) for m in ("cos", "chord", "ccos")}
    out = {"tool": "panel_analysis.py", "source": a.panel, "n_windows": N,
           "n_episodes": int(len(np.unique(eid))), "box_names": BN, "n_designed": nd,
           "goal_dim": J.get("goal_dim"), "model": J.get("model", {}),
           "icem_iter0": {"n": int(len(jerk)), "jerk_raw_min": float(jerk.min()),
                          "jerk_raw_median": float(np.median(jerk)),
                          "penalty_shipped_median": float(np.median(pen_ship)),
                          "penalty_shipped_p90": float(np.percentile(pen_ship, 90)),
                          "penalty_shipped_max": float(pen_ship.max())},
           "metrics": {}}
    for m in ("cos", "chord", "ccos"):
        g = G[m]; gcv = g[:, i_cv]
        spread = g.max(1) - g.min(1)
        kspread = g[:, [i_cv] + kidx].max(1) - g[:, [i_cv] + kidx].min(1)
        lr = np.abs(g[:, iL] - g[:, iR]); lr_rel = lr / np.maximum(0.5 * (g[:, iL] + g[:, iR]), 1e-300)
        # closed-form iteration-0 exclusion, model-free: penalty >= goal(cv) - min_n goal(n)
        best = gcv - g.min(1)                       # the most a candidate could gain on the goal term
        ex_med = float((pen_ship >= np.median(gcv)).mean() * 100)
        ex_max = float((pen_ship >= gcv.max()).mean() * 100)
        ex_best_med = float((pen_ship >= np.median(best)).mean() * 100)
        # cv's TOTAL cost must equal its goal term exactly (its penalties are zero)
        cost_cv = C[m][:, i_cv]
        out["metrics"][m] = {
            "goal_cv": {"median": float(np.median(gcv)), "p10": float(np.percentile(gcv, 10)),
                        "p90": float(np.percentile(gcv, 90)), "max": float(gcv.max()),
                        "min": float(gcv.min()), "n_exactly_0": int((gcv == 0).sum()),
                        "n_exactly_1": int((gcv == 1.0).sum()),
                        "ci95_median_episode_cluster": ci_median(gcv, eid)},
            "cost_cv_equals_goal_cv_max_abs": float(np.abs(cost_cv - gcv).max()),
            "term_median_over_box": float(np.median(g)),
            "box_spread_median": float(np.median(spread)),
            "kappa_subbox_spread_median": float(np.median(kspread)),
            "LR_gap_median": float(np.median(lr)),
            "LR_rel_to_term_median_pct": float(np.median(lr_rel) * 100),
            "LR_rel_ci95": [x * 100 for x in ci_median(lr_rel, eid)],
            "exclusion_iter0_shipped": {"at_median_window_pct": ex_med, "at_max_window_pct": ex_max,
                                        "using_goal_cv_minus_best_candidate_median_pct": ex_best_med},
        }
    # leverage factors vs cos on n = 282
    cos = G["cos"]
    for m in ("chord", "ccos"):
        g = G[m]
        cells = cos > 0
        ratio = g[cells] / cos[cells]
        sp_c = cos.max(1) - cos.min(1); sp_f = g.max(1) - g.min(1); keep = sp_c > 0
        lr_c = np.abs(cos[:, iL] - cos[:, iR]); lr_f = np.abs(g[:, iL] - g[:, iR]); kl = lr_c > 0
        Fv = float(np.median(ratio)); Fd = float(np.median(sp_f[keep] / sp_c[keep]))
        out["metrics"][m]["leverage_vs_cos"] = {
            "value_median_cell_ratio": Fv, "value_ci95": ci_median(np.median(np.where(cells, g / np.where(cells, cos, 1), np.nan), axis=1)[~np.isnan(np.nanmedian(np.where(cells, g / np.where(cells, cos, 1), np.nan), axis=1))], eid[~np.isnan(np.nanmedian(np.where(cells, g / np.where(cells, cos, 1), np.nan), axis=1))]) if cells.any() else None,
            "decision_median_box_spread_ratio": Fd, "decision_ci95": ci_median(sp_f[keep] / sp_c[keep], eid[keep]),
            "LR_gap_median_ratio": float(np.median(lr_f[kl] / lr_c[kl])),
            "n_cells": int(cells.sum()), "n_windows": int(keep.sum()),
            "compensated_triple_decision": [W_JERK * Fd, W_KAPPA * Fd, W_VEND * Fd],
            "compensated_triple_value": [W_JERK * Fv, W_KAPPA * Fv, W_VEND * Fv],
        }
        for tag, F in (("decision", Fd), ("value", Fv)):
            pen = W_JERK * F * jerk + W_KAPPA * F * kap
            gcv = g[:, i_cv]
            out["metrics"][m][f"exclusion_iter0_compensated_{tag}"] = {
                "factor": F, "at_median_window_pct": float((pen >= np.median(gcv)).mean() * 100),
                "at_max_window_pct": float((pen >= gcv.max()).mean() * 100)}
    # the pilot factor the arm-2 weights were actually set from
    if a.factor_json and os.path.exists(a.factor_json):
        out["pilot_devbox40"] = json.load(open(a.factor_json, encoding="utf-8")).get("factors")
    # CONTROLS on the real function
    ctl = [r["controls"] for r in rows]
    ccos64 = np.array([r["ccos64"][:nd] for r in rows], float)
    out["controls"] = {
        "zero_model_ptp_max": {m: float(max(c[f"zero_model_ptp_{m}"] for c in ctl)) for m in ("cos", "chord", "ccos")},
        "zero_model_value_ccos_unique": sorted({round(c["zero_model_value_ccos"], 8) for c in ctl})[:5],
        "identity_max_abs": {m: float(max(abs(c[f"ident_{m}"]) for c in ctl)) for m in ("cos", "chord", "ccos")},
        "ccos_cv_from_plan_closure": {
            "median": float(np.median([c["ccos_cv"] for c in ctl])),
            "min": float(min(c["ccos_cv"] for c in ctl)), "max": float(max(c["ccos_cv"] for c in ctl)),
            "n_exactly_1": int(sum(c["ccos_cv"] == 1.0 for c in ctl)),
            "note": "cv row scored ALONE against the captured z_ref: bit-identical fields give exactly 1.0; "
                    "the in-batch value (basecost_cv_cl in the arm dumps) is the one the planner sees"},
        "float64_crosscheck_max_abs_ccos32_minus_ccos64": float(np.abs(G["ccos"] - ccos64).max()),
        "float64_crosscheck_median_abs": float(np.median(np.abs(G["ccos"] - ccos64))),
        "goalresp_rel": {"median": float(np.median([r["goalresp_rel"] for r in rows])),
                         "n_below_1e-6": int(sum(r["goalresp_rel"] < 1e-6 for r in rows)),
                         "note": "||g - z_ref|| / ||z_ref||: float rounding when the decoded goal is HOLD"},
        "refnorm_median": float(np.median([r["refnorm"] for r in rows])),
    }
    if a.probe and os.path.exists(a.probe):
        P = json.load(open(a.probe, encoding="utf-8"))
        PB = P["box_names"]; prow = {(r["ep"], r["t"]): r for r in P["rows"]}
        shared = [n for n in BN[:nd] if n in PB]
        ia = [BN.index(n) for n in shared]; ib = [PB.index(n) for n in shared]
        agree = {}
        for m in ("cos", "chord", "ccos"):
            d = []
            for r in rows:
                q = prow.get((r["ep"], r["t"]))
                if q is None:
                    continue
                d.append(np.abs(np.array(r[f"goal_{m}"])[ia] - np.array(q[m])[ib]))
            d = np.concatenate(d)
            agree[m] = {"max_abs": float(d.max()), "median_abs": float(np.median(d)), "n_cells": int(d.size)}
        out["crosscheck_second_probe"] = {"source": a.probe, "shared_box_rows": shared,
                                          "n_windows_matched": int(sum((r["ep"], r["t"]) in prow for r in rows)),
                                          "agreement": agree,
                                          "note": "two independently written capture tools, same checkpoint, same windows; "
                                                  "differences are batch-composition ulps (the probe's cv row is rolled with "
                                                  "a different box) — agreement to ~1e-6 is the pass"}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    # ---- markdown ---------------------------------------------------------------
    L = []
    L.append(f"n = {N} windows / {out['n_episodes']} episode clusters; box = {nd} designed candidates (+ seeds/proposal not used here); goal dim {J.get('goal_dim')}; REAL `refa_v1._goal_term` float32 (Thor)")
    L.append("")
    L.append("| metric | goal(cv) median [CI] | goal(cv) max | exactly 0 / exactly 1 | term median over box | box spread | L/R decision as % of term [CI] | iter-0 EXCLUDED @median / @max window (shipped w) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in ("cos", "chord", "ccos"):
        v = out["metrics"][m]; gc = v["goal_cv"]; ex = v["exclusion_iter0_shipped"]
        L.append(f"| `{m}` | {gc['median']:.3g} [{gc['ci95_median_episode_cluster'][0]:.3g}, {gc['ci95_median_episode_cluster'][1]:.3g}] | {gc['max']:.3g} | {gc['n_exactly_0']} / {gc['n_exactly_1']} | {v['term_median_over_box']:.3g} | {v['box_spread_median']:.3g} | **{v['LR_rel_to_term_median_pct']:.2f} %** [{v['LR_rel_ci95'][0]:.2f}, {v['LR_rel_ci95'][1]:.2f}] | **{ex['at_median_window_pct']:.2f} % / {ex['at_max_window_pct']:.2f} %** |")
    L.append("")
    L.append("| form vs `cos` | value leverage (median cell ratio) | decision leverage (median box-spread ratio) [CI] | L/R-gap ratio | compensated triple (decision) | iter-0 excluded @median, compensated decision / value |")
    L.append("|---|---|---|---|---|---|")
    for m in ("chord", "ccos"):
        v = out["metrics"][m]; lv = v["leverage_vs_cos"]
        cd, cv_ = v["exclusion_iter0_compensated_decision"], v["exclusion_iter0_compensated_value"]
        L.append(f"| `{m}` | {lv['value_median_cell_ratio']:,.1f}x | **{lv['decision_median_box_spread_ratio']:,.2f}x** [{lv['decision_ci95'][0]:,.1f}, {lv['decision_ci95'][1]:,.1f}] | {lv['LR_gap_median_ratio']:,.1f}x | ({', '.join(f'{x:.4g}' for x in lv['compensated_triple_decision'])}) | {cd['at_median_window_pct']:.2f} % / {cv_['at_median_window_pct']:.2f} % |")
    c = out["controls"]
    L.append("")
    L.append("Controls (real function, real fields): zero-model ptp max = " + ", ".join(f"`{m}` {c['zero_model_ptp_max'][m]:.3g}" for m in c["zero_model_ptp_max"]) +
             "; identity max|.| = " + ", ".join(f"`{m}` {c['identity_max_abs'][m]:.3g}" for m in c["identity_max_abs"]) +
             f"; ccos(cv) scored alone vs z_ref: exactly 1.0 on {c['ccos_cv_from_plan_closure']['n_exactly_1']}/{N} (min {c['ccos_cv_from_plan_closure']['min']:.6f}, max {c['ccos_cv_from_plan_closure']['max']:.6f})"
             f"; float64 cross-check max|ccos32-ccos64| = {c['float64_crosscheck_max_abs_ccos32_minus_ccos64']:.3g}; ||g-z_ref||/||z_ref|| median {c['goalresp_rel']['median']:.3g}, < 1e-6 on {c['goalresp_rel']['n_below_1e-6']}/{N} windows (the HOLD stratum).")
    if "crosscheck_second_probe" in out:
        x = out["crosscheck_second_probe"]
        L.append("Second probe (independently written): matched " + str(x["n_windows_matched"]) + " windows; max|Δ| " +
                 ", ".join(f"`{m}` {x['agreement'][m]['max_abs']:.3g}" for m in x["agreement"]) + ".")
    md = "\n".join(L)
    print(md)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
