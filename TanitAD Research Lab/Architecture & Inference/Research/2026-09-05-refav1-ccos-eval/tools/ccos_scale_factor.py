"""D-REFAV1-CCOS-EVAL step 2 — the WEIGHT-NEUTRALITY factor of ``ccos`` against the
shipped ``(W_JERK, W_KAPPA, W_VEND) = (0.02, 0.05, 0.10)``, and the compensating triple.

Zero GPU. Reads the banked FORM comparison (``cost_forms_devbox.json``: n = 40 windows /
20 clips, dev-box RTX 4060, checkpoint step 21,109, the 26-candidate designed box + the
planner's own seeds, every form computed in float64 from the SAME captured (x, y) the
shipped ``_goal_term`` was handed) and optionally the 282-window probe of the REAL
implementation (``ccos_panel_probe.json``, Thor) when it exists.

THE TWO FACTORS, stated separately because they answer different questions:

  value leverage     median over (window, candidate) cells of  ccos / cos
                     — the ratio of the TERM's magnitude. This is the analogue of the
                     chord's "5,792.6x" (`.../2026-09-03-cost-repair/RESULT.md`), which
                     was a ratio of term VALUES.
  decision leverage  median over windows of  spread_box(ccos) / spread_box(cos)
                     where spread_box = max - min of the term over the designed box
                     — the ratio of what the term can actually TRADE against the
                     penalties. It is the one that governs ranking.

COMPENSATION: multiplying all three weights by a factor F keeps the goal:penalty
balance where the SHIPPED cos left it. ``W_VEND`` is never charged on the eval path
(``target_speed=None``), so it is carried for completeness only. The DECISION-scale
triple is the one the compensated arm runs under; the VALUE-scale triple is reported
and its a-priori consequence (100 % exclusion by the closed-form bound) is stated.

The closed-form ITERATION-0 exclusion bound (`analyse8_exclusion.py`, same rule):
a candidate n can beat cv only if  penalty(n) < goal(cv) - goal(n) <= goal(cv)  because
the goal term is non-negative; for ccos goal(cv) = 1.0 by construction (centred vector
is the zero vector), so the model-free bound is  penalty(n) < 1.0.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

W_JERK, W_KAPPA, W_VEND = 0.02, 0.05, 0.10


def icem_iter0_population(stack):
    sys.path.insert(0, stack)
    import torch
    from tanitad.refs.refa_v1_plan import colored_noise, PlanConfig, _clip
    pc = PlanConfig(horizon=10, dt=0.2, seed=0)
    gen = torch.Generator().manual_seed(0)
    S = _clip(torch.zeros(pc.horizon, 2) + colored_noise(
        (pc.n_samples, pc.horizon, 2), pc.beta, generator=gen), pc)
    jerk = ((S[:, 1:, 0] - S[:, :-1, 0]) / pc.dt).pow(2).mean(-1).numpy()
    kap = S[..., 1].pow(2).mean(-1).numpy()
    return jerk, kap


def analyse(J, forms=("cos", "chord", "ccos"), tag=""):
    BN, rows = J["box_names"], J["rows"]
    A = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    i_cv = BN.index("cv")
    kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]
    iL, iR = BN.index("kap+0.1"), BN.index("kap-0.1")
    G = {f: A(f) for f in forms}
    n_box = len(BN)
    out = {"n_windows": len(rows), "n_box": n_box, "box_names": BN, "tag": tag,
           "forms": {}}
    for f in forms:
        g = G[f]
        gcv = g[:, i_cv]
        sub = [i_cv] + kidx
        spread_kbox = g[:, sub].max(1) - g[:, sub].min(1)
        spread_box = g[:, :n_box].max(1) - g[:, :n_box].min(1)
        lr_gap = np.abs(g[:, iL] - g[:, iR])
        lr_rel = lr_gap / np.maximum(0.5 * (g[:, iL] + g[:, iR]), 1e-300)
        out["forms"][f] = {
            "median_over_cells": float(np.median(g[:, :n_box])),
            "goal_cv": {"median": float(np.median(gcv)), "p90": float(np.percentile(gcv, 90)),
                        "max": float(gcv.max()), "min": float(gcv.min()),
                        "n_exactly_1": int((gcv == 1.0).sum())},
            "spread_over_box_median": float(np.median(spread_box)),
            "spread_over_kappa_subbox_median": float(np.median(spread_kbox)),
            "LR_gap_median": float(np.median(lr_gap)),
            "LR_gap_rel_to_term_median_pct": float(np.median(lr_rel) * 100),
        }
    # leverage of each non-default form against the shipped cos
    cos = G["cos"]
    for f in forms:
        if f == "cos":
            continue
        g = G[f]
        cells = (cos[:, :n_box] > 0) & np.isfinite(g[:, :n_box])
        ratio = g[:, :n_box][cells] / cos[:, :n_box][cells]
        sp_c = (cos[:, :n_box].max(1) - cos[:, :n_box].min(1))
        sp_f = (g[:, :n_box].max(1) - g[:, :n_box].min(1))
        keep = sp_c > 0
        kb_c = cos[:, [i_cv] + kidx].max(1) - cos[:, [i_cv] + kidx].min(1)
        kb_f = g[:, [i_cv] + kidx].max(1) - g[:, [i_cv] + kidx].min(1)
        keep_k = kb_c > 0
        lr_c = np.abs(cos[:, iL] - cos[:, iR]); lr_f = np.abs(g[:, iL] - g[:, iR])
        keep_lr = lr_c > 0
        out["forms"][f]["leverage_vs_cos"] = {
            "value_median_ratio_over_cells": float(np.median(ratio)),
            "value_ratio_p10_p90": [float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))],
            "n_cells": int(cells.sum()),
            "decision_median_ratio_of_box_spread": float(np.median(sp_f[keep] / sp_c[keep])),
            "decision_ratio_p10_p90": [float(np.percentile(sp_f[keep] / sp_c[keep], 10)),
                                       float(np.percentile(sp_f[keep] / sp_c[keep], 90))],
            "decision_median_ratio_of_kappa_subbox_spread": float(np.median(kb_f[keep_k] / kb_c[keep_k])),
            "LR_gap_median_ratio": float(np.median(lr_f[keep_lr] / lr_c[keep_lr])),
            "n_windows": int(keep.sum()),
        }
    return out


def exclusion(out, jerk, kap, settings):
    """iteration-0 exclusion under each (form, weights): penalty >= goal(cv) at the
    median window and at the most favourable window (max goal(cv))."""
    res = []
    for lbl, f, wj, wk in settings:
        gcv = out["forms"][f]["goal_cv"]
        pen = wj * jerk + wk * kap
        res.append({"setting": lbl, "form": f, "W_JERK": wj, "W_KAPPA": wk,
                    "goal_cv_median": gcv["median"], "goal_cv_max": gcv["max"],
                    "excluded_at_median_window_pct": float((pen >= gcv["median"]).mean() * 100),
                    "excluded_at_max_window_pct": float((pen >= gcv["max"]).mean() * 100),
                    "n_population": int(len(pen))})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms-json", required=True)
    ap.add_argument("--probe-json", default=None, help="282-window REAL-implementation probe")
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    jerk, kap = icem_iter0_population(a.stack)
    J = json.load(open(a.forms_json, encoding="utf-8"))
    out = {"tool": "ccos_scale_factor.py", "shipped_weights": [W_JERK, W_KAPPA, W_VEND],
           "icem_iter0": {"n": int(len(jerk)), "jerk_raw_min": float(jerk.min()),
                          "jerk_raw_median": float(np.median(jerk)),
                          "penalty_shipped_median": float(np.median(W_JERK * jerk + W_KAPPA * kap)),
                          "penalty_shipped_max": float((W_JERK * jerk + W_KAPPA * kap).max())},
           "sources": {"forms_json": a.forms_json, "probe_json": a.probe_json}}
    out["devbox40"] = analyse(J, tag="cost_forms_devbox.json n=40 windows / 20 clips, float64 re-implementation on captured fields")
    lev = out["devbox40"]["forms"]["ccos"]["leverage_vs_cos"]
    Fv = lev["value_median_ratio_over_cells"]; Fd = lev["decision_median_ratio_of_box_spread"]
    out["factors"] = {
        "value_leverage": Fv, "decision_leverage": Fd,
        "compensated_triple_decision_scale": [W_JERK * Fd, W_KAPPA * Fd, W_VEND * Fd],
        "compensated_triple_value_scale": [W_JERK * Fv, W_KAPPA * Fv, W_VEND * Fv],
        "rule": ("compensated = shipped x F; F_decision preserves the goal:penalty balance at the "
                 "scale that trades against the penalties (ranking); F_value preserves it at the "
                 "term-magnitude scale (the chord's 5,792.6x convention)")}
    settings = [
        ("S0 cos shipped", "cos", W_JERK, W_KAPPA),
        ("chord shipped (deliberate regression)", "chord", W_JERK, W_KAPPA),
        ("ccos NAIVE flip @ shipped", "ccos", W_JERK, W_KAPPA),
        ("ccos COMPENSATED decision-scale", "ccos", W_JERK * Fd, W_KAPPA * Fd),
        ("ccos COMPENSATED value-scale", "ccos", W_JERK * Fv, W_KAPPA * Fv),
    ]
    out["exclusion_iter0_devbox40"] = exclusion(out["devbox40"], jerk, kap, settings)
    if a.probe_json and os.path.exists(a.probe_json):
        P = json.load(open(a.probe_json, encoding="utf-8"))
        out["probe282"] = analyse(P, tag=P.get("tag", "ccos_panel_probe.json (REAL _goal_term, float32)"))
        lev2 = out["probe282"]["forms"]["ccos"]["leverage_vs_cos"]
        out["factors"]["probe282_value_leverage"] = lev2["value_median_ratio_over_cells"]
        out["factors"]["probe282_decision_leverage"] = lev2["decision_median_ratio_of_box_spread"]
        out["exclusion_iter0_probe282"] = exclusion(out["probe282"], jerk, kap, settings)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["factors"], indent=1))
    for k in ("exclusion_iter0_devbox40", "exclusion_iter0_probe282"):
        if k in out:
            print(k)
            for r in out[k]:
                print(f"  {r['setting']:40s} goal(cv) med {r['goal_cv_median']:.4g} max {r['goal_cv_max']:.4g} "
                      f"excluded@median {r['excluded_at_median_window_pct']:.2f}% @max {r['excluded_at_max_window_pct']:.2f}%")
    for src in ("devbox40", "probe282"):
        if src in out:
            print(src)
            for f, v in out[src]["forms"].items():
                print(f"  {f:6s} med {v['median_over_cells']:.4g}  goal(cv) med {v['goal_cv']['median']:.4g} "
                      f"(exactly 1.0 on {v['goal_cv']['n_exactly_1']}/{out[src]['n_windows']})  "
                      f"box spread {v['spread_over_box_median']:.4g}  L/R rel {v['LR_gap_rel_to_term_median_pct']:.2f}%")


if __name__ == "__main__":
    main()
