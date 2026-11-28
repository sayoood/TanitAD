#!/usr/bin/env python
"""P1 — THE ANATOMY OF THE 0.297-LOGIT MARGIN: calibration, or separability?

⭐ WHY THIS EXISTS AND WHAT IT ADDS TO `decision_ceiling.py`. That instrument
established the ROC (AUC 0.712 vs shuffled 0.506) and the headline
`median_margin_gap_logits = 0.297`. It answered "is the information there?"
(yes) and "does argmax sit badly on the ROC?" (no). It did NOT answer the
question that decides the training objective:

    is the gap SMALL, or is the gap merely EXPRESSED IN A UNIT THAT HAS NO SCALE?

⛔⛔ THE CONTROL THAT SETTLES IT, AND IT INVALIDATES A PROPOSED OBJECTIVE.
`lat_head` is `LayerNorm -> Linear`. Multiply that Linear's weight AND bias by
`c > 0` and EVERY logit scales by `c`, so:
  * every argmax is bit-identical,
  * every ROC point, AUC and balanced accuracy is bit-identical,
  * and `median_margin_gap_logits` becomes `c * 0.297`.
⇒ **A RAW-LOGIT MARGIN IS NOT A PROPERTY OF THE DECISION — IT IS A PROPERTY OF
THE WEIGHT NORM.** Any training objective phrased as "widen the margin gap in
logits" is therefore satisfiable by scaling the head and changing NOTHING, which
is the same failure family as CLAUDE.md's probe traps: a true number whose
implication is wrong. This script MEASURES that with a `logit_scale` control
that must reproduce the gap x c and the AUC exactly.

⇒ The admissible, scale-free quantities are reported instead:
  * AUC (rank-only),
  * Cohen's d = gap / pooled SD  (the gap in units of the score's own spread),
  * the OVERLAP COEFFICIENT of the two class-conditional score densities,
  * max balanced accuracy over all thresholds  (the Bayes rate of ANY monotone
    rule on this score).

CALIBRATION vs SEPARABILITY, decided by a stated rule rather than a vibe:
  * CALIBRATION  <=> the score separates well but the operating threshold is
    misplaced: `bal_acc_at_best_threshold` is high AND much better than at the
    argmax threshold.
  * SEPARABILITY <=> even the best threshold is poor: the densities overlap, so
    no scalar bias can buy recall without buying false turns at a steep rate.
The verdict prints the numbers that decided it, both ways.

THE OPERATING CURVE. The lever `--lat-logit-bias` in its `commit b` form is a
single negative entry on LANE_KEEP, which is EXACTLY a threshold on
`s = max(curv logits) - LANE_KEEP`. This sweeps it densely and reports, per
setting, decode turn-recall and false-turn-on-straight.

⛔ OPERATING POINT, COMMITTED IN ADVANCE (before any number was read from this
script; see SPEC.md in this package):
  * PRIMARY   : max Youden's J = TPR - FPR  (equivalently max balanced accuracy).
    Justified by symmetry of harm — a missed turn departs the road, a false turn
    steers off a straight road; neither is the cheap error.
  * SECONDARY : max recall subject to false-turn <= 0.10 (a budget chosen as
    ~argmax's own 0.073 plus headroom), reported for the case where gate 2's
    refusal makes decode false turns cheaper than decode misses.

⛔ CONTROLS (each must read a value fixed in advance or the run is void):
  1. b=0 reproduces the banked argmax decode: recall 0.2045, false 0.0733.
  2. shuffled labels -> AUC ~ 0.5.
  3. `logit_scale=3` -> AUC and every decode IDENTICAL, raw gap x3. This is the
     control that voids the raw-logit-margin objective.
  4. `n` printed for every cell.

Run: python margin_anatomy.py --logits <npz> --gt-kappa <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def roc(score: np.ndarray, lab: np.ndarray):
    """Descending-score ROC. Returns (sorted score, tpr, fpr, auc)."""
    o = np.argsort(-score)
    sl, ll = score[o], lab[o]
    tp = np.cumsum(ll)
    fp = np.cumsum(1 - ll)
    P, N = ll.sum(), (1 - ll).sum()
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    auc = float(np.trapezoid(np.concatenate([[0], tpr]),
                             np.concatenate([[0], fpr])))
    return sl, tpr, fpr, auc


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    """Sum over bins of min(p_a, p_b) — 1.0 = identical, 0.0 = disjoint.

    Histogram-based on a COMMON support so the two densities are comparable.
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.minimum(pa, pb).sum())


def pct(x: np.ndarray, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--gt-kappa", required=True)
    ap.add_argument("--kappa-turn", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logit-scale", type=float, default=3.0,
                    help="CONTROL 3: scale factor whose decode must be identical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z = np.load(a.logits, allow_pickle=False)
    lg = Z["lat_logits"].astype(np.float64)
    names = [str(x) for x in Z["lat_names"]]
    LK = names.index("LANE_KEEP")
    CURV = [i for i, t in enumerate(names)
            if t.startswith(("TURN_", "NUDGE_", "LANE_CHANGE_"))]

    G = np.load(a.gt_kappa, allow_pickle=False)
    key = {(int(c), int(w)): i for i, (c, w) in
           enumerate(zip(Z["clip_index"], Z["ws"]))}
    gt = np.full(lg.shape[0], np.nan)
    for c, w, k in zip(G["clip_index"], G["ws"], G["gt_kappa"]):
        i = key.get((int(c), int(w)))
        if i is not None:
            gt[i] = k
    m = np.isfinite(gt)
    lg_all = lg
    lg, gt = lg[m], gt[m]
    eid = Z["eid"][m] if "eid" in Z.files else None
    y = (np.abs(gt) > a.kappa_turn).astype(int)
    n = len(y)

    s = lg[:, CURV].max(1) - lg[:, LK]          # the monotone decision score
    dec = lg.argmax(-1)
    prop = np.isin(dec, CURV)

    out: dict = {
        "panel": {"logits": a.logits, "gt_kappa": a.gt_kappa,
                  "n_windows_with_gt": int(n),
                  "n_gt_turn": int(y.sum()), "n_gt_straight": int((1 - y).sum()),
                  "n_windows_total_in_npz": int(lg_all.shape[0]),
                  "n_episodes": int(len(np.unique(eid))) if eid is not None else None,
                  "kappa_turn_threshold": a.kappa_turn,
                  "score": "max(TURN_/NUDGE_/LANE_CHANGE_ logits) - LANE_KEEP logit"},
    }

    # ---------------------------------------------------------------- #
    # 1. THE FULL MARGIN DISTRIBUTION, by GT class and by correctness
    # ---------------------------------------------------------------- #
    s_turn, s_str = s[y == 1], s[y == 0]
    missed = (y == 1) & (~prop)
    held = (y == 0) & (~prop)
    hit = (y == 1) & prop
    false = (y == 0) & prop

    pooled_sd = float(np.sqrt(((len(s_turn) - 1) * s_turn.var(ddof=1)
                               + (len(s_str) - 1) * s_str.var(ddof=1))
                              / max(len(s_turn) + len(s_str) - 2, 1)))
    cohen_d = float((s_turn.mean() - s_str.mean()) / pooled_sd) if pooled_sd else None

    out["margin_distribution"] = {
        "note": ("s > 0 means the head decodes a curvature-carrying token. "
                 "The two class-conditional distributions below are what any "
                 "threshold rule must separate."),
        "gt_turn": {"n": int(len(s_turn)), "mean": float(s_turn.mean()),
                    "sd": float(s_turn.std(ddof=1)), **pct(s_turn)},
        "gt_straight": {"n": int(len(s_str)), "mean": float(s_str.mean()),
                        "sd": float(s_str.std(ddof=1)), **pct(s_str)},
        "separation_scalefree": {
            "auc": None,                       # filled below
            "cohen_d": cohen_d,
            "pooled_sd_logits": pooled_sd,
            "overlap_coefficient": overlap_coefficient(s_turn, s_str),
            "mean_gap_logits": float(s_turn.mean() - s_str.mean()),
            "median_gap_logits": float(np.median(s_turn) - np.median(s_str)),
        },
        "by_outcome_lane_keep_won_by": {
            "missed_turns": {"n": int(missed.sum()), **pct(-s[missed])}
            if missed.sum() else {"n": 0},
            "correctly_held": {"n": int(held.sum()), **pct(-s[held])}
            if held.sum() else {"n": 0},
            "median_gap_logits": (float(np.median(-s[held]) - np.median(-s[missed]))
                                  if missed.sum() and held.sum() else None),
            "REPRODUCES_banked_0p297": None,
        },
        "by_outcome_curv_won_by": {
            "correct_turns": {"n": int(hit.sum()), **pct(s[hit])} if hit.sum() else {"n": 0},
            "false_turns": {"n": int(false.sum()), **pct(s[false])} if false.sum() else {"n": 0},
        },
    }
    g = out["margin_distribution"]["by_outcome_lane_keep_won_by"]["median_gap_logits"]
    out["margin_distribution"]["by_outcome_lane_keep_won_by"][
        "REPRODUCES_banked_0p297"] = (g is not None and abs(g - 0.2973119) < 1e-4)

    # where the 0.297 sits INSIDE the distribution it summarises
    if missed.sum() and held.sum():
        allneg = np.concatenate([-s[missed], -s[held]])
        out["margin_distribution"]["where_0p297_sits"] = {
            "gap_logits": g,
            "iqr_of_pooled_lane_keep_margins": float(
                np.percentile(allneg, 75) - np.percentile(allneg, 25)),
            "gap_as_frac_of_that_iqr": float(g / (np.percentile(allneg, 75)
                                                  - np.percentile(allneg, 25))),
            "gap_as_frac_of_pooled_sd": float(g / pooled_sd) if pooled_sd else None,
            "interpretation": ("a gap far SMALLER than the spread it must "
                               "separate means the two populations are "
                               "interleaved, not offset"),
        }

    # ---------------------------------------------------------------- #
    # 2. ROC + CONTROLS
    # ---------------------------------------------------------------- #
    sl, tpr, fpr, auc = roc(s, y)
    out["margin_distribution"]["separation_scalefree"]["auc"] = auc

    rng = np.random.default_rng(a.seed)
    aucs = [roc(s, rng.permutation(y))[3] for _ in range(200)]
    ctrl = {
        "CONTROL_2_shuffled_label_auc": {
            "mean": float(np.mean(aucs)), "sd": float(np.std(aucs)),
            "expect": 0.5, "n_shuffles": 200,
            "passes": bool(abs(float(np.mean(aucs)) - 0.5) < 0.05)},
    }

    # CONTROL 1 — argmax reproduces the banked operating point
    op_recall = float(prop[y == 1].mean())
    op_false = float(prop[y == 0].mean())
    ctrl["CONTROL_1_argmax_reproduces_banked"] = {
        "turn_recall": op_recall, "expect": 0.20454545,
        "false_turn_on_straight": op_false, "expect_false": 0.07333333,
        "passes": bool(abs(op_recall - 0.20454545) < 1e-6
                       and abs(op_false - 0.07333333) < 1e-6)}

    # ⛔ CONTROL 3 — the one that voids the raw-logit-margin objective
    c = a.logit_scale
    lg_s = lg * c
    s_s = lg_s[:, CURV].max(1) - lg_s[:, LK]
    dec_s = lg_s.argmax(-1)
    _, _, _, auc_s = roc(s_s, y)
    miss_s = (y == 1) & (~np.isin(dec_s, CURV))
    held_s = (y == 0) & (~np.isin(dec_s, CURV))
    gap_s = (float(np.median(-s_s[held_s]) - np.median(-s_s[miss_s]))
             if miss_s.sum() and held_s.sum() else None)
    ctrl["CONTROL_3_logit_scale_voids_raw_margin_objective"] = {
        "scale_c": c,
        "decode_bit_identical": bool(np.array_equal(dec, dec_s)),
        "auc_unchanged": bool(abs(auc_s - auc) < 1e-12),
        "auc_scaled": auc_s,
        "raw_median_gap_logits_scaled": gap_s,
        "raw_median_gap_logits_base": g,
        "ratio_should_equal_c": (float(gap_s / g) if g else None),
        "cohen_d_unchanged": bool(
            abs(float((s_s[y == 1].mean() - s_s[y == 0].mean())
                      / np.sqrt(((len(s_turn) - 1) * s_s[y == 1].var(ddof=1)
                                 + (len(s_str) - 1) * s_s[y == 0].var(ddof=1))
                                / max(n - 2, 1))) - cohen_d) < 1e-9),
        "VERDICT": ("a raw-logit margin gap is NOT a property of the decision; "
                    "scaling the head's Linear by c multiplies it by c while "
                    "every decode, AUC and Cohen's d is unchanged. An objective "
                    "phrased as 'widen the margin gap in logits' is therefore "
                    "GAMEABLE BY WEIGHT NORM and inadmissible as stated."),
    }
    out["controls"] = ctrl

    # ---------------------------------------------------------------- #
    # 3. CALIBRATION vs SEPARABILITY — decided by a stated rule
    # ---------------------------------------------------------------- #
    bal = 0.5 * (tpr + (1.0 - fpr))
    j_star = int(np.argmax(bal))
    youden = tpr - fpr
    jy = int(np.argmax(youden))
    bal_argmax = 0.5 * (op_recall + (1.0 - op_false))

    out["calibration_vs_separability"] = {
        "bal_acc_at_argmax_threshold": float(bal_argmax),
        "bal_acc_at_best_threshold": float(bal[j_star]),
        "best_threshold_on_s": float(sl[j_star]),
        "gain_from_rethresholding": float(bal[j_star] - bal_argmax),
        "chance_bal_acc": 0.5,
        "auc": auc,
        "cohen_d": cohen_d,
        "overlap_coefficient": out["margin_distribution"][
            "separation_scalefree"]["overlap_coefficient"],
        "recall_cost_per_true_turn": None,     # filled from the sweep
        "RULE": ("CALIBRATION if bal_acc_at_best_threshold >= 0.80 (the score "
                 "separates; only the threshold is wrong). SEPARABILITY if "
                 "bal_acc_at_best_threshold < 0.80 (no threshold on this score "
                 "operates acceptably; the score itself must be re-fit). The "
                 "0.80 bar and this rule are committed in SPEC.md before the "
                 "numbers were read."),
        "VERDICT": None,
    }

    # ---------------------------------------------------------------- #
    # 4. THE lat_logit_bias OPERATING CURVE (the `commit b` family)
    # ---------------------------------------------------------------- #
    grid = [round(x, 3) for x in np.arange(0.0, 4.001, 0.05)]
    curve = []
    for b in grid:
        bias = np.zeros(lg.shape[1])
        bias[LK] = -b
        d2 = (lg + bias[None, :]).argmax(-1)
        p2 = np.isin(d2, CURV)
        r = float(p2[y == 1].mean())
        f = float(p2[y == 0].mean())
        curve.append({
            "commit_b": b,
            "turn_recall_decode": r,
            "false_turn_on_straight_decode": f,
            "balanced_acc": float(0.5 * (r + (1.0 - f))),
            "youden_j": float(r - f),
            "frac_curvature_carrying": float(p2.mean()),
            "n_curv_windows": int(p2.sum()),
        })
    out["operating_curve_commit_b"] = {
        "note": ("`commit b` = lat_logit_bias with a single entry -b on "
                 "LANE_KEEP. This is the DECODE curve (gate 1) at zero GPU; "
                 "the EXECUTED-turn rate additionally passes gate 2 and is "
                 "measured on the GPU in P4."),
        "grid": curve,
    }

    # committed operating points
    best_j = max(curve, key=lambda r: (r["youden_j"], -r["commit_b"]))
    cons = [r for r in curve if r["false_turn_on_straight_decode"] <= 0.10]
    best_c = max(cons, key=lambda r: (r["turn_recall_decode"], -r["commit_b"])) if cons else None
    out["committed_operating_points"] = {
        "PRIMARY_max_youden_j": best_j,
        "SECONDARY_max_recall_false_turn_le_0p10": best_c,
        "CONTROL_argmax_b0": curve[0],
        "criterion_committed_in": "SPEC.md (this package), before numbers read",
    }

    # cost of recall in false turns, over the whole curve
    rs = np.array([r["turn_recall_decode"] for r in curve])
    fs = np.array([r["false_turn_on_straight_decode"] for r in curve])
    if rs.max() > rs.min():
        slope = float(np.polyfit(rs, fs, 1)[0])
    else:
        slope = None
    out["calibration_vs_separability"]["recall_cost_per_true_turn"] = slope

    ba = out["calibration_vs_separability"]["bal_acc_at_best_threshold"]
    out["calibration_vs_separability"]["VERDICT"] = (
        "CALIBRATION" if ba >= 0.80 else "SEPARABILITY")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # -------- console summary (ascii only; cp1252 console) ------------- #
    print(f"n={n}  gt_turn={int(y.sum())}  gt_straight={int((1-y).sum())}")
    print(f"AUC={auc:.4f}  cohen_d={cohen_d:.4f}  "
          f"overlap={out['margin_distribution']['separation_scalefree']['overlap_coefficient']:.4f}")
    print(f"bal_acc argmax={bal_argmax:.4f}  best_threshold={bal[j_star]:.4f}  "
          f"-> {out['calibration_vs_separability']['VERDICT']}")
    print(f"raw median gap={g:.4f} logits  (= {out['margin_distribution']['where_0p297_sits']['gap_as_frac_of_pooled_sd']:.3f} pooled SD)")
    print("CONTROLS: " + json.dumps({k: (v.get("passes") if isinstance(v, dict) else v)
                                     for k, v in ctrl.items()}))
    print(f"C3 decode identical under x{c}: "
          f"{ctrl['CONTROL_3_logit_scale_voids_raw_margin_objective']['decode_bit_identical']}"
          f"  gap ratio={ctrl['CONTROL_3_logit_scale_voids_raw_margin_objective']['ratio_should_equal_c']}")
    print(f"PRIMARY  b={best_j['commit_b']}  recall={best_j['turn_recall_decode']:.4f} "
          f"false={best_j['false_turn_on_straight_decode']:.4f} J={best_j['youden_j']:.4f}")
    if best_c:
        print(f"SECONDARY b={best_c['commit_b']}  recall={best_c['turn_recall_decode']:.4f} "
              f"false={best_c['false_turn_on_straight_decode']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
