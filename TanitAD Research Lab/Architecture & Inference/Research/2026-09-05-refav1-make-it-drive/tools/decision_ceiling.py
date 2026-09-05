#!/usr/bin/env python
"""THE CEILING OF STEP 1: what can ANY decision rule buy on these logits?

⭐ THE QUESTION THIS ANSWERS IS THE PI'S. "Must we retrain?" A decision rule
only RE-THRESHOLDS a score the head already computed; it cannot add
information. So the honest bound on Step 1 is the **ROC of the head's own
turn-vs-hold score**, and that bound decides whether the cheap fix can work at
all or whether the head must be re-trained (Step 2).

Score: ``max(turn-ish logits) - LANE_KEEP logit`` — a monotone rule on this
scalar is exactly what a commit threshold or a single-slot bias implements, so
its ROC is the reachable frontier for that whole family.

Read it like this:
  * **AUC near 0.5** ⇒ the head cannot separate turns from holds at all, no
    threshold helps, and Step 2 (re-train the head) is REQUIRED.
  * **AUC well above 0.5 but recall poor at argmax** ⇒ the information IS there
    and merely mis-thresholded ⇒ Step 1 suffices, and the ROC says exactly how
    much recall is available at any false-turn budget.

Also reported: the **margin distribution on MISSED turns**. A missed turn that
lost by 0.2 logits is a calibration failure; one that lost by 8 is a
representation failure. That distinction is the difference between a flag and a
GPU-week, and it cannot be read off a recall number.

⛔ CONTROLS:
  * a SHUFFLED-LABEL control must read AUC ~0.5 — without it a high AUC could
    be an artefact of the scoring code;
  * `n` is printed for every cell;
  * the argmax operating point is located ON the ROC and must reproduce the
    banked 0.2045 / 0.0733, or the score is not the one the head is using.

Run: python decision_ceiling.py --logits <npz> --gt-kappa <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--gt-kappa", required=True)
    ap.add_argument("--kappa-turn", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
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
    lg, gt = lg[m], gt[m]
    y = (np.abs(gt) > a.kappa_turn).astype(int)          # 1 = GT turns
    n = len(y)

    # the monotone score a threshold rule acts on
    s = lg[:, CURV].max(1) - lg[:, LK]

    def roc(score, lab):
        o = np.argsort(-score)
        sl, ll = score[o], lab[o]
        tp = np.cumsum(ll); fp = np.cumsum(1 - ll)
        P, N = ll.sum(), (1 - ll).sum()
        tpr = tp / max(P, 1); fpr = fp / max(N, 1)
        auc = float(np.trapezoid(np.concatenate([[0], tpr]),
                                 np.concatenate([[0], fpr])))
        return sl, tpr, fpr, auc

    sl, tpr, fpr, auc = roc(s, y)

    # CONTROL: shuffled labels must read AUC ~0.5
    rng = np.random.default_rng(a.seed)
    aucs = [roc(s, rng.permutation(y))[3] for _ in range(200)]
    auc_shuf = float(np.mean(aucs)); auc_shuf_sd = float(np.std(aucs))

    # the ARGMAX operating point, and where it sits on the frontier
    dec = lg.argmax(-1)
    prop = np.isin(dec, CURV)
    op = {"turn_recall": float(prop[y == 1].mean()),
          "false_turn_on_straight": float(prop[y == 0].mean())}
    # recall reachable by a THRESHOLD at the same false-turn rate
    j = int(np.searchsorted(fpr, op["false_turn_on_straight"], side="right")) - 1
    op["best_threshold_recall_at_same_false_rate"] = (
        float(tpr[max(j, 0)]) if len(tpr) else None)

    # recall at a set of false-turn budgets — the frontier itself
    frontier = {}
    for b in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        j = int(np.searchsorted(fpr, b, side="right")) - 1
        frontier[f"false_turn<={b}"] = {
            "max_turn_recall": float(tpr[max(j, 0)]) if len(tpr) else None,
            "threshold": float(sl[max(j, 0)]) if len(sl) else None}

    # ⭐ MARGIN ON MISSED TURNS: calibration failure vs representation failure
    missed = (y == 1) & (~prop)
    mm = -s[missed]                      # how far LANE_KEEP won by, in logits
    margins = {"n_missed_turns": int(missed.sum())}
    if missed.sum():
        margins.update({
            "lane_keep_won_by_p10": float(np.percentile(mm, 10)),
            "lane_keep_won_by_p50": float(np.percentile(mm, 50)),
            "lane_keep_won_by_p90": float(np.percentile(mm, 90)),
            "frac_missed_within_1_logit": float((mm <= 1.0).mean()),
            "frac_missed_within_2_logits": float((mm <= 2.0).mean()),
            "frac_missed_beyond_4_logits": float((mm > 4.0).mean())})
    # ⛔ THE CONTROL THAT MAKES "every miss is a NEAR miss" MEAN ANYTHING.
    # If LANE_KEEP also wins by ~the same margin on CORRECTLY-held windows,
    # then the score is nearly constant, the margins say nothing about turns,
    # and a uniform threshold shift would flip holds and turns together. The
    # two distributions must SEPARATE, and by how much is the whole story.
    held = (y == 0) & (~prop)
    hh = -s[held]
    if held.sum():
        margins["CONTROL_correctly_held"] = {
            "n": int(held.sum()),
            "lane_keep_won_by_p10": float(np.percentile(hh, 10)),
            "lane_keep_won_by_p50": float(np.percentile(hh, 50)),
            "lane_keep_won_by_p90": float(np.percentile(hh, 90)),
            "median_gap_vs_missed_turns": float(np.median(hh) - np.median(mm))
            if missed.sum() else None,
            "note": "must be LARGER than the missed-turn margin, else the "
                    "score is flat and no threshold separates them"}

    R = {"logits": a.logits, "n_windows_with_gt": int(n),
         "n_gt_turn": int(y.sum()), "n_gt_straight": int((1 - y).sum()),
         "score": "max(TURN_/NUDGE_/LANE_CHANGE_ logits) - LANE_KEEP logit",
         "AUC": auc,
         "CONTROL_AUC_shuffled_labels": {"mean": auc_shuf, "sd": auc_shuf_sd,
                                         "expect": 0.5, "n_shuffles": 200},
         "argmax_operating_point": op, "frontier": frontier,
         "missed_turn_margins": margins}
    ctrl_ok = abs(auc_shuf - 0.5) < 0.05
    R["control_passes"] = bool(ctrl_ok)

    # ⚠️ THE VERDICT USES THE MARGIN GAP, NOT ONLY THE AUC — because "every
    # missed turn is within 2 logits" is TRUE AND MISLEADING on its own. If
    # LANE_KEEP also wins by ~the same margin where it is RIGHT, the score is
    # nearly flat and a uniform shift flips holds and turns together. The gap
    # between the two medians is what says whether a threshold can separate
    # them cheaply, and it is the number to quote.
    gap = (margins.get("CONTROL_correctly_held") or {}).get(
        "median_gap_vs_missed_turns")
    R["separation_summary"] = {
        "auc": auc, "median_margin_gap_logits": gap,
        "recall_cost_per_true_turn": None}
    f10 = frontier["false_turn<=0.1"]["max_turn_recall"]
    f20 = frontier["false_turn<=0.2"]["max_turn_recall"]
    if f20 is not None and f10 is not None and (f20 - f10) > 0:
        R["separation_summary"]["recall_cost_per_true_turn"] = float(
            0.10 / (f20 - f10))          # extra false-turn rate per extra recall

    if auc <= 0.6:
        v = ("STEP 1 CANNOT FIX THIS: the head barely separates turns from "
             "holds, so no threshold rule recovers recall — re-train the head")
    elif gap is not None and gap < 0.5:
        v = ("STEP 1 IS REAL BUT WEAK AND EXPENSIVE: the head DOES separate "
             f"(AUC {auc:.3f} vs shuffled {auc_shuf:.3f}), but the margin "
             f"distributions overlap heavily — LANE_KEEP wins by a median "
             f"{gap:.3f} logits MORE where it is right than where it is wrong. "
             "A uniform threshold shift therefore flips holds and turns nearly "
             "together, and recall is bought at a steep false-turn price. "
             "⇒ Step 1 helps but is UNLIKELY TO BE SUFFICIENT; Step 2 "
             "(fine-tune the goal head to SHARPEN this separation) is "
             "indicated — and note the separation, not the capability, is what "
             "is missing: the information is present.")
    else:
        v = ("STEP 1 SUFFICES: the head separates turns from holds and argmax "
             "is merely a bad threshold on a well-separated score")
    R["verdict"] = v
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps(R, indent=1))
    print("\nCONTROL", "PASS" if ctrl_ok else "FAIL",
          f"(shuffled AUC {auc_shuf:.4f} +/- {auc_shuf_sd:.4f})")
    print("VERDICT:", R["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
