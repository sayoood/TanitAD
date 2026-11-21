"""How close is REFe's pick to flipping? The admissible cross-GPU tolerance for the parity gate is set from THIS.

Reads the dev box's own E-6 artifacts for snapshot 016 on sub200 (read-only, CPU, ~1 MB):
  proptable/sub200_ep016/proposals.npz  -- the 64 proposals + the scorer's raw six logits per token (the landed run)
  proptable/sub200_ep016/table.npz      -- the NAVSIM v1 PDMS of EVERY proposal (64 harness runs, W3's harness)
and recomputes the planner's selection with the SAME formula as refe/planner.py `aggregate` (rule navsim_v1,
float32 torch-free numpy re-implementation; checked against the stored pick, which must agree on 200/200).

Reports, per token: the aggregate margin between the best and the runner-up proposal, and the PDMS change the
sub200 mean would see if the pick flipped to the runner-up (the worst case a cross-GPU flip can cause).
Writes pick_margin_ep016.json beside this file.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep016"
LANDED_CSV = "D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep016/refe_sub200_ep016.csv"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    p = np.load(f"{D}/proposals.npz")
    t = np.load(f"{D}/table.npz")
    assert list(p["token"]) == list(t["token"])
    lg = p["logits"].astype(np.float32)                       # [N, M, 6] NC DAC EP TTC C DDC
    s = sig(lg.astype(np.float32))
    agg = s[..., 0] * s[..., 1] * (5 * s[..., 2] + 5 * s[..., 3] + 2 * s[..., 4]) / np.float32(12.0)
    pick = agg.argmax(-1)
    agree = int((pick == p["pick"]).sum())
    srt = np.sort(agg, axis=-1)
    top1, top2 = srt[:, -1], srt[:, -2]
    margin = (top1 - top2).astype(np.float64)
    rel = margin / np.maximum(top1.astype(np.float64), 1e-12)
    runner = np.argsort(agg, axis=-1)[:, -2]
    n = len(pick)
    pdms = t["pdms"]                                          # [N, M] per-proposal PDMS (0..1)
    d_flip = (pdms[np.arange(n), runner] - pdms[np.arange(n), pick]) / n * 100.0   # PDMS points on the mean
    # the landed per-token score at the pick reproduces the table (G2), so the table is the right reference
    mean_at_pick = float(pdms[np.arange(n), pick].mean() * 100.0)
    qs = [0.0, 0.01, 0.05, 0.1, 0.5]
    rep = {
        "snapshot": "snap_epoch016.pt (sub200, W3 A1_sub200_tokens.json, 200 tokens / 93 logs)",
        "rule": str(p["rule"]), "repair_last_heading": bool(p["repair_last_heading"]),
        "pick_recomputed_agrees_with_stored": f"{agree}/{n}",
        "pdms_mean_at_pick_x100": round(mean_at_pick, 4),
        "aggregate_margin_top1_minus_top2": {
            "min": float(margin.min()), "quantiles": {str(q): float(np.quantile(margin, q)) for q in qs},
            "n_below_1e-6": int((margin < 1e-6).sum()), "n_below_1e-5": int((margin < 1e-5).sum()),
            "n_below_1e-4": int((margin < 1e-4).sum()), "n_below_1e-3": int((margin < 1e-3).sum()),
            "n_exact_ties": int((margin == 0).sum())},
        "relative_margin_min": float(rel.min()),
        "flip_to_runner_up_pdms_delta_on_mean": {
            "max_abs_single_token": float(np.abs(d_flip).max()),
            "n_tokens_where_flip_changes_score": int((np.abs(d_flip) > 0).sum()),
            "worst5": sorted(((float(m), float(d)) for m, d in zip(margin, d_flip)),
                             key=lambda x: -abs(x[1]))[:5]},
        "closest5_margin_and_flip_delta": sorted(((float(m), float(d)) for m, d in zip(margin, d_flip)))[:5],
        "what": "aggregate = sig(NC)*sig(DAC)*(5 sig(EP)+5 sig(TTC)+2 sig(C))/12 over the stored float32 logits "
                "(planner.py rule navsim_v1); margin = best - runner-up per token; flip delta = the change in the "
                "200-token PDMS mean (points) if that token's pick moved to its runner-up, read from the E-6 table",
    }
    json.dump(rep, open(os.path.join(HERE, "pick_margin_ep016.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0 if agree == n else 1


if __name__ == "__main__":
    sys.exit(main())
