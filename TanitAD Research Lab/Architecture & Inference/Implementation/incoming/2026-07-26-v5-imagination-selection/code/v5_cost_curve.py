#!/usr/bin/env python3
"""E-V5-3 — the COST-vs-QUALITY curve for imagination-scored selection.

Pre-registration: ``V5_IMAGINATION_SELECTION.md`` §0 (E-V5-3 block).

⚠️ BOOST_PROGRAM §7.3 IS BINDING AND IS DISCHARGED HERE, BEFORE ANY NUMBER.
H2's efficiency claim ("selective camera activation saves ~85 %") was true,
measured and INFORMATION-FREE: never escalating saves 85.7 %, a perfect oracle
85.6 %, the operating point 84.8 % — the entire span between a useless gate and a
perfect one was 0.1 pp.  A benefit metric that cannot separate a good design from
a useless one carries no evidence.  So, committed in advance:

  WHAT WOULD BE DISAPPOINTING
  ---------------------------
  D1  **A FLAT QUALITY AXIS.**  If ade(n, k) varies across the whole (n, k) grid
      by LESS than the paired CI half-width of the A1-vs-A0 comparison, the curve
      has no dynamic range: rolling 20 steps is then indistinguishable from
      rolling 2, and the imagination is not doing work beyond its first step.
      In that case I report "no admissible efficiency axis", NOT a compute saving.
  D2  **QUALITY BEST AT n = 2.**  If restricting to the top-2 candidates by base
      score is as good as all 256, the fan's remaining 254 candidates are
      decoration and any "efficiency" is trivially maximal and meaningless.
  D3  **HIERARCHY BUYS NOTHING.**  If quality at q candidates chosen by the
      tactical commit equals quality at q candidates chosen by base score alone,
      the hierarchy is not the lever — candidate COUNT is.

  The informative axis, constructed rather than assumed: **quality at a FIXED
  imagination budget** (predictor-steps per decision), where the budget can be
  spent on breadth (n candidates) or depth (k steps) or bought back by hierarchy.

Runs entirely OFF-GPU from ``v5_*_windows.pt`` — the k-step roll-out is a strict
causal PREFIX of the 20-step one, so no re-rolling is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _cluster_boot(v, ep, n_boot=2000, seed=20260726):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    by = [np.where(ep == e)[0] for e in eps]
    d = []
    for _ in range(n_boot):
        p = rng.integers(0, len(eps), len(eps))
        s = np.concatenate([by[j] for j in p])
        d.append(float(v[s].mean()))
    return np.asarray(d)


def paired_ci(a, b, ep):
    d = _cluster_boot(a - b, ep)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float((a - b).mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "ci95": round(float((hi - lo) / 2), 4),
            "separated": bool(lo > 0 or hi < 0),
            "estimator": "paired_episode_cluster_bootstrap",
            "_orientation": "a - b; NEGATIVE = a is BETTER"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/workspace/_v5/v5_v4_windows.pt")
    ap.add_argument("--result", default="/workspace/_v5/v5_v4.json")
    ap.add_argument("--out", default="/workspace/_v5/v5_cost_curve.json")
    a = ap.parse_args()

    D = torch.load(a.dump, map_location="cpu", weights_only=False)
    res = json.loads(Path(a.result).read_text())
    tim = res.get("_timing", {})
    fan = D["fan"].float()                       # [W, N, 20, 2]
    imag = D["imag"].float()
    err4 = D["fan_err4"].float()                 # [W, N]   4-wp error per candidate
    ep = D["ep"].numpy()
    W, N, K, _ = fan.shape

    R: dict = {
        "_experiment": "E-V5-3 -- cost-vs-quality curve for imagination-scored "
                       "selection over the frozen v4 fan",
        "_evidence_class": "MEASURED (ours)",
        "_disappointing_committed_in_advance": {
            "D1_flat_quality_axis": "ade(n,k) span < the paired CI half-width of "
                                    "A1-vs-A0 => NO admissible efficiency axis; "
                                    "report that, not a compute saving.",
            "D2_best_at_n2": "top-2 by base score as good as all 256 => the fan's "
                             "other 254 candidates are decoration.",
            "D3_hierarchy_buys_nothing": "tactical-commit top-q == base-score "
                                         "top-q => candidate COUNT is the lever, "
                                         "not hierarchy.",
            "_source": "BOOST_PROGRAM 7.3 -- H2's 85% saving was information-free "
                       "because never-escalate saved 85.7% and a perfect oracle "
                       "85.6%. State the disappointing value BEFORE quoting.",
        },
        "_grid": {"W": W, "N": N, "K": K},
        "_timing_measured": tim,
    }

    # ---- the quality surface: ade(n, k) -----------------------------------
    # ⛔ LABEL CORRECTED 2026-07-27 (E-H1 section 9.2/9.5, verified 881/881 rows).
    # This was called `base_rank` and documented as "the top-n candidates by the
    # AS-TRAINED base ranking". IT IS NOT A RANKING. It is exactly
    #
    #     nested_order[w] = [ the as-trained pick ] ++ [ anchors 0..N-1, pick removed ]
    #
    # i.e. column 0 is the deployed pick (so n=1 reproduces 0.8563 exactly) and
    # columns 1.. are ANCHOR INDEX ORDER, which carries no score information.
    # The E-V5-3 CONCLUSIONS survive -- "letting the imagination rule consider
    # more candidates makes it worse" holds for ANY nested family of candidate
    # sets, and index order is still a nested family -- but the LABEL does not,
    # and a later stream nearly published a false mechanism off the old name.
    # Root-cause class: A TENSOR'S SEMANTICS TAKEN FROM ITS NAME RATHER THAN
    # FROM ITS CONSTRUCTION SITE. If you want a real score ranking here, dump
    # `sel_score.argsort(descending=True)` -- it is one key.
    sel0 = D["ref_sel_idx"]
    order = torch.arange(N).repeat(W, 1)
    order = torch.cat([sel0[:, None], order], dim=1)
    # stable-unique per row, keeping first occurrence
    keep = torch.ones(W, N + 1, dtype=torch.bool)
    keep.scatter_(1, (sel0 + 1)[:, None], False)
    keep[:, 0] = True
    nested_order = order[keep].reshape(W, N)
    base_rank = nested_order      # legacy alias: the dumped key name is kept so
                                  # already-staged .pt files stay readable

    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ks = [1, 2, 4, 6, 8, 10, 14, 20]
    surface: dict = {}
    span_lo, span_hi = 1e9, -1e9
    for k in ks:
        cost_k = (imag[:, :, :k] - fan[:, :, :k]).norm(dim=-1).mean(dim=-1)
        row = {}
        for n in ns:
            cand = nested_order[:, :n]     # NOT "top-n by score" — see above
            c = cost_k.gather(1, cand)
            pick = cand.gather(1, c.argmin(dim=1, keepdim=True)).squeeze(1)
            ade = err4.gather(1, pick[:, None]).squeeze(1).numpy()
            row[f"n{n}"] = round(float(ade.mean()), 4)
            span_lo, span_hi = min(span_lo, ade.mean()), max(span_hi, ade.mean())
        surface[f"k{k}"] = row
    R["quality_surface_ade_0_2s"] = surface
    R["quality_span"] = {"min": round(float(span_lo), 4),
                         "max": round(float(span_hi), 4),
                         "span": round(float(span_hi - span_lo), 4)}

    # ---- D1: is the axis flat relative to the decision-grade interval? -----
    a1 = D["ade_by_arm"]["A1_imag_consistency"].numpy()
    a0 = D["ade_by_arm"]["A0_as_trained"].numpy()
    pc = paired_ci(a1, a0, ep)
    R["D1_flatness_test"] = {
        "quality_span_over_grid": R["quality_span"]["span"],
        "paired_ci95_halfwidth_A1_vs_A0": pc["ci95"],
        "axis_has_dynamic_range": bool(
            R["quality_span"]["span"] > pc["ci95"]),
        "paired_A1_minus_A0": pc,
    }

    # ---- D2: is n=2 already as good as n=256? -----------------------------
    k20 = surface["k20"]
    R["D2_breadth_test"] = {
        "ade_at_n2_k20": k20["n2"], "ade_at_n256_k20": k20["n256"],
        "breadth_buys_m": round(k20["n2"] - k20["n256"], 4),
        "disappointing_if": "breadth_buys_m ~ 0",
    }

    # ---- the COST axis, in real units -------------------------------------
    # imagination cost = ceil(n / batch) * k predictor steps.  At the measured
    # batch-2048 throughput one predictor step serves 2048 candidate-rolls, so a
    # single decision's n<=256 candidates fit in ONE batched step per k.
    st = tim.get("predictor_step_ms_batch256")
    enc = tim.get("encode_window_ms_batch1")
    if st and enc:
        cost_rows = {}
        for k in ks:
            cost_rows[f"k{k}"] = {
                "predictor_steps_per_decision": k,
                "ms_at_batch256": round(st * k, 3),
                "x_full_camera_pass": round(st * k / enc, 3),
            }
        R["cost_axis"] = {
            "predictor_step_ms_batch256": st,
            "encode_window_ms_batch1": enc,
            "rows": cost_rows,
            "_read": "a 256-candidate fan fits in ONE batched predictor call per "
                     "imagination step, so the cost of scoring the WHOLE fan k "
                     "steps deep is k predictor steps -- NOT 256*k.",
        }

    # ---- the informative axis: quality at a FIXED budget -------------------
    # budget = n_candidates * k  (candidate-steps).  Spend it on breadth or depth.
    budget: dict = {}
    for B in (64, 128, 256, 512, 1024, 2048, 5120):
        best = None
        for k in ks:
            for n in ns:
                if n * k > B:
                    continue
                v = surface[f"k{k}"][f"n{n}"]
                if best is None or v < best[0]:
                    best = (v, n, k)
        if best:
            budget[f"budget_{B}"] = {"best_ade": best[0], "n": best[1],
                                     "k": best[2]}
    R["quality_at_fixed_budget"] = {
        "rows": budget,
        "_read": "budget = n_candidates x k = candidate-steps per decision. THE "
                 "informative axis: whether the optimum moves with the budget. If "
                 "the same (n, k) wins at every budget there is no trade-off to "
                 "design.",
    }

    # ---- D3: hierarchy vs plain breadth, at matched candidate count --------
    hier = Path(str(a.out).replace("cost_curve", "hier")).with_suffix(".json")
    R["D3_hierarchy_note"] = (
        "the matched-count comparison (tactical-commit top-q vs base-score top-q) "
        "is computed in E-V5-2 (v5_hier.json, arms H_graft_q* / H_imag_q*); this "
        f"file supplies the base-score top-q leg. hier artifact: {hier}")

    # ---- the REPO-SIZED reduced dump --------------------------------------
    # The full dump carries fan/imag/ctrv ([W,256,20,2] each, ~150 MB total) and
    # stays pod-side. Everything a reader needs to RECOMPUTE any bar in this
    # stream with no GPU is [W,256] matrices: the per-candidate 4-wp error and
    # the per-candidate cost of every scoring rule, incl. one per k. ~7 MB.
    red = {
        "ep": D["ep"], "t": D["t"], "v0": D["v0"],
        "canary_err": D["canary_err"], "ref_sel_idx": D["ref_sel_idx"],
        "fan_err4": err4,
        "costs": {k: v for k, v in D["costs"].items()},
        "cost_A1_by_k": {
            f"k{k}": (imag[:, :, :k] - fan[:, :, :k]).norm(dim=-1).mean(dim=-1)
            for k in ks},
        "base_rank": nested_order,          # legacy key name — see _base_rank_IS
        "nested_order": nested_order,        # the correct name
        "_base_rank_IS": "[the as-trained pick] ++ [anchors 0..N-1 in INDEX "
                         "order, pick removed]. NOT a score ranking — verified "
                         "881/881 rows, E-H1 section 9.2. Column 0 reproduces "
                         "F_flat = 0.8563 exactly; columns 1.. carry NO score "
                         "information. Use `sel_score.argsort(descending=True)` "
                         "if you need a real ranking.",
        "picks": D["picks"], "ade_by_arm": D["ade_by_arm"],
        "_note": "REDUCED dump for the repo. fan_err4 [W,256] is the 4-wp error "
                 "of EVERY candidate, so ANY selection rule's ade_0_2s is "
                 "fan_err4.gather(1, pick).mean(). cost_A1_by_k gives the whole "
                 "E-V5-3 depth axis. Full fan/imag/ctrv tensors stay pod-side "
                 "(~150 MB) at /workspace/_v5/.",
    }
    torch.save(red, str(Path(a.out).with_name(
        Path(a.dump).stem + "_reduced.pt")))

    Path(a.out).write_text(json.dumps(R, indent=2))
    print(json.dumps({"quality_span": R["quality_span"],
                      "D1": R["D1_flatness_test"],
                      "D2": R["D2_breadth_test"],
                      "budget": budget}, indent=2))


if __name__ == "__main__":
    main()
