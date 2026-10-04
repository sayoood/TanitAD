"""G0 DETECTION-cell diagnosis (zero GPU): is the in-run-vs-replay deviation on low-support detection cells
the SAME KIND of move that a NUMERICS-ONLY lever produces on the dev box itself?

For every DETECTION term of a G0 artifact (classes by `g0_refcv7.term_class`, support by
`g0_refcv7.detection_support`, i.e. the SAME functions the gate uses) it tabulates
  * the in-run value, the replay (mean / sd over the inference seeds; sd 0 = seed-invariant),
  * the numerics-only arm `fp32_s0` (SPEC A6: same weights / flags / windows / code; trunk fp32 + NCHW,
    cuDNN TF32 off) and the three mutations at seed 0,
  * for a single-GT cell (support n = 1) the implied RANK of the true positive, rank = 1 / AP (all-point AP of a
    score-ordered list with one GT, `box3d_head.ap_from_rows`): the number of class-c detections the replay and
    the in-run row put above it -- the count of discrete box decisions the gap requires at minimum.
and compares, per support stratum, how many cells each arm moves by more than the registered DETECTION
tolerance (abs 0.02): if the numerics-only arm moves as many low-support cells as the in-run row does, while
both leave the n >= 30 cells inside tolerance, the in-run gap is at the instrument's own numerics floor.

    python g0_det_cells.py --g0 raw/step50400/g0.json --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
        --out raw/g0diag_step50400/det_cells.json [--fp32-from <supp g0 json>]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402

TOL = 0.02


def _num(v):
    return None if G._isnull(v) else float(v)


def implied_rank(ap):
    """1/AP when AP is (to 5 dp) the reciprocal of an integer, else None."""
    if ap is None or ap <= 0:
        return None
    r = round(1.0 / ap)
    return int(r) if abs(1.0 / r - ap) <= 6e-6 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--fp32-from", default=None, help="a supplement G0 json carrying a6.fp32_s0 (step 30,000)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g0 = json.load(open(a.g0, encoding="utf-8"))
    step = int(g0["step"])
    ev = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in ev if r.get("step") == step and "eval_loss" in r]
    assert len(ev) == 1, f"{len(ev)} in-run rows at step {step}"
    inrun = ev[0]
    seeds = sorted(g0["by_seed"], key=int)
    s0 = g0["by_seed"]["0"]["row"]
    src = g0
    if a.fp32_from:
        src = json.load(open(a.fp32_from, encoding="utf-8"))
        # the supplement's seed 0 must be G0's seed 0 (else its fp32 arm is a floor for a different replay)
        ss0 = src["by_seed"]["0"]["row"]
        bad = [k for k in s0 if k.startswith("eval_") and k in ss0 and not (
            (G._isnull(s0[k]) and G._isnull(ss0[k])) or s0[k] == ss0[k]
            or abs(float(s0[k]) - float(ss0[k])) <= 1e-5 * max(1.0, abs(float(s0[k]))))]
        assert not bad, f"supplement seed 0 differs on {len(bad)} keys: {bad[:5]}"
    fp32 = ((src.get("a6") or {}).get("fp32_s0") or {}).get("row") or {}
    muts = {m: (v or {}).get("row") or {} for m, v in (g0.get("mutations") or {}).items()}
    eps0 = ((g0.get("diagnostic_arms") or {}).get("eps0") or {}).get("row") or {}
    cells = {}
    for k in sorted(inrun):
        if not k.startswith("eval_") or G.term_class(k) != "DETECTION":
            continue
        x = _num(inrun.get(k))
        vals = [_num(g0["by_seed"][s]["row"].get(k)) for s in seeds]
        if x is None or any(v is None for v in vals):
            continue
        sup = G.detection_support(k, inrun)
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        bounded01 = not (k.endswith(G.MODEL_DEP_COUNT_SUFFIX) or k.endswith(("_conf_ratio", "_centre_err_p50")))
        rec = {"support_n": sup, "bounded01": bounded01, "inrun": x, "replay_mean": mean, "replay_sd": sd,
               "fp32_s0": _num(fp32.get(k)), "eps0": _num(eps0.get(k)),
               **{f"{m}": _num(r.get(k)) for m, r in muts.items()}}
        rec["dev_inrun"] = abs(x - mean)
        rec["dev_fp32"] = None if rec["fp32_s0"] is None else abs(rec["fp32_s0"] - mean)
        if sup is not None and sup == 1 and bounded01:
            rec["rank_inrun"] = implied_rank(x)
            rec["rank_replay"] = implied_rank(mean)
            rec["rank_fp32"] = implied_rank(rec["fp32_s0"])
        cells[k] = rec

    def stratum(r):
        if r["support_n"] is None:
            return "pooled"
        return "lowsupport_n<30" if r["support_n"] < G.A2_MIN_SUPPORT else "gating_n>=30"

    summary = {}
    for st in ("lowsupport_n<30", "gating_n>=30", "pooled"):
        rs = {k: r for k, r in cells.items() if stratum(r) == st and r["bounded01"]}
        if not rs:
            continue
        d_in = [r["dev_inrun"] for r in rs.values()]
        d_fp = [r["dev_fp32"] for r in rs.values() if r["dev_fp32"] is not None]
        summary[st] = {
            "n_cells": len(rs),
            "inrun_over_tol": sum(d > TOL for d in d_in),
            "fp32_over_tol": sum(d > TOL for d in d_fp),
            "inrun_nonzero": sum(d > 1e-5 for d in d_in),
            "fp32_nonzero": sum(d > 1e-5 for d in d_fp),
            "inrun_max": max(d_in), "fp32_max": max(d_fp) if d_fp else None,
            "inrun_median_nonzero": statistics.median([d for d in d_in if d > 1e-5] or [0.0]),
            "fp32_median_nonzero": statistics.median([d for d in d_fp if d > 1e-5] or [0.0]),
            "inrun_over_tol_terms": sorted(k for k, r in rs.items() if r["dev_inrun"] > TOL),
            "fp32_over_tol_terms": sorted(k for k, r in rs.items() if (r["dev_fp32"] or 0) > TOL),
            **{f"{m}_over_tol": sum(1 for r in rs.values() if r.get(m) is not None
                                    and abs(r[m] - r["replay_mean"]) > TOL) for m in muts},
        }
    # the failing cells, by name (the 13 at 50,400) with the numbers that decide them
    fails = {k: r for k, r in cells.items() if r["bounded01"] and r["dev_inrun"] > TOL}
    out = {"tool": "g0_det_cells.py", "g0": a.g0, "step": step, "ckpt_md5": g0.get("ckpt_md5"),
           "fp32_source": a.fp32_from or a.g0, "tol_abs": TOL, "a2_min_support": G.A2_MIN_SUPPORT,
           "seeds": len(seeds), "summary": summary, "failing_cells": fails, "cells": cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1)[:5000])
    print("failing cells (|in-run - replay| > 0.02):")
    for k, r in fails.items():
        print(f"  {k}: n={r['support_n']} inrun={r['inrun']} replay={r['replay_mean']:.5f} sd={r['replay_sd']:.2g} "
              f"fp32={r['fp32_s0']} m1={r.get('m1')} eps0={r['eps0']} "
              f"rank in/rep/fp32={r.get('rank_inrun')}/{r.get('rank_replay')}/{r.get('rank_fp32')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
