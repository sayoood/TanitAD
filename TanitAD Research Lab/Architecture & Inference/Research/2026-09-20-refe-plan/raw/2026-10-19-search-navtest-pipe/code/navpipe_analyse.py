#!/usr/bin/env python3
"""The READ analysis of SEARCH-NAVTEST-PIPE, on MERGED chunk outputs in the canonical (sorted-token) order.

    --rule c1   the SEARCH-P2-CONFIRM READ C1 analysis, `spc_analyse.analyse_c1` UNCHANGED (this is the FIXTURE route: it must reproduce Step B's banked P2HEAD read on carve B exactly)
    --rule c2   `analyse_stepB.analyse` UNCHANGED (the endpoint, the interval, the 3-seed spread, the four sub-score safety points, the INERT test and the families) plus the DECISION RULE TERMS of MM_RULING_SPC DQ3:
                lower CI > 0  AND  point >= bar_dpdms (+0.10 points, the minimum worthwhile effect)  AND  point > the 3-seed range  AND  each of dNC, dDAC, dTTC, dC point >= bar_safety (-0.25).
                The bars come from the SEALED C2 amendment's pins (`--bars-json`), never from a constant here.  The OUTCOME NAMES are not set here: the sealed amendment names them; this file reports the terms.
Both: the interval is the paired log-cluster bootstrap (B = 10,000, seed 20260927) = "another draw of LOGS" only; one training seed; STRATEGIC PAUSED.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEPB_CODE = os.environ.get("SPC_STEPB_CODE", "C:/Users/Admin/par18/l2stepb/code")
L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
SPC_CODE = os.environ.get("SPC_CODE", "C:/Users/Admin/par18/spc/code")
for _p in (HERE, STEPB_CODE, L1_CODE, SPC_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analyse_stepB as AB  # noqa: E402
import stepA as A  # noqa: E402

C2_TIER = ("T1 = NAVSIM v1.1 PDMS, pseudo-simulation of an open-loop plan against logged agents (the W3 harness, seam route; NOT closed loop); navtest, 12,146 tokens / 136 logs when complete; "
           "the interval is the paired log-cluster bootstrap (B = 10,000, seed 20260927) = 'another draw of LOGS' only; ONE training seed (H-ESTIM-SEED-1 open) and 3 search seeds (the spread is reported beside it); "
           "no head substitution; STRATEGIC PAUSED (PI ruling 2026-09-19, ITEM 25); the sealed C2 amendment sets the remaining stamps")


def c2_terms(dpdms: dict, seed_range: float, safety_points: dict, bars: dict) -> dict:
    """pure; every term has its own regression arm in test_navpipe.py.  Returns the terms and `rule_met` (the DQ3 conjunction)."""
    b, bs = float(bars["bar_dpdms_points"]), float(bars["bar_safety_points"])
    terms = {"lower_ci_gt_0": bool(dpdms["lo"] > 0.0), "point_ge_bar": bool(dpdms["point"] >= b), "point_gt_seed_range": bool(dpdms["point"] > seed_range)}
    safety = {k: bool(safety_points[k] >= bs) for k in AB.SAFETY_TERMS}
    return {"terms": terms, "safety_terms": safety, "rule_met": bool(all(terms.values()) and all(safety.values())), "bar_dpdms_points": b, "bar_safety_points": bs}


def analyse_c2(bundle: dict, so: dict, lab: dict, bars: dict, B: int = A.N_BOOT, seed: int = A.SEED) -> dict:
    res = AB.analyse(bundle, so, lab, B=B, seed=seed)
    dp, sr = res["endpoint_dPDMS_points"], res["per_seed_points"]["range"]
    safety = {k: res["sub_score_deltas_points"][k]["point"] for k in AB.SAFETY_TERMS}
    verdict = c2_terms(dp, sr, safety, bars)
    # T1 has a banked label for the PICK only: the in-fan block (arm argmax vs pick, which needs a label of a non-pick proposal) is not computable here
    pk = np.asarray(bundle["pick_real"] if "pick_real" in bundle else bundle["pick"], np.int64)[np.asarray(so["bundle_index"], np.int64)]
    am = np.asarray(so["arm_argmax_real"] if "arm_argmax_real" in so else so["arm_argmax"], np.int64)
    am_eq_pick = float((am == pk).mean())
    res["in_fan_decomposition_descriptive"] = {"status": "NOT COMPUTABLE at T1 (the harness labels the executed plans only; no non-pick proposal has a label)", "arm_argmax_equals_p2_pick_share": am_eq_pick}
    return {"read": "C2", "arm": "P2HEAD (no head substitution)", "decision_rule_terms": verdict, "rule_met": verdict["rule_met"], "endpoint_dPDMS_points": dp, "per_seed_points": res["per_seed_points"],
            "safety_points": safety, "sub_score_deltas_points": res["sub_score_deltas_points"], "inert": res["inert"], "families": res["families"],
            "in_fan_decomposition_descriptive": res["in_fan_decomposition_descriptive"], "moved_share": res["moved_share"], "n_tokens": res["n_tokens"], "n_logs": res["n_logs"], "units": res["units"],
            "stamps": C2_TIER, "STRATEGIC": AB.STRATEGIC, "outcome_names": "NOT SET HERE: the sealed C2 amendment names the outcomes; this record carries the DQ3 terms"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rule", required=True, choices=("c1", "c2"))
    ap.add_argument("--bars-json", default=None, help="c2: {'bar_dpdms_points': .., 'bar_safety_points': ..} from the sealed amendment's pins")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.rule == "c2" and not a.bars_json:
        raise SystemExit("--rule c2 needs --bars-json from the sealed amendment's pins (no default bar exists here)")
    b = A.load_npz(a.bundle)
    so = {k: v for k, v in np.load(a.search, allow_pickle=False).items()}
    lab = {k: v for k, v in np.load(a.labels, allow_pickle=False).items()}
    if a.rule == "c1":
        import spc_analyse as SA
        res = SA.analyse_c1(b, so, lab, with_concordance=False)
        head = ("ZZANALYSE_NAVPIPE", "c1", res["outcome"])
    else:
        if not a.bars_json:
            raise SystemExit("--rule c2 needs --bars-json from the sealed amendment's pins (no default bar exists here)")
        res = analyse_c2(b, so, lab, json.load(open(a.bars_json, encoding="utf-8")))
        head = ("ZZANALYSE_NAVPIPE", "c2", "RULE_MET" if res["rule_met"] else "RULE_NOT_MET")
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o)))
    print(*head, json.dumps(res["endpoint_dPDMS_points"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
