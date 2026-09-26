#!/usr/bin/env python3
"""Paired interval on the STAGE-2 statistic S2-EPDMS-u — a DIAGNOSTIC, and labelled as one.

⛔ WHY THIS IS NOT "THE SETTLED ESTIMATOR". ``taniteval/adapters/navsim_ci.py`` rules on exactly two
aggregates (``AGGREGATIONS``): the official two-stage mapping-key mean and the single-stage token
mean. S2-EPDMS-u — the only statistic that describes refcv4b on navhard, because its stage 1 is a CV
stand-in — is NEITHER. So the FORMAL interval field for the model arm is
``{status: UNAVAILABLE, reason, n}``, as the mission requires, and this file does not overwrite it.

⭐ WHAT THIS FILE IS. The settled estimator's own resampling code —
``taniteval.ci.paired_episode_cluster_bootstrap(eid=log_name, reduce=nanmean)``, the exact draw
generator ``navsim_ci.paired_log_cluster_bootstrap`` calls, same cluster unit (``log_name``), same B
(2000), seed (0), alpha (0.05), RG-14 floor (8 clusters) — applied to the S2-EPDMS-u units. It
answers "is refcv4b's gap to STOP large relative to between-LOG variation?", and nothing more.
⛔ It is emitted under its own name (``s2_group_uniform_paired_log_bootstrap_DIAGNOSTIC``), never
under the settled estimator's.

THE STATISTIC, reproduced exactly (``summarize.py::s2_group_uniform``): for every mapping key of the
split yaml, two groups — ``[p[0] for p in pairs]`` and ``[p[1] for p in pairs]`` — each contributing
the UNIFORM mean of its stage-2 ``score``; S2-EPDMS-u = the mean over the 450 groups. The unit
resampled is the GROUP; its cluster is the ``log_name`` of its tokens (asserted single-valued).
⭐ Control: the full-sample point MUST reproduce each arm's ``statistics.S2_EPDMS_u.value`` from the
suite's own summary.json to 1e-12, or the file refuses — a bootstrap of a different aggregate is
precise about the wrong thing.

⚠️ THE QUESTION IT ANSWERS: *would another draw of LOGS say this?* It is BLIND to training variance
(H-ESTIM-SEED-1). The inference-variance question is closed by construction for refcv4b (the decoder
noise is `torch.zeros_like` outside training; `MODEL_REGISTRY.md` §4.6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))
SUMMARY_ROWS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                "extended_pdm_score_combined", "average_all_frames")


def stage2_scores(csv_path: Path) -> dict:
    d = pd.read_csv(csv_path, index_col=0)
    d = d[~d["token"].isin(SUMMARY_ROWS)]
    d = d[d["ego_progress_stage_one"].isna()]            # stage-2 rows only
    return {str(t): float(s) for t, s in zip(d["token"], d["score"])}


def group_units(mapping, tok_log: dict):
    """-> [(tokens, log_name)] for the 450 groups, in the yaml's order."""
    units = []
    for orig, prev, pairs in mapping:
        for grp in ([str(p[0]) for p in pairs], [str(p[1]) for p in pairs]):
            logs = {tok_log.get(t) for t in grp}
            if len(logs) != 1 or None in logs:
                raise SystemExit(f"a group spans {len(logs)} logs or has an unknown log: {sorted(map(str, logs))[:3]}")
            units.append((grp, logs.pop()))
    return units


def contributions(scores: dict, units) -> np.ndarray:
    out = []
    for grp, _ in units:
        v = [scores[t] for t in grp]                       # KeyError = a missing token: fail loud
        out.append(float(np.mean(v)))
    return np.asarray(out, dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--vs", default="STOP,CV,ECHO")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from taniteval import ci as _ci
    from taniteval.bench.navsim import profiles as P
    y = P.read_split_yaml("navhard_two_stage")
    doc = json.load(open(a.inputs, encoding="utf-8"))["tokens"]
    tok_log = {t: r["log_name"] for t, r in doc.items()}
    units = group_units(y["mapping"], tok_log)
    clusters = [lg for _, lg in units]
    S = json.load(open(os.path.join(a.run, "summary.json"), encoding="utf-8"))
    arms = [a.arm] + [x.strip() for x in a.vs.split(",") if x.strip()]
    C, point_check = {}, {}
    for arm in arms:
        C[arm] = contributions(stage2_scores(Path(a.run) / "scores" / f"{arm}.csv"), units)
        mine = float(np.nanmean(C[arm]))
        suite = float(S["arms"][arm]["statistics"]["S2_EPDMS_u"]["value"])
        point_check[arm] = {"recomputed": mine, "suite_summary_json": suite, "abs_diff": abs(mine - suite)}
        if abs(mine - suite) > 1e-12:
            raise SystemExit(f"REFUSED: {arm} recomputed S2-EPDMS-u {mine!r} != summary.json {suite!r}")
    n_clusters = len(set(clusters))
    out = {"_what": ("DIAGNOSTIC paired interval on S2-EPDMS-u (stage-2 group-uniform mean) — NOT the "
                     "settled estimator, whose AGGREGATIONS has no key for this statistic"),
           "name": "s2_group_uniform_paired_log_bootstrap_DIAGNOSTIC",
           "machine": "taniteval.ci.paired_episode_cluster_bootstrap(eid=log_name, reduce=nanmean) — "
                      "the draw generator navsim_ci.paired_log_cluster_bootstrap itself calls",
           "unit": "stage-2 GROUP (2 per mapping key)", "n_units": len(units),
           "cluster_unit": "log_name", "n_clusters": n_clusters, "min_clusters": 8,
           "n_boot": 2000, "seed": 0, "alpha": 0.05,
           "question_answered": ("would another draw of LOGS say this? — evaluation-set variance only; "
                                 "BLIND to training variance; inference variance closed by construction "
                                 "for refcv4b (deterministic decoder at inference)"),
           "point_reproduces_suite": point_check, "pairs": {}}
    if n_clusters < 8:
        out["status"] = "UNAVAILABLE"
        out["reason"] = f"{n_clusters} logs < RG-14 floor of 8"
    else:
        for other in arms[1:]:
            pr = _ci.paired_episode_cluster_bootstrap(C[a.arm], C[other], clusters, n_boot=2000, seed=0,
                                                      alpha=0.05, reduce=lambda v: float(np.nanmean(v)))
            out["pairs"][f"{a.arm}-{other}"] = {
                "point_delta": round(float(np.nanmean(C[a.arm]) - np.nanmean(C[other])), 6),
                "lo": pr.get("lo"), "hi": pr.get("hi"), "separated": pr.get("separated"),
                "n_clusters": n_clusters}
        out["status"] = "OK_DIAGNOSTIC"
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"status": out["status"], "n_units": len(units), "n_clusters": n_clusters,
                      "point_reproduces_suite": {k: v["abs_diff"] for k, v in point_check.items()},
                      "pairs": out["pairs"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
