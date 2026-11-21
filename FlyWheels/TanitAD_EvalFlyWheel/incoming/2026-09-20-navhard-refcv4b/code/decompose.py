#!/usr/bin/env python3
"""WHERE an arm loses on EPDMS — sub-metric, stage, speed band, log, and which multiplier zeroed.

⛔ WHY THIS IS NOT OPTIONAL (mission brief + RULE ZERO). EPDMS is a PRODUCT of multipliers with a
weighted-average term: **one zeroed multiplier sinks a whole scene to 0.0 regardless of how well the
rest drove**. A headline delta therefore says nothing about the mechanism, and "arm X loses to STOP"
is a waypoint, never a deliverable. This turns the verdict into a ranked list of levers.

⭐ THE CONTROL THAT MAKES THE ZERO-ATTRIBUTION HONEST. A scene can be zeroed by SEVERAL multipliers
at once, so "NC caused 412 zeros" is only true as a *co-occurrence*. This reports, per term,
(a) how many zeroed scenes have that term at 0, and (b) how many have it **UNIQUELY** at 0 — the
second is the attributable count, the first is the upper bound, and quoting only the first is how a
term gets blamed for another's failures.

⚠️ SCOPE. On a split whose stage-1 rows are a CV STAND-IN for the model arm, only the STAGE-2
comparison is about the model. Stage 1 is reported anyway, labelled, so the asymmetry is visible
rather than silently dropped.

    python decompose.py --run <run dir> --arm A1 --vs STOP,CV,ECHO \
        --inputs <export json> --out raw/decomposition.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

SUB = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
       "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
       "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
#: ⛔ EXACTLY FOUR MULTIPLIERS. `summarize.py::epdms_formula` (promoted VERBATIM from the devkit's
#: docs/metrics.md and pinned by test_bench_suite_promotion.py) is
#:     prod(NC, DAC, DDC, TLC) * sum(w*m)/sum(w)
#: ⚠️ **TTC IS NOT A MULTIPLIER** — it is a WEIGHTED term with w = 5.0, the same weight as EP. W7
#: wrote it into this list from memory on the first pass and the DATA caught it: TTC zeroed 242 STOP
#: scenes but was the UNIQUE zero in exactly **0** of them, which is impossible for a real
#: multiplier. ⇒ the weights are read from the promoted source, never recalled. Same family as the
#: units trap in CLAUDE.md: a correct formula applied with the wrong term roles reads like an answer.
MULTIPLIERS = ("NC", "DAC", "DDC", "TLC")
#: weights of the averaged terms (summarize.py::W); EC's weight is DROPPED when EC is NaN (/14).
WEIGHTS = {"EP": 5.0, "TTC": 5.0, "LK": 2.0, "HC": 2.0, "EC": 2.0}
SUMMARY_ROWS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                "extended_pdm_score_combined")


def load(run: str, arm: str):
    import pandas as pd
    p = os.path.join(run, "scores", f"{arm}.csv")
    d = pd.read_csv(p, index_col=0)
    tok = d[~d.token.isin(SUMMARY_ROWS)].set_index("token")
    return tok, p


def stage_of(tok) -> dict:
    out = {}
    for t, r in tok.iterrows():
        s1 = r.get("ego_progress_stage_one")
        out[t] = 1 if (s1 == s1 and s1 is not None) else 2   # NaN-safe: s1 != s1 iff NaN
    return out


def band(v: float) -> str:
    for lo, hi in ((0, 2), (2, 5), (5, 8), (8, 12), (12, 99)):
        if lo <= v < hi:
            return f"{lo}-{hi} m/s" if hi < 99 else ">=12 m/s"
    return "unknown"


def epdms(vals: dict) -> float:
    """``prod(NC,DAC,DDC,TLC) * sum(w*m)/sum(w)``; EC's weight dropped when EC is NaN (/14).
    A literal re-implementation of ``summarize.py::epdms_formula`` — it exists so the counterfactual
    below can move ONE term, and its correctness is asserted against the devkit's own `score`
    column before any counterfactual is reported."""
    prod = 1.0
    for k in MULTIPLIERS:
        prod *= float(vals[k])
    num = den = 0.0
    for k, w in WEIGHTS.items():
        v = float(vals[k])
        if k == "EC" and v != v:                       # NaN
            continue
        num += w * v
        den += w
    return prod * num / den


def wtl(a, b, eps: float = 1e-12) -> dict:
    w = int(sum(1 for x, y in zip(a, b) if x - y > eps))
    l = int(sum(1 for x, y in zip(a, b) if y - x > eps))
    return {"win": w, "tie": int(len(a) - w - l), "loss": l, "n": int(len(a))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--vs", required=True, help="comma list of comparison arms")
    ap.add_argument("--inputs", required=True, help="the export JSON (speeds, log names)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import numpy as np

    doc = json.load(open(a.inputs, encoding="utf-8"))["tokens"]
    A, apath = load(a.run, a.arm)
    st = stage_of(A)
    v0 = {t: math.hypot(*doc[t]["ego_statuses"][-1]["ego_velocity"][:2]) for t in A.index if t in doc}
    log_of = {t: doc[t]["log_name"] for t in A.index if t in doc}
    out = {"_what": ("per-sub-metric / per-stage / per-speed-band / per-log decomposition of the "
                     "official per-token EPDMS `score`, plus zero-attribution over the multipliers"),
           "run": a.run, "arm": a.arm, "scores_csv": apath,
           "n_tokens": int(len(A)), "n_stage1": sum(1 for s in st.values() if s == 1),
           "n_stage2": sum(1 for s in st.values() if s == 2),
           "multiplier_terms": list(MULTIPLIERS),
           "weighted_average_terms": [k for k in SUB if k not in MULTIPLIERS],
           "vs": {}}

    def col(d, key, stage):
        return d[f"{SUB[key]}_stage_{'one' if stage == 1 else 'two'}"]

    # ---- zero attribution for THIS arm, per stage ------------------------------------
    za = {}
    for stage in (1, 2):
        idx = [t for t in A.index if st[t] == stage]
        if not idx:
            continue
        sub = A.loc[idx]
        sc = sub["score"].to_numpy(float)
        zero = sc <= 1e-12
        terms = {k: col(sub, k, stage).to_numpy(float) for k in MULTIPLIERS}
        zmask = {k: (terms[k] <= 1e-12) for k in MULTIPLIERS}
        n_zero_terms = np.sum([zmask[k] for k in MULTIPLIERS], axis=0)
        za[f"stage_{stage}"] = {
            "n": int(len(idx)), "n_zero_score": int(zero.sum()),
            "frac_zero_score": round(float(zero.mean()), 6),
            "per_term": {k: {"n_zeroed_scenes_with_this_term_0": int((zero & zmask[k]).sum()),
                             "n_zeroed_scenes_where_ONLY_this_term_is_0":
                                 int((zero & zmask[k] & (n_zero_terms == 1)).sum()),
                             "term_mean": round(float(np.nanmean(terms[k])), 6)}
                         for k in MULTIPLIERS},
            "_read": ("the FIRST count is a co-occurrence upper bound (several multipliers can be 0 "
                      "in one scene); the SECOND is the attributable count. Quoting only the first "
                      "blames a term for other terms' failures.")}
    out["zero_attribution"] = za

    # ---- LEVER RANKING: the counterfactual ceiling of repairing ONE term ---------------
    # ⭐ CLAUDE.md: "prefer the lever with the largest MEASURED effect, not the most interesting
    # one". Each entry answers: if this arm scored a PERFECT 1.0 on exactly this term and changed
    # nothing else, what would its stage-2 scene mean become? That is a CEILING, not a prediction —
    # repairing a term usually costs another (braking buys NC and loses EP) — and it is stated as
    # one. ⛔ The reconstruction is asserted against the devkit's own `score` column FIRST: a
    # counterfactual built on a formula that cannot reproduce the actual number is fiction.
    lev = {}
    for stage in (1, 2):
        idx = [t for t in A.index if st[t] == stage]
        if not idx:
            continue
        sub = A.loc[idx]
        vals = {k: col(sub, k, stage).to_numpy(float) for k in SUB}
        base = np.array([epdms({k: vals[k][i] for k in SUB}) for i in range(len(idx))])
        actual = sub["score"].to_numpy(float)
        chk = {"max_abs_diff_vs_devkit_score": float(np.nanmax(np.abs(base - actual))), "tol": 1e-9}
        chk["pass"] = bool(chk["max_abs_diff_vs_devkit_score"] <= chk["tol"])
        blk = {"formula_selfcheck": chk, "actual_scene_mean": round(float(actual.mean()), 6)}
        if chk["pass"]:
            per = {}
            for k in SUB:
                cf = np.array([epdms({kk: (1.0 if kk == k else vals[kk][i]) for kk in SUB})
                               for i in range(len(idx))])
                per[k] = {"scene_mean_if_this_term_were_perfect": round(float(cf.mean()), 6),
                          "gain": round(float(cf.mean() - actual.mean()), 6),
                          "term_mean_now": round(float(np.nanmean(vals[k])), 6),
                          "role": "multiplier" if k in MULTIPLIERS else f"weighted (w={WEIGHTS[k]})"}
            blk["per_term"] = dict(sorted(per.items(), key=lambda kv: -kv[1]["gain"]))
            blk["_read"] = ("a CEILING under a single-term repair with everything else frozen; the "
                            "terms TRADE (braking buys NC/TTC and loses EP), so these do not add up "
                            "and none is a promise")
        else:
            blk["per_term"] = {"status": "REFUSED",
                               "reason": ("the local EPDMS reconstruction does not reproduce the "
                                          "devkit `score` column, so no counterfactual on it is "
                                          "admissible")}
        lev[f"stage_{stage}"] = blk
    out["lever_ranking_counterfactual"] = lev

    # ---- per-stage sub-metric means + paired deltas ----------------------------------
    for other in [x.strip() for x in a.vs.split(",") if x.strip()]:
        if other == a.arm:
            continue
        try:
            B, bpath = load(a.run, other)
        except Exception as e:                                          # noqa: BLE001
            out["vs"][other] = {"status": "UNAVAILABLE", "reason": f"{type(e).__name__}: {e}"[:200]}
            continue
        common = [t for t in A.index if t in B.index]
        if len(common) != len(A.index):
            out["vs"][other] = {"status": "UNAVAILABLE", "n": len(common),
                                "reason": ("the two arms are NOT on identical tokens — a paired "
                                           "comparison is refused rather than computed on an "
                                           "intersection that neither arm's headline describes")}
            continue
        blk = {"status": "OK", "scores_csv": bpath, "n_tokens": len(common), "per_stage": {}}
        for stage in (1, 2):
            idx = [t for t in common if st[t] == stage]
            if not idx:
                continue
            sa, sb = A.loc[idx, "score"].to_numpy(float), B.loc[idx, "score"].to_numpy(float)
            terms = {}
            for k in SUB:
                ta = col(A.loc[idx], k, stage).to_numpy(float)
                tb = col(B.loc[idx], k, stage).to_numpy(float)
                terms[k] = {"arm": round(float(np.nanmean(ta)), 6),
                            "other": round(float(np.nanmean(tb)), 6),
                            "delta": round(float(np.nanmean(ta) - np.nanmean(tb)), 6),
                            "is_multiplier": k in MULTIPLIERS}
            blk["per_stage"][f"stage_{stage}"] = {
                "n": len(idx), "scene_mean_arm": round(float(sa.mean()), 6),
                "scene_mean_other": round(float(sb.mean()), 6),
                "scene_mean_delta": round(float(sa.mean() - sb.mean()), 6),
                "wtl": wtl(sa, sb), "submetrics": terms,
                "n_zero_arm": int((sa <= 1e-12).sum()), "n_zero_other": int((sb <= 1e-12).sum()),
                "n_flip_other0_arm_positive": int(((sb <= 1e-12) & (sa > 1e-12)).sum()),
                "n_flip_arm0_other_positive": int(((sa <= 1e-12) & (sb > 1e-12)).sum())}
        # ---- speed bands (stage 2 = the model's own rows on a stand-in split) --------
        bands = {}
        for stage in (1, 2):
            idx = [t for t in common if st[t] == stage and t in v0]
            if not idx:
                continue
            bb = {}
            for t in idx:
                bb.setdefault(band(v0[t]), []).append(t)
            bands[f"stage_{stage}"] = {
                k: {"n": len(v),
                    "arm": round(float(A.loc[v, "score"].mean()), 6),
                    "other": round(float(B.loc[v, "score"].mean()), 6),
                    "delta": round(float(A.loc[v, "score"].mean() - B.loc[v, "score"].mean()), 6),
                    "wtl": wtl(A.loc[v, "score"].to_numpy(float), B.loc[v, "score"].to_numpy(float))}
                for k, v in sorted(bb.items())}
        blk["speed_bands_by_v0_at_t0"] = bands
        # ---- per log (stage 2) -------------------------------------------------------
        idx2 = [t for t in common if st[t] == 2 and t in log_of]
        per_log, groups = {}, {}
        for t in idx2:
            groups.setdefault(log_of[t], []).append(t)
        for ln, ts in groups.items():
            per_log[ln] = {"n": len(ts),
                           "arm": round(float(A.loc[ts, "score"].mean()), 6),
                           "other": round(float(B.loc[ts, "score"].mean()), 6),
                           "delta": round(float(A.loc[ts, "score"].mean() - B.loc[ts, "score"].mean()), 6)}
        blk["per_log_stage2"] = {"n_logs": len(per_log),
                                 "n_logs_arm_ahead": sum(1 for v in per_log.values() if v["delta"] > 0),
                                 "logs": per_log}
        out["vs"][other] = blk
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    brief = {"arm": a.arm, "zero_attribution": {k: {"n_zero_score": v["n_zero_score"], "n": v["n"]}
                                                for k, v in za.items()},
             "vs": {o: (b.get("per_stage", {}).get("stage_2", {}).get("scene_mean_delta")
                        if b.get("status") == "OK" else b.get("status"))
                    for o, b in out["vs"].items()}}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
