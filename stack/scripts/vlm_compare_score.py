"""Score a VLM head-to-head produced by ``vlm_model_compare.py``. NO POD, NO GPU.

Ground truth is the KINEMATIC v2.1 route label (``refb_labels.route_from_future_v21``)
carried inline on every record, so this runs anywhere the raw records exist.

WHAT IS AND IS NOT ADMISSIBLE
  * Only **Pass A** may enter accuracy statistics. Pass A is not shown the
    numeric future ego track; Pass B is, so its ROUTE is downstream of our own
    kinematics and agreeing with them measures nothing. Pass B is scored here
    for schema/slot behaviour only and is refused entry to every accuracy and
    agreement number (``--pass B`` will raise).
  * Windows where the kinematic labeler itself says ``unknown`` (``no_arc`` /
    ``gray_zone``) have no ground truth. They are EXCLUDED from accuracy and
    reported separately as a distribution — dropping them silently would let an
    arm look good by answering only the easy windows.
  * Abstention is an OUTCOME, not a missing value. Every rate is reported twice:
    ``over_answered`` (denominator = windows the model actually answered
    left/straight/right) and ``over_all`` (denominator = every GT-valid window,
    with ``unknown`` / ``u_turn`` / parse failures counted as wrong). A model
    that abstains everywhere scores 0 on ``over_all`` — which is the honest
    number.

PAIRED, ALWAYS. Both arms see the same window list, so single-arm intervals
must never be combined in quadrature (retired 2026-07-20). The difference is
tested with **McNemar** (exact binomial on the discordant pairs) and intervalled
with the **paired episode-cluster bootstrap** from ``taniteval.ci`` — clustered
on EPISODE because windows inside an episode are correlated.

Usage:
  python3 vlm_compare_score.py --out <dir with windows.json + <tag>/ dirs> \
      --arms reason1,reason2 --json results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

CLS3 = ("left", "straight", "right")
VLM_OUTCOMES = ("left", "straight", "right", "u_turn", "unknown", "no_answer")


def _find_taniteval():
    """Locate ``taniteval`` by walking up from this file — repo-layout agnostic."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cand = os.path.join(d, "taniteval")
        if os.path.isdir(os.path.join(cand, "taniteval")):
            return cand
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "ci.py")):
            return d
        d = os.path.dirname(d)
    return None


_TE = _find_taniteval()
if _TE and _TE not in sys.path:
    sys.path.insert(0, _TE)
try:
    from taniteval.ci import (episode_cluster_bootstrap,
                              paired_episode_cluster_bootstrap)
    _CI_OK = True
except Exception as _e:                      # loud: an interval without its
    _CI_OK = False                           # estimator is inadmissible here
    _CI_ERR = str(_e)


# --------------------------------------------------------------------- load
def load_arm(out_dir: str, tag: str) -> dict:
    """{(episode, t): record} for one arm.

    Accepts EITHER ``<out>/<tag>/ep_*_t*.json`` (what the pod writes) or a
    consolidated ``<out>/<tag>.jsonl`` — the form committed to the repo, so the
    comparison re-scores with no pod and no 400-file directory."""
    recs = {}
    jl = os.path.join(out_dir, tag + ".jsonl")
    if os.path.isfile(jl):
        with open(jl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[(r["episode"], int(r["t"]))] = r
        return recs
    for f in sorted(glob.glob(os.path.join(out_dir, tag, "ep_*_t*.json"))):
        r = json.load(open(f))
        recs[(r["episode"], int(r["t"]))] = r
    return recs


def vlm_outcome(rec: dict) -> str:
    """The arm's Pass-A answer as one of VLM_OUTCOMES. Never repaired."""
    a = rec.get("pass_A") or {}
    if a.get("outcome") != "ok":
        return "no_answer"                   # parse fail / enum violation / error
    r = a.get("ROUTE")
    return r if r in VLM_OUTCOMES else "no_answer"


# ------------------------------------------------------------------ metrics
def confusion(gt: list, pred: list) -> dict:
    M = {g: {p: 0 for p in VLM_OUTCOMES} for g in CLS3}
    for g, p in zip(gt, pred):
        M[g][p] += 1
    return M


def _rate(num, den):
    return None if den == 0 else round(num / den, 4)


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (no scipy).

    NOTE it treats windows as independent, which they are NOT (5 windows per
    episode). It is therefore ANTI-CONSERVATIVE and is reported only next to
    the episode-cluster bootstrap interval, never instead of it."""
    if n == 0:
        return float("nan")
    from math import comb
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return float(min(1.0, sum(x for x in pmf if x <= obs * (1 + 1e-9))))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p on [[a, b], [c, d]] (sum-of-small-p). No scipy."""
    from math import comb
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    if n == 0 or r1 == 0 or r2 == 0 or c1 == 0 or c1 == n:
        return 1.0
    def _p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    obs = _p(a)
    return float(min(1.0, sum(_p(x) for x in range(lo, hi + 1)
                              if _p(x) <= obs * (1 + 1e-9))))


def score_arm(recs: dict, keys: list, wins: dict) -> dict:
    """Every per-arm number in the brief, on the SHARED key list."""
    gt = [wins[k]["kin_v21"] for k in keys]
    pred = [vlm_outcome(recs[k]) for k in keys]
    a = [recs[k].get("pass_A") or {} for k in keys]
    wins_eid = [k[0] for k in keys]                  # episode = cluster unit
    wins_nf = [wins[k]["n_future_frames"] for k in keys]

    valid = [i for i, g in enumerate(gt) if g["valid"]]
    gt_v = [gt[i]["route"] for i in valid]
    pr_v = [pred[i] for i in valid]

    # --- 1. turn detection recall (GT turn -> model says any turn) ------------
    turn_i = [i for i, g in zip(valid, gt_v) if g in ("left", "right")]
    n_turn = len(turn_i)
    det = sum(1 for i in turn_i if pred[i] in ("left", "right", "u_turn"))
    # --- 2. direction | detected ---------------------------------------------
    dir_i = [i for i in turn_i if pred[i] in ("left", "right")]
    dir_ok = sum(1 for i in dir_i if pred[i] == gt[i]["route"])
    # --- 3. 3-class accuracy --------------------------------------------------
    ans_i = [i for i in valid if pred[i] in CLS3]
    ans_ok = sum(1 for i in ans_i if pred[i] == gt[i]["route"])
    all_ok = ans_ok                          # non-CLS3 answers are wrong by def
    # --- straight-only recall (the trivial-majority check) --------------------
    st_i = [i for i in valid if gt[i]["route"] == "straight"]
    st_ok = sum(1 for i in st_i if pred[i] == "straight")

    # --- 4. parse / enum failure taxonomy ------------------------------------
    outc = {}
    for x in a:
        outc[x.get("outcome", "missing")] = outc.get(x.get("outcome", "missing"), 0) + 1
    bad_tokens = sorted({str(x.get("raw_route")) for x in a
                         if x.get("outcome") == "enum_violation"})
    n_trunc = sum(1 for x in a if x.get("truncated"))
    n_geom_bad = sum(1 for x in a if x.get("road_geometry_ok") is False)

    # --- 5. slot fill ---------------------------------------------------------
    slots = {}
    for x in a:
        for s, filled in (x.get("slot_filled") or {}).items():
            d = slots.setdefault(s, [0, 0])
            d[0] += bool(filled)
            d[1] += 1
    slot_fill = {s: _rate(v[0], v[1]) for s, v in sorted(slots.items())}

    # --- 6. cost --------------------------------------------------------------
    gens = [x.get("n_gen_tokens", 0) for x in a if x.get("n_gen_tokens")]
    secs = [x.get("gen_seconds", 0.0) for x in a if x.get("gen_seconds")]
    prompts = [x.get("n_prompt_tokens", 0) for x in a if x.get("n_prompt_tokens")]
    peak = max((r.get("peak_vram_gib", 0.0) for r in recs.values()), default=0.0)

    # --- abstention -----------------------------------------------------------
    n_unknown = sum(1 for i in valid if pred[i] == "unknown")
    n_uturn = sum(1 for i in valid if pred[i] == "u_turn")
    n_noans = sum(1 for i in valid if pred[i] == "no_answer")

    # --- what the arm says where kinematics abstains --------------------------
    inval = [i for i, g in enumerate(gt) if not g["valid"]]
    unk_dist = {}
    for i in inval:
        unk_dist[pred[i]] = unk_dist.get(pred[i], 0) + 1

    # --- 2b. IS THE DIRECTION CALL BETTER THAN A COIN FLIP? -------------------
    # The decisive question for a route labeler. Conditional on the arm having
    # committed to left/right, test the hit rate against p=0.5 two ways: an
    # exact binomial (ignores episode clustering -> anti-conservative) AND an
    # episode-cluster bootstrap interval (the decision-grade one).
    dir_hits = [int(pred[i] == gt[i]["route"]) for i in dir_i]
    dir_eid = [wins_eid[i] for i in dir_i]
    dir_chance = {
        "n_detected_lr": len(dir_i), "n_correct": dir_ok,
        "accuracy": _rate(dir_ok, len(dir_i)),
        "binomial_p_vs_0.5": (round(binom_two_sided(dir_ok, len(dir_i)), 6)
                              if dir_i else None),
        "binomial_caveat": "ignores episode clustering; anti-conservative",
        "cluster_bootstrap": _boot(dir_hits, dir_eid, 0)}
    if isinstance(dir_chance["cluster_bootstrap"], dict) and \
            "lo" in dir_chance["cluster_bootstrap"]:
        b = dir_chance["cluster_bootstrap"]
        dir_chance["beats_chance_95ci"] = bool(b["lo"] > 0.5)

    # --- 2c. DOES THE DIRECTION CALL CARRY ANY INFORMATION? ------------------
    # Fisher exact on the 2x2 {GT left|right} x {said left|right}. This is the
    # bias-free version of the question: a model can score ~50 % "accuracy" by
    # calling everything left on a left-heavy corpus, and that model's odds
    # ratio is 1. Accuracy alone cannot tell those apart; this can.
    t11 = sum(1 for i in dir_i if gt[i]["route"] == "left" and pred[i] == "left")
    t12 = sum(1 for i in dir_i if gt[i]["route"] == "left" and pred[i] == "right")
    t21 = sum(1 for i in dir_i if gt[i]["route"] == "right" and pred[i] == "left")
    t22 = sum(1 for i in dir_i if gt[i]["route"] == "right" and pred[i] == "right")
    assoc = {"table_gt_left": {"said_left": t11, "said_right": t12},
             "table_gt_right": {"said_left": t21, "said_right": t22},
             "odds_ratio": (round((t11 * t22) / (t12 * t21), 3)
                            if t12 and t21 else None),
             "fisher_p_two_sided": round(fisher_exact_2x2(t11, t12, t21, t22), 6),
             "acc_given_detected_gt_left": _rate(t11, t11 + t12),
             "acc_given_detected_gt_right": _rate(t22, t21 + t22),
             "test": "fisher_exact_2x2_two_sided (windows treated as "
                     "independent — see clustering caveat)"}

    # --- left/right prediction bias ------------------------------------------
    n_pred_l = sum(1 for i in valid if pred[i] == "left")
    n_pred_r = sum(1 for i in valid if pred[i] == "right")
    lr_bias = {"n_pred_left": n_pred_l, "n_pred_right": n_pred_r,
               "n_gt_left": sum(1 for i in valid if gt[i]["route"] == "left"),
               "n_gt_right": sum(1 for i in valid if gt[i]["route"] == "right"),
               "left_share_of_predicted_turns": _rate(n_pred_l, n_pred_l + n_pred_r),
               "left_share_of_gt_turns": _rate(
                   sum(1 for i in valid if gt[i]["route"] == "left"), n_turn),
               "recall_left": _rate(
                   sum(1 for i in valid
                       if gt[i]["route"] == "left" and pred[i] == "left"),
                   sum(1 for i in valid if gt[i]["route"] == "left")),
               "recall_right": _rate(
                   sum(1 for i in valid
                       if gt[i]["route"] == "right" and pred[i] == "right"),
                   sum(1 for i in valid if gt[i]["route"] == "right"))}

    # --- stratified by how much FUTURE the window actually had ---------------
    strata = {}
    for i in valid:
        nf = wins_nf[i]
        d = strata.setdefault(str(nf), {"n": 0, "correct": 0, "n_turn": 0,
                                        "turn_detected": 0})
        d["n"] += 1
        d["correct"] += int(pred[i] == gt[i]["route"])
        if gt[i]["route"] in ("left", "right"):
            d["n_turn"] += 1
            d["turn_detected"] += int(pred[i] in ("left", "right", "u_turn"))
    for d in strata.values():
        d["acc3_over_all"] = _rate(d["correct"], d["n"])
        d["turn_detection_recall"] = _rate(d["turn_detected"], d["n_turn"])

    return {
        "model": next(iter(recs.values()))["model"],
        "arch": next(iter(recs.values())).get("arch"),
        "n_windows": len(keys),
        "n_gt_valid": len(valid),
        "n_gt_turn": n_turn,
        "n_gt_straight": len(st_i),
        "turn_detection_recall": _rate(det, n_turn),
        "turn_detected_n": det,
        "direction_accuracy_given_detected": _rate(dir_ok, len(dir_i)),
        "direction_n_detected_lr": len(dir_i),
        "direction_n_correct": dir_ok,
        "direction_vs_chance": dir_chance,
        "direction_association": assoc,
        "left_right_bias": lr_bias,
        "by_n_future_frames": strata,
        "straight_recall": _rate(st_ok, len(st_i)),
        "acc3_over_answered": _rate(ans_ok, len(ans_i)),
        "acc3_n_answered": len(ans_i),
        "acc3_over_all": _rate(all_ok, len(valid)),
        "abstention_rate_unknown": _rate(n_unknown, len(valid)),
        "u_turn_rate": _rate(n_uturn, len(valid)),
        "no_answer_rate": _rate(n_noans, len(valid)),
        "confusion_gt_rows": confusion(gt_v, pr_v),
        "outcome_counts": outc,
        "parse_failure_rate": _rate(
            sum(v for k, v in outc.items()
                if k in ("no_json", "json_invalid", "missing_route_key",
                         "exception")), len(a)),
        "enum_violation_rate": _rate(outc.get("enum_violation", 0), len(a)),
        "enum_violation_tokens": bad_tokens[:20],
        "road_geometry_violation_rate": _rate(n_geom_bad, len(a)),
        "truncation_rate": _rate(n_trunc, len(a)),
        "slot_fill_rate": slot_fill,
        "gen_tokens_mean": round(float(np.mean(gens)), 1) if gens else None,
        "gen_tokens_p95": round(float(np.percentile(gens, 95)), 1) if gens else None,
        "prompt_tokens_mean": round(float(np.mean(prompts)), 1) if prompts else None,
        "gen_seconds_mean": round(float(np.mean(secs)), 3) if secs else None,
        # the median is the contention-robust throughput number: a co-tenant GPU
        # job inflates the mean/p95 of whichever arm happened to overlap it
        "gen_seconds_median": round(float(np.median(secs)), 3) if secs else None,
        "gen_seconds_p95": round(float(np.percentile(secs, 95)), 3) if secs else None,
        "peak_vram_gib": round(peak, 3),
        "on_kinematic_unknown_windows": {"n": len(inval), "vlm_says": unk_dist},
    }


# ------------------------------------------------------- paired inferentials
def cohens_kappa(a: list, b: list) -> dict:
    """Agreement between the two ARMS (not with kinematics), all outcomes."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return {"kappa": None}
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    exp = sum(pa[c] * pb[c] for c in cats)
    k = (obs - exp) / (1 - exp) if abs(1 - exp) > 1e-12 else float("nan")
    return {"kappa": round(float(k), 4), "observed_agreement": round(obs, 4),
            "expected_agreement": round(exp, 4), "n": n}


def mcnemar(a_ok: list, b_ok: list) -> dict:
    """Exact two-sided McNemar on the discordant pairs. No scipy."""
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)   # a right, b wrong
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)   # b right, a wrong
    n = b + c
    if n == 0:
        return {"b_a_only": 0, "c_b_only": 0, "n_discordant": 0, "p_value": 1.0,
                "test": "mcnemar_exact_binomial"}
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return {"b_a_only": b, "c_b_only": c, "n_discordant": n,
            "p_value": round(min(1.0, p), 6), "test": "mcnemar_exact_binomial"}


def paired_block(name, a_ok, b_ok, eid, seed=0):
    out = {"metric": name, "mcnemar": mcnemar(a_ok, b_ok),
           "mean_a": round(float(np.mean(a_ok)), 4) if a_ok else None,
           "mean_b": round(float(np.mean(b_ok)), 4) if b_ok else None,
           "n": len(a_ok), "n_episodes": len(set(eid))}
    if _CI_OK and a_ok:
        out["paired_ci"] = paired_episode_cluster_bootstrap(
            np.array(a_ok, float), np.array(b_ok, float), eid, seed=seed)
    else:
        # An empty subset is reported as such, never silently as 0/NaN.
        out["paired_ci"] = {"error": _CI_ERR if not _CI_OK else "empty_subset"}
    return out


def _boot(vals, eid, seed):
    if not _CI_OK:
        return {"error": _CI_ERR}
    if len(vals) == 0:
        return {"error": "empty_subset"}
    return episode_cluster_bootstrap(np.asarray(vals, float), eid, seed=seed)


# -------------------------------------------------- Pass B: SLOT ADHERENCE ONLY
def _passb_enums() -> dict:
    """{slot: allowed tokens} for every categorical field Pass B asks for.

    Imported, never duplicated — a hand-copied enum table would drift from the
    frozen vocabulary within a week. This is the ONLY part of the scorer that
    needs `vlm_route_labels` (and therefore torch); the Pass-A path stays
    numpy-only.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.dirname(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import vlm_route_labels as VL
    from tanitad.lake import vocab as V
    e = {}
    e.update({f"SCENARIO.{k}": v for k, v in VL.ENUMS_SCENARIO.items()})
    e.update({f"OBS.{k}": v for k, v in VL.ENUMS_OBS.items()})
    e["STRATEGIC.ROUTE"] = VL.ROUTE_ENUM
    for k in ("MISSION", "LANEOBJ", "SPEEDPOLICY", "STYLE", "RISK", "ODD"):
        e[f"STRATEGIC.{k}"] = tuple(V.STRATEGIC_TOKENS[k])
    for k in ("LATMANEUVER", "LONMODE", "VSOURCE", "HEADWAY", "DYN", "RULECTX",
              "SIGNAL", "INTERACT", "TACPOINT", "LIGHTSTATE"):
        e[f"TACTICAL.{k}"] = tuple(V.TACTICAL_TOKENS[k])
    return e


_PASSB_PATHS = {
    "SCENARIO.road_type": ("SCENARIO", "road_type"),
    "SCENARIO.weather": ("SCENARIO", "environment", "weather"),
    "SCENARIO.time_of_day": ("SCENARIO", "environment", "time_of_day"),
    "SCENARIO.lighting": ("SCENARIO", "environment", "lighting"),
    "SCENARIO.surface": ("SCENARIO", "surface"),
    "SCENARIO.traffic_density": ("SCENARIO", "traffic_density"),
    "SCENARIO.road_geometry": ("SCENARIO", "road_geometry"),
    "SCENARIO.scenario_tag": ("SCENARIO", "scenario_tag"),
    "SCENARIO.difficulty": ("SCENARIO", "difficulty"),
    "STRATEGIC.ROUTE": ("STRATEGIC", "ROUTE"),
    "OBS.lead_lane": ("OBSERVATIONS", "lead_vehicle", "lane"),
    "OBS.distance_bucket": ("OBSERVATIONS", "lead_vehicle", "distance_bucket"),
    "OBS.relative_motion": ("OBSERVATIONS", "lead_vehicle", "relative_motion"),
    "OBS.markings": ("OBSERVATIONS", "lane_info", "markings"),
    "OBS.lane_type": ("OBSERVATIONS", "lane_info", "lane_type"),
    "OBS.light_state": ("OBSERVATIONS", "traffic_light", "state"),
}
for _k in ("MISSION", "LANEOBJ", "SPEEDPOLICY", "STYLE", "RISK", "ODD"):
    _PASSB_PATHS[f"STRATEGIC.{_k}"] = ("STRATEGIC", _k)
for _k in ("LATMANEUVER", "LONMODE", "VSOURCE", "HEADWAY", "DYN", "RULECTX",
           "SIGNAL", "INTERACT", "TACPOINT", "LIGHTSTATE"):
    _PASSB_PATHS[f"TACTICAL.{_k}"] = ("TACTICAL", _k)


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def score_passb_slots(recs: dict) -> dict:
    """Per-slot IN-VOCAB / VIOLATION / UNKNOWN / MISSING rates for Pass B.

    Schema behaviour ONLY. Pass B is shown the numeric future ego track, so
    nothing here may be read as accuracy — in particular ``STRATEGIC.ROUTE`` is
    reported for adherence and is inadmissible as a route measurement.
    """
    enums = _passb_enums()
    out, n = {}, len(recs)
    parsed_ok = err = trunc = 0
    gens, secs = [], []
    for r in recs.values():
        b = r.get("pass_B") or {}
        js = b.get("parsed") or {}
        parsed_ok += bool(js)
        err += bool(b.get("error"))
        trunc += bool(b.get("truncated"))
        if b.get("n_gen_tokens"):
            gens.append(b["n_gen_tokens"])
        if b.get("gen_seconds"):
            secs.append(b["gen_seconds"])
        for slot, path in _PASSB_PATHS.items():
            allowed = enums.get(slot)
            if allowed is None:
                continue
            d = out.setdefault(slot, {"in_vocab": 0, "violation": 0,
                                      "unknown": 0, "missing": 0,
                                      "bad_tokens": set()})
            v = _dig(js, path)
            if v is None:
                d["missing"] += 1
            elif isinstance(v, str) and v.strip().lower() in allowed:
                if v.strip().lower() in ("unknown", "none"):
                    d["unknown"] += 1
                else:
                    d["in_vocab"] += 1
            else:
                d["violation"] += 1
                d["bad_tokens"].add(str(v)[:40])
    slots = {}
    for s, d in sorted(out.items()):
        answered = d["in_vocab"] + d["unknown"] + d["violation"]
        slots[s] = {
            "in_vocab_rate": _rate(d["in_vocab"] + d["unknown"], max(answered, 1)),
            "violation_rate": _rate(d["violation"], max(answered, 1)),
            "answered_rate": _rate(answered, n),
            "informative_rate": _rate(d["in_vocab"], n),   # non-unknown/none
            "n_unknown": d["unknown"], "n_missing": d["missing"],
            "bad_tokens": sorted(d["bad_tokens"])[:6]}
    viol = sum(v["violation_rate"] * 1 for v in slots.values() if v["violation_rate"])
    return {"n_windows": n,
            "json_parse_rate": _rate(parsed_ok, n),
            "error_rate": _rate(err, n),
            "truncation_rate": _rate(trunc, n),
            "gen_tokens_mean": round(float(np.mean(gens)), 1) if gens else None,
            "gen_seconds_median": round(float(np.median(secs)), 2) if secs else None,
            "mean_slot_violation_rate": round(viol / max(1, len(slots)), 4),
            "mean_informative_rate": round(
                float(np.mean([v["informative_rate"] for v in slots.values()])), 4),
            "n_slots": len(slots), "slots": slots,
            "caveat": "SCHEMA ADHERENCE ONLY — Pass B sees the numeric future "
                      "ego track; STRATEGIC.ROUTE here is NOT a route measurement"}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser("vlm_compare_score")
    ap.add_argument("--out", required=True, help="dir with windows.json + arms")
    ap.add_argument("--arms", default="reason1,reason2")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pass", dest="which", default="A", choices=["A"],
                    help="Pass A only — Pass B ROUTE is downstream of kinematics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passb-slots", action="store_true",
                    help="score PASS B SCHEMA ADHERENCE ONLY for --arms and "
                         "exit; emits no accuracy or agreement numbers")
    args = ap.parse_args()

    if args.passb_slots:
        res = {"mode": "pass_B_slot_adherence_only",
               "arms": {t.strip(): score_passb_slots(load_arm(args.out, t.strip()))
                        for t in args.arms.split(",") if t.strip()}}
        txt = json.dumps(res, indent=1)
        if args.json:
            open(args.json, "w").write(txt)
            print(f"wrote {args.json}")
        else:
            print(txt)
        return

    W = json.load(open(os.path.join(args.out, "windows.json")))
    wins = {(w["episode"], int(w["t"])): w for w in W["windows"]}
    tags = [t.strip() for t in args.arms.split(",") if t.strip()]
    arms = {t: load_arm(args.out, t) for t in tags}
    for t, r in arms.items():
        if not r:
            raise SystemExit(f"arm {t!r} has no records under {args.out}/{t}")

    # THE SHARED KEY LIST — every arm scored on exactly these windows.
    keys = sorted(set(wins) & set.intersection(*[set(r) for r in arms.values()]))
    res = {"out_dir": os.path.abspath(args.out),
           "prompt_version": W.get("prompt_version"),
           "val_build": W.get("val"), "stride": W.get("stride"),
           "n_windows_manifest": W.get("n_windows"),
           "n_windows_scored": len(keys),
           "n_episodes": len({k[0] for k in keys}),
           "unpaired_dropped": {t: len(set(r) - set(keys)) for t, r in arms.items()},
           "ci_estimator": ("paired_episode_cluster_bootstrap" if _CI_OK
                            else f"UNAVAILABLE: {_CI_ERR}"),
           "ground_truth": "kinematic refb_labels.route_from_future_v21 (v2.1)",
           "arms": {}}
    for t in tags:
        res["arms"][t] = score_arm(arms[t], keys, wins)
        run = os.path.join(args.out, f"run_{t}.json")
        if os.path.exists(run):
            res["arms"][t]["run_summary"] = json.load(open(run))

    # ---- inter-arm agreement (independent of kinematics) --------------------
    if len(tags) >= 2:
        A, B = tags[0], tags[1]
        pa = [vlm_outcome(arms[A][k]) for k in keys]
        pb = [vlm_outcome(arms[B][k]) for k in keys]
        res["inter_arm_agreement_all_windows"] = cohens_kappa(pa, pb)
        vk = [k for k in keys if wins[k]["kin_v21"]["valid"]]
        res["inter_arm_agreement_gt_valid"] = cohens_kappa(
            [vlm_outcome(arms[A][k]) for k in vk],
            [vlm_outcome(arms[B][k]) for k in vk])

        # ---- paired tests on the GT-valid windows ---------------------------
        eid = [k[0] for k in vk]
        gt = [wins[k]["kin_v21"]["route"] for k in vk]
        a3 = [vlm_outcome(arms[A][k]) for k in vk]
        b3 = [vlm_outcome(arms[B][k]) for k in vk]
        res["paired_tests"] = {"arm_a": A, "arm_b": B, "n_episodes": len(set(eid)),
                               "tests": []}
        res["paired_tests"]["tests"].append(paired_block(
            "acc3_over_all", [int(x == g) for x, g in zip(a3, gt)],
            [int(x == g) for x, g in zip(b3, gt)], eid, args.seed))
        res["paired_tests"]["tests"].append(paired_block(
            "turn_detected", *[[int(p in ("left", "right", "u_turn"))
                                for p, g in zip(pp, gt) if g in ("left", "right")]
                               for pp in (a3, b3)],
            [e for e, g in zip(eid, gt) if g in ("left", "right")], args.seed))
        # direction correctness on GT turns, scored over ALL GT turns so the two
        # arms share a denominator (a detection-conditioned denominator differs
        # per arm and is therefore NOT pairable)
        res["paired_tests"]["tests"].append(paired_block(
            "direction_correct_over_all_gt_turns",
            *[[int(p == g) for p, g in zip(pp, gt) if g in ("left", "right")]
              for pp in (a3, b3)],
            [e for e, g in zip(eid, gt) if g in ("left", "right")], args.seed))

        # ---- single-arm intervals, same estimator ---------------------------
        turn_eid = [e for e, g in zip(eid, gt) if g in ("left", "right")]
        res["single_arm_ci"] = {}
        for t, pp in ((A, a3), (B, b3)):
            res["single_arm_ci"][t] = {
                "acc3_over_all": _boot(
                    [int(x == g) for x, g in zip(pp, gt)], eid, args.seed),
                "direction_correct_over_all_gt_turns": _boot(
                    [int(p == g) for p, g in zip(pp, gt)
                     if g in ("left", "right")], turn_eid, args.seed)}

    txt = json.dumps(res, indent=1)
    if args.json:
        open(args.json, "w").write(txt)
        print(f"wrote {args.json}")
    print(txt if not args.json else json.dumps(
        {k: v for k, v in res.items() if k != "arms"}, indent=1))


if __name__ == "__main__":
    main()
