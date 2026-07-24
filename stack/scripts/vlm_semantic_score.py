"""Score / QA a semantic labeling run from `vlm_semantic_labels.py`. NO POD, NO GPU.

Everything this needs travels ON the records: the kinematic v2.1 route label is
stamped inline, the enum table is written beside them as `enums.json`, and the
exact prompt text is written as `prompt_A.txt` / `prompt_B.txt`. So the run can
be re-scored from the repo forever, and the enum table can never drift from what
was actually asked (a hand-copied or re-imported vocabulary would).

WHAT IS AND IS NOT ADMISSIBLE HERE
  * **ROUTE is a CROSS-CHECK, never a label.** Reason2's direction call is at
    chance (57.1 %, episode-cluster CI [0.400, 0.745], Fisher OR 1.32 p=0.78) and
    its errors are correlated toward `left`. Every ROUTE number below exists to
    monitor that finding, not to mint anything.
  * **Only Pass A ROUTE may enter agreement statistics.** Pass B is shown the
    numeric future ego track, so its ROUTE is downstream of our own kinematics
    and agreeing with them measures nothing. `passb_slots` scores Pass B for
    SCHEMA BEHAVIOUR only and says so on every emission.
  * **Scenario slots have NO ground truth** — that is the entire reason the
    labeling run exists. Nothing here is called accuracy. What IS measurable
    without truth, and is reported instead: schema adherence, informativeness,
    event-time validity (is the emitted offset one we actually offered?),
    cross-pass reproducibility of the geometry call, and a weak KINEMATIC
    CO-OCCURRENCE check. Read §`scenario_kinematic_crosscheck`'s own caveat
    before quoting it: a junction traversed straight is a correct `junction`
    label with no turn signature, so disagreement there is not error.

Usage:
  python3 vlm_semantic_score.py --out <run dir> --arms val_full --json r.json
  python3 vlm_semantic_score.py --out <run dir> --arms base,dense_early --compare
  python3 vlm_semantic_score.py --out <run dir> --arms val_full --to-jsonl <f>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_compare_score import (_boot, _rate, cohens_kappa,   # noqa: E402
                               fisher_exact_2x2, mcnemar)

CLS3 = ("left", "straight", "right")
# the `road_geometry` enum as shipped; MEASURED bleed into this slot: Pass A
# answers ROUTE tokens (`left`/`right`) ~3 % of the time and Pass B answers
# `intersection` (a `road_type` token). Neither may become a scenario stratum.
GEOM_ENUM = ("straight", "curve_left", "curve_right", "junction",
             "roundabout", "merge", "fork", "unknown")
BANDS = ("high", "medium", "low")
NGRAM = 6                       # words; long enough that overlap means copying


# --------------------------------------------------------------------- load
def load_arm(out_dir: str, tag: str) -> dict:
    """{(episode, t): record}. Accepts the pod's per-window JSON directory OR
    the consolidated `<tag>.jsonl` that is what actually gets committed."""
    recs = {}
    jl = os.path.join(out_dir, tag + ".jsonl")
    if os.path.isfile(jl):
        with open(jl, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    recs[(r["episode"], int(r["t"]))] = r
        return recs
    for f in sorted(glob.glob(os.path.join(out_dir, tag, "ep_*_t*.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        recs[(r["episode"], int(r["t"]))] = r
    return recs


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _b_parsed(b: dict):
    """(Pass-B object, mode). Prefers the STRICT parse; falls back to the
    salvaged prefix of a truncated reply.

    Truncation is a tail failure — the object is emitted in schema order and one
    late free-text field runs away — so a truncated record still carries an
    intact `SCENARIO` block, the very block the scenario metrics are blocked on.
    Every rate that uses this reports how many rows came from salvage, so a
    reader can always subtract them."""
    if not isinstance(b, dict):
        return {}, "none"
    strict = b.get("parsed") or {}
    if strict:
        return strict, "complete"
    salv = b.get("parsed_salvaged") or {}
    if salv:
        return salv, b.get("parse_mode") or "partial"
    # Records written BEFORE the salvage existed carry no `parsed_salvaged`,
    # only `raw`. Re-deriving it here means the fix applies retroactively to
    # every record we already own instead of only to future runs. Lazy + guarded
    # because the salvage lives next to the harness (which imports torch) and
    # this scorer must keep running where torch does not.
    raw = b.get("raw")
    if raw:
        try:
            import vlm_semantic_labels as _VS
        except Exception:
            return {}, b.get("parse_mode") or "none"
        obj, mode = _VS.salvage_json(raw)
        if mode == "partial" and obj:
            return obj, "partial_from_raw"
    return {}, b.get("parse_mode") or "none"


def _ngrams(text: str, n: int = NGRAM) -> set:
    w = re.findall(r"[a-z]+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


# ----------------------------------------------------------- PASS A defects
def _a_fields(a: dict) -> dict:
    """The Pass-A answer fields, whichever record layout they arrived in.

    `vlm_semantic_labels` nests the parsed reply under `parsed`;
    `vlm_model_compare` (the head-to-head / enum-order probe harness) writes the
    same field NAMES flat on `pass_A`. Accepting both lets one scorer read every
    record we own — which matters here because the probe's 200-window arm is a
    far better-powered v1 baseline for the evidence-contamination rate than the
    40-window prompt arm."""
    return a.get("parsed") or a


def pass_a_block(recs: dict, prompt_a: str | None) -> dict:
    """The four measured prompt defects + the ROUTE cross-check, in one place."""
    keys = sorted(recs)
    A = [recs[k].get("pass_A") or {} for k in keys]
    n = len(A)
    outc = {}
    for a in A:
        o = a.get("outcome", "missing")
        outc[o] = outc.get(o, 0) + 1

    # --- defect 1: truncation ------------------------------------------------
    trunc = sum(1 for a in A if a.get("truncated"))

    # --- defect 2: confidence ------------------------------------------------
    bands, floats = {}, []
    for a in A:
        js = _a_fields(a)
        v = js.get("route_confidence_band", js.get("route_confidence"))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            floats.append(float(v))
            bands["<float>"] = bands.get("<float>", 0) + 1
        else:
            t = str(v).strip().lower() if v is not None else "missing"
            bands[t] = bands.get(t, 0) + 1
    top = max(bands.values()) if bands else 0
    conf = {"distribution": dict(sorted(bands.items(), key=lambda x: -x[1])),
            "modal_share": _rate(top, n),
            "n_distinct": len(bands),
            "degenerate": bool(n and top / n >= 0.90),
            "float_values_seen": sorted(set(floats))[:10] or None,
            "rule": "degenerate = one answer on >=90 % of windows; such a "
                    "field cannot threshold anything and should be dropped"}

    # --- defect 3: evidence contamination ------------------------------------
    pset = _ngrams(prompt_a) if prompt_a else set()
    ev = [(_a_fields(a).get("route_evidence") or "") for a in A]
    is_turn = [(a.get("ROUTE") in ("left", "right")) for a in A]
    copied, frags = [], {}
    for e in ev:
        hit = sorted(_ngrams(e) & pset)
        copied.append(bool(hit))
        for h in hit[:1]:
            frags[h] = frags.get(h, 0) + 1
    n_turn = sum(is_turn)
    evidence = {
        "n_with_evidence": sum(1 for e in ev if e.strip()),
        "n_unique_strings": len({e.strip().lower() for e in ev if e.strip()}),
        "unique_rate": _rate(len({e.strip().lower() for e in ev if e.strip()}),
                             max(1, sum(1 for e in ev if e.strip()))),
        "prompt_copy_rate_all": _rate(sum(copied), n),
        "prompt_copy_rate_turn_calls": _rate(
            sum(1 for c, t in zip(copied, is_turn) if c and t), n_turn),
        "n_turn_calls": n_turn,
        "top_copied_fragments": dict(sorted(frags.items(),
                                            key=lambda x: -x[1])[:5]),
        "detector": f"{NGRAM}-word n-gram overlap with the shipped prompt text"}
    if prompt_a is None:
        evidence["prompt_copy_rate_all"] = None
        evidence["note"] = "prompt_A.txt absent — copy rate not computable"

    # --- event-time validity (new in v2) -------------------------------------
    ev_time = _event_time_block(
        [(recs[k], _a_fields(recs[k].get("pass_A") or {})) for k in keys],
        [("route_event_time_s",), ("geometry_event_time_s",)])

    # --- ROUTE cross-check ---------------------------------------------------
    gt = [recs[k].get("kin_v21") or {} for k in keys]
    pred = [(a.get("ROUTE") if a.get("outcome") == "ok" else "no_answer")
            for a in A]
    eid = [k[0] for k in keys]
    val = [i for i, g in enumerate(gt) if g.get("valid")]
    turn_i = [i for i in val if gt[i]["route"] in ("left", "right")]
    det = sum(1 for i in turn_i if pred[i] in ("left", "right", "u_turn"))
    dir_i = [i for i in turn_i if pred[i] in ("left", "right")]
    dir_ok = sum(1 for i in dir_i if pred[i] == gt[i]["route"])
    t11 = sum(1 for i in dir_i if gt[i]["route"] == "left" and pred[i] == "left")
    t12 = sum(1 for i in dir_i if gt[i]["route"] == "left" and pred[i] == "right")
    t21 = sum(1 for i in dir_i if gt[i]["route"] == "right" and pred[i] == "left")
    t22 = sum(1 for i in dir_i if gt[i]["route"] == "right" and pred[i] == "right")
    n_pl = sum(1 for i in val if pred[i] == "left")
    n_pr = sum(1 for i in val if pred[i] == "right")
    route = {
        "CAVEAT": "CROSS-CHECK ONLY — ROUTE stays kinematic; these numbers "
                  "monitor a closed finding and may never mint a label",
        "n_gt_valid": len(val), "n_gt_turn": len(turn_i),
        "turn_detection_recall": _rate(det, len(turn_i)),
        "direction_accuracy_given_detected": _rate(dir_ok, len(dir_i)),
        "direction_n_detected_lr": len(dir_i),
        "direction_ci": _boot([int(pred[i] == gt[i]["route"]) for i in dir_i],
                              [eid[i] for i in dir_i], 0),
        "acc3_over_all": _rate(sum(1 for i in val if pred[i] == gt[i]["route"]),
                               len(val)),
        "fisher": {"table_gt_left": {"said_left": t11, "said_right": t12},
                   "table_gt_right": {"said_left": t21, "said_right": t22},
                   "odds_ratio": (round((t11 * t22) / (t12 * t21), 3)
                                  if t12 and t21 else None),
                   "p_two_sided": round(fisher_exact_2x2(t11, t12, t21, t22), 5),
                   "acc_given_detected_gt_left": _rate(t11, t11 + t12),
                   "acc_given_detected_gt_right": _rate(t22, t21 + t22)},
        "left_share_of_predicted_turns": _rate(n_pl, n_pl + n_pr),
        "left_share_of_gt_turns": _rate(
            sum(1 for i in val if gt[i]["route"] == "left"), len(turn_i))}

    # --- confidence band vs correctness (is the band worth anything?) --------
    cal = {}
    for i in val:
        js = _a_fields(A[i])
        b = str(js.get("route_confidence_band",
                       js.get("route_confidence"))).strip().lower()
        d = cal.setdefault(b, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(pred[i] == gt[i]["route"])
    for d in cal.values():
        d["acc3"] = _rate(d["correct"], d["n"])
    route["accuracy_by_confidence_band"] = dict(
        sorted(cal.items(), key=lambda x: -x[1]["n"]))

    gens = [a.get("n_gen_tokens", 0) for a in A if a.get("n_gen_tokens")]
    secs = [a.get("gen_seconds", 0.0) for a in A if a.get("gen_seconds")]
    prm = [a.get("n_prompt_tokens", 0) for a in A if a.get("n_prompt_tokens")]
    return {
        "n_windows": n,
        "outcome_counts": outc,
        "parse_failure_rate": _rate(
            sum(v for k, v in outc.items()
                if k in ("no_json", "json_invalid", "missing_route_key",
                         "exception")), n),
        "enum_violation_rate": _rate(outc.get("enum_violation", 0), n),
        "truncation_rate": _rate(trunc, n),
        "confidence": conf,
        "evidence": evidence,
        "event_times": ev_time,
        "route_crosscheck": route,
        "prompt_tokens_mean": round(float(np.mean(prm)), 1) if prm else None,
        "gen_tokens_mean": round(float(np.mean(gens)), 1) if gens else None,
        "gen_tokens_p95": (round(float(np.percentile(gens, 95)), 1)
                           if gens else None),
        "gen_seconds_median": (round(float(np.median(secs)), 3)
                               if secs else None),
        "peak_vram_gib": max((r.get("peak_vram_gib", 0.0)
                              for r in recs.values()), default=0.0)}


# ------------------------------------------------------------- event times
def _event_time_block(pairs, paths) -> dict:
    """Is every emitted time offset one we ACTUALLY OFFERED?

    The whole defence of asking a VLM for a time is that it is not a
    measurement — it must COPY one of the frame offsets it was shown. An offset
    that is not in that list is a fabricated number and is counted as one."""
    out = {}
    for path in paths:
        name = ".".join(path)
        n = ok = null = bad = 0
        badvals = {}
        for rec, js in pairs:
            offered = set(rec.get("future_offsets_s") or [])
            v = _dig(js, path)
            n += 1
            if v is None:
                null += 1
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if float(v) in offered:
                    ok += 1
                else:
                    bad += 1
                    badvals[str(v)] = badvals.get(str(v), 0) + 1
            else:
                bad += 1
                badvals[str(v)[:20]] = badvals.get(str(v)[:20], 0) + 1
        out[name] = {"n": n, "in_offered_set_rate": _rate(ok, n),
                     "null_rate": _rate(null, n),
                     "fabricated_rate": _rate(bad, n),
                     "top_fabricated": dict(sorted(badvals.items(),
                                                   key=lambda x: -x[1])[:5])}
    return out


# ------------------------------------------------- PASS B: SCHEMA ADHERENCE
_PATHS = {
    "SCENARIO.road_type": ("SCENARIO", "road_type"),
    "SCENARIO.weather": ("SCENARIO", "environment", "weather"),
    "SCENARIO.time_of_day": ("SCENARIO", "environment", "time_of_day"),
    "SCENARIO.lighting": ("SCENARIO", "environment", "lighting"),
    "SCENARIO.surface": ("SCENARIO", "surface"),
    "SCENARIO.traffic_density": ("SCENARIO", "traffic_density"),
    "SCENARIO.road_geometry": ("SCENARIO", "road_geometry"),
    "SCENARIO.scenario_tag": ("SCENARIO", "scenario_tag"),
    "SCENARIO.difficulty": ("SCENARIO", "difficulty"),
    "ROUTE": ("STRATEGIC", "ROUTE"),
    "OBS.lead_lane": ("OBSERVATIONS", "lead_vehicle", "lane"),
    "OBS.distance_bucket": ("OBSERVATIONS", "lead_vehicle", "distance_bucket"),
    "OBS.relative_motion": ("OBSERVATIONS", "lead_vehicle", "relative_motion"),
    "OBS.markings": ("OBSERVATIONS", "lane_info", "markings"),
    "OBS.lane_type": ("OBSERVATIONS", "lane_info", "lane_type"),
    "OBS.light_state": ("OBSERVATIONS", "traffic_light", "state"),
}
for _k in ("MISSION", "LANEOBJ", "SPEEDPOLICY", "STYLE", "RISK", "ODD"):
    _PATHS[f"STRATEGIC.{_k}"] = ("STRATEGIC", _k)
for _k in ("LATMANEUVER", "LONMODE", "VSOURCE", "HEADWAY", "DYN", "RULECTX",
           "TACPOINT", "LIGHTSTATE", "INTERACT", "SIGNAL"):
    _PATHS[f"TACTICAL.{_k}"] = ("TACTICAL", _k)


def pass_b_block(recs: dict, enums: dict) -> dict:
    """Per-slot in-vocab / violation / unknown / missing. SCHEMA ONLY."""
    keys = sorted(recs)
    n = len(keys)
    slots, parsed_ok, err, trunc = {}, 0, 0, 0
    gens, secs, prm = [], [], []
    asked = set()
    modes = {}
    for k in keys:
        b = recs[k].get("pass_B") or {}
        js, mode = _b_parsed(b)
        modes[mode] = modes.get(mode, 0) + 1
        parsed_ok += bool(b.get("parsed"))
        err += bool(b.get("error"))
        trunc += bool(b.get("truncated"))
        if b.get("n_gen_tokens"):
            gens.append(b["n_gen_tokens"])
        if b.get("gen_seconds"):
            secs.append(b["gen_seconds"])
        if b.get("n_prompt_tokens"):
            prm.append(b["n_prompt_tokens"])
        na = set(recs[k].get("not_asked") or ())
        for slot, path in _PATHS.items():
            if slot.split(".")[-1] in na:
                continue
            asked.add(slot)
            allowed = enums.get(slot) or enums.get(slot.split(".", 1)[-1])
            if allowed is None:
                continue
            d = slots.setdefault(slot, {"in_vocab": 0, "violation": 0,
                                        "unknown": 0, "missing": 0,
                                        "bad": {}})
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
                d["bad"][str(v)[:40]] = d["bad"].get(str(v)[:40], 0) + 1
    tbl = {}
    for s, d in sorted(slots.items()):
        ans = d["in_vocab"] + d["unknown"] + d["violation"]
        tbl[s] = {"in_vocab_rate": _rate(d["in_vocab"] + d["unknown"],
                                         max(ans, 1)),
                  "violation_rate": _rate(d["violation"], max(ans, 1)),
                  "answered_rate": _rate(ans, n),
                  "informative_rate": _rate(d["in_vocab"], n),
                  "n_unknown": d["unknown"], "n_missing": d["missing"],
                  "bad_tokens": dict(sorted(d["bad"].items(),
                                            key=lambda x: -x[1])[:5])}
    ev = _event_time_block(
        [(recs[k], _b_parsed(recs[k].get("pass_B") or {})[0]) for k in keys],
        [("SCENARIO", "geometry_event_time_s"),
         ("SCENARIO", "geometry_event_end_time_s"),
         ("SCENARIO", "scenario_event_time_s")])
    viol = [v["violation_rate"] or 0.0 for v in tbl.values()]
    return {"CAVEAT": "SCHEMA ADHERENCE ONLY — Pass B sees the numeric future "
                      "ego track; nothing here is accuracy and ROUTE here is "
                      "not a route measurement",
            "n_windows": n,
            "json_parse_rate_strict": _rate(parsed_ok, n),
            "json_usable_rate_incl_salvage": _rate(
                sum(v for k, v in modes.items() if k.startswith(("complete",
                                                                 "partial"))), n),
            "parse_mode_counts": modes,
            "n_rows_from_salvage": sum(v for k, v in modes.items()
                                       if k.startswith("partial")),
            "error_rate": _rate(err, n),
            "truncation_rate": _rate(trunc, n),
            "prompt_tokens_mean": round(float(np.mean(prm)), 1) if prm else None,
            "gen_tokens_mean": round(float(np.mean(gens)), 1) if gens else None,
            "gen_tokens_p95": (round(float(np.percentile(gens, 95)), 1)
                               if gens else None),
            "gen_seconds_median": (round(float(np.median(secs)), 2)
                                   if secs else None),
            "n_slots": len(tbl),
            "mean_slot_violation_rate": round(float(np.mean(viol)), 4)
            if viol else None,
            "mean_informative_rate": round(float(np.mean(
                [v["informative_rate"] or 0.0 for v in tbl.values()])), 4)
            if tbl else None,
            "event_times": ev,
            "slots": tbl}


# ------------------------------------------------------ SCENARIO usefulness
GEOM_EVENTFUL = ("junction", "roundabout", "merge", "fork")


def scenario_block(recs: dict) -> dict:
    """What the labels are FOR: are the scenario strata populated and coherent?

    There is no ground truth for any of this — producing it is the point of the
    run. So the report is: coverage, distribution, the two reproducibility
    checks that need no truth, and one weak kinematic co-occurrence check with
    its caveat attached."""
    keys = sorted(recs)
    n = len(keys)
    ga, gb, tags, diff = [], [], {}, {}
    for k in keys:
        a = _a_fields(recs[k].get("pass_A") or {})
        b = _b_parsed(recs[k].get("pass_B") or {})[0]
        ga.append(str(a.get("road_geometry", "missing")).strip().lower())
        gb.append(str(_dig(b, ("SCENARIO", "road_geometry"))
                      or "missing").strip().lower())
        t = str(_dig(b, ("SCENARIO", "scenario_tag")) or "missing").lower()
        tags[t] = tags.get(t, 0) + 1
        d = str(_dig(b, ("SCENARIO", "difficulty")) or "missing").lower()
        diff[d] = diff.get(d, 0) + 1
    dist_a, dist_b = {}, {}
    for x in ga:
        dist_a[x] = dist_a.get(x, 0) + 1
    for x in gb:
        dist_b[x] = dist_b.get(x, 0) + 1

    # cross-pass reproducibility: Pass A has no future track, Pass B does.
    # Agreement is not accuracy, but a geometry call that flips when the ego
    # track is added is not a stable label either.
    both = [(x, y) for x, y in zip(ga, gb)
            if x not in ("missing", "") and y not in ("missing", "")]
    kap = (cohens_kappa([x for x, _ in both], [y for _, y in both])
           if both else {"kappa": None})

    # weak kinematic co-occurrence, in BOTH directions
    ev_idx = [i for i, k in enumerate(keys)
              if gb[i] in GEOM_EVENTFUL or ga[i] in GEOM_EVENTFUL]
    kin_turn = []
    for i, k in enumerate(keys):
        g = recs[k].get("kin_v21") or {}
        kin_turn.append(bool(g.get("valid")
                             and g.get("route") in ("left", "right")))
    n_ev = len(ev_idx)
    n_turn = sum(kin_turn)
    both_n = sum(1 for i in ev_idx if kin_turn[i])
    cross = {
        "n_vlm_eventful_geometry": n_ev,
        "vlm_eventful_rate": _rate(n_ev, n),
        "n_kinematic_turn": n_turn,
        "kinematic_turn_rate": _rate(n_turn, n),
        "P(kinematic turn | VLM says junction/roundabout/merge/fork)":
            _rate(both_n, n_ev),
        "P(VLM says junction/roundabout/merge/fork | kinematic turn)":
            _rate(both_n, n_turn),
        "fisher_p": round(fisher_exact_2x2(
            both_n, n_ev - both_n, n_turn - both_n,
            n - n_ev - n_turn + both_n), 6),
        "CAVEAT": "NOT an accuracy check. A junction traversed STRAIGHT is a "
                  "correct `junction` label with no turn signature, and a 2 s "
                  "kinematic window cannot see a 5-20 s event at all. A "
                  "significant association is weak positive evidence that the "
                  "geometry call tracks reality; its absence would not falsify "
                  "the labels."}

    # event-offset reach: how far past the 2 s planning window do labels point?
    reach = {}
    for k in keys:
        b = _b_parsed(recs[k].get("pass_B") or {})[0]
        v = _dig(b, ("SCENARIO", "geometry_event_time_s"))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            band = ("<=2s" if v <= 2 else "2-5s" if v <= 5 else
                    "5-10s" if v <= 10 else "10-20s")
            reach[band] = reach.get(band, 0) + 1
    return {
        "n_windows": n,
        "road_geometry_passA": dict(sorted(dist_a.items(), key=lambda x: -x[1])),
        "road_geometry_passB": dict(sorted(dist_b.items(), key=lambda x: -x[1])),
        "scenario_tag_passB": dict(sorted(tags.items(), key=lambda x: -x[1])),
        "difficulty_passB": dict(sorted(diff.items(), key=lambda x: -x[1])),
        "cross_pass_geometry_agreement": kap,
        "geometry_event_time_reach": dict(sorted(reach.items())),
        "scenario_kinematic_crosscheck": cross,
        "strata_populated": {k: v for k, v in sorted(
            dist_b.items(), key=lambda x: -x[1]) if k not in ("missing",)},
    }


# ------------------------------------------------------------ arm comparison
def compare(arms: dict, enums: dict) -> dict:
    """Paired comparison across arms on the SHARED windows (frame ablation)."""
    tags = list(arms)
    keys = sorted(set.intersection(*[set(a) for a in arms.values()]))
    out = {"n_shared_windows": len(keys),
           "n_episodes": len({k[0] for k in keys}), "arms": tags,
           "cost": {}, "route": {}, "slot_agreement": {}}
    gt = [(arms[tags[0]][k].get("kin_v21") or {}) for k in keys]
    eid = [k[0] for k in keys]
    for t in tags:
        A = [arms[t][k].get("pass_A") or {} for k in keys]
        B = [arms[t][k].get("pass_B") or {} for k in keys]
        r0 = arms[t][keys[0]]
        gA = [a.get("n_gen_tokens", 0) for a in A if a.get("n_gen_tokens")]
        pA = [a.get("n_prompt_tokens", 0) for a in A if a.get("n_prompt_tokens")]
        sA = [a.get("gen_seconds", 0) for a in A if a.get("gen_seconds")]
        sB = [b.get("gen_seconds", 0) for b in B if b.get("gen_seconds")]
        # frame counts must be MEANS, not the first window's: a window at t=0
        # has no history and a late-clip window has no 20 s future, so any
        # single record understates the plan (base is 3+5 images by definition
        # but 1+4 at t=0). The plan's declared offsets are reported separately.
        n_img = [arms[t][k].get("n_images") or 0 for k in keys]
        n_fut = [arms[t][k].get("n_future_frames") or 0 for k in keys]
        out["cost"][t] = {
            "frames_plan": r0.get("frames_plan"), "image_px": r0.get("image_px"),
            "n_images_mean": round(float(np.mean(n_img)), 2) if n_img else None,
            "n_future_frames_mean": (round(float(np.mean(n_fut)), 2)
                                     if n_fut else None),
            "n_future_frames_max": max(n_fut) if n_fut else None,
            "hist_offsets_s_example": r0.get("hist_offsets_s"),
            "future_offsets_s_example": r0.get("future_offsets_s"),
            "promptA_tokens_mean": round(float(np.mean(pA)), 1) if pA else None,
            "genA_tokens_mean": round(float(np.mean(gA)), 1) if gA else None,
            "genA_seconds_median": round(float(np.median(sA)), 3) if sA else None,
            "genB_seconds_median": round(float(np.median(sB)), 3) if sB else None,
            "peak_vram_gib": max((arms[t][k].get("peak_vram_gib", 0.0)
                                  for k in keys), default=0.0),
            "passA_parse_fail": _rate(
                sum(1 for a in A if a.get("outcome") not in ("ok", None)), len(A)),
            # None, not 0.0 — a Pass-A-only ablation arm has no truncation rate,
            # and a zero here would read as "this plan never truncates"
            "passB_truncation": (_rate(sum(1 for b in B if b.get("truncated")),
                                       sum(1 for b in B if b))
                                 if any(B) else None),
        }
        val = [i for i, g in enumerate(gt) if g.get("valid")]
        pr = [(a.get("ROUTE") if a.get("outcome") == "ok" else "no_answer")
              for a in A]
        ti = [i for i in val if gt[i]["route"] in ("left", "right")]
        di = [i for i in ti if pr[i] in ("left", "right")]
        out["route"][t] = {
            "turn_detection_recall": _rate(
                sum(1 for i in ti if pr[i] in ("left", "right", "u_turn")),
                len(ti)),
            "direction_accuracy_given_detected": _rate(
                sum(1 for i in di if pr[i] == gt[i]["route"]), len(di)),
            "n_detected_lr": len(di),
            "acc3_over_all": _rate(
                sum(1 for i in val if pr[i] == gt[i]["route"]), len(val)),
            "acc3_ci": _boot([int(pr[i] == gt[i]["route"]) for i in val],
                             [eid[i] for i in val], 0),
            "left_share_of_turn_calls": _rate(
                sum(1 for i in val if pr[i] == "left"),
                sum(1 for i in val if pr[i] in ("left", "right")))}

    # paired ROUTE correctness between the first two arms
    if len(tags) >= 2:
        a1, a2 = tags[0], tags[1]
        val = [i for i, g in enumerate(gt) if g.get("valid")]
        ok = {}
        for t in (a1, a2):
            A = [arms[t][k].get("pass_A") or {} for k in keys]
            pr = [(a.get("ROUTE") if a.get("outcome") == "ok" else "no_answer")
                  for a in A]
            ok[t] = [int(pr[i] == gt[i]["route"]) for i in val]
        out["paired_route_acc3"] = {
            "arm_a": a1, "arm_b": a2,
            "mean_a": round(float(np.mean(ok[a1])), 4) if ok[a1] else None,
            "mean_b": round(float(np.mean(ok[a2])), 4) if ok[a2] else None,
            "mcnemar": mcnemar(ok[a1], ok[a2]),
            "n": len(val), "n_episodes": len({eid[i] for i in val})}

    # per-slot agreement between arms — the only reliability signal available
    # where there is no ground truth: a slot whose answer changes when the
    # frame plan changes is not a stable label.
    if len(tags) >= 2:
        a1, a2 = tags[0], tags[1]
        for slot, path in sorted(_PATHS.items()):
            if slot not in enums and slot.split(".", 1)[-1] not in enums:
                continue
            x, y = [], []
            for k in keys:
                p1 = _b_parsed(arms[a1][k].get("pass_B") or {})[0]
                p2 = _b_parsed(arms[a2][k].get("pass_B") or {})[0]
                v1, v2 = _dig(p1, path), _dig(p2, path)
                if isinstance(v1, str) and isinstance(v2, str):
                    x.append(v1.lower())
                    y.append(v2.lower())
            if len(x) >= 20:
                out["slot_agreement"][slot] = cohens_kappa(x, y)
        pa1 = [str(_a_fields(arms[a1][k].get("pass_A") or {})
                   .get("road_geometry", "")).lower() for k in keys]
        pa2 = [str(_a_fields(arms[a2][k].get("pass_A") or {})
                   .get("road_geometry", "")).lower() for k in keys]
        out["slot_agreement"]["passA.road_geometry"] = cohens_kappa(pa1, pa2)
    return out


# ---------------------------------------------------------------------- main
def to_jsonl(recs: dict, path: str, drop_raw: bool = False) -> int:
    """Consolidate a per-window directory into the form the repo carries.

    `--drop-raw` strips the stored model reply. Keep it ON for the production
    corpus (it is ~4x the size) and OFF for anything a defect has to be
    re-measured from: the evidence copy-rate and every failure taxonomy are
    computed from `raw`."""
    with open(path, "w", encoding="utf-8") as fh:
        for k in sorted(recs):
            r = dict(recs[k])
            if drop_raw:
                for p in ("pass_A", "pass_B"):
                    if isinstance(r.get(p), dict):
                        r[p] = {kk: vv for kk, vv in r[p].items() if kk != "raw"}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(recs)


def audit_sheet(recs: dict, path: str, n: int = 60, seed: int = 0) -> int:
    """A stratified spot-check sheet for a HUMAN.

    Nothing in this file has ground truth, so schema adherence and
    reproducibility are the only automatic evidence available — and neither can
    tell us whether `junction` means a junction. The cheapest thing that can is
    a person looking at 60 rows. This writes them: over-sampling the eventful
    geometries (which are rare and are the ones the metric suite needs), with
    the model's own evidence sentence beside each so the check is a read, not
    an investigation.
    """
    import random as _r
    rows = []
    for k in sorted(recs):
        r = recs[k]
        b, bmode = _b_parsed(r.get("pass_B") or {})
        a = _a_fields(r.get("pass_A") or {})
        offered = set(r.get("future_offsets_s") or [])

        def _t(v):
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and float(v) in offered else None

        # The sheet must show what we SHIP, or a human verifies the wrong thing.
        # Per §2's doctrine fix the shipped geometry and event time are PASS A's
        # (independent); Pass B's are kept beside them for comparison and are
        # marked, because they are downstream of our own kinematics.
        raw_a = str(a.get("road_geometry") or "").lower()
        geom_a = raw_a if raw_a in GEOM_ENUM else "unknown"
        raw_b = str(_dig(b, ("SCENARIO", "road_geometry")) or "").lower()
        rows.append({
            "episode": k[0], "t": k[1], "episode_id": r.get("episode_id"),
            "val_build": r.get("val_build"), "parse_mode": bmode,
            "SHIPPED_geometry": geom_a,
            "shipped_geom_out_of_enum": raw_a if raw_a and geom_a != raw_a else "",
            "SHIPPED_event_t_s": _t(a.get("geometry_event_time_s")),
            "shipped_geom_band": str(a.get("road_geometry_confidence_band") or ""),
            "scenario_tag": str(_dig(b, ("SCENARIO", "scenario_tag")) or ""),
            "passB_geometry_CONTAMINATED": raw_b,
            "passB_event_t_s_CONTAMINATED": _dig(
                b, ("SCENARIO", "geometry_event_time_s")),
            "kin_route": (r.get("kin_v21") or {}).get("route"),
            "kin_net_dyaw_deg": (r.get("kin_v21") or {}).get("net_dyaw_deg"),
            "route_evidence_A": (a.get("route_evidence") or "")[:180],
            "coc_observation": str(_dig(b, ("COC", "observation")) or "")[:180],
            "HUMAN_VERDICT_geometry": "", "HUMAN_VERDICT_tag": "",
            "HUMAN_NOTE": ""})
    rare = [r for r in rows if r["SHIPPED_geometry"] in GEOM_EVENTFUL
            or r["passB_geometry_CONTAMINATED"] in GEOM_EVENTFUL]
    rest = [r for r in rows if r not in rare]
    rng = _r.Random(seed)
    rng.shuffle(rare)
    rng.shuffle(rest)
    take = rare[:max(1, n // 2)] + rest[:n - len(rare[:max(1, n // 2)])]
    take.sort(key=lambda x: (x["episode"], x["t"]))
    cols = list(take[0].keys()) if take else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in take:
            fh.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ")
                               for c in cols) + "\n")
    return len(take)


def main():
    ap = argparse.ArgumentParser("vlm_semantic_score")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--to-jsonl", default=None,
                    help="write <arm>.jsonl beside --out and exit")
    ap.add_argument("--audit-sheet", default=None,
                    help="write a stratified TSV spot-check sheet for a human "
                         "(the only way to settle what schema metrics cannot)")
    ap.add_argument("--audit-n", type=int, default=60)
    ap.add_argument("--drop-raw", action="store_true")
    args = ap.parse_args()

    tags = [t.strip() for t in args.arms.split(",") if t.strip()]
    arms = {t: load_arm(args.out, t) for t in tags}
    for t, r in arms.items():
        if not r:
            raise SystemExit(f"arm {t!r} has no records under {args.out}")

    if args.to_jsonl:
        for t in tags:
            p = (args.to_jsonl if len(tags) == 1
                 else os.path.join(os.path.dirname(args.to_jsonl) or ".",
                                   t + ".jsonl"))
            n = to_jsonl(arms[t], p, args.drop_raw)
            print(f"wrote {p}: {n} records")
        return

    if args.audit_sheet:
        for t in tags:
            p = (args.audit_sheet if len(tags) == 1
                 else os.path.join(os.path.dirname(args.audit_sheet) or ".",
                                   t + "_audit.tsv"))
            n = audit_sheet(arms[t], p, args.audit_n)
            print(f"wrote {p}: {n} rows for human verification")
        return

    epath = os.path.join(args.out, "enums.json")
    enums = json.load(open(epath)) if os.path.exists(epath) else {}
    if not enums:
        print("WARNING: enums.json missing — Pass B slot adherence unavailable",
              file=sys.stderr)

    res = {"out_dir": os.path.abspath(args.out), "arms": tags,
           "enums_source": epath if enums else None}
    if args.compare:
        res["comparison"] = compare(arms, enums)
    for t in tags:
        r0 = next(iter(arms[t].values()))
        ppath = os.path.join(args.out, t, "prompt_A.txt")
        prompt_a = (open(ppath, encoding="utf-8").read()
                    if os.path.exists(ppath) else None)
        has_b = any((v.get("pass_B") or {}) for v in arms[t].values())
        blk = {"n_records": len(arms[t]),
               "prompt_version": r0.get("prompt_version"),
               "model": r0.get("model"), "val_build": r0.get("val_build"),
               "frames_plan": r0.get("frames_plan"),
               "image_px": r0.get("image_px"),
               "enum_order": r0.get("enum_order"),
               "n_episodes": len({k[0] for k in arms[t]}),
               "pass_A": pass_a_block(arms[t], prompt_a)}
        if has_b:
            blk["pass_B"] = pass_b_block(arms[t], enums)
            blk["scenario"] = scenario_block(arms[t])
        res[t] = blk

    txt = json.dumps(res, indent=1)
    if args.json:
        open(args.json, "w", encoding="utf-8").write(txt)
        print(f"wrote {args.json}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
