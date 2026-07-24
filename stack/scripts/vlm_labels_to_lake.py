"""Convert a semantic-labeling run into the two shapes the rest of the stack
can actually consume. NO POD, NO GPU, stdlib only.

WHY THIS EXISTS. `vlm_semantic_labels.py` emits one record per WINDOW. Nothing
downstream wants that shape:

  * `stack/tanitad/lake/enrich.py` defines EPISODE-level sidecar skeletons
    (`vlm_pending_scene_tags`, `vlm_pending_lead_state`, `vlm_pending_sign_reads`,
    `vlm_pending_language`) which the deferred Cosmos pass was always meant to
    fill IN PLACE. Scene tags genuinely are episode properties — weather does not
    change mid-clip — so they aggregate.
  * TanitEval v2's scenario strata (§3.5 S2) want the opposite: a per-WINDOW row
    keyed `(episode, t)`, because a 2 s window is the unit every metric is
    computed on and one clip contains both a straight stretch and a junction.

So this writes both, from the same records, and refuses to blur the difference.

THE METRIC-FIELD REFUSAL, IN CODE. `vlm_pending_lead_state()` has three metric
slots — `gap_m`, `closing_speed_ms`, `ttc_s`. The 48-clip pilot measured this
model fabricating band edges on 48 % of clips, so it is never asked for them and
this converter **leaves all three `None`** with `_metric_fields_unavailable`
stating why. What it does fill is the coarse categorical lead state
(`lane` / `distance_bucket` / `relative_motion`), which is enough to STRATIFY a
metric and is not enough to compute a TTC. A downstream consumer that wants
headway in seconds must still be refused, and now it is refused by the data.

Usage:
  python3 vlm_labels_to_lake.py --jsonl <run>.jsonl \
      --sidecars-out <dir> --windows-out <file>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

UNKNOWN = "unknown"
# TanitEval keys a window by its START index and the model observes
# `[start, start+TANITEVAL_WINDOW)` before predicting, so its "now" is
# `start + TANITEVAL_WINDOW` (`taniteval/rollout.py::collect(window=8,
# stride=8)`). These labels are keyed by "now" — the index `ego_block` and
# `route_from_future_v21` are evaluated at. Emitting the converted key stops
# every future consumer from re-deriving an off-by-eight, and the labeling
# stride (40) is a multiple of the eval stride (8), so the grids do line up.
TANITEVAL_WINDOW = 8
SCENE_AXES = ("weather", "time_of_day", "road_type", "surface",
              "traffic_density")
VRU_TYPES = ("pedestrian", "cyclist", "motorcycle", "animal")
# Geometries that name a multi-second event rather than a shape of the road
EVENTFUL_GEOM = ("junction", "roundabout", "merge", "fork")
# The `road_geometry` enum exactly as shipped in the prompt. MEASURED need: on
# 100 windows the model answered `left` twice and `right` once in this slot —
# ROUTE's vocabulary reaching into the geometry slot, the same "wrong slot's
# tokens" failure the head-to-head found on Reason1. Without this check an
# out-of-vocabulary token would enter the scenario strata as if it were a
# geometry, so the converter validates and COUNTS instead of passing it on.
GEOM_ENUM = ("straight", "curve_left", "curve_right", "junction",
             "roundabout", "merge", "fork", "unknown")


def _dig(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def last_balanced_json(s: str):
    """The last balanced {...} in a reply, or None. Cosmos-Reason emits a
    reasoning preamble, so the answer is the final object."""
    depth, start, best = 0, None, None
    for i, c in enumerate(s or ""):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                best = s[start:i + 1]
    if best is None:
        return None
    for cand in (best, re.sub(r",\s*([}\]])", r"\1", best)):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def salvage_json(s: str):
    """(object, mode) from a possibly TRUNCATED reply — complete/partial/none.

    Lives HERE, in the stdlib-only module, so both the GPU harness and this
    converter use one implementation. Truncation was measured at 32.5 % of
    Pass-B windows at a 2200-token budget and got *worse* at 3500 (61.5 %): the
    model spends whatever ceiling it is given, and the failure is always the
    same shape — the object is emitted correctly in schema order and then one
    free-text field near the end runs away. So a truncated record still carries
    an intact `SCENARIO` block, which is exactly what the scenario metrics are
    blocked on. Closing the open braces at the last safe boundary returns what
    was actually completed; the caller is always told it is `partial`.
    """
    strict = last_balanced_json(s)
    if strict is not None:
        return strict, "complete"
    i = (s or "").find("{")
    if i < 0:
        return {}, "none"
    buf = s[i:]
    instr = esc = False
    stack, safe = [], []
    for j, c in enumerate(buf):
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack:
                stack.pop()
            safe.append((j + 1, tuple(stack)))
        elif c == ",":
            safe.append((j, tuple(stack)))          # cut BEFORE the comma
    for cut, st in reversed(safe):
        if not st:
            continue
        try:
            obj = json.loads(buf[:cut] + "".join(reversed(st)))
        except Exception:
            continue
        if isinstance(obj, dict) and obj:
            return obj, "partial"
    return {}, "none"


def _b_parsed(b: dict):
    """(Pass-B object, parse mode). Strict parse first, salvaged prefix second.

    A truncated reply is a TAIL failure — the object comes out in schema order
    and one late free-text field runs away — so `SCENARIO` survives. Dropping
    the row would discard a correct geometry label because an unrelated later
    field rambled. Every emitted row carries `parse_mode`, so a consumer that
    wants strict-only can filter and one that wants coverage does not have to."""
    if not isinstance(b, dict):
        return {}, "none"
    if b.get("parsed"):
        return b["parsed"], "complete"
    if b.get("parsed_salvaged"):
        return b["parsed_salvaged"], b.get("parse_mode") or "partial"
    # Records written BEFORE the salvage existed carry only `raw`. Re-deriving
    # here applies the fix retroactively to every record we already own.
    if b.get("raw"):
        obj, mode = salvage_json(b["raw"])
        if mode == "partial" and obj:
            return obj, "partial_from_raw"
    return {}, b.get("parse_mode") or "none"


def _tok(v):
    return v.strip().lower() if isinstance(v, str) and v.strip() else None


def _mode(vals, allow_unknown=False):
    """Most common non-null token, plus the share that agreed.

    The share is not decoration: an episode whose windows split 3/2 on `weather`
    has not measured the weather, and a consumer needs to see that rather than
    inherit a coin flip as a fact."""
    v = [x for x in vals if x and (allow_unknown or x != UNKNOWN)]
    if not v:
        return UNKNOWN, 0.0, 0
    c = Counter(v).most_common()
    return c[0][0], round(c[0][1] / len(v), 3), len(v)


# ------------------------------------------------------------ per-window rows
def window_rows(recs: list) -> list:
    """One row per (episode, t) — the shape TanitEval v2's scenario strata want."""
    rows = []
    for r in recs:
        b, bmode = _b_parsed(r.get("pass_B") or {})
        a = (r.get("pass_A") or {}).get("parsed") or (r.get("pass_A") or {})
        sc, obs = b.get("SCENARIO") or {}, b.get("OBSERVATIONS") or {}
        lead = obs.get("lead_vehicle") or {}
        offered = r.get("future_offsets_s") or []

        def _t(v):
            # an offset we never showed is a fabricated number, not a reading
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and float(v) in set(offered) else None

        # PRIMARY = PASS A. Pass B is shown a plain-language reading of our own
        # numeric future ego track, and it was MEASURED parroting it back: its
        # `geometry_event_time_s` reproduces our kinematic onset to the decimal
        # (11.9 -> 11.9, 15.4 -> 15.4, 11.0 -> 11.0, 14.4 -> 14.4), so 38-69 % of
        # its "event times" are not in the offered frame-offset set at all.
        # Pass A, which never sees that block, is 100 % compliant. The same
        # doctrine that quarantines Pass B's ROUTE therefore applies to every
        # Pass-B quantity that our conditioning could have supplied — including
        # `road_geometry`, which "junction" is an easy read off "turns left 78
        # deg". A stratum used to judge a model against our kinematics must not
        # be derived from those kinematics.
        raw_geom_a = _tok(a.get("road_geometry"))
        raw_geom_b = _tok(sc.get("road_geometry"))
        geom_a = raw_geom_a if raw_geom_a in GEOM_ENUM else UNKNOWN
        geom_b = raw_geom_b if raw_geom_b in GEOM_ENUM else UNKNOWN
        geom_a_violation = (raw_geom_a is not None
                            and raw_geom_a not in GEOM_ENUM)
        rows.append({
            "episode": r.get("episode"), "t": r.get("t"),
            "episode_id": r.get("episode_id"), "val_build": r.get("val_build"),
            # the join key TanitEval actually uses; null where no eval window
            # can exist (t=0 has no observation history to key off)
            "taniteval_window_start": (r["t"] - TANITEVAL_WINDOW
                                       if isinstance(r.get("t"), int)
                                       and r["t"] >= TANITEVAL_WINDOW else None),
            "parse_mode": bmode,
            "provenance": "vlm", "model": r.get("model"),
            "prompt_version": r.get("prompt_version"),
            "frames_plan": r.get("frames_plan"),
            "future_offsets_s": offered,
            # --- the strata the metric suite is blocked on -------------------
            "road_geometry": geom_a,                       # INDEPENDENT (Pass A)
            "road_geometry_source": "pass_A_independent",
            # an out-of-enum answer becomes `unknown`, never a stratum, and the
            # token it actually said is kept so the rate stays auditable
            "road_geometry_enum_violation": geom_a_violation,
            "road_geometry_raw": raw_geom_a,
            "road_geometry_band": _tok(a.get("road_geometry_confidence_band")),
            "geometry_event_time_s": _t(a.get("geometry_event_time_s")),
            "is_eventful_geometry": geom_a in EVENTFUL_GEOM,
            # --- the context-rich but CONTAMINATED twin, kept for comparison --
            "road_geometry_passB_CONTAMINATED": geom_b,
            "geometry_event_time_s_passB_CONTAMINATED":
                _t(sc.get("geometry_event_time_s")),
            "geometry_event_end_time_s_passB_CONTAMINATED":
                _t(sc.get("geometry_event_end_time_s")),
            "passB_is_downstream_of_our_kinematics": True,
            # scenario_tag has no Pass-A twin yet — see the note in the research
            # doc; it inherits Pass B's contamination and is marked as such
            "scenario_tag": _tok(sc.get("scenario_tag")) or "none",
            "scenario_tag_source": "pass_B_context_conditioned",
            "scenario_band": _tok(sc.get("scenario_confidence_band")),
            "scenario_event_time_s": _t(sc.get("scenario_event_time_s")),
            "difficulty": _tok(sc.get("difficulty")),
            "road_type": _tok(sc.get("road_type")),
            "traffic_density": _tok(sc.get("traffic_density")),
            # --- coarse lead state: STRATIFICATION ONLY ----------------------
            "lead_present": lead.get("present"),
            "lead_lane": _tok(lead.get("lane")),
            "lead_distance_bucket": _tok(lead.get("distance_bucket")),
            "lead_relative_motion": _tok(lead.get("relative_motion")),
            "lead_is_metric_grade": False,
            # --- kinematic ground truth carried alongside, never merged ------
            "kin_route_v21": (r.get("kin_v21") or {}).get("route"),
            "kin_route_valid": (r.get("kin_v21") or {}).get("valid"),
            "kin_net_dyaw_deg": (r.get("kin_v21") or {}).get("net_dyaw_deg"),
            "kin_strata": r.get("strata"),
            # --- the VLM's ROUTE: recorded, NEVER a label --------------------
            "vlm_route_passA_CROSSCHECK_ONLY": (r.get("pass_A") or {}).get("ROUTE"),
        })
    return rows


# ---------------------------------------------------------- episode sidecars
def episode_sidecar(ep_recs: list) -> dict:
    """Fill the four `vlm_pending_*` skeletons from `enrich.py`, shape-for-shape."""
    bs = [_b_parsed(r.get("pass_B") or {})[0] for r in ep_recs]
    scs = [b.get("SCENARIO") or {} for b in bs]
    envs = [s.get("environment") or {} for s in scs]
    obss = [b.get("OBSERVATIONS") or {} for b in bs]

    axis_src = {"weather": [_tok(e.get("weather")) for e in envs],
                "time_of_day": [_tok(e.get("time_of_day")) for e in envs],
                "road_type": [_tok(s.get("road_type")) for s in scs],
                "surface": [_tok(s.get("surface")) for s in scs],
                "traffic_density": [_tok(s.get("traffic_density")) for s in scs]}
    scene_tags, agree = {}, {}
    for ax in SCENE_AXES:
        tok, share, n = _mode(axis_src[ax])
        scene_tags[ax] = {"token": tok, "prov": "vlm"}
        agree[ax] = {"agreement": share, "n_windows": n}

    vru = any(_tok(ca.get("type")) in VRU_TYPES
              for o in obss for ca in (o.get("critical_agents") or [])
              if isinstance(ca, dict))
    scene_tags["vru_present"] = {"token": "true" if vru else "false",
                                 "prov": "vlm"}
    # notable events keep their WINDOW and their OFFSET: an intersection lasts
    # 5-20 s and a 2 s window cannot contain the event that explains it.
    # The geometry event takes its offset from PASS A (independent, 100 %
    # in-offered); Pass B's is our own kinematic onset echoed back.
    notable = []
    for r, s in zip(ep_recs, scs):
        pa = (r.get("pass_A") or {}).get("parsed") or (r.get("pass_A") or {})
        offered = set(r.get("future_offsets_s") or [])

        def _off(v):
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and float(v) in offered else None

        tag = _tok(s.get("scenario_tag"))
        g_raw = _tok(pa.get("road_geometry"))
        geom = g_raw if g_raw in GEOM_ENUM else UNKNOWN
        if tag and tag != "none":
            notable.append({"event": tag, "t": r.get("t"),
                            "offset_s": _off(s.get("scenario_event_time_s")),
                            "band": _tok(s.get("scenario_confidence_band")),
                            "source": "pass_B_context_conditioned",
                            "prov": "vlm"})
        if geom in EVENTFUL_GEOM:
            notable.append({"event": f"geometry.{geom}", "t": r.get("t"),
                            "offset_s": _off(pa.get("geometry_event_time_s")),
                            "band": _tok(pa.get("road_geometry_confidence_band")),
                            "source": "pass_A_independent",
                            "prov": "vlm"})
    scene_tags["notable_events"] = notable
    scene_tags["_pending"] = False
    scene_tags["_axis_agreement"] = agree

    leads = [o.get("lead_vehicle") or {} for o in obss]
    present = [bool(x.get("present")) for x in leads if x.get("present") is not None]
    lane, lane_share, _ = _mode([_tok(x.get("lane")) for x in leads])
    dist, dist_share, _ = _mode([_tok(x.get("distance_bucket")) for x in leads])
    mot, mot_share, _ = _mode([_tok(x.get("relative_motion")) for x in leads])
    lead_state = {
        "present": (sum(present) / len(present) >= 0.5) if present else None,
        # REFUSED BY DESIGN — see module docstring
        "gap_m": None, "closing_speed_ms": None, "ttc_s": None,
        "lane": lane, "distance_bucket": dist, "relative_motion": mot,
        "present_rate": round(sum(present) / len(present), 3) if present else None,
        "agreement": {"lane": lane_share, "distance_bucket": dist_share,
                      "relative_motion": mot_share},
        "_metric_fields_unavailable":
            "gap_m / closing_speed_ms / ttc_s are metric quantities a forward "
            "camera VLM cannot measure; the 48-clip pilot measured this model "
            "fabricating band edges on 48 % of clips. The coarse categorical "
            "state is sufficient to STRATIFY a metric and insufficient to "
            "compute headway or TTC.",
        "prov": "vlm", "_pending": False}

    signs = []
    for o in obss:
        for s in (o.get("sign_reads") or []):
            if not isinstance(s, dict):
                continue
            val = s.get("value")
            if s.get("type") is None and val is None:
                continue
            signs.append({"type": _tok(s.get("type")), "value": val,
                          "unit": _tok(s.get("unit")), "band": _tok(s.get("band")),
                          "prov": "vlm"})
    if not signs:
        signs = [{"_pending": False, "prov": "vlm", "_note": "no sign read"}]

    cocs = [b.get("COC") or {} for b in bs]
    first = next((c for c in cocs if c.get("observation")), {})
    language = {
        "caption": first.get("observation"),
        "coc_trace": {"observation": first.get("observation"),
                      "critical_agents": [ca for o in obss
                                          for ca in (o.get("critical_agents") or [])
                                          if isinstance(ca, dict)][:8],
                      "justification": first.get("inference"),
                      "decision": first.get("decision"),
                      "action": None, "physics_flag": None},
        "qa": [], "prov": "vlm", "_pending": False,
        "_per_window_coc": [{"t": r.get("t"), **{k: c.get(k) for k in
                                                 ("observation", "inference",
                                                  "decision")}}
                            for r, c in zip(ep_recs, cocs) if c],
    }
    r0 = ep_recs[0]
    return {"episode": r0.get("episode"), "episode_id": r0.get("episode_id"),
            "val_build": r0.get("val_build"), "n_windows": len(ep_recs),
            "scene_tags": scene_tags, "lead_state": lead_state,
            "sign_reads": signs, "language": language,
            "vlm_stamp": {"model": r0.get("model"),
                          "prompt_version": r0.get("prompt_version"),
                          "frames_plan": r0.get("frames_plan"),
                          "provenance": "vlm",
                          "labeler": "scripts/vlm_semantic_labels.py"}}


def main():
    ap = argparse.ArgumentParser("vlm_labels_to_lake")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--sidecars-out", default=None,
                    help="dir for <episode>.goal.vlm.json episode sidecars")
    ap.add_argument("--windows-out", default=None,
                    help="per-window JSONL for the TanitEval scenario strata")
    args = ap.parse_args()

    recs = []
    with open(args.jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    recs.sort(key=lambda r: (r.get("episode", ""), r.get("t", 0)))

    if args.windows_out:
        rows = window_rows(recs)
        with open(args.windows_out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_ev = sum(1 for r in rows if r["is_eventful_geometry"])
        print(f"wrote {args.windows_out}: {len(rows)} window rows "
              f"({n_ev} with an eventful geometry)")

    if args.sidecars_out:
        os.makedirs(args.sidecars_out, exist_ok=True)
        by_ep = {}
        for r in recs:
            by_ep.setdefault(r.get("episode"), []).append(r)
        for ep, rs in sorted(by_ep.items()):
            sc = episode_sidecar(rs)
            with open(os.path.join(args.sidecars_out, f"{ep}.goal.vlm.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(sc, fh, indent=1, ensure_ascii=False)
        print(f"wrote {len(by_ep)} episode sidecars to {args.sidecars_out}")


if __name__ == "__main__":
    main()
