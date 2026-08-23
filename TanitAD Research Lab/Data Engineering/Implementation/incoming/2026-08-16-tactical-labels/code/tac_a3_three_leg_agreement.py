#!/usr/bin/env python3
"""A3 — THREE-LEG AGREEMENT. The core deliverable: **the disagreement rate**.

⛔ THE PREMISE THIS TESTS. A tactical label built by fusing Alpamayo + VLM + ego
is only trustworthy if the legs agree more than chance. The S2 review is the
cautionary case: a derivation whose inputs could not distinguish the classes it
emitted, shipped anyway, ≈78 % wrong.

⚠️ **MISSING IS NEVER A NEGATIVE.** Every cell counts three states —
AGREE / DISAGREE / leg SILENT — and the silent count is reported separately and
never folded into either. (The S2 defect was exactly this: an empty census
rendered as the confident claim `"no agents"`.)

⚠️ **THE TWO n's ARE DIFFERENT AND MUST NOT BE CONFLATED** (PI correction
2026-08-16):
  * LABEL-SIDE n — Alpamayo alone — is **4,729 clips**, complete, produced from
    the full camera rig. Nothing about it is coverage-limited.
  * AGREEMENT-SIDE n — clips where ≥2 legs speak — is bounded by our own w120
    video/pose cache, **201 clips today**. That is a VIDEO-availability bound,
    not a label-availability bound.

⛔ INDEPENDENCE IS NOT ASSUMED, IT IS MEASURED. `ph1_fuse.emit_vocab` counts
`ego` and `vlm` as two of three votes. But the v2.2 VLM ran with an
`ego_state` block in its prompt carrying `motion` and `turning` — and those are
the very fields the ego voter reads (`_EGO_TURN_VOTE[turning]`, `motion ==
"steady"`). A 2-of-3 majority satisfied by {ego, vlm} may therefore be ONE
source counted twice. This script measures how often that happens.

Inputs (all local, all primary):
  --alpamayo-per-clip  A1's per-clip JSONL (from records.parquet, n=4,729)
  --fused              the fused aug120 dir (VLM symbols + ph1 vocab)
  --v2                 the merged ph0_v2.json (source truth for _ego_prompt_mode)
  --engine-a           engine_a_aug120.jsonl (the ego geometric spine)

Usage:
  python tac_a3_three_leg_agreement.py --alpamayo-per-clip … --fused … \
      --v2 … --engine-a … --out <json>
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

# --------------------------------------------------------------------------- #
# The production mappings, RE-USED VERBATIM from stack/scripts/ph1_fuse.py so   #
# this measures the SHIPPED derivation and not a new one I invented.           #
# ph1_fuse.py:60-69 (LAT_RULES / LON_RULES), :279-281 (_EGO_TURN_VOTE).        #
# --------------------------------------------------------------------------- #
LAT_RULES = (("lane_change_l", "LANE_CHANGE_L"), ("left_lane", "LANE_CHANGE_L"),
             ("lane_change_r", "LANE_CHANGE_R"), ("right_lane", "LANE_CHANGE_R"),
             ("change_left", "LANE_CHANGE_L"), ("change_right", "LANE_CHANGE_R"),
             ("nudge_l", "NUDGE_L"), ("nudge_r", "NUDGE_R"),
             ("keep", "LANE_KEEP"), ("hold_corridor", "LANE_KEEP"),
             ("straight", "LANE_KEEP"))
LON_RULES = (("brake", "BRAKE_TO"), ("stop", "BRAKE_TO"), ("decel", "BRAKE_TO"),
             ("yield", "YIELD_MERGE"), ("merge", "YIELD_MERGE"),
             ("creep", "CREEP"), ("hold", "HOLD"), ("wait", "HOLD"),
             ("follow", "FOLLOW"), ("cruise", "CRUISE"), ("accel", "CRUISE"))
_EGO_TURN_VOTE = {"turning_left": "NUDGE_L", "turning_right": "NUDGE_R",
                  "left": "NUDGE_L", "right": "NUDGE_R",
                  "straight": "LANE_KEEP"}


def map_rules(text: str, rules) -> str | None:
    t = (text or "").lower()
    for sub, tok in rules:
        if sub in t:
            return tok
    return None


# --------------------------------------------------------------------------- #
# THE COMMON COMPARISON AXES. Three vocabularies cannot be compared directly;   #
# they are projected onto the coarsest axis all three can express.             #
# ⛔ THE PROJECTION IS LOSSY AND THE LOSS IS THE POINT — "Sharp Steer Left" and #
# "Steer Left" both become `left` because our label set has no severity. Any    #
# agreement number below is therefore an UPPER BOUND on agreement at full       #
# resolution. (Precedent: a2_parse_meta_action.py:44 LAT2DIR.)                  #
# --------------------------------------------------------------------------- #
LAT3 = ("left", "straight", "right")
LON3 = ("decelerate", "maintain", "accelerate")

#: Alpamayo LATERAL axis -> our 3-way direction.
ALP_LAT3 = {
    "Go Straight": "straight",
    "Steer Left": "left", "Sharp Steer Left": "left", "Slight Steer Left": "left",
    "Steer Right": "right", "Sharp Steer Right": "right",
    "Slight Steer Right": "right",
    "Reverse Left": None, "Reverse Right": None,     # out of scope, not coerced
}
#: Alpamayo LONGITUDINAL axis -> our 3-way band.
ALP_LON3 = {
    "Maintain Speed": "maintain",
    "Gentle Deceleration": "decelerate", "Strong Deceleration": "decelerate",
    "Stop": "decelerate",
    "Gentle Acceleration": "accelerate", "Strong Acceleration": "accelerate",
    "Reverse": None,
}
#: Alpamayo LANE axis -> the v6 TACTICAL_LAT_ACTIONS vocabulary. ⚠️ This is a
#: DIAGNOSTIC projection for the agreement table only — it is NOT a proposed
#: label mapping (that question is A4's, and it is only asked if the legs agree).
ALP_LANE_V6 = {
    "Lane Keep": "LANE_KEEP",
    "Left Lane Change": "LANE_CHANGE_L", "Right Lane Change": "LANE_CHANGE_R",
    "Slightly Shift Left": "NUDGE_L", "Slightly Shift Right": "NUDGE_R",
    "Turn Left": None, "Turn Right": None,   # ⛔ NO v6 LAT ACTION TOKEN EXISTS
}
#: v6 lat action -> 3-way direction, for the common-axis table.
V6LAT_3 = {"LANE_KEEP": "straight", "LANE_CHANGE_L": "left", "NUDGE_L": "left",
           "LANE_CHANGE_R": "right", "NUDGE_R": "right", "ABORT_LC": None}
#: v6 lon action -> 3-way band.
V6LON_3 = {"BRAKE_TO": "decelerate", "CRUISE": "maintain", "FOLLOW": "maintain",
           "HOLD": "decelerate", "CREEP": "decelerate",
           "YIELD_MERGE": "decelerate"}

# ego geometric thresholds — stated, not hidden
EGO_DYAW_RAD = 0.15       # same as a2_parse_meta_action.DIR_YAW_RAD
EGO_DV_MS = 1.0           # net speed change that counts as a band change


def ego_geom_lat3(ea: dict) -> str | None:
    """3-way lateral from the ENGINE-A geometric spine (not the VLM's ego block).

    Uses `route.maneuver_dyaw_rad`, the same net-heading quantity
    `a2_parse_meta_action.net_yaw` computes. Sign convention is read from the
    data, not assumed — see the calibration block in the results.
    """
    if not ea:
        return None
    r = ea.get("route") or {}
    dy = r.get("maneuver_dyaw_rad")
    if dy is None:
        return None
    if dy > EGO_DYAW_RAD:
        return "left"
    if dy < -EGO_DYAW_RAD:
        return "right"
    return "straight"


def ego_geom_lon3(ea: dict) -> str | None:
    if not ea:
        return None
    sp = ea.get("speed_profile") or {}
    if sp.get("stops"):
        return "decelerate"
    dv = sp.get("net_dv_ms")
    if dv is None:
        return None
    if dv > EGO_DV_MS:
        return "accelerate"
    if dv < -EGO_DV_MS:
        return "decelerate"
    return "maintain"


def ego_geom_lane_v6(ea: dict) -> str | None:
    """v6 lat ACTION from engine-A's lane-change events.

    ⛔ NOTE: `lane_change_events` is the output of exactly the geometric gate the
    PI ORDERED REMOVED on 2026-08-16 (`_gated_lc_event`, ≈78 % wrong on the
    adjudicated sample). It is included here ONLY so its disagreement with the
    other legs can be MEASURED — never as a candidate label source.
    """
    if not ea:
        return None
    ev = ea.get("lane_change_events") or []
    if not ev:
        return "LANE_KEEP"
    toks = {e.get("token") for e in ev}
    if "lc_left" in toks and "lc_right" in toks:
        return None                              # ambiguous, not coerced
    if "lc_left" in toks:
        return "LANE_CHANGE_L"
    if "lc_right" in toks:
        return "LANE_CHANGE_R"
    return None


def confusion(pairs) -> dict:
    """(a, b) pairs -> confusion + agreement + Cohen's kappa. Missing excluded
    and counted separately by the caller — NEVER imputed."""
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    cm: collections.Counter = collections.Counter(both)
    n = len(both)
    agree = sum(1 for a, b in both if a == b)
    labs = sorted({a for a, _ in both} | {b for _, b in both})
    po = agree / n if n else None
    pe = sum((sum(1 for a, _ in both if a == c) / n) *
             (sum(1 for _, b in both if b == c) / n) for c in labs) if n else 0
    kap = (round((po - pe) / (1 - pe), 4)
           if n and abs(1 - pe) > 1e-9 else None)
    return {
        "n_both_speak": n,
        "n_a_silent": sum(1 for a, b in pairs if a is None and b is not None),
        "n_b_silent": sum(1 for a, b in pairs if b is None and a is not None),
        "n_both_silent": sum(1 for a, b in pairs if a is None and b is None),
        "agreement": round(po, 4) if po is not None else None,
        "disagreement": round(1 - po, 4) if po is not None else None,
        "cohens_kappa": kap,
        "chance_agreement": round(pe, 4) if n else None,
        "confusion": [{"a": k[0], "b": k[1], "n": v} for k, v in cm.most_common()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpamayo-per-clip", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    alp = {}
    with open(a.alpamayo_per_clip, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                alp[r["clip_id"]] = r

    v2rows = json.load(open(a.v2, encoding="utf-8"))
    v2rows = v2rows if isinstance(v2rows, list) else v2rows.get("clips", v2rows)
    v2 = {r["clip_id"]: r for r in v2rows if isinstance(r, dict) and r.get("clip_id")}

    ea = {}
    with open(a.engine_a, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("clip_id"):
                    ea[r["clip_id"]] = r.get("engine_a")

    fused = {}
    for f in glob.glob(os.path.join(a.fused, "*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f, encoding="utf-8"))
        if r.get("clip_id"):
            fused[r["clip_id"]] = r

    res: dict = {
        "_evidence_class": "MEASURED (ours; primary artifacts, this directory)",
        "_two_n_rule": {
            "label_side_n_alpamayo_only": len(alp),
            "agreement_side_n_clips_with_our_video": len(fused),
            "_note": ("these are DIFFERENT quantities and must never be "
                      "conflated. The agreement n is bounded by OUR w120 video "
                      "cache, not by Alpamayo label availability."),
        },
    }

    # ---------------------------------------------------------------- #
    # STEP 0 — IS THE VLM LEG INDEPENDENT OF THE EGO LEG? (measured)    #
    # ---------------------------------------------------------------- #
    modes = collections.Counter(v2[c].get("_ego_prompt_mode") for c in v2)
    stamped_vision = sum(
        1 for c, r in fused.items()
        if (r.get("_provenance") or {}).get("vlm") == "vision")
    stamp_lie = sum(
        1 for c, r in fused.items()
        if (r.get("_provenance") or {}).get("vlm") == "vision"
        and (v2.get(c, {}).get("_ego_prompt_mode") or "none") != "none")
    res["leg_independence"] = {
        "_question": ("ph1_fuse.emit_vocab counts `ego` and `vlm` as two "
                      "independent votes. Are they?"),
        "v2_ego_prompt_mode_tally": {str(k): v for k, v in modes.items()},
        "_what_the_vlm_was_shown": sorted(
            (v2[next(iter(v2))].get("ego_state") or {}).keys()) if v2 else [],
        "_what_the_ego_voter_reads": ["turning", "motion"],
        "_overlap": ("`turning` and `motion` are BOTH in the VLM's prompt AND "
                     "are the ONLY fields the ph1 ego voter reads "
                     "(ph1_fuse.py:318-325). The two voters share their input."),
        "n_fused_stamped_vlm_vision": stamped_vision,
        "n_provenance_stamp_contradicted_by_source": stamp_lie,
        "_stamp_finding": (
            "a fused record stamped `_provenance.vlm == 'vision'` whose v2 "
            "source says `_ego_prompt_mode != 'none'` is a FALSE stamp. "
            "ph1_fuse.py:556-561 at HEAD computes this correctly; the banked "
            "corpus predates the fix and has not been re-fused."),
    }

    # ---------------------------------------------------------------- #
    # STEP 1 — build the per-clip three-leg table                       #
    # ---------------------------------------------------------------- #
    rows = []
    for cid in sorted(set(fused) | set(ea) | set(alp)):
        A = alp.get(cid)
        F = fused.get(cid)
        E = ea.get(cid)
        V = v2.get(cid)
        sym = ((F or {}).get("semantics") or {}).get("symbols") or \
              ((V or {}).get("symbols") or {})
        acts = sym.get("actions") or []
        # VLM -> v6 tokens via the PRODUCTION rules
        vlm_lat = vlm_lon = None
        for act in acts:
            txt = f"{act.get('verb', '')}_{act.get('direction', '')}"
            vlm_lat = vlm_lat or map_rules(txt, LAT_RULES)
            vlm_lon = vlm_lon or map_rules(txt, LON_RULES)
        egost = (V or {}).get("ego_state") or {}
        rows.append({
            "clip_id": cid,
            "has_alpamayo": A is not None,
            "has_vlm": bool(sym),
            "has_engine_a": E is not None,
            # --- Alpamayo
            "alp_lat_raw": (A or {}).get("lateral"),
            "alp_lon_raw": (A or {}).get("longitudinal"),
            "alp_lane_raw": (A or {}).get("lane"),
            "alp_lat3": ALP_LAT3.get((A or {}).get("lateral")),
            "alp_lon3": ALP_LON3.get((A or {}).get("longitudinal")),
            "alp_lane_v6": ALP_LANE_V6.get((A or {}).get("lane")),
            # --- VLM (production mapping)
            "vlm_lat_v6": vlm_lat, "vlm_lon_v6": vlm_lon,
            "vlm_lat3": V6LAT_3.get(vlm_lat), "vlm_lon3": V6LON_3.get(vlm_lon),
            "vlm_goal_kind": sym.get("goal_kind"),
            "vlm_n_actions": len(acts),
            # --- ego, PRODUCTION vote (the ph1 one, from the VLM-visible block)
            "egoprompt_lat_v6": _EGO_TURN_VOTE.get(str(egost.get("turning", ""))),
            "egoprompt_turning": egost.get("turning"),
            "egoprompt_motion": egost.get("motion"),
            # --- ego, GEOMETRIC spine (engine A — never shown to this VLM run)
            "egogeom_lat3": ego_geom_lat3(E),
            "egogeom_lon3": ego_geom_lon3(E),
            "egogeom_lane_v6": ego_geom_lane_v6(E),
            "egogeom_route": ((E or {}).get("route") or {}).get("token"),
        })
    res["n_rows"] = len(rows)
    res["leg_presence"] = {
        "alpamayo_only": sum(1 for r in rows if r["has_alpamayo"]
                             and not r["has_vlm"] and not r["has_engine_a"]),
        "all_three": sum(1 for r in rows if r["has_alpamayo"]
                         and r["has_vlm"] and r["has_engine_a"]),
        "at_least_two": sum(1 for r in rows if sum(
            (r["has_alpamayo"], r["has_vlm"], r["has_engine_a"])) >= 2),
        "has_alpamayo": sum(1 for r in rows if r["has_alpamayo"]),
        "has_vlm": sum(1 for r in rows if r["has_vlm"]),
        "has_engine_a": sum(1 for r in rows if r["has_engine_a"]),
    }

    tri = [r for r in rows if r["has_alpamayo"] and r["has_vlm"]
           and r["has_engine_a"]]
    res["_agreement_cohort_n"] = len(tri)

    # --- sign calibration for the ego dyaw convention (read, not assumed) ---
    cal = collections.Counter(
        (r["egogeom_route"], r["egogeom_lat3"]) for r in tri
        if r["egogeom_route"] in ("turn_left", "turn_right"))
    res["_ego_dyaw_sign_calibration"] = {
        "_why": ("the sign convention of maneuver_dyaw_rad is READ from clips "
                 "whose engine-A route token already says which way they "
                 "turned, rather than assumed. If turn_left maps to 'right' "
                 "the axis is flipped and every lateral number below is void."),
        "table": [{"route_token": k[0], "derived_lat3": k[1], "n": v}
                  for k, v in cal.most_common()],
    }

    # ---------------------------------------------------------------- #
    # STEP 2 — THE CONFUSIONS. Headline = disagreement rate.            #
    # ---------------------------------------------------------------- #
    pairs = {
        # the genuinely independent contrast: Alpamayo vs OUR geometry
        "LAT3__alpamayo_vs_egogeom": [(r["alp_lat3"], r["egogeom_lat3"]) for r in tri],
        "LON3__alpamayo_vs_egogeom": [(r["alp_lon3"], r["egogeom_lon3"]) for r in tri],
        # Alpamayo vs the VLM
        "LAT3__alpamayo_vs_vlm": [(r["alp_lat3"], r["vlm_lat3"]) for r in tri],
        "LON3__alpamayo_vs_vlm": [(r["alp_lon3"], r["vlm_lon3"]) for r in tri],
        # the VLM vs our geometry
        "LAT3__vlm_vs_egogeom": [(r["vlm_lat3"], r["egogeom_lat3"]) for r in tri],
        "LON3__vlm_vs_egogeom": [(r["vlm_lon3"], r["egogeom_lon3"]) for r in tri],
        # ⚠️ NOT INDEPENDENT — the VLM saw this block. Measured to SHOW the echo.
        "LAT_V6__vlm_vs_egopromptvote": [(r["vlm_lat_v6"], r["egoprompt_lat_v6"])
                                         for r in tri],
        # v6-vocabulary level, the level a label would actually be emitted at
        "LATV6__alpamayo_lane_vs_vlm": [(r["alp_lane_v6"], r["vlm_lat_v6"])
                                        for r in tri],
        "LATV6__alpamayo_lane_vs_egogeom": [(r["alp_lane_v6"], r["egogeom_lane_v6"])
                                            for r in tri],
    }
    res["confusions"] = {k: confusion(v) for k, v in pairs.items()}

    # ---------------------------------------------------------------- #
    # STEP 3 — 3-way: do ALL THREE legs agree on one clip?              #
    # ---------------------------------------------------------------- #
    for axis, keys in (("LAT3", ("alp_lat3", "vlm_lat3", "egogeom_lat3")),
                       ("LON3", ("alp_lon3", "vlm_lon3", "egogeom_lon3"))):
        vals = [tuple(r[k] for k in keys) for r in tri]
        spoke = [v for v in vals if all(x is not None for x in v)]
        unan = sum(1 for v in spoke if v[0] == v[1] == v[2])
        two = sum(1 for v in spoke
                  if len(set(v)) == 2)
        three = sum(1 for v in spoke if len(set(v)) == 3)
        res[f"three_way_{axis}"] = {
            "n_all_three_speak": len(spoke),
            "n_at_least_one_silent": len(vals) - len(spoke),
            "n_unanimous": unan,
            "unanimous_pct": round(100.0 * unan / len(spoke), 2) if spoke else None,
            "n_two_of_three": two,
            "n_all_three_differ": three,
            "_silence_by_leg": {
                k: sum(1 for r in tri if r[k] is None) for k in keys},
        }

    # ---------------------------------------------------------------- #
    # STEP 4 — the VLM verb -> v6 mapping COVERAGE (a code gap, not data)#
    # ---------------------------------------------------------------- #
    verb_cov: dict[str, dict] = {}
    for r in rows:
        F = fused.get(r["clip_id"]) or v2.get(r["clip_id"]) or {}
        sym = ((F.get("semantics") or {}).get("symbols")
               or F.get("symbols") or {})
        for act in (sym.get("actions") or []):
            txt = f"{act.get('verb', '')}_{act.get('direction', '')}"
            e = verb_cov.setdefault(txt, {"n": 0, "lat": None, "lon": None})
            e["n"] += 1
            e["lat"] = map_rules(txt, LAT_RULES)
            e["lon"] = map_rules(txt, LON_RULES)
    res["vlm_verb_mapping_coverage"] = {
        "_question": ("does ph1_fuse's substring mapping actually cover the "
                      "VLM's emitted vocabulary? An unmapped verb is a SILENT "
                      "DROP — the leg looks silent when it in fact spoke."),
        "table": [{"verb_direction": k, "n": v["n"],
                   "lat_token": v["lat"], "lon_token": v["lon"]}
                  for k, v in sorted(verb_cov.items(),
                                     key=lambda kv: -kv[1]["n"])],
        "n_emissions_with_no_lon_token": sum(
            v["n"] for v in verb_cov.values() if v["lon"] is None),
        "n_emissions_with_no_lat_token": sum(
            v["n"] for v in verb_cov.values() if v["lat"] is None),
        "n_emissions_total": sum(v["n"] for v in verb_cov.values()),
    }

    # per-clip table for audit
    per = a.out.replace(".json", "_per_clip.jsonl")
    with open(per, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    res["_per_clip_path"] = per

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("confusions",)}, indent=1,
                     ensure_ascii=False)[:5000])
    print(f"\n[out] {a.out}\n[out] {per}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
