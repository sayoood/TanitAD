#!/usr/bin/env python3
"""M1 — MEASURE the ph1_fuse tactical-mapping fix on the REAL banked corpus.

Not a unit test and not a claim: a replay. The PRE-FIX mapping is carried here
VERBATIM as a negative control (it was deleted from `ph1_fuse.py`, so it has
to live somewhere to be replayed), the POST-FIX mapping is IMPORTED from the
shipped module, and both are run over the same inputs the production fuse saw.

Inputs (all local, all CPU, no GPU):
  * `<sp>/aug120/fused_aug120/*.json` — 201 banked fused records. Their
    `alpamayo` field is EXACTLY the `{task: raw_json}` dict `ph1_fuse.main`
    builds, so the Alpamayo leg replays bit-for-bit including the
    `json.dumps(...)[:400]` truncation that is half of defect 3.
  * `<sp>/aug120/merged/ph0_v2.json` — the 201 v2 records (symbols.actions,
    ego_state, speed_profile) that drive the VLM and ego legs.

⛔ THIS SCRIPT DOES NOT RE-FUSE ANYTHING. It writes one JSON of counts. The
re-fuse is escalated separately and has three independent reasons pending
(SAM3 backfill, the `_provenance.vlm` mis-stamp, and these mapping fixes).

Usage:
  python tacfix_m1_before_after.py --fused <dir> --v2 <ph0_v2.json> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

_STACK = Path(__file__).resolve().parents[6] / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import ph1_fuse  # noqa: E402
from tanitad.models.v6 import (TACTICAL_LAT_ACTIONS,  # noqa: E402
                               TACTICAL_LON_ACTIONS)

# --------------------------------------------------------------------------- #
# THE NEGATIVE CONTROL — `ph1_fuse.py:60-77` as it stood before this change.    #
# Copied verbatim so the "before" column is the real thing, not a reconstruction.
# --------------------------------------------------------------------------- #
OLD_LAT_RULES = (("lane_change_l", "LANE_CHANGE_L"),
                 ("left_lane", "LANE_CHANGE_L"),
                 ("lane_change_r", "LANE_CHANGE_R"),
                 ("right_lane", "LANE_CHANGE_R"),
                 ("change_left", "LANE_CHANGE_L"),
                 ("change_right", "LANE_CHANGE_R"),
                 ("nudge_l", "NUDGE_L"), ("nudge_r", "NUDGE_R"),
                 ("keep", "LANE_KEEP"), ("hold_corridor", "LANE_KEEP"),
                 ("straight", "LANE_KEEP"))
OLD_LON_RULES = (("brake", "BRAKE_TO"), ("stop", "BRAKE_TO"),
                 ("decel", "BRAKE_TO"), ("yield", "YIELD_MERGE"),
                 ("merge", "YIELD_MERGE"), ("creep", "CREEP"),
                 ("hold", "HOLD"), ("wait", "HOLD"), ("follow", "FOLLOW"),
                 ("cruise", "CRUISE"), ("accel", "CRUISE"))


def old_map_rules(text: str, rules) -> str | None:
    t = (text or "").lower()
    for sub, tok in rules:
        if sub in t:
            return tok
    return None


def old_majority(votes, valid):
    """The retired 2-of-3. Returns (token, voters)."""
    counts: dict[str, list] = {}
    for src, tok in votes:
        if tok in valid:
            counts.setdefault(tok, []).append(src)
    if not counts:
        return None, []
    tok = max(counts, key=lambda k: len(counts[k]))
    return tok, counts[tok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v2raw = json.load(open(a.v2, encoding="utf-8"))
    v2raw = v2raw if isinstance(v2raw, list) else v2raw.get("clips", v2raw)
    v2_by = {r["clip_id"]: r for r in v2raw if isinstance(r, dict)}

    fused = {}
    for f in sorted(os.listdir(a.fused)):
        if f.endswith(".json") and not f.startswith("_"):
            r = json.load(open(os.path.join(a.fused, f), encoding="utf-8"))
            fused[r["clip_id"]] = r

    res: dict = {
        "_evidence_class": "MEASURED (ours; replay over the banked aug120 "
                           "fused corpus + its own v2 source)",
        "_what_this_is_not": "NOT a re-fuse. Counts only; no record rewritten.",
        "n_fused_records": len(fused),
        "n_v2_records": len(v2_by),
    }

    verb_tab: collections.Counter = collections.Counter()
    verb_map: dict = {}
    alp_rows, lat_rows, lon_rows = [], [], []
    old_lat_dist, new_lat_dist = collections.Counter(), collections.Counter()
    old_lon_dist, new_lon_dist = collections.Counter(), collections.Counter()
    echo_lat = echo_lon = 0            # 2-of-3 carried by {ego,vlm} alone
    corrob_lat = corrob_lon = 0
    n_alp_lon_old_spoke = n_alp_lon_offaxis = 0
    n_alp_lat_changed = 0
    per_clip = []

    for cid, r in sorted(fused.items()):
        v2 = v2_by.get(cid)
        if v2 is None:
            continue
        sym = v2.get("symbols") or {}
        ego = v2.get("ego_state") or {}
        sp = v2.get("speed_profile") or {}

        # ---- ego leg (unchanged by this fix) ------------------------------
        ego_lat = ph1_fuse._EGO_TURN_VOTE.get(str(ego.get("turning", "")))
        ego_lon = None
        if (sp.get("stops") or 0) > 0:
            ego_lon = "BRAKE_TO"
        elif str(ego.get("motion")) == "steady":
            ego_lon = "CRUISE"

        # ---- VLM leg: OLD vs NEW, per emission ----------------------------
        o_lat_v, o_lon_v, n_lat_v, n_lon_v = [], [], [], []
        for act in (sym.get("actions") or []):
            verb, direc = act.get("verb"), act.get("direction")
            key = f"{verb}_{direc}"
            txt = f"{verb or ''}_{direc or ''}"
            ol = old_map_rules(txt, OLD_LAT_RULES)
            oo = old_map_rules(txt, OLD_LON_RULES)
            nl, nn, _note = ph1_fuse.map_vlm_action(verb, direc)
            verb_tab[key] += 1
            verb_map[key] = {"old_lat": ol, "old_lon": oo,
                             "new_lat": nl, "new_lon": nn}
            o_lat_v.append(ol)
            o_lon_v.append(oo)
            n_lat_v.append(nl)
            n_lon_v.append(nn)

        # ---- Alpamayo leg: OLD (substring over the blob) vs NEW (axes) ----
        alp = r.get("alpamayo") or {}
        o_alp_lat = o_alp_lon = n_alp_lat = n_alp_lon = None
        axes = {}
        if alp.get("meta_action"):
            blob = json.dumps(alp["meta_action"])[:400]
            o_alp_lat = old_map_rules(blob, OLD_LAT_RULES)
            o_alp_lon = old_map_rules(blob, OLD_LON_RULES)
            axes = ph1_fuse.parse_alpamayo_axes(alp["meta_action"])
            n_alp_lat, n_alp_lon, _ = ph1_fuse.map_alpamayo_axes(axes)
            if o_alp_lon in TACTICAL_LON_ACTIONS:
                n_alp_lon_old_spoke += 1
                # did the OLD longitudinal vote come from text that is NOT the
                # Longitudinal AXIS? i.e. was it the `cot` (or another axis)
                # that decided? Replay the rules on the AXIS LINE alone.
                axis_only = old_map_rules(
                    str(axes.get("longitudinal") or ""), OLD_LON_RULES)
                if axis_only != o_alp_lon:
                    n_alp_lon_offaxis += 1
                    alp_rows.append({
                        "clip_id": cid,
                        "axis_longitudinal": axes.get("longitudinal"),
                        "old_vote_from_blob": o_alp_lon,
                        "old_vote_from_axis_alone": axis_only,
                        "cot": (axes.get("cot") or "")[:160]})
            if o_alp_lat != n_alp_lat:
                n_alp_lat_changed += 1

        # ---- assemble both vote sets and both decisions -------------------
        old_lat_votes = ([("ego", ego_lat)] + [("vlm", t) for t in o_lat_v]
                         + ([("alpamayo", o_alp_lat)] if alp.get("meta_action")
                            else []))
        old_lon_votes = ([("ego", ego_lon)] + [("vlm", t) for t in o_lon_v]
                         + ([("alpamayo", o_alp_lon)] if alp.get("meta_action")
                            else []))
        new_lat_votes = ([("ego", ego_lat)] + [("vlm", t) for t in n_lat_v]
                         + ([("alpamayo", n_alp_lat)] if alp.get("meta_action")
                            else []))
        new_lon_votes = ([("ego", ego_lon)] + [("vlm", t) for t in n_lon_v]
                         + ([("alpamayo", n_alp_lon)] if alp.get("meta_action")
                            else []))

        ol_tok, ol_src = old_majority(old_lat_votes, TACTICAL_LAT_ACTIONS)
        oo_tok, oo_src = old_majority(old_lon_votes, TACTICAL_LON_ACTIONS)
        nl = ph1_fuse.block_vote(new_lat_votes, TACTICAL_LAT_ACTIONS)
        nn = ph1_fuse.block_vote(new_lon_votes, TACTICAL_LON_ACTIONS)

        # ⛔ THE ECHO: the retired majority decided by >=2 voters that were
        # ALL inside the {ego, vlm} block — one source counted twice.
        if len(ol_src) >= 2 and set(ol_src) <= {"ego", "vlm"}:
            echo_lat += 1
            lat_rows.append({"clip_id": cid, "token": ol_tok,
                             "voters": ol_src})
        if len(oo_src) >= 2 and set(oo_src) <= {"ego", "vlm"}:
            echo_lon += 1
            lon_rows.append({"clip_id": cid, "token": oo_tok,
                             "voters": oo_src})
        corrob_lat += int(nl["corroborated"])
        corrob_lon += int(nn["corroborated"])
        old_lat_dist[ol_tok] += 1
        new_lat_dist[nl["token"]] += 1
        old_lon_dist[oo_tok] += 1
        new_lon_dist[nn["token"]] += 1
        per_clip.append({
            "clip_id": cid,
            "old_lat": ol_tok, "old_lat_voters": ol_src,
            "new_lat": nl["token"], "new_lat_prov": nl["provenance"],
            "new_lat_corroborated": nl["corroborated"],
            "old_lon": oo_tok, "old_lon_voters": oo_src,
            "new_lon": nn["token"], "new_lon_prov": nn["provenance"],
            "new_lon_corroborated": nn["corroborated"],
            "alp_axes": {k: axes.get(k) for k in
                         ("longitudinal", "lateral", "lane")},
        })

    n = len(per_clip)
    res["n_clips_replayed"] = n
    res["vlm_emissions"] = {
        "_note": "one row per (verb, direction) EMISSION, not per clip",
        "n_emissions": sum(verb_tab.values()),
        "table": {k: {"n": verb_tab[k], **verb_map[k]}
                  for k in sorted(verb_tab, key=lambda x: -verb_tab[x])},
    }
    n_em = sum(verb_tab.values())
    dropped_old = sum(v for k, v in verb_tab.items()
                      if verb_map[k]["old_lat"] is None
                      and verb_map[k]["old_lon"] is None)
    lonwrong_old = sum(v for k, v in verb_tab.items()
                       if k.startswith("hold_corridor"))
    res["defect_1_reduce_to_dropped"] = {
        "n_emissions_silent_on_BOTH_axes_before": dropped_old,
        "pct": round(100.0 * dropped_old / n_em, 2) if n_em else None,
        "n_after": sum(v for k, v in verb_tab.items()
                       if verb_map[k]["new_lat"] is None
                       and verb_map[k]["new_lon"] is None),
    }
    res["defect_2_hold_corridor_lon"] = {
        "n_hold_corridor_emissions": lonwrong_old,
        "pct": round(100.0 * lonwrong_old / n_em, 2) if n_em else None,
        "old_lon_token": "HOLD", "new_lon_token": None,
    }
    res["defect_3_alpamayo_substring_over_the_blob"] = {
        "_question": "did the OLD Alpamayo longitudinal vote come from text "
                     "that is NOT the Longitudinal axis line (i.e. the `cot` "
                     "rationale or another axis)?",
        "n_clips_old_alp_lon_spoke": n_alp_lon_old_spoke,
        "n_decided_off_axis": n_alp_lon_offaxis,
        "pct_off_axis": (round(100.0 * n_alp_lon_offaxis / n_alp_lon_old_spoke,
                               2) if n_alp_lon_old_spoke else None),
        "examples": alp_rows[:12],
        "n_clips_alp_LAT_changed_by_the_fix": n_alp_lat_changed,
    }
    res["priority_2_echo"] = {
        "_question": "how often did the retired 2-of-3 majority get decided "
                     "by >=2 voters ALL inside the {ego, vlm} block?",
        "lat_n": echo_lat, "lat_pct": round(100.0 * echo_lat / n, 2) if n else None,
        "lon_n": echo_lon, "lon_pct": round(100.0 * echo_lon / n, 2) if n else None,
        "lat_examples": lat_rows[:8], "lon_examples": lon_rows[:8],
        "after_n_corroborated_lat": corrob_lat,
        "after_n_corroborated_lon": corrob_lon,
        "_corroborated_means": ">=2 INDEPENDENT blocks spoke and agreed; "
                               "{ego,vlm} is ONE block and can never satisfy it",
    }
    res["token_distribution"] = {
        "lat_before": dict(old_lat_dist), "lat_after": dict(new_lat_dist),
        "lon_before": dict(old_lon_dist), "lon_after": dict(new_lon_dist),
    }
    res["_vocabulary_unchanged"] = {
        "TACTICAL_LAT_ACTIONS": list(TACTICAL_LAT_ACTIONS),
        "TACTICAL_LON_ACTIONS": list(TACTICAL_LON_ACTIONS),
        "_note": "REPORTED, never edited — the tuples size embedding tables "
                 "(v6.py:3297-3298) and a shape change breaks the live 30k "
                 "v6F strict resume.",
    }
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    pc = os.path.join(os.path.dirname(a.out), "m1_before_after_per_clip.jsonl")
    with open(pc, "w", encoding="utf-8") as fh:
        for row in per_clip:
            fh.write(json.dumps(row) + "\n")
    print(f"M1_DONE n={n} -> {a.out}")
    print(json.dumps({k: res[k] for k in
                      ("defect_1_reduce_to_dropped",
                       "defect_2_hold_corridor_lon",
                       "defect_3_alpamayo_substring_over_the_blob",
                       "priority_2_echo")}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
