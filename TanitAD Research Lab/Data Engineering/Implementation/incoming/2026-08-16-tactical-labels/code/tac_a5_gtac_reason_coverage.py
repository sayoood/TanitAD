#!/usr/bin/env python3
"""A5 — CAN THE REASON FIELD CARRY THE `g_tac` GOAL VOCABULARY?

⭐ WHY THIS IS THE DECIDING MEASUREMENT, and why the PI's question is exactly
right to name "meta actions AND reasoning" together.

A1/A3 established the split:
  * `meta_action` is **AXIS + MAGNITUDE** — "Gentle Deceleration / Steer Right /
    Lane Keep". It says WHAT the vehicle does.
  * `cot` is the **REASON** — "Keep distance to the lead vehicle because it is
    directly ahead in the same lane". It says WHY.

And the v6 vocabularies are split the same way:
  * `TACTICAL_LAT_ACTIONS` is magnitude/direction-typed (LANE_KEEP,
    LANE_CHANGE_L/R, NUDGE_L/R) ⇒ recoverable from the meta_action LANE axis.
  * `TACTICAL_LON_ACTIONS` is **REASON-typed** (FOLLOW, YIELD_MERGE, CREEP,
    HOLD, BRAKE_TO, CRUISE). "Gentle Deceleration" alone cannot say whether it
    is FOLLOW (closing on a lead) or BRAKE_TO (a stop line) — ⇒ NOT recoverable
    from the axis alone.
  * `TACTICAL_GOAL_TOKENS_LON` is **PURELY reason-typed** (GAP_TARGET, YIELD_AT,
    STOP_POINT, TRAFFIC_LIGHT_REACT, WAIT_FOR_ONCOMING, SPEED_BAND) ⇒ can come
    from NOTHING BUT the reason.

⇒ This script measures what fraction of the corpus the reason text can place
into a `g_tac` token at all — the ceiling on any reason-derived tactical goal
label. It does NOT propose the production mapping.

⛔ WHAT THIS IS NOT. The keyword mapping below is a **FEASIBILITY PROBE**, not
a labeller. It measures REACHABILITY (does the reason mention the thing a token
needs?), never correctness (is the token right?). A high number here means
"worth building and validating"; it is NOT evidence that a label built this way
would be correct. Confusing those two is the S2 failure.

⚠️ MULTI-HIT IS REPORTED, NOT RESOLVED. A reason matching two token families is
counted as AMBIGUOUS and reported separately — silently taking the first match
would manufacture precision that the text does not contain.

Usage:
  python tac_a5_gtac_reason_coverage.py --records <records.parquet> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import re

#: reason-family -> the v6 `g_tac` token it would serve. ⛔ FEASIBILITY ONLY.
#: Token names are the real ones from `tanitad.models.v6`; the PATTERNS are
#: mine and are deliberately conservative (a miss is honest, a false hit is not).
GTAC_PROBES = (
    ("GAP_TARGET",          "lon", r"lead vehicle|vehicle ahead|car ahead|"
                                   r"queued vehicles|keep distance|safe distance|"
                                   r"following distance"),
    ("TRAFFIC_LIGHT_REACT", "lon", r"traffic light|red light|green light|"
                                   r"light turns|signal is"),
    ("STOP_POINT",          "lon", r"stop sign|stop line|come to a stop|"
                                   r"speed bump|crosswalk|pedestrian"),
    ("YIELD_AT",            "lon", r"yield|roundabout|give way|merging traffic|"
                                   r"oncoming traffic"),
    ("SPEED_BAND",          "lon", r"curve|bend|speed limit|road ahead is clear|"
                                   r"lane ahead is clear|resume speed|"
                                   r"maintain (?:the )?speed"),
    ("WAIT_FOR_ONCOMING",   "lon", r"oncoming|wait for"),
    ("EVADE_IN_CORRIDOR",   "lat", r"nudge|parked car|parked vehicle|obstacle|"
                                   r"debris|cone|around the"),
    ("CORRIDOR_OFFSET",     "lat", r"shift (?:left|right)|keep (?:left|right)|"
                                   r"stay (?:left|right)|position"),
    ("ANCHOR_GOAL",         "lat", r"lane change|change lane|merge into|"
                                   r"move (?:in)?to the (?:left|right) lane"),
)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(a.records)

    def unwrap(d, k):
        v = d.get(k)
        return (v[0] if v else None) if isinstance(v, list) else v

    # union the two reason legs — A2 measured them only 22.2 % identical, so
    # they are semi-independent draws and both are read.
    ma: dict[str, str] = {}
    for _, r in df[df.task == "meta_action"].iterrows():
        ma[r["clip_id"]] = unwrap(json.loads(r["raw_json"]), "cot") or ""
    al: dict[str, str] = {}
    for _, r in df[df.task == "auto_labeling"].iterrows():
        j = unwrap(json.loads(r["raw_json"]), "cot_auto_labeling_json")
        if isinstance(j, dict) and isinstance(j.get("chain_of_causation"), str):
            al[r["clip_id"]] = j["chain_of_causation"]

    rx = [(tok, ax, re.compile(p, re.I)) for tok, ax, p in GTAC_PROBES]
    res: dict = {
        "_evidence_class": "MEASURED (ours; records.parquet, n=4,729 clips)",
        "_what_this_measures": (
            "REACHABILITY — does the reason text mention the thing a g_tac "
            "token needs? NOT correctness. A high number means 'worth building "
            "and validating', never 'this label would be right'."),
        "n_clips": len(ma),
    }

    for legname, corpus in (("meta_action_cot", ma),
                            ("auto_labeling_coc", al),
                            ("union_of_both_legs", None)):
        if corpus is None:
            corpus = {c: (ma.get(c, "") + " || " + al.get(c, ""))
                      for c in set(ma) | set(al)}
        per_tok: collections.Counter = collections.Counter()
        n_hit = n_amb = n_none = 0
        amb_pairs: collections.Counter = collections.Counter()
        by_axis = {"lat": 0, "lon": 0}
        for cid, txt in corpus.items():
            hits = [(tok, ax) for tok, ax, r in rx if r.search(txt or "")]
            for tok, _ in hits:
                per_tok[tok] += 1
            lon = {t for t, ax in hits if ax == "lon"}
            lat = {t for t, ax in hits if ax == "lat"}
            by_axis["lon"] += int(bool(lon))
            by_axis["lat"] += int(bool(lat))
            if not hits:
                n_none += 1
            else:
                n_hit += 1
                if len(lon) > 1:
                    n_amb += 1
                    amb_pairs[tuple(sorted(lon))] += 1
        n = len(corpus)
        res[legname] = {
            "n_clips": n,
            "n_reaching_at_least_one_token": n_hit,
            "reach_pct": round(100.0 * n_hit / n, 2) if n else None,
            "n_reaching_no_token": n_none,
            "no_token_pct": round(100.0 * n_none / n, 2) if n else None,
            "n_lon_axis_reached": by_axis["lon"],
            "lon_reach_pct": round(100.0 * by_axis["lon"] / n, 2) if n else None,
            "n_lat_axis_reached": by_axis["lat"],
            "lat_reach_pct": round(100.0 * by_axis["lat"] / n, 2) if n else None,
            "n_ambiguous_multi_lon_token": n_amb,
            "ambiguous_pct": round(100.0 * n_amb / n, 2) if n else None,
            "per_token": [{"token": t, "n": c,
                           "pct": round(100.0 * c / n, 2)}
                          for t, c in per_tok.most_common()],
            "top_ambiguous_pairs": [{"tokens": list(k), "n": v}
                                    for k, v in amb_pairs.most_common(10)],
        }

    # ---- the two legs as a CONSISTENCY check on the derived token ---------
    # if both legs reach a lon token, do they reach the SAME one? This is the
    # closest thing to a repeat measurement Alpamayo gives us.
    both = [c for c in ma if c in al]
    same = diff = onlyone = neither = 0
    for c in both:
        la = {t for t, ax, r in rx if ax == "lon" and r.search(ma[c] or "")}
        lb = {t for t, ax, r in rx if ax == "lon" and r.search(al[c] or "")}
        if not la and not lb:
            neither += 1
        elif not la or not lb:
            onlyone += 1
        elif la == lb:
            same += 1
        else:
            diff += 1
    res["cross_leg_lon_token_consistency"] = {
        "_question": ("Alpamayo's two reason legs are semi-independent draws "
                      "(A2: only 22.2 % identical strings). When BOTH reach a "
                      "lon token, do they reach the SAME one? This is the "
                      "nearest thing to a repeat measurement we have."),
        "n_clips_with_both_legs": len(both),
        "n_both_reach_same_token_set": same,
        "n_both_reach_different_token_set": diff,
        "n_only_one_leg_reaches": onlyone,
        "n_neither_reaches": neither,
        "consistency_pct_when_both_speak": (
            round(100.0 * same / (same + diff), 2) if (same + diff) else None),
    }

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
