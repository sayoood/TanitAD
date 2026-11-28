#!/usr/bin/env python3
"""A2 — the REASONING legs, both of them, measured for CONTENT not presence.

The PI asked about "reasoning". A1 established that `meta_action.raw_json.cot`
exists at 100 % coverage. This script asks the harder question: **is there a
SECOND reasoning source, and does either carry the structure a tactical GOAL
label would need?**

Two candidate legs, and they are NOT the same field:

  * `meta_action.cot`      — one sentence, the "Chain-of-Causation" preamble
                             emitted inline before the three axis lines.
  * `auto_labeling.cot_auto_labeling_json` — a JSON object with FOUR declared
                             sub-fields (`critical_components_analysis`,
                             `ego_vehicle_motion_analysis`,
                             `trajectory_analysis`, `chain_of_causation`).

⛔ A DECLARED SUB-FIELD THAT IS ALWAYS `null` IS NOT A SOURCE. The first row
inspected showed three of the four as `null`; presence in a schema is not
presence in the data, and this script counts non-null per sub-field so a
schema cannot be mistaken for a signal. (Same family as the S2 review's
`", ".join(...) or "no agents"` — a shape that looks like a finding.)

⚠️ Whether the two `chain_of_causation` strings AGREE is measured here too: a
duplicated field is one source, not two, and quoting both would be
double-counting.

Usage:
  python tac_a2_reasoning_legs.py --records <records.parquet> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import re

#: Tactical-goal-relevant REASON families, as regex over the cot text. These
#: are DIAGNOSTIC BUCKETS for measuring what the reason field talks about —
#: ⛔ NOT a proposed label mapping. The mapping question is A4's, and it is
#: only asked after the agreement measurement says the sources are trustworthy.
REASON_PROBES = (
    ("lead_vehicle",  r"lead vehicle|vehicle ahead|car ahead|queued|traffic ahead"),
    ("traffic_light", r"traffic light|red light|green light|signal"),
    ("sign",          r"\bsign\b|stop sign|yield sign|speed limit"),
    ("curve",         r"\bcurve\b|bend|curving"),
    ("intersection",  r"intersection|junction|crossroad"),
    ("roundabout",    r"roundabout"),
    ("pedestrian",    r"pedestrian|crosswalk|crossing person"),
    ("parked_obstacle", r"parked|obstacle|debris|cone"),
    ("lane_structure", r"\blane\b"),
    ("speed_bump",    r"speed bump|bump|hump"),
    ("merge",         r"merge|merging|on-ramp|off-ramp|exit"),
    ("clear_road",    r"road ahead is clear|lane ahead is clear|is clear"),
)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(a.records)
    res: dict = {"_evidence_class": "MEASURED (ours; records.parquet)"}

    def unwrap(d, k):
        v = d.get(k)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    # ---- leg 1: meta_action.cot -------------------------------------------
    ma_cot: dict[str, str] = {}
    for _, r in df[df.task == "meta_action"].iterrows():
        d = json.loads(r["raw_json"])
        ma_cot[r["clip_id"]] = unwrap(d, "cot") or ""

    # ---- leg 2: auto_labeling.cot_auto_labeling_json -----------------------
    al_rows = df[df.task == "auto_labeling"]
    sub_nonnull: collections.Counter = collections.Counter()
    sub_seen: collections.Counter = collections.Counter()
    al_coc: dict[str, str] = {}
    n_no_json = 0
    for _, r in al_rows.iterrows():
        d = json.loads(r["raw_json"])
        j = unwrap(d, "cot_auto_labeling_json")
        if not isinstance(j, dict):
            # fall back to parsing the raw string, so a formatting change does
            # not read as an absent field
            try:
                j = json.loads(unwrap(d, "cot_auto_labeling") or "")
            except Exception:                                    # noqa: BLE001
                j = None
        if not isinstance(j, dict):
            n_no_json += 1
            continue
        for k, v in j.items():
            sub_seen[k] += 1
            if v not in (None, "", [], {}):
                sub_nonnull[k] += 1
        coc = j.get("chain_of_causation")
        if isinstance(coc, str) and coc:
            al_coc[r["clip_id"]] = coc

    res["auto_labeling_leg"] = {
        "n_rows": int(len(al_rows)),
        "n_rows_without_parsable_json": n_no_json,
        "_sub_field_table": [
            {"sub_field": k, "n_declared": sub_seen[k],
             "n_non_null": sub_nonnull.get(k, 0),
             "non_null_pct": round(100.0 * sub_nonnull.get(k, 0) / sub_seen[k], 2)}
            for k in sorted(sub_seen)],
        "_rule": ("a declared sub-field that is always null is NOT a source; "
                  "read n_non_null, never the schema."),
    }

    # ---- are the two chain-of-causation strings the SAME source? ----------
    both = [c for c in al_coc if c in ma_cot and ma_cot[c]]
    exact = sum(1 for c in both if al_coc[c].strip().rstrip(".") ==
                ma_cot[c].strip().rstrip("."))
    def toks(s):
        return set(re.findall(r"[a-z]+", s.lower()))
    jacc = [len(toks(al_coc[c]) & toks(ma_cot[c])) /
            max(1, len(toks(al_coc[c]) | toks(ma_cot[c]))) for c in both]
    res["two_reason_legs_are_independent"] = {
        "_question": ("is auto_labeling.chain_of_causation a SECOND source, "
                      "or a duplicate of meta_action.cot? Quoting a duplicate "
                      "as corroboration would be double-counting."),
        "n_clips_with_both": len(both),
        "n_exact_string_match": exact,
        "exact_match_pct": round(100.0 * exact / len(both), 2) if both else None,
        "token_jaccard_mean": round(sum(jacc) / len(jacc), 4) if jacc else None,
        "token_jaccard_median": (round(sorted(jacc)[len(jacc) // 2], 4)
                                 if jacc else None),
    }

    # ---- what do the reasons TALK ABOUT? ----------------------------------
    for name, corpus in (("meta_action_cot", ma_cot),
                         ("auto_labeling_coc", al_coc)):
        tally = {}
        n = len(corpus)
        for probe, pat in REASON_PROBES:
            rx = re.compile(pat, re.I)
            hits = sum(1 for s in corpus.values() if rx.search(s or ""))
            tally[probe] = {"n": hits,
                            "pct": round(100.0 * hits / n, 2) if n else None}
        uncovered = sum(1 for s in corpus.values()
                        if not any(re.search(p, s or "", re.I)
                                   for _, p in REASON_PROBES))
        res[f"{name}_reason_content"] = {
            "n_clips": n, "probes": tally,
            "n_matching_no_probe": uncovered,
            "pct_matching_no_probe": (round(100.0 * uncovered / n, 2)
                                      if n else None),
            "_note": ("DIAGNOSTIC BUCKETS over the reason text — what the "
                      "field talks about. NOT a label mapping; buckets "
                      "overlap and do not sum to 100 %."),
        }
    res["auto_labeling_coc_distinct"] = len(set(al_coc.values()))
    res["meta_action_cot_distinct"] = len(set(v for v in ma_cot.values() if v))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:7000])
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
