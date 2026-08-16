#!/usr/bin/env python3
"""A1 — CHARACTERISE THE ALPAMAYO `meta_action` LEG. Measurement, not design.

Answers, at full corpus n and from the PRIMARY artifact (`records.parquet`,
sha256 ecae276d…, 23,644 rows / 4,729 clips, banked in
`…/2026-08-06-alpamayo-augmentation/a2_records_stats.json`):

  1. What is the actual taxonomy? Enumerate DISTINCT values per axis with
     frequencies — free text, enum, or structured?
  2. Does a `reasoning` field exist AS DATA, or is that an assumption?
     (The PI's question names "reasoning" explicitly. `raw_json.cot` is the
     candidate; this script establishes its presence, coverage and content.)
  3. How stable is the emission — how many rows fail to parse, and what
     fraction of the axis vocabulary is degenerate?

⛔ THE VOCABULARY IS READ, NEVER ASSERTED. `a2_parse_meta_action.py` kept
`LAT2DIR` as an OBSERVED list on purpose: a value outside it is a FINDING
about the taxonomy, not something to coerce into a neighbour. This script
enumerates rather than validates, so an unseen token appears as itself.

⚠️ Sampling: the generation is `temperature 0.6`, seed 42, one draw per clip.
ONE DRAW IS NOT THE MODEL'S MODE. Stability across draws is UNMEASURED and
every number here inherits that caveat.
⚠️ Contamination: the clips are PhysicalAI-AV, which Alpamayo lists as TRAINING
data. Overlap is UNRESOLVED — treat Alpamayo agreement as a TEACHER signal,
never as an independent ground truth.

Usage:
  python tac_a1_alpamayo_taxonomy.py --records <records.parquet> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import re

#: The three axes Alpamayo declares. Read from the emission, not assumed.
AXES = ("Longitudinal", "Lateral", "Lane")


def parse_axes(text: str) -> dict:
    """Split one meta_action generation into its axes + the chain-of-causation.

    Mirrors `…/2026-08-05-alpamayo2-super/tools/a2_parse_meta_action.py:53`
    so the two readings of the same field cannot drift.
    """
    out: dict = {}
    for axis in AXES:
        m = re.search(rf"{axis}:\s*([^.\n<]+)", text)
        out[axis.lower()] = m.group(1).strip() if m else None
    first = min((text.find(f"{a}:") for a in AXES if f"{a}:" in text), default=-1)
    out["cot_inline"] = text[:first].strip() if first > 0 else None
    return out


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(a.records)
    m = df[df.task == "meta_action"]

    res: dict = {
        "_evidence_class": "MEASURED (ours; primary artifact records.parquet)",
        "_sampling_caveat": (
            "temperature 0.6, seed 42, ONE draw per clip. One draw is not the "
            "model's mode; cross-draw stability is UNMEASURED."),
        "_contamination_caveat": (
            "clips are PhysicalAI-AV, which Alpamayo lists as TRAINING data; "
            "overlap UNRESOLVED. Alpamayo is a TEACHER, not ground truth."),
        "n_meta_action_rows": int(len(m)),
        "n_unique_clips": int(m.clip_id.nunique()),
    }

    # ---- 1. is `raw_json` structured, and which keys does it carry? --------
    keysets: collections.Counter = collections.Counter()
    rows: list[dict] = []
    bad_json = 0
    for _, r in m.iterrows():
        try:
            d = json.loads(r["raw_json"])
        except Exception:                                        # noqa: BLE001
            bad_json += 1
            continue
        keysets[tuple(sorted(d.keys()))] += 1
        # every field in this record is a LIST of one — unwrap honestly
        def one(k):
            v = d.get(k)
            if isinstance(v, list):
                return v[0] if v else None
            return v
        raw = one("raw_outputs") or ""
        ma = one("meta_action")
        rows.append({
            "clip_id": r["clip_id"],
            "raw": raw,
            "meta_action_field": ma,
            "cot_field": one("cot"),
            "answer_field": one("answer"),
            "box_field": one("box"),
            "cot_auto_labeling_field": one("cot_auto_labeling"),
        })
    res["n_bad_json"] = bad_json
    res["raw_json_keysets"] = {"|".join(k): v for k, v in keysets.most_common()}

    # ---- 2. THE REASONING FIELD — does it exist as DATA? -------------------
    n = len(rows)
    cot_present = sum(1 for r in rows if r["cot_field"])
    cot_lens = sorted(len(r["cot_field"] or "") for r in rows)
    cot_texts = [r["cot_field"] for r in rows if r["cot_field"]]
    cot_tally = collections.Counter(cot_texts)
    cotal_present = sum(1 for r in rows if r["cot_auto_labeling_field"])
    box_present = sum(1 for r in rows if r["box_field"])
    res["reasoning_field"] = {
        "_question": "the PI names 'reasoning' — does it exist as data?",
        "field_name": "raw_json.cot (Alpamayo's 'Chain-of-Causation')",
        "n_rows": n,
        "n_non_empty": cot_present,
        "coverage_pct": round(100.0 * cot_present / n, 2) if n else None,
        "n_distinct_strings": len(cot_tally),
        "distinct_pct_of_rows": (round(100.0 * len(cot_tally) / cot_present, 2)
                                 if cot_present else None),
        "len_chars_median": cot_lens[len(cot_lens) // 2] if cot_lens else None,
        "len_chars_p05": cot_lens[int(0.05 * len(cot_lens))] if cot_lens else None,
        "len_chars_p95": cot_lens[int(0.95 * len(cot_lens))] if cot_lens else None,
        "top_50_strings": cot_tally.most_common(50),
        "_sibling_fields": {
            "cot_auto_labeling_non_empty": cotal_present,
            "box_non_empty": box_present,
        },
    }
    # `answer` vs `cot`: are they the same string? (a duplicated field is not
    # a second source — this is the check that stops us double-counting)
    same = sum(1 for r in rows
               if (r["cot_field"] or "") == (r["answer_field"] or ""))
    res["reasoning_field"]["answer_equals_cot_rows"] = same
    res["reasoning_field"]["answer_equals_cot_pct"] = (
        round(100.0 * same / n, 2) if n else None)

    # ---- 3. THE TAXONOMY — enumerate, never validate ----------------------
    parsed = [dict(parse_axes(r["raw"]), clip_id=r["clip_id"]) for r in rows]
    tax = {}
    for ax in ("longitudinal", "lateral", "lane"):
        c = collections.Counter(p[ax] for p in parsed)
        tot = sum(c.values())
        tax[ax] = {
            "n_distinct_values": len([k for k in c if k is not None]),
            "n_null_unparsed": c.get(None, 0),
            "values": [{"value": k, "n": v,
                        "pct": round(100.0 * v / tot, 3)}
                       for k, v in c.most_common() if k is not None],
        }
    res["taxonomy"] = tax
    res["_taxonomy_kind"] = (
        "STRUCTURED-ENUM-IN-TEXT: a fixed small vocabulary emitted as free "
        "text on three labelled lines. It is not a typed enum field (no "
        "schema enforces it) and it is not open free text (the observed "
        "value set is small and closed). It must be PARSED, and an unparsed "
        "row must be counted, never coerced.")

    # the joint distribution — the whole point of a 3-axis taxonomy
    joint = collections.Counter(
        (p["longitudinal"], p["lateral"], p["lane"]) for p in parsed)
    res["joint_lon_lat_lane"] = [
        {"longitudinal": k[0], "lateral": k[1], "lane": k[2], "n": v}
        for k, v in joint.most_common(60)]
    res["n_distinct_joint_cells"] = len(joint)

    # ⭐ THE FACTORISATION EVIDENCE: how often do two axes move at once?
    # (this is the claim that our 5-way mixed softmax cannot represent)
    def nontrivial_lon(v):
        return v is not None and v != "Constant Speed"
    def nontrivial_lat(v):
        return v is not None and v != "Go Straight"
    both = sum(1 for p in parsed
               if nontrivial_lon(p["longitudinal"]) and nontrivial_lat(p["lateral"]))
    res["simultaneous_lon_and_lat"] = {
        "_definition": ("longitudinal not in {null, 'Constant Speed'} AND "
                        "lateral not in {null, 'Go Straight'} — i.e. the "
                        "vehicle is declared to be doing something on BOTH "
                        "axes at once, which a single 5-way softmax over "
                        "[lane_keep, turn_left, turn_right, accelerate, "
                        "brake_stop] cannot represent."),
        "n": both, "n_total": n,
        "pct": round(100.0 * both / n, 2) if n else None,
    }

    # per-clip export so the agreement stage (A3) can join on clip_id
    res["_per_clip_path"] = a.out.replace(".json", "_per_clip.jsonl")
    with open(res["_per_clip_path"], "w", encoding="utf-8") as fh:
        for p, r in zip(parsed, rows):
            fh.write(json.dumps({
                "clip_id": p["clip_id"],
                "longitudinal": p["longitudinal"],
                "lateral": p["lateral"],
                "lane": p["lane"],
                "cot": r["cot_field"],
            }) + "\n")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    slim = {k: v for k, v in res.items()
            if k not in ("joint_lon_lat_lane",)}
    slim["reasoning_field"] = {
        k: v for k, v in slim["reasoning_field"].items()
        if k != "top_50_strings"}
    print(json.dumps(slim, indent=1, ensure_ascii=False)[:6000])
    print(f"\n[out] {a.out}\n[out] {res['_per_clip_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
