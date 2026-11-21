"""STEP 5 — verify the re-fuse BY CONTENT, and measure what the unified floor moved.

⭐ THIS RUN HAS TWO PRE-REGISTERED PREDICTIONS, BOTH COMMITTED BEFORE IT RAN, AND
EITHER OUTCOME IS A RESULT:

  P1  THE 115 CLIPS' FUSED RECORDS ARE BYTE-IDENTICAL BEFORE AND AFTER.
      Their SAM3 input did not change, the fuser did not change, and the Engine A
      sidecar did not change. If any of them moves, something is non-deterministic
      or a shared input was mutated — and that would invalidate the 86's delta too,
      because the two would no longer be separable. This is the CONTROL and it is
      the reason the delta is attributable at all.

  P2  NOT ONE OF THE FOUR LABEL FAMILIES MOVES ON ANY OF THE 201.
      `…/2026-08-17-aug120-refuse/AUG120_REFUSE.md` §1 MEASURED that swapping 115
      clips from absent-perception to full v2 perception changed **zero** g_str,
      a_str, a_tac_lat and a_tac_lon tokens, because `emit_vocab` reads the VLM,
      Alpamayo and Engine A while SAM3 reaches only `corroborate()` and the census.
      If that structural claim is right, re-detecting the 86 must also move zero.
      ⚠️ A NON-ZERO HERE WOULD REFUTE THE ORTHOGONALITY CLAIM, not confirm a
      perception win — and it would be the more interesting outcome.

⛔ AND THE COMPARISON IS NOT A v1-vs-v2 PER-CONCEPT DELTA. The 86 change floor AND
extraction at once, on a population with 2.63x the detections; presenting
"car 1 413 -> N" as a threshold effect would be the `--v2` conflation defect. What is
reported instead is what the corpus can now DO — pool a rate across all 201 — and the
per-leg numbers stay labelled with the leg that produced them.

Writes: raw/f5_refuse_delta.json
        raw/fused_aug120_v3_index.jsonl   (per-clip tokens + md5, 201 rows)
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os

FAMILIES = ("g_str", "a_str", "a_tac_lat", "a_tac_lon")


def load(d: str) -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(p)
        if b.startswith("_"):
            continue
        raw = open(p, "rb").read()
        rec = json.loads(raw.decode("utf-8"))
        rec["_md5"] = hashlib.md5(raw).hexdigest()
        rec["_bytes"] = len(raw)
        out[rec.get("clip_id") or b[:-5]] = rec
    return out


def tok(rec: dict, field: str):
    """⚠️ THE FOUR FAMILIES LIVE UNDER `vocab`, NOT AT TOP LEVEL — MEASURED on a
    banked record, not assumed. Reading `rec[field]` returns None for every clip,
    which would make "zero labels changed" TRUE FOR THE WRONG REASON: the
    comparison would be None-vs-None on all 201 and P2 would 'hold' while
    measuring nothing. That is a verification that cannot fail, which is the
    C77/C18 shape."""
    v = (rec.get("vocab") or {}).get(field)
    if isinstance(v, dict):
        return v.get("token")
    return v


def perception_of(rec: dict) -> dict:
    p = rec.get("perception") or {}
    cen = p.get("census") or {}
    return {
        "absent": bool(p.get("absent")),
        "engine": (p.get("engine") or {}),
        "census_state": cen.get("state"),
        "n_agents": cen.get("n_agents"),
        "n_tracks": len(p.get("tracks") or []),
        "scene": bool(p.get("scene")),
        "n_scene_det": ((p.get("scene") or {}).get("n_scene_det_total") or 0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="fused_aug120_v2 (mixed floor)")
    ap.add_argument("--after", required=True, help="fused_aug120_v3 (unified)")
    ap.add_argument("--leg86", required=True,
                    help="json list OR dir naming the 86 re-detected clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-out", required=True)
    a = ap.parse_args(argv)

    before, after = load(a.before), load(a.after)
    if os.path.isdir(a.leg86):
        the86 = {os.path.basename(p)[:-5]
                 for p in glob.glob(os.path.join(a.leg86, "*.json"))
                 if not os.path.basename(p).startswith("_")}
    else:
        the86 = set(json.load(open(a.leg86, encoding="utf-8")))
    the115 = sorted(set(after) - the86)

    assert set(before) == set(after), (
        f"the two corpora cover different clips: only-before "
        f"{sorted(set(before)-set(after))[:3]}, only-after "
        f"{sorted(set(after)-set(before))[:3]}")
    assert len(the86) == 86, f"leg86 names {len(the86)} clips"

    # ---- P1: the 115 are the CONTROL ------------------------------------- #
    moved115 = [c for c in the115 if before[c]["_md5"] != after[c]["_md5"]]
    moved86 = [c for c in sorted(the86) if before[c]["_md5"] != after[c]["_md5"]]

    # ---- P2: the four label families -------------------------------------- #
    # ⛔ POSITIVE CONTROL FOR THE COMPARISON ITSELF. "zero tokens changed" is
    # also what a comparison reading the WRONG FIELD reports — None vs None on
    # all 201, a check that cannot fail. Refuse to report P2 unless the reader
    # can actually see tokens.
    seen_tok = {f: sum(1 for c in after if tok(after[c], f) is not None)
                for f in FAMILIES}
    assert seen_tok["g_str"] > 0 and seen_tok["a_str"] > 0, (
        f"the token reader sees nothing ({seen_tok}) — `vocab.<family>.token` "
        "has moved and P2 would 'hold' while measuring nothing")
    fam = {}
    for f in FAMILIES:
        changed = [c for c in sorted(after)
                   if tok(before[c], f) != tok(after[c], f)]
        fam[f] = {
            "n_changed": len(changed),
            "n_changed_in_86": len([c for c in changed if c in the86]),
            "n_changed_in_115": len([c for c in changed if c not in the86]),
            "examples": [{"clip_id": c, "before": tok(before[c], f),
                          "after": tok(after[c], f)} for c in changed[:5]],
            "hist_after": dict(collections.Counter(
                tok(after[c], f) for c in sorted(after)).most_common()),
        }

    # ---- perception, per leg (NEVER pooled across the before-state) -------- #
    def leg(recs, clips):
        per = collections.Counter()
        absent = tracks = scene = nscene = 0
        states = collections.Counter()
        floors = collections.Counter()
        for c in clips:
            p = perception_of(recs[c])
            absent += int(p["absent"])
            tracks += int(p["n_tracks"] or 0)
            scene += int(p["scene"])
            nscene += int(p["n_scene_det"] or 0)
            states[p["census_state"]] += 1
            floors[str((p["engine"] or {}).get("confidence_threshold"))] += 1
            for t in (recs[c].get("perception") or {}).get("tracks") or []:
                per[t.get("concept")] += 1
        return {"n_clips": len(clips), "perception_absent": absent,
                "n_tracks": tracks, "clips_with_scene": scene,
                "n_scene_det_total": nscene,
                "census_states": dict(states),
                "engine_floors": dict(floors),
                "per_concept_tracks": dict(per.most_common())}

    out = {
        "class": "MEASURED",
        "n_clips": len(after),
        "P1_control_115_byte_identical": {
            "prediction": "the 115 clips' fused records do not move",
            "n_moved": len(moved115), "moved": moved115[:10],
            "HOLDS": not moved115},
        "P2_label_families_unmoved": {
            "prediction": "zero token changes on all four families, because "
                          "emit_vocab does not read SAM3",
            "tokens_visible_positive_control": seen_tok,
            "families": fam,
            "HOLDS": all(v["n_changed"] == 0 for v in fam.values())},
        "n_86_records_that_moved": len(moved86),
        "before_mixed_floor": {"the86_leg": leg(before, sorted(the86)),
                               "the115_leg": leg(before, the115)},
        "after_unified_floor": {"the86_leg": leg(after, sorted(the86)),
                                "the115_leg": leg(after, the115),
                                "ALL_201_POOLED": leg(after, sorted(after))},
        "pooling_note": "⭐ `after_unified_floor.ALL_201_POOLED` is the first "
                        "aug120 perception total that is legitimate to quote: "
                        "before the re-run the 201 spanned two detection floors "
                        "and no per-concept rate over them meant anything. "
                        "⚠️ the two `before_mixed_floor` legs are NOT to be "
                        "differenced against each other.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(a.index_out, "w", encoding="utf-8") as fh:
        for c in sorted(after):
            fh.write(json.dumps({
                "clip_id": c, "md5": after[c]["_md5"],
                "bytes": after[c]["_bytes"],
                "moved_vs_mixed_floor": after[c]["_md5"] != before[c]["_md5"],
                **{f: tok(after[c], f) for f in FAMILIES},
                "perception": perception_of(after[c])}) + "\n")

    print(f"[f5] P1 control: {len(moved115)} of 115 moved -> "
          f"HOLDS={not moved115}")
    for f in FAMILIES:
        print(f"[f5] P2 {f}: {fam[f]['n_changed']} changed "
              f"({fam[f]['n_changed_in_86']} in the 86)")
    pooled = out["after_unified_floor"]["ALL_201_POOLED"]
    print(f"[f5] pooled 201: tracks {pooled['n_tracks']} · absent "
          f"{pooled['perception_absent']} · scene clips "
          f"{pooled['clips_with_scene']} · floors {pooled['engine_floors']}")
    print("F5_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
