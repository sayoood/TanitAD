"""Assemble the SAM3 perception leg for the aug120 RE-FUSE, and assert the
facts the assembly depends on BEFORE anything is written.

⛔ THE ASSEMBLY IS TWO RUNS AT TWO DETECTION FLOORS AND THAT IS UNAVOIDABLE
TODAY. `sam3_backfill_v2/` re-ran the 115 clips the batch pipeline never
reached (floor 0.25, schema 2); the other 86 clips still hold ONLY the batch
pipeline's own SAM3 (vendor default 0.5, pre-schema, and it stamps neither
field). Re-running those 86 at 0.25 needs a GPU, which this task does not have.

So the mixture is DECLARED here, MEASURED (disjointness + exact union), and
stamped per record by `ph1_fuse.perception_engine` — never smoothed over.

Writes:  <work>/sam3_refuse/<clip>.json  (201 files, one per clip)
         <work>/inputs_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

V1_FLOOR_NOTE = (
    "batch-pipeline SAM3 leg: vendor default confidence_threshold=0.5, "
    "pre-schema (stamps neither schema_version nor engine.confidence_"
    "threshold). Floor known from the RUN, not from the record.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True, help="pulled sam3_backfill_v2/")
    ap.add_argument("--aug120", required=True, help="cached v1 work dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    merged = os.path.join(a.aug120, "merged")
    v2_labels = json.load(open(os.path.join(merged, "ph0_v2.json"),
                               encoding="utf-8"))["clips"]
    todo = {c["clip_id"] for c in v2_labels}
    s1 = json.load(open(os.path.join(merged, "sam3.json"),
                        encoding="utf-8"))["clips"]
    s1_by = {c["clip_id"]: c for c in s1}
    v2_paths = {os.path.basename(p)[:-5]: p
                for p in glob.glob(os.path.join(a.v2_dir, "*.json"))}

    # ---- the assertions the merge policy rests on ------------------------- #
    assert len(todo) == 201, f"population moved: {len(todo)}"
    assert set(v2_paths) <= todo, "v2 corpus carries clips outside the cohort"
    assert set(s1_by) <= todo, "v1 batch leg carries clips outside the cohort"
    overlap = set(v2_paths) & set(s1_by)
    assert not overlap, (
        f"{len(overlap)} clips appear in BOTH SAM3 legs — the floors would be "
        "silently mixed PER CLIP and the pick would be arbitrary; stop and "
        f"decide explicitly (e.g. {sorted(overlap)[:3]})")
    union = set(v2_paths) | set(s1_by)
    assert union == todo, (
        f"SAM3 legs cover {len(union)} of {len(todo)} clips — "
        f"{len(todo - union)} would fuse as named partials")

    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "*.json")):
        os.remove(f)
    n2 = n1 = 0
    for cid, p in sorted(v2_paths.items()):
        shutil.copyfile(p, os.path.join(a.out, f"{cid}.json"))
        n2 += 1
    for cid, rec in sorted(s1_by.items()):
        with open(os.path.join(a.out, f"{cid}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rec, fh)
        n1 += 1

    man = {
        "class": "MEASURED",
        "n_clips": len(todo), "n_from_sam3_backfill_v2": n2,
        "n_from_batch_pipeline_v1": n1,
        "legs_disjoint": True, "union_equals_cohort": True,
        "v2_leg": {"prefix": "sam3_backfill_v2/", "schema_version": 2,
                   "confidence_threshold": 0.25,
                   "clips": sorted(v2_paths)},
        "v1_leg": {"source": "labels/batch_*/sam3/sam3.json (deduped)",
                   "schema_version": None, "confidence_threshold": None,
                   "note": V1_FLOOR_NOTE, "clips": sorted(s1_by)},
        "MIXED_FLOOR": True,
        "mixed_floor_note": (
            "⛔ 115 clips at floor 0.25 + 86 clips at floor 0.5. Per-concept "
            "detection RATES may not be pooled across the two legs. The "
            "mixture is stamped per record in perception.engine and censused "
            "in _summary.json.perception_engines."),
    }
    json.dump(man, open(os.path.join(os.path.dirname(a.out) or ".",
                                     "inputs_manifest.json"), "w"), indent=1)
    print(f"[inputs] {n2} v2 + {n1} v1 = {n2 + n1} sam3 records -> {a.out}")
    print("BUILD_INPUTS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
