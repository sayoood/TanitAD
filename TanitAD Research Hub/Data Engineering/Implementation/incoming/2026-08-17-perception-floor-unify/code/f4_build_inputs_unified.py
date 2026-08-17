"""STEP 4 — assemble the SAM3 leg for the re-fuse, now that there is only ONE leg.

⛔ WHY THIS IS NOT `…/2026-08-17-aug120-refuse/code/build_inputs.py`, AND WHY THAT
SCRIPT IS NOT BROKEN. That script exists to MERGE two SAM3 legs, and it refuses
unless they are DISJOINT:

    overlap = set(v2_paths) & set(s1_by)
    assert not overlap, "... clips appear in BOTH SAM3 legs — the floors would be
                         silently mixed PER CLIP and the pick would be arbitrary"

Once the 86 are re-detected, all 201 clips are in the v2 corpus and the batch
pipeline's own leg becomes a strict subset of it — so the overlap is 86 and that
assertion FIRES. ⭐ That is the script working, not failing: with both legs holding
the same clip at two different floors, picking one silently is exactly the defect it
was written to prevent. It is left untouched and its refusal is reported as a
result.

What replaces it is not a looser check but a STRICTER one, because a single leg
admits stronger assertions than a merge ever could: every one of the 201 records must
carry schema >= 2, floor == 0.25, a live liveness control and zero errors — i.e. the
corpus must be homogeneous BEFORE anything is fused from it, not merely non-
overlapping.

Writes: <work>/sam3_refuse/<clip>.json   (201 — the name `refuse_run.py` expects)
        <work>/inputs_manifest_unified.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    *[".."] * 6))
sys.path.insert(0, os.path.join(REPO, "colab"))
import s2_lab_lib as L                                            # noqa: E402

CONF, SCHEMA = 0.25, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+",
                    help="dirs holding the unified v2 records (115 + 86)")
    ap.add_argument("--aug120", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)

    cohort = {c["clip_id"] for c in json.load(
        open(os.path.join(a.aug120, "merged", "ph0_v2.json"),
             encoding="utf-8"))["clips"]}

    paths: dict[str, str] = {}
    dupes = []
    for d in a.corpus:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            cid = os.path.basename(p)[:-5]
            if cid.startswith("_"):
                continue
            if cid in paths:
                dupes.append(cid)
            paths[cid] = p

    # ---- the assertions the single-leg policy rests on -------------------- #
    assert len(cohort) == 201, f"population moved: {len(cohort)}"
    assert not dupes, (f"{len(dupes)} clips appear in two corpus dirs "
                       f"(e.g. {sorted(set(dupes))[:3]}) — the pick would be "
                       "arbitrary and the floors could differ")
    extra = set(paths) - cohort
    assert not extra, f"corpus carries {len(extra)} clips outside the cohort"
    missing = cohort - set(paths)
    assert not missing, (
        f"{len(missing)} of 201 clips have NO v2 record (e.g. "
        f"{sorted(missing)[:3]}) — they would fuse as named partials at the old "
        "floor, which is the mixed corpus rebuilt. Finish the re-run first: "
        "…/2026-08-17-perception-floor-unify/code/f2_drive86.py")

    # ⛔ HOMOGENEITY IS ASSERTED BEFORE ANY FUSION, not audited after it.
    cen = L.content_census_local(os.path.dirname(next(iter(paths.values()))),
                                 require_schema=SCHEMA, require_conf=CONF) \
        if len(a.corpus) == 1 else None
    floors, schemas, bad = set(), set(), []
    for cid, p in sorted(paths.items()):
        rec = json.loads(open(p, "rb").read())
        eng = rec.get("engine") or {}
        floors.add(eng.get("confidence_threshold"))
        schemas.add(rec.get("schema_version"))
        lv = rec.get("liveness")
        alive = lv is not None and any(
            int(v) > 0 for v in (lv.get("n_det") or {}).values())
        if not alive or int(rec.get("n_err_total") or 0):
            bad.append(cid)
    assert floors == {CONF}, (
        f"⛔ the corpus is MIXED-FLOOR: {sorted(map(str, floors))}. No "
        "per-concept rate may be pooled across it and nothing fused from it is "
        "attributable.")
    assert all(s is not None and int(s) >= SCHEMA for s in schemas), \
        f"corpus spans schemas {sorted(map(str, schemas))}"
    assert not bad, (f"{len(bad)} records have a dead/absent liveness control "
                     f"or carry errors (e.g. {bad[:3]}) — C77")

    out = os.path.join(a.work, "sam3_refuse")
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "*.json")):
        os.remove(f)
    for cid, p in sorted(paths.items()):
        shutil.copyfile(p, os.path.join(out, f"{cid}.json"))

    man = {
        "class": "MEASURED",
        "n_clips": len(cohort), "n_from_v2_corpus": len(paths),
        "n_from_batch_pipeline_v1": 0,
        "single_leg": True,
        "MIXED_FLOOR": False,
        "confidence_threshold": CONF,
        "schema_version_min": SCHEMA,
        "distinct_floors": sorted(map(str, floors)),
        "distinct_schemas": sorted(map(str, schemas)),
        "supersedes": "…/2026-08-17-aug120-refuse/raw/inputs_manifest.json "
                      "(MIXED_FLOOR: true, 115 @0.25 + 86 @0.5)",
        "note": "⭐ every per-concept detection RATE is now poolable across the "
                "201 — the constraint the mixed corpus imposed is lifted.",
        "clips": sorted(paths),
    }
    mp = os.path.join(a.work, "inputs_manifest_unified.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(f"[inputs] {len(paths)} v2 records (single leg, floor {CONF}, "
          f"schema {sorted(map(str, schemas))}) -> {out}")
    if cen:
        print(f"[inputs] census cross-check: complete {cen['n_complete']} · "
              f"wrong_conf {cen['wrong_conf']} · wrong_schema "
              f"{cen['wrong_schema']} · errors "
              f"{sum(cen['error_census'].values())}")
    print("BUILD_INPUTS_UNIFIED_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
