"""STEP 3 — bank the HOMOGENEITY MANIFEST: the fact the repo can hold, and the test can pin.

⛔ THE CORPUS ITSELF CANNOT BE THE PIN. It is ~24 MB of data, it is not in git (the
v1 and v2 corpora are not either), and a unit test has no network. So what goes into
the repo is the manifest: per clip its schema, its detection floor, its md5, its
detection counts and its liveness verdict — and, at the top, the distinct SETS of
floors and schemas across the whole corpus.

⭐ THAT SET IS THE WHOLE POINT. `len(floors) == 1` is the invariant that was violated
for the entire life of this corpus and that NOTHING could see, because a detection
floor is invisible in a payload — it shows up only as rows that are not there. A
manifest that publishes the set turns an invisible property into a checkable one, and
`stack/tests/test_perception_floor_homogeneity.py` fails the suite the day it grows a
second member.

⚠️ AND THE MANIFEST MUST NOT BE ABLE TO LIE ABOUT ITS OWN COVERAGE. `covers_cohort`
is DERIVED here (`n_records == cohort_n`) and re-derived in the test, and an
incomplete corpus is required to carry its residual by name. An "everything is fine"
manifest over 115 of 201 clips would be this defect's own shape one level up.

Writes: raw/floor_homogeneity_manifest.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    *[".."] * 6))
sys.path.insert(0, os.path.join(REPO, "colab"))
import s2_lab_lib as L                                            # noqa: E402

CONF, SCHEMA = 0.25, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+",
                    help="one or more dirs of <clip_id>.json v2 records")
    ap.add_argument("--aug120", required=True)
    ap.add_argument("--out", required=True)
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
    # ⛔ A clip present in two source dirs is exactly the ambiguity this package
    # exists to remove — the pick would be arbitrary and the floors could differ.
    assert not dupes, (f"{len(dupes)} clips appear in more than one corpus dir "
                       f"(e.g. {sorted(set(dupes))[:3]}) — refusing an arbitrary pick")

    rows = []
    floors: collections.Counter = collections.Counter()
    schemas: collections.Counter = collections.Counter()
    errs: collections.Counter = collections.Counter()
    per = collections.Counter()
    sper = collections.Counter()
    n_live = n_dead = n_nocontrol = 0
    n_noconf = n_noschema = 0
    zero = []
    for cid, p in sorted(paths.items()):
        raw = open(p, "rb").read()
        rec = json.loads(raw.decode("utf-8"))
        assert rec.get("clip_id") == cid, f"{p} carries {rec.get('clip_id')!r}"
        eng = rec.get("engine") or {}
        conf = eng.get("confidence_threshold")
        sch = rec.get("schema_version")
        floors[conf] += 1
        schemas[sch] += 1
        n_noconf += int(conf is None)
        n_noschema += int(sch is None)
        lv = rec.get("liveness")
        if lv is None:
            n_nocontrol += 1
            alive = False
        else:
            alive = any(int(v) > 0 for v in (lv.get("n_det") or {}).values())
            n_live += int(alive)
            n_dead += int(not alive)
        nd = int(rec.get("n_det_total") or 0)
        ns = int(rec.get("n_scene_det_total") or 0)
        ne = int(rec.get("n_err_total") or 0)
        if rec.get("err_kinds"):
            errs.update({k: int(v) for k, v in rec["err_kinds"].items()})
        for k, v in (rec.get("per_concept_hits") or {}).items():
            per[k] += int(v)
        for k, v in (rec.get("per_scene_hits") or {}).items():
            sper[k] += int(v)
        if nd == 0:
            zero.append({"clip_id": cid, "liveness_live": alive,
                         "liveness_n_det": (lv or {}).get("n_det") or {}})
        rows.append({"clip_id": cid, "md5": hashlib.md5(raw).hexdigest(),
                     "bytes": len(raw), "schema_version": sch,
                     "confidence_threshold": conf, "n_det_total": nd,
                     "n_scene_det_total": ns, "n_err_total": ne,
                     "liveness_live": alive})

    n = len(rows)
    covers = n == len(cohort)
    residual = sorted(cohort - set(paths))
    floor_set = sorted(str(k) for k in floors)
    schema_set = sorted(str(k) for k in schemas)
    homogeneous = (len(floors) == 1 and len(schemas) == 1
                   and n_noconf == 0 and n_noschema == 0
                   and float(next(iter(floors))) == CONF
                   and int(next(iter(schemas))) >= SCHEMA)

    man = {
        "class": "MEASURED",
        "what": "per-clip detection floor + schema of the aug120 perception "
                "layer, so a mixed corpus becomes a checkable fact",
        "corpus_dirs": [os.path.abspath(d) for d in a.corpus],
        "cohort": "aug120 201-clip cohort (records.parquet ∩ w120 − w120val_600)",
        "cohort_n": len(cohort),
        "n_records": n,
        "covers_cohort": covers,
        "n_residual": len(residual),
        "residual": residual,
        "expected_confidence_threshold": CONF,
        "expected_schema_version_min": SCHEMA,
        "distinct_confidence_thresholds": floor_set,
        "distinct_schema_versions": schema_set,
        "confidence_threshold_histogram": {str(k): v for k, v in floors.items()},
        "schema_version_histogram": {str(k): v for k, v in schemas.items()},
        "n_records_without_confidence_stamp": n_noconf,
        "n_records_without_schema_stamp": n_noschema,
        "liveness_live": n_live, "liveness_dead": n_dead,
        "records_without_control": n_nocontrol,
        "error_census": dict(errs.most_common()),
        "n_det_total": sum(r["n_det_total"] for r in rows),
        "n_scene_det_total": sum(r["n_scene_det_total"] for r in rows),
        "per_concept_totals": dict(per.most_common()),
        "per_scene_totals": dict(sper.most_common()),
        "zero_det_split": {
            "empty_scene_control_live": sum(1 for z in zero if z["liveness_live"]),
            "dead_control": sum(1 for z in zero if not z["liveness_live"]),
            "clips": zero},
        "HOMOGENEOUS": bool(homogeneous),
        "UNIFIED": bool(homogeneous and covers),
        "pushed_to_hf": False,
        "clips": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)

    # Independent second opinion from the shared C77 predicate.
    cen = L.census_records((
        (r["clip_id"], json.loads(open(paths[r["clip_id"]], "rb").read()))
        for r in rows), want=cohort, require_schema=SCHEMA, require_conf=CONF)
    print(f"[f3] {n}/{len(cohort)} records · floors {floor_set} · "
          f"schemas {schema_set} · live {n_live} dead {n_dead} "
          f"nocontrol {n_nocontrol}")
    print(f"[f3] det {man['n_det_total']} · scene {man['n_scene_det_total']} · "
          f"errors {sum(errs.values())} · zero-split "
          f"{man['zero_det_split']['empty_scene_control_live']} empty / "
          f"{man['zero_det_split']['dead_control']} dead")
    print(f"[f3] census cross-check: n_complete {cen['n_complete']} · "
          f"wrong_conf {cen['wrong_conf']} · wrong_schema {cen['wrong_schema']} "
          f"· pass_ {cen['pass_']}")
    print(f"[f3] HOMOGENEOUS {man['HOMOGENEOUS']} · UNIFIED {man['UNIFIED']} · "
          f"residual {len(residual)}")
    print("F3_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
