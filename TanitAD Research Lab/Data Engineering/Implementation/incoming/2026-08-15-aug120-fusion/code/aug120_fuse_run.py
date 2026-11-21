"""Fuse the aug120 batches (PH1) — merge the overlapping label trees, then run
`stack/scripts/ph1_fuse.py` ONCE, exactly as the val-600 production run did.

WHY A MERGE STEP EXISTS (MEASURED 2026-08-15 against the far side of
`Sayood/tanitad-ph0-aug120`): the aug120 labels were produced by TWO
overlapping pipeline passes — a partial BATCH=8 pass (tags batch_00000..184
step 8; its 192 slice was never pushed) and a completed BATCH=40 pass (tags
batch_00000/00040/.../00160/00200, the `AUG120_DONE` provenance). The union
covers all 201 clips exactly once each for v2; naive per-directory fusion
would fuse 152 clips twice. Duplicate v2 records are content-identical
(151/152 differ ONLY in the `_calls` meta field; 0 substantive diffs), which
is asserted here — the source-choice policy below is only valid because of it.

SOURCE POLICY (declared, recorded per clip in _label_sources.json):
  * sam3 source = the tag carrying the clip's sam3 record; if several,
    prefer a completed-run tag (00000/00040/00080/00120/00160/00200), then
    lexicographic.
  * v2 source = the sam3 source tag when its v2 also holds the clip (keeps
    the (v2, sam3) pair from one pipeline pass), else the same preference.

SAM3 COVERAGE IS PARTIAL BY DEFECT: `aug120_pipeline.py` never passed `--n`
to `ph0_sam3.py`, whose default is 4 (`ph0_sam3.py:411`), so each sam3.json
holds only the first 4 clips of its batch's v2 order — 86/201 distinct.
batch_00184's sam3 stage additionally produced nothing (runbook §6.11).
The 115 uncovered clips are fused as NAMED PARTIALS via
`--missing-sam3-ok AUG120_SAM3_STAGE_GAP` (perception marked absent per
record; SAM3-dependent checks not_computable — never fabricated).

Usage:
  PYTHONPATH=<stack> python aug120_fuse_run.py --work <scratch/aug120> \
      [--stack <repo>/stack]
Expects under --work:  labels/batch_*/{v2/ph0_v2.json,sam3/sam3.json},
ego/<cid>.npz (201), aux/records.parquet, aux/w120_loc.json,
aux/w120val_600__clips.json.  Writes merged/, fused_aug120/, accounting.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import OrderedDict

MISSING_REASON = "AUG120_SAM3_STAGE_GAP"
COMPLETED_RUN_TAGS = {"batch_00000", "batch_00040", "batch_00080",
                      "batch_00120", "batch_00160", "batch_00200"}


def strip_meta(d: dict) -> str:
    return json.dumps({k: v for k, v in d.items() if not k.startswith("_")},
                      sort_keys=True)


def choose(tags: list[str]) -> str:
    return sorted(tags, key=lambda t: (t not in COMPLETED_RUN_TAGS, t))[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("aug120_fuse_run")
    ap.add_argument("--work", required=True)
    ap.add_argument("--stack", default=None,
                    help="stack dir (default: resolved from this file's repo)")
    a = ap.parse_args(argv)
    work = os.path.abspath(a.work)
    stack = a.stack or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), *[".."] * 5, "stack")
    stack = os.path.abspath(stack)
    sys.path.insert(0, stack)
    sys.path.insert(0, os.path.join(stack, "scripts"))
    import pandas as pd  # noqa: E402

    # ---- population: the reconstructed todo list -------------------------- #
    aux = os.path.join(work, "aux")
    rec = set(pd.read_parquet(os.path.join(aux, "records.parquet"))["clip_id"]
              .astype(str).unique())
    loc = json.load(open(os.path.join(aux, "w120_loc.json")))
    done = set(json.load(open(os.path.join(aux, "w120val_600__clips.json"))))
    todo = sorted((rec & set(loc)) - done)
    assert len(todo) == 201, f"todo reconstruction broke: {len(todo)}"

    # ---- load every far-side batch file ----------------------------------- #
    v2_files, s3_files = OrderedDict(), OrderedDict()
    for p in sorted(glob.glob(os.path.join(work, "labels", "batch_*", "v2",
                                           "ph0_v2.json"))):
        tag = p.replace("\\", "/").split("/")[-3]
        v2_files[tag] = {c["clip_id"]: c for c in
                         json.load(open(p, encoding="utf-8"))["clips"]}
    for p in sorted(glob.glob(os.path.join(work, "labels", "batch_*", "sam3",
                                           "sam3.json"))):
        tag = p.replace("\\", "/").split("/")[-3]
        s3_files[tag] = {c["clip_id"]: c for c in
                         json.load(open(p, encoding="utf-8"))["clips"]}

    v2_where: dict[str, list] = {}
    s3_where: dict[str, list] = {}
    for tag, m in v2_files.items():
        for cid in m:
            v2_where.setdefault(cid, []).append(tag)
    for tag, m in s3_files.items():
        for cid in m:
            s3_where.setdefault(cid, []).append(tag)
    assert set(v2_where) == set(todo), (
        f"v2 union != todo: missing {sorted(set(todo) - set(v2_where))[:3]} "
        f"extra {sorted(set(v2_where) - set(todo))[:3]}")
    assert set(s3_where) <= set(todo)

    # duplicate v2 records must be content-identical (modulo _meta) — the
    # source policy is only admissible under this measured fact
    n_dup_diff = 0
    for cid, tags in v2_where.items():
        if len({strip_meta(v2_files[t][cid]) for t in tags}) > 1:
            n_dup_diff += 1
    assert n_dup_diff == 0, (
        f"{n_dup_diff} clips have SUBSTANTIVELY different v2 records across "
        "batch files — the pick-one policy is not admissible; stop and diff")

    # ---- choose sources, build merged inputs ------------------------------ #
    sources = {}
    merged_v2, merged_s3 = [], []
    for cid in todo:
        s3_tag = choose(s3_where[cid]) if cid in s3_where else None
        if s3_tag and cid in v2_files.get(s3_tag, {}):
            v2_tag = s3_tag
        else:
            v2_tag = choose(v2_where[cid])
        sources[cid] = {"v2": v2_tag, "sam3": s3_tag}
        merged_v2.append(v2_files[v2_tag][cid])
        if s3_tag:
            merged_s3.append(s3_files[s3_tag][cid])

    mdir = os.path.join(work, "merged")
    os.makedirs(mdir, exist_ok=True)
    json.dump({"clips": merged_v2, "n": len(merged_v2),
               "_note": "deduped union of batch_*/v2/ph0_v2.json; "
                        "per-clip source in fused _label_sources.json"},
              open(os.path.join(mdir, "ph0_v2.json"), "w"), indent=1)
    json.dump({"engine": "C_sam3", "n_clips": len(merged_s3),
               "clips": merged_s3,
               "_note": "deduped union of batch_*/sam3/sam3.json (86/201 — "
                        "ph0_sam3 --n default-4 defect, see run doc)"},
              open(os.path.join(mdir, "sam3.json"), "w"), indent=1)
    print(f"[merge] v2={len(merged_v2)} sam3={len(merged_s3)} "
          f"of todo={len(todo)}", flush=True)

    # ego must be complete BEFORE the run — the fuser degrades silently
    no_ego = [c for c in todo
              if not os.path.exists(os.path.join(work, "ego", f"{c}.npz"))]
    assert not no_ego, f"missing ego npz for {len(no_ego)} clips: {no_ego[:3]}"

    # ---- the fuse: one invocation, val-600 shape + the named partial ------ #
    out = os.path.join(work, "fused_aug120")
    from ph1_fuse import main as fuse_main  # noqa: E402
    rc = fuse_main(["--v2-json", os.path.join(mdir, "ph0_v2.json"),
                    "--sam3", os.path.join(mdir, "sam3.json"),
                    "--ego-root", os.path.join(work, "ego"),
                    "--records", os.path.join(aux, "records.parquet"),
                    "--missing-sam3-ok", MISSING_REASON,
                    "--out", out])
    assert rc == 0, f"fuse rc={rc}"

    # ---- per-batch accounting (fused stats grouped by v2 source tag) ------ #
    fused = {}
    for p in glob.glob(os.path.join(out, "*.json")):
        if os.path.basename(p).startswith("_"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        fused[r["clip_id"]] = r
    assert set(fused) == set(todo), "fused set != todo set"

    acct = {}
    for tag in sorted(set(v2_files) | set(s3_files)):
        rows = [fused[c] for c in todo if sources[c]["v2"] == tag]
        acct[tag] = {
            "v2_records_in": len(v2_files.get(tag, {})),
            "sam3_records_in": len(s3_files.get(tag, {})),
            "clips_attributed_here": len(rows),
            "sam3_attributed_here": sum(1 for c in todo
                                        if sources[c]["sam3"] == tag),
            "corroborated": sum(
                sum(1 for c in r["corroboration"].values()
                    if c.get("verdict") == "corroborated") for r in rows),
            "conflicts": sum(len(r["_conflicts"]) for r in rows),
            "with_alpamayo": sum(1 for r in rows if r["alpamayo"]),
            "sam3_absent": sum(1 for r in rows
                               if r["perception"].get("absent")),
        }
    json.dump(acct, open(os.path.join(out, "_batch_accounting.json"), "w"),
              indent=1)
    json.dump({"policy": "sam3 source tag preferred (completed-run tags "
                         "first), v2 from the same tag when it holds the "
                         "clip; duplicates verified content-identical "
                         "modulo _meta before merging",
               "missing_sam3_reason": MISSING_REASON,
               "sources": sources},
              open(os.path.join(out, "_label_sources.json"), "w"), indent=1)
    print("[acct] per-batch accounting + label sources written", flush=True)
    print("AUG120_FUSE_RUN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
