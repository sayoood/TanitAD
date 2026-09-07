#!/usr/bin/env python3
"""Slim the stage-1 comparison for banking, without losing anything irreproducible."""
import collections
import json

NOTE = (
    "Slimmed for banking. OMITTED and why: "
    "(a) the 1,445 IDENTICAL rows carry no information beyond the count; "
    "(b) the ABSENT_FROM_REPO_HEAD rows are stored as PATHS ONLY, because all 3,837 are "
    "byte-identical to the blob at Thor's checkout HEAD 30d6d60 -- every one is recoverable "
    "with: git cat-file blob 30d6d60:<path> -- so the shas add nothing; "
    "(c) the 9,787 REPO_ONLY paths are a property of the repo, not of Thor, and regenerate "
    "in ~2 s with: GIT_INDEX_FILE=<local-disk-path> git read-tree HEAD && git ls-files -s. "
    "KEPT IN FULL: every DRIFT row, and the irreproducible box-state measurements "
    "(thor_head_blobs.json, thor_census_md5.json, thor_untracked_blobs.json)."
)

d = json.load(open("adjudicate_stage1.json", encoding="utf-8"))
rows = d["rows"]
ab = [r for r in rows if r["state"] == "ABSENT_FROM_REPO_HEAD"]


def td(p):
    return p.split("/")[0]


out = {
    "repo_head": d["repo_head"],
    "thor_head": d["thor_head"],
    "note": NOTE,
    "counts": dict(collections.Counter(r["state"] for r in rows)),
    "absent_from_repo_head_all_equal_thor_head_blob":
        all(r["thor_bytes_equal_thor_head"] for r in ab),
    "absent_from_repo_head_by_top_dir": dict(collections.Counter(td(r["path"]) for r in ab)),
    "absent_from_repo_head_paths": [r["path"] for r in ab],
    "n_repo_only_paths": len(d["repo_only"]),
    "repo_only_by_top_dir": dict(collections.Counter(td(p) for p in d["repo_only"])),
    "drift_rows": [r for r in rows if r["state"] == "DRIFT"],
}
json.dump(out, open("drift_stage1_slim.json", "w", encoding="utf-8"), indent=1)
print("counts:", out["counts"])
print("absent all == thor head blob:", out["absent_from_repo_head_all_equal_thor_head_blob"])
