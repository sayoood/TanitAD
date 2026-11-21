#!/usr/bin/env python3
"""P6d - "no member named <clip>.lidar_top_360fov.parquet": is the LiDAR ABSENT, or spelled
differently? Absence at one spelling is not absence (CLAUDE.md operating standard rule 2).

For each fetch-stage failure in `bev_gt_extra/manifest.jsonl`: read the chunk zip's central
directory and list EVERY member whose name contains the clip id, plus a same-breath control --
the member count of the zip must be > 0 and another clip of the same chunk must be findable.
Writes `raw/p6d_missing_lidar_probe.json` (sha12 only).
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6b_extra_corpus as X  # noqa: E402
from lidar_fetch import LIDAR_TMPL, REPO_ID, chunk_map, hf_token, redact_text, sha12  # noqa: E402


def main() -> int:
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfFileSystem
    fails = [json.loads(l) for l in (X.GT_EXTRA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    fails = {r["clip_sha12"] for r in fails if r.get("stage") == "fetch"}
    import glob, gzip
    files = glob.glob(X.TRAIN_LABELS_GLOB, recursive=True)
    ids = []
    with gzip.open(files[0], "rt", encoding="utf-8") as fh:
        for line in fh:
            ids.append(json.loads(line)["clip_id"])
    by = {sha12(c): c for c in ids}
    chunks = chunk_map()
    fs = HfFileSystem(token=hf_token(), skip_instance_cache=True)
    out = {"schema": "tanitad.missing_lidar_probe/1", "evidence_class": "MEASURED (ours, zip central directories)",
           "clips": []}
    for s in sorted(fails):
        cid = by[s]
        ch = chunks[cid]
        path = f"datasets/{REPO_ID}/" + LIDAR_TMPL.format(chunk=ch)
        with fs.open(path, "rb") as fh:
            names = zipfile.ZipFile(fh).namelist()
        hits = [n for n in names if cid in n]
        same_chunk_others = [c for c in ids if chunks.get(c) == ch and c != cid]
        ctrl_found = sum(1 for c in same_chunk_others if f"{c}.lidar_top_360fov.parquet" in names)
        rec = {"clip": s, "chunk": int(ch), "zip_members": len(names),
               "members_containing_clip_id": [redact_text(h) for h in hits],
               "control_same_chunk_train_clips": len(same_chunk_others),
               "control_found_under_expected_name": ctrl_found}
        out["clips"].append(rec)
        print(json.dumps(rec), flush=True)
    (HERE.parent / "raw" / "p6d_missing_lidar_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
