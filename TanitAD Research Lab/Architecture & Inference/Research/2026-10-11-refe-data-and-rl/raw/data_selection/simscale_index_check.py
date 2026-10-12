#!/usr/bin/env python3
"""SimScale round-4 history index check (2026-10-12; the PI approved downloading this ONE 0.9 MB file).

Reads the downloaded index `simscale_pdm_v1.0-4_sensor_blobs_hist.json` (shard name -> list of scene tokens) and answers what it CAN answer:
scene count, token suffixes, overlap of the base tokens with navtrain / navtest, and (from the HF file listing, metadata only) the size of every
history shard.  What it CANNOT answer: the camera set -- the index holds no sensor names (MEASURED; DATA_SELECTION_REVIEW 2.3 had expected it to).

Run (dev box, from a directory that is NOT the download dir):
  python -I simscale_index_check.py <index.json> <PKG>/splits/navtrain.yaml <PKG>/raw/p2_r2_instrument/baseline/navtest_full_tokens.json <out.json>
"""
import hashlib
import json
import re
import sys

REPO, REV = "OpenDriveLab/SimScale", "06a4fc88f48d8c3ac7c16cc1df05f3caab6397ae"
SHARDS = "SimScale_data/synthetic_reaction_pdm_v1.0-4/simscale_pdm_v1.0-4_sensor_blobs_hist"


def main() -> int:
    idx, navtrain, navtest, out = sys.argv[1:5]
    raw = open(idx, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    toks = [t for v in d.values() for t in v]
    base = {t.split("-")[0] for t in toks}
    txt = open(navtrain, encoding="utf-8").read()
    tr = set(re.findall(r"^  - '?([0-9a-f]{16})'?\s*$", txt.split("\ntokens:")[1], re.M))
    te = set(json.load(open(navtest, encoding="utf-8"))["tokens"])
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    sizes = {e.path.rsplit("/", 1)[-1]: int(getattr(e, "size", 0) or 0)
             for e in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REV, path_in_repo=SHARDS)}
    per = {k: len(v) for k, v in d.items()}
    res = {
        "source": {"repo": REPO, "revision": REV, "file": SHARDS + ".json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "index_shards": len(d), "scene_tokens": len(toks), "unique_tokens": len(set(toks)), "unique_base_tokens": len(base),
        "token_suffixes": sorted({t.split("-")[1] for t in toks if "-" in t}),
        "tokens_per_shard_min_max": [min(per.values()), max(per.values())],
        "navtrain_tokens": len(tr), "navtest_tokens": len(te),
        "base_in_navtrain": len(base & tr), "base_in_navtest": len(base & te), "base_in_neither": len(base - tr - te),
        "hist_shard_files": len(sizes), "hist_total_bytes": sum(sizes.values()),
        "hist_shard_bytes_min_max": [min(sizes.values()), max(sizes.values())],
        "camera_set_answerable_from_index": False,
        "evidence": "MEASURED 2026-10-12 (dev box): index downloaded at the pinned revision; shard sizes from the HF listing (no shard downloaded)",
    }
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
