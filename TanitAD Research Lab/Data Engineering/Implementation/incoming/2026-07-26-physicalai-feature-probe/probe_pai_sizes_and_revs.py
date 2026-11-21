#!/usr/bin/env python3
"""Per-feature byte sizes + revision history for nvidia/PhysicalAI-Autonomous-Vehicles.

Answers three things the L1/L2 tree walk could not:
  (a) size of one chunk per feature  -> extrapolated feature size over 3,146 chunks
  (b) does an EARLIER tagged revision carry features the current one does not
      (CLAUDE.md: absence at ONE location is not absence — probe a 2nd revision)
  (c) the chunk_0000 size of labels/obstacle.offline, to price the schema probe

METADATA ONLY: `paths_info` returns sizes without downloading blobs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
RTYPE = "dataset"


def _bootstrap():
    for p in HERE.parents:
        if (p / "stack" / "tanitad" / "keys.py").exists():
            sys.path.insert(0, str(p / "stack"))
            break
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    assert os.environ.get("HF_TOKEN")


def main():
    _bootstrap()
    from huggingface_hub import HfApi
    import pandas as pd
    api = HfApi()

    feats = pd.read_csv(HERE / "pai_features.csv")
    out = {"n_features_in_features_csv": int(len(feats)), "sizes": {}}

    # (a)+(c) one chunk per feature, sized via paths_info (no blob fetched).
    # chunk 0000 does not exist for the srr_3 / lrr_1 / mrr_2 radars (they start
    # at 0095/0104) — fall back to the first chunk the tree actually listed.
    l3 = json.loads((HERE / "pai_tree_l3_sample.json").read_text())
    first_of = {}
    for d, ents in l3.items():
        files = [e["path"] for e in ents if e.get("type") == "RepoFile"]
        if files:
            first_of[d] = sorted(files)[0]

    probe_paths = sorted(first_of.values())
    infos = api.get_paths_info(REPO, probe_paths, repo_type=RTYPE, expand=False)
    by_path = {i.path: i for i in infos}
    for d, p in sorted(first_of.items()):
        i = by_path.get(p)
        size = getattr(i, "size", None)
        lfs = getattr(getattr(i, "lfs", None), "size", None)
        out["sizes"][d] = {"probe_file": p, "size_bytes": lfs or size}

    # number of chunk files per feature dir (full listing, names only)
    counts = {}
    for d in sorted(first_of):
        try:
            n = sum(1 for _ in api.list_repo_tree(REPO, repo_type=RTYPE,
                                                  path_in_repo=d,
                                                  recursive=False))
            counts[d] = n
        except Exception as e:  # noqa: BLE001
            counts[d] = f"ERROR {e!r}"
        print(f"[size] {d}: n_chunks={counts[d]} "
              f"first={out['sizes'][d]['size_bytes']}", flush=True)
    out["n_chunk_files"] = counts

    # (b) revisions: does an older tag hold features the head does not?
    revs = {}
    try:
        rinfo = api.list_repo_refs(REPO, repo_type=RTYPE)
        revs["branches"] = [b.name for b in rinfo.branches]
        revs["tags"] = [t.name for t in rinfo.tags]
        revs["converts"] = [c.name for c in getattr(rinfo, "converts", [])]
    except Exception as e:  # noqa: BLE001
        revs["ERROR"] = repr(e)
    print(f"[revs] {revs}", flush=True)

    revs["per_rev_top_level"] = {}
    for r in (revs.get("tags") or []) + (revs.get("branches") or []):
        try:
            ents = [e.path for e in api.list_repo_tree(
                REPO, repo_type=RTYPE, revision=r, recursive=False)]
            revs["per_rev_top_level"][r] = sorted(ents)
        except Exception as e:  # noqa: BLE001
            revs["per_rev_top_level"][r] = [f"ERROR {e!r}"]
        print(f"[revs] {r}: {revs['per_rev_top_level'][r]}", flush=True)
    out["revisions"] = revs

    (HERE / "pai_sizes_and_revs.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[size] done", flush=True)


if __name__ == "__main__":
    main()
