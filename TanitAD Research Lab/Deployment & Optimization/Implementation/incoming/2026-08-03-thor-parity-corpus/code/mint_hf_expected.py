#!/usr/bin/env python3
"""Mint the per-episode SIZE + SHA256 expectation table off the HF LFS metadata.

This is the SOURCE's own record of itself. Verifying a transfer against it is
therefore a genuine end-to-end check; verifying against a number a human typed is
not. (HF stores the sha256 of every LFS object as part of the pointer file, so the
digests come out of the tree API for free — no download.)

🔒 SAFE TO COMMIT: episode uids here are POSITIONAL (``ep_%05d.pt``), never clip ids.
Clip ids are gated-confidential PhysicalAI-AV content and live only on pods
(``parity.py`` §9). Nothing in this table identifies a clip.

Usage:
    python mint_hf_expected.py --path epcache-256px-phase0/physicalai-train-e438721ae894 \
        --out hf_expected_train.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]


def hf_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    txt = (REPO_ROOT / "Keys.txt").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit("no hf_ token in Keys.txt")
    return m.group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Sayood/tanitad-physicalai-w120-256x640cyl")
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:                       # noqa: BLE001
        pass
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    files, no_lfs = {}, []
    for it in api.list_repo_tree(a.repo, repo_type="dataset", recursive=True,
                                 path_in_repo=a.path):
        sz = getattr(it, "size", None)
        if sz is None:
            continue
        name = os.path.basename(it.path)
        lfs = getattr(it, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs else None
        if sha is None:
            no_lfs.append(name)
        files[name] = {"size": int(sz), "sha256": sha}
    out = {"repo": a.repo, "path_in_repo": a.path, "n_files": len(files),
           "n_without_lfs_sha": len(no_lfs), "without_lfs_sha": sorted(no_lfs)[:20],
           "total_bytes": sum(v["size"] for v in files.values()),
           "files": dict(sorted(files.items()))}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
