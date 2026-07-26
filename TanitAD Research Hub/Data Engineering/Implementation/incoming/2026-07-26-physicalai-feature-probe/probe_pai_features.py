#!/usr/bin/env python3
"""Probe the 36 features of nvidia/PhysicalAI-Autonomous-Vehicles (gated).

METADATA-ONLY. Walks the HF *tree API* (never `list_repo_files`, which hangs on
this repo's ~millions of files), records every top-level feature directory, its
child layout, file counts and byte sizes, and downloads the dataset card.

Security: the HF token is read in place from the git-ignored ``Keys.txt`` via
``tanitad.keys.load_keys()`` -> os.environ. It is never printed or passed in argv.
TLS goes through the OS trust store (``truststore``) because the dev box sits
behind an intercepting proxy.

Outputs (JSON, next to this script):
    pai_tree_l1.json        top-level entries
    pai_tree_l2.json        children of every top-level dir
    pai_tree_l3_sample.json children of a sampled subdir per feature
    pai_card.md             the raw dataset card (README.md)
    pai_probe_summary.json  the derived 36-feature table
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
RTYPE = "dataset"


def _bootstrap():
    stack = HERE
    for p in HERE.parents:
        if (p / "stack" / "tanitad" / "keys.py").exists():
            stack = p / "stack"
            break
    sys.path.insert(0, str(stack))
    from tanitad.keys import enable_tls, load_keys  # noqa: E402
    ok_tls = enable_tls()
    names = load_keys()
    assert os.environ.get("HF_TOKEN"), "no HF token found in Keys.txt"
    print(f"[probe] tls={ok_tls} keys={sorted(set(names))}", flush=True)


def _ent(e):
    d = {"path": e.path, "type": type(e).__name__}
    for a in ("size", "blob_id", "lfs"):
        v = getattr(e, a, None)
        if a == "lfs" and v is not None:
            d["lfs_size"] = getattr(v, "size", None)
        elif v is not None and a != "lfs":
            d[a] = v
    return d


def tree(api, path, recursive=False, expand=False):
    out = []
    for e in api.list_repo_tree(REPO, repo_type=RTYPE, path_in_repo=path or None,
                                recursive=recursive, expand=expand):
        out.append(_ent(e))
    return out


def main():
    _bootstrap()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    # --- dataset card + repo info -------------------------------------- #
    info = api.dataset_info(REPO, files_metadata=False)
    meta = {
        "id": info.id, "sha": info.sha, "private": info.private,
        "gated": getattr(info, "gated", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": list(getattr(info, "tags", []) or []),
        "last_modified": str(getattr(info, "last_modified", None)),
        "card_data": (info.card_data.to_dict()
                      if getattr(info, "card_data", None) else None),
    }
    (HERE / "pai_repo_info.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[probe] repo sha={info.sha} gated={meta['gated']}", flush=True)

    try:
        card = hf_hub_download(REPO, "README.md", repo_type=RTYPE)
        (HERE / "pai_card.md").write_text(Path(card).read_text(encoding="utf-8"),
                                          encoding="utf-8")
        print("[probe] card saved", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] card FAILED: {e!r}", flush=True)

    # --- L1: top level -------------------------------------------------- #
    l1 = tree(api, "", expand=True)
    (HERE / "pai_tree_l1.json").write_text(json.dumps(l1, indent=2, default=str))
    dirs1 = [e["path"] for e in l1 if e["type"] == "RepoFolder"]
    print(f"[probe] L1: {len(l1)} entries, dirs={dirs1}", flush=True)

    # --- L2: children of each top-level dir ----------------------------- #
    l2 = {}
    for d in dirs1:
        t0 = time.time()
        try:
            l2[d] = tree(api, d, expand=True)
        except Exception as e:  # noqa: BLE001
            l2[d] = [{"ERROR": repr(e)}]
        print(f"[probe] L2 {d}: {len(l2[d])} entries "
              f"({time.time() - t0:.1f}s)", flush=True)
    (HERE / "pai_tree_l2.json").write_text(json.dumps(l2, indent=2, default=str))

    # --- L3: sample one subdir per L2 dir (schema + size shape) --------- #
    l3 = {}
    for d, ents in l2.items():
        subdirs = [e["path"] for e in ents if e.get("type") == "RepoFolder"]
        for sd in subdirs:                      # every L2 dir, first page only
            try:
                page = []
                it = api.list_repo_tree(REPO, repo_type=RTYPE, path_in_repo=sd,
                                        recursive=False, expand=False)
                for i, e in enumerate(it):
                    page.append(_ent(e))
                    if i >= 40:
                        page.append({"path": "...TRUNCATED@41", "type": "note"})
                        break
                l3[sd] = page
            except Exception as e:  # noqa: BLE001
                l3[sd] = [{"ERROR": repr(e)}]
            print(f"[probe] L3 {sd}: {len(l3[sd])}", flush=True)
    (HERE / "pai_tree_l3_sample.json").write_text(
        json.dumps(l3, indent=2, default=str))

    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
