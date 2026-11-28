"""MM 2026-11-19: bank E:/Projects/TanitAD-artifacts/v8f-recover (22 checkpoints + small files) to a PRIVATE HF dataset, then verify by sha256.
The checkpoints are uploaded one file per commit (retry, skip if the hub already holds an equal sha256). All non-.pt files go in one tar.
Writes BANK_V8F_RECOVER.json with the local and hub sha256 of every uploaded object, and ends on ZZ BANKRECOVER DONE ZZ / ZZ BANKRECOVER FAIL ZZ <reason>."""
import hashlib
import json
import os
import sys
import tarfile
import time

from huggingface_hub import HfApi

SRC = "E:/Projects/TanitAD-artifacts/v8f-recover"
OUT = "F:/TanitAD-scratch/bank_recover"
REPO = "Sayood/tanitad-v8f-recover-runs"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hub_sha(api):
    out = {}
    for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
        lfs = getattr(e, "lfs", None)
        if lfs is not None:
            out[e.path] = lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
    return out


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        print("ZZ BANKRECOVER FAIL ZZ repo is not private", flush=True)
        sys.exit(1)
    pts, small = [], []
    for root, _, files in os.walk(SRC):
        for f in files:
            p = os.path.join(root, f).replace("\\", "/")
            (pts if f.endswith(".pt") else small).append(p)
    pts.sort(); small.sort()
    tar_p = os.path.join(OUT, "v8f_recover_small.tar")
    with tarfile.open(tar_p, "w") as t:
        for p in small:
            t.add(p, arcname=os.path.relpath(p, SRC).replace("\\", "/"))
    items = [(p, os.path.relpath(p, SRC).replace("\\", "/")) for p in pts] + [(tar_p, "v8f_recover_small.tar")]
    local = {rel: sha(p) for p, rel in items}
    print(f"local hashed: {len(local)} objects", flush=True)
    have = hub_sha(api)
    for p, rel in items:
        if have.get(rel) == local[rel]:
            print(f"skip (equal on hub) {rel}", flush=True)
            continue
        for k in range(4):
            try:
                api.upload_file(path_or_fileobj=p, path_in_repo=rel, repo_id=REPO, repo_type="dataset",
                                commit_message=f"bank {rel}")
                print(f"uploaded {rel} {os.path.getsize(p)}", flush=True)
                break
            except Exception as e:
                print(f"retry {k} {rel}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 * (k + 1))
        else:
            print(f"ZZ BANKRECOVER FAIL ZZ upload failed {rel}", flush=True)
            sys.exit(1)
    hub = hub_sha(api)
    rows = {rel: {"local_sha256": local[rel], "hub_lfs_sha256": hub.get(rel), "equal": hub.get(rel) == local[rel]} for _, rel in items}
    bad = [r for r, v in rows.items() if not v["equal"]]
    with open(os.path.join(OUT, "BANK_V8F_RECOVER.json"), "w", encoding="utf-8") as f:
        json.dump({"repo": REPO, "repo_type": "dataset", "private": True, "src": SRC, "n_objects": len(rows),
                   "n_small_files_in_tar": len(small), "objects": rows, "not_equal": bad}, f, indent=1)
    if bad:
        print(f"ZZ BANKRECOVER FAIL ZZ {len(bad)} objects not equal on hub: {bad[:5]}", flush=True)
        sys.exit(1)
    print(f"ZZ BANKRECOVER DONE ZZ {len(rows)} objects equal on hub ({len(pts)} checkpoints + 1 tar of {len(small)} files)", flush=True)


if __name__ == "__main__":
    main()
