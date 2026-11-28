"""MM 2026-11-19: bank F:/TanitAD-scratch/datascale/read3/t1 (the DS read3 T1 files of BL_s0 and UE_s0, with dumps and logs; the
UE_s0 file is the one the coordinator binds k5 on) to the PRIVATE HF dataset Sayood/tanitad-v8f-dscale10k-runs as one tar,
then verify by sha256. Writes BANK_READ3_T1.json and ends on ZZ BANKREAD3T1 DONE ZZ / ZZ BANKREAD3T1 FAIL ZZ <reason>."""
import hashlib
import json
import os
import sys
import tarfile
import time

from huggingface_hub import HfApi

SRC = "F:/TanitAD-scratch/datascale/read3/t1"
OUT = "F:/TanitAD-scratch/datascale/bank_read3t1"
REPO = "Sayood/tanitad-v8f-dscale10k-runs"
REL = "read3_t1/read3_t1.tar"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hub_sha(api):
    for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
        if e.path == REL:
            lfs = getattr(e, "lfs", None)
            if lfs is not None:
                return lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
    return None


def main():
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        print("ZZ BANKREAD3T1 FAIL ZZ repo is not private", flush=True)
        sys.exit(1)
    files = []
    for root, _, fs in os.walk(SRC):
        for f in fs:
            files.append(os.path.join(root, f).replace("\\", "/"))
    files.sort()
    per_file = {os.path.relpath(p, SRC).replace("\\", "/"): sha(p) for p in files}
    tar_p = os.path.join(OUT, "read3_t1.tar")
    with tarfile.open(tar_p, "w") as t:
        for p in files:
            t.add(p, arcname="t1/" + os.path.relpath(p, SRC).replace("\\", "/"))
    local = sha(tar_p)
    print(f"tar {os.path.getsize(tar_p)} bytes, {len(files)} files, sha256 {local}", flush=True)
    if hub_sha(api) != local:
        for k in range(4):
            try:
                api.upload_file(path_or_fileobj=tar_p, path_in_repo=REL, repo_id=REPO, repo_type="dataset",
                                commit_message=f"bank {REL} (DS read3 T1 files, BL_s0 + UE_s0)")
                break
            except Exception as e:
                print(f"retry {k}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 * (k + 1))
        else:
            print("ZZ BANKREAD3T1 FAIL ZZ upload failed", flush=True)
            sys.exit(1)
    hub = hub_sha(api)
    rec = {"repo": REPO, "repo_type": "dataset", "private": True, "src": SRC, "path_in_repo": REL,
           "tar_local_sha256": local, "tar_hub_lfs_sha256": hub, "equal": hub == local,
           "n_files": len(files), "files_sha256": per_file}
    with open(os.path.join(OUT, "BANK_READ3_T1.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    if hub != local:
        print("ZZ BANKREAD3T1 FAIL ZZ hub sha256 not equal", flush=True)
        sys.exit(1)
    os.remove(tar_p)
    print(f"deleted local tar {tar_p} (banked, hub equal)", flush=True)
    print("ZZ BANKREAD3T1 DONE ZZ", flush=True)


if __name__ == "__main__":
    main()
