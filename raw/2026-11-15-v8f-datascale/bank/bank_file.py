"""MM 2026-11-19: bank ONE local file to a PRIVATE HF dataset and verify it by sha256.
Usage: bank_file.py <repo> <local file> <path_in_repo> <record json out>
Ends on ZZ BANKFILE DONE ZZ / ZZ BANKFILE FAIL ZZ <reason>. The token comes from the environment (HF_TOKEN), never from argv."""
import hashlib
import json
import os
import sys
import time

from huggingface_hub import HfApi


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hub_sha(api, repo, rel):
    for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True, expand=True):
        if e.path == rel:
            lfs = getattr(e, "lfs", None)
            if lfs is not None:
                return lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
    return None


def main():
    repo, src, rel, rec_p = sys.argv[1:5]
    api = HfApi()
    if not api.repo_info(repo, repo_type="dataset").private:
        print("ZZ BANKFILE FAIL ZZ repo is not private", flush=True)
        sys.exit(1)
    local = sha(src)
    print(f"{src} {os.path.getsize(src)} bytes sha256 {local}", flush=True)
    if hub_sha(api, repo, rel) != local:
        for k in range(4):
            try:
                api.upload_file(path_or_fileobj=src, path_in_repo=rel, repo_id=repo, repo_type="dataset", commit_message=f"bank {rel}")
                break
            except Exception as e:
                print(f"retry {k}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 * (k + 1))
        else:
            print("ZZ BANKFILE FAIL ZZ upload failed", flush=True)
            sys.exit(1)
    hub = hub_sha(api, repo, rel)
    with open(rec_p, "w", encoding="utf-8") as f:
        json.dump({"repo": repo, "repo_type": "dataset", "private": True, "local_path": src, "path_in_repo": rel,
                   "bytes": os.path.getsize(src), "local_sha256": local, "hub_lfs_sha256": hub, "equal": hub == local}, f, indent=1)
    if hub != local:
        print("ZZ BANKFILE FAIL ZZ hub sha256 not equal", flush=True)
        sys.exit(1)
    print("ZZ BANKFILE DONE ZZ", flush=True)


if __name__ == "__main__":
    main()
