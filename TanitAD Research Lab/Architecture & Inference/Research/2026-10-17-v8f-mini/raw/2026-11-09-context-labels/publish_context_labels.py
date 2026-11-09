"""Publish the per-frame context labels PRIVATE on the PI's HF account (binding: PI ruling 2026-11-08, SPEC_HWM section 6) and verify by sha256 after a re-download.
Parquet labels only, no frame or image is uploaded. The token is read in place from Keys.txt, held in a variable, never printed or written.

  publish_context_labels.py --release F:/strat_out/release --card SL/CONTEXT_LABELS_CARD.md [--dry-run]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
NAME = "tanitad-context-labels-v1"
KEYS = "E:/Projects/TanitAD/Keys.txt"
VERIFY = "F:/TanitAD-scratch/ctxrel/verify"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tokens():
    seen = []
    for m in re.findall(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8", errors="ignore").read()):
        if m not in seen:
            seen.append(m)
    return seen


def authenticated():
    from huggingface_hub import HfApi
    for t in tokens():
        try:
            who = HfApi(token=t).whoami()
        except Exception:
            continue
        if who.get("type") == "user" or who.get("name"):
            return HfApi(token=t), t, who["name"]
    raise SystemExit("no hf_ token in Keys.txt authenticates")


def listing(release):
    fs = sorted(glob.glob(release + "/frames/frames_*.parquet")) + [release + "/clips/clips.parquet"]
    return [(p, os.path.relpath(p, release).replace("\\", "/")) for p in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="F:/strat_out/release")
    ap.add_argument("--card", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = listing(a.release)
    assert files and all(os.path.getsize(p) > 0 for p, _ in files)
    bad = [r for _, r in files if not r.endswith(".parquet")]
    assert not bad, bad
    manifest = {"name": NAME, "files": {r: {"sha256": sha(p), "bytes": os.path.getsize(p)} for p, r in files}}
    manifest["files"]["README.md"] = {"sha256": sha(a.card), "bytes": os.path.getsize(a.card)}
    mpath = a.release + "/manifest.json"
    json.dump(manifest, open(mpath, "w"), indent=1, sort_keys=True)
    print(json.dumps({"n_files": len(files) + 2, "total_gb": round(sum(v["bytes"] for v in manifest["files"].values()) / 1e9, 3)}))
    if a.dry_run:
        return 0
    from huggingface_hub import hf_hub_download
    api, tok, user = authenticated()
    repo = f"{user}/{NAME}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(repo, repo_type="dataset").private is True, "the repo is not private"
    for p, r in files + [(a.card, "README.md"), (mpath, "manifest.json")]:
        api.upload_file(path_or_fileobj=p, path_in_repo=r, repo_id=repo, repo_type="dataset", commit_message=f"{NAME}: {r}")
    assert api.repo_info(repo, repo_type="dataset").private is True, "the repo is not private after the upload"
    head = api.repo_info(repo, repo_type="dataset").sha
    os.makedirs(VERIFY, exist_ok=True)
    errs = []
    for r, v in manifest["files"].items():
        q = hf_hub_download(repo, r, repo_type="dataset", revision=head, token=tok, local_dir=VERIFY)
        if sha(q) != v["sha256"]:
            errs.append(r)
    json.dump({"repo": repo, "revision": head, "private": True, "n_files_verified": len(manifest["files"]) - len(errs), "sha256_mismatch": errs},
              open(a.release + "/PUBLISH_VERIFY.json", "w"), indent=1)
    shutil.rmtree("F:/TanitAD-scratch/ctxrel/verify")
    print(json.dumps({"repo": repo, "revision": head, "verified": len(manifest["files"]) - len(errs), "mismatch": errs}))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
