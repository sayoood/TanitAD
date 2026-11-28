"""Bank F:/strat_out/band_labels (the v10 band labels and the strategic geometry and token files) PRIVATE on the PI's HF account, verified by sha256 after a re-download. Arrays and small logs only, no frame or image.

  publish_band_labels.py [--dry-run]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import publish_context_labels as PC

ROOT = "F:/strat_out/band_labels"
NAME = "tanitad-band-labels-v10"
VERIFY = "F:/TanitAD-scratch/ctxrel/verify_band"
BAD_EXT = (".png", ".jpg", ".jpeg", ".mp4", ".webp")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = {}
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            p = os.path.join(d, f)
            r = os.path.relpath(p, ROOT).replace(os.sep, "/")
            assert not r.lower().endswith(BAD_EXT), r
            if os.path.getsize(p) > 0:
                files[r] = p
    manifest = {r: {"sha256": sha(p), "bytes": os.path.getsize(p)} for r, p in sorted(files.items())}
    print(json.dumps({"n_files": len(manifest), "total_mb": round(sum(v["bytes"] for v in manifest.values()) / 1e6, 1)}))
    if a.dry_run:
        return 0
    from huggingface_hub import hf_hub_download
    api, tok, user = PC.authenticated()
    repo = f"{user}/{NAME}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(repo, repo_type="dataset").private is True, "the repo is not private"
    api.upload_folder(folder_path=ROOT, repo_id=repo, repo_type="dataset", allow_patterns=sorted(manifest), commit_message=f"{NAME}: band labels")
    info = api.repo_info(repo, repo_type="dataset")
    assert info.private is True, "the repo is not private after the upload"
    os.makedirs(VERIFY, exist_ok=True)
    errs = []
    for r, v in manifest.items():
        if sha(hf_hub_download(repo, r, repo_type="dataset", revision=info.sha, token=tok, local_dir=VERIFY)) != v["sha256"]:
            errs.append(r)
    shutil.rmtree(VERIFY)
    print(json.dumps({"repo": repo, "revision": info.sha, "verified": len(manifest) - len(errs), "mismatch": errs}))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
