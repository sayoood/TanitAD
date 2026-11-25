"""bank_leadspeed_ckpt.py -- banks the card ADD149 head checkpoints (pulled from box 2, sha256 checked against FULL.json) to the private HF dataset of this fork.
Runs on the dev box. The token is read in this process from Keys.txt and never printed. Verifies by the hub's own LFS sha256. Deletes and moves nothing.

  python bank_leadspeed_ckpt.py <local folder> <path in repo> <file> [<file> ...]
"""
import hashlib
import json
import os
import re
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi  # noqa: E402

REPO_NAME = "tanitad-v8proof-fork-bank"
KEYS = "E:/Projects/TanitAD/Keys.txt"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    src, dst, files = sys.argv[1], sys.argv[2].strip("/"), sys.argv[3:]
    tok = re.findall(r"hf_[A-Za-z0-9]+", open(KEYS, encoding="utf-8", errors="ignore").read())[0]
    api = HfApi(token=tok)
    repo = "%s/%s" % (api.whoami()["name"], REPO_NAME)
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    if not info.private:
        raise SystemExit("repo is not private; stop")
    local = {f: sha(os.path.join(src, f)) for f in files}
    for f in files:
        api.upload_file(path_or_fileobj=os.path.join(src, f), path_in_repo=dst + "/" + f, repo_id=repo, repo_type="dataset", commit_message="bank %s/%s" % (dst, f))
    hub = {}
    for p in api.get_paths_info(repo_id=repo, paths=[dst + "/" + f for f in files], repo_type="dataset"):
        lfs = getattr(p, "lfs", None)
        hub[p.path.split("/")[-1]] = (lfs.sha256 if lfs else None, getattr(p, "blob_id", None))
    out = {"repo_name": REPO_NAME, "private": True, "path_in_repo": dst,
           "files": {f: {"sha256_local": local[f], "sha256_hub_lfs": hub.get(f, (None, None))[0], "git_blob_sha1": hub.get(f, (None, None))[1],
                         "equal": hub.get(f, (None, None))[0] == local[f]} for f in files}}
    out["all_equal"] = all(v["equal"] for v in out["files"].values())
    json.dump(out, open(os.path.join(src, "BANK_VERIFY.json"), "w"), indent=1)
    print(json.dumps({"all_equal": out["all_equal"], "n": len(files), "path_in_repo": dst}))


if __name__ == "__main__":
    main()
