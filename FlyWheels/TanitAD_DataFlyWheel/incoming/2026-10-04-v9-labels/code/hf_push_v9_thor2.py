"""Push the refcv8 v9 label release to the PRIVATE HF dataset repo from Thor (Thor's own HF login; no token handled here).
v2: exits non-zero unless every local file is verified remotely (size, and sha256 for LFS/xet files)."""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi
folder = Path("/home/nvidia/refcv8_v9labels/hfup")
api = HfApi()
repo = "Sayood/tanitad-refcv8-v9-labels"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
assert api.dataset_info(repo).private is True, "repo is not private -- refusing"
api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset",
                  commit_message="refcv8 v9 label release (per-frame tactical actions/goals with constraints over [NOW+2, NOW+8] s, per-frame nav, route checkpoint) + corpus manifest")
info = api.dataset_info(repo, files_metadata=True)
remote = {s.rfilename: (s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings}
local = sorted(p for p in folder.rglob("*") if p.is_file())
ok, bad, n_sha = 0, [], 0
for f in local:
    rel = f.relative_to(folder).as_posix(); size, sha = remote.get(rel, (None, None))
    if size != f.stat().st_size: bad.append((rel, "size", size)); continue
    if sha:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != sha: bad.append((rel, "sha256")); continue
        n_sha += 1
    ok += 1
res = {"repo": repo, "private": info.private, "n_local": len(local), "files_ok": ok, "sha256_checked": n_sha,
       "bad": bad, "n_remote": len(remote), "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET", "0"), "sha": info.sha}
print("ZZHFV9", json.dumps(res), flush=True)
sys.exit(0 if (not bad and ok == len(local) and info.private is True) else 3)
