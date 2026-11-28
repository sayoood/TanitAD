"""Download files from a private HF dataset and verify each against its hub LFS sha256 (or git blob size for non-LFS).
usage: fetch_verify.py <repo> <local_dir> <path_in_repo> [<path_in_repo> ...]"""
import hashlib, json, os, re, sys
from huggingface_hub import HfApi, hf_hub_download

repo, local_dir, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok)
info = {i.path: i for i in api.get_paths_info(repo, paths, repo_type="dataset")}
out = []
for p in paths:
    i = info.get(p)
    if i is None:
        raise SystemExit(f"not on hub: {p}")
    lfs = getattr(i, "lfs", None)
    want = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    f = hf_hub_download(repo, p, repo_type="dataset", local_dir=local_dir, token=tok)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    got = h.hexdigest()
    ok = (got == want) if want else (os.path.getsize(f) == getattr(i, "size", -1))
    out.append({"path": p, "local": f, "bytes": os.path.getsize(f), "sha256": got, "hub_lfs_sha256": want, "ok": ok})
    print(("OK " if ok else "MISMATCH ") + p, os.path.getsize(f), got[:16], flush=True)
with open(os.path.join(local_dir, "FETCH_VERIFY.json"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"repo": repo, "files": out}) + "\n")
if not all(r["ok"] for r in out):
    sys.exit(1)
