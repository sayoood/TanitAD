"""List the cache shard tars of the private 10k dataset with their hub LFS sha256 and size (read-only)."""
import json, os, sys
from huggingface_hub import HfApi

REPO = "Sayood/tanitad-datascale-10k"
OUT = sys.argv[1]
api = HfApi()
rows = []
for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="cache", recursive=True, expand=True):
    p = getattr(f, "path", "")
    if not p.endswith(".tar"):
        continue
    lfs = getattr(f, "lfs", None)
    sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    size = (lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)) if lfs else getattr(f, "size", None)
    rows.append({"path": p, "sha256": sha, "size": int(size)})
rows.sort(key=lambda r: r["path"])
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"repo": REPO, "n": len(rows), "total_bytes": sum(r["size"] for r in rows), "shards": rows}, fh, indent=1)
print(len(rows), sum(r["size"] for r in rows), sum(1 for r in rows if not r["sha256"]))
for r in rows[:3] + rows[-2:]:
    print(r)
