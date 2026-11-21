"""Folder-level summary of the corpus repo: files and bytes per folder (2 levels), plus the top-level files."""
import collections, re
import truststore; truststore.inject_into_ssl()
from huggingface_hub import HfApi
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok); repo = "Sayood/tanitad-v7-training-corpus"
n = collections.Counter(); b = collections.Counter(); ex = {}
for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True):
    if type(f).__name__ != "RepoFile":
        continue
    parts = f.path.split("/"); k = "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else "<top>")
    n[k] += 1; b[k] += f.size or 0; ex.setdefault(k, f.path)
for k in sorted(n):
    print(f"{k:45s} files {n[k]:6d}  {b[k] / 1e9:8.3f} GB  e.g. {ex[k]}")
print("total files", sum(n.values()), "bytes GB", round(sum(b.values()) / 1e9, 2))
