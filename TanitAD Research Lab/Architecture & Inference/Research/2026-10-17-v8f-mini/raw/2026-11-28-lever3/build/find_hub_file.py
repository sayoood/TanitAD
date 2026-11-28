"""Search our private HF datasets for files whose name contains a pattern (read-only)."""
import re, sys
from huggingface_hub import HfApi

pat = sys.argv[1]
tok = re.search(r"hf_[A-Za-z0-9]+", open("E:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok)
repos = [d.id for d in api.list_datasets(author="Sayood")]
print(len(repos), "datasets")
for r in repos:
    try:
        for f in api.list_repo_tree(r, repo_type="dataset", recursive=True, expand=True):
            p = getattr(f, "path", "")
            if pat in p:
                lfs = getattr(f, "lfs", None)
                sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                print(r, p, getattr(f, "size", None), sha)
    except Exception as e:  # noqa: BLE001
        print("ERR", r, repr(e)[:120])
