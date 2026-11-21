"""Private/public storage used by every repo of the account (Hub usedStorage per repo = LFS/xet bytes incl. history)."""
import re, time
import truststore; truststore.inject_into_ssl()
from huggingface_hub import HfApi
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok); who = api.whoami(); me = who["name"]
print("account", me, "| isPro", who.get("isPro"))
tot = {True: 0, False: 0}; rows = []
for kind, lister, info in (("model", api.list_models, api.model_info), ("dataset", api.list_datasets, api.dataset_info), ("space", api.list_spaces, api.space_info)):
    for r in lister(author=me):
        try:
            i = info(r.id, expand=["usedStorage", "private"]); u = getattr(i, "used_storage", None) or 0; priv = bool(i.private)
        except Exception as e:
            print("  ?", kind, r.id, type(e).__name__); continue
        tot[priv] += u; rows.append((u, kind, r.id, priv))
for u, kind, rid, priv in sorted(rows, reverse=True)[:30]:
    print(f"{u / 1e9:9.2f} GB  {kind:8s} private={priv!s:5s} {rid}")
print(f"repos {len(rows)} | PRIVATE used {tot[True] / 1e9:.2f} GB | public used {tot[False] / 1e9:.2f} GB")
