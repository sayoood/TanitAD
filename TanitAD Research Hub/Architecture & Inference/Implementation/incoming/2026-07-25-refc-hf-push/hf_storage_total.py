"""Report total storage used by every Sayood/ model+dataset repo. Token from stdin."""
import sys

TOK = sys.stdin.readline().strip().lstrip("﻿")
from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=TOK)

grand = 0
rows = []
for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets)):
    try:
        repos = list(lister(author="Sayood"))
    except Exception as e:
        print(f"list {kind}s failed: {type(e).__name__} {str(e)[:150]}")
        continue
    for r in repos:
        total = 0
        nfiles = 0
        try:
            for f in api.list_repo_tree(r.id, repo_type=kind, recursive=True):
                sz = getattr(f, "size", None)
                if sz:
                    total += sz
                    nfiles += 1
        except Exception as e:
            rows.append((kind, r.id, None, f"TREE FAILED {type(e).__name__}"))
            continue
        grand += total
        rows.append((kind, r.id, total,
                     f"private={getattr(r,'private','?')} gated={getattr(r,'gated','?')} files={nfiles}"))

rows.sort(key=lambda x: -(x[2] or 0))
print(f"{'kind':8} {'repo':46} {'GB':>8}  meta")
for kind, rid, total, meta in rows:
    gb = f"{total/1e9:.3f}" if total is not None else "  n/a"
    print(f"{kind:8} {rid:46} {gb:>8}  {meta}")
print(f"\nGRAND TOTAL Sayood/: {grand/1e9:.3f} GB ({grand} bytes) across {len(rows)} repos")
print("STORAGE_DONE")
