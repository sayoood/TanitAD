"""Every Sayood/* repo: visibility, the Hub's usedStorage, and whether its FILE NAMES carry
raw clip ids. READ-ONLY: list and metadata calls only; nothing is created, changed or
deleted. Token read in place from the git-ignored Keys.txt; never printed, never argv.

A name "carries a raw id" if it contains a full UUID that is one of the 306,152 PhysicalAI
clip ids, or a hex-bounded 8-hex token that is the prefix of exactly one of them.
The universe is passed in as a path and is never banked (it is itself a list of raw ids).
Output carries counts only -- no id, no prefix.
"""
import json
import re
import sys
from pathlib import Path

import truststore
truststore.inject_into_ssl()
from huggingface_hub import HfApi  # noqa: E402

UNIVERSE, OUT = sys.argv[1], Path(sys.argv[2])
ids = json.load(open(UNIVERSE))["ids"]
IDS = set(ids)
PRE = {}
for i in ids:
    PRE[i[:8]] = PRE.get(i[:8], 0) + 1
U = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
H8 = re.compile(r"(?<![0-9a-f])([0-9a-f]{8})(?![0-9a-f-])")      # not the head of a full uuid

tok = re.search(r"hf_[A-Za-z0-9]+", Path("D:/Projects/TanitAD/Keys.txt")
                .read_text(encoding="utf-8", errors="replace")).group(0)
api = HfApi(token=tok)
user = api.whoami()["name"]
rows = []
for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets),
                     ("space", api.list_spaces)):
    for r in lister(author=user):
        info = api.repo_info(r.id, repo_type=kind, expand=["usedStorage", "private", "siblings"])
        names = [s.rfilename for s in (info.siblings or [])]
        full = sum(1 for n in names if any(u.lower() in IDS for u in U.findall(n)))
        pref = sum(1 for n in names if not any(u.lower() in IDS for u in U.findall(n))
                   and any(PRE.get(t) == 1 for t in H8.findall(n.lower())))
        rows.append({"repo": r.id, "type": kind, "private": info.private,
                     "used_gb": round((getattr(info, "used_storage", 0) or 0) / 1e9, 3),
                     "n_files": len(names), "names_with_full_clip_id": full,
                     "names_with_clip_prefix_only": pref})
rows.sort(key=lambda x: -x["used_gb"])
summary = {
    "repos": len(rows),
    "public": sum(1 for x in rows if x["private"] is False),
    "private": sum(1 for x in rows if x["private"] is True),
    "public_gb": round(sum(x["used_gb"] for x in rows if x["private"] is False), 3),
    "private_gb": round(sum(x["used_gb"] for x in rows if x["private"] is True), 3),
    "public_repos_with_raw_ids_in_names": [x["repo"] for x in rows if x["private"] is False
                                           and (x["names_with_full_clip_id"] or x["names_with_clip_prefix_only"])],
    "private_repos_with_raw_ids_in_names": [x["repo"] for x in rows if x["private"] is True
                                            and (x["names_with_full_clip_id"] or x["names_with_clip_prefix_only"])],
}
OUT.write_text(json.dumps({"summary": summary, "repos": rows}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
for x in rows:
    flag = "RAW-IDS" if (x["names_with_full_clip_id"] or x["names_with_clip_prefix_only"]) else ""
    print("  %-7s %-8s %-50s %9.3f GB %6d files  full %5d  prefix %4d  %s" % (
        "public" if x["private"] is False else "PRIVATE", x["type"], x["repo"][:50], x["used_gb"],
        x["n_files"], x["names_with_full_clip_id"], x["names_with_clip_prefix_only"], flag))
