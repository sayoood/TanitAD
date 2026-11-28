"""REHEARSAL helper (runs inside WSL against the mock pod): print the path of the first KEPT twin's part under a cache root, computed with P6's own
online_labeller.parts_path (the consumer's path function).  usage: python3 kept_part.py <bank_dir> <cache_root> <p6_refe_dir>"""
import json
import sys

bank, root, p6refe = sys.argv[1:4]
sys.path.insert(0, p6refe)
import online_labeller as OL  # noqa: E402

with open(bank + "/targets_rank1.jsonl", encoding="utf-8") as fh:
    for ln in fh:
        if '"lane_rank"' not in ln:
            continue
        r = json.loads(ln)
        a = r.get("aug")
        if isinstance(a, dict) and a.get("kind") == "lane_rank":
            print(OL.parts_path(root, r["log_name"], r["token"], a))
            break
