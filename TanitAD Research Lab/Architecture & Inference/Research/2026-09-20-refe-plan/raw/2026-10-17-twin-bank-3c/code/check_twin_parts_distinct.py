#!/usr/bin/env python3
"""D7 acceptance from ARTIFACTS with TWO roots: a rank-K twin's route digest (centreline + route lane ids, `p6_prep.route_digest`) must DIFFER from the logged route's
for the same token (the pre-fix files were byte-identical: differ=0).  `p6_prep.py lane-distinct` assumes one shared root; the 3-C twin parts live in their own root.
    python check_twin_parts_distinct.py --bank <dir> --twin-root R --logged-root L --k K [--n 40] [--min-frac 0.9]     (pod, teacher venv, PYTHONPATH incl. $PKG/refe)
Prints ZZD7_DISTINCT_OK / ZZD7_DISTINCT_BAD like the original."""
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True); ap.add_argument("--twin-root", required=True); ap.add_argument("--logged-root", required=True)
ap.add_argument("--k", type=int, required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--min-frac", type=float, default=0.9)
a = ap.parse_args()
import online_labeller as OL
import p6_prep as PP
seen, sampled, differ, rows = set(), 0, 0, 0
rr = [json.loads(l) for l in open(Path(a.bank) / "targets_rank1.jsonl", encoding="utf-8") if '"lane_rank"' in l]
rr = [r for r in rr if isinstance(r.get("aug"), dict) and r["aug"].get("kind") == "lane_rank" and int(r["aug"]["value"]) == a.k]
step = max(len(rr) // max(a.n * 3, 1), 1)                        # an even stride over the bank, not its first n (the first logs are not the population)
for r in rr[::step]:
    key = (r["log_name"], r["token"])
    if key in seen: continue
    tw, lg = OL.parts_path(a.twin_root, key[0], key[1], r["aug"]), OL.parts_path(a.logged_root, key[0], key[1], None)
    if not (tw.is_file() and lg.is_file()): continue
    seen.add(key); sampled += 1
    differ += int(PP.route_digest(OL.load_parts(tw)) != PP.route_digest(OL.load_parts(lg)))
    if sampled >= a.n: break
frac = differ / sampled if sampled else 0.0
ok = sampled > 0 and frac >= a.min_frac
print(f"{'ZZD7_DISTINCT_OK' if ok else 'ZZD7_DISTINCT_BAD'} k={a.k} sampled={sampled} differ={differ} frac={round(frac, 4)} min_frac={a.min_frac}")
sys.exit(0 if ok else 1)
