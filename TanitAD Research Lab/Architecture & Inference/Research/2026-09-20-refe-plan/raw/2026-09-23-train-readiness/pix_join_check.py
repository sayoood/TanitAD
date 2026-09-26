"""Producer->consumer seam: do our navtrain target rows' camera paths exist in OpenScene's pixels?

Producer: build_teacher_rollouts.py writes row["image"] = 4 x "<log>/<CAM>/<hash>.jpg", chosen by
camera_arrays(db) from the nuPlan DB `image` table. Consumer: the pod's `pix` stage keeps only what
OpenScene's navtrain_current_{1..32}.tgz ships. If our frame->image association differs from
OpenScene's, the file is simply NOT in the tarball and the trainer cannot load that row.

Reference: members.txt = the full member list of navtrain_current_25.tgz (downloaded 2026-09-22).
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

MEM = "D:/Projects/TanitAD/data/openscene_probe/members.txt"
BANKS = sorted(glob.glob("D:/Projects/TanitAD/data/refe_navtrain/r0_shard*/targets_rank0.jsonl"))
YAML = sys.argv[1] if len(sys.argv) > 1 else None

members = set()
shard_logs = set()
per_log_cam = defaultdict(Counter)
with open(MEM, encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        parts = ln.split("/")
        if len(parts) == 4 and parts[3].endswith(".jpg"):
            members.add("/".join(parts[1:]))          # "<log>/<CAM>/<hash>.jpg"
            shard_logs.add(parts[1])
            per_log_cam[parts[1]][parts[2]] += 1
print(f"shard 25: {len(shard_logs)} logs, {len(members):,} jpg members")

rows_total = 0
rows_in_shard = 0
hit = Counter()
miss_examples = []
rows_all4 = 0
tok_in_shard = set()
bank_logs = set()
for b in BANKS:
    with open(b, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue                                 # a live file's last line may be partial
            rows_total += 1
            bank_logs.add(r["log_name"])
            if r["log_name"] not in shard_logs:
                continue
            rows_in_shard += 1
            tok_in_shard.add((r["log_name"], r["token"]))
            ok = 0
            for p in r["image"]:
                cam = p.split("/")[1]
                if p in members:
                    hit[cam] += 1
                    ok += 1
                elif len(miss_examples) < 6:
                    miss_examples.append(p)
            rows_all4 += ok == len(r["image"])

overlap = bank_logs & shard_logs
print(f"local bank: {rows_total:,} rows over {len(bank_logs)} logs; logs shared with shard 25: {len(overlap)}")
print(f"rows whose log is in shard 25: {rows_in_shard:,}  (distinct tokens {len(tok_in_shard):,})")
for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
    print(f"  {cam}: {hit[cam]:,}/{rows_in_shard:,} paths found in the tarball")
print(f"rows with ALL 4 cameras present: {rows_all4:,}/{rows_in_shard:,}")
if miss_examples:
    print("missing examples:", miss_examples)
if rows_in_shard == 0:
    print("ZZJOIN_NO_OVERLAPZZ")
elif rows_all4 == rows_in_shard:
    print("ZZJOIN_COMPLETEZZ")
else:
    print(f"ZZJOIN_PARTIAL_{rows_all4}_OF_{rows_in_shard}ZZ")

# frames the shard ships per log vs tokens navtrain lists for that log (needs the split yaml)
if YAML:
    import yaml
    y = yaml.safe_load(open(YAML, encoding="utf-8"))
    toks = y.get("tokens") or y.get("scene_filter", {}).get("tokens") or []
    logs = y.get("log_names") or []
    print(f"split yaml: {len(toks):,} tokens, {len(logs):,} logs")
