import json, random, hashlib
src = "/workspace/ops/reroll_probe/bank_in/targets_rank1.jsonl"
prod = "/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl"
lane = open(src, "rb").read().split(b"\n")
lane = [l for l in lane if l]
goal = []
with open(prod, "rb") as fh:
    for ln in fh:
        if b'"lane_rank"' in ln: continue
        goal.append(ln.rstrip(b"\n"))
        if len(goal) >= 150: break
rng = random.Random(20261017)
rows = list(lane)
for g in goal:
    rows.insert(rng.randint(0, len(rows)), g)
import os
os.makedirs("/workspace/ops/bank3c/controls/bank_i3", exist_ok=True)
open("/workspace/ops/bank3c/controls/bank_i3/targets_rank1.jsonl", "wb").write(b"\n".join(rows) + b"\n")
sub = [r for r in rows if b'"lane_rank"' in r]
print("lane", len(lane), "goal", len(goal), "total", len(rows), "lane subsequence == bank_in:", sub == lane)
print(hashlib.sha256(open(src,'rb').read()).hexdigest())
