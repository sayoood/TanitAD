"""pick_metrics.py <thor_reads dir> <step> -- print the newest metrics copy carrying EXACTLY ONE eval row at <step>
(empty output = none). Used by chain_g0_r8.sh; asserts on content, not on file names."""
import glob
import json
import sys

d, step = sys.argv[1], int(sys.argv[2])
for p in sorted(glob.glob(d.rstrip("/") + "/metrics_*.jsonl"), reverse=True):
    n = 0
    for ln in open(p, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("step") == step and "eval_loss" in r:
            n += 1
    if n == 1:
        print(p.replace("\\", "/"))
        break
