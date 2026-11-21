"""Recover each level's BUILT IDM weight from a v8f_mini_train train_log.jsonl by the loss identity
level/L = pred + rollout_w*rollout + beta*sigreg + gamma*idm (+ act_sigreg*asg at L>=2)  =>  gamma_hat = residual / idm.
Usage: python gamma_from_log.py <train_log.jsonl> [beta=0.08] [rollout_w=1.0] [act_sigreg=0.005]"""
import json, sys
p = sys.argv[1]; beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
rw = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; asw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
g = {1: [], 2: [], 3: []}
for ln in open(p, encoding="utf-8"):
    r = json.loads(ln)
    if "wm/level/L1" not in r:
        continue
    for L in (1, 2, 3):
        res = r[f"wm/level/L{L}"] - r[f"wm/pred/L{L}"] - rw * r.get(f"wm/rollout/L{L}", 0.0) - beta * r[f"wm/sigreg/L{L}"] - (asw * r[f"wm/act_sigreg/L{L}"] if L >= 2 else 0.0)
        g[L].append(res / r[f"wm/idm/L{L}"])
out = {f"L{L}": {"n": len(v), "min": min(v), "max": max(v)} for L, v in g.items()}
print(json.dumps(out))
