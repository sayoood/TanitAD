"""Regression pin: the new bench.py must leave the legacy `heldout` block
numerically UNCHANGED for an already-published arm. If this fails, every
cross-arm comparison the gate makes is invalid."""
import json, sys
sys.path.insert(0, "/root/taniteval")
from taniteval import bench, rollout

for key in ("flagship-30k", "flagship-v2-6k"):
    win = rollout.load_windows(f"/root/taniteval/results/windows_{key}.pt")
    stored = json.loads(open(f"/root/taniteval/results/{key}.json").read())
    fresh = bench.run(win)
    ok = True
    for m in ("ade_0_2s", "fde@2s", "miss_rate@2m"):
        s = stored["heldout"]["model"][m]["mean"]
        f = fresh["heldout"]["model"][m]["mean"]
        d = abs(s - f)
        flag = "OK " if d < 1e-9 else "DIFF"
        if d >= 1e-9: ok = False
        print(f"[pin] {key:16s} {m:14s} stored={s:.6f} fresh={f:.6f} {flag}")
    cb = fresh.get("cluster_bootstrap", {}).get("model", {}).get("ade_0_2s")
    print(f"[pin] {key:16s} primary_ci={fresh.get('primary_ci')} "
          f"cluster_bootstrap ade_0_2s={cb}")
    print(f"[pin] {key:16s} legacy-block reproduced: {ok}\n")
