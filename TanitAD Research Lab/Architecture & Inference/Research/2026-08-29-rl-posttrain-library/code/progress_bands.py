"""Is the progress clamp saturating? Split by v0 band — NOT pooled.

The pooled mean cannot answer this: the prediction was that saturation
DISAPPEARS on fast windows and APPEARS honestly on stopped ones. A single
average over both regimes hides exactly that split (TRAIN-C2's lesson).
"""
import sys, glob, os, importlib.util
import numpy as np, torch
sys.path.insert(0, "stack")
from tanitad.rl import rewards as R
spec = importlib.util.spec_from_file_location("a0", "stack/scripts/rl_a0_coverage.py")
a0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a0)

SP = sys.argv[1]
fan = a0.build_anchor_fan(64, 0)
eps = sorted(glob.glob(os.path.join(SP, "sp2/cache/physicalai-val130-heldout/*.v2ep.pt")))
agents = a0.read_agents(os.path.join(SP, "sp2/val130_agents.jsonl"),
                        {os.path.basename(p).split(".")[0] for p in eps})
rng = np.random.default_rng(0)
rows = []
n = 0
for path in eps:
    cid = os.path.basename(path).split(".")[0]
    if cid not in agents or n >= 240: continue
    d = torch.load(path, map_location="cpu", weights_only=False)
    poses = d["poses"].numpy(); T = poses.shape[0]
    if T < 22: continue
    cand = [t for t in sorted(agents[cid]) if 0 <= t < T - 21]
    if not cand: continue
    for t0 in sorted(int(t) for t in rng.choice(cand, size=min(6, len(cand)), replace=False)):
        v0 = float(poses[t0, 3]); n += 1
        new = R.COMPONENTS["progress"](fan, {"dt": 0.1, "v0": v0})
        old = R.COMPONENTS["progress"](fan, {"dt": 0.1, "progress_ref_m": 30.0})
        rows.append((v0,
                     float((new >= 1.5 - 1e-6).float().mean()),
                     float((old >= 1.5 - 1e-6).float().mean())))
rows = np.array(rows)
print(f"windows {len(rows)} | ego v0: median {np.median(rows[:,0]):.1f} m/s, "
      f"p10 {np.percentile(rows[:,0],10):.1f}, p90 {np.percentile(rows[:,0],90):.1f}")
print(f"{'v0 band':<14}{'n':>5}{'clamped NEW':>13}{'clamped OLD':>13}")
bands = [(0, 2, "stopped <2"), (2, 8, "slow 2-8"), (8, 15, "mid 8-15"),
         (15, 100, "fast >15")]
for lo, hi, name in bands:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    if not m.any(): continue
    print(f"{name:<14}{int(m.sum()):>5}{rows[m,1].mean():>12.1%}{rows[m,2].mean():>13.1%}")
