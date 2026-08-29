"""A/B the OLD saturating headway vs the NEW graded one, same windows.

Honesty check: A0's first FAIL was partly a METRIC bug (median over windows
where the term is undefined). Did the redesign actually change anything, or
would the old shape have passed once the metric was fixed?
"""
import sys, glob, json, os
import numpy as np, torch
sys.path.insert(0, "stack")
from tanitad.rl import rewards as R
sys.path.insert(0, os.path.dirname(os.path.abspath("stack/scripts/rl_a0_coverage.py")))
import importlib.util
spec = importlib.util.spec_from_file_location("a0", "stack/scripts/rl_a0_coverage.py")
a0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a0)

def old_headway(traj, ctx):
    """The pre-2026-08-29 shape: min_t (gap/v)/T*, clamped [0,1]."""
    lead = ctx.get("lead_path")
    if lead is None:
        return torch.ones(traj.shape[:-2])
    t_star = float(ctx.get("target_time_gap_s", 2.0))
    kin = R.kinematics(traj, ctx.get("dt", 0.1))
    gap = (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1)
    gap = (gap - float(ctx.get("lead_len_m", 4.5))).clamp_min(0.0)
    tg = gap / kin.speed.clamp_min(0.5)
    return (tg.min(dim=-1).values / t_star).clamp(0.0, 1.0)

SP = sys.argv[1]
fan = a0.build_anchor_fan(64, 0)
eps = sorted(glob.glob(os.path.join(SP, "sp2/cache/physicalai-val130-heldout/*.v2ep.pt")))
agents = a0.read_agents(os.path.join(SP, "sp2/val130_agents.jsonl"),
                        {os.path.basename(p).split(".")[0] for p in eps})
rng = np.random.default_rng(0)
old_sp, new_sp = [], []
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
        x0, y0, yaw0 = poses[t0, 0], poses[t0, 1], poses[t0, 2]
        obs = []
        for wx, wy, _ in a0.agents_to_world(agents[cid].get(t0, []), x0, y0, yaw0):
            ex, ey = a0.ego_frame(np.array(wx), np.array(wy), x0, y0, yaw0)
            if 0.0 < float(ex) < 80.0: obs.append((float(ex), float(ey)))
        inl = [(ex, ey) for ex, ey in obs if abs(ey) <= 2.0]
        n += 1
        if not inl: continue
        lx, ly = min(inl, key=lambda p: p[0])
        ctx = {"dt": 0.1, "lead_path": torch.tensor([[lx, ly]] * 20, dtype=torch.float32)}
        o = old_headway(fan, ctx); nw = R.COMPONENTS["headway"](fan, ctx)
        old_sp.append(float(o.max() - o.min())); new_sp.append(float(nw.max() - nw.min()))

old_sp, new_sp = np.array(old_sp), np.array(new_sp)
print(f"lead-present windows: {len(old_sp)}")
for nm, v in (("OLD saturating", old_sp), ("NEW graded", new_sp)):
    live = v[v > 1e-9]
    print(f"  {nm:<16} ranks on {len(live)}/{len(v)} = {len(live)/max(len(v),1):.1%} "
          f"| median spread when ranking {np.median(live) if live.size else 0:.4f} "
          f"| mean spread {v.mean():.4f}")
