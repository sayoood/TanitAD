"""ADDENDUM 2 follow-up — is the extra REACH at a larger denoise budget USABLE?

Addendum 2 measured that the fan's reachable set scales with `decoder_steps`
(1.68 -> 3.19 m mean; 1.79 -> 6.08 m at the 2 s endpoint, crossing the 3.67 m gap
a near-miss candidate must close). That reopens "structurally dead" as "dead at
this inference budget" -- but ONLY if the extra reach is not bought by wrecking
the fan.

⛔ THE TRAP THIS AVOIDS: reporting "reach doubles at steps=8" as a lever without
checking what steps=8 does to the QUALITY of the fan. A decoder can move further
by moving worse. So this measures, on the SAME windows, at every budget:

  R1  composed reward (DEFAULT spec, so it stays comparable to every banked arm)
  R2  fan collision rate
  R3  selection ADE  -- the metric the drift was measured in

ZERO training. Cold model only. If R3 holds while reach grows, the extra reach is
free except for latency, and the RL arms should be re-run at the larger budget.
If R3 degrades, the reach is an illusion and the structural reading stands.

Tier: T0 instrument measurement. NON-PARITY corpus. Evidence class: MEASURED.
"""
import importlib.util, sys, json
import numpy as np, torch

sys.path.insert(0, "stack")
spec_ = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(P)
from tanitad.rl import PostTrainConfig, RewardSpec

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
dev = "cuda"

model = P.load_model(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt", dev)
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

rows = {}
for st in (2, 4, 8, 16):
    cfg = PostTrainConfig(method="grpo", group_size=4, steps=1, batch=1, lr=1e-5,
                          seed=0, dt=P.DT_TRAJ, decoder_steps=st)
    r = P.readout(model, src, RewardSpec(dt=P.DT_TRAJ), cfg, dev)
    rows[st] = r
    print(f"steps={st:>2}: R1 {r['R1_fan_reward']['mean']:+.4f}"
          f" · R2 {r['R2_fan_collision_rate']['mean']*100:6.3f}%"
          f" · R3 {r['R3_sel_ade_m']['mean']:.4f} m")

base = rows[2]
print("\n=== vs the deployed steps=2, PAIRED over the same 120 windows ===")
for st in (4, 8, 16):
    r = rows[st]
    # paired per-episode delta on the metric the drift was measured in
    d = [r["per_episode"][k]["ade"] - base["per_episode"][k]["ade"]
         for k in base["per_episode"] if k in r["per_episode"]]
    d = np.array(d)
    bs = np.array([np.mean(rng_.choice(d, d.size, replace=True))
                   for rng_ in [np.random.default_rng(s) for s in range(4000)]])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sep = "SEPARATED" if (lo > 0) == (hi > 0) else "not separated"
    print(f"  steps={st:>2}: dR3 {d.mean():+.4f} m  [{lo:+.4f}, {hi:+.4f}]  {sep}"
          f"   dR1 {r['R1_fan_reward']['mean']-base['R1_fan_reward']['mean']:+.4f}"
          f"   dR2 {(r['R2_fan_collision_rate']['mean']-base['R2_fan_collision_rate']['mean'])*100:+.3f} pp")

json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "per_episode"}
           for k, v in rows.items()}, open(f"{O}/denoise_budget.json", "w"), indent=1)
print(f"\n-> {O}/denoise_budget.json")
