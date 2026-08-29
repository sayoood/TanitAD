"""Quantify TRAIN-C5: the P-RC21 readout ran with model.training=True.

ego_dropout 0.5 + route_dropout 0.5 + stochastic diffusion were ACTIVE in every
readout. This measures the same cold-start checkpoint both ways, 3 seeds each,
so the defect is stated as a number rather than a worry.
"""
import importlib.util, sys, json
import numpy as np, torch
sys.path.insert(0, "stack")
spec = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
from tanitad.rl import RewardSpec, PostTrainConfig

dev = "cuda"
O = r"C:/Users/Admin/tanitad-data/rl-pilot"
m = P.load_model(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt", dev)
cfg = PostTrainConfig(decoder_steps=2, dt=P.DT_TRAJ)
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)
spec_r = RewardSpec(dt=P.DT_TRAJ)

out = {}
for mode in ("train", "eval"):
    getattr(m, mode)()
    runs = []
    for rep in range(3):
        torch.manual_seed(1000 + rep)
        r = P.readout(m, src, spec_r, cfg, dev)
        runs.append((r["R1_fan_reward"]["mean"], r["R2_fan_collision_rate"]["mean"],
                     r["R3_sel_ade_m"]["mean"]))
    a = np.array(runs)
    out[mode] = {"R1": a[:,0].tolist(), "R2": a[:,1].tolist(), "R3": a[:,2].tolist()}
    print(f"{mode:6s} R1 {a[:,0].mean():+.4f} (spread {np.ptp(a[:,0]):.4f}) | "
          f"R2 {100*a[:,1].mean():.3f}% (spread {100*np.ptp(a[:,1]):.3f}pp) | "
          f"R3 {a[:,2].mean():.3f} m (spread {np.ptp(a[:,2]):.3f} m)", flush=True)
json.dump(out, open(f"{O}/eval_mode_impact.json","w"), indent=1)
tr, ev = np.array(out["train"]["R3"]), np.array(out["eval"]["R3"])
print(f"\nR3 train-mode mean {tr.mean():.3f} m vs eval-mode {ev.mean():.3f} m "
      f"=> the readout was reporting {100*(tr.mean()/ev.mean()-1):+.1f}% vs deployed")
print(f"train-mode run-to-run R3 spread {np.ptp(tr):.3f} m; eval-mode {np.ptp(ev):.3f} m")
