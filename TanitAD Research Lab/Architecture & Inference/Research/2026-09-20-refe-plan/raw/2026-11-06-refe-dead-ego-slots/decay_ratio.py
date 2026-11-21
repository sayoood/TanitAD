"""Is the trained-minus-init change of a column a PURE uniform shrink (weight decay with zero gradient)?  ratio = W/W0 per entry: one constant => no gradient."""
import sys, torch
sys.path.insert(0, sys.argv[1]); import ckpt_io, model as M
ck = sys.argv[2]; o = torch.load(ck, map_location="cpu", weights_only=False)
base = M.REFeConfig.for_backbone(o["meta"]["backbone"]); base.n_cameras = int(o["meta"]["n_cameras"]); cfg = ckpt_io.config_for_checkpoint(base, ck)
torch.manual_seed(0); m = M.REFe(cfg)
W0 = m.ego_enc[0].weight.detach().double(); W = o["model"]["ego_enc.0.weight"].double()
for j, n in enumerate(["vx", "vy", "ax", "ay", "yaw_rate", "steer", "speed", "goal0"]):
    r = W[:, j] / W0[:, j]
    print(f"{n:9s} ratio W/W0: min {r.min():.6f} max {r.max():.6f} spread {float(r.max() - r.min()):.2e}")
