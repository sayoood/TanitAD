"""Residual of each ego column from the best pure shrink c*W0 (c = median of W/W0): a zero-gradient column leaves only float rounding (~1e-6 abs);
any Adam step with a non-zero gradient moves an entry by ~lr regardless of the gradient's size (Adam normalises), so a trained column cannot hide."""
import sys, torch
sys.path.insert(0, sys.argv[1]); import ckpt_io, model as M
ck = sys.argv[2]; o = torch.load(ck, map_location="cpu", weights_only=False)
base = M.REFeConfig.for_backbone(o["meta"]["backbone"]); base.n_cameras = int(o["meta"]["n_cameras"]); cfg = ckpt_io.config_for_checkpoint(base, ck)
torch.manual_seed(0); m = M.REFe(cfg)
W0 = m.ego_enc[0].weight.detach().double(); W = o["model"]["ego_enc.0.weight"].double()
print("ckpt dtype", o["model"]["ego_enc.0.weight"].dtype)
for j, n in enumerate(["vx", "vy", "ax", "ay", "yaw_rate", "steer", "speed", "goal0"]):
    c = float((W[:, j] / W0[:, j]).median()); r = (W[:, j] - c * W0[:, j]).abs()
    print(f"{n:9s} c={c:.6f}  max|W - c*W0| {float(r.max()):.3e}  mean {float(r.mean()):.3e}")
