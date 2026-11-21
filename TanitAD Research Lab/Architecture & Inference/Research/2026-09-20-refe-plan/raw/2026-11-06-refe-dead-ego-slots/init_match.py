"""Rebuild REFe at its training init (train.py:1174-1175: torch.manual_seed(seed); REFe(cfg)) and compare ego_enc.0.weight column by column with
the trained checkpoint.  A column that received gradient EXACTLY 0 for the whole run keeps its init bytes; 256 random floats cannot match by chance.
Controls: the columns of signals that vary in training (vx, ax, speed) must DIFFER from init; else the rebuild or the probe is wrong."""
import sys, torch
sys.path.insert(0, sys.argv[1])
import ckpt_io, model as M
ck = sys.argv[2]; seed = int(sys.argv[3])
o = torch.load(ck, map_location="cpu", weights_only=False)
base = M.REFeConfig.for_backbone(o["meta"]["backbone"]); base.n_cameras = int(o["meta"]["n_cameras"])
if hasattr(base, "per_sample_calib"): base.per_sample_calib = o["meta"].get("per_sample_calib") in (True, "True")
cfg = ckpt_io.config_for_checkpoint(base, ck)
print("cfg ego_dim", cfg.ego_dim, "nav_mode", getattr(cfg, "nav_mode", None), "n_cameras", cfg.n_cameras)
torch.manual_seed(seed)
m = M.REFe(cfg)
W0 = m.ego_enc[0].weight.detach().float(); W = o["model"]["ego_enc.0.weight"].float()
print("shapes", tuple(W0.shape), tuple(W.shape))
lab = ["vx", "vy", "ax", "ay", "yaw_rate", "steer", "speed"] + [f"goal{j}" for j in range(W.shape[1] - 7)]
for j in range(W.shape[1]):
    d = float((W[:, j] - W0[:, j]).abs().max())
    if j < 12 or d == 0.0:
        print(f"  col {j:2d} {lab[j]:9s} max|trained - init| {d:.3e}  {'IDENTICAL TO INIT' if d == 0.0 else ''}")
print("bias max|diff|", float((o['model']['ego_enc.0.bias'].float() - m.ego_enc[0].bias.detach().float()).abs().max()))
