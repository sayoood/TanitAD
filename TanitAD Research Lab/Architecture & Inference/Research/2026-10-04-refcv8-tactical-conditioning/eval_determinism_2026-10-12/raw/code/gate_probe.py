"""The launch gate's OWN control (`launch_gate._forward_out`: compute_losses_v3 under fork_rng + manual_seed(1234),
stopped at the model output by a hook), twice, on the rig model with the provisional r8 recipe. CPU."""
import importlib, json, sys, torch
import repro_lib as L
G = importlib.import_module("launch_gate")
T = L.T
emit_live = sys.argv[1] == "1"
kind = sys.argv[2] if len(sys.argv) > 2 else "r8"
cfg, m = L.build(kind)
if kind == "r8":
    m.core.decoder.r8_emit_live = emit_live
m.eval()
eps = T._synth_episodes(2, cfg.core, seed=0)
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
batch = torch.utils.data.default_collate([ds[0], ds[1]])
batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
batch["nav_valid"] = torch.tensor([True, True])
if "pose_hist" not in batch:
    ph = L.R.batch(cfg, b=2)["pose_hist"].clone()
    v0b = batch["pose_last"][:, 3].reshape(-1).to(torch.float32)
    ph[:, :, 3] = v0b[:, None]
    ph[:, :, 0] = v0b[:, None] * 0.1 * torch.arange(ph.shape[1], dtype=torch.float32)[None]
    batch["pose_hist"] = ph
batch["v_max_ms"] = torch.tensor([13.8889, 27.7778]); batch["v_max_valid"] = torch.ones(2)
for a in ("_w_goal_point", "_w_tac_goal"):
    setattr(m, a, 0.0)
m._tac_goal_pos_weight = None; m._tac_goal_class_mask = None

class A: mode = "diffusion"
o1 = G.output_digests(G._forward_out(T, m, batch, A, 1234))
o2 = G.output_digests(G._forward_out(T, m, batch, A, 1234))
d = sorted(k for k in o1 if o2.get(k) != o1[k])
print(json.dumps({"tanitad": __import__("tanitad").__file__, "kind": kind, "emit_live": emit_live,
                  "n_outputs": len(o1), "n_differ": len(d), "differ": d}))
