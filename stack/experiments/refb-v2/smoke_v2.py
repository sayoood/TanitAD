"""CPU shape/param/loss/norms smoke for REF-B v2 (B1 time-anchored + B2 yr0).
Loads the MODIFIED refb.py (as tanitad.refs.refb) + trainer via importlib so
the live package files are untouched. Checks: ego_emb=2, anchored decoder wired,
no dangling wp_heads, anchor-cls + WTA loss computes, H26 norms, gating intact.
"""
import sys, importlib.util
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import tanitad.refs.refb as refb          # real package module (now modified)
rbt = _load("refb_train", "/root/refb_train_v3.py")
torch.manual_seed(0)

# ---- gating: default model must be pre-v2 (has wp_heads, no ego/anchor) ------
base = refb.RefBModel(refb.refb_smoke_config())    # tiny; gating check only
assert hasattr(base.tactical, "wp_heads") and not base.tactical.anchored
assert base.ego_emb is None and base.speed_emb is None
print("[gate] default refb_config: wp_heads present, no anchored decoder, "
      "no ego_emb -> pre-v2 identity OK")
del base

# ---- v2 smoke config ---------------------------------------------------------
cfg = refb.refb_smoke_config()
cfg.speed_input = cfg.aux_accel = True
cfg.aux_yaw = True
cfg.ego_dropout = 0.5
cfg.path_dists = (2, 5, 10, 20)          # refbpatch distance path head KEPT
cfg.yaw_input = True                     # B2
cfg.anchored_tactical = True             # B1
cfg.anchor_space = "time"                # FINAL
cfg.anchor_n = 16; cfg.anchor_pool = 256
cfg.anchor_d = 32; cfg.anchor_heads = 4; cfg.anchor_layers = 2
m = refb.RefBModel(cfg)

# ---- assembly assertions -----------------------------------------------------
assert m.ego_emb is not None and m.ego_emb.in_features == 2, "ego_emb must be Linear(2,·)"
assert m.speed_emb is None, "speed_emb must be None when yaw_input"
assert hasattr(m.tactical, "wp_decoder") and m.tactical.anchored
assert not hasattr(m.tactical, "wp_heads"), "unimodal wp_heads must be DROPPED"
assert m.path_heads is not None, "refbpatch fixed-distance path head must be KEPT"
assert m.tactical.wp_decoder.anchors.shape == (16, 4, 2)
pb = refb.param_breakdown(m)
print(f"[assembly] ego_emb=Linear(2,{m.ego_emb.out_features}) | wp_decoder "
      f"anchors={tuple(m.tactical.wp_decoder.anchors.shape)} | wp_heads DROPPED "
      f"| path_head KEPT | params={pb}")

# ---- synthetic batch (smoke shapes: window 4, 1ch 64px) ----------------------
B, W, C, H = 4, cfg.window, cfg.encoder.in_channels, cfg.encoder.image_size
max_h = max(max(cfg.tactical.waypoint_horizons), cfg.operative.action_seq - 1)
batch = {
    "frames": torch.rand(B, W, C, H, H),
    "actions": torch.randn(B, W, 2) * 0.1,
    "future_actions": torch.randn(B, max_h, 2) * 0.1,
    "future_poses": torch.randn(B, max_h, 4),
    "pose_last": torch.randn(B, 4),
    "pose_prev": torch.randn(B, 4),           # B2 backward-diff yr0 source (t-1)
    "nav_cmd": torch.randint(0, 3, (B,)),
    "nav_valid": torch.ones(B, dtype=torch.bool),
    "route_target": torch.randint(0, 3, (B,)),
}
assert "future_frames" not in batch

# ---- LEAKAGE GUARD: the yr0 INPUT must read ONLY past/window (no t+1) ---------
# Capture the yr0 the trainer feeds the model; it must be INVARIANT to
# future_poses (the leaked yaw[t+1]) and RESPOND to pose_prev (window t-1).
cap = {}
_h = m.register_forward_pre_hook(
    lambda mod, a, k: cap.__setitem__(
        "yr0", None if k.get("yr0") is None else k["yr0"].detach().clone()),
    with_kwargs=True)
_cl = lambda b: {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                 for kk, vv in b.items()}
rbt.compute_losses(m, _cl(batch), "cpu"); yrA = cap["yr0"]
bB = _cl(batch); bB["future_poses"][:, 0, 2] += 5.0     # corrupt ONLY the future
rbt.compute_losses(m, bB, "cpu"); yrB = cap["yr0"]
bC = _cl(batch); bC["pose_prev"][:, 2] += 1.0           # change window t-1
rbt.compute_losses(m, bC, "cpu"); yrC = cap["yr0"]
_h.remove()
assert yrA is not None, "yr0 not fed to the model"
assert torch.allclose(yrA, yrB), "yr0 CHANGED with future_poses -> LEAKAGE!"
assert not torch.allclose(yrA, yrC), "yr0 ignored pose_prev -> not window-wired"
print("[leakage] yr0 INVARIANT to future_poses[:,0], RESPONDS to pose_prev "
      "-> past-only backward-diff, NO t+1 access")

# ---- model-forward output surface --------------------------------------------
m.train()
with torch.no_grad():
    mo = m(batch["frames"], nav_cmd=batch["nav_cmd"],
           v0=batch["pose_last"][:, 3], yr0=torch.randn(B))
for k in ("anchor_logits", "anchor_traj", "sel_idx", "waypoints",
          "path_waypoints"):
    assert k in mo, f"model out missing {k}"
assert "mm_path" not in mo, "time-anchored must NOT emit mm_path"
print("[forward] out has anchor_logits/anchor_traj/sel_idx + waypoints "
      "(multimodal selected) + path_waypoints (distance aux KEPT)")

# ---- loss + backward ---------------------------------------------------------
out = rbt.compute_losses(m, batch, "cpu")
for k in ("cls", "wta", "anchor_acc", "n_modes", "conf_norm", "prior_norm"):
    assert k in out, f"loss dict missing {k}"
loss = out["loss"]
assert torch.isfinite(loss), f"loss not finite: {loss}"
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
show = ["loss", "action", "seq", "wp", "cls", "wta", "man", "route", "inv",
        "conf", "path", "aux_accel", "aux_yaw", "anchor_acc"]
print("[loss] " + "  ".join(f"{k}={float(out[k]):.4f}" for k in show))
print(f"[loss] finite=all  gnorm={float(gnorm):.3f}  n_modes={out['n_modes']}  "
      f"(utilization >1 mode over this batch)")
print(f"[H26 norms] conf_norm={float(out['conf_norm']):.3f}  "
      f"prior_norm={float(out['prior_norm']):.3f}  "
      f"ratio(prior/conf)={float(out['prior_norm'])/max(float(out['conf_norm']),1e-6):.2f} "
      f"(<~1 => prior BIASES, does not swamp)")

# ---- eval-mode forward (ego_dropout off) -------------------------------------
m.eval()
with torch.no_grad():
    o2 = rbt.compute_losses(m, batch, "cpu")
assert torch.isfinite(o2["loss"])
print(f"[eval] forward OK, loss={float(o2['loss']):.4f}")

# ---- B2 yr0 actually consumed. The shared CausalBlock FiLM is ZERO-INIT
# (identity at step 0), so cmd (v0 AND yr0) has no downstream effect UNTIL the
# modulation trains (standard AdaLN/FiLM-zero stable init — same for the pre-v2
# v0). Take a few steps so the FiLM is non-identity, then confirm yr0 moves the
# continuous anchor outputs (waypoints are the DISCRETE argmax pick). ----------
v0 = batch["pose_last"][:, 3]
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
m.train()
for _ in range(30):
    opt.zero_grad()
    rbt.compute_losses(m, batch, "cpu")["loss"].backward()
    opt.step()
m.eval()
with torch.no_grad():
    oa = m(batch["frames"], nav_cmd=batch["nav_cmd"], v0=v0, yr0=torch.zeros(B))
    ob = m(batch["frames"], nav_cmd=batch["nav_cmd"], v0=v0,
           yr0=torch.full((B,), 10.0))
dlog = float((oa["anchor_logits"] - ob["anchor_logits"]).abs().mean())
dtraj = float((oa["anchor_traj"] - ob["anchor_traj"]).abs().mean())
assert dlog > 1e-5 or dtraj > 1e-5, "yr0 has NO effect after training -> B2 broken!"
print(f"[B2] after 30 steps (FiLM non-identity), yr0 0 vs 10: "
      f"|Δlogits|={dlog:.4f} |Δtraj|={dtraj:.4f} -> yr0 CONSUMED "
      f"(ego_emb->strategic->ctx->decoder)")

# ---- distance-variant still BUILDS (switch prepped, not chosen) --------------
cfgd = refb.refb_smoke_config()
cfgd.speed_input = cfgd.aux_accel = cfgd.aux_yaw = True
cfgd.ego_dropout = 0.5; cfgd.path_dists = (2, 5, 10, 20)
cfgd.yaw_input = True; cfgd.anchored_tactical = True
cfgd.anchor_space = "distance"
cfgd.anchor_n = 16; cfgd.anchor_pool = 256
cfgd.anchor_d = 32; cfgd.anchor_heads = 4; cfgd.anchor_layers = 2
md = refb.RefBModel(cfgd)
with torch.no_grad():
    od = md(torch.rand(2, cfgd.window, 1, 64, 64),
            nav_cmd=torch.zeros(2, dtype=torch.long), v0=torch.rand(2),
            yr0=torch.rand(2))
assert "mm_path" in od
print("[switch] distance-anchored variant also builds+forwards (mm_path present)")

print("\nSMOKE_OK")
