"""R4 audit probe -- MEASURED on the TIP source (c36b6ddd), read-only.

Question: does any TACTICAL output enter the OPERATIVE plan (V6Stack.emit's
controls/waypoints), and does the operative plan loss's gradient reach it?

Method (counterfactual, same input batch, same weights except the lever):
  * wake the zero-init emission head (its final Linear is zero-init for the CV
    warm start, train_v58f_unicycle_head.py:205-206, so at init NOTHING upstream
    can move the plan -- a probe at init would read "absent" for every path);
  * GOAL lever: force goal_head_tac's type_head bias to a large one-hot (token i
    vs token j) -> does plan["waypoints"] move?
  * BEHAVIOUR lever: the same on act_head_lat / act_head_lon -> does the plan move?
  * gradient reach: backward of the planner's own WTA loss
    (train_v6_staged.v6_loss_step's `plan` term shape) -> which layer_tac tensors
    receive gradient?
Usage: python r4_audit_probe.py <stack_root>   (the TIP copy)
"""
import copy
import json
import sys

stack_root = sys.argv[1]
sys.path.insert(0, stack_root)
sys.path.insert(0, stack_root + "/scripts")
import torch  # noqa: E402

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402


def small(**kw):
    return V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2, horizons=(1,),
                                  action_dim=3, residual=True),
        d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32, f_blocks=1,
        aux_hidden=16, sigreg_slices=8, plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2),
        tac_band_s=(0.2, 0.6), d_plan_feat=16, emission_hidden=16, d_goal_embed=16,
        n_candidates=4, **kw)


def wake_emission(s, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (s.emission.net[-1].weight, s.emission.net[-1].bias):
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def force_token(head, tok, big=40.0):
    with torch.no_grad():
        head.type_head.bias.zero_()
        head.type_head.bias[tok] = big


def plan_of(s, b):
    with torch.no_grad():
        return s.forward(**b)["plan"]["waypoints"].clone()


out = {"source": stack_root, "_evidence_class": "MEASURED (tip source, CPU, seed 0, toy geometry)"}
for vv in ("v7.0",):
    torch.manual_seed(0)
    s = V6Stack(small(tac_vocab_version=vv))
    s.eval()
    wake_emission(s)
    b = s.synthetic_batch(2, seed=1)
    # --- GOAL lever
    s_g = copy.deepcopy(s)
    force_token(s_g.goal_head_tac, 0); p0 = plan_of(s_g, b)
    force_token(s_g.goal_head_tac, 7); p1 = plan_of(s_g, b)
    goal_moves = float((p0 - p1).abs().max())
    # --- BEHAVIOUR levers
    s_l = copy.deepcopy(s)
    force_token(s_l.act_head_lat, 0); q0 = plan_of(s_l, b)
    force_token(s_l.act_head_lat, 1); q1 = plan_of(s_l, b)
    force_token(s_l.act_head_lon, 0); q2 = plan_of(s_l, b)
    force_token(s_l.act_head_lon, 7); q3 = plan_of(s_l, b)
    lat_moves = float((q0 - q1).abs().max())
    lon_moves = float((q2 - q3).abs().max())
    # sanity: the behaviour lever DID move the tactical decision
    with torch.no_grad():
        force_token(s_l.act_head_lat, 0); a0 = s_l.forward(**b)["a_lat"]["probs"]
        force_token(s_l.act_head_lat, 1); a1 = s_l.forward(**b)["a_lat"]["probs"]
    lat_decision_moved = float((a0 - a1).abs().max())
    # --- gradient reach of the planner WTA loss (the v6_loss_step `plan` term)
    s_r = copy.deepcopy(s)
    s_r.train()
    for p in s_r.parameters():
        p.requires_grad_(True)
    o = s_r.forward(**b)
    fan = o["plan"]["waypoints"].float()
    tgt = torch.randn(fan.shape[0], fan.shape[2], 2, generator=torch.Generator().manual_seed(5))
    err = (fan - tgt[:, None]).norm(dim=-1).mean(dim=-1)
    lp = err[torch.arange(fan.shape[0]), err.argmin(dim=1)].mean()
    names = [n for n, _ in s_r.named_parameters()]
    grads = torch.autograd.grad(lp, [p for _, p in s_r.named_parameters()], allow_unused=True)
    live = sorted(n for n, g in zip(names, grads) if g is not None and float(g.abs().max()) > 0)
    tac_live = [n for n in live if s_r.group_of(n) == "layer_tac"]
    out[vv] = {
        "plan_moves_when_GOAL_token_changes (max|dwp| m)": goal_moves,
        "plan_moves_when_LAT_BEHAVIOUR_changes (max|dwp| m)": lat_moves,
        "plan_moves_when_LON_BEHAVIOUR_changes (max|dwp| m)": lon_moves,
        "lat_decision_probs_moved (control: the lever is real)": lat_decision_moved,
        "plan_loss_grad_reaches_layer_tac_tensors": tac_live,
        "plan_loss_grad_reaches_act_head_lat_or_lon": any(n.startswith(("act_head_lat.", "act_head_lon.")) for n in live),
        "plan_loss_grad_reaches_goal_head_tac": any(n.startswith("goal_head_tac.") for n in live),
        "groups_reached": sorted({s_r.group_of(n) for n in live}),
        "live_by_group": {g: sorted(n for n in live if s_r.group_of(n) == g)
                          for g in sorted({s_r.group_of(n) for n in live})},
    }
    # at init (emission zero-init) -- the path is dead for EVERY lever
    torch.manual_seed(0)
    s0 = V6Stack(small(tac_vocab_version=vv)); s0.eval()
    force_token(s0.goal_head_tac, 0); z0 = plan_of(s0, b)
    force_token(s0.goal_head_tac, 7); z1 = plan_of(s0, b)
    out[vv]["AT_INIT plan_moves_when_GOAL_changes (emission zero-init)"] = float((z0 - z1).abs().max())
print(json.dumps(out, indent=1))
