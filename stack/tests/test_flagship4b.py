"""Flagship 4-brain tests (D-030 recovery) — CPU, synthetic.

Pins the full trained-and-wired 4-brain stack:
  (a) budget: WorldModel(flagship4b_config) total within +-5 % of ~261 M, and
      the per-brain breakdown covers the model;
  (b) tactical policy emits maneuver dist + 2 s goal (waypoints + target latent)
      + intent token, and the intent FiLM-CHANGES the operative output;
  (c) strategic context FiLM-CHANGES the tactical output (the ctx->tactical path);
  (d) the strategic->tactical->operative hierarchy composes end to end;
  (e) hierarchical grounding gradients reach EACH level's params (op/tac/str);
  (f) SIGReg position-relaxation leaves the free dims ungrouped (their SIGReg
      gradient is exactly 0) while the complement is regularized;
  (g) one joint training step is finite with ALL loss components present;
  (h) vanilla load of a base WorldModel from a 4b checkpoint (policy keys extra,
      grounding heads saved separately);
  (i) the trained-policy-PROPOSE + grounded-rollout-SCORE tactical selection;
  (j) vocab counts agree with tanitad.refs.refb (the brains size their heads by
      integer counts; the label module owns the strings);
  (k) the pred-only SIGReg variant (the REF-A path) runs on the shared loss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_flagship4b import FlagshipWindowDataset  # noqa: E402

from tanitad.config import (base250cam_config, flagship4b_config,  # noqa: E402
                            flagship4b_smoke_config)
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.models.fourbrain import (Maneuver, TacticalSelector,  # noqa: E402
                                      WorldModel, run_hierarchy)
from tanitad.models.metric_dynamics import grounding_losses  # noqa: E402
from tanitad.models.sigreg import SigReg, position_relaxed  # noqa: E402
from tanitad.train.flagship_losses import (build_grounding,  # noqa: E402
                                           flagship_loss, horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)   # small horizons -> max_h 4


# --------------------------------------------------------------------------- #
# synthetic kinematics (unicycle) -> contract episodes                        #
# --------------------------------------------------------------------------- #
def _poses(T: int, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T, eid, yaw_rate=0.0, accel=0.0, size=64):
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


def _batch(n=4, T=260, cfg=None, **fwd):
    """A batch from turning episodes (T large so early windows are route-valid)."""
    cfg = cfg or flagship4b_smoke_config()
    plan = horizon_plan(cfg, **(fwd or FAST))
    eps = [_episode(T, 0, yaw_rate=0.06), _episode(T, 1, yaw_rate=-0.06),
           _episode(T, 2, yaw_rate=0.0), _episode(T, 3, accel=-1.2)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    batch = default_collate([ds[i] for i in range(n)])
    return cfg, plan, batch


# --------------------------------------------------------------------------- #
# (a) budget + breakdown                                                       #
# --------------------------------------------------------------------------- #
def test_budget_within_5pct_and_breakdown():
    with torch.device("meta"):
        m = WorldModel(flagship4b_config())
    total = sum(p.numel() for p in m.parameters())
    rel = abs(total / 1e6 - 261) / 261
    assert rel <= 0.05, f"flagship total {total/1e6:.2f}M is {rel:+.1%} vs 261 M"
    c = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
    brains = (c(m.encoder) + c(m.readout) + c(m.predictor) + c(m.inv_dyn)
              + c(m.tactical_pred) + c(m.tactical_policy) + c(m.strategic_policy)
              + c(m.imagination) + c(m.sigreg))
    assert brains == total                       # breakdown covers the model
    # the two trained policy brains are real, budget-proven modules
    assert 18_000_000 < c(m.tactical_policy) < 26_000_000
    assert 5_000_000 < c(m.strategic_policy) < 12_000_000
    # the operative predictor is intent-conditioned (the closing FiLM link)
    assert m.predictor.intent_proj is not None


def test_vocab_counts_match_refb():
    from tanitad.refs.refb import (MANEUVER_CLASSES, NAV_COMMANDS,
                                   ROUTE_CLASSES)
    cfg = flagship4b_config()
    assert cfg.tactical_policy.n_maneuvers == len(MANEUVER_CLASSES)
    assert cfg.strategic_policy.n_route == len(ROUTE_CLASSES)
    assert cfg.strategic_policy.n_commands == len(NAV_COMMANDS)


# --------------------------------------------------------------------------- #
# (b) tactical policy outputs + intent FiLM sensitivity on the operative       #
# --------------------------------------------------------------------------- #
def test_tactical_policy_outputs_and_intent_sensitivity():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    B, W = 3, cfg.predictor.window
    states = torch.randn(B, W, m.state_dim)
    ctx = torch.randn(B, cfg.strategic_policy.d_ctx)
    tac = m.tactical_policy(states, ctx)
    assert tac["maneuver_logits"].shape == (B, cfg.tactical_policy.n_maneuvers)
    assert set(tac["waypoints"]) == set(cfg.tactical_policy.waypoint_horizons)
    for k in cfg.tactical_policy.waypoint_horizons:
        assert tac["waypoints"][k].shape == (B, 2)         # 2 s ego sub-waypoints
    assert tac["target_latent"].shape == (B, m.state_dim)  # goal latent
    assert tac["intent"].shape == (B, cfg.tactical_policy.d_intent)
    probs = tac["maneuver_logits"].softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)   # a dist

    # intent FiLM-conditions the operative predictor: identity start, then live.
    actions = torch.randn(B, W, 2)
    i1, i2 = tac["intent"], torch.randn(B, cfg.tactical_policy.d_intent)
    with torch.no_grad():
        # zero-init FiLM => intent has no effect yet (identity start)
        o_a = m.imagine(states, actions, intent=i1)[1]
        o_b = m.imagine(states, actions, intent=i2)[1]
        assert torch.equal(o_a, o_b)
        for blk in m.predictor.blocks:
            nn.init.normal_(blk.film.to_scale_shift.weight, std=0.1)
        o1 = m.imagine(states, actions, intent=i1)[1]
        o2 = m.imagine(states, actions, intent=i2)[1]
        assert float((o1 - o2).abs().max()) > 1e-4          # intent now steers op
        assert torch.equal(o1, m.imagine(states, actions, intent=i1)[1])


# --------------------------------------------------------------------------- #
# (c) strategic context FiLM-changes the tactical output                       #
# --------------------------------------------------------------------------- #
def test_strategic_ctx_conditions_tactical():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    B, W = 2, cfg.predictor.window
    states = torch.randn(B, W, m.state_dim)
    c1 = torch.randn(B, cfg.strategic_policy.d_ctx)
    c2 = torch.randn(B, cfg.strategic_policy.d_ctx)
    with torch.no_grad():
        t1, t2 = m.tactical_policy(states, c1), m.tactical_policy(states, c2)
        # zero-init FiLM => ctx has no effect yet (identity start)
        assert torch.equal(t1["maneuver_logits"], t2["maneuver_logits"])
        assert torch.equal(t1["intent"], t2["intent"])
        for blk in m.tactical_policy.blocks:
            nn.init.normal_(blk.film.to_scale_shift.weight, std=0.1)
        t1, t2 = m.tactical_policy(states, c1), m.tactical_policy(states, c2)
        assert float((t1["maneuver_logits"] - t2["maneuver_logits"]).abs()
                     .max()) > 1e-4
        assert float((t1["intent"] - t2["intent"]).abs().max()) > 1e-4


# --------------------------------------------------------------------------- #
# (d) full strategic->tactical->operative hierarchy composes                   #
# --------------------------------------------------------------------------- #
def test_run_hierarchy_composes():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    B, W = 3, cfg.predictor.window
    states = torch.randn(B, W, m.state_dim)
    actions = torch.randn(B, W, 2)
    out = run_hierarchy(m, states, actions, nav_cmd=torch.tensor([0, 1, 2]))
    for key in ("ctx", "route_logits", "maneuver_logits", "waypoints",
                "target_latent", "intent", "preds"):
        assert key in out
    assert out["ctx"].shape == (B, cfg.strategic_policy.d_ctx)
    assert out["route_logits"].shape == (B, cfg.strategic_policy.n_route)
    for k in cfg.predictor.horizons:
        assert out["preds"][k].shape == (B, m.state_dim)
    assert torch.isfinite(out["preds"][1]).all()
    # base model (no policies) refuses the hierarchy
    base = WorldModel(base_cfg_no_policies(cfg))
    with pytest.raises(ValueError):
        run_hierarchy(base, states, actions)


def base_cfg_no_policies(cfg):
    import dataclasses
    return dataclasses.replace(cfg, tactical_policy=None, strategic_policy=None)


# --------------------------------------------------------------------------- #
# (e) hierarchical grounding gradients reach EACH level's params               #
# --------------------------------------------------------------------------- #
def test_grounding_grads_reach_each_level():
    torch.manual_seed(0)
    cfg, plan, batch = _batch(n=4)
    m = WorldModel(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    total, parts, log = grounding_losses(
        grounding, m.predictor, states, fut_states, batch["actions"],
        batch["future_actions"], batch["pose_last"].float(),
        batch["future_poses"].float(), plan.idx_of, plan.level_cfg,
        pose_scale=10.0)
    m.zero_grad(); grounding.zero_grad()
    total.backward()
    for lvl in ("op", "tac", "str"):
        gi = sum(float(p.grad.abs().sum()) for p in
                 grounding.invdyn[lvl].parameters() if p.grad is not None)
        gs = sum(float(p.grad.abs().sum()) for p in
                 grounding.step[lvl].parameters() if p.grad is not None)
        assert gi > 0, f"metric-invdyn[{lvl}] got no gradient"
        assert gs > 0, f"step-readout[{lvl}] got no gradient"
    # encoder + predictor are shaped by the grounding too
    assert float(m.encoder.patch.weight.grad.abs().sum()) > 0
    assert float(m.predictor.heads["1"].weight.grad.abs().sum()) > 0
    for lvl in ("op", "tac", "str"):
        assert math.isfinite(log[f"g_{lvl}_fwd_ade_m"])


# --------------------------------------------------------------------------- #
# (f) SIGReg position-relaxation leaves the free dims ungrouped                #
# --------------------------------------------------------------------------- #
def test_sigreg_position_relaxation_free_dims_ungrouped():
    torch.manual_seed(0)
    sr = SigReg(n_slices=64, beta=1.0)
    free = 8
    z = torch.randn(128, 40, requires_grad=True)
    loss = position_relaxed(sr, z, free)
    loss.backward()
    assert float(z.grad[:, :free].abs().sum()) == 0.0     # exempt subspace
    assert float(z.grad[:, free:].abs().sum()) > 0.0      # complement regularized
    # free_dims=0 == plain SIGReg on the whole latent (all dims grouped)
    z2 = torch.randn(128, 40, requires_grad=True)
    position_relaxed(sr, z2, 0).backward()
    assert float(z2.grad[:, :free].abs().sum()) > 0.0
    with pytest.raises(ValueError):
        position_relaxed(sr, torch.randn(8, 8), 8)        # empty complement


# --------------------------------------------------------------------------- #
# (g) one joint training step: finite, ALL components present                  #
# --------------------------------------------------------------------------- #
def test_joint_step_finite_all_components():
    torch.manual_seed(0)
    cfg, plan, batch = _batch(n=6)
    m = WorldModel(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    from tanitad.train.flagship_losses import LossWeights
    params = list(m.parameters()) + list(grounding.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    total, log, parts = flagship_loss(
        m, grounding, batch, states, fut_states, plan, cfg,
        weights=LossWeights(), sigreg_variant="full_relaxed",
        sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
        fwd_step_weight=0.5, device="cpu")
    assert torch.isfinite(total)
    for name, v in parts.items():
        assert torch.isfinite(v).all(), f"non-finite part {name}"
    # every advertised component present in the log
    for k in ("pred", "tacpred", "goal", "wp", "man", "route", "sigreg", "inv",
              "ground", "g_op_mid", "g_op_fwd", "g_tac_mid", "g_tac_fwd",
              "g_str_mid", "g_str_fwd"):
        assert k in log and math.isfinite(log[k]), k
    # turning + long episodes -> route CE is actually exercised (> 0)
    assert log["nav_valid_frac"] > 0.0 and log["route"] > 0.0
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    for name, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_pred_only_sigreg_variant_runs():
    """The REF-A SIGReg target (predictor outputs only) also runs on the shared
    loss — the flagship vs REF-A difference is exactly this one argument."""
    torch.manual_seed(0)
    cfg, plan, batch = _batch(n=4)
    m = WorldModel(cfg)
    grounding = build_grounding(m.state_dim, hidden=32)
    from tanitad.train.flagship_losses import LossWeights
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    total, log, _ = flagship_loss(
        m, grounding, batch, states, fut_states, plan, cfg,
        weights=LossWeights(), sigreg_variant="pred_only",
        sigreg_free_dims=0, pose_scale=10.0, fwd_step_weight=0.5, device="cpu")
    assert torch.isfinite(total) and math.isfinite(log["sigreg"])


# --------------------------------------------------------------------------- #
# (h) vanilla base WorldModel loads a 4b checkpoint                            #
# --------------------------------------------------------------------------- #
def test_vanilla_worldmodel_loads_4b_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    flag = WorldModel(cfg)
    grounding = build_grounding(flag.state_dim, hidden=32)
    ckpt = {"model": flag.state_dict(), "grounding": grounding.state_dict(),
            "step": 0}
    torch.save(ckpt, tmp_path / "ckpt.pt")
    ck = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=True)
    # grounding heads are NOT inside ck["model"] (separate key)
    assert not any(k.startswith(("invdyn", "step")) for k in ck["model"])
    # a policy-less base model loads the 4b model dict with NO missing base keys
    base = WorldModel(base_cfg_no_policies(cfg))
    missing, unexpected = base.load_state_dict(ck["model"], strict=False)
    assert not missing, f"base WorldModel missing keys from 4b ckpt: {missing[:6]}"
    assert any("tactical_policy" in k or "strategic_policy" in k
               or "intent_proj" in k for k in unexpected)
    # and the loaded base still runs its operative forward (D1/D2/D3 eval path)
    B, W = 2, cfg.predictor.window
    with torch.no_grad():
        st = base.encode_window(torch.rand(B, W, 1, 64, 64))
        z = base.imagine(st, torch.randn(B, W, 2))[1]
    assert z.shape == (B, base.state_dim) and torch.isfinite(z).all()


# --------------------------------------------------------------------------- #
# (i) trained-policy PROPOSE + grounded-rollout SCORE tactical selection       #
# --------------------------------------------------------------------------- #
def test_tactical_selector_grounded_rescoring():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    m = WorldModel(cfg).eval()
    grounding = build_grounding(m.state_dim, hidden=32)
    sel = TacticalSelector(m, probe_imag=None)
    W = cfg.predictor.window
    states = torch.randn(1, W, m.state_dim)
    past = torch.randn(1, W, 2)
    ctx = torch.randn(1, cfg.strategic_policy.d_ctx)
    maneuvers = [
        Maneuver("lane_keep", torch.zeros(3, 2), maneuver_class=0),
        Maneuver("turn_left", torch.tensor([[0.3, 0.0]] * 3), maneuver_class=1),
        Maneuver("brake", torch.tensor([[0.0, -2.0]] * 3), maneuver_class=4),
    ]
    subgoal = torch.tensor([5.0, 0.0])
    best, scores = sel.propose_and_score(states, past, maneuvers, subgoal,
                                         grounding.step["tac"], ctx)
    assert 0 <= best < len(maneuvers)
    assert scores.shape == (len(maneuvers),) and torch.isfinite(scores).all()
