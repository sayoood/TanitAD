"""REF-A 4-brain trainer + model tests (scripts/refa_train4b.py, CPU/synthetic).

Pins the frozen-DINO twin of the flagship:
  (a) the feature-window dataset threads the SAME flagship label fields
      (nav_cmd / nav_valid / route_target / maneuver_label) onto DINO features;
  (b) one grounded 4-brain step via the SHARED flagship_loss (pred_only SIGReg)
      is finite with ALL components present; grads reach the adapter + predictor;
      the frozen DINO features stay grad-free;
  (c) a smoke --data toy run logs the SAME components as the flagship trainer,
      round-trips its checkpoint, and reuses the frozen standardizer on resume;
  (d) vanilla-load safe: a policy-less RefAModel loads ckpt["model"] (the policy
      keys are simply extra), and the grounding heads are saved separately.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refa_train4b  # noqa: E402  (scripts/refa_train4b.py)

from tanitad.config import (flagship4b_config,  # noqa: E402
                            refa4b_config, refa4b_smoke_config)
from tanitad.refs.refa import RefAModel  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)   # small horizons -> max_h 4
N_TOK = 16                                          # 4x4 grid -> readout 512


def _cfg_plan(rollout_k=4):
    cfg = refa4b_smoke_config()
    cfg.train.rollout_k = rollout_k
    return cfg, horizon_plan(cfg, **FAST)


def _model(cfg) -> RefAModel:
    return RefAModel.from_stack_config(cfg, n_tokens=N_TOK)


def _batch(cfg, plan, batch=8, T=180, n_eps=4):
    """A batch of the SAME label fields the flagship emits, over synthetic DINO
    feature episodes. T large + turning episodes so early windows are route-valid
    and the maneuver classes vary (the flagship-test recipe, on features)."""
    eps = refa_train4b._toy_episodes(n_eps, T, N_TOK, 768)
    ds = refa_train4b.FlagshipFeatureWindowDataset(
        eps, cfg.predictor.window, plan.max_horizon, plan.maneuver_h)
    return eps, ds, default_collate([ds[i] for i in range(batch)])


def _argv(out_dir, steps):
    return ["--data", "toy", "--config", "smoke", "--n-tokens", str(N_TOK),
            "--op-fwd-k", "2", "--tac-fwd-k", "3", "--str-fwd-k", "4",
            "--rollout-k", "4", "--batch-size", "4", "--steps", str(steps),
            "--episodes", "6", "--log-every", "1", "--device", "cpu",
            "--out", str(out_dir)]


# --------------------------------------------------------------------------- #
# full-config horizon plan covers the change-weighted goal ref (regression)     #
# --------------------------------------------------------------------------- #
def test_full_config_horizon_plan_covers_goal_change_ref():
    """The tactical GOAL latent loss is change-weighted, so flagship_loss reads
    BOTH goal_h-1 and goal_h-2 from the encoded futures. At the real config
    (goal_h=20) horizon_plan must encode index 18 too, or the shared loss
    KeyErrors on the very first step — dormant at smoke (goal_h-2 is a tacpred
    target there), so only the full config catches it. Guards both arms (REF-A
    reuses the SAME shared plan)."""
    for cfg in (flagship4b_config(), refa4b_config()):
        for fwd in (dict(op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20), FAST):
            plan = horizon_plan(cfg, **fwd)
            assert plan.goal_h - 1 in plan.idx_of
            assert plan.goal_h - 2 in plan.idx_of, (cfg.tactical_policy, fwd)


# --------------------------------------------------------------------------- #
# (a) the dataset threads the flagship label fields onto DINO features          #
# --------------------------------------------------------------------------- #
def test_feature_dataset_threads_flagship_labels():
    cfg, plan = _cfg_plan()
    _, _, batch = _batch(cfg, plan, batch=8)
    for k in ("feats", "actions", "future_feats", "future_actions",
              "future_poses", "pose_last", "nav_cmd", "nav_valid",
              "route_target", "maneuver_label"):
        assert k in batch, k
    B = batch["feats"].shape[0]
    assert batch["nav_cmd"].shape == (B,) and batch["nav_cmd"].dtype == torch.long
    assert batch["nav_valid"].dtype == torch.bool
    assert batch["route_target"].shape == (B,)
    assert batch["maneuver_label"].shape == (B,)
    # label ranges agree with the shared vocab counts.
    assert int(batch["maneuver_label"].max()) < cfg.tactical_policy.n_maneuvers
    assert int(batch["route_target"].max()) < cfg.strategic_policy.n_route
    # T=180 turning episodes: early windows clear NAV_MIN_STEPS (route-valid).
    assert bool(batch["nav_valid"].any())


# --------------------------------------------------------------------------- #
# (b) one grounded 4-brain step via the shared loss: finite, all components     #
# --------------------------------------------------------------------------- #
def test_grounded_4b_step_finite_and_grad_free_features():
    torch.manual_seed(0)
    cfg, plan = _cfg_plan()
    model = _model(cfg)
    model.standardizer.fit(torch.randn(64, N_TOK, 768) for _ in range(2))
    grounding = build_grounding(model.state_dim, hidden=32)
    _, _, batch = _batch(cfg, plan, batch=8)

    states = model.encode_window(batch["feats"])
    fut_states = model.encode_window(batch["future_feats"][:, plan.needed_fut])
    total, log, parts = flagship_loss(
        model, grounding, batch, states, fut_states, plan, cfg,
        weights=LossWeights(), sigreg_variant=refa_train4b.SIGREG_VARIANT,
        sigreg_free_dims=0, pose_scale=10.0, fwd_step_weight=0.5, device="cpu")

    assert refa_train4b.SIGREG_VARIANT == "pred_only"
    assert torch.isfinite(total)
    for name, v in parts.items():
        assert torch.isfinite(v).all(), f"non-finite part {name}"
    # the SAME components the flagship trainer logs (H15 excepted — not shared).
    for k in ("pred", "tacpred", "roll", "goal", "wp", "man", "route", "sigreg",
              "inv", "ground", "g_op_mid", "g_op_fwd", "g_tac_mid", "g_tac_fwd",
              "g_str_mid", "g_str_fwd", "nav_valid_frac"):
        assert k in log and math.isfinite(log[k]), k
    assert log["nav_valid_frac"] > 0 and log["route"] > 0    # route CE exercised

    total.backward()
    gsum = lambda ps: sum(float(p.grad.abs().sum()) for p in ps  # noqa: E731
                          if p.grad is not None)
    assert gsum(model.adapter.parameters()) > 0        # trainable encoder axis
    assert gsum(model.predictor.parameters()) > 0
    assert gsum(grounding.parameters()) > 0
    # frozen DINO features carry no grad (the correct main-vs-REF-A asymmetry).
    assert batch["feats"].requires_grad is False and batch["feats"].grad is None
    assert batch["future_feats"].grad is None
    for name, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------------------- #
# (c) smoke --data toy run: flagship columns, ckpt round-trip, frozen stats     #
# --------------------------------------------------------------------------- #
def test_train_smoke_columns_and_resume(tmp_path):
    out_dir = tmp_path / "refa4b"
    summary = refa_train4b.main(_argv(out_dir, 2))
    # logs the SAME shared components as the flagship trainer (+ REF-A monitors).
    for k in ("pred", "tacpred", "roll", "goal", "wp", "man", "route", "sigreg",
              "inv", "ground", "g_op_fwd_ade_m", "adapter_std", "gnorm_adapter"):
        assert k in summary["final"] and math.isfinite(summary["final"][k]), k
    assert summary["final_step"] == 1
    assert summary["sigreg_variant"] == "pred_only"
    ckpt = out_dir / "ckpt.pt"
    assert ckpt.exists()

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    # grounding heads saved under a SEPARATE key (not inside ck["model"]).
    assert "grounding" in ck
    assert not any(k.startswith(("invdyn", "step")) for k in ck["model"])
    # standardizer stats travel inside ck["model"] and are frozen.
    assert bool(ck["model"]["standardizer.fitted"])

    # resume: rerun with more steps -> picks up at step 2, finishes at 3, and
    # reuses the stored standardizer stats (never refit).
    summary2 = refa_train4b.main(_argv(out_dir, 4))
    assert summary2["final_step"] == 3
    ck2 = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert ck2["step"] == 3
    assert torch.equal(ck2["model"]["standardizer.mean"],
                       ck["model"]["standardizer.mean"])


# --------------------------------------------------------------------------- #
# (d) vanilla-load safe: a policy-less RefAModel loads a 4b checkpoint          #
# --------------------------------------------------------------------------- #
def test_vanilla_refamodel_loads_4b_ckpt(tmp_path):
    out_dir = tmp_path / "refa4b_v"
    refa_train4b.main(_argv(out_dir, 2))
    ck = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=True)

    # a base (policy-less) REF-A loads the 4b model dict: no missing base keys,
    # and the policy/intent keys are simply extra (a strict subset relationship).
    base = RefAModel(refa4b_smoke_config().predictor, adapter_kind="grid",
                     n_tokens=N_TOK, grid=4, grid_d_readout=32)
    missing, unexpected = base.load_state_dict(ck["model"], strict=False)
    assert not missing, f"base RefAModel missing keys from 4b ckpt: {missing[:6]}"
    assert any("tactical_policy" in k or "strategic_policy" in k
               or "intent_proj" in k for k in unexpected)
    # and the base model still runs its operative forward (the eval path).
    B, W = 2, base.pred_cfg.window
    with torch.no_grad():
        st = base.encode_window(torch.randn(B, W, N_TOK, 768).half())
        z = base.predict(st, torch.randn(B, W, 2))[1]
    assert z.shape == (B, base.state_dim) and torch.isfinite(z).all()
