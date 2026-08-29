"""Synthetic-episode smoke for the post-training loop — with COUNTERS asserted.

⛔ THE TRAP THIS FILE IS BUILT AGAINST
--------------------------------------
*An exit 0 over an empty set is indistinguishable from a real pass.* A smoke
that iterates zero batches, scores zero windows and steps zero times returns
cleanly and looks exactly like success. So the loop carries counters, refuses to
write its done-marker when a required counter is zero, and this file tests BOTH
directions — that a real run counts, and that an empty one is refused.

No GPU, no real corpus: a tiny synthetic policy stands in for refcv3's diffusion
decoder through the injectable ``sample_fn`` seam.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from torch import nn

from tanitad.rl import audit as AUD
from tanitad.rl import rewards as R
from tanitad.rl.config import PostTrainConfig
from tanitad.rl.posttrain import (SmokeCounters, apply_exploration_noise,
                                  rl_objective, run_posttrain,
                                  select_trainable)


# ---------------------------------------------------------------------------
# A tiny stand-in policy with a trunk and a head, so freezing is meaningful
# ---------------------------------------------------------------------------

class TinyCore(nn.Module):
    """Mirrors the REAL RefCV3Model layout: core.encoder + core.decoder."""

    def __init__(self, n_anchors=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU(),
                                     nn.Linear(8, n_anchors))


class TinyPlanner(nn.Module):
    """A stand-in whose MODULE NAMES match the real model's.

    ⚠️ An earlier version named its parts ``core`` / ``scorer`` /
    ``lat_head_tac`` — chosen to match the config's then-default
    ``trainable_prefixes``. It passed while the real wiring was inverted
    (selector trainable, generator frozen). A mock whose names are chosen to
    satisfy the code under test cannot catch a naming bug; it manufactures the
    agreement it is supposed to check. These names now mirror
    ``RefCV3Model``: ``core.encoder`` = trunk, ``core.decoder`` = generator,
    ``scorer`` = selector (forbidden).
    """

    def __init__(self, n_anchors=3, n_steps=21):
        super().__init__()
        self.core = TinyCore(n_anchors)
        self.scorer = nn.Linear(8, n_anchors)      # the SELECTOR — never trained
        self.n_anchors, self.n_steps = n_anchors, n_steps


def make_sample_fn(model, n_anchors=3, n_steps=21, dt=0.1):
    """(batch, cfg) -> (traj [B,N,G,S,2], logp [B,N,G], ctx)."""
    def sample_fn(batch, cfg):
        b = 2
        g = cfg.group_size
        feats = torch.ones(b, 4)
        h = model.core.encoder(feats)                          # [b, 8]
        head = model.core.decoder(h)                           # [b, N]

        t = torch.arange(n_steps, dtype=torch.float32) * dt
        base = torch.stack([10.0 * t, torch.zeros_like(t)], dim=-1)
        traj = base.expand(b, n_anchors, g, n_steps, 2).clone()
        # lateral offset per anchor so candidates actually differ
        lat = torch.linspace(-1.0, 1.0, n_anchors).view(1, n_anchors, 1, 1)
        traj[..., 1] = traj[..., 1] + lat
        offset = torch.randn(b, n_anchors, g, 1, 1) * 0.05
        traj = traj + apply_exploration_noise(offset, cfg)

        logp = head.unsqueeze(-1).expand(b, n_anchors, g)
        logp = torch.log_softmax(logp.reshape(b, -1), dim=-1).reshape(
            b, n_anchors, g)
        ctx = {"gt_traj": base, "obstacles": torch.tensor([[18.0, 0.0]])}
        return traj, logp, ctx
    return sample_fn


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def test_frozen_trunk_leaves_only_the_generator_trainable():
    m = TinyPlanner()
    rep = select_trainable(m, PostTrainConfig(freeze_trunk=True))
    assert rep["freeze_trunk"] is True
    assert rep["frozen_params"] > 0
    assert 0 < rep["trainable_params"] < rep["total_params"]
    for name, p in m.named_parameters():
        assert p.requires_grad == name.startswith("core.decoder"), name


def test_the_selector_is_frozen_even_in_the_mock():
    m = TinyPlanner()
    select_trainable(m, PostTrainConfig())
    assert not m.scorer.weight.requires_grad


def test_no_trainable_parameters_is_refused():
    """An optimiser over an empty parameter list steps and changes nothing."""
    m = TinyPlanner()
    cfg = PostTrainConfig(freeze_trunk=True, trainable_prefixes=("nonexistent",))
    with pytest.raises(RuntimeError, match="no parameters are trainable"):
        select_trainable(m, cfg)


# ---------------------------------------------------------------------------
# Counters — both directions
# ---------------------------------------------------------------------------

def test_empty_run_is_REFUSED_not_reported_as_success():
    """⛔ The false-green guard."""
    c = SmokeCounters()
    with pytest.raises(RuntimeError, match="FALSE-GREEN REFUSED"):
        c.assert_nonzero()


def test_counters_pass_when_every_phase_did_work():
    c = SmokeCounters(steps=1, windows_scored=2, samples_scored=8,
                      reward_evals=1, advantage_evals=1, optimizer_steps=1)
    c.assert_nonzero()


def test_partially_empty_run_names_the_zero_counter():
    c = SmokeCounters(steps=1, windows_scored=0, samples_scored=8,
                      reward_evals=1, advantage_evals=1, optimizer_steps=1)
    with pytest.raises(RuntimeError, match="windows_scored"):
        c.assert_nonzero()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def test_grpo_smoke_runs_counts_and_writes_a_done_marker(tmp_path):
    m = TinyPlanner()
    cfg = PostTrainConfig(method="grpo", group_size=4, steps=3, lr=1e-3,
                          out_dir=str(tmp_path))
    before = m.core.encoder[0].weight.detach().clone()

    summary = run_posttrain(m, make_sample_fn(m), cfg)

    assert summary["done"] is True
    assert summary["steps"] == 3
    for k in SmokeCounters.REQUIRED:
        assert summary["counters"][k] > 0, k
    # the done-marker really landed on disk
    with open(os.path.join(tmp_path, "summary.json"), encoding="utf-8") as fh:
        assert json.load(fh)["done"] is True
    # and config.json records every knob
    with open(os.path.join(tmp_path, "config.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    assert rec["method"] == "grpo" and rec["group_size"] == 4
    assert rec["freeze_trunk"] is True and "reward_weights" in rec
    # ⭐ the trunk really did not move
    assert torch.allclose(before, m.core.encoder[0].weight), "frozen trunk MOVED"


def test_smoke_reports_the_reward_audit_in_its_summary(tmp_path):
    """The loop's own audit runs with NO scene context, so it must say so.

    ⚠️ It would be easy — and wrong — to let the summary say `clean` here. The
    loop audits the reward SPEC, not the run's data, so `collision` and
    `headway` have nothing to fire on and the verdict cannot rule. The
    honest verdict is INCONCLUSIVE, and the run record carries it.
    """
    m = TinyPlanner()
    cfg = PostTrainConfig(steps=2, out_dir=str(tmp_path))
    summary = run_posttrain(m, make_sample_fn(m), cfg)
    assert "reward_audit" in summary
    assert summary["reward_audit"]["verdict"] == "INCONCLUSIVE"
    assert summary["reward_audit"]["flagged"] is False
    assert set(summary["reward_audit"]["dead_components"]) == {
        "collision", "headway"}


def test_smoke_with_the_hackable_reward_reports_FLAGGED(tmp_path):
    """The regression arm survives end-to-end, not just in the audit unit test.

    progress-only has NO dead components (progress is its only weighted term),
    so the audit CAN rule here — and it must rule FLAGGED.
    """
    m = TinyPlanner()
    cfg = PostTrainConfig(steps=2, out_dir=str(tmp_path),
                          reward_weights=dict(R.HACKABLE_WEIGHTS))
    summary = run_posttrain(m, make_sample_fn(m), cfg)
    assert summary["reward_audit"]["verdict"] == "FLAGGED", (
        "a progress-only run must be flagged in its own summary.json")
    assert summary["reward_audit"]["flagged"] is True


def test_awr_requires_an_imitation_loss():
    m = TinyPlanner()
    cfg = PostTrainConfig(method="awr", steps=1)
    traj, logp, ctx = make_sample_fn(m)(0, cfg)
    with pytest.raises(ValueError, match="imitation_loss"):
        rl_objective(traj, logp, ctx, cfg, R.RewardSpec())


def test_objective_refuses_a_selector_bearing_context():
    m = TinyPlanner()
    cfg = PostTrainConfig(steps=1)
    traj, logp, ctx = make_sample_fn(m)(0, cfg)
    with pytest.raises(ValueError, match="DISJOINTNESS VIOLATED"):
        rl_objective(traj, logp, {**ctx, "sel_score": torch.zeros(2)}, cfg,
                     R.RewardSpec())


# ---------------------------------------------------------------------------
# Exploration noise
# ---------------------------------------------------------------------------

def test_multiplicative_noise_scales_with_magnitude():
    """DDv2's ablation preferred multiplicative (90.1 vs 89.7 PDMS)."""
    torch.manual_seed(0)
    cfg = PostTrainConfig(noise_mode="multiplicative", noise_scale=0.5)
    big = apply_exploration_noise(torch.full((4096, 1), 10.0), cfg)
    small = apply_exploration_noise(torch.full((4096, 1), 0.1), cfg)
    assert big.std() > small.std() * 50


def test_additive_noise_does_not_scale():
    torch.manual_seed(0)
    cfg = PostTrainConfig(noise_mode="additive", noise_scale=0.5)
    big = apply_exploration_noise(torch.full((4096, 1), 10.0), cfg)
    small = apply_exploration_noise(torch.full((4096, 1), 0.1), cfg)
    assert big.std() == pytest.approx(float(small.std()), rel=0.2)


def test_zero_noise_is_identity():
    cfg = PostTrainConfig(noise_scale=0.0)
    x = torch.randn(16, 2)
    assert torch.allclose(apply_exploration_noise(x, cfg), x)
