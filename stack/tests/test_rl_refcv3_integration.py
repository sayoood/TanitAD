"""Integration: the RL library against the REAL RefCV3Model, not a stand-in.

⛔ WHY THIS FILE EXISTS — IT CAUGHT A REAL INVERSION
----------------------------------------------------
The unit smoke in ``test_rl_posttrain.py`` uses a ``TinyPlanner`` mock whose
module names were chosen to match the config's ``trainable_prefixes``. It passed
happily while the DEFAULT prefixes were **wrong**: they named ``scorer`` (the
SELECTOR, 87 params) and omitted ``core.decoder`` (the diffusion decoder — the
GENERATOR). Against the real model that default trained the selector, froze the
generator, reported ``0.81 % trainable``, and failed nothing.

That is the precise inversion of the library's own doctrine (``advantage.py``:
*"this library targets the generator, never the selector"*), and it would have
trained the one component DDv2 names as the weak link — selector over-reliance —
with a reward built specifically to avoid depending on it.

⇒ **A mock validates the plumbing; only the real model validates the wiring.**
"""

from __future__ import annotations

import pytest
import torch

from tanitad.refs import refc_v3 as v3
from tanitad.rl import PostTrainConfig, select_trainable


@pytest.fixture(scope="module")
def model():
    return v3.RefCV3Model(v3.refc_v3_smoke_config(hier=True))


def test_default_prefixes_train_the_DECODER_on_the_real_model(model):
    rep = select_trainable(model, PostTrainConfig())
    assert rep["trainable_params"] > 0
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names, "nothing trainable"
    assert all(n.startswith("core.decoder") for n in names), (
        f"non-decoder parameters are trainable: "
        f"{[n for n in names if not n.startswith('core.decoder')][:8]}")
    # the decoder is a real fraction of the model, not a rounding error
    assert rep["trainable_fraction"] > 0.05, (
        f"only {rep['trainable_fraction']:.2%} trainable — that is the shape of "
        "the bug this file was written for")


def test_the_SELECTOR_is_never_trainable_by_default(model):
    """⛔ The inversion guard."""
    select_trainable(model, PostTrainConfig())
    for name, p in model.named_parameters():
        if name.startswith("scorer"):
            assert not p.requires_grad, (
                f"{name} is trainable — the library must post-train the "
                "GENERATOR, never the SELECTOR")


def test_the_TRUNK_is_never_trainable_by_default(model):
    """The encoder carries every representation number we have measured."""
    select_trainable(model, PostTrainConfig())
    for name, p in model.named_parameters():
        if name.startswith("core.encoder"):
            assert not p.requires_grad, f"trunk parameter {name} is trainable"


def test_explicitly_naming_the_selector_is_REFUSED(model):
    """Even a deliberate override cannot smuggle the selector in."""
    cfg = PostTrainConfig(trainable_prefixes=("scorer",))
    with pytest.raises(RuntimeError, match="FORBIDDEN PARAMETERS"):
        select_trainable(model, cfg)


def test_freeze_report_records_the_prefixes_it_used(model):
    rep = select_trainable(model, PostTrainConfig())
    assert rep["trainable_prefixes"] == ["core.decoder"]
    assert rep["forbidden_prefixes"] == ["scorer"]


def test_unfrozen_still_refuses_the_selector(model):
    """freeze_trunk=False widens the trunk, not the forbidden set."""
    with pytest.raises(RuntimeError, match="FORBIDDEN PARAMETERS"):
        select_trainable(model, PostTrainConfig(freeze_trunk=False))


def test_real_model_fan_shape_matches_the_reward_contract(model):
    """`anchor_traj` must be [B, N, S, 2] — the shape the rewards consume."""
    cfg = model.cfg.core
    enc = cfg.encoder
    b = 1
    frames = torch.zeros(b, cfg.window, enc.in_channels, enc.image_size,
                         enc.image_width or enc.image_size)
    with torch.no_grad():
        out = model(frames, v0=torch.zeros(b))
    fan = out["anchor_traj"]
    assert fan.dim() == 4 and fan.shape[0] == b and fan.shape[-1] == 2, (
        f"anchor_traj is {tuple(fan.shape)}; the reward components expect "
        "[..., S, 2] waypoints")
    from tanitad.rl import RewardSpec
    r = RewardSpec()(fan, {"dt": 0.1})
    assert r.shape == fan.shape[:2], (
        f"reward {tuple(r.shape)} must be one scalar per candidate "
        f"{tuple(fan.shape[:2])}")
    assert torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# The adapter: an END-TO-END post-training step on the REAL model
# ---------------------------------------------------------------------------

def test_sample_offsets_shapes_and_differentiability():
    from tanitad.rl.refcv3_adapter import sample_offsets
    off = torch.randn(2, 3, 5, 2, requires_grad=True)
    cfg = PostTrainConfig(group_size=4)
    s, logp = sample_offsets(off, cfg)
    assert s.shape == (2, 3, 4, 5, 2)
    assert logp.shape == (2, 3, 4)
    logp.sum().backward()
    assert off.grad is not None, "logp must be differentiable w.r.t. the decoder"


def test_sample_offsets_survives_a_zero_offset():
    """⚠️ A zero offset makes the multiplicative scale 0 -> logp -inf -> NaN."""
    from tanitad.rl.refcv3_adapter import sample_offsets
    _, logp = sample_offsets(torch.zeros(1, 2, 4, 2), PostTrainConfig(group_size=2))
    assert torch.isfinite(logp).all(), "zero-offset anchor produced a non-finite logp"


def test_sample_offsets_refuses_group_of_one():
    from tanitad.rl.refcv3_adapter import sample_offsets
    with pytest.raises(ValueError, match="group_size"):
        sample_offsets(torch.randn(1, 2, 4, 2), PostTrainConfig(group_size=1))


def test_end_to_end_posttrain_step_on_the_real_model(model, tmp_path):
    """⭐ The whole chain: real refcv3 -> sampler -> reward -> advantage -> step."""
    from tanitad.rl.posttrain import run_posttrain
    from tanitad.rl.refcv3_adapter import gt_context, make_refcv3_sample_fn

    cfg_core = model.cfg.core
    enc = cfg_core.encoder
    frames = torch.zeros(1, cfg_core.window, enc.in_channels, enc.image_size,
                         enc.image_width or enc.image_size)
    n_steps = len(cfg_core.trajectory.horizons)
    t = torch.arange(n_steps, dtype=torch.float32) * 0.1
    batch = {"frames": frames, "v0": torch.zeros(1),
             "gt_traj": torch.stack([10.0 * t, torch.zeros_like(t)], dim=-1),
             "obstacles": torch.tensor([[40.0, 0.0]])}

    cfg = PostTrainConfig(method="grpo", group_size=3, steps=2, lr=1e-4,
                          out_dir=str(tmp_path))
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if n.startswith("core.encoder")}

    sample_fn = make_refcv3_sample_fn(model, cfg, build_ctx=gt_context)
    # ⚠️ real batches must be PASSED — with `batches=None` the loop iterates
    # `range(steps)` and hands the adapter integers. Caught here, which is the
    # point of running the adapter end-to-end rather than unit-testing it alone.
    summary = run_posttrain(model, sample_fn, cfg, batches=[batch, batch])

    assert summary["done"] is True
    for k in ("steps", "windows_scored", "samples_scored", "optimizer_steps"):
        assert summary["counters"][k] > 0, k
    # the trunk did not move, on the REAL model
    for n, p in model.named_parameters():
        if n in before:
            assert torch.allclose(before[n], p), f"frozen trunk parameter {n} MOVED"
