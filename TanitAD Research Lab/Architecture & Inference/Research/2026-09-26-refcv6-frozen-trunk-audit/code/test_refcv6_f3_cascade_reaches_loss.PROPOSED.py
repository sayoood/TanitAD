"""⛔ refcv6 F3: the per-stage (cascade) loss must REACH the trainer on the PRODUCTION forward.

PROPOSED for `stack/tests/` by the A16 audit (2026-09-26,
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv6-frozen-trunk-audit/`).

MEASURED on the live run `refcv6-r101-s0`: 0 of 668 training rows and 0 of 66 in-run eval rows
carry `cascade`, and the stage-0..2 heads and AdaLN are bit-identical at steps 1,000 / 5,000 /
30,000. `AnchoredDiffusionDecoder` exports `layer_u0_hat` / `layer_logits`, but
`RefCModel.forward` copies decoder outputs through a WHITELIST that omitted them, so
`compute_losses_v3`'s `if f3_per_layer and "layer_u0_hat" in out` was always false. Every existing
F3 test (`test_refcv6_diffusion.py`) calls the DECODER directly, and the programme's gradient-reach
census (`test_built_heads_receive_gradient.py`) classifies TOP-LEVEL children, where `core` reads
GRADIENT_REACHES whatever its sub-modules do. This file asserts on the CONSUMER, at LEAF level.

Expectations are LITERALS (the leaf names), never derived from the code under test. The
deliberate-regression arm re-introduces the historical defect exactly (the two keys removed from
the production forward's output) and must be REFUSED.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(os.environ.get("A16_STACK_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

F_ON = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
        "--f5-emitting-conf", "--f6-w-u0-zero"]
ARGV = ["--arm", "hier", "--smoke", "--device", "cpu", "--batch", "2", "--sampler", "ddim",
        "--anchor-v0-conditioned", "--out", "unused"] + F_ON

#: the smoke decoder has TWO layers => stage 0 is supervised ONLY by the per-stage loss (stage 1
#: is the emitted fan). Written as literals on purpose.
STAGE0_LEAVES = ("core.decoder.cascade.control_heads.0",
                 "core.decoder.cascade.conf_heads.0",
                 "core.decoder.adaln.0.scale_shift_mlp.1")


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_for_f3reach", str(ROOT / "scripts" / "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_and_loss(T, drop_keys: bool = False):
    from tanitad.refs import refc
    args = T.build_parser().parse_args(ARGV)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    # a real control ladder (the decoder refuses an exactly degenerate v0-conditioned bank)
    ac = model.core.decoder.anchor_controls
    n = int(ac.shape[0])
    with torch.no_grad():
        ac.copy_(torch.stack([torch.linspace(-2.0, 2.0, n),
                              torch.linspace(-1.5, 1.5, n)], dim=-1).to(ac.dtype))
    model._w_goal_point = 0.0
    model._w_tac_goal = 0.0
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    orig = refc.RefCModel.forward

    def fwd_drop(self, *a, **k):              # the HISTORICAL defect, re-introduced exactly
        o = orig(self, *a, **k)
        o.pop("layer_u0_hat", None)
        o.pop("layer_logits", None)
        return o
    if drop_keys:
        refc.RefCModel.forward = fwd_drop
    try:
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    finally:
        refc.RefCModel.forward = orig
    return model, losses


def test_the_F3_per_stage_loss_is_IN_the_trainers_loss_dict():
    T = _trainer()
    _m, losses = _build_and_loss(T)
    assert "cascade" in losses, (
        "F3 is built but compute_losses_v3 returned no `cascade` term: the per-stage loss was "
        "skipped (the refcv6-r101-s0 defect, A16 2026-09-26)")
    assert math.isfinite(float(losses["cascade"])) and float(losses["cascade"]) > 0.0


def test_stage0_heads_and_adaln_RECEIVE_gradient_at_leaf_level():
    T = _trainer()
    model, losses = _build_and_loss(T)
    losses["loss"].backward()
    mods = dict(model.named_modules())
    dead = [nm for nm in STAGE0_LEAVES
            if all(p.grad is None for p in mods[nm].parameters(recurse=False))]
    assert not dead, f"stage-0 leaves took NO gradient (never in the graph): {dead}"


def test_DELIBERATE_REGRESSION_dropping_the_keys_is_REFUSED_not_skipped():
    T = _trainer()
    with pytest.raises(SystemExit, match="layer_u0_hat"):
        _build_and_loss(T, drop_keys=True)
