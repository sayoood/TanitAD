"""⛔⛔ PI RULING 2026-09-26 (SPEC_REFCV7 §7 / A2, D-REFCV7-E1): the speed-ceiling argmax filter
is INFERENCE-ONLY.

MEASURED from source before the ruling: the filter (`refc.py`, `SpeedCeilingFilter` masking `rank`
before `idx = rank.argmax`) had no `self.training` gate, so in TRAINING it changed which candidate
is selected -- and the selected `traj` feeds `law_pred = law_head(cat(pooled, traj))`, whose MSE is
a training loss. A filter driven by an EGO-FUTURE channel was shaping the LAW head's input.

Pinned here on the REAL `RefCModel.forward` (smoke rig, CPU), with a CRAFTED ceiling that excludes
the top-ranked candidate of each row that has a slower alternative:

* EVAL: the ceiling bites -- the pick moves off the excluded winner onto a compliant candidate;
* TRAINING: the ceiling is inert -- same seed, same pick, same fan, and the LAW head's input is the
  UNMASKED selection;
* RED ARM: the pre-ruling behaviour (`speed_ceiling_in_training = True`) makes the training
  assertion fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tanitad.refs import refc  # noqa: E402
from tanitad.refs.refc_agents import AgentSeamConfig  # noqa: E402
from tanitad.refs.refcv6_selection import planned_max_speed  # noqa: E402

B = 8
SEED = 7


def _model():
    cfg = refc.refc_smoke_config()
    cfg.agents = AgentSeamConfig(enable=True)      # the scene hook needs agent tokens
    cfg.decoder.cross_agent = True
    cfg.speed_ceiling_filter = True
    torch.manual_seed(0)
    return refc.RefCModel(cfg), cfg


def _inputs(cfg):
    g = torch.Generator().manual_seed(1)
    frames = torch.rand(B, cfg.window, 1, 64, 64, generator=g)
    v0 = torch.linspace(4.0, 18.0, B)
    return frames, v0


def _run(model, frames, v0, v_lim):
    """One forward through the REAL model with the ceiling fed by a scene hook, capturing the
    LAW head's input. Same seed on every call, so train-mode randomness is identical."""
    captured = {}

    def pre(_mod, args):
        captured["x"] = args[0].detach().clone()
    h = model.law_head.register_forward_pre_hook(pre)
    try:
        torch.manual_seed(SEED)
        with torch.no_grad():
            out = model(frames, v0=v0, steps=0,
                        scene_hook=lambda *a, **k: {"v_limit_ms": v_lim})
    finally:
        h.remove()
    return out, captured["x"]


def _craft(model, out):
    """Per row: the next-slower candidate's planned max speed, so the top-ranked candidate is
    excluded and at least one survivor remains; +inf (no bite) where the winner is the slowest."""
    dec = model.decoder
    x, idx = out["anchor_traj"], out["sel_idx"]
    vmax = planned_max_speed(x, horizons=dec.anchor_horizons, tick_s=dec.anchor_dt)   # [B, N]
    ar = torch.arange(x.shape[0])
    win = vmax[ar, idx]
    slower = torch.where(vmax < win[:, None] - 1e-3, vmax, torch.full_like(vmax, -1.0))
    nxt = slower.max(dim=1).values
    bite = nxt >= 0.0
    v_lim = torch.where(bite, nxt, torch.full_like(nxt, float("inf")))
    return v_lim, bite, vmax


def _assert_bites_in_eval(model, frames, v0):
    model.eval()
    ref, _ = _run(model, frames, v0, torch.full((B,), float("inf")))
    v_lim, bite, vmax = _craft(model, ref)
    assert int(bite.sum()) >= 2, "the crafted batch has no row to bite -- the test proves nothing"
    got, _ = _run(model, frames, v0, v_lim)
    ar = torch.arange(B)
    moved = got["sel_idx"] != ref["sel_idx"]
    assert bool(moved[bite].all()), "at EVAL the ceiling must move the pick off the excluded winner"
    assert bool((vmax[ar, got["sel_idx"]][bite] <= v_lim[bite] + 1e-6).all())
    assert bool((~moved[~bite]).all()), "rows with no bite must keep their pick"


def _assert_inert_in_training(model, frames, v0):
    model.train()
    ref, law_ref = _run(model, frames, v0, torch.full((B,), float("inf")))
    v_lim, bite, _ = _craft(model, ref)
    assert int(bite.sum()) >= 2, "the crafted batch has no row to bite -- the test proves nothing"
    got, law_got = _run(model, frames, v0, v_lim)
    assert torch.equal(got["anchor_traj"], ref["anchor_traj"]), "same seed must give the same fan"
    assert torch.equal(got["sel_idx"], ref["sel_idx"]), \
        "in TRAINING the ceiling must not change the pick (PI 2026-09-26 A2)"
    ar = torch.arange(B)
    unmasked = ref["anchor_traj"][ar, ref["sel_idx"]].reshape(B, -1)
    n = unmasked.shape[1]
    assert torch.equal(law_got[:, -n:], unmasked), \
        "the LAW head's input in training must be the UNMASKED selection"
    assert torch.equal(law_got, law_ref)


def test_the_ceiling_BITES_at_eval():
    model, cfg = _model()
    _assert_bites_in_eval(model, *_inputs(cfg))


def test_the_ceiling_is_INERT_in_training_and_the_LAW_input_is_unmasked():
    model, cfg = _model()
    _assert_inert_in_training(model, *_inputs(cfg))


def test_RED_ARM_the_pre_ruling_behaviour_goes_RED(monkeypatch):
    """The mask active in training again (the class switch the ruling turned off): the
    training-inertness assertion must FAIL."""
    model, cfg = _model()
    monkeypatch.setattr(refc.AnchoredDiffusionDecoder, "speed_ceiling_in_training", True)
    with pytest.raises(AssertionError, match="must not change the pick"):
        _assert_inert_in_training(model, *_inputs(cfg))


def test_the_default_is_the_ruling():
    assert refc.AnchoredDiffusionDecoder.speed_ceiling_in_training is False
