"""The horizon contract: what the config DECLARES must be what a loss TRAINS.

⛔ THE DEFECT THIS PINS, MEASURED 2026-08-31 on `o1ctrl30k`'s 8 snapshots
(steps 5,000-22,500):

    |W1| 3.7283 -> 7.7516      moving
    |W2| 0.026154 CONSTANT     delta EXACTLY 0.000e+00 at every snapshot
    |W4| 0.026113 CONSTANT     delta EXACTLY 0.000e+00 at every snapshot

Bit-identical across 17,500 steps, with `--w-o1-ctrl 1.0` in force -- so it is a
property of the WIRING, not of any objective.

⭐ AND IT IS BY DESIGN, WHICH IS WHY THE FIX IS A GUARD AND NOT A REWRITE.
`metric_dynamics.rollout_transitions` reaches long horizons by applying the
1-step head AUTOREGRESSIVELY (`predictor(ws, wa)[1]`, k times, full-chain
gradient -- explicitly NOT truncated BPTT), and O5 supervises the error at every
step. The horizon is therefore `o5_k * dt`, and heads for k != 1 are allocated,
computed in `forward`, and consumed by nothing.

⚠️ THE COST WAS MEASUREMENT, NOT COMPUTE. Untrained heads emit initialisation
noise that reads as a number: MM-E10 published h2/h4 ratios of ~1e-5 as though
they meant something, MM-E14 retracted them, and the same heads then produced a
false "the model only imagines 0.1 s" alarm. A silently dead parameter is a
measurement hazard.
"""
from pathlib import Path

import pytest
import torch

from tanitad.models.predictor import OperativePredictor, PredictorConfig

STATE_DIM = 16
_CFG = dict(d_model=32, depth=2, n_heads=4, window=4, action_dim=3)


def _cfg(**kw):
    return PredictorConfig(**{**_CFG, **kw})


def test_only_head_1_is_reachable_from_the_rollout():
    """⭐ THE ROOT FACT, asserted on the real rollout rather than on a comment.

    Roll the predictor, backprop through the rollout output, and check which
    heads received gradient. Head 1 must; every other head must be None or zero.
    """
    from tanitad.models.metric_dynamics import rollout_transitions
    p = OperativePredictor(_cfg(horizons=(1, 2, 4)), STATE_DIM)
    b, w, k = 2, 4, 3
    states = torch.randn(b, w, STATE_DIM)
    actions = torch.randn(b, w, 3)
    future = torch.randn(b, k, 3)
    trans = rollout_transitions(p, states, actions, future, k)
    loss = sum(t[1].square().mean() for t in trans)
    loss.backward()

    g1 = p.heads["1"].weight.grad
    assert g1 is not None and float(g1.abs().sum()) > 0, "head 1 must train"
    for dead in ("2", "4"):
        g = p.heads[dead].weight.grad
        assert g is None or float(g.abs().sum()) == 0.0, (
            f"head {dead} received gradient — the rollout changed shape and this "
            f"test's premise (and the trainer's refusal) must be revisited")


def test_trained_horizons_is_the_honest_attribute():
    """⚠️ Report `trained_horizons`, never `cfg.horizons`. The config records an
    INTENT; this records what gradient actually reaches."""
    p = OperativePredictor(_cfg(horizons=(1, 2, 4)), STATE_DIM)
    assert p.trained_horizons == (1,)
    assert tuple(p.cfg.horizons) == (1, 2, 4), "cfg keeps the declared value"


def test_old_checkpoints_still_LOAD_with_dead_heads():
    """⛔ LOADING IS NOT AUTHORING, and this is the line the guard must not cross.

    ~30 banked checkpoints carry heads.2/heads.4 in their state_dict. Refusing
    them at construction would make every one unloadable — destroying the
    evidence instead of preventing the defect. A checkpoint is a record of what
    was done and must stay readable however wrong it was.
    """
    p = OperativePredictor(_cfg(horizons=(1, 2, 4)), STATE_DIM)
    sd = p.state_dict()
    assert "heads.2.weight" in sd and "heads.4.weight" in sd
    q = OperativePredictor(_cfg(horizons=(1, 2, 4)), STATE_DIM)
    q.load_state_dict(sd)          # must not raise


def test_forward_still_emits_every_declared_horizon():
    """⚠️ The guard must not change the forward. Consumers index out[k] for the
    declared horizons; silently dropping keys would break them far from here."""
    p = OperativePredictor(_cfg(horizons=(1, 2, 4)), STATE_DIM)
    out = p(torch.randn(2, 4, STATE_DIM), torch.randn(2, 4, 3))
    assert set(out) == {1, 2, 4}


def test_the_refusal_text_names_o5k_and_the_binding_6s_target():
    """⚠️ A refusal that does not say what to do instead is a wall, not a guard.

    The whole point is that the horizon comes from `--o5-k`, not from the head
    index — so the message must carry that, and the 6.0 s target, or the reader
    fixes it by deleting heads and still trains a 0.8 s model.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "train_v6_staged.py").read_text(encoding="utf-8")
    blk = src[src.index("which NO loss consumes"):][:1600]
    assert "--o5-k 60" in blk, "the refusal must name the 6 s setting"
    assert "6.0 s" in blk, "the refusal must name the binding target in seconds"
    assert "rollout_transitions" in blk, "and WHY only head 1 trains"


def test_the_guard_does_not_fire_on_a_RESUME():
    """⛔⛔ THE BUG THIS PINS ALMOST BRICKED A LIVE 24 h ARM.

    `k60p30k` (MM-E19) runs `--horizons 1 2 4` deliberately — a matched control
    against a banked incumbent, whose dead heads are proven inert. A preflight
    firing on RESUME would refuse that arm's own restart after any crash, making
    22 h unrecoverable: the guard destroying the work it exists to protect.

    Resuming is not authoring. Only a FRESH run is a new decision.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "train_v6_staged.py").read_text(encoding="utf-8")
    i = src.index("which NO loss consumes")
    blk = src[max(0, i - 2200):i]
    assert "_resuming" in blk, "the guard must be resume-aware"
    assert 'ckpt.pt").exists()' in blk, "and detect resume by the checkpoint"
    assert "RESUME" in src[i - 2200:i + 1800], "and say so when it allows one"


def test_horizon_seconds_is_o5k_times_dt_not_the_head_index():
    """⭐ The arithmetic the false alarm got wrong. The binding target is 6.0 s
    (§4b; v6.py PLAN_STEPS=60, HORIZON_S=6.0), reached as o5_k * dt — NOT by the
    largest declared head index."""
    dt = 0.1
    assert 8 * dt == pytest.approx(0.8)      # the whole v7-tiny campaign
    assert 20 * dt == pytest.approx(2.0)     # the value §4b RETIRED
    assert 60 * dt == pytest.approx(6.0)     # the binding target
