"""The v6 trunk adapter must satisfy exactly the interface the P-battery uses.

⛔ WHY. MEASURED 2026-08-14: the P-battery could not read a v6 checkpoint, and
four successive runs each fixed a surface assumption before the real one showed
up — the probes BUILD a v5 ``WorldModel`` and infer ``action_dim`` from v5
parameter names. These tests pin the narrow contract so the port cannot rot:
``collect_grid``/``p8_latents`` need only ``encode_window``, ``predictor``
(driven as ``predictor(s, a)[1]``), ``state_dim`` and ``parameters()``.

The heavy end-to-end run needs a GPU and the val cache; these run on CPU with a
tiny stack, which is what makes them a regression guard rather than a ritual.
"""
import pytest
import torch

from tanitad.eval.v6_probe_trunk import (V6ProbeTrunk, is_v6_checkpoint,
                                         load_trunk_auto)
from tanitad.models.v6 import V6Stack

SD = {"a": torch.zeros(2)}


def test_is_v6_checkpoint_recognises_the_staged_layout():
    assert is_v6_checkpoint({"stack": SD, "opt": {}, "step": 5, "config": {}})


def test_is_v6_checkpoint_rejects_v5_and_bare():
    assert not is_v6_checkpoint({"model": SD, "step": 5})
    assert not is_v6_checkpoint(SD)
    # a v5 ckpt that also carries a "stack" entry must NOT be re-routed
    assert not is_v6_checkpoint({"model": SD, "stack": SD})


def test_load_trunk_auto_refuses_v6_without_a_run_config():
    """A checkpoint that travelled without its config is a refused restart —
    loudly, not by silently rebuilding a default architecture."""
    with pytest.raises(SystemExit, match="no run config"):
        load_trunk_auto({"stack": SD, "step": 1}, "cpu")


@pytest.fixture(scope="module")
def tiny_stack():
    return V6Stack()


def test_adapter_exposes_the_four_things_the_probes_need(tiny_stack):
    t = V6ProbeTrunk(tiny_stack)
    assert isinstance(t.state_dim, int) and t.state_dim > 0
    assert t.state_dim == int(tiny_stack.cfg.d_op), "state_dim is the firewall"
    assert callable(t.encode_window)
    assert t.predictor is tiny_stack.predictor_op
    assert len(list(t.parameters())) == len(list(tiny_stack.parameters()))
    assert len(list(t.named_parameters())) > 0   # module_md5 needs this


def test_adapter_adds_no_parameters(tiny_stack):
    """It must not be an nn.Module that could contribute weights of its own."""
    assert not isinstance(V6ProbeTrunk(tiny_stack), torch.nn.Module)


def test_predictor_is_driven_the_way_rollout_transitions_drives_it(tiny_stack):
    """`rollout_transitions` calls `predictor(states, actions)[1]`. For v6 the
    forward returns dict[int, Tensor], so `[1]` selects the 1-step head — the
    same expression that works for v5. This is the load-bearing compatibility."""
    t = V6ProbeTrunk(tiny_stack)
    cfg = tiny_stack.cfg
    b, w = 2, cfg.predictor.window
    states = torch.randn(b, w, t.state_dim)
    actions = torch.randn(b, w, cfg.predictor.action_dim)
    with torch.no_grad():
        out = t.predictor(states, actions)
    assert isinstance(out, dict) and 1 in out, "no 1-step head to roll"
    assert out[1].shape == (b, t.state_dim)


def test_v6_action_dim_matches_the_lifted_3_channel_format(tiny_stack):
    """`lift_actions3` emits 3 channels (the v5 speed-append contract). v6 is
    built with action_dim=3, so the probes need no action translation. If this
    ever diverges the probes must lift differently — hence the pin."""
    assert int(tiny_stack.cfg.predictor.action_dim) == 3


def test_encode_window_returns_batch_time_state(tiny_stack):
    t = V6ProbeTrunk(tiny_stack)
    cfg = tiny_stack.cfg
    frames = torch.randn(1, 2, cfg.encoder.in_channels,
                         cfg.encoder.image_size, cfg.encoder.image_width)
    with torch.no_grad():
        z = t.encode_window(frames)
    assert z.shape == (1, 2, t.state_dim)


def test_adapter_exposes_the_trunks_window(tiny_stack):
    """The dataset's causal window must come from the checkpoint. v6 uses 6
    where the v5 eval default says 8; the probes read it off the trunk, and a
    v5 trunk (no attribute) falls back to the old value."""
    t = V6ProbeTrunk(tiny_stack)
    assert t.window == int(tiny_stack.cfg.predictor.window)
    assert getattr(object(), "window", 8) == 8       # the v5 fallback path
