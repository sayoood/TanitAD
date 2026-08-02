"""P1 validation (deep review 2026-08-02): imagination conditioning is FEEDABLE, end to end.

The review found ``cond_imagination`` — the conditioning ``flagship_v15`` itself calls
"THE NOVEL PART" — hard-wired False in the trainer since ``real_smoke``, so the v5 planner
trained with ZERO imagination tokens. The fix exposes ``--cond-imagination`` and feeds
``imagine_probes`` output at both head call sites. These tests hold the three seams:

1. the imagination path runs END TO END on the real head (tiny dims, CPU) and delivers
   exactly ``n_probes * len(imag_read)`` tokens — not zero, not a fallback;
2. the helper is FAIL-LOUD: flag on + no probe vocabulary must raise, never silently skip
   (a head that expects 32 tokens and sees none is the bug this whole change removes);
3. the trainer wiring is real — the flag exists and the main head site reads it (the two
   deliberate smoke/proof hard-wires stay, and their COUNT is pinned so a third one cannot
   creep back in unnoticed).
"""
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.train_flagship_v4 import (_imagination_inputs, _smoke_head_cfg)
from tanitad.models.flagship_v4 import FlagshipV4Head
from tanitad.models.flagship_v15 import imagine_probes

STATE_DIM, WINDOW, B = 64, 4, 2


class _StubPredictor(nn.Module):
    """Deterministic 1-step head with the predictor's calling contract:
    ``predictor(win_states, win_actions) -> (_, z_next)``."""

    def __init__(self, s=STATE_DIM):
        super().__init__()
        self.lin = nn.Linear(s + 3, s)

    def forward(self, ws, wa):
        z = self.lin(torch.cat([ws[:, -1], wa[:, -1]], dim=-1))
        return None, z


def _imag_cfg():
    cfg = _smoke_head_cfg(STATE_DIM, WINDOW)
    cfg.cond_imagination = True
    cfg.n_probes = 4
    cfg.probe_steps = 6
    cfg.imag_read = (2, 4, 6)
    return cfg


def test_imagination_tokens_flow_end_to_end():
    torch.manual_seed(0)
    cfg = _imag_cfg()
    head = FlagshipV4Head(cfg)
    assert head.n_imag_tokens == cfg.n_probes * len(cfg.imag_read) > 0, \
        "cond_imagination=True must buy a non-zero imagination token budget"

    pred = _StubPredictor()
    states = torch.randn(B, WINDOW, STATE_DIM)
    actions = torch.randn(B, WINDOW, 3)
    probes = torch.randn(cfg.n_probes, cfg.probe_steps, 2)
    v0n = torch.rand(B)

    imag = imagine_probes(pred, states, actions, probes, cfg.imag_read, v0n)
    assert imag.shape == (B, cfg.n_probes * len(cfg.imag_read), STATE_DIM)

    out = head(states, v0n * 10.0, imagined=imag)
    assert torch.isfinite(out["traj"]).all()
    assert out["traj"].shape[0] == B

    # the tokens must MATTER: a different imagination must change the decode.
    out2 = head(states, v0n * 10.0, imagined=imag + 1.0)
    assert not torch.allclose(out["traj"], out2["traj"]), \
        "imagined tokens changed but the decode did not — the conditioning is dead"


def test_head_refuses_missing_imagination():
    """flagship_v15's own contract: cond_imagination on + no imagined -> ValueError."""
    cfg = _imag_cfg()
    head = FlagshipV4Head(cfg)
    with pytest.raises(ValueError):
        head(torch.randn(B, WINDOW, STATE_DIM), torch.rand(B), imagined=None)


def test_imagination_inputs_helper_contract():
    cfg = _imag_cfg()

    class _W(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = _StubPredictor()

    world = _W()
    batch = {"pose_last": torch.zeros(B, 4).index_fill_(1, torch.tensor(3), 8.0),
             "actions": torch.randn(B, WINDOW, 2)}          # 2-ch: helper lifts to 3
    states = torch.randn(B, WINDOW, STATE_DIM)

    # off -> exactly {}
    cfg_off = _smoke_head_cfg(STATE_DIM, WINDOW)
    assert _imagination_inputs(world, cfg_off, batch, states, None) == {}

    # on + no probes -> fail LOUD, never a silent skip
    with pytest.raises(RuntimeError, match="probe vocabulary"):
        _imagination_inputs(world, cfg, batch, states, None)

    # on + probes -> the imagined tensor, right shape, no grad into the predictor
    probes = torch.randn(cfg.n_probes, cfg.probe_steps, 2)
    kw = _imagination_inputs(world, cfg, batch, states.requires_grad_(True), probes)
    assert set(kw) == {"imagined"}
    assert kw["imagined"].shape == (B, cfg.n_probes * len(cfg.imag_read), STATE_DIM)
    assert not kw["imagined"].requires_grad, \
        "imagination must be a no-grad INPUT, not a 20-step backprop path"


def test_trainer_wiring_is_real_and_hardwires_are_pinned():
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "train_flagship_v4.py").read_text(encoding="utf-8")
    assert '"--cond-imagination"' in src, "the launch flag vanished"
    assert 'bool(getattr(a, "cond_imagination", False))' in src, \
        "the MAIN head site no longer reads the flag"
    # the two DELIBERATE off-sites (real_smoke proof + _smoke_head_cfg) stay; a third
    # hard-wire would silently re-create the v5 defect. Pin the count.
    assert src.count("cond_imagination = False") == 2, (
        "unexpected number of hard-wired cond_imagination=False sites — if this is a "
        "deliberate new smoke path, update this pin with its justification")
