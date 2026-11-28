"""Truncated BPTT on the token-field rollouts — the gradient path is BOUNDED.

⛔ WHY. Until 2026-09-02 `TokenFieldPredictor.rollout` applied the predictor K
times with no detach: a **30-deep** back-prop chain through shared parameters at
the live config. MEASURED on the first refav1 launch: gnorm **3.6e3** at step 50,
**5.7e7** at 150, **inf** at 300.

The Research Lab's ASK-1 pass found **no published recipe in our reference class
back-propagates that far**, and four independent ones all avoid it — TD-MPC2
(H=3 + terminal value), DreamerV3 (H=15 + AGC), Looped-WM (truncate at
ceil(mu_rec/2)), InfinityDrive (window curriculum). None lowers the clip.

⭐ THE PROPERTY THAT MATTERS, and it is why "it runs" is not evidence:
truncation must change the GRADIENT and leave the FORWARD pass untouched. A
"truncation" that altered predictions would be a different model, not a bounded
one.
"""
from __future__ import annotations

import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config, TokenFieldPredictor

K = 6


def _pred(trunc: int, seed: int = 0) -> TokenFieldPredictor:
    cfg = RefAV1Config()
    cfg.d_state, cfg.n_tokens = 16, 4
    torch.manual_seed(seed)
    p = TokenFieldPredictor(cfg, cfg.d_state, 1, 2, None)
    p.bptt_truncate = trunc
    return p


def test_full_chain_reaches_the_input():
    """The deliberate-regression control. Without it a PASS below means nothing:
    a rollout that never propagated to the input would 'pass' truncation
    trivially."""
    p = _pred(0)
    f = torch.randn(2, 4, 16, requires_grad=True)
    p.rollout(f, torch.randn(2, K, RefAV1Config().a_dim))[:, -1].sum().backward()
    assert f.grad is not None and float(f.grad.abs().sum()) > 1e-6


def test_truncation_cuts_the_chain_to_the_input():
    """MEASURED: 127.998 -> 0 at trunc=2, K=6."""
    p = _pred(2)
    f = torch.randn(2, 4, 16, requires_grad=True)
    p.rollout(f, torch.randn(2, K, RefAV1Config().a_dim))[:, -1].sum().backward()
    g = 0.0 if f.grad is None else float(f.grad.abs().sum())
    assert g == 0.0, f"gradient {g} still reaches the input past the cut"


def test_the_forward_pass_is_UNCHANGED_by_truncation():
    """⭐ Truncation bounds gradients; it must not alter a single prediction.
    Every step still sees the true previous state — only the carried state is
    detached, and the appended OUTPUT keeps its own path."""
    f = torch.randn(2, 4, 16)
    a = torch.randn(2, K, RefAV1Config().a_dim)
    with torch.no_grad():
        d = float((_pred(0).rollout(f, a) - _pred(2).rollout(f, a)).abs().max())
    assert d == 0.0, f"truncation changed the forward pass by {d}"


def test_last_only_path_truncates_too():
    """The planning path (`last_only=True`) is a separate loop and was fixed
    with the same cut — a fix applied to one of two loops is half a fix."""
    p = _pred(2)
    f = torch.randn(2, 4, 16, requires_grad=True)
    out = p.rollout(f, torch.randn(2, K, RefAV1Config().a_dim), last_only=True)
    # ⛔ the RETURNED tensor must still carry a gradient path — the first
    # implementation detached it whenever K was a multiple of `trunc`, handing
    # the planning path a gradient-free result while every other test passed
    assert out.requires_grad, "last_only returned a fully DETACHED field"
    out.sum().backward()
    g = 0.0 if f.grad is None else float(f.grad.abs().sum())
    assert g == 0.0, f"last_only path still back-propagates {g} to the input"


def test_the_default_is_the_literature_value_and_reaches_both_predictors():
    """15 = DreamerV3's imagination horizon AND Looped-WM's ceil(mu_rec/2) at
    our K=30 — two independent recipes agreeing. ⛔ `tactical` is the SAME class
    rolling its own chain; fixing only the one that happened to blow up would
    leave the identical defect next door."""
    assert RefAV1Config().bptt_truncate == 15
    m = RefAV1(RefAV1Config(strategic_cfg=None, tactical_cfg=None))
    assert m.operative.bptt_truncate == 15
    assert m.tactical.bptt_truncate == 15
