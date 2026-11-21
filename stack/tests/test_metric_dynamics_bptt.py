"""Truncated BPTT on `metric_dynamics.rollout_transitions` — the v7 trainer's
k-step rollout gets the SAME bounded gradient path refav1's token-field rollout
got on 2026-09-02, pinned by the SAME five properties
(`test_refa_v1_bptt_truncation.py`), plus the one that is specific to a
WINDOWED state.

⛔ WHY. `rollout_transitions` applied the 1-step head k times with no detach —
"explicitly NOT truncated BPTT" in its own docstring. MEASURED: the k=60 O5
rollout diverged (gnorm 2.1e9, killed at 9,000; PREREG_MM_E19), the same
shared-parameter-chain failure refav1 hit (gnorm 3.6e3 -> 5.7e7 -> inf within
300 steps) and fixed with `bptt_truncate=15`. The v7 trainer had no such knob.

⭐ THE PROPERTY THAT MATTERS, and why "it runs" is not evidence: truncation must
change the GRADIENT and leave the FORWARD untouched. A "truncation" that altered
a single prediction would be a different model, not a bounded one.

⚠️ THE WINDOWED TRAP (specific to this function, found while writing this
file): the carried state is a SLIDING WINDOW of W latents. Detaching only the
newly appended latent — the literal transcription of refav1's `z_.detach()` —
leaves the W-1 older entries attached, and the chain to the input survives
through them. The last test below is the deliberate-regression control for
exactly that: it implements the naive cut inline and asserts it does NOT cut.
"""
from __future__ import annotations

import torch
from torch import nn

from tanitad.models.metric_dynamics import rollout_transitions

K, W, S, A, B = 6, 3, 8, 3, 2


class _TinyPred(nn.Module):
    """Reads EVERY window entry (not only the newest), so an older attached
    latent would carry gradient — the shape that makes the windowed trap
    observable."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(W * S + A, S)

    def forward(self, ws, wa):
        x = torch.cat([ws.reshape(ws.shape[0], -1), wa[:, -1]], dim=-1)
        return None, torch.tanh(self.lin(x))              # [0] unused, [1] = 1-step head


def _inputs(seed: int = 0, grad: bool = True):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, W, S, generator=g).requires_grad_(grad)
    a = torch.randn(B, W, A, generator=g)
    fa = torch.randn(B, K - 1, A, generator=g)
    return s, a, fa


def _grad_to_input(trunc: int, which: int = -1, ckpt: bool = False):
    p = _TinyPred()
    s, a, fa = _inputs()
    tr = rollout_transitions(p, s, a, fa, K, grad_checkpoint=ckpt,
                             bptt_truncate=trunc)
    tr[which][1].sum().backward()
    g = 0.0 if s.grad is None else float(s.grad.abs().sum())
    return g, tr, p


def test_full_chain_reaches_the_input():
    """The deliberate-regression control. Without it a PASS below means
    nothing: a rollout that never propagated to the input would 'pass'
    truncation trivially."""
    g, _, _ = _grad_to_input(0)
    assert g > 1e-6


def test_truncation_cuts_the_chain_to_the_input():
    """trunc=2 < K=6: the LAST step's error no longer reaches the input."""
    g, _, _ = _grad_to_input(2)
    assert g == 0.0, f"gradient {g} still reaches the input past the cut"


def test_the_forward_pass_is_UNCHANGED_by_truncation():
    """⭐ Truncation bounds gradients; it must not alter a single value —
    neither the recorded z_prev nor the recorded z_hat, at any step."""
    for ckpt in (False, True):
        for grad in (True, False):
            s, a, fa = _inputs(grad=grad)
            p = _TinyPred()
            full = rollout_transitions(p, s, a, fa, K, grad_checkpoint=ckpt)
            cut = rollout_transitions(p, s, a, fa, K, grad_checkpoint=ckpt,
                                      bptt_truncate=2)
            assert len(full) == len(cut) == K
            for (zp0, zh0), (zp1, zh1) in zip(full, cut):
                assert torch.equal(zp0.detach(), zp1.detach())
                assert torch.equal(zh0.detach(), zh1.detach())


def test_every_returned_output_keeps_its_gradient_path():
    """⛔ The RETURNED tensors must still carry a gradient path — including
    when K is a multiple of the cut (K=6 with 2 and 3), which is exactly the
    refav1 `last_only` defect: a fully detached return that every other test
    let through. A readout decoding ANY step must still train the predictor."""
    for trunc in (2, 3):
        _, tr, p = _grad_to_input(trunc)
        assert all(zh.requires_grad for _, zh in tr), \
            f"a returned z_hat is DETACHED at trunc={trunc}"
        assert p.lin.weight.grad is not None and \
            float(p.lin.weight.grad.abs().sum()) > 0.0, \
            "the last step's error no longer trains the predictor"


def test_the_chain_is_bounded_not_severed():
    """The FIRST step always sees the input (no cut can precede it), so its
    error must still reach the input at any truncation; only steps beyond the
    cut are decoupled. Bounded, not blind."""
    g_first, _, _ = _grad_to_input(2, which=0)
    assert g_first > 1e-6
    g_last, _, _ = _grad_to_input(2, which=-1)
    assert g_last == 0.0


def test_a_cut_beyond_k_is_a_no_op_and_matches_the_full_chain_exactly():
    """trunc >= K never fires; the gradient must EQUAL the full-chain one bit
    for bit (the trainer's preflight refuses this as an inert flag — the
    function itself stays honest about it)."""
    g0, _, _ = _grad_to_input(0)
    for trunc in (K, K + 5):
        g, _, _ = _grad_to_input(trunc)
        assert g == g0


def test_truncation_composes_with_grad_checkpoint():
    """The k=60 run NEEDS checkpointing (a measured 37.97/44 GiB OOM) — the
    cut must hold under it too, and the forward stays identical (covered
    above); here: the gradient properties."""
    g_full, _, _ = _grad_to_input(0, ckpt=True)
    g_cut, _, _ = _grad_to_input(2, ckpt=True)
    assert g_full > 1e-6 and g_cut == 0.0


def test_negative_truncation_is_refused():
    p = _TinyPred()
    s, a, fa = _inputs()
    try:
        rollout_transitions(p, s, a, fa, K, bptt_truncate=-1)
    except ValueError:
        return
    raise AssertionError("a negative bptt_truncate was accepted")


def test_NEGATIVE_CONTROL_detaching_only_the_newest_latent_would_NOT_cut():
    """⚠️ The windowed trap, made falsifiable. The naive transcription of
    refav1's fix — detach only the appended z_hat — leaves the W-1 older window
    entries attached, so the input still reaches the last step. This control
    MUST read non-zero; if it ever reads zero the test above proves nothing
    (the predictor stopped reading the older entries)."""
    p = _TinyPred()
    s, a, fa = _inputs()
    win_s, win_a = s, a
    z_hat = None
    for j in range(K):
        z_hat = p(win_s, win_a)[1]
        if j < K - 1:
            carry = z_hat.detach() if (j + 1) % 2 == 0 else z_hat   # NAIVE cut
            win_s = torch.cat([win_s[:, 1:], carry.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], fa[:, j].unsqueeze(1)], dim=1)
    z_hat.sum().backward()
    g = 0.0 if s.grad is None else float(s.grad.abs().sum())
    assert g > 1e-6, "the naive cut severed the chain — the control is broken"
