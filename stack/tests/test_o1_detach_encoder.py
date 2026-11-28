"""H-RANK-22 — O1 confined to the predictor: does the flag actually cut the
gradient path it claims to cut?

WHY THIS EXISTS.  MEASURED 2026-08-23 (`h_rank18_readout.json`), on matched 2k
arms over the same clips:

    arm         terms          participation   action-divergence (x100 actions)
    lewm        o5+o6                   4.43                              0.000
    lewm_o1     o5+o6+O1                2.94                            516.573
    fixed       all six                 2.94                            474.908

Adding O1 to the collapse-free two-term recipe restored action-sensitivity from
*exactly zero* and simultaneously gave back the ENTIRE rank gain -- landing on
the six-term arm to three significant figures.  So O1 is the single term that
both buys action-conditioning and causes the collapse (H-RANK-18 REFUTED: the
two are coupled, not separable by adding O1 at full weight).

THE MECHANISM HYPOTHESIS this flag tests: O1's gradient reaches the ENCODER, and
the cheapest way for the encoder to satisfy "different actions must give
different rollouts" is to spend scene variance on action-aligned directions --
which is a participation drop by construction.  If that is the mechanism, then
detaching the encoder for the O1 term ONLY should give action-conditioning at no
cost in rank.

⛔ The flag is only worth running if it does what it says, so this file asserts
the gradient topology rather than the flag's presence:

  1. default OFF -- the incumbent loss must be bit-identical for every old run
  2. flag ON  => the encoder receives EXACTLY ZERO gradient from O1
  3. NEGATIVE CONTROL: flag ON => the PREDICTOR still receives gradient from O1
     (otherwise the flag has merely switched O1 off, which would "fix" the rank
     for the trivial reason and reproduce the `lewm` arm)
  4. NEGATIVE CONTROL: flag OFF => the encoder DOES receive gradient from O1
     (otherwise tests 2 is vacuous -- it would pass on a stack where O1 never
     touched the encoder in the first place)

Tests 3 and 4 are the ones that make 2 mean anything; C2026-08-22 cost four
false probe results that were caught ONLY by a control reading a known value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_v6_staged import stack, tiny_cfg  # noqa: F401,E402  (shared fixtures)
from train_v6_staged import V6LossWeights, v6_loss_step  # noqa: E402
from test_v6_staged import synthetic_train_batch  # noqa: E402


def _o1_only(detach: bool) -> V6LossWeights:
    """Every term except O1 at zero, so any gradient observed IS O1's."""
    return V6LossWeights(
        o1_ctrl=1.0, o1_fact=1.0, o1_scene=0.3,
        o2_nearfield=0.0, o3_masked=0.0, o5_rollout=0.0, o6_sigreg=0.0,
        t1_latent=0.0, s1_latent=0.0, lambda_plan=0.0, seam_op=0.0,
        o1_detach_encoder=detach)


def _grad_sums(stack_, detach: bool) -> tuple[float, float]:
    """(encoder grad mass, predictor grad mass) from an O1-only backward."""
    for p in stack_.parameters():
        p.grad = None
    torch.manual_seed(7)
    b = synthetic_train_batch(stack_, batch=2, k=12, seed=3)
    b["gt_wp"] = torch.randn(2, 10, 2)
    L = v6_loss_step(stack_, b, stage="S-W", weights=_o1_only(detach),
                     o1_k=10, o5_k=12)
    assert torch.isfinite(L["loss"]), "O1-only loss is not finite"
    L["loss"].backward()

    enc = pred = 0.0
    for n, p in stack_.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.abs().sum())
        if n.startswith("encoder.") or n.startswith("readout."):
            enc += g
        elif n.startswith("predictor_op.") or n.startswith("step_readout_op."):
            pred += g
    return enc, pred


def test_default_is_off_so_every_old_run_is_bit_identical():
    assert V6LossWeights().o1_detach_encoder is False, (
        "the default must stay False -- flipping it silently changes the loss "
        "of every checkpoint and every queued run")


def test_flag_is_logged_so_a_run_says_which_variant_it_was(stack):  # noqa: F811
    b = synthetic_train_batch(stack, batch=2, k=12, seed=5)
    b["gt_wp"] = torch.randn(2, 10, 2)
    L = v6_loss_step(stack, b, stage="S-W", weights=_o1_only(True),
                     o1_k=10, o5_k=12)
    assert L["log"]["o1_detach_encoder"] is True, (
        "a run must record the variant in its own log -- otherwise two arms "
        "are indistinguishable after the fact")


def test_NEGATIVE_CONTROL_with_the_flag_OFF_o1_does_reach_the_encoder(stack):  # noqa: F811
    """If this fails, the ON-test below is vacuous."""
    enc, pred = _grad_sums(stack, detach=False)
    assert pred > 0, "O1 produced no predictor gradient at all -- test is broken"
    assert enc > 0, (
        "with the flag OFF, O1 must reach the encoder; if it never did, the "
        "detach flag cannot be the mechanism behind the measured rank drop")


def test_with_the_flag_ON_o1_reaches_the_encoder_EXACTLY_ZERO(stack):  # noqa: F811
    enc, pred = _grad_sums(stack, detach=True)
    assert enc == 0.0, (
        f"encoder still received {enc:.3e} of gradient from O1 -- the detach "
        "did not cut the path it claims to cut")
    assert pred > 0, (
        "NEGATIVE CONTROL FAILED: the predictor received no O1 gradient either, "
        "so the flag has merely switched O1 OFF. That would 'fix' the rank for "
        "the trivial reason and just reproduce the two-term arm")


def test_the_two_modes_are_actually_different(stack):  # noqa: F811
    off_enc, off_pred = _grad_sums(stack, detach=False)
    on_enc, on_pred = _grad_sums(stack, detach=True)
    assert off_enc > 0 and on_enc == 0.0
    assert on_pred > 0 and off_pred > 0, (
        "the predictor must be trained by O1 in BOTH modes -- the flag changes "
        "WHERE O1's gradient goes, never WHETHER O1 is applied")
