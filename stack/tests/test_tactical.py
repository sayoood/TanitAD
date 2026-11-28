"""Tests for E4.2 + E4.3: the tactical layer stage 0
(tanitad/models/tactical.py). CPU-only, synthetic, small dims.

Pins:
  (1) shapes end-to-end (PhiTac -> fan -> f_tac -> selector -> losses);
  (2) param budgets at the reference sizes: PhiTac in [1 M, 4 M] (~2 M),
      FTac in [2 M, 6 M] (~4 M) — MEASURED via n_params();
  (3) TCN causality: perturbing the OLDEST window position changes the
      output (a future-position perturbation beyond the window is impossible
      by construction — the module consumes exactly W steps — so the
      complementary pin is batch-order equivariance);
  (4) fan diversity at init (anchor queries => N candidates not identical);
  (5) ranking_loss drives the hindsight winner's logit above a loser's after
      a few optimizer steps on synthetic data;
  (6) WTA regression decreases on a synthetic overfit (20 steps, tiny dims);
  (7) valid-mask handling: invalid tau rows contribute ZERO gradient;
  (8) admissibility guard: TacticalGoalFan.forward accepts ONLY z_tac —
      no **kwargs backdoor (2026-08-03 binding rule);
  (9) sel_gap_tac triple is exact (selected/oracle/gap, gap >= 0);
 (10) the tau constants match the E4.1 labeler's (cross-pin vs refb_labels).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
from torch import nn

from tanitad.models.tactical import (FTac, PhiTac, TacticalGoalFan,
                                     TacticalSelector, TacticalStage0,
                                     TacticalStage0Config, fan_goal_error,
                                     ranking_loss, sel_gap_tac,
                                     wta_regression_loss)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

torch.manual_seed(0)

# tiny reference dims for the CPU tests (budget tests use the real ones)
TINY = dict(d_op=16, d_tac=12, window=4, n_goals=3, horizon_taus=(2, 4),
            phi_hidden=8, f_hidden=16, fan_hidden=8, sel_hidden=8)


def tiny_stage() -> TacticalStage0:
    return TacticalStage0(TacticalStage0Config(**TINY))


def tiny_batch(B: int = 4, with_next: bool = True) -> dict:
    g = torch.Generator().manual_seed(7)
    batch = {
        "z_op": torch.randn(B, TINY["window"], TINY["d_op"], generator=g),
        "goal": torch.randn(B, 2, 4, generator=g),
        "goal_valid": torch.ones(B, 2, dtype=torch.bool),
    }
    if with_next:
        batch["z_op_next"] = torch.randn(B, TINY["window"], TINY["d_op"],
                                         generator=g)
    return batch


# ---------------------------------------------------------------------------
# (1) shapes end-to-end
# ---------------------------------------------------------------------------

def test_shapes_end_to_end():
    s = tiny_stage()
    B, N, K = 5, TINY["n_goals"], len(TINY["horizon_taus"])
    out = s(torch.randn(B, TINY["window"], TINY["d_op"]))
    assert out["z_tac"].shape == (B, TINY["d_tac"])
    assert out["goals"].shape == (B, N, K, 4)
    assert out["logits"].shape == (B, N)
    assert out["scores"].shape == (B, N)
    assert out["sel_idx"].shape == (B,)
    # heading channel emitted wrapped, matching the E4.1 (-pi, pi] convention
    h = out["goals"][..., 2]
    assert bool((h > -torch.pi - 1e-6).all() and (h <= torch.pi + 1e-6).all())

    losses = s.losses(tiny_batch())
    for k in ("wta", "rank_fan", "rank_sel", "ftac", "sel_err", "oracle_err",
              "sel_gap", "total"):
        assert k in losses and losses[k].ndim == 0, k
        assert torch.isfinite(losses[k]), k
    losses["total"].backward()  # the composed loss reaches every trained part
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in s.phi_tac.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in s.goal_fan.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in s.f_tac.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in s.selector.parameters())


# ---------------------------------------------------------------------------
# (2) param budgets at the REFERENCE sizes (MEASURED, asserted in band)
# ---------------------------------------------------------------------------

def test_param_budget_phi_tac():
    n = PhiTac(d_op=512, d_tac=512, window=4, hidden=256).n_params()
    assert 1_000_000 <= n <= 4_000_000, n     # ~2 M target (measured 1.71 M)


def test_param_budget_f_tac():
    n = FTac(d_tac=512, d_goal=12, hidden=512).n_params()
    assert 2_000_000 <= n <= 6_000_000, n     # ~4 M target (measured 3.68 M)


# ---------------------------------------------------------------------------
# (3) TCN causality + batch-order equivariance
# ---------------------------------------------------------------------------

def test_tcn_oldest_input_reaches_output():
    phi = PhiTac(**{k: TINY[k] for k in ("d_op", "d_tac", "window")},
                 hidden=TINY["phi_hidden"]).eval()
    x = torch.randn(1, TINY["window"], TINY["d_op"])
    y0 = phi(x)
    x2 = x.clone()
    x2[:, 0] += 1.0                      # oldest position (t-3 s)
    y2 = phi(x2)
    # the receptive field must cover the whole window: the oldest sample
    # changes z_tac. (A future-position test is impossible by construction —
    # the module consumes exactly W past steps and nothing else.)
    assert (y0 - y2).abs().max() > 1e-6


def test_tcn_batch_order_equivariance():
    phi = PhiTac(**{k: TINY[k] for k in ("d_op", "d_tac", "window")},
                 hidden=TINY["phi_hidden"]).eval()
    x = torch.randn(3, TINY["window"], TINY["d_op"])
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        assert torch.allclose(phi(x)[perm], phi(x[perm]), atol=1e-5)


# ---------------------------------------------------------------------------
# (4) fan diversity at init
# ---------------------------------------------------------------------------

def test_fan_diversity_at_init():
    fan = TacticalGoalFan(TINY["d_tac"], TINY["n_goals"],
                          TINY["horizon_taus"], TINY["fan_hidden"]).eval()
    goals, logits = fan(torch.randn(2, TINY["d_tac"]))
    # anchor queries must yield distinct candidates untrained — diversity by
    # construction, not something training has to discover.
    for i in range(TINY["n_goals"]):
        for j in range(i + 1, TINY["n_goals"]):
            assert (goals[:, i] - goals[:, j]).abs().max() > 1e-6, (i, j)


# ---------------------------------------------------------------------------
# (5) ranking loss orders the logits after a few steps
# ---------------------------------------------------------------------------

def test_ranking_loss_drives_winner_above_losers():
    err = torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.05, 0.6]])  # winners: 0, 1
    logits = nn.Parameter(torch.zeros(2, 3))
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(30):
        opt.zero_grad()
        loss = ranking_loss(logits, err, margin=0.1)
        loss.backward()
        opt.step()
    lg = logits.detach()
    assert lg[0, 0] > lg[0, 1] and lg[0, 0] > lg[0, 2]
    assert lg[1, 1] > lg[1, 0] and lg[1, 1] > lg[1, 2]
    # and the margin objective is actually satisfied (loss ~ 0)
    assert float(ranking_loss(lg, err, margin=0.1)) < 1e-3


def test_ranking_loss_degenerate_single_candidate_is_zero():
    loss = ranking_loss(torch.randn(3, 1), torch.rand(3, 1))
    assert float(loss) == 0.0


# ---------------------------------------------------------------------------
# (6) WTA loss decreases on a synthetic overfit
# ---------------------------------------------------------------------------

def test_wta_overfit_decreases():
    torch.manual_seed(1)
    s = tiny_stage()
    batch = tiny_batch(B=4)
    opt = torch.optim.Adam(s.parameters(), lr=3e-3)
    first = None
    for _ in range(20):
        opt.zero_grad()
        losses = s.losses(batch)
        if first is None:
            first = float(losses["wta"].detach())
        losses["total"].backward()
        opt.step()
    last = float(s.losses(batch)["wta"].detach())
    assert last < first, (first, last)


# ---------------------------------------------------------------------------
# (7) valid mask: invalid tau rows contribute ZERO gradient
# ---------------------------------------------------------------------------

def test_valid_mask_zero_grad_on_invalid_rows():
    torch.manual_seed(2)
    fan = TacticalGoalFan(TINY["d_tac"], TINY["n_goals"],
                          TINY["horizon_taus"], TINY["fan_hidden"])
    z = torch.randn(3, TINY["d_tac"])
    goals, logits = fan(z)
    goals.retain_grad()
    labels = torch.randn(3, 2, 4)
    valid = torch.tensor([[True, False],       # tau 1 invalid for sample 0
                          [True, True],
                          [False, True]])      # tau 0 invalid for sample 2
    loss = wta_regression_loss(goals, labels, valid)
    loss.backward()
    g = goals.grad
    assert g is not None
    # invalid rows: zero grad for EVERY candidate (masked before reduction)
    assert g[0, :, 1, :].abs().max() == 0.0
    assert g[2, :, 0, :].abs().max() == 0.0
    # valid rows: the winning candidate did get gradient
    assert g[:, :, :, :].abs().sum() > 0
    assert g[0, :, 0, :].abs().sum() > 0
    assert g[2, :, 1, :].abs().sum() > 0


def test_valid_mask_all_invalid_is_zero_loss():
    fan = TacticalGoalFan(TINY["d_tac"], TINY["n_goals"],
                          TINY["horizon_taus"], TINY["fan_hidden"])
    goals, _ = fan(torch.randn(2, TINY["d_tac"]))
    labels = torch.randn(2, 2, 4)
    valid = torch.zeros(2, 2, dtype=torch.bool)
    loss = wta_regression_loss(goals, labels, valid)
    assert float(loss.detach()) == 0.0
    loss.backward()                            # still differentiable, all-zero


# ---------------------------------------------------------------------------
# (8) admissibility guard (binding rule 2026-08-03): z_tac ONLY
# ---------------------------------------------------------------------------

def test_goal_fan_forward_accepts_only_z_tac():
    sig = inspect.signature(TacticalGoalFan.forward)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "z_tac"]
    # no *args / **kwargs backdoor through which a situation-classifier
    # output or ego state could ever be threaded at inference.
    assert all(p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD) for p in params)


# ---------------------------------------------------------------------------
# (9) sel_gap triple
# ---------------------------------------------------------------------------

def test_sel_gap_tac_exact():
    err = torch.tensor([[0.4, 0.1, 0.9],
                        [0.2, 0.5, 0.3]])
    sel_idx = torch.tensor([0, 2])
    sel, oracle, gap = sel_gap_tac(err, sel_idx)
    assert torch.allclose(sel, torch.tensor([0.4, 0.3]))
    assert torch.allclose(oracle, torch.tensor([0.1, 0.2]))
    assert torch.allclose(gap, torch.tensor([0.3, 0.1]))
    assert bool((gap >= 0).all())
    # selector agreeing with the oracle => gap exactly 0
    _, _, g0 = sel_gap_tac(err, err.argmin(dim=1))
    assert float(g0.abs().max()) == 0.0


def test_selector_scores_shape_and_grad():
    sel = TacticalSelector(TINY["d_tac"], d_goal=8, hidden=TINY["sel_hidden"])
    f = FTac(TINY["d_tac"], d_goal=8, hidden=TINY["f_hidden"])
    z = torch.randn(2, TINY["d_tac"])
    goals = torch.randn(2, TINY["n_goals"], 2, 4)
    scores = sel(z, goals, f)
    assert scores.shape == (2, TINY["n_goals"])
    # ranking through the selector reaches BOTH the scorer and f_tac (the
    # selector scores ROLLED futures, so f_tac is on the path)
    err = torch.rand(2, TINY["n_goals"])
    ranking_loss(scores, err).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in sel.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in f.parameters())


# ---------------------------------------------------------------------------
# (10) cross-pins vs the E4.1 labeler + aux heads
# ---------------------------------------------------------------------------

def test_tau_and_class_constants_match_refb_labels():
    import refb_labels as R  # noqa: E402  (scripts/ on sys.path above)
    from tanitad.models import tactical as T
    assert T.GOAL_TAC_TAUS_STEPS == R.GOAL_TAC_TAUS_STEPS
    assert T.N_LAT3 == len(R.LAT3_NAMES)
    assert T.N_LON3 == len(R.LON3_NAMES)
    assert T.N_LANE3 == len(R.LANE3_NAMES)


def test_aux_heads_off_by_default_on_when_weighted():
    s = tiny_stage()
    batch = tiny_batch()
    batch["man3"] = torch.zeros(4, 3, dtype=torch.long)
    losses = s.losses(batch)
    # weights default 0 => aux terms not computed at all
    assert not any(k.startswith("aux_") for k in losses)

    s2 = TacticalStage0(TacticalStage0Config(**TINY, w_lat=0.5, w_lon=0.5,
                                             w_lane=0.5))
    losses2 = s2.losses(batch)
    for k in ("aux_lat", "aux_lon", "aux_lane"):
        assert k in losses2 and torch.isfinite(losses2[k])
    assert float(losses2["total"].detach()) > 0
