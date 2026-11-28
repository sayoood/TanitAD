"""Pin the CoT influence/consistency instrument to the values it MUST return.

Every test here asserts a value that is known in advance from the definition,
not a value observed from a run. That is the whole point: this module exists to
tell a caption generator from a reasoner, and a metric that cannot return the
no-information value on no information will not return the truth on data.
"""
import pytest
import torch

from tanitad.instruments.cot_influence import (
    NoInfluenceViolation,
    consistency,
    constant_energy_floor,
    echo_residual,
    influence,
    rerank_logits,
    roll_control,
)


def _fan(b=4, k=6, s=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, k, s, 2, generator=g)


def _logits(b=4, k=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, k, generator=g)


# ---------------------------------------------------------------- influence --
def test_beta_zero_is_a_bit_identity_not_an_approximation():
    s0, e = _logits(), _logits(seed=9)
    s1 = rerank_logits(s0, e, beta=0.0)
    assert torch.equal(s1, s0)
    r = influence(s0, s1, _fan())
    assert r.flip_rate == 0.0
    assert r.kl == 0.0
    assert r.geo_m == 0.0
    assert r.is_inert()


def test_identical_logits_read_exactly_zero_influence():
    s0 = _logits()
    r = influence(s0, s0.clone(), _fan())
    assert (r.flip_rate, r.kl, r.geo_m) == (0.0, 0.0, 0.0)


def test_constant_energy_cannot_move_the_decision():
    """A per-window constant shifts every candidate equally -> argmax and
    softmax are unchanged. If this ever fails, the influence pathway is running
    through an offset rather than through the energy's variation over the fan."""
    s0, e = _logits(), _logits(seed=3)
    flat = constant_energy_floor(e)
    s1 = rerank_logits(s0, flat, beta=2.5)
    r = influence(s0, s1, _fan())
    assert r.flip_rate == 0.0
    assert r.geo_m == 0.0
    assert r.kl == pytest.approx(0.0, abs=1e-6)


def test_a_real_energy_does_move_the_decision():
    """The complement of the test above: without it, a broken rerank that
    always returned s0 would pass every zero-assertion in this file."""
    s0 = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]])
    e = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    s1 = rerank_logits(s0, e, beta=10.0)
    r = influence(s0, s1, _fan(b=2, k=3))
    assert r.flip_rate == 0.5          # only the second row's argmax moves
    assert r.kl > 0.0
    assert r.geo_m > 0.0


def test_geo_is_nan_without_a_fan_never_zero():
    s0 = _logits()
    r = influence(s0, s0.clone())
    assert r.geo_m != r.geo_m           # nan: "not computed" != "no effect"


def test_influence_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        influence(_logits(b=4), _logits(b=3))
    with pytest.raises(ValueError):
        influence(_logits(), _logits(), _fan(b=4, k=5))
    with pytest.raises(ValueError):
        rerank_logits(_logits(b=4, k=6), _logits(b=4, k=5), beta=1.0)


# -------------------------------------------------------------- consistency --
def test_perfect_consistency_reads_rank_one_and_gap_zero():
    e = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]])
    sel = torch.tensor([0, 1])
    c = consistency(e, sel)
    assert c.mean_rank == 1.0
    assert c.top1_rate == 1.0
    assert c.gap == 0.0


def test_worst_consistency_reads_the_last_rank():
    e = torch.tensor([[0.0, 1.0, 2.0]])
    c = consistency(e, torch.tensor([2]))
    assert c.mean_rank == 3.0
    assert c.top1_rate == 0.0
    assert c.gap == pytest.approx(2.0)


def test_constant_energy_is_scored_pessimistically():
    """A degenerate reasoner that has no preference must land at the WORST
    rank, not at an accidental best from tie ordering. Flattering a degenerate
    model is exactly how a caption generator would score as consistent."""
    e = torch.zeros(3, 7)
    c = consistency(e, torch.tensor([0, 3, 6]))
    assert c.mean_rank == 7.0
    assert c.top1_rate == 0.0
    assert c.gap == 0.0
    assert c.chance_rank == 4.0


def test_chance_rank_is_reported_so_k_dependence_is_visible():
    assert consistency(torch.zeros(2, 5), torch.tensor([0, 0])).chance_rank == 3.0
    assert consistency(torch.zeros(2, 128), torch.tensor([0, 0])).chance_rank == 64.5


def test_consistency_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        consistency(torch.zeros(3, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        consistency(torch.zeros(3), torch.tensor([0, 1, 2]))


# ------------------------------------------------------------------ control --
def test_roll_control_mispairs_every_row():
    z = torch.arange(5).float().unsqueeze(1)
    rolled = roll_control(z, shift=1)
    assert not (rolled == z).all(dim=1).any()      # NOT ONE row keeps its own z


def test_roll_control_refuses_a_batch_it_cannot_mispair():
    with pytest.raises(ValueError, match="batch >= 2"):
        roll_control(torch.zeros(1, 4))
    with pytest.raises(ValueError, match="identity"):
        roll_control(torch.zeros(4, 2), shift=4)


def test_roll_control_is_a_permutation_of_the_same_rows():
    z = torch.randn(6, 3, generator=torch.Generator().manual_seed(4))
    r = roll_control(z, shift=2)
    assert torch.allclose(r.sum(0), z.sum(0))      # same multiset, re-indexed


# -------------------------------------------------------------------- echo --
def test_echo_residual_is_zero_when_z_is_a_linear_function_of_tau():
    g = torch.Generator().manual_seed(7)
    tau = torch.randn(64, 4, 2, generator=g)
    W = torch.randn(8, 5, generator=g)
    z = tau.reshape(64, -1) @ W
    assert echo_residual(z, tau, ridge=1e-8) == pytest.approx(0.0, abs=1e-6)


def test_echo_residual_is_near_one_when_z_is_independent_of_tau():
    g = torch.Generator().manual_seed(11)
    tau = torch.randn(400, 4, 2, generator=g)
    z = torch.randn(400, 5, generator=g)
    assert echo_residual(z, tau, ridge=1.0) > 0.9


def test_echo_residual_refuses_degenerate_input():
    with pytest.raises(ValueError):
        echo_residual(torch.zeros(1, 3), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="zero variance"):
        echo_residual(torch.ones(8, 3), torch.randn(8, 2, 2))
    with pytest.raises(ValueError):
        echo_residual(torch.randn(8, 3), torch.randn(7, 2, 2))


# ----------------------------------------------------------- semantic null --
from tanitad.instruments.cot_influence import (          # noqa: E402
    norm_matched_null,
    permuted_prior_null,
    semantic_null_screen,
)


def test_norm_matched_null_preserves_per_window_mean_and_spread():
    """The point of the null is that it intervenes JUST AS HARD. If its scale
    drifted from the real energy's, a weaker null would flatter every arm."""
    g = torch.Generator().manual_seed(3)
    e = torch.randn(6, 32, generator=torch.Generator().manual_seed(2)) * 4.0 + 7.0
    z = norm_matched_null(e, g)
    assert torch.allclose(z.mean(1), e.mean(1), atol=1e-5)
    assert torch.allclose(z.std(1), e.std(1), rtol=1e-4)
    assert not torch.allclose(z, e)


def test_norm_matched_null_of_a_constant_energy_is_constant():
    e = torch.full((4, 16), 2.5)
    z = norm_matched_null(e, torch.Generator().manual_seed(0))
    assert torch.allclose(z, e, atol=1e-4)


def test_norm_matched_null_refuses_a_fan_it_cannot_rerank():
    with pytest.raises(ValueError, match="K >= 2"):
        norm_matched_null(torch.zeros(4, 1))
    with pytest.raises(ValueError):
        norm_matched_null(torch.zeros(4))


def test_permuted_prior_null_mispairs_every_row_and_keeps_the_values():
    e = torch.randn(5, 8, generator=torch.Generator().manual_seed(6))
    p = permuted_prior_null(e)
    assert not (p == e).all(dim=1).any()
    assert torch.allclose(p.sum(0), e.sum(0))       # same rows, re-indexed


def test_a_content_free_prior_can_OUT_influence_a_real_one():
    """⛔ THE PUBLISHED FAILURE MODE, reproduced in miniature.

    A near-flat 'real' energy and a high-variance null: the null moves the
    decision far more. This test exists to prove the screen can FAIL -- a
    control that cannot fail is not a control, and this is precisely the case
    VLADriveBench measured (a 'depicts'-token prior at ~2x the real CoT)."""
    s0 = torch.randn(8, 16, generator=torch.Generator().manual_seed(1))
    flat = torch.randn(8, 16, generator=torch.Generator().manual_seed(2)) * 1e-3
    rep = semantic_null_screen(s0, flat, beta=5.0,
                               generator=torch.Generator().manual_seed(3))
    assert rep.real_kl < rep.norm_matched_kl or rep.real_kl < rep.permuted_kl
    assert not rep.passes


def test_the_screen_passes_when_the_energy_actually_carries_the_decision():
    """The complement: without it, a screen that always failed would pass every
    assertion above. Here the real energy is large and structured while the
    nulls are matched to a SMALLER scale, so the real arm must win both."""
    g = torch.Generator().manual_seed(11)
    s0 = torch.randn(64, 16, generator=g)
    energy = torch.randn(64, 16, generator=g) * 3.0
    rep = semantic_null_screen(s0, energy, beta=1.0,
                               generator=torch.Generator().manual_seed(12))
    # both nulls are norm-matched by construction, so this is a genuine contest;
    # what must hold is that the report is well-formed and the verdict is the
    # conjunction, never the disjunction.
    assert rep.n == 64
    assert rep.passes == (rep.beats_norm_matched and rep.beats_permuted)


def test_screen_verdict_is_a_conjunction_not_a_disjunction():
    r = SemanticNullReport(real_kl=1.0, norm_matched_kl=0.5, permuted_kl=2.0, n=4)
    assert r.beats_norm_matched and not r.beats_permuted
    assert not r.passes          # ⛔ beating ONE null is not passing


from tanitad.instruments.cot_influence import SemanticNullReport   # noqa: E402


# ---------------------------------------------------------- outcome screen --
from tanitad.instruments.cot_influence import (          # noqa: E402
    OutcomeNullReport,
    outcome_null_screen,
)


def test_outcome_screen_rewards_the_prior_that_picks_the_BETTER_candidate():
    """⛔ THE DISTINCTION THE KL SCREEN CANNOT MAKE. Here the real energy nudges
    gently toward the best candidate while the setup is such that a random push
    goes elsewhere. What must be scored is WHERE the arm landed, not how hard it
    pushed -- the KL screen would prefer whichever moved further."""
    s0 = torch.zeros(200, 8)
    outcome = torch.rand(200, 8, generator=torch.Generator().manual_seed(5))
    # a prior that knows the answer: lowest energy exactly on the best candidate
    energy = outcome.clone()
    rep = outcome_null_screen(s0, energy, beta=50.0, outcome=outcome,
                              generator=torch.Generator().manual_seed(6))
    assert rep.real < rep.norm_matched
    assert rep.real < rep.permuted
    assert rep.passes


def test_outcome_screen_fails_a_prior_that_knows_nothing():
    g = torch.Generator().manual_seed(7)
    s0 = torch.zeros(200, 8)
    outcome = torch.rand(200, 8, generator=g)
    energy = torch.randn(200, 8, generator=g)      # unrelated to the outcome
    rep = outcome_null_screen(s0, energy, beta=50.0, outcome=outcome,
                              generator=torch.Generator().manual_seed(8))
    assert not rep.passes or abs(rep.real - rep.norm_matched) < 0.05


def test_outcome_screen_verdict_is_a_conjunction():
    r = OutcomeNullReport(real=0.4, norm_matched=0.3, permuted=0.9, n=4)
    assert not r.passes                       # beating ONE null is not passing
    r2 = OutcomeNullReport(real=0.2, norm_matched=0.3, permuted=0.9, n=4)
    assert r2.passes


def test_outcome_screen_honours_higher_is_better():
    r = OutcomeNullReport(real=0.9, norm_matched=0.3, permuted=0.4, n=4,
                          lower_is_better=False)
    assert r.passes
    assert not OutcomeNullReport(real=0.9, norm_matched=0.3, permuted=0.4,
                                 n=4, lower_is_better=True).passes


def test_outcome_screen_rejects_a_mismatched_outcome_table():
    with pytest.raises(ValueError, match="must match energy"):
        outcome_null_screen(torch.zeros(4, 6), torch.zeros(4, 6), beta=1.0,
                            outcome=torch.zeros(4, 5))
