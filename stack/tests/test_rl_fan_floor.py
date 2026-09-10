"""Tests for ``tanitad.rl.fan_floor`` — the RL rung's primary readout.

⭐ THE DESIGN RULE THESE TESTS ENFORCE: every control must read a **KNOWN VALUE**,
placed where it can contradict the result. Four of the five failures the
2026-08-22 ridge-probe retraction records were caught only because a control read
the same number as the thing being measured.

⛔ The load-bearing test is :func:`test_collapse_raises_floor_and_kills_diversity`:
it CONSTRUCTS the failure mode (mode collapse) and asserts that ``fan_floor@k``
alone is fooled by it while the floor+diversity PAIR is not. A gate that cannot
fail a knowingly-broken fan cannot certify an honest one.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.rl import fan_floor as F


def _fan(W=7, K=16, S=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(W, K, S, 2)
    t = torch.linspace(0.0, 1.0, S)
    for w in range(W):
        for k in range(K):
            speed = 8.0 + 4.0 * torch.rand(1, generator=g).item()
            lat = (torch.rand(1, generator=g).item() - 0.5) * 6.0
            base[w, k, :, 0] = speed * t * 2.0
            base[w, k, :, 1] = lat * t ** 2
    return base


def _quality(W=7, K=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(W, K, generator=g)


# ---------------------------------------------------------------------------
# fan_floor@k — the order statistic itself
# ---------------------------------------------------------------------------

def test_floor_at_k_is_the_kth_best_exactly():
    q = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    out = F.fan_floor_at_k(q, ks=(1, 2, 3, 4))
    assert out[1].item() == pytest.approx(0.9)
    assert out[2].item() == pytest.approx(0.7)
    assert out[3].item() == pytest.approx(0.5)
    assert out[4].item() == pytest.approx(0.1)


def test_floor_is_monotone_non_increasing_in_k():
    q = _quality()
    out = F.fan_floor_at_k(q, ks=(1, 2, 4, 8, 16))
    ks = sorted(out)
    for a, b in zip(ks, ks[1:]):
        assert torch.all(out[a] >= out[b] - 1e-7), f"@{a} < @{b}"


def test_floor_is_invariant_to_candidate_ORDER():
    """⭐ CONTROL WITH A KNOWN VALUE: permuting candidates changes nothing."""
    q = _quality()
    g = torch.Generator().manual_seed(3)
    perm = torch.argsort(torch.rand(q.shape, generator=g), dim=1)
    q2 = q.gather(1, perm)
    a = F.fan_floor_at_k(q, ks=(1, 5, 8))
    b = F.fan_floor_at_k(q2, ks=(1, 5, 8))
    for k in a:
        assert torch.equal(a[k], b[k]), f"@{k} moved under a permutation"


def test_k_greater_than_K_is_omitted_not_faked():
    q = _quality(K=4)
    out = F.fan_floor_at_k(q, ks=(1, 2, 64))
    assert set(out) == {1, 2}, "an undefined @k must be absent, never invented"


def test_all_k_out_of_range_raises():
    with pytest.raises(F.FanFloorError):
        F.fan_floor_at_k(_quality(K=4), ks=(99,))


# ---------------------------------------------------------------------------
# ⛔ THE MANDATORY best-of-k CONTROL
# ---------------------------------------------------------------------------

def test_random_best_of_1_equals_the_fan_MEAN_not_its_max():
    """⭐ KNOWN VALUE. A no-skill selector at budget 1 draws uniformly, so its
    expectation is the fan MEAN — not ``fan_floor@1`` (the max). If these two
    were ever equal the control would be vacuous."""
    q = _quality(W=5, K=32, seed=11)
    rb = F.random_best_of_k(q, ks=(1,), draws=400, seed=0)[1]
    assert torch.allclose(rb, q.mean(dim=1), atol=0.03), \
        "E[max over a random 1-subset] must be the fan mean"
    assert not torch.allclose(rb, q.max(dim=1).values, atol=0.03), \
        "the control must NOT coincide with fan_floor@1, or it proves nothing"


def test_random_best_of_K_equals_the_fan_MAX_exactly():
    """At k = K the random subset IS the whole fan, so the control must collapse
    onto ``fan_floor@1`` exactly. This pins the two ends of the control."""
    q = _quality(W=5, K=8, seed=12)
    rb = F.random_best_of_k(q, ks=(8,), draws=5, seed=0)[8]
    assert torch.allclose(rb, q.max(dim=1).values, atol=1e-6)


def test_random_best_of_k_is_monotone_increasing_in_k():
    q = _quality(W=6, K=32, seed=13)
    rb = F.random_best_of_k(q, ks=(1, 2, 4, 8, 16, 32), draws=60, seed=0)
    ks = sorted(rb)
    for a, b in zip(ks, ks[1:]):
        assert rb[a].mean() <= rb[b].mean() + 1e-6


def test_summarise_fan_ALWAYS_carries_the_random_control():
    """It is not an option — the retraction it exists for was a decision."""
    fan, q = _fan(), _quality()
    r = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(1, 8))
    assert set(r.rand_best) == {1, 8}
    assert any(n.startswith("rand_best@") for n in r.flat())


# ---------------------------------------------------------------------------
# ⛔⛔ THE LOAD-BEARING TEST — the floor is gameable, the pair is not
# ---------------------------------------------------------------------------

def test_collapse_raises_floor_and_kills_diversity():
    """⛔ DELIBERATE REGRESSION, constructed rather than hoped for.

    Collapsing the fan toward its own mean must (a) drive diversity monotonically
    to ~0 and (b) RAISE ``fan_floor@k`` for k > 1 toward ``@1``. That is the
    proof that a floor gain alone cannot distinguish a better fan from a narrower
    one — the reason :func:`floor_verdict` refuses a verdict without diversity.
    """
    fan = _fan(W=6, K=16, seed=5)
    # quality = a monotone function of the candidate's own geometry, so it moves
    # with the collapse exactly as a real rule-based reward would.
    def qual(f):
        return -(f[:, :, -1, 1].abs())          # prefer small lateral offset

    divs, floors = [], []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = F.collapse_fan(fan, a)
        divs.append(float(F.fan_diversity(c).mean()))
        floors.append(float(F.fan_floor_at_k(qual(c), ks=(8,))[8].mean()))

    for x, y in zip(divs, divs[1:]):
        assert y <= x + 1e-9, f"diversity not monotone under collapse: {divs}"
    assert divs[-1] == pytest.approx(0.0, abs=1e-6), "total collapse must read 0"
    for x, y in zip(floors, floors[1:]):
        assert y >= x - 1e-6, f"floor@8 must RISE under collapse: {floors}"
    assert floors[-1] > floors[0] + 1e-6, \
        "the whole point: collapse buys floor. If this fails the test is vacuous."


def test_floor_verdict_REFUSES_without_diversity():
    fan, q = _fan(), _quality()
    a = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(8,))
    b = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(8,))
    a.diversity = None
    with pytest.raises(F.FanFloorError):
        F.floor_verdict(a, b, k=8)


def test_floor_verdict_flags_COLLAPSE_SUSPECT_on_a_collapsed_fan():
    fan = _fan(W=6, K=16, seed=5)
    def qual(f):
        return -(f[:, :, -1, 1].abs())
    base = F.summarise_fan(fan, qual(fan), quality_name="lat_offset", ks=(8,))
    col = F.collapse_fan(fan, 0.9)
    arm = F.summarise_fan(col, qual(col), quality_name="lat_offset", ks=(8,))
    v = F.floor_verdict(arm, base, k=8)
    assert v["d_floor"] > 0, "the collapsed fan must LOOK like a floor gain"
    assert v["verdict"] == "COLLAPSE-SUSPECT", v


def test_floor_verdict_says_NO_GAIN_when_nothing_moved():
    fan, q = _fan(), _quality()
    a = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(8,))
    b = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(8,))
    v = F.floor_verdict(a, b, k=8)
    assert v["verdict"] == "NO-GAIN"
    assert v["d_floor"] == pytest.approx(0.0, abs=1e-12)
    assert v["d_diversity"] == pytest.approx(0.0, abs=1e-12)


def test_floor_verdict_VOIDs_a_mismatched_K():
    fan16, fan8 = _fan(K=16, seed=5), _fan(K=8, seed=6)
    a = F.summarise_fan(fan16, _quality(K=16), quality_name="r", ks=(8,))
    b = F.summarise_fan(fan8, _quality(K=8, seed=9), quality_name="r", ks=(8,))
    assert F.floor_verdict(a, b, k=8)["verdict"] == "VOID-K"


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def test_identical_candidates_read_diversity_EXACTLY_zero():
    """⭐ CONTROL WITH A KNOWN VALUE."""
    one = _fan(W=3, K=1, seed=2)[:, :1]
    fan = one.repeat(1, 12, 1, 1)
    assert float(F.fan_diversity(fan).abs().max()) == pytest.approx(0.0, abs=1e-6)
    assert float(F.fan_diversity(fan, mode="endpoint_std").abs().max()) == \
        pytest.approx(0.0, abs=1e-6)


def test_diversity_of_two_paths_is_their_own_rms_distance():
    """A closed-form value, so the implementation cannot drift silently."""
    fan = torch.zeros(1, 2, 4, 2)
    fan[0, 1, :, 1] = 3.0                       # a constant 3 m lateral offset
    assert float(F.fan_diversity(fan)[0]) == pytest.approx(3.0, abs=1e-6)


def test_diversity_scales_linearly_with_the_fan():
    fan = _fan(W=4, K=8, seed=7)
    c = fan.mean(dim=1, keepdim=True)
    scaled = c + (fan - c) * 2.0
    assert float(F.fan_diversity(scaled).mean()) == \
        pytest.approx(2.0 * float(F.fan_diversity(fan).mean()), rel=1e-5)


def test_diversity_is_order_invariant():
    fan = _fan(W=4, K=8, seed=8)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(F.fan_diversity(fan), F.fan_diversity(fan[:, perm]))


# ---------------------------------------------------------------------------
# Collision rate over the SET
# ---------------------------------------------------------------------------

def test_collision_rate_all_and_none_read_their_known_values():
    W, K = 5, 10
    assert torch.all(F.fan_collision_rate(torch.ones(W, K)) == 1.0)
    assert torch.all(F.fan_collision_rate(torch.zeros(W, K)) == 0.0)


def test_topk_collision_reads_the_ranked_subset_not_the_whole_fan():
    """The generator's rate and the selector's view are DIFFERENT populations —
    `D-RL-GEN-COLLIDES-1`'s whole point."""
    flag = torch.zeros(1, 4)
    flag[0, 0] = 1.0                                   # only the best-ranked collides
    rank = torch.tensor([[9.0, 1.0, 0.5, 0.1]])
    assert float(F.fan_collision_rate(flag)[0]) == pytest.approx(0.25)
    assert float(F.fan_collision_rate(flag, rank=rank, top_k=1)[0]) == pytest.approx(1.0)
    assert float(F.fan_collision_rate(flag, rank=rank, top_k=4)[0]) == pytest.approx(0.25)


def test_topk_without_rank_raises():
    with pytest.raises(F.FanFloorError):
        F.fan_collision_rate(torch.zeros(2, 4), top_k=2)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_echo_quality_is_REFUSED_by_name():
    fan, q = _fan(), _quality()
    for bad in ("ade", "minADE", "gt_similarity", "speed_match"):
        with pytest.raises(F.FanFloorError):
            F.summarise_fan(fan, q, quality_name=bad)


def test_admissible_quality_names_pass():
    for ok in ("reward_composed", "robust_contact", "feasibility", "progress"):
        assert F.assert_quality_admissible(ok) == ok


def test_assert_equal_k_refuses_mismatched_fans():
    with pytest.raises(F.FanFloorError):
        F.assert_equal_k(_quality(K=16), _quality(K=8, seed=4))
    assert F.assert_equal_k(_quality(K=16), _quality(K=16, seed=4)) == 16
    assert F.assert_equal_k(_fan(K=12), _quality(K=12)) == 12


def test_non_finite_quality_is_refused_as_a_SENTINEL():
    """`rank` carries −inf on masked candidates; standardising that column made
    every prediction NaN and argmin silently returned index 0 — the constant
    control. The instrument refuses the input rather than propagating it."""
    q = _quality()
    q[0, 0] = float("-inf")
    with pytest.raises(F.FanFloorError, match="sentinel"):
        F.fan_floor_at_k(q)


def test_shape_errors_are_loud():
    with pytest.raises(F.FanFloorError):
        F.fan_diversity(torch.zeros(4, 8, 5))          # missing the xy axis
    with pytest.raises(F.FanFloorError):
        F.fan_floor_at_k(torch.zeros(4, 8, 2))         # quality must be [W, K]
    with pytest.raises(F.FanFloorError):
        F.summarise_fan(_fan(K=16), _quality(K=8, seed=2), quality_name="r")


def test_collapse_alpha_is_bounded():
    with pytest.raises(F.FanFloorError):
        F.collapse_fan(_fan(), 1.5)


# ---------------------------------------------------------------------------
# The K-free form
# ---------------------------------------------------------------------------

def test_quantile_tau0_is_the_max_and_tau1_is_the_min():
    q = _quality(W=4, K=16, seed=21)
    out = F.fan_quantile(q, taus=(0.0, 1.0))
    assert torch.allclose(out[0.0], q.max(dim=1).values)
    assert torch.allclose(out[1.0], q.min(dim=1).values)


def test_quantile_is_K_INVARIANT_where_at_k_is_not():
    """⭐ The reason the K-free form exists: duplicate every candidate and the
    fan's quality distribution is unchanged, but ``@k`` reads a different place
    in it while the quantile does not."""
    q = _quality(W=4, K=16, seed=22)
    q2 = q.repeat_interleave(2, dim=1)                 # same distribution, K = 32
    assert torch.allclose(F.fan_quantile(q, (0.5,))[0.5],
                          F.fan_quantile(q2, (0.5,))[0.5], atol=1e-6)
    a8 = F.fan_floor_at_k(q, ks=(8,))[8]
    b8 = F.fan_floor_at_k(q2, ks=(8,))[8]
    assert not torch.allclose(a8, b8), \
        "if @k were K-invariant the guard would be pointless"


def test_flat_names_are_stable_and_unique():
    fan, q = _fan(), _quality()
    r = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(1, 8),
                        collision_flag=(q > 0.8).to(torch.uint8), rank=q)
    names = r.flat()
    assert "fan_floor@8" in names and "rand_best@8" in names
    assert "fan_diversity" in names and "fan_endpoint_std" in names
    assert "fan_collision_all" in names and "fan_collision_top8" in names
    assert len(names) == len(set(names))
    for v in names.values():
        assert v.shape == (fan.shape[0],), "every metric must be per-window [W]"
        assert torch.isfinite(v).all()


def test_readout_records_its_own_W_and_K():
    fan, q = _fan(W=9, K=12, seed=31), _quality(W=9, K=12, seed=32)
    r = F.summarise_fan(fan, q, quality_name="reward_composed", ks=(1,))
    assert (r.n_windows, r.n_candidates) == (9, 12)
    assert not math.isnan(float(r.diversity.mean()))
