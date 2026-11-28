"""`L2-SPD`'s speed-appropriateness term — the controls `PREREG_D9_REWARD_REPAIR` §13 fixed.

⛔ WHY THE TERM EXISTS. `H-DDV2RL-2` seed 1 over-sped the human by **+0.372 m/s**
(separated) while **not** driving off-road. The reward could not see it: `ego_progress`
SATURATES at the route end, so candidates at 10 / 12 / 14 / 20 m/s against a human at 10
all score the same progress (MEASURED 2026-09-17). Speed above the human's was a **flat
direction** — neither rewarded nor penalised — and a flat direction under a stochastic
policy gradient drifts.

⛔ The first attempted repair — a two-sided EP, `min(raw,ref)/max(raw,ref)` — was
implemented, **failed its own test within the hour**, and was reverted: reshaping the ratio
changes nothing because its input is already clipped. This term **ADDS** a penalty instead.

Every assertion below is a control §13.3 named **before** the term existed.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tanitad.rl import pdm_proxy as P                        # noqa: E402

N = P.PROXY.n_ticks


def _states(*speeds: float) -> torch.Tensor:
    """`[M, T, 4]` straight-ahead candidates; row 0 is the HUMAN, as the term requires."""
    t = torch.arange(N + 1, dtype=torch.float32) * P.PROXY.dt
    st = torch.zeros((len(speeds), N + 1, 4))
    for i, v in enumerate(speeds):
        st[i, :, 0] = v * t
        st[i, :, 3] = v
    return st


def _score(states, cfg):
    tracks = P.AgentTracks.from_frames([[] for _ in range(N + 1 + 9)],
                                       torch.zeros((N + 1 + 9, 4)))
    human = states[0]
    return P.score_candidates(states[1:], human, tracks, human[:, :2].clone(), cfg=cfg)


# --------------------------------------------------------------------------- #
# §13.3 — the four controls that must read known values                        #
# --------------------------------------------------------------------------- #

def test_CONTROL_w_spd_zero_is_BYTE_IDENTICAL_to_the_arm_without_the_term():
    """⛔ A new term that changes the score at zero weight is not a term, it is a bug."""
    st = _states(10.0, 10.0, 14.0, 6.0)
    off = _score(st, P.ProxyConfig(w_spd=0.0))
    # the reference: a config that predates the term is expressed by w_spd = 0
    base = _score(st, P.PROXY)
    assert torch.equal(off["pdms"], base["pdms"])
    for k in ("nc", "dac", "ep", "ttc", "comfort"):
        assert torch.equal(off[k], base[k])


def test_CONTROL_the_human_scores_SPD_exactly_1():
    """Its excess over itself is zero. Any other value means `v_human` and `v_cand` are
    being computed from different sources."""
    st = _states(10.0, 12.0)
    out = _score(st, P.ProxyConfig(w_spd=2.0))
    assert float(out["human"]["spd"]) == pytest.approx(1.0, abs=1e-9)


def test_CONTROL_a_candidate_AT_the_human_speed_scores_exactly_1():
    """The penalty starts AT the reference, not before it."""
    st = _states(10.0, 10.0)
    out = _score(st, P.ProxyConfig(w_spd=2.0))
    assert float(out["spd"][0]) == pytest.approx(1.0, abs=1e-9)


def test_CONTROL_a_SLOWER_candidate_scores_exactly_1_one_sidedness_asserted():
    """⛔ Being slower is ALREADY penalised by EP. Penalising it here too would be
    double-counting and would change the meaning of the existing EP result."""
    st = _states(10.0, 8.0, 6.0, 2.0)
    out = _score(st, P.ProxyConfig(w_spd=2.0))
    for i in range(3):
        assert float(out["spd"][i]) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# the term's shape                                                             #
# --------------------------------------------------------------------------- #

def test_the_penalty_is_linear_in_the_excess_and_clamps_at_v_tol():
    cfg = P.ProxyConfig(w_spd=2.0, v_tol_mps=1.5)
    st = _states(10.0, 10.75, 11.5, 13.0)          # excess 0.75, 1.5, 3.0
    out = _score(st, cfg)
    assert float(out["spd"][0]) == pytest.approx(0.5, abs=1e-6)
    assert float(out["spd"][1]) == pytest.approx(0.0, abs=1e-6)
    assert float(out["spd"][2]) == pytest.approx(0.0, abs=1e-6), "clamped, never negative"


def test_v_tol_changes_the_slope_and_the_LADDER_values_are_distinguishable():
    """⚠️ §13.2 fixes the ladder {0.75, 1.5, 3.0} and forbids choosing it from the
    failure. These must at least be distinguishable, or the ladder is decorative."""
    st = _states(10.0, 10.75)                       # the failing seed's scale of excess
    got = [float(_score(st, P.ProxyConfig(w_spd=2.0, v_tol_mps=v))["spd"][0])
           for v in (0.75, 1.5, 3.0)]
    assert got == sorted(got), "a larger tolerance must be more forgiving"
    assert len(set(round(g, 6) for g in got)) == 3, "the three rungs must differ"


# --------------------------------------------------------------------------- #
# ⭐ the design choice, asserted rather than commented                          #
# --------------------------------------------------------------------------- #

def test_SPD_is_ADDITIVE_not_multiplicative_so_a_zero_does_NOT_annihilate_the_score():
    """⭐ The distinction from NC and DAC, which ARE multipliers.

    A collision or leaving the road is a CONSTRAINT and annihilates the score. Driving
    too fast is a QUALITY failure: it must cost, not erase. If SPD ever annihilates,
    a mild over-speed becomes catastrophic and the term reintroduces — from the other
    side — the all-or-nothing behaviour that made a constant DAC so damaging.
    """
    cfg = P.ProxyConfig(w_spd=2.0, v_tol_mps=1.5)
    st = _states(10.0, 20.0)                        # excess 10 m/s >> v_tol -> SPD 0
    out = _score(st, cfg)
    assert float(out["spd"][0]) == pytest.approx(0.0, abs=1e-9)
    assert float(out["pdms"][0]) > 0.0, (
        "SPD is a weighted component, not a multiplier: a zero must reduce the score, "
        "never annihilate it")


def test_enabling_the_term_LOWERS_an_over_speeding_candidate_and_leaves_a_matching_one():
    cfg_on, cfg_off = P.ProxyConfig(w_spd=2.0), P.ProxyConfig(w_spd=0.0)
    st = _states(10.0, 10.0, 12.0)
    on, off = _score(st, cfg_on), _score(st, cfg_off)
    assert float(on["pdms"][0]) == pytest.approx(float(off["pdms"][0]), rel=1e-6), \
        "a candidate at the human's speed is untouched (SPD == 1 enters as a full term)"
    assert float(on["pdms"][1]) < float(off["pdms"][1]), \
        "an over-speeding candidate must score lower once the term is weighted"


def test_spd_is_REPORTED_even_when_unweighted():
    """⚠️ The D9 DAC lesson: a sub-score that is not logged cannot be seen to be wrong.
    `spd` must appear in the output at w_spd = 0 too, so a run can SEE the excess it is
    not yet penalising."""
    st = _states(10.0, 14.0)
    out = _score(st, P.ProxyConfig(w_spd=0.0))
    assert "spd" in out and "spd" in out["human"]
    assert float(out["spd"][0]) < 1.0, "the term is computed even when it is not weighted"
