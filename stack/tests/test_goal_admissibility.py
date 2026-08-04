"""Tests for tanitad/eval/goal_admissibility.py — the goal-input leak instrument.

The instrument exists because two goal-signal defects got through prose review:
the nav echo (a route head that was an exact bijection of its own input and
scored 1.0000) and a supplied route on a corpus whose only route supplier is the
ego's own future. Each is reproduced here as a test, so the instrument is proved
against the failure it was built for rather than against a toy.
"""
from __future__ import annotations

import numpy as np
import pytest

from tanitad.eval.goal_admissibility import (ECHO_FLAG_RATE, audit_goal_signal,
                                             echo_score, horizon_disjoint,
                                             incremental_information,
                                             situation_disjoint)


# --- 1. the nav echo -------------------------------------------------------- #
def test_echo_score_catches_an_exact_bijection_the_way_the_nav_head_was():
    """flagship-v1's route head: 369/369 and 81/81 recoverable from the nav it is
    fed, scoring 1.0000. A perfect relabelling must read as an ECHO, not skill."""
    rng = np.random.default_rng(0)
    nav = rng.integers(0, 4, 369)
    route = np.array([{0: 2, 1: 0, 2: 3, 3: 1}[int(v)] for v in nav])
    r = echo_score(nav, route)
    assert r["functional_agreement"] == 1.0
    assert r["bijection"] is True
    assert r["is_echo"] is True
    assert r["n"] == 369


def test_echo_score_does_not_fire_on_an_independent_output():
    rng = np.random.default_rng(1)
    r = echo_score(rng.integers(0, 4, 800), rng.integers(0, 3, 800))
    assert r["functional_agreement"] < ECHO_FLAG_RATE
    assert r["is_echo"] is False and r["bijection"] is False


def test_echo_score_reports_UNPOWERED_rather_than_clean_on_zero_windows():
    """Absence found at n=0 is not absence — the standing rule, in code."""
    r = echo_score(np.array([]), np.array([]))
    assert r["status"] == "UNPOWERED" and "NOT established" in r["reason"]
    assert "is_echo" not in r, "an empty probe must not emit a clean verdict"


def test_echo_score_refuses_misaligned_arrays():
    with pytest.raises(ValueError):
        echo_score(np.zeros(5), np.zeros(6))


# --- 2. the horizon guard --------------------------------------------------- #
def test_horizon_disjoint_flags_the_vtarget_overlap_and_clears_the_guarded_mint():
    """The scored horizon is {l .. l+20}. The v2 mint reads [l+1, l+200) and
    overlaps it; the guarded mint reads [l+21, l+200) and does not."""
    ell = 7
    bad = horizon_disjoint(ell + 1, ell + 200, ell, ell + 21)
    good = horizon_disjoint(ell + 21, ell + 200, ell, ell + 21)
    assert bad["disjoint"] is False and bad["n_overlap_steps"] == 20
    assert good["disjoint"] is True and good["n_overlap_steps"] == 0


# --- 3. the substantive residual -------------------------------------------- #
def test_incremental_information_is_zero_for_a_signal_that_adds_nothing():
    rng = np.random.default_rng(2)
    eid = np.repeat(np.arange(10), 40)
    legal = rng.normal(size=(400, 2))
    y = legal @ np.array([1.0, -2.0]) + rng.normal(0, 0.05, 400)
    noise = rng.normal(size=(400, 1))
    r = incremental_information(y, legal, noise, eid)
    assert r["n_episodes"] == 10
    assert abs(r["delta_r2"]) < 0.02


def test_incremental_information_finds_a_planted_privileged_signal():
    rng = np.random.default_rng(3)
    eid = np.repeat(np.arange(10), 40)
    legal = rng.normal(size=(400, 2))
    hidden = rng.normal(size=400)
    y = legal[:, 0] + 3.0 * hidden + rng.normal(0, 0.05, 400)
    r = incremental_information(y, legal, hidden[:, None], eid)
    assert r["delta_r2"] > 0.5
    assert r["se_legal_only"].shape == (400,), "per-window SE is the paired unit"


# --- 4. the situation-classifier clause ------------------------------------- #
def test_situation_disjoint_flags_a_shared_symbol_and_a_derived_one():
    assert situation_disjoint(["pooled", "v0"], ["sit_posterior"])["disjoint"]
    assert not situation_disjoint(["pooled", "sit_posterior"],
                                  ["sit_posterior"])["disjoint"]
    d = situation_disjoint(["pooled", "sit_posterior_argmax"], ["sit_posterior"])
    assert not d["disjoint"] and d["name_derived_symbols"]


# --- 5. the combined verdict ------------------------------------------------ #
def test_a_privileged_signal_is_INADMISSIBLE_supplied_and_ADMISSIBLE_as_a_label():
    """The PI's ruling, in code: labels may use the future; inference may not.
    The SAME evidence must therefore produce two different verdicts."""
    h = horizon_disjoint(8, 207, 7, 28)
    inc = {"delta_r2": 0.0996}
    sup = audit_goal_signal(name="vt_oracle", supplied_at_inference=True,
                            horizon=h, increment=inc)
    lab = audit_goal_signal(name="vt_oracle", supplied_at_inference=False,
                            horizon=h, increment=inc)
    assert sup["verdict"] == "INADMISSIBLE" and sup["failures"]
    assert lab["verdict"] == "ADMISSIBLE" and lab["warnings"]


def test_a_guarded_but_still_informative_supplied_signal_still_FAILS():
    """⭐ The finding this instrument had to be able to express: excision makes
    the read window disjoint and does NOT make a supplied signal admissible."""
    v = audit_goal_signal(
        name="vt_guarded_supplied", supplied_at_inference=True,
        horizon=horizon_disjoint(28, 207, 7, 28),
        increment={"delta_r2": 0.0996})
    assert v["checks"]["horizon"]["disjoint"] is True
    assert v["verdict"] == "INADMISSIBLE"
    assert any("PRIVILEGED INCREMENT" in f for f in v["failures"])


def test_a_predicted_goal_from_legal_inputs_is_ADMISSIBLE():
    v = audit_goal_signal(
        name="vt_pred", supplied_at_inference=False,
        echo=echo_score(np.arange(200) % 4, np.arange(200) % 7),
        horizon=horizon_disjoint(28, 207, 7, 28),
        increment={"delta_r2": 0.05},
        provenance=situation_disjoint(["pooled", "v0"], ["sit_posterior"]))
    assert v["verdict"] == "ADMISSIBLE" and not v["failures"]


def test_supplied_at_inference_has_no_default():
    with pytest.raises(TypeError):
        audit_goal_signal(name="x")          # type: ignore[call-arg]
