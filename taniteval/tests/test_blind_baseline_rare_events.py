"""FIX 4 — the firewall on RARE-EVENT and VARIABLE-ARITY targets.

THE DEFECT, reproduced from the real failure. ``blind_conditioning_baseline``
returned ``CIRCULAR`` on **all three** situation targets
(``…/2026-07-26-situation-classifier/artifacts/sc_results.json``) while its own
``context_leaks = 0`` refuted that verdict on every one of them:

    roundabout     blind 0.9970  majority 0.9970  real 0.9864  skill  0.0
    intersection   blind 0.9743  majority 0.9743  real 0.8194  skill  0.0
    lane_change    blind 0.9787  majority 0.9788  real 0.9193  skill -7.6e-05

Two degenerate routes, both of raw accuracy on a ~0.003-positive target:
  1. ``blind >= 1 - eps`` fires on the MAJORITY CLASS itself;
  2. ``vision_buys_nothing`` compares ACCURACIES, and a recall-seeking
     rare-event model must lose that comparison to "always predict negative".

**A firewall that returns CIRCULAR on a clean target is worse than no
firewall** — it retires an admissible decision problem.

Separately: the packaged module cannot express AlpaSim S1's variable-arity
option set, and padding it to a fixed arity is a strictly WEAKER attack, i.e. a
LOWER BOUND on the leak — the wrong direction for a firewall to err in.

Every assertion here is driven with input DESIGNED to make the guard fail, and
each repair is checked in BOTH directions: the false positive must go away AND
the real leak must still be caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from taniteval import blind_baseline as BB


def _rare(n=6000, pos_rate=0.0030, seed=7, n_ep=60):
    """A rare-positive target the context knows NOTHING about.

    ``context_leaks`` is 0 by construction — the context is pure noise — so any
    ``CIRCULAR`` verdict here is definitionally wrong.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    ctx = rng.integers(0, 4, n)                    # independent of y
    return ctx, y, [f"ep{i % n_ep:03d}" for i in range(n)]


# ------------------------------------------------------------- route 1 ------
def test_a_rare_target_with_a_NOISE_context_is_no_longer_called_CIRCULAR():
    """The majority class itself clears ``blind >= 1 - eps``.

    MEASURED analogue: ``situation:roundabout`` — blind 0.9970 == majority
    0.9970 at a positive rate of 0.0030, ``deterministic = True``,
    ``context_leaks = False``. The old logic called that CIRCULAR.
    """
    ctx, y, eids = _rare()
    fw = BB.blind_conditioning_baseline({"c": ctx}, y, eids,
                                        problem="rare_noise", n_boot=200)
    # the degeneracy must be REAL, or this test proves nothing
    assert fw["majority_base_rate"]["mean"] >= 1.0 - BB.DETERMINISTIC_EPS
    assert fw["degeneracy_audit"]["deterministic_test_degenerate"] is True
    assert fw["degeneracy_audit"]["leak_test_degenerate"] is True
    assert fw["degeneracy_audit"]["max_possible_blind_skill"] < BB.SKILL_EPS
    # ...and the verdict must not be a libel
    assert fw["verdict"] != "CIRCULAR", fw["summary"]
    assert fw["verdict"] in ("CLEAN", "REFUSED"), fw["summary"]
    assert fw["context_leaks"] is False
    assert fw["statistic"] in ("balanced_accuracy", "none")


# ------------------------------------------------------------- route 2 ------
def test_a_recall_seeking_real_model_no_longer_forces_CIRCULAR():
    """``vision_buys_nothing`` compares ACCURACIES — a comparison a rare-event
    model cannot win.

    MEASURED analogue: ``situation:intersection`` — blind 0.9743 == majority
    0.9743, real 0.8194 => ``vision_buys_nothing = True`` => CIRCULAR, with
    ``context_leaks = False``.
    """
    ctx, y, eids = _rare(pos_rate=0.026, seed=11)
    rng = np.random.default_rng(3)
    # a real model that finds 60 % of the positives at an 18 % FPR: HIGH recall,
    # LOWER accuracy than the majority predictor — the exact shape that broke it
    real = np.where(y == 1, (rng.random(len(y)) < 0.60).astype(int),
                    (rng.random(len(y)) < 0.18).astype(int))
    fw = BB.blind_conditioning_baseline({"c": ctx}, y, eids, real_pred=real,
                                        problem="rare_recall", n_boot=200)
    assert fw["real_model_accuracy"]["mean"] < fw["majority_base_rate"]["mean"], (
        "the setup must reproduce the accuracy inversion, or nothing is tested")
    assert fw["degeneracy_audit"]["vision_buys_nothing_test_degenerate"] is True
    assert fw["vision_buys_nothing"] is False
    assert fw["verdict"] != "CIRCULAR", fw["summary"]
    # the balanced statistic DOES see the real model's genuine recall skill
    assert fw["degeneracy_audit"]["balanced_accuracy_real"] > 0.5 + BB.SKILL_EPS


# ------------------------------------------- the other direction: still bites
def test_the_repair_still_catches_a_REAL_leak_on_a_rare_target():
    """Disarming the degenerate tests must NOT disarm the firewall, or FIX 4
    traded a false positive for a false negative."""
    rng = np.random.default_rng(5)
    n, n_ep = 6000, 60
    ctx = rng.integers(0, 200, n)
    y = (ctx < 1).astype(int)                    # rare AND fully determined
    eids = [f"ep{i % n_ep:03d}" for i in range(n)]
    fw = BB.blind_conditioning_baseline({"c": ctx}, y, eids,
                                        problem="rare_lookup", n_boot=200)
    assert fw["degeneracy_audit"]["raw_accuracy_scale_is_degenerate"] is True
    assert fw["verdict"] == "CIRCULAR", fw["summary"]
    assert fw["admissible"] is False
    assert fw["statistic"] == "balanced_accuracy"


def test_a_balanced_target_is_still_decided_on_RAW_accuracy():
    """The repair must not change behaviour where accuracy was never broken."""
    rng = np.random.default_rng(9)
    n = 1200
    ctx = rng.integers(0, 3, n)
    y = rng.integers(0, 3, n)
    eids = [f"ep{i % 20:02d}" for i in range(n)]
    fw = BB.blind_conditioning_baseline({"c": ctx}, y, eids,
                                        problem="balanced", n_boot=200)
    assert fw["statistic"] == "accuracy"
    assert fw["degeneracy_audit"]["raw_accuracy_scale_is_degenerate"] is False
    assert fw["verdict"] == "CLEAN", fw["summary"]


def test_REFUSED_blocks_registration_without_libelling_the_target():
    assert "REFUSED" in BB.VERDICTS
    assert "UNADJUDICATED" in BB._READ["REFUSED"]
    rec = {"block": BB.BLOCK, "verdict": "REFUSED", "admissible": False,
           "summary": "x", "_read": BB._READ["REFUSED"]}
    with pytest.raises(BB.CircularTarget):
        BB.register_decision_problem("x", target="t", conditioning=["c"],
                                     firewall=rec)


def test_balanced_accuracy_floor_is_1_over_n_class_at_any_imbalance():
    for rate in (0.5, 0.05, 0.003):
        n = 20000
        rng = np.random.default_rng(0)
        y = (rng.random(n) < rate).astype(int)
        const = np.zeros(n, int)                  # the majority predictor
        bal, present = BB.balanced_accuracy(y, const, 2)
        assert present == 2
        assert bal == pytest.approx(0.5, abs=1e-9), rate


# --------------------------------------------------- variable arity (S1) ----
def _options(seed=0, n_groups=90, leak=0.0):
    """Variable-arity decision points: K in 2..5, exactly one chosen each."""
    rng = np.random.default_rng(seed)
    ctx, grp, chosen, eid = [], [], [], []
    for g in range(n_groups):
        k = int(rng.integers(2, 6))
        c = int(rng.integers(0, k))
        for j in range(k):
            ctx.append(1.0 if (leak and j == c) else float(rng.normal()))
            grp.append(f"g{g:03d}")
            chosen.append(j == c)
            eid.append(f"ep{g % 20:02d}")
    return {"feat": np.array(ctx)}, grp, np.array(chosen), eid


def test_fixed_class_entry_point_REFUSES_a_variable_arity_problem():
    """DESIGNED TO FAIL: hand the fixed-class firewall an S1-shaped problem."""
    ctx, y, eids = _rare(n=400, pos_rate=0.3, seed=2, n_ep=20)
    k = np.random.default_rng(0).integers(2, 6, len(y))
    with pytest.raises(BB.FirewallRefused) as e:
        BB.blind_conditioning_baseline({"c": ctx}, y, eids, problem="s1_like",
                                       n_boot=50, n_options=k)
    assert "blind_option_baseline" in str(e.value)
    assert "LOWER BOUND" in str(e.value)
    # a genuinely fixed-arity declaration must still be accepted
    fw = BB.blind_conditioning_baseline({"c": ctx}, y, eids, problem="fixed",
                                        n_boot=50,
                                        n_options=np.full(len(y), 3))
    assert fw["verdict"] in BB.VERDICTS


def test_option_baseline_is_arity_exact_and_CLEAN_on_a_noise_context():
    oc, grp, chosen, eid = _options(seed=1, leak=0.0)
    fw = BB.blind_option_baseline(oc, grp, chosen, eid, problem="s1_noise",
                                  n_boot=200)
    assert fw["arity"]["variable"] is True
    assert fw["arity"]["min"] == 2 and fw["arity"]["max"] == 5
    # chance is mean(1/K_i) — NOT 1/K_max, NOT a majority class
    assert 0.2 < fw["chance_accuracy"]["mean"] < 0.5
    assert fw["verdict"] == "CLEAN", fw["summary"]
    assert fw["blind_vs_chance_paired"]["estimator"] == \
        "paired_episode_cluster_bootstrap"
    assert fw["statistic"] == "option_choice_accuracy"


def test_option_baseline_CATCHES_a_leak_the_padded_attack_would_miss():
    """DESIGNED TO FAIL: the per-option features name the chosen option."""
    oc, grp, chosen, eid = _options(seed=2, leak=1.0)
    fw = BB.blind_option_baseline(oc, grp, chosen, eid, problem="s1_leak",
                                  n_boot=200)
    assert fw["verdict"] == "CIRCULAR", fw["summary"]
    assert fw["admissible"] is False
    assert fw["blind_skill_over_chance"] > BB.SKILL_EPS


def test_option_baseline_refuses_a_one_option_decision_point():
    oc, grp, chosen, eid = _options(seed=3)
    grp = ["g999"] + list(grp[1:])                # a group with a single option
    chosen = np.asarray(chosen).copy()
    chosen[0] = True
    with pytest.raises((BB.FirewallRefused, ValueError)):
        BB.blind_option_baseline(oc, grp, chosen, eid, problem="bad", n_boot=50)


def test_option_baseline_demands_exactly_one_chosen_option_per_group():
    oc, grp, chosen, eid = _options(seed=4)
    chosen = np.asarray(chosen).copy()
    chosen[:] = True                              # every option "chosen"
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        BB.blind_option_baseline(oc, grp, chosen, eid, problem="bad", n_boot=50)
