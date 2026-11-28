"""The K1 degeneracy guard — pinned in BOTH directions.

⛔ WHY BOTH DIRECTIONS, EXPLICITLY. This programme built one useless guard in
each direction inside 24 hours:

* **C92** — a ridge that penalised its own intercept made no-signal arms **FAIL
  by construction** (predictions collapsed toward zero, never toward the mean).
* **C97** — the repair for C92 made no-signal arms **PASS by construction** (a
  fully-shrunk repaired ridge *is* the train MEAN, while C-CONST is the train
  MEDIAN, so on a skewed target K1 becomes a mean-vs-median contest and a pure
  ``torch.randn`` null "PASSES" ``n_agents_all`` at −1.884 with ``pred_sd``
  0.715 against ``gt_sd`` 46.459).

⇒ C95's rule: *when you loosen a criterion, test the direction you were not
trying to fix.* **A guard that rejects everything is as useless as one that
rejects nothing**, so this file pins (a) that a degenerate arm IS caught, and
(b) that a genuinely-signalled arm is NOT — weak signal included, because
rejecting weak-but-real readouts is the failure mode a degeneracy guard invites.

⛔ T0-DIAGNOSTIC fixtures throughout. Nothing here is a driving number.
"""
from pathlib import Path

import numpy as np
import pytest

from taniteval.degeneracy import (SD_RATIO_FLAT_FLOOR, k1_guard,
                                  screen_banked_k1)

N_EP = 40                      # the programme's real val-episode count
PER_EP = 60
N_BOOT = 400                   # tests only; production paths use 2000


def _episodes(n_ep=N_EP, per_ep=PER_EP):
    return np.repeat(np.arange(n_ep), per_ep)


def _skewed_targets(seed=0, n_ep=N_EP, per_ep=PER_EP):
    """A RIGHT-SKEWED target with a train/eval shift — C97's actual shape.

    ``n_agents_all`` there had train median 34.0 against train mean 62.8, and on
    the eval episodes the train MEAN was the better MAE constant (36.051 vs
    37.936). That only happens with a train/eval location shift, so the fixture
    builds one rather than pretending a same-distribution draw would do it.
    """
    rng = np.random.default_rng(seed)
    y_tr = rng.lognormal(mean=np.log(34.0), sigma=1.05, size=4000)
    # eval episodes sit HIGHER, so the train mean lands nearer their centre
    eid = _episodes(n_ep, per_ep)
    y_ev = np.concatenate([
        rng.lognormal(mean=np.log(rng.uniform(50.0, 78.0)), sigma=0.55,
                      size=per_ep)
        for _ in range(n_ep)])
    return y_tr, y_ev, eid


# --------------------------------------------------------------------------- #
# DIRECTION (a) — a DEGENERATE arm must be CAUGHT                              #
# --------------------------------------------------------------------------- #
def test_the_C97_case_a_flat_line_PASSES_K1_and_the_guard_CATCHES_it():
    """⭐ The whole reason this module exists, reproduced end to end."""
    y_tr, y_ev, eid = _skewed_targets(seed=1)
    c_med, c_mean = float(np.median(y_tr)), float(y_tr.mean())
    rng = np.random.default_rng(7)
    # a fully-shrunk repaired ridge: the train MEAN plus float-level wobble
    pred = c_mean + rng.normal(0.0, 0.715, size=y_ev.size)

    g = k1_guard(pred, y_ev, eid, c_med, n_boot=N_BOOT, c_mean=c_mean)

    # 1. the defect is real: K1 alone declares a PASS for a zero-information arm
    assert g["K1_PASSES"], (
        "fixture is not reproducing C97 — K1 must PASS for the flat line, "
        f"got K1={g['K1_delta']} separated={g['K1_separated']}")
    # 2. the guard catches it
    assert g["guard_verdict"] in ("DEGENERATE-CONSTANT", "CONSTANT-OFFSET-ONLY")
    assert not g["K1_quotable_as_latent_evidence"]
    # 3. and it says WHY: essentially none of K1 is latent-attributable
    assert abs(g["K1B_delta"]) < 0.05 * abs(g["K1_delta"]), (
        "a flat line has no latent-attributable component")
    assert g["sd_ratio"] < SD_RATIO_FLAT_FLOOR


def test_the_guard_catches_a_spurious_separated_FAIL_too_not_only_a_PASS():
    """⚠️ DIRECTION SYMMETRY — C92 biased FAILs, C97 biases PASSes.

    A guard that watched only PASSes would be the next member of this family.
    Same flat-line arm, a target where the train MEAN is the WORSE constant:
    K1 now separates as a FAIL, and it is still a fact about which constant.
    """
    rng = np.random.default_rng(3)
    eid = _episodes()
    y_tr = rng.lognormal(mean=np.log(34.0), sigma=1.05, size=4000)
    y_ev = rng.lognormal(mean=np.log(34.0), sigma=1.05, size=eid.size)
    c_med, c_mean = float(np.median(y_tr)), float(y_tr.mean())
    pred = c_mean + rng.normal(0.0, 0.7, size=y_ev.size)

    g = k1_guard(pred, y_ev, eid, c_med, n_boot=N_BOOT, c_mean=c_mean)

    assert g["K1_separated"] and g["K1_delta"] > 0, (
        "fixture must produce a separated FAIL for the flat line")
    assert g["guard_verdict"] in ("DEGENERATE-CONSTANT", "CONSTANT-OFFSET-ONLY")
    assert not g["K1_quotable_as_latent_evidence"]


def test_a_constant_shifted_arm_with_real_spread_is_still_flagged_if_K1B_dies():
    """CONSTANT-OFFSET-ONLY: spread large enough to clear the flat-line floor,
    but the spread is pure NOISE, so it buys nothing over the arm's own mean."""
    y_tr, y_ev, eid = _skewed_targets(seed=5)
    c_med, c_mean = float(np.median(y_tr)), float(y_tr.mean())
    rng = np.random.default_rng(11)
    pred = c_mean + rng.normal(0.0, 0.35 * y_ev.std(), size=y_ev.size)

    g = k1_guard(pred, y_ev, eid, c_med, n_boot=N_BOOT, c_mean=c_mean)

    assert g["sd_ratio"] > SD_RATIO_FLAT_FLOOR, "fixture must clear layer 3"
    assert not g["K1B_PASSES"], "noise spread must not beat the arm's own mean"
    assert g["guard_verdict"] != "OK"
    assert not g["K1_quotable_as_latent_evidence"]


# --------------------------------------------------------------------------- #
# DIRECTION (b) — a GENUINELY-SIGNALLED arm must NOT be caught                 #
# --------------------------------------------------------------------------- #
def test_a_strong_readout_is_NOT_caught():
    """The GT-ORACLE-DIRECT shape: r +0.9932, err 0.580 m against gt_sd 6.200."""
    rng = np.random.default_rng(2)
    eid = _episodes()
    y_tr = rng.normal(15.0, 6.2, size=4000)
    y_ev = rng.normal(15.0, 6.2, size=eid.size)
    pred = y_ev + rng.normal(0.0, 0.73, size=y_ev.size)     # r ~ +0.993

    g = k1_guard(pred, y_ev, eid, float(np.median(y_tr)), n_boot=N_BOOT)

    assert g["K1_PASSES"]
    assert g["guard_verdict"] == "OK", g["guard_note"]
    assert g["K1_quotable_as_latent_evidence"]
    assert g["K1B_PASSES"]


def test_a_WEAK_but_genuine_readout_is_NOT_caught():
    """⭐ The false-positive direction that matters most.

    A guard tuned to kill flat lines will happily kill a correctly-SHRUNK weak
    readout, which looks flat-ish by construction — and the v6 arms are weak.
    A shrunk readout at r ~ 0.5 keeps real per-window skill and must survive.
    """
    rng = np.random.default_rng(4)
    eid = _episodes()
    y_tr = rng.normal(15.0, 6.2, size=4000)
    y_ev = rng.normal(15.0, 6.2, size=eid.size)
    pred = 15.0 + 0.5 * (y_ev - 15.0) + rng.normal(0.0, 1.0, size=y_ev.size)

    g = k1_guard(pred, y_ev, eid, float(np.median(y_tr)), n_boot=N_BOOT)

    assert g["sd_ratio"] > SD_RATIO_FLAT_FLOOR
    assert g["K1_PASSES"], "a shrunk r~0.5 readout does beat the constant"
    assert g["guard_verdict"] == "OK", g["guard_note"]
    assert g["K1_quotable_as_latent_evidence"]


def test_a_HEAVY_TAILED_target_does_not_turn_a_real_readout_into_a_false_alarm():
    """⛔ THE FALSE-POSITIVE THIS GUARD'S FIRST DRAFT WOULD HAVE MANUFACTURED.

    ``sd_ratio`` compares against ``gt_sd``, which a few extreme windows inflate
    without making the target harder in the bulk. ``n_agents_all`` is exactly
    that shape (gt_sd 46.459 against a median of 34.0). A readout that tracks
    the bulk can therefore look "flat" (ratio < 5 %) while genuinely beating its
    own mean, paired and separated.

    ⇒ K1B must OVERRIDE the flat-line label. If this test ever fails, the layer
    ordering has been inverted back and the guard has become the
    rejects-everything kind.
    """
    rng = np.random.default_rng(31)
    eid = _episodes()
    n = eid.size
    y = rng.normal(10.0, 1.0, size=n)
    tail = rng.random(n) < 0.03                     # 3 % extreme windows
    y[tail] = rng.normal(5000.0, 50.0, size=int(tail.sum()))
    pred = 10.0 + 0.9 * (np.clip(y, 0, 20) - 10.0)  # tracks the BULK only

    g = k1_guard(pred, y, eid, 10.0, n_boot=N_BOOT)

    assert g["sd_ratio"] < SD_RATIO_FLAT_FLOOR, (
        f"fixture must look flat, got sd_ratio={g['sd_ratio']}")
    assert g["K1B_separated"] and g["K1B_delta"] < 0, "the skill is real"
    assert g["guard_verdict"] == "OK", (
        "a flat-LOOKING readout that beats its own mean is NOT degenerate — "
        f"got {g['guard_verdict']}: {g['guard_note']}")
    assert g["K1_quotable_as_latent_evidence"]


def test_the_guard_does_not_reject_everything_a_sweep_over_signal_strength():
    """⛔ THE ANTI-'REJECTS-EVERYTHING' PIN.

    As the readout's correlation rises from 0 to 1 the verdict must MOVE from
    caught to OK, and stay OK. A guard that never says OK is as useless as one
    that never says anything else.
    """
    rng = np.random.default_rng(6)
    eid = _episodes()
    y_ev = rng.normal(15.0, 6.2, size=eid.size)
    c_med = 15.0
    verdicts = {}
    for beta in (0.0, 0.15, 0.35, 0.6, 0.9):
        pred = 15.0 + beta * (y_ev - 15.0) + rng.normal(0.0, 0.8, size=y_ev.size)
        verdicts[beta] = k1_guard(pred, y_ev, eid, c_med,
                                  n_boot=N_BOOT)["guard_verdict"]
    assert verdicts[0.0] != "OK", verdicts
    assert verdicts[0.6] == "OK" and verdicts[0.9] == "OK", verdicts


# --------------------------------------------------------------------------- #
# The algebra the verdict rests on                                            #
# --------------------------------------------------------------------------- #
def test_the_decomposition_K1_equals_K1B_plus_K1C_is_EXACT():
    y_tr, y_ev, eid = _skewed_targets(seed=8)
    rng = np.random.default_rng(9)
    pred = 40.0 + 0.4 * (y_ev - 40.0) + rng.normal(0.0, 3.0, size=y_ev.size)
    g = k1_guard(pred, y_ev, eid, float(np.median(y_tr)), n_boot=N_BOOT)
    assert abs(g["decomposition_residual"]) < 1e-9, g["decomposition_residual"]
    assert abs(g["K1_delta"] - (g["K1B_delta"] + g["K1C_delta"])) < 1e-5


@pytest.mark.parametrize("seed", range(12))
def test_the_bound_is_a_THEOREM_abs_K1B_never_exceeds_pred_mad(seed):
    """|K1B| <= mean|pred - mean(pred)| <= pred_sd, for ANY pred and y.

    Reverse triangle inequality per window, then Jensen. Fuzzed, because layer 1
    (the zero-compute screen over 214 banked rows) is only valid if this holds.
    """
    rng = np.random.default_rng(seed)
    n = 600
    eid = np.repeat(np.arange(20), n // 20)
    y = rng.lognormal(mean=rng.uniform(0, 4), sigma=rng.uniform(0.2, 1.5), size=n)
    pred = rng.normal(rng.uniform(-50, 50), rng.uniform(0.01, 40), size=n)
    g = k1_guard(pred, y, eid, float(np.median(y)) + rng.normal(0, 5),
                 n_boot=50)
    assert g["bound_holds"]
    assert abs(g["K1B_delta"]) <= g["pred_mad"] + 1e-6
    assert g["pred_mad"] <= g["pred_sd"] + 1e-9


def test_K1B_is_INVARIANT_to_the_choice_of_C_CONST():
    """⭐ Why the mean-vs-median question is dissolved rather than adjudicated.

    K1 moves when C-CONST moves; K1B does not. So the guard's decision cannot be
    argued into or out of existence by picking a different constant.
    """
    y_tr, y_ev, eid = _skewed_targets(seed=13)
    rng = np.random.default_rng(14)
    pred = 45.0 + 0.5 * (y_ev - 45.0) + rng.normal(0.0, 4.0, size=y_ev.size)
    a = k1_guard(pred, y_ev, eid, float(np.median(y_tr)), n_boot=N_BOOT)
    b = k1_guard(pred, y_ev, eid, float(y_tr.mean()), n_boot=N_BOOT)
    assert abs(a["K1_delta"] - b["K1_delta"]) > 1e-6, "fixture: constants differ"
    assert abs(a["K1B_delta"] - b["K1B_delta"]) < 1e-12
    assert a["K1B_separated"] == b["K1B_separated"]


# --------------------------------------------------------------------------- #
# LAYER 1 — the zero-compute screen, on the REAL banked numbers                #
# --------------------------------------------------------------------------- #
def test_layer1_screen_flags_the_REAL_C97_row():
    """C97, verbatim: n_agents_all, K1 −1.884, pred_sd 0.715, gt_sd 46.459."""
    s = screen_banked_k1(-1.884, 0.715, 46.459, k1_separated=True)
    assert s["k1_exceeds_own_spread"], "pred_sd 0.715 < |K1| 1.884 is provable"
    assert s["flat_line"]
    assert s["min_constant_component"] == pytest.approx(1.169, abs=1e-3)


def test_layer1_screen_does_NOT_flag_the_REAL_GT_oracle_row():
    """GT-ORACLE-DIRECT repaired: K1 −4.553, pred_sd 0.987 x gt_sd 6.200."""
    s = screen_banked_k1(-4.553, 0.987 * 6.200, 6.200, k1_separated=True)
    assert not s["k1_exceeds_own_spread"]
    assert not s["flat_line"]
    assert s["min_constant_component"] == 0.0


def test_layer1_screen_is_only_a_SCREEN_and_says_so():
    """Passing layer 1 must not read as 'attributable' — that needs layer 2."""
    s = screen_banked_k1(-0.10, 5.0, 6.0)
    assert not s["k1_exceeds_own_spread"]
    assert "layer 2 still required" in s["screen_verdict"]


def test_a_non_separated_K1_has_NO_VERDICT_TO_GUARD():
    rng = np.random.default_rng(21)
    eid = _episodes()
    y_ev = rng.normal(15.0, 6.2, size=eid.size)
    pred = 15.0 + rng.normal(0.0, 0.02, size=y_ev.size)
    g = k1_guard(pred, y_ev, eid, 15.0, n_boot=N_BOOT)
    assert g["guard_verdict"] == "NO-VERDICT-TO-GUARD"
    assert not g["K1_quotable_as_latent_evidence"]


# --------------------------------------------------------------------------- #
# The guard must be WIRED IN, not merely available                            #
# --------------------------------------------------------------------------- #
_HUB = (Path(__file__).resolve().parents[2] / "TanitAD Research Hub"
        / "Architecture & Inference" / "Implementation" / "incoming")
_PRODUCERS = {
    "pc6_linear_readout.py":
        _HUB / "2026-08-17-probe-positive-control" / "code"
        / "pc6_linear_readout.py",
    "ll1_ladder.py":
        _HUB / "2026-08-17-latent-linear-ladder" / "code" / "ll1_ladder.py",
}


@pytest.mark.parametrize("name", sorted(_PRODUCERS))
def test_every_ridge_verdict_producer_actually_CALLS_the_guard(name):
    """⛔ AN INSTRUMENT NOBODY CALLS IS NOT AN INSTRUMENT.

    This programme has left an orthogonality probe unmerged for 10 days and a
    LAL-v2 anticipation for 12, each because "it exists" was mistaken for "it
    runs". Both producers of a K1 verdict must emit the guard beside it, or the
    next reader quotes a bare K1 exactly as C97 did.
    """
    path = _PRODUCERS[name]
    if not path.exists():                                    # pragma: no cover
        pytest.skip(f"producer not present: {path}")
    src = path.read_text(encoding="utf-8")
    assert "from taniteval.degeneracy import k1_guard" in src, (
        f"{name} must import the guard from taniteval, not re-derive it")
    assert "k1_guard(" in src, f"{name} imports the guard but never calls it"
    assert '"k1_guard"' in src, f"{name} must emit the guard block in its JSON"
    assert "K1_PASSES_GUARDED" in src, (
        f"{name} must emit a GUARDED pass flag beside the raw K1_PASSES")


def test_the_guard_module_names_the_retractions_that_earned_it():
    """A fix whose reason is not written at the code gets rediscovered."""
    import taniteval.degeneracy as mod
    doc = mod.__doc__ or ""
    for token in ("C92", "C97", "C95", "K1B", "MEDIAN"):
        assert token in doc, f"degeneracy docstring must name {token!r}"


def test_the_guard_never_uses_the_forbidden_estimator():
    y_tr, y_ev, eid = _skewed_targets(seed=17)
    g = k1_guard(np.full(y_ev.size, 40.0), y_ev, eid, float(np.median(y_tr)),
                 n_boot=50)
    assert g["estimator"] == "taniteval.ci.paired_episode_cluster_bootstrap"
    assert g["forbidden"] == "overlapping_holdout_se"
    assert g["eval_tier"] == "T0-DIAGNOSTIC"
