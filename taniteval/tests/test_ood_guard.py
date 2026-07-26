"""``taniteval.ood`` — the OOD guard that can no longer hide its own saturation.

⚠️ THE DEFECT (MEASURED, 2026-07-26)
-------------------------------------
``OODMap.ratio_arr`` maps a deviation to an ADE-ratio through ``np.interp``,
which **CLAMPS** at ``|dlat| = 3.0 m`` / ``|dyaw| = 12 deg``. Beyond the envelope
the ratio **SATURATES**, so:

* every long-horizon OOD ratio the program has quoted is a **LOWER BOUND**, and
* the ``ratio > ~1.5x`` criterion **structurally cannot fire** out of envelope —
  it is uninformative exactly where it matters most.

At the flagship-v4 30 k gate the block read **1.2741** ("under 1.5") while
**54.63 % of steps exceeded 3 m** and **90.24 % of windows left the envelope**,
and emitted *"within the measured envelope on average"*.

E1a's rule was always the DISJUNCTION (``e1a_horizon.py:28-30``): ratio > ~1.5x
**OR steps leave the measured envelope**. Only the ratio half was implemented.

THE LOAD-BEARING TESTS are the two impossibility pins:
:func:`TestImpossibleToLaunder::test_a_majority_outside_cannot_read_as_in_envelope`
and ``test_the_30k_numbers_now_read_EXTRAPOLATION`` — fed the exact MEASURED 30 k
numbers, the guard must refuse the string that survived.

Pure-Python + numpy on committed artifacts. No torch model, no GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from taniteval import ood as _ood

_REPO = Path(__file__).resolve().parents[2]
_E1A = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
        / "Implementation" / "incoming" / "2026-07-25-closedloop-horizon-and-shift"
        / "e1a_horizon_heldout44_K185.json")
_GATE = (_REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
         / "incoming" / "2026-07-26-v4-30k-gate")
_CORRIDOR_V4 = _GATE / "coprimary" / "corridor_v4_30k_K185.json"

# MEASURED at the 30 k gate (GATE_30K_RESULTS.md 6.4). Named so a drift goes red.
GATE_RATIO = 1.2741
GATE_FRAC_STEPS_LAT = 0.5463
GATE_FRAC_WINDOWS = 0.9024
# MEASURED in E1a's own K=185 artifact.
E1A_FRAC_STEPS_LAT = 0.5281
E1A_FRAC_WINDOWS = 0.9070

LEGACY_STRING = "within the measured envelope on average"


def _env(lat, yaw):
    return _ood.envelope_fractions(np.asarray(lat, float), np.asarray(yaw, float))


# =========================================================================== #
# clause 2 is FIRST CLASS                                                     #
# =========================================================================== #
class TestEnvelopeFractions:
    def test_every_fraction_names_its_own_denominator(self):
        lat = np.array([[0.0, 4.0, 1.0], [0.5, 0.5, 0.5]])
        yaw = np.array([[1.0, 1.0, 20.0], [1.0, 1.0, 1.0]])
        f = _env(lat, yaw)
        assert f["frac_steps_lat_over_3m"] == pytest.approx(1 / 6, abs=1e-5)
        assert f["frac_steps_yaw_over_12deg"] == pytest.approx(1 / 6, abs=1e-5)
        assert f["frac_steps_any"] == pytest.approx(2 / 6, abs=1e-5)
        assert f["frac_windows_any_step_out_of_envelope"] == pytest.approx(0.5)

    def test_a_clean_rollout_reports_zero(self):
        f = _env(np.zeros((4, 10)), np.zeros((4, 10)))
        assert f["frac_steps_any"] == 0.0
        assert f["frac_windows_any_step_out_of_envelope"] == 0.0

    def test_the_envelope_constants_are_the_P1_MEASURED_ones(self):
        from taniteval.corridor import ENV_LAT_MAX, ENV_YAW_MAX
        assert (_ood.ENV_LAT_MAX, _ood.ENV_YAW_MAX) == (ENV_LAT_MAX, ENV_YAW_MAX)
        assert (_ood.ENV_LAT_MAX, _ood.ENV_YAW_MAX) == (3.0, 12.0)

    def test_you_cannot_get_a_ratio_without_the_fractions(self):
        """`ratio_and_fractions` is the intended entry point: the ratio never
        travels alone."""
        m = _fake_map()
        r, f = m.ratio_and_fractions(np.array([[0.5, 5.0]]), np.array([[1.0, 1.0]]))
        assert r.shape == (1, 2)
        assert f["frac_steps_lat_over_3m"] == 0.5


def _fake_map():
    """A P1-shaped envelope JSON: baseline 0.40, rising to 3 m / 12 deg."""
    return _ood.OODMap({
        "baseline_real_frames": {"mean": 0.40},
        "conditions": {
            "lat": [{"amount": 0.0, "ade2s_ci": {"mean": 0.40}},
                    {"amount": 3.0, "ade2s_ci": {"mean": 0.47}}],
            "yaw": [{"amount": 0.0, "ade2s_ci": {"mean": 0.40}},
                    {"amount": 12.0, "ade2s_ci": {"mean": 0.46}}]}})


# =========================================================================== #
# the SATURATION is real, and is declared                                     #
# =========================================================================== #
class TestSaturation:
    def test_np_interp_clamps_so_the_ratio_stops_growing(self):
        """The mechanism itself: 3 m and 300 m map to the SAME ratio."""
        m = _fake_map()
        at_edge = float(m.ratio_arr(np.array([3.0]), np.array([0.0]))[0])
        far_out = float(m.ratio_arr(np.array([300.0]), np.array([0.0]))[0])
        assert far_out == at_edge          # SATURATED — a 100x deviation, same number
        assert far_out < _ood.RATIO_EXTRAPOLATION_X   # ...and it never reaches 1.5

    def test_the_ratio_criterion_cannot_fire_out_of_envelope(self):
        """Why the 1.5x bar is uninformative exactly where it matters: even an
        absurd excursion cannot push the clamped ratio past it."""
        m = _fake_map()
        lat = np.full((4, 50), 500.0)
        yaw = np.full((4, 50), 500.0)
        node = _ood.verdict(lat, yaw, ["a", "a", "b", "b"], m, 185, n_boot=200)
        assert node["criterion_1_ratio_over_1p5"]["fires"] is False
        assert node["criterion_1_ratio_over_1p5"]["informative"] is False
        # ...and the OTHER clause carries the verdict.
        assert node["criterion_2_steps_outside_measured_envelope"]["fires"] is True
        assert node["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_EXTRAPOLATION
        assert node["ratio_is_lower_bound"] is True

    def test_a_clean_rollout_is_still_allowed_to_say_MEASUREMENT(self):
        """The fix must not make every verdict pessimistic."""
        m = _fake_map()
        node = _ood.verdict(np.full((4, 20), 0.2), np.full((4, 20), 1.0),
                            ["a", "a", "b", "b"], m, 20, n_boot=200)
        assert node["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_MEASUREMENT
        assert node["ratio_is_lower_bound"] is False
        assert node["criterion_1_ratio_over_1p5"]["informative"] is True


# =========================================================================== #
# ⭐ IMPOSSIBLE TO LAUNDER                                                     #
# =========================================================================== #
class TestImpossibleToLaunder:
    def test_a_majority_outside_cannot_read_as_in_envelope(self):
        """**Make it impossible to emit "within the measured envelope" when a
        majority of steps are outside it.**"""
        node = {"EXTRAPOLATION_VERDICT": LEGACY_STRING,
                "EXTRAPOLATION_frac_steps_lat_over_3m": GATE_FRAC_STEPS_LAT,
                "EXTRAPOLATION_frac_windows_any_step_out_of_envelope":
                    GATE_FRAC_WINDOWS}
        with pytest.raises(_ood.EnvelopeVerdictError, match="MEASUREMENT"):
            _ood.assert_envelope_verdict_consistent(node)

    def test_even_ONE_step_outside_forbids_a_MEASUREMENT_verdict(self):
        node = {"EXTRAPOLATION_VERDICT": _ood.VERDICT_MEASUREMENT,
                "EXTRAPOLATION_frac_steps_any": 0.0001,
                "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": 0.01}
        with pytest.raises(_ood.EnvelopeVerdictError):
            _ood.assert_envelope_verdict_consistent(node)

    def test_a_majority_outside_forbids_even_PARTIAL(self):
        node = {"EXTRAPOLATION_VERDICT": _ood.VERDICT_PARTIAL,
                "EXTRAPOLATION_frac_steps_any": 0.6,
                "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": 0.9}
        with pytest.raises(_ood.EnvelopeVerdictError, match="MAJORITY"):
            _ood.assert_envelope_verdict_consistent(node)

    def test_a_ratio_with_steps_outside_must_declare_its_saturation(self):
        """A saturating estimator must declare its own saturation."""
        node = {"EXTRAPOLATION_VERDICT": _ood.VERDICT_EXTRAPOLATION,
                "ood_peak_ratio": {"mean": 1.27},
                "EXTRAPOLATION_frac_steps_any": 0.55,
                "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": 0.90}
        with pytest.raises(_ood.EnvelopeVerdictError, match="lower_bound"):
            _ood.assert_envelope_verdict_consistent(node)
        node["ratio_is_lower_bound"] = True
        assert _ood.assert_envelope_verdict_consistent(node) is node

    def test_verdict_cannot_be_constructed_inconsistently(self):
        """The guard runs INSIDE `verdict`, so it cannot be forgotten."""
        m = _fake_map()
        node = _ood.verdict(np.full((4, 20), 9.0), np.zeros((4, 20)),
                            ["a", "a", "b", "b"], m, 185, n_boot=200)
        assert node["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_EXTRAPOLATION


# =========================================================================== #
# the verdict CLASS, not the wording                                          #
# =========================================================================== #
class TestVerdictClass:
    @pytest.mark.parametrize("s,cls", [
        (LEGACY_STRING, _ood.CLASS_MEASUREMENT),
        (_ood.VERDICT_MEASUREMENT, _ood.CLASS_MEASUREMENT),
        (_ood.VERDICT_PARTIAL, _ood.CLASS_PARTIAL),
        (_ood.VERDICT_EXTRAPOLATION, _ood.CLASS_EXTRAPOLATION),
        ("PARTIAL EXTRAPOLATION — a minority of windows leave the envelope",
         _ood.CLASS_PARTIAL),
        (None, _ood.CLASS_UNKNOWN), ("", _ood.CLASS_UNKNOWN)])
    def test_class_of(self, s, cls):
        assert _ood.verdict_class(s) == cls

    def test_the_legacy_string_classifies_as_a_MEASUREMENT_claim(self):
        """It asserts the envelope held — which is exactly why it was wrong."""
        assert _ood.verdict_class(LEGACY_STRING) == _ood.CLASS_MEASUREMENT


# =========================================================================== #
# re-adjudication from fields the emitters ALREADY wrote                      #
# =========================================================================== #
class TestReadjudicate:
    def test_the_30k_numbers_now_read_EXTRAPOLATION(self):
        """⭐ The exact MEASURED 30 k node, re-adjudicated."""
        fixed = _ood.readjudicate({
            "EXTRAPOLATION_VERDICT": LEGACY_STRING,
            "ood_peak_ratio": {"mean": GATE_RATIO},
            "EXTRAPOLATION_frac_steps_lat_over_3m": GATE_FRAC_STEPS_LAT,
            "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": GATE_FRAC_WINDOWS})
        assert fixed["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_EXTRAPOLATION
        assert fixed["_class_before"] == _ood.CLASS_MEASUREMENT
        assert fixed["_class_after"] == _ood.CLASS_EXTRAPOLATION
        assert fixed["_class_changed"] is True
        assert fixed["ratio_is_lower_bound"] is True
        assert fixed["criterion_1_ratio_over_1p5"]["fires"] is False
        assert fixed["criterion_2_steps_outside_measured_envelope"]["fires"] is True

    def test_it_does_not_mutate_the_input(self):
        node = {"EXTRAPOLATION_VERDICT": LEGACY_STRING,
                "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": 0.9,
                "EXTRAPOLATION_frac_steps_any": 0.6}
        _ood.readjudicate(node)
        assert node["EXTRAPOLATION_VERDICT"] == LEGACY_STRING

    def test_a_rewording_is_not_counted_as_a_retraction(self):
        """Comparing strings would make every re-wording look like a class
        flip; only the CLASS is load-bearing."""
        fixed = _ood.readjudicate({
            "EXTRAPOLATION_VERDICT":
                "PARTIAL EXTRAPOLATION — a minority of windows leave the envelope",
            "EXTRAPOLATION_frac_steps_any": 0.05,
            "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": 0.10})
        assert fixed["_class_changed"] is False


# =========================================================================== #
# against the COMMITTED artifacts                                             #
# =========================================================================== #
@pytest.mark.skipif(not _E1A.exists(), reason="E1a artifact absent")
def test_e1a_K185_is_extrapolation_not_in_distribution():
    """GATE_PROTOCOL 0.1, RETRACTION_LOG C6 and LOOP_STATE all say the K=185
    result is "genuine in-distribution failure, not extrapolation" because the
    ratio stays <= 1.30. E1a's OWN artifact refutes that."""
    d = json.loads(_E1A.read_text(encoding="utf-8"))
    node = d["all_windows"]["185"]["overall"]
    assert node["EXTRAPOLATION_frac_steps_lat_over_3m"] == E1A_FRAC_STEPS_LAT
    assert node["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"] == E1A_FRAC_WINDOWS
    assert node["ood_peak_ratio"]["mean"] < 1.30      # the ratio DOES stay low
    fixed = _ood.readjudicate(node)                   # ...and it does not matter
    assert fixed["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_EXTRAPOLATION
    assert fixed["criterion_1_ratio_over_1p5"]["informative"] is False


@pytest.mark.skipif(not _CORRIDOR_V4.exists(), reason="30 k corridor artifact absent")
def test_the_committed_v4_artifact_still_carries_a_false_string():
    """The sweep's finding, pinned: `fix_ood_verdict.py` corrected the
    `all_windows` nodes of this file but NOT its `paired_common_start` nodes,
    which still say the envelope held at K=185."""
    d = json.loads(_CORRIDOR_V4.read_text(encoding="utf-8"))
    stale = d["paired_common_start"]["185"]["ood"]["overall"]
    assert _ood.verdict_class(stale["EXTRAPOLATION_VERDICT"]) == _ood.CLASS_MEASUREMENT
    fixed = _ood.readjudicate(stale)
    assert fixed["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_EXTRAPOLATION
    # ...while all_windows WAS corrected in the same file.
    good = d["all_windows"]["185"]["ood"]["overall"]
    assert _ood.verdict_class(good["EXTRAPOLATION_VERDICT"]) == _ood.CLASS_EXTRAPOLATION


# =========================================================================== #
# the run_gate MIRROR may not drift                                           #
# =========================================================================== #
def test_run_gate_mirror_agrees():
    """`stack` cannot import `taniteval`, so `run_gate._ood_verdict` MIRRORS this
    rule. This test is the only thing keeping the two from drifting — the exact
    failure mode `closedloop.py` warns about ("a second copy of the rule
    drifting from the first, which is how overlapping_holdout_se survived")."""
    sys.path.insert(0, str(_REPO / "stack" / "scripts"))
    import run_gate as rg
    assert (rg.ENV_LAT_MAX, rg.ENV_YAW_MAX) == (_ood.ENV_LAT_MAX, _ood.ENV_YAW_MAX)
    assert rg.RATIO_EXTRAPOLATION_X == _ood.RATIO_EXTRAPOLATION_X
    assert rg.OOD_MAJORITY_FRAC == _ood.MAJORITY_FRAC
    for peak, fs, fw in [(GATE_RATIO, GATE_FRAC_STEPS_LAT, GATE_FRAC_WINDOWS),
                         (1.6, 0.0, 0.0), (1.0, 0.0, 0.0), (1.1, 0.01, 0.05),
                         (1.2, 0.51, 0.99), (None, 0.0, 0.0), (1.0, 0.49, 0.5)]:
        mine = rg._ood_verdict(peak, fs, fw)
        theirs = _ood.readjudicate({
            "ood_peak_ratio": None if peak is None else {"mean": peak},
            "EXTRAPOLATION_frac_steps_any": fs,
            "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": fw})
        assert (_ood.verdict_class(mine["EXTRAPOLATION_VERDICT"])
                == _ood.verdict_class(theirs["EXTRAPOLATION_VERDICT"])), (peak, fs, fw)
        assert mine["ratio_is_lower_bound"] == theirs["ratio_is_lower_bound"]
