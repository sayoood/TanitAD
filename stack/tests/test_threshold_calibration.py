"""P4-13 — every threshold constant carries its calibration, or the suite fails.

⛔ THE FAILURE THIS PREVENTS. ``proximity_safe_m = 5.0`` was a round number with
no derivation, and it flags the human driver's OWN future on 34.9 % of windows.
It cost a full RL campaign's worth of nulls before anyone asked what the constant
was calibrated against — because nothing ever required a constant to be justified.

⚠️ THIS SUITE DOES NOT RE-MEASURE. The rates live in ``THRESHOLD_CALIBRATION``,
produced by a probe over a real corpus that CI cannot run. What these tests
enforce is a COVERAGE CONTRACT: a new threshold constant FAILS until somebody
measures it. The value of the audit is not the audit — it is that the next
constant gets one automatically, instead of by another campaign.
"""
import re

import pytest

from tanitad.rl import rewards as RW

#: Module constants that are NOT thresholds against the world — shape/exponent
#: knobs, sampling rates and numerical guards. Each carries WHY, so the
#: exemption list cannot grow silently into a way of dodging the contract.
NOT_A_THRESHOLD = {
    "DT_S": "sampling rate, not a limit",
    "EPS": "numerical guard",
}


def _module_constants():
    """UPPER_CASE numeric constants defined at module level in rewards.py."""
    return {k: v for k, v in vars(RW).items()
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", k)
            and isinstance(v, (int, float)) and not isinstance(v, bool)}


def test_every_registered_constant_has_a_measured_rate_and_an_n():
    for name, rec in RW.THRESHOLD_CALIBRATION.items():
        assert "value" in rec, name
        assert isinstance(rec["flags_human_frac"], float), name
        assert 0.0 <= rec["flags_human_frac"] <= 1.0, name
        assert rec["n"] > 0, f"{name} has no sample size — an unmeasured rate"
        assert rec["verdict"] in {"OK", "REVIEW", "MISCALIBRATED"}, name
        assert rec.get("note"), f"{name} needs a note saying what the rate means"


def test_registry_values_match_the_live_constants():
    """⛔ A registry that drifts from the code it describes is worse than none —
    it certifies a value nobody is using."""
    live = {
        "kappa_max_1pm": RW.KAPPA_MAX_1PM,
        "jerk_max_mps3": RW.JERK_MAX_MPS3,
        "a_max_mps2": RW.A_MAX_MPS2,
        "lat_acc_max_mps2": RW.LAT_ACC_MAX_MPS2,
    }
    for k, v in live.items():
        assert RW.THRESHOLD_CALIBRATION[k]["value"] == pytest.approx(v), (
            f"{k}: registry says {RW.THRESHOLD_CALIBRATION[k]['value']}, "
            f"code says {v}")


def test_ctx_default_thresholds_match_the_code():
    """The ctx-overridable defaults are thresholds too — and they are INLINE
    ``ctx.get(..., default)`` values rather than module constants, which is
    exactly how they escaped every previous audit."""
    src = open(RW.__file__, encoding="utf-8").read()
    for key in ("proximity_safe_m", "target_time_gap_s", "ttc_min_s"):
        rec = RW.THRESHOLD_CALIBRATION[key]
        assert re.search(rf'ctx\.get\(\s*"{key}",\s*{rec["value"]}\s*\)', src), (
            f"{key}: registry says {rec['value']}, but no ctx.get default in "
            f"rewards.py matches it — the registry has drifted from the code")


def test_a_new_module_threshold_constant_must_be_registered():
    """⭐ THE ONE THAT PROTECTS THE FUTURE.

    Adding ``SOME_MAX_MPS2 = 7.0`` to rewards.py without measuring what it flags
    fails HERE, the moment it is introduced — not in a campaign six weeks later.
    """
    registered = {k.upper() for k in RW.THRESHOLD_CALIBRATION}
    unaccounted = sorted(k for k in _module_constants()
                         if k not in NOT_A_THRESHOLD and k not in registered)
    assert not unaccounted, (
        "these module constants are neither registered in "
        "THRESHOLD_CALIBRATION nor listed in NOT_A_THRESHOLD with a reason:\n  "
        + "\n  ".join(unaccounted)
        + "\n⇒ measure what the constant flags on the demonstration "
          "distribution (see THRESHOLD_CALIBRATION_SOURCE), then register it. "
          "A threshold nobody calibrated is how proximity_safe_m cost a whole "
          "campaign.")


def test_miscalibrated_constants_name_their_successor():
    """A constant known to be wrong must point at the experiment that fixes it,
    or the finding decays into a comment nobody acts on."""
    for name, rec in RW.THRESHOLD_CALIBRATION.items():
        if rec["verdict"] == "MISCALIBRATED":
            assert "PREREG" in rec["note"], (
                f"{name} is MISCALIBRATED but names no pre-registration")


def test_the_source_artifact_is_named_with_its_caveat():
    s = RW.THRESHOLD_CALIBRATION_SOURCE
    assert "threshold_sweep" in s and "MEASURED" in s
    assert "NON-PARITY" in s, "the corpus caveat must travel with the rates"
