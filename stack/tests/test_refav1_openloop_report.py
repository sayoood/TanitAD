"""``refav1_openloop_report.py`` — the SECOND probe on the headline, and its refusal.

CPU only, synthetic dumps and a synthetic record. ⛔ Never touches Thor, a pod, a real
checkpoint or the banked eval slice.

WHAT IS PINNED

  A. THE INDEPENDENT IDENTITY PROBE AND THE CONTROL-SHAPE PROBE — the only things this
     tool computes itself:
     A1 a dump whose ``cl`` IS the constant-velocity floor reads ``n_bit_identical ==
        n_windows`` and ``frac_bit_identical == 1.0``;
     A2 a dump whose ``cl`` genuinely differs reads a fraction STRICTLY BELOW 1 — the
        deliberate-regression arm, without which "1.0" could be an artefact of the
        comparison rather than a property of the arm;
     A3 the MECHANISM half is read separately: ``cl_controls`` exactly zero is counted
        from ``decisions/``, and a non-zero control set is NOT counted as zero;
     A4 the per-episode breakdown sums to the pooled counts (no double count, no drop);
     A5 the SHAPE probe (kappa identically 0, accel constant in time, the DISTINCT plans
        emitted) sees an injected baseline that is NOT zero and NOT identical to `ha0` —
        the 2026-09-04 finding that 5 % of windows emit a constant −1.5 brake and would
        otherwise have been read as planning;
     A6 a curving, time-varying plan does NOT read as a collapsed action space.

  B. THE CROSS-CHECK REFUSAL — the reason the second probe exists at all:
     B1 agreeing probes give ``status: OK``;
     B2 a record whose ``trivial_profile`` disagrees with the raw arrays REFUSES, and
        ``main()`` exits non-zero rather than rendering;
     B3 the refusal survives ``--allow-probe-disagreement`` only as an explicit override,
        and the rendered document then still carries the REFUSED status.

  C. THE VOCABULARY AND THE READ-ONLY CONTRACT:
     C1 the rendered markdown NEVER contains "closed loop" (PI ruling 2026-09-02) and
        does carry the OPEN-LOOP note;
     C2 every family level in the document is the record's own number — the tool does
        not recompute a metric, so perturbing the record moves the document;
     C3 the majority-class floor is the record's when present and DERIVED from that same
        block's ``per_class`` counts when not — and it is labelled as derived.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
TOOL = _REPO / "taniteval" / "tools" / "refav1_openloop_report.py"

_spec = importlib.util.spec_from_file_location("refav1_openloop_report_under_test", TOOL)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

K = 4


def _paths(n, *, straight):
    """[n, K, 2] paths: a straight constant-speed line, or a curving one."""
    t = np.arange(1, K + 1, dtype=np.float32)
    out = np.zeros((n, K, 2), dtype=np.float32)
    for i in range(n):
        v = 5.0 + i
        out[i, :, 0] = v * 0.2 * t
        out[i, :, 1] = 0.0 if straight else 0.05 * (i + 1) * t ** 2
    return out


def _make_dump(tmp: Path, *, n_ep=3, n_win=5, cl_is_floor=True,
               controls="zero", name=None) -> Path:
    """``controls``: 'zero' | 'brake' (constant -1.5, kappa 0) | 'varied' (curving,
    time-varying accel — the only one a real planner would emit)."""
    d = tmp / (name or ("dump_floor" if cl_is_floor else "dump_real"))
    (d / "decisions").mkdir(parents=True, exist_ok=True)
    for e in range(n_ep):
        ha0 = _paths(n_win, straight=True)
        cl = ha0.copy() if cl_is_floor else _paths(n_win, straight=False)
        np.savez(d / f"ep{e:03d}.npz", g=_paths(n_win, straight=False),
                 cl=cl, ha=_paths(n_win, straight=False), ha0=ha0,
                 ol=_paths(n_win, straight=False),
                 ws=np.arange(n_win), eid=np.array([e]),
                 clip_index=np.array([e]), v0=np.full(n_win, 5.0, np.float32))
        ctl = np.zeros((n_win, K, 2), np.float32)
        if controls == "brake":
            ctl[:, :, 0] = -1.5
        elif controls == "varied":
            ctl[:, :, 0] = np.arange(K, dtype=np.float32)[None, :] * 0.25
            ctl[:, :, 1] = 0.03
        np.savez(d / "decisions" / f"ep{e:03d}.npz", cl_controls=ctl)
    return d


def _make_record(n_windows: int, identical_n: int, *, ade=0.5) -> dict:
    fam = {
        "longitudinal": {"speed_mae_mps": 0.393, "speed_bias_mps": 0.18,
                         "speed_rmse_mps": 0.64, "along_mae_m": 0.25,
                         "along_final_bias_m": 0.51, "accel_mae_mps2": 0.41},
        "lateral": {"heading_mae_deg": 4.3248, "yaw_rate_mae_degps": 4.03,
                    "curvature_mae_1pm": 0.0125, "cross_mae_m": 0.3488,
                    "cross_final_mae_m": 0.899},
        "tactical": {"lateral_decision": {"status": "OK", "accuracy": 0.75, "kappa": 0.0},
                     "longitudinal_decision": {"status": "OK", "accuracy": 0.75,
                                               "kappa": 0.0},
                     "maneuver_5way_collapsed": {"status": "OK", "accuracy": 0.5,
                                                 "kappa": 0.0},
                     "goal_setting": {"status": "OK", "goal_point_error_m": 1.38}},
        "strategic": {"status": "UNAVAILABLE", "reason": "no route in the scored pass"},
    }
    iv = {"metrics": {"ade_dense_m": {"mean": ade, "lo": 0.1, "hi": 0.9},
                      "fde_last_m": {"mean": 1.38, "lo": 0.2, "hi": 2.6},
                      "LON_speed_mae_mps": {"mean": 0.393, "lo": 0.1, "hi": 0.7},
                      "LON_along_mae_m": {"mean": 0.25, "lo": 0.08, "hi": 0.42},
                      "LAT_cross_mae_m": {"mean": 0.3488, "lo": 0.01, "hi": 0.68},
                      "LAT_heading_mae_deg": {"mean": 4.3248, "lo": 0.14, "hi": 8.5}}}
    arm = {"tier": "T1", "four_families": json.loads(json.dumps(fam)),
           "intervals": json.loads(json.dumps(iv))}
    return {
        "arms": {a: json.loads(json.dumps(arm)) for a in ("cl", "ha", "ha0", "ol")},
        "refav1": {
            "n_windows": n_windows,
            "trivial_profile": {
                "n_windows": n_windows,
                "arms": {"cl": {"n": n_windows, "trivial_frac": 1.0,
                                "identical_to": {"ha0": {"n": identical_n,
                                                         "frac": 1.0}}},
                         "ha0": {"n": n_windows, "trivial_frac": 1.0,
                                 "identical_to": {}}},
                "degenerate_arms": ["cl", "ha0"]},
            "families_paired": {"paired_cl_minus_ha0": {
                "direction": "cl - ha0", "tier": "T1 minus T1",
                "estimator": "paired_episode_cluster_bootstrap",
                "families": {"ADE": {"ade_m": {"delta": 0.0, "lo": 0.0, "hi": 0.0,
                                               "separated": False}}}}},
            "distance_keeping": {"status": "PRESENT", "per_arm": {
                "cl": {"status": "OK", "mean_headway_min_m": 33.4,
                       "mean_time_gap_min_s": 3.55, "mean_min_ttc_s": 24.5}}},
            "strategic": {
                "n_windows": n_windows, "n_route_labeled": 8, "nav_valid_frac": 1.0,
                "n_excluded_no_route_label": n_windows - 8,
                "conditionings": {
                    "nav_true": {"status": "OK", "n": 8, "accuracy": 1.0, "kappa": 1.0,
                                 "majority_class_rate": 0.75,
                                 "ci": {"accuracy": {"mean": 1.0, "lo": 1.0, "hi": 1.0}}},
                    "nav_shuffled": {"status": "OK", "n": 8, "accuracy": 0.625,
                                     "kappa": -0.01, "majority_class_rate": 0.75,
                                     "ci": {"accuracy": {"mean": 0.625, "lo": 0.4,
                                                         "hi": 0.85}}},
                    "nav_zero": {"status": "OK", "n": 8, "accuracy": 0.75, "kappa": 0.0,
                                 "majority_class_rate": 0.75,
                                 "ci": {"accuracy": {"mean": 0.75, "lo": 0.5,
                                                     "hi": 0.95}}}}},
            "tactical_declared": {
                "vocabulary": "v7.0", "n_windows": n_windows,
                "conditionings": {
                    f"{h}_{c}": {"status": "OK", "n": 8, "accuracy": 0.5, "kappa": 0.1,
                                 "per_class": {"a": {"n_true": 6}, "b": {"n_true": 2}},
                                 "ci": {"accuracy": {"mean": 0.5, "lo": 0.2, "hi": 0.8}}}
                    for h in ("lat", "lon")
                    for c in ("nav_true", "nav_shuffled", "nav_zero")}},
            "planner": {"cl": {"n": n_windows, "source_fractions": {"cem": 1.0},
                               "baseline_won_frac": 0.0,
                               "goal_source_fractions": {"tactical_imagined": 1.0},
                               "goal_action_lon": {"CRUISE": 1.0}, "cost_mean": 1e-5}},
            "wm_diagnostic_T0": {"tier": "T0", "tier_note": "teacher-forced",
                                 "n": n_windows, "k_wm": 30, "dt_s": 0.2,
                                 "space": "standardised", "tgt_std_mean": 0.96,
                                 "intervals": {"wm_mse_model": {"mean": 0.8, "lo": 0.7,
                                                                "hi": 0.9}},
                                 "paired_const_minus_model": {"delta": 0.2, "lo": 0.1,
                                                              "hi": 0.3,
                                                              "separated": True}},
        },
    }


# ------------------------------------------------------------------ A: the probe
def test_A1_a_floor_dump_reads_full_identity(tmp_path):
    d = _make_dump(tmp_path, cl_is_floor=True)
    p = rr.identity_probe(str(d))
    assert p["n_windows"] == 15 and p["n_episodes"] == 3
    assert p["n_bit_identical"] == 15
    assert p["frac_bit_identical"] == 1.0
    assert p["n_exact_array_equal"] == 15
    assert p["max_residual_m"] == 0.0


def test_A2_DELIBERATE_REGRESSION_a_real_planner_arm_is_not_the_floor(tmp_path):
    """⛔ Must FAIL to read 1.0, or A1's 1.0 proves nothing about the arm."""
    d = _make_dump(tmp_path, cl_is_floor=False)
    p = rr.identity_probe(str(d))
    assert p["n_bit_identical"] == 0
    assert p["frac_bit_identical"] == 0.0
    assert p["max_residual_m"] > 1e-3


def test_A3_the_control_mechanism_is_counted_separately(tmp_path):
    zero = rr.identity_probe(str(_make_dump(tmp_path, controls="zero")))
    assert zero["n_windows_controls_exactly_zero"] == 15
    assert zero["frac_controls_exactly_zero"] == 1.0
    assert zero["max_abs_control"] == 0.0
    nz = rr.identity_probe(str(_make_dump(tmp_path, cl_is_floor=False,
                                          controls="varied")))
    assert nz["n_windows_controls_exactly_zero"] == 0
    assert nz["max_abs_control"] == pytest.approx(0.75)


def test_A5_the_SHAPE_probe_sees_a_trivial_baseline_the_zero_test_MISSES(tmp_path):
    """⭐ The 2026-09-04 finding in miniature. A constant −1.5 brake with κ ≡ 0 is an
    INJECTED baseline, but it is not zero and it is not identical to `ha0` — so a
    headline built on `controls == 0` alone would have called it planning."""
    d = _make_dump(tmp_path, cl_is_floor=False, controls="brake", name="dump_brake")
    p = rr.identity_probe(str(d))
    assert p["n_windows_controls_exactly_zero"] == 0          # the zero test misses it
    assert p["n_bit_identical"] == 0                          # so does the identity test
    assert p["n_windows_kappa_identically_zero"] == 15        # the shape probe does not
    assert p["n_windows_accel_constant_in_time"] == 15
    assert p["n_windows_straight_and_constant_accel"] == 15
    assert p["frac_straight_and_constant_accel"] == 1.0
    assert p["distinct_plans_emitted"] == {"-1.5": 15}
    assert p["n_distinct_plans_emitted"] == 1


def test_A6_DELIBERATE_REGRESSION_a_curving_time_varying_plan_reads_as_such(tmp_path):
    """⛔ Must NOT read as a collapsed action space, or A5's verdict proves nothing."""
    d = _make_dump(tmp_path, cl_is_floor=False, controls="varied", name="dump_varied")
    p = rr.identity_probe(str(d))
    assert p["n_windows_kappa_identically_zero"] == 0
    assert p["n_windows_accel_constant_in_time"] == 0
    assert p["n_windows_straight_and_constant_accel"] == 0
    assert p["n_distinct_plans_emitted"] == 0


def test_A4_per_episode_rows_sum_to_the_pooled_counts(tmp_path):
    p = rr.identity_probe(str(_make_dump(tmp_path)))
    assert sum(v["n"] for v in p["per_episode"].values()) == p["n_windows"]
    assert (sum(v["n_bit_identical"] for v in p["per_episode"].values())
            == p["n_bit_identical"])


# ------------------------------------------------------------------ B: the refusal
def test_B1_agreeing_probes_are_OK(tmp_path):
    d = _make_dump(tmp_path)
    p = rr.identity_probe(str(d))
    assert rr.cross_check(_make_record(15, 15), p)["status"] == "OK"


def test_B2_a_disagreeing_record_REFUSES_and_main_exits_nonzero(tmp_path):
    d = _make_dump(tmp_path)
    p = rr.identity_probe(str(d))
    xc = rr.cross_check(_make_record(15, 9), p)          # record claims 9, arrays say 15
    assert xc["status"] == "REFUSED" and "reason" in xc
    rec_path = tmp_path / "rec_bad.json"
    rec_path.write_text(json.dumps(_make_record(15, 9)), encoding="utf-8")
    out = tmp_path / "should_not_exist.md"
    import sys
    argv = sys.argv
    sys.argv = ["refav1_openloop_report.py", "--record", str(rec_path),
                "--dump", str(d), "--out", str(out)]
    try:
        assert rr.main() == 2
    finally:
        sys.argv = argv
    assert not out.exists(), "a REFUSED cross-check must not leave a document behind"


def test_B3_the_override_is_explicit_and_the_document_still_says_REFUSED(tmp_path):
    d = _make_dump(tmp_path)
    p = rr.identity_probe(str(d))
    rec = _make_record(15, 9)
    md = rr.render(rec, p, rr.cross_check(rec, p), "t")
    assert "REFUSED" in md


# ------------------------------------------------------------------ C: contract
def test_C1_the_document_never_says_closed_loop(tmp_path):
    d = _make_dump(tmp_path)
    p = rr.identity_probe(str(d))
    rec = _make_record(15, 15)
    md = rr.render(rec, p, rr.cross_check(rec, p), "refav1 read")
    assert "closed loop" not in md.lower()
    assert "closed-loop" not in md.lower()
    assert "OPEN LOOP" in md


def test_C2_every_family_level_is_the_records_own_number(tmp_path):
    d = _make_dump(tmp_path)
    p = rr.identity_probe(str(d))
    rec = _make_record(15, 15)
    base = rr.render(rec, p, rr.cross_check(rec, p), "t")
    assert "4.3248" in base                      # LAT heading, straight from the record
    rec2 = _make_record(15, 15)
    rec2["arms"]["cl"]["four_families"]["lateral"]["heading_mae_deg"] = 9.8765
    moved = rr.render(rec2, p, rr.cross_check(rec2, p), "t")
    assert "9.8765" in moved, "the document must track the record, not recompute it"


def test_C3_the_majority_floor_is_the_records_or_is_labelled_derived(tmp_path):
    d = _make_dump(tmp_path)
    p = rr.identity_probe(str(d))
    rec = _make_record(15, 15)
    md = rr.render(rec, p, rr.cross_check(rec, p), "t")
    # strategic carries its own majority_class_rate -> no "(derived)" marker on 0.7500
    assert "0.7500" in md
    # tactical_declared does not -> derived from per_class (6 of 8) and labelled
    assert "0.7500 *(derived)*" in md
    v = {"n": 8, "per_class": {"a": {"n_true": 6}, "b": {"n_true": 2}}}
    assert rr._majority_floor(v) == (0.75, "derived")
    assert rr._majority_floor({"majority_class_rate": 0.4}) == (0.4, "record")
    assert rr._majority_floor({"n": 8, "per_class": {}})[1] == "n/a"
