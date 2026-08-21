"""CPU tests for the four-family rescore that closes the v5.8f release row.

⛔ NO GPU, NO CORPUS, NO CHECKPOINT. Every fixture is a synthetic dump whose
family values are hand-computable from the geometry, so a wrong number fails
against ARITHMETIC rather than against a previously-recorded output — a golden
file would only pin whatever the code did on the day it was written.

Two groups:

* the FAMILY MATHS (``taniteval.four_families``) — the trajectory-derived
  TACTICAL family, the target-speed accuracy member, and the STRATEGIC n/a;
* the REFUSALS (``taniteval/tools/ff_rescore.py``) — cross-grid join, missing
  tier stamp, banned estimator, mismatched lead block. Each is exercised as a
  SUBPROCESS with its exit code checked, because a refusal that only prints is
  not a refusal.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_REPO = os.path.dirname(os.path.dirname(_HERE))              # <repo>
_STACK = os.path.join(_REPO, "stack")
_TE = os.path.join(_REPO, "taniteval")
_TOOL = os.path.join(_TE, "tools", "ff_rescore.py")
for _p in (_STACK, _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

ff = pytest.importorskip("taniteval.four_families")


def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([_STACK, _TE, e.get("PYTHONPATH", "")])
    # ⛔ torch spawns ~113 threads per process; a test suite that shells out
    # several times without this can sit at 0-6 % sm making no progress and look
    # exactly like a hang (MEASURED 2026-07-27).
    e["OMP_NUM_THREADS"] = "2"
    return e


def _run(args, expect=0):
    p = subprocess.run([sys.executable, _TOOL] + args, env=_env(),
                       capture_output=True, text=True, encoding="utf-8", timeout=900)
    out = p.stdout + p.stderr
    assert p.returncode == expect, (
        f"expected exit {expect}, got {p.returncode}\n{out[-3000:]}")
    return out


# --------------------------------------------------------------------------- #
# Fixtures — straight-line GT so every family value is closed-form             #
# --------------------------------------------------------------------------- #
DT = 0.1
K = 20
SPEED = 10.0            # m/s -> 1.0 m per 0.1 s step


def straight(n=40, speed=SPEED, k=K, dt=DT):
    """GT: dead straight at a constant speed. dyaw = 0, dv = 0, v0 = v1 = speed."""
    t = np.arange(1, k + 1) * dt
    x = np.outer(np.full(n, speed), t)
    return np.stack([x, np.zeros_like(x)], -1)


def write_dump(d, arms: dict, gt=None, n_ep=4, n_per=10, k=K):
    """A T1-schema dump dir: ep*.npz with 'g' + one key per arm."""
    os.makedirs(d, exist_ok=True)
    files = []
    for e in range(n_ep):
        g = straight(n_per, k=k) if gt is None else gt[e * n_per:(e + 1) * n_per]
        payload = {"g": g, "ws": np.arange(n_per)}
        for name, fn in arms.items():
            payload[name] = fn(g)
        f = os.path.join(d, f"ep{e:03d}.npz")
        np.savez(f, **payload)
        files.append(f)
    return d


# ============================================================================ #
# 1. The manoeuvre kinematics a decision is read from                          #
# ============================================================================ #
def test_kinematics_of_a_straight_constant_speed_path_are_exactly_zero():
    p = torch.as_tensor(straight(5)).float()
    dyaw, dv, v0, v1, prov = ff.maneuver_kinematics(p, DT)
    assert torch.allclose(dyaw, torch.zeros(5), atol=1e-5)
    assert torch.allclose(dv, torch.zeros(5), atol=1e-4)
    assert torch.allclose(v0, torch.full((5,), SPEED), atol=1e-4)
    assert torch.allclose(v1, torch.full((5,), SPEED), atol=1e-4)
    assert prov["n_never_moved"] == 0
    assert prov["n_heading_held_from_last_moving_step"] == 0
    assert prov["dt_s"] == DT


def test_dyaw_is_the_path_tangent_at_the_horizon():
    """A path bending to a known final tangent must report that tangent.

    Final segment is (1.0, 1.0) -> tangent = +45 deg = +pi/4 rad, which is above
    YAW_TURN_RAD (0.15) and must therefore classify as turn_left.
    """
    p = np.zeros((1, K, 2))
    for i in range(K):
        p[0, i] = [i + 1, 0.0] if i < K - 1 else [K, 1.0]
    dyaw, _, _, _, _ = ff.maneuver_kinematics(torch.as_tensor(p).float(), DT)
    assert dyaw[0] == pytest.approx(math.pi / 4, abs=1e-4)


def test_brake_to_stop_holds_heading_instead_of_deleting_the_window():
    """⛔ The branch that would otherwise bias brake_stop out of the denominator.

    A window that decelerates to a standstill ends with ds -> 0, where the path
    tangent is undefined. It must be HELD at the last moving step (a stopped
    vehicle keeps its heading), the fallback must be COUNTED, and the window
    must survive to be classified.
    """
    p = np.zeros((1, K, 2))
    x = 0.0
    for i in range(K):
        step = max(0.0, 1.0 - i * 0.12)          # decelerates, then dead stop
        x += step
        p[0, i] = [x, 0.0]
    t = torch.as_tensor(p).float()
    dyaw, dv, v0, v1, prov = ff.maneuver_kinematics(t, DT)
    assert prov["n_heading_held_from_last_moving_step"] == 1
    assert prov["n_never_moved"] == 0
    assert float(v1) < float(v0)                  # it really did slow down
    assert float(dyaw) == pytest.approx(0.0, abs=1e-5)
    lat, lon = _factor(dyaw, dv, v0, v1)
    from tanitad.refs.refc_tactical import LON_BRAKE_STOP
    assert int(lon[0]) == LON_BRAKE_STOP


def _factor(dyaw, dv, v0, v1):
    from tanitad.refs.refc_tactical import factor_from_kinematics
    return factor_from_kinematics(dyaw, dv, v0, v1)


def test_a_window_that_never_moved_is_lane_keep_not_a_nan():
    z = torch.zeros(3, K, 2)
    dyaw, dv, v0, v1, prov = ff.maneuver_kinematics(z, DT)
    assert prov["n_never_moved"] == 3
    assert torch.allclose(dyaw, torch.zeros(3))
    lat, _ = _factor(dyaw, dv, v0, v1)
    from tanitad.refs.refc_tactical import LAT_LANE_KEEP
    assert set(lat.tolist()) == {LAT_LANE_KEEP}


# ============================================================================ #
# 2. Cohen's kappa — including the case that must NOT return a fake 1.0        #
# ============================================================================ #
def test_kappa_perfect_agreement_is_one():
    a = [0, 1, 2, 0, 1, 2]
    assert ff._kappa_k(a, a, 3) == pytest.approx(1.0)


def test_kappa_is_zero_when_agreement_is_exactly_chance():
    # 2x2 with po = 0.5 and pe = 0.5 -> kappa = 0
    g = [0, 0, 1, 1]
    p = [0, 1, 0, 1]
    assert ff._kappa_k(g, p, 2) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_None_not_one_when_both_streams_are_constant():
    """⛔ Two raters that both always say 'lane_keep' agree perfectly and know
    nothing. Returning 1.0 there would read as perfect tactical skill on a
    window set where nothing happened."""
    assert ff._kappa_k([0] * 20, [0] * 20, 3) is None


def test_kappa_covers_all_five_classes_not_only_the_first_three():
    """Regression guard: hierarchy._kappa is hard-coded to {0,1,2} and would
    silently ignore classes 3/4 of the 5-way label."""
    g = [3, 3, 4, 4]
    p = [3, 3, 4, 4]
    assert ff._kappa_k(g, p, 5) == pytest.approx(1.0)


# ============================================================================ #
# 3. The TACTICAL family from trajectories                                     #
# ============================================================================ #
def _eids(n, per=10):
    return [f"ep{i // per:03d}" for i in range(n)]


def test_tactical_perfect_arm_scores_one_and_kappa_is_honest():
    gt = torch.as_tensor(straight(40)).float()
    out = ff.tactical_from_trajectory(gt.clone(), gt, DT, _eids(40), n_boot=50,
                                      tier="T1")
    assert out["status"] == "OK"
    assert out["lateral_decision"]["accuracy"] == 1.0
    assert out["longitudinal_decision"]["accuracy"] == 1.0
    assert out["maneuver_5way_collapsed"]["accuracy"] == 1.0
    # every window is lane_keep/steady, so kappa is UNDEFINED, not 1.0
    assert out["lateral_decision"]["kappa"] is None
    assert out["lateral_decision"]["kappa_undefined_reason"]


def test_tactical_detects_a_lateral_decision_error_ade_would_hide():
    """The arm turns left where the human goes straight. LAT accuracy must be 0
    and 'lane_keep' must show up as NEVER PREDICTED."""
    gt = torch.as_tensor(straight(40)).float()
    t = torch.arange(1, K + 1).float() * DT
    pred = gt.clone()
    pred[:, :, 1] = 1.5 * t ** 2                       # curving left
    out = ff.tactical_from_trajectory(pred, gt, DT, _eids(40), n_boot=50,
                                      tier="T1")
    lat = out["lateral_decision"]
    assert lat["accuracy"] == 0.0
    assert lat["never_predicted"] == ["lane_keep"]
    assert lat["per_class"]["turn_left"]["n_pred"] == 40
    assert lat["per_class"]["turn_left"]["n_true"] == 0
    assert lat["ci"]["accuracy"]["estimator"] == "episode_cluster_bootstrap"


def test_tactical_reports_lat_and_lon_separately_not_only_the_collapsed_5way():
    """⛔ The collapsed 5-way cannot show lat/lon mixing — the programme's single
    largest known architectural defect. Both axes must be present."""
    gt = torch.as_tensor(straight(40)).float()
    out = ff.tactical_from_trajectory(gt.clone(), gt, DT, _eids(40), n_boot=20)
    for k in ("lateral_decision", "longitudinal_decision",
              "maneuver_5way_collapsed", "goal_setting"):
        assert k in out, k
    assert out["lateral_decision"]["classes"] == ["lane_keep", "turn_left",
                                                  "turn_right"]
    assert out["longitudinal_decision"]["classes"] == ["brake_stop", "steady",
                                                       "accelerate"]


def test_tactical_says_out_loud_it_is_not_selected_vs_executed():
    gt = torch.as_tensor(straight(20)).float()
    out = ff.tactical_from_trajectory(gt.clone(), gt, DT, _eids(20), n_boot=20)
    assert "NOT 'selected vs executed'" in out["_is_not"]


def test_tactical_at_T0_carries_the_action_echo_warning():
    """A T0 path is teacher-forced by the RECORDED actions, so a
    trajectory-derived manoeuvre is substantially an echo of the label's source
    (§1.12). The block must say so itself."""
    gt = torch.as_tensor(straight(20)).float()
    t0 = ff.tactical_from_trajectory(gt.clone(), gt, DT, _eids(20), n_boot=20,
                                     tier="T0")
    t1 = ff.tactical_from_trajectory(gt.clone(), gt, DT, _eids(20), n_boot=20,
                                     tier="T1")
    assert "⛔_tier_warning" in t0 and "echo" in t0["⛔_tier_warning"].lower()
    assert "⛔_tier_warning" not in t1


def test_tactical_without_eids_reports_no_interval_rather_than_a_bare_number():
    gt = torch.as_tensor(straight(20)).float()
    out = ff.tactical_from_trajectory(gt.clone(), gt, DT, eid=None, n_boot=20)
    assert out["lateral_decision"]["ci"]["status"] == "UNAVAILABLE"
    assert "decision-grade" in out["lateral_decision"]["ci"]["reason"]


# ============================================================================ #
# 4. Tactical GOAL setting                                                     #
# ============================================================================ #
def test_goal_metrics_are_the_hand_computed_geometry():
    """GT endpoint (20, 0); arm endpoint (20, 6).
    error = 6; bearing = atan2(6,20) = 16.699 deg; range ratio = |(20,6)|/20."""
    gt = torch.as_tensor(straight(10)).float()
    pred = gt.clone()
    pred[:, -1, 1] = 6.0
    g = ff.tactical_goal(pred, gt, _eids(10), n_boot=50, tier="T1")
    assert g["goal_point_error_m"] == pytest.approx(6.0, abs=1e-3)
    assert g["goal_bearing_mae_deg"] == pytest.approx(
        math.degrees(math.atan2(6.0, 20.0)), abs=1e-2)
    assert g["goal_range_ratio"] == pytest.approx(
        math.hypot(20.0, 6.0) / 20.0, abs=1e-3)
    assert g["goal_lat_bias_m"] == pytest.approx(6.0, abs=1e-3)
    assert g["goal_long_bias_m"] == pytest.approx(0.0, abs=1e-3)
    assert g["n"] == 10


def test_goal_anchor_selection_is_unavailable_with_reason_and_n():
    """⛔ Clause 5: an uncomputable member states its reason AND its n. An arm
    that commits to one path has no fan, so anchor SELECTION cannot be scored —
    and n must be the windows it would have had, not 0."""
    gt = torch.as_tensor(straight(15)).float()
    g = ff.tactical_goal(gt.clone(), gt, _eids(15), n_boot=20)
    a = g["anchor_selection"]
    assert a["status"] == "UNAVAILABLE"
    assert a["n"] == 15
    assert "fan" in a["reason"] and "WORK ITEM" in a["reason"]


def test_goal_point_error_is_labelled_as_fde_and_not_sold_as_new():
    gt = torch.as_tensor(straight(8)).float()
    g = ff.tactical_goal(gt.clone(), gt, _eids(8), n_boot=20)
    assert "FDE" in g["_goal_point_error_is"]


# ============================================================================ #
# 5. The STRATEGIC n/a — reason + n, never a silent drop                       #
# ============================================================================ #
def test_strategic_unavailable_states_reason_and_the_n_it_would_have_had():
    s = ff.strategic_unavailable(881, tier="T1")
    assert s["status"] == "UNAVAILABLE"
    assert s["n"] == 881 and s["n_windows_it_would_have_had"] == 881
    assert "we do not include open maps data" in s["reason"]
    assert "lat/lon" in s["reason"]
    assert "WORK ITEM" in s["_is_a_work_item"]
    assert s["tier"] == "T1"


def test_strategic_names_the_instrument_that_would_close_it():
    s = ff.strategic_unavailable(10)
    assert "PH2" in s["instrument_that_would_close_it"]
    assert "VLM" in s["instrument_that_would_close_it"]


def test_strategic_no_label_is_opt_in_and_never_inferred():
    """A corpus that DOES carry a map must not be handed the excuse."""
    gt = torch.as_tensor(straight(10)).float()
    win = {"pred_dense": gt.clone(), "gt_dense": gt, "pred": gt[:, ::5],
           "gt": gt[:, ::5], "dt_s": DT, "eid": _eids(10)}
    default = ff.all_families(win, tactical_from_traj=True, n_boot=20)
    assert "open maps data" not in json.dumps(default["strategic"])
    declared = ff.all_families(win, tactical_from_traj=True,
                               strategic_no_label=True, n_boot=20)
    assert "open maps data" in declared["strategic"]["reason"]


# ============================================================================ #
# 6. LONGITUDINAL target-speed accuracy                                        #
# ============================================================================ #
def test_target_speed_accuracy_bands_are_hand_computable():
    """Arm is exactly 0.75 m/s too fast at every step: inside the 1.0 and 2.0
    bands, outside the 0.5 band."""
    gt = torch.as_tensor(straight(10, speed=10.0)).float()
    pred = torch.as_tensor(straight(10, speed=10.75)).float()
    lon = ff.longitudinal(pred, gt, DT)
    acc = lon["target_speed_acc"]
    assert acc["within_0.5_mps"] == pytest.approx(0.0, abs=1e-6)
    assert acc["within_1.0_mps"] == pytest.approx(1.0, abs=1e-6)
    assert acc["within_2.0_mps"] == pytest.approx(1.0, abs=1e-6)
    assert lon["speed_bias_mps"] == pytest.approx(0.75, abs=1e-3)


def test_distance_keeping_without_a_lead_block_is_a_work_item_not_a_pass():
    gt = torch.as_tensor(straight(10)).float()
    dk = ff.longitudinal(gt.clone(), gt, DT)["distance_keeping"]
    assert dk["status"] == "UNAVAILABLE"
    assert "WORK ITEM" in dk["reason"]


# ============================================================================ #
# 7. all_families bookkeeping                                                  #
# ============================================================================ #
def _win(n=20):
    gt = torch.as_tensor(straight(n)).float()
    return {"pred_dense": gt.clone(), "gt_dense": gt,
            "pred": gt[:, [4, 9, 14, 19]], "gt": gt[:, [4, 9, 14, 19]],
            "wp_steps": [5, 10, 15, 20], "dt_s": DT, "eid": _eids(n)}, gt


def test_rule_satisfied_separates_compliant_from_complete():
    """⛔ Two different questions. A STRATEGIC n/a WITH a reason and an n obeys
    the binding rule (clause 5) while the block is permanently incomplete."""
    win, _ = _win()
    fam = ff.all_families(win, tactical_from_traj=True, strategic_no_label=True,
                          tier="T1", n_boot=20)
    assert fam["_rule_satisfied"] is True
    assert fam["_complete"] is False          # strategic n/a + no lead block
    assert fam["_families_unavailable"] == ["strategic"]


def test_tier_is_stamped_on_every_family():
    win, _ = _win()
    fam = ff.all_families(win, tactical_from_traj=True, strategic_no_label=True,
                          tier="T1", n_boot=20)
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        assert fam[k]["tier"] == "T1", k
    assert fam["_tier"] == "T1"


def test_tactical_from_trajectory_is_opt_in_so_existing_callers_are_unchanged():
    """t1_eval.py and eval_four_families.py call all_families() with no flag;
    their behaviour must not change under them."""
    win, _ = _win()
    fam = ff.all_families(win, n_boot=20)
    assert fam["tactical"]["status"] == "UNAVAILABLE"
    assert "how_to_populate" in fam["tactical"]


def test_a_real_hierarchy_result_still_beats_the_trajectory_fallback():
    """A DECLARED decision is strictly more informative than one read back off
    the driven path, so `hier` must win when both are offered."""
    win, _ = _win()
    hier = {"consistency": {"maneuver_vs_trajectory": {"kappa": 0.42,
                                                       "agreement": 0.8}},
            "n_windows": 20}
    fam = ff.all_families(win, hier=hier, tactical_from_traj=True, n_boot=20)
    assert fam["tactical"]["source"] == "hierarchy.run"


# ============================================================================ #
# 8. THE REFUSALS — each as a subprocess with its exit code checked            #
# ============================================================================ #
def test_tool_happy_path_writes_per_arm_and_comparison_json(tmp_path):
    d = write_dump(str(tmp_path / "dump"),
                   {"cl": lambda g: g + 0.20,
                    "ha": lambda g: g + 0.40})
    out = str(tmp_path / "out")
    log = _run(["--dump", f"A={d}", "--out-dir", out, "--strategic-no-label",
                "--n-boot", "40"])
    assert "FF_EXIT=0" in log
    a = json.load(open(os.path.join(out, "ff_A_cl.json"), encoding="utf-8"))
    assert a["tier"] == "T1"                        # resolved from the shared table
    assert a["intervals"]["estimator"] == "episode_cluster_bootstrap"
    assert a["intervals"]["point_estimate"] == "full_set pooled mean over windows"
    fam = a["four_families"]
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        assert k in fam and fam[k].get("tier") == "T1", k
    assert fam["tactical"]["status"] == "OK"
    assert fam["strategic"]["status"] == "UNAVAILABLE" and fam["strategic"]["n"] > 0
    c = json.load(open(os.path.join(out, "ff_comparison.json"), encoding="utf-8"))
    assert c["status"] == "OK"
    k = "A:ha_minus_A:cl"
    assert k in c["paired"]
    assert c["paired"][k]["ade_dense_m"]["estimator"] == \
        "paired_episode_cluster_bootstrap"


def test_tool_reports_all_four_families_never_ade_alone(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.1})
    out = str(tmp_path / "o")
    _run(["--dump", f"A={d}", "--out-dir", out, "--strategic-no-label",
          "--n-boot", "20"])
    fam = json.load(open(os.path.join(out, "ff_A_cl.json"), encoding="utf-8"))["four_families"]
    assert fam["_rule_satisfied"] is True
    # the binding members, per family
    assert {"speed_mae_mps", "target_speed_acc", "distance_keeping"} <= set(
        fam["longitudinal"])
    assert {"heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps",
            "cross_mae_m"} <= set(fam["lateral"])
    assert {"lateral_decision", "longitudinal_decision", "goal_setting"} <= set(
        fam["tactical"])


def test_tool_REFUSES_a_cross_grid_join_and_BANKS_the_refusal(tmp_path):
    """⛔ Same row count, different windows. A cross-grid join produces a
    PLAUSIBLE number, not an error — so it must be refused, not warned."""
    a = write_dump(str(tmp_path / "a"), {"cl": lambda g: g + 0.1})
    b = write_dump(str(tmp_path / "b"), {"cl": lambda g: g + 0.1},
                   gt=straight(40, speed=4.0))       # different GT, same shape
    out = str(tmp_path / "o")
    log = _run(["--dump", f"A={a}", "--dump", f"B={b}", "--out-dir", out,
                "--strategic-no-label", "--n-boot", "20"], expect=3)
    assert "CROSS-GRID" in log or "distinct window grids" in log
    c = json.load(open(os.path.join(out, "ff_comparison.json"), encoding="utf-8"))
    assert c["status"] == "REFUSED"
    assert "paired" not in c
    assert len(c["grid_groups"]) == 2
    # the per-arm records are still valid on their OWN grid and are written
    assert os.path.exists(os.path.join(out, "ff_A_cl.json"))
    assert os.path.exists(os.path.join(out, "ff_B_cl.json"))


def test_tool_REFUSES_an_arm_with_no_tier_stamp(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"mystery": lambda g: g + 0.1})
    log = _run(["--dump", f"C={d}", "--out-dir", str(tmp_path / "o"),
                "--n-boot", "20"], expect=2)
    assert "NO TIER" in log
    assert "EVAL_DOCTRINE" in log


def test_tool_accepts_an_explicit_tier_for_an_unknown_arm(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"mystery": lambda g: g + 0.1})
    out = str(tmp_path / "o")
    _run(["--dump", f"C={d}", "--tier", "C:mystery=T0", "--out-dir", out,
          "--strategic-no-label", "--n-boot", "20"])
    rec = json.load(open(os.path.join(out, "ff_C_mystery.json"), encoding="utf-8"))
    assert rec["tier"] == "T0"
    assert "NEVER quotable as driving performance" in rec["tier_note"]


def test_tool_REFUSES_the_biased_estimator_by_name(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.1})
    log = _run(["--dump", f"A={d}", "--out-dir", str(tmp_path / "o"),
                "--estimator", "overlapping_holdout_se"], expect=2)
    assert "BIASES THE POINT ESTIMATE" in log
    assert "SIGN FLIP" in log


def test_tool_REFUSES_a_lead_block_on_a_different_grid(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.1})   # 40 windows
    lead = str(tmp_path / "lead.npz")
    np.savez(lead, leads=np.full((39, K, 2), np.nan),                # 39 rows
             lead_lens=np.full(39, np.nan), speeds=np.zeros(39))
    log = _run(["--dump", f"A={d}", "--out-dir", str(tmp_path / "o"),
                "--lead", lead, "--n-boot", "20"], expect=2)
    assert "different window grids" in log
    assert "do NOT" in log and "truncate" in log


def test_tool_REFUSES_a_lead_block_whose_times_do_not_land_on_the_grid(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.1})
    lead = str(tmp_path / "lead.npz")
    np.savez(lead, leads=np.full((40, 3, 2), np.nan),
             lead_lens=np.full(40, np.nan), speeds=np.zeros(40),
             ts_rel_s=np.array([0.33, 0.66, 0.99]))     # not multiples of dt
    log = _run(["--dump", f"A={d}", "--out-dir", str(tmp_path / "o"),
                "--lead", lead, "--n-boot", "20"], expect=2)
    assert "does not land on this" in log


def test_tool_joins_a_coarser_lead_grid_by_TIME_not_by_truncation(tmp_path):
    """A 4-sample lead track at 0.5/1.0/1.5/2.0 s against a 20-step 0.1 s path
    must map to steps 4/9/14/19 — never to the first four steps."""
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.1})
    lead = str(tmp_path / "lead.npz")
    leads = np.full((40, 4, 2), np.nan)
    leads[:, :, 0] = np.array([25.0, 30.0, 35.0, 40.0])       # 15 m ahead
    leads[:, :, 1] = 0.0
    np.savez(lead, leads=leads, lead_lens=np.full(40, 4.0),
             speeds=np.full(40, SPEED),
             ts_rel_s=np.array([0.5, 1.0, 1.5, 2.0]))
    out = str(tmp_path / "o")
    log = _run(["--dump", f"A={d}", "--out-dir", out, "--lead", lead,
                "--strategic-no-label", "--n-boot", "20"])
    assert "scored on steps [4, 9, 14, 19]" in log
    dk = json.load(open(os.path.join(out, "ff_A_cl.json"), encoding="utf-8"))[
        "four_families"]["longitudinal"]["distance_keeping"]
    assert dk["status"] == "OK"
    assert dk["n"] == 40


def test_tool_derives_dt_from_wp_steps_and_never_assumes_0_1(tmp_path):
    """⛔ The defect that inflated every published speed x5 and every accel x25:
    a sparse 4-waypoint view at ticks (5,10,15,20) is a 0.5 s grid."""
    import torch as _t
    from taniteval import rollout
    gt = _t.as_tensor(straight(40, k=4)).float()
    p = str(tmp_path / "w.pt")
    rollout.save_windows({"pred": gt.clone(), "gt": gt, "eid": _eids(40),
                          "wp_steps": [5, 10, 15, 20]}, p)
    out = str(tmp_path / "o")
    _run(["--dump", f"W={p}", "--tier", "W=T0", "--out-dir", out,
          "--strategic-no-label", "--n-boot", "20"])
    rec = json.load(open(os.path.join(out, "ff_W.json"), encoding="utf-8"))
    assert rec["dt_s"] == 0.5
    assert "wp_steps" in rec["dt_provenance"]
    assert rec["four_families"]["longitudinal"]["dt_s"] == 0.5


def test_tool_output_never_carries_the_deprecated_estimator_anywhere(tmp_path):
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.1,
                                         "ol": lambda g: g + 0.2})
    out = str(tmp_path / "o")
    _run(["--dump", f"A={d}", "--out-dir", out, "--strategic-no-label",
          "--n-boot", "20"])
    for f in os.listdir(out):
        blob = json.load(open(os.path.join(out, f), encoding="utf-8"))
        for path, val in _walk(blob):
            if not path.endswith("estimator"):
                continue
            s = str(val).strip()
            # the field must never NAME the deprecated estimator as the one used
            assert s.split()[0].strip(".,") not in (
                "overlapping_holdout_se", "jackknife", "overlapping"), \
                f"{f}:{path} = {val}"
            # …and where the prose mentions it at all, it must be to REFUSE it
            if "overlapping" in s or "jackknife" in s:
                assert "NOT used" in s, f"{f}:{path} mentions it without refusing"


def test_tool_stamps_a_cross_tier_delta_as_not_a_capability_comparison(tmp_path):
    """T1-minus-T0 measures the TIER as much as the arm (the §1.12 action-echo
    contrast). It must never read as a leaderboard row."""
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.3,
                                         "ol": lambda g: g + 0.1})
    out = str(tmp_path / "o")
    _run(["--dump", f"A={d}", "--out-dir", out, "--strategic-no-label",
          "--n-boot", "20"])
    c = json.load(open(os.path.join(out, "ff_comparison.json"), encoding="utf-8"))
    blk = list(c["paired"].values())[0]
    assert "minus" in blk["tier"]
    assert "⛔_cross_tier_warning" in blk


def test_paired_sign_convention_is_per_metric_not_a_blanket_sentence(tmp_path):
    """⛔ Most rows are ERRORS (lower better) but the TACTICAL decision rows are
    ACCURACIES (higher better). One blanket sentence would invert the reading of
    exactly the family this rescore exists to add."""
    d = write_dump(str(tmp_path / "d"), {"cl": lambda g: g + 0.3,
                                         "ha": lambda g: g + 0.1})
    out = str(tmp_path / "o")
    _run(["--dump", f"A={d}", "--out-dir", out, "--strategic-no-label",
          "--n-boot", "20"])
    c = json.load(open(os.path.join(out, "ff_comparison.json"), encoding="utf-8"))
    sc = list(c["paired"].values())[0]["sign_convention"]
    assert isinstance(sc, dict)
    assert "ACCURACY" in sc["TAC_lat_decision_correct"]
    assert "ACCURACY" in sc["TAC_lon_decision_correct"]
    assert "ERROR" in sc["ade_dense_m"]
    assert "ERROR" in sc["LON_speed_mae_mps"]


def _walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{p}[{i}]")
    else:
        yield p, o


# ============================================================================ #
# 9. eid canonicalisation — a NAME difference is not a GRID difference         #
# ============================================================================ #
def test_canon_eid_matches_two_naming_conventions_for_the_same_episodes():
    sys.path.insert(0, os.path.join(_TE, "tools"))
    import ff_rescore as fr
    assert fr.canon_eid([0, 0, 1, 1]) == fr.canon_eid(
        ["ep_00000", "ep_00000", "ep_00001", "ep_00001"])


def test_runs_detects_a_genuinely_different_partition():
    sys.path.insert(0, os.path.join(_TE, "tools"))
    import ff_rescore as fr
    assert fr._runs([0, 0, 1, 1]) == fr._runs(["a", "a", "b", "b"])
    assert fr._runs([0, 0, 1, 1]) != fr._runs([0, 1, 1, 1])
