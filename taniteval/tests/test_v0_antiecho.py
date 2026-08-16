"""⛔ THE CAN-IT-FIRE PROOF for the v0 anti-echo controls (PI ruling 2026-08-16).

**A guard nobody has watched fail is a HYPOTHESIS.** That is not a slogan here —
it caught two real defects in this programme in the last day alone. So the centre
of this file is not "the code runs": it is a **synthetic pure-echo planner**
(output ≡ hold-v0) that MUST be flagged, beside a **planner with genuine
acceleration structure** that MUST NOT be, on the same windows, through the same
entry point the real evals use.

The ruling being enforced (``Project Steering/V6F_PLANNER_DESIGN.md`` §1.4):

    "We can use v0 as input since it is measured and is not the future, but we
    should assure that the model/planner later is not cheating by just
    outputting v0 as longitudinal plan."  — Sayed, 2026-08-16

CPU-only, numpy/torch, no pod, no checkpoint. Runs in well under a second.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from taniteval import v0_antiecho as AE
from taniteval import four_families as FF

DT = 0.5
H = 4


# --------------------------------------------------------------------------- #
# synthetic arms — the whole point of the file                                 #
# --------------------------------------------------------------------------- #
def _v0(n, rng):
    """Entry speeds spread over a realistic urban band."""
    return torch.as_tensor(rng.uniform(3.0, 16.0, size=n).astype(np.float32))


def _path_from_accel(v0: torch.Tensor, accel: torch.Tensor, dt=DT, h=H):
    """Ego-frame straight path for a per-window CONSTANT acceleration [n]."""
    t = torch.arange(1, h + 1, dtype=torch.float32) * dt
    along = v0[:, None] * t[None, :] + 0.5 * accel[:, None] * t[None, :] ** 2
    return torch.stack([along, torch.zeros_like(along)], -1)


def _echo_arm(v0, **kw):
    """⛔ THE CHEAT. The longitudinal plan IS hold-v0. Zero commanded accel."""
    return AE.hold_v0_path(v0, H, DT)


def _honest_arm(v0, accel_gt, noise=0.0, rng=None):
    """A planner with GENUINE acceleration structure: it tracks the human's own
    commanded acceleration (imperfectly when ``noise`` > 0)."""
    a = accel_gt.clone()
    if noise:
        a = a + torch.as_tensor(
            rng.normal(0.0, noise, size=len(a)).astype(np.float32))
    return _path_from_accel(v0, a)


def _corpus(n=240, n_ep=12, seed=0, act_frac=0.7, accel_scale=1.6):
    """Windows where the human DOES accelerate/brake on ``act_frac`` of them.

    ⭐ The corpus must contain real longitudinal events or the test proves
    nothing: on a corpus of pure cruising, hold-v0 IS the right answer and an
    echo is indistinguishable from skill BY CONSTRUCTION. That is a fact about
    driving, not a weakness of the detector, and it is why ``echo_index`` is
    always published beside ``echo_index_gt``.
    """
    rng = np.random.default_rng(seed)
    v0 = _v0(n, rng)
    a = torch.as_tensor(rng.normal(0.0, accel_scale, size=n).astype(np.float32))
    a[rng.uniform(size=n) > act_frac] = 0.0
    gt = _path_from_accel(v0, a)
    eid = [f"ep{i % n_ep:02d}" for i in range(n)]
    return {"v0": v0, "accel": a, "gt": gt, "eid": eid, "n": n}


def _win(pred, c):
    return {"pred": pred, "gt": c["gt"], "v0": c["v0"], "eid": c["eid"],
            "wp_steps": [1, 2, 3, 4], "dt_s": DT}


# ============================================================================ #
# 1. the baseline plan is the programme's ONE hold-v0, not a second one        #
# ============================================================================ #
def test_hold_v0_path_is_bit_identical_to_the_driving_floor():
    """⛔ The programme already has a hold-v0 (``driving.hold_v0``, tier-0). A
    second implementation that drifts from it would let the four-family control
    and the tier-0 floor disagree about what "the baseline" is."""
    from taniteval.driving import hold_v0 as driving_hold_v0, DT_WP
    v = torch.tensor([0.0, 3.5, 12.4565, 21.0])
    assert DT_WP == 0.5, "the tier-0 waypoint spacing moved; re-pin this test"
    assert torch.equal(AE.hold_v0_path(v, 4, DT_WP), driving_hold_v0(v, 4))


def test_hold_v0_path_is_straight_and_commands_zero_acceleration():
    v = torch.tensor([4.0, 11.0])
    p = AE.hold_v0_path(v, 6, 0.25)
    assert torch.allclose(p[..., 1], torch.zeros_like(p[..., 1]))     # straight
    g = FF._seq_geometry(p, 0.25)
    assert torch.allclose(g["speed"], v[:, None].expand(-1, 6), atol=1e-4)
    assert float(g["accel"].abs().max()) < 1e-4                       # no command


def test_hold_v0_path_refuses_a_degenerate_grid():
    with pytest.raises(ValueError):
        AE.hold_v0_path(torch.tensor([5.0]), 0, 0.5)
    with pytest.raises(ValueError):
        AE.hold_v0_path(torch.tensor([5.0]), 4, 0.0)


# ============================================================================ #
# 2. ⭐ THE CAN-IT-FIRE PROOF — the detector on a pure echo vs an honest arm    #
# ============================================================================ #
def test_copy_detector_FIRES_on_a_pure_echo_planner():
    """⭐ A planner whose output IS hold-v0 must be flagged ECHO, and its
    headline scalar must be exactly 1.0 — not "high", exactly 1."""
    c = _corpus()
    det = AE.copy_detector(_echo_arm(c["v0"]), c["gt"], c["v0"], DT)
    assert det["verdict"] == "ECHO"
    assert det["echo_index"] == 1.0
    assert det["echo_index"] > det["echo_index_gt"]
    assert det["echo_index_excess"] > 0
    assert det["cmd_accel_mae_mps2"] == 0.0
    assert det["speed_dev_rms_mps"] == 0.0
    assert det["dev_ratio"] == 0.0
    # ⭐ the departure correlation is UNDEFINED for a pure echo, and that is the
    # signature — a number here would be an invented verdict.
    assert det["dev_r"] is None and det["dev_r_degenerate"] is True
    # on every window where the human demonstrably acted, the echo held anyway
    assert det["n_windows_human_acted"] > 0
    assert det["echo_frac_where_human_acted"] == 1.0


def test_copy_detector_does_NOT_fire_on_a_planner_with_real_accel_structure():
    """⭐ The other half of the proof. A guard that fires on everything is not a
    guard — it is a constant."""
    c = _corpus()
    rng = np.random.default_rng(7)
    arm = _honest_arm(c["v0"], c["accel"], noise=0.35, rng=rng)
    det = AE.copy_detector(arm, c["gt"], c["v0"], DT)
    assert det["verdict"] == "CLEAN"
    assert det["echo_index"] < AE.ECHO_INDEX_ECHO
    assert det["echo_index_excess"] < AE.ECHO_EXCESS_SUSPECT
    assert det["cmd_accel_mae_mps2"] > AE.ECHO_ACCEL_MPS2
    assert det["dev_r"] is not None and det["dev_r"] > 0.8   # tracks the human
    assert det["dev_ratio"] > 0.5


def test_the_two_arms_are_separated_by_the_headline_scalar_alone():
    """The whole reason ``echo_index`` exists: one number, reported always, that
    orders the cheat above the honest arm without reading any prose."""
    c = _corpus()
    rng = np.random.default_rng(11)
    echo = AE.copy_detector(_echo_arm(c["v0"]), c["gt"], c["v0"], DT)
    real = AE.copy_detector(_honest_arm(c["v0"], c["accel"], 0.35, rng),
                            c["gt"], c["v0"], DT)
    assert echo["echo_index"] - real["echo_index"] > 0.5


def test_r_alone_has_almost_no_dynamic_range_which_is_why_it_is_never_alone():
    """⛔ The instrument defect this design avoids, MEASURED rather than asserted.

    Both the arm's along-track profile and hold-v0's are monotone ramps from the
    origin, so ``r`` is pinned near 1 for *any* forward-moving plan. Measured
    here: a **pure echo** scores 1.0000 and a planner braking at a violent
    **−2.0 m/s²** still scores ~0.987 — the whole usable range of ``r`` between
    "literal copy" and "emergency stop" is under **2 %**. Any threshold on it is
    knife-edge. ``echo_index`` separates the same two arms by > 0.5."""
    c = _corpus()
    braking = _path_from_accel(c["v0"], torch.full((c["n"],), -2.0))
    hard = AE.copy_detector(braking, c["gt"], c["v0"], DT)
    echo = AE.copy_detector(_echo_arm(c["v0"]), c["gt"], c["v0"], DT)
    assert hard["r_vs_holdv0_mean"] > 0.98          # r still says "basically a copy"
    r_range = echo["r_vs_holdv0_mean"] - hard["r_vs_holdv0_mean"]
    assert r_range < 0.02                           # ...over the whole range
    assert echo["echo_index"] - hard["echo_index"] > 0.5   # the joint one does not
    assert hard["verdict"] != "ECHO"
    assert hard["cmd_accel_mae_mps2"] > 1.0         # because it commands a lot


def test_the_dev_r_degeneracy_gate_is_physical_not_an_exact_zero_test():
    """⛔ REGRESSION PIN for a defect the can-it-fire proof actually caught.

    ``hold_v0_path -> _seq_geometry -> speed`` is **not bit-exact in float32**
    (MEASURED residue 1.9073e-06 m/s max, 5.6974e-07 RMS — the geometry recovers
    speed from ``norm(diff(positions))/dt``). With an exact ``== 0`` degeneracy
    test the detector **correlated that float noise** and reported
    ``dev_r = -0.0133`` for a **PURE ECHO** — a number a reader would take as a
    genuine-but-weak planner instead of a copy."""
    c = _corpus()
    echo = _echo_arm(c["v0"])
    resid = (FF._seq_geometry(echo, DT)["speed"] - c["v0"][:, None]).abs()
    assert 0.0 < float(resid.max()) < 1e-4          # non-zero, and tiny
    assert float(resid.max()) < AE.DEV_R_MIN_RMS_MPS
    det = AE.copy_detector(echo, c["gt"], c["v0"], DT)
    assert det["dev_r"] is None and det["dev_r_degenerate"] is True
    assert det["dev_r_rms_arm_mps"] <= AE.DEV_R_MIN_RMS_MPS
    # and the gate does NOT swallow a real departure
    rng = np.random.default_rng(4)
    real = AE.copy_detector(_honest_arm(c["v0"], c["accel"], 0.2, rng),
                            c["gt"], c["v0"], DT)
    assert real["dev_r_degenerate"] is False and real["dev_r"] is not None


# ============================================================================ #
# 3. the hold-v0 BASELINE — the binding admissibility verdict                   #
# ============================================================================ #
def test_a_pure_echo_cannot_beat_hold_v0_because_it_IS_hold_v0():
    """⛔ Not a small miss. The echo's delta against hold-v0 is exactly zero, so
    it can never be separated, so no longitudinal claim is admissible from it."""
    c = _corpus()
    b = AE.holdv0_baseline(_echo_arm(c["v0"]), c["gt"], c["v0"], DT, c["eid"],
                           n_boot=200)
    assert b["verdict"] == "NOT_SEPARATED"
    assert b["admissible"] is False
    assert b["metrics"]["speed_mae_mps"]["delta"] == 0.0
    assert "learned nothing beyond its own v0 input" in b["verdict_reason"]
    assert "paired_episode_cluster_bootstrap" in b["estimator"]


def test_an_arm_that_really_predicts_acceleration_BEATS_hold_v0_separated():
    c = _corpus()
    rng = np.random.default_rng(3)
    b = AE.holdv0_baseline(_honest_arm(c["v0"], c["accel"], 0.30, rng),
                           c["gt"], c["v0"], DT, c["eid"], n_boot=400)
    assert b["verdict"] == "BEATS_HOLDV0"
    assert b["admissible"] is True
    assert b["metrics"]["speed_mae_mps"]["separated"] is True
    assert b["metrics"]["speed_mae_mps"]["delta"] < 0
    assert b["metrics"]["along_abs_m"]["verdict"] == "BEATS_HOLDV0"


def test_an_arm_WORSE_than_hold_v0_is_reported_as_losing_not_as_a_near_miss():
    c = _corpus()
    rng = np.random.default_rng(5)
    # a planner that gets the SIGN of the acceleration wrong
    bad = _honest_arm(c["v0"], -1.5 * c["accel"], 0.1, rng)
    b = AE.holdv0_baseline(bad, c["gt"], c["v0"], DT, c["eid"], n_boot=400)
    assert b["verdict"] == "LOSES_TO_HOLDV0"
    assert b["admissible"] is False
    assert "0-parameter copy" in b["verdict_reason"]


def test_without_eid_there_is_no_separation_and_therefore_no_admission():
    """⛔ A point delta cannot discharge a SEPARATION requirement, and a
    quadrature combination of two single-arm intervals is not offered."""
    c = _corpus()
    rng = np.random.default_rng(9)
    b = AE.holdv0_baseline(_honest_arm(c["v0"], c["accel"], 0.3, rng),
                           c["gt"], c["v0"], DT, eid=None)
    assert b["verdict"] == "NO_INTERVAL"
    assert b["admissible"] is False
    row = b["metrics"]["speed_mae_mps"]
    assert row["estimator"] == "UNAVAILABLE" and row["separated"] is False
    assert "quadrature" in row["reason"]


def test_the_baseline_never_uses_the_deprecated_estimator():
    c = _corpus()
    b = AE.holdv0_baseline(_echo_arm(c["v0"]), c["gt"], c["v0"], DT, c["eid"],
                           n_boot=100)
    blob = repr(b)
    assert "overlapping_holdout_se" not in blob or "NOT " in b["estimator"]
    for m in AE.BASELINE_METRICS:
        assert b["metrics"][m]["estimator"] == "paired_episode_cluster_bootstrap"


# ============================================================================ #
# 4. the v0 SHUFFLE — reused, not reinvented                                    #
# ============================================================================ #
def test_the_shuffle_control_points_at_the_EXISTING_machinery():
    """⛔ ``--speed-echo-control`` already exists and produced the P1 row
    (R² 0.995 → −0.72). This module must NAME it, not grow a rival."""
    for token in ("probe_latent_state.py", "--speed-echo-control",
                  "speed_echo_control=True", "pbattery_watcher.py"):
        assert token in AE.SHUFFLE_PRODUCER


def test_the_existing_machinery_is_actually_there_at_the_cited_lines():
    """⛔ 'Absence found at one location is not absence' has a twin: a CITATION
    that was never checked. This one is checked — against the real file."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "stack" / "scripts" / "probe_latent_state.py").read_text(
        encoding="utf-8", errors="replace")
    assert "speed_echo_control" in src
    assert '"--speed-echo-control"' in src
    assert "torch.randperm" in src


def test_shuffle_v0_permutes_and_reports_its_own_degeneracy():
    v = torch.arange(50, dtype=torch.float32)
    a, pa = AE.shuffle_v0(v, seed=1)
    b, pb = AE.shuffle_v0(v, seed=1)
    assert torch.equal(a, b)                      # seeded == reproducible
    assert sorted(a.tolist()) == sorted(v.tolist())    # a permutation
    assert not torch.equal(a, v)
    assert set(pa.tolist()) == set(range(50))


def test_shuffle_control_is_a_work_item_not_a_pass_when_the_rerun_is_absent():
    c = _corpus()
    s = AE.shuffle_control(_echo_arm(c["v0"]), c["gt"], c["v0"], DT)
    assert s["status"] == "UNAVAILABLE" and s["n"] == 0
    assert "WORK ITEM" in s["reason"]
    assert "probe_latent_state.py" in s["reason"]
    assert s["n_windows_available"] == c["n"]


def test_shuffle_control_FIRES_on_an_echo_that_follows_the_lie():
    """⭐ The falsifier's can-it-fire proof. Re-run a pure echo with v0 permuted:
    its plan follows the FALSE speed exactly, so it tracks the lie."""
    c = _corpus()
    vs, _ = AE.shuffle_v0(c["v0"], seed=0)
    s = AE.shuffle_control(_echo_arm(c["v0"]), c["gt"], c["v0"], DT,
                           pred_shuffled=_echo_arm(vs), v0_shuffled=vs,
                           eid=c["eid"], n_boot=200)
    assert s["verdict"] == "ECHO"
    assert s["tracks_shuffled_v0"] == 0.0
    assert s["rms_to_shuffled_v0_mps"] == 0.0


def test_shuffle_control_does_NOT_fire_on_a_planner_that_ignores_the_false_v0():
    """A planner whose speed profile is anchored in the scene, not in its v0
    input, keeps predicting the same thing when v0 is permuted.

    ⛔ And the verdict is named **NOT_AN_ECHO**, not "USES_SPEED": this very arm
    ignores v0 completely — its degradation is exactly 0.0 — so a label claiming
    it *uses* speed would be the instrument asserting more than it measured."""
    c = _corpus()
    vs, _ = AE.shuffle_v0(c["v0"], seed=0)
    rng = np.random.default_rng(2)
    arm = _honest_arm(c["v0"], c["accel"], 0.2, rng)
    s = AE.shuffle_control(arm, c["gt"], c["v0"], DT,
                           pred_shuffled=arm,        # unchanged by the lie
                           v0_shuffled=vs, eid=c["eid"], n_boot=200)
    assert s["verdict"] == "NOT_AN_ECHO"
    assert s["tracks_shuffled_v0"] > 0.9
    assert s["degradation"]["delta"] == 0.0            # it ignored v0 entirely
    assert "establishes nothing else" in s["verdict_reason"]
    assert "USES_SPEED" not in s["verdict"]


def test_shuffle_control_refuses_a_rerun_on_different_windows():
    c = _corpus()
    s = AE.shuffle_control(_echo_arm(c["v0"]), c["gt"], c["v0"], DT,
                           pred_shuffled=torch.zeros(3, H, 2))
    assert s["status"] == "UNAVAILABLE" and "refusing" in s["reason"]


# ============================================================================ #
# 5. v0 RESOLUTION — and the refusal that keeps the control falsifiable         #
# ============================================================================ #
def test_v0_is_never_imputed_from_the_future():
    """⛔ THE LOAD-BEARING REFUSAL. Deriving v0 from gt/pred would make every
    control compare the arm against a quantity the arm's own error moves — the
    exact unfalsifiability the ruling exists to prevent."""
    c = _corpus()
    win = {"pred": _echo_arm(c["v0"]), "gt": c["gt"], "eid": c["eid"],
           "wp_steps": [1, 2, 3, 4], "dt_s": DT}
    v, why = AE.resolve_v0(win, c["n"])
    assert v is None
    assert "NOT imputed from" in why and "WORK ITEM" in why


@pytest.mark.parametrize("key", ["v0", "speed"])
def test_v0_is_read_from_either_canonical_key(key):
    c = _corpus(n=40, n_ep=4)
    v, prov = AE.resolve_v0({key: c["v0"]}, 40)
    assert v is not None and torch.allclose(v, c["v0"])
    assert key in prov


def test_v0_falls_back_to_the_lead_blocks_own_ego_speed():
    """A caller who supplied a lead block already supplied v0 without knowing —
    ``lead['speeds']`` IS the ego speed at t0 (the time-gap denominator)."""
    c = _corpus(n=30, n_ep=3)
    v, prov = AE.resolve_v0({"lead": {"speeds": c["v0"].numpy()}}, 30)
    assert v is not None and torch.allclose(v, c["v0"])
    assert "lead" in prov


def test_a_wrong_length_v0_is_refused_rather_than_broadcast():
    v, why = AE.resolve_v0({"v0": torch.tensor([1.0, 2.0])}, 40)
    assert v is None and "n=40" in why


# ============================================================================ #
# 6. ⭐ THE WIRING — automatic, through the entry point the real evals use       #
# ============================================================================ #
def test_all_families_attaches_the_controls_WITHOUT_being_asked():
    """⛔ Item 4 of the ruling: reported automatically with any LONGITUDINAL
    number, never on request. No flag is passed here."""
    c = _corpus()
    fam = FF.all_families(_win(_echo_arm(c["v0"]), c), n_boot=100)
    ae = fam["longitudinal"]["anti_echo"]
    assert ae["status"] == "OK"
    assert set(ae) >= {"holdv0_baseline", "copy_detector", "shuffle_control"}
    assert ae["longitudinal_claim_admissible"] is False
    assert fam["_longitudinal_claim_admissible"] is False
    assert "holdv0=" in fam["_anti_echo_summary"]


def test_the_echo_is_flagged_end_to_end_and_the_honest_arm_is_not():
    """⭐ THE PROOF, at the entry point that matters. Same windows, same call,
    two arms: the cheat is refused and the real planner is admitted."""
    c = _corpus()
    rng = np.random.default_rng(13)
    echo = FF.all_families(_win(_echo_arm(c["v0"]), c), n_boot=300)
    real = FF.all_families(
        _win(_honest_arm(c["v0"], c["accel"], 0.30, rng), c), n_boot=300)
    assert echo["_longitudinal_claim_admissible"] is False
    assert real["_longitudinal_claim_admissible"] is True
    assert echo["longitudinal"]["anti_echo"]["flagged"] is True
    assert real["longitudinal"]["anti_echo"]["flagged"] is False
    assert echo["longitudinal"]["anti_echo"]["copy_detector"]["verdict"] == "ECHO"


def test_the_echo_arm_would_have_LOOKED_FINE_on_the_metrics_it_is_beside():
    """⛔ THE WHOLE REASON THE CONTROL IS BINDING. On this corpus the pure echo
    carries a respectable-looking speed MAE and a target-speed accuracy that a
    report would happily quote. Only the control says it is a copy."""
    c = _corpus()
    lon = FF.all_families(_win(_echo_arm(c["v0"]), c), n_boot=100)["longitudinal"]
    assert lon["target_speed_acc"]["within_2.0_mps"] > 0.5    # looks like skill
    assert math.isfinite(lon["speed_mae_mps"])
    assert lon["anti_echo"]["longitudinal_claim_admissible"] is False


def test_the_block_is_incomplete_while_the_controls_cannot_run():
    """No v0 in the window ⇒ the condition is UNDISCHARGED, and ``_complete``
    must say so rather than reading as a compliant block."""
    c = _corpus()
    win = {"pred": _echo_arm(c["v0"]), "gt": c["gt"], "eid": c["eid"],
           "wp_steps": [1, 2, 3, 4], "dt_s": DT}
    fam = FF.all_families(win)
    ae = fam["longitudinal"]["anti_echo"]
    assert ae["status"] == "UNAVAILABLE" and ae["n"] == 0
    assert fam["_complete"] is False
    assert "UNDISCHARGED" in ae["⛔_consequence"]


def test_the_controls_survive_the_dense_path_and_carry_their_grid():
    """The rate metrics are dt-sensitive; the controls must be computed on the
    SAME grid the family reports, not on an assumed 0.1 s tick."""
    c = _corpus(n=60, n_ep=6)
    win = _win(_echo_arm(c["v0"]), c)
    win["pred_dense"] = win.pop("pred")
    win["gt_dense"] = c["gt"]
    win["dt_s"] = DT
    fam = FF.all_families(win, n_boot=100)
    ae = fam["longitudinal"]["anti_echo"]
    assert ae["status"] == "OK"
    assert ae["holdv0_baseline"]["dt_s"] == fam["_grid"]["dt_s"] == DT


def test_the_ruling_and_the_measured_context_travel_with_the_block():
    """⚠️ Admitting v0 removed an ARGUMENT, not a DEFICIT — and that must not be
    lost between this artifact and a report quoting it."""
    c = _corpus(n=40, n_ep=4)
    ae = FF.all_families(_win(_echo_arm(c["v0"]), c),
                         n_boot=100)["longitudinal"]["anti_echo"]
    assert "2026-08-16" in ae["ruling"]
    assert "3.527" in ae["⚠️_context"] and "1.1888" in ae["⚠️_context"]
    assert ae["version"] == AE.VERSION


# ============================================================================ #
# 7. clause 5 — an unavailable branch states its REASON and its n              #
# ============================================================================ #
def test_every_unavailable_branch_carries_a_reason_and_an_n():
    c = _corpus(n=20, n_ep=2)
    no_v0 = AE.anti_echo(_echo_arm(c["v0"]), c["gt"], DT, {"eid": c["eid"]})
    assert no_v0["status"] == "UNAVAILABLE"
    assert no_v0["reason"] and no_v0["n"] == 0
    ok = AE.anti_echo(_echo_arm(c["v0"]), c["gt"], DT,
                      {"v0": c["v0"], "eid": c["eid"]}, n_boot=100)
    sc = ok["shuffle_control"]
    assert sc["status"] == "UNAVAILABLE" and sc["reason"] and sc["n"] == 0
