"""K2DC G-LIVE COST (MM 4a3d05, 2026-11-14) -- the final gate at 68bda0d5 (Thor, 03:52:26Z, smoke 20261114T024837) refused K2DC as
PI-DECISION on "15.4067 s/step is 2.41x refcv6's 6.4 (> 1.25x)". That number was neither the smoke's nor the run's cost:

  * the map-hires cost judge (``judge_new2_live``) dropped the gradient-share reading interval but NOT the dose controller's, so ONE
    583.76 s controller reading (fp32 replay: 204.14 s forward + 379.59 s measure) was averaged into 86 steady 8.6 s intervals;
  * the top-level G-LIVE cost of the same smoke dropped both and read 8.6217 -- which UNDER-states the run, because the controller is
    part of K2DC's training method and fires 24 times in its 6,000 steps (~3.9 h of Thor).

The fix judges ``steady s/step + the controller's readings amortized over the RUN`` (``launch_gate.dose_ctl_amortized``): 8.62 + 24 x
583.76 / 6,000 = 10.96 s/step -- above the PI's recorded 10.5, so still PI-DECISION, now on the arm's real cost. Arms without the
controller are judged exactly as before. No GPU, no data, no torch model.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RUN_ARGV = json.loads((ROOT / "ops" / "runs.d" / "r9e2_K2DC.argv.json").read_text(encoding="utf-8"))["argv"]
#: the smoke's cadence: 100 steps, a gradient-share reading at row 50, the controller at row 100 (as in the 68bda0d5 final gate)
SMOKE_ARGV = ["--steps", "100", "--grad-share-every", "50", "--grad-share-at", "50",
              "--map-hires-share-target", "0.10", "--map-hires-share-every", "100", "--map-hires-share-group", "trunk"]
DC_ROW = {"step": 100, "dc_w_eff": 21.65864288915684, "dc_share_read": 0.12566355752056854, "dc_clamped": 0.0,
          "dcp_fp32_replay": 1.0, "dcp_fp32_replay_s": 583.7555390670896, "dcp_fp32_fwd_s": 204.14013614226133,
          "dcp_measure_s": 379.5914282184094}
STEADY = 8.6
APPROVAL_10_5 = {"max_s_per_step": 10.5, "pi_verbatim": "test"}


def _step_times():
    """100 opt.step timestamps (steps 1..100): steady 8.6 s, the gradient-share replay (768 s) inside interval 49 and the controller's
    replay (583.76 s) inside interval 98 -- where the 68bda0d5 smoke put them (its max interval read 592.13 s)."""
    iv = [STEADY] * 99
    iv[49] += 768.04
    iv[98] += DC_ROW["dcp_fp32_replay_s"]
    t, out = 100.0, [100.0]
    for v in iv:
        t += v
        out.append(t)
    return out


def _judge(ctl_cost, approval=APPROVAL_10_5):
    prof = LG.PROFILES["refcv9-e2"]
    run1 = {"fwd_train_calls": 100, "fmap_hits": 100, "hires_first": ([], {}), "step_times": _step_times()}
    reasons, needs_pi, det = LG.judge_new2_live(prof, run1, None, SMOKE_ARGV, {"groups": []}, cuda=True, approval=approval,
                                                ctl_cost=ctl_cost)
    return reasons, needs_pi, det["cost"]


def _cost(reasons):
    """The judge's COST reasons only: this fixture builds no leaf groups and no class-weight file, so NEW-2's other checks refuse it."""
    return [r for r in reasons if "s/step" in r or "reading cost" in r]


def _ctl():
    return LG.dose_ctl_amortized(RUN_ARGV, [{"step": 50, "gs_fp32_replay_s": 768.0}, DC_ROW, {"step": 101}])


# ============================================================================================ the amortized term ==
def test_the_run_argv_fires_the_controller_24_times_in_6000_steps():
    assert LG.dc_reading_rows(RUN_ARGV) == list(range(250, 6001, 250))
    c = _ctl()
    assert (c["n_readings"], c["run_steps"], c["smoke_reading_rows"]) == (24, 6000, [100])
    assert c["reading_s_mean"] == 583.76
    assert c["s_per_step"] == round(24 * DC_ROW["dcp_fp32_replay_s"] / 6000, 4) == 2.335


def test_no_controller_means_no_term():
    off = [a for a in RUN_ARGV]
    i = off.index("--map-hires-share-target")
    del off[i:i + 2]
    assert LG.dose_ctl_amortized(off, [DC_ROW]) is None


def test_a_firing_without_measured_seconds_gives_no_term_and_says_why():
    c = LG.dose_ctl_amortized(RUN_ARGV, [{"step": 100, "dc_w_eff": 25.0}])
    assert c["s_per_step"] is None and "dcp_fp32_replay_s" in c["why"]


def test_without_a_replay_the_measure_seconds_are_the_reading_cost():
    row = {"step": 100, "dc_w_eff": 25.0, "dcp_measure_s": 12.0}
    assert LG.dose_ctl_amortized(RUN_ARGV, [row])["s_per_step"] == round(24 * 12.0 / 6000, 4)


# ============================================================================================ the judgement ==
def test_the_controller_interval_no_longer_enters_the_steady_mean():
    """MEASURED refusal: the controller's interval stayed in, 15.4067 s/step. Here: one 592 s interval among 86."""
    _, _, cost = _judge(None)
    assert cost["s_per_step_mean"] == pytest.approx(STEADY, abs=1e-3), cost
    assert cost["reading_rows"] == [50, 100]


def test_the_judged_cost_adds_the_readings_and_goes_to_the_pi_above_the_approval():
    reasons, needs_pi, cost = _judge(_ctl())
    assert cost["s_per_step_judged"] == round(STEADY + 2.335, 4)
    assert needs_pi and not _cost(reasons)
    assert cost["verdict"].startswith(f"PI-DECISION: {round(STEADY + 2.335, 4)} s/step (steady ")
    assert "the dose controller's 24 readings x 583.76 s over 6000 steps = 2.335" in cost["verdict"]
    assert cost["dose_ctl"]["n_readings"] == 24


def test_an_approval_covering_the_judged_cost_passes():
    reasons, needs_pi, cost = _judge(_ctl(), approval={"max_s_per_step": 12.3})
    assert not needs_pi and not _cost(reasons)
    assert cost["verdict"] == "above +25 % but within the PI's recorded approval"


def test_the_steady_cost_alone_would_have_passed_the_approval():
    """What the amortized term changes: without it the same smoke clears 10.5 and the ~3.9 h of readings is never seen."""
    _, needs_pi, cost = _judge(None)
    assert not needs_pi and cost["verdict"] == "above +25 % but within the PI's recorded approval"


def test_an_unmeasured_controller_cost_is_a_reason_not_a_pass():
    reasons, needs_pi, cost = _judge({"n_readings": 24, "run_steps": 6000, "reading_s_mean": None, "s_per_step": None, "why": "x"})
    assert any("reading cost was not measured" in r for r in reasons) and not needs_pi
    assert "verdict" not in cost


def test_an_arm_without_the_controller_is_judged_as_before():
    _, _, cost = _judge(None, approval=None)
    assert "s_per_step_judged" not in cost and "dose_ctl" not in cost
    assert cost["verdict"] == (f"PI-DECISION: {cost['s_per_step_mean']} s/step is {cost['ratio']:.2f}x refcv6's 6.4 (> 1.25x) "
                               f"-- SPEC 6.2 item 5")


def test_the_gate_passes_the_amortized_term_to_the_judge():
    """Reachability: the G-LIVE caller must hand ``dose_ctl_amortized`` the LAUNCH argv (``ctx.argv``) -- the smoke argv fires at the
    smoke's cadence, not the run's."""
    src = (ROOT / "scripts" / "launch_gate.py").read_text(encoding="utf-8")
    assert re.search(r"judge_new2_live\(ctx\.prof, run1, cfg1, argv1, gt,.{0,300}?"
                     r"ctl_cost=dose_ctl_amortized\(ctx\.argv, train_rows\(rows1\)\)\)", src, re.S)
