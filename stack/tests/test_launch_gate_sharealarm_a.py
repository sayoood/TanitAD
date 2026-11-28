"""Launch gate, SHARE-ALARM option A (MM 2026-11-15): three gradient-share readings in the smoke (steps 25 / 50 / 75) and the alarm judges their MEAN,
together with the dose controller's own `dc_share_read` rows when the controller is on. The 0.05 floor is unchanged.

Why: the K2DC final gate FAILED on 2026-11-15 09:04Z with `SHARE-ALARM: the banded map takes 0.0170 of the trunk gradient`, on ONE reading at step 50
of a config whose two earlier smokes read 0.0854 / 0.0817 (all at map weight w = 25; the controller's own reading at step 100: 0.0178 / 0.1257 / 0.1045).
The fixtures below are those three real smokes' rows (Thor: /home/nvidia/r8ns/r9e2/gate3/K2DC/smoke/<ts>/metrics.jsonl), reduced to the keys the gate reads.

Deliberate-regression arms (each must go RED if the mechanism is removed):
  M1  K2M1N20 at w = 1 (0.00125 of the trunk gradient)         -> FAIL
  M2  no reading at all                                        -> FAIL ("cannot be read")
  M3  the OLD statistic (min of the gs rows) on the 11-15 rows -> trips, i.e. option A is what changes the outcome
  M4  a controller arm whose readings are all low              -> FAIL (the mean is not a free pass)
  M5  a non-finite reading is not a reading
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

FULL = LG.PROFILES["refcv9-full"]
RUNS = ROOT / "ops" / "runs.d"
K2DC = json.loads((RUNS / "r9e2_K2DC.argv.json").read_text(encoding="utf-8"))["argv"]
K2M1N20 = json.loads((RUNS / "r9e2_K2M1N20.argv.json").read_text(encoding="utf-8"))["argv"]
ALARM_FLAG = FULL["map_share_alarm"]["flag"]
KEY = FULL["map_share_alarm"]["row_key"]

#: the three real K2DC smokes (config identical, w = 25 at every gs reading): gs@50 and the controller's firing row @100
REAL = {
    "20261113T214512": [{"step": 50, KEY: 0.08541779343610231, "gs_fp32_replay": 1.0},
                        {"step": 100, "dc_w_before": 25.0, "dc_w_eff": 25.0, "dc_share_read": 0.017757934240579372,
                         "dc_share_pred": 0.01775793424057937, "dc_clamped": 1.0, "dcp_fp32_replay": 1.0}],
    "20261114T024837": [{"step": 50, KEY: 0.08165220103818757, "gs_fp32_replay": 1.0},
                        {"step": 100, "dc_w_before": 25.0, "dc_w_eff": 21.65864288915684, "dc_share_read": 0.12566355752056854,
                         "dc_share_pred": 0.10000000000000002, "dc_clamped": 0.0, "dcp_fp32_replay": 1.0}],
    "20261115T080114": [{"step": 50, KEY: 0.01699545192889461, "gs_fp32_replay": 1.0},
                        {"step": 100, "dc_w_before": 25.0, "dc_w_eff": 24.313256823392102, "dc_share_read": 0.10445404708593331,
                         "dc_share_pred": 0.1, "dc_clamped": 0.0, "dcp_fp32_replay": 1.0}],
}


def live(argv, rows, prof=FULL):
    return LG.full_profile_live_reasons(dict(prof, lr_schedule_rule=None), argv, argv, rows, None)


def alarm(reasons):
    return [r for r in reasons if r.startswith("SHARE-ALARM")]


def test_the_floor_and_the_estimator_inputs_are_what_the_brief_says():
    assert LG._R9F_SHARE["floor"] == 0.05 and KEY == "gs_trunk_proj_map_hires"        # the floor is NOT moved
    assert LG.flag_values(K2DC, "--w-map-hires") == ["25.0"] and LG.flag_values(K2DC, "--map-hires-w-bounds") == ["1,25"]


@pytest.mark.parametrize("ts", sorted(REAL))
def test_the_three_real_smokes_all_pass_under_the_mean(ts):
    r, d = live(K2DC, REAL[ts])
    assert alarm(r) == [] and r == [], r
    sa = d["share_alarm"]
    assert sa["n"] == 2 and sa["floor"] == 0.05 and "mean" in sa["statistic"]
    vals = [x for x in (REAL[ts][0][KEY], REAL[ts][1]["dc_share_read"])]
    assert sa["readings"] == vals and abs(sa["mean"] - sum(vals) / 2) < 1e-15
    assert {x["src"] for x in sa["reading_rows"]} == {KEY, "dc_share_read"}


def test_M3_the_old_min_statistic_on_the_1115_rows_trips_and_the_mean_does_not():
    rows = REAL["20261115T080114"]
    old = min(float(r[KEY]) for r in rows if KEY in r)                                   # today's statistic: min over the row_key rows
    assert old < 0.05 and abs(old - 0.0170) < 5e-5                                       # => the 11-15 09:04Z FAIL
    r, d = live(K2DC, rows)
    assert alarm(r) == [] and d["share_alarm"]["mean"] > 0.05                            # option A: 0.0607
    assert abs(d["share_alarm"]["mean"] - (0.01699545192889461 + 0.10445404708593331) / 2) < 1e-12


def test_M1_K2M1N20_at_w1_reading_0_00125_FAILS_and_names_the_mean():
    argv = LG.set_flag(K2M1N20, "--w-map-hires", ["1.0"])
    r, d = live(argv, [{"step": 50, KEY: 0.00125}])
    assert len(r) == 1 and r[0].startswith("SHARE-ALARM: the banded map takes 0.0013"), r
    assert d["share_alarm"]["n"] == 1
    r, _ = live(argv, [{"step": 25, KEY: 0.00125}, {"step": 50, KEY: 0.00125}, {"step": 75, KEY: 0.00125}])
    assert len(alarm(r)) == 1


def test_M2_no_reading_FAILS_as_today_and_banded_off_is_inert():
    r, d = live(K2DC, [{"step": 50, "lr": 1e-4}])
    assert len(alarm(r)) == 1 and "cannot be read" in alarm(r)[0] and d["share_alarm"]["n"] == 0
    r, d = live(K2M1N20, [{"step": 50, "lr": 1e-4}])
    assert len(alarm(r)) == 1 and "cannot be read" in alarm(r)[0]
    assert live([x for x in K2M1N20 if x != ALARM_FLAG], [{"step": 50}])[0] == []         # banded OFF: inert
    assert "share_alarm" not in live(K2DC, [{"step": 50}], dict(FULL, map_share_alarm=None))[1]


def test_M4_a_controller_arm_with_every_reading_low_still_FAILS():
    rows = [{"step": 25, KEY: 0.01}, {"step": 50, KEY: 0.02}, {"step": 75, KEY: 0.015},
            {"step": 100, "dc_w_before": 25.0, "dc_w_eff": 25.0, "dc_share_read": 0.03, "dc_share_pred": 0.03, "dc_clamped": 1.0}]
    r, d = live(K2DC, rows)
    assert len(alarm(r)) == 1 and d["share_alarm"]["n"] == 4 and abs(d["share_alarm"]["mean"] - 0.0189 * 1.0) < 5e-4, (r, d)


def test_three_gs_readings_average_and_a_fixed_dose_arm_uses_only_its_gs_rows():
    r, d = live(K2M1N20, [{"step": 25, KEY: 0.02}, {"step": 50, KEY: 0.017}, {"step": 75, KEY: 0.12}])
    assert alarm(r) == [] and d["share_alarm"]["n"] == 3 and abs(d["share_alarm"]["mean"] - 0.052333333) < 1e-6
    r, d = live(K2M1N20, [{"step": 25, KEY: 0.02}, {"step": 50, KEY: 0.017}, {"step": 75, KEY: 0.09}])       # mean 0.0423 < floor
    assert len(alarm(r)) == 1 and abs(d["share_alarm"]["mean"] - 0.042333333) < 1e-6
    # a fixed-dose arm has no controller: a stray `dc_share_read` row is NOT a reading there
    r, d = live(K2M1N20, [{"step": 50, KEY: 0.02}, {"step": 100, "dc_w_eff": 25.0, "dc_share_read": 0.9}])
    assert d["share_alarm"]["n"] == 1 and len(alarm(r)) == 1


def test_M5_a_non_finite_reading_is_not_a_reading():
    r, d = live(K2M1N20, [{"step": 50, KEY: float("nan")}])
    assert d["share_alarm"]["n"] == 0 and len(alarm(r)) == 1 and "cannot be read" in alarm(r)[0]
    r, d = live(K2M1N20, [{"step": 25, KEY: float("nan")}, {"step": 50, KEY: 0.01}])
    assert d["share_alarm"]["n"] == 1 and len(alarm(r)) == 1


# ------------------------------------------------------------------------------------------------------------ the smoke argv + the accounting
def test_smoke_argv_takes_three_readings_with_the_alarm_flag_and_keeps_every_trainer_cadence_constraint():
    torch = pytest.importorskip("torch")                                                  # noqa: F841
    import refc_v3_train as T
    from tanitad.train import grad_share as GS
    from tanitad.train import map_dose_ctl as M
    run, dep = LG.smoke_argv(K2DC, 100, 50, alarm_flag=ALARM_FLAG)
    assert LG.flag_values(run, "--grad-share-at") == ["25,50,75"] and LG.flag_values(run, "--grad-share-every") == ["25"]
    assert LG.flag_values(run, "--log-every") == ["25"] and LG.flag_values(run, "--map-hires-share-every") == ["100"]
    assert LG.gs_reading_rows(run) == [25, 50, 75] and LG.dc_reading_rows(run) == [100]
    assert "--grad-share-every/--grad-share-at/--log-every" in dep and "--map-hires-share-every" in dep
    args = T.build_parser().parse_args(run)
    assert args.grad_share_every % args.log_every == 0 and args.map_hires_share_every % args.log_every == 0
    GS.parse_at(args.grad_share_at, args.grad_share_every, args.steps)                    # the trainer's own parser accepts it
    assert M.controller_for(args) is not None
    M.pin_args(args)                                                                      # the controller's own refusals accept it
    # the cost judgement drops the intervals of EVERY reading row (3 gs + 1 controller), automatically
    st = [float(i) * 8.6 for i in range(101)]
    for r in (25, 50, 75, 100):
        for j in range(r, 101):
            st[j] += 600.0                                                                # a 600 s reading before row r shifts everything after it
    ex = sorted(set(LG.gs_reading_rows(run)) | set(LG.dc_reading_rows(run)))
    assert ex == [25, 50, 75, 100]
    c = LG.step_cost(st, 10, exclude_rows=ex)
    assert c["reading_rows"] == ex and c["s_per_step_max"] < 20.0, c
    # replay accounting: 100 + 3 gs replays + 1 controller replay = 104 loss calls, counted from the rows' own stamps
    steps = [{"keys": [], "nonfinite": [], "vals": {}} for _ in range(104)]
    rows = [{"step": 25, "gs_fp32_replay": 1.0}, {"step": 50, "gs_fp32_replay": 1.0}, {"step": 75, "gs_fp32_replay": 1.0},
            {"step": 100, "dcp_fp32_replay": 1.0}]
    r, d = LG.judge_live([], [], steps, expect_steps=100, summary={"done": True, "step": 100}, train_rows=rows, grads={}, prior=None, exit_msg=None)
    assert not [x for x in r if "loss calls observed" in x], r
    assert d["fp32_replay_steps"] == [25, 50, 75] and d["dc_fp32_replay_steps"] == [100]
    r, _ = LG.judge_live([], [], steps[:101], expect_steps=100, summary={"done": True, "step": 100}, train_rows=rows, grads={}, prior=None, exit_msg=None)
    assert [x for x in r if "loss calls observed" in x]                                   # an unexplained shortfall still FAILS


def test_smoke_argv_is_unchanged_without_the_alarm_flag_or_without_the_instrument_or_without_the_banded_head():
    old, _ = LG.smoke_argv(K2DC, 100, 50)
    assert LG.flag_values(old, "--grad-share-at") == ["50"] and LG.flag_values(old, "--grad-share-every") == ["50"]
    assert LG.flag_values(old, "--log-every") == ["50"]
    assert LG.smoke_argv(K2DC, 100, 50, alarm_flag=None)[0] == old                        # default = today's behaviour
    off = LG.set_flag(LG.set_flag(K2DC, "--grad-share-every", ["0"]), "--grad-share-at", None)
    assert LG.smoke_argv(off, 100, 50, alarm_flag=ALARM_FLAG)[0] == LG.smoke_argv(off, 100, 50)[0]       # instrument off: nothing added
    nob = [x for x in K2DC if x != ALARM_FLAG]
    assert LG.smoke_argv(nob, 100, 50, alarm_flag=ALARM_FLAG)[0] == LG.smoke_argv(nob, 100, 50)[0]        # banded head not in the argv
    # K2M1N20 (fixed dose, no controller) also reads three times; there is no controller cadence to move
    run, _ = LG.smoke_argv(K2M1N20, 100, 50, alarm_flag=ALARM_FLAG)
    assert LG.flag_values(run, "--grad-share-at") == ["25,50,75"] and not LG.has_flag(run, "--map-hires-share-every")


def test_stamps_written_for_applies_the_same_rule_so_the_gates_own_record_is_not_refused():
    run, _ = LG.smoke_argv(K2DC, 100, 50, alarm_flag=ALARM_FLAG)
    same, d = LG.stamps_written_for(run, K2DC, 100, 50, alarm_flag=ALARM_FLAG)
    assert same and d["rule"] == "smoke_argv"
    same, _ = LG.stamps_written_for(run, K2DC, 100, 50)                                   # without the flag the rule differs -> refused (control)
    assert not same
