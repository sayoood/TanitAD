"""Launch gate, SHARE-ALARM floor exemption for the pre-registered K2DC02cl clamp arm (PI 2026-11-22 "Yes, use the clamp").

Why: K2DC02cl's design dose is ~0.02 of the trunk gradient, so the 0.05 floor refused it BY CONSTRUCTION (MEASURED 2026-11-23, Thor gate3/K2DC02cl/
FAIL_a4631f4a2b90.json: mean 0.0018 of 4 readings [0.0054, 0.0097, -0.0061, -0.002]). `SHARE_ALARM_DOSE_ARMS` exempts ONLY the floor comparison, and only when
the argv's --map-hires-share-target / --map-hires-w-bounds equal the table's.

  (a) listed arm + measured readings -> no SHARE-ALARM reason, `exempt` recorded     (b) a non-listed arm -> today's floor reason
  (c) listed arm, other target / bounds / missing flag -> the NEW mismatch reason    (d) listed arm, no readings -> the existing missing-alarm reason
  (e) listed arm + stop-gradient flag -> the existing stop-gradient behaviour        (f) terms flag not separating map_hires still refuses
  MUTATION: delete the table entry -> (a) goes RED (scratch-copy check, see the hold's HANDOFF.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
CL = json.loads((RUNS / "r9e2_K2DC02cl.argv.json").read_text(encoding="utf-8"))["argv"]
K2DC02 = json.loads((RUNS / "r9e2_K2DC02.argv.json").read_text(encoding="utf-8"))["argv"]
K2M1N20 = json.loads((RUNS / "r9e2_K2M1N20.argv.json").read_text(encoding="utf-8"))["argv"]
FULL = LG.PROFILES["refcv9-full"]
KEY = FULL["map_share_alarm"]["row_key"]
MEASURED = [0.0054, 0.0097, -0.0061, -0.002]          # Thor gate3/K2DC02cl/FAIL_a4631f4a2b90.json


def rows_of(*vals):
    return [{"step": 25 * (i + 1), KEY: v, "gs_fp32_replay": 1.0} for i, v in enumerate(vals)]


def live(argv, rows):
    return LG.full_profile_live_reasons(dict(FULL, lr_schedule_rule=None), argv, argv, rows, None)


def alarm(reasons):
    return [r for r in reasons if r.startswith("SHARE-ALARM")]


def test_real_cl_argv_is_what_the_table_expects():
    assert LG.wcost_out_name(CL) == "r9e2-K2DC02cl" in LG.SHARE_ALARM_DOSE_ARMS
    assert LG.flag_values(CL, "--map-hires-share-target") == ["0.02"] and LG.flag_values(CL, "--map-hires-w-bounds") == ["2.5,3.5"]


def test_a_listed_arm_with_the_measured_readings_is_exempt_and_recorded():
    r, d = live(CL, rows_of(*MEASURED))
    assert alarm(r) == [], r
    ex = d["share_alarm"]["exempt"]
    assert ex["mean"] == pytest.approx(0.0018, abs=1e-4) and ex["floor"] == 0.05
    assert ex["target"] == 0.02 and ex["bounds"] == [2.5, 3.5] and ex["prereg"].endswith("PREREG_K2DC02CL_2026-11-22.md")


def test_a2_a_reading_at_or_above_the_floor_is_not_marked_exempt():
    r, d = live(CL, rows_of(0.08, 0.09, 0.1))
    assert alarm(r) == [] and "exempt" not in d["share_alarm"]


@pytest.mark.parametrize("name,argv", [("K2DC02", K2DC02), ("K2M1N20", K2M1N20)])
def test_b_a_non_listed_arm_keeps_the_floor_reason(name, argv):
    r, d = live(argv, rows_of(*MEASURED))
    a = alarm(r)
    assert len(a) == 1 and a[0].startswith("SHARE-ALARM: the banded map takes 0.0018") and "below the floor 0.05" in a[0], (name, r)
    assert "exempt" not in d["share_alarm"]


def test_b2_the_cl_argv_under_another_arm_key_keeps_the_floor_reason():
    argv = LG.set_flag(list(CL), "--out", ["/home/nvidia/r8ns/r9e2/runs/r9e2-NOTCL"])
    r, _ = live(argv, rows_of(*MEASURED))
    assert len(alarm(r)) == 1 and "below the floor 0.05" in r[0]


@pytest.mark.parametrize("flag,vals", [("--map-hires-share-target", ["0.05"]), ("--map-hires-w-bounds", ["2.0,4.0"]),
                                       ("--map-hires-share-target", None), ("--map-hires-w-bounds", None),
                                       ("--map-hires-w-bounds", ["2.5,x"])])
def test_c_listed_arm_with_a_mismatched_argv_gets_the_new_refusal(flag, vals):
    argv = LG.set_flag(list(CL), flag, vals)
    r, d = live(argv, rows_of(*MEASURED))
    a = [x for x in alarm(r) if "listed in SHARE_ALARM_DOSE_ARMS" in x]
    assert len(a) == 1 and "does not match" in a[0] and "r9e2-K2DC02cl" in a[0], (flag, vals, r)
    assert "exempt" not in d["share_alarm"] and not any("below the floor" in x for x in r)


def test_d_listed_arm_with_no_readings_is_still_a_fail():
    r, _ = live(CL, [])
    assert len(alarm(r)) == 1 and "no `gs_trunk_proj_map_hires` reading although the banded map head is ON" in r[0] and "never a pass" in r[0], r


def test_e_listed_arm_stop_gradient_behaviour_is_unchanged():
    sg0 = LG.set_flag(list(CL), "--map-hires-trunk-grad-scale", ["0.0"])
    assert live(sg0, rows_of(0.0, 0.0))[0] == [] or all(not x.startswith("SHARE-ALARM") for x in live(sg0, rows_of(0.0, 0.0))[0])
    r, _ = live(sg0, rows_of(0.0, 1e-9))
    assert len(alarm(r)) == 1 and "OP-VOID" in r[0]
    r, _ = live(sg0, [])
    assert len(alarm(r)) == 1 and "never a pass" in r[0]
    half = LG.set_flag(list(CL), "--map-hires-trunk-grad-scale", ["0.5"])
    r, _ = live(half, rows_of(*MEASURED))
    assert len(alarm(r)) == 1 and "partial trunk-gradient scale" in r[0]


def test_f_listed_arm_whose_terms_flag_does_not_separate_map_hires_still_refuses():
    argv = LG.set_flag(list(CL), "--grad-share-terms", ["agent,box3d"])
    r, _ = live(argv, rows_of(*MEASURED))
    assert len(alarm(r)) == 1 and "does not separate" in r[0], r
