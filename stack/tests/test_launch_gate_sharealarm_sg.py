"""Launch gate, SHARE-ALARM under the map->trunk stop-gradient (MM 2026-11-19, arm K2SG).

Why: K2SG = K2M1N20 + `--map-hires-trunk-grad-scale 0.0`. At S = 0 the map term's gradient on every trunk parameter is exactly zero, so the smoke's
`gs_trunk_proj_map_hires` reads exactly 0.0 BY DESIGN, and the floor alarm (0.05) would refuse the arm with "the map weight is too small to shape layer2"
-- the wrong diagnosis. PREREG_K2SG s2 commits the right check: at S = 0 every reading must be exactly 0.0, else OP-VOID.

  S absent or 1.0  -> today's alarm, byte-for-byte        S = 0.0 -> exact-zero identity        0 < S < 1, out of range, unparsable -> REFUSE
  (g) MUTATION: the S = 0 branch deleted from the gate's source makes (b) go RED (it falls back to the floor alarm).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

FULL = LG.PROFILES["refcv9-full"]
RUNS = ROOT / "ops" / "runs.d"
K2M1N20 = json.loads((RUNS / "r9e2_K2M1N20.argv.json").read_text(encoding="utf-8"))["argv"]
K2SG = json.loads((RUNS / "r9e2_K2SG.argv.json").read_text(encoding="utf-8"))["argv"]
KEY = FULL["map_share_alarm"]["row_key"]
FLAG = "--map-hires-trunk-grad-scale"


def with_s(s):
    """K2M1N20's argv with the stop-gradient scale `s` (None = flag absent)."""
    return LG.set_flag(list(K2M1N20), FLAG, None if s is None else [str(s)])


def rows_of(*vals):
    return [{"step": 25 * (i + 1), KEY: v, "gs_fp32_replay": 1.0} for i, v in enumerate(vals)]


def live(argv, rows, gate=LG):
    return gate.full_profile_live_reasons(dict(gate.PROFILES["refcv9-full"], lr_schedule_rule=None), argv, argv, rows, None)


def alarm(reasons):
    return [r for r in reasons if r.startswith("SHARE-ALARM")]


# what the tip's code emits (copied from the tip's f-strings) for the two floor outcomes
PASS_ROWS = rows_of(0.08, 0.09, 0.1)
FAIL_ROWS = rows_of(0.001, 0.002, 0.003)
FAIL_REASON = ("SHARE-ALARM: the banded map takes 0.0020 of the trunk gradient (mean of 3 reading(s) [0.001, 0.002, 0.003] of `gs_trunk_proj_map_hires`), "
               "below the floor 0.05 (E0j: %s) -- the map weight is too small to shape layer2" % FULL["map_share_alarm"]["e0j"])


def test_flag_name_is_pinned_to_the_trainer():
    from tanitad.train import map_trunk_gs as M
    assert LG._MAP_TRUNK_GS_FLAG == M.FLAG == FLAG
    assert LG.flag_values(K2SG, FLAG) == ["0.0"] and LG.flag_values(K2M1N20, FLAG) is None


@pytest.mark.parametrize("s", [None, 1.0, "1", "1.00"])
def test_a_S_absent_or_one_is_todays_alarm_byte_for_byte(s):
    r, d = live(with_s(s), PASS_ROWS)
    assert r == [] and "mode" not in d["share_alarm"]
    assert d["share_alarm"]["readings"] == [0.08, 0.09, 0.1] and d["share_alarm"]["n"] == 3
    r, d = live(with_s(s), FAIL_ROWS)
    assert r == [FAIL_REASON] and "mode" not in d["share_alarm"]
    r, _ = live(with_s(s), [])
    assert len(r) == 1 and r[0].startswith("SHARE-ALARM: the smoke carries no `gs_trunk_proj_map_hires` reading although the banded map head is ON")
    # absent and 1.0 are indistinguishable
    assert live(with_s(None), FAIL_ROWS) == live(with_s(1.0), FAIL_ROWS)


def test_b_S0_all_readings_exactly_zero_passes_the_row():
    r, d = live(K2SG, rows_of(0.0, 0.0, 0.0))
    assert alarm(r) == [] and r == []
    assert d["share_alarm"]["mode"] == "stop-gradient: every reading exactly 0.0 (PREREG_K2SG §2)"
    assert d["share_alarm"]["readings"] == [0.0, 0.0, 0.0]
    assert live(with_s(0), rows_of(0.0))[0] == [] and live(with_s("0"), rows_of(-0.0))[0] == []


def test_c_S0_one_nonzero_reading_is_op_void():
    r, _ = live(K2SG, rows_of(0.0, 1e-9, 0.0))
    assert len(alarm(r)) == 1 and "OP-VOID" in r[0] and "does not reach the trunk" in r[0] and "[1e-09]" in r[0], r
    r, _ = live(K2SG, rows_of(0.5, 0.5, 0.5))                       # a healthy floor-passing share is a VOID here
    assert "OP-VOID" in r[0]


def test_d_S0_no_reading_is_a_fail():
    r, d = live(K2SG, [])
    assert len(r) == 1 and "no `gs_trunk_proj_map_hires` reading" in r[0] and "never a pass" in r[0], r
    assert d["share_alarm"]["n"] == 0


@pytest.mark.parametrize("s", [0.5, 1e-6, 0.999])
def test_e_partial_scale_is_refused_whatever_the_readings(s):
    for rows in (rows_of(0.0, 0.0), PASS_ROWS, []):
        r, _ = live(with_s(s), rows)
        assert len(r) == 1 and "partial trunk-gradient scale" in r[0] and "no pre-registered share-alarm rule" in r[0], r


@pytest.mark.parametrize("s", [1.5, -0.1, "nan", "inf", "abc", ""])
def test_f_out_of_range_or_unparsable_scale_is_refused(s):
    r, _ = live(with_s(s), PASS_ROWS)
    assert len(r) == 1 and r[0].startswith("SHARE-ALARM:") and ("outside [0, 1]" in r[0] or "not a number" in r[0]), r


def test_flag_without_value_is_refused():
    r, _ = live(list(K2M1N20) + [FLAG], PASS_ROWS)
    assert len(r) == 1 and "not a number" in r[0], r


def test_flag_alone_without_the_banded_map_is_inert():
    argv = [a for a in with_s(0.0) if a != "--r9-map-banded"]
    assert live(argv, [])[0] == []


def _mutated_gate(drop_branch: bool, drop_partial: bool = False):
    src = (ROOT / "scripts" / "launch_gate.py").read_text(encoding="utf-8")
    needle = "        elif sg_s == 0.0:\n"
    assert src.count(needle) == 1, "the S = 0 branch moved: update the mutation"
    if drop_branch:
        src = src.replace(needle, "        elif False:\n")
    if drop_partial:
        n2 = "        elif sg_s != 1.0:\n"
        assert src.count(n2) == 1, "the partial-scale branch moved: update the mutation"
        src = src.replace(n2, "        elif False:\n")
    name = "launch_gate_mut_%d%d" % (int(drop_branch), int(drop_partial))
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / "scripts" / "launch_gate.py")
    sys.modules[name] = mod
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def test_g_MUTATION_deleting_the_S0_branch_turns_b_red():
    ok = _mutated_gate(False)                                          # control: the unmutated source passes (b)
    assert live(K2SG, rows_of(0.0, 0.0, 0.0), ok)[0] == []
    bad = _mutated_gate(True)                                          # S = 0 branch deleted: (b) is RED (the partial-scale refusal catches it)
    r, _ = live(K2SG, rows_of(0.0, 0.0, 0.0), bad)
    assert r and "partial trunk-gradient scale" in r[0], r
    # with the partial-scale refusal gone too (= the tip's behaviour) S = 0 falls back to the floor alarm and refuses the arm by design
    tip = _mutated_gate(True, True)
    r, _ = live(K2SG, rows_of(0.0, 0.0, 0.0), tip)
    assert r and "below the floor 0.05" in r[0] and "too small to shape layer2" in r[0], r
    r, _ = live(K2SG, rows_of(0.0, 1e-9, 0.0), tip)                  # (c) goes wrong the same way: "below the floor", not OP-VOID
    assert "OP-VOID" not in r[0]
