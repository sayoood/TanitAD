"""BANDED-NL (MM 2026-11-04) -- `required_values_when`: refcv9-e1rig holds a `--r9-map-banded` argv to near lift 10, every other to 20.

MEASURED 2026-11-04 (the f078b7e dev-box gate of M1): `--map-hires-near-lift-m 20 is REQUIRED for a refcv9-e1rig launch ... and the argv
has ['10']` -- the profile's required_values carried refcv8's 20 unconditionally, while the banded map head REFUSES any near lift but 10
(MapHiresConfig(banded); `_E1_MAP_BANDED["near_lift_m"]`, the value its own /2-banded G-MAP-OVERFIT record ran).  M1 / M2a / M2b were
built, tested and record-bound -- and unreachable from the launch gate.

Every arm reads the REAL runs.d argv.  RED arms: a banded argv with 20, a non-banded argv with 10, and the rule deleted from the profile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
FLAG, NL = "--r9-map-banded", "--map-hires-near-lift-m"


def argv(name: str) -> list[str]:
    return list(json.loads((RUNS / f"r9e1_{name}.argv.json").read_text(encoding="utf-8"))["argv"])


def nl_reasons(prof: dict, av: list[str]) -> list[str]:
    reasons, _ = LG.required_on_reasons(prof, av)
    return [r for r in reasons if r.startswith(NL)]


def with_value(av: list[str], flag: str, value: str) -> list[str]:
    out = list(av)
    out[out.index(flag) + 1] = value
    return out


def test_the_rule_is_the_banded_heads_own_literal():
    assert RIG["required_values_when"] == {FLAG: {NL: "10"}, "--r9-map-banded-near-m=20": {NL: "20"}}      # E1-NEAR20: M1N20 (near band 0-20 m) -> 20
    assert float(RIG["required_values_when"][FLAG][NL]) == LG._E1_MAP_BANDED["near_lift_m"] == 10.0
    assert RIG["required_values"][NL] == "20"                          # every non-banded arm keeps refcv8's 20
    assert FLAG in RIG["required_values_when_why"]


@pytest.mark.parametrize("arm", ["M1", "M2a", "M2b"])
def test_the_banded_arms_pass_the_near_lift_check(arm):
    av = argv(arm)
    assert FLAG in av and LG.flag_values(av, NL) == ["10"]
    assert nl_reasons(RIG, av) == []


@pytest.mark.parametrize("arm", ["C0", "C1", "B1", "B2", "T1"])
def test_the_non_banded_arms_are_still_held_to_20(arm):
    av = argv(arm)
    assert FLAG not in av and LG.flag_values(av, NL) == ["20"]
    assert nl_reasons(RIG, av) == []


def test_RED_a_banded_argv_with_20_is_refused_naming_the_trigger():
    r = nl_reasons(RIG, with_value(argv("M1"), NL, "20"))
    assert len(r) == 1 and "10 is REQUIRED" in r[0] and "--r9-map-banded is passed" in r[0] and "['20']" in r[0]


def test_RED_a_non_banded_argv_with_10_is_refused():
    r = nl_reasons(RIG, with_value(argv("C0"), NL, "10"))
    assert len(r) == 1 and "20 is REQUIRED" in r[0] and "['10']" in r[0]


def test_RED_without_the_rule_the_banded_arm_is_refused_as_on_f078b7e():
    prof = {k: v for k, v in RIG.items() if k != "required_values_when"}
    r = nl_reasons(prof, argv("M1"))
    assert len(r) == 1 and "20 is REQUIRED" in r[0] and "['10']" in r[0]


def test_a_profile_without_the_key_is_unchanged():
    assert "required_values_when" not in LG.PROFILES["refcv8"] and "admitted_values_when" not in LG.PROFILES["refcv8"]
    # THE FLIP (MM 2026-11-07): the full family carries `required_values_when` for the crowned T1h BUNDLE; K2-ARGV (MM 2026-11-07) MERGES the near-lift rules into the same dict (a banded
    # argv -> near lift 10, `--r9-map-banded-near-m=20` -> 20: `LG._NEAR_LIFT_WHEN`, the SAME constant refcv9-e1rig uses) and the W-MAP-2 co-requirement. Still no map-weight entry and
    # no `admitted_values_when` (the weight is `map_weight_reasons`'s: free and positive with the banded head). Its pins: test_launch_gate_r9f_t1h.py, test_r9e2_k2_argv.py
    for name in ("refcv9-e2", "refcv9-full", "refcv9-full-d1"):
        rvw = LG.PROFILES[name]["required_values_when"]
        assert set(rvw) == set(LG._R9F_T1H_ADMITTED) | {FLAG, "--r9-map-banded-near-m=20", "--r9-plan-map-channels"}, name
        assert {k: v for k, v in rvw.items() if k in LG._NEAR_LIFT_WHEN} == LG._NEAR_LIFT_WHEN and "admitted_values_when" not in LG.PROFILES[name]
        assert {f for k, over in rvw.items() if k in LG._R9F_T1H_ADMITTED for f in over} <= set(LG._R9F_T1H_ADMITTED)
        assert not any("--w-map-hires" in over for over in rvw.values()), name


# ---- the map weight: a banded arm may carry ONE of the E1 map arms' pre-registered weights; every other arm keeps refcv8's 1.0 ----
W = "--w-map-hires"


def w_reasons(prof: dict, av: list[str]) -> list[str]:
    reasons, _ = LG.required_on_reasons(prof, av)
    return [r for r in reasons if r.startswith(W)]


def test_the_admitted_weights_are_the_argv_builders_map_arms():
    import r9_e1_make_argv as MA
    assert set(LG._E1_MAP_BANDED["w_map_hires_admitted"]) == set(MA.MAP_ARMS.values()) == {1.0, 0.511, 1.018}
    assert RIG["admitted_values_when"] == {FLAG: {W: ("1", "0.511", "1.018")}}
    assert RIG["required_values"][W] == "1.0"


@pytest.mark.parametrize("arm,w", [("M1", "1.0"), ("M2a", "0.511"), ("M2b", "1.018")])
def test_each_banded_arm_carries_its_own_weight_and_passes(arm, w):
    av = argv(arm)
    assert LG.flag_values(av, W) == [w]
    assert w_reasons(RIG, av) == []
    _, det = LG.required_on_reasons(RIG, av)
    assert W not in det["required_values"] and det["admitted_values"] == {W: ["1", "0.511", "1.018"]}


@pytest.mark.parametrize("bad", ["0.7", "2.0", "nan", "abc"])
def test_RED_a_banded_argv_with_any_other_weight_is_refused(bad):
    r = w_reasons(RIG, with_value(argv("M2a"), W, bad))        # nan / abc are ALSO refused by required_positive (a second reason)
    adm = [x for x in r if "must be ONE of ['1', '0.511', '1.018']" in x]
    assert len(adm) == 1 and "--r9-map-banded is passed" in adm[0] and f"['{bad}']" in adm[0]


def test_RED_a_non_banded_argv_with_an_m2_weight_is_refused():
    r = w_reasons(RIG, with_value(argv("C0"), W, "0.511"))
    assert len(r) == 1 and "1.0 is REQUIRED" in r[0] and "['0.511']" in r[0]


def test_RED_without_the_admitted_set_m2a_is_refused_as_on_f078b7e():
    prof = {k: v for k, v in RIG.items() if k != "admitted_values_when"}
    r = w_reasons(prof, argv("M2a"))
    assert len(r) == 1 and "1.0 is REQUIRED" in r[0] and "['0.511']" in r[0]


def test_every_e1_arm_argv_passes_the_profile_argv_rules_except_the_host_only_init():
    """The whole refusal list of the gate's own argv rules, per committed E1 argv (the f078b7e run refused M1 / M2a / M2b here)."""
    for p in sorted(RUNS.glob("r9e1_*.argv.json")):
        refusals, pend = LG.profile_argv_rules(RIG, argv(p.name[len("r9e1_"):-len(".argv.json")]), {}, [])
        real = [r for r in refusals if "not readable on this host" not in r and "missing on this host" not in r]
        assert real == [] and pend == [], (p.name, real, pend)
