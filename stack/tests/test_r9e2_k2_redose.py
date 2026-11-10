"""K2M1N20 RE-DOSE (MM 4a3d05, 2026-11-10): `--w-map-hires 25`, pre-registered in
E:/Projects/TanitAD-artifacts/dagland/e2/PREREG_K2_REDOSE_2026-11-10.md (sha256 2c5c147bf10368bb88677f964f9474acf2b6eef4b96dfe06f72fa8e22043b5df).
K2M1N20 failed G-LIVE's SHARE-ALARM at w 1.0 (0.00125 < 0.05). Pins: the composer emits 25 for K2M1N20 and 1.0 for K0 / every other variant; the full family
(refcv9-e2 / -full / -full-d1) admits 25 on a banded argv and refuses a free value (24); E1's admitted set is UNCHANGED (25 refused there); a MUTATION (the composer emitting 1.0) goes RED."""
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
import r9_e2_make_argv as MK  # noqa: E402
from test_launch_gate_refcv9_full import K0, _full_argv  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
W = "--w-map-hires"
PREREG_SHA = "2c5c147bf10368bb88677f964f9474acf2b6eef4b96dfe06f72fa8e22043b5df"


def doc(v):
    return json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))


def w_of(argv):
    return LG.flag_values(argv, W)


def compose_k2m1n20():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    return MK.compose(json.loads(k0b.decode("utf-8")), "K2M1N20", 0.06472)


def assert_k2m1n20_is_25(argv):
    assert w_of(argv) == ["25.0"]


def test_composer_and_committed_files_carry_25_for_K2M1N20_and_1_for_everything_else():
    assert_k2m1n20_is_25(compose_k2m1n20())
    assert_k2m1n20_is_25(doc("K2M1N20")["argv"])
    assert w_of(doc("K0")["argv"]) == ["1.0"] == w_of(K0)
    for v in ("K2M3", "K2M3N20"):
        assert w_of(doc(v)["argv"]) == ["1.0"], v
    assert W in doc("K2M1N20")["argv"] and doc("K2M1N20")["lever"]["w_map_hires"] == 25.0
    assert PREREG_SHA in doc("K2M1N20")["lever"]["w_map_hires_redose"] and "PREREG_K2_REDOSE_2026-11-10" in doc("K2M1N20")["lever"]["w_map_hires_redose"]
    assert MK.REDOSE_PREREG_SHA256 == PREREG_SHA and MK.MAP_W_REDOSE == {"K2M1N20": 25.0}


def test_W_COST_weight_is_unchanged_by_the_redose():
    assert LG.flag_values(doc("K2M1N20")["argv"], "--w-r8-drivable") == ["0.06472"]


def test_MUTATION_a_composer_emitting_1_for_K2M1N20_goes_RED(monkeypatch):
    monkeypatch.setattr(MK, "MAP_W_REDOSE", {})
    assert w_of(compose_k2m1n20()) == ["1.0"]
    with pytest.raises(AssertionError):
        assert_k2m1n20_is_25(compose_k2m1n20())


def _argv_for(prof_name, w):
    base = doc("K2M1N20")["argv"]
    a = LG.set_flag(list(base), W, [w])
    return _full_argv(a) if prof_name != "refcv9-e2" else a


def _w_reasons(prof_name, w):
    reasons, _ = LG.required_on_reasons(LG.PROFILES[prof_name], _argv_for(prof_name, w))
    return [r for r in reasons if r.startswith(W)]


@pytest.mark.parametrize("name", ["refcv9-e2", "refcv9-full", "refcv9-full-d1"])
def test_the_full_family_admits_25_and_refuses_a_free_value_on_a_banded_argv(name):
    for ok in ("25", "25.0", "1.0", "0.511", "1.018"):
        assert _w_reasons(name, ok) == [], (name, ok)
    for bad in ("24", "26", "15", "0.01", "100"):
        r = _w_reasons(name, bad)
        assert len(r) == 1 and "must be ONE of" in r[0] and "'25'" in r[0], (name, bad, r)
    # `map_weight_reasons` (finite, > 0 with the banded head) still passes 25
    assert LG.map_weight_reasons(LG.PROFILES[name], _argv_for(name, "25.0"))[0] == []


def test_a_non_banded_full_argv_still_requires_refcv8s_1_0_and_25_does_not_leak_to_it():
    a = LG.set_flag(list(K0), W, ["25.0"])
    assert LG.map_weight_reasons(LG.PROFILES["refcv9-e2"], a)[0], "25 on a NON-banded argv must be refused by W-MAP-HIRES"
    assert LG.map_weight_reasons(LG.PROFILES["refcv9-e2"], K0)[0] == []


def test_E1s_admitted_set_is_unchanged_and_the_rig_refuses_25():
    assert tuple(LG._E1_MAP_BANDED["w_map_hires_admitted"]) == (1.0, 0.511, 1.018)
    assert LG._R9F_MAP_W_ADMITTED == (1.0, 0.511, 1.018, 25.0)
    assert LG.PROFILES["refcv9-e1rig"]["admitted_values_when"] == {"--r9-map-banded": {W: ("1", "0.511", "1.018")}}
    rig = LG.PROFILES["refcv9-e1rig"]
    banded = LG.set_flag(list(doc("K2M1N20")["argv"]), W, ["25.0"])
    reasons, _ = LG.required_on_reasons(rig, banded)
    assert any(r.startswith(W) and "must be ONE of ['1', '0.511', '1.018']" in r for r in reasons)
