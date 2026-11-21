"""Pins for build_deploy_gains_r8.py (RETIME_DEPLOY, SHIP-R8 AMENDMENT 1).  Expectations are LITERALS (the frozen train file's floats, copied out of it with repr;
AMENDMENT 1's own 3-decimal rendering), never an expression over the code under test.  Each guard has a MUTATION that must go RED.
    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_deploy_gains_r8.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_deploy_gains_r8 as BD  # noqa: E402

# the frozen train file's floats (raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json, sha256 b208c9c1...), copied as literals
S2_INST = [0.5, 1.0, 1.5, 2.0]
S2_TRAIN = [0.9673903790217738, 0.6192916835387219, 0.5511341845460037, 0.530539399347386]
S6_INST = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
S6_TRAIN = [0.6393548580383511, 0.5969720822621201, 0.6546139969520486, 0.6994940271226132, 0.7222699235551424, 0.7453601677670005]
# the DEPLOYED knots, literally
KT = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
KG = [0.9673903790217738, 0.6192916835387219, 0.5511341845460037, 0.530539399347386,
      0.6546139969520486, 0.6994940271226132, 0.7222699235551424, 0.7453601677670005]
S6_DEPLOY = [0.6192916835387219, 0.530539399347386, 0.6546139969520486, 0.6994940271226132, 0.7222699235551424, 0.7453601677670005]
G_075 = 0.7933410312802478          # (0.9673903790217738 + 0.6192916835387219) / 2


def fz():
    return {"surfaces": {"S2": {"instants_s": list(S2_INST), "all": {"RETIME": list(S2_TRAIN)}},
                         "S6": {"instants_s": list(S6_INST), "all": {"RETIME": list(S6_TRAIN)}}}}


def test_knots_are_the_literal_floats():
    kt, kg = BD.knots_from_frozen(fz())
    assert kt == KT and kg == KG


def test_register_three_decimal_values():
    assert [round(g, 3) for g in KG] == [0.967, 0.619, 0.551, 0.531, 0.655, 0.699, 0.722, 0.745]


def test_gain_at_literals():
    assert BD.gain_at(0.75, KT, KG) == pytest.approx(G_075, abs=1e-15)       # mean of the 0.5 s and 1.0 s knots
    assert BD.gain_at(0.2, KT, KG) == KG[0]                                     # held before 0.5 s
    assert BD.gain_at(7.0, KT, KG) == KG[-1]                                    # held after 6.0 s
    assert BD.gain_at(0.0, KT, KG) == KG[0] and BD.gain_at(100.0, KT, KG) == KG[-1]
    for t, g in zip(KT, KG):                                                    # every knot is exact (bitwise)
        assert BD.gain_at(t, KT, KG) == g


def test_gain_at_is_linear_between_knots_at_a_nonmidpoint():
    # 2.5 s is the midpoint of the 2.0 s and 3.0 s knots (the knot spacing changes there)
    assert BD.gain_at(2.5, KT, KG) == pytest.approx((0.530539399347386 + 0.6546139969520486) / 2, abs=1e-15)
    # 1.2 s: 40 % of the way from the 1.0 s knot (0.6192916835387219) to the 1.5 s knot (0.5511341845460037)
    assert BD.gain_at(1.2, KT, KG) == pytest.approx(0.6192916835387219 + 0.4 * (0.5511341845460037 - 0.6192916835387219), abs=1e-15)


def test_built_tables_are_the_literals():
    d = BD.build(fz())
    assert d["surfaces"]["S2"]["RETIME_DEPLOY"] == S2_TRAIN                      # EXACT: every S2 instant is a knot
    assert d["surfaces"]["S2"]["identical_to_train"] is True and d["surfaces"]["S2"]["changed_vs_train"] == []
    assert d["surfaces"]["S6"]["RETIME_DEPLOY"] == S6_DEPLOY                     # [S2@1.0, S2@2.0, S6@3, S6@4, S6@5, S6@6]
    assert d["surfaces"]["S6"]["RETIME_DEPLOY"] == [S2_TRAIN[1], S2_TRAIN[3], S6_TRAIN[2], S6_TRAIN[3], S6_TRAIN[4], S6_TRAIN[5]]
    assert [c["t_s"] for c in d["surfaces"]["S6"]["changed_vs_train"]] == [1.0, 2.0]
    assert d["surfaces"]["S6"]["changed_vs_train"][0] == {"t_s": 1.0, "train": 0.6393548580383511, "deploy": 0.6192916835387219}
    assert d["surfaces"]["S6"]["changed_vs_train"][1] == {"t_s": 2.0, "train": 0.5969720822621201, "deploy": 0.530539399347386}
    assert BD.check_deploy(d) == []


# ------------------------------------------------------------------ mutations: each reintroduces a real defect and must be CAUGHT
def test_mutation_perturbed_s2_knot_breaks_the_identity():
    z = fz()
    z["surfaces"]["S2"]["all"]["RETIME"][1] += 0.01             # the S2 train table the deploy is built FROM is what the knot reads: perturb only the S6 reading instead
    d = BD.build(z)
    # a knot perturbation that does not also move the train table: edit the deployed knot after the build
    d2 = BD.build(fz())
    d2["surfaces"]["S2"]["RETIME_DEPLOY"][1] += 0.01
    d2["surfaces"]["S2"]["identical_to_train"] = (d2["surfaces"]["S2"]["RETIME_DEPLOY"] == d2["surfaces"]["S2"]["RETIME_TRAIN"])
    assert BD.check_deploy(d2) != []
    assert BD.check_deploy(d) != []                              # knot 1.0 s no longer matches the register's 0.619 at 3 dp


def test_mutation_s6_uses_its_own_1s_2s_gains():
    """The bug the amendment removes: knots taken from S6 at 1 s / 2 s (0.639 / 0.597) instead of the S2 values."""
    d = BD.build(fz())
    d["surfaces"]["S6"]["RETIME_DEPLOY"] = list(S6_TRAIN)
    d["surfaces"]["S6"]["changed_vs_train"] = []
    assert any("S6 deploy table" in b or "changed instants" in b for b in BD.check_deploy(d))


def test_mutation_linear_extrapolation_instead_of_hold():
    """A gain_at that extrapolates past 6.0 s fails the literal gain_at(7) == the 6.0 s knot."""
    def bad(t, kt, kg):
        if t > kt[-1]:
            return kg[-1] + (t - kt[-1]) * (kg[-1] - kg[-2]) / (kt[-1] - kt[-2])
        return BD.gain_at(t, kt, kg)
    assert bad(7.0, KT, KG) != KG[-1]
    assert BD.gain_at(7.0, KT, KG) == KG[-1]


def test_mutation_knot_not_an_instant_is_refused():
    z = fz()
    z["surfaces"]["S6"]["instants_s"][2] = 3.5                    # S6 no longer has a 3.0 s instant
    with pytest.raises(SystemExit):
        BD.knots_from_frozen(z)


def test_mutation_wrong_register_value():
    d = BD.build(fz())
    d2 = copy.deepcopy(d)
    d2["knots"]["gain"][4] = 0.65                                 # 0.655 -> 0.650
    assert any("3 dp" in b for b in BD.check_deploy(d2))


# ------------------------------------------------------------------ the real artifact
def test_pinned_source_hash_and_written_artifact():
    assert BD.SOURCE_SHA256 == "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
    if not BD.SOURCE.exists():
        pytest.skip("frozen train gains not present")
    assert hashlib.sha256(BD.SOURCE.read_bytes()).hexdigest() == BD.SOURCE_SHA256
    if not BD.DEPLOY_JSON.exists():
        pytest.skip("deploy gains not built")
    d = json.loads(BD.DEPLOY_JSON.read_text(encoding="utf-8"))
    assert d["source_sha256"] == "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
    assert d["knots"]["t_s"] == KT and d["knots"]["gain"] == KG
    assert d["surfaces"]["S2"]["RETIME_DEPLOY"] == S2_TRAIN and d["surfaces"]["S6"]["RETIME_DEPLOY"] == S6_DEPLOY
    assert hashlib.sha256(BD.DEPLOY_JSON.read_bytes()).hexdigest() == BD.DEPLOY_SHA256_FILE.read_text().strip()
