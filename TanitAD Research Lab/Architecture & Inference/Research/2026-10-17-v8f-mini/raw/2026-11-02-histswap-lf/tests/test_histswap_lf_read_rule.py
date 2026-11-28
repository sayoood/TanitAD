"""The pure reading rule of ``code/histswap_lf_read.py`` (H-HISTSWAP-LF-1), on synthetic cells: ``decide_m1`` (HISTSWAP's M1 reading per arm), ``interpret`` (the card's committed table), ``exec_chain_dev`` (KC3),
and a mutation per rule.  Run:  python -I -m pytest -q tests/test_histswap_lf_read_rule.py"""
import importlib.util
import math
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), "/")
CODE = f"{os.path.dirname(HERE)}/code/histswap_lf_read.py"


def load(path, name="histswap_lf_read"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load(CODE)


def row(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}


def cell(xy_delta, yp_delta=0.06, sep=True):
    xy = row(xy_delta, xy_delta - 0.01, xy_delta + 0.01) if sep else row(xy_delta, xy_delta - 0.05, xy_delta + 0.05)
    return R.m1_cell(xy, row(yp_delta, yp_delta - 0.01, yp_delta + 0.01))


def four(*cs):
    return dict(zip(("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1"), cs))


def test_closes_needs_all_four_separated_better_and_mean_share_ge_half():
    assert R.decide_m1(four(*[cell(-0.04)] * 4))["reading"] == "CLOSES"                          # share 0.667 each
    assert R.decide_m1(four(*[cell(-0.03)] * 4))["reading"] == "CLOSES"                          # exactly 0.5 each: >= is inclusive
    assert R.decide_m1(four(*[cell(-0.02)] * 4))["reading"] == "MIXED"                           # 4 / 4 separated, share 0.333 < 0.5
    assert R.decide_m1(four(cell(-0.06), cell(-0.06), cell(-0.06), cell(-0.0301)))["reading"] == "CLOSES"   # the mean, not each cell


def test_three_of_four_is_mixed_and_fewer_is_not_closes():
    assert R.decide_m1(four(cell(-0.05), cell(-0.05), cell(-0.05), cell(-0.05, sep=False)))["reading"] == "MIXED"
    assert R.decide_m1(four(cell(-0.05), cell(-0.05), cell(-0.05, sep=False), cell(-0.05, sep=False)))["reading"] == "NOT CLOSES"
    assert R.decide_m1(four(*[cell(-0.05, sep=False)] * 4))["reading"] == "NOT CLOSES"


def test_wrong_side_negative_share_and_undefined_excess():
    c = cell(+0.05)                                                                              # separated but X is WORSE than Y: not 'better', share negative
    assert not c["better"] and c["share_closed"] < 0
    d = R.decide_m1(four(c, c, c, c))
    assert d["reading"] == "NOT CLOSES" and d["mean_share_closed"] < 0 and not d["closes"]
    u = cell(-0.05, yp_delta=-0.01)                                                              # Y has no excess over P10 here: share undefined, counts 0
    assert u["share_closed"] is None
    d = R.decide_m1(four(cell(-0.06), cell(-0.06), cell(-0.06), u))
    assert d["reading"] == "CLOSES" and abs(d["mean_share_closed"] - 0.75) < 1e-9


def test_refuses_anything_but_four_cells():
    with pytest.raises(ValueError):
        R.decide_m1(dict(list(four(*[cell(-0.04)] * 4).items())[:3]))


def test_interpretation_table_is_the_committed_one():
    closes, mixed, no = (R.decide_m1(four(*[cell(x, sep=s)] * 4)) for x, s in ((-0.05, True), (-0.02, True), (-0.05, False)))
    assert (closes["reading"], mixed["reading"], no["reading"]) == ("CLOSES", "MIXED", "NOT CLOSES")
    assert R.interpret(closes, closes)["reading"] == "A_INERTIA" and R.interpret(closes, no)["reading"] == "A_INERTIA"
    assert R.interpret(closes, no)["hlvl_discordant"] and not R.interpret(closes, closes)["hlvl_discordant"]
    assert R.interpret(no, closes)["reading"] == "B_LEVEL" and R.interpret(mixed, closes)["reading"] == "B_LEVEL"
    assert R.interpret(no, no)["reading"] == "C_INTERACTION" and R.interpret(mixed, mixed)["reading"] == "C_INTERACTION"
    assert R.interpret(mixed, no)["hcv_mixed"] and not R.interpret(no, no)["hcv_mixed"]


def test_exec_chain_dev_is_zero_on_a_consistent_chain_and_large_on_a_broken_one():
    m = 10
    plan = np.stack([np.arange(1, 31) * 1.6, np.arange(1, 31) * 0.05], 1)
    pose0 = np.array([5.0, -2.0, 0.4, 8.0])
    c, s = math.cos(0.4), math.sin(0.4)
    k = plan[m - 1]
    p1 = pose0[:2] + np.array([k[0] * c - k[1] * s, k[0] * s + k[1] * c])
    t = {"tick": np.array([0, 1]), "win": np.array([7, 7]), "pose": np.array([pose0, [p1[0], p1[1], 0.5, 9.0]]), "plan": np.stack([plan, plan])}
    assert R.exec_chain_dev(t)[0] < 1e-12
    t_bad = dict(t, pose=np.array([pose0, [p1[0] + 1.0, p1[1], 0.5, 9.0]]))
    assert abs(R.exec_chain_dev(t_bad)[0] - 1.0) < 1e-9
    t_logged = dict(t, pose=np.array([pose0, [p1[0], p1[1] + 0.3, 0.5, 9.0]]))                  # committing from a different pose breaks it
    assert R.exec_chain_dev(t_logged)[0] > 0.2


def test_mutations_turn_the_rules_red():
    src = open(CODE, encoding="utf-8").read()
    for old, new, probe in (("if n_better == 4 and mean_share >= SHARE_BAR:", "if n_better >= 3 and mean_share >= SHARE_BAR:", lambda M: M.decide_m1(four(cell(-0.05), cell(-0.05), cell(-0.05), cell(-0.05, sep=False)))["reading"] == "MIXED"),
                            ("elif n_better < 3:", "elif n_better < 2:", lambda M: M.decide_m1(four(cell(-0.05), cell(-0.05), cell(-0.05, sep=False), cell(-0.05, sep=False)))["reading"] == "NOT CLOSES"),
                            ("    if hcv[\"closes\"]:\n        r = \"A_INERTIA\"", "    if hcv[\"closes\"] and hlvl[\"closes\"]:\n        r = \"A_INERTIA\"", lambda M: M.interpret(M.decide_m1(four(*[cell(-0.05)] * 4)), M.decide_m1(four(*[cell(-0.05, sep=False)] * 4)))["reading"] == "A_INERTIA")):
        assert src.count(old) == 1, old
        spec = importlib.util.spec_from_loader("histswap_lf_read_mut", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = CODE
        exec(compile(src.replace(old, new), "histswap_lf_read_mut", "exec"), mod.__dict__)
        assert probe(R), "the probe must hold on the real rule"
        assert not probe(mod), f"MUTATION did not turn the rule red: {old}"
