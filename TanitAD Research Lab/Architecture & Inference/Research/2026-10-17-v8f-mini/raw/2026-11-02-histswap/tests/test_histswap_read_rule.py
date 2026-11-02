"""The pure reading rule of ``code/histswap_t1_read.py`` (H-HISTSWAP-1), on synthetic cells.  Run:  python -I -m pytest -q tests/test_histswap_read_rule.py"""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
spec = importlib.util.spec_from_file_location("histswap_t1_read", f"{os.path.dirname(HERE)}/code/histswap_t1_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def row(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}


def cell(xy_delta, yp_delta=0.06, sep=True):
    xy = row(xy_delta, xy_delta - 0.01, xy_delta + 0.01) if sep else row(xy_delta, xy_delta - 0.05, xy_delta + 0.05)
    return R.m1_cell(xy, row(yp_delta, yp_delta - 0.01, yp_delta + 0.01))


def four(*cs):
    return dict(zip(("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1"), cs))


def test_supported_needs_all_four_separated_better_and_mean_share_ge_half():
    assert R.decide(four(*[cell(-0.04)] * 4))["outcome"] == "SUPPORTED"                       # share 0.667 each
    assert R.decide(four(*[cell(-0.03)] * 4))["outcome"] == "SUPPORTED"                       # exactly 0.5 each: >= is inclusive
    assert R.decide(four(*[cell(-0.02)] * 4))["outcome"] == "MIXED"                           # 4 / 4 separated, share 0.333 < 0.5
    assert R.decide(four(cell(-0.06), cell(-0.06), cell(-0.06), cell(-0.0301)))["outcome"] == "SUPPORTED"      # the mean, not each cell


def test_three_of_four_is_mixed_and_fewer_is_not_supported():
    assert R.decide(four(cell(-0.05), cell(-0.05), cell(-0.05), cell(-0.05, sep=False)))["outcome"] == "MIXED"
    assert R.decide(four(cell(-0.05), cell(-0.05), cell(-0.05, sep=False), cell(-0.05, sep=False)))["outcome"] == "NOT SUPPORTED"
    assert R.decide(four(*[cell(-0.05, sep=False)] * 4))["outcome"] == "NOT SUPPORTED"


def test_wrong_side_and_undefined_excess():
    c = cell(+0.05)                                                                            # separated but X is WORSE than Y: not 'better'
    assert not c["better"]
    assert R.decide(four(c, c, c, c))["outcome"] == "NOT SUPPORTED"
    u = cell(-0.05, yp_delta=-0.01)                                                            # Y has no excess over P10 here: share undefined, counts 0
    assert u["share_closed"] is None
    d = R.decide(four(cell(-0.06), cell(-0.06), cell(-0.06), u))
    assert d["outcome"] == "SUPPORTED" and abs(d["mean_share_closed"] - 0.75) < 1e-9             # 4 / 4 separated-better, (1 + 1 + 1 + 0) / 4 >= 0.5
    d2 = R.decide(four(cell(-0.03), cell(-0.03), u, u))
    assert d2["outcome"] == "MIXED" and abs(d2["mean_share_closed"] - 0.25) < 1e-9             # 4 / 4 separated-better, (0.5 + 0.5 + 0 + 0) / 4 < 0.5


def test_share_arithmetic_and_pooled():
    d = R.decide(four(cell(-0.03, 0.06), cell(-0.06, 0.06), cell(-0.045, 0.06), cell(-0.045, 0.06)))
    assert abs(d["mean_share_closed"] - 0.75) < 1e-9 and abs(d["pooled_share_closed"] - 0.75) < 1e-9 and d["n_cells_separated_better"] == 4


def test_refuses_anything_but_four_cells():
    with pytest.raises(ValueError):
        R.decide(dict(list(four(*[cell(-0.04)] * 4).items())[:3]))


def test_mutation_the_bar_turns_the_boundary_case_red(monkeypatch):
    monkeypatch.setattr(R, "SHARE_BAR", 0.51)
    with pytest.raises(AssertionError):
        assert R.decide(four(*[cell(-0.03)] * 4))["outcome"] == "SUPPORTED"
