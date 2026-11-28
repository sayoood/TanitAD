"""test_navrb.py -- a mutation per rule goes RED; a future-read mutant goes RED.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_navrb.py   (CPU)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb as R  # noqa: E402


def arrays(n=12):
    ex = np.full(n, -1, np.int32)
    flag = np.zeros(n, np.uint8)
    return ex, flag


def mut(**over):
    r = dict(R.RULES)
    r.update(over)
    return r


def test_exit_convention_and_mutants():
    exp = {1: 2, 2: 0, 3: 1, 4: 0, 7: 0, -1: 0}
    assert all(R.exit_side(k) == v for k, v in exp.items())
    swapped = lambda n: {1: 1, 3: 2}.get(int(n), 0) if n >= 0 else 0
    assert any(swapped(k) != v for k, v in exp.items())                       # a left/right swap would change an expected value
    ex, flag = arrays()
    ex[3] = 1
    assert R.predict(ex, flag, 5, 0) == 2
    assert R.predict(ex, flag, 5, 0, mut(exit_side=swapped)) != 2             # mutant RED


def test_latest_causal_scan():
    ex, flag = arrays()
    ex[2], ex[6] = 1, 3
    assert R.latest_exit_causal(ex, flag, 5, 0) == 1
    assert R.latest_exit_causal(ex, flag, 8, 0) == 3
    assert R.latest_exit_causal(ex, flag, 1, 0) == -1
    flag[2] = 1                                                               # a flagged row is not a router row
    assert R.latest_exit_causal(ex, flag, 5, 0) == -1
    assert R.latest_exit_causal(ex, flag, 5, 3) == -1                         # never reads rows before the clip's offset


def test_current_row_only_differs():
    ex, flag = arrays()
    ex[2] = 1
    assert R.latest_exit_current(ex, flag, 5, 0) == -1 and R.latest_exit_causal(ex, flag, 5, 0) == 1


def test_state_rule_needs_flag_zero():
    isrb = np.zeros(6, bool)
    flag = np.zeros(6, np.uint8)
    isrb[2] = True
    assert R.is_state(isrb, flag, 2) and not R.is_state(isrb, flag, 3)
    flag[2] = 1
    assert not R.is_state(isrb, flag, 2)
    assert R.RULES["state"](isrb, np.zeros(6, np.uint8), 3) is False


def future_probe(latest, rows=range(0, 10)):
    """rows after NOW replaced by garbage must not change the prediction; returns the number of windows that differ"""
    rng = np.random.default_rng(0)
    bad = 0
    for r in rows:
        ex, flag = arrays(30)
        ex[r + 3 if r + 3 < 30 else 29] = 1
        ex2 = ex.copy()
        ex2[r + 1:] = rng.integers(0, 9, len(ex2[r + 1:]))
        rules = mut(latest=latest)
        bad += int(R.predict(ex, flag, r, 0, rules) != R.predict(ex2, flag, r, 0, rules))
    return bad


def test_causal_rule_has_no_future_read():
    assert future_probe(R.latest_exit_causal) == 0


def test_future_read_mutant_goes_red():
    ahead = lambda ex, flag, r, o: R.latest_exit_ahead(ex, flag, r, o)
    assert future_probe(ahead) > 0


def test_grammar_roundabout_rows_have_no_side():
    texts = ["At roundabout, take exit 2 in 60m", "Enter roundabout", "Exit roundabout in 20m", "Exit rotary", "Continue around the roundabout and take the next exit"]
    P = [R.NT.parse_full(t) for t in texts]
    g, n_unp, _ = R.grammar_fields(texts, P)
    assert set(g) == {"RB_TAKE_EXIT", "RB_ENTER", "RB_EXIT", "ROTARY_EXIT", "RB_NEXT_EXIT"}
    assert all(v["side_field_nonempty"] == 0 and v["text_has_left_right_word"] == 0 for v in g.values())
    assert g["RB_TAKE_EXIT"]["with_exit_number"] == 1 and g["RB_TAKE_EXIT"]["with_distance"] == 1
    P2 = [R.NT.parse_full("Make a left at the roundabout")]
    assert P2[0].cls == "UNPARSED"                                            # a text outside the grammar is reported, not guessed
