"""Tests for read_pilot_s.py -- the AMENDMENT-1 sec. A3 reading as code.  Every expected outcome is a LITERAL worked out by hand from the amendment's table (the
arithmetic is in the comments), never an expression over ``verdict``; the cases that pin one clause carry a mutant that drops that clause and must disagree.

    python run_isolated.py --stack <stack_min> -m pytest tests/test_read_pilot_s.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import read_pilot_s as R  # noqa: E402

OK = ([0.97, 0.97], [True, True])


def v(pick, shuf, auc=(0.97, 0.97), fin=(True, True)):
    return R.verdict(list(pick), list(shuf), list(auc), list(fin))["outcome"]


def test_the_references_are_the_pilots_literals():
    assert (R.OWN, R.ORACLE, R.E_PICK) == (0.99895, 0.35874, {0: 1.5892, 1: 1.5921})
    assert abs(abs(1.0714 - 1.1627) - R.SPREAD_F) < 1e-9 and abs(abs(1.5892 - 1.5921) - R.SPREAD_E) < 1e-9


def test_outcome_rows_by_hand():
    # nu = 0.02, M_own = 0.0913; g_own = 0.19895 / 0.17895 > M_own; h = 0.30 / 0.33 > nu
    assert v((0.80, 0.82), (1.10, 1.15)) == "O1 BEATS-OWN"
    # g_own < 0; nu = 0.02 = M_E; g_E = 0.2892 / 0.2721 > 0.02; h = 0.25 / 0.28 > 0.02
    assert v((1.30, 1.32), (1.55, 1.60)) == "O2 READS-THE-SCENE"
    # same S, but shuffling the scene barely changes the pick: h = 0.01 / 0.005 <= nu = 0.02  -> the gain over E is not the scene
    assert v((1.30, 1.32), (1.31, 1.325)) == "O3 NO-SCENE-SIGNAL"
    # S ts0 is 0.0042 m better than E, ts1 is 0.0079 m WORSE: not "at both seeds"
    assert v((1.585, 1.60), (2.0, 2.0)) == "O3 NO-SCENE-SIGNAL"


def test_boundaries_are_strict_and_the_margin_is_the_larger_of_the_floors():
    # nu = 0: M_own = 0.0913.  own - 0.9077 = 0.09125 < 0.0913 -> not O1 (falls to O2: g_E = 0.68 > 0.0029, h = 0.29 > 0)
    assert v((0.9077, 0.9077), (1.2, 1.2)) == "O2 READS-THE-SCENE"
    assert v((0.9076, 0.9076), (1.2, 1.2)) == "O1 BEATS-OWN"          # own - 0.9076 = 0.09135 > 0.0913
    # S's own seed spread raises the bar: nu = 0.15 -> M_own = 0.15 > 0.0913; g_own = 0.29895 / 0.14895 -> ts1 is under 0.15 -> not O1;
    # O2 instead (g_E = 0.8892 / 0.7421 > 0.15, h = 0.50 / 0.35 > 0.15)
    assert v((0.70, 0.85), (1.2, 1.2)) == "O2 READS-THE-SCENE"


def test_withheld_on_k7_nonfinite_or_unreadable_control():
    assert R.verdict([0.8, 0.82], [1.1, 1.15], [0.97, 0.94], [True, True])["outcome"] == "WITHHELD"
    assert R.verdict([0.8, 0.82], [1.1, 1.15], [0.97, 0.94], [True, True])["suspect_seeds"] == [1]
    assert v((0.80, 0.82), (1.10, 1.15), fin=(False, True)) == "WITHHELD"
    assert v((0.80, 0.82), (1.10, float("nan"))) == "WITHHELD"
    assert R.verdict([0.8, 0.82], [1.1, 1.15], [None, 0.97], [True, True])["outcome"] == "WITHHELD"


def test_mutants_that_drop_a_clause_disagree_with_the_reading():
    def no_shuffle_clause(pick, shuf):                     # drops C-S1 (h > nu)
        nu = abs(pick[0] - pick[1])
        gE = [R.E_PICK[s] - pick[s] for s in (0, 1)]
        return all(g > max(nu, R.SPREAD_E) for g in gE)
    assert no_shuffle_clause((1.30, 1.32), (1.31, 1.325)) is True and v((1.30, 1.32), (1.31, 1.325)) != "O2 READS-THE-SCENE"

    def floor_only(pick):                                  # drops the max(nu, ...) of M_own
        return all(R.OWN - p > R.SPREAD_F for p in pick)
    assert floor_only((0.70, 0.85)) is True and v((0.70, 0.85), (1.2, 1.2)) != "O1 BEATS-OWN"


def _log(seed, losses, picks, shufs, own=0.99895, oracle=0.35874, sha="5af23a842e87", comf=0.97, mode="S"):
    eps = []
    for i, (l, p, s) in enumerate(zip(losses, picks, shufs)):
        eps.append({"epoch": i, "train_loss": {"total": 5.0}, "val": {"n": 980, "loss": l, "pick_cost_m": p, "own_pick_cost_m": own,
                                                                      "oracle_cost_m": oracle, "head_auc": {"comf": comf, "nc": 0.7}},
                    "val_scene_shuffled": {"pick_cost_m": s}})
    return {"args": {"mode": mode, "seed": seed}, "n_val": 980, "val_rows_sha12": sha, "epochs": eps}


def test_the_reader_takes_the_min_val_loss_epoch_not_the_min_pick_epoch_and_refuses_another_slice():
    with tempfile.TemporaryDirectory() as td:
        def w(name, log):
            p = os.path.join(td, name)
            json.dump(log, open(p, "w"))
            return p
        # min LOSS is epoch 1 (pick 1.30); the min PICK is epoch 2 (pick 0.50) and must not be read
        a = w("a.json", _log(0, [5.0, 4.9, 5.1], [1.4, 1.30, 0.50], [1.6, 1.55, 0.9]))
        b = w("b.json", _log(1, [5.0, 4.8, 5.2], [1.4, 1.32, 0.60], [1.6, 1.60, 1.0]))
        out = R.read(a, b)
        assert [r["epoch"] for r in out["per_seed"]] == [1, 1] and [r["pick"] for r in out["per_seed"]] == [1.30, 1.32]
        assert out["verdict"]["outcome"] == "O2 READS-THE-SCENE"            # the same numbers as the O2 case above
        for bad in (_log(0, [5.0], [1.3], [1.5], own=1.0), _log(0, [5.0], [1.3], [1.5], sha="000000000000"), _log(0, [5.0], [1.3], [1.5], mode="E")):
            with pytest.raises(SystemExit):
                R.read(w("bad.json", bad), b)
        nos = _log(0, [5.0], [1.3], [1.5])
        del nos["epochs"][0]["val_scene_shuffled"]
        with pytest.raises(SystemExit, match="C-S1"):
            R.read(w("nos.json", nos), b)
        with pytest.raises(SystemExit, match="expected S seed 1"):
            R.read(a, w("seed0again.json", _log(0, [5.0], [1.3], [1.5])))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
