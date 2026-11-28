"""Tests of card ADD147 (RULEE1A10K): the generated queue and the reader's pure parts, synthetic data only."""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import read_rulee1a10k as E  # noqa: E402

spec = importlib.util.spec_from_file_location("q_e1a", os.path.join(HERE, "box2_rulee1a_queue.py"))
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def test_queue_argv_equals_partner():
    runs = Q.build_runs("D:/tanitad/rhacc10k")
    assert [r["arm"] for r in runs] == ["H_RQ50KE1_P30_s0", "H_RQ50KE1_P30_s1", "H_RQ50KE1_P30_s2"]
    for r in runs:
        ref = json.load(open(os.path.join(V8, "rhacc10k", "pull", "H_RQ50K_P30_s%d" % r["m"], "argv.json"), encoding="utf-8"))["argv"]
        got = [x.replace(r["arm"], "H_RQ50K_P30_s%d" % r["m"]) for x in r["argv"]]
        assert got[got.index("--rule-every") + 1] == "1"
        got[got.index("--rule-every") + 1] = "5"
        assert got == ref


def test_module_patched():
    assert E.M.P_ARMS == (E.NEW, E.PART) and E.M.R3_PAIR == (E.NEW, E.PART) and E.M.SPREAD_ARM_R3 == E.PART
    assert E.M.register_arms is E.register_arms

    class R:
        ALL_RUNS = {}
    E.register_arms(R)
    assert R.ALL_RUNS["H_RQ50KE1_P30_s1"][6] == 1 and R.ALL_RUNS["H_RQ50K_P30_s2"][6] == 5 and "H_RLcA_P0_s0" in R.ALL_RUNS


def cell(lo, hi):
    return {"interval": [lo, hi], "per_seed": [0, 0, 0], "pooled": (lo + hi) / 2}


def blk(u, c, fams=None):
    f = {k: "NOT-SEPARATED" for k in E.FAMS}
    f.update(fams or {})
    return {"headway": {"U": u, "C": c, "F": cell(-1, 1)}, "families": f}


def test_judge():
    chk, name = E.judge(blk(cell(-0.02, -0.01), cell(-0.01, 0.01)))
    assert name == "RULEE1A10K-U-LOWER" and chk["consequence"].startswith("every-step stays")
    chk, name = E.judge(blk(cell(-0.01, 0.01), cell(-0.01, 0.01)))
    assert name == "RULEE1A10K-U-NOT-SEPARATED" and chk["consequence"].startswith("every-step stays")
    chk, _ = E.judge(blk(cell(-0.02, -0.01), cell(0.001, 0.01)))
    assert chk["consequence"].startswith("to the PI")
    chk, _ = E.judge(blk(cell(-0.02, -0.01), cell(-0.01, 0.01), {"LATERAL": "WORSE"}))
    assert chk["consequence"].startswith("to the PI")


def test_later_looks(tmp_path):
    class A:
        pass
    a = A()
    a.pull = str(tmp_path)
    for m in (0, 1, 2):
        for arm, qs in ((E.NEW, range(1, 10)), (E.PART, [5])):
            d = tmp_path / ("%s_s%d" % (arm, m)) / "ticks"
            d.mkdir(parents=True)
            np.savez(d / ("rule_log_mid_%s_s%d.npz" % (arm, m)), q=np.array(list(qs)))
    assert E.c_later(a, (0, 1, 2))["PASS"] is True
