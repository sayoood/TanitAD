"""Tests of card ADD145 (RULEE1R10K): the generated queue and the reader's pure parts, synthetic data only."""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import read_rulee1r10k as E  # noqa: E402

RD = E.RD
spec = importlib.util.spec_from_file_location("q_e1r", os.path.join(HERE, "box2_rulee1r_queue.py"))
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def test_queue_rows():
    runs = Q.build_runs("D:/tanitad/rhacc10k")
    assert [r["arm"] for r in runs] == ["H_RQ50KE1_s0", "H_RQ50KE1_s1", "H_RQ50KE1_s2"]
    for r in runs:
        a = r["argv"]
        assert a[1].endswith("/build/v8f_t1_eval_reader_rule_rzgfh.py") and a[a.index("--form") + 1] == "RR" and a[a.index("--rule-every") + 1] == "1"


def test_argv_equals_partner():
    n = 0
    for r in Q.build_runs("D:/tanitad/rhacc10k"):
        p = os.path.join(V8, "rhacc10k", "pull", "H_RQ50K_s%d" % r["m"], "argv.json")
        ref = json.load(open(p, encoding="utf-8"))["argv"]
        got = [x.replace(r["arm"], "H_RQ50K_s%d" % r["m"]) for x in r["argv"]]
        got[got.index("--rule-every") + 1] = "5"
        assert got == ref
        n += 1
    assert n == 3


def cell(lo, hi, per):
    return {"interval": [lo, hi], "per_seed": per, "pooled": (lo + hi) / 2}


def blk(u, c, fams=None):
    f = {k: "NOT-SEPARATED" for k in RD.FAMS}
    f.update(fams or {})
    return {"headway": {"U": u, "C": c, "F": cell(-1, 1, [0, 0, 0])}, "families": f}


def test_judge_consequences():
    out, names = RD.judge({"RE1": blk(cell(-0.02, -0.005, [-1, -1, -1]), cell(-0.01, 0.01, [0, 0, 0]))})
    assert names["RE1"] == "RULEE1R10K-RE1-U-LOWER" and out["RE1"]["consequence"].startswith("every-step is the rule")
    out, _ = RD.judge({"RE1": blk(cell(-0.02, -0.005, [-1, -1, -1]), cell(0.001, 0.01, [1, 1, 1]))})
    assert out["RE1"]["consequence"].startswith("to the PI") and "C higher: True" in out["RE1"]["consequence"]
    out, _ = RD.judge({"RE1": blk(cell(-0.02, -0.005, [-1, -1, -1]), cell(-0.01, 0.01, [0, 0, 0]), {"TACTICAL": "WORSE"})})
    assert out["RE1"]["consequence"].startswith("to the PI")
    out, names = RD.judge({"RE1": blk(cell(-0.02, 0.005, [-1, -1, 1]), cell(-0.01, 0.01, [0, 0, 0]))})
    assert names["RE1"].endswith("U-NOT-SEPARATED") and out["RE1"]["consequence"].startswith("the reader form keeps")


def test_later_looks(tmp_path):
    class A:
        pass
    a = A()
    a.pull = str(tmp_path)
    for m in (0, 1, 2):
        for arm, qs in (("H_RQ50KE1", range(1, 10)), ("H_RQ50K", [5])):
            d = tmp_path / ("%s_s%d" % (arm, m)) / "ticks"
            d.mkdir(parents=True)
            np.savez(d / ("rule_log_mid_%s_s%d.npz" % (arm, m)), q=np.array(list(qs) * 3))
    assert E.c_later(a, (0, 1, 2))["PASS"] is True
    np.savez(tmp_path / "H_RQ50KE1_s1" / "ticks" / "rule_log_mid_H_RQ50KE1_s1.npz", q=np.array([5, 5]))
    assert E.c_later(a, (0, 1, 2))["PASS"] is False


def test_evaluate_controls_wired():
    import inspect
    src = inspect.getsource(RD.evaluate) if False else open(RD.__file__, encoding="utf-8").read()
    assert "c_r1(a.r1)" in src                      # the source file is unedited
    assert RD.evaluate.__code__.co_names.count("c_later") == 1 and "c_r1" not in RD.evaluate.__code__.co_names
