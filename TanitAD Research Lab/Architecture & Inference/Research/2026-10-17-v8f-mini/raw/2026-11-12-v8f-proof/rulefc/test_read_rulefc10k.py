"""Tests of read_rulefc10k.py on synthetic data (pure parts: the name rule, the consequence, the partner sha check, the argv check)."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_rulefc10k as RD  # noqa: E402


def cell(lo, hi, per):
    return {"interval": [lo, hi], "per_seed": per, "pooled": (lo + hi) / 2}


def block(u, fams):
    f = {k: "NOT-SEPARATED" for k in RD.FAMS}
    f.update(fams)
    return {"headway": {"U": u, "C": cell(-1, 1, [0, 0, 0]), "F": cell(-1, 1, [0, 0, 0])}, "families": f}


def test_u_card_interval_only():
    assert RD.u_card(cell(-0.03, -0.001, [-0.02, 0.001, -0.01])) == "LOWER"       # a seed above zero does not change the card's rule
    assert RD.u_card(cell(0.001, 0.03, [0.01, 0.02, 0.0])) == "HIGHER"
    assert RD.u_card(cell(-0.01, 0.0, [-0.01, -0.01, -0.01])) == "NOT-SEPARATED"


def test_judge_names_and_consequence():
    blocks = {"E1": block(cell(-0.02, 0.01, [-0.01, 0, 0.01]), {}),
              "FC": block(cell(-0.05, -0.01, [-0.03, -0.02, -0.04]), {}),
              "E1FC": block(cell(-0.06, -0.02, [-0.03, -0.05, -0.04]), {"LONGITUDINAL": "WORSE"})}
    out, names = RD.judge(blocks)
    assert names == {"E1": "RULEFC10K-E1-U-NOT-SEPARATED", "FC": "RULEFC10K-FC-U-LOWER", "E1FC": "RULEFC10K-E1FC-U-LOWER"}
    assert out["FC"]["consequence"].startswith("the rule of the candidate's label form")
    assert out["E1FC"]["consequence"].startswith("to the PI") and "LONGITUDINAL" in out["E1FC"]["consequence"]
    assert out["E1"]["U_per_seed_signs"] == ["-", "0", "+"]


def test_partner_sha(tmp_path):
    pull = tmp_path / "pull"
    lines = []
    for m in (0, 1, 2):
        key = "H_RLcK_s%d" % m
        d = pull / key / ("dump_%s_m10_causal_heads" % key)
        d.mkdir(parents=True)
        for e in range(3):
            p = d / ("ep%04d.npz" % e)
            np.savez(p, cl=np.zeros((2, 30, 2)) + e + m)
            lines.append("%s %s/%s/%s" % (hashlib.sha256(p.read_bytes()).hexdigest(), key, d.name, p.name))
    sha = tmp_path / "sha.txt"
    sha.write_text("\n".join(lines) + "\n", encoding="utf-8")

    class A:
        pass
    a = A()
    a.pull, a.partners_sha, a.expect_files = str(pull), str(sha), 3
    assert RD.c_partner(a, (0, 1, 2))["PASS"] is True
    np.savez(pull / "H_RLcK_s1" / "dump_H_RLcK_s1_m10_causal_heads" / "ep0001.npz", cl=np.ones((2, 30, 2)) * 9)
    r = RD.c_partner(a, (0, 1, 2))
    assert r["PASS"] is False and r["per_run"]["H_RLcK_s1"]["differing"] == 1
    a.partners_sha = str(tmp_path / "missing.txt")
    assert RD.c_partner(a, (0, 1, 2))["PASS"] is False


def test_r1_reads_pass_flag(tmp_path):
    p = tmp_path / "r1.json"
    p.write_text(json.dumps({"pass": True, "checks": {"dumps": {"pass": True}}}), encoding="utf-8")
    assert RD.c_r1(str(p))["PASS"] is True
    p.write_text(json.dumps({"pass": False, "checks": {"dumps": {"pass": False}}}), encoding="utf-8")
    assert RD.c_r1(str(p))["PASS"] is False
    assert RD.c_r1(str(tmp_path / "none.json"))["PASS"] is False


def test_queue_rows_cover_new_arms():
    q = RD.B.load_queue(RD.DEFAULT_QUEUE)
    arms = {r["arm"] for r in q.build_runs()}
    for m in (0, 1, 2):
        for arm in RD.NEW_ARMS:
            assert "%s_s%d" % (arm, m) in arms
