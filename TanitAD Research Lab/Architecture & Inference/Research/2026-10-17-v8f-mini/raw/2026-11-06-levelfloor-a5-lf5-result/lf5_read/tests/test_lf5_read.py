"""Tests of the LF5 reader (PREREG_LEVELFLOOR_A5 sha256 2aac1dbe...): the label logic (sigma*, the joint reading of card section 5), the 20-rollout table and flags, the noise instruments, the reuse of lfo_read.py's ceiling labels, and that lf5_chain.sh's table is the reader's.
Run:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lf5_read/pt/run <this file>   (LF5_READER = the reader under test; code/run_mutations_lf5_read.py points it at a mutated copy)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
READER = Path(os.environ.get("LF5_READER", str(PKG / "code" / "lf5_read.py")))
spec = importlib.util.spec_from_file_location("lf5_read_under_test", str(READER))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
R.init()


def test_rollouts_are_the_twenty_of_the_card_in_priority_order():
    rows = R.rollouts()
    tags = [r[0] for r in rows]
    assert len(rows) == 20 and len(set(tags)) == 20
    assert tags[:4] == ["RILF100_s0", "RILF100_s1", "RILF100C_s0", "RILF100C_s1"]                                   # the admissible arm first
    assert tags[4:8] == ["RILFO100_s0", "RILFO100_s1", "RILFO100C_s0", "RILFO100C_s1"]
    assert tags[8:12] == ["RILFN100x10_s0", "RILFN100x10_s1", "RILFN100x10C_s0", "RILFN100x10C_s1"] and tags[12:16][0] == "RILFN100x20_s0"
    assert tags[16:] == ["LF100_s0", "LF100_s1", "LF100C_s0", "LF100C_s1"] and all(r[4] == "logged" for r in rows[16:]) and all(r[4] == "reindexed" for r in rows[:16])
    fl = {r[0]: R.expected_flags(r[6], r[7], r[4], r[5], r[3]) for r in rows}
    assert all("--floor-kappa 1.0 " in f for f in fl.values())
    assert "--floor-vhat-noise 0.10 --floor-vhat-noise-seed 20261106" in fl["RILFN100x10C_s1"] and "--floor-vhat-noise 0.20 --floor-vhat-noise-seed 20261106" in fl["RILFN100x20_s0"]
    assert "--floor-vhat-noise" not in fl["RILFO100_s0"] and "--floor-vhat-source logged" in fl["RILFO100_s0"]
    assert "--floor-vhat-source heads" in fl["LF100_s0"] and "--scene logged" in fl["LF100C_s1"] and "--feasible kbox" in fl["LF100C_s1"] and "--feasible" not in fl["RILF100_s0"]


def test_the_chain_table_is_the_readers():
    out = subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-c", f"bash '{(PKG / 'lf5_chain.sh').as_posix()}' --dry-run"], capture_output=True, text=True, stdin=subprocess.DEVNULL, env=dict(os.environ, LF5_PK=PKG.as_posix().replace("E:/", "/e/"))).stdout
    rows = [ln.split("\t") for ln in out.splitlines() if ln.count("\t") == 5]
    want = [[r[0], str(r[3]), r[4], r[5], r[6], "-" if not r[7] else ("%.2f" % r[7])] for r in R.rollouts()]
    assert rows == want


def test_sigma_star_is_the_largest_sigma_where_p_and_g_hold():
    assert R.sigma_star({0.0: True, 0.10: True, 0.20: True})["sigma_star"] == 0.20
    assert R.sigma_star({0.0: True, 0.10: True, 0.20: False})["sigma_star"] == 0.10
    assert R.sigma_star({0.0: True, 0.10: False, 0.20: False})["sigma_star"] == 0.0
    assert R.sigma_star({0.0: False, 0.10: False, 0.20: False})["sigma_star"] is None
    nm = R.sigma_star({0.0: True, 0.10: False, 0.20: True})
    assert nm["sigma_star"] == 0.20 and nm["monotone"] is False


def test_joint_reading_follows_card_section_5():
    j = R.joint_reading
    d = j("CEILING-SHORT", "D-HOLDS", None)
    assert d["counted_dominant"] is True and "three doses" in d["next"]
    c = j("CEILING-CLEARS", "P-FAILS-M1-MOVED", 0.10)
    assert c["case"] == "FORM-WORKS-GIVEN-A-GOOD-LEVEL" and c["estimator_levers_signalled"] is True and not c["counted_dominant"] and "temporal filter" in c["next"]
    z = j("CEILING-CLEARS", "P-FAILS-M1-MOVED", 0.0)
    assert z["case"] == "FORM-NEEDS-AN-EXACT-LEVEL" and z["estimator_levers_signalled"] is False
    assert j("CEILING-CLEARS", "LEAK-BORNE", 0.20)["estimator_levers_signalled"] is True
    h = j("CEILING-HARMS", "P-FAILS-M1-MOVED", 0.20)
    assert h["case"] == "OPERATOR-IS-THE-PROBLEM" and h["estimator_levers_signalled"] is False
    s = j("CEILING-SHORT", "P-FAILS-M1-MOVED", None)
    assert s["case"] == "PROGRESS-FLOOR-FORM-EXHAUSTED" and s["estimator_levers_signalled"] is False and "T2" in s["next"] and "A40" in s["next"]
    assert j("F-UNREACHED", "F-UNREACHED", None)["case"] == "F-UNREACHED"


def test_ceiling_labels_are_lfo_reads_unchanged():
    reach_bad, ok = {"PASS": False}, {"PASS": True}
    assert R.LFO.decide(reach_bad)["label"] == "F-UNREACHED"
    assert R.LFO.decide(ok, {"PASS": False}, {"PASS": True})["label"] == "CEILING-SHORT"
    assert R.LFO.decide(ok, {"PASS": True}, {"PASS": False, "per_seed": {"s0": {"non_crawl_separated_worse": ["LAT"], "ade6_within_guard": True}, "s1": {"non_crawl_separated_worse": [], "ade6_within_guard": True}}})["label"] == "CEILING-HARMS"
    assert R.LFO.decide(ok, {"PASS": True}, {"PASS": True})["label"] == "CEILING-CLEARS"


def test_noise_instrument_reads_the_realised_log_ratio_and_the_heads_overshoot():
    rng = np.random.default_rng(3)
    v = np.full(4000, 6.0)
    eps = rng.standard_normal(4000)
    b = R._err_block(v * np.exp(0.2 * eps), v, np.ones(4000, bool))
    assert abs(b["log_ratio_std"] - 0.2) < 0.01 and abs(b["log_ratio_mean"]) < 0.02 and 0.14 < b["relative_mae"] < 0.18 and abs(b["share_vhat_exceeds_logged"] - 0.5) < 0.03
    assert 0.8 < b["mae_mps"] < 1.2
    h = R._err_block(np.full(10, 8.0), np.full(10, 6.0), np.ones(10, bool))
    assert h["share_vhat_exceeds_logged"] == 1.0 and abs(h["mae_mps"] - 2.0) < 1e-12 and abs(h["relative_mae"] - 1 / 3) < 1e-12
    nan = R._err_block(np.array([np.nan, 7.0]), np.array([6.0, 6.0]), np.ones(2, bool))
    assert nan["n"] == 1
