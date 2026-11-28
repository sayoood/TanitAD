"""Tests of the card ADD152 reads: hc_common controls and judgement on synthetic data; the read modules import (their asserted source replacements hold)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hc_common as HC  # noqa: E402

SHA = "ab" * 32


def table(tmp, n_clip=2, T=20):
    ci = np.repeat(np.arange(n_clip), T)
    fr = np.tile(np.arange(T), n_clip)
    est = np.where(fr < 9, np.nan, 0.1 * fr + ci)
    p = tmp / "HEAD_TABLE.npz"
    np.savez(p, clip_index=ci, frame=fr, est=est)
    return p, HC.load_table(str(p))


def write_run(tmp, key, T, shift=0, mid_at_tick=False, pre=False):
    td = tmp / key / "ticks"
    td.mkdir(parents=True)
    rng = np.random.default_rng(0)

    def rows(ci, j, q, t0):
        # as the wrapper logs it: j = the frame of the look (tick frame + q), k = j - t0
        tv = HC.table_at(T, ci, j - (q if mid_at_tick else 0) + shift)
        eo = np.isfinite(tv)
        ce = np.where(eo, tv, np.nan)
        pr = rng.random(len(ci)) < 0.6
        cs = np.where(pr, np.where(eo, ce, 0.0), 0.0).astype(np.float32).astype(np.float64)
        vl = rng.uniform(0, 12, len(ci))
        return dict(ci=ci, j=j, k=j - t0, t0=t0, c_est=ce, est_ok=eo, c_src=cs, v_l=vl, v_L=np.maximum(0, vl + cs), present=pr)
    ci = np.array([0, 0, 1, 1, 1])
    t0 = np.array([3, 0, 9, 2, 5])
    j = t0 + np.array([0, 10, 0, 10, 10])
    np.savez(td / ("rule_log_%s.npz" % key), **rows(ci, j, np.zeros(5, int), t0))
    cim, qm, t0m = np.repeat(ci, 3), np.tile([1, 2, 3], 5), np.repeat(t0, 3)
    jm = np.repeat(j, 3) + qm
    np.savez(td / ("rule_log_mid_%s.npz" % key), q=qm, **rows(cim, jm, qm, t0m))
    if pre:
        np.savez(td / ("rule_log_pre_%s.npz" % key), **rows(ci, j - 2, np.zeros(5, int), t0 - 2))
    (td / ("run_%s.txt" % key)).write_text("ckpt x\nest looks scored full 10 short 2 preroll full 0 short 0 file %s\n" % SHA)


def test_fed_pass(tmp_path):
    _, T = table(tmp_path)
    write_run(tmp_path, "A_s0", T, pre=True)
    r = HC.fed_one(str(tmp_path / "A_s0"), "A_s0", T, SHA)
    assert r["PASS"], r
    assert r["rule_log_mid_X.npz"]["est_ok_rows"] > 0 and r["present_looks_with_est_ok_share"] is not None


def test_fed_mid_read_at_tick_frame_fails(tmp_path):
    _, T = table(tmp_path)
    write_run(tmp_path, "A_s0", T, mid_at_tick=True)
    r = HC.fed_one(str(tmp_path / "A_s0"), "A_s0", T, SHA)
    assert not r["PASS"] and r["rule_log_mid_X.npz"]["c_est_differs"] > 0


def test_fed_one_frame_late_fails(tmp_path):
    _, T = table(tmp_path)
    write_run(tmp_path, "A_s0", T, shift=1)
    assert not HC.fed_one(str(tmp_path / "A_s0"), "A_s0", T, SHA)["PASS"]


def test_fed_wrong_run_line_sha_fails(tmp_path):
    _, T = table(tmp_path)
    write_run(tmp_path, "A_s0", T)
    r = HC.fed_one(str(tmp_path / "A_s0"), "A_s0", T, "cd" * 32)
    assert not r["PASS"] and not r["t4_run_line"]["PASS"]


def test_fed_c_src_not_fed_fails(tmp_path):
    _, T = table(tmp_path)
    write_run(tmp_path, "A_s0", T)
    p = tmp_path / "A_s0" / "ticks" / "rule_log_A_s0.npz"
    z = dict(np.load(p))
    z["c_src"] = np.zeros_like(z["c_src"])
    np.savez(p, **z)
    r = HC.fed_one(str(tmp_path / "A_s0"), "A_s0", T, SHA)
    assert not r["PASS"] and r["rule_log_X.npz"]["c_src_not_fed_value_on_present"] > 0


def test_c_tables(tmp_path):
    td = tmp_path / "table"
    td.mkdir()
    np.savez(td / "HEAD_TABLE.npz", a=np.ones(2))
    np.savez(td / "ZERO_TABLE.npz", a=np.zeros(2))
    hs, zs = HC.sha(str(td / "HEAD_TABLE.npz")), HC.sha(str(td / "ZERO_TABLE.npz"))
    (td / "TABLE.json").write_text(json.dumps({"controls_all_pass": True, "tables": {"head": hs, "zero": zs}}))
    for r in ("H_RQ50KVz_s0", "H_RQ50KVz_P30_s0"):
        (tmp_path / "pull" / r).mkdir(parents=True)
        (tmp_path / "pull" / r / "CONTROL_ZERO.json").write_text(json.dumps({"PASS": True, "logs": {"x": {"PASS": True}}}))
    assert HC.c_tables(str(tmp_path / "pull"), str(td), head_sha=hs, zero_sha=zs)["PASS"]
    assert not HC.c_tables(str(tmp_path / "pull"), str(td), head_sha=zs, zero_sha=zs)["PASS"]
    (tmp_path / "pull" / "H_RQ50KVz_P30_s0" / "CONTROL_ZERO.json").write_text(json.dumps({"PASS": False}))
    assert not HC.c_tables(str(tmp_path / "pull"), str(td), head_sha=hs, zero_sha=zs)["PASS"]


def cell(lo, hi):
    return {"interval": [lo, hi], "per_seed": [0, 0, 0], "pooled": (lo + hi) / 2}


def blk(u, c, fams=None, pu=None):
    f = {k: "NOT-SEPARATED" for k in HC.FAMS4}
    f.update(fams or {})
    return {"headway": {"U": u, "C": c, "F": cell(-1, 1)}, "families": f}


LO, NS, HI = cell(-0.02, -0.005), cell(-0.01, 0.01), cell(0.002, 0.02)


def test_combine_branches():
    r = HC.combine(blk(LO, NS), blk(NS, NS), 1.8)
    assert r["name"] == "HEADCLOSE10K-U-LOWER" and r["consequence"].startswith("ADOPT") and not r["run_ceiling"]
    assert HC.combine(blk(LO, NS), blk(NS, HI), 1.8)["consequence"].startswith("TO-THE-PI")
    assert HC.combine(blk(LO, NS), blk(HI, NS), 1.8)["consequence"].startswith("TO-THE-PI")
    assert HC.combine(blk(LO, NS, {"LATERAL": "WORSE"}), blk(NS, NS), 1.8)["consequence"].startswith("TO-THE-PI")
    assert HC.combine(blk(LO, NS), blk(NS, NS, {"TACTICAL": "WORSE"}), 1.8)["consequence"].startswith("TO-THE-PI")
    r = HC.combine(blk(NS, NS), blk(LO, NS), 1.8)
    assert r["name"] == "HEADCLOSE10K-U-NOT-SEPARATED" and r["name_p30"] == "HEADCLOSE10K-P30-U-LOWER" and r["run_ceiling"]
    r = HC.combine(blk(HI, NS), blk(NS, NS), 1.6)
    assert r["name"].endswith("U-HIGHER") and not r["run_ceiling"] and "BLOCKED" in r["consequence"]


def test_ceiling_verdict():
    assert HC.ceiling_verdict(blk(LO, NS), blk(HI, NS))["lever"].startswith("HEAD-ACCURACY")
    assert HC.ceiling_verdict(blk(NS, NS), blk(NS, NS))["lever"].startswith("PRESENCE-AND-GAP")


def test_read_modules_import():
    import importlib
    g = importlib.import_module("read_headclose10k")
    assert g.RD.PAIRS == {"V1": ("H_RQ50KV1", "H_RQ50KE1")} and g.RD.SPREAD_ARM == "H_RQ50KE1"
    assert "c_fed_card" in g.RD.evaluate.__code__.co_names and "c_r1" not in g.RD.evaluate.__code__.co_names
    p = importlib.import_module("read_headclose10k_p30")
    assert p.M.R3_PAIR == ("H_RQ50KV1_P30", "H_RQ50KE1_P30") and "c_fed_card" in p.E.evaluate.__code__.co_names
    assert "WRAPPER_OF" in p.M.q2_argv.__code__.co_names


def test_fed_look_step_off_tick_fails(tmp_path):
    _, T = table(tmp_path)
    write_run(tmp_path, "A_s0", T)
    p = tmp_path / "A_s0" / "ticks" / "rule_log_mid_A_s0.npz"
    z = dict(np.load(p))
    z["k"] = z["k"] + 1
    np.savez(p, **z)
    r = HC.fed_one(str(tmp_path / "A_s0"), "A_s0", T, SHA)
    assert not r["PASS"] and r["rule_log_mid_X.npz"]["look_step_not_tick_plus_q"] > 0
