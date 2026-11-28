"""strategic_paired.py -- the paired STRATEGIC rows (S-GOAL) for the refcv9 E3 read, on SYNTHETIC refcv3_arm-form dumps.  CPU only, no scipy.

  * the per-window producer is HELD EQUAL to ``nav_compliance.from_refcv3_dump`` (rate, n, tau per conditioning);
  * schema; pairing by ``(clip_id, ws)`` (a permuted table row order changes nothing); a planted per-window difference is recovered with the RIGHT SIGN;
  * the echo control: present, or the whole S-GOAL is refused (deliberate regression: nav_zero dropped -> NOT-COMPUTABLE, no row);
  * S-DEC reads NOT-COMPUTABLE with the flag + argv file named and n = 0;
  * refusals: window-set mismatch, tau mismatch, informative-mask mismatch, a dump without the nav-zero forward;
  * ``r9_e3_verdict.strategic_report`` reads the output as REPORT-ONLY (S-GOAL rows readable, S-DEC carried under ``not_computable`` with its reason, never a pass).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)                      # <repo>/taniteval
for _p in (_PKG_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.append(_p)
_STACK = os.path.join(os.path.dirname(_PKG_PARENT), "stack")
if os.path.isdir(_STACK) and _STACK not in sys.path:
    sys.path.append(_STACK)

from taniteval import nav_compliance as nc                # noqa: E402
import test_nav_compliance_selbank as fx                  # noqa: E402  (the synthetic dump builder)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SP = _load("strategic_paired_under_test", os.path.join(_PKG_PARENT, "tools", "strategic_paired.py"))
VERDICT = os.path.join(_STACK, "scripts", "r9_e3_verdict.py")


def _degrade(dd, eps, cond="nav_true"):
    """Rewrite episodes' plan under ``cond`` to a straight line (a non-follower on those windows)."""
    for e in eps:
        p = os.path.join(dd, "decisions", f"ep{e:03d}.npz")
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d[f"plan_full_{cond}"] = np.stack([fx._path(0.0)] * d["ws"].shape[0])
        np.savez(p, **d)


@pytest.fixture()
def arms(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    dd_a, lp, _ = fx.make_dump(a_dir)
    dd_b, lp_b, _ = fx.make_dump(b_dir)
    _degrade(dd_b, (0, 2, 4, 6))                   # B stops following the command on 4 of the 6 turn clips; A is a perfect follower
    ta = SP.table_from_dump(dd_a, lp, "A")
    tb = SP.table_from_dump(dd_b, lp_b, "B", tau=ta["tau"])
    return {"dd_a": dd_a, "lp": lp, "ta": ta, "tb": tb}


def _rows(out):
    return {k: v for k, v in out["STRATEGIC"].items() if isinstance(v, dict) and "better" in v}


# ---------------------------------------------------------------- producer
def test_table_schema_and_equals_from_refcv3_dump(arms):
    ta = arms["ta"]
    assert ta["schema"] == SP.SCHEMA and ta["tier"] == "T1*" and ta["readout"] == "plan"
    n = ta["n_windows"]
    assert n == len(ta["clip_id"]) == len(ta["ws"]) == len(ta["informative"]) == 144
    assert set(ta["conds"]) == {"nav_true", "nav_shuffled", "nav_zero"}
    assert all(len(ta["comply"][c]) == n for c in ta["conds"])
    rep = nc.from_refcv3_dump(arms["dd_a"], labels_path=arms["lp"], n_boot=50, seed=0)
    blk = rep["readouts"]["plan"]
    inf = np.asarray(ta["informative"], bool)
    assert int(inf.sum()) == ta["n_informative"] == blk["n_informative_windows"] > 0
    assert abs(ta["tau"] - blk["tau"]) < 1e-9
    for c in ta["conds"]:
        mine = float(np.asarray(ta["comply"][c])[inf].mean())
        ref = blk["conditionings"][c]["compliance_with_TRUE_command"]["mean"]
        assert abs(mine - ref) < 1e-4, (c, mine, ref)   # the reference rounds to 4 dp
    # the synthetic follower: perfect under true, never under zero
    assert float(np.asarray(ta["comply"]["nav_true"])[inf].mean()) == 1.0
    assert float(np.asarray(ta["comply"]["nav_zero"])[inf].mean()) == 0.0


def test_dump_without_the_nav_zero_forward_is_refused(tmp_path):
    dd, lp, _ = fx.make_dump(tmp_path)
    for e in range(fx.N_EP):
        p = os.path.join(dd, "decisions", f"ep{e:03d}.npz")
        with np.load(p) as z:
            d = {k: z[k] for k in z.files if "nav_zero" not in k}
        np.savez(p, **d)
    with pytest.raises(SP.Refused, match="nav_zero|echo control|os_navzero"):
        SP.table_from_dump(dd, lp, "X")


# ---------------------------------------------------------------- pairing
def test_planted_difference_recovered_with_the_right_sign(arms):
    out = SP.pair_tables(arms["ta"], arms["tb"], a_name="E3", b_name="refcv8", n_boot=500)
    rows = _rows(out)
    assert out["STRATEGIC"]["status"] == "REPORT-ONLY"
    assert set(rows) == {"S-GOAL.plan.nav_true", "S-GOAL.plan.ECHO.nav_zero", "S-GOAL.plan.ECHO.nav_shuffled",
                         "S-GOAL.plan.EFFECT.true_minus_zero", "S-GOAL.plan.EFFECT.true_minus_shuffled"}
    r = rows["S-GOAL.plan.nav_true"]
    assert r["delta"] > 0 and r["separated"] is True and r["lo"] > 0           # A (perfect follower) - B (4 of 6 clips not following) = 1 - 2/6
    assert abs(r["delta"] - 4 / 6) < 1e-3 and r["n_episodes"] == 6 and r["tier"] == "T1*"
    assert r["estimator"] == "paired_episode_cluster_bootstrap"
    assert rows["S-GOAL.plan.EFFECT.true_minus_zero"]["delta"] > 0
    z = rows["S-GOAL.plan.ECHO.nav_zero"]                                      # both arms are at 0 under nav-zero: the echo control reads exactly 0
    assert z["delta"] == 0.0 and z["separated"] is False
    # reverse the arms: the sign flips
    rev = _rows(SP.pair_tables(arms["tb"], arms["ta"], n_boot=500))
    assert rev["S-GOAL.plan.nav_true"]["delta"] < 0 and rev["S-GOAL.plan.nav_true"]["separated"]


def test_pairing_is_by_clip_and_ws_not_by_row_position(arms):
    tb = copy.deepcopy(arms["tb"])
    perm = np.random.default_rng(1).permutation(len(tb["clip_id"]))
    for k in ("clip_id", "ws", "informative"):
        tb[k] = [tb[k][i] for i in perm]
    tb["comply"] = {c: [v[i] for i in perm] for c, v in tb["comply"].items()}
    a = _rows(SP.pair_tables(arms["ta"], arms["tb"], n_boot=200))
    b = _rows(SP.pair_tables(arms["ta"], tb, n_boot=200))
    assert a == b


def test_window_set_mismatch_is_refused_not_intersected(arms):
    tb = copy.deepcopy(arms["tb"])
    for k in ("clip_id", "ws", "informative"):
        tb[k] = tb[k][:-1]
    tb["comply"] = {c: v[:-1] for c, v in tb["comply"].items()}
    with pytest.raises(SP.Refused, match="window sets differ"):
        SP.pair_tables(arms["ta"], tb)


def test_tau_mismatch_and_informative_mismatch_are_refused(arms):
    tb = copy.deepcopy(arms["tb"])
    tb["tau"] = tb["tau"] + 0.01
    with pytest.raises(SP.Refused, match="tau differs"):
        SP.pair_tables(arms["ta"], tb)
    tb = copy.deepcopy(arms["tb"])
    i = int(np.flatnonzero(tb["informative"])[0])
    tb["informative"][i] = False
    with pytest.raises(SP.Refused, match="informative mask differs"):
        SP.pair_tables(arms["ta"], tb)


# ---------------------------------------------------------------- the echo control
def test_deliberate_regression_echo_control_dropped_refuses_every_row(arms):
    for who in ("a", "b"):
        ta, tb = copy.deepcopy(arms["ta"]), copy.deepcopy(arms["tb"])
        (ta if who == "a" else tb)["comply"].pop("nav_zero")
        out = SP.pair_tables(ta, tb)
        st = out["STRATEGIC"]
        assert st["S-GOAL_status"] == "NOT-COMPUTABLE" and "INADMISSIBLE" in st["S-GOAL_reason"]
        assert st["S-GOAL_rows"] == [] and _rows(out) == {}
        assert st["S-GOAL"]["status"] == "NOT-COMPUTABLE" and st["S-GOAL"]["reason"] == st["S-GOAL_reason"] and st["S-GOAL"]["n"] == 0


def test_shuffle_rows_dropped_when_one_arm_lacks_it_but_zero_control_stays(arms):
    tb = copy.deepcopy(arms["tb"])
    tb["comply"].pop("nav_shuffled")
    rows = _rows(SP.pair_tables(arms["ta"], tb, n_boot=100))
    assert "S-GOAL.plan.ECHO.nav_zero" in rows and "S-GOAL.plan.ECHO.nav_shuffled" not in rows


def test_s_dec_not_computable_with_reason_and_n_zero(arms):
    sd = SP.pair_tables(arms["ta"], arms["tb"], n_boot=50)["STRATEGIC"]["S-DEC"]
    assert sd["status"] == "NOT-COMPUTABLE" and sd["n"] == 0 and "better" not in sd
    assert "--no-strategic" in sd["reason"] and "r9e3_K2DC02.argv.json" in sd["reason"]


# ---------------------------------------------------------------- the judge
def test_judge_reads_it_as_report_only(arms):
    V = _load("r9_e3_verdict_under_test", VERDICT)
    out = SP.pair_tables(arms["ta"], arms["tb"], n_boot=200)
    out = json.loads(json.dumps(out))                              # through JSON, as the judge's input arrives
    rep = V.strategic_report(out["STRATEGIC"])
    assert rep["status"] == "REPORT-ONLY"
    goal = [k for k in rep["rows"] if k.startswith("S-GOAL.")]
    assert len(goal) == 5 and all(rep["rows"][k]["ok"] for k in goal)
    assert rep["rows"]["S-GOAL.plan.nav_true"]["separated_better"] is True
    assert "S-DEC" not in rep["rows"] and rep["unreadable_rows"] == []          # a stamp, not a row (MM fix 2026-11-23)
    assert rep["not_computable"]["S-DEC"]["status"] == "NOT-COMPUTABLE" and "--no-strategic" in rep["not_computable"]["S-DEC"]["reason"]  # reason kept
    assert "PASS" not in rep["status"]
    # a refused S-GOAL leaves only the S-DEC stub: still REPORT-ONLY, still not a pass
    ta = copy.deepcopy(arms["ta"])
    ta["comply"].pop("nav_zero")
    rep2 = V.strategic_report(SP.pair_tables(ta, arms["tb"])["STRATEGIC"])
    assert rep2["status"] == "NOT-SUPPLIED" and rep2["rows"] == {}                 # no row left: NOT-SUPPLIED, never a pass
    assert set(rep2["not_computable"]) == {"S-DEC", "S-GOAL"} and "INADMISSIBLE" in rep2["not_computable"]["S-GOAL"]["reason"]   # both reasons kept


def test_cli_round_trip(arms, tmp_path):
    ta_p, tb_p, out_p = tmp_path / "ta.json", tmp_path / "tb.json", tmp_path / "out.json"
    ta_p.write_text(json.dumps(arms["ta"]), encoding="utf-8")
    tb_p.write_text(json.dumps(arms["tb"]), encoding="utf-8")
    assert SP.main(["pair", "--a", str(ta_p), "--b", str(tb_p), "--out", str(out_p)]) == 0
    assert "S-GOAL.plan.nav_true" in json.loads(out_p.read_text(encoding="utf-8"))["STRATEGIC"]
    tb = json.loads(tb_p.read_text(encoding="utf-8"))
    tb["schema"] = "nope"
    tb_p.write_text(json.dumps(tb), encoding="utf-8")
    assert SP.main(["pair", "--a", str(ta_p), "--b", str(tb_p), "--out", str(out_p)]) == 2
