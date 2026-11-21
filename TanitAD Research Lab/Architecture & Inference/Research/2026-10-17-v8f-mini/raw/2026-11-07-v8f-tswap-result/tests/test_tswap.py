"""tests for the TSWAP controls and the result rule: the outcome flips with ONE M1 interval; the T share has a known answer; G-LIVE fails on the mutant's artifacts; G-ID fails on one flipped bit.
Needs the subset outputs F:/TanitAD-scratch/motion_c/tswap/out/S_* (run_subset.sh)."""
import os
import sys
import json
import shutil
import importlib.util
import numpy as np
import pytest

PK = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/tswap"
OUT = "F:/TanitAD-scratch/motion_c/tswap/out"
sys.dont_write_bytecode = True


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("tswap_result_t", f"{PK}/tswap_result.py")
G = _load("tswap_gates_t", f"{PK}/tswap_gates.py")
GD = _load("gapdecomp_t", "E:/Projects/TanitAD-artifacts/dagland/gapdecomp/gapdecomp.py")
row = lambda d, lo, hi: {"delta": d, "lo": lo, "hi": hi}


def facts():
    f = {"ts_vs_own": row(-0.02, -0.03, -0.01), "ts_vs_p10": row(0.001, -0.01, 0.012), "own_vs_p10": row(0.4, 0.3, 0.5)}
    return {0: dict(f), 1: dict(f)}


def test_outcome_flips_with_one_m1_interval():
    f = facts()
    assert R.outcome(f, True) == "T-CARRIES"
    f[1]["ts_vs_own"] = row(-0.02, -0.03, 0.001)                     # one interval now contains 0 at seed 1
    assert R.outcome(f, True) == "T-NOT"
    f = facts()
    f[0]["ts_vs_p10"] = row(0.05, 0.01, 0.09)                        # the excess remains at seed 0
    assert R.outcome(f, True) == "T-PARTIAL"
    f = facts()
    f[0]["ts_vs_own"] = row(0.02, 0.01, 0.03)                        # separated-WORSE is not better
    assert R.outcome(f, True) == "T-NOT"
    assert R.outcome(facts(), False) == "INVALID"


def test_t_share_known_answer():
    rng = np.random.default_rng(1)
    n = 600
    eid = np.repeat(np.arange(30), 20).astype(str)
    b = rng.uniform(0.0, 0.1, n)
    a_own = b + 0.4 + rng.normal(0, 0.01, n)
    a_ts = a_own - 0.5 * (a_own - b)                                  # TSWAP removes exactly half of the excess
    r = R.t_share(a_own, a_ts, b, eid, 400, GD)
    assert abs(r["share"] - 0.5) < 1e-9 and r["lo"] <= 0.5 <= r["hi"] and r["n_windows"] == n
    a_ts2 = b.copy()
    assert abs(R.t_share(a_own, a_ts2, b, eid, 400, GD)["share"] - 1.0) < 1e-9
    assert abs(R.t_share(a_own, a_own, b, eid, 400, GD)["share"]) < 1e-12


need = pytest.mark.skipif(not os.path.isdir(f"{OUT}/S_MUT/ticks"), reason="subset outputs absent")


@need
def test_g_live_fails_on_the_mutant():
    rec = json.load(open(f"{OUT}/receipts/receipt_S_MUT.json"))
    assert G.gate_live(f"{OUT}/S_MUT", "S_MUT", rec, f"{OUT}/S_OWN", "S_OWN")["PASS"] is False        # a twin that swaps nothing must fail the live control
    rec2 = json.load(open(f"{OUT}/receipts/receipt_S_TSWAP.json"))
    assert G.gate_live(f"{OUT}/S_TSWAP", "S_TSWAP", rec2, f"{OUT}/S_OWN", "S_OWN")["PASS"] is True


@need
def test_g_id_fails_on_one_flipped_bit(tmp_path):
    d = tmp_path / "X"
    shutil.copytree(f"{OUT}/S_OWN", d)
    for sub, name in (("dump_S_OWN", None), ("ticks", None)):
        pass
    assert G.records_equal(str(d), "S_OWN", f"{OUT}/S_OWN", "S_OWN")["PASS"] is True
    f = sorted(os.listdir(d / "dump_S_OWN"))[0]
    z = dict(np.load(d / "dump_S_OWN" / f, allow_pickle=True))
    z["cl"] = z["cl"].copy(); z["cl"].flat[0] = np.nextafter(z["cl"].flat[0], np.inf)
    np.savez(d / "dump_S_OWN" / f, **z)
    assert G.records_equal(str(d), "S_OWN", f"{OUT}/S_OWN", "S_OWN")["PASS"] is False


@need
def test_g_tick0_fails_when_plan_differs_at_tick0(tmp_path):
    d = tmp_path / "Y"
    shutil.copytree(f"{OUT}/S_TSWAP", d)
    t = dict(np.load(d / "ticks" / "ticks_S_TSWAP.npz", allow_pickle=True))
    i = int(np.where(t["tick"] == 0)[0][0])
    t["plan"] = t["plan"].copy(); t["plan"][i, 0, 0] += 1e-9
    np.savez(d / "ticks" / "ticks_S_TSWAP.npz", **t)
    rec = json.load(open(f"{OUT}/receipts/receipt_S_TSWAP.json"))
    assert G.gate_tick0(str(d), "S_TSWAP", f"{OUT}/S_OWN", "S_OWN", rec)["PASS"] is False
