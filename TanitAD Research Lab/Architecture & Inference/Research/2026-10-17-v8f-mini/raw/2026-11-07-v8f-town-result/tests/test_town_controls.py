"""Controls of PREREG_V8F_TOWN.md as tests: real 10-episode smoke runs (F:/TanitAD-scratch/motion_c/town/out/S_*) plus synthetic known answers. CPU only."""
import os
import sys
import json
import shutil

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import town_gates as G                                                       # noqa: E402
import town_read as R                                                        # noqa: E402

OUT = "F:/TanitAD-scratch/motion_c/town/out"
need = lambda tag: pytest.mark.skipif(not os.path.exists(f"{OUT}/{tag}/ticks/twin_{tag}.npz") and not os.path.exists(f"{OUT}/{tag}/ticks/ticks_{tag}.npz"), reason=f"{tag} not run")
rcpt = lambda tag: json.load(open(f"{OUT}/receipts/receipt_{tag}.json"))


@need("S_TOWN")
def test_identity_dump_run_equals_tool_of_record_same_threads():
    r = G.gate_id(f"{OUT}/S_TOWN", "S_TOWN", f"{OUT}/S_BARE", "S_BARE")
    assert r["PASS"] and r["max_abs_delta_tracks"] == 0.0 and r["episode_files"] == 10 and r["ticks_equal"]


@need("S_TOWN")
def test_twin_controls_pass_on_real_run():
    r = G.gate_twin(f"{OUT}/S_TOWN", "S_TOWN", rcpt("S_TOWN"))
    assert r["PASS"] and r["G-TWIN0"]["tick0_rows"] == 170 and r["G-TWINL"]["later_swap_unequal"] == r["G-TWINL"]["later_rows"] == 340


@need("S_MUT")
def test_mutant_twin_that_swaps_nothing_fails_the_logged_twin_gate():
    r = G.gate_twin(f"{OUT}/S_MUT", "S_MUT", rcpt("S_MUT"))
    assert r["G-TWIN0"]["PASS"] and not r["G-TWINL"]["PASS"] and not r["PASS"] and r["G-TWINL"]["later_swap_unequal"] == 0


@need("S_REG")
def test_regression_arm_zeroes_t_on_every_call_and_restores_weights():
    rc = rcpt("S_REG")
    assert G.gate_reg(rc, 36)["PASS"] and rc["zero_t_ego"] is True
    assert not G.gate_reg(rcpt("S_TOWN"), 36)["PASS"]                          # the unzeroed run must not pass G-REG


def test_gate_id_detects_one_flipped_value(tmp_path):
    a = tmp_path / "a" / "dump_x"; b = tmp_path / "b" / "dump_y"; a.mkdir(parents=True); b.mkdir(parents=True)
    z = {"cl": np.arange(60, dtype=np.float64).reshape(2, 15, 2), "fl": np.zeros((2, 15, 2)), "ol": np.ones((2, 15, 2))}
    np.savez(a / "ep0000.npz", **z)
    z2 = {k: v.copy() for k, v in z.items()}
    np.savez(b / "ep0000.npz", **z2)
    assert G.gate_id(str(tmp_path / "a"), "x", str(tmp_path / "b"), "y", only_prefix_episodes=True)["PASS"]
    z2["cl"][1, 3, 1] = np.nextafter(z2["cl"][1, 3, 1], 1e9)                  # one ulp
    np.savez(b / "ep0000.npz", **z2)
    r = G.gate_id(str(tmp_path / "a"), "x", str(tmp_path / "b"), "y", only_prefix_episodes=True)
    assert not r["PASS"] and r["max_abs_delta_tracks"] > 0


def test_slope_known_answers():
    rng = np.random.default_rng(1)
    eid = np.repeat(np.arange(40).astype(str), 25)
    dv = rng.normal(0, 2, eid.size)
    r = R.slope_ci(dv, -0.3 * dv + rng.normal(0, .2, eid.size), eid, 300)
    assert r["below0_separated"] and abs(r["slope"] + 0.3) < 0.02
    r0 = R.slope_ci(dv, rng.normal(0, 1, eid.size), eid, 300)
    assert r0["not_separated"]


def test_rank_order_is_the_committed_one():
    names = ["HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP"]          # LON9
    order = [names[i] for i in np.argsort(R.RANK)]
    assert order == ["HOLD", "STOP", "CREEP", "DECELERATE", "FOLLOW", "KEEP", "ACCELERATE"]


def test_softmax_expectation_is_exact_on_one_hot():
    lg = np.full((1, 7), -50.0); lg[0, 4] = 50.0
    assert float((R.softmax(lg) @ R.RANK)[0]) == pytest.approx(R.RANK[4], abs=1e-9)
