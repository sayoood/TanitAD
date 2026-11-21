"""Tests for the P6 read (CPU, banked data only; reads nothing of P6).
Run:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_p6_read.py --basetemp=F:/TanitAD_p6read_scratch/pytest
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selc6_analyze as A  # noqa: E402


def stat(**kw):
    s = dict(G_TTC=0.0, G_NC=0.0, S_TTC=-0.1, S_NC=-0.1, T_TTC=-0.1, T_NC=-0.1, share=0.15, dA_hi=-0.1, dB_hi=-0.05)
    s.update(kw)
    return s


@pytest.mark.parametrize("kw,verdict,sub", [
    ({}, "HOLDS", []),
    ({"G_TTC": -0.16}, "NOT PROVEN", ["N2"]),                      # pessimism persists but behaviour (C) holds -> not REFUTED
    ({"G_TTC": -0.16, "share": 0.3}, "REFUTED", []),
    ({"G_NC": -0.4, "dB_hi": 0.01}, "REFUTED", []),
    ({"share": 0.3}, "NOT PROVEN", ["N1"]),
    ({"dB_hi": 0.0}, "NOT PROVEN", ["N1"]),                          # CI-hi must be strictly < 0
    ({"S_TTC": -0.5}, "NOT PROVEN", ["N2", "N3"]),                   # A holds, B fails, C holds -> both named
    ({"S_TTC": -0.5, "share": 0.3}, "NOT PROVEN", ["N3"]),
    ({"G_TTC": 0.2, "share": 0.3}, "NOT PROVEN", None),              # over-corrected and behaviour fails: matches none of N1-N3
    ({"G_TTC": 0.15, "G_NC": -0.15}, "HOLDS", []),                   # bars are inclusive
    ({"share": 0.2}, "HOLDS", []),
])
def test_decide(kw, verdict, sub):
    d = A.decide(stat(**kw))
    assert d["VERDICT"] == verdict
    if sub is None:
        assert d["subcases"][0].startswith("N-unnamed")
    else:
        assert d["subcases"] == sub


def test_decide_nan_is_instrument_failure():
    assert A.decide(stat(G_TTC=float("nan")))["VERDICT"] == "INSTRUMENT FAILURE"


def test_tie_ranks_vs_argsort_of_argsort():
    """The retraction-class bug: argsort(argsort) breaks ties by proposal INDEX, so a constant-with-ties label correlates with 'speed' when speed rises with index."""
    speed = np.arange(10, dtype=float)[None]
    label = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.]])
    M = np.ones((1, 10), bool)
    assert A.per_token_spearman(label, speed, M, "tie")[0] == pytest.approx(__import__("scipy.stats", fromlist=["x"]).spearmanr(label[0], speed[0])[0], abs=1e-12)
    const = np.ones((1, 10))
    assert np.isnan(A.per_token_spearman(const, speed, M, "tie")[0])          # constant label: undefined
    assert A.per_token_spearman(const, speed, M, "bug_argsort")[0] == pytest.approx(1.0)   # the bug 'measures' index
    assert np.isnan(A.per_token_spearman(label[:, :4], speed[:, :4], M[:, :4], "tie")[0])  # < 5 valid: undefined


def test_fp32_gate_catches_bf16_grid():
    rng = np.random.default_rng(0)
    traj = np.zeros((400, 64, 20, 3), np.float32)
    traj[..., 19, 0] = rng.uniform(30, 66, size=(400, 64)).astype(np.float32)
    assert A.g6_fp32({"traj": traj})["PASS"]
    q = (np.round(traj * 4) / 4).astype(np.float32)
    assert not A.g6_fp32({"traj": q})["PASS"]
    thin = traj[:2]
    assert not A.g6_fp32({"traj": thin})["PASS"]                              # too few values in the band: the check cannot see -> RED


def test_control_reproduces_u400_and_red_arms_fail():
    g, (Du, B, S, r3) = A.g6_repro(HERE + "/u400_reference.pinned.json")
    assert g["PASS"], g
    assert g["max_abs_diff_vs_pinned_reference"] == 0.0
    t = S["R2_spearman_tie"]
    assert t["TTC"]["pred"][0] == pytest.approx(-0.546, abs=5e-4) and t["NC"]["pred"][0] == pytest.approx(-0.491, abs=5e-4)
    assert S["R1_weighted_cruise_slow_share"]["point"] == pytest.approx(0.412, abs=5e-4)
    assert r3["weighted_slow_share"]["point"] == pytest.approx(0.366, abs=5e-4)
    assert g["RED_argsort_of_argsort_tie_block_max_abs_diff"] > 1e-6


def test_pinned_reference_hash():
    assert A.sha256_file(HERE + "/u400_reference.pinned.json") == "1ad5776a19a02179cb22a63fc9d20e9a04db34d4a2ffd6b6f14796e8121682e5"


def _synthetic_p6(tmp, bf16=False, labeller="online"):
    """A stand-in 'P6' made from u400's banked data (6 of its 7 logits, rule navsim_v1) written in the exact dump/table/seam/csv formats."""
    Du = A.run_control_u400()
    L6 = Du["logits"][..., :6]
    pick = A.sc.pick_from_logits(L6, "torch", lane=False)
    N = len(pick)
    traj = Du["traj"].astype(np.float32)
    if bf16:
        traj = (np.round(traj * 4) / 4).astype(np.float32)
    fp = np.array([f"fp{i}" for i in range(N)])
    np.savez(tmp / "dump_p6.npz", token=Du["token"], log=Du["log"], fingerprint=fp, logits=L6.astype(np.float32), pick=pick, traj=traj, props8=Du["props8"], ego=Du["ego"],
             gt_valid=Du["gt_valid"], gt_pose20=Du["gt_pose20"], gt_t20=Du["gt_t20"], human8=Du["human8"], frame_ctrl_m=np.zeros(N), rule=np.array("navsim_v1"),
             ckpt=np.array("synthetic"), ckpt_sha256=np.array("0" * 64), nav_mode=np.array("command"), select=np.array("best"))
    # reverse the token order in the files: the loader must align by token, not by position
    d = np.load(tmp / "dump_p6.npz")
    rev = {k: d[k][::-1] for k in d.files if d[k].ndim > 0 and d[k].shape[0] == N}
    rest = {k: d[k] for k in d.files if k not in rev}
    np.savez(tmp / "dump_p6.npz", **rev, **rest)
    np.savez(tmp / "table_p6.npz", token=Du["token"][::-1], pdms=Du["pdms"][::-1], sub=Du["sub"][::-1], valid=Du["valid"][::-1],
             red_pick_pdms=(Du["pdms"][np.arange(N), pick] + 0.01)[::-1])
    np.savez(tmp / "seam.npz", token=Du["token"], poses=Du["props8"][np.arange(N), pick].astype(np.float32))
    with open(tmp / "p6.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "valid", "score"] + ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"])
        for i, t in enumerate(Du["token"]):
            w.writerow([t, "True", repr(float(Du["pdms"][i, pick[i]]))] + [0] * 6)
        w.writerow(["average", "True", "0"] + [0] * 6)
    json.dump({"argv": ["train.py"], "args": {"labeller": labeller, "label_backend": "pdm", "nav_mode": "command", "metric_cache": "mc", "yaw_wrap": "off", "epochs": 25},
               "meta": {"nav_mode": "command"}}, open(tmp / "config.json", "w"))
    open(tmp / "train_logs.txt", "w").write("\n".join(f"fake_train_log_{i}" for i in range(150)))
    return Du


def _args(tmp, out):
    class Ns:
        pass
    a = Ns()
    a.dump, a.table, a.out = str(tmp / "dump_p6.npz"), str(tmp / "table_p6.npz"), str(out)
    a.run_config, a.train_logs, a.p6_seam, a.p6_csv = str(tmp / "config.json"), str(tmp / "train_logs.txt"), str(tmp / "seam.npz"), str(tmp / "p6.csv")
    a.expect_ckpt_sha, a.u400_reference = None, HERE + "/u400_reference.pinned.json"
    return a


def test_full_read_on_synthetic_p6(tmp_path):
    _synthetic_p6(tmp_path)
    out = tmp_path / "out"
    assert A.full_read(_args(tmp_path, out)) == 0, json.load(open(out / "p6_speed_result.json"))["red_gates"]
    r = json.load(open(out / "p6_speed_result.json"))
    assert r["red_gates"] == [] and r["verdict"]["VERDICT"] in ("HOLDS", "REFUTED", "NOT PROVEN")
    assert set(r["gates"]) >= {"G6-PIN", "G6-ARGV", "G6-DISJOINT", "G6-0", "G6-1", "G6-2", "G6-3", "G6-4", "G6-FP32", "G6-REPRO"}
    assert r["analysis"]["descriptive"]["STRATEGIC"]["status"] == "PAUSED" and r["analysis"]["descriptive"]["TACTICAL"]["status"] == "n/a"
    assert (out / "P6_SPEED_RESULT.md").read_text(encoding="utf-8").startswith("# P6_SPEED_RESULT")


def test_bf16_dump_is_instrument_failure(tmp_path):
    _synthetic_p6(tmp_path, bf16=True)
    rc = A.full_read(_args(tmp_path, tmp_path / "out"))
    r = json.load(open(tmp_path / "out" / "p6_speed_result.json"))
    assert rc == 2 and "G6-FP32" in r["red_gates"] and r["verdict"]["VERDICT"] == "INSTRUMENT FAILURE" and "analysis" not in r


def test_not_online_voids_hypothesis(tmp_path):
    _synthetic_p6(tmp_path, labeller="bank")
    A.full_read(_args(tmp_path, tmp_path / "out"))
    r = json.load(open(tmp_path / "out" / "p6_speed_result.json"))
    assert "G6-ARGV" in r["red_gates"] and r["verdict"]["H_P6_SPEED_1_VOID"] is True


def test_missing_inputs_are_red_not_skipped(tmp_path):
    _synthetic_p6(tmp_path)
    a = _args(tmp_path, tmp_path / "out")
    a.p6_seam = a.p6_csv = a.train_logs = a.run_config = None
    assert A.full_read(a) == 2
    r = json.load(open(tmp_path / "out" / "p6_speed_result.json"))
    assert {"G6-ARGV", "G6-DISJOINT", "G6-1", "G6-3"} <= set(r["red_gates"])


def test_gate_red_arms_each_fail():
    Du = A.run_control_u400()
    assert A.g6_0(Du)["PASS"] and A.g6_2(Du)["PASS"] and A.g6_4(Du)["PASS"]
    w = dict(Du)
    w["pick"] = (Du["pick"] + 1) % 64
    assert not A.g6_0(w)["PASS"]
    s = dict(Du)
    s["sub"] = Du["sub"].copy()
    s["sub"][..., 3] *= 0.9
    assert not A.g6_4(s)["PASS"]


def test_pin_detects_drift(tmp_path, monkeypatch):
    p = tmp_path / "PREREG.md"
    p.write_text("edited")
    assert not A.g6_pin(prereg_path=str(p))["PASS"]
