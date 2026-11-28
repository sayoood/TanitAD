"""Tests of rhk_bank_check.py (the full-bank check for readout K) on SYNTHETIC banks: the shard reader is replaced by an in-memory table, bank.json / PROVENANCE.json / provenance files are real files in the shape the
generator writes (the pod's rh_pilot_check.py reads exactly these keys). It proves the check's logic and its refusals (the mutation arms), NOT a real K bank. CPU only."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
os.environ["RH_TREE"] = os.path.join(HERE, "ref", "tree")
sys.path.insert(0, os.path.join(HERE, "ref", "tree", "stack"))
sys.path.insert(0, os.path.join(B, "pod_ro"))
sys.path.insert(0, B)
import rhk_bank_check as K  # noqa: E402

C = K.C
N = 6
K_GEN = {"readout": "causal_g", "acc_gain": 0.0, "path_exec": True, "kappa_max": "0.2"}
CSTAR_GEN = {"readout": "causal_g", "acc_gain": 1.0, "path_exec": True, "kappa_max": "0.2"}


def arrays(own_acc, moving=True):
    """N records: pose (N, 3, 3) with executed speeds s1 = 5 m/s and s2 = 7 m/s (executed acceleration (7-5)/0.2 = 10), or constant speed when moving is False; kin3[:, 1] = own_acc."""
    pose = np.zeros((N, 3, 3))
    pose[:, 1, 0] = 1.0
    pose[:, 2, 0] = 1.0 + (1.4 if moving else 1.0)
    kin3 = np.zeros((N, 3))
    kin3[:, 1] = own_acc
    return {"ci": np.arange(N), "reason": np.zeros(N, np.int64), "pose": pose, "kin3": kin3}


def make_bank(tmp, own_acc=0.0, gen=K_GEN, coverage=0.8, blocked=False, moving=True, partial=False, windows=100):
    bd = str(tmp)
    os.makedirs(bd, exist_ok=True)
    table = {}
    for k in range(4):
        f = os.path.join(bd, f"shard_{k:03d}of004.npz")
        open(f, "wb").write(b"x")
        table[f] = arrays(own_acc, moving)
        json.dump({"kb1_replay": {"mismatch_chunks": 0}, "generator": gen}, open(os.path.join(bd, f"shard_{k:03d}of004.provenance.json"), "w"))
    json.dump({"coverage": {"fraction": coverage, "blocked": blocked}, "partial": partial, "common": {"generator": gen, "train": {"n_windows": windows}}}, open(os.path.join(bd, "bank.json"), "w"))
    json.dump({"controls": {"K-B4": {"PASS": True}, "K-B5": {"PASS": True}}}, open(os.path.join(bd, "PROVENANCE.json"), "w"))
    return bd, table


def run(tmp, monkeypatch, capsys, **kw):
    bd, table = make_bank(tmp, **{k: v for k, v in kw.items() if k != "expect"})
    monkeypatch.setattr(C.DG, "load_shard", lambda f: (table[f], None))
    out = os.path.join(bd, "BANK_CHECK.json")
    rc = K.main(["--bank", bd, "--expect-windows", str(kw.get("expect", 100)), "--out", out])
    return rc, json.load(open(out)), capsys.readouterr().out


def test_a_k_bank_passes_and_the_record_is_for_readout_k(tmp_path, monkeypatch, capsys):
    rc, rep, out = run(tmp_path, monkeypatch, capsys)
    assert rc == 0 and rep["pass"] and rep["readout"] == "causal_g0x" and out.strip().endswith("ZZ RH-PILOT-CHECK PASS")
    b = rep["checks"]["b_no_own_acceleration"]
    assert b["worst_abs"] == 0.0 and b["n_records"] == 4 * N and abs(b["executed_accel_max"] - 10.0) < 1e-6


def test_the_pod_file_table_is_untouched_by_name_but_extended_in_this_process_only():
    assert set(C.WANT) == {"causal_g1x", "causal_g1", "causal_g0x"}                       # the module's own table, extended in this process
    src = open(os.path.join(B, "pod_ro", "rh_pilot_check.py"), encoding="utf-8").read()
    assert "causal_g0x" not in src                                                         # the pod file on disk was not edited


def test_a_cstar_bank_is_refused_by_the_k_check(tmp_path, monkeypatch, capsys):          # mutation arm: C* (gain 1: stored acceleration = the executed 10) and its header
    rc, rep, out = run(tmp_path, monkeypatch, capsys, own_acc=10.0, gen=CSTAR_GEN)
    assert rc == 1 and not rep["pass"]
    assert not rep["checks"]["b_no_own_acceleration"]["pass"] and not rep["checks"]["d_header_generator"]["pass"]


def test_a_k_header_over_c_star_values_and_a_c_star_header_over_k_values_are_each_refused(tmp_path, monkeypatch, capsys):
    rc, rep, _ = run(tmp_path / "a", monkeypatch, capsys, own_acc=10.0)                   # header K, stored acceleration = the executed one: b refuses
    assert rc == 1 and not rep["checks"]["b_no_own_acceleration"]["pass"] and rep["checks"]["d_header_generator"]["pass"]
    rc, rep, _ = run(tmp_path / "b", monkeypatch, capsys, gen=CSTAR_GEN)                  # K values, C* header: d refuses
    assert rc == 1 and rep["checks"]["b_no_own_acceleration"]["pass"] and not rep["checks"]["d_header_generator"]["pass"]


def test_vacuous_b_is_refused(tmp_path, monkeypatch, capsys):                              # a bank of standing-still poses reads stored acceleration 0 trivially: the control must refuse it
    rc, rep, _ = run(tmp_path, monkeypatch, capsys, moving=False)
    assert rc == 1 and not rep["checks"]["b_no_own_acceleration"]["pass"] and rep["checks"]["b_no_own_acceleration"]["executed_accel_max"] < 1e-3


@pytest.mark.parametrize("kw,why", [({"coverage": 0.3}, "coverage"), ({"blocked": True}, "coverage"), ({"partial": True}, "partial"), ({"expect": 99}, "n_windows")])
def test_coverage_blocked_and_incomplete_banks_are_refused(tmp_path, monkeypatch, capsys, kw, why):
    rc, rep, out = run(tmp_path, monkeypatch, capsys, **kw)
    assert rc == 1 and not rep["pass"]
    assert why in json.dumps(rep["checks"]) or why in out
    if why == "coverage":
        assert "coverage" in out                                                          # the chain greps this word to print its BLOCKED line
