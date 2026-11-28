"""Tests of rhk_argv.py, rhk_pair.py (the stock pair check on a token-swapped copy) and rhk_gate_record.py (the pod's rh_gate_record.py + 3 anchored edits) on the REAL pod argv files (pulled read-only from the finished
RH / B0 arms) and the stock pair check. CPU only. The readout-K bank, its sha and its BANK_CHECK are SYNTHETIC here (the bank does not exist yet): the tests prove the code paths and refusals, not a real K bank."""
import argparse
import hashlib
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
PO = os.path.join(B, "pod_ro")
sys.path.insert(0, B)
os.environ["RH_TREE"] = os.path.join(HERE, "ref", "tree")
sys.path.insert(0, os.path.join(HERE, "ref", "tree", "stack"))
sys.path.insert(0, os.path.join(HERE, "ref", "tree", "stack", "scripts"))
import importlib.util  # noqa: E402

import rhk_argv as A  # noqa: E402
import rhk_pair as KP  # noqa: E402
import rollhist_pair_check as P  # noqa: E402

KSHA = {0: "ab" * 32, 1: "cd" * 32}


def load(n):
    return A.load(os.path.join(PO, n))


def rh_sha(s):
    return hashlib.sha256(open(os.path.join(PO, f"bank_s{s}.json"), "rb").read()).hexdigest()


def build(s, rh=None, **kw):
    kw.setdefault("k_bank_sha", KSHA[s])
    kw.setdefault("k_generator", dict(A.K_GEN))
    kw.setdefault("rh_bank_sha", rh_sha(s))
    return A.build(rh or load(f"argv_RH_s{s}.json"), s, "/root/rollhist", **kw)


@pytest.mark.parametrize("s", [0, 1])
def test_real_rh_argv_gives_rhk_with_exactly_the_bank_flags_and_out_changed(s):
    rh = load(f"argv_RH_s{s}.json")
    new, bad = build(s)
    assert bad == [], bad
    f, g = A.flags(new), A.flags(rh)
    assert sorted(k for k in set(f) | set(g) if f.get(k) != g.get(k)) == ["--dagger-bank", "--dagger-bank-readout", "--dagger-bank-sha256", "--out"]
    assert f["--dagger-bank"] == [f"/root/rollhist/bank_k/s{s}"] and f["--dagger-bank-sha256"] == [KSHA[s]] and f["--dagger-bank-readout"] == ["causal_g0x"]
    assert f["--out"] == [f"/root/rollhist/runs/RHK_s{s}/train"]
    assert f["--dagger-p"] == ["0.1"] == g["--dagger-p"] and f["--dagger-bank-kappa"] == ["0.2"]            # RH's dose and heading bound
    assert f["--init-ckpt"] == g["--init-ckpt"] and f["--steps"] == ["3000"] and f["--seed"] == [str(s)] and f["--dagger-seed"] == g["--dagger-seed"]


@pytest.mark.parametrize("s", [0, 1])
def test_stock_pair_check_through_the_wrapper_passes_rhk_against_the_existing_b0(s):
    new, _ = build(s)
    b0 = load(f"argv_B0_s{s}.json")
    assert KP.rhk_problems(new, b0, KSHA[s]) == []
    assert P.pair_problems(new, b0, bank_sha256=KSHA[s]) != []                                   # the stock file alone refuses the K token: that is why the wrapper exists
    assert P.RH_READOUT == "causal_g1x"                                                          # and it was not patched


def test_the_wrapper_refuses_what_the_stock_check_refuses_and_a_wrong_token():
    new, _ = build(0)
    b0 = load("argv_B0_s0.json")
    assert KP.rhk_problems(A.setf(new, "--dagger-bank-readout", ["causal_g1x"]), b0, KSHA[0])                 # C* under the RHK wrapper
    assert KP.rhk_problems(A.setf(new, "--dagger-bank-sha256", ["00" * 32]), b0, KSHA[0])                     # the bank sha is not the bank's
    assert KP.rhk_problems(A.setf(new, "--lr", ["2e-4"]), b0, KSHA[0])                                        # a second changed flag vs B0
    assert KP.rhk_problems(A.setf(new, "--dagger-p", ["0"]), b0, KSHA[0])                                     # the swap is off
    assert KP.rhk_problems(new, A.setf(b0, "--dagger-p", ["0.1"]), KSHA[0])                                   # B0 with the swap on


def test_refusals():
    rh = load("argv_RH_s0.json")
    for mut, key in ((A.setf(rh, "--dagger-p", ["0.2"]), "dagger-p"), (A.setf(rh, "--dagger-bank", ["/elsewhere/bank"]), "dagger-bank"), (A.setf(rh, "--out", ["/root/rollhist/runs/other/train"]), "out"),
                     (A.setf(rh, "--dagger-bank-readout", ["causal_g1"]), "declaration")):
        _, bad = build(0, rh=mut)
        assert bad and any(key in b for b in bad), (key, bad)
    _, bad = build(0, k_generator={"readout": "causal_g", "acc_gain": 1.0, "path_exec": True, "kappa_max": "0.2"})        # a C* bank at bank_k
    assert bad and any("generator" in b for b in bad), bad
    _, bad = build(0, k_generator={"readout": "causal_g", "acc_gain": 0.0, "path_exec": False, "kappa_max": "0.2"})       # K without the executed path
    assert bad
    _, bad = build(0, k_bank_sha=rh_sha(0))                                                                              # the RH bank's bytes under bank_k
    assert bad and any("byte-identical" in b for b in bad), bad
    _, bad = build(1, rh=load("argv_RH_s0.json"))                                                                        # seed-0 argv under seed 1
    assert bad


def _world(tmp, with_k=True):
    R = tmp / "rh"
    for s in (0, 1):
        (R / "runs" / f"RH_s{s}").mkdir(parents=True)
        (R / "runs" / f"B0_s{s}").mkdir(parents=True)
        (R / "bank" / f"s{s}").mkdir(parents=True)
        shutil.copy(os.path.join(PO, f"bank_s{s}.json"), R / "bank" / f"s{s}" / "bank.json")
        rh = [str(x).replace("/root/rollhist", str(R).replace("\\", "/")) for x in load(f"argv_RH_s{s}.json")]
        json.dump({"argv": rh}, open(R / "runs" / f"RH_s{s}" / "argv.json", "w"))
        json.dump({"argv": load(f"argv_B0_s{s}.json")}, open(R / "runs" / f"B0_s{s}" / "argv.json", "w"))
        if with_k:
            (R / "bank_k" / f"s{s}").mkdir(parents=True)
            json.dump({"common": {"generator": dict(A.K_GEN)}, "note": "synthetic K bank header"}, open(R / "bank_k" / f"s{s}" / "bank.json", "w"))
    return R


def test_render_is_idempotent_refuses_a_differing_file_and_a_missing_bank(tmp_path, monkeypatch):
    R = _world(tmp_path)
    monkeypatch.setattr(A, "R", str(R).replace("\\", "/"))
    assert A.main(["render"]) == 0
    first = open(R / "runs" / "RHK_s0" / "argv.json").read()
    assert A.main(["render"]) == 0 and open(R / "runs" / "RHK_s0" / "argv.json").read() == first
    assert json.loads(first)["argv"][json.loads(first)["argv"].index("--dagger-p") + 1] == "0.1"
    open(R / "runs" / "RHK_s0" / "argv.json", "w").write(first.replace("causal_g0x", "causal_g1x"))
    assert A.main(["render"]) == 1                                    # never silently rewritten
    R2 = _world(tmp_path / "nok", with_k=False)
    monkeypatch.setattr(A, "R", str(R2).replace("\\", "/"))
    assert A.main(["render"]) == 1 and not os.path.exists(R2 / "runs" / "RHK_s0" / "argv.json")      # the K bank is not built yet: no argv


# ---------------------------------------------------------------------------------------------------------------- the chain-local gate record (pod rh_gate_record.py + 3 anchored edits)
def load_rec():
    spec = importlib.util.spec_from_file_location("rhk_gate_record_t", os.path.join(B, "rhk_gate_record.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rec_world(tmp, s=0, sc_token="causal_g0x", check_readout="causal_g0x", check_pass=True, bank_check=True):
    R = str(tmp)
    d = f"{R}/runs/RHK_s{s}"
    os.makedirs(f"{d}/selfcheck", exist_ok=True)
    os.makedirs(f"{R}/bank_k/s{s}", exist_ok=True)
    os.makedirs(f"{R}/pilot", exist_ok=True)
    os.makedirs(f"{R}/done", exist_ok=True)
    kb = {"common": {"generator": dict(A.K_GEN)}, "partial": False}
    json.dump(kb, open(f"{R}/bank_k/s{s}/bank.json", "w"))
    ksha = hashlib.sha256(open(f"{R}/bank_k/s{s}/bank.json", "rb").read()).hexdigest()
    new, bad = A.build(load(f"argv_RH_s{s}.json"), s, "/root/rollhist", k_bank_sha=ksha, k_generator=dict(A.K_GEN), rh_bank_sha=rh_sha(s))
    assert not bad
    json.dump({"argv": new}, open(f"{d}/argv.json", "w"))
    sc = open(os.path.join(PO, f"selfcheck_RH_s{s}.json")).read().replace('"causal_g1x"', f'"{sc_token}"')           # SYNTHETIC: the RH selfcheck with its lever token rewritten
    open(f"{d}/selfcheck/selfcheck.json", "w").write(sc)
    if bank_check:
        json.dump({"pass": check_pass, "readout": check_readout}, open(f"{R}/bank_k/s{s}/BANK_CHECK.json", "w"))
    shutil.copy(f"{PO}/fake_R/gsuite_result_rh.json", f"{R}/gsuite_result_rh.json")
    shutil.copy(f"{PO}/fake_R/pilot/PILOT_RH.json", f"{R}/pilot/PILOT_RH.json")
    shutil.copy(f"{PO}/fake_R/done/tree.done", f"{R}/done/tree.done")
    open(f"{d}/pair_RHK_s{s}.log", "w").write("ZZ ROLLHIST-PAIR PASS (readout causal_g0x; stock checker on a token-swapped copy)\n")
    return R, d


@pytest.mark.parametrize("s", [0, 1])
def test_record_mints_and_verifies_for_the_rhk_tag(tmp_path, s):
    R, d = rec_world(tmp_path, s)
    G = load_rec()
    G.R = R
    assert G.mint(argparse.Namespace(tag=f"RHK_s{s}", readout="causal_g0x")) == 0
    assert G.verify(argparse.Namespace(tag=f"RHK_s{s}")) == 0
    rec = json.load(open(f"{d}/gate/RH_GATE_RECORD.json"))
    assert rec["pass"] and rec["tag"] == f"RHK_s{s}" and f"/bank_k/s{s}/bank.json" in rec["binds"]["bank_json"]["path"]


def test_record_refuses(tmp_path):
    G = load_rec()
    for i, kw in enumerate(({"check_readout": "causal_g1x"},              # a C* full-bank check under RHK
                            {"check_pass": False},                          # a failed full-bank check (coverage BLOCKED among them)
                            {"sc_token": "causal_g1x"},                     # the trainer built a C* bank
                            {"bank_check": False})):                        # no full-bank check at all
        R, d = rec_world(tmp_path / f"r{i}", 0, **kw)
        G.R = R
        assert G.mint(argparse.Namespace(tag="RHK_s0", readout="causal_g0x")) == 1, kw
        assert os.path.isfile(f"{d}/gate/RH_GATE_REFUSED.json") and not os.path.isfile(f"{d}/gate/RH_GATE_RECORD.json")
    R, d = rec_world(tmp_path / "rt", 0)
    G.R = R
    assert G.mint(argparse.Namespace(tag="RHK_s0", readout="causal_g1x")) == 1       # the record mints readout K only
    assert G.mint(argparse.Namespace(tag="RH_s0", readout="causal_g0x")) == 1        # ... and RHK tags only
    assert G.mint(argparse.Namespace(tag="RHK_s0", readout="causal_g0x")) == 0
    bj = json.load(open(f"{R}/bank_k/s0/bank.json"))
    bj["partial"] = True
    json.dump(bj, open(f"{R}/bank_k/s0/bank.json", "w"))
    assert G.verify(argparse.Namespace(tag="RHK_s0")) == 1                           # the bank changed after the mint
