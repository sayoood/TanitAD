"""rh_gate_record.py (the chain-local record, UNCHANGED, tag-prefix keyed) on the REAL pod files for the new tag RHp3_s0 / RHp3_s1: the real G-SUITE result, pilot, bank.json, BANK_CHECK.json and the finished RH arm's selfcheck.
This is the reachability proof that the tag name RHp3_sK passes the record's own file naming and its declaration / dose / bank-sha clauses."""
import argparse
import importlib.util
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
PO = os.path.join(B, "pod_ro")
sys.path.insert(0, B)
import rhp3_argv as A  # noqa: E402


def load_mod():
    spec = importlib.util.spec_from_file_location("rh_gate_record_ro", os.path.join(PO, "rh_gate_record.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def world(tmp, s=0, p="0.3"):
    R = str(tmp)
    d = f"{R}/runs/RHp3_s{s}"
    os.makedirs(f"{d}/selfcheck", exist_ok=True)
    os.makedirs(f"{R}/bank/s{s}", exist_ok=True)
    os.makedirs(f"{R}/pilot", exist_ok=True)
    os.makedirs(f"{R}/done", exist_ok=True)
    rh = A.load(os.path.join(PO, f"argv_RH_s{s}.json"))
    bsha = __import__("hashlib").sha256(open(f"{PO}/bank_s{s}.json", "rb").read()).hexdigest()
    new, bad = A.build(rh, s, "/root/rollhist", bank_sha=bsha)
    assert not bad
    new = A.setf(new, "--dagger-p", [p])
    json.dump({"argv": new}, open(f"{d}/argv.json", "w"))
    shutil.copy(f"{PO}/selfcheck_RH_s{s}.json", f"{d}/selfcheck/selfcheck.json")
    shutil.copy(f"{PO}/bank_s{s}.json", f"{R}/bank/s{s}/bank.json")
    shutil.copy(f"{PO}/BANK_CHECK_s{s}.json", f"{R}/bank/s{s}/BANK_CHECK.json")
    shutil.copy(f"{PO}/fake_R/gsuite_result_rh.json", f"{R}/gsuite_result_rh.json")
    shutil.copy(f"{PO}/fake_R/pilot/PILOT_RH.json", f"{R}/pilot/PILOT_RH.json")
    shutil.copy(f"{PO}/fake_R/done/tree.done", f"{R}/done/tree.done")
    open(f"{d}/pair_RHp3_s{s}.log", "w").write("ZZ ROLLHIST-PAIR PASS (readout causal_g1x; checker stock)\n")
    return R, d


@pytest.mark.parametrize("s", [0, 1])
def test_record_mints_and_verifies_for_the_rhp3_tag(tmp_path, s):
    R, d = world(tmp_path, s)
    G = load_mod()
    G.R = R
    assert G.mint(argparse.Namespace(tag=f"RHp3_s{s}", readout="causal_g1x")) == 0
    assert os.path.isfile(f"{d}/gate/RH_GATE_RECORD.json")
    assert G.verify(argparse.Namespace(tag=f"RHp3_s{s}")) == 0
    rec = json.load(open(f"{d}/gate/RH_GATE_RECORD.json"))
    assert rec["pass"] and rec["tag"] == f"RHp3_s{s}" and "bank_json" in rec["binds"]


def test_record_refuses_the_swap_off(tmp_path):
    R, d = world(tmp_path, 0, p="0")                                  # an RHp3 argv with the swap off: refused by the record's own dose clause
    G = load_mod()
    G.R = R
    assert G.mint(argparse.Namespace(tag="RHp3_s0", readout="causal_g1x")) == 1
    assert os.path.isfile(f"{d}/gate/RH_GATE_REFUSED.json")


def test_record_refuses_a_bank_that_is_not_the_argvs(tmp_path):
    R, d = world(tmp_path, 0)
    bj = json.load(open(f"{R}/bank/s0/bank.json"))
    bj["partial"] = True
    json.dump(bj, open(f"{R}/bank/s0/bank.json", "w"))
    G = load_mod()
    G.R = R
    assert G.mint(argparse.Namespace(tag="RHp3_s0", readout="causal_g1x")) == 1
