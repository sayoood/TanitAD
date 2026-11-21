"""``--assemble-only`` rebuilds a seam from banked parts WITHOUT asking the device gate.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_bridge_assemble_only.py

⛔ WHY THE BYPASS IS THE POINT. Assembling is file concatenation: no model, no tensor, no device.
A gate call there refuses the recovery path in exactly the state it exists for — a busy box, a
killed run, parts on disk — and the operator then cannot read work that is already paid for.

The gate is stubbed to EXPLODE if consulted, so the bypass is proven by a mutation, not inspected:
the same call **without** ``--assemble-only`` must hit that stub. A test that only ran the
assemble path would pass just as happily against a build where the gate was never wired at all.
"""
import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "code"


def _mod():
    spec = importlib.util.spec_from_file_location("w3_bridge_ao", CODE / "run_bridge_navtest.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_bridge_ao"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()
TOKS = ["tokA", "tokB"]


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """A minimal export + bank index + one banked part, and a gate that must not be called."""
    exp = tmp_path / "inputs.json.gz"
    with gzip.open(exp, "wt", encoding="utf-8") as f:
        json.dump({"tokens": {t: {"fingerprint": f"fp{t}", "log_name": "log1",
                                  "timestamps_us": [0, 500000],
                                  "ego_statuses": []} for t in TOKS}}, f)
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "index.json").write_text(json.dumps({"tokens": {t: {"rig_key": "r0"} for t in TOKS}}),
                                     encoding="utf-8")
    out = tmp_path / "out"
    (out / "chunks_A1_ego_cmd").mkdir(parents=True)
    M.chunk_write(out / "chunks_A1_ego_cmd", 0, {
        "token": np.asarray(TOKS), "fingerprint": np.asarray([f"fp{t}" for t in TOKS]),
        "source": np.asarray(["refcv4b"] * 2), "device": np.asarray(["cpu"] * 2),
        "poses": np.stack([np.full((8, 3), 1.0, np.float32), np.full((8, 3), 2.0, np.float32)]),
        "knots": np.stack([np.full((8, 2), 1.0, np.float32), np.full((8, 2), 2.0, np.float32)])})

    called = {"n": 0}

    def _explode(args, log=print):
        called["n"] += 1
        raise AssertionError("the device gate was consulted")

    fake = types.ModuleType("taniteval.bench.plugins.navsim_v1")
    fake.device_decision = _explode
    monkeypatch.setitem(sys.modules, "taniteval.bench.plugins.navsim_v1", fake)
    return exp, bank, out, called


def _argv(exp, bank, out, *extra):
    return ["--arms", "A1_ego_cmd", "--inputs", str(exp), "--bank", str(bank),
            "--out", str(out), "--device", "cpu", *extra]


def test_assemble_only_rebuilds_the_seam_without_consulting_the_gate(rig):
    exp, bank, out, called = rig
    assert M.main(_argv(exp, bank, out, "--assemble-only")) == 0
    assert called["n"] == 0                                  # LITERAL: the gate was never asked
    z = np.load(out / "seam_A1_ego_cmd.npz", allow_pickle=False)
    assert z["token"].tolist() == TOKS
    assert z["poses"][0][0][0] == 1.0 and z["poses"][1][0][0] == 2.0
    man = json.load(open(out / "seam_A1_ego_cmd.manifest.json", encoding="utf-8"))
    assert man["n"] == 2 and man["partial"] is False and man["devices"] == {"cpu": 2}


def test_mutation_the_same_call_without_the_flag_does_consult_the_gate(rig):
    """⛔ THE CONTROL. If this passes silently the test above proves nothing — it would be green
    against a build with no gate at all."""
    exp, bank, out, called = rig
    with pytest.raises(AssertionError, match="device gate was consulted"):
        M.main(_argv(exp, bank, out))
    assert called["n"] == 1


def test_a_partial_assembly_is_marked_partial_and_carries_its_denominator(rig, tmp_path):
    """A seam over part of the split must SAY so — `n_requested` travels with `n`, in the npz and
    in the manifest, so a subset can never be read as the full split by a later consumer."""
    exp, bank, out, _ = rig
    exp2 = tmp_path / "inputs3.json.gz"
    doc = json.load(gzip.open(exp, "rt", encoding="utf-8"))
    doc["tokens"]["tokC"] = dict(doc["tokens"]["tokA"], fingerprint="fptokC")
    with gzip.open(exp2, "wt", encoding="utf-8") as f:
        json.dump(doc, f)
    (bank / "index.json").write_text(
        json.dumps({"tokens": {t: {"rig_key": "r0"} for t in TOKS + ["tokC"]}}), encoding="utf-8")
    assert M.main(_argv(exp2, bank, out, "--assemble-only")) == 0
    man = json.load(open(out / "seam_A1_ego_cmd.manifest.json", encoding="utf-8"))
    assert man["n"] == 2 and man["n_requested"] == 3 and man["partial"] is True
    z = np.load(out / "seam_A1_ego_cmd.npz", allow_pickle=False)
    assert int(z["n_requested"]) == 3 and len(z["token"]) == 2
