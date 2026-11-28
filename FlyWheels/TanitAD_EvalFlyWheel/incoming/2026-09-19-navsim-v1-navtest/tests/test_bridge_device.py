"""``run_bridge_navtest.run_model_dev`` is E2's ``run_model`` with ONE parameter added.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_bridge_device.py

A COPY of someone else's function is only admissible if it is proven equal to the original where
both apply. On CPU the two must return BIT-IDENTICAL trajectories on the same inputs — the
control that the added ``device`` parameter changed nothing else. Skips (never silently passes)
when the checkpoint or the smoke frame bank is absent.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "code"
BANK = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank_smoke")
EXPORT = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs_shard0log1.json.gz")
CKPT = "D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not (Path(CKPT).exists() and BANK.exists() and EXPORT.exists()),
                    reason="checkpoint / smoke bank / export absent")
def test_run_model_dev_equals_e2_run_model_on_cpu():
    import gzip
    R = _mod("w3_bridge", CODE / "run_bridge_navtest.py")
    B = R.load_e2_bridge()
    doc = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    bank = R.Bank(str(BANK))
    toks = [t for t in sorted(doc) if t in bank.index["tokens"]][:2]
    assert toks, "no smoke token in the bank"
    model, cfg, targs, prov, arm_mod = B.load_refcv4b(CKPT)
    tr = arm_mod.trainer()
    steps, W = int(prov["decoder_steps"]), int(prov["window"])
    for tok in toks:
        r = doc[tok]
        decl = B.declare(r["ego_statuses"], "A1_ego_cmd")
        fr, _, _ = bank.get(tok)
        rows = B.pack_frames(fr, B.slot_sources(R.times_rel_t0(r), "ST", W))
        mine = R.run_model_dev(B, model, tr, rows, decl, steps, "cpu")
        theirs = B.run_model(model, tr, rows, decl, steps)
        assert np.array_equal(mine["traj"], theirs["traj"]), f"{tok}: trajectories differ"
        assert mine["diag"] == theirs["diag"] and mine["ego"] == theirs["ego"]
        assert mine["nav"] == theirs["nav"]
        assert np.isfinite(mine["traj"]).all() and mine["traj"].shape == (8, 2)


@pytest.mark.skipif(not BANK.exists(), reason="smoke bank absent")
def test_bank_refuses_a_tampered_sha():
    R = _mod("w3_bridge2", CODE / "run_bridge_navtest.py")
    bank = R.Bank(str(BANK))
    tok = sorted(bank.index["tokens"])[0]
    bank.get(tok)                                        # the honest read passes
    bank.index["tokens"][tok]["sha256"] = "0" * 16       # the mutation must go RED
    with pytest.raises(ValueError):
        bank.get(tok)
