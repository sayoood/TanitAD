"""Tests of tok_probe_u_ck.py (card PREREG_V8F_PROOF_ADD123, control q1). CPU only, torch, a tiny synthetic batch stream fed to the banked ``run_training`` directly.
The banked tool is the file tok_probe_u.py in the folder TP_DIR (default: two folders above this file, the banked kit folder). The study tree is not needed: the model's head and its lead term are replaced
by small stand-ins (the saving tool touches neither). Every check holds at least one planted wrong statement that must fail (``must_fail``).

Run:  PYCAP_MB=3000 bash F:/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest test_ck_tool.py -q -p no:cacheprovider --basetemp <NEW folder>
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import pytest
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TP_DIR = Path(os.environ.get("TP_DIR", str(HERE.parent.parent)))
CKFILE = HERE / "tok_probe_u_ck.py"
TOTAL, NB, FRAMES = 12, 12, 7
ARMS = ("TK16_s0", "TK16_s1")
torch.set_num_threads(2)


def load_ck():
    spec = importlib.util.spec_from_file_location("tok_probe_u_ck_under_test", str(CKFILE))
    m = importlib.util.module_from_spec(spec)
    sys.modules["tok_probe_u_ck_under_test"] = m
    spec.loader.exec_module(m)
    return m


class StubHead(nn.Module):
    def __init__(self, dim, hist):
        super().__init__()
        self.hist = hist
        self.fc = nn.Linear(dim * hist, 3)

    def forward(self, x):                                                     # [B, F, dim] -> [B, F - hist + 1, 3]
        return self.fc(x.unfold(1, self.hist, 1).flatten(-2))


def stub_terms(out, batch, hist):
    t = (out - 0.1).pow(2).mean()
    z = torch.tensor(float(out.shape[0]))
    return {"lead_sup": t, "lead_sup_bce": t.detach() * 0.5, "lead_sup_gap": t.detach() * 0.25, "lead_sup_clos": t.detach() * 0.25,
            "lead_sup_n_gap": z, "lead_sup_n_pres": z, "lead_sup_n_clos": z}


def make_batches():
    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(NB):
        out.append({"feats": torch.randn(2, FRAMES, 40, 1024, generator=g), "frame_ok": torch.ones(2, FRAMES), "lead_pres": torch.ones(2, FRAMES),
                    "lead_gapm": torch.ones(2, FRAMES), "lead_clos": torch.ones(2, FRAMES)})
    return out


BATCHES = make_batches()


def lr_at(step):
    return 1e-2 * 0.5 * (1 + math.cos(math.pi * step / TOTAL)) + 1e-4          # does not depend on the run's total


@pytest.fixture()
def world():
    ck = load_ck()
    tpu = ck.load_banked(str(TP_DIR))
    tpu._TREE.clear()
    tpu._TREE.update({"M": types.SimpleNamespace(LeadSupHead=StubHead, lead_sup_terms=stub_terms)})
    orig = {k: getattr(tpu, k) for k in ("build_probe", "step_probe", "run_training", "main")}
    yield types.SimpleNamespace(ck=ck, tpu=tpu)
    for k, v in orig.items():
        setattr(tpu, k, v)
    tpu._TREE.clear()


def train(tpu, out, total=TOTAL, arms=ARMS):
    return tpu.run_training(iter(BATCHES), list(arms), total, lr_at, lr=1e-2, wd=0.01, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(out), argv_sha="abc123", grid="pool4x10")


def state(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def must_fail(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("a planted wrong statement was accepted: the check is blind")


def _assert(x):
    assert x


def test_last_weights_with_and_without_the_saving_tool_are_bit_equal(world, tmp_path):
    tpu, ck = world.tpu, world.ck
    a, b = tmp_path / "plain", tmp_path / "saved"
    train(tpu, a)
    rng_a = torch.get_rng_state()
    saver = ck.Saver(tpu, [4, 8]).install()
    train(tpu, b)
    saver.restore()
    rng_b = torch.get_rng_state()
    for arm in ARMS:
        assert same(state(a / arm / "ckpt_last.pt"), state(b / arm / "ckpt_last.pt"))
    assert torch.equal(rng_a, rng_b)                                          # no random number was drawn by the saving
    la = [json.loads(x) for x in open(a / ARMS[0] / "train_log.jsonl", encoding="utf-8") if x.strip()][1:]
    lb = [json.loads(x) for x in open(b / ARMS[0] / "train_log.jsonl", encoding="utf-8") if x.strip()][1:]
    assert [r["lead_sup"] for r in la] == [r["lead_sup"] for r in lb]
    # planted wrong: a run with another learning rate must NOT read as equal
    c = tmp_path / "other"
    tpu.run_training(iter(BATCHES), list(ARMS), TOTAL, lambda s: 2e-2, lr=1e-2, wd=0.01, clip_grad=1.0, w_loss=1.0, device="cpu", out_dir=str(c), argv_sha="abc123", grid="pool4x10")
    must_fail(lambda: _assert(same(state(a / ARMS[0] / "ckpt_last.pt"), state(c / ARMS[0] / "ckpt_last.pt"))))
    assert not same(state(a / ARMS[0] / "ckpt_last.pt"), state(c / ARMS[0] / "ckpt_last.pt"))


def test_a_step_file_equals_the_last_weights_of_a_run_stopped_at_that_step(world, tmp_path):
    tpu, ck = world.tpu, world.ck
    full = tmp_path / "full"
    saver = ck.Saver(tpu, [4, 8]).install()
    train(tpu, full)
    saver.restore()
    for n in (4, 8):
        short = tmp_path / ("stop%d" % n)
        train(tpu, short, total=n)
        for arm in ARMS:
            got = state(full / arm / ("ckpt_step%d.pt" % n))
            assert same(got, state(short / arm / "ckpt_last.pt"))
            blob = torch.load(str(full / arm / ("ckpt_step%d.pt" % n)), map_location="cpu", weights_only=False)
            assert blob["step"] == n and blob["arm"] == arm and blob["argv_sha"] == "abc123" and blob["n_tok_seen"] == 40 and blob["kind"] == "tokprobe"
    # planted wrong: the step-4 file is not the step-8 weights, nor the last weights
    s8 = state(tmp_path / "stop8" / ARMS[0] / "ckpt_last.pt")
    must_fail(lambda: _assert(same(state(full / ARMS[0] / "ckpt_step4.pt"), s8)))
    assert not same(state(full / ARMS[0] / "ckpt_step4.pt"), state(full / ARMS[0] / "ckpt_last.pt"))


def test_the_step_files_load_through_the_banked_load_probe(world, tmp_path):
    tpu, ck = world.tpu, world.ck
    out = tmp_path / "run"
    saver = ck.Saver(tpu, [4, 8]).install()
    train(tpu, out)
    saver.restore()
    for n in (4, 8):
        for arm in ARMS:
            p, blob = tpu.load_probe(str(out / arm / ("ckpt_step%d.pt" % n)))
            assert blob["step"] == n and blob["arm"] == arm
            assert same({k: v.detach().cpu() for k, v in p.state_dict().items()}, state(out / arm / ("ckpt_step%d.pt" % n)))
            with torch.no_grad():
                y = p(BATCHES[0]["feats"])
            assert tuple(y.shape) == (2, FRAMES - 5 + 1, 3)
    junk = tmp_path / "junk.pt"
    torch.save({"kind": "other"}, str(junk))
    with pytest.raises(SystemExit):                                            # a file that is not a tokprobe blob is refused by the banked reader
        tpu.load_probe(str(junk))


def test_a_ck_step_not_below_the_total_is_refused_with_exit_3(world, tmp_path):
    tpu, ck = world.tpu, world.ck
    for bad in ([4, TOTAL], [4, TOTAL + 5]):
        saver = ck.Saver(tpu, bad).install()
        with pytest.raises(SystemExit) as e:
            train(tpu, tmp_path / ("bad%d" % bad[1]))
        saver.restore()
        assert e.value.code == 3
        assert not (tmp_path / ("bad%d" % bad[1]) / ARMS[0] / "ckpt_step4.pt").exists()
    # the same refusal at the command line, before anything is built
    called = []
    tpu.main = lambda argv=None: called.append(list(argv)) or 0
    with pytest.raises(SystemExit) as e:
        ck.main(["train", "--steps", str(TOTAL), "--ck-steps", "4,%d" % TOTAL, "--out", "x"], tpu=tpu)
    assert e.value.code == 3 and called == []
    # planted wrong: a valid list must not be refused
    saver = ck.Saver(tpu, [4, TOTAL - 1]).install()
    train(tpu, tmp_path / "ok")
    saver.restore()
    assert (tmp_path / "ok" / ARMS[0] / ("ckpt_step%d.pt" % (TOTAL - 1))).exists()
    must_fail(lambda: _assert(not (tmp_path / "ok" / ARMS[0] / ("ckpt_step%d.pt" % (TOTAL - 1))).exists()))


def test_main_removes_its_own_option_and_passes_the_rest_through(world, tmp_path):
    tpu, ck = world.tpu, world.ck
    seen = {}

    def fake_main(argv=None):
        seen["argv"] = list(argv)
        assert tpu.run_training is not seen["orig_rt"]                         # the saving is installed while the banked main runs
        train(tpu, tmp_path / "viamain")
        return 0

    seen["orig_rt"] = tpu.run_training
    tpu.main = fake_main
    rc = ck.main(["train", "--out", "x", "--ck-steps", "4,8", "--arms", "TK16_s0,TK16_s1"], tpu=tpu)
    assert rc == 0 and seen["argv"] == ["train", "--out", "x", "--arms", "TK16_s0,TK16_s1"]
    assert tpu.run_training is seen["orig_rt"]                                 # restored afterwards
    assert (tmp_path / "viamain" / ARMS[1] / "ckpt_step8.pt").exists()
    # other sub-commands and runs without the option go to the banked main untouched
    seen2 = []
    tpu.main = lambda argv=None: seen2.append(list(argv)) or 0
    assert ck.main(["extract", "--ckpt", "f.pt"], tpu=tpu) == 0 and ck.main(["train", "--out", "y"], tpu=tpu) == 0
    assert seen2 == [["extract", "--ckpt", "f.pt"], ["train", "--out", "y"]]
    must_fail(lambda: _assert(seen["argv"].count("--ck-steps") == 1))


def test_ck_record_has_exactly_the_four_keys(world, tmp_path):
    tpu, ck = world.tpu, world.ck
    out = tmp_path / "rec"
    saver = ck.Saver(tpu, [8, 4]).install()
    train(tpu, out)
    saver.restore()
    rec = json.load(open(out / "CK_RECORD.json", encoding="utf-8"))
    assert sorted(rec) == ["ck_steps", "files", "tok_probe_u_sha256", "tool_sha256"]
    assert rec["ck_steps"] == [4, 8] and all(isinstance(x, int) for x in rec["ck_steps"])
    assert rec["files"] == {a: ["ckpt_step4.pt", "ckpt_step8.pt"] for a in ARMS}
    assert len(rec["tok_probe_u_sha256"]) == 64 and set(rec["tok_probe_u_sha256"]) <= set("0123456789abcdef")
    assert rec["tok_probe_u_sha256"] == hashlib.sha256((TP_DIR / "tok_probe_u.py").read_bytes()).hexdigest()
    assert rec["tool_sha256"] == hashlib.sha256(CKFILE.read_bytes()).hexdigest()
    must_fail(lambda: _assert(sorted(rec) == ["ck_steps", "files", "tool_sha256"]))
    must_fail(lambda: _assert(rec["tool_sha256"] == rec["tok_probe_u_sha256"]))


def test_the_banked_tool_file_is_not_edited_by_the_tool(world):
    before = hashlib.sha256((TP_DIR / "tok_probe_u.py").read_bytes()).hexdigest()
    world.ck.Saver(world.tpu, [4]).install().restore()
    assert hashlib.sha256((TP_DIR / "tok_probe_u.py").read_bytes()).hexdigest() == before
    assert not (HERE / "tok_probe_u.py").exists()                              # no copy of the banked tool in this folder
