"""Tests for topk_shim.py (CPU, no model weights).  Run: python -m pytest -q test_topk_shim.py"""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topk_shim as T  # noqa: E402

TEXT = T.source_text()


def fake_module():
    m = types.ModuleType("fake_v8f_mini")
    m.__file__ = "fake"
    m.torch, m.F, m.Tensor = torch, F, torch.Tensor
    m.N_GOAL, m.N_LAT, m.N_LON, m.N_CONS = 22, 5, 7, 26

    class V8FMini:
        def forward_infer(self, batch):
            return {"code": batch["code"], "traj30": batch["code"][:, :2].unsqueeze(1).expand(-1, 30, -1).clone()}

        def forward_diag(self, batch, code):
            return {"traj30": code[:, 22:24].unsqueeze(1).expand(-1, 30, -1).clone()}
    m.V8FMini = V8FMini
    return m


def make_code(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    goal = torch.rand(B, 22, generator=g)
    lat = torch.softmax(torch.randn(B, 5, generator=g), -1)
    lon = torch.softmax(torch.randn(B, 7, generator=g), -1)
    cons = torch.randn(B, 26, generator=g)
    return torch.cat([goal, lat, lon, cons], 1)


def test_pins_match_the_source_commit():
    a, b = T.segments(TEXT)
    assert T.sha(a) == T.PINS["topk_codes"] and T.sha(b) == T.PINS["forward_topk"]


def test_mutated_source_is_refused():
    bad = TEXT.replace("pr, idx = joint.topk(K, 1)", "pr, idx = joint.topk(K, 1, largest=False)")
    assert bad != TEXT
    with pytest.raises(AssertionError):
        T.attach(fake_module(), bad)


def test_attach_and_topk_structure():
    m = fake_module()
    rec = T.attach(m, TEXT)
    assert rec["attached"] and rec["sha256"] == T.PINS
    code = make_code()
    r = m.V8FMini().forward_topk({"code": code}, 3)
    assert r["traj30"].shape == (4, 3, 30, 2) and r["prob"].shape == (4, 3)
    assert torch.equal(r["code"], code)
    assert torch.equal(r["traj30_top1"], m.V8FMini().forward_infer({"code": code})["traj30"])
    assert bool((r["prob"][:, :-1] >= r["prob"][:, 1:]).all())
    joint = (code[:, 22:27, None] * code[:, None, 27:34]).flatten(1)
    assert torch.allclose(r["prob"][:, 0], joint.max(1).values)
    codes, _ = m.topk_codes(code, 3)
    for c in codes:
        assert torch.equal(c[:, :22], code[:, :22]) and torch.equal(c[:, 34:], code[:, 34:])
        assert torch.equal(c[:, 22:27].sum(1), torch.ones(4)) and torch.equal(c[:, 27:34].sum(1), torch.ones(4))
    top = joint.argmax(1)
    assert torch.equal(codes[0][:, 22:27].argmax(1), top // 7) and torch.equal(codes[0][:, 27:34].argmax(1), top % 7)


def test_no_attach_when_present():
    m = fake_module()
    m.V8FMini.forward_topk = lambda self, b, K=3: None
    assert T.attach(m, TEXT) == {"attached": False, "reason": "V8FMini already has forward_topk"}
