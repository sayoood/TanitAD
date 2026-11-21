"""The trainer -> gate naming contract (BACKLOG ZG-WIRE item 2), 2026-10-18.

``refc_v3_train.py`` now writes ``param_names`` into ``ckpt.pt`` -- the parameter names in OPTIMIZER-ID order -- and
``stack/scripts/zerograd_gate.py`` prefers them to its shape-alignment heuristic. Pinned here:

* ``_opt_param_names`` walks ``opt.param_groups`` the way ``Optimizer.state_dict()`` numbers ids. That is NOT
  ``model.named_parameters()`` order under ``--opt dd`` (encoder group first), which is the case a naive
  ``[n for n, _ in model.named_parameters()]`` gets wrong;
* the real synthetic ``train()`` rig (adam AND dd) saves names that match the optimizer's state shapes id for id, and
  the gate resolves them as ``param_names`` / source ``ckpt`` -- exact, not inferred;
* the gate still computes its VERDICT from the ids: a checkpoint whose embedded names do not fit (stale, reordered) is
  not blocked -- it falls back to the shape alignment and says so (``param_names_rejected``);
* ``error_kind`` on the ERROR report, which is what lets the supervisor tell "no optimizer state" (gate does not
  apply) from "the gate could not look"; and an internal crash is exit 2, not the exit 1 of a traceback (== FAIL).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_refc_v3_save_before_eval as RIG  # noqa: E402  the synthetic 1-step train() rig
import zerograd_gate as ZG  # noqa: E402

T = RIG.T
GATE = ROOT / "scripts" / "zerograd_gate.py"


# ------------------------------------------------------------------ the helper
class _Toy(torch.nn.Module):
    """``core.head`` is registered BEFORE ``core.encoder``, so named_parameters() order != the dd group order."""

    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.head = torch.nn.Linear(5, 2)
        self.core.encoder = torch.nn.Linear(3, 5)
        self.scorer = torch.nn.Linear(5, 4)

    def forward(self, x):
        return self.scorer(torch.relu(self.core.encoder(x))) + self.core.head(torch.relu(self.core.encoder(x))).sum()


def _args(**kw):
    ns = type("A", (), {})()
    ns.opt, ns.lr, ns.encoder_lr_mult, ns.weight_decay = "adam", 1e-3, 0.5, 1e-4
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_param_names_follow_the_optimizer_id_order_not_named_parameters_order():
    m = _Toy()
    named = [n for n, _ in m.named_parameters()]
    adam = T.build_optimizer(m, _args(opt="adam"))
    assert T._opt_param_names(m, adam) == named                                   # same list for plain Adam
    dd = T.build_optimizer(m, _args(opt="dd"))
    got = T._opt_param_names(m, dd)
    assert got != named, "the toy must make dd order differ, or this test proves nothing"
    assert got[:2] == ["core.encoder.weight", "core.encoder.bias"]                # the encoder group is FIRST
    assert sorted(got) == sorted(named)
    # id i of opt.state_dict() IS got[i]: the tensors line up one for one
    by_name = dict(m.named_parameters())
    flat = [p for g in dd.param_groups for p in g["params"]]
    assert [by_name[n] is p for n, p in zip(got, flat)] == [True] * len(flat)
    # a tensor the model does not name is reported, not guessed
    dd.param_groups[1]["params"].append(torch.nn.Parameter(torch.zeros(2)))
    assert T._opt_param_names(m, dd)[-1].startswith("<unnamed:")


# ------------------------------------------------------------------ the real trainer
def _train(out, extra=(), steps=2):
    mp = pytest.MonkeyPatch()
    RIG._rig(mp)
    try:
        args = T.build_parser().parse_args(RIG._argv(out) + ["--steps", str(steps)] + list(extra))
        assert T.train(args) == {"step": steps}
    finally:
        mp.undo()
    return torch.load(Path(out) / "ckpt.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("extra", [[], ["--opt", "dd", "--warmup", "1"]], ids=["adam", "dd"])
def test_the_trainers_checkpoint_carries_names_that_fit_the_optimizer_state(tmp_path, extra):
    ck = _train(tmp_path / "run", extra)
    names = ck["param_names"]
    ids = [i for g in ck["opt"]["param_groups"] for i in g["params"]]
    assert len(names) == len(ids) >= 100 and len(set(names)) == len(names)
    sd = ck["model"]
    for i, n in zip(ids, names):
        assert n in sd, n
        if i in ck["opt"]["state"]:                                                   # shape agreement id for id
            assert tuple(ck["opt"]["state"][i]["exp_avg"].shape) == tuple(sd[n].shape), n
    rep = ZG.analyze(ck)                                                              # the gate takes them as exact
    assert rep["alignment"]["mode"] == "param_names" and rep["alignment"]["param_names_source"] == "ckpt"
    assert rep["alignment"]["unique"] and rep["alignment"]["n_ids_name_ambiguous"] == 0
    assert rep["alignment"]["param_names_rejected"] is None
    # the weights-only milestone file has no optimizer state at all: that is the gate's "not applicable" case
    ms = torch.load(tmp_path / "run" / "ckpt_1.pt", map_location="cpu", weights_only=False)
    with pytest.raises(ZG.GateError) as ei:
        ZG.analyze(ms)
    assert ei.value.kind == "no_optimizer_state"


def test_the_gate_prefers_the_embedded_names_and_degrades_to_the_alignment_without_them(tmp_path):
    ck = _train(tmp_path / "run", ["--opt", "dd", "--warmup", "1"])
    exact = ZG.analyze(ck)
    bare = {k: v for k, v in ck.items() if k != "param_names"}                        # an older checkpoint
    inferred = ZG.analyze(bare)
    assert exact["alignment"]["mode"] == "param_names" and inferred["alignment"]["mode"] != "param_names"
    assert exact["verdict"] == inferred["verdict"]                                    # naming never decides the verdict
    # a reordered (stale) embedded list is REJECTED loudly and the gate still answers
    bad = dict(ck)
    names = list(ck["param_names"])
    j = next(k for k in range(1, len(names)) if tuple(ck["model"][names[k]].shape) != tuple(ck["model"][names[0]].shape))
    names[0], names[j] = names[j], names[0]
    bad["param_names"] = names
    rep = ZG.analyze(bad)
    assert rep["alignment"]["param_names_rejected"] and rep["alignment"]["mode"] != "param_names"
    assert rep["alignment"]["param_names_source"] is None and rep["verdict"] == exact["verdict"]
    assert "REJECTED" in ZG.render(dict(rep, ckpt="x"))
    # ... whereas an EXPLICIT --param-names that does not fit stays an error: the caller asked for exactness
    with pytest.raises(ZG.GateError) as ei:
        ZG.analyze(bare, param_names=names)
    assert ei.value.kind == "bad_param_names"


# ------------------------------------------------------------------ error_kind and the exit code of a crash
def _cli(*argv):
    return subprocess.run([sys.executable, str(GATE), *map(str, argv)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)


def test_error_kind_is_in_the_report_and_a_crash_exits_2_not_1(tmp_path):
    w = tmp_path / "weights_only.pt"
    torch.save({"model": {"a.weight": torch.zeros(2, 2)}, "step": 3}, w)
    out = tmp_path / "r.json"
    r = _cli("--ckpt", w, "--json-out", out)
    rep = json.loads(out.read_text())
    assert r.returncode == 2 and rep["verdict"] == "ERROR" and rep["error_kind"] == "no_optimizer_state"
    r = _cli("--ckpt", tmp_path / "missing.pt", "--json-out", out)
    assert r.returncode == 2 and json.loads(out.read_text())["error_kind"] == "unreadable"
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    r = _cli("--ckpt", junk, "--json-out", out)
    assert r.returncode == 2 and json.loads(out.read_text())["error_kind"] == "unreadable"
    # a crash inside the gate (here: --param-names points at a file that does not exist) used to be an uncaught
    # exception == exit 1 == the SAME code as a FAIL verdict
    ok = tmp_path / "ok.pt"
    lin = torch.nn.Linear(2, 2)
    o = torch.optim.Adam(lin.parameters())
    lin(torch.ones(1, 2)).sum().backward()
    o.step()
    torch.save({"model": lin.state_dict(), "opt": o.state_dict(), "step": 1}, ok)
    r = _cli("--ckpt", ok, "--param-names", tmp_path / "nope.json", "--json-out", out)
    rep = json.loads(out.read_text())
    assert r.returncode == 2 and rep["verdict"] == "ERROR" and rep["error_kind"] == "internal_error", r.stderr
    assert _cli("--ckpt", ok).returncode == 0                                         # the control: the same ckpt PASSES
