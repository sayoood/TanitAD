"""PREREG_T2_DAGGER (DT2): the T2 own-state bank swap, hop by hop -- argv -> make_t2bank -> bad_t2bank (G-DVB) -> the swap -> the batch the LOSS receives -> the masked terms (exactly 0.0) and the target law (non-zero),
with a p = 0 control and deliberate-regression arms (unmask a label, drop the aux gate). Synthetic world only (the repo's own generator); nothing here reads the real corpus."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_recover as R  # noqa: E402
from tanitad.data import v8f_t2bank as TB  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_mini import make_batch, tiny_model, world  # noqa: E402,F401

torch.set_num_threads(1)
TRAIN_SRC = (REPO / "stack" / "scripts" / "v8f_mini_train.py").read_text(encoding="utf-8")
MASKED = ("ckpt", "goal", "lat", "lon", "cons", "acc", "wm", "pen_SPEED", "pen_STOP", "pen_TURN", "pen_LEAD", "pens_SPEED", "pens_STOP", "pens_TURN", "pens_LEAD")


def make_bank(world, n=6, seed=5, ids=None):
    b, _, _ = make_batch(world, n=n, seed=seed)
    g = torch.Generator().manual_seed(1)
    a = {k: b[k].clone() for k in TB.INPUT_KEYS}
    a["gt_logged"] = b["gt"].double().clone()
    a["dy"] = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5)
    a["dpsi"] = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 0.1
    ids = ids or [f"scene-{i % 3}" for i in range(n)]
    return TB.T2Bank(a, [{"scene_id": s, "fold": "A", "seed": 0, "f0": i, "tick": 0} for i, s in enumerate(ids)])


def swap_of(bank, p=1.0, **kw):
    bank.sha256 = bank.sha256 or "a" * 64
    return TB.T2BankSwap(p=p, bank=bank, bank_path="x/bank.json", bank_sha256=bank.sha256, seed=3, **kw)


def terms(model, batch):
    model.zero_grad(set_to_none=True)
    return model.loss(batch)["terms"]


def test_p0_is_the_same_batch_and_the_unchanged_loss_path(world):
    b, J, _ = make_batch(world, n=8)
    sw = TB.T2BankSwap(p=0.0)
    assert sw.apply(b, 0) is b and not sw.active
    model, _ = tiny_model(J)
    t0, t1 = terms(model, b), terms(model, sw.apply(b, 0))
    assert all(torch.equal(t0[k], t1[k]) for k in t0)                     # p = 0: the very same batch object, hence the very same (aux-free) loss path
    t2 = terms(model, {**b, "aux_ok": torch.ones(8, dtype=torch.bool)})  # the gate with every row on is the same loss (the gating arithmetic)
    for k in t0:
        assert torch.allclose(t0[k], t2[k], atol=1e-5, rtol=1e-5), k


def test_bank_rows_carry_only_the_target_law(world):
    b, J, _ = make_batch(world, n=8)
    model, cfg = tiny_model(J, w_acc=3.0)
    bank = make_bank(world)
    out = swap_of(bank).apply(b, 7)
    assert out is not b and int((~out["aux_ok"]).sum()) == 8
    T = terms(model, out)
    assert {k: float(T[k]) for k in MASKED if k in T and float(T[k]) != 0.0} == {}
    assert float(T["ade"]) > 0.0 and float(T["fde"]) > 0.0
    T["total"].backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0.0
    # the row's target is RECOVER-FT's law, applied to the drawn bank row
    idx = swap_of(bank).apply(b, 7) and None
    sw = swap_of(bank)
    out = sw.apply(b, 7)
    want = R.rejoin_target(bank.a["gt_logged"][sw.last["bank_idx"]].double(), bank.a["dy"][sw.last["bank_idx"]].double(), bank.a["dpsi"][sw.last["bank_idx"]].double(), mode="rejoin", steps=10).float()
    assert torch.allclose(out["gt"], want, atol=1e-6) and bool(out["gt_ok"].all())


def test_masked_terms_have_exactly_zero_gradient_on_bank_rows_and_logged_rows_keep_theirs(world):
    b, J, _ = make_batch(world, n=8)
    model, _ = tiny_model(J, w_acc=3.0)
    g0 = torch.Generator().manual_seed(0)
    with torch.no_grad():                                                     # the zero-init residual head makes every input gradient 0.0 at init: move the weights so the controls can read non-zero
        for q in model.parameters():
            q.add_(0.05 * torch.randn(q.shape, generator=g0))
    sw = TB.T2BankSwap(p=0.5, bank=(bk := make_bank(world)), bank_path="x/bank.json", bank_sha256="a" * 64, seed=11)
    bk.sha256 = "a" * 64
    out = sw.apply(b, 2)
    sel = sw.last["sel"]
    assert 0 < int(sel.sum()) < 8
    feats = out["feats"].float().requires_grad_(True)
    for k in ("ckpt", "goal", "acc", "pen_SPEED", "pens_TURN"):
        model.zero_grad(set_to_none=True)
        feats.grad = None
        T = model.loss({**out, "feats": feats})["terms"]
        T[k].backward(retain_graph=False)
        g = feats.grad.abs().reshape(8, -1).sum(1)
        assert float(g[sel].sum()) == 0.0, f"{k}: a masked term has a gradient on a bank row"
    model.zero_grad(set_to_none=True)
    feats.grad = None
    model.loss({**out, "feats": feats})["terms"]["ade"].backward()                  # the target-law term reaches the bank rows
    assert float(feats.grad.abs().reshape(8, -1).sum(1)[sel].sum()) > 0.0


def test_mutation_unmask_a_label_goes_red(world, monkeypatch):
    b, J, _ = make_batch(world, n=8)
    model, _ = tiny_model(J)
    bank = make_bank(world)

    def goal_after(masks):
        monkeypatch.setattr(TB, "MASKED_LABELS", masks)
        return float(terms(model, swap_of(bank).apply(b, 7))["goal"])
    full = TB.MASKED_LABELS
    assert goal_after(full) == 0.0
    assert goal_after(tuple(x for x in full if x != ("lab_t", "has"))) != 0.0      # delete the one mask line: the tactical term comes back


def test_mutation_drop_the_aux_gate_goes_red(world):
    b, J, _ = make_batch(world, n=8)
    model, _ = tiny_model(J, w_acc=3.0)
    out = swap_of(make_bank(world)).apply(b, 7)
    assert float(terms(model, out)["acc"]) == 0.0
    ungated = {k: v for k, v in out.items() if k != "aux_ok"}
    assert float(terms(model, ungated)["acc"]) != 0.0 or float(terms(model, ungated)["pen_SPEED"]) != 0.0


def test_bank_roundtrip_identity_and_tamper_is_refused(world, tmp_path):
    bank = make_bank(world, ids=["s1", "s2", "s1", "s3", "s2", "s1"])
    sha = bank.save(tmp_path / "bk")
    back = TB.T2Bank.load(tmp_path / "bk", expect_sha256=sha)
    assert back.scene_ids == ["s1", "s2", "s3"] and all(torch.equal(back.a[k], bank.a[k]) for k in TB.ROW_KEYS)
    with pytest.raises(TB.BankRefused):
        TB.T2Bank.load(tmp_path / "bk", expect_sha256="0" * 64)
    raw = bytearray((tmp_path / "bk" / TB.SHARD).read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (tmp_path / "bk" / TB.SHARD).write_bytes(bytes(raw))
    with pytest.raises(TB.BankRefused):
        TB.T2Bank.load(tmp_path / "bk", expect_sha256=sha)


def test_opened_scene_ids_are_recorded(world):
    b, _, _ = make_batch(world, n=8)
    bank = make_bank(world, ids=["s1", "s1", "s2", "s2", "s3", "s3"])
    sw = swap_of(bank)
    sw.apply(b, 1)
    rec = sw.opened_record()
    assert rec["scene_ids"] and set(rec["scene_ids"]) <= {"s1", "s2", "s3"} and sum(rec["draws"].values()) == 8


def test_flags_reach_the_built_swap_and_the_gate_reads_them_back(world, tmp_path):
    bank = make_bank(world)
    sha = bank.save(tmp_path / "bk")
    base = ["--t2-bank-p", "0.3", "--t2-bank", str(tmp_path / "bk"), "--t2-bank-sha256", sha, "--t2-bank-seed", "4"]
    a = TR.build_parser().parse_args(base + ["--out", "o"])
    assert (a.t2_bank_p, a.t2_bank_seed) == (0.3, 4)
    info = {}
    t2b = TR.make_t2bank(a, info)
    assert t2b.active and t2b.built_levers()["t2_bank_sha256"] == sha and info["t2bank"]["n_rows"] == 6
    assert TR.bad_t2bank(a, t2b) == []
    with pytest.raises(SystemExit):                                              # a declared sha256 that is not the bank's
        TR.make_t2bank(TR.build_parser().parse_args(base[:-2] + ["--t2-bank-sha256", "0" * 64, "--out", "o"]), {})
    a2 = TR.build_parser().parse_args(base + ["--out", "o", "--recover-p", "0.3", "--recover-dev-law", "apriori"])
    assert any("exclusive" in m for m in TR.bad_t2bank(a2, TR.make_t2bank(a2, {})))
    a3 = TR.build_parser().parse_args(base + ["--out", "o", "--wm-only"])                                        # MRATE's WM-only mode runs no planner loss: the bank lever is refused with it
    assert any("--wm-only" in m for m in TR.bad_t2bank(a3, TR.make_t2bank(a3, {})))
    a0 = TR.build_parser().parse_args(["--out", "o", "--t2-bank-p", "0", "--t2-bank", str(tmp_path / "bk")])                # a flag at p = 0 does nothing: refused
    assert TR.bad_t2bank(a0, TR.make_t2bank(a0, {}))
    assert TR.bad_t2bank(TR.build_parser().parse_args(["--out", "o"]), TR.make_t2bank(TR.build_parser().parse_args(["--out", "o"]), {})) == []   # the default argv: nothing built, nothing declared


def test_reachability_from_the_caller_source():
    assert all(k in TR.FLAG_CHECKS for k in ("t2_bank_p", "t2_bank", "t2_bank_sha256", "t2_bank_seed"))
    assert "bad += bad_t2bank(a, t2b, data_info)" in TRAIN_SRC and "bad = declared_vs_built(a, cfg, model, opt, info, enc, ego, lr_at, dag, comp, rcv, t2b)" in TRAIN_SRC
    assert re.search(r"^\s+b = t2b\.apply\(b, step\)$", TRAIN_SRC, re.M) and "ZZT2BANK-NOT-APPLIEDZZ" in TRAIN_SRC and "t2bank_probe(model, probe, cfg, t2b)" in TRAIN_SRC
    assert TR.build_parser().get_default("t2_bank_p") == 0.0                          # the default disables it: an arm that tests it must NAME --t2-bank-p


# ------------------------------------------------------------------------------------------------ the WM auxiliary: bank rows are not in its computation (the test the first DT2 agent could not run: the tiny config built no WM)
def _wm_world(world, n=8):
    from test_v8f_mini import TINY, _args, M
    lv = TR.declared_levers(_args(["--arm", "M2b"]))
    assert lv["wm"] != "off"
    b, J, _ = make_batch(world, n=n, hmax=16)
    cfg = M.V8FMiniConfig(**{**TINY, "hmax": 16, "navonly": lv["navonly"], "consistency": lv["consistency"], "wm": lv["wm"]})
    mu, sd = M.cons_stats(J)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd), b, J


def _sel_batch(world, model, b, J, p):
    g = torch.Generator().manual_seed(1)
    n = int(b["feats"].shape[0])
    a = {k: b[k].clone() for k in TB.INPUT_KEYS}
    a["gt_logged"], a["dy"], a["dpsi"] = b["gt"].double().clone(), torch.rand(n, generator=g, dtype=torch.float64) - 0.5, (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 0.1
    bank = TB.T2Bank(a, [{"scene_id": f"scene-{i % 3}", "fold": "A", "seed": 0, "f0": i, "tick": 0} for i in range(n)])
    out = swap_of(bank, p=p).apply(b, 7)
    return out


def test_wm_term_reads_only_the_rows_that_carry_it_and_the_gate_is_what_does_it(world):
    model, b, J = _wm_world(world)
    out = _sel_batch(world, model, b, J, p=0.5)
    aux = out["aux_ok"]
    assert 0 < int(aux.sum()) < int(aux.numel()), "need a mixed batch (some bank rows, some logged rows)"

    def wm_at(batch):                                                                          # the WM stack draws random numbers: every read is under the SAME seed
        torch.manual_seed(1234)
        return float(terms(model, batch)["wm"])
    base = wm_at(out)
    assert base != 0.0 and np.isfinite(base)
    assert wm_at(out) == base                                                                  # CONTROL: the same batch under the same seed reads the same value, so a difference below is the batch's
    bank_rows, log_rows = (~aux).nonzero(as_tuple=True)[0], aux.nonzero(as_tuple=True)[0]

    def with_act(rows, d):
        o = dict(out)
        a = out["act"].clone()
        a[rows] = a[rows] + d
        o["act"] = a
        return o
    assert wm_at(with_act(bank_rows, 1.7)) == base                                             # a bank row's act is not in the WM computation: exactly equal
    assert wm_at(with_act(log_rows, 1.7)) != base                                              # a logged row's act is: the control that proves the probe can read a change
    no_gate = {k: v for k, v in with_act(bank_rows, 1.7).items() if k != "aux_ok"}
    assert wm_at(no_gate) != base                                                              # MUTATION: delete the aux_ok gate and the bank row leaks into the WM term


def test_wm_term_on_an_all_bank_batch_is_exactly_zero_with_no_wm_gradient(world):
    model, b, J = _wm_world(world)
    out = _sel_batch(world, model, b, J, p=1.0)
    assert int(out["aux_ok"].sum()) == 0
    model.zero_grad(set_to_none=True)
    T = model.loss(out)["terms"]
    assert float(T["wm"]) == 0.0
    T["total"].backward()
    wm_g = [p.grad for n_, p in model.named_parameters() if n_.startswith("wm.") and p.grad is not None]
    assert all(float(g.abs().sum()) == 0.0 for g in wm_g)                                     # the WM parameters receive nothing (no grad, or exactly 0)
