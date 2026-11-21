"""PREREG_V8F_TEGOCUT (claim row H-V8F-TEGOCUT-1): T's ego token replaced by a CONSTANT, at training AND inference. ``--t-ego {full,const}`` in ``stack/scripts/v8f_mini_train.py``, ``t_ego`` / ``t_ego_c`` in ``stack/tanitad/models/v8f_mini.py``.

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a lever whose library function is correct and whose CALLER never reaches it. So the lever is proven hop by hop FROM ARGV,
and every proof has a deliberate-regression arm (``raw/2026-10-27-tegocut-train/mutate_tegocut.py`` deletes the single line / flips the default / makes c trainable, on a COPY of the tree, and requires this file to go RED):

  argv --t-ego -> make_config (cfg.t_ego) -> the init checkpoint FILE's e_ego.bias -> model.set_t_ego_const (a frozen BUFFER) -> declared_vs_built (bad_t_ego: G-DVB) -> t_ego_probe (G-LIVE)
       -> V8FMini.run_t's ego slot -> the loss the trainer backpropagates -> the checkpoint (cfg.t_ego + the t_ego_c buffer) -> the eval loaders' strict rebuild

  (A) flag        default ``full``, a FLAG_CHECKS entry, the config field and its validation
  (B) model       the default model IS the saved pre-change model (state dict, levers, loss, gradients, a 20-step trace, forward_infer: bit for bit); const gives T the constant on every row and ONLY T's slot changes;
                  the constant is a buffer (not a Parameter), saved in the state dict, loaded only by a model BUILT const, validated, and refused when absent
  (C) G-DVB       each clause of ``bad_t_ego`` refuses on its own: declared vs built, const needs --init-ckpt, the constant equals the init FILE's e_ego.bias bit for bit, no Parameter added, no combination with the other levers
  (D) G-LIVE      ``t_ego_probe`` passes on both values and records the controls; each clause fires on a broken model made to violate exactly that clause
  (E) the caller  a real ``TR.main`` gives the loss the constant at every step; the checkpoint round-trips through the eval loaders; a resumed run equals the uninterrupted one; K-STREAM: the flag adds no RNG draw,
                  no Parameter and no change of the batches (so the existing ORC-0 is a valid paired control)
  (F) defaults    the default run IS the saved pre-change trainer + model: the 3-step loss trace, the final weights and the other probes' records, bit for bit
Everything that needs the saved pre-change modules is SKIPPED, not failing, in a checkout that has only ``stack/``."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402,F401
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
RAWT = PKG / "raw" / "2026-10-27-tegocut-train"
REF_MODEL_P = RAWT / "pre_tegocut_v8f_mini.py.ref"
REF_TRAIN_P = RAWT / "pre_tegocut_v8f_mini_train.py.ref"
MODEL_P = REPO / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
HERE_T = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_ref(name, path):
    """A module loaded under a private name from a file whose suffix is not .py (the saved reference). Registered in sys.modules BEFORE it executes (a dataclass with ``from __future__ import annotations`` looks its module up)."""
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        loader.exec_module(m)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return m


REF_M = _load_ref("v8f_mini_pre_tegocut", REF_MODEL_P) if REF_MODEL_P.exists() else None
needs_ref = pytest.mark.skipif(REF_M is None or not REF_TRAIN_P.exists(), reason="raw/2026-10-27-tegocut-train (the saved pre-change model + trainer) is not in this checkout")
_P = _load("test_v8f_pgt_for_tegocut", HERE_T / "test_v8f_pgt.py")                  # the batch and the synthetic world the other v8F-mini tests use
world, tiny_model, make_batch = _P.world, _P.tiny_model, _P.make_batch
_T = _P._T                                                                           # test_v8f_mini
COMMON = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]


@pytest.fixture(autouse=True)
def _release_tmp_path(tmp_path):
    """pytest keeps every test's tmp_path until the session ends; on a disk-limited exFAT box (1 MiB clusters) that piles up. Remove a test's OWN tmp_path when it finishes (module-scoped fixtures live elsewhere)."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ helpers
def _no_crash(fn, *a, **k):
    """Call ``fn``; an exception out of the instrument under test is a FAILED assertion (it crashed on an input it must judge), never an error. SystemExit (a refusal) passes through."""
    try:
        return fn(*a, **k)
    except Exception as e:                                                             # noqa: BLE001
        err = f"{getattr(fn, '__name__', fn)} raised {type(e).__name__}: {str(e)[:300]}"
    pytest.fail(err)


def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])


def _J(world):
    return D.load_join(Path(world["root"]) / "join_train.npz")


def _mk(mod, J, seed=3, **over):
    """A tiny model built from module ``mod`` (the real one or the saved reference), weights seeded; ``over`` are config fields."""
    cfg = mod.V8FMiniConfig(**{**_T.TINY, "wm": "off", **over})
    mu, sd = mod.cons_stats(J)
    torch.manual_seed(seed)
    return mod.V8FMini(cfg, mod.P10Prior.synthetic(0), mu, sd)


def _wake(model, seed=11):
    """The residual head is zero-init: at build O's output is P10 whatever it reads, so NO wire could be seen. Give it weights (seeded) so a probe can see a change."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = model.head_o[2].weight
        w.copy_(torch.randn(w.shape, generator=g) * 0.05)
    return model


def _batch(world, n=16, seed=1):
    b, _, _ = make_batch(world, n=n, seed=seed, hmax=10)
    B = b["feats"].shape[0]
    i = torch.arange(B)
    b["lab_t"]["has"] = (i % 3 != 0)
    b["lab_s"]["has"] = (i % 4 != 1)
    g = torch.Generator().manual_seed(123)
    b["lab_s"]["rc_ok"] = torch.rand(B, 3, generator=g) > 0.35
    return b


def _const(world, seed=3, c_seed=7, **over):
    """A tiny model BUILT const, its constant set (random, so it is not any window's live token), and that constant."""
    m = _wake(_mk(M, _J(world), seed=seed, t_ego="const", **over))
    c = torch.randn(m.cfg.d, generator=torch.Generator().manual_seed(c_seed)) * 0.3
    m.set_t_ego_const(c)
    return m, c


def _cond_t(m, b, **kw):
    """The cond that ``read_t`` is handed in a forward of ``m`` on ``b`` (captured at the reader's own input), and the forward's outputs."""
    cap = {}
    f0 = m.read_t.forward

    def fwd(cond, *a, **k):
        cap["cond"] = cond.detach().clone()
        return f0(cond, *a, **k)
    m.read_t.forward = fwd
    try:
        with torch.no_grad():
            out = m._forward(b, 0.0, False, **kw)
    finally:
        m.read_t.__dict__.pop("forward", None)
    return cap["cond"], out


def _gdvb_parts(world, argv, tmp_path=None, build=None, c=None, no_buffer=False, record=True, **over):
    """The launch gate's inputs for a tiny model: the args, the BUILT config / model / optimiser, and the data facts the gate compares (as test_v8f_oacc builds them). ``--t-ego const`` gets an init checkpoint FILE holding
    ``c`` as its ``e_ego.bias`` (``--init-ckpt`` appended to the argv, its init record in ``info``) and a model whose constant is ``c`` (``build`` = the value the MODEL is built with when it should differ from the argv's)."""
    argv = list(argv)
    c = torch.randn(32, generator=torch.Generator().manual_seed(7)) * 0.3 if c is None else c
    decl = argv[argv.index("--t-ego") + 1] if "--t-ego" in argv else "full"
    init_path = None
    if tmp_path is not None and decl == "const":
        init_path = Path(tmp_path) / "init.pt"
        torch.save({"model": {"e_ego.bias": c.clone()}, "cfg": {}}, init_path)
        argv += ["--init-ckpt", str(init_path)]
    a = _args(argv)
    cfg = TR.make_config(a)
    keep = {k: v for k, v in _T.TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over, **({"t_ego": build} if build else {})})
    mu, sd = M.cons_stats(_J(world))
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    if cfg.t_ego == "const" and not no_buffer:
        model.set_t_ego_const(c)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    ws = cfg.window_start_effective
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True, "kin_mode": a.kin_mode, "p10_units": None, "window_start": ws, "window_t_min": ws, "ds_window_start": ws}
    if init_path is not None:
        info["init_ckpt"] = {"path": os.path.abspath(init_path), "sha256": "0" * 64, "applied": True, "skipped": None, "n_tensors": 1, "n_mismatch_after_load": 0,
                             **({"e_ego_bias_sha256": M.tensor_sha256(c)} if record else {})}
    return a, cfg, model, opt, info


# =============================================================================================================================================
# (A) THE FLAG
# =============================================================================================================================================
def test_the_flag_exists_defaults_to_full_and_has_a_declared_vs_built_entry():
    ap = TR.build_parser()
    assert ap.get_default("t_ego") == "full"
    act = next(x for x in ap._actions if x.dest == "t_ego")
    assert tuple(act.choices) == M.T_EGO_MODES == ("full", "const") and act.option_strings == ["--t-ego"]
    assert TR.FLAG_CHECKS["t_ego"] == "lever:t_ego"
    assert {x.dest for x in ap._actions if x.dest != "help"} == set(TR.FLAG_CHECKS), "a flag without an entry, or an entry without a flag, is a launch blocker"
    assert M.V8FMiniConfig().t_ego == "full", "the config default is the pre-change behaviour too"
    a, c = _args(), _args(["--t-ego", "const"])
    assert a.t_ego == "full" and TR.make_config(a).t_ego == "full" and c.t_ego == "const" and TR.make_config(c).t_ego == "const"
    with pytest.raises(ValueError, match="t_ego must be one of"):
        M.V8FMiniConfig(t_ego="zero")
    with pytest.raises(SystemExit):
        _args(["--t-ego", "zero"])
    assert {k: v for k, v in TR.make_config(a).__dict__.items() if k != "t_ego"} == {k: v for k, v in TR.make_config(c).__dict__.items() if k != "t_ego"}, "nothing else moves: the two configs differ in t_ego and in nothing else"


def test_a_flag_without_an_entry_is_refused(world, monkeypatch):
    a, cfg, model, opt, info = _gdvb_parts(world, [])
    assert _no_crash(TR.declared_vs_built, a, cfg, model, opt, info) == []
    monkeypatch.setattr(TR, "FLAG_CHECKS", {k: v for k, v in TR.FLAG_CHECKS.items() if k != "t_ego"})
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any("flags without a declared-vs-built entry" in m and "t_ego" in m for m in bad), bad


# =============================================================================================================================================
# (B) THE MODEL
# =============================================================================================================================================
@needs_ref
def test_the_default_model_is_the_saved_pre_change_model_bit_for_bit(world):
    """State dict, built levers (additions only), loss terms, every gradient, forward_infer and a 20-step AdamW trace: identical to the saved pre-change module, with no Parameter, no buffer and no RNG draw added."""
    J = _J(world)
    cfg = M.V8FMiniConfig(**{**_T.TINY, "wm": "off"})
    kw = {k: v for k, v in cfg.__dict__.items() if k != "t_ego"}
    mu, sd = M.cons_stats(J)
    torch.manual_seed(3)
    new = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    torch.manual_seed(3)
    old = REF_M.V8FMini(REF_M.V8FMiniConfig(**kw), REF_M.P10Prior.synthetic(0), mu, sd)
    sn, so = new.state_dict(), old.state_dict()
    assert sn.keys() == so.keys() and all(torch.equal(sn[k], so[k]) for k in sn), "identical built model, key for key"
    assert new.t_ego_c is None and "t_ego_c" not in sn and "t_ego_c" not in dict(new.named_buffers())
    ln, lo = new.built_levers(), old.built_levers()
    assert set(ln) - set(lo) == {"t_ego", "t_ego_c_sha256"} and set(lo) <= set(ln) and {k: v for k, v in ln.items() if k in lo} == lo and (ln["t_ego"], ln["t_ego_c_sha256"]) == ("full", None)
    assert {f.name for f in dataclasses.fields(M.V8FMiniConfig)} - {f.name for f in dataclasses.fields(REF_M.V8FMiniConfig)} == {"t_ego"}
    assert [(n, tuple(p.shape)) for n, p in new.named_parameters()] == [(n, tuple(p.shape)) for n, p in old.named_parameters()]
    _wake(new), _wake(old)
    b = _batch(world)
    for p in (0.0, 0.3, 1.0):
        torch.manual_seed(5)
        rn = new.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        rng_n = torch.get_rng_state()
        torch.manual_seed(5)
        ro = old.loss(b, p_gt=p, gen=torch.Generator().manual_seed(4))
        assert torch.equal(rng_n, torch.get_rng_state()), "no RNG draw added"
        assert rn["terms"].keys() == ro["terms"].keys() and all(torch.equal(rn["terms"][k], ro["terms"][k]) for k in ro["terms"])
        assert all(torch.equal(rn[k], ro[k]) for k in ro if isinstance(ro[k], torch.Tensor))
        new.zero_grad(), old.zero_grad()
        rn["terms"]["total"].backward(), ro["terms"]["total"].backward()
        assert all(torch.equal(pn.grad, po.grad) for (_, pn), (_, po) in zip(new.named_parameters(), old.named_parameters()) if pn.grad is not None)
    new.eval(), old.eval()
    with torch.no_grad():
        fn_, fo_ = new.forward_infer(b), old.forward_infer(b)
    assert fn_.keys() == fo_.keys() and all(torch.equal(fn_[k], fo_[k]) for k in fo_ if isinstance(fo_[k], torch.Tensor)) and fn_["_keys_read"] == fo_["_keys_read"]
    new.train(), old.train()
    opn = torch.optim.AdamW([p for p in new.parameters() if p.requires_grad], lr=1e-3)
    opo = torch.optim.AdamW([p for p in old.parameters() if p.requires_grad], lr=1e-3)
    for s in range(20):
        ts = []
        for mdl, op in ((new, opn), (old, opo)):
            torch.manual_seed(100 + s)
            op.zero_grad()
            t = mdl.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(s))["terms"]["total"]
            t.backward()
            op.step()
            ts.append(float(t.detach()))
        assert ts[0] == ts[1], f"step {s}"
    sn, so = new.state_dict(), old.state_dict()
    assert all(torch.equal(sn[k], so[k]) for k in so), "the weights after 20 steps are the saved module's exactly"


def test_const_gives_t_the_constant_on_every_row_whatever_its_own_ego_state_is(world):
    m, c = _const(world)
    b = _batch(world)
    cond, _ = _cond_t(m, b)
    assert cond.shape[1] == 4 and torch.equal(cond[:, 0], c[None].expand(cond.shape[0], -1)), "T's ego slot is c on every row, bit for bit"
    for name, p10f in (("donor", b["p10f"].roll(1, 0)), ("zeros", torch.zeros_like(b["p10f"])), ("x100", b["p10f"] * 100.0)):
        cond2, _ = _cond_t(m, {**b, "p10f": p10f})
        assert torch.equal(cond2[:, 0], c[None].expand(cond.shape[0], -1)), name
    f = copy.deepcopy(m)
    f.cfg.t_ego = "full"
    live, _ = _cond_t(f, b)
    assert not torch.equal(live[:, 0], cond[:, 0]) and not (live[:, 0] == c[None]).all(1).any(), "the control: the full token is the live one and is not c on any row"
    B = b["feats"].shape[0]
    adapt, last_t = m._encode(b), torch.full((B,), b["feats"].shape[1] - 1, dtype=torch.long) - b["a_t"]
    ck, ok = m.ck_nom[None].expand(B, -1, -1), torch.ones(B, 3)
    with torch.no_grad():
        t0, t1 = m.run_t(b, adapt, last_t, ck, ok), m.run_t({**b, "p10f": b["p10f"].roll(1, 0)}, adapt, last_t, ck, ok)
        f0, f1 = f.run_t(b, adapt, last_t, ck, ok), f.run_t({**b, "p10f": b["p10f"].roll(1, 0)}, adapt, last_t, ck, ok)
    assert all(torch.equal(t0[k], t1[k]) for k in ("goal_logits", "lat_logits", "lon_logits", "cons_std")), "const: every T output moves by exactly 0.0 under a donor p10f (the checkpoint input held fixed)"
    assert all(float((f0[k] - f1[k]).abs().max()) > 0.0 for k in ("goal_logits", "lat_logits", "lon_logits", "cons_std")), "the control: full moves"


def test_only_ts_ego_slot_changes_s_and_o_and_the_prior_are_untouched(world):
    m, c = _const(world)
    f = copy.deepcopy(m)
    f.cfg.t_ego = "full"
    b = _batch(world)
    caps = {}
    for tag, mm in (("const", m), ("full", f)):
        cap = {}
        for nm in ("read_s", "read_t", "read_o"):
            rd = getattr(mm, nm)

            def fwd(cond, *a, _nm=nm, _f0=rd.forward, **k):
                cap[_nm] = cond.detach().clone()
                return _f0(cond, *a, **k)
            rd.forward = fwd
        with torch.no_grad():
            caps[tag] = (cap, mm._forward(b, 0.0, False))
    (cc, oc), (cf, of) = caps["const"], caps["full"]
    assert torch.equal(cc["read_s"], cf["read_s"]), "S's whole cond (ego, nav, limit) is the full model's"
    assert torch.equal(oc["ckpt"], of["ckpt"]), "S's checkpoints are the full model's"
    assert torch.equal(oc["p10"], of["p10"]), "the P10 prior reads p10f[last_now] and base: untouched"
    assert torch.equal(cc["read_o"][:, 0], cf["read_o"][:, 0]), "O's ego token is its own"
    assert torch.equal(cc["read_t"][:, 1:], cf["read_t"][:, 1:]), "T's nav, limit and checkpoint tokens are the full model's (S's output is unchanged)"
    assert all(float((oc[k] - of[k]).abs().max()) > 0.0 for k in ("goal_logits", "lat_logits", "lon_logits")), "T's outputs differ: the cut changes T"
    assert float((oc["traj60"] - of["traj60"]).abs().max()) > 0.0, "and O's plan moves through T's code (the cut is felt downstream, through the hierarchy)"


def test_e_ego_gets_no_gradient_from_ts_own_terms_under_const_and_still_trains_through_s_and_o(world):
    m, c = _const(world)
    b = _batch(world)
    for mode, want_zero in (("const", True), ("full", False)):
        mm = copy.deepcopy(m)
        mm.cfg.t_ego, mm.cfg.detach_upper = mode, True
        mm.train()
        T = mm.loss(b, p_gt=0.0)["terms"]
        g = torch.autograd.grad(T["goal"] + T["lat"] + T["lon"] + T["cons"], [mm.e_ego.weight, mm.e_ego.bias], allow_unused=True)
        gsum = sum(0.0 if x is None else float(x.abs().sum()) for x in g)
        assert (gsum == 0.0) if want_zero else (gsum > 0.0), (mode, gsum)
        mm2 = copy.deepcopy(m)
        mm2.cfg.t_ego = mode
        mm2.train()
        mm2.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(3))["terms"]["total"].backward()
        assert float(mm2.e_ego.weight.grad.abs().sum()) > 0.0 and float(mm2.e_ego.bias.grad.abs().sum()) > 0.0, f"{mode}: e_ego still trains through S and O"


def test_the_constant_is_a_buffer_in_the_state_dict_and_a_default_model_has_none(world):
    J = _J(world)
    f = _mk(M, J)
    assert f.t_ego_c is None and "t_ego_c" not in f.state_dict() and "t_ego_c" not in dict(f.named_buffers())
    m, c = _const(world)
    sd = m.state_dict()
    assert torch.equal(sd["t_ego_c"], c) and sd["t_ego_c"].dtype == torch.float32 and "t_ego_c" in dict(m.named_buffers())
    assert not m.t_ego_c.requires_grad and all(p is not m.t_ego_c for p in m.parameters()) and "t_ego_c" not in dict(m.named_parameters())
    assert [(n, tuple(p.shape)) for n, p in m.named_parameters()] == [(n, tuple(p.shape)) for n, p in f.named_parameters()], "the flag adds no Parameter"
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == sum(p.numel() for p in f.parameters() if p.requires_grad)
    assert copy.deepcopy(m).t_ego_c is not m.t_ego_c and torch.equal(copy.deepcopy(m).t_ego_c, c)
    lv = m.built_levers()
    assert lv["t_ego"] == "const" and lv["t_ego_c_sha256"] == M.tensor_sha256(c) == hashlib.sha256(c.numpy().tobytes()).hexdigest()
    assert lv["n_params_trainable"] == f.built_levers()["n_params_trainable"]
    assert f.built_levers()["t_ego"] == "full" and f.built_levers()["t_ego_c_sha256"] is None


def test_c_is_the_token_the_model_itself_emits_for_the_mean_kinematic_state(world):
    """The PREREG's identity: c = the init checkpoint's e_ego.bias = e_ego(standardise(xm)) EXACTLY, because standardise(xm) = (xm - xm) / xs = 0. So c is in distribution at init (a token M2b emitted)."""
    J = _J(world)
    rng = torch.Generator().manual_seed(5)
    xm, xs = torch.randn(8, generator=rng) * 3.0, torch.rand(8, generator=rng) + 0.5
    cfg = M.V8FMiniConfig(**{**_T.TINY, "wm": "off"})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(3)
    m = M.V8FMini(cfg, M.P10Prior(xm, xs, torch.zeros(2 * M.K30)), mu, sd)
    z = m.prior.standardise(m.prior.xm[None])
    assert float(z.abs().max()) == 0.0, "standardise(xm) is exactly 0"
    with torch.no_grad():
        m.e_ego.weight.copy_(torch.randn(m.e_ego.weight.shape, generator=rng))
        m.e_ego.bias.copy_(torch.randn(m.e_ego.bias.shape, generator=rng))
        assert torch.equal(m._ego(m.prior.xm[None]), m.e_ego.bias[None]), "e_ego(standardise(xm)) IS e_ego.bias, bit for bit"


def test_set_t_ego_const_validates_and_only_a_model_built_const_takes_one(world):
    m, c = _const(world)
    for bad, msg in ((torch.zeros(5), "shape"), (torch.zeros(1, 32), "shape"), (torch.full((32,), float("nan")), "finite"), (torch.full((32,), float("inf")), "finite")):
        with pytest.raises(ValueError, match=msg):
            m.set_t_ego_const(bad)
        assert torch.equal(m.t_ego_c, c), "a refused constant leaves the model unchanged"
    m.set_t_ego_const(c.double())
    assert m.t_ego_c.dtype == torch.float32 and torch.equal(m.t_ego_c, c)
    with pytest.raises(ValueError, match="needs a model built with t_ego='const'"):
        _mk(M, _J(world)).set_t_ego_const(c)
    assert m.to(torch.float64).t_ego_c is not None, "the buffer follows the module"


def test_a_const_model_without_the_buffer_refuses_at_the_forward_and_never_falls_back_to_the_live_token(world):
    m = _wake(_mk(M, _J(world), t_ego="const"))
    b = _batch(world)
    with pytest.raises(RuntimeError, match="t_ego_c buffer is absent"):
        m.forward_infer(b)
    with pytest.raises(RuntimeError, match="t_ego_c buffer is absent"):
        m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(1))


def test_state_dict_round_trip_and_who_may_load_the_constant(world):
    J = _J(world)
    m, c = _const(world)
    fresh = _mk(M, J, t_ego="const")
    assert fresh.t_ego_c is None
    fresh.load_state_dict(m.state_dict())
    assert torch.equal(fresh.t_ego_c, c) and fresh.built_levers()["t_ego_c_sha256"] == m.built_levers()["t_ego_c_sha256"], "a model BUILT const takes the constant a state dict carries (the eval loaders' route)"
    full = _mk(M, J)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        full.load_state_dict(m.state_dict())
    assert full.t_ego_c is None, "a full model REFUSES a state dict that carries a constant: a checkpoint whose config lost t_ego is never evaluated with the live token"
    nb = _mk(M, J, t_ego="const")
    nb.load_state_dict(full.state_dict())                              # the init checkpoint (M2b's) carries no constant: the strict load passes ...
    assert nb.t_ego_c is None
    with pytest.raises(RuntimeError, match="t_ego_c buffer is absent"):
        nb.forward_infer(_batch(world))                                # ... and the model refuses at the forward
    for bad, msg in ((torch.zeros(5), "shape"), (torch.full((32,), float("nan")), "finite")):
        sd = m.state_dict()
        sd["t_ego_c"] = bad
        with pytest.raises(ValueError, match=msg):
            _mk(M, J, t_ego="const").load_state_dict(sd)


def test_the_flag_adds_no_rng_draw(world):
    """K-STREAM at the model: the RNG state after a teacher-forced loss is identical for full and const (the same seed, the same weights): the number and order of draws do not depend on the flag."""
    m, c = _const(world)
    f = copy.deepcopy(m)
    f.cfg.t_ego = "full"
    b = _batch(world)
    states = []
    for mm in (f, m):
        torch.manual_seed(9)
        mm.loss(b, p_gt=0.5, gen=None)
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])
    g = [torch.Generator().manual_seed(4) for _ in range(2)]
    f.loss(b, p_gt=0.5, gen=g[0]), m.loss(b, p_gt=0.5, gen=g[1])
    assert torch.equal(g[0].get_state(), g[1].get_state())


def test_a_bf16_autocast_forward_runs_and_the_slot_is_still_c_exactly(world):
    m, c = _const(world)
    b = _batch(world)
    cap = {}
    f0 = m.read_t.forward

    def fwd(cond, *a, **k):
        cap["cond"] = cond.detach().clone()
        return f0(cond, *a, **k)
    m.read_t.forward = fwd
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(1))
    assert torch.isfinite(out["terms"]["total"]) and torch.equal(cap["cond"][:, 0].float(), c[None].expand(cap["cond"].shape[0], -1)), "torch.stack promotes: c is not rounded to bf16"


# =============================================================================================================================================
# (C) G-DVB: bad_t_ego
# =============================================================================================================================================
@pytest.mark.parametrize("argv", [[], ["--t-ego", "full"], ["--t-ego", "const"]])
def test_g_dvb_passes_what_was_built(world, tmp_path, argv):
    a, cfg, model, opt, info = _gdvb_parts(world, argv, tmp_path)
    assert _no_crash(TR.declared_vs_built, a, cfg, model, opt, info) == [], argv
    assert model.built_levers()["t_ego"] == a.t_ego == cfg.t_ego


@pytest.mark.parametrize("kw, what", [
    (dict(argv=["--t-ego", "const"], build="full"), "lever t_ego: declared 'const' but built 'full'"),
    (dict(argv=["--t-ego", "full"], build="const"), "lever t_ego: declared 'full' but built 'const'"),
    (dict(argv=["--t-ego", "const"], no_buffer=True), "no t_ego_c buffer was built"),
    (dict(argv=["--t-ego", "const"], record=False), "the init record carries no e_ego_bias_sha256"),
])
def test_g_dvb_refuses_what_was_not_built(world, tmp_path, kw, what):
    a, cfg, model, opt, info = _gdvb_parts(world, tmp_path=tmp_path, **kw)
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any(what in m for m in bad), bad


def test_g_dvb_refuses_const_without_an_init_checkpoint(world):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], None)
    assert a.t_ego == "const" and not a.init_ckpt
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any("--t-ego const needs --init-ckpt" in m for m in bad), bad


def test_g_dvb_refuses_a_constant_that_is_not_the_init_files_e_ego_bias_bit_for_bit(world, tmp_path):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    assert _no_crash(TR.declared_vs_built, a, cfg, model, opt, info) == []
    c = model.t_ego_c.clone()
    c[3] = torch.nextafter(c[3], torch.tensor(float("inf")))             # ONE ulp off
    model.set_t_ego_const(c)
    bad = _no_crash(TR.declared_vs_built, a, cfg, model, opt, info)
    assert any("not bit-identical to the init checkpoint file's e_ego.bias" in m for m in bad) and any("!= the init checkpoint's e_ego.bias sha256" in m for m in bad), bad
    a2, cfg2, model2, opt2, info2 = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    info2["init_ckpt"]["e_ego_bias_sha256"] = "0" * 64                    # the record disagrees with the built constant, the file agrees
    bad = _no_crash(TR.declared_vs_built, a2, cfg2, model2, opt2, info2)
    assert any("!= the init checkpoint's e_ego.bias sha256" in m for m in bad) and not any("not bit-identical" in m for m in bad), bad
    a3, cfg3, model3, opt3, info3 = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    torch.save({"model": {"e_ego.bias": model3.t_ego_c + 1.0}, "cfg": {}}, a3.init_ckpt)      # the FILE moved under a model and a record built from the old one
    bad = _no_crash(TR.declared_vs_built, a3, cfg3, model3, opt3, info3)
    assert any("not bit-identical to the init checkpoint file's e_ego.bias" in m for m in bad) and not any("!= the init checkpoint's e_ego.bias sha256" in m for m in bad), "the gate re-reads the FILE, not only the record"
    torch.save({"model": {}, "cfg": {}}, a3.init_ckpt)
    bad = _no_crash(TR.declared_vs_built, a3, cfg3, model3, opt3, info3)
    assert any("cannot be re-read for t_ego_c" in m for m in bad), bad


def test_g_dvb_refuses_a_buffer_that_is_not_a_finite_float32_d_vector(world, tmp_path):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    for bad_t in (model.t_ego_c.double(), model.t_ego_c[:5].clone(), torch.full((32,), float("nan"))):
        model.t_ego_c = bad_t                                             # the buffer slot, bypassing set_t_ego_const's own validation
        bad = _no_crash(TR.bad_t_ego, a, model, info)
        assert any("must be a finite float32" in m for m in bad), (bad_t.dtype, tuple(bad_t.shape), bad)


def test_g_dvb_refuses_a_constant_on_a_full_arm_and_a_trainable_constant_and_a_changed_parameter_list(world, tmp_path):
    a, cfg, model, opt, info = _gdvb_parts(world, [], tmp_path)
    model.register_buffer("t_ego_c", torch.zeros(32))
    bad = _no_crash(TR.bad_t_ego, a, model, info)
    assert any("--t-ego full is declared but the model carries a t_ego_c buffer" in m for m in bad) and any("a constant is recorded" in m for m in bad), bad
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    c = model.t_ego_c.clone()
    del model._buffers["t_ego_c"]
    model.t_ego_c = nn.Parameter(c)                                       # the mutant: c is a Parameter
    bad = _no_crash(TR.bad_t_ego, a, model, info)
    assert any("t_ego_c is trainable" in m for m in bad), bad
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    model.extra_t_bias = nn.Parameter(torch.zeros(32))                    # a flag that added a Parameter
    bad = _no_crash(TR.bad_t_ego, a, model, info)
    assert any("changed the model's parameter list" in m for m in bad), bad


@pytest.mark.parametrize("flag, val", [("ego_perturb_p", 0.3), ("dagger_p", 0.1)])
def test_g_dvb_refuses_const_combined_with_another_lever(world, tmp_path, flag, val):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    assert _no_crash(TR.bad_t_ego, a, model, info) == []
    setattr(a, flag, val)
    bad = _no_crash(TR.bad_t_ego, a, model, info)
    assert any("the levers are never combined in one arm" in m for m in bad), bad
    a2, _, model2, _, info2 = _gdvb_parts(world, [], tmp_path)
    setattr(a2, flag, val)
    assert _no_crash(TR.bad_t_ego, a2, model2, info2) == [], "the combination is refused for const only: a full arm is not this lever's business"


def test_the_gate_does_not_move_the_rng_stream(world, tmp_path):
    a, cfg, model, opt, info = _gdvb_parts(world, ["--t-ego", "const"], tmp_path)
    torch.manual_seed(4)
    s0 = torch.get_rng_state()
    _no_crash(TR.bad_t_ego, a, model, info)
    assert torch.equal(s0, torch.get_rng_state()), "the twin built to compare parameter lists draws from a forked RNG"


def test_read_init_t_ego_c_reads_the_file_as_a_float32_copy(tmp_path):
    c = torch.randn(32)
    torch.save({"model": {"e_ego.bias": c.double(), "x": torch.zeros(1)}, "cfg": {}}, tmp_path / "a.pt")
    got = TR.read_init_t_ego_c(tmp_path / "a.pt")
    assert got.dtype == torch.float32 and torch.equal(got, c.double().float())
    torch.save({"model": {"x": torch.zeros(1)}}, tmp_path / "b.pt")
    with pytest.raises(ValueError, match="e_ego.bias"):
        TR.read_init_t_ego_c(tmp_path / "b.pt")


# =============================================================================================================================================
# (D) G-LIVE: t_ego_probe
# =============================================================================================================================================
def test_the_probe_passes_on_both_values_and_records_the_controls(world):
    b = _batch(world)
    m, c = _const(world)
    r = _no_crash(TR.t_ego_probe, m, b, m.cfg)
    assert r["pass"] is True and r["t_ego"] == "const" and set(r["branches"]) == {"const", "full"}
    k, f = r["branches"]["const"], r["branches"]["full"]
    assert k["a_token_is_c_every_row"] and not k["a_token_is_the_live_token"] and f["a_token_is_the_live_token"] and not f["a_token_is_c_every_row"]
    assert all(v == 0.0 for v in k["b_max_abs_dT_under_donor_p10f"].values()) and any(v > 0.0 for v in f["b_max_abs_dT_under_donor_p10f"].values())
    assert k["c_grad_T_terms_wrt_e_ego"] == 0.0 and f["c_grad_T_terms_wrt_e_ego"] > 0.0 and k["c_grad_T_terms_wrt_read_t"] > 0.0
    assert all(v > 0.0 for v in k["d_grad_by_group"].values()) and all(v > 0.0 for v in f["d_grad_by_group"].values())
    assert k["e_max_abs_change_of_c_after_3_steps"] == 0.0 and k["e_is_a_buffer_not_a_parameter"] and k["h_s_token_is_its_own"] and k["h_o_token_is_its_own"]
    assert k["g_determinism_max_abs_diff"] == 0.0 and k["g_record_identical"] and f["g_determinism_max_abs_diff"] == 0.0
    assert r["i_effect_max_abs_dT_const_vs_full"] > 0.0 and k["f_residual_dT_donor_p10f"] > 0.0, "back door 2 (S's checkpoint token) is open by design, and its size is recorded"
    assert "p10f" in r["keys_read"] and set(r["keys_read"]) <= set(M.INFERENCE_INPUT_KEYS)
    full = _wake(_mk(M, _J(world)))
    r2 = _no_crash(TR.t_ego_probe, full, b, full.cfg)
    assert r2["pass"] is True and r2["t_ego"] == "full" and r2["branches"]["const"]["a_token_is_c_every_row"] and r2["branches"]["full"]["a_token_is_the_live_token"], "a default launch runs BOTH branches too"
    assert all(v == 0.0 for v in r2["branches"]["const"]["b_max_abs_dT_under_donor_p10f"].values())


def test_the_probe_does_not_change_the_model_or_the_global_rng(world):
    m, c = _const(world)
    b = _batch(world)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    torch.manual_seed(3)
    s0 = torch.get_rng_state()
    TR.t_ego_probe(m, b, m.cfg)
    assert torch.equal(s0, torch.get_rng_state()) and all(torch.equal(before[k], v) for k, v in m.state_dict().items()) and m.cfg.t_ego == "const" and not m.cfg.detach_upper


def test_a_single_window_or_a_batch_that_cannot_tell_a_donor_is_skipped_in_a_full_run_and_refused_in_a_const_run(world):
    b = _batch(world)
    one = {k: (v[:1] if isinstance(v, torch.Tensor) else v) for k, v in b.items() if not isinstance(v, dict)}
    one["lab"], one["lab_t"], one["lab_s"] = ({k: v[:1] for k, v in b[n].items()} for n in ("lab", "lab_t", "lab_s"))
    f = _wake(_mk(M, _J(world)))
    r = TR.t_ego_probe(f, one, f.cfg)
    assert r["pass"] is True and "single window" in r["skipped"]
    m, _ = _const(world)
    with pytest.raises(AssertionError, match="single window"):
        TR.t_ego_probe(m, one, m.cfg)
    same = {**b, "p10f": b["p10f"][:1].expand_as(b["p10f"]).clone()}      # every window has the same ego history: a donor cannot be told from the window's own
    r = TR.t_ego_probe(f, same, f.cfg)
    assert r["pass"] is True and "cannot discriminate" in r["skipped"]
    with pytest.raises(AssertionError, match="cannot tell a donor"):
        TR.t_ego_probe(m, same, m.cfg)


def _probe_fires(world, match):
    m, _ = _const(world)
    b = _batch(world)
    with pytest.raises(AssertionError, match=match):
        TR.t_ego_probe(m, b, m.cfg)


def test_the_probe_fires_when_the_token_is_not_the_constant(world, monkeypatch):
    monkeypatch.setattr(M.V8FMini, "_t_ego_const", lambda self, B: (self.t_ego_c * 2.0)[None].expand(B, -1))
    _probe_fires(world, "not the constant on every row")


def test_the_probe_fires_when_the_live_token_still_reaches_the_slot(world, monkeypatch):
    m, _ = _const(world)
    b = _batch(world)
    assert TR.t_ego_probe(m, b, m.cfg)["pass"], "(the control: the unmutated model passes)"
    orig = M.V8FMini.run_t

    def live_slot(self, b_, adapt, last_t, ck_t, ck_ok):
        sv = self.cfg.t_ego
        self.cfg.t_ego = "full"                                           # M1: the const line deleted -- T reads the live token
        try:
            return orig(self, b_, adapt, last_t, ck_t, ck_ok)
        finally:
            self.cfg.t_ego = sv
    monkeypatch.setattr(M.V8FMini, "run_t", live_slot)
    with pytest.raises(AssertionError, match="not the constant on every row"):
        TR.t_ego_probe(m, b, m.cfg)


def test_the_probe_fires_when_t_depends_on_its_own_p10f_by_another_route(world, monkeypatch):
    orig = M.V8FMini.run_t

    def run_t(self, b, adapt, last_t, ck_t, ck_ok):
        out = orig(self, b, adapt, last_t, ck_t, ck_ok)
        out["goal_logits"] = out["goal_logits"] + self._at(b["p10f"], last_t)[:, :1] * 0.1      # the slot is c; T still reads p10f
        return out
    monkeypatch.setattr(M.V8FMini, "run_t", run_t)
    _probe_fires(world, "moved by .* under a donor p10f")


def test_the_probe_fires_when_the_gradient_of_ts_terms_still_reaches_e_ego(world, monkeypatch):
    orig = M.V8FMini.run_t

    def run_t(self, b, adapt, last_t, ck_t, ck_ok):
        out = orig(self, b, adapt, last_t, ck_t, ck_ok)
        live = self._ego(self._at(b["p10f"], last_t))
        out["goal_logits"] = out["goal_logits"] + (live - live.detach()).sum()      # forward: exactly + 0.0 (T's outputs and the token are unchanged); backward: a gradient into e_ego
        return out
    monkeypatch.setattr(M.V8FMini, "run_t", run_t)
    _probe_fires(world, r"gradient of T's loss terms w.r.t. e_ego is .*, not EXACTLY 0.0")


def test_the_probe_fires_when_the_full_branch_is_not_the_control_it_must_be(world, monkeypatch):
    orig = M.V8FMini.run_t

    def always_const(self, b, adapt, last_t, ck_t, ck_ok):
        sv = self.cfg.t_ego
        self.cfg.t_ego = "const"
        try:
            return orig(self, b, adapt, last_t, ck_t, ck_ok)
        finally:
            self.cfg.t_ego = sv
    monkeypatch.setattr(M.V8FMini, "run_t", always_const)
    _probe_fires(world, "the full branch .the control. does not read T's own token")


def test_the_probe_fires_when_the_full_control_cannot_move_or_has_no_gradient(world, monkeypatch):
    orig_fwd = M.Reader.forward

    def deaf(self, cond, scene, pad):
        if cond.shape[1] == 4:                                             # T's reader: its ego slot is zeroed (the probe captured the cond before this)
            cond = torch.cat([torch.zeros_like(cond[:, :1]), cond[:, 1:]], 1)
        return orig_fwd(self, cond, scene, pad)
    monkeypatch.setattr(M.Reader, "forward", deaf)
    _probe_fires(world, "the full branch .the control. did not move under a donor p10f")
    monkeypatch.setattr(M.Reader, "forward", orig_fwd)
    orig = M.V8FMini.run_t
    ego0 = M.V8FMini._ego

    def detached(self, b, adapt, last_t, ck_t, ck_ok):
        self._ego = lambda f8: ego0(self, f8).detach()                    # T's live token carries no gradient
        try:
            return orig(self, b, adapt, last_t, ck_t, ck_ok)
        finally:
            self.__dict__.pop("_ego", None)
    monkeypatch.setattr(M.V8FMini, "run_t", detached)
    _probe_fires(world, "the full branch .the control. has a zero gradient")


def test_the_probe_fires_when_a_group_the_cut_must_leave_trainable_is_dead(world, monkeypatch):
    orig = M.V8FMini.run_t

    def run_t(self, b, adapt, last_t, ck_t, ck_ok):
        out = orig(self, b, adapt, last_t, ck_t, ck_ok)
        out["goal_logits"] = out["goal_logits"].detach()
        out["goal_prob"] = torch.sigmoid(out["goal_logits"])
        return out
    monkeypatch.setattr(M.V8FMini, "run_t", run_t)
    _probe_fires(world, r"no gradient on \['h_goal'\]")


def test_the_probe_fires_when_s_or_o_lost_their_own_ego_token(world, monkeypatch):
    orig_s = M.V8FMini.run_s

    def run_s(self, b, adapt, last_s, age_rel=None):
        return orig_s(self, {**b, "p10f": b["p10f"].roll(1, 0)}, adapt, last_s, age_rel)       # S reads another window's ego state
    monkeypatch.setattr(M.V8FMini, "run_s", run_s)
    _probe_fires(world, r"S's / O's ego token is not its own live token \(S: False")


def test_the_probe_fires_when_it_is_not_deterministic(world, monkeypatch):
    orig = M.V8FMini.run_t

    def run_t(self, b, adapt, last_t, ck_t, ck_ok):
        out = orig(self, b, adapt, last_t, ck_t, ck_ok)
        out["goal_logits"] = out["goal_logits"] + torch.rand(1) * 1e-3
        return out
    monkeypatch.setattr(M.V8FMini, "run_t", run_t)
    _probe_fires(world, "the probe is not deterministic")


def test_the_probe_fires_when_the_constant_is_a_parameter(world):
    m, c = _const(world)
    del m._buffers["t_ego_c"]
    m.t_ego_c = nn.Parameter(c.clone())
    b = _batch(world)
    with pytest.raises(AssertionError, match="Parameter / trainable / not a registered buffer"):
        TR.t_ego_probe(m, b, m.cfg)


def test_the_probe_fires_when_the_constant_moves_under_the_optimiser(world, monkeypatch):
    orig = M.V8FMini.loss

    def loss(self, batch, p_gt=None, gen=None):
        out = orig(self, batch, p_gt, gen)
        if self.t_ego_c is not None and self.training:
            self.t_ego_c.add_(1e-3)                                       # a buffer that something else writes to: the bytes move
        return out
    monkeypatch.setattr(M.V8FMini, "loss", loss)
    _probe_fires(world, r"t_ego_c moved by .* under three optimiser steps")


def test_the_probe_fires_when_the_branches_read_different_keys_or_a_label(world, monkeypatch):
    orig = M.V8FMini.run_t

    def run_t(self, b, adapt, last_t, ck_t, ck_ok):
        if self.cfg.t_ego == "const":
            b["gt"]                                                       # a label read at inference, in the const branch only
        return orig(self, b, adapt, last_t, ck_t, ck_ok)
    monkeypatch.setattr(M.V8FMini, "run_t", run_t)
    _probe_fires(world, "the keys forward_infer reads differ between the branches or leave the admissible set")


# =============================================================================================================================================
# (E) THE CALLER: a real TR.main
# =============================================================================================================================================
class Pack:
    pass


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    """One trainer-written checkpoint (the init: the role M2b's checkpoint plays) on the trainer's own synthetic world, and its e_ego.bias."""
    root = tmp_path_factory.mktemp("tegc")
    k = Pack()
    k.root, k.init_out = root, root / "init"
    assert TR.main(["--out", str(k.init_out), *COMMON, "--steps", "2"]) == 0
    k.ck = k.init_out / "ckpt_last.pt"
    k.bias = torch.load(k.ck, map_location="cpu", weights_only=False)["model"]["e_ego.bias"].clone()
    k.sha_bias = M.tensor_sha256(k.bias)
    return k


def _run(out, extra=(), common=COMMON):
    return ["--out", str(out), *common, *extra]


def CONST(pk):
    return ["--init-ckpt", str(pk.ck), "--t-ego", "const"]


def _summary(out):
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def _selfcheck(out):
    return json.loads((Path(out) / "selfcheck.json").read_text(encoding="utf-8"))


def _log(out):
    return [json.loads(x) for x in (Path(out) / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def _ck(out):
    return torch.load(Path(out) / "ckpt_last.pt", map_location="cpu", weights_only=False)


def test_a_default_run_is_full_records_no_constant_and_still_runs_the_probe(pack, tmp_path):
    assert TR.main(_run(tmp_path / "d", ["--steps", "4"])) == 0
    s, sc, ck = _summary(tmp_path / "d"), _selfcheck(tmp_path / "d"), _ck(tmp_path / "d")
    assert s["levers"]["t_ego"] == "full" and s["levers"]["t_ego_c_sha256"] is None and ck["cfg"]["t_ego"] == "full" and "t_ego_c" not in ck["model"]
    assert sc["t_ego_probe"]["pass"] is True and sc["declared_vs_built"] == "PASS" and sc["levers"]["t_ego"] == "full"
    assert sc["t_ego_probe"]["branches"]["const"]["a_token_is_c_every_row"] and sc["t_ego_probe"]["branches"]["full"]["a_token_is_the_live_token"]
    assert TR.main(_run(tmp_path / "di", ["--steps", "2", "--init-ckpt", str(pack.ck)])) == 0
    assert "e_ego_bias_sha256" not in _selfcheck(tmp_path / "di")["info"]["init_ckpt"], "a full arm's init record is the pre-change record"


def test_flag_to_config_to_built_model_to_the_records_and_the_checkpoint(pack, tmp_path):
    n = 12
    assert TR.main(_run(tmp_path / "c", ["--steps", str(n), *CONST(pack)])) == 0
    assert TR.main(_run(tmp_path / "f", ["--steps", str(n), "--init-ckpt", str(pack.ck)])) == 0
    s, sc, ck, fk = _summary(tmp_path / "c"), _selfcheck(tmp_path / "c"), _ck(tmp_path / "c"), _ck(tmp_path / "f")
    assert s["levers"]["t_ego"] == "const" and s["levers"]["t_ego_c_sha256"] == pack.sha_bias == sc["levers"]["t_ego_c_sha256"] == sc["info"]["init_ckpt"]["e_ego_bias_sha256"], "summary.json levers read const and the init file's sha256"
    assert ck["cfg"]["t_ego"] == "const" and torch.equal(ck["model"]["t_ego_c"], pack.bias), f"the final checkpoint's t_ego_c is c after {n} steps: a change of EXACTLY 0.0"
    assert not torch.equal(ck["model"]["e_ego.bias"], pack.bias), "e_ego.bias is not c: S and O trained it, which proves the run trained at all"
    assert s["levers"]["n_params_trainable"] == _summary(tmp_path / "f")["levers"]["n_params_trainable"], "the flag adds no trainable parameter"
    assert "t_ego_c" not in fk["model"] and fk["model"].keys() | {"t_ego_c"} == ck["model"].keys()
    og, fg = ck["opt"]["param_groups"], fk["opt"]["param_groups"]
    assert [len(g["params"]) for g in og] == [len(g["params"]) for g in fg] and ck["opt"]["state"].keys() == fk["opt"]["state"].keys(), "AdamW's state layout is the full arm's"
    assert all(ck["opt"]["state"][i]["exp_avg"].shape == fk["opt"]["state"][i]["exp_avg"].shape for i in fk["opt"]["state"])
    assert not all(torch.equal(ck["model"][k], fk["model"][k]) for k in fk["model"]), "the control: the flag does change the training"
    assert sc["t_ego_probe"]["pass"] is True and sc["declared_vs_built"] == "PASS"
    assert s["steps"] == n and all(abs(r["total"]) < 1e6 for r in _log(tmp_path / "c")), "it trained"


@pytest.fixture
def spy_t_slot(monkeypatch):
    """Record, at every call of the model's ``loss`` that comes from the TRAINING LOOP (gen None and p_gt not 0: the probes pass a generator or p_gt 0.0), T's ego slot as ``read_t`` is handed it."""
    seen = []
    orig = M.V8FMini.loss

    def loss(self, batch, p_gt=None, gen=None):
        if gen is None and p_gt not in (None, 0.0) and self.training:
            f0 = self.read_t.forward

            def fwd(cond, *a, **k):
                seen.append(cond[:, 0].detach().clone())
                return f0(cond, *a, **k)
            self.read_t.forward = fwd
            try:
                return orig(self, batch, p_gt, gen)
            finally:
                self.read_t.__dict__.pop("forward", None)
        return orig(self, batch, p_gt, gen)
    monkeypatch.setattr(M.V8FMini, "loss", loss)
    return seen


def test_the_loss_the_trainer_backpropagates_gets_c_in_ts_ego_slot_at_every_step(pack, tmp_path, spy_t_slot, monkeypatch):
    n = 6
    assert TR.main(_run(tmp_path / "c", ["--steps", str(n), *CONST(pack)])) == 0
    assert len(spy_t_slot) == n and all(torch.equal(x, pack.bias[None].expand(x.shape[0], -1)) for x in spy_t_slot), "every training step: T's ego slot is c on every row"
    spy_t_slot.clear()
    assert TR.main(_run(tmp_path / "f", ["--steps", str(n), "--init-ckpt", str(pack.ck)])) == 0
    assert len(spy_t_slot) == n and not any(torch.equal(x, pack.bias[None].expand(x.shape[0], -1)) for x in spy_t_slot), "the control: the same spy on a full run reads the live token, so it can see the difference"
    assert all(float(x.std(0).max()) > 0.0 for x in spy_t_slot), "and the live token differs between rows"


def test_the_gate_refuses_a_model_whose_t_slot_is_not_c_and_writes_nothing(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(M.V8FMini, "_t_ego_const", lambda self, B: (self.t_ego_c * 2.0)[None].expand(B, -1))
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ t-ego: t_ego probe: const is declared but the ego slot"):
        TR.main(_run(tmp_path / "bad", ["--steps", "2", *CONST(pack)]))
    for f in ("selfcheck.json", "argv.json", "ckpt_last.pt", "summary.json", "train_log.jsonl"):
        assert not (tmp_path / "bad" / f).exists(), f"a refused launch writes no {f}"


@pytest.mark.parametrize("extra, msg", [
    (["--t-ego", "const"], "ZZDVB-REFUSEDZZ declared-vs-built: .*--t-ego const needs --init-ckpt"),
    (["--t-ego", "const", "--init-ckpt", "{ck}", "--ego-perturb-p", "0.3"], "ZZDVB-REFUSEDZZ declared-vs-built: .*never combined in one arm"),
])
def test_a_refused_declaration_stops_the_real_trainer_before_it_writes_anything(pack, tmp_path, extra, msg):
    extra = [x.replace("{ck}", str(pack.ck)) for x in extra]
    with pytest.raises(SystemExit, match=msg):
        TR.main(_run(tmp_path / "bad", ["--steps", "2", *extra]))
    for f in ("selfcheck.json", "argv.json", "ckpt_last.pt", "summary.json", "train_log.jsonl"):
        assert not (tmp_path / "bad" / f).exists(), f"a refused launch writes no {f}"


def test_an_init_checkpoint_without_an_e_ego_bias_is_refused_with_const(pack, tmp_path):
    blob = torch.load(pack.ck, map_location="cpu", weights_only=False)
    del blob["model"]["e_ego.bias"]
    torch.save(blob, tmp_path / "nobias.pt")
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ init-ckpt .*--t-ego const needs the init checkpoint's e_ego.bias"):
        TR.main(_run(tmp_path / "bad", ["--steps", "2", "--init-ckpt", str(tmp_path / "nobias.pt"), "--t-ego", "const"]))
    assert not (tmp_path / "bad" / "selfcheck.json").exists()


def test_a_resumed_const_run_is_the_uninterrupted_run_and_takes_c_from_the_file_not_the_live_model(pack, tmp_path, monkeypatch):
    """On a resume the init checkpoint is NOT applied (the run's own ckpt_last.pt wins), so the live model's e_ego.bias is a random initialisation at gate time: a constant taken from it would be refused by the gate."""
    common = [*COMMON, "--steps", "5", "--ckpt-every", "1", *CONST(pack)]
    assert TR.main(["--out", str(tmp_path / "full"), *common]) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "3")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(tmp_path / "res"), *common])
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(["--out", str(tmp_path / "res"), *common]) == 0
    a, b = _ck(tmp_path / "full"), _ck(tmp_path / "res")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "the final weights are those of the uninterrupted run"
    assert torch.equal(b["model"]["t_ego_c"], pack.bias)
    la, lb = {r["step"]: r["total"] for r in _log(tmp_path / "full")}, {r["step"]: r["total"] for r in _log(tmp_path / "res")}
    assert la == lb, "and so is every logged loss"
    sc = _selfcheck(tmp_path / "res")
    assert sc["info"]["init_ckpt"]["applied"] is False and sc["info"]["init_ckpt"]["skipped"] == "resume"
    assert sc["levers"]["t_ego_c_sha256"] == pack.sha_bias == sc["info"]["init_ckpt"]["e_ego_bias_sha256"], "the resumed gate holds the file's constant, though the live e_ego.bias is not it"


def test_the_eval_loaders_rebuild_the_cut_model_from_the_checkpoint_alone(pack, tmp_path, monkeypatch):
    """G-EVAL / K-CUT-EVAL. Both eval tools run the same three lines (anchored below): the config from ck["cfg"], a model built with a zero prior, a STRICT load of ck["model"]."""
    seen = {}
    orig_dvb, orig_probe = TR.declared_vs_built, TR.t_ego_probe

    def dvb(a, cfg, model, *r, **k):
        seen["model"] = model
        return orig_dvb(a, cfg, model, *r, **k)

    def probe(model, batch, cfg):
        seen["batch"] = batch
        return orig_probe(model, batch, cfg)
    monkeypatch.setattr(TR, "declared_vs_built", dvb)
    monkeypatch.setattr(TR, "t_ego_probe", probe)
    assert TR.main(_run(tmp_path / "c", ["--steps", "6", *CONST(pack)])) == 0
    ck = _ck(tmp_path / "c")

    def load(cfg_dict, sd):
        cfg = M.V8FMiniConfig(**cfg_dict)
        model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        model.load_state_dict(sd)
        return model.eval()
    ev = load(ck["cfg"], ck["model"])
    assert ev.built_levers()["t_ego"] == "const" and torch.equal(ev.t_ego_c, ck["model"]["t_ego_c"]) and torch.equal(ev.t_ego_c, pack.bias) and ev.built_levers()["t_ego_c_sha256"] == pack.sha_bias
    tm = seen["model"].cpu().eval()
    b = seen["batch"]
    with torch.no_grad():
        a_, b_ = tm.forward_infer(b), ev.forward_infer(b)
    assert a_.keys() == b_.keys() and all(torch.equal(a_[k], b_[k]) for k in a_ if isinstance(a_[k], torch.Tensor)), "forward_infer of the eval-loaded model is bit-identical to the trainer's model"
    r = TR.t_ego_probe(ev, b, ev.cfg)
    assert all(v == 0.0 for v in r["branches"]["const"]["b_max_abs_dT_under_donor_p10f"].values()) and r["branches"]["const"]["a_token_is_c_every_row"], "K-CUT-EVAL: clause (b) reads EXACTLY 0.0 through the eval loader on the final checkpoint"
    init = torch.load(pack.ck, map_location="cpu", weights_only=False)
    fm = load(init["cfg"], init["model"])
    rf = TR.t_ego_probe(fm, b, fm.cfg)
    assert fm.built_levers()["t_ego"] == "full" and any(v > 0.0 for v in rf["branches"]["full"]["b_max_abs_dT_under_donor_p10f"].values()), "the same probe on the full (ORC-0-like) checkpoint reads > 0"
    lost = {k: v for k, v in ck["cfg"].items() if k != "t_ego"}                 # M5: the saved config loses t_ego
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load(lost, ck["model"])
    for f, need in (("v8f_mini_eval.py", 'cfg = M.V8FMiniConfig(**ck["cfg"])'), ("v8f_t1_eval.py", 'cfg = M.V8FMiniConfig(**ck["cfg"])')):
        src = (REPO / "stack" / "scripts" / f).read_text(encoding="utf-8")
        assert src.count(need) == 1 and src.count('model.load_state_dict(ck["model"])') == 1 and "strict=False" not in src, f"{f}: the eval loader needs no edit and loads strictly"


def _stream_run(out, extra, monkeypatch, patch=None):
    """K-STREAM: per call of the model's ``loss``: the sha256 of the batch's (ci, t, a_t, a_s), p_gt, the sha256 of every ``torch.rand`` draw (the two teacher-forcing masks) and the RNG state after the call."""
    log, state = [], {"in": False, "calls": []}
    orig_loss, orig_rand = M.V8FMini.loss, torch.rand

    def rand(*a, **k):
        r = orig_rand(*a, **k)
        if state["in"]:
            state["calls"].append((tuple(r.shape), hashlib.sha256(r.numpy().tobytes()).hexdigest()))
        return r

    def loss(self, batch, p_gt=None, gen=None):
        h = hashlib.sha256()
        for k in ("ci", "t", "a_t", "a_s"):
            h.update(batch[k].numpy().tobytes())
        state["in"], state["calls"] = True, []
        try:
            res = orig_loss(self, batch, p_gt, gen)
        finally:
            state["in"] = False
        log.append((h.hexdigest(), p_gt, gen is None, tuple(state["calls"]), hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
        return res
    with monkeypatch.context() as mp:
        mp.setattr(torch, "rand", rand)
        mp.setattr(M.V8FMini, "loss", loss)
        if patch is not None:
            patch(mp)
        assert TR.main(_run(out, extra)) == 0
    return log


def test_k_stream_the_flag_adds_no_rng_draw_no_parameter_and_no_change_of_the_batches(pack, tmp_path, monkeypatch):
    """The property PREREG_V8F_TEGOCUT section 3 relies on to reuse ORC-0 as the paired control: 20 steps of `--t-ego full` and `--t-ego const` from the same init see the same batches (ci, t, a_t, a_s) and the same two
    teacher-forcing masks at every step, and leave the RNG in the same state."""
    lf = _stream_run(tmp_path / "f", ["--steps", "20", "--init-ckpt", str(pack.ck)], monkeypatch)
    lc = _stream_run(tmp_path / "c", ["--steps", "20", *CONST(pack)], monkeypatch)
    train = lambda lg: [x for x in lg if x[2] and x[1] not in (None, 0.0)]      # noqa: E731  the training loop's calls (the probes pass a generator or p_gt 0.0)
    assert len(train(lf)) == len(train(lc)) == 20 and all(len(x[3]) == 2 for x in train(lc)), "20 steps, two teacher-forcing draws each (S's and T's code)"
    assert lf == lc, "every call of the loss, probes included, has the same batch hash, the same masks and the same RNG state after it"
    assert len({x[0] for x in train(lc)}) == 20, "and the batches differ from step to step (the hash can tell one batch from another)"
    assert _ck(tmp_path / "c")["opt"]["state"].keys() == _ck(tmp_path / "f")["opt"]["state"].keys()
    # the deliberate regression: ONE extra draw in the const branch (mutant M6) changes the stream, and this instrument SEES it
    orig = M.V8FMini._t_ego_const

    def drawing(self, B):
        torch.rand(1)
        return orig(self, B)
    ld = _stream_run(tmp_path / "d", ["--steps", "20", *CONST(pack)], monkeypatch, patch=lambda mp: mp.setattr(M.V8FMini, "_t_ego_const", drawing))
    assert ld != lc and [x[0] for x in train(ld)] == [x[0] for x in train(lc)], "the batches are the same and the stream is not: the extra draw is visible"


# =============================================================================================================================================
# (F) THE DEFAULT RUN IS THE PRE-CHANGE RUN
# =============================================================================================================================================
CHILD = r'''
import sys, json, importlib.machinery, importlib.util
repo, mode, out, extra, ref_train, ref_model = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), sys.argv[5], sys.argv[6]
for p in (repo + "/stack", repo + "/taniteval", repo + "/stack/scripts"):
    sys.path.insert(0, p)
import torch
torch.set_num_threads(1)
def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    loader.exec_module(m)
    return m
if mode == "ref":
    import tanitad.models
    tanitad.models.v8f_mini = load("tanitad.models.v8f_mini", ref_model)       # the saved pre-change MODEL, under the name the trainer imports
    TR = load("v8f_mini_train", ref_train)                                      # and the saved pre-change TRAINER on top of it
else:
    import v8f_mini_train as TR
sys.exit(TR.main(["--out", out] + extra))
'''


def _child(mode, out, extra):
    r = subprocess.run([sys.executable, "-c", CHILD, str(REPO), mode, str(out), json.dumps(extra), str(REF_TRAIN_P), str(REF_MODEL_P)], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, (mode, r.stdout[-1500:], r.stderr[-2500:])


@needs_ref
def test_defaults_reproduce_the_pre_change_trainer_and_model_bit_for_bit(tmp_path):
    """The control that must read exactly equal: the saved pre-change trainer + model vs this tree's, the same argv (no --t-ego), 3 steps: every logged number, every logged KEY, every final weight, the optimiser state and the
    other probes' records."""
    extra = ["--steps", "3", *COMMON]
    _child("ref", tmp_path / "ref", extra)
    _child("new", tmp_path / "new", extra)
    l_ref, l_new = _log(tmp_path / "ref"), _log(tmp_path / "new")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "sec"} for r in rows]       # noqa: E731
    assert len(l_ref) == 3 and strip(l_ref) == strip(l_new), "the loss trace AND the set of logged keys are identical (json round-trips floats exactly)"
    a, b = _ck(tmp_path / "ref"), _ck(tmp_path / "new")
    assert a["model"].keys() == b["model"].keys() and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "the final weights are bit-identical, key for key (no t_ego_c)"
    assert all(torch.equal(a["opt"]["state"][i]["exp_avg"], b["opt"]["state"][i]["exp_avg"]) for i in a["opt"]["state"]), "and so is the optimiser state"
    assert {k: v for k, v in b["cfg"].items() if k != "t_ego"} == a["cfg"] and b["cfg"]["t_ego"] == "full", "the only new config key is t_ego"
    rs, ns = _summary(tmp_path / "ref"), _summary(tmp_path / "new")
    assert set(ns) == set(rs), "no new summary key"
    assert {k: v for k, v in ns["levers"].items() if k not in ("t_ego", "t_ego_c_sha256")} == rs["levers"] and ns["levers"]["t_ego"] == "full" and ns["levers"]["t_ego_c_sha256"] is None
    sr, sn = _selfcheck(tmp_path / "ref"), _selfcheck(tmp_path / "new")
    assert set(sn) - set(sr) == {"t_ego_probe"} and set(sr) <= set(sn), "the only new selfcheck record is t_ego_probe"
    for k in ("declared_vs_built", "live", "code_mix_probe", "o_interface_probe", "acc_probe", "admissibility", "ego_perturb_probe", "dagger_probe", "steps_total"):
        assert sn[k] == sr[k], f"{k}: every other probe's record is unchanged"
    assert sn["t_ego_probe"]["pass"] is True


@needs_ref
def test_the_saved_references_are_the_tip_trainer_and_model_with_nothing_of_the_lever_in_them():
    s, m = REF_TRAIN_P.read_text(encoding="utf-8"), REF_MODEL_P.read_text(encoding="utf-8")
    assert "t_ego" not in s and "--t-ego" not in s and "t_ego" not in m and "tegocut" not in s.lower() and "TEGOCUT" not in m
    assert "dagger" in s and "ego_perturb" in s and "init_ckpt" in s, "it is the tree the lever was added to (the DAgger build's trainer), not an older one"
    assert "o_interface" in m and "hacc_w" in m


# =============================================================================================================================================
# the mutation anchors
# =============================================================================================================================================
def test_the_single_applying_lines_exist_exactly_once():
    """``raw/2026-10-27-tegocut-train/mutate_tegocut.py`` deletes these; an anchor that is not found exactly once would make a mutation test vacuous."""
    mdl, trn = MODEL_P.read_text(encoding="utf-8"), TRAIN_P.read_text(encoding="utf-8")
    assert mdl.count('        ego_tok = self._t_ego_const(i.shape[0]) if cfg.t_ego == "const" else self._ego(self._at(b["p10f"], i)) ') == 1
    assert mdl.count('        cond = torch.stack([ego_tok, self.e_nav(self._at(b["nav"], i)),') == 1, "T's cond is built from the swapped token, once"
    assert mdl.count('        self.register_buffer("t_ego_c", t)') == 1
    assert mdl.count('    t_ego: str = "full" ') == 1
    assert trn.count('    bad += bad_t_ego(a, model, data_info) ') == 1
    assert trn.count("        tegop = t_ego_probe(model, probe, cfg) ") == 1
    assert trn.count("model.set_t_ego_const(read_init_t_ego_c(a.init_ckpt))") == 1
    assert trn.count('ap.add_argument("--t-ego", dest="t_ego", choices=M.T_EGO_MODES, default="full",') == 1
    assert trn.count('    sc["t_ego_probe"] = tegop') == 1
