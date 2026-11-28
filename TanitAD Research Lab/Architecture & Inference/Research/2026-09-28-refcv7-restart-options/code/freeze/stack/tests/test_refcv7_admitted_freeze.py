"""⛔⛔ refcv7 RESTART FREEZE -- G-LIVE's ten ADMITTED dead groups, frozen and declared model-side.

At launch (fec3a0d) the launch gate ADMITTED ten leaf groups by FLAG
(`launch_gate.PROFILES["refcv7"]["live_dead_admitted"]`) and owed the model-side declaration at the
next restart, because it touches code the binding runs recorded. MEASURED dead twice: G-LIVE's launch
smoke (10 of 474 leaf groups) and the live run's step-1,500 checkpoint (the ONLY 20 of 808 optimiser
entries with no AdamW state). The freeze uses the batch-3 mechanism
(`tanitad.models._gradreach.declare_grad_unreachable`): kept BUILT for strict loads, out of the
optimiser, declared with a reason, held against ARGV by G-DVB (`declared_vs_built`).

Pinned here, every expectation a LITERAL:
1. the two independently written tables (model side keyed on the CORE config, argv side keyed on
   flags) and the gate's admission list are the SAME literal ten, with the literal flag map;
2. on the refcv7 rig (the eval-loader test's rig: every refcv7 block, captured from the REAL
   train() before any data is read) the build declares the literal THIRTEEN (the three batch-3
   ones + the ten), freezes exactly their 26 tensors (20 of them the ten's), and G-DVB agrees;
3. RED: the rule table without `--no-strategic` / without `--graft-tac8-prior` names exactly the
   8 / 2 declarations argv no longer justifies; an UNDECLARED build is named on all ten;
4. the declaration is MEASURED TRUE for the eight `--no-strategic` modules on the trainer's own loss
   (smoke rig, `probe_grad_unreachable`: re-opened for one backward, `.grad is None` on all);
5. the freeze changes NO VALUE and consumes NO RNG: with the flags OFF the build is untouched, and
   with them ON its state dict equals an undeclared build's bit for bit;
6. the trainer's `build_optimizer` (the dd recipe) excludes EXACTLY the twenty tensors;
7. the resume converter (`scripts/refcv7_ckpt_freeze_convert.py`): a pre-freeze optimiser state
   loads into the frozen optimiser after conversion and NOT before (the loud refusal is pinned),
   and a dropped entry that CARRIES state is REFUSED.
CPU; the rig is the eval-loader test's (resnet18, 128 x 576, sentinel data paths).
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

STACK = Path(__file__).resolve().parents[1]
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import refc_v3_train as T                                          # noqa: E402
from tanitad.models import _gradreach as gr                        # noqa: E402
from tanitad.refs import refc_v3 as v3                             # noqa: E402
from tanitad.train import declared_vs_built as dvb                 # noqa: E402

#: ⛔ THE ten, as G-LIVE measured them -- a LITERAL, never read from a table under test
TEN_STRATEGIC = ("core.strategic.gru", "core.strategic.proj", "nav_to_str", "str_goal_head",
                 "gstr_embed", "gstr_film", "core.decoder.ctx_to_cond", "core.route_head")
TEN_TAC8 = ("core.decoder.lat_to_anchor", "core.decoder.lon_to_anchor")
TEN = set(TEN_STRATEGIC) | set(TEN_TAC8)
BATCH3 = {"core.decoder.control_head", "core.decoder.offset_head", "scorer.goal_point"}
#: the ten's tensors: GRU 4, proj 2, nav_to_str 2, str_goal_head 2, gstr_embed 2, gstr_film 2,
#: ctx_to_cond 2, route_head 2, lat/lon_to_anchor 1 each (bias=False)
N_TEN_TENSORS = 20
F1_F6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
         "--f5-emitting-conf", "--f6-w-u0-zero"]
R7_CORE = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
           "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
           "--residual-prior", "ha0_ext_pose"]


def _under(n: str, prefixes) -> bool:
    return any(n == p or n.startswith(p + ".") for p in prefixes)


@pytest.fixture(scope="module")
def rig():
    """The refcv7 rig: every refcv7 build block, the REAL train() captured at the model build."""
    import test_refcv7_eval_loader as R
    d = Path(tempfile.mkdtemp(prefix="r7freeze_"))
    R._synth_inputs(d)
    argv = R._argv(d)
    cap = LG.run_trainer_until(T, argv, "model")
    return cap["model"], cap["args"], argv


A_LON = (-2.0, -1.0, 0.0, 1.0, 2.0)
A_LAT = (-1.5, -0.5, 0.0, 0.5)


def _smoke(argv, seed: int = 0):
    """test_grad_unreachable_declared.py's smoke build, with its real control ladder (F9)."""
    args = T.build_parser().parse_args(list(argv) + ["--out", "z"])
    cfg = T._pin_trainer_cfg(v3.refc_v3_smoke_config(args.arm == "hier"), args)
    torch.manual_seed(seed)
    model = v3.RefCV3Model(cfg)
    dec = model.core.decoder
    if bool(getattr(dec, "anchor_v0_cond", False)):
        with torch.no_grad():
            grid = torch.cartesian_prod(torch.tensor(A_LON), torch.tensor(A_LAT))
            dec.anchor_controls.copy_(grid.to(dec.anchor_controls.dtype))
    return args, cfg, model


def _smoke_batch(cfg):
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    ds.ego_history = cfg.core.ego_history is not None
    return torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])


def _loss_backward(model, batch, seed: int = 1):
    model.train()
    torch.manual_seed(seed)
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    losses["loss"].backward()
    return losses


# ---- 1. the three tables are the SAME literal ten --------------------------------------------- #
def test_1_model_table_argv_table_and_gate_admissions_are_the_literal_ten():
    model_tab = {p: f for p, f, _w in v3.BYPASS_DECLARATIONS}
    argv_tab = {p: f for p, f, _w in dvb.GRAD_UNREACHABLE_BYPASS_RULES}
    gate_tab = {g: f for g, f, _w in LG.PROFILES["refcv7"]["live_dead_admitted"]}
    assert set(model_tab) == set(argv_tab) == set(gate_tab) == TEN
    assert model_tab == {**{p: "no_strategic" for p in TEN_STRATEGIC},
                         **{p: "graft_tac8_prior" for p in TEN_TAC8}}
    assert argv_tab == gate_tab == {**{p: "--no-strategic" for p in TEN_STRATEGIC},
                                    **{p: "--graft-tac8-prior" for p in TEN_TAC8}}
    assert all(len(w) > 20 for _p, _f, w in v3.BYPASS_DECLARATIONS)      # each carries a reason
    assert {p for p, _r, _w in dvb.GRAD_UNREACHABLE_RULES} == BATCH3    # batch 3 is untouched


# ---- 2. the refcv7 rig declares the literal thirteen ------------------------------------------ #
def test_2_the_refcv7_rig_declares_the_LITERAL_thirteen_and_G_DVB_agrees(rig):
    model, args, _argv = rig
    assert set(gr.grad_unreachable_prefixes(model)) == BATCH3 | TEN
    assert set(model._bypass_declared) == TEN
    frozen = sorted(n for n, p in model.named_parameters() if not p.requires_grad)
    assert frozen == sorted(n for n, _p in model.named_parameters() if _under(n, BATCH3 | TEN))
    assert len([n for n in frozen if _under(n, TEN)]) == N_TEN_TENSORS
    assert dvb.check_grad_unreachable(model, args) == []
    assert set(dvb.expected_grad_unreachable(args)) == BATCH3 | TEN


# ---- 3. RED arms ------------------------------------------------------------------------------ #
def test_3a_RED_without_no_strategic_argv_no_longer_justifies_the_eight(rig):
    import argparse
    model, args, _argv = rig
    off = argparse.Namespace(**{**vars(args), "no_strategic": False})
    got = sorted((m.lever, m.declared, m.built) for m in dvb.check_grad_unreachable(model, off))
    assert got == sorted((f"grad_unreachable[{p}]", "not declared (argv: reached)", "declared")
                         for p in TEN_STRATEGIC)


def test_3b_RED_without_graft_tac8_prior_argv_no_longer_justifies_the_two(rig):
    import argparse
    model, args, _argv = rig
    off = argparse.Namespace(**{**vars(args), "graft_tac8_prior": False})
    got = sorted((m.lever, m.declared, m.built) for m in dvb.check_grad_unreachable(model, off))
    assert got == sorted((f"grad_unreachable[{p}]", "not declared (argv: reached)", "declared")
                         for p in TEN_TAC8)


def test_3c_RED_an_UNDECLARED_build_is_named_on_all_ten(rig, monkeypatch):
    _model, _args, argv = rig
    monkeypatch.setattr(v3, "declare_bypassed_by_flag", lambda model: {})
    cap = LG.run_trainer_until(T, argv, "model")
    got = sorted((m.lever, m.built) for m in dvb.check_grad_unreachable(cap["model"], cap["args"]))
    assert got == sorted((f"grad_unreachable[{p}]", "not declared") for p in TEN)


# ---- 4. MEASURED on the trainer's own loss (the eight --no-strategic modules) ----------------- #
def test_4_the_no_strategic_declarations_are_MEASURED_true():
    args, cfg, model = _smoke(R7_CORE + F1_F6 + ["--no-strategic"])
    assert set(model._bypass_declared) == set(TEN_STRATEGIC)        # no tac8 graft on this rig
    assert dvb.check_grad_unreachable(model, args) == []
    batch = _smoke_batch(cfg)
    got = dvb.probe_grad_unreachable(model, lambda: _loss_backward(model, batch))
    assert got == [], [str(m) for m in got]


# ---- 5. no value changed, no RNG consumed ----------------------------------------------------- #
def test_5a_flags_OFF_the_build_is_untouched():
    _a, _c, model = _smoke(R7_CORE + F1_F6)
    assert model._bypass_declared == {}
    assert set(gr.grad_unreachable_prefixes(model)) == BATCH3


def test_5b_flags_ON_the_state_dict_equals_an_undeclared_build_BIT_FOR_BIT(monkeypatch):
    argv = R7_CORE + F1_F6 + ["--no-strategic"]
    _a1, _c1, frozen = _smoke(argv, seed=3)
    monkeypatch.setattr(v3, "declare_bypassed_by_flag", lambda model: {})
    _a2, _c2, plain = _smoke(argv, seed=3)
    s1, s2 = frozen.state_dict(), plain.state_dict()
    assert list(s1) == list(s2)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


# ---- 6. the optimiser ------------------------------------------------------------------------- #
def test_6_build_optimizer_excludes_EXACTLY_the_twenty(rig):
    model, args, _argv = rig
    import argparse
    a = argparse.Namespace(**{**vars(args), "opt": "dd"})
    opt = T.build_optimizer(model, a)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    out = sorted(n for n, p in model.named_parameters() if id(p) not in in_opt)
    ten_out = [n for n in out if _under(n, TEN)]
    assert len(ten_out) == N_TEN_TENSORS
    assert all(_under(n, TEN | BATCH3) for n in out)
    assert [g["name"] for g in opt.param_groups] == ["encoder", "head"]


# ---- 7. the resume converter ------------------------------------------------------------------ #
def _converter():
    spec = importlib.util.spec_from_file_location(
        "refcv7_ckpt_freeze_convert", str(STACK / "scripts" / "refcv7_ckpt_freeze_convert.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pre_freeze_opt_state(monkeypatch, argv):
    """A pre-freeze build (the declaration removed), two REAL steps of the dd optimiser."""
    import argparse
    with monkeypatch.context() as m:
        m.setattr(v3, "declare_bypassed_by_flag", lambda model: {})
        args, cfg, pre = _smoke(argv, seed=0)
    a = argparse.Namespace(**{**vars(args), "opt": "dd"})
    opt = T.build_optimizer(pre, a)
    batch = _smoke_batch(cfg)
    for s in range(2):
        opt.zero_grad(set_to_none=True)
        _loss_backward(pre, batch, seed=10 + s)
        opt.step()
    return a, pre, opt.state_dict()


def test_7a_a_pre_freeze_state_loads_ONLY_after_conversion(monkeypatch):
    C = _converter()
    argv = R7_CORE + F1_F6 + ["--no-strategic"]
    a, pre, sd = _pre_freeze_opt_state(monkeypatch, argv)
    _a2, _c2, frozen = _smoke(argv, seed=0)
    frozen.load_state_dict(pre.state_dict(), strict=True)      # requires_grad is not serialised
    opt = T.build_optimizer(frozen, a)
    with pytest.raises(ValueError, match="doesn't match the size"):
        opt.load_state_dict(sd)                                 # the LOUD refusal, pinned
    new, rec = C.convert_opt_state(sd, frozen)
    assert rec["dropped_with_state"] == [] and len(rec["dropped"]) == 18   # the eight's tensors
    assert rec["n_state_entries_out"] == rec["n_state_entries"]
    opt2 = T.build_optimizer(frozen, a)
    opt2.load_state_dict(new)
    # every surviving parameter's moments are the pre-freeze ones, BY NAME (not by position)
    orders = C.old_and_new_orders(frozen)
    old_names = orders["old"]["encoder"] + orders["old"]["head"]
    idx = [i for g in sd["param_groups"] for i in g["params"]]
    old_by_name = {n: sd["state"][i] for i, n in zip(idx, old_names) if i in sd["state"]}
    fr_names = {id(p): n for n, p in frozen.named_parameters()}
    n_checked = 0
    for g in opt2.param_groups:
        for p in g["params"]:
            n = fr_names[id(p)]
            assert not _under(n, set(frozen._bypass_declared))   # the eight; lat/lon LIVE here
            st = opt2.state.get(p)
            if not st:
                continue
            for k in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(st[k], old_by_name[n][k]), (n, k)
            n_checked += 1
    assert n_checked == rec["n_state_entries_out"] == len(old_by_name)


def test_7b_RED_a_dropped_entry_that_CARRIES_state_is_refused(monkeypatch):
    C = _converter()
    argv = R7_CORE + F1_F6 + ["--no-strategic"]
    _a, _pre, sd = _pre_freeze_opt_state(monkeypatch, argv)
    _a2, _c2, frozen = _smoke(argv, seed=0)
    orders = C.old_and_new_orders(frozen)
    old_names = orders["old"]["encoder"] + orders["old"]["head"]
    idx = [i for g in sd["param_groups"] for i in g["params"]]
    victim = next(i for i, n in zip(idx, old_names) if n == "core.route_head.weight")
    shape = dict(frozen.named_parameters())["core.route_head.weight"].shape
    sd["state"][victim] = {"step": torch.tensor(2.0), "exp_avg": torch.zeros(shape),
                           "exp_avg_sq": torch.zeros(shape)}
    with pytest.raises(SystemExit, match="CARRY AdamW state"):
        C.convert_opt_state(sd, frozen)


# ---- 8. a bypass row whose module this build did not construct is not demanded --------------- #
def test_8_a_module_the_build_did_not_construct_is_not_demanded():
    """`--goal-point-inject` turns `nav_inject` OFF (PREREG E15), so `nav_to_str` is never built:
    the argv side must not demand its declaration (a module that does not exist cannot be a dead
    trainable group) -- and must still demand the other seven."""
    args, _cfg, model = _smoke(R7_CORE + F1_F6 + ["--no-strategic", "--goal-point-inject"])
    assert getattr(model, "nav_to_str", None) is None
    assert set(model._bypass_declared) == set(TEN_STRATEGIC) - {"nav_to_str"}
    assert "nav_to_str" in dvb.expected_grad_unreachable(args)          # argv alone names it...
    assert dvb.check_grad_unreachable(model, args) == []                  # ...the build is judged
