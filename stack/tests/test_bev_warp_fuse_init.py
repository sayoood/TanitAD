"""refcv9 T1h-INIT -- ``--r9-bev-warp-fuse-init {rezero,identity}``: the COPY init of the T1 fuse.  CPU, the tiny FULL model of ``test_map_only_ft._build()``.

WHY (MEASURED by the MM on E1 arm T1): the trained 1x1 fuse (``_map_hires.bev_warp.fuse.weight``, [64, 130, 1, 1]) is still at its PyTorch default init (abs-mean 0.0442 / fro 4.69 vs
the kaiming_uniform(a=sqrt5) expectation 0.0439 / 4.62) and its gate random-walked (|g| 0.0077 ~ lr*sqrt(steps)); a zero-training probe found the gate gradient's sign consistency at the null
under BOTH the default and the history-extended mask: a ReZero DEAD START (random W -> no consistent gate push -> gate ~ 0 -> W's gradient ~ gate ~ 0).  ``identity`` writes
``W[c, p*(C+1)+c] = 1/P`` (every other entry 0) so ``fuse(x)`` is the MEAN of the lags' warped past BEV; the gate stays exactly 0, so the step-0 forward is the baseline's.

  IDENTITY    fuse(x) equals the mean of the warped inputs EXACTLY (P = 2) / to float precision (P = 3); the mask columns carry weight 0; gate 0 => ``mix`` returns ``bev`` bit for bit
  ANALYTIC    the gate's gradient at step 0 equals sum_cells dL/dbev_fused * mean_p(warped_p) (rel 1e-5), on the FULL model's real loss
  DEFAULT     ``rezero`` (and an absent attribute) => the weight is bit-equal to the tip's construction at the same seed, and NO other state_dict entry moves
  STEP 0      identity vs rezero at gate 0: the loss dict and EVERY gradient but the gate's are bit-identical (the init changes nothing else at step 0)
  FRESH ONLY  attach() runs BEFORE every checkpoint load (resume's strict load, the eval loader's strict load, --init-from which exempts the fuse): a strict load after the build restores
              the TRAINED weights, never the identity; ``init_mode`` is build-time provenance, not state
  MUTATION    the ``init_identity_`` call removed from ``attach`` (source-level) / ``init_identity_`` a no-op => the IDENTITY check goes RED and G-DVB names the build
  GATE        a G-DVB entry (kind ``built``); a non-default value without ``--r9-bev-warp`` / an unknown value is refused; G-EVAL reads ``fuse_init`` off both builds
"""
from __future__ import annotations

import copy
import inspect
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_bev_warp_full_model as FM  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.models import bev_warp as BW  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402

FUSE = "--r9-bev-warp-fuse-init"
RUNS = ROOT / "ops" / "runs.d"


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _built_init(init, *, seed=None, gate=0.0, **over):
    """A NEW identically-initialised tiny FULL model with T1 built and ``--r9-bev-warp-fuse-init <init>`` (``None`` = the attribute is ABSENT, an old namespace).  The global RNG is
    re-seeded right before ``attach`` when ``seed`` is given (the default init draws from it)."""
    model, cfg = R._build()
    args = FM._args(**over)
    if init is not None:
        args.r9_bev_warp_fuse_init = init
    BF.pin_args(args)
    if seed is not None:
        torch.manual_seed(seed)
    BF.attach(model, args)
    model._map_hires.bev_warp.gate.data.fill_(gate)
    return model, cfg, args


def _is_identity(w: torch.Tensor, c: int, p: int) -> bool:
    """``w`` ([C, P*(C+1), 1, 1]) is EXACTLY the copy pattern: 1/P at [c, k*(C+1)+c], 0.0 everywhere else (the mask columns included)."""
    exp = torch.zeros_like(w)
    for k in range(p):
        exp[torch.arange(c), k * (c + BW.MASK_CH) + torch.arange(c), 0, 0] = 1.0 / p
    return bool(torch.equal(w, exp))


def _tip_fuse_weight(model, seed):
    """The tip's construction of the fuse weight: ``BevWarpFuse(...)`` inside ``fork_rng`` at the same global seed (what ``attach`` did before this change)."""
    mh = model._map_hires
    rows = BF.lag_rows_of(FM._args())
    torch.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        f = BW.BevWarpFuse(int(mh.cfg.lift_width), len(rows), float(mh.cfg.x_max_m), float(mh.cfg.y_half_m), rows)
    return f.fuse.weight.detach().clone()


def _fuse_inputs(c=8, p=2, b=2, x=6, y=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    warped = [torch.randn(b, c, x, y, generator=g) for _ in range(p)]
    masks = [(torch.rand(b, 1, x, y, generator=g) > 0.5).float() for _ in range(p)]
    return warped, masks, torch.randn(b, c, x, y, generator=g)


def _module(c=8, p=2, identity=True):
    f = BW.BevWarpFuse(c, p, 20.0, 10.0, tuple(5 * (i + 1) for i in range(p)))
    if identity:
        f.init_identity_()
    return f


def _nz(g) -> int:
    return 0 if g is None else int(torch.count_nonzero(g))


# ===================================================================================================================== #
# IDENTITY: fuse(x) is the mean of the warped inputs                                                                     #
# ===================================================================================================================== #
def test_IDENTITY_the_weight_is_exactly_the_copy_pattern_and_the_gate_stays_exactly_zero():
    for c, p in ((8, 2), (8, 3), (64, 2)):
        f = _module(c, p)
        assert tuple(f.fuse.weight.shape) == (c, p * (c + 1), 1, 1) and _is_identity(f.fuse.weight.detach(), c, p)
        assert int(torch.count_nonzero(f.fuse.weight)) == c * p, "exactly C*P non-zero weights: the mask columns and every cross-channel entry are 0.0"
        assert float(f.gate.abs().max()) == 0.0 and f.init_mode == "identity"
        assert f.fuse.bias is None
    assert _module(8, 2, identity=False).init_mode == "rezero"


def test_IDENTITY_fuse_of_x_equals_the_mean_of_the_warped_inputs_EXACTLY_for_two_lags():
    f = _module(8, 2)
    warped, masks, bev = _fuse_inputs(8, 2)
    x = torch.cat([t for pair in zip(warped, masks) for t in pair], dim=1)
    got = f.fuse(x)
    assert torch.equal(got, torch.stack(warped).mean(0)), "P = 2: 0.5 is a power of two, the copy is exact"
    # the mask columns carry weight 0: another mask, the same output
    masks2 = [torch.ones_like(m) for m in masks]
    x2 = torch.cat([t for pair in zip(warped, masks2) for t in pair], dim=1)
    assert torch.equal(f.fuse(x2), got)
    # a lag with no history (warped 0, mask 0): the mean still divides by P (the other lag's content / 2)
    w0 = [warped[0], torch.zeros_like(warped[1])]
    m0 = [masks[0], torch.zeros_like(masks[1])]
    x0 = torch.cat([t for pair in zip(w0, m0) for t in pair], dim=1)
    assert torch.equal(f.fuse(x0), warped[0] * 0.5)


def test_IDENTITY_three_lags_agree_with_the_mean_to_float_precision_and_mix_is_the_baseline_at_gate_0():
    f3 = _module(8, 3)
    warped, masks, bev = _fuse_inputs(8, 3)
    x = torch.cat([t for pair in zip(warped, masks) for t in pair], dim=1)
    torch.testing.assert_close(f3.fuse(x), torch.stack(warped).mean(0), rtol=1e-6, atol=1e-7)
    f2 = _module(8, 2)
    w2, m2, bev2 = _fuse_inputs(8, 2, seed=4)
    assert torch.equal(f2.mix(bev2, w2, m2), bev2), "gate 0: the fused BEV IS the BEV, bit for bit (the step-0 forward is the baseline's)"
    f2.gate.data.fill_(0.3)
    torch.testing.assert_close(f2.mix(bev2, w2, m2), bev2 + 0.3 * torch.stack(w2).mean(0), rtol=1e-6, atol=1e-7)


# ===================================================================================================================== #
# ANALYTIC: the gate's gradient at step 0, on the FULL model's real loss                                                 #
# ===================================================================================================================== #
def _gate_grad_vs_mean_warped(monkeypatch, init):
    """-> (model, gate grad, reference sum_cells dL/dbev * mean_p(warped_p) [float64], relative error) at step 0 on the FULL model's real loss."""
    model, cfg, args = _built_init(init)
    fz = model._map_hires.bev_warp
    assert float(fz.gate.abs().max()) == 0.0
    cap: dict = {}
    orig = BW.BevWarpFuse.mix

    def spy(self, bev, warped, masks):
        out = orig(self, bev, warped, masks)
        out.retain_grad()                                       # dL/d(bev_fused): the sum over EVERY consumer of the fused BEV
        cap["out"], cap["warped"] = out, [w.detach() for w in warped]
        return out
    monkeypatch.setattr(BW.BevWarpFuse, "mix", spy)
    out = FM._loss(model, FM._batch(cfg, model, args))
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    g = fz.gate.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    dl = cap["out"].grad.double()
    mean_w = torch.stack(cap["warped"]).double().mean(0)
    ref = (dl * mean_w).sum(dim=(0, 2, 3))                     # d loss / d gate_c = sum over (b, x, y) of dL/dout * fuse(x)_c; under identity fuse(x) = mean_p warped_p
    assert float(ref.abs().sum()) > 0.0, "the identity is not vacuous: the history reached the loss"
    assert _nz(fz.fuse.weight.grad) == 0, "ReZero: at gate 0 the fuse weight's gradient is EXACTLY 0.0 (the gate's push comes first)"
    return model, g, ref, float((g.double() - ref).norm() / ref.norm())


def test_ANALYTIC_the_gate_gradient_at_step_0_equals_sum_dL_dbev_times_the_mean_warped(monkeypatch):
    _, _, _, rel = _gate_grad_vs_mean_warped(monkeypatch, "identity")
    assert rel < 1e-5, ("gate gradient != sum_cells dL/dbev * mean_warped", rel)


def test_ANALYTIC_CONTROL_under_the_default_init_the_same_formula_does_NOT_hold_so_the_probe_discriminates(monkeypatch):
    _, _, _, rel = _gate_grad_vs_mean_warped(monkeypatch, "rezero")
    assert rel > 1e-2, ("the identity formula must NOT describe a random-W fuse", rel)


# ===================================================================================================================== #
# DEFAULT: bit-identical to the tip                                                                                      #
# ===================================================================================================================== #
@pytest.mark.parametrize("default", ["rezero", None])
def test_DEFAULT_the_weight_is_bit_equal_to_the_tips_construction_at_the_same_seed_and_nothing_else_moves(default):
    model, _, _ = _built_init(default, seed=7)
    assert torch.equal(model._map_hires.bev_warp.fuse.weight.detach(), _tip_fuse_weight(model, 7)), "the default init draws are the tip's"
    assert model._map_hires.bev_warp.init_mode == "rezero"
    ident, _, _ = _built_init("identity", seed=7)
    sd_d, sd_i = model.state_dict(), ident.state_dict()
    assert list(sd_d) == list(sd_i), "the lever adds no key"
    diff = [k for k in sd_d if not torch.equal(sd_d[k], sd_i[k])]
    assert diff == ["_map_hires.bev_warp.fuse.weight"], diff                  # NO other parameter's draw moves (the fork_rng / no-draw argument, measured)
    w = model._map_hires.bev_warp.fuse.weight.detach()
    assert not _is_identity(w, model._map_hires.bev_warp.d_bev, model._map_hires.bev_warp.n_past) and float(w.abs().mean()) > 0.0


def test_STEP_0_identity_changes_neither_the_loss_nor_any_gradient_but_the_gates():
    m_id, cfg, a_id = _built_init("identity", seed=3)
    m_rz, _, a_rz = _built_init("rezero", seed=3)
    b_id, b_rz = FM._batch(cfg, m_id, a_id), FM._batch(cfg, m_rz, a_rz)
    out_id, out_rz = FM._loss(m_id, b_id), FM._loss(m_rz, b_rz)
    assert FM._dict_equal(out_id, out_rz), "gate 0: the forward (and every returned loss tensor) is the baseline's, whatever the fuse weight"
    g_id, g_rz = FM._backward(m_id, out_id), FM._backward(m_rz, out_rz)
    gate, fw = "_map_hires.bev_warp.gate", "_map_hires.bev_warp.fuse.weight"
    rest = lambda g: {k: v for k, v in g.items() if k not in (gate, fw)}      # noqa: E731
    assert FM._grads_equal(rest(g_id), rest(g_rz)), "every OTHER parameter's step-0 gradient is bit-identical"
    assert _nz(g_id[fw]) == 0 and _nz(g_rz[fw]) == 0, "ReZero: the fuse weight's step-0 gradient is exactly 0.0 under both"
    assert _nz(g_id[gate]) > 0 and not torch.equal(g_id[gate], g_rz[gate]), "ONLY the gate's gradient differs: <dL/dbev, mean warped> vs <dL/dbev, W_random x>"


# ===================================================================================================================== #
# FRESH BUILD ONLY                                                                                                       #
# ===================================================================================================================== #
def test_FRESH_ONLY_a_strict_load_after_the_build_restores_the_TRAINED_fuse_never_the_identity():
    trained, _, _ = _built_init("identity")
    fz = trained._map_hires.bev_warp
    g = torch.Generator().manual_seed(11)
    fz.fuse.weight.data = torch.randn(tuple(fz.fuse.weight.shape), generator=g) * 0.05          # what training did to it
    fz.gate.data = torch.randn(tuple(fz.gate.shape), generator=g) * 0.1
    sd = copy.deepcopy(trained.state_dict())
    assert not any("init_mode" in k for k in sd), "init_mode is build-time provenance, not state"
    for init in ("identity", "rezero"):                       # the eval loader / resume build first (either declared init), THEN load strictly
        fresh, _, _ = _built_init(init)
        if init == "identity":
            assert _is_identity(fresh._map_hires.bev_warp.fuse.weight.detach(), fz.d_bev, fz.n_past), "a fresh build IS the identity"
        res = fresh.load_state_dict(sd, strict=True)
        assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
        w = fresh._map_hires.bev_warp
        assert torch.equal(w.fuse.weight, fz.fuse.weight) and torch.equal(w.gate, fz.gate)
        assert not _is_identity(w.fuse.weight.detach(), fz.d_bev, fz.n_past), "the trained weights, not the init"


def test_FRESH_ONLY_attach_precedes_every_checkpoint_load_in_the_trainer_and_the_eval_loader():
    tr = inspect.getsource(T.train)
    i_attach = tr.index("_bw.attach(model, args)")
    assert i_attach < tr.index('model.load_state_dict(state["model"])'), "resume's strict load comes AFTER the build"
    assert i_attach < tr.index("args.init_from"), "the --init-from warm start comes AFTER the build"
    ld = (ROOT / "tanitad" / "eval" / "refcv7_loader.py").read_bytes().decode("utf-8")
    assert ld.index("tr._bw.attach(model, args)") < ld.index('model.load_state_dict(ck["model"], strict=strict)'), "the eval loader's strict load comes AFTER the build"
    from tanitad.refs import refcv9_perc_rig as RIG
    assert BF.KEY_PREFIX in RIG.bev_warp_exemptions(_built_init("identity")[0]), "--init-from exempts the fuse keys: a warm start keeps the BUILD init (identity), never a source's"
    assert BF.warm_start_prefixes(_built_init("identity")[0]) == (BF.KEY_PREFIX,)


# ===================================================================================================================== #
# MUTATION                                                                                                              #
# ===================================================================================================================== #
def _identity_problems(model) -> list:
    """The IDENTITY reading, as one check (``[]`` iff the declared identity init reached the weights)."""
    fz = model._map_hires.bev_warp
    probs = []
    if not _is_identity(fz.fuse.weight.detach(), fz.d_bev, fz.n_past):
        probs.append("the weight is not the copy pattern")
    if fz.init_mode != "identity":
        probs.append("init_mode is not identity")
    return probs


def test_MUTATION_deleting_the_init_call_from_attach_turns_the_IDENTITY_check_and_G_DVB_RED():
    model, _, args = _built_init("identity")
    assert _identity_problems(model) == [] and D.REGISTRY["r9_bev_warp_fuse_init"].check(model, args) == []        # the real attach is GREEN
    src = inspect.getsource(BF.attach)
    src, n = re.subn(r'^    if fuse_init_of\(args\) == "identity":\n        fuse\.init_identity_\(\)[^\n]*\n', "", src, flags=re.M)
    assert n == 1, n
    g = dict(BF.__dict__)
    exec(compile(src, "<attach variant>", "exec"), g)
    mutant = g["attach"]
    assert "init_identity_" not in mutant.__code__.co_names
    model2, _ = R._build()
    args2 = FM._args(r9_bev_warp_fuse_init="identity")
    mutant(model2, args2)
    assert _identity_problems(model2) == ["the weight is not the copy pattern", "init_mode is not identity"], "RED: the argv declared identity and nothing reached the weights"
    mism = D.REGISTRY["r9_bev_warp_fuse_init"].check(model2, args2)
    assert [m.lever for m in mism] == ["r9_bev_warp_fuse_init"] and mism[0].declared == "identity" and mism[0].built == "rezero"


def test_MUTATION_an_init_that_writes_nothing_is_caught_by_the_analytic_probe_not_only_by_the_attribute(monkeypatch):
    monkeypatch.setattr(BW.BevWarpFuse, "init_identity_", lambda self: None)
    model, _, args = _built_init("identity")
    assert _identity_problems(model) == ["the weight is not the copy pattern", "init_mode is not identity"]
    f = model._map_hires.bev_warp
    warped, masks, _ = _fuse_inputs(f.d_bev, f.n_past)
    x = torch.cat([t for pair in zip(warped, masks) for t in pair], dim=1)
    assert not torch.allclose(f.fuse(x), torch.stack(warped).mean(0), rtol=1e-3, atol=1e-3), "fuse(x) is NOT the mean of the warped inputs: the IDENTITY check would fail"
    assert D.REGISTRY["r9_bev_warp_fuse_init"].check(model, args), "and G-DVB names it"


# ===================================================================================================================== #
# the flag's refusals, G-DVB, the stamp, G-EVAL                                                                          #
# ===================================================================================================================== #
def test_pin_args_refuses_a_non_default_init_without_r9_bev_warp_and_an_unknown_value():
    with pytest.raises(SystemExit, match=r"--r9-bev-warp-fuse-init without --r9-bev-warp: nothing would read it \(a dead flag\)"):
        BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_fuse_init="identity"))
    BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_fuse_init="rezero"))        # the DEFAULT value without T1: not a lever, nothing refused
    BF.pin_args(FM._args(r9_bev_warp=False))
    with pytest.raises(SystemExit, match="not one of"):
        BF.pin_args(FM._args(r9_bev_warp_fuse_init="kaiming"))
    with pytest.raises(SystemExit, match="not one of"):
        BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_fuse_init="kaiming"))
    BF.pin_args(FM._args(r9_bev_warp_fuse_init="identity"))
    BF.pin_args(FM._args(r9_bev_warp_fuse_init="identity", r9_bev_warp_supervise_hist=True, r9_bev_warp_shuffle=True))
    p = T.build_parser()
    assert {a.dest: a.default for a in p._actions}["r9_bev_warp_fuse_init"] == "rezero", "default rezero: the flag must be NAMED in the arm's argv"
    ns = p.parse_args(["--arm", "hier", "--out", "x", FUSE, "identity"])
    assert ns.r9_bev_warp_fuse_init == "identity" and ns.r9_bev_warp is False
    with pytest.raises(SystemExit, match="dead flag"):
        BF.pin_args(ns)
    with pytest.raises(SystemExit):                                                  # argparse's own choices
        p.parse_args(["--arm", "hier", "--out", "x", FUSE, "kaiming"])


def test_the_G_DVB_entry_is_kind_built_and_catches_both_directions():
    lever = D.REGISTRY["r9_bev_warp_fuse_init"]
    assert lever.kind == "built" and lever.check is BF.dvb_check_fuse_init
    ident, _, a_id = _built_init("identity")
    rez, _, a_rz = _built_init("rezero")
    assert lever.check(ident, a_id) == [] and lever.check(rez, a_rz) == []
    m = lever.check(rez, a_id)                                                       # argv says identity, the build is the default
    assert [x.lever for x in m] == ["r9_bev_warp_fuse_init"] and (m[0].declared, m[0].built) == ("identity", "rezero") and m[0].read_from == "model._map_hires.bev_warp.init_mode"
    m = lever.check(ident, a_rz)                                                     # the reverse
    assert [(x.declared, x.built) for x in m] == [("rezero", "identity")]
    plain, _ = R._build()
    assert lever.check(plain, FM._args(r9_bev_warp=False)) == [], "T1 not built, default declared: inert"
    assert lever.check(plain, FM._args(r9_bev_warp=False, r9_bev_warp_fuse_init="identity")), "identity declared, no fuse built: a mismatch"


def test_the_stamp_records_identity_and_is_unchanged_for_the_default():
    ident, _, a_id = _built_init("identity")
    rez, _, a_rz = _built_init("rezero")
    s_id, s_rz = BF.stamp(ident, a_id), BF.stamp(rez, a_rz)
    assert s_id["fuse_init"]["mode"] == "identity" and s_id["fuse_init"]["flag"] == FUSE and "fresh build" in s_id["fuse_init"]["applied"]
    assert "fuse_init" not in s_rz and set(s_id) - {"fuse_init"} == set(s_rz), "default: the T1 stamp is key-for-key what it was"


def test_G_EVAL_reads_the_init_off_both_builds_and_names_a_disagreement():
    import launch_gate as LG
    ident, _, a_id = _built_init("identity")
    ident2, _, a_id2 = _built_init("identity")
    rez, _, a_rz = _built_init("rezero")
    r, d = LG.rig_eval_t1_build_reasons(a_id, ident, a_id2, ident2)
    assert r == [] and d["trainer"]["fuse_init"] == "identity" == d["loader"]["fuse_init"]
    r, _ = LG.rig_eval_t1_build_reasons(a_id, ident, a_rz, rez)                      # trainer identity, loader forgot the flag
    assert [x.split(" differs")[0] for x in r] == ["T1 (--r9-bev-warp): fuse_init"], r
    r, _ = LG.rig_eval_t1_build_reasons(a_rz, rez, a_rz, rez)
    assert r == [], "the existing arms: rezero / rezero, inert"


def test_the_hop_is_named_at_its_source_line():
    src = inspect.getsource(BF.attach)
    assert src.count('if fuse_init_of(args) == "identity":') == 1 and src.count("fuse.init_identity_()") == 1
    assert src.index("with torch.random.fork_rng(devices=[])") < src.index("fuse.init_identity_()") < src.index("mh.add_module(HEAD_ATTR, fuse.to(dev))")
    assert inspect.getsource(BW.BevWarpFuse.init_identity_).count('self.init_mode = "identity"') == 1
    assert "--r9-bev-warp-fuse-init" in {s for a in T.build_parser()._actions for s in a.option_strings}
