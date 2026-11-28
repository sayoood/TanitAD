"""refcv9 T1h-FILL -- ``--r9-bev-warp-fill-only``: the gated history term is added ONLY on BEV cells the CURRENT lift does not observe.  CPU, the tiny FULL model of ``test_map_only_ft._build()``.

WHY (MEASURED by the MM post hoc, GADD_CONSISTENCY.txt): the gate gradient from the HISTORY-ONLY cells alone is sign-consistent (0.383 trained-W / 0.391 identity-W, above the sign-flip null p99
0.30 / 0.34; under the identity init -mean g > 0 on 72 % of channels), while the gradient over ALL supervised cells sits at the null: the ~10x more camera-seen cells, where history is redundant, swamp it.
``out = bev + gate * fuse(x) * (1 - valid_now)``, ``valid_now`` = the lift's own ``valid.any(dim=1)`` at the fuse's resolution (the one ``lift_hook`` already holds).

  (a) OFF       the flag absent / False => ``mix`` is the tip's expression bit for bit; the FULL model's loss dict and EVERY gradient are bit-identical to a tip ``mix``
  (b) FILL      a cell valid now is EXACTLY ``bev`` (any gate, any history); a cell invalid now (hist-valid or not) is ``bev + gate * fuse(x)``; on the FULL model the ``valid_now`` ``mix`` receives IS the
                forward's ``map_hires_lift_valid`` and the fused BEV equals the lift's BEV on every valid-now cell
  (c) ANALYTIC  the gate's gradient equals the gradient from the not-valid-now cells ONLY: sum over (b, x, y) of dL/dbev * fuse(x) * (1 - valid_now) (rel 1e-5), and NOT the all-cells sum (control)
  (d) MUTATION  the ``(1 - valid_now)`` factor dropped from ``mix`` (source-level) => (b) goes RED; a fill-only ``mix`` called without ``valid_now`` REFUSES; the hook's pass-through removed => REFUSES
  GATE          a G-DVB entry (kind ``built``); the flag without ``--r9-bev-warp`` is refused; G-EVAL reads ``fuse_fill_only`` off both builds; the stamp records it
"""
from __future__ import annotations

import inspect
import re
import sys
import textwrap
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

FILL = "--r9-bev-warp-fill-only"


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ===================================================================================================================== #
# helpers                                                                                                               #
# ===================================================================================================================== #
def _built(fill, *, init="identity", gate=0.0, **over):
    """A NEW identically-initialised tiny FULL model with T1 built; ``fill`` = True / False / None (the attribute ABSENT: an old namespace)."""
    model, cfg = R._build()
    args = FM._args(r9_bev_warp_fuse_init=init, **over)
    if fill is not None:
        args.r9_bev_warp_fill_only = fill
    BF.pin_args(args)
    BF.attach(model, args)
    model._map_hires.bev_warp.gate.data.fill_(gate)
    return model, cfg, args


def _module(c=8, p=2, fill=True, gate=0.3, seed=0):
    f = BW.BevWarpFuse(c, p, 20.0, 10.0, tuple(5 * (i + 1) for i in range(p)))
    f.fill_only = fill
    f.gate.data = torch.randn(c, generator=torch.Generator().manual_seed(seed + 1)) * gate + gate        # a NON-ZERO gate on every channel
    return f


def _inputs(c=8, p=2, b=2, x=6, y=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    warped = [torch.randn(b, c, x, y, generator=g) for _ in range(p)]
    masks = [(torch.rand(b, 1, x, y, generator=g) > 0.5).float() for _ in range(p)]
    valid = torch.rand(b, x, y, generator=g) > 0.5
    return warped, masks, torch.randn(b, c, x, y, generator=g), valid


def _tip_mix(self, bev, warped, masks):
    """``BevWarpFuse.mix`` AS THE TIP HAD IT (verbatim)."""
    if len(warped) != self.n_past or len(masks) != self.n_past:
        raise ValueError("[bev-warp] fuse built for %d past frames, got %d / %d" % (self.n_past, len(warped), len(masks)))
    x = torch.cat([t for pair in zip(warped, masks) for t in pair], dim=1)
    return bev + self.gate.view(1, -1, 1, 1).to(bev.dtype) * self.fuse(x)


def _fill_problems(f, warped, masks, bev, valid) -> list:
    """The (b) reading, as ONE check (``[]`` iff fill-only behaves): valid-now cells are EXACTLY bev; invalid-now cells are bev + gate*fuse(x)."""
    out = f.mix(bev, warped, masks, valid_now=valid)
    x = torch.cat([t for pair in zip(warped, masks) for t in pair], dim=1)
    full = bev + f.gate.view(1, -1, 1, 1) * f.fuse(x)
    vm = valid.unsqueeze(1).expand_as(bev)
    probs = []
    if not torch.equal(out[vm], bev[vm]):
        probs.append("a cell valid now is not exactly bev")
    if not torch.equal(out[~vm], full[~vm]):
        probs.append("a cell invalid now is not bev + gate*fuse")
    return probs


def _nz(g) -> int:
    return 0 if g is None else int(torch.count_nonzero(g))


# ===================================================================================================================== #
# (a) OFF: bit-identical to the tip                                                                                      #
# ===================================================================================================================== #
def test_a_OFF_mix_is_the_tips_expression_bit_for_bit_and_never_reads_valid_now():
    f = _module(fill=False)
    assert f.fill_only is False
    warped, masks, bev, valid = _inputs()
    ref = _tip_mix(f, bev, warped, masks)
    assert torch.equal(f.mix(bev, warped, masks), ref)
    assert torch.equal(f.mix(bev, warped, masks, valid_now=valid), ref), "OFF: valid_now is IGNORED, the output is the tip's"
    assert torch.equal(f.mix(bev, warped, masks, valid_now=torch.ones_like(valid)), ref)


def test_a_OFF_the_FULL_models_loss_and_every_gradient_are_bit_identical_to_a_tip_mix(monkeypatch):
    m_new, cfg, a_new = _built(False, gate=0.3)
    m_tip, _, a_tip = _built(None, gate=0.3)
    assert m_new._map_hires.bev_warp.fill_only is False and m_tip._map_hires.bev_warp.fill_only is False
    b_new, b_tip = FM._batch(cfg, m_new, a_new), FM._batch(cfg, m_tip, a_tip)
    out_new = FM._loss(m_new, b_new)
    g_new = FM._backward(m_new, out_new)
    monkeypatch.setattr(BW.BevWarpFuse, "mix", _tip_mix)             # the tip's mix, installed on the class BEFORE the hook looks it up
    out_tip = FM._loss(m_tip, b_tip)
    g_tip = FM._backward(m_tip, out_tip)
    assert FM._dict_equal(out_new, out_tip) and FM._grads_equal(g_new, g_tip), "gate 0.3, fill-only OFF: nothing moved"


# ===================================================================================================================== #
# (b) FILL: where the history is, and where it is not                                                                   #
# ===================================================================================================================== #
def test_b_a_cell_valid_now_is_EXACTLY_bev_at_any_gate_and_an_invalid_one_gets_bev_plus_gate_fuse():
    f = _module(fill=True)
    warped, masks, bev, valid = _inputs()
    assert bool(valid.any()) and bool((~valid).any())
    assert _fill_problems(f, warped, masks, bev, valid) == []
    for gate in (0.0, 1.0, -7.5, 1e3):
        f.gate.data.fill_(gate)
        out = f.mix(bev, warped, masks, valid_now=valid)
        vm = valid.unsqueeze(1).expand_as(bev)
        assert torch.equal(out[vm], bev[vm]), ("valid-now cells must be exactly bev at gate", gate)
    f.gate.data.fill_(0.5)
    out = f.mix(bev, warped, masks, valid_now=valid)
    assert not torch.equal(out, bev), "and the history DID land somewhere: the invalid-now cells moved"
    assert bool((out[~valid.unsqueeze(1).expand_as(bev)] != bev[~valid.unsqueeze(1).expand_as(bev)]).any())
    # all-valid: the fused BEV IS the BEV; none valid: the fuse is the unrestricted one
    assert torch.equal(f.mix(bev, warped, masks, valid_now=torch.ones_like(valid)), bev)
    assert torch.equal(f.mix(bev, warped, masks, valid_now=torch.zeros_like(valid)), _tip_mix(f, bev, warped, masks))


def test_b_on_the_FULL_model_mix_receives_the_forwards_own_lift_valid_and_valid_now_cells_stay_the_lifts_BEV(monkeypatch):
    model, cfg, args = _built(True, gate=0.4)
    cap: dict = {}
    orig = BW.BevWarpFuse.mix

    def spy(self, bev, warped, masks, **kw):
        out = orig(self, bev, warped, masks, **kw)
        cap["bev"], cap["out"], cap["valid_now"] = bev.detach().clone(), out.detach().clone(), kw.get("valid_now")
        return out
    monkeypatch.setattr(BW.BevWarpFuse, "mix", spy)
    real_loss = T._map_hires_loss
    seen: dict = {}

    def wrap(model_, out, batch, device, extra, with_metrics=True):
        seen["lv"] = out["perception"]["map_hires_lift_valid"].detach().clone()
        return real_loss(model_, out, batch, device, extra, with_metrics)
    monkeypatch.setattr(T, "_map_hires_loss", wrap)
    FM._loss(model, FM._batch(cfg, model, args))
    vn = cap["valid_now"]
    assert vn is not None and vn.dtype == torch.bool and torch.equal(vn, seen["lv"]), "valid_now IS the forward's `valid.any(dim=1)` (the lift's own validity)"
    assert bool(vn.any()) and bool((~vn).any())
    vm = vn.unsqueeze(1).expand_as(cap["bev"])
    assert torch.equal(cap["out"][vm], cap["bev"][vm]), "every valid-now cell of the FUSED BEV is exactly the lift's BEV (gate 0.4)"
    assert bool((cap["out"][~vm] != cap["bev"][~vm]).any()), "and the history landed on the invalid-now cells"


# ===================================================================================================================== #
# (c) ANALYTIC: the gate gradient comes from the not-valid-now cells only                                                #
# ===================================================================================================================== #
def _gate_grad_parts(monkeypatch):
    """-> (gate grad, ref over NOT-valid-now cells, ref over ALL cells) -- sum_(b,x,y) dL/dbev * fuse(x) [* (1 - valid_now)], float64, identity init, gate 0, the FULL model's real loss."""
    model, cfg, args = _built(True)
    fz = model._map_hires.bev_warp
    assert float(fz.gate.abs().max()) == 0.0
    cap: dict = {}
    orig = BW.BevWarpFuse.mix

    def spy(self, bev, warped, masks, **kw):
        out = orig(self, bev, warped, masks, **kw)
        out.retain_grad()
        cap["out"], cap["warped"], cap["valid_now"] = out, [w.detach() for w in warped], kw.get("valid_now")
        return out
    monkeypatch.setattr(BW.BevWarpFuse, "mix", spy)
    out = FM._loss(model, FM._batch(cfg, model, args))
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    dl = cap["out"].grad.double()
    fuse_x = torch.stack(cap["warped"]).double().mean(0)                       # identity init: fuse(x) = the mean of the warped past BEV
    lv = cap["valid_now"]
    assert lv is not None, "fill-only: the hook passed valid_now"
    keep = (~lv).unsqueeze(1).double()
    return fz.gate.grad, (dl * fuse_x * keep).sum(dim=(0, 2, 3)), (dl * fuse_x).sum(dim=(0, 2, 3))


def test_c_the_gate_gradient_equals_the_gradient_from_the_not_valid_now_cells_only(monkeypatch):
    g, ref_fill, ref_all = _gate_grad_parts(monkeypatch)
    assert float(ref_fill.abs().sum()) > 0.0 and float(g.abs().sum()) > 0.0
    rel = float((g.double() - ref_fill).norm() / ref_fill.norm())
    assert rel < 1e-5, ("gate gradient != sum over not-valid-now cells of dL/dbev * fuse(x)", rel)
    rel_all = float((g.double() - ref_all).norm() / ref_all.norm())
    assert rel_all > 1e-2, ("the all-cells sum must NOT describe the fill-only gradient: the valid-now cells carry a material part", rel_all)


def test_c_CONTROL_with_the_flag_OFF_the_gate_gradient_IS_the_all_cells_sum(monkeypatch):
    model, cfg, args = _built(False)
    fz = model._map_hires.bev_warp
    cap: dict = {}
    orig = BW.BevWarpFuse.mix

    def spy(self, bev, warped, masks, **kw):
        out = orig(self, bev, warped, masks, **kw)
        out.retain_grad()
        cap["out"], cap["warped"] = out, [w.detach() for w in warped]
        return out
    monkeypatch.setattr(BW.BevWarpFuse, "mix", spy)
    out = FM._loss(model, FM._batch(cfg, model, args))
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    ref_all = (cap["out"].grad.double() * torch.stack(cap["warped"]).double().mean(0)).sum(dim=(0, 2, 3))
    assert float((fz.gate.grad.double() - ref_all).norm() / ref_all.norm()) < 1e-5


# ===================================================================================================================== #
# (d) MUTATION                                                                                                          #
# ===================================================================================================================== #
def test_d_MUTATION_dropping_the_one_minus_valid_now_factor_turns_the_FILL_check_RED(monkeypatch):
    f = _module(fill=True)
    warped, masks, bev, valid = _inputs()
    assert _fill_problems(f, warped, masks, bev, valid) == [], "the real mix is GREEN"
    src = textwrap.dedent(inspect.getsource(BW.BevWarpFuse.mix))
    src, n = re.subn(r" \* \(~valid_now\.bool\(\)\)\.unsqueeze\(1\)\.to\(bev\.dtype\)", "", src)
    assert n == 1, n
    g = dict(BW.__dict__)
    exec(compile(src, "<mix variant>", "exec"), g)
    monkeypatch.setattr(BW.BevWarpFuse, "mix", g["mix"])
    assert _fill_problems(f, warped, masks, bev, valid) == ["a cell valid now is not exactly bev"], "RED: the history leaks onto the cells the camera sees"


def test_d_MUTATION_a_fill_only_mix_without_valid_now_and_a_hook_that_forgets_to_pass_it_both_REFUSE(monkeypatch):
    f = _module(fill=True)
    warped, masks, bev, _ = _inputs()
    with pytest.raises(ValueError, match="no valid_now reached mix"):
        f.mix(bev, warped, masks)
    src = textwrap.dedent(inspect.getsource(BW.BevWarpFuse.lift_hook))
    src, n = re.subn(r"^[ ]+if self\.fill_only:\n[ ]+return self\.mix\(out, warped, masks, valid_now=src_valid\)[^\n]*\n", "", src, flags=re.M)
    assert n == 1, n
    g = dict(BW.__dict__)
    exec(compile(src, "<lift_hook variant>", "exec"), g)
    monkeypatch.setattr(BW.BevWarpFuse, "lift_hook", g["lift_hook"])        # BEFORE the build: attach registers the bound method it finds
    model, cfg, args = _built(True, gate=0.3)
    with pytest.raises(ValueError, match="no valid_now reached mix"):
        FM._loss(model, FM._batch(cfg, model, args))


# ===================================================================================================================== #
# the flag's refusals, G-DVB, the stamp, G-EVAL, the hop                                                                 #
# ===================================================================================================================== #
def test_pin_args_refuses_the_flag_without_r9_bev_warp_and_the_parser_default_is_off():
    with pytest.raises(SystemExit, match=r"--r9-bev-warp-fill-only without --r9-bev-warp: nothing would read it \(a dead flag\)"):
        BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_fill_only=True))
    with pytest.raises(SystemExit, match="dead flag"):
        BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_fill_only=True, r9_bev_warp_shuffle=True))
    BF.pin_args(FM._args(r9_bev_warp=False, r9_bev_warp_fill_only=False))
    BF.pin_args(FM._args(r9_bev_warp_fill_only=True))
    BF.pin_args(FM._args(r9_bev_warp_fill_only=True, r9_bev_warp_supervise_hist=True, r9_bev_warp_fuse_init="identity", r9_bev_warp_shuffle=True))
    p = T.build_parser()
    assert {a.dest: a.default for a in p._actions}["r9_bev_warp_fill_only"] is False, "default OFF: the flag must be NAMED in the arm's argv"
    ns = p.parse_args(["--arm", "hier", "--out", "x", FILL])
    assert ns.r9_bev_warp_fill_only is True and ns.r9_bev_warp is False
    with pytest.raises(SystemExit, match="dead flag"):
        BF.pin_args(ns)


def test_the_G_DVB_entry_is_kind_built_and_catches_both_directions():
    lever = D.REGISTRY["r9_bev_warp_fill_only"]
    assert lever.kind == "built" and lever.check is BF.dvb_check_fill_only
    on, _, a_on = _built(True)
    off, _, a_off = _built(False)
    assert lever.check(on, a_on) == [] and lever.check(off, a_off) == []
    m = lever.check(off, a_on)
    assert [x.lever for x in m] == ["r9_bev_warp_fill_only"] and (m[0].declared, m[0].built) == (True, False) and m[0].read_from == "model._map_hires.bev_warp.fill_only"
    assert [(x.declared, x.built) for x in lever.check(on, a_off)] == [(False, True)]
    plain, _ = R._build()
    assert lever.check(plain, FM._args(r9_bev_warp=False)) == [] and lever.check(plain, FM._args(r9_bev_warp=False, r9_bev_warp_fill_only=True)) != []


def test_the_stamp_records_the_flag_and_is_unchanged_when_off_and_G_EVAL_names_a_disagreement():
    import launch_gate as LG
    on, _, a_on = _built(True)
    on2, _, a_on2 = _built(True)
    off, _, a_off = _built(False)
    s_on, s_off = BF.stamp(on, a_on), BF.stamp(off, a_off)
    assert s_on["fill_only"]["flag"] == FILL and "(1 - valid_now)" in s_on["fill_only"]["rule"]
    assert "fill_only" not in s_off and set(s_on) - {"fill_only"} == set(s_off), "OFF: the stamp is key-for-key what it was"
    r, d = LG.rig_eval_t1_build_reasons(a_on, on, a_on2, on2)
    assert r == [] and d["trainer"]["fuse_fill_only"] is True == d["loader"]["fuse_fill_only"]
    r, _ = LG.rig_eval_t1_build_reasons(a_on, on, a_off, off)
    assert [x.split(" differs")[0] for x in r] == ["T1 (--r9-bev-warp): fuse_fill_only"], r
    assert LG.rig_eval_t1_build_reasons(a_off, off, a_off, off)[0] == []


def test_the_hops_are_named_at_their_source_lines():
    attach = inspect.getsource(BF.attach)
    assert attach.count("fuse.fill_only = bool(getattr(args, FILL_DEST, False))") == 1
    hook = inspect.getsource(BW.BevWarpFuse.lift_hook)
    assert hook.count("return self.mix(out, warped, masks, valid_now=src_valid)") == 1 and "src_valid = valid.any(dim=1)" in hook
    assert hook.index("src_valid = valid.any(dim=1)") < hook.index("valid_now=src_valid")
    mix = inspect.getsource(BW.BevWarpFuse.mix)
    assert mix.count("(~valid_now.bool()).unsqueeze(1).to(bev.dtype)") == 1 and mix.count("no valid_now reached mix") == 1
    assert "--r9-bev-warp-fill-only" in {s for a in T.build_parser()._actions for s in a.option_strings}


# ===================================================================================================================== #
# the arm AS LAUNCHED: all three T1h levers together, a few real optimiser steps                                         #
# ===================================================================================================================== #
def test_the_T1h_arm_as_launched_builds_steps_and_wakes_the_gate_and_then_the_fuse(monkeypatch):
    """supervise-hist + fuse-init identity + fill-only (+ T1 itself) on the tiny FULL model, 3 Adam steps: finite losses, the gate leaves its exact 0 on its FIRST step (the ReZero start is no longer dead),
    the fuse weight then receives a non-zero gradient, the history-supervision counter is live, and the fuse weight is still the copy pattern + whatever the optimiser did (not re-initialised)."""
    model, cfg, args = _built(True, r9_bev_warp_supervise_hist=True)
    assert model._r9_bev_warp.supervise_hist and model._map_hires.bev_warp.fill_only and model._map_hires.bev_warp.init_mode == "identity"
    batch = FM._batch(cfg, model, args)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    fz = model._map_hires.bev_warp
    w0 = fz.fuse.weight.detach().clone()
    gate_nz, fuse_nz, counters = [], [], []
    real = T._map_hires_loss
    cap: dict = {}

    def wrap(model_, out, batch_, device, extra, with_metrics=True):
        cap["extra"] = extra
        return real(model_, out, batch_, device, extra, with_metrics)
    monkeypatch.setattr(T, "_map_hires_loss", wrap)
    for _ in range(3):
        out = FM._loss(model, batch)
        assert bool(torch.isfinite(out["loss"]))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gate_nz.append(_nz(fz.gate.grad))
        fuse_nz.append(_nz(fz.fuse.weight.grad))
        counters.append(cap["extra"].get("n_map_hires_hist_cells025"))
        opt.step()
    assert gate_nz[0] > 0, "the gate hears the history from step 0 (identity init + fill-only)"
    assert fuse_nz[0] == 0 and fuse_nz[1] > 0 and fuse_nz[2] > 0, "the fuse weight's gradient is exactly 0 at step 0 (ReZero) and wakes after the gate's first move"
    assert float(fz.gate.abs().max()) > 0.0 and not torch.equal(fz.fuse.weight.detach(), w0)
    assert all(c is not None and c > 0 for c in counters), "the history-supervision counter is live on every step"
