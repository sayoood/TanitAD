"""Pins ``tanitad.rl.ddv2_refc_chain`` — the DDv2 chain bound to a REF-C WP-4 decoder.

The strongest check is PARITY: the binding's callable, driven down the DEPLOYED ladder with the
decoder's own schedule, must reproduce ``AnchoredDiffusionDecoder._sample`` BIT FOR BIT. A
binding that differs from the model in any way (a missing residual, a different speed, a
different integrator) cannot pass it. Deliberate regressions of each are required to fail it.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs import refc
from tanitad.rl import ddv2_refc_chain as C
from tanitad.rl import ddv2_rl as D


def _dec(seed=0, n=5, horizons=(5, 10, 15, 20), **cfg_kw):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler="ddim", **cfg_kw)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(horizons), d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(n, len(horizons), 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False, grounded_selector=False,
        horizons=horizons, v0_conditioned=True, control_units="alat")
    dec.anchor_controls.copy_(torch.stack([torch.linspace(-2.0, 2.0, n),
                                           torch.linspace(-1.5, 1.5, n)], dim=-1))
    # a TRAINED-looking control head (zero-init would make x0_hat == x_t and hide bugs)
    torch.manual_seed(seed + 7)
    with torch.no_grad():
        dec.control_head.weight.normal_(0, 0.3)
        dec.control_head.bias.normal_(0, 0.3)
    return dec.eval()


def _inputs(b=2):
    g = torch.Generator().manual_seed(123)
    return (torch.randn(b, 16, 3, 5, generator=g), torch.randn(b, 8, generator=g),
            torch.tensor([12.0, 3.5][:b]))


def _captured(dec, seed=5, **fwd_kw):
    fmap, m, v0 = _inputs()
    with C.capture_sampler_inputs(dec) as rec:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = dec(fmap, m, steps=2, v_ms=v0, **fwd_kw)
    assert len(rec) == 1
    return rec[0], out


def test_parity_the_binding_reproduces_the_deployed_sampler_bitwise():
    dec = _dec()
    inp, out = _captured(dec)
    fan_ref, u0_ref, _, tele = inp.out
    assert tele["sampler_ladder"] == [10, 0]
    eps = inp.replay_eps(dec)
    with torch.no_grad():
        fan, u0 = C.native_sample(dec, inp, eps)
    assert torch.equal(u0, u0_ref)
    assert torch.equal(fan, fan_ref)
    assert torch.equal(out["u0_hat"], u0_ref)          # and that IS what forward emitted


def test_capture_restores_the_bound_method():
    dec = _dec()
    before = dec._sample.__func__
    _captured(dec)
    assert "_sample" not in dec.__dict__ and dec._sample.__func__ is before


def test_REGRESSION_a_binding_without_the_residual_breaks_parity(monkeypatch):
    dec = _dec()
    inp, _ = _captured(dec)
    eps = inp.replay_eps(dec)
    real = C.make_x0_fn

    def no_residual(decoder, inputs, *, input_clamp):
        f = real(decoder, inputs, input_clamp=input_clamp)
        return lambda x, t: f(x, t) - x                  # du only: the mutant
    monkeypatch.setattr(C, "make_x0_fn", no_residual)
    with torch.no_grad():
        _, u0 = C.native_sample(dec, inp, eps)
    assert not torch.equal(u0, inp.out[1])


def test_REGRESSION_the_wrong_speed_breaks_parity():
    dec = _dec()
    inp, _ = _captured(dec)
    eps = inp.replay_eps(dec)
    wrong = C.SamplerInputs(kv=inp.kv, cond=inp.cond, bank=inp.bank, v_ms=inp.v_ms,
                            v=torch.full_like(inp.v, dec.anchor_ref_speed), steps=inp.steps)
    with torch.no_grad():
        _, u0 = C.native_sample(dec, wrong, eps)
    assert not torch.equal(u0, inp.out[1])


def test_REGRESSION_the_release_input_clamp_is_a_different_pass_outside_the_box():
    dec = _dec()
    inp, _ = _captured(dec)
    x = torch.full((2, 5, 4, 2), 1.7)                     # outside [-1, 1]
    with torch.no_grad():
        a = C.make_x0_fn(dec, inp, input_clamp=False)(x, 10)
        b = C.make_x0_fn(dec, inp, input_clamp=True)(x, 10)
        c = C.make_x0_fn(dec, inp, input_clamp=True)(x.clamp(-1, 1), 10)
    assert not torch.equal(a, b)
    assert torch.equal(b, c)                              # the clamp IS the difference


def _group_independence(dec, **fwd_kw):
    inp, _ = _captured(dec, **fwd_kw)
    fn = C.make_x0_fn(dec, inp, input_clamp=True)
    g = torch.Generator().manual_seed(9)
    groups = [torch.randn(2, 5, 4, 2, generator=g) * 0.5 for _ in range(4)]
    others = [groups[0]] + [torch.randn(2, 5, 4, 2, generator=g) * 3 for _ in range(3)]
    with torch.no_grad():
        whole = fn(torch.cat(groups, dim=1), 14)
        whole_others = fn(torch.cat(others, dim=1), 14)
        parts = torch.cat([fn(x, 14) for x in groups], dim=1)
    return whole, whole_others, parts


def test_groups_are_independent_queries():
    """G*N queries in one call carry no cross-query information.

    EXACT form: in a call of the SAME shape, replacing every OTHER query leaves group 0's output
    BIT-IDENTICAL (MEASURED max |d| = 0.0). Cross-shape form: one G*N call vs four N calls
    agree to float32 kernel rounding (MEASURED max rel 2.0e-6 on this rig), bounded at 1e-5.
    """
    whole, whole_others, parts = _group_independence(_dec())
    assert torch.equal(whole[:, :5], whole_others[:, :5])
    assert torch.allclose(whole, parts, rtol=1e-5, atol=1e-6)


def test_REGRESSION_a_layer_that_mixes_queries_is_caught(monkeypatch):
    real = refc.CrossAttnLayer.forward

    # ⭐ `*args, **kwargs` since 2026-09-17, deliberately. This double used to
    # PIN the layer's argument list (`agent_tokens, agent_pad, agent_index`) and
    # broke the moment refcv6 gave `CrossAttnLayer.forward` its BEV and tactical
    # arguments — a TypeError in the DOUBLE that looks exactly like a failure of
    # the thing under test. The mutation being made here is to `q` alone; every
    # other argument is pass-through and nothing is gained by naming them.
    def mixing(self, q, kv, cond, *args, **kwargs):
        return real(self, q + 0.1 * q.mean(dim=1, keepdim=True), kv, cond,
                    *args, **kwargs)
    monkeypatch.setattr(refc.CrossAttnLayer, "forward", mixing)
    whole, whole_others, _ = _group_independence(_dec())
    assert not torch.equal(whole[:, :5], whole_others[:, :5])


def test_capture_refuses_agent_tokens_reaching_the_sampler():
    """refcv5-v2 ran --agents off; a chain that silently took agent tokens would be another model.
    (Added after the 2026-09-15 mutation sweep found this refusal untested: mutant C04 survived.)"""
    dec = _dec()
    fmap, m, v0 = _inputs()
    kv = dec.feat_proj(fmap.flatten(2).transpose(1, 2)).detach()
    cond = dec.cond_proj(m).detach()
    bank = dec.roll_bank(v0, None, 2, torch.float32)
    with pytest.raises(D.Ddv2ConfigError):
        with C.capture_sampler_inputs(dec):
            dec._sample(kv, cond, bank, v0, 2, agents=torch.zeros(2, 3, 32))


def test_capture_refuses_a_decoder_without_a_sampler():
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler="none")
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.randn(5, 4, 2), cfg=cfg, hierarchy=False, graft_maneuver=False,
        graft_target_latent=False, grounded_selector=False, horizons=(5, 10, 15, 20),
        v0_conditioned=True, control_units="alat")
    with pytest.raises(D.Ddv2ConfigError):
        with C.capture_sampler_inputs(dec):
            pass


SAMPLER_PATH = ("traj_proj", "time_mlp", "layers", "control_head")


def test_one_chain_update_moves_only_the_sampler_path():
    """Rollout -> per-step grad pass on a captured window: gradient reaches traj_proj /
    time_mlp / layers / control_head and NOTHING upstream of the capture (kv, cond detached)."""
    dec = _dec()
    inp, _ = _captured(dec)
    table = D.diffusers_alphas_cumprod()
    fn = C.make_x0_fn(dec, inp, input_clamp=True)
    start, _ = D.truncated_start(C.anchor_state(dec, 2), 4, table,
                                 generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.manual_seed(2)
        roll = D.rollout_chain(fn, start, table, keep_x0=True)
    adv = torch.zeros(2, 20)
    adv[0, 3], adv[1, 7] = 1.2, -1.0
    w = D.step_loss_weights(adv, len(roll["labels"]))
    for p in dec.parameters():
        p.grad = None
    for i in range(len(roll["labels"])):
        lp, x0 = D.chain_step_logprob(fn, roll["chain"], i, table, labels=roll["labels"])
        il = (C.state_to_path(dec, x0, inp.v) - 0.0).abs().mean()
        D.per_step_loss(lp, il, w, i).backward()
    for n, p in dec.named_parameters():
        on_path = n.split(".")[0] in SAMPLER_PATH
        has = p.grad is not None and float(p.grad.abs().sum()) > 0
        if n.split(".")[0] in ("feat_proj", "cond_proj", "conf_head", "offset_head"):
            assert not has, n
        if n.startswith("control_head") or n.startswith("traj_proj"):
            assert has, n
    rates = C.clamp_rates(roll["chain"], roll["x0"])
    assert set(rates) >= {"x0_clamp_frac_a_lon", "input_clamp_frac_a_lat"}


# --------------------------------------------------------------------------- #
# refcv7 NEW-1: the chain on a RESIDUAL build, where the plan is P + Delta.    #
# The Thor full-suite gate of 2026-09-27 went RED on every test above that     #
# captures: `_sample` gained `prior=` and the fixed-arity capture hook raised  #
# TypeError. The hook now forwards the prior VERBATIM and RECORDS it; these    #
# pin that the chain composes on it exactly as the deployed sampler does, and  #
# that an off build still calls `_sample` the refcv6 way.                      #
# --------------------------------------------------------------------------- #
import dataclasses  # noqa: E402

from tanitad.models import kinematic_prior as KP  # noqa: E402

RESIDUAL = {"residual_prior": "ha0_ext_pose"}


def _prior(b=2):
    """A NON-trivial prior ``(a0 [m/s^2], kappa0 [1/m])``. A zero prior would hide a chain that
    ignores it: with ``a0 = kappa0 = 0`` the residual roll IS the vocabulary's own roll."""
    return torch.tensor([0.8, -1.2][:b]), torch.tensor([0.02, -0.035][:b])


def _spy_sample_kwargs(dec, **fwd_kw):
    """The keyword arguments the decoder's forward passes to ``_sample``."""
    seen, real = [], dec._sample

    def spy(*a, **kw):
        seen.append(dict(kw))
        return real(*a, **kw)
    dec._sample = spy
    try:
        fmap, m, v0 = _inputs()
        with torch.no_grad():
            dec(fmap, m, steps=2, v_ms=v0, **fwd_kw)
    finally:
        del dec._sample
    assert len(seen) == 1
    return seen[0]


def test_NEW1_an_OFF_build_calls_sample_exactly_as_refcv6_did_and_records_no_prior():
    """OFF parity at the INTERFACE: no ``prior=`` keyword reaches ``_sample`` at all, so a
    wrapper written against the refcv6 signature keeps working on every refcv6 build."""
    dec = _dec()
    assert "prior" not in _spy_sample_kwargs(dec)
    inp, _ = _captured(dec)
    assert inp.prior is None
    # ...and on a residual build the prior DOES travel, as the one keyword
    kw = _spy_sample_kwargs(_dec(**RESIDUAL), residual_prior=_prior())
    assert set(kw) == {"prior"} and len(kw["prior"]) == 3


def test_NEW1_parity_RESIDUAL_the_binding_reproduces_the_deployed_sampler_bitwise():
    dec = _dec(**RESIDUAL)
    inp, out = _captured(dec, residual_prior=_prior())
    # the capture RECORDED the prior exactly as the forward composed it; `v` is its speed
    assert inp.prior is not None
    for got, want in zip(inp.prior, KP.prior_from_out(out)):
        assert torch.equal(got, want)
    assert inp.v is inp.prior[2]
    fan_ref, u0_ref, _, tele = inp.out
    assert tele["sampler_ladder"] == [10, 0]
    eps = inp.replay_eps(dec)
    with torch.no_grad():
        fan, u0 = C.native_sample(dec, inp, eps)
    assert torch.equal(u0, u0_ref)
    assert torch.equal(fan, fan_ref)
    assert torch.equal(out["u0_hat"], u0_ref)         # ABSOLUTE controls, as emitted


def test_NEW1_parity_RESIDUAL_holds_on_a_WITHHELD_row_where_the_prior_speed_is_not_v0():
    """On a withheld row the prior is zeroed and rolled at the bank's withheld speed, NOT the
    raw v0 -- the hook's pre-NEW-1 rule (``v = v_ms``) would break parity exactly here."""
    dec = _dec(**RESIDUAL)
    inp, _ = _captured(dec, residual_prior=_prior(), ego_keep=torch.tensor([True, False]))
    a0, k0, pv = inp.prior
    assert float(a0[1]) == 0.0 and float(k0[1]) == 0.0
    assert float(pv[1]) != float(inp.v_ms[1])         # the withheld row is not rolled at v0
    with torch.no_grad():
        fan, u0 = C.native_sample(dec, inp, inp.replay_eps(dec))
    assert torch.equal(u0, inp.out[1]) and torch.equal(fan, inp.out[0])


def test_NEW1_REGRESSION_a_chain_that_rolls_Delta_without_the_prior_breaks_parity(monkeypatch):
    dec = _dec(**RESIDUAL)
    inp, _ = _captured(dec, residual_prior=_prior())
    eps = inp.replay_eps(dec)
    real_roll = dec._roll_state

    def no_prior(z, v, metre, prior):                 # the mutant: Delta rolled as the plan
        zero = (torch.zeros_like(prior[0]), torch.zeros_like(prior[1]), prior[2])
        return real_roll(z, v, metre, zero)
    monkeypatch.setattr(dec, "_roll_state", no_prior)
    with torch.no_grad():
        fan, _ = C.native_sample(dec, inp, eps)
    assert not torch.equal(fan, inp.out[0])


def test_NEW1_REGRESSION_a_capture_that_loses_the_prior_is_REFUSED_not_rolled():
    dec = _dec(**RESIDUAL)
    inp, _ = _captured(dec, residual_prior=_prior())
    eps = inp.replay_eps(dec)
    with pytest.raises(D.Ddv2ConfigError, match="RESIDUAL"):
        C.native_sample(dec, dataclasses.replace(inp, prior=None), eps)
    wrong_v = dataclasses.replace(inp, v=torch.full_like(inp.v, dec.anchor_ref_speed))
    with pytest.raises(D.Ddv2ConfigError, match="roll speed"):
        C.native_sample(dec, wrong_v, eps)
    off = _dec()
    inp_off, _ = _captured(off)
    with pytest.raises(D.Ddv2ConfigError, match="residual_prior='off'"):
        C.state_to_path(off, C.anchor_state(off, 2), inp_off.v, inp.prior)


def test_NEW1_groups_are_independent_queries_on_a_RESIDUAL_build():
    whole, whole_others, parts = _group_independence(_dec(**RESIDUAL),
                                                     residual_prior=_prior())
    assert torch.equal(whole[:, :5], whole_others[:, :5])
    assert torch.allclose(whole, parts, rtol=1e-5, atol=1e-6)
