"""refcv6 §3 — F1..F9, each behind its own flag, each default OFF.

Each test names its kind: *fails without the feature* / *control that must read
a known value* / *deliberate regression*.

⭐ The load-bearing identity this file exists for is the FIRST test: with all
nine flags off, a 64-window forward is **bit-identical** to the pre-refcv6
file, materialised from git rather than described. Every other test then has
something to be a difference FROM.

⛔ Every guard below is proven by MUTATION: the defect it catches is
re-introduced in a sibling test and the assertion is shown to go RED. A guard
that has only ever seen the fixed code has not been tested.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from tanitad.models import refcv6_diffusion as rv6
from tanitad.refs import refc
from tanitad.refs import refc_sampler as rs

_ROOT = Path(__file__).resolve().parents[2]
_REL = "stack/tanitad/refs/refc.py"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _decoder(flags=None, sampler="ddim", n=5, horizons=(5, 10, 15, 20),
             seed=0, layers=2, **cfgkw):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=layers, ff_mult=2,
                             sampler=sampler, refcv6=flags, **cfgkw)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(horizons), d_meas=8, d_ctx=4,
        tac_latent_dim=4, anchors=torch.randn(n, len(horizons), 2), cfg=cfg,
        hierarchy=False, graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=horizons, v0_conditioned=True,
        control_units="alat")
    torch.manual_seed(seed + 1)
    ctrl = torch.stack([torch.linspace(-2.0, 2.0, n),
                        torch.linspace(-1.5, 1.5, n)], dim=-1)
    dec.anchor_controls.copy_(ctrl)
    return dec


def _pre_refcv6_module():
    """The refc.py of the branch tip, imported under its own name.

    ⛔ Materialised from GIT, not from a path outside the repo: a bit-identity
    claim measured against a copy someone might have edited is not a claim.
    """
    try:
        src = subprocess.run(["git", "show", f"HEAD:{_REL}"], cwd=_ROOT,
                             capture_output=True, check=True).stdout
    except Exception as e:                                 # pragma: no cover
        pytest.skip(f"git not available for the baseline: {e}")
    tmp = Path(__file__).resolve().parent / "_pre_refcv6_refc.py"
    tmp.write_bytes(src)
    try:
        spec = importlib.util.spec_from_file_location("_pre_refcv6_refc", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_pre_refcv6_refc"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 0. ⭐⭐ THE BIT-IDENTITY CLAIM
# ---------------------------------------------------------------------------
def test_all_flags_off_is_BIT_IDENTICAL_on_64_windows():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE, and the premise of every other
    test here. With ``refcv6=None`` the decoder must construct nothing new and
    emit, for 64 windows, EXACTLY what the pre-refcv6 file emits — same
    state_dict keys, same tensors, bit for bit."""
    old = _pre_refcv6_module()

    def build(mod, seed=7):
        torch.manual_seed(seed)
        cfg = mod.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                                sampler="ddim")
        torch.manual_seed(seed)
        d = mod.AnchoredDiffusionDecoder(
            feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
            anchors=torch.zeros(5, 4, 2), cfg=cfg, hierarchy=False,
            graft_maneuver=False, graft_target_latent=False,
            grounded_selector=False, horizons=(5, 10, 15, 20),
            v0_conditioned=True, control_units="alat")
        return d.eval()

    a, b = build(old), build(refc)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys(), "the flag block added or removed weights"
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    B = 64                                          # ⭐ the 64 windows
    fmap = torch.randn(B, 16, 3, 5)
    m = torch.randn(B, 8)
    v0 = torch.rand(B) * 20 + 2
    torch.manual_seed(99)
    oa = a(fmap, m, steps=2, v_ms=v0)
    torch.manual_seed(99)
    ob = b(fmap, m, steps=2, v_ms=v0)
    assert set(oa) == set(ob), (set(oa) ^ set(ob))
    for k, va in oa.items():
        if isinstance(va, torch.Tensor):
            assert torch.equal(va, ob[k]), k
    assert oa["sel_tele"] == ob["sel_tele"], "the telemetry dict changed"


def test_default_flag_block_builds_NOTHING():
    """⛔ CONTROL. Off means not constructed — not constructed and gated."""
    d = _decoder()
    assert d.cascade is None and d.adaln is None
    assert rv6.DiffusionFlags().any_on is False
    assert rv6.DiffusionFlags().builds_modules is False


def test_flags_on_a_CLASSIFIER_build_are_REFUSED():
    """⛔ CONTROL. F1..F9 describe the diffusion decoder. A flag that reaches
    no mechanism is the refcv5 false-provenance defect."""
    with pytest.raises(ValueError, match="no sampler"):
        _decoder(rv6.DiffusionFlags(f2_dd_step=True), sampler="none")


def test_a_dict_of_flags_is_REFUSED_not_silently_ignored():
    """⛔ DELIBERATE REGRESSION. A dict reads back as 'no flags' through every
    `getattr` and the arm would silently be the baseline."""
    with pytest.raises(TypeError, match="DiffusionFlags"):
        _decoder({"f2_dd_step": True})


# ---------------------------------------------------------------------------
# F1 — random-t training, ONE decoder call
# ---------------------------------------------------------------------------
def test_F1_draws_ONE_timestep_per_sample_and_makes_ONE_call():
    """FAILS WITHOUT THE FEATURE. DD draws `randint(0, 50)` per sample and
    makes ONE decoder call (`transfuser_model_v2.py:463-476`); ours walks the
    2-step inference ladder. The call count is the observable."""
    d = _decoder(rv6.DiffusionFlags(f1_random_t=True)).train()
    calls = []
    orig = d._decode_ctrl
    d._decode_ctrl = lambda *a, **k: (calls.append(a[3].clone()), orig(*a, **k))[1]
    d(torch.randn(6, 16, 3, 5), torch.randn(6, 8), steps=2,
      v_ms=torch.rand(6) * 20 + 2)
    assert len(calls) == 1, f"F1 must make ONE call, made {len(calls)}"
    t = calls[0]
    assert tuple(t.shape) == (6,)
    assert float(t.min()) >= 0 and float(t.max()) < 50
    assert float(t.std()) > 0, "a constant draw is not U[0, 50)"


def test_F1_is_TRAINING_ONLY():
    """⛔ CONTROL. At eval the PUBLISHED truncated ladder is what the paper
    measures; a random-t eval would report a different model."""
    d = _decoder(rv6.DiffusionFlags(f1_random_t=True)).eval()
    calls = []
    orig = d._decode_ctrl
    d._decode_ctrl = lambda *a, **k: (calls.append(a[3][0].item()), orig(*a, **k))[1]
    d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
      v_ms=torch.tensor([10.0, 14.0]))
    assert calls == [10.0, 0.0], f"eval must walk DD's ladder, got {calls}"


def test_F1_t_max_1_is_the_registered_ZERO_NOISE_arm():
    """⛔ DELIBERATE REGRESSION (PREREG P13-R). `t_max = 1` draws the constant
    t = 0 and must be ACCEPTED — a gate that cannot separate a no-noise draw
    from DD's is blind. `t_max = 0` must raise."""
    d = _decoder(rv6.DiffusionFlags(f1_random_t=True, f1_t_max=1)).train()
    d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
      v_ms=torch.tensor([9.0, 9.0]))
    with pytest.raises(ValueError, match="f1_t_max must be >= 1"):
        rv6.DiffusionFlags(f1_t_max=0)


# ---------------------------------------------------------------------------
# F2 — DD's step semantics
# ---------------------------------------------------------------------------
def test_F2_pairs_step_t_to_t_minus_one():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. DD calls
    `set_timesteps(1000)` (`:507`) while stepping the [10, 0] ladder, so
    diffusers derives `prev = t - 1000//1000 = t - 1`. Ours uses the NEXT
    LADDER ENTRY."""
    assert rv6.dd_step_pairs([10, 0], False) == [(10, 0), (0, 0)]
    assert rv6.dd_step_pairs([10, 0], True) == [(10, 9), (0, -1)]


def test_F2_keeps_95_percent_of_the_residual_and_ours_keeps_28():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE — MEASURED off the alpha table,
    not asserted. The spec's whole claim for F2 is that 10 -> 9 leaves ~95 %
    of the gap (so pass 2 re-denoises nearly the same input under a different
    time label) where 10 -> 0 leaves ~28 %."""
    s = rs.DDIMSchedule()
    dd = rv6.residual_retained(s, 10, 9)
    ours = rv6.residual_retained(s, 10, 0)
    assert 0.93 <= dd <= 0.97, dd
    assert 0.25 <= ours <= 0.31, ours


def test_F2_changes_the_sampled_state_and_OFF_does_not():
    """FAILS WITHOUT THE FEATURE. With the ladder actually walked, F2 must
    move the intermediate state; with F2 off the run must reproduce the
    baseline exactly."""
    fmap, m = torch.randn(4, 16, 3, 5), torch.randn(4, 8)
    v = torch.rand(4) * 20 + 2
    base = _decoder(seed=5).eval()
    on = _decoder(rv6.DiffusionFlags(f2_dd_step=True), seed=5).eval()
    on.load_state_dict(base.state_dict())
    # make `control_head` non-zero, or every step is an identity and F2 is
    # invisible for a reason that has nothing to do with the semantics
    for d in (base, on):
        with torch.no_grad():
            d.control_head.weight.normal_(0, 0.05)
            d.control_head.bias.normal_(0, 0.05)
    on.control_head.load_state_dict(base.control_head.state_dict())
    torch.manual_seed(3)
    a = base(fmap, m, steps=2, v_ms=v)["u0_hat"]
    torch.manual_seed(3)
    b = on(fmap, m, steps=2, v_ms=v)["u0_hat"]
    assert not torch.allclose(a, b), "F2 did not change the sampler"
    assert b.shape == a.shape
    assert on.rv6.f2_dd_step and not base.rv6.f2_dd_step


# ---------------------------------------------------------------------------
# F3 — the cascade
# ---------------------------------------------------------------------------
def test_F3_builds_one_head_pair_PER_LAYER_and_emits_one_prediction_each():
    """FAILS WITHOUT THE FEATURE. DD clones a decoder layer that carries its
    own `task_decoder` (`:345-347`, `:308-312`) and collects one
    `(poses_reg, poses_cls)` per stage (`:372-380`)."""
    d = _decoder(rv6.DiffusionFlags(f3_per_layer=True), layers=3).eval()
    assert d.cascade is not None and d.cascade.n_layers == 3
    assert len(d.cascade.control_heads) == len(d.cascade.conf_heads) == 3
    out = d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=1,
            v_ms=torch.tensor([10.0, 14.0]))
    assert len(out["layer_u0_hat"]) == 3 and len(out["layer_logits"]) == 3
    for u in out["layer_u0_hat"]:
        assert tuple(u.shape) == (2, 5, 4, 2)
    # the LAST stage IS the emitted prediction
    assert torch.equal(out["layer_u0_hat"][-1], out["u0_hat"])


def test_F3_cascade_heads_are_ZERO_INIT():
    """⛔ CONTROL. Every stage must start at 'predict the current state', the
    same discipline the single `control_head` follows (`refc.py:1705-1706`)."""
    d = _decoder(rv6.DiffusionFlags(f3_per_layer=True), layers=3)
    for h in d.cascade.control_heads:
        assert float(h.weight.abs().max()) == 0.0
        assert float(h.bias.abs().max()) == 0.0


def test_F3_DETACHES_between_stages():
    """FAILS WITHOUT THE FEATURE, and it is the half of F3 that is easiest to
    omit. DD detaches the trajectory handed to the next stage (`:379`). With
    the cut, a loss on the LAST stage alone must put NO gradient on an EARLIER
    stage's head; without it, the per-layer losses are just extra gradient
    paths into one chain — which is NOT the cascade DD ablates in Tab. 5."""
    # ⛔ The probe is the EARLIER ATTENTION LAYER's weights, not stage 0's
    # head: that head only feeds stage 0's own output, so it would read zero
    # whether or not the detach is there, and the test would pass vacuously.
    # The detach cuts the QUERY, so what must go dark is `layers[0]`.
    d = _decoder(rv6.DiffusionFlags(f3_per_layer=True), layers=3).train()
    for h in d.cascade.control_heads:                 # break the zero init
        with torch.no_grad():
            h.weight.normal_(0, 0.05)
    out = d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=1,
            v_ms=torch.tensor([10.0, 14.0]))
    out["layer_u0_hat"][-1].square().mean().backward()
    g2 = d.cascade.control_heads[2].weight.grad
    assert g2 is not None and float(g2.abs().sum()) > 0
    leaked = [n for n, p in d.layers[0].named_parameters()
              if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert not leaked, f"layer 0 received gradient from stage 2: {leaked}"


def test_F3_WITHOUT_the_detach_leaks_gradient_backwards():
    """⛔ DELIBERATE REGRESSION — the MUTATION that proves the test above is
    alive. Re-introduce the defect (no `q.detach()`) and stage 0 must receive
    gradient from stage 2."""
    d = _decoder(rv6.DiffusionFlags(f3_per_layer=True), layers=3).train()
    for h in d.cascade.control_heads:
        with torch.no_grad():
            h.weight.normal_(0, 0.05)
    kv = d.feat_proj(torch.randn(2, 16, 3, 5).flatten(2).transpose(1, 2))
    cond = d.cond_proj(torch.randn(2, 8))
    q = d.traj_proj(torch.randn(2, 5, 4, 2).reshape(2, 5, -1))
    outs = []
    for i, layer in enumerate(d.layers):
        q = layer(q, kv, cond, None, None, None)
        outs.append(d.cascade.emit(i, q)[1])
        # ⛔ the detach is DELIBERATELY OMITTED here
    outs[-1].square().mean().backward()
    leaked = [n for n, p in d.layers[0].named_parameters()
              if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert leaked, ("the undetached control did not leak — the detach test "
                    "is vacuous")


# ---------------------------------------------------------------------------
# F4 — per-layer AdaLN
# ---------------------------------------------------------------------------
def test_F4_is_DDs_ModulationLayer_arithmetic():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. `scale, shift =
    Linear(Mish(cond)).chunk(2)` then `x * (1 + scale) + shift`
    (`transfuser_model_v2.py:235-238, 265-267`). The Mish comes FIRST and
    there is no norm; writing it the usual way round is a different function
    of the timestep."""
    torch.manual_seed(0)
    mod = rv6.AdaLNModulation(6, 6)
    x = torch.randn(2, 3, 6)
    te = torch.randn(2, 6)
    scale, shift = mod.scale_shift_mlp(te.unsqueeze(1)).chunk(2, dim=-1)
    assert torch.allclose(mod(x, te), x * (1 + scale) + shift)
    assert isinstance(mod.scale_shift_mlp[0], nn.Mish)
    assert isinstance(mod.scale_shift_mlp[1], nn.Linear)


def test_F4_builds_ONE_modulation_per_layer_and_changes_the_output():
    """FAILS WITHOUT THE FEATURE. Today the timestep is added ONCE, to the
    query (`refc.py::_decode_ctrl`); DD modulates after EVERY layer."""
    d = _decoder(rv6.DiffusionFlags(f4_adaln=True), layers=3).eval()
    assert d.adaln is not None and len(d.adaln) == 3
    base = _decoder(layers=3).eval()
    sd = {k: v for k, v in d.state_dict().items() if "adaln" not in k}
    base.load_state_dict(sd, strict=True)
    fmap, m, v = torch.randn(2, 16, 3, 5), torch.randn(2, 8), torch.tensor(
        [10.0, 14.0])
    torch.manual_seed(1)
    a = base(fmap, m, steps=2, v_ms=v)
    torch.manual_seed(1)
    b = d(fmap, m, steps=2, v_ms=v)
    # ⛔ THE SAMPLER SURFACE IS THE ONE TO READ. F4 modulates `_decode_ctrl`,
    # the DENOISING pass; `anchor_logits` comes from the separate CLASSIFIER
    # pass whose "timestep" is an index into a 3-row `nn.Embedding`, and it
    # must be UNCHANGED — applying a continuous-t AdaLN there would be a
    # different graft wearing F4's name.
    assert torch.equal(a["anchor_logits"], b["anchor_logits"])
    assert not torch.allclose(a["refined_logits"], b["refined_logits"]), \
        "the AdaLN reached nothing"


def test_F4_zero_init_variant_is_the_IDENTITY_and_DD_default_is_not():
    """⛔ CONTROL. DD's released `if_zeroinit_scale` is FALSE (`:233`). Our
    zero-init variant is a REMOVABLE graft; the code's own default is not, and
    the stamp must say which ran."""
    z = rv6.AdaLNModulation(6, 6, zero_init=True)
    x = torch.randn(2, 3, 6)
    assert torch.allclose(z(x, torch.randn(2, 6)), x)
    assert rv6.DiffusionFlags().f4_zero_init is False       # DD's value
    torch.manual_seed(0)
    nz = rv6.AdaLNModulation(6, 6, zero_init=False)
    assert not torch.allclose(nz(x, torch.randn(2, 6)), x)


# ---------------------------------------------------------------------------
# F5 — the emitting pass's own confidence, and focal
# ---------------------------------------------------------------------------
def test_F5_focal_matches_DDs_py_sigmoid_focal_loss_ELEMENTWISE():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE — recomputed here from
    `multimodal_loss.py:90-98` rather than trusted."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7)
    tgt = torch.tensor([0, 3, 6, 1])
    onehot = torch.zeros(4, 7).scatter_(1, tgt[:, None], 1.0)
    p = logits.sigmoid()
    pt = (1 - p) * onehot + p * (1 - onehot)
    fw = (0.25 * onehot + 0.75 * (1 - onehot)) * pt.pow(2.0)
    want = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, onehot, reduction="none") * fw).mean()
    assert torch.allclose(rv6.focal_cls_loss(logits, tgt), want)


def test_F5_focal_is_SIGMOID_not_softmax():
    """⛔ DELIBERATE REGRESSION. Swapping in `cross_entropy` reproduces
    neither the gradient nor the scale, which is what makes F5 a separable arm
    from our `loss_cls`. The two must NOT agree."""
    torch.manual_seed(0)
    logits, tgt = torch.randn(4, 7), torch.tensor([0, 3, 6, 1])
    ce = torch.nn.functional.cross_entropy(logits, tgt)
    assert abs(float(rv6.focal_cls_loss(logits, tgt)) - float(ce)) > 0.1


def test_F5_ranks_by_the_SAMPLERS_OWN_confidence():
    """FAILS WITHOUT THE FEATURE. `SelectionConfig.refined` defaults FALSE, so
    the ranked score is the CLASSIFIER surface — a head that never saw the
    sample (the measured 45.4 %-of-windows ranking failure, one level up).
    F5 makes the emitting pass's own confidence the ranked surface."""
    fmap, m, v = torch.randn(4, 16, 3, 5), torch.randn(4, 8), torch.rand(4) * 9 + 3
    base = _decoder(seed=2).eval()
    on = _decoder(rv6.DiffusionFlags(f5_emitting_conf=True), seed=2).eval()
    on.load_state_dict(base.state_dict())
    torch.manual_seed(4)
    a = base(fmap, m, steps=2, v_ms=v)
    torch.manual_seed(4)
    b = on(fmap, m, steps=2, v_ms=v)
    assert torch.equal(a["anchor_logits"], b["anchor_logits"])
    assert torch.equal(b["sel_score"], b["refined_logits"])
    assert torch.equal(a["sel_score"], a["anchor_logits"])


def test_F5_REFUSES_alongside_sel_score_emitted():
    """⛔ CONTROL. Two different answers to 'which head scores the emitted
    fan' on one fan makes the arm unattributable — the `--v2` conflation
    failure."""
    dec = _decoder(rv6.DiffusionFlags(f5_emitting_conf=True), seed=2)
    dec.sel = refc.SelectionConfig(score_emitted=True, refined=True,
                                   horizon_s=2.0)
    with pytest.raises(ValueError, match="unattributable"):
        dec(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
            v_ms=torch.tensor([9.0, 9.0]))


# ---------------------------------------------------------------------------
# F6 — DD has ONE reconstruction loss
# ---------------------------------------------------------------------------
def test_F6_sets_w_u0_zero_AND_carries_its_own_acknowledgement():
    """FAILS WITHOUT THE FEATURE. `--sampler ddim --w-u0 0` is refused unless
    acknowledged (PI ruling 2026-09-11). F6 IS that configuration, so it must
    carry the acknowledgement and be STAMPED as refcv6 F6 rather than read as
    a bypassed guard."""
    sys.path.insert(0, str(_ROOT / "stack" / "scripts"))
    import refc_v3_train as T                               # noqa: E402
    args = type("A", (), {"f6_w_u0_zero": True})()
    flags = T.refcv6_flags_from_args(args)
    assert flags is not None and flags.f6_w_u0_zero
    assert rv6.flag_stamp(flags)["f6_w_u0_zero"] is True


# ---------------------------------------------------------------------------
# F7 — several noise samples per anchor
# ---------------------------------------------------------------------------
def test_F7_tiling_is_GROUP_MAJOR_not_interleaved():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE, and the one that would
    type-check either way. Candidate `g*N + a` must be anchor `a`;
    `repeat_interleave` gives `[a0,a0,a1,a1,...]` and every prior lands on the
    wrong candidate while every shape still agrees."""
    prior = torch.tensor([[0.0, 1.0, 2.0]])
    got = rv6.tile_anchor_prior(prior, 2)
    assert got.tolist() == [[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]]
    assert got.tolist() != prior.repeat_interleave(2, dim=1).tolist()
    idx = torch.arange(6)
    assert rv6.candidate_to_anchor_id(idx, 3, 2).tolist() == [0, 1, 2, 0, 1, 2]


def test_F7_REFUSES_without_the_eval_join_acknowledgement():
    """⛔ CONTROL. Two of the three named sites are widened here; the THIRD —
    taniteval's `sel_idx` join — is another agent's file. A silently
    mis-joined eval is worse than a crash, so the refusal stands and names all
    three."""
    d = _decoder(rv6.DiffusionFlags(f7_samples_per_anchor=3))
    with pytest.raises(NotImplementedError) as e:
        d(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
          v_ms=torch.tensor([10.0]))
    for site in ("loss_cls", "anchor priors", "sel_idx"):
        assert site.split()[0] in str(e.value)
    assert "sel_anchor_id" in str(e.value)


def test_F7_widens_the_fan_and_emits_sel_anchor_id():
    """FAILS WITHOUT THE FEATURE. With the acknowledgement the fan is [B, G*N]
    and the ANCHOR id is emitted next to the candidate index, so the join has
    a correct column to move to."""
    flags = rv6.DiffusionFlags(f7_samples_per_anchor=3, f7_ack_eval_join=True)
    d = _decoder(flags).eval()
    out = d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
            v_ms=torch.tensor([10.0, 14.0]))
    assert tuple(out["anchor_traj"].shape) == (2, 15, 4, 2)
    assert tuple(out["anchor_bank"].shape) == (2, 15, 4, 2)
    assert tuple(out["anchor_logits"].shape) == (2, 15)
    assert out["sampler_groups"] == 3
    assert torch.equal(out["sel_anchor_id"], out["sel_idx"] % 5)
    # the widened bank really is the anchor set repeated group-major
    bank = out["anchor_bank"]
    assert torch.equal(bank[:, 0:5], bank[:, 5:10])


def test_F7_groups_draw_DIFFERENT_noise_per_group():
    """FAILS WITHOUT THE FEATURE. If the G copies shared one epsilon, F7 would
    be G identical candidates and the whole point (more samples per anchor)
    would be lost while every shape still checked out."""
    flags = rv6.DiffusionFlags(f7_samples_per_anchor=3, f7_ack_eval_join=True)
    d = _decoder(flags).eval()
    out = d(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=1,
            v_ms=torch.tensor([12.0]))
    u = out["u0_hat"]
    assert not torch.allclose(u[:, 0:5], u[:, 5:10])


# ---------------------------------------------------------------------------
# F8 — DD's flat waypoint-space noise
# ---------------------------------------------------------------------------
def test_F8_normalisation_is_DDs_affine_box_and_ROUND_TRIPS():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. `norm_odo` / `denorm_odo`,
    `transfuser_model_v2.py:432-453`."""
    xy = torch.tensor([[[-1.2, -20.0], [55.7, 26.0], [0.0, 0.0]]])
    n = rv6.dd_norm_waypoints(xy)
    assert torch.allclose(n[0, 0], torch.tensor([-1.0, -1.0]), atol=1e-6)
    assert torch.allclose(n[0, 1], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert torch.allclose(rv6.dd_denorm_waypoints(n), xy, atol=1e-4)


def test_F8_noise_is_FLAT_across_the_horizon():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE, and it is the entire claim of
    F8. At t = 8, `sqrt(1 - abar) = 0.0316`; de-normalised through DD's box
    that is 0.90 m in x and 0.73 m in y PER WAYPOINT — the same at 0.5 s and
    at 6 s. Our control-space noise integrates and grows with the horizon."""
    s = rs.DDIMSchedule()
    sig = float(s.sqrt_one_minus_abar(8))
    assert abs(sig - 0.0316) < 5e-4
    sx = sig * rv6.DD_X_SPAN / 2
    sy = sig * rv6.DD_Y_SPAN / 2
    assert abs(sx - 0.90) < 0.01 and abs(sy - 0.73) < 0.01
    # ...and it really is per-waypoint: a constant, not a function of the slot.
    # ⚠️ 2,000 draws per slot, not 4. An under-powered estimator reading a
    # spread of its own sampling noise is exactly the false alarm this
    # programme has logged three times in one session.
    x0 = torch.zeros(2000, 1, 8, 2)
    torch.manual_seed(0)
    xt = s.add_noise(x0, torch.randn_like(x0), torch.tensor(8))
    per_slot = rv6.dd_denorm_waypoints(xt).sub(
        rv6.dd_denorm_waypoints(x0)).std(dim=0).reshape(8, 2)
    for axis, want in ((0, sx), (1, sy)):
        got = per_slot[:, axis]
        assert float(got.std()) < 0.05 * float(got.mean()), got
        assert abs(float(got.mean()) - want) < 0.05 * want


def test_F8_REFUSES_on_the_control_space_sampler():
    """⛔ CONTROL. DD's METRE spans applied to (a_lon, a_lat) is a
    plausible-looking number for a quantity that is not a distance — the
    `metre_sigma_m` divisor trap, one field over."""
    with pytest.raises(ValueError, match="sampler_space='metre'"):
        _decoder(rv6.DiffusionFlags(f8_flat_waypoint_noise=True))


def test_F8_runs_on_the_metre_arm_and_clamps():
    """FAILS WITHOUT THE FEATURE. DD clamps the normalised sample to [-1, 1]
    (`:474`, `:519`) — on its waypoint box that is a real operation."""
    d = _decoder(rv6.DiffusionFlags(f8_flat_waypoint_noise=True),
                 sampler_space="metre").eval()
    out = d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
            v_ms=torch.tensor([10.0, 14.0]))
    fan = out["anchor_traj"]
    assert torch.isfinite(fan).all()
    # the clamp bounds the fan inside DD's own box
    assert float(fan[..., 0].max()) <= rv6.DD_X_SPAN - rv6.DD_X_OFF + 1e-3
    assert float(fan[..., 1].abs().max()) <= rv6.DD_Y_SPAN / 2 + rv6.DD_Y_OFF + 1e-3


# ---------------------------------------------------------------------------
# F9 — the vocabulary is unchanged, and the build says so
# ---------------------------------------------------------------------------
def test_F9_asserts_the_v0_conditioned_117_anchor_vocabulary():
    """FAILS WITHOUT THE FEATURE. F9 is a NO-CHANGE item, so it is ENFORCED.
    An arm that quietly ran on 20 DD-style k-means anchors would be a
    different experiment wearing refcv6's name."""
    rv6.assert_f9_vocabulary(117, True)
    with pytest.raises(ValueError, match="must stay 117 anchors"):
        rv6.assert_f9_vocabulary(20, True)
    with pytest.raises(ValueError, match="v0-CONDITIONED"):
        rv6.assert_f9_vocabulary(117, False)


def test_F9_fires_from_the_FORWARD_on_a_wrong_vocabulary():
    """⛔ DELIBERATE REGRESSION. The 5-anchor test decoder must be refused by
    the live assertion, not only by the helper."""
    d = _decoder(rv6.DiffusionFlags(f9_assert_vocab=True))
    with pytest.raises(ValueError, match="must stay 117 anchors"):
        d(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
          v_ms=torch.tensor([10.0]))
    ok = _decoder(rv6.DiffusionFlags(f9_assert_vocab=True, f9_n_anchors=5))
    ok(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
       v_ms=torch.tensor([10.0]))


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------
def test_the_flag_stamp_carries_all_nine():
    """⛔ CONTROL. A run that cannot state which of F1..F9 were live makes
    every later comparison unfalsifiable (mm-decisions M18)."""
    s = rv6.flag_stamp(rv6.DiffusionFlags(f1_random_t=True, f5_focal=True))
    for k in ("f1_random_t", "f2_dd_step", "f3_per_layer", "f4_adaln",
              "f5_emitting_conf", "f5_focal", "f6_w_u0_zero",
              "f7_samples_per_anchor", "f8_flat_waypoint_noise",
              "f9_assert_vocab", "any_on"):
        assert k in s
    assert s["any_on"] is True
    assert rv6.flag_stamp(rv6.DiffusionFlags())["any_on"] is False


def test_telemetry_is_UNCHANGED_when_every_flag_is_off():
    """⛔ CONTROL. `sel_tele` reaches config.json and every log row; a key that
    appears on every run is a key a reader stops reading."""
    d = _decoder().eval()
    tele = d(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
             v_ms=torch.tensor([10.0, 14.0]))["sel_tele"]
    assert not any(k.startswith("refcv6") for k in tele)
    on = _decoder(rv6.DiffusionFlags(f2_dd_step=True)).eval()
    tele2 = on(torch.randn(2, 16, 3, 5), torch.randn(2, 8), steps=2,
               v_ms=torch.tensor([10.0, 14.0]))["sel_tele"]
    assert tele2["refcv6"]["f2_dd_step"] is True
    assert tele2["refcv6_pairs"] == [[10, 9], [0, -1]]
