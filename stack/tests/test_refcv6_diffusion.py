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
import shutil
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

# ⛔⛔ THE BASELINE REVISION, PINNED. This used to read `HEAD`, and on
# 2026-09-22 that made the whole bit-identity claim a TAUTOLOGY: `HEAD:refc.py`
# IS the worktree file, so the test compared `refc.py` against itself. MEASURED
# with the 40-char shape assertion — HEAD blob `a8d8f905…` == worktree blob
# `a8d8f905…`, and `HEAD:refc.py` carried **83** "refcv6" mentions against
# **1** in the true pre-refcv6 file. What survived was a check for uncommitted
# worktree edits, which also goes red on a comment change.
#
# ⭐ THE CLAIM ITSELF HOLDS: re-run against this revision the forward is
# BIT_IDENTICAL — same state_dict keys, zero weight mismatches, zero tensor
# diffs over 64 windows, identical `sel_tele`
# (`…/2026-09-22-refcv6-review/raw/bitidentity_baseline.json`). Only the guard
# was inert. `8c7d215` ("refcv6 core: … all of F1-F9") is the commit that
# introduced `refcv6_diffusion` into `refc.py`, found with
# `git log --reverse -S"refcv6_diffusion"`; its PARENT is the last refcv6-free
# state. The full hash is written out because `8c7d215^` is a rev-expression
# over a short hash and this must not silently re-resolve.
_BASELINE_REV = "cbadba5844a2db70a968172ab0b0bbe3a0140af0"   # == 8c7d215^
_BASELINE_CHILD = "8c7d215a18631f836c93d0f39b26f4ce79d70646"
#: MEASURED on `_BASELINE_REV:stack/tanitad/refs/refc.py` — a LITERAL, not a
#: count taken from the file under test.
_BASELINE_REFCV6_MENTIONS = 1
#: The same-breath CONTROL: a token that must read the SAME known value in the
#: baseline and in HEAD, so "0 hits" can never be a claim about a failed read.
_CONTROL_TOKEN = b"class AnchoredDiffusionDecoder"
_CONTROL_EXPECT = 1
#: A FLOOR on HEAD's count (83 at the time of writing). It only has to be big
#: enough that a baseline which accidentally resolved to HEAD cannot pass
#: `_BASELINE_REFCV6_MENTIONS == 1`.
_HEAD_REFCV6_FLOOR = 50


def _git_blob(rev: str) -> bytes:
    """``git show <rev>:refc.py`` -> bytes, or ABORT.

    ⛔ Never returns an empty bytestring as if it were content: an empty result
    from this mount is indistinguishable from a failed query, and a bit-identity
    test whose baseline is empty would fail for the wrong reason (or, with a
    lenient comparison, pass). Only a POSITIVE read is admissible.
    """
    p = subprocess.run(["git", "show", f"{rev}:{_REL}"], cwd=_ROOT,
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        err = (p.stderr or b"").decode("utf-8", errors="replace").strip()
        raise AssertionError(
            f"INVALID: could not read `{rev}:{_REL}` (rc={p.returncode}, "
            f"{len(p.stdout or b'')} bytes). This test's baseline must be a "
            f"POSITIVE read; it is not skipped, because a missing baseline "
            f"means the bit-identity claim was NOT checked. git said: {err}")
    return p.stdout


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
    """The TRUE pre-refcv6 refc.py, imported under its own name.

    ⛔ Materialised from GIT, not from a path outside the repo: a bit-identity
    claim measured against a copy someone might have edited is not a claim.
    ⛔ And pinned to :data:`_BASELINE_REV`, not to ``HEAD`` — see the comment
    there for what ``HEAD`` measured instead.
    """
    if not shutil.which("git"):                            # pragma: no cover
        pytest.skip("git is not on PATH; the baseline cannot be materialised")
    src = _git_blob(_BASELINE_REV)
    head = _git_blob("HEAD")
    # ---- the baseline is REALLY pre-refcv6, asserted, not assumed ---------- #
    n_base = src.count(b"refcv6")
    n_head = head.count(b"refcv6")
    c_base = src.count(_CONTROL_TOKEN)
    c_head = head.count(_CONTROL_TOKEN)
    assert c_base == _CONTROL_EXPECT and c_head == _CONTROL_EXPECT, (
        f"INVALID: the same-breath control read {c_base}/{c_head} instead of "
        f"{_CONTROL_EXPECT}/{_CONTROL_EXPECT}, so a count of 'refcv6' from "
        f"these bytes says nothing")
    assert n_base == _BASELINE_REFCV6_MENTIONS, (
        f"INVALID baseline: {_BASELINE_REV[:7]}:{_REL} carries {n_base} "
        f"'refcv6' mentions, expected the literal {_BASELINE_REFCV6_MENTIONS}. "
        f"This revision is not the pre-refcv6 file; the guard would be a "
        f"tautology again. (HEAD reads {n_head}.)")
    assert n_head >= _HEAD_REFCV6_FLOOR, (
        f"INVALID: HEAD:{_REL} carries only {n_head} 'refcv6' mentions "
        f"(floor {_HEAD_REFCV6_FLOOR}) — either the baseline resolved to HEAD "
        f"or refcv6 has left this file")
    assert src != head, "INVALID: the baseline blob EQUALS HEAD's blob"
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
        # ⚠️ 2026-09-23: this build used to leave `anchor_controls` at its
        # registered ZEROS while declaring `v0_conditioned=True`, so the bank
        # it compared was EXACTLY degenerate — all 5 candidates the same
        # straight line (MEASURED spread 0.000000000 m). Both sides were
        # degenerate, so the comparison was valid but WEAK: it could not have
        # seen a change in how the anchors are rolled. Real controls make the
        # 64-window comparison a comparison of a real fan, and they are what
        # the decoder's tensor-level v0 refusal now requires.
        d.anchor_controls.copy_(torch.stack(
            [torch.linspace(-2.0, 2.0, 5),
             torch.linspace(-1.5, 1.5, 5)], dim=-1))
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


# ===========================================================================
# 10. ⛔⛔ THE 2026-09-23 FIXES — each with its CONTROLS and its MUTATION arm
#
# Every guard below was written against a MEASURED defect
# (`TanitAD Research Lab/Architecture & Inference/Research/
#   2026-09-22-refcv6-review/DIFFUSION_PAPER_REVIEW.md`) and every expectation
# is a LITERAL or an analytically-known value. The deliberate-regression arms
# live in `…/2026-09-23-refcv6-fixes/code/mutation_harness.py`, which rewrites
# the SOURCE on a copied tree and records which of these go RED.
#
# ⚠️ THE INSTRUMENT ARTIFACT THAT MAKES THESE HARD TO WRITE, restated because
# it produced two wrong readings during the review: `control_head` and every
# `CascadeHeads.control_head` are ZERO-INIT (`refc.py:1705-1706`,
# `refcv6_diffusion.py:304-305`), so at construction `du = 0`, `u0_hat = x_n`,
# and THE EMITTED TRAJECTORY IS INDEPENDENT OF EVERY DECODER WEIGHT. A liveness
# probe that mutates an upstream module and watches `traj` reads 0.0 on a
# perfectly wired model. Every arm below therefore carries BOTH states: gate
# CLOSED (removability) and gate OPENED (the effect).
# ===========================================================================

from tanitad.models import refc_bev_coupling as bevc      # noqa: E402


def _ctrl_ladder(n):
    return torch.stack([torch.linspace(-2.0, 2.0, n),
                        torch.linspace(-1.5, 1.5, n)], dim=-1)


def _bev_decoder(flags=None, n=5, horizons=(5, 10, 15, 20), d_bev=6,
                 layers=2, seed=0, **cfgkw):
    """A decoder with coupling (1) ATTACHED — the build the review hooked."""
    dec = _decoder(flags=flags, n=n, horizons=horizons, layers=layers,
                   seed=seed, **cfgkw)
    built = dec.attach_bev_coupling(
        bevc.BEVCouplingConfig(enable=True, d_model=int(dec.cfg.d),
                               d_bev=d_bev, n_points=len(horizons)),
        d_bev=d_bev)
    assert built > 0, "INVALID: the coupling did not build, so 0 fires is moot"
    return dec


class _BevRecorder:
    """Counts `BEVWaypointSampler.forward` calls THROUGH the decoder forward.

    ⭐ This is the whole point of the guard: the nine existing coupling-(1)
    tests (`test_refcv6_perception.py:394-518`) all call `s(q, wp, bev)`
    STANDALONE and were green while the consumer called it ZERO times.
    """

    def __init__(self):
        self.n = 0
        self.shapes: list[tuple] = []
        self._h = []

    def attach(self, dec):
        for ly in dec.layers:
            if getattr(ly, "bev_wp", None) is not None:
                self._h.append(ly.bev_wp.register_forward_pre_hook(self._hook))
        assert self._h, "INVALID: nothing to hook — the coupling is not built"
        return self

    def _hook(self, _mod, args):
        self.n += 1
        self.shapes.append(tuple(tuple(a.shape) if torch.is_tensor(a)
                                 else None for a in args))

    def remove(self):
        for h in self._h:
            h.remove()
        self._h = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()
        return False


def _bev_inputs(dec, b=2, d_bev=6):
    torch.manual_seed(11)
    return dict(fmap=torch.randn(b, 16, 3, 5), m=torch.randn(b, 8),
                v_ms=torch.tensor([10.0, 14.0])[:b],
                bev=torch.randn(b, d_bev, 120, 64))


# ---------------------------------------------------------------------------
# FINDING 1 — coupling (1) was BUILT and NEVER CALLED
# ---------------------------------------------------------------------------
def test_coupling1_REACHES_the_sampler_and_the_classifier():
    """⛔⛔ FAILS WITHOUT THE FIX — and it did: MEASURED 2026-09-22 a forward
    hook on ``BEVWaypointSampler.forward`` fired **0 times** through
    ``AnchoredDiffusionDecoder.forward``, in the sampler AND in the classifier,
    with ``bev`` handed straight in. ``_sample``'s ``bev`` is its NINTH
    parameter and the call passed EIGHT positionals.

    The expectations are LITERALS derived from the architecture, not from the
    code: 2 layers x (1 classifier pass + 2 ladder passes) = **6**.
    """
    dec = _bev_decoder().eval()
    inp = _bev_inputs(dec)
    with _BevRecorder().attach(dec) as rec, torch.no_grad():
        dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"], bev=inp["bev"])
    assert rec.n == 6, (
        f"coupling (1) fired {rec.n} times, expected the literal 6 "
        f"(2 layers x (1 classifier + 2 denoise passes))")
    # the module is addressed BY THE CANDIDATE'S OWN WAYPOINTS: [B, N, S, 2]
    q_shape, wp_shape, bev_shape = rec.shapes[0]
    assert q_shape[0] == 2 and len(q_shape) == 3
    assert wp_shape[0] == 2 and wp_shape[2] == 4 and wp_shape[3] == 2
    assert bev_shape == (2, 6, 120, 64)


def test_coupling1_fires_on_the_CLASSIFIER_ONLY_pass():
    """⛔ The second of the three breaks: ``refc.py``'s DEFAULT classifier
    ``_decode`` omitted ``bev`` too, so even a ``steps=0`` build was dead.
    A ``ddim`` build runs the sampler either way (``_sample`` falls back to
    ``cfg.sampler_steps``), so the literal here is the same 6."""
    dec = _bev_decoder().eval()
    inp = _bev_inputs(dec)
    with _BevRecorder().attach(dec) as rec, torch.no_grad():
        dec(inp["fmap"], inp["m"], steps=0, v_ms=inp["v_ms"], bev=inp["bev"])
    assert rec.n == 6, f"classifier-path fires = {rec.n}, expected 6"


def test_coupling1_CONTROL_no_bev_means_no_call():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. The counter must read **0**
    when no BEV map is supplied — otherwise the test above is measuring the
    hook, not the wiring. (MEASURED in the review as arm C4: identical to the
    broken arm, which is exactly why 0 fires proved nothing on its own.)"""
    dec = _bev_decoder().eval()
    inp = _bev_inputs(dec)
    with _BevRecorder().attach(dec) as rec, torch.no_grad():
        dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])
    assert rec.n == 0, f"fired {rec.n} times with bev=None"


def test_coupling1_CONTROL_direct_call_fires_exactly_once():
    """⛔ CONTROL. The instrument works: one direct call, one fire. This is
    what the nine existing coupling-(1) tests measure, and it stayed GREEN
    through the entire defect."""
    dec = _bev_decoder().eval()
    q = torch.randn(2, 6, dec.cfg.d)
    wp = torch.randn(2, 6, 4, 2)
    with _BevRecorder().attach(dec) as rec, torch.no_grad():
        dec.layers[0].bev_wp(q, wp, torch.randn(2, 6, 120, 64))
    assert rec.n == 1


def test_coupling1_is_REMOVABLE_at_the_zero_init_gate():
    """⛔ THE BIT-IDENTITY HALF. ``BEVWaypointSampler.gate`` is
    ``nn.Parameter(torch.zeros(1))``, so a freshly built coupling must change
    the emitted plan by **exactly 0.0** — the fix costs the banked baseline
    nothing.

    ⚠️ The seed is pinned because ``_sample`` draws a fresh ``eps`` every
    forward (`refc.py:2538`): an unseeded comparison measures INFERENCE noise
    and made the correct fix read `BIT_IDENTICAL: false` on the review's first
    run. This is `CLAUDE.md`'s third variance in miniature."""
    dec = _bev_decoder().eval()
    inp = _bev_inputs(dec)
    with torch.no_grad():
        torch.manual_seed(1234)
        off = dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])["traj"]
        torch.manual_seed(1234)
        on = dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"],
                 bev=inp["bev"])["traj"]
    assert torch.equal(off, on), (
        f"the zero-init gate is not an identity: max |delta| = "
        f"{float((off - on).abs().max()):.9f}")


def test_coupling1_is_GATED_not_DEAD():
    """⛔ THE OTHER HALF, and the one a removability proof cannot give you: a
    gate that is an identity because it is DEAD looks exactly like a gate that
    is an identity because it is CLOSED. Open the gate AND un-zero the emitting
    head (the zero-init `control_head` bottleneck would otherwise hold `traj`
    at 0.0 delta on a perfectly wired model) — the plan must MOVE.

    ⛔⛔ ``u0_hat``, NOT ONLY ``traj``, AND THE MUTATION HARNESS IS WHY.
    MEASURED 2026-09-23: with the historical defect restored (``_sample``
    called with eight positionals) this test STAYED GREEN on ``traj`` alone —
    because the CLASSIFIER pass still reads the BEV map, its confidences move,
    the argmax moves, and a DIFFERENT candidate is emitted. That is `traj`
    moving for a reason that has nothing to do with the sampler. ``u0_hat`` is
    produced ONLY by ``_sample``, so it is the assertion that can distinguish
    them. A guard my own regression arm could not turn red was not a guard.
    """
    dec = _bev_decoder().eval()
    inp = _bev_inputs(dec)
    with torch.no_grad():
        torch.manual_seed(5)
        dec.control_head.weight.normal_(0.0, 0.5)
        torch.manual_seed(1234)
        a = dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"],
                bev=inp["bev"])
        base_t, base_u = a["traj"].clone(), a["u0_hat"].clone()
        for ly in dec.layers:
            ly.bev_wp.gate.fill_(1.0)
        torch.manual_seed(1234)
        b = dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"],
                bev=inp["bev"])
    du = float((base_u - b["u0_hat"]).abs().max())
    dt = float((base_t - b["traj"]).abs().max())
    assert du > 1e-3, (
        f"the SAMPLER's own output did not move: max |delta u0_hat| = "
        f"{du:.9f} — coupling (1) is not reaching `_sample`")
    assert dt > 1e-3, f"the emitted plan did not move: {dt:.9f}"


# ---------------------------------------------------------------------------
# FINDING 2 — the ranked score was BLIND to the emitted trajectory
# ---------------------------------------------------------------------------
def _perturb_sampler_only(dec):
    """Move ONLY the sampler's own head. The classifier surface reads
    ``conf_head`` over the classifier pass and cannot see this."""
    with torch.no_grad():
        torch.manual_seed(7)
        dec.control_head.weight.add_(torch.randn_like(
            dec.control_head.weight) * 0.5)


def _traj_and_score(dec, inp, seed=4321):
    with torch.no_grad():
        torch.manual_seed(seed)
        o = dec(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])
    return o["traj"].clone(), o["sel_score"].clone(), o["sel_tele"]


def test_DEFAULT_ranked_score_is_BLIND_to_the_emitted_trajectory():
    """⛔⛔ THE DEFECT, PINNED AS A MEASUREMENT so it can never be
    'discovered' again. On the default configuration
    ``base = refined if (sel.refined or f5_emitting_conf) else conf`` takes the
    CLASSIFIER branch, and the classifier pass runs over the RAW anchor bank.
    MEASURED 2026-09-22: a sampler-only perturbation moved ``traj`` 12.20 m and
    ``sel_score`` by **exactly 0.0**. The literal here is that 0.0."""
    dec = _decoder().eval()
    inp = _bev_inputs(dec)
    with torch.no_grad():
        torch.manual_seed(5)
        dec.control_head.weight.normal_(0.0, 0.5)   # past the zero-init gate
    t0, s0, tele0 = _traj_and_score(dec, inp)
    _perturb_sampler_only(dec)
    t1, s1, _ = _traj_and_score(dec, inp)
    assert float((t0 - t1).abs().max()) > 1e-3, (
        "INVALID: the perturbation did not move the emitted plan, so this "
        "says nothing about the ranking")
    assert float((s0 - s1).abs().max()) == 0.0, (
        f"expected the literal 0.0 — the default ranked score must be blind; "
        f"got {float((s0 - s1).abs().max()):.9f}")
    assert tele0["sampler_ranks_the_fan"] is False


def test_F5_makes_the_ranked_score_SEE_the_emitted_trajectory():
    """⛔ FAILS WITHOUT THE FEATURE, and the DD-faithful arm: DD gathers
    ``poses_reg`` by the argmax of the SAME pass's ``poses_cls``
    (``transfuser_model_v2.py:554-557``). With F5 the sampler's own last-pass
    confidence is the ranked surface, so a sampler-only perturbation MUST move
    the score."""
    dec = _decoder(rv6.DiffusionFlags(f5_emitting_conf=True)).eval()
    inp = _bev_inputs(dec)
    with torch.no_grad():
        torch.manual_seed(5)
        dec.control_head.weight.normal_(0.0, 0.5)
    t0, s0, tele0 = _traj_and_score(dec, inp)
    _perturb_sampler_only(dec)
    t1, s1, _ = _traj_and_score(dec, inp)
    assert float((t0 - t1).abs().max()) > 1e-3
    assert float((s0 - s1).abs().max()) > 0.0, (
        "the F5 ranked surface did not move with the sample")
    assert tele0["sampler_ranks_the_fan"] is True


def test_sampler_ranks_the_fan_TELEMETRY_no_longer_lies_on_an_F5_arm():
    """⛔ THE STAMP WAS FALSE ON AN ARM THAT DOES RANK THE FAN. It read
    ``bool(self.sel.refined)`` alone while the ranked surface is chosen by
    ``sel.refined OR f5_emitting_conf``. A reader tells which arm they are
    looking at from this key, so a key that contradicts the line it describes
    is worse than no key. Four literals, one per corner."""
    inp = _bev_inputs(_decoder())
    cases = {
        (False, False): False,
        (True, False): True,          # --f5-emitting-conf
        (False, True): True,          # --sel-refined
        (True, True): True,
    }
    for (f5, refined), expect in cases.items():
        d = _decoder(rv6.DiffusionFlags(f5_emitting_conf=f5)).eval()
        d.sel.refined = refined
        with torch.no_grad():
            tele = d(inp["fmap"], inp["m"], steps=2,
                     v_ms=inp["v_ms"])["sel_tele"]
        assert tele["sampler_ranks_the_fan"] is expect, (
            f"f5={f5} refined={refined} stamped "
            f"{tele['sampler_ranks_the_fan']}, expected {expect}")


def test_f5_refuse_blind_rank_REFUSES_and_is_OPT_IN():
    """⛔ The guard, and its control. Default OFF means every banked arm is
    untouched; ON, a sampler build whose ranked surface is the classifier
    REFUSES BEFORE THE COMPUTE."""
    inp = _bev_inputs(_decoder())
    blind = _decoder(rv6.DiffusionFlags(f5_refuse_blind_rank=True)).eval()
    with pytest.raises(ValueError, match="rank the fan with the CLASSIFIER"):
        blind(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])
    ok = _decoder(rv6.DiffusionFlags(f5_refuse_blind_rank=True,
                                     f5_emitting_conf=True)).eval()
    with torch.no_grad():                       # CONTROL: F5 satisfies it
        ok(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])
    ok2 = _decoder(rv6.DiffusionFlags(f5_refuse_blind_rank=True)).eval()
    ok2.sel.refined = True                      # CONTROL: --sel-refined too
    with torch.no_grad():
        ok2(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])
    off = _decoder().eval()                     # CONTROL: OFF is the default
    with torch.no_grad():
        off(inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"])
    assert rv6.DiffusionFlags().f5_refuse_blind_rank is False
    assert rv6.DiffusionFlags(f5_refuse_blind_rank=True).any_on is True


# ---------------------------------------------------------------------------
# FINDING 3 — every v0 guard tested the DECLARATION, not the TENSOR
# ---------------------------------------------------------------------------
def test_v0_refusal_READS_THE_TENSOR_not_the_flag():
    """⛔⛔ FAILS WITHOUT THE FIX. MEASURED 2026-09-22 with
    ``anchor_v0_cond=True`` and ``anchor_controls`` left at its registered
    zeros: the forward did NOT raise, F9 PASSED, and the bank was exactly
    degenerate — ``bank_max_spread_across_anchors_m 0.000000000``. Reachable
    from argv as ``--sampler ddim --anchor-v0-conditioned`` WITHOUT
    ``--anchors``."""
    dec = _decoder().eval()
    with torch.no_grad():
        dec.anchor_controls.zero_()
    dec._v0_controls_nonzero = False            # the latch must not hide it
    with pytest.raises(ValueError, match="all zeros"):
        dec(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
            v_ms=torch.tensor([10.0]))


def test_v0_degenerate_bank_spread_is_EXACTLY_ZERO_and_the_control_is_not():
    """⛔ THE ANALYTIC IDENTITY the refusal rests on, measured on the tensor the
    sampler eats. All-zero controls roll every anchor to the SAME straight
    line, so the spread across anchors is **exactly 0.0** — not small, zero.
    The CONTROL, with the same build and real controls, must read > 0."""
    dec = _decoder().eval()
    v = torch.tensor([10.0, 14.0])
    with torch.no_grad():
        real = dec.roll_bank(v, None, 2, torch.float32)
        spread_real = float((real - real[:, :1]).abs().max())
        dec.anchor_controls.zero_()
        dead = dec.roll_bank(v, None, 2, torch.float32)
        spread_dead = float((dead - dead[:, :1]).abs().max())
    assert spread_dead == 0.0, f"expected the literal 0.0, got {spread_dead!r}"
    assert spread_real > 0.0, (
        "INVALID CONTROL: the real vocabulary is degenerate too, so 0.0 above "
        "measures nothing")


def test_assert_f9_vocabulary_READS_the_controls_tensor():
    """⛔ The function's signature was ``(n_anchors, v0_conditioned, expect_n)``
    — three declarations and no tensor — so its own third message was
    unenforceable. Literals: zeros raise, non-zero passes, ``None`` keeps the
    declaration-only behaviour for a caller that has no tensor."""
    zeros = torch.zeros(7, 2)
    live = _ctrl_ladder(7)
    with pytest.raises(ValueError, match="all zeros"):
        rv6.assert_f9_vocabulary(7, True, 7, anchor_controls=zeros)
    rv6.assert_f9_vocabulary(7, True, 7, anchor_controls=live)
    rv6.assert_f9_vocabulary(7, True, 7, anchor_controls=None)
    rv6.assert_f9_vocabulary(7, True, 7)
    # CONTROL: the pre-existing declaration checks still fire first
    with pytest.raises(ValueError, match="must stay 117 anchors"):
        rv6.assert_f9_vocabulary(7, True, 117, anchor_controls=live)
    with pytest.raises(ValueError, match="v0-CONDITIONED"):
        rv6.assert_f9_vocabulary(7, False, 7, anchor_controls=live)


def test_F9_reaches_the_tensor_THROUGH_THE_FORWARD():
    """⛔ The CONSUMER-side half — the review's own lesson was that a module
    proven correct in isolation can have a consumer that never calls it."""
    dec = _decoder(rv6.DiffusionFlags(f9_assert_vocab=True, f9_n_anchors=5))
    dec = dec.eval()
    with torch.no_grad():
        dec.anchor_controls.zero_()
    dec._v0_controls_nonzero = False
    with pytest.raises(ValueError, match="all zeros"):
        dec(torch.randn(1, 16, 3, 5), torch.randn(1, 8), steps=2,
            v_ms=torch.tensor([10.0]))


# ---------------------------------------------------------------------------
# FINDING 4 — the bit-identity baseline had rotted to a tautology
# ---------------------------------------------------------------------------
def test_bitidentity_baseline_is_PINNED_and_really_pre_refcv6():
    """⛔⛔ THE GUARD ON THE GUARD. ``_pre_refcv6_module`` materialised its
    baseline from ``HEAD``, whose blob IS the worktree file — so the test
    compared ``refc.py`` against itself while carrying 83 'refcv6' mentions on
    both sides. Literals, with a same-breath control that must read a KNOWN
    value in both blobs."""
    if not shutil.which("git"):                            # pragma: no cover
        pytest.skip("git is not on PATH")
    base = _git_blob(_BASELINE_REV)
    head = _git_blob("HEAD")
    assert base.count(_CONTROL_TOKEN) == _CONTROL_EXPECT
    assert head.count(_CONTROL_TOKEN) == _CONTROL_EXPECT
    assert base.count(b"refcv6") == _BASELINE_REFCV6_MENTIONS
    assert head.count(b"refcv6") >= _HEAD_REFCV6_FLOOR
    assert base != head
    assert len(_BASELINE_REV) == 40 and len(_BASELINE_CHILD) == 40
    # the pinned parent really is `_BASELINE_CHILD`'s parent -- a POSITIVE
    # assertion, because a rev-expression over a short hash can re-resolve.
    p = subprocess.run(["git", "rev-parse", f"{_BASELINE_CHILD}^"],
                       cwd=_ROOT, capture_output=True)
    got = (p.stdout or b"").decode("utf-8", errors="replace").strip()
    assert len(got) == 40, f"INCONCLUSIVE: rev-parse returned {got!r}"
    assert got == _BASELINE_REV


def test_bitidentity_baseline_MUTATION_pointing_at_HEAD_goes_RED():
    """⛔ THE DELIBERATE REGRESSION, re-introducing the EXACT historical
    defect: point the baseline back at ``HEAD``. The guard must ABORT as
    INVALID rather than quietly comparing the file with itself."""
    import test_refcv6_diffusion as me
    old = me._BASELINE_REV
    try:
        me._BASELINE_REV = "HEAD"
        with pytest.raises(AssertionError, match="INVALID baseline"):
            me._pre_refcv6_module()
    finally:
        me._BASELINE_REV = old
    assert me._BASELINE_REV == old


def test_missing_baseline_ABORTS_rather_than_SKIPPING():
    """⛔ An arm whose anchor does not apply must be INVALID, never silently
    skipped: a skipped bit-identity test reports as 'not failed'."""
    with pytest.raises(AssertionError, match="INVALID: could not read"):
        _git_blob("refs/tanitad/definitely-not-a-revision")


# ---------------------------------------------------------------------------
# FINDING 5 — F8 clamped ONCE, OUTSIDE the ladder
# ---------------------------------------------------------------------------
def _f8_probe(dec, inp, steps=2):
    """-> (ladder_states, emit_states, step_outputs), all as ``max |x_n|``.

    ⛔ A LADDER denorm and the EMIT denorm are NOT the same statement and must
    not be pooled. ``_sample`` calls ``denorm`` twice for two different reasons:
    at the top of each pass on the current sample ``x_n`` (DD's ``img``, which
    DD clamps — ``transfuser_model_v2.py:519``), and ONCE after the loop on the
    model's x0 prediction ``x0_hat_n`` (DD's ``poses_reg``, which DD does NOT
    clamp — it is emitted straight out of the decoder at ``:545, 554-556`` and
    only ``norm_odo``'d on its way into ``scheduler.step``).

    They are separated STRUCTURALLY, not by index: a denorm is a LADDER denorm
    iff a ``_decode_ctrl`` pass follows it. Slicing ``[:-1]`` would be an
    expression over the implementation, and would silently re-pool the two the
    day an F3 cascade adds per-stage exports inside the loop.
    """
    events: list[tuple] = []
    seen_step: list[float] = []
    orig_denorm = rv6.dd_denorm_waypoints
    orig_step = dec.sched.step
    orig_ctrl = dec._decode_ctrl

    def denorm(z):
        events.append(("denorm", float(z.abs().max())))
        return orig_denorm(z)

    def step(*a, **k):
        out = orig_step(*a, **k)
        seen_step.append(float(out.abs().max()))
        return out

    def ctrl(*a, **k):
        events.append(("decode", None))
        return orig_ctrl(*a, **k)

    rv6.dd_denorm_waypoints = denorm
    dec.sched.step = step
    dec._decode_ctrl = ctrl
    try:
        with torch.no_grad():
            torch.manual_seed(1234)
            dec(inp["fmap"], inp["m"], steps=steps, v_ms=inp["v_ms"])
    finally:
        rv6.dd_denorm_waypoints = orig_denorm
        del dec.sched.step
        del dec._decode_ctrl
    ladder = [v for i, (k, v) in enumerate(events)
              if k == "denorm" and i + 1 < len(events)
              and events[i + 1][0] == "decode"]
    emit = [v for i, (k, v) in enumerate(events)
            if k == "denorm" and not (i + 1 < len(events)
                                      and events[i + 1][0] == "decode")]
    return ladder, emit, seen_step


def test_F8_clamps_INSIDE_the_ladder_like_DiffusionDrive():
    """⛔⛔ FAILS WITHOUT THE FIX. DD re-clamps at the TOP OF EVERY ITERATION —
    ``x_boxes = torch.clamp(img, min=-1, max=1)``
    (``transfuser_model_v2.py:519``) — and denormalises the CLAMPED state. Ours
    clamped ONCE, before the loop, and MEASURED 2026-09-22 ``|x_n|`` reached
    **15.04** after ``sched.step``, 15x outside DD's box.

    The expectation is the box literal **1.0**, read from the released source,
    not from our code. The DISCRIMINATING CONTROL is the second assertion: the
    pre-clamp state must EXCEED the box, or the first assertion is satisfied by
    a sample that was never out of it."""
    dec = _bev_decoder(rv6.DiffusionFlags(f8_flat_waypoint_noise=True),
                       sampler_space="metre").eval()
    with torch.no_grad():                       # past the zero-init bottleneck
        torch.manual_seed(5)
        dec.control_head.weight.normal_(0.0, 0.5)
    ladder, emit, stepped = _f8_probe(dec, _bev_inputs(dec))
    assert len(ladder) == 2, (
        f"INVALID: the [10, 0] ladder ran {len(ladder)} passes, expected the "
        f"literal 2 — there is no second pass to check")
    assert max(ladder) <= 1.0, (
        f"a ladder pass denormalised a sample OUTSIDE DD's box: "
        f"max |x_n| = {max(ladder):.6f} > 1.0  (per-pass {ladder})")
    assert stepped and max(stepped) > 1.0, (
        f"INVALID CONTROL: `sched.step` never left the box "
        f"(max {max(stepped) if stepped else float('nan'):.6f}), so the clamp "
        f"is not load-bearing on this fixture and the assertion above is "
        f"vacuous")
    # ⭐ AND THE NEGATIVE HALF, stated so nobody 'fixes' it later: DD does NOT
    # clamp what it EMITS. `best_reg` is gathered from `poses_reg` straight out
    # of the decoder (`:545, 554-556`); only the sample fed back into
    # `scheduler.step` is boxed. Our emit denorm is therefore expected to be
    # free, and on this fixture (a deliberately un-zeroed random head) it is.
    assert len(emit) == 1, f"expected exactly one emit denorm, got {emit}"


def test_F8_clamp_is_IDEMPOTENT_so_pass_zero_is_unchanged():
    """⛔ CONTROL. ``clamp`` is idempotent, so moving it inside the loop leaves
    the FIRST pass bit-identical; only passes >= 1 change. A one-step ladder is
    therefore an exact identity, which is what bounds the blast radius of this
    fix to the never-run F8 arm."""
    def run(steps):
        d = _bev_decoder(rv6.DiffusionFlags(f8_flat_waypoint_noise=True),
                         sampler_space="metre").eval()
        inp = _bev_inputs(d)
        with torch.no_grad():
            torch.manual_seed(5)
            d.control_head.weight.normal_(0.0, 0.5)
            torch.manual_seed(1234)
            return d(inp["fmap"], inp["m"], steps=steps,
                     v_ms=inp["v_ms"])["traj"]
    a, b = run(1), run(1)
    assert torch.equal(a, b), "the 1-step arm is not even reproducible"
    assert rv6.flag_stamp(rv6.DiffusionFlags(
        f8_flat_waypoint_noise=True))["f8_clamp_in_ladder"] is True


def test_F8_OFF_never_reaches_the_DD_box_at_all():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE: with F8 off the metre arm
    normalises by the derived sigma, not DD's affine box, so
    ``dd_denorm_waypoints`` must be called **exactly 0 times**. That is what
    makes the probe above a probe of the F8 path."""
    dec = _bev_decoder(sampler_space="metre").eval()
    ladder, emit, stepped = _f8_probe(dec, _bev_inputs(dec))
    assert ladder == [] and emit == [], (
        f"dd_denorm_waypoints ran {len(ladder) + len(emit)} times with F8 off")
    assert stepped, "INVALID: the ladder did not step at all"


# ---------------------------------------------------------------------------
# FINDING 1, BREAK B — no caller ever supplied the map, and a BUILT coupling
# with no map is indistinguishable from a coupling that helps nothing
# ---------------------------------------------------------------------------
def _smoke_model_with_coupling(d_bev=6):
    from tanitad.refs.refc import RefCModel, refc_smoke_config
    cfg = refc_smoke_config()
    m = RefCModel(cfg).eval()
    built = m.decoder.attach_bev_coupling(
        bevc.BEVCouplingConfig(enable=True, d_model=int(m.decoder.cfg.d),
                               d_bev=d_bev, n_points=m.decoder.n_steps),
        d_bev=d_bev)
    assert built > 0
    return m, cfg


def test_a_BUILT_coupling_with_NO_map_is_REFUSED_before_the_compute():
    """⛔⛔ THE SELF-ENFORCING HALF. A coupling that is built, counted in
    ``param_breakdown``, stamped into ``config.json`` by
    ``bev_coupling_provenance()`` and then handed NO map is EXACTLY the state
    the review measured (0 forward-hook fires) — and from the outside it is
    indistinguishable from a coupling that helps nothing. An arm in that state
    would publish 'the BEV coupling does not help' while never having had a BEV
    map, which is a refutation manufactured by a missing seam."""
    m, cfg = _smoke_model_with_coupling()
    frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    with pytest.raises(ValueError, match="no BEV map reached"):
        with torch.no_grad():
            m(frames, steps=2)


def test_the_map_REACHES_the_decoder_through_a_full_RefCModel_forward():
    """⛔ CONTROL, and the consumer-side proof one level up from the decoder:
    the same model RUNS once a map is supplied, and the sampler fires the
    coupling. The literal is again 2 layers x 3 passes = 6."""
    m, cfg = _smoke_model_with_coupling()
    frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    bev = torch.randn(2, 6, 120, 64)
    with _BevRecorder().attach(m.decoder) as rec, torch.no_grad():
        m(frames, steps=2, bev=bev)
    assert rec.n == 6, f"fired {rec.n} times through RefCModel, expected 6"


def test_a_model_WITHOUT_the_coupling_is_untouched_by_the_guard():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. The guard is scoped to builds
    that carry the coupling, so every shipped arm — none of which does — is
    bitwise unaffected and does not refuse."""
    from tanitad.refs.refc import RefCModel, refc_smoke_config
    cfg = refc_smoke_config()
    m = RefCModel(cfg).eval()
    assert m.decoder.bev_coupling_params() == 0
    frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    with torch.no_grad():
        torch.manual_seed(3)
        a = m(frames, steps=2)["traj"]
        torch.manual_seed(3)
        b = m(frames, steps=2, bev=torch.randn(2, 6, 120, 64))["traj"]
    assert torch.equal(a, b), (
        "passing a map to a build with no coupling changed the plan")


def _with_s16(m, c=5, hw=(4, 8)):
    """Give the smoke trunk a stride-16 output so the §4 perception seam is
    REACHABLE on CPU.

    ⚠️ THIS IS A DOUBLE FOR THE TRUNK, NEVER FOR THE CODE UNDER TEST. `refc.py`
    exposes `fmap_s16` only when the encoder has `forward_features` — true on
    `--trunk timm`, false on the in-repo REF-C CNN, which emits stride 32 alone.
    Without this the whole `bev_hook` branch is unreachable on this box and
    Break B would be argued rather than measured.
    """
    enc = m.encoder
    real = enc.forward

    def forward_features(x, _real=real, _c=c, _hw=hw):
        fmap, pooled = _real(x)
        return torch.zeros(x.shape[0], _c, *_hw), fmap, pooled

    enc.forward_features = forward_features
    return m


def test_BREAK_B_the_perception_branch_FEEDS_the_coupling():
    """⛔⛔ FAILS WITHOUT THE FIX. `RefCModel.forward` DECLARED `bev` and
    forwarded it, and AST-scanned, NO CALLER IN `stack/` EVER SUPPLIED IT:
    `refc_v3.py`'s two `self.core(...)` calls pass `scene_hook` / `bev_hook` /
    `bev_tokens` and never `bev`. So even with the arity fixed the coupling had
    no map. The branch's DENSE map is `bev_feats`
    (`refcv6_perception_branch.py:434`, `[B, C, X, Y]` from
    `bev_encoder.py:263`) — NOT `bev_tokens`, which is the §4 tactical
    SEQUENCE. Literal: 2 layers x 3 passes = 6, with NO explicit `bev=`."""
    m, cfg = _smoke_model_with_coupling()
    _with_s16(m)
    frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    dense = torch.randn(2, 6, 120, 64)
    with _BevRecorder().attach(m.decoder) as rec, torch.no_grad():
        out = m(frames, steps=2, bev_hook=lambda s16: {"bev_feats": dense})
    assert rec.n == 6, f"fired {rec.n} times via the hook, expected 6"
    assert out["perception"]["bev_feats"] is dense


def test_BREAK_B_refuses_TWO_suppliers_for_one_tensor():
    """⛔ CONTROL. The hook's map carries the trunk's graph and an explicit
    tensor usually does not, so silently preferring one is how an arm trains on
    the tensor it did not mean — the same refusal `bev_hook` vs `bev_tokens`
    already carries."""
    m, cfg = _smoke_model_with_coupling()
    _with_s16(m)
    frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    dense = torch.randn(2, 6, 120, 64)
    with pytest.raises(ValueError, match="BOTH an explicit"):
        with torch.no_grad():
            m(frames, steps=2, bev=dense,
              bev_hook=lambda s16: {"bev_feats": dense})


def test_BREAK_B_is_SCOPED_a_tactical_arm_with_no_coupling_is_untouched():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE, and the blast-radius bound. A
    refcv6 §4 TACTICAL arm supplies the same `bev_hook` and builds NO
    `bev_wp`; for it `bev` must stay `None` and the emitted plan must be
    bit-identical to a run with no hook at all."""
    from tanitad.refs.refc import RefCModel, refc_smoke_config
    cfg = refc_smoke_config()
    m = _with_s16(RefCModel(cfg).eval())
    assert m.decoder.bev_coupling_params() == 0
    frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
    dense = torch.randn(2, 6, 120, 64)
    with torch.no_grad():
        torch.manual_seed(3)
        a = m(frames, steps=2)["traj"]
        torch.manual_seed(3)
        b = m(frames, steps=2,
              bev_hook=lambda s16: {"bev_feats": dense})["traj"]
    assert torch.equal(a, b), (
        "the §4 hook changed a no-coupling arm's plan — the feed is not scoped")
