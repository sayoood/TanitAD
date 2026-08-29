"""MM-E4 drift-lever trainer features (PREREG_DRIFT_ATTACK_LADDER + Amendment A).

One flag per ladder cell, composable:
  L1  --o6-innovation [--o6-innovation-shuffle]   (innovation-SIGReg)
  L2  --o5-target frozen                          (frozen-teacher targets)
  L4  --o5-target-crop <frac>                     (azimuthal-crop targets)

⛔ LOAD-BEARING: ``test_the_defaults_are_bit_identical`` (all three off ⇒ no
new params, no new state_dict keys, and the loss step routes through the
incumbent path bit-for-bit) and
``test_resync_after_init_load_copies_live_into_frozen`` — without the resync,
an ``--init-from`` arm trains against a FROZEN RANDOM teacher and the run
looks healthy while its target is noise (the EMA suite's lesson, applied to
the teacher that never moves at all).

The innovation tests pin the prereg's ANALYTIC anchor: z_{t+1} == z_t makes
the innovations exactly 0 — a point mass the sketched test penalises hard
where plain SIGReg on the same states reads near-null — and the CROSS-FIT
contract (never fit and scored on the same rows, the 2026-08-22 probe rules)
via a two-regime batch a leaky fit would zero out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.models.sigreg import SigReg  # noqa: E402
from train_v6_staged import (  # noqa: E402
    STAGE_MAY_INTRODUCE, _resync_ema_o5, _weights_from_args,
    azimuthal_target_crop, build_parser, build_stack_from_args,
    o6_innovation_rows, o6_sigreg_loss, synthetic_train_batch, v6_loss_step,
)

BASE = ["--out", "UNUSED", "--stage", "S-W", "--frame-h", "64",
        "--frame-w", "160", "--enc-dim", "32", "--enc-depth", "1",
        "--enc-heads", "2", "--readout-grid", "2", "--readout-grid-w", "4",
        "--readout-dim", "16", "--pred-dim", "32", "--pred-depth", "1",
        "--pred-heads", "2", "--window", "4", "--d-tac", "16", "--d-str", "8"]


def _args(*extra):
    return build_parser().parse_args(BASE + list(extra))


def _stack(*extra):
    return build_stack_from_args(_args(*extra))


def _batch(st, *, batch=2, k=4, seed=7):
    b = synthetic_train_batch(st, batch=batch, k=k, seed=seed)
    b["gt_wp"] = torch.randn(batch, 3, 2,
                             generator=torch.Generator().manual_seed(seed + 1))
    return b


# ===========================================================================
# defaults: bit-identical, nothing constructed, nothing logged
# ===========================================================================

def test_the_defaults_are_bit_identical():
    a = _args()
    assert a.o5_target == "live" and float(a.o5_target_crop) == 0.0
    assert a.o6_innovation is False and a.o6_innovation_shuffle is False
    st = _stack()
    assert not hasattr(st, "frozen_o5_enc") and not hasattr(st, "ema_o5_enc")
    assert not any(("frozen_o5" in k) or ("ema_o5" in k)
                   for k in st.state_dict())
    assert not any(("frozen_o5" in n) or ("ema_o5" in n)
                   for n, _ in st.named_parameters())
    # the loss step with the new kwargs at their defaults == not passing them
    st.eval()          # dropout off: the comparison must be bit-exact
    b = _batch(st)
    w = _weights_from_args(a)
    outs = []
    for explicit in (False, True):
        torch.manual_seed(123)   # any residual global draw, held equal
        kw = dict(o6_innovation=False, o6_innovation_shuffle=False) \
            if explicit else {}
        L = v6_loss_step(st, b, stage="S-W", weights=w, o1_k=3, o5_k=3,
                         generator=torch.Generator().manual_seed(9),
                         sigreg_generator=torch.Generator().manual_seed(5),
                         **kw)
        outs.append(L)
    assert float(outs[0]["loss"].detach()) == float(outs[1]["loss"].detach())
    for L in outs:
        assert "o6_input" not in L["log"]


def test_the_new_flags_parse_and_shuffle_requires_innovation():
    a = _args("--o6-innovation", "--o6-innovation-shuffle",
              "--o5-target", "frozen", "--o5-target-crop", "0.8")
    assert a.o6_innovation and a.o6_innovation_shuffle
    assert a.o5_target == "frozen" and float(a.o5_target_crop) == 0.8
    st = _stack()
    w = _weights_from_args(_args())
    with pytest.raises(ValueError, match="o6-innovation"):
        v6_loss_step(st, {}, stage="S-W", weights=w,
                     o6_innovation_shuffle=True)   # shuffle without innovation


def test_crop_out_of_range_is_refused_at_build():
    with pytest.raises(SystemExit, match="o5-target-crop"):
        _stack("--o5-target-crop", "1.0")
    with pytest.raises(SystemExit, match="o5-target-crop"):
        _stack("--o5-target-crop", "-0.3")


# ===========================================================================
# L1 — innovation-SIGReg
# ===========================================================================

def test_innovation_swaps_the_o6_input_and_fails_on_predictable_dynamics():
    # the prereg's analytic anchor: z_{t+1} == z_t (trivially predictable
    # dynamics) ⇒ dz == 0 ⇒ ridge intercept 0 ⇒ innovations EXACTLY 0 — a
    # point mass the sketched test must penalise hard, while plain SIGReg on
    # the very same states (Gaussian across samples) reads near-null.
    g = torch.Generator().manual_seed(0)
    B, W, d = 32, 3, 16
    states = torch.randn(B, 1, d, generator=g).expand(B, W, d).clone()
    rows = o6_innovation_rows(states)
    assert rows.shape == (B * (W - 1), d)
    assert float(rows.abs().max()) < 1e-4          # analytic: innovations ~ 0
    sr = SigReg(n_slices=512)
    plain = o6_sigreg_loss(sr, states.reshape(-1, d), 0,
                           generator=torch.Generator().manual_seed(1))
    innov = o6_sigreg_loss(sr, rows, 0,
                           generator=torch.Generator().manual_seed(1))
    assert float(innov) > 2.0 * float(plain), (float(plain), float(innov))


def test_g_is_cross_fitted_never_fit_on_scored_rows():
    # two half-batches with DIFFERENT trivial dynamics: half A constant
    # (dz = 0), half B a constant drift (dz = 1 per step). Cross-fitting
    # scores each half under the OTHER half's g, so NEITHER innovation set
    # can vanish; a leaky fit-on-scored g would zero BOTH. This is the
    # 2026-08-22 "tuned on the data it scores" rule, mechanised.
    g = torch.Generator().manual_seed(5)
    B, W, d = 8, 4, 6
    states = torch.randn(B, 1, d, generator=g).expand(B, W, d).clone()
    shift = torch.zeros(B, 1, d)
    shift[B // 2:] = 1.0
    steps = torch.arange(W, dtype=torch.float32).view(1, W, 1)
    states = states + shift * steps
    rows = o6_innovation_rows(states).reshape(B, W - 1, d)
    assert float(rows[:B // 2].abs().mean()) > 0.3   # scored under drift-g
    assert float(rows[B // 2:].abs().mean()) > 0.3   # scored under const-g


def test_shuffled_g_carries_no_dynamics():
    # with g fit on PERMUTED z_t the map must be ~noise: the innovations stay
    # ~ the centred dz (nothing of the dynamics was removed).
    g = torch.Generator().manual_seed(3)
    B, W, d = 16, 5, 8
    states = torch.randn(B, W, d, generator=g)
    dz = (states[:, 1:] - states[:, :-1]).reshape(-1, d)
    rows = o6_innovation_rows(states, shuffle=True,
                              generator=torch.Generator().manual_seed(4))
    dzc = dz - dz.mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(
        rows.reshape(1, -1), dzc.reshape(1, -1)).item()
    assert cos > 0.7, cos


def test_shuffle_mode_is_statistically_close_to_plain_sigreg():
    # the deliberate-regression arm must DEGENERATE to ~plain SIGReg. Craft
    # the null so both readings sit at the same Epps-Pulley operating point:
    # states = per-sample constant + iid noise with variances 0.5/0.5, so the
    # states AND dz are both unit-variance Gaussians. Averaged over seeds,
    # the shuffled-innovation value must sit within a factor band of plain —
    # while the analytic collapse case above sits far outside it.
    vals_plain, vals_shuf = [], []
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        B, W, d = 16, 5, 8
        c = torch.randn(B, 1, d, generator=g) * (0.5 ** 0.5)
        u = torch.randn(B, W, d, generator=g) * (0.5 ** 0.5)
        states = c + u
        rows = o6_innovation_rows(
            states, shuffle=True,
            generator=torch.Generator().manual_seed(seed + 100))
        sr = SigReg(n_slices=512)
        vals_plain.append(float(o6_sigreg_loss(
            sr, states.reshape(-1, d), 0,
            generator=torch.Generator().manual_seed(seed + 200))))
        vals_shuf.append(float(o6_sigreg_loss(
            sr, rows, 0,
            generator=torch.Generator().manual_seed(seed + 200))))
    mp = sum(vals_plain) / len(vals_plain)
    ms = sum(vals_shuf) / len(vals_shuf)
    assert 0.33 < ms / mp < 3.0, (mp, ms)


def test_innovation_needs_two_samples_and_two_steps():
    with pytest.raises(ValueError, match="batch >= 2"):
        o6_innovation_rows(torch.randn(1, 4, 8))
    with pytest.raises(ValueError, match="batch >= 2"):
        o6_innovation_rows(torch.randn(4, 1, 8))


def test_innovation_rows_carry_gradient_to_the_states():
    states = torch.randn(4, 4, 8, requires_grad=True)
    rows = o6_innovation_rows(states)
    rows.pow(2).sum().backward()
    assert states.grad is not None
    assert float(states.grad.abs().sum()) > 0


# ===========================================================================
# L2 — frozen-teacher targets
# ===========================================================================

def test_frozen_mode_attaches_fixed_copies_mapped_to_aux():
    st = _stack("--o5-target", "frozen")
    assert hasattr(st, "frozen_o5_enc") and hasattr(st, "frozen_o5_ro")
    # the per-step teacher update keys on `ema_o5_enc` — frozen mode must
    # leave that gate cold or the "frozen" teacher would quietly track EMA.
    assert not hasattr(st, "ema_o5_enc")
    seen = 0
    for n, p in st.named_parameters():
        if n.startswith(("frozen_o5_enc.", "frozen_o5_ro.")):
            seen += 1
            assert not p.requires_grad
            assert st.group_of(n) == "aux"
    assert seen > 0
    # same config by construction — the no-projection contract
    live = [tuple(p.shape) for p in st.encoder.parameters()]
    froz = [tuple(p.shape) for p in st.frozen_o5_enc.module.parameters()]
    assert live == froz


def test_the_frozen_prefixes_are_S_W_introducible():
    for pref in ("frozen_o5_enc.", "frozen_o5_ro."):
        assert pref in STAGE_MAY_INTRODUCE["S-W"]


def test_resync_after_init_load_copies_live_into_frozen():
    st = _stack("--o5-target", "frozen")
    # simulate an init-from load: perturb the LIVE encoder after construction
    with torch.no_grad():
        for p in st.encoder.parameters():
            p.add_(1.0)
    lp = next(st.encoder.parameters())
    tp = next(st.frozen_o5_enc.module.parameters())
    assert not torch.allclose(lp, tp)          # teacher is stale (the bug)
    _resync_ema_o5(st)
    tp = next(st.frozen_o5_enc.module.parameters())
    assert torch.allclose(lp, tp)              # resync fixed it


def test_frozen_teacher_never_moves_under_training():
    # mirrors the REAL main-loop order: build → (init) resync → stage freeze
    # → optimiser over requires_grad params → step. ⚠️ apply_stage_freeze
    # flips requires_grad True for every trainable-group param — the aux-
    # mapped teacher copies INCLUDED (the shipped EMA precedent) — so the
    # property that actually protects the frozen teacher is that its forward
    # runs under no_grad: grads stay None, AdamW skips it, no update call is
    # wired. This test pins exactly that mechanism.
    from tanitad.models.v6 import apply_stage_freeze
    a = _args("--o5-target", "frozen")
    st = build_stack_from_args(a)
    _resync_ema_o5(st)
    apply_stage_freeze(st, "S-W")
    before = {k: v.clone() for k, v in st.state_dict().items()
              if k.startswith(("frozen_o5_enc.", "frozen_o5_ro."))}
    assert before
    live_before = next(st.encoder.parameters()).clone()
    w = _weights_from_args(a)
    trainable = [p for p in st.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    for step in range(2):
        b = _batch(st, seed=20 + step)
        with torch.no_grad():                  # the main loop's target path
            x = b["frames"][:, -1]
            st.frozen_o5_ro.module(st.frozen_o5_enc.module(x))
        L = v6_loss_step(st, b, stage="S-W", weights=w, o1_k=3, o5_k=3)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        # the protective mechanism: no gradient ever reaches the teacher
        for n, p in st.named_parameters():
            if n.startswith(("frozen_o5_enc.", "frozen_o5_ro.")):
                assert p.grad is None, n
        opt.step()
        # the trainer's per-step teacher update keys on `ema_o5_enc` —
        # absent in frozen mode, so nothing may move the frozen pair
        assert not hasattr(st, "ema_o5_enc")
    assert not torch.allclose(next(st.encoder.parameters()),
                              live_before)     # the LIVE net did move …
    after = st.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), k     # … the frozen teacher did NOT


def test_frozen_teacher_forward_is_no_grad_and_matches_live_at_init():
    st = _stack("--o5-target", "frozen")
    st.eval()
    x = torch.randn(2, 9, 64, 160)
    zf = st.frozen_o5_ro.module(st.frozen_o5_enc.module(x))
    zl = st.readout(st.encoder(x))
    assert zf.shape == zl.shape                # no projection needed: same dim
    assert zf.grad_fn is None or not zf.requires_grad
    assert torch.allclose(zf, zl, atol=1e-6)   # a faithful copy at init


# ===========================================================================
# L4 — azimuthal-crop targets
# ===========================================================================

def test_crop_targets_differ_inputs_untouched_same_window_across_k():
    B, k, C, H, W = 3, 4, 2, 8, 32
    ramp = torch.arange(W, dtype=torch.float32).view(1, 1, 1, 1, W)
    ff = ramp.expand(B, k, C, H, W).clone()
    ff_before = ff.clone()
    out = azimuthal_target_crop(ff, 0.5,
                                generator=torch.Generator().manual_seed(2))
    assert out.shape == ff.shape               # resized back to full width
    assert torch.equal(ff, ff_before)          # the input tensor is untouched
    assert not torch.allclose(out, ff)         # and a real crop happened
    for bi in range(B):
        for kk in range(1, k):                 # SAME window across a sample's
            assert torch.allclose(out[bi, 0], out[bi, kk])   # k futures
    # a column ramp stays a monotone ramp spanning ~cw columns: the crop is a
    # pure FOV restriction (cylindrical column = azimuth), height untouched
    row = out[0, 0, 0, 0]
    assert float((row[1:] - row[:-1]).min()) >= -1e-4
    assert float(row.min()) >= -1e-3 and float(row.max()) <= W - 1 + 1e-3
    span = float(row.max() - row.min())
    assert 12.0 <= span <= 17.0, span          # cw = 16 of 32, antialias slack


def test_crop_offsets_are_per_sample():
    B, k, C, H, W = 3, 2, 1, 4, 64
    ff = torch.arange(W, dtype=torch.float32).view(1, 1, 1, 1, W) \
        .expand(B, k, C, H, W).clone()
    out = azimuthal_target_crop(ff, 0.5,
                                generator=torch.Generator().manual_seed(0))
    starts = [round(float(out[bi, 0, 0, 0, 0])) for bi in range(B)]
    assert len(set(starts)) >= 2, starts       # per-sample, not per-batch


def test_crop_refuses_out_of_range_and_silent_noop_fractions():
    ff = torch.zeros(2, 2, 1, 4, 16)
    for bad in (0.0, 1.0, 1.5, -0.2, 0.99):    # 0.99 rounds to full width
        with pytest.raises(ValueError):
            azimuthal_target_crop(ff, bad)


# ===========================================================================
# composability — L3 = L1 + L2 builds and steps
# ===========================================================================

def test_L1_plus_L2_compose_build_and_step():
    a = _args("--o5-target", "frozen", "--o6-innovation")
    st = build_stack_from_args(a)
    _resync_ema_o5(st)
    w = _weights_from_args(a)
    L = v6_loss_step(st, _batch(st, batch=4, seed=31), stage="S-W",
                     weights=w, o1_k=3, o5_k=3,
                     o6_innovation=bool(a.o6_innovation),
                     o6_innovation_shuffle=bool(a.o6_innovation_shuffle))
    assert torch.isfinite(L["loss"])
    assert L["log"]["o6_input"] == "innovation"
    # innovation rows: B*(W-1), and the incumbent renorm machinery untouched
    assert L["log"]["o6_rows"] == 4 * (st.cfg.predictor.window - 1)
    assert L["log"]["o6_row_renorm"] == 1.0
    trainable = [p for p in st.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    L["loss"].backward()
    grads = [p.grad for p in st.encoder.parameters() if p.grad is not None]
    assert grads and any(float(t.abs().sum()) > 0 for t in grads)
    opt.step()                                 # a full step with both levers
