"""The ladder's targets must sit at the TIME each level claims to predict.

⛔ WHY THIS FILE EXISTS. `test_change_9_all_three_rates_reach_exactly_six_seconds`
asserted `dt * steps == 6.0` -- on the CONFIG. It passed for every shipped
version and never once looked at which future frame a prediction was regressed
onto. MEASURED when someone finally did: **10 of 10 tactical and 4 of 4
strategic targets were wrong**, each shifted early by a full stride, because the
targets were sliced `[::stride]` instead of `[stride-1::stride]`. Both abstracted
levels were being trained as 0.2 s predictors -- change #9's temporal
abstraction, which is the PI's three-planner directive and the one thing MM-E15
says the programme most needs, was not in the model at all.

⭐ THE RULE THIS ENCODES: a test that asserts the advertisement is not a test of
the code. Assert the realised behaviour -- here, the target INDEX the forward
path actually selects, recovered from the tensor rather than read from source.
"""
import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _target_indices(cfg: RefAV1Config, level: str):
    """⭐ OBSERVED, NOT RECOMPUTED. Runs the real forward path and reads the
    indices the model reports it used. A helper that re-derived the slice would
    agree with the bug as happily as with the fix."""
    m = RefAV1(cfg)
    b = 1
    out = m(torch.randn(b, cfg.op_window, cfg.n_tokens, cfg.d_enc),
            torch.zeros(b, cfg.op_steps, cfg.a_dim),
            future_feats=torch.randn(b, cfg.op_steps, cfg.n_tokens, cfg.d_enc))
    return out[f"{level}_target_idx"]


# --------------------------------------------------------- the alignment law --
@pytest.mark.parametrize("level", ["tac", "str"])
def test_EVERY_target_sits_at_the_time_its_level_claims(level):
    """⭐⭐ THE LOAD-BEARING TEST. Future index k is the observation at
    (k+1)*op_dt, because rollout()[:,0] is the state after ONE step. So the
    target for step j of a `dt`-rate level must land at exactly (j+1)*dt."""
    c = _cfg()
    dt, steps = (c.tac_dt, c.tac_steps) if level == "tac" else (c.str_dt, c.str_steps)
    idx = _target_indices(c, level)
    assert len(idx) == steps, (
        f"{level}: only {len(idx)} of {steps} targets exist inside a "
        f"{c.op_steps}-step rollout -- the rest are silently dropped")
    for j, k in enumerate(idx):
        assert (k + 1) * c.op_dt == pytest.approx((j + 1) * dt, abs=1e-9), (
            f"{level} step {j} is regressed onto the frame at "
            f"{(k + 1) * c.op_dt:.2f} s but claims to predict {(j + 1) * dt:.2f} s")


@pytest.mark.parametrize("level", ["tac", "str"])
def test_the_last_target_reaches_the_full_six_seconds(level):
    """⛔ The shipped default reached 5.6 s (tactical) and 5.0 s (strategic)
    while its config said 6.0 -- a horizon claim no instrument checked."""
    c = _cfg()
    dt, steps = (c.tac_dt, c.tac_steps) if level == "tac" else (c.str_dt, c.str_steps)
    idx = _target_indices(c, level)
    assert (idx[-1] + 1) * c.op_dt == pytest.approx(6.0, abs=1e-9)


def test_THE_DELIBERATE_REGRESSION_the_old_slice_is_caught():
    """⚠️ A gate that does not FAIL the defect it exists to catch proves
    nothing. This re-introduces `[::stride]` and requires the law to reject it."""
    c = _cfg()
    stride = int(round(c.tac_dt / c.op_dt))
    old = list(range(0, c.op_steps, stride))[:c.tac_steps]      # the defect
    misaligned = sum(1 for j, k in enumerate(old)
                     if abs((k + 1) * c.op_dt - (j + 1) * c.tac_dt) > 1e-9)
    assert misaligned == c.tac_steps, (
        "the old slice must be caught on EVERY step, not merely on some")


# ------------------------------------------------- the guard on the config ----
def test_a_rate_that_is_not_a_multiple_of_op_dt_is_REFUSED():
    """⛔ THE SHIPPED DEFAULT. str_dt 1.5 / op_dt 0.2 = 7.5, rounded to 8, so
    the strategic rung silently ran at 1.6 s. The abstracted levels subsample
    the operative grid, so a non-integer ratio is not a rounding nicety -- it
    is a rate the model cannot represent."""
    with pytest.raises(ValueError, match="not an integer multiple"):
        _cfg(str_dt=1.5, str_steps=4).sanity()


@pytest.mark.parametrize("dt,steps", [(1.0, 6), (1.2, 5), (2.0, 3), (3.0, 2)])
def test_the_overrun_guard_is_a_BACKSTOP_and_cannot_fire_today(dt, steps):
    """⚠️ SAYING WHAT IS TRUE RATHER THAN STAGING A PASS. I wrote a test to make
    the overrun guard fire and could not: while all three horizons are pinned to
    exactly 6.0 s, `stride * steps == (dt/op_dt) * steps == 6.0/op_dt ==
    op_steps` IDENTICALLY -- an overrun is arithmetically impossible, and every
    config that would overrun is caught by the non-integer check first.

    ⇒ the guard stays as a backstop against a future change to the 6.0 s
    constant, and this test pins the invariant that makes it unreachable.
    A guard whose test cannot fail it is not evidence, so this file does not
    pretend otherwise."""
    c = _cfg(str_dt=dt, str_steps=steps)
    c.sanity()
    assert int(round(dt / c.op_dt)) * steps == c.op_steps


def test_the_PI_DECIDED_default_3_0x2_lands_exactly():
    """⭐ PI decision 2026-08-31: str_dt 3.0 x 2 — the 1 : 3 : 15 ladder.
    3.0/0.2 = 15 exactly, 2*15 = 30 = op_steps; targets OBSERVED at operative
    indices 14 and 29 = 3.0 s and 6.0 s."""
    c = _cfg()
    c.sanity()
    assert _target_indices(c, "str") == [14, 29]
    assert _target_indices(c, "tac") == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]


def test_a_valid_VARIANT_ladder_1_2x5_still_aligns():
    """The interim repair stays exercised as a variant — the alignment law is
    rate-generic, not a property of one default."""
    c = _cfg(str_dt=1.2, str_steps=5)
    c.sanity()
    assert _target_indices(c, "str") == [5, 11, 17, 23, 29]


# ------------------------------------------------------- end-to-end shapes ----
def test_the_fixed_slicing_still_produces_a_finite_backpropagating_loss():
    """⚠️ An alignment fix that breaks the shapes is not a fix."""
    c = _cfg()
    m = RefAV1(c)
    b = 2
    feats = torch.randn(b, c.op_window, c.n_tokens, c.d_enc)
    actions = torch.randn(b, c.op_steps, c.a_dim)
    future = torch.randn(b, c.op_steps, c.n_tokens, c.d_enc)
    out = m(feats, actions, future_feats=future)
    assert out["tac_pred"].shape[1] == c.tac_steps
    assert out["str_pred"].shape[1] == c.str_steps
    for k in ("loss", "loss_feat_op", "loss_feat_tac", "loss_feat_str"):
        assert torch.isfinite(torch.as_tensor(out[k])), k
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.tactical.parameters())
