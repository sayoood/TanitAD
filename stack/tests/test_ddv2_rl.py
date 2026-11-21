"""Pins ``tanitad.rl.ddv2_rl`` to the RELEASED DiffusionDriveV2 arithmetic.

Three kinds of evidence, in order of strength (CLAUDE.md: a cross-check must be derived
independently of the value it checks):

1. ⭐ AN INDEPENDENTLY AUTHORED REFERENCE — the release's own ``DDIMScheduler_with_logprob``
   class, vendored VERBATIM (``fixtures/ddv2_released_scheduler.py``, excerpt sha256 pinned)
   and executed on the same inputs and RNG state: the port must match it BIT FOR BIT.
2. ANALYTIC LITERALS — values written by hand, never an expression over the code under test.
3. DELIBERATE REGRESSIONS — the defects a port is most likely to introduce (clamping the last
   alpha to ``table[0]``, per-coordinate noise, a missing floor, a missing ≥GT mask, the
   library's across-anchor centring, all-sample averaging) are re-introduced and the check
   that should catch each one is required to FAIL on it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import pathlib

import pytest
import torch

from tanitad.rl import ddv2_rl as D

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "ddv2_released_scheduler.py"


def _released():
    pytest.importorskip("diffusers")
    spec = importlib.util.spec_from_file_location("ddv2_released_scheduler", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sched = mod.DDIMScheduler_with_logprob(num_train_timesteps=1000, steps_offset=1,
                                           beta_schedule="scaled_linear",
                                           prediction_type="sample")
    sched.set_timesteps(1000, "cpu")            # rl.py:804 — makes prev_timestep = t - 1
    return mod, sched


# --------------------------------------------------------------------------- #
# 1. the vendored reference is what it says it is                              #
# --------------------------------------------------------------------------- #
def test_the_vendored_excerpt_is_byte_identical_to_its_pinned_hash():
    text = FIXTURE.read_text(encoding="utf-8")
    body = text[text.index("class DDIMScheduler_with_logprob"):]
    got = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert got == "ae98ea7680374c177529176a5e7be02a985594d794de2125fcfb98fc4f6f505f"


#: sha256 of the banked release's **line-ending-normalised** bytes. Same content, same pin
#: value as before: the LF bytes were always what `2789081a…` was computed over.
BANK_SHA256 = "2789081af3b62ae3c97744a1f86d61d31a9b1de9453bc19c583f3334288a6e46"


def _bank_path():
    # sparse worktrees omit the Research Lab bank; TANITAD_BANK_ROOT points at a full checkout
    root = pathlib.Path(os.environ.get("TANITAD_BANK_ROOT", str(HERE.parents[1])))
    return (root / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
            / "2026-09-05-diffusiondrive-v2-analysis" / "raw" / "ddv2_src"
            / "diffusiondrivev2_model_rl.py")


def _lf(raw: bytes) -> bytes:
    """⚠️ **HASH THE LINE-ENDING-NORMALISED BYTES, NOT THE RAW ONES.**

    The repo has no ``.gitattributes`` for this path, so with the dev box's ``core.autocrlf=
    true`` the working tree is CRLF while every Linux pod / cloud checkout is LF — and a raw
    ``read_bytes()`` hash then reports the SAME FILE as changed depending only on where it ran.

    MEASURED 2026-09-16 and independently verified by the coordinator: in a fresh NON-sparse
    worktree this file hashes ``a9075847ee…`` raw (53,674 bytes, 1,170 CRLF pairs) while
    ``git cat-file -p HEAD:<path> | sha256sum`` is ``2789081af3b6…`` (52,504 bytes) — exactly
    the pin. The content was right; only the checkout differed. The test passed historically
    only because the predecessor worktrees were SPARSE and took the skip path below, so a fresh
    full checkout would have gone red on the platform and taught the reader to re-baseline on
    red — which is precisely how a real edit would slip through.

    ⛔ This forgives line endings and NOTHING else: proven by
    :func:`test_MUTATION_the_normalised_bank_hash_still_catches_a_one_character_change`.
    Same convention as ``tests/test_v5_trainer_v2_val.py``'s trainer pin.
    """
    return raw.replace(b"\r\n", b"\n")


def test_the_excerpt_matches_the_banked_release_when_the_bank_is_checked_out():
    bank = _bank_path()
    if not bank.exists():
        pytest.skip("banked release not in this (sparse) checkout; the excerpt hash above "
                    "still pins the fixture")
    raw = _lf(bank.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == BANK_SHA256, (
        "the banked DiffusionDriveV2 RL source changed. This is the PRIMARY SOURCE every "
        "`rl.py:NNN` citation in ddv2_rl.py, the SPEC and the DIFF resolves against. Confirm "
        "the edit is intended before touching this pin — and note that a line-ending "
        "difference can no longer cause this, because the bytes are normalised first.")
    lines = raw.decode("utf-8").splitlines(keepends=True)
    text = FIXTURE.read_text(encoding="utf-8")
    assert "".join(lines[517:676]) == text[text.index("class DDIMScheduler_with_logprob"):]


def test_MUTATION_the_normalised_bank_hash_still_catches_a_one_character_change():
    """⛔ A pin that forgives line endings must not forgive CONTENT. Both branches proven.

    Fixing the CRLF false positive by normalising is only safe if the normalised pin is still
    a real guard — so the forgiven transform is required to PASS and three content changes,
    one of them a single character and one of them whitespace-only, are required to go RED.
    """
    bank = _bank_path()
    if not bank.exists():
        pytest.skip("banked release not in this (sparse) checkout")
    lf = _lf(bank.read_bytes())
    sha = lambda b: hashlib.sha256(_lf(b)).hexdigest()          # noqa: E731

    # (a) the real file passes, on LF and on a CRLF checkout of the SAME content
    assert sha(lf) == BANK_SHA256
    crlf = lf.replace(b"\n", b"\r\n")
    assert crlf != lf and len(crlf) > len(lf), "fixture: the CRLF expansion must differ"
    assert sha(crlf) == BANK_SHA256, "the whole point: a CRLF checkout is the same content"

    # (b) a ONE-CHARACTER content change must go red
    i = lf.index(b"class DDIMScheduler_with_logprob")
    one_char = lf[:i] + b"C" + lf[i + 1:]                        # 'class' -> 'Class'
    assert len(one_char) == len(lf) and one_char != lf
    assert sha(one_char) != BANK_SHA256

    # (c) a WHITESPACE-ONLY change that is not a line ending must go red too
    ws = lf.replace(b"class DDIMScheduler_with_logprob",
                    b"class  DDIMScheduler_with_logprob", 1)
    assert ws != lf and sha(ws) != BANK_SHA256

    # (d) and a deletion must go red
    assert sha(lf[:-1]) != BANK_SHA256


# --------------------------------------------------------------------------- #
# 2. the step, against the release, bit for bit                                #
# --------------------------------------------------------------------------- #
def test_the_released_constructor_really_runs_with_clip_sample_on():
    """The release passes no ``clip_sample=`` (rl.py:703-708): the framework default decides.

    If a future diffusers flips the default, THIS goes red first, before the bitwise tests
    start comparing two clamp-less paths and passing for the wrong reason.
    """
    _, sched = _released()
    assert sched.config.clip_sample is True
    assert float(sched.config.clip_sample_range) == 1.0
    assert sched.config.thresholding is False
    assert float(sched.final_alpha_cumprod) == 1.0          # SPEC §3.2 (set_alpha_to_one)


@pytest.mark.parametrize("eta", [1.0, 0.0])
@pytest.mark.parametrize("scale", [0.3, 1.5])
def test_step_is_bitwise_the_released_step_at_every_rollout_label(eta, scale):
    """``scale=1.5`` puts ~50 % of the x0_hat coordinates outside [-1, 1], so the clamp is live."""
    mod, sched = _released()
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 5, 8, 2, generator=g) * scale
    xt = x0 + 0.03 * torch.randn(2, 5, 8, 2, generator=g)
    if scale > 1:
        assert float((x0.abs() > 1).float().mean()) > 0.3
    for t in D.rollout_labels():
        torch.manual_seed(1234 + t)
        r_prev, r_logp, r_mean = sched.step(model_output=x0, timestep=torch.tensor(t),
                                            sample=xt, eta=eta)
        torch.manual_seed(1234 + t)
        p_prev, p_logp, p_mean = D.ddim_logprob_step(x0, xt, t, t - 1,
                                                     sched.alphas_cumprod, eta=eta)
        assert torch.equal(p_mean, r_mean), f"mean differs at t={t}"
        assert torch.equal(p_prev, r_prev), f"sample differs at t={t}"
        assert torch.equal(p_logp, r_logp), f"log-prob differs at t={t}"


def test_logprob_at_a_supplied_prev_sample_is_bitwise_the_release():
    mod, sched = _released()
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(3, 4, 8, 2, generator=g)
    xt = torch.randn(3, 4, 8, 2, generator=g)
    prev = torch.randn(3, 4, 8, 2, generator=g)
    for t in (18, 8, 0):
        _, r_logp, _ = sched.step(model_output=x0, timestep=torch.tensor(t), sample=xt,
                                  eta=1.0, prev_sample=prev)
        _, p_logp, _ = D.ddim_logprob_step(x0, xt, t, t - 1, sched.alphas_cumprod,
                                           eta=1.0, prev_sample=prev)
        assert torch.equal(p_logp, r_logp)


def test_REGRESSION_a_port_without_the_clamp_is_not_the_release():
    """The WIP port of 2026-09-15 had no clamp; this is the check that caught it."""
    _, sched = _released()
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 5, 8, 2, generator=g) * 1.5
    xt = x0 + 0.03 * torch.randn(2, 5, 8, 2, generator=g)
    torch.manual_seed(7)
    _, r_logp, r_mean = sched.step(model_output=x0, timestep=torch.tensor(8), sample=xt, eta=1.0)
    torch.manual_seed(7)
    _, m_logp, m_mean = D.ddim_logprob_step(x0, xt, 8, 7, sched.alphas_cumprod, eta=1.0,
                                            clip_sample=False)
    assert not torch.equal(m_mean, r_mean)
    assert not torch.equal(m_logp, r_logp)


def test_clamped_coordinate_gradient_analytic():
    """At the FINAL step a clamped x0_hat coordinate gets exactly zero gradient; at t>0 it gets
    only the eps_hat path, whose coefficient is NEGATIVE: -sqrt(1-abar_p-s^2)*sqrt(abar_t)/sqrt(1-abar_t)."""
    table = _alphas()
    x0 = torch.full((1, 1, 8, 2), 2.0, requires_grad=True)      # every coordinate outside [-1, 1]
    xt = torch.zeros(1, 1, 8, 2)
    _, _, mean0 = D.ddim_logprob_step(x0, xt, 0, -1, table, eta=1.0)
    (g0,) = torch.autograd.grad(mean0.sum(), x0)
    assert torch.equal(g0, torch.zeros_like(g0))
    _, _, mean18 = D.ddim_logprob_step(x0, xt, 18, 17, table, eta=1.0)
    (g18,) = torch.autograd.grad(mean18.sum(), x0)
    a_t, a_p = float(table[18]), float(table[17])
    var = (1 - a_p) / (1 - a_t) * (1 - a_t / a_p)
    coef = -math.sqrt(max(1 - a_p - var, 0.0)) * math.sqrt(a_t) / math.sqrt(1 - a_t)
    assert coef < 0
    assert torch.allclose(g18, torch.full_like(g18, coef), rtol=1e-4)


def test_rollout_labels_are_the_released_literals():
    assert D.rollout_labels() == [18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    assert D.rollout_labels(2) == [10, 0]            # forward_test_rl, rl.py:943-949


def _alphas():
    betas = torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, 1000, dtype=torch.float32) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


def test_the_final_step_mean_is_exactly_x0_hat():
    # literals INSIDE the clip box, so this checks abar_prev = 1.0 and nothing else
    x0 = torch.tensor([[[[0.625, -0.375]] * 8]])
    xt = x0 + 0.5
    _, _, mean = D.ddim_logprob_step(x0, xt, 0, -1, _alphas(), eta=1.0)
    assert torch.equal(mean, x0)


def test_the_final_step_mean_is_the_CLAMPED_x0_hat_outside_the_box():
    x0 = torch.tensor([[[[3.0, -1.5]] * 8]])
    _, _, mean = D.ddim_logprob_step(x0, x0 + 0.5, 0, -1, _alphas(), eta=1.0)
    assert torch.equal(mean, torch.tensor([[[[1.0, -1.0]] * 8]]))


def test_REGRESSION_clamping_the_last_alpha_to_table0_moves_the_final_mean(monkeypatch):
    """The refc_sampler convention (idx clamped to 0) is NOT the release's (1.0).

    In-box literals: with x0 outside the box the clamp alone would move the mean and this
    mutant check would pass for the wrong reason (it did, for one edit, on 2026-09-15).
    """
    x0 = torch.tensor([[[[0.625, -0.375]] * 8]])
    xt = x0 + 0.5
    monkeypatch.setattr(D, "abar_at", lambda table, t: table[max(int(t), 0)])
    _, _, mean = D.ddim_logprob_step(x0, xt, 0, -1, _alphas(), eta=1.0)
    assert not torch.equal(mean, x0), "the mutant must be visible to the final-mean check"


# --------------------------------------------------------------------------- #
# 3. exploration: two scalars, floor 0.04                                      #
# --------------------------------------------------------------------------- #
def _ratio_spread_along_s(prev, mean):
    r = prev / mean
    return float((r.amax(dim=2) - r.amin(dim=2)).abs().max())


def test_exploration_is_two_scalars_per_trajectory_constant_along_the_waypoints():
    torch.manual_seed(0)
    mean_like = torch.rand(4, 20, 8, 2) + 0.5
    prev, _, mean = D.ddim_logprob_step(mean_like, mean_like, 0, -1, _alphas(), eta=1.0)
    assert _ratio_spread_along_s(prev, mean) < 1e-5          # analytic: 0
    r = (prev / mean)[:, :, 0, :]
    assert float(r[..., 0].std()) > 0.01 and float(r[..., 1].std()) > 0.01


def test_REGRESSION_per_coordinate_noise_breaks_the_two_scalar_invariant():
    torch.manual_seed(0)
    mean = torch.rand(4, 20, 8, 2) + 0.5
    mutant = mean * (1.0 + 0.04 * torch.randn_like(mean))    # the paper's "additive-like" rank-16 draw
    assert _ratio_spread_along_s(mutant, mean) > 1e-2


def test_exploration_std_is_the_0p04_floor_at_every_label():
    table = _alphas()
    mean_like = torch.ones(1, 200_000, 8, 2)
    for t in (18, 8, 2, 0):
        torch.manual_seed(t)
        prev, _, mean = D.ddim_logprob_step(mean_like, mean_like, t, t - 1, table, eta=1.0)
        ratio = (prev / mean)[0, :, 0, 0]
        assert abs(float(ratio.std()) - 0.04) < 0.0005, (t, float(ratio.std()))


def test_REGRESSION_removing_the_floor_is_visible():
    table = _alphas()
    mean_like = torch.ones(1, 200_000, 8, 2)
    torch.manual_seed(18)
    prev, _, mean = D.ddim_logprob_step(mean_like, mean_like, 18, 17, table, eta=1.0,
                                        explore_std_floor=0.0)
    ratio = (prev / mean)[0, :, 0, 0]
    assert abs(float(ratio.std()) - 0.04) > 0.02            # sigma_t(18) = 0.0120, not 0.04


def test_likelihood_uses_the_0p1_floor_and_sums_16_terms_analytically():
    table = _alphas()
    x0 = torch.zeros(1, 1, 8, 2, requires_grad=True)
    xt = torch.zeros(1, 1, 8, 2)
    prev = torch.full((1, 1, 8, 2), 0.05)
    _, logp, _ = D.ddim_logprob_step(x0, xt, 0, -1, table, eta=1.0, prev_sample=prev)
    # literal: 16 * [ -(0.05^2)/(2*0.01) - ln 0.1 - ln sqrt(2 pi) ]
    expect = 16 * (-(0.05 ** 2) / 0.02 - math.log(0.1) - math.log(math.sqrt(2 * math.pi)))
    assert abs(float(logp) - expect) < 1e-4
    logp.sum().backward()
    # d logp / d x0 at the final step = (prev - mean)/0.01 = 5.0 per coordinate
    assert torch.allclose(x0.grad, torch.full_like(x0, 5.0), atol=1e-4)


# --------------------------------------------------------------------------- #
# 4. the advantage                                                              #
# --------------------------------------------------------------------------- #
def _toy_rewards():
    # B=1, G=4, N=2. anchor 0 rewards 0.9 0.5 0.7 0.3 ; anchor 1 rewards 0.2 0.2 0.2 0.6
    r = torch.tensor([[[0.9, 0.2], [0.5, 0.2], [0.7, 0.2], [0.3, 0.6]]])
    return r


def test_advantage_literals_by_hand():
    r = _toy_rewards()
    fail = torch.zeros_like(r, dtype=torch.bool)
    fail[0, 1, 1] = True                               # sample (g=1, n=1) collides
    out = D.intra_anchor_advantage(r, torch.tensor([0.65]), fail)
    a = out["advantage"]
    # anchor 0: mean 0.6, unbiased std sqrt(0.2/3)=0.2581989 ; (r-0.6)/(std+1e-4)
    s0 = math.sqrt(0.2 / 3) + 1e-4
    assert abs(float(a[0, 0, 0]) - 0.3 / s0) < 1e-5     # 0.9 > bar 0.65 -> kept
    assert float(a[0, 1, 0]) == 0.0                      # 0.5: negative -> 0
    # 0.7: (0.7-0.6)/s0 = +0.387 and 0.7 clears the 0.65 bar -> kept POSITIVE.
    # (2026-09-15: the WIP literal here was 0.0, contradicting its own comment; fixed by hand.)
    assert abs(float(a[0, 2, 0]) - 0.1 / s0) < 1e-5
    assert float(a[0, 3, 0]) == 0.0                      # 0.3: negative -> 0


def test_advantage_literal_the_bar_admits_0p7_and_the_constraint_overrides():
    r = _toy_rewards()
    fail = torch.zeros_like(r, dtype=torch.bool)
    fail[0, 1, 1] = True
    a = D.intra_anchor_advantage(r, torch.tensor([0.65]), fail)["advantage"]
    s0 = math.sqrt(0.2 / 3) + 1e-4
    s1 = math.sqrt(0.12 / 3) + 1e-4                     # anchor 1: mean 0.3, var_unb 0.12/3
    expect = torch.tensor([[[0.3 / s0, 0.0], [0.0, -1.0], [0.1 / s0, 0.0], [0.0, 0.0]]])
    # anchor 1 sample 3: r=0.6 -> (0.6-0.3)/s1 = +1.5 positive, but 0.6 < bar 0.65 -> 0
    assert torch.allclose(a, expect, atol=1e-5), a
    assert s1 > 0  # used above in the comment's arithmetic


def test_REGRESSION_without_the_bar_the_sub_bar_positive_survives():
    r = _toy_rewards()
    fail = torch.zeros_like(r, dtype=torch.bool)
    a = D.intra_anchor_advantage(r, torch.tensor([0.65]), fail, use_gt_bar=False)["advantage"]
    s1 = math.sqrt(0.12 / 3) + 1e-4
    assert abs(float(a[0, 3, 1]) - 0.3 / s1) < 1e-5     # the bar is what zeroed it


def test_REGRESSION_veto_before_the_mask_would_erase_the_minus_one():
    r = _toy_rewards()
    fail = torch.zeros_like(r, dtype=torch.bool)
    fail[0, 1, 1] = True
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    adv = torch.where(fail, torch.full_like(adv, -1.0), adv)          # veto FIRST (wrong)
    adv = adv.clamp(min=0) * (r > 0.65 - 1e-6).float()
    assert float(adv[0, 1, 1]) == 0.0                                  # the mutant loses it
    good = D.intra_anchor_advantage(r, torch.tensor([0.65]), fail)["advantage"]
    assert float(good[0, 1, 1]) == -1.0


def test_REGRESSION_the_librarys_composite_is_a_different_object():
    from tanitad.rl.advantage import composite_advantage
    r = _toy_rewards()                                       # [B, G, N] here
    lib = composite_advantage(r.transpose(1, 2), gt_bar=torch.tensor([0.65]))["total"]
    ours = D.intra_anchor_advantage(r, torch.tensor([0.65]),
                                    torch.zeros_like(r, dtype=torch.bool))["advantage"]
    assert not torch.allclose(lib.transpose(1, 2), ours, atol=1e-3)


def test_a_group_of_one_is_refused():
    with pytest.raises(D.Ddv2ConfigError):
        D.intra_anchor_advantage(torch.zeros(1, 1, 3), torch.zeros(1),
                                 torch.zeros(1, 1, 3, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# 5. discount and loss                                                          #
# --------------------------------------------------------------------------- #
def test_discount_literals():
    w = D.discount_weights()
    expect = torch.tensor([0.8 ** 9, 0.8 ** 8, 0.8 ** 7, 0.8 ** 6, 0.8 ** 5, 0.8 ** 4,
                           0.8 ** 3, 0.8 ** 2, 0.8, 1.0])
    assert torch.allclose(w, expect)


def test_rl_loss_averages_over_nonzero_advantage_samples_only():
    logp = torch.zeros(2, 5, 3, requires_grad=True)
    adv = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0],        # row 0: one admitted sample
                        [1.0, 1.0, 1.0, 1.0, -1.0]])      # row 1: five non-zero
    disc = torch.tensor([0.5, 1.0, 1.0])
    out = D.rl_loss_per_row(logp, adv, disc)
    # row 0: per step -(2*gamma_t)/1 ; mean over steps = -2*(0.5+1+1)/3
    assert abs(float(out["rl_loss_b"][0]) - (-2.0 * 2.5 / 3)) < 1e-6
    # row 1: per step -(sum adv * gamma_t)/5 = -(3*gamma_t)/5
    assert abs(float(out["rl_loss_b"][1]) - (-(3.0 / 5) * 2.5 / 3)) < 1e-6
    out["rl_loss_b"].mean().backward()
    # d/dlogp[0,0,t] = -2*gamma_t/(1*T*B)
    assert abs(float(logp.grad[0, 0, 0]) - (-2.0 * 0.5 / (3 * 2))) < 1e-7
    assert float(logp.grad[0, 1, 0]) == 0.0


def test_REGRESSION_averaging_over_all_samples_changes_the_row_loss():
    logp = torch.zeros(1, 5, 1)
    adv = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    ours = float(D.rl_loss_per_row(logp, adv, torch.ones(1))["rl_loss_b"][0])
    mutant = float(-(adv.unsqueeze(-1)).mean())             # library-style mean over all M
    assert ours == -2.0 and abs(mutant - (-0.4)) < 1e-7 and ours != mutant


def test_il_row_weights_and_batch_global_il():
    adv_t = torch.tensor([[[0.0], [0.0]], [[0.3], [-1.0]]])   # row0 none positive, row1 has one
    w = D.il_row_weights(adv_t)
    assert w.tolist() == [1.0, pytest.approx(0.1)]
    rl_b = torch.tensor([0.0, -0.5])
    total = D.total_loss(rl_b, torch.tensor(2.0), w)
    assert abs(float(total) - ((0.0 + 1.0 * 2.0) + (-0.5 + 0.1 * 2.0)) / 2) < 1e-6
    with pytest.raises(D.Ddv2ConfigError):
        D.total_loss(rl_b, torch.tensor([2.0, 2.0]), w)


# --------------------------------------------------------------------------- #
# 6. the chain: table, truncated start, rollout, grad pass (resume pass)        #
# --------------------------------------------------------------------------- #
def test_alpha_table_is_bitwise_the_diffusers_table():
    _, sched = _released()
    assert torch.equal(D.diffusers_alphas_cumprod(), sched.alphas_cumprod)


def test_truncated_start_is_bitwise_diffusers_add_noise_and_group_major():
    _, sched = _released()
    g = torch.Generator().manual_seed(3)
    anchors = torch.randn(2, 5, 8, 2, generator=g) * 0.4          # B=2, N=5
    x_t, noise = D.truncated_start(anchors, 4, sched.alphas_cumprod, generator=g)
    tiled = anchors.unsqueeze(1).repeat(1, 4, 1, 1, 1).view(2, 20, 8, 2)   # rl.py:813-815
    ref = sched.add_noise(original_samples=tiled, noise=noise,
                          timesteps=torch.ones((2,), dtype=torch.long) * 8)  # rl.py:819-820
    assert torch.equal(x_t, ref)
    # group-major: m = g*N + n  <=>  .view(B, G, N) un-tiles it (rl.py:886)
    assert torch.equal(D.tile_groups(anchors, 4).view(2, 4, 5, 8, 2)[:, 3, 2], anchors[:, 2])


def test_REGRESSION_anchor_major_tiling_breaks_the_untile():
    anchors = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 8, 2)
    wrong = anchors.repeat_interleave(4, dim=1)                   # anchor-major
    assert not torch.equal(wrong.view(1, 4, 5, 8, 2)[:, 1, 0], anchors[:, 0])
    assert torch.equal(D.tile_groups(anchors, 4).view(1, 4, 5, 8, 2)[:, 1, 0], anchors[:, 0])


def _toy_x0_fn(x, t):
    return 0.9 * x + 0.01 * t


def _released_rollout(sched, x_start, x0_fn):
    """rl.py:825-862 transcribed literally, driving the vendored class."""
    sched.set_timesteps(1000, "cpu")
    step_num = 10
    roll = (torch.arange(0, step_num).double() * (20 / step_num)).round().flip(0).long()
    x = x_start
    states, logps = [x_start], []
    for k in roll:
        prev, lp, _ = sched.step(model_output=x0_fn(x, int(k)), timestep=k, sample=x, eta=1.0)
        states.append(prev)
        logps.append(lp)
        x = prev
    return torch.stack(states, dim=-1), torch.stack(logps, dim=-1)


def test_rollout_chain_is_bitwise_the_released_rollout_loop():
    _, sched = _released()
    g = torch.Generator().manual_seed(5)
    x_start = torch.randn(2, 8, 8, 2, generator=g) * 0.6
    torch.manual_seed(99)
    r_chain, r_logp = _released_rollout(sched, x_start, _toy_x0_fn)
    torch.manual_seed(99)
    out = D.rollout_chain(_toy_x0_fn, x_start, sched.alphas_cumprod)
    assert out["labels"] == [18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    assert torch.equal(out["chain"], r_chain)
    assert torch.equal(out["logp"], r_logp)


def test_REGRESSION_label_to_label_transitions_are_not_the_release():
    """Stepping 18 -> 16 -> ... (our inference ladder's convention) is a different chain."""
    _, sched = _released()
    g = torch.Generator().manual_seed(5)
    x_start = torch.randn(2, 8, 8, 2, generator=g) * 0.6
    torch.manual_seed(99)
    r_chain, _ = _released_rollout(sched, x_start, _toy_x0_fn)
    torch.manual_seed(99)
    labels = D.rollout_labels()
    x, states = x_start, [x_start]
    for i, t in enumerate(labels):
        t_prev = labels[i + 1] if i + 1 < len(labels) else -1
        x, _, _ = D.ddim_logprob_step(_toy_x0_fn(x, t), x, t, t_prev, sched.alphas_cumprod)
        states.append(x)
    assert not torch.equal(torch.stack(states, dim=-1), r_chain)


class _ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.lin = torch.nn.Linear(16 + 1, 16)

    def forward(self, x, t):
        b, m = x.shape[:2]
        h = torch.cat([x.reshape(b, m, 16), torch.full((b, m, 1), t / 20.0)], dim=-1)
        return x + 0.1 * torch.tanh(self.lin(h)).reshape(b, m, 8, 2)


def _il_step(x0_hat, gt):                          # batch-global mean L1, "metres" = state here
    return (x0_hat - gt[:, None]).abs().mean()


def _toy_problem():
    table = D.diffusers_alphas_cumprod()
    net = _ToyNet()
    g = torch.Generator().manual_seed(21)
    x_start = torch.randn(3, 6, 8, 2, generator=g) * 0.5
    gt = torch.randn(3, 8, 2, generator=g) * 0.5
    with torch.no_grad():
        torch.manual_seed(4)
        roll = D.rollout_chain(net, x_start, table)
    adv = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],            # row 0: nothing admitted
                        [1.5, 0.0, -1.0, 0.0, 0.3, 0.0],           # row 1: mixed
                        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])     # row 2: all vetoed
    return table, net, roll, gt, adv


def test_per_step_backward_equals_the_released_one_graph_loss():
    table, net, roll, gt, adv = _toy_problem()
    labels, T = roll["labels"], len(roll["labels"])
    # (a) the RELEASED form: one graph over all steps, rl_loss_per_row + total_loss
    net.zero_grad()
    logps, il = [], torch.zeros(())
    for i in range(T):
        lp, x0 = D.chain_step_logprob(net, roll["chain"], i, table, labels=labels)
        logps.append(lp)
        il = il + _il_step(x0, gt)
    il = il / T
    rl = D.rl_loss_per_row(torch.stack(logps, dim=-1), adv, D.discount_weights(T))
    loss_ref = D.total_loss(rl["rl_loss_b"], il, D.il_row_weights(rl["adv_discounted"]))
    loss_ref.backward()
    g_ref = [p.grad.clone() for p in net.parameters()]
    # (b) the per-step decomposition
    net.zero_grad()
    w = D.step_loss_weights(adv, T)
    total = 0.0
    for i in range(T):
        lp, x0 = D.chain_step_logprob(net, roll["chain"], i, table, labels=labels)
        li = D.per_step_loss(lp, _il_step(x0, gt), w, i)
        li.backward()
        total += float(li)
    assert abs(total - float(loss_ref)) < 1e-5 * max(1.0, abs(float(loss_ref)))
    for a, b in zip(g_ref, [p.grad for p in net.parameters()]):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), float((a - b).abs().max())
    assert w["il_weight_b"].tolist() == [1.0, pytest.approx(0.1), 1.0]


def test_REGRESSION_all_sample_denominator_in_the_decomposition_changes_the_gradient():
    table, net, roll, gt, adv = _toy_problem()
    labels, T = roll["labels"], len(roll["labels"])
    w = D.step_loss_weights(adv, T)
    good = []
    net.zero_grad()
    for i in range(T):
        lp, x0 = D.chain_step_logprob(net, roll["chain"], i, table, labels=labels)
        D.per_step_loss(lp, _il_step(x0, gt), w, i).backward()
    good = [p.grad.clone() for p in net.parameters()]
    bad_w = dict(w)
    disc = D.discount_weights(T)
    bad_w["coef_rl"] = adv.unsqueeze(-1) * disc / adv.shape[1] / (T * adv.shape[0])  # / M
    net.zero_grad()
    for i in range(T):
        lp, x0 = D.chain_step_logprob(net, roll["chain"], i, table, labels=labels)
        D.per_step_loss(lp, _il_step(x0, gt), bad_w, i).backward()
    bad = [p.grad for p in net.parameters()]
    assert any(not torch.allclose(a, b, rtol=1e-3) for a, b in zip(good, bad))


def test_zeroed_policy_coefficient_gives_exactly_the_il_only_gradient():
    """The NORL control (driver `--arm norl`): coef_rl := 0 while the IL weights stay the
    advantage-derived ones. KNOWN VALUE: the accumulated gradient is BIT-IDENTICAL to a pass that
    never builds the policy term — and differs from the RL pass. (Added 2026-09-15 after the
    logged `rl_part` of the real NORL arm read 4e-8 of float round-off: this pins the mechanism
    at the gradient itself, where round-off in a logging subtraction cannot reach.)"""
    table, net, roll, gt, adv = _toy_problem()
    labels, T = roll["labels"], len(roll["labels"])
    w = D.step_loss_weights(adv, T)

    def grads(fn_loss):
        net.zero_grad()
        for i in range(T):
            lp, x0 = D.chain_step_logprob(net, roll["chain"], i, table, labels=labels)
            fn_loss(lp, _il_step(x0, gt), i).backward()
        return [p.grad.clone() for p in net.parameters()]

    w0 = dict(w)
    w0["coef_rl"] = torch.zeros_like(w["coef_rl"])
    g_norl = grads(lambda lp, il, i: D.per_step_loss(lp, il, w0, i))
    g_il = grads(lambda lp, il, i: w["il_coef"] * il)
    g_rl = grads(lambda lp, il, i: D.per_step_loss(lp, il, w, i))
    assert all(torch.equal(a, b) for a, b in zip(g_norl, g_il))
    assert any(not torch.allclose(a, b) for a, b in zip(g_norl, g_rl))


def test_the_constants_are_the_published_values():
    c = D.DDV2.to_dict()
    for k, v in {"group_size": 4, "rollout_steps": 10, "label_span": 20, "trunc_t": 8,
                 "explore_std_floor": 0.04, "likelihood_std_floor": 0.10, "gamma": 0.8,
                 "eta": 1.0, "clip_sample": True, "clip_sample_range": 1.0,
                 "adv_std_eps": 1e-4, "bar_eps": 1e-6, "veto_value": -1.0,
                 "il_weight_with_positive": 0.1, "il_weight_no_positive": 1.0,
                 "lr": 2e-4, "weight_decay": 1e-4, "epochs": 10, "total_batch": 512}.items():
        assert c[k] == v, k
