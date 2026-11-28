"""The >=GT truncation is REACHED FROM THE CALLER — proven by MUTATION, not census.

⛔ WHY THIS FILE EXISTS, AND WHY THE TWO FILES BESIDE IT WERE NOT ENOUGH
------------------------------------------------------------------------
* ``test_rl_v2_faithful.py`` pins what the mask COMPUTES.
* ``test_rl_gt_bar_gradient_path.py`` pins that the mask reaches a ``.grad`` when
  ``rl_objective`` is handed a bar directly.
* **Neither pins that any CALLER ever hands it one.** MEASURED 2026-09-10 on the
  2,000-step pilot: ``use_gt_bar = False``, ``frac_above_bar_mean = None``, and
  ``scripts/rl_pilot_refc21.py`` exposed **thirteen flags, none of which reached
  the truncation**. The mechanism was built, tested, gradient-proven — and
  unreachable. ⇒ what ran was a V1-style baseline reported as an RL arm.

⭐ THE FAILURE CLASS, NAMED: ``tac_goal_tok_head`` — 11,286 parameters that were
built, tested, rollable, and took ``grad_abs_sum`` **exactly 0 for all 40,284
steps** because nothing called them. *Rollable and trained are different claims.*
This file asserts the CALLING, at the `refcv3_adapter` -> `posttrain` ->
`advantage` boundary the pilot actually uses.

⛔ HOW THE EXPECTATIONS ARE WRITTEN — LITERALS AND ANALYTIC IDENTITIES ONLY.
*"A check that shares the defect it checks for is green forever."* An AST census
of this same wiring read **0 suspects on BOTH the fixed and the broken trainer**,
so inspection is not admissible here. Every assertion below is either a hand
written literal or one of two analytic identities that need no knowledge of the
reward function at all:

  * a bar BELOW every achievable reward is the IDENTITY mask ⇒ the gradient must
    equal the no-bar gradient **bit-for-bit**;
  * a bar ABOVE every achievable reward admits nothing ⇒ with ``w_intra = 0`` the
    loss loses every dependence on ``logp`` ⇒ the gradient is **exactly 0.0**.

Each section ships a DELIBERATE-REGRESSION arm that restores the real historical
defect (the caller that never emits a bar) and asserts the discriminator fires.

Evidence class: MEASURED (ours). Tier: N/A — instrument, not a capability claim.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
from torch import nn

from tanitad.rl import PostTrainConfig, RewardSpec
from tanitad.rl.posttrain import rl_objective, run_posttrain, score_gt_bar
from tanitad.rl.refcv3_adapter import make_refcv3_sample_fn

B, N, G, S = 2, 3, 4, 5
_SCALE = torch.tensor([0.4, 0.8, 1.2]).view(1, N, 1, 1)
_JITTER = torch.linspace(0.9, 1.1, G).view(1, 1, G, 1)


# --------------------------------------------------------------------------- #
# a stand-in policy with ONE leaf scalar, driven through the REAL adapter        #
# --------------------------------------------------------------------------- #
class _OneKnobPolicy(nn.Module):
    """Emits ``anchor_traj``/``offset`` like refc, through a single parameter.

    ⚠️ Deliberately NOT a mock of the adapter: the adapter is the thing under
    test. Only the model is stood in, so the sampling, the ctx assembly, the
    extras assembly and the objective are all the production code paths.
    """

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1))
        self.decoder = nn.Linear(1, 1)          # so select_trainable finds a prefix

    def forward(self, frames, **kw):
        x = torch.linspace(0.0, 8.0, S).view(1, 1, S) * _SCALE.view(1, N, 1)
        x = x.expand(B, N, S) + self.theta
        anchor = torch.stack([x, torch.zeros_like(x)], dim=-1)     # [B,N,S,2]
        return {"anchor_traj": anchor, "offset": 0.25 * anchor}


def _ctx(gt_reach_m: float = 8.0) -> dict:
    """A reward context whose GT is a straight path of a CHOSEN along-track reach.

    ⚠️ ``gt_reach_m`` scales the GT's EXTENT, it does not translate it — and that
    distinction cost a first draft of this file. ``_progress`` reads
    ``traj[..., -1, 0] - traj[..., 0, 0]``, a DIFFERENCE, so adding a constant to
    every waypoint moves the GT a megametre and changes its score by exactly
    nothing. A "bar" built that way is a lever that does not move, wearing a
    lever's name.
    """
    gt_x = torch.linspace(0.0, gt_reach_m, S).view(1, S).expand(B, S)
    return {"dt": 0.5,
            "v0": torch.full((B,), 4.0),
            "gt_traj": torch.stack([gt_x, torch.zeros_like(gt_x)], dim=-1)}


def _cfg(*, use_gt_bar: bool, w_intra: float = 0.0) -> PostTrainConfig:
    return PostTrainConfig(
        method="grpo", group_size=G, steps=1, batch=B, lr=0.0, dt=0.5,
        reward_weights={"progress": 1.0}, w_intra=w_intra, w_inter=1.0,
        veto_enabled=False, w_anchor=0.0, use_gt_bar=use_gt_bar,
        trainable_prefixes=("decoder",), noise_scale=0.0)


def _grad_through_the_adapter(*, use_gt_bar: bool, gt_reach_m: float = 8.0,
                              bar_value: float | None = None,
                              w_intra: float = 0.0, emit_bar: bool | None = None):
    """One objective step driven through `make_refcv3_sample_fn`; returns
    (grad, out, extras).

    ``bar_value`` hands a LITERAL bar through the adapter's emitter slot — the
    route for the two analytic identities, which need a bar outside the reward's
    own clamped range and therefore cannot be produced by any GT.
    ``gt_reach_m`` instead exercises the REAL `score_gt_bar` on a GT of a chosen
    extent. ``emit_bar`` lets a regression arm switch the EMITTER off while
    leaving the CONFIG on — the exact 2026-09-10 defect."""
    torch.manual_seed(0)
    model = _OneKnobPolicy()
    cfg = _cfg(use_gt_bar=use_gt_bar, w_intra=w_intra)
    spec = RewardSpec(weights=dict(cfg.reward_weights), dt=cfg.dt)
    ctx = _ctx(gt_reach_m)
    emit = use_gt_bar if emit_bar is None else emit_bar
    gt_bar_fn = None
    if emit and bar_value is not None:
        def gt_bar_fn(batch, c, n_steps):                       # noqa: F811
            return torch.full((B,), float(bar_value))
    elif emit:
        def gt_bar_fn(batch, c, n_steps):                       # noqa: F811
            return score_gt_bar(spec, c, n_steps=n_steps)
    sample_fn = make_refcv3_sample_fn(
        model, cfg, build_ctx=lambda b, out=None: dict(ctx),
        gt_bar_fn=gt_bar_fn, strict_conditioning=False)
    got = sample_fn({"frames": torch.zeros(B, 1)}, cfg)
    traj, logp, c = got[0], got[1], got[2]
    extras = got[3] if len(got) > 3 else {}
    # logp from the adapter is the sampled log-prob; make it carry theta so the
    # policy-gradient term has a leaf to differentiate through.
    logp = logp + 0.0 * traj.sum(dim=(-1, -2))
    out = rl_objective(traj, logp, c, cfg, spec, gt_bar=extras.get("gt_bar"))
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    return model.theta.grad.clone(), out, extras


# --------------------------------------------------------------------------- #
# 1. THE ADAPTER EMITS THE BAR AT ALL — and did not before                      #
# --------------------------------------------------------------------------- #

def test_the_adapter_emits_no_bar_by_default_so_banked_arms_reproduce():
    """⛔ DEFAULT OFF IS A REQUIREMENT, not an oversight. The banked 2,000-step
    baseline must stay reproducible bit-for-bit, so no ``gt_bar_fn`` means the
    pre-2026-09-11 THREE-tuple, unchanged."""
    model = _OneKnobPolicy()
    cfg = _cfg(use_gt_bar=False)
    sample_fn = make_refcv3_sample_fn(
        model, cfg, build_ctx=lambda b, out=None: dict(_ctx()),
        strict_conditioning=False)
    assert sample_fn.emits_gt_bar is False
    got = sample_fn({"frames": torch.zeros(B, 1)}, cfg)
    assert len(got) == 3, ("the no-bar, no-reference path must return the exact "
                           f"three-tuple it always did, got {len(got)} elements")


def test_the_adapter_emits_a_PER_WINDOW_bar_when_asked():
    _, _, extras = _grad_through_the_adapter(use_gt_bar=True)
    assert "gt_bar" in extras, (
        "the adapter did not put a bar in extras — this is the 2026-09-10 defect "
        "exactly: config on, mask never reached")
    assert tuple(extras["gt_bar"].shape) == (B,), (
        f"the bar must be per-window [B]={B}, got {tuple(extras['gt_bar'].shape)}")


# --------------------------------------------------------------------------- #
# 2. THE MUTATION — move ONLY the bar, watch the gradient                       #
# --------------------------------------------------------------------------- #

def test_the_instrument_carries_signal_before_it_is_used_to_conclude_anything():
    """⛔ THE CONTROL THAT MAKES THE TWO IDENTITIES BELOW MEAN ANYTHING.

    A gradient that is 0.0 for an unrelated reason would satisfy the
    admit-nothing identity trivially. So first: with NO bar, the gradient must be
    non-zero — the instrument is live."""
    g, _, extras = _grad_through_the_adapter(use_gt_bar=False)
    assert extras == {}, "no bar was requested; extras must be empty"
    assert float(g.abs().sum()) > 0.0, (
        "the no-bar gradient is already 0 — nothing below can discriminate")


def test_an_impossible_bar_ZEROES_the_gradient_and_a_permissive_one_does_not():
    """The two analytic identities, through the REAL caller wiring.

    ⛔ Both targets are analytic, not empirical: they need no knowledge of what
    ``progress`` computes.
      * bar far BELOW everything achievable -> identity mask -> gradient must be
        BIT-IDENTICAL to the no-bar gradient;
      * bar far ABOVE everything achievable -> admits nothing -> with
        ``w_intra = 0`` the loss loses every dependence on ``logp`` -> EXACTLY 0.
    """
    g_none, _, _ = _grad_through_the_adapter(use_gt_bar=False)
    g_admit_all, out_all, _ = _grad_through_the_adapter(
        use_gt_bar=True, bar_value=-1e9)
    g_admit_none, out_none, _ = _grad_through_the_adapter(
        use_gt_bar=True, bar_value=+1e9)

    assert torch.equal(g_admit_all, g_none), (
        "a bar below every achievable reward is the identity mask, so the "
        f"gradient must be bit-identical to the no-bar one: {g_admit_all} vs "
        f"{g_none}")
    assert float(out_all["frac_above_bar"]) == 1.0
    assert float(g_admit_none.abs().sum()) == 0.0, (
        "a bar above every achievable reward admits nothing, so with w_intra=0 "
        f"the gradient must be EXACTLY 0.0, got {g_admit_none}")
    assert float(out_none["frac_above_bar"]) == 0.0


def test_the_REAL_scorer_ZEROES_SOME_anchors_at_a_bar_in_between():
    """⭐ THE ONE THE TWO EXTREMES CANNOT ANSWER, and it runs the REAL
    `score_gt_bar` rather than a literal.

    A mask that only ever admits everything or nothing is not a truncation, it is
    a switch. Sweeping the GT's own along-track reach must produce a bar that
    admits only SOME anchors — a fraction STRICTLY between 0 and 1 — and the
    inter-anchor advantage must then contain BOTH a zeroed entry and a surviving
    positive one. ⛔ This is the assertion that says the mask *zeroes samples*,
    which is a different claim from "the flag is set"."""
    hits = {}
    for reach in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        _, out, _ = _grad_through_the_adapter(use_gt_bar=True, gt_reach_m=reach)
        hits[reach] = float(out["frac_above_bar"])
    partial = [r for r, f in hits.items() if 0.0 < f < 1.0]
    assert partial, (
        "no GT reach admitted only SOME anchors — the mask never discriminated. "
        f"frac_above_bar by GT reach (m): {hits}")
    assert any(f == 1.0 for f in hits.values()) or any(f == 0.0 for f in hits.values()), (
        f"the sweep never reached a saturating bar either; hits={hits}")

    _, out, _ = _grad_through_the_adapter(use_gt_bar=True, gt_reach_m=partial[0])
    inter = out["adv_inter"]
    assert float((inter == 0.0).sum()) > 0, "no anchor was zeroed by the bar"
    assert float((inter > 0.0).sum()) > 0, "the bar zeroed every anchor"


def test_DELIBERATE_REGRESSION_a_caller_that_never_emits_the_bar_is_CAUGHT():
    """⛔ THE HISTORICAL DEFECT, REINTRODUCED. ``use_gt_bar=True`` with a caller
    that emits nothing is EXACTLY what the 2,000-step arm would have done had the
    flag existed without the plumbing. It must RAISE, not run — a configured mask
    that silently never applies still exits 0 and reports a number."""
    with pytest.raises(ValueError, match="no gt_bar was handed to rl_objective"):
        _grad_through_the_adapter(use_gt_bar=True, emit_bar=False)


def test_DELIBERATE_REGRESSION_a_bar_scored_over_the_wrong_horizon_is_CAUGHT():
    """⛔ A bar scored over a DIFFERENT horizon is a threshold on a different
    reward function — ``_progress`` normalises by ``(S-1)*dt`` — and it looks
    entirely plausible. Literal: the fan has S=5 here, so a 4-waypoint GT must
    refuse."""
    spec = RewardSpec(weights={"progress": 1.0}, dt=0.5)
    ctx = _ctx()
    ctx["gt_traj"] = ctx["gt_traj"][:, :4, :]           # S=4 against a fan of S=5
    with pytest.raises(ValueError, match="SAME horizon"):
        score_gt_bar(spec, ctx, n_steps=S)


def test_score_gt_bar_REFUSES_a_spec_that_weights_gt_similarity():
    """The GT scores that term's own maximum against itself, so the bar would be
    inflated by exactly that weight for every window and nothing could clear it."""
    with pytest.raises(ValueError, match="gt_similarity is weighted"):
        score_gt_bar(RewardSpec(weights={"progress": 1.0, "gt_similarity": 0.5},
                                dt=0.5), _ctx(), n_steps=S)


# --------------------------------------------------------------------------- #
# 3. COMMENSURABILITY — the bar and the fan must be ONE reward function          #
# --------------------------------------------------------------------------- #

class _Unused(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(2, 2)


def test_run_posttrain_REFUSES_a_spec_that_disagrees_with_the_bars_config():
    """⛔ NOW LIVE, no longer latent: the pilot passes ``spec=`` as of
    2026-09-11 precisely so the bar and the candidates are one object."""
    cfg = _cfg(use_gt_bar=True)

    def never_called(batch, c):                                # pragma: no cover
        raise AssertionError("the guard must fire before any sampling")

    with pytest.raises(ValueError, match="use_gt_bar=True with an explicit spec"):
        run_posttrain(_Unused(), never_called, cfg,
                      spec=RewardSpec(weights={"progress": 0.5}, dt=0.5),
                      batches=[])
    with pytest.raises(ValueError, match="use_gt_bar=True with an explicit spec"):
        run_posttrain(_Unused(), never_called, cfg,
                      spec=RewardSpec(weights={"progress": 1.0}, dt=0.25),
                      batches=[])


def test_the_matching_spec_is_ACCEPTED_so_the_guard_is_not_a_blanket_refusal():
    """⛔ THE CONTROL. A guard that refuses everything passes the test above and
    breaks every legitimate caller. Reaching the loop's OWN false-green guard is
    the proof the spec check let us through."""
    cfg = _cfg(use_gt_bar=True)
    with pytest.raises(RuntimeError, match="FALSE-GREEN REFUSED"):
        run_posttrain(_Unused(), lambda b, c: None, cfg,
                      spec=RewardSpec(weights={"progress": 1.0}, dt=0.5),
                      batches=[])


# --------------------------------------------------------------------------- #
# 4. THE PILOT'S OWN FLAG SURFACE — the thing that was actually missing          #
# --------------------------------------------------------------------------- #

def _pilot_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "rl_pilot_refc21.py")
    spec = importlib.util.spec_from_file_location("rl_pilot_refc21_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_pilot_EXPOSES_gt_bar_and_it_defaults_OFF():
    """⚠️ Inspection, and it is here as a REGRESSION TRIPWIRE only — the mutation
    sections above are what prove the mask is reached. The literals are the two
    facts the 2026-09-10 run got wrong: the flag must EXIST, and it must default
    to False so the banked baseline reproduces."""
    ns = _pilot_module().build_argparser().parse_args([
        "--ckpt", "x", "--train-epdir", "x", "--train-agents", "x",
        "--val-epdir", "x", "--val-agents", "x", "--out", "x"])
    assert hasattr(ns, "gt_bar"), (
        "rl_pilot_refc21.py still exposes no --gt-bar; the V2 truncation is "
        "unreachable from the only production RL caller in the tree")
    assert ns.gt_bar is False, "--gt-bar must default OFF"

    on = _pilot_module().build_argparser().parse_args([
        "--ckpt", "x", "--train-epdir", "x", "--train-agents", "x",
        "--val-epdir", "x", "--val-agents", "x", "--out", "x", "--gt-bar"])
    assert on.gt_bar is True


def test_score_gt_bar_broadcasts_on_THE_PILOTS_OWN_BATCH_SHAPES():
    """⛔ THE SHAPE TRAP, pinned against the shapes `rl_pilot_refc21.collate`
    actually emits -- not a convenient synthetic pair.

    The pilot's reward context is BATCH-SHAPED for a fan `[B, N, G, S, 2]`:
    `obstacles` is `[B, 1, 1, K, 2]`, `lead_path` `[B, 1, 1, S, 2]`, `v0` `[B]`,
    and `gt_traj` is `[B, S, 2]`. A GT handed in at its own rank would broadcast
    against those on the WRONG axes -- and the failure mode that matters is not
    a raise, it is a plausible-looking number from a scrambled pairing. The bar
    is therefore lifted to `[B, 1, 1, S, 2]`, the candidate rank with one anchor
    and one sample, and must come back per-window.
    """
    B, S, K, N, Gg = 2, 4, 3, 5, 4          # S = 4 = len(rl_pilot.HORIZONS)
    torch.manual_seed(0)
    ctx = {"gt_traj": torch.randn(B, S, 2).cumsum(1) + torch.tensor([4.0, 0.0]),
           "obstacles": torch.rand(B, 1, 1, K, 2) * 30.0,
           "lead_path": (torch.tensor([25.0, 0.0]).view(1, 1, 1, 1, 2)
                         + torch.zeros(B, 1, 1, S, 2)),
           "v0": torch.full((B,), 8.0), "dt": 0.5, "proximity_safe_m": 5.0}
    spec = RewardSpec(weights=dict(__import__("tanitad.rl.rewards", fromlist=["x"])
                                   .DEFAULT_WEIGHTS), dt=0.5)
    bar = score_gt_bar(spec, ctx, n_steps=S)
    assert tuple(bar.shape) == (B,), tuple(bar.shape)
    assert torch.isfinite(bar).all()

    fan = torch.randn(B, N, Gg, S, 2).cumsum(-2) + torch.tensor([4.0, 0.0])
    assert tuple(spec(fan, ctx).shape) == (B, N, Gg), (
        "the fan and the bar must be scored by the SAME context without either "
        "of them reshaping it")


def test_the_pilot_ALSO_EXPOSES_noise_mode_and_it_defaults_to_the_banked_value():
    """⚠️ THE SAME DEFECT, FOUND IN PASSING, IN THE SECOND V2 INGREDIENT.

    MEASURED 2026-09-11 with a same-breath control: `grep -c noise_mode
    rl_pilot_refc21.py` read **0** while `grep -c PostTrainConfig` read **4** --
    so the file WAS read and the token was genuinely absent. DDv2's released
    two-scalar sampler (`PostTrainConfig.noise_mode="two_scalar"`, landed
    2026-09-05) was therefore unreachable from the only caller the P-RC21 line
    runs. ⚠️ Scope: it was NOT unreachable from the programme --
    `stack/scripts/rl_refcv3_min.py` reaches it.

    ⛔ The literal that matters is the DEFAULT: every banked arm recorded
    `noise_mode: "multiplicative"`, so the flag must not change any of them.
    """
    base = ["--ckpt", "x", "--train-epdir", "x", "--train-agents", "x",
            "--val-epdir", "x", "--val-agents", "x", "--out", "x"]
    mod = _pilot_module()
    assert mod.build_argparser().parse_args(base).noise_mode == "multiplicative"
    assert mod.build_argparser().parse_args(
        base + ["--noise-mode", "two_scalar"]).noise_mode == "two_scalar"
