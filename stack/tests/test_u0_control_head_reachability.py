"""⭐ The `--sampler ddim --w-u0 0` refusal rests on a premise that is FALSE.

⛔ WHAT THIS FILE SETTLES
-------------------------
`refc_v3_train.py:468-475` refuses `--sampler ddim --w-u0 0`, and its stated reason is
a claim about GRADIENTS::

    "`control_head` is zero-init, so it would stay at exactly zero and the arm
     would silently be the anchored Gaussian with no denoiser at all"

repeated at `refc.py:2429-2435`: *"without that loss `control_head` stays at its zero
init forever"*.

**MEASURED 2026-09-10 — it does not.** `control_head` receives gradient from
``out["anchor_traj"]`` alone, which is precisely the tensor the trainer's
matched-anchor L1 gathers (`refc_v3_train.py:2245-2247`,
``recon = out["anchor_traj"][ar, a_star]``). So with the u0 loss entirely absent from
the graph the head is still trained — through the integrated path rather than on its
own prediction.

⚠️ SCOPE IT HONESTLY, BECAUSE THIS IS A JUSTIFICATION BUG, NOT A LICENCE.
  * What is refuted is the REASON ("stays at exactly zero"), not necessarily the
    DECISION. A sampler supervised only through the integrator is a weaker thing than
    one supervised on its own x0 prediction, and whether that arm is worth running is
    an experiment-design question for the PI — not something a test may decide.
  * ⛔ This file therefore does NOT remove the refusal. It pins the measurement so the
    decision is made against what the code does rather than against a comment.

⭐ WHY IT MATTERS BEYOND THE COMMENT. refcv6's cheapest planned arm is `--w-u0 0`, and
it was believed to be a one-flag change. It is not: on `--sampler ddim` it is a hard
startup refusal, and the refusal's premise is the claim measured here. Its status is a
PI decision, and this file is the evidence that decision needs.

⛔ HOW THE MEASUREMENT IS MADE DISCRIMINATING. A "gradient is non-zero" reading proves
nothing without something that must read zero. Three outputs of the SAME forward are
measured in the same breath and MUST read exactly 0.0 — ``anchor_logits``, ``offset``
and ``sel_score``. If any of them ever reads non-zero, this rig has stopped
discriminating and its positive result is inadmissible.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.refs import refc                                        # noqa: E402

_HORIZONS = (5, 10, 15, 20)
_N = 5

#: MEASURED 2026-09-10 on this rig. Recorded as literals so a change in magnitude is
#: visible, but the ASSERTIONS below are on the analytic properties (zero / non-zero),
#: never on these numbers — a float equality on a backward sum would be brittle noise.
_MEASURED = {"anchor_traj": 93867.19, "u0_hat": 106868.49, "traj": 21659.14,
             "anchor_logits": 0.0, "offset": 0.0, "sel_score": 0.0}

#: The outputs that CANNOT reach `control_head`. These are the controls.
_MUST_BE_ZERO = ("anchor_logits", "offset", "sel_score")


def _v0_decoder(seed: int = 0, sampler: str = "ddim"):
    """The helper from `tests/test_refc_sampler.py:121-135`, kept identical so the
    two files are talking about the same object."""
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                             sampler=sampler)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(_HORIZONS), d_meas=8, d_ctx=4,
        tac_latent_dim=4, anchors=torch.randn(_N, len(_HORIZONS), 2), cfg=cfg,
        hierarchy=False, graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=_HORIZONS, v0_conditioned=True,
        control_units="alat")
    torch.manual_seed(seed + 1)
    dec.anchor_controls.copy_(torch.stack(
        [torch.linspace(-2.0, 2.0, _N), torch.linspace(-1.5, 1.5, _N)], dim=-1))
    return dec


def _grad_through(key: str) -> float:
    """`control_head`'s grad_abs_sum after one backward through ``out[key].sum()``.

    ⛔ The u0 loss is not merely zero-weighted here — it is ABSENT from the graph.
    That is the strongest form of ``w_u0 = 0``.
    """
    dec = _v0_decoder()
    dec.train()
    torch.manual_seed(77)
    fmap = torch.randn(2, 16, 3, 5)
    m = torch.randn(2, 8)
    dec.zero_grad(set_to_none=True)
    out = dec(fmap, m, steps=2, v_ms=torch.tensor([12.0, 20.0]))
    t = out[key]
    if not torch.is_tensor(t) or not t.requires_grad or not t.is_floating_point():
        return 0.0
    t.sum().backward()
    return sum(float(p.grad.detach().abs().sum())
               for p in dec.control_head.parameters() if p.grad is not None)


def test_control_head_starts_at_EXACTLY_zero():
    """The refusal's own premise, as a literal. If this ever fails the whole
    argument changes shape and the rest of this file is about something else."""
    dec = _v0_decoder()
    assert float(dec.control_head.weight.detach().abs().sum()) == 0.0
    assert float(dec.control_head.bias.detach().abs().sum()) == 0.0


@pytest.mark.parametrize("key", _MUST_BE_ZERO)
def test_THE_CONTROLS_must_read_exactly_zero(key):
    """⛔ WITHOUT THESE THE POSITIVE RESULT IS INADMISSIBLE. Three outputs of the
    same forward that `control_head` cannot reach. They make 'non-zero' mean
    something."""
    assert _grad_through(key) == 0.0, (
        f"{key} reached control_head — this rig no longer discriminates between "
        f"'reachable' and 'unreachable', so its positive result is void")


def test_the_rig_can_detect_gradient_at_all():
    """⛔ THE POSITIVE CONTROL. Backward through the sampler's own prediction must
    move the head, or the zeros above would be uninformative."""
    assert _grad_through("u0_hat") > 0.0


def test_control_head_IS_REACHED_from_anchor_traj_with_NO_u0_loss():
    """⭐ THE MEASUREMENT. `anchor_traj` is the tensor the trainer's matched-anchor
    L1 gathers (`refc_v3_train.py:2245-2247`). Gradient reaches `control_head`
    through it with the u0 loss absent from the graph entirely.

    ⇒ "without that loss `control_head` stays at its zero init forever"
      (`refc.py:2429-2435`, and the refusal at `refc_v3_train.py:468-475`)
      is FALSE as written.
    """
    g = _grad_through("anchor_traj")
    assert g > 0.0, (
        "control_head received no gradient from anchor_traj — if this fails, the "
        "refusal's premise HOLDS and refcv6 arm D really is blocked")
    # and it is not a rounding artefact: it is the same order as the positive control
    assert g > 1.0


def test_the_emitted_fan_also_reaches_it():
    """`traj` is the emitted fan. It reaches the head too, so the effect is not an
    artefact of one particular output key."""
    assert _grad_through("traj") > 0.0


def test_DELIBERATE_REGRESSION_a_sampler_none_build_has_no_control_head():
    """⛔ THE REGRESSION ARM. Everything above is a claim about the DDIM seam. With
    `sampler='none'` the head does not exist at all, so a rig that silently built
    the wrong decoder would be caught here rather than reporting a reachability
    result about a module that is not in the arm."""
    dec = _v0_decoder(sampler="none")
    assert dec.control_head is None


def test_the_measured_magnitudes_are_recorded_for_the_PI_decision():
    """Not an assertion about correctness — a recorded fact, so the PI's decision on
    arm D is made against numbers rather than against a comment. Tolerance is wide
    on purpose: the ORDER is the evidence, the digits are not."""
    g_traj = _grad_through("anchor_traj")
    g_u0 = _grad_through("u0_hat")
    assert g_traj == pytest.approx(_MEASURED["anchor_traj"], rel=0.5)
    assert g_u0 == pytest.approx(_MEASURED["u0_hat"], rel=0.5)
    # the integrated path carries a LARGE share of what the direct x0 loss does
    assert g_traj > 0.5 * g_u0
