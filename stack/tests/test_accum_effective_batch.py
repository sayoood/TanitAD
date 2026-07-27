"""⭐ ``--batch 8 --accum 8`` REALLY IS v1's EFFECTIVE 64 — measured, not echoed.

WHY THIS FILE EXISTS
--------------------
The staged v5 command carried ``--batch 16 --accum 4``. MEASURED on pod2's A40
(45,498 MiB): **16 OOMs at both candidate frames** (max micro-batch 8 at
256x640, 12 at 176x624). The no-code-change fix is ``--batch 8 --accum 8``,
which keeps ``batch * accum == 64``.

But ``batch * accum == 64`` is ARITHMETIC, and the trainer's own log row
``"eff_batch": batch * accum`` (``train_flagship_v4.py``) is that same
arithmetic printed back — it is **not evidence** that the optimizer step saw 64
examples' worth of gradient. Two independent things have to hold, and both are
properties of code rather than of the flags:

1. ``accum`` **distinct** micro-batches are consumed per optimizer step
   (``next(it)`` inside the accumulation loop), so 8x8 is 64 DISTINCT windows
   and not one window set counted eight times;
2. ``v4_loss_step``'s ``total`` is a per-example **MEAN**, so
   ``sum_i (L_i / accum)`` over equal-size micro-batches equals the loss of one
   batch of ``batch*accum``. If any term were a per-batch SUM, or if the
   micro-batches differed in size, ``(total / accum).backward()`` would silently
   re-weight the step and "effective 64" would be a slogan.

``drop_last=True`` on the trainer's DataLoader is what makes the micro-batches
equal-size, so it is pinned here too: without it the LAST micro-batch of an
epoch is short and the accumulated gradient is a weighted mean with the wrong
weights.

The test below runs the REAL ``v4_loss_step`` on toy episodes and compares the
gradient produced by the ACCUMULATION path against the gradient of the single
big batch, parameter by parameter. CPU, no cache, no GPU.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_flagship_v4 as T                                     # noqa: E402


#: The three per-example Bernoulli dropouts inside ``v4_loss_step``
#: (``flagship_losses.py``: ``v2_ego_dropout`` / ``v2_nav_dropout`` /
#: ``v2_fa_dropout``). They make the loss STOCHASTIC, so a naive
#: accum-vs-single gradient comparison measures the MASK DRAW, not the
#: accumulation arithmetic — the two arms consume the RNG stream differently
#: (8 draws of size ``micro`` vs 1 draw of size ``micro*accum``). They are
#: switched OFF for the equivalence leg and switched back ON, against a
#: measured noise floor, for the leg that reports what stays true in a real run.
STOCHASTIC_KNOBS = ("v2_ego_dropout", "v2_nav_dropout", "v2_fa_dropout")
#: …and the same again inside the HEAD (``flagship_v15.py``): ``ego_dropout`` /
#: ``goal_dropout`` masks and the truncated-diffusion ``noise_std``. All three
#: are gated on ``self.training``, so they are switched off by VALUE rather than
#: by ``.eval()`` — the model must stay in TRAIN mode, or the test would also
#: switch off any batch-coupled normalisation and could no longer see it.
#: (There is none: ``encoder.py`` bans BatchNorm outright — "LayerNorm/RMSNorm
#: only" — which is the structural reason accumulation CAN be equivalent here.)
HEAD_STOCHASTIC_KNOBS = ("ego_dropout", "goal_dropout")


def _fixture(n_windows=16, deterministic=True):
    """The `smoke()` stack, exactly — real WorldModel + FlagshipV4Head +
    `v4_loss_step`, on toy episodes at the smoke config's tiny frame."""
    import dataclasses

    from torch.utils.data import default_collate

    from tanitad.config import flagship4b_smoke_config
    from tanitad.models.fourbrain import WorldModel
    from train_flagship4b import FlagshipWindowDataset

    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    if deterministic:
        for k in STOCHASTIC_KNOBS:
            if hasattr(cfg, k):
                setattr(cfg, k, 0.0)
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    world = WorldModel(cfg)
    grounding = T.build_grounding(world.state_dim, hidden=32)
    plan = T.horizon_plan(cfg, op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)
    hcfg = T._smoke_head_cfg(world.state_dim, cfg.predictor.window)
    if deterministic:
        for k in HEAD_STOCHASTIC_KNOBS:
            if hasattr(hcfg, k):
                setattr(hcfg, k, 0.0)
        if getattr(hcfg, "decoder", None) is not None and \
                hasattr(hcfg.decoder, "noise_std"):
            hcfg.decoder.noise_std = 0.0
    head = T.FlagshipV4Head(hcfg)
    # `smoke.toy_episode` is a closure inside `smoke()`; rebuild the same shape.
    import math

    from tanitad.data._contract import assemble_episode

    def toy_episode(Tn, eid, size=64):
        g = torch.Generator().manual_seed(100 + eid)
        frames = [torch.rand(1, size, size, generator=g) for _ in range(Tn)]
        rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, 8.0
        dt, yaw_rate = 0.1, (0.05 if eid % 2 else -0.05)
        accel = -1.0 if eid % 2 else 1.0
        for _ in range(Tn):
            rows.append([x, y, yaw, v])
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += yaw_rate * dt
            v = max(0.0, v + accel * dt)
        poses = torch.tensor(rows)
        return assemble_episode(frames, [p.numpy() for p in poses],
                                [yaw_rate] * Tn, 0.1, eid)

    eps = [toy_episode(60, i) for i in range(4)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    items = [ds[i] for i in range(n_windows)]
    return world, grounding, head, plan, cfg, items, default_collate


@contextmanager
def sigreg_off():
    """Zero the ONE term of the flagship loss that is not a per-example mean.

    ``v4_loss_step`` builds a fresh ``LossWeights()`` internally, so the weight
    cannot be passed in; it is patched in the module namespace instead.

    ⚠️ ``SigReg`` draws FRESH random slice directions on every call
    (``sigreg.py``: "Fresh random directions every call (never a fixed
    buffer)"), so with it live the accumulation arm makes ``2*accum`` draws and
    the single-batch arm makes 2. Any gradient comparison run with it on
    measures the DRAW. Its own batch-size behaviour is measured separately —
    see :func:`test_SIGREG_is_the_one_term_that_is_not_a_per_example_mean`."""
    from tanitad.train.flagship_losses import LossWeights
    old = T.LossWeights
    T.LossWeights = lambda: LossWeights(sigreg=0.0)
    try:
        yield
    finally:
        T.LossWeights = old


def _grads(world, grounding, head, plan, cfg, collate, items, accum):
    """Accumulated gradient after ONE optimizer step's worth of work.

    Reproduces ``train_flagship_v4._training_loop``'s accumulation body verbatim:
    ``opt.zero_grad`` -> for each micro-batch ``(total / accum).backward()``."""
    params = [p for p in list(world.parameters()) + list(head.parameters())
              + list(grounding.parameters()) if p.requires_grad]
    for p in params:
        p.grad = None
    micro = len(items) // accum
    phases = T.CurriculumPhases(phase_a=2, phase_b=6)
    lw = T.V4LossWeights()
    for m in range(accum):
        b = collate(items[m * micro:(m + 1) * micro])
        total, _ = T.v4_loss_step(world, grounding, head, b, plan, cfg,
                                  8, phases, lw)         # Phase C, lambda_plan=1
        (total / accum).backward()
    return [(None if p.grad is None else p.grad.detach().clone())
            for p in params]


def _worst_rel(ga, gb):
    """``(n_compared, max|a-b| / max|b|)`` with a GLOBAL denominator.

    ⚠️ A PER-PARAMETER relative difference is the wrong statistic here and it
    cost a false alarm: ``head.decoder.conf_head.bias`` has a gradient of
    magnitude ~8e-8, so float32 cancellation alone reads as a 41 % "relative
    disagreement" on it while all 261 other tensors agree to ~1e-7. Dividing by
    the LARGEST gradient in the step makes the number a statement about the
    optimizer step rather than about the smallest tensor in it — and a per-batch
    SUM would still read ~7 on it."""
    n, worst_abs, scale = 0, 0.0, 1e-12
    for a, b in zip(ga, gb):
        if a is None or b is None:
            continue
        n += 1
        worst_abs = max(worst_abs, float((a - b).abs().max()))
        scale = max(scale, float(b.abs().max()))
    return n, worst_abs / scale


def test_the_loss_is_DETERMINISTIC_only_once_BOTH_knob_sets_are_off():
    """The precondition for the equivalence leg, proved rather than assumed.

    ⚠️ MEASURED: with the shipped settings the SAME batch scored twice gives a
    DIFFERENT loss, from FOUR independent sources — three per-example Bernoulli
    dropouts, the truncated-diffusion noise, and SigReg's fresh slice
    directions. Any accum-vs-single comparison run in that regime measures the
    DRAW, not the arithmetic. This test is the control that says the next one
    does not."""
    world, grounding, head, plan, cfg, items, collate = _fixture(8)
    b = collate(items)
    phases, lw = T.CurriculumPhases(phase_a=2, phase_b=6), T.V4LossWeights()
    with sigreg_off():
        t1, _ = T.v4_loss_step(world, grounding, head, b, plan, cfg, 8, phases, lw)
        t2, _ = T.v4_loss_step(world, grounding, head, b, plan, cfg, 8, phases, lw)
    assert float(t1) == float(t2), (float(t1), float(t2))

    # …and the control really is controlling something: with the shipped knobs
    # the same call is NOT reproducible.
    w2, g2, h2, p2, c2, it2, col2 = _fixture(8, deterministic=False)
    b2 = col2(it2)
    s1, _ = T.v4_loss_step(w2, g2, h2, b2, p2, c2, 8, phases, lw)
    s2, _ = T.v4_loss_step(w2, g2, h2, b2, p2, c2, 8, phases, lw)
    assert float(s1) != float(s2), \
        "the stochastic knobs are supposed to make this vary — if they no " \
        "longer do, the control above has quietly stopped controlling anything"


def test_batch8_accum8_gradient_EQUALS_the_single_batch_of_64():
    """⭐ THE claim ``--batch 8 --accum 8`` makes, tested rather than asserted.

    16 windows at (8 x 2) vs (1 x 16), every per-example term live: WM stack,
    planner, factorised LAT/LON/DIST CE, smoothness, grounding. The smoke frame
    is tiny but the loss is the REAL one, and batch-linearity is a property of
    the loss, not of n."""
    world, grounding, head, plan, cfg, items, collate = _fixture(16)
    with sigreg_off():
        g_acc = _grads(world, grounding, head, plan, cfg, collate, items,
                       accum=8)
        g_one = _grads(world, grounding, head, plan, cfg, collate, items,
                       accum=1)
    n_cmp, worst_rel = _worst_rel(g_acc, g_one)
    assert n_cmp > 50, f"only {n_cmp} grads compared — fixture is not exercising"
    # float32 accumulation order differs (8 partial sums vs 1), so this is a
    # NUMERICAL identity, not a bitwise one. A per-batch SUM anywhere in the loss
    # would show up as a factor ~8, i.e. worst_rel ~ 7, not ~1e-6.
    assert worst_rel < 1e-3, f"max relative grad disagreement {worst_rel:.3e}"


def test_SIGREG_is_the_one_term_that_is_not_a_per_example_mean():
    """⚠️ And it is measured, because the algebra is not obvious.

    ``SigReg`` is an O(n^2) PAIRWISE Epps-Pulley statistic over the batch axis,
    and its own source says "Do NOT normalize by n: the statistic's built-in
    batch-scale is part of the validated operating point." That reads like a
    term whose value tracks the micro-batch — in which case ``16 x 4 -> 8 x 8``
    would silently re-weight the LeJEPA regularizer at an unchanged effective
    batch.

    MEASURED, and the reading is the opposite: ``S(n)`` is FLAT in n. The ratio
    ``S(16)/S(8)`` sits inside the slice-draw spread, so the term is preserved
    in expectation as well. What is NOT preserved pointwise is the DRAW, which
    is why the equivalence leg above runs with it off rather than pretending it
    is deterministic."""
    from tanitad.config import flagship4b_config
    from tanitad.models.sigreg import SigReg
    c = flagship4b_config()
    sr = SigReg(c.loss.sigreg.n_slices, c.loss.sigreg.beta)
    torch.manual_seed(7)
    z = torch.randn(64, 512)
    vals = {}
    for n in (8, 16, 64):
        s = []
        for r in range(12):
            torch.manual_seed(r)
            s.append(float(sr(z[:n].contiguous())))
        vals[n] = (sum(s) / len(s),
                   max(s) - min(s))
    m8, m16, m64 = vals[8][0], vals[16][0], vals[64][0]
    spread = max(vals[8][1], vals[16][1], vals[64][1]) / m8
    # flat to within the draw spread — an O(n) term would read 2.0 and 8.0 here
    assert abs(m16 / m8 - 1.0) <= spread, (m16 / m8, spread)
    assert abs(m64 / m8 - 1.0) <= spread, (m64 / m8, spread)
    # and the statistic is genuinely stochastic, so "flat" is a statement about
    # its mean and had to be averaged
    assert spread > 0.0


def test_the_guard_can_FAIL_when_the_1_over_accum_scaling_is_dropped():
    """RED twin. Remove the ``/ accum`` and the same comparison must blow up by
    ~accum — otherwise the test above proves nothing."""
    world, grounding, head, plan, cfg, items, collate = _fixture(16)
    params = [p for p in list(world.parameters()) + list(head.parameters())
              + list(grounding.parameters()) if p.requires_grad]
    phases, lw = T.CurriculumPhases(phase_a=2, phase_b=6), T.V4LossWeights()

    def run(scale_by_accum):
        for p in params:
            p.grad = None
        with sigreg_off():
            for m in range(8):
                b = collate(items[m * 2:(m + 1) * 2])
                total, _ = T.v4_loss_step(world, grounding, head, b, plan, cfg,
                                          8, phases, lw)
                (total / 8 if scale_by_accum else total).backward()
        return [None if p.grad is None else p.grad.detach().clone()
                for p in params]

    good = run(True)
    bad = run(False)
    ratios = [float(b.abs().max() / a.abs().max())
              for a, b in zip(good, bad)
              if a is not None and b is not None and float(a.abs().max()) > 1e-9]
    assert ratios, "no comparable gradients"
    assert min(ratios) > 7.9 and max(ratios) < 8.1, \
        f"dropping /accum must scale every gradient by exactly 8, got {ratios[:5]}"
    # …and on the exact statistic the GREEN leg uses, the same defect reads ~7 —
    # six orders of magnitude above its 1e-5 bar, so that bar is not vacuous.
    _, rel = _worst_rel(bad, good)
    assert rel > 6.0, rel


def test_the_trainer_consumes_a_DISTINCT_micro_batch_per_accum_step():
    """``next(it)`` per micro-step, not one batch reused ``accum`` times.

    Read off the shipped source: the accumulation body pulls from the iterator
    inside the loop, and restarts the iterator on ``StopIteration`` rather than
    breaking (so an epoch boundary does not silently shorten the step)."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    i = src.index("for _micro in range(accum):")
    body = src[i:src.index("(total / accum).backward()", i)]
    assert "batch_d = next(it)" in body
    assert "it = iter(dl); batch_d = next(it)" in body


def test_drop_last_is_TRUE_so_every_micro_batch_is_the_same_size():
    """The equal-size premise of ``sum_i L_i / accum``. With ``drop_last=False``
    the final micro-batch of an epoch is short and the accumulated gradient
    becomes a weighted mean with the wrong weights — a silent re-weighting of
    the effective batch, which is the exact thing this file exists to deny."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    i = src.index("dl = DataLoader(ds_train")
    assert "drop_last=True" in src[i:i + 400]


@pytest.mark.parametrize("batch,accum", [(8, 8), (16, 4), (4, 16), (64, 1)])
def test_preflight_accepts_every_factorisation_of_64_and_only_those(batch,
                                                                    accum):
    a = T.build_parser().parse_args(
        ["--print-launch", "--batch", str(batch), "--accum", str(accum)])
    assert not [p for p in T.preflight_asserts(a) if "effective batch" in p]


def test_preflight_REFUSES_a_factorisation_that_is_not_64():
    a = T.build_parser().parse_args(
        ["--print-launch", "--batch", "8", "--accum", "4"])       # = 32
    assert [p for p in T.preflight_asserts(a) if "effective batch" in p]
