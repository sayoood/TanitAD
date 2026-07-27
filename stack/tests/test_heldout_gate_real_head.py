"""⭐ THE MID-RUN HELD-OUT GATE, AGAINST THE REAL HEAD — the tripwire, INVERTED.

⭐ **FIXED 2026-07-27 (the vtband-WIRING stream). These tests were a TRIPWIRE
pinning a live defect; they are now the regression guard for its fix**, inverted
exactly as the original file instructed rather than deleted.

THE CHOICE THE FIX HAD TO MAKE, recorded here because this file demanded it
---------------------------------------------------------------------------
``HeldoutGateConfig.goal_option`` now defaults to
:data:`~tanitad.train.heldout_gate.GOAL_OPTION_DEFAULT` == **``"dropped"``** —
``VT_DROPPED`` (= ``N_VTARGET_BANDS`` = 23) / ``ROUTE_DROPPED`` (= 4), the learned
withheld embedding rows — **NOT** the band-0 zeros this file's own "obvious patch"
paragraph warned about.

⚠️ **That default is the WIRING stream's choice PENDING THE PI's OVERRIDE.**
``VTBAND_DECISION.md`` priced all the options and deliberately declined to choose;
``--heldout-goal band0|produced`` flips it and nothing else changes. Why
``dropped`` (MEASURED, arm ``flagship-v4-fromscratch`` @ step 15000, 528 windows /
8 held-out episodes):

* ``goal_dropout = 0.5`` ships in ``V15Config``, is inherited by ``V4Config`` and
  is never overridden by the trainer — so those rows are trained on ~50 % of every
  batch. They are the **most frequently trained value of the channel**;
* ``band0`` is not a neutral zero: ``VTARGET_TOKENS[0] == "v_stop"`` and
  ``route 0 == ROUTE_LEFT``. MEASURED cost **−0.0621 [−0.0878, −0.0371]**
  (separated, paired episode-cluster bootstrap, B=2000), **93 % of it from the
  VTARGET channel alone**, by making the planner travel 9.1 % less;
* ⛔ the literal "build the zeros the provenance promises" patch is WORSE than
  either: it also zeroes ``vt_speed``, so the selector chases ``(v0 − 5)⁺`` and
  ranks up the maximally decelerating candidate — and a braking plan NaNs
  ``recovery`` by construction, so **the composite goes UP and the gate reads
  healthier while probing a planner that brakes.** It survives only as the named
  ``zeros_naive`` option and is BLOCKED by ``preflight_asserts``.

The fix is a **SIGNATURE change**: ``goal_kwargs_fn(states, v0)``, not
``(batch_size, device)`` — which carried neither, and therefore could not express
even ``band0`` faithfully (``vt_speed = v0``) nor ``produced`` at all.

WHAT HAPPENED — the defect these tests were written against
------------------------------------------------------------
(MEASURED 2026-07-27, pod2, a REAL v5 run on the REAL caches)
``train_flagship_v4 --heldout-gate`` on the v5 configuration
(``--v2-subframe 176x624``, ``--from-scratch``, the real ``FlagshipV4Head``)
**CRASHED at its first probe**::

    heldout_gate.py:426   probe -> pseudosim.pseudo_evaluate
    pseudosim.py:501      planner.traj(fw, v0, g)
    heldout_gate.py:223   out = self.head(states, v0, **kw)
    flagship_v15.py:456   ValueError: cond_vtarget is on but no vt_band supplied

``DeployableSurfacePlanner.traj`` builds ``kw = {}`` when no ``goal_kwargs_fn``
is given, and ``HeldoutGate.probe`` is called by the trainer without one. The
real head has ``cond_vtarget=True`` and **requires** ``vt_band``.

⚠️ **The consequence was a step-class blocker, not a quality note.** In the staged
v5 command ``--heldout-every 2000``, so the run trained for 2 000 optimizer steps
— several GPU-hours — and then died with a ``ValueError``. The gate exists to
save ~29.5 GPU-h (``flagship-v5-retrain.PREP.md`` cause #1); as shipped it cost
the run instead.

WHY NOTHING CAUGHT IT — and it is why this file keeps using the REAL head
--------------------------------------------------------------------------
Every other test of this path uses a STUB head. Stubs accept ``**kw`` and never
assert on it, so the whole gate was green against a head that *cannot refuse*.
The one component that can refuse had never been on the path. **These tests keep
it there**, which is the only reason they still earn their runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.train.heldout_gate import DeployableSurfacePlanner     # noqa: E402

import train_flagship_v4 as T                                      # noqa: E402


def _real_stack():
    """The REAL WorldModel + FlagshipV4Head at the SHIPPED goal conditioning.

    ⚠️ Sizes come from ``_smoke_head_cfg`` so this runs on CPU in milliseconds,
    but ``cond_vtarget`` / ``cond_route`` are put BACK to what ``v4_config()``
    ships — which is what a real run builds (``train_flagship_v4.py`` line
    ~1207: ``hcfg = v4_config()``). ``_smoke_head_cfg`` sets them to ``False``,
    and that single line is why every existing test walks past this defect."""
    import dataclasses

    from tanitad.config import flagship4b_smoke_config
    from tanitad.models.flagship_v4 import v4_config
    from tanitad.models.fourbrain import WorldModel

    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    world = WorldModel(cfg)
    hcfg = T._smoke_head_cfg(world.state_dim, cfg.predictor.window)
    shipped = v4_config()
    assert shipped.cond_vtarget and shipped.cond_route, \
        "v4_config() no longer conditions on vt_band/route — re-read this file"
    hcfg.cond_vtarget = shipped.cond_vtarget          # as a real run builds it
    hcfg.cond_route = shipped.cond_route
    head = T.FlagshipV4Head(hcfg)
    return world, head, cfg


def test_the_REAL_head_STILL_refuses_a_planner_given_NO_goal_at_all():
    """⛔ INVERTED HALF 1 — the head's refusal is PRESERVED, not patched away.

    The original tripwire pinned this ``ValueError`` as the defect. It is not the
    defect: it is :class:`FlagshipV4Head`'s own ``ego_guard`` — refusing to score
    a conditioned checkpoint blind rather than silently emitting a number. The
    defect was that ``HeldoutGate.probe`` walked into it.

    So this assertion **stays**, now guarding the opposite thing: that a future
    "fix" never makes the head tolerate a missing ``vt_band``. ``goal_kwargs_fn is
    None`` is the ``crash_today`` RED baseline, reachable only by naming it."""
    world, head, cfg = _real_stack()
    assert head.cfg.cond_vtarget, \
        "the shipped v4/v5 head conditions on vt_band; if that changed, this " \
        "whole file needs re-reading rather than deleting"

    planner = DeployableSurfacePlanner(world, head, device="cpu")
    assert planner.goal_kwargs_fn is None
    b, W = 2, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    with pytest.raises(ValueError, match="cond_vtarget is on but no vt_band"):
        planner.traj(frames, torch.full((b,), 8.0))


def test_the_provenance_no_longer_CLAIMS_a_default_the_code_does_not_build():
    """⭐ INVERTED HALF 2 — the string that lied is gone.

    ``DeployableSurfacePlanner`` used to advertise "withheld/unknown defaults
    (zeros) — the deployed no-route state" in its ``provenance`` while passing
    ``{}``. Every probe record carried that string, which is *why the gap read as
    covered*. A provenance field that describes a default the code does not build
    is worse than an absent one."""
    world, head, _ = _real_stack()
    p = DeployableSurfacePlanner(world, head, device="cpu")
    cond = p.provenance["goal_conditioning"]
    assert "zeros" not in cond, cond
    assert "NONE" in cond and "REFUSES" in cond, cond
    assert p.provenance["goal_protocol"] == "goal_kwargs_fn(states, v0)"


def test_the_gates_OWN_DEFAULT_now_plans_on_the_real_head():
    """⭐ THE FIX ITSELF: no caller-supplied fn, no explicit kwargs — the shipped
    default alone must make the real head plan.

    This is the assertion the whole file exists for. Previously the *only* green
    path was one where the test supplied the kwargs by hand, which proved the head
    was fine and told you nothing about what a run would do."""
    import tanitad.train.heldout_gate as HG
    from tanitad.train.heldout_goal import make_goal_kwargs_fn

    world, head, cfg = _real_stack()
    assert HG.GOAL_OPTION_DEFAULT == "dropped", \
        "the default changed — update this file's docstring, which RECORDS the " \
        "choice, rather than only this assertion"
    assert HG.HeldoutGateConfig().goal_option == HG.GOAL_OPTION_DEFAULT

    fn = make_goal_kwargs_fn(HG.GOAL_OPTION_DEFAULT, head.cfg)
    planner = DeployableSurfacePlanner(world, head, device="cpu",
                                       goal_kwargs_fn=fn,
                                       goal_option=HG.GOAL_OPTION_DEFAULT)
    b, W = 2, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    traj = planner.traj(frames, torch.full((b,), 8.0))
    assert traj.shape[0] == b and traj.shape[-1] == 2
    assert torch.isfinite(traj).all()
    # and the record says WHICH goal, in words, not just a boolean
    assert planner.provenance["goal_option"] == "dropped"
    assert "VT_DROPPED" in planner.provenance["goal_option_meaning"]


def test_the_trainer_now_passes_the_goal_through_probe():
    """The other half of the mechanism, read off the shipped trainer.

    The old call site passed only ``device=``, so ``goal_kwargs_fn`` stayed
    ``None`` and the empty-kwargs path was the one a real v5 run took. It now
    forwards ``goal_head`` (needed by ``--heldout-goal produced``) and the gate
    builds the fn from ``cfg.goal_option``."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    i = src.index("heldout_gate.probe(")
    call = src[i:i + 260]
    assert "goal_head=goal_head" in call, call
    # the option reaches the config, and the flag exists to override it
    assert "goal_option=a.heldout_goal" in src
    assert '"--heldout-goal"' in src


def test_zeros_naive_and_crash_today_are_BLOCKED_by_preflight():
    """⛔ The two options that must never reach a 30k run by accident.

    ``zeros_naive`` is the patch the old provenance string invited and is the
    WORST option: it zeroes ``vt_speed`` too, so the selector chases
    ``(v0 − 5)⁺``, the plan brakes, ``recovery`` goes NaN on the gate's
    ``dlat = 0`` grid — and the composite goes UP. **A gate patched that way
    reports a healthier run while probing a braking planner.**"""
    for opt, needle in (("zeros_naive", "PRICED TRAP"),
                        ("crash_today", "RED baseline"),
                        ("band0_vt_only", "DIAGNOSTIC"),
                        ("band0_route_only", "DIAGNOSTIC")):
        a = _launch_args(heldout_goal=opt)
        problems = T.preflight_asserts(a)
        hits = [p for p in problems if "HELDOUT-GATE" in p and needle in p]
        assert hits, f"{opt} is not blocked by preflight: {problems}"
    # ...and every real candidate passes
    for opt in ("dropped", "band0", "produced"):
        a = _launch_args(heldout_goal=opt)
        assert not [p for p in T.preflight_asserts(a)
                    if "HELDOUT-GATE" in p], opt


def _launch_args(**over):
    """The SHIPPED parser's defaults with overrides — never a hand-built
    Namespace, which would silently drift from the flags a run actually parses."""
    argv = ["--print-launch"]
    for k, v in over.items():
        argv += ["--" + k.replace("_", "-"), str(v)]
    return T.build_parser().parse_args(argv)


def test_no_existing_heldout_test_uses_the_real_head():
    """Why the suite was green. Named so the next person does not have to
    re-derive it — and so that adding a real-head test elsewhere trips this and
    prompts a re-read rather than silently duplicating coverage."""
    here = Path(__file__).parent
    for name in ("test_heldout_gate.py", "test_v5_trainer_v2_val.py"):
        s = (here / name).read_text(encoding="utf-8")
        assert "FlagshipV4Head" not in s, \
            f"{name} now builds a real head — re-read this file's premise"
