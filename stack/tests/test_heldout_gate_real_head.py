"""⛔ THE MID-RUN HELD-OUT GATE HAS NEVER BEEN RUN AGAINST THE REAL HEAD.

WHAT HAPPENED (MEASURED 2026-07-27, pod2, a REAL v5 run on the REAL caches)
---------------------------------------------------------------------------
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

WHY NOTHING CAUGHT IT — and it is the reason this file exists
--------------------------------------------------------------
Every existing test of this path uses a STUB head: ``test_heldout_gate.py``'s
``_FakeHead``/``_VaryingHead``, and ``test_v5_trainer_v2_val.py``'s ``_Planner``
(whose trainer-level test additionally REPLACES ``probe`` outright). Stubs accept
``**kw`` and never assert on it, so the whole gate is green against a head that
cannot refuse. The one component that can refuse had never been on the path.

⚠️ **The consequence is a step-class blocker, not a quality note.** In the staged
v5 command ``--heldout-every 2000``, so the run trains for 2 000 optimizer steps
— several GPU-hours — and then dies with a ``ValueError``. The gate exists to
save ~29.5 GPU-h (``flagship-v5-retrain.PREP.md`` cause #1); as shipped it costs
the run instead.

WHAT THIS FILE DOES — a TRIPWIRE, deliberately not a fix
---------------------------------------------------------
The tests below pin the defect **as it is today**, with the real head config, so
that:

* it cannot be re-discovered by a GPU-week rather than by a unit test;
* whoever fixes it is told by a failing test that these assertions must be
  INVERTED, and is pointed at the semantic question the fix has to answer.

⛔ **It is not fixed here on purpose.** The obvious patch — have
``DeployableSurfacePlanner`` build the kwargs its own docstring already promises
("withheld/unknown defaults (zeros)") — is not obviously the right one:

* ``train_flagship_v4._goal_inputs`` falls back to ``vt_band = zeros``, and
  **zero is BAND 0, a real target-speed band** — not the withheld state;
* the genuinely withheld values are ``flagship_v15.VT_DROPPED`` (=
  ``N_VTARGET_BANDS``) and ``ROUTE_DROPPED`` (= 4), which the head's own dropout
  path uses;
* the two measure DIFFERENT deployable surfaces, and ``GATE_PROTOCOL`` §0.8 is
  explicit that what the model is given changes what the measurement is OF.

Choosing between them decides what the v5 early-stop stops on. That is an owner
decision, not a silent default.
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


def test_the_REAL_head_requires_goal_kwargs_the_gates_default_does_not_supply():
    """⛔ THE DEFECT, pinned. Reproduces pod2's crash in 40 ms instead of 2 000
    optimizer steps.

    ⚠️ WHEN THIS TEST FAILS the defect has been fixed — do NOT delete it,
    INVERT it, and record in the fix which goal state was chosen (band-0 zeros
    vs VT_DROPPED/ROUTE_DROPPED) and why."""
    world, head, cfg = _real_stack()
    assert head.cfg.cond_vtarget, \
        "the shipped v4/v5 head conditions on vt_band; if that changed, this " \
        "whole file needs re-reading rather than deleting"

    planner = DeployableSurfacePlanner(world, head, device="cpu")
    assert planner.goal_kwargs_fn is None
    assert planner.provenance["goal_conditioning"].startswith("withheld/unknown")

    b, W = 2, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    with pytest.raises(ValueError, match="cond_vtarget is on but no vt_band"):
        planner.traj(frames, torch.full((b,), 8.0))


def test_the_provenance_string_CLAIMS_a_default_the_code_does_not_build():
    """⚠️ The docstring is not merely silent — it is WRONG, and that is why the
    gap read as covered.

    ``DeployableSurfacePlanner`` advertises "withheld/unknown defaults (zeros)"
    in its own ``provenance``, and every probe record carries that string. What
    the code actually passes is ``{}`` — no keys at all."""
    import tanitad.train.heldout_gate as HG
    world, head, _ = _real_stack()
    p = DeployableSurfacePlanner(world, head, device="cpu")
    assert "zeros" in p.provenance["goal_conditioning"]
    src = Path(HG.__file__).read_text(encoding="utf-8")
    i = src.index("def traj(self, frames, v0, goal=None):")
    body = src[i:i + 900]
    assert "else {})" in body, \
        "traj no longer defaults to an EMPTY kwargs dict — if it now builds " \
        "the zeros it advertises, invert this test and the one above"


def test_supplying_the_kwargs_EXPLICITLY_makes_the_real_head_work():
    """⭐ The GREEN half: the head is fine, the gate's default is not.

    This is what makes the two tests above a statement about
    ``DeployableSurfacePlanner`` rather than about ``FlagshipV4Head`` — and it is
    also, mechanically, the shape of the fix."""
    world, head, cfg = _real_stack()

    def goal_kwargs(b, device):
        return T._goal_inputs(head.cfg, {}, torch.zeros(b, device=device))

    planner = DeployableSurfacePlanner(world, head, device="cpu",
                                       goal_kwargs_fn=goal_kwargs)
    b, W = 2, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    traj = planner.traj(frames, torch.full((b,), 8.0))
    assert traj.shape[0] == b and traj.shape[-1] == 2
    assert torch.isfinite(traj).all()
    assert planner.provenance["goal_conditioning"] == "caller-supplied goal_kwargs_fn"


def test_the_trainer_calls_probe_WITHOUT_a_goal_kwargs_fn():
    """The other half of the mechanism, read off the shipped trainer.

    ``HeldoutGate.probe`` takes ``goal_kwargs_fn=None`` by default and the
    trainer's call site passes only ``device=``/``verbose=`` — so the empty-kwargs
    path above is the one a real v5 run takes."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    i = src.index("heldout_gate.probe(")
    call = src[i:i + 260]
    assert "goal_kwargs_fn" not in call, call


def test_no_existing_heldout_test_uses_the_real_head():
    """Why the suite was green. Named so the next person does not have to
    re-derive it — and so that adding a real-head test elsewhere trips this and
    prompts a re-read rather than silently duplicating coverage."""
    here = Path(__file__).parent
    for name in ("test_heldout_gate.py", "test_v5_trainer_v2_val.py"):
        s = (here / name).read_text(encoding="utf-8")
        assert "FlagshipV4Head" not in s, \
            f"{name} now builds a real head — re-read this file's premise"
