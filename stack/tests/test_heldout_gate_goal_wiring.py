"""⭐ THE WIRING — ``--heldout-gate`` reaches its probe instead of dying at step 2 000.

WHAT THIS FILE GUARDS (and what the other two files do NOT)
-----------------------------------------------------------
* ``test_heldout_gate_real_head.py`` proves the real head PLANS under the default.
* ``test_vtband_options.py`` proves the OPTIONS are what they claim to be.
* **This file proves the two are actually CONNECTED** — that
  :meth:`HeldoutGate.probe` builds a goal by itself, that the trainer's flag
  reaches it, and that the choice is recorded in the emitted record.

That gap is the whole reason the fix was needed twice over: ``heldout_goal.py``
shipped on 2026-07-27 with every option implemented and tested, **and nothing
imported it**, so ``--heldout-gate`` still crashed. A module can be correct,
tested and inert at the same time; only a wiring test can tell you which.

⚠️ **THE DEFAULT IS AN AGENT'S CHOICE PENDING THE PI's OVERRIDE.**
``VTBAND_DECISION.md`` priced the options and deliberately declined to choose.
The wiring stream picked ``"dropped"`` so the gate could run at all, and the
tests below pin that **flipping it costs exactly one flag** — that property is
load-bearing, not cosmetic, because it is what makes the default overridable
without a code change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))

import tanitad.train.heldout_gate as HG                            # noqa: E402
import tanitad.train.heldout_goal as HGOAL                         # noqa: E402

import train_flagship_v4 as T                                      # noqa: E402


def _real_stack():
    """The REAL WorldModel + FlagshipV4Head at the SHIPPED goal conditioning.

    Sizes from ``_smoke_head_cfg`` (CPU, milliseconds); ``cond_vtarget`` /
    ``cond_route`` put back to what ``v4_config()`` ships — the single line whose
    absence let every other test walk past the defect."""
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
    # ⚠️ the head's window must match what pseudo_evaluate actually FEEDS it
    # (``pseudosim.WINDOW``), not the predictor's — the gate slices the frame
    # window itself, so a head built at a different window is a fixture bug that
    # would look like a wiring failure.
    from taniteval.pseudosim import WINDOW
    hcfg = T._smoke_head_cfg(world.state_dim, WINDOW)
    shipped = v4_config()
    hcfg.cond_vtarget = shipped.cond_vtarget
    hcfg.cond_route = shipped.cond_route
    hcfg.goal_dropout = shipped.goal_dropout
    return world, T.FlagshipV4Head(hcfg), cfg


def _legacy_frame():
    """⚠️ These probes run on a SYNTHETIC 64x64 raster, which no CanonicalFrame
    describes. ``clhorizon.LEGACY_WARP`` says so outright — it pins the shipped
    266/128 constants on whatever raster, and is never a published geometry. A
    real run passes the TRAIN frame (``train_flagship_v4`` wires
    ``resolve_v2_frames``' second return value into ``HeldoutGateConfig.frame``),
    and the end-to-end proof of THAT lives on pod2, not in a unit test."""
    from taniteval.clhorizon import LEGACY_WARP
    return LEGACY_WARP


def _episodes(cfg, n=4, T_steps=64):
    """Held-out episodes: ``.poses`` = [x, y, yaw, v], ``.frames``.

    ⚠️ Two fixture properties are load-bearing, and both were learned by watching
    the gate refuse:

    1. ``v0`` is ``poses[:, 3]`` and must be NON-ZERO — the whole ``vt_band``
       question is a target speed *relative to* ``v0``, so a stationary fixture
       makes every option agree by accident.
    2. speed and curvature must VARY ACROSS EPISODES. ``discriminative_range``
       refuses to emit a composite whose components have no range ("a metric that
       cannot fail is not a metric"), so a fixture of identical straight lines
       makes the gate raise ``GateNotUsableError`` — which looks exactly like a
       wiring failure and is not one."""
    import math

    ch = cfg.encoder.in_channels
    size = cfg.encoder.image_size
    eps = []
    for e in range(n):
        ep = type("Ep", (), {})()
        g = torch.Generator().manual_seed(7 + e)
        ep.frames = torch.rand(T_steps, ch, size, size, generator=g)
        v = 3.0 + 2.0 * e                     # 3, 5, 7, 9 m/s
        yaw_rate = 0.02 * (e - 1.5)           # both turn directions + ~straight
        yaw = torch.arange(T_steps).float() * yaw_rate
        dx = torch.cos(yaw) * v * 0.1
        dy = torch.sin(yaw) * v * 0.1
        ep.poses = torch.stack([torch.cumsum(dx, 0), torch.cumsum(dy, 0), yaw,
                                torch.full((T_steps,), float(v))], dim=-1)
        eps.append(ep)
    assert math.isfinite(float(eps[0].poses.sum()))
    return eps


# --------------------------------------------------------------------------- #
# 1. the gate builds its own goal — the thing that was missing                 #
# --------------------------------------------------------------------------- #
def test_probe_builds_a_goal_ITSELF_and_the_real_head_no_longer_refuses():
    """⭐ THE HEADLINE. ``HeldoutGate.probe`` on the REAL head, with NO
    ``goal_kwargs_fn`` argument — exactly the trainer's call shape.

    Before the wiring this raised ``ValueError: cond_vtarget is on but no
    vt_band supplied`` from inside ``pseudo_evaluate``. The run reached this
    point ~2 000 optimizer steps in."""
    pytest.importorskip("taniteval.pseudosim")
    world, head, cfg = _real_stack()
    gate = HG.HeldoutGate(HG.HeldoutGateConfig(
        every=1, episodes=3, stride=4, horizon=4, batch=4, n_boot=50,
        frame=_legacy_frame()))
    rec = gate.probe(0, world, head, _episodes(cfg), device="cpu")

    assert rec["primary"] == HG.PRIMARY_NAME
    assert rec["n_episodes"] >= 2 and rec["n_windows"] >= 2
    # ...and it says WHICH goal it used, in the record a reader will see
    assert rec["pseudosim"]["goal_option"] == "dropped"
    assert "PENDING THE PI" in rec["pseudosim"]["goal_option_provenance"]
    assert rec["pseudosim"]["surface"]["goal_protocol"] == "goal_kwargs_fn(states, v0)"


def test_the_option_is_ONE_FLAG_and_the_alternatives_actually_change_the_probe():
    """⚠️ 'Overridable in one flag' is a CLAIM about behaviour, so it is tested as
    one: the same gate under ``band0`` must produce a *different* probe value,
    not merely a different provenance string.

    If they came out equal the flag would be decorative — and that is not a
    hypothetical: at random init ``band0`` and ``zeros_naive`` are bit-identical
    (``sel_gate = 0.0`` exactly), which is why ``VTBAND_DECISION.md``'s numbers
    had to come from a TRAINED head."""
    pytest.importorskip("taniteval.pseudosim")
    world, head, cfg = _real_stack()
    eps = _episodes(cfg)
    vals = {}
    for opt in ("dropped", "band0"):
        gate = HG.HeldoutGate(HG.HeldoutGateConfig(
            every=1, episodes=3, stride=4, horizon=4, batch=4, n_boot=50,
            frame=_legacy_frame(), goal_option=opt))
        rec = gate.probe(0, world, head, eps, device="cpu")
        vals[opt] = rec["primary_value"]
        assert rec["pseudosim"]["goal_option"] == opt
    assert vals["dropped"] != vals["band0"], vals


def test_the_default_is_dropped_and_is_marked_as_an_AGENTS_choice():
    """The default and its provenance are both part of the contract.

    ⚠️ A default nobody can tell was *chosen* is indistinguishable from one that
    was adjudicated. The string travels in every probe record for that reason."""
    assert HG.GOAL_OPTION_DEFAULT == "dropped"
    assert HG.HeldoutGateConfig().goal_option == "dropped"
    prov = HG.GOAL_OPTION_PROVENANCE
    assert "PENDING THE PI" in prov and "--heldout-goal" in prov
    assert "VTBAND_DECISION" in prov


def test_the_trainer_flag_reaches_the_gate_config_and_the_staged_command():
    """⛔ A goal option that did not survive ``--print-launch`` would let a
    reconstructed launch silently re-decide what the early-stop measures."""
    a = T.build_parser().parse_args(["--print-launch", "--heldout-goal", "band0"])
    assert a.heldout_goal == "band0"
    cmd = T.launch_command(a) if hasattr(T, "launch_command") else None
    if cmd is None:                     # fall back to the source contract
        src = Path(T.__file__).read_text(encoding="utf-8")
        assert '("--heldout-goal", a.heldout_goal)' in src
    else:
        assert "--heldout-goal band0" in cmd


def test_the_default_option_survives_a_round_trip_through_the_parser():
    a = T.build_parser().parse_args(["--print-launch"])
    assert a.heldout_goal == HG.GOAL_OPTION_DEFAULT
    cfg = HG.HeldoutGateConfig(goal_option=a.heldout_goal)
    assert cfg.goal_option == "dropped"


# --------------------------------------------------------------------------- #
# 2. the refusals stay refusals                                                #
# --------------------------------------------------------------------------- #
def test_probe_REFUSES_produced_without_a_goal_head_rather_than_downgrading():
    """⛔ Falling back to a cheaper option would make the probe's meaning depend
    on a call-site accident — the exact class ``GATE_PROTOCOL`` §0.8 forbids."""
    world, head, cfg = _real_stack()
    gate = HG.HeldoutGate(HG.HeldoutGateConfig(
        every=1, episodes=3, stride=4, horizon=4, batch=4, n_boot=50,
        frame=_legacy_frame(), goal_option="produced"))
    with pytest.raises(ValueError, match="goal_head"):
        gate.probe(0, world, head, _episodes(cfg), device="cpu")


def test_probe_REFUSES_dropped_on_a_config_that_turned_the_dropout_OFF():
    """The ``VT_DROPPED`` rows are only admissible because ``goal_dropout = 0.5``
    trains them. At 0 they are ``N(0, 0.02)`` init noise and the probe would be
    measuring initialisation."""
    world, head, cfg = _real_stack()
    head.cfg.goal_dropout = 0.0
    gate = HG.HeldoutGate(HG.HeldoutGateConfig(
        every=1, episodes=3, stride=4, horizon=4, batch=4, n_boot=50,
        frame=_legacy_frame()))
    with pytest.raises(ValueError, match="goal_dropout > 0"):
        gate.probe(0, world, head, _episodes(cfg), device="cpu")


def test_the_dropped_refusal_is_SCOPED_to_heads_that_have_the_rows():
    """⚠️ …but a head conditioned on NEITHER channel never receives a categorical
    at all, so refusing it would disable the gate for no benefit — a
    silently-unusable early-stop is the failure this module removes.

    (This is why the guard checks ``cond_vtarget or cond_route`` and not just
    ``goal_dropout``.)"""
    _, head, cfg = _real_stack()
    head.cfg.goal_dropout = 0.0
    head.cfg.cond_vtarget = head.cfg.cond_route = False
    fn = HGOAL.make_goal_kwargs_fn("dropped", head.cfg)
    assert fn(None, torch.full((2,), 8.0)) == {}


def test_crash_today_STILL_reproduces_the_original_crash_through_the_gate():
    """The RED baseline must remain reachable and must remain red — otherwise
    the regression this file guards could be re-introduced untested."""
    pytest.importorskip("taniteval.pseudosim")
    world, head, cfg = _real_stack()
    gate = HG.HeldoutGate(HG.HeldoutGateConfig(
        every=1, episodes=3, stride=4, horizon=4, batch=4, n_boot=50,
        frame=_legacy_frame(), goal_option="crash_today"))
    with pytest.raises(ValueError, match="cond_vtarget is on but no vt_band"):
        gate.probe(0, world, head, _episodes(cfg), device="cpu")


def test_an_unknown_option_is_refused_at_the_probe_not_silently_ignored():
    world, head, cfg = _real_stack()
    gate = HG.HeldoutGate(HG.HeldoutGateConfig(
        every=1, episodes=3, stride=4, horizon=4, batch=4, n_boot=50,
        frame=_legacy_frame(), goal_option="whatever"))
    with pytest.raises(ValueError, match="option must be one of"):
        gate.probe(0, world, head, _episodes(cfg), device="cpu")


# --------------------------------------------------------------------------- #
# 3. the module is no longer INERT                                             #
# --------------------------------------------------------------------------- #
def test_heldout_goal_is_actually_IMPORTED_by_the_shipped_path():
    """⭐ The inertness check, stated as a test because prose said it was wired
    once already and it was not.

    ``heldout_goal.py`` shipped complete, tested and unreferenced. 'Nothing
    imports it' is a property no unit test of the module itself can detect."""
    gate_src = Path(HG.__file__).read_text(encoding="utf-8")
    assert "heldout_goal" in gate_src, \
        "heldout_gate no longer references heldout_goal — the option registry " \
        "is inert again and --heldout-gate is back to crashing at step 2000"
    trainer_src = Path(T.__file__).read_text(encoding="utf-8")
    assert "heldout_goal" in trainer_src


def test_the_provenance_that_LIED_is_gone_from_the_planner():
    """The old string advertised "withheld/unknown defaults (zeros)" while the
    code passed ``{}``. It read as coverage and was the reason the gap survived
    review."""
    world, head, cfg = _real_stack()
    fn = HGOAL.make_goal_kwargs_fn("dropped", head.cfg)
    p = HG.DeployableSurfacePlanner(world, head, device="cpu",
                                    goal_kwargs_fn=fn, goal_option="dropped")
    assert "zeros" not in p.provenance["goal_conditioning"]
    assert p.provenance["goal_option"] == "dropped"
