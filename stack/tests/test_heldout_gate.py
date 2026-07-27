"""The MID-RUN HELD-OUT GATE — tanitad/train/heldout_gate.py.

Every test here is driven with input DESIGNED TO MAKE THE GUARD FAIL. A guard
exercised only with input it accepts has not been tested, and this module is the
fix for the largest single measured waste in the program (~29.5 GPU-h — half the
v4 30k run — spent training past the best checkpoint while every training term
improved). If it is wrong it is worse than nothing, because it will be trusted.

The load-bearing claims pinned below:

1. the primary is the map-free COMPOSITE, and a run that decays on it while
   ``ade_0_2s`` improves is STOPPED — with the ADE-driven control shown NOT to
   stop, so the axis choice is proven load-bearing rather than asserted;
2. ONE separated-worse probe does not stop (an unpowered read is not a
   refutation); TWO consecutive ones do;
3. a worse-but-UNSEPARATED probe does not stop at all;
4. the incumbent advances only on a SEPARATED improvement;
5. misaligned windows RAISE instead of being silently compared;
6. the streak survives a checkpoint round-trip — a restart cannot launder a
   decayed run past the gate;
7. an unusable probe RAISES rather than silently disabling the gate;
8. the probe grid is inside the MEASURED envelope and still carries a perturbed
   heading (without one, ``recovery`` is NaN by construction and the gate is
   blind to exactly the collapse it exists to catch);
9. the planner adapter reads the SELECTED trajectory, not the oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # stack/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))

from tanitad.train.heldout_gate import (  # noqa: E402
    DeployableSurfacePlanner, GateNotUsableError, HeldoutGate,
    HeldoutGateConfig, NonDensePlanError, PRIMARY_NAME, REFUSED_PRIMARY,
    WindowAlignmentError, probe_grid)

# fast but still a real interval; the shipped default is B=2000
NB = 400
N_EP, N_PER_EP = 10, 6


def _eid():
    return [f"ep{e}" for e in range(N_EP) for _ in range(N_PER_EP)]


def _series(level, *, spread=0.02, seed=0):
    """Per-window composite at a given mean ``level``, same window order always.

    The per-window noise is drawn from a FIXED seed so the two arms of a paired
    comparison share their window-level difficulty — which is the situation the
    paired estimator exists for, and the situation a real probe is in.
    """
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0.0, spread, N_EP * N_PER_EP)).tolist()


def _gate(**kw):
    kw.setdefault("n_boot", NB)
    return HeldoutGate(HeldoutGateConfig(**kw))


# =========================================================================== #
# 1. THE AXIS — the composite decides, ADE does not                            #
# =========================================================================== #
def test_a_run_that_decays_on_the_composite_while_ADE_IMPROVES_is_stopped():
    """⭐ THE CENTRAL TEST, built from the v4 30k failure's own shape.

    Every training term improved. Held-out selection was separated-worse. A gate
    on ``ade_0_2s`` sees the first and stops nothing; a gate on the map-free
    composite sees the second and stops the run.

    Input is constructed adversarially: ADE falls monotonically (looks great)
    while the composite falls too (higher-is-better, so that is a DECAY).
    """
    g = _gate()
    ade = [0.62, 0.55, 0.48, 0.41]
    comp = [0.80, 0.66, 0.52, 0.40]
    recs = [g.observe(step=i * 2000, per_window=_series(c),
                      eid=_eid(), diagnostics={"ade_0_2s": a})
            for i, (c, a) in enumerate(zip(comp, ade))]

    assert g.stop, ("the gate did NOT stop a run that decayed on the primary — "
                    f"history={[r['primary_value'] for r in recs]}")
    assert recs[1]["separated_worse"] and recs[2]["separated_worse"]
    # it stopped on the SECOND consecutive separated-worse probe, not the first
    assert recs[1]["stop"] is False and recs[2]["stop"] is True
    assert PRIMARY_NAME in g.stop_reason
    # the ADE it was handed only ever travelled as a diagnostic
    assert recs[2]["diagnostics"]["ade_0_2s"] == 0.48
    assert recs[2]["_diagnostics_are_not_the_rule"] == REFUSED_PRIMARY


def test_the_ADE_control_does_NOT_stop_on_the_same_run():
    """The falsifier for the test above: the axis choice must be load-bearing.

    Feed the SAME run's ``ade_0_2s`` as the primary (negated, so lower-is-better
    becomes higher-is-better and the rule's direction is respected). ADE is
    improving, so an ADE-gated run sails on — while the composite-gated run
    above stopped. If both stopped, the axis argument would be decorative.
    """
    g = _gate()
    for i, a in enumerate([0.62, 0.55, 0.48, 0.41]):
        g.observe(step=i * 2000, per_window=_series(-a), eid=_eid())
    assert not g.stop, "the ADE control stopped — the two axes did not disagree"


# =========================================================================== #
# 2/3. PATIENCE and SEPARATION                                                 #
# =========================================================================== #
def test_ONE_separated_worse_probe_does_not_stop():
    """A single separated probe is a sample. The program's standing rule is that
    an unpowered/isolated read is not a refutation — patience is 2."""
    g = _gate()
    g.observe(0, _series(0.80), _eid())
    r1 = g.observe(2000, _series(0.60), _eid())
    assert r1["separated_worse"] and r1["worse_streak"] == 1
    assert not g.stop


def test_a_recovery_between_two_bad_probes_RESETS_the_streak():
    """Designed to make a naive counter fail: bad, good, bad must NOT stop.

    A gate that counted separated-worse probes cumulatively instead of
    CONSECUTIVELY would stop here — and would have killed a run that recovered.
    """
    g = _gate()
    g.observe(0, _series(0.80), _eid())
    assert g.observe(2000, _series(0.60), _eid())["worse_streak"] == 1
    assert g.observe(4000, _series(0.82), _eid())["worse_streak"] == 0
    r = g.observe(6000, _series(0.62), _eid())
    assert r["worse_streak"] == 1 and not g.stop


def _shifted(base, per_episode_delta):
    """``base`` shifted by an explicit PER-EPISODE delta — the paired estimator's
    own unit, so the construction is exact rather than seed-lucky."""
    return [v + per_episode_delta[i // N_PER_EP] for i, v in enumerate(base)]


def test_a_worse_but_UNSEPARATED_probe_never_counts():
    """Noise must not stop a run. The delta is negative but the CI covers zero.

    Constructed exactly: half the episodes move down 0.50 and half move up 0.48,
    so the point estimate is worse (-0.01) while the episode-level spread is
    enormous and the interval cannot separate. This is precisely the case a
    point-estimate comparison gets wrong — and it is the reason the rule is
    'separated-worse', not 'worse'.
    """
    base = _series(0.80, seed=1)
    deltas = [-0.50 if e % 2 == 0 else +0.48 for e in range(N_EP)]
    g = _gate()
    g.observe(0, base, _eid())
    r = g.observe(2000, _shifted(base, deltas), _eid())
    assert r["paired"]["delta"] < 0, r["paired"]
    assert not r["paired"]["separated"], r["paired"]
    assert r["separated_worse"] is False and r["worse_streak"] == 0
    r2 = g.observe(4000, _shifted(base, deltas), _eid())
    assert not g.stop and r2["worse_streak"] == 0


# =========================================================================== #
# 4. THE INCUMBENT                                                             #
# =========================================================================== #
def test_the_incumbent_advances_only_on_a_SEPARATED_improvement():
    """A lucky point estimate must never become the bar the stop rule fires at.

    Probe 1 is nominally better but not separated -> the incumbent must NOT move.
    Probe 2 is separated better -> it must.
    """
    base = _series(0.80, seed=1)
    blip = [+0.50 if e % 2 == 0 else -0.48 for e in range(N_EP)]   # mean +0.01
    g = _gate()
    g.observe(0, base, _eid())
    r1 = g.observe(2000, _shifted(base, blip), _eid())
    assert r1["paired"]["delta"] > 0 and not r1["paired"]["separated"]
    assert r1["incumbent_step"] == 0, "the incumbent moved on an unseparated blip"

    r2 = g.observe(4000, _series(1.60), _eid())
    assert r2["separated_better"] and r2["incumbent_step"] == 4000


# =========================================================================== #
# 5. ALIGNMENT                                                                 #
# =========================================================================== #
def test_misaligned_windows_RAISE_instead_of_being_compared():
    """An unpaired 'paired' test is not a weaker test, it is a wrong one.

    Driven with a probe over a DIFFERENT episode set — the shape a changed
    stride/grid/episode-count would produce — which must fail loud.
    """
    g = _gate()
    g.observe(0, _series(0.80), _eid())
    other = [f"OTHER{e}" for e in range(N_EP) for _ in range(N_PER_EP)]
    with pytest.raises(WindowAlignmentError, match="DIFFERENT window set"):
        g.observe(2000, _series(0.60), other)


# =========================================================================== #
# 6. RESUME                                                                    #
# =========================================================================== #
def test_the_streak_survives_a_checkpoint_roundtrip():
    """A restart must not launder a decayed run past the gate.

    The falsifier is run first: a FRESH gate given only the post-restart probe
    does not stop. The restored gate does. If ``load_state_dict`` were a no-op
    this test would fail on the second assertion only — so both are asserted.
    """
    g = _gate()
    g.observe(0, _series(0.80), _eid())
    g.observe(2000, _series(0.60), _eid())
    assert g.worse_streak == 1 and not g.stop
    sd = g.state_dict()

    fresh = _gate()                                   # the pod restarted
    fresh.observe(4000, _series(0.40), _eid())
    assert not fresh.stop, ("control: a gate with no history cannot stop on its "
                            "first probe — which is exactly why the streak must "
                            "be restored")

    restored = _gate()
    restored.load_state_dict(sd)
    assert restored.worse_streak == 1
    r = restored.observe(4000, _series(0.40), _eid())
    assert r["separated_worse"] and restored.stop


# =========================================================================== #
# 7. FAIL LOUD                                                                 #
# =========================================================================== #
def test_an_unusable_probe_RAISES_rather_than_silently_disabling_the_gate():
    """A silently-disabled early-stop reproduces the exact defect this removes."""
    g = _gate()
    with pytest.raises(GateNotUsableError, match="FAILS LOUD"):
        g.observe(0, [0.5] * 6, ["only_one_episode"] * 6)

    g2 = _gate()
    with pytest.raises(GateNotUsableError):
        g2.observe(0, [float("nan")] * 12, [f"ep{i // 6}" for i in range(12)])


def test_mismatched_lengths_raise():
    g = _gate()
    with pytest.raises(ValueError, match="length mismatch"):
        g.observe(0, [0.1, 0.2, 0.3], ["a", "b"])


# =========================================================================== #
# 8. THE PROBE GRID                                                            #
# =========================================================================== #
def test_probe_grid_is_inside_the_measured_envelope():
    """0 % out of envelope, or the probe does not run — and the falsifier fires."""
    from taniteval import pseudosim as PS
    proof = PS.assert_grid_in_envelope(probe_grid())
    assert proof["EXTRAPOLATION_frac_steps_any"] == 0.0
    assert proof["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"] == 0.0

    # the guard is exercised with input designed to break it
    with pytest.raises(PS.EnvelopeViolation):
        PS.assert_grid_in_envelope(probe_grid(dyaw_deg=(0.0, 40.0)))


def test_probe_grid_keeps_a_perturbed_heading_or_recovery_is_all_NaN():
    """⭐ The cheap-grid trap, driven with the input that triggers it.

    ``recovery`` is undefined at the unperturbed point BY CONSTRUCTION. A probe
    grid of only ``dyaw=0`` therefore measures progress and comfort and is
    structurally blind to the error-recovery collapse the gate exists to catch —
    while still producing a perfectly plausible-looking composite. Here the
    degenerate grid is built and shown to be degenerate.
    """
    from taniteval import pseudosim as PS
    assert any(abs(y) > 0.0 for y in probe_grid().dyaw_deg), (
        "the probe grid has no perturbed heading — recovery would be all-NaN")

    flat = PS.GridSpec(dyaw_deg=(0.0,), dlon_steps=(0,))
    pw = _toy_pw(flat)
    rc = PS.score_windows(pw)["recovery"]
    assert not np.isfinite(rc).any(), (
        "a dyaw=0-only grid produced finite recovery values — the NaN-by-"
        "construction property this test relies on has changed")

    curved = _toy_pw(probe_grid())
    assert np.isfinite(PS.score_windows(curved)["recovery"]).any()


def _toy_pw(grid):
    """A minimal pseudo_evaluate-shaped record, no model involved."""
    pts = grid.points()
    n = len(pts) * 4
    dlat = torch.tensor([p[0] for p in pts]).repeat_interleave(4)
    dyaw = torch.tensor([p[1] for p in pts]).repeat_interleave(4)
    dlon = torch.tensor([float(p[2]) for p in pts]).repeat_interleave(4)
    t = torch.arange(1, 21).float()[None].expand(n, 20)
    traj = torch.stack([t * 0.5, torch.zeros_like(t)], dim=-1)
    ref = torch.stack([torch.arange(0, 21).float()[None].expand(n, 21) * 0.5,
                       torch.zeros(n, 21)], dim=-1)
    return {"traj": traj, "ref_path": ref, "ref_yaw": torch.zeros(n),
            "pt_dlat": dlat, "pt_dyaw": dyaw, "pt_dlon": dlon,
            "v0": torch.full((n,), 5.0),
            "eid": [f"ep{i % 4}" for i in range(n)]}


# =========================================================================== #
# 9. THE DEPLOYABLE SURFACE                                                    #
# =========================================================================== #
class _FakeWorld(torch.nn.Module):
    """Encodes each window to a state that VARIES with its pixels, so the probe's
    per-window scores have real spread (a constant planner is refused by
    ``discriminative_range``, correctly)."""

    def encode_window(self, frames):
        b = frames.shape[0]
        m = frames.reshape(b, -1).mean(-1)
        return m[:, None, None] * torch.ones(b, 4, 8)


class _FakeHead(torch.nn.Module):
    """Emits a fan whose ORACLE is excellent and whose SELECTED pick is awful."""

    def __init__(self, horizons=tuple(range(1, 21))):
        super().__init__()
        self.cfg = type("C", (), {"horizons": horizons})()
        self.seen_train_mode = None

    def forward(self, states, v0, **kw):
        b, s = states.shape[0], len(self.cfg.horizons)
        self.seen_train_mode = self.training
        good = torch.ones(b, s, 2)                      # the oracle candidate
        bad = torch.full((b, s, 2), -9.0)               # what select() picked
        return {"wp_seq": bad, "traj": bad, "fan": torch.stack([good, bad], 1),
                "oracle": good}


class _VaryingHead(torch.nn.Module):
    """A plausible planner: drive forward at ``v0`` with a per-window curvature.

    Its plans differ window to window, which is what a real arm does and what the
    composite's discriminative-range gate requires before it will emit at all.
    """

    def __init__(self, horizons=tuple(range(1, 21))):
        super().__init__()
        self.cfg = type("C", (), {"horizons": horizons})()

    def forward(self, states, v0, **kw):
        s = len(self.cfg.horizons)
        t = torch.arange(1, s + 1, dtype=torch.float32) * 0.1        # [S]
        c = (states[:, -1, 0] - 0.5) * 2.0                            # [B]
        x = v0[:, None].float() * t[None]
        y = c[:, None] * t[None] ** 2
        wp = torch.stack([x, y], dim=-1)
        return {"wp_seq": wp, "traj": wp}


def test_planner_reads_the_SELECTED_trajectory_not_the_oracle():
    """v4's regression was in SELECTION. A probe that read the oracle would have
    reported the run healthy while it decayed — so this pins the source key."""
    p = DeployableSurfacePlanner(_FakeWorld(), _FakeHead(), device="cpu")
    out = p.traj(torch.zeros(3, 4, 1, 8, 8), torch.full((3,), 5.0))
    assert out.shape == (3, 20, 2)
    assert torch.allclose(out, torch.full((3, 20, 2), -9.0)), (
        "the planner returned something other than the SELECTED wp_seq")
    assert p.provenance["selected_not_oracle"] is True


def test_planner_evaluates_in_EVAL_mode_and_restores_training_mode():
    head, world = _FakeHead(), _FakeWorld()
    head.train(); world.train()
    p = DeployableSurfacePlanner(world, head, device="cpu")
    p.traj(torch.zeros(2, 4, 1, 8, 8), torch.full((2,), 5.0))
    assert head.seen_train_mode is False, "probe ran with dropout/BN in train mode"
    assert head.training is True, "the planner did not restore training mode"


def test_planner_REFUSES_a_non_dense_head():
    """A coarse 5 s plan scored as if dense divides every derivative by the wrong
    dt. Driven with the tactical instance's own horizon tuple."""
    with pytest.raises(NonDensePlanError, match="DENSE"):
        DeployableSurfacePlanner(_FakeWorld(), _FakeHead(tuple(range(5, 51, 5))),
                                 device="cpu")


# =========================================================================== #
# cadence + end-to-end                                                         #
# =========================================================================== #
def test_due_is_a_fixed_cadence_never_data_dependent():
    g = _gate(every=2000)
    assert g.due(0) and g.due(2000) and g.due(4000)
    assert not g.due(1999) and not g.due(2001)
    assert not _gate(enabled=False).due(2000)
    assert not _gate(every=0).due(2000)


class _Ep:
    """A pseudo_evaluate-shaped held-out episode: .poses + .frames."""

    def __init__(self, T=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.frames = torch.rand(T, 1, 32, 32, generator=g)
        x = torch.arange(T).float() * 0.5
        self.poses = torch.stack([x, torch.zeros(T), torch.zeros(T),
                                  torch.full((T,), 5.0)], dim=-1)


def test_probe_end_to_end_runs_pseudosim_on_the_deployable_surface():
    """The integration proof: a real pseudo_evaluate -> composite -> observe pass
    on toy episodes, with the envelope proof and the estimator riding along."""
    g = _gate(episodes=4, stride=8, horizon=20, batch=8)
    eps = [_Ep(seed=i) for i in range(4)]
    rec = g.probe(0, _FakeWorld(), _VaryingHead(), eps, device="cpu")
    assert rec["primary"] == PRIMARY_NAME
    assert rec["n_episodes"] >= 2
    ps = rec["pseudosim"]
    assert ps["rollout_steps_executed"] == 0        # no accumulation, by construction
    assert ps["envelope_proof"] is not None
    assert ps["surface"]["selected_not_oracle"] is True
    assert ps["components_admitted"], "no component was admitted into the composite"


def test_the_admitted_component_set_is_PINNED_at_the_first_probe():
    """Re-deriving admissibility per probe would let the composite change
    definition mid-run and compare two different metrics.

    Driven by mutating the pinned ranges after probe 0 and showing probe 1 uses
    the PINNED set rather than re-deriving one from its own data.
    """
    g = _gate(episodes=4, stride=8, horizon=20, batch=8)
    eps = [_Ep(seed=i) for i in range(4)]
    g.probe(0, _FakeWorld(), _VaryingHead(), eps, device="cpu")
    pinned = dict(g._pinned_admitted)
    assert pinned

    # a poisoned range table: if the gate re-derived admissibility it would
    # ignore this and admit its own set; pinned means it must honour it.
    g._pinned_ranges = {k: {"admissible": (k == "comfort")}
                        for k in ("ego_progress", "recovery", "comfort",
                                  "no_collision", "ttc")}
    val, eid, _ = g._composite_of(_toy_pw(probe_grid()))
    assert np.isfinite(np.asarray(val, float)).any()
    assert g._pinned_admitted == pinned, "the pinned admitted set was overwritten"
