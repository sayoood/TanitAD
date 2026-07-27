"""THE ``vt_band`` DECISION, pinned — what each candidate goal state MEANS.

⛔ **No default changes and nothing is wired in.** These tests price the options
in ``tanitad.train.heldout_goal`` so the PI's choice rests on facts that a test
run re-proves, not on prose. ``test_heldout_gate_real_head.py`` remains the
tripwire for the defect itself; this file is the decision's evidence.

The load-bearing facts, each with its own test:

1. index 0 of BOTH categorical goal channels is a real class — ``v_stop`` and
   ``ROUTE_LEFT`` — and ``route=LEFT`` beside ``route_graded=0.0`` is
   self-inconsistent against the label's own definition;
2. ``VT_DROPPED`` / ``ROUTE_DROPPED`` are IN-DISTRIBUTION because
   ``goal_dropout = 0.5`` ships and the trainer never overrides it;
3. the two options differ in the SELECTOR (``vt_keep``), not only the embedding;
4. today's ``goal_kwargs_fn(b, device)`` signature cannot express any option
   faithfully, and the "obvious" all-zeros patch brakes the probe;
5. every option except ``crash_today`` lets the head plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.train import heldout_goal as HG                       # noqa: E402
from tanitad.train.heldout_goal import (StatesAwareSurfacePlanner,  # noqa: E402
                                        make_goal_kwargs_fn)

import train_flagship_v4 as T                                      # noqa: E402


def _real_stack():
    """The REAL WorldModel + FlagshipV4Head at the SHIPPED goal conditioning.

    Sizes from ``_smoke_head_cfg`` (CPU, milliseconds); ``cond_vtarget`` /
    ``cond_route`` / ``goal_dropout`` put back to what ``v4_config()`` ships,
    which is what a real v5 run builds."""
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
    hcfg.cond_vtarget = shipped.cond_vtarget
    hcfg.cond_route = shipped.cond_route
    hcfg.goal_dropout = shipped.goal_dropout
    return world, T.FlagshipV4Head(hcfg), cfg


# --------------------------------------------------------------------------- #
# 1. index 0 is not a null                                                     #
# --------------------------------------------------------------------------- #
def test_vt_band_zero_is_v_stop_not_a_null():
    """⛔ THE FACT THE WHOLE DECISION TURNS ON.

    ``_goal_inputs`` falls back to ``vt_band = zeros``. Index 0 of the VTARGET
    vocabulary is ``v_stop`` — the STOPPED target-speed band. Probing at band 0
    tells the planner it is coming to a stop on every held-out window."""
    from tanitad.lake.vocab import VTARGET_TOKENS
    assert VTARGET_TOKENS[0] == "v_stop", VTARGET_TOKENS[:3]
    assert HG.describe_index_zero()["vt_band_0"] == "v_stop"
    # and it is genuinely distinct from the withheld row
    assert HG.VT_DROPPED == HG.N_VTARGET_BANDS == 23
    assert HG.VT_DROPPED != 0


def test_route_zero_is_ROUTE_LEFT_and_contradicts_route_graded_zero():
    """⚠️ The zeros fallback is wrong on the ROUTE channel too, and inconsistently.

    ``refb_labels``: ``ROUTE_LEFT, ROUTE_STRAIGHT, ROUTE_RIGHT = range(3)``, so
    ``route = 0`` is LEFT. It is passed beside ``route_graded = 0.0``, and the
    label defines ``route_graded = tanh(mean_curv / CURV_TURN_PER_M)`` with a
    genuine LEFT window carrying ``|graded| >= tanh(1)``. The pair (LEFT, 0.0) is
    therefore a combination the training distribution essentially never contains."""
    import refb_labels as rl
    assert (rl.ROUTE_LEFT, rl.ROUTE_STRAIGHT, rl.ROUTE_RIGHT) == (0, 1, 2)
    assert rl.ROUTE_UNKNOWN == 3 and HG.ROUTE_DROPPED == 4
    thr = float(torch.tanh(torch.tensor(1.0)))
    assert thr > 0.76
    # a real LEFT window's graded magnitude is >= tanh(1); the fallback gives 0.0
    assert abs(0.0) < thr


# --------------------------------------------------------------------------- #
# 2. is VT_DROPPED in-distribution? — established IN CODE                      #
# --------------------------------------------------------------------------- #
def test_VT_DROPPED_is_IN_DISTRIBUTION_because_goal_dropout_ships_at_half():
    """⭐ The ``v2_ego_dropout``-precedent question, answered from the source.

    ``V15Config.goal_dropout = 0.5`` is inherited by ``V4Config`` and the trainer
    NEVER overrides it, so a real v5 run masks ~50 % of every batch to
    ``VT_DROPPED``/``ROUTE_DROPPED``. Those are their own learned embedding rows,
    not a zero-fill — a STRONGER precedent than ``v2_ego_dropout``'s."""
    from tanitad.models.flagship_v4 import v4_config
    from tanitad.models.flagship_v15 import V15Config

    assert V15Config().goal_dropout == 0.5
    assert v4_config().goal_dropout == 0.5, \
        "v4_config no longer ships goal_dropout=0.5 — the 'DROPPED rows are " \
        "trained' argument is void and VTBAND_DECISION.md must be re-read"

    # ⚠️ AST, not a substring scan. The original substring form tripped on
    # ``--heldout-goal``'s own help text, which *documents* goal_dropout = 0.5
    # rather than setting it — a false positive that would have pressured the
    # next reader to delete the check. What matters is ASSIGNMENT: any
    # ``x.goal_dropout = …`` or ``Cfg(goal_dropout=…)`` anywhere in the trainer.
    import ast
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    writes = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tgts:
                name = (t.attr if isinstance(t, ast.Attribute)
                        else t.id if isinstance(t, ast.Name) else None)
                if name == "goal_dropout":
                    writes.append(f"assign@{n.lineno}")
        if isinstance(n, ast.Call):
            writes += [f"kwarg@{n.lineno}" for k in n.keywords
                       if k.arg == "goal_dropout"]
    assert not writes, \
        f"the trainer now WRITES goal_dropout ({writes}) — re-derive whether " \
        f"the DROPPED rows are still trained at 0.5 before trusting " \
        f"--heldout-goal dropped"


def test_the_dropped_rows_are_real_embedding_rows_distinct_from_UNKNOWN():
    """The DROPPED index addresses a row that exists, and is not the labeler's
    UNKNOWN sentinel — the source says the two states are different on purpose."""
    _, head, _ = _real_stack()
    assert head.vtarget_emb.num_embeddings == HG.N_VTARGET_BANDS + 1
    assert head.route_emb.num_embeddings == HG.N_ROUTE_CLASSES + 1
    import refb_labels as rl
    assert HG.ROUTE_DROPPED != rl.ROUTE_UNKNOWN


def test_make_goal_kwargs_fn_REFUSES_dropped_when_the_rows_were_never_trained():
    """⭐ The guard that stops this option from being cargo-culted onto a config
    that turned the dropout off — where the rows are N(0, 0.02) init noise."""
    _, head, _ = _real_stack()
    head.cfg.goal_dropout = 0.0
    with pytest.raises(ValueError, match="goal_dropout > 0"):
        make_goal_kwargs_fn("dropped", head.cfg)


# --------------------------------------------------------------------------- #
# 3. the options differ in the SELECTOR, not only the embedding                #
# --------------------------------------------------------------------------- #
def test_band0_keeps_the_longitudinal_selector_ON_and_dropped_masks_it_OFF():
    """⛔ The difference is not cosmetic and not confined to the condition vector.

    ``condition()`` returns ``vt_keep = (band != VT_DROPPED)`` and ``select()``
    multiplies the longitudinal penalty by it. So band-0 probes a planner whose
    longitudinal ranking term is live; VT_DROPPED probes one whose term is masked."""
    _, head, _ = _real_stack()
    head.eval()
    v0 = torch.full((4,), 9.0)

    _, _, keep_b0 = head.condition(v0, torch.zeros(4, dtype=torch.long),
                                   torch.zeros(4, dtype=torch.long),
                                   torch.zeros(4))
    _, _, keep_dp = head.condition(v0,
                                   torch.full((4,), HG.VT_DROPPED, dtype=torch.long),
                                   torch.full((4,), HG.ROUTE_DROPPED, dtype=torch.long),
                                   torch.zeros(4))
    assert keep_b0 is not None and bool(keep_b0.all()), "band 0 must keep the term"
    assert keep_dp is not None and not bool(keep_dp.any()), \
        "VT_DROPPED must mask the longitudinal term off"


def test_vt_speed_zero_makes_the_selector_chase_a_5mps_DECELERATION():
    """⛔ THE TRAP. 'Build the zeros the provenance promises' also zeroes
    ``vt_speed``, and ``select()`` clamps it into the reachable band around v0:

        v_goal = max(min(vt_speed, v0+reach), (v0-reach).clamp_min(0)),
        reach  = sel_accel_max * horizons[-1] * 0.1 = 5.0 m/s

    With ``vt_speed = 0`` that is ``(v0 - 5)+`` on every window — the selector is
    told to brake as hard as it reachably can. A braking plan then has
    ``s_along -> 0``, and with the gate's ``dlat = 0`` grid
    ``xt_hold = s_along*|tan(dpsi)| -> 0``, so ``recovery`` is NaN BY
    CONSTRUCTION and the composite goes UP. The patch that looks conservative
    produces a gate that reads healthier while probing a braking planner.

    ⚠️ ``reach`` is read off the REAL ``v4_config()`` (dense horizons 1..20), not
    off ``_smoke_head_cfg`` — the smoke head has 4 horizons and would give
    ``reach = 1.0``, understating the trap fivefold."""
    from tanitad.models.flagship_v4 import v4_config
    real = v4_config()
    reach = real.sel_accel_max * real.horizons[-1] * 0.1
    assert real.horizons[-1] == 20
    assert reach == pytest.approx(5.0)
    v0 = torch.tensor([12.0, 8.0, 3.0])

    def v_goal(vt_speed):
        return torch.max(torch.min(vt_speed, v0 + reach),
                         (v0 - reach).clamp_min(0.0))

    assert torch.allclose(v_goal(v0), v0)                     # faithful: hold-v0
    naive = v_goal(torch.zeros(3))
    assert torch.allclose(naive, torch.tensor([7.0, 3.0, 0.0]))
    assert bool((naive < v0).all()), "the naive patch decelerates every window"


def test_the_gates_probe_grid_has_dlat_zero_so_a_slow_plan_NaNs_recovery():
    """The mechanism behind the trap above, pinned on the SHIPPED grid.

    ``probe_grid()`` is dyaw-only (``dlat_m = (0.0,)``, the lateral axis is
    refused by default), so ``xt_hold = |dlat + s_along*tan(dpsi)|`` collapses to
    ``s_along*|tan(dpsi)|`` and vanishes with along-track travel.

    ⚠️ NOT ``importorskip``: this assertion is load-bearing for the decision, so a
    box without ``taniteval`` on the path must FAIL rather than quietly pass. The
    repo layout puts it at ``<repo>/taniteval``, which is exactly what
    ``heldout_gate._taniteval()`` inserts."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))
    from tanitad.train.heldout_gate import probe_grid
    g = probe_grid()
    assert tuple(g.dlat_m) == (0.0,)
    assert tuple(g.dyaw_deg) == (-8.0, 0.0, 8.0)
    assert tuple(g.dlon_steps) == (0,)


# --------------------------------------------------------------------------- #
# 4. the signature gap                                                         #
# --------------------------------------------------------------------------- #
def test_the_shipped_signature_now_CARRIES_v0_and_states():
    """⭐ INVERTED 2026-07-27 — this pinned the signature gap; it now pins the fix.

    The old shipped call was ``self.goal_kwargs_fn(b, self.device)``: ``v0`` was
    in scope at that line and not passed, and ``states`` did not exist yet, so
    ``band0`` could not be expressed faithfully (``vt_speed = v0``) and
    ``produced`` could not be expressed at all. Both halves are now inverted —
    the protocol AND the ordering, because either alone would leave ``produced``
    unreachable."""
    import tanitad.train.heldout_gate as HGate
    src = Path(HGate.__file__).read_text(encoding="utf-8")
    i = src.index("def traj(self, frames, v0, goal=None):")
    body = src[i:i + 1400]
    assert "self.goal_kwargs_fn(b, self.device)" not in body, body
    assert "self.goal_kwargs_fn(states, v0d)" in body, body
    # states is now computed BEFORE kw is built -> `produced` is reachable
    assert body.index("encode_window") < body.index("kw = ("), body


def test_every_option_filters_to_the_heads_own_cond_switches():
    """No option can inject a channel the head is not conditioned on."""
    _, head, _ = _real_stack()
    v0 = torch.full((3,), 7.0)
    for opt in ("zeros_naive", "band0", "dropped"):
        kw = make_goal_kwargs_fn(opt, head.cfg)(None, v0)
        assert set(kw) == {"vt_band", "vt_speed", "route", "route_graded"}
    head.cfg.cond_route = False
    kw = make_goal_kwargs_fn("band0", head.cfg)(None, v0)
    assert set(kw) == {"vt_band", "vt_speed"}


def test_oracle_is_refused_with_its_reason_not_silently_unimplemented():
    av = HG.oracle_availability()
    assert av["mechanically_reachable"] is True
    assert "REFUSED" in av["verdict"]
    assert "perturbed" in av["blocker_3_FATAL_undefined_at_the_probe_points"]
    with pytest.raises(NotImplementedError, match="refused, not unimplemented"):
        make_goal_kwargs_fn("oracle", None)


# --------------------------------------------------------------------------- #
# 5. does the head plan under each option?                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("option", ["zeros_naive", "band0", "dropped"])
def test_the_real_head_PLANS_under_each_candidate(option):
    """⭐ PRIORITY 1: every candidate lets the gate run; the shipped default does not."""
    world, head, cfg = _real_stack()
    fn = make_goal_kwargs_fn(option, head.cfg)
    planner = StatesAwareSurfacePlanner(world, head, device="cpu",
                                        goal_kwargs_fn=fn, option=option)
    b, W = 3, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    traj = planner.traj(frames, torch.full((b,), 8.0))
    assert traj.shape == (b, len(head.cfg.horizons), 2)
    assert torch.isfinite(traj).all()
    assert planner.provenance["vtband_option"] == option


@pytest.mark.parametrize("option", ["band0_vt_only", "band0_route_only"])
def test_the_channel_isolation_diagnostics_move_exactly_ONE_channel(option):
    """The two diagnostics that split band0's penalty must each corrupt one
    categorical and leave the other at its learned DROPPED row."""
    _, head, _ = _real_stack()
    v0 = torch.full((3,), 7.0)
    kw = make_goal_kwargs_fn(option, head.cfg)(None, v0)
    if option == "band0_vt_only":
        assert int(kw["vt_band"][0]) == 0
        assert int(kw["route"][0]) == HG.ROUTE_DROPPED
    else:
        assert int(kw["vt_band"][0]) == HG.VT_DROPPED
        assert int(kw["route"][0]) == 0
    # both keep vt_speed faithful to _goal_inputs, so neither reintroduces the
    # zeros_naive braking confound into the isolation
    assert torch.equal(kw["vt_speed"], v0)


def test_the_diagnostics_are_not_advertised_as_candidates():
    """A diagnostic that reads as a candidate is how a decision table gets an
    extra column nobody meant to offer."""
    assert set(HG.CANDIDATES) == {"band0", "dropped", "produced"}
    for d in ("band0_vt_only", "band0_route_only"):
        assert d not in HG.CANDIDATES
        assert HG.OPTION_MEANING[d].startswith("DIAGNOSTIC")


def test_crash_today_is_the_RED_baseline():
    """The shipped default, through the same harness — it must still refuse."""
    world, head, cfg = _real_stack()
    assert make_goal_kwargs_fn("crash_today", head.cfg) is None
    planner = StatesAwareSurfacePlanner(world, head, device="cpu",
                                        goal_kwargs_fn=None, option="crash_today")
    b, W = 2, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    with pytest.raises(ValueError, match="cond_vtarget is on but no vt_band"):
        planner.traj(frames, torch.full((b,), 8.0))


def test_the_harness_is_equivalent_to_the_shipped_planner_when_the_goal_matches():
    """⭐ The control that makes every number in VTBAND_DECISION.md attributable.

    ``StatesAwareSurfacePlanner`` must differ from ``DeployableSurfacePlanner``
    ONLY in how ``kw`` is built. Fed the same kwargs, the two must emit the same
    trajectory bit-for-bit — otherwise an "option effect" could be a harness
    effect.

    ⭐ Since 2026-07-27 the shipped planner takes the SAME ``(states, v0)``
    protocol, so this equivalence is now structural rather than coincidental —
    the shim delegates instead of copying ``traj``'s body. The test is kept
    because the numbers in ``VTBAND_DECISION.md`` were taken through the shim,
    and it is also what would catch a future re-divergence."""
    from tanitad.train.heldout_gate import DeployableSurfacePlanner
    world, head, cfg = _real_stack()
    b, W = 3, cfg.predictor.window
    frames = torch.rand(b, W, cfg.encoder.in_channels, 64, 64)
    v0 = torch.full((b,), 8.0)

    shipped = DeployableSurfacePlanner(
        world, head, device="cpu",
        goal_kwargs_fn=lambda states, v0_: HG.band0_kwargs(head.cfg, v0_))
    mine = StatesAwareSurfacePlanner(
        world, head, device="cpu", option="band0",
        goal_kwargs_fn=make_goal_kwargs_fn("band0", head.cfg))
    assert torch.equal(shipped.traj(frames, v0), mine.traj(frames, v0))


def test_the_planner_restores_train_mode_like_the_shipped_one():
    """Regression guard: a probe that left the model in eval() would silently
    disable dropout for the rest of training."""
    world, head, cfg = _real_stack()
    world.train(); head.train()
    planner = StatesAwareSurfacePlanner(
        world, head, device="cpu", option="band0",
        goal_kwargs_fn=make_goal_kwargs_fn("band0", head.cfg))
    b, W = 2, cfg.predictor.window
    planner.traj(torch.rand(b, W, cfg.encoder.in_channels, 64, 64),
                 torch.full((b,), 8.0))
    assert head.training and world.training
