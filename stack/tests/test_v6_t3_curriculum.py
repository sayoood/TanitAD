"""F-9 / catalog T3 — THE INTERACTION CURRICULUM: both directions.

THE SPEC, quoted (two independent locations, established BEFORE a line of the
implementation was written):

  * ``…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:66`` —
    *"T3 | interaction curriculum: windows ranked by MULTI-AGENT kinematic
    entropy measured from the O-layer's own predicted occupancy
    (self-supervised, after O2/O3 make it non-degenerate) | curriculum from
    free-flow -> dense interaction | P7 calibration rho >=0.3 held on
    interaction-rich strata, not just pooled"*
  * ``…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:59`` — *"O4 is
    the ego-kinematic version only; T3's multi-agent extension needs the P8
    occupancy readout in the loop. Fix F-9 (gated on P8 maturity)"*, and
    ``:214`` — *"F-9 | P3 | T3 interaction curriculum (multi-agent entropy from
    the P8 occupancy readout) — gated on P8 maturity."*

⛔ THE CENTRAL FACT THIS FILE PINS is that the OBVIOUS implementation of
"entropy of the occupancy field" is **MAXIMAL ON AN EMPTY ROAD** — it is not
merely imprecise, it is INVERTED, and a curriculum built on it would drive
training towards empty scenes while its name said the opposite. The failure is
demonstrated here on the bare functional and then shown to be absent from the
shipped one. Same discipline as F-8's flat-plan control: know your functional's
degenerate input before you rank a corpus with it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.v6 import (  # noqa: E402
    T3_CONTROL_MIN_N, T3_MASS_SCALE, T3Curriculum, V6Config, V6Stack,
    multi_agent_kinematic_entropy, saliency_weights, t3_rank_control)


# ---------------------------------------------------------------------------
# synthetic occupancy rollouts — the three regimes the score must separate
# ---------------------------------------------------------------------------

def _empty(k: int = 4, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    """Open road: a near-zero field with sensor noise. THE degenerate input."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, k, h, w, generator=g) * 1e-4


def _moving(cells, k: int = 4, h: int = 16, w: int = 16) -> torch.Tensor:
    """`cells` agents, each stepping one cell per tick."""
    o = torch.zeros(1, k, h, w)
    for (r, c) in cells:
        for j in range(k):
            o[0, j, (r + j) % h, (c + j) % w] = 1.0
    return o


def _parked(cells, k: int = 4, h: int = 16, w: int = 16) -> torch.Tensor:
    """Agents present but STATIONARY — occupancy without kinematics."""
    o = torch.zeros(1, k, h, w)
    for (r, c) in cells:
        o[0, :, r, c] = 1.0
    return o


def _bare_spatial_entropy(occ: torch.Tensor) -> float:
    """⛔ THE DEGENERATE READING, implemented here ONLY so its failure can be
    measured. Shannon entropy of the normalised occupancy raster."""
    p = occ[:, -1].reshape(occ.shape[0], -1)
    p = p / p.sum(dim=-1, keepdim=True)
    return float(-(p * (p + 1e-12).log()).sum(dim=-1) / math.log(p.shape[-1]))


# ===========================================================================
# 1. ⛔ THE DEGENERACY — measured on the bare functional, absent from ours
# ===========================================================================

def test_bare_spatial_entropy_is_maximal_on_an_empty_raster():
    """The failure this cell exists to avoid, MEASURED rather than asserted.

    Normalising a near-zero field divides noise by noise: the result is
    near-uniform, and uniform is the maximum of Shannon entropy. So the naive
    reading ranks an EMPTY road ABOVE a dense one.
    """
    empty, dense = _empty(), _moving([(2, 2), (2, 10), (10, 3), (11, 11)])
    h_empty, h_dense = _bare_spatial_entropy(empty), _bare_spatial_entropy(dense)
    assert h_empty > 0.9, h_empty          # ~0.965 MEASURED
    assert h_dense < 0.5, h_dense          # ~0.250 MEASURED
    # the inversion, stated as a ratio so the magnitude is on the record
    assert h_empty / h_dense > 3.0


def test_the_shipped_score_is_not_inverted():
    """The same three rasters, through the shipped functional: correctly
    ordered, and the empty road is at the BOTTOM."""
    empty = float(multi_agent_kinematic_entropy(_empty()))
    one = float(multi_agent_kinematic_entropy(_moving([(3, 4)])))
    dense = float(multi_agent_kinematic_entropy(
        _moving([(2, 2), (2, 10), (10, 3), (11, 11)])))
    assert empty < one < dense
    assert empty < 0.05          # the mass gate crushes noise-only mass
    assert dense > 2 * one       # several movers beat one mover


def test_an_exactly_empty_raster_scores_exactly_zero():
    """The mass gate is ``1 - exp(-M/scale)``, which is EXACTLY 0 at M=0 — not
    small, zero. This is the guarantee the degenerate reading cannot make."""
    assert float(multi_agent_kinematic_entropy(torch.zeros(1, 4, 8, 8))) == 0.0


def test_parked_agents_score_below_moving_ones():
    """*KINEMATIC* entropy: presence is not interaction. Identical agents, one
    set stationary, and the stationary set scores exactly zero because the
    per-cell CHANGE field is empty."""
    cells = [(2, 2), (2, 10), (10, 3), (11, 11)]
    assert float(multi_agent_kinematic_entropy(_parked(cells))) == 0.0
    assert float(multi_agent_kinematic_entropy(_moving(cells))) > 0.4


def test_more_agents_score_higher_at_equal_motion():
    """The 'multi-agent' half: same per-agent motion, more agents, higher
    score. Monotone over 1/2/4/8 movers."""
    prev = -1.0
    for n in (1, 2, 4, 8):
        cells = [(2 * i % 15, 3 * i % 15) for i in range(n)]
        s = float(multi_agent_kinematic_entropy(_moving(cells)))
        assert s > prev, (n, s, prev)
        prev = s


# ===========================================================================
# 2. the functional's guards — and their opposites (C95/C97)
# ===========================================================================

def test_the_score_refuses_a_single_tick():
    with pytest.raises(ValueError, match="K >= 2"):
        multi_agent_kinematic_entropy(torch.rand(1, 1, 8, 8))


def test_the_score_refuses_a_snapshot_without_a_tick_axis():
    with pytest.raises(ValueError, match=r"\[B, K, H, W\]"):
        multi_agent_kinematic_entropy(torch.rand(1, 8, 8))


def test_the_score_refuses_logits():
    """P8 emits LOGITS. Entropy of a logit field is not a quantity anyone
    specified, and it would silently produce a ranking."""
    with pytest.raises(ValueError, match="PROBABILITIES"):
        multi_agent_kinematic_entropy(torch.randn(1, 4, 8, 8) * 5)


def test_a_legal_call_raises_NOTHING():
    """⭐ C95/C97: this programme shipped a rejects-everything guard and a
    passes-everything guard within one day. Every refusal above is paired."""
    out = multi_agent_kinematic_entropy(_moving([(1, 1), (5, 5)]))
    assert out.shape == (1,)
    assert 0.0 <= float(out) <= 1.0


def test_the_score_is_bounded_in_the_unit_interval_on_random_input():
    g = torch.Generator().manual_seed(7)
    out = multi_agent_kinematic_entropy(torch.rand(16, 5, 12, 12, generator=g))
    assert out.shape == (16,)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_the_mass_scale_is_a_declared_constant_not_a_hidden_default():
    assert T3_MASS_SCALE == 4.0
    hot = _moving([(1, 1), (5, 5), (9, 9)])
    assert (float(multi_agent_kinematic_entropy(hot, mass_scale=0.1))
            > float(multi_agent_kinematic_entropy(hot, mass_scale=100.0)))
    with pytest.raises(ValueError, match="mass_scale"):
        multi_agent_kinematic_entropy(hot, mass_scale=0.0)


# ===========================================================================
# 3. the CURRICULUM — direction, ramp, and the parity property
# ===========================================================================

def test_the_default_curriculum_runs_free_flow_to_dense():
    """The catalog's direction, as a property of the DRAW, not of a number.

    At progress 0 the free-flow window must be more likely than the dense one;
    at progress 1, the reverse. This is what 'free-flow -> dense' means.
    """
    c = T3Curriculum()
    scores = torch.tensor([0.0, 1.0])         # [free-flow, dense]
    w0, w1 = c.weights_at(scores, 0.0), c.weights_at(scores, 1.0)
    assert float(w0[0]) > float(w0[1]), w0    # starts on free flow
    assert float(w1[1]) > float(w1[0]), w1    # ends on dense
    assert pytest.approx(1.0, abs=1e-6) == float(w0.sum())
    assert pytest.approx(1.0, abs=1e-6) == float(w1.sum())


def test_alpha_ramps_linearly_then_holds():
    c = T3Curriculum(alpha_start=-1.0, alpha_end=1.0, warmup_frac=0.5)
    assert c.alpha_at(0.0) == pytest.approx(-1.0)
    assert c.alpha_at(0.25) == pytest.approx(0.0)
    assert c.alpha_at(0.5) == pytest.approx(1.0)
    assert c.alpha_at(0.75) == pytest.approx(1.0)   # HELD, not overshooting
    assert c.alpha_at(1.0) == pytest.approx(1.0)


def test_every_window_stays_reachable_at_every_alpha():
    """⛔ THE PARITY PROPERTY. O4's ``floor>0`` note applies verbatim: the
    curriculum REWEIGHTS the draw and never removes a window, so every arm
    still sees all 2376 episodes. A weight of exactly 0 anywhere would be a
    silent re-selection of the corpus."""
    c = T3Curriculum()
    scores = torch.tensor([0.0, 0.001, 0.5, 1.0, 5.0])
    for prog in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0):
        w = c.weights_at(scores, prog)
        assert float(w.min()) > 0.0, (prog, w)
        assert torch.isfinite(w).all()


def test_a_static_arm_is_expressible_as_the_no_curriculum_control():
    """``alpha_start == alpha_end`` is O4's shape with T3's score — the
    attributability control for 'did the SCHEDULE do anything?'."""
    c = T3Curriculum(alpha_start=1.0, alpha_end=1.0)
    s = torch.tensor([0.0, 0.4, 1.0])
    assert torch.allclose(c.weights_at(s, 0.0), c.weights_at(s, 1.0))


def test_the_curriculum_refuses_a_reversed_direction():
    with pytest.raises(ValueError, match="REVERSED"):
        T3Curriculum(alpha_start=1.0, alpha_end=-1.0)


def test_the_curriculum_refuses_a_zero_floor():
    """With alpha<0 a zero floor makes a zero-score window infinitely likely —
    and it would also break reachability."""
    with pytest.raises(ValueError, match="floor"):
        T3Curriculum(floor=0.0)


def test_the_curriculum_refuses_a_degenerate_warmup():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="warmup_frac"):
            T3Curriculum(warmup_frac=bad)


def test_progress_outside_the_unit_interval_is_refused():
    c = T3Curriculum()
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="progress"):
            c.alpha_at(bad)


def test_a_legal_curriculum_construction_raises_NOTHING():
    """C95/C97 pairing for the constructor guards above."""
    c = T3Curriculum(alpha_start=-0.5, alpha_end=2.0, warmup_frac=1.0,
                     floor=0.1)
    assert c.alpha_at(0.5) == pytest.approx(0.75)


def test_o4s_saliency_weights_guard_was_NOT_weakened():
    """⭐ T3 needs a NEGATIVE exponent; O4's shared helper refuses one. The
    resolution was T3's own weighting, NOT an edit to the shared module.
    F-7's lesson, pinned so a later change cannot quietly take the other
    route."""
    with pytest.raises(ValueError, match="alpha"):
        saliency_weights(torch.tensor([0.0, 1.0]), alpha=-1.0)


# ===========================================================================
# 4. the CONTROL — and its minimum-n rule
# ===========================================================================

def test_the_control_detects_a_working_ranking():
    g = torch.Generator().manual_seed(3)
    n = 64
    dense = torch.zeros(2 * n, dtype=torch.bool)
    dense[n:] = True
    scores = torch.cat([torch.rand(n, generator=g) * 0.2,
                        0.6 + torch.rand(n, generator=g) * 0.4])
    out = t3_rank_control(scores, dense)
    assert out["verdict"] == "OK"
    assert out["ratio"] > 1.0
    assert out["sem_dense"] > 0 and out["sem_free"] > 0


def test_the_control_reports_INVERTED_on_the_degenerate_functional():
    """⭐ THE POINT OF THE CONTROL. Score the same corpus with the BARE spatial
    entropy — the empty-road-is-maximal reading — and the control must call it
    INVERTED rather than letting the run proceed."""
    n = 64
    dense = torch.zeros(2 * n, dtype=torch.bool)
    dense[n:] = True
    scores = torch.tensor(
        [_bare_spatial_entropy(_empty(seed=i)) for i in range(n)]
        + [_bare_spatial_entropy(_moving([(2, 2), (2, 10), (10, 3)]))] * n)
    out = t3_rank_control(scores, dense)
    assert out["verdict"] == "INVERTED"
    assert out["ratio"] < 1.0


def test_the_control_reports_DEGENERATE_when_every_score_is_zero():
    n = 40
    dense = torch.zeros(2 * n, dtype=torch.bool)
    dense[n:] = True
    out = t3_rank_control(torch.zeros(2 * n), dense)
    assert out["verdict"] == "DEGENERATE_ALL_ZERO"


def test_the_control_refuses_a_verdict_below_its_sample_floor():
    """⛔ MEASURED for the sibling T2 control at the NULL (true ratio 1 by
    construction): at n=4 the ratio spanned 0.397-3.361. A verdict from a
    handful of windows is noise wearing a number's clothes."""
    assert T3_CONTROL_MIN_N == 32
    n = 4
    dense = torch.zeros(2 * n, dtype=torch.bool)
    dense[n:] = True
    out = t3_rank_control(torch.rand(2 * n), dense)
    assert out["verdict"] == "REFUSED_TOO_FEW"
    assert "ratio" not in out            # no number to quote out of context
    assert out["n_dense"] == n and out["n_free"] == n


def test_the_control_refuses_a_one_sided_split_even_when_total_n_is_large():
    """n is PER SIDE. 1000 free-flow windows and 4 dense ones is still n=4."""
    dense = torch.zeros(1004, dtype=torch.bool)
    dense[1000:] = True
    assert t3_rank_control(torch.rand(1004),
                           dense)["verdict"] == "REFUSED_TOO_FEW"


def test_the_control_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        t3_rank_control(torch.rand(64), torch.zeros(63, dtype=torch.bool))


# ===========================================================================
# 5. ⛔ INERT AT DEFAULT — the constraint every cell in this programme carries
# ===========================================================================

def test_default_build_is_untouched_at_the_production_geometry():
    """⛔ 87,893,449 params / 405 keys. The live tensor-strict v6F S-W resume
    depends on this being exactly true. F-9 adds ZERO parameters: nothing it
    introduces is an ``nn.Module``."""
    m = V6Stack(V6Config())
    assert sum(p.numel() for p in m.parameters()) == 87_893_449
    assert len(m.state_dict()) == 405


def test_nothing_in_F9_is_a_module_or_carries_state():
    """The structural reason the pin above cannot drift: a curriculum is a
    schedule and a score is a number. Neither is a parameter."""
    assert not isinstance(T3Curriculum(), torch.nn.Module)
    assert not hasattr(T3Curriculum(), "state_dict")
    m = V6Stack(V6Config())
    assert not any("t3" in k for k in m.state_dict())


def test_F9_needs_no_stage_may_introduce_entry():
    """⭐ THE INSERTION-POINT ANSWER, pinned. F-7 needed
    ``STAGE_MAY_INTRODUCE["S-T"] += ("t2_head.",)`` because it added keys. F-9
    adds none, so there is nothing for the allowlist to adjudicate and it may
    be enabled in ANY stage's run, including one resuming tensor-strict."""
    from train_v6_staged import STAGE_MAY_INTRODUCE
    for stage, allowed in STAGE_MAY_INTRODUCE.items():
        assert not any("t3" in p for p in allowed), (stage, allowed)


def test_the_06b8782_class_does_not_apply_to_F9():
    """⚠️ Commit ``06b8782`` changed what S-J trains by APPENDING to
    ``MODULE_GROUPS`` without touching S-J's declaration. F-9 touches neither
    ``MODULE_GROUPS`` nor ``_GROUP_PREFIXES`` — it introduces no parameter to
    assign to a group at all."""
    from tanitad.models.v6 import LADDER_UNTRAINED_GROUPS, MODULE_GROUPS
    assert MODULE_GROUPS == ("encoder", "readout", "predictor_op", "layer_tac",
                             "layer_str", "planner", "aux", "interp")
    assert LADDER_UNTRAINED_GROUPS == frozenset({"interp"})


# ===========================================================================
# 6. the SCORE ARTIFACT loader — every refusal, and its opposite
# ===========================================================================

def _artifact(tmp_path, scores, provenance={"scorer": "p8", "ckpt": "abc"}):
    import torch as _t
    p = tmp_path / "t3.pt"
    blob = {"scores": _t.as_tensor(scores, dtype=_t.float32)}
    if provenance is not None:
        blob["provenance"] = provenance
    _t.save(blob, p)
    return str(p)


def test_the_loader_accepts_a_well_formed_declared_artifact(tmp_path):
    """C95/C97's passes-everything twin for the six refusals below."""
    from train_v6_staged import load_t3_scores
    s, prov = load_t3_scores(_artifact(tmp_path, [0.0, 0.5, 1.0]), n_windows=3)
    assert s.tolist() == [0.0, 0.5, 1.0]
    assert prov == {"scorer": "p8", "ckpt": "abc"}


def test_the_loader_refuses_an_UNDECLARED_artifact(tmp_path):
    """⛔ THE ADMISSIBILITY GUARD. T3's score descends from a decoder trained on
    the obstacle join — a LABEL path — while O4's docstring says the join is
    *"never a training-time selector"*. The F-10 precedent makes a label-derived
    SAMPLER input admissible **as a declared data mix**; this refusal is what
    makes the declaration real rather than aspirational."""
    from train_v6_staged import load_t3_scores
    with pytest.raises(SystemExit, match="provenance"):
        load_t3_scores(_artifact(tmp_path, [0.1, 0.2], provenance=None),
                       n_windows=2)
    with pytest.raises(SystemExit, match="provenance"):
        load_t3_scores(_artifact(tmp_path, [0.1, 0.2], provenance={}),
                       n_windows=2)


def test_the_loader_refuses_a_length_mismatch(tmp_path):
    """⚠️ The window count depends on max_horizon, which is derived from the
    STAGE's live loss terms — so a score file built for one stage does not
    transfer, and silently reweighting the wrong windows is the failure."""
    from train_v6_staged import load_t3_scores
    with pytest.raises(SystemExit, match="scores for"):
        load_t3_scores(_artifact(tmp_path, [0.1, 0.2, 0.3]), n_windows=94)


def test_the_loader_refuses_non_finite_and_negative_scores(tmp_path):
    from train_v6_staged import load_t3_scores
    with pytest.raises(SystemExit, match="non-finite"):
        load_t3_scores(_artifact(tmp_path, [0.1, float("nan")]), n_windows=2)
    with pytest.raises(SystemExit, match="NEGATIVE"):
        load_t3_scores(_artifact(tmp_path, [0.1, -0.3]), n_windows=2)


def test_the_loader_refuses_an_ALL_ZERO_score_file(tmp_path):
    """A curriculum that is uniform at every alpha is a curriculum the run does
    not have — the empty-rollout case the score functional documents."""
    from train_v6_staged import load_t3_scores
    with pytest.raises(SystemExit, match="every T3 score"):
        load_t3_scores(_artifact(tmp_path, [0.0, 0.0, 0.0]), n_windows=3)


def test_the_loader_refuses_a_file_that_is_not_an_artifact(tmp_path):
    import torch as _t
    from train_v6_staged import load_t3_scores
    p = tmp_path / "bare.pt"
    _t.save(_t.zeros(3), p)
    with pytest.raises(SystemExit, match="not a T3 score artifact"):
        load_t3_scores(str(p), n_windows=3)


# ===========================================================================
# 7. the CURRICULUM actually moves the DRAW — end-to-end through the sampler
# ===========================================================================

def test_the_sampler_draw_shifts_from_free_flow_to_dense():
    """⭐ The property that matters is not the weight vector but WHAT GETS
    DRAWN. One episode, half free-flow windows and half dense ones: at
    progress 0 the draw must favour free flow, at progress 1 dense."""
    from tanitad.models.v6 import InteractionSampler
    n = 200
    index = [(0, t) for t in range(n)]
    scores = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    c = T3Curriculum()

    def _dense_frac(prog: float) -> float:
        g = torch.Generator().manual_seed(4)
        smp = InteractionSampler(index, c.weights_at(scores, prog),
                                 eps_per_batch=1, generator=g)
        drawn = smp(800)
        return sum(1 for i in drawn if i >= n // 2) / len(drawn)

    early, late = _dense_frac(0.0), _dense_frac(1.0)
    assert early < 0.35, early      # starts on free flow
    assert late > 0.65, late        # ends on dense
    assert late > 2 * early


def test_free_flow_windows_are_still_drawn_at_full_dense_bias():
    """The parity property, at the DRAW rather than at the weight: even at
    alpha_end every window class remains reachable."""
    from tanitad.models.v6 import InteractionSampler
    n = 200
    index = [(0, t) for t in range(n)]
    scores = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    g = torch.Generator().manual_seed(9)
    smp = InteractionSampler(index, T3Curriculum().weights_at(scores, 1.0),
                             eps_per_batch=1, generator=g)
    drawn = smp(800)
    assert any(i < n // 2 for i in drawn), "free flow became unreachable"


def test_the_curriculum_refresh_precedes_the_draw_in_the_TRAIN_LOOP():
    """⛔ A DEFECT FOUND IN THIS CELL'S OWN FIRST WIRING, pinned so it cannot
    return. The refresh was originally placed AFTER ``idx = sample(...)``, so
    every step drew under the PREVIOUS step's exponent and the final update was
    never used at all. It would have been invisible in the logs: the alpha
    printed and the alpha drawn under are both 'correct', one step apart.

    Asserted against the SOURCE because the ordering lives inside ``train()``'s
    step loop, which needs a corpus to execute — the same idiom
    ``test_v6_t5_consistency`` uses to assert a loss never reads ``waypoints``.
    """
    src = (ROOT / "scripts" / "train_v6_staged.py").read_text(encoding="utf-8")
    loop = src.index("for step in range(start_step + 1, a.steps + 1):")
    refresh = src.index("if t3_curr is not None:", loop)
    draw = src.index("idx = sample(a.batch)", loop)
    pair_draw = src.index("if t5_partner:", loop)
    assert refresh < draw, "the curriculum refresh must precede the draw"
    assert refresh < pair_draw, "…including the F-8 paired draw"
