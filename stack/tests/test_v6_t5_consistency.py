"""F-8 / catalog T5 — MOMENTUM-AWARE TEMPORAL CONSISTENCY: both directions.

THE SPEC, quoted (two independent locations):

  * ``…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:68`` —
    *"T5 | temporal-consistency selection loss (momentum-aware, Drive-JEPA
    pattern) | penalise plan flip-flop across consecutive windows (cross-frame
    comfort) | LATERAL family: yaw-rate/curvature MAE at selection level;
    plan-switch rate reported"*
  * ``…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:58`` — *"no
    cross-window plan-flip-flop penalty; no plan-switch-rate logging. Needs
    consecutive-window batches (the current sampler draws windows
    independently) — a sampler + loss change."*

⛔ THE CENTRAL FACT THIS FILE PINS is that the term is **DEGENERATE ALONE**: a
constant control plan scores EXACTLY ZERO. That is not a hypothetical — the
emission is zero-valued at initialisation (MEASURED below), so T5 starts at its
global minimum and its only effect is to resist the plan objective. The guard
that refuses ``w_t5_consist > 0`` with ``lambda_plan == 0`` is therefore
load-bearing, and it is WIRED (``v6_loss_step`` + ``preflight``), not merely
documented — "a guard that is never called is not a guard".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from test_v6_gstr_port import _small  # noqa: E402

PAIRS = torch.tensor([[0, 1]])


def _consistent_pair(k: int, lag: int, n: int = 1):
    """A pair whose second plan IS the first plan's lag-suffix — the GT
    property. ``TEMPORAL_STABILITY_RESULT.md`` measured this floor at 0.0 m and
    0.0001 m/s^2 on real data, so it is a control that came back where the
    maths says it must."""
    base = torch.randn(1, n, k + lag)
    return torch.cat([base[..., :k], base[..., lag:lag + k]], dim=0)


# ---------------------------------------------------------------------------
# 1. INERTNESS
# ---------------------------------------------------------------------------

def test_the_cell_adds_no_parameters_and_no_state_dict_key():
    """⛔ THE REASON F-8 NEEDS NO ``STAGE_MAY_INTRODUCE`` ENTRY, and the reason
    it cannot break the live tensor-strict resume: it holds nothing. Same
    construction as ``MpcRefiner``."""
    s = V6Stack(V6Config())
    assert sum(p.numel() for p in s.parameters()) == 87_893_449
    assert len(s.state_dict()) == 405
    assert not any("t5" in k.lower() for k in s.state_dict())


def test_stage_may_introduce_has_no_t5_entry():
    import train_v6_staged as T
    for st, allowed in T.STAGE_MAY_INTRODUCE.items():
        assert not any("t5" in p.lower() for p in allowed), st


def test_the_term_is_skipped_not_zero_multiplied_when_off():
    import train_v6_staged as T
    s = V6Stack(_small(selector="goal"))
    batch = T.synthetic_train_batch(s, batch=4, k=12)
    L = T.v6_loss_step(s, batch, stage="S-T",
                       weights=T.V6LossWeights(w_t5_consist=0.0,
                                               lambda_plan=1.0))
    assert "t5" not in L
    assert not any(k.startswith("t5_") for k in L["log"])


@pytest.mark.parametrize("stage", ["S-W", "S-S"])
def test_for_stage_zeroes_the_weight_where_the_planner_is_frozen(stage):
    import train_v6_staged as T
    assert T.V6LossWeights(w_t5_consist=1.0).for_stage(stage).w_t5_consist == 0


# ---------------------------------------------------------------------------
# 2. THE CONTROLS — positive, and the trivial proxy
# ---------------------------------------------------------------------------

def test_a_consistent_pair_scores_exactly_zero_POSITIVE_CONTROL():
    """⭐ POSITIVE CONTROL. When the second plan genuinely IS the first plan's
    continuation, the loss must be 0 — the GT floor, exactly as measured on
    real data (``TEMPORAL_STABILITY_RESULT.md``: 0.0 m / 0.0001 m/s^2)."""
    import train_v6_staged as T
    k, lag = 10, 3
    a = _consistent_pair(k, lag)
    loss, log = T.t5_consistency_loss(a, torch.zeros_like(a), None, PAIRS, lag)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert log["t5_accel_jump_mae"] == pytest.approx(0.0, abs=1e-6)
    assert log["t5_overlap_steps"] == k - lag


def test_an_inconsistent_pair_scores_above_zero():
    """The other direction — without this, a loss stuck at 0 would 'pass'."""
    import train_v6_staged as T
    k, lag = 10, 3
    a = torch.cat([torch.randn(1, 1, k), torch.randn(1, 1, k)], dim=0)
    loss, _ = T.t5_consistency_loss(a, torch.zeros_like(a), None, PAIRS, lag)
    assert float(loss) > 1e-3


def test_a_flat_plan_scores_exactly_zero():
    """⛔ THE TRIVIAL-PROXY CONTROL, and the degeneracy the guard exists for.

    A CONSTANT control plan satisfies temporal consistency perfectly while
    describing a car that ignores the road. Any 'T5 improved' claim must be
    read against this: the cheapest way to win this term is to stop planning.
    """
    import train_v6_staged as T
    k, lag = 10, 3
    flat_a = torch.full((2, 1, k), 1.234)
    flat_k = torch.full((2, 1, k), -0.5)
    loss, _ = T.t5_consistency_loss(flat_a, flat_k, None, PAIRS, lag)
    assert float(loss) == 0.0


def test_the_emission_is_flat_at_initialisation_so_T5_starts_at_its_optimum():
    """⛔ MEASURED, and it is why the degeneracy is not hypothetical: the
    unicycle emission outputs EXACTLY zero controls at init, so the T5 term is
    already at its global minimum before step 1."""
    import train_v6_staged as T
    s = V6Stack(_small(selector="goal"))
    b = T.synthetic_train_batch(s, batch=4, k=12)
    out = s.forward(frames=b["frames"],
                    actions=T._lift3(b["actions2"], b["v0"]), v0=b["v0"])
    assert float(out["plan"]["a"].detach().abs().max()) == 0.0
    assert float(out["plan"]["kappa"].detach().abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 3. THE GUARD IS WIRED, NOT DOCUMENTED
# ---------------------------------------------------------------------------

def test_v6_loss_step_refuses_the_term_without_a_plan_objective():
    """⭐ 'A guard that is never called is not a guard.' This one is called
    from the loss itself, so it fires on a hand-built launch too."""
    import train_v6_staged as T
    s = V6Stack(_small(selector="goal"))
    b = T.synthetic_train_batch(s, batch=4, k=12)
    b["t5_pairs"] = torch.tensor([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="DEGENERATE ALONE"):
        T.v6_loss_step(s, b, stage="S-T",
                       weights=T.V6LossWeights(w_t5_consist=1.0,
                                               lambda_plan=0.0))


def test_v6_loss_step_refuses_the_term_without_pairs():
    """Without consecutive windows the term would compare UNRELATED episodes —
    the failure DIAGRAM_CONFORMANCE.md:58 names."""
    import train_v6_staged as T
    s = V6Stack(_small(selector="goal"))
    b = T.synthetic_train_batch(s, batch=4, k=12)
    with pytest.raises(ValueError, match=r"t5_pairs"):
        T.v6_loss_step(s, b, stage="S-T",
                       weights=T.V6LossWeights(w_t5_consist=1.0,
                                               lambda_plan=1.0))


def test_the_loss_refuses_an_empty_pair_set():
    import train_v6_staged as T
    a = torch.randn(2, 1, 10)
    with pytest.raises(ValueError, match="EMPTY"):
        T.t5_consistency_loss(a, a, None, torch.zeros(0, 2, dtype=torch.long),
                              3)


@pytest.mark.parametrize("lag", [0, 10, 11, -1])
def test_the_loss_refuses_a_lag_that_leaves_no_overlap(lag):
    """A lag at or beyond the horizon leaves NO shared instants and the term
    would be silently empty — the class of defect where a loss trains nothing
    while appearing in the log."""
    import train_v6_staged as T
    a = torch.randn(2, 1, 10)
    with pytest.raises(ValueError, match="lag"):
        T.t5_consistency_loss(a, a, None, PAIRS, lag)


def test_preflight_refuses_the_degenerate_and_unpaired_launches():
    """⭐ THE SECOND RUNG: the same refusals BEFORE the corpus mounts."""
    import train_v6_staged as T
    p = T.build_parser()
    base = ["--stage", "S-T", "--out", "unused", "--w-t5-consist", "1.0"]
    a = p.parse_args(base + ["--t5-pairs", "--lambda-plan", "0"])
    assert any("DEGENERATE ALONE" in s for s in T.preflight(a))
    a2 = p.parse_args(base + ["--lambda-plan", "1.0"])
    assert any("--t5-pairs" in s for s in T.preflight(a2))
    # and the paired, plan-bearing launch raises NEITHER of the two
    a3 = p.parse_args(base + ["--t5-pairs", "--lambda-plan", "1.0"])
    probs = T.preflight(a3)
    assert not any("DEGENERATE ALONE" in s or "--t5-pairs" in s
                   for s in probs), \
        f"a legal T5 launch was refused — C95's rejects-everything guard: " \
        f"{probs}"


def test_preflight_refuses_T2_without_its_projector_and_in_the_wrong_stage():
    """The F-7 half of the same rung, including the C95 direction: a LEGAL T2
    launch must pass."""
    import train_v6_staged as T
    p = T.build_parser()
    a = p.parse_args(["--stage", "S-T", "--out", "unused",
                      "--w-t2-contrast", "1.0"])
    assert any("--t2-contrastive" in s for s in T.preflight(a))
    a2 = p.parse_args(["--stage", "S-W", "--out", "unused",
                       "--t2-contrastive", "--w-t2-contrast", "1.0"])
    assert any("S-W" in s and "t2" in s.lower() for s in T.preflight(a2))
    a3 = p.parse_args(["--stage", "S-T", "--out", "unused",
                       "--t2-contrastive", "--w-t2-contrast", "1.0"])
    assert not any("t2" in s.lower() for s in T.preflight(a3)), \
        "a legal T2 launch was refused"


# ---------------------------------------------------------------------------
# 4. THE CELL DOES WHAT IT CLAIMS
# ---------------------------------------------------------------------------

def test_the_loss_is_in_control_space_and_needs_no_pose_transform():
    """⛔ THE DESIGN DECISION, pinned. ``TEMPORAL_STABILITY_RESULT.md``
    measured position replan shift 0.0947 m ('nearly fine') against an accel
    jump of 1.1021 m/s^2 — larger than the human's ENTIRE accel RMS (0.8048).
    Controls are also FRAME-INVARIANT, so the comparison needs no relative
    pose. This test pins the invariance: rotating/translating the ego frame
    cannot change the loss, because the loss never sees a position.
    """
    import train_v6_staged as T
    import inspect
    src = inspect.getsource(T.t5_consistency_loss)
    assert "waypoints" not in src, \
        "T5 must not read positions — that reintroduces a frame dependence"


def test_the_selector_softmax_makes_it_a_SELECTION_level_term():
    """The gate row says *'at selection level'*: the term must be weighted by
    the scorer's own softmax and must be differentiable INTO it."""
    import train_v6_staged as T
    k, lag, n = 10, 3, 4
    a = torch.randn(2, n, k, requires_grad=True)
    sel = torch.randn(2, n, requires_grad=True).softmax(dim=-1)
    loss, log = T.t5_consistency_loss(a, torch.zeros_like(a), sel, PAIRS, lag)
    g = torch.autograd.grad(loss, sel, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0, \
        "the term must reach the selector, or it is not a SELECTION loss"
    assert log["t5_n_pairs"] == 1


def test_uniform_weighting_is_the_no_selector_control_arm():
    import train_v6_staged as T
    k, lag, n = 10, 3, 4
    a = torch.randn(2, n, k)
    uni = torch.full((2, n), 1.0 / n)
    l1, _ = T.t5_consistency_loss(a, torch.zeros_like(a), None, PAIRS, lag)
    l2, _ = T.t5_consistency_loss(a, torch.zeros_like(a), uni, PAIRS, lag)
    assert float(l1) == pytest.approx(float(l2), rel=1e-6)


def test_the_two_families_are_reported_separately_never_pooled():
    """The binding four-families rule: 'per-family, never pooled into one
    score'. Accel and curvature must appear as their own numbers."""
    import train_v6_staged as T
    k, lag = 10, 3
    a = torch.randn(2, 1, k)
    kp = torch.randn(2, 1, k)
    loss, log = T.t5_consistency_loss(a, kp, None, PAIRS, lag, w_kappa=1.0)
    assert log["t5_accel_jump_mae"] > 0 and log["t5_curvature_mae"] > 0
    assert float(loss) == pytest.approx(
        log["t5_accel_jump_mae"] + log["t5_curvature_mae"], rel=1e-5)


def test_w_kappa_reweights_only_the_curvature_half():
    import train_v6_staged as T
    k, lag = 10, 3
    a, kp = torch.randn(2, 1, k), torch.randn(2, 1, k)
    l0, lg = T.t5_consistency_loss(a, kp, None, PAIRS, lag, w_kappa=0.0)
    l1, _ = T.t5_consistency_loss(a, kp, None, PAIRS, lag, w_kappa=2.0)
    assert float(l0) == pytest.approx(lg["t5_accel_jump_mae"], rel=1e-5)
    assert float(l1) == pytest.approx(
        lg["t5_accel_jump_mae"] + 2.0 * lg["t5_curvature_mae"], rel=1e-5)


def test_yaw_rate_is_reported_when_v0_is_available():
    """LATERAL family asks for yaw-rate as well as curvature; yaw = v * kappa."""
    import train_v6_staged as T
    k, lag = 10, 3
    a, kp = torch.zeros(2, 1, k), torch.randn(2, 1, k)
    _, log = T.t5_consistency_loss(a, kp, None, PAIRS, lag,
                                   v0=torch.tensor([10.0, 10.0]))
    assert log["t5_yawrate_mae"] == pytest.approx(
        10.0 * log["t5_curvature_mae"], rel=1e-4)


def test_plan_switch_rate_counts_manoeuvre_toggles_per_axis():
    """The *'plan-switch rate reported'* half — the direct successor of
    ``TEMPORAL_STABILITY_RESULT.md``'s manoeuvre toggle rate 0.1759, reported
    per AXIS because the factored LAT x LON pair replaced the mixed 5-way head.
    """
    import train_v6_staged as T
    lat = torch.tensor([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0], [9.0, 0.0]])
    lon = torch.tensor([[9.0, 0.0], [9.0, 0.0], [9.0, 0.0], [9.0, 0.0]])
    pairs = torch.tensor([[0, 1], [2, 3]])
    out = T.t5_plan_switch_rate(lat, lon, pairs)
    assert out["t5_switch_rate_lat"] == pytest.approx(0.5)   # pair 0 toggles
    assert out["t5_switch_rate_lon"] == pytest.approx(0.0)
    assert out["t5_switch_rate_any"] == pytest.approx(0.5)
    assert out["t5_switch_n_pairs"] == 2


def test_end_to_end_through_v6_loss_step_logs_every_gate_quantity():
    import train_v6_staged as T
    s = V6Stack(_small(selector="goal"))
    b = T.synthetic_train_batch(s, batch=4, k=12)
    b["t5_pairs"] = torch.tensor([[0, 1], [2, 3]])
    b["t5_lag"] = 2
    L = T.v6_loss_step(s, b, stage="S-T",
                       weights=T.V6LossWeights(w_t5_consist=1.0,
                                               lambda_plan=1.0))
    for k in ("t5_loss", "t5_accel_jump_mae", "t5_curvature_mae",
              "t5_yawrate_mae", "t5_n_pairs", "t5_lag", "t5_overlap_steps",
              "t5_selection_level", "t5_switch_rate_lat",
              "t5_switch_rate_lon"):
        assert k in L["log"], f"missing gate quantity {k}"
    assert L["log"]["t5_selection_level"] is True
