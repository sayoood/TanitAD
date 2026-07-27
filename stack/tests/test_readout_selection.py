"""C8 — tanitad/models/readout_selection.py.

The brief was explicit on both halves: **ship the rule, not a fitter**, because
fitting the switch to ADE costs 3.1x of ``T_blind`` to buy 0.7 % of ADE. So the
tests here pin the rule's VALUES, the byte-identity of the one-level path against
the existing ``rollout_decode`` (so C8 is a strict generalisation and not a
re-implementation), the graceful degradation on single-readout arms, and — the
adversarial one — that the module **cannot** fit anything, enforced by inspecting
every public signature for a ground-truth argument.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.models import readout_selection as RS  # noqa: E402
from tanitad.models.metric_dynamics import (  # noqa: E402
    HierarchicalGrounding, rollout_decode)


# ------------------------------------------------------------- the rule ------
def test_the_rule_is_op_early_and_str_late_at_the_registered_switch():
    assert RS.C8_SWITCH_STEP == 5                       # 0.5 s at dt = 0.1
    assert RS.C8_SWITCH_RANGE_S == (0.5, 0.8)
    assert (RS.C8_EARLY_LEVEL, RS.C8_LATE_LEVEL) == ("op", "str")

    plan = RS.readout_plan(20)
    assert plan[:5] == ["op"] * 5, "the first 0.5 s must read op"
    assert plan[5:] == ["str"] * 15, "everything past 0.5 s must read str"
    assert len(plan) == 20


def test_the_switch_is_NOT_at_the_ADE_optimum():
    """⭐ The trade the measurement rejected, pinned as a value.

    The ADE-optimal switch is 1.0 s (10 steps): it buys 0.0063 m (0.7 %) of
    ``ade_0_2s`` and collapses deployable ``T_blind`` from 2.5 s to 0.8 s — 3.1x.
    If someone 'optimises' this constant, this test is what fails.
    """
    assert RS.C8_SWITCH_STEP != 10, RS.C8_DO_NOT_FIT
    assert RS.C8_SWITCH_STEP * RS.DT <= RS.C8_SWITCH_RANGE_S[1]
    assert "3.1x" in RS.C8_DO_NOT_FIT and "T_blind" in RS.C8_DO_NOT_FIT


def test_the_module_ships_a_rule_and_CANNOT_fit_one():
    """⭐ Driven adversarially: every public callable is inspected for any way to
    hand it ground truth. A fitter needs a target; if no signature can accept
    one, no fitter can exist here by construction."""
    forbidden = ("target", "targets", "gt", "ground_truth", "y", "labels",
                 "ade", "error", "loss", "traj_tgt", "future_poses", "poses",
                 "train", "fit", "objective", "metric")
    for name in RS.__all__:
        obj = getattr(RS, name)
        if not callable(obj):
            continue
        assert "fit" not in name.lower(), f"{name} looks like a fitter"
        params = inspect.signature(obj).parameters
        bad = [p for p in params if p.lower() in forbidden]
        assert not bad, (
            f"{name} accepts {bad} — that is enough to fit the switch to error. "
            + RS.C8_DO_NOT_FIT)
    # no CALLABLE anywhere in the module is named like a fitter (the constant
    # C8_DO_NOT_FIT is the refusal itself, not a fitter)
    fitters = [n for n in dir(RS) if not n.startswith("_")
               and "fit" in n.lower() and callable(getattr(RS, n))]
    assert fitters == [], fitters


def test_the_provenance_travels_and_states_its_scope_limit():
    p = RS.C8_PROVENANCE
    assert "v1" in p["measured_on"] and "v1 ONLY" in p["scope_limit"]
    assert "hold_v0" in p["not_a_rescue"], (
        "the rule must carry the fact that it does not rescue the regime")
    assert p["switch_step"] == RS.C8_SWITCH_STEP


# ------------------------------------------------- graceful degradation ------
class _Bare(torch.nn.Module):
    """A REF-A/B/C-shaped arm: ONE readout, no op/tac/str bank."""

    def __init__(self, s):
        super().__init__()
        self.net = torch.nn.Linear(2 * s, 3)

    def forward(self, z, zn):
        return self.net(torch.cat([z, zn], dim=-1))


def test_a_single_readout_arm_degrades_LOUDLY_not_silently():
    """REF-A has no bank. It must not raise, must not pretend, and must say so."""
    bare = _Bare(8)
    bank, tele = RS.resolve_readout_bank(bare)
    assert set(bank) == {"op"} and bank["op"] is bare
    assert tele["c8_available"] is False
    assert "NOT a C8 number" in tele["note"]
    # and the plan collapses to one level rather than KeyError-ing on "str"
    assert set(RS.readout_plan(20, available=("op",))) == {"op"}


def test_a_real_bank_is_detected():
    g = HierarchicalGrounding(8, hidden=8)
    bank, tele = RS.resolve_readout_bank(g)
    assert sorted(bank) == ["op", "str", "tac"] and tele["c8_available"] is True


# --------------------------------------------------------- the decode --------
class _Pred(torch.nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.f = torch.nn.Linear(s, s)

    def forward(self, win_s, win_a):
        return None, self.f(win_s[:, -1])


def _fix(b=3, s=8, w=4, k=8, seed=0):
    torch.manual_seed(seed)
    return (_Pred(s), torch.randn(b, w, s), torch.randn(b, w, 3),
            torch.randn(b, k, 3), HierarchicalGrounding(s, hidden=8), k)


def test_c8_with_ONE_level_is_BIT_IDENTICAL_to_the_existing_rollout_decode():
    """C8 must be a strict generalisation of the shipped path, not a rewrite.

    Driven at ``switch_step = 0`` and ``switch_step >= k`` (the two degenerate
    settings), where the answer must equal ``rollout_decode`` on that one head
    EXACTLY — Δ == 0, not 'close'.
    """
    pred, st, ac, fa, g, k = _fix()
    ref_op, _ = rollout_decode(pred, st, ac, fa, g.step["op"], k)
    ref_str, _ = rollout_decode(pred, st, ac, fa, g.step["str"], k)

    all_op, t1 = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k,
                                              switch_step=k)
    assert torch.equal(all_op, ref_op), "switch>=k must be pure op"
    assert t1["switch_discontinuity_m"] == 0.0

    all_str, _ = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k,
                                              switch_step=0)
    assert torch.equal(all_str, ref_str), "switch==0 must be pure str"


def test_c8_reads_each_HEADS_OWN_full_path_not_a_spliced_accumulation():
    """The validated semantics: decode the whole rollout with each head, then read
    path r[j] at index j. A per-step Δpose splice inside one SE(2) accumulation is
    a different object that was never validated — this pins which one we ship."""
    pred, st, ac, fa, g, k = _fix()
    ref_op, _ = rollout_decode(pred, st, ac, fa, g.step["op"], k)
    ref_str, _ = rollout_decode(pred, st, ac, fa, g.step["str"], k)

    wp, tele = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k, switch_step=3)
    assert torch.equal(wp[:, :3], ref_op[:, :3])
    assert torch.equal(wp[:, 3:], ref_str[:, 3:])
    assert tele["per_step_level"] == ["op"] * 3 + ["str"] * (k - 3)
    assert tele["levels_used"] == ["op", "str"]


def test_the_rollout_is_executed_ONCE_which_is_the_free_in_the_claim():
    pred, st, ac, fa, g, k = _fix()
    calls = {"n": 0}
    inner = pred.forward

    def counting(win_s, win_a):
        calls["n"] += 1
        return inner(win_s, win_a)

    pred.forward = counting
    _, tele = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k, switch_step=3)
    assert calls["n"] == k, (
        f"the predictor ran {calls['n']} times for k={k}; two levels must SHARE "
        f"one roll or the 'zero GPU' claim is false")
    assert tele["rollouts_executed"] == 1 and tele["levels_decoded"] == 2


def test_the_switch_DISCONTINUITY_is_measured_and_reported_not_hidden():
    """⭐ The artefact of the validated semantics. Driven with two heads forced far
    apart, so the jump is unmistakable and must appear in the telemetry."""
    pred, st, ac, fa, g, k = _fix()
    with torch.no_grad():                      # make str disagree with op, loudly
        for p in g.step["str"].parameters():
            p.add_(5.0)
    _, tele = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k, switch_step=3)
    assert tele["switch_discontinuity_m"] > 0.0
    assert tele["switch_discontinuity_max_m"] >= tele["switch_discontinuity_m"]
    assert "not interpolated" in tele["_discontinuity_note"]

    # a single-level path has nothing to jump across
    _, t2 = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k, switch_step=k)
    assert t2["switch_discontinuity_m"] == 0.0


def test_shapes_and_finiteness():
    pred, st, ac, fa, g, k = _fix()
    wp, _ = RS.calibrated_rollout_decode(pred, st, ac, fa, g, k)
    assert wp.shape == (3, k, 2) and torch.isfinite(wp).all()


def test_a_bare_arm_still_produces_a_path_flagged_as_non_c8():
    pred, st, ac, fa, _g, k = _fix()
    bare = _Bare(8)
    wp, tele = RS.calibrated_rollout_decode(pred, st, ac, fa, bare, k)
    ref, _ = rollout_decode(pred, st, ac, fa, bare, k)
    assert torch.equal(wp, ref)
    assert tele["c8_available"] is False
