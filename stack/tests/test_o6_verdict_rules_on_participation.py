"""D1 / C132 REPAIR -- the O6 collapse verdict must rule on the ENERGY statistic.

THE DEFECT. ``o6_rank_verdict`` set ``pass`` from ``effective_rank`` (p ~ sigma,
AMPLITUDE) and never consulted ``participation_ratio`` (p ~ sigma^2, ENERGY),
even though its own ``statistic_note`` said "COLLAPSE is an energy question --
decide on participation". The two statistics do not merely disagree in
magnitude; they INVERT the ordering.

THE DELIBERATE REGRESSION IN THIS FILE. ``_OLD_RULE_frozen`` below is a verbatim
re-implementation of the pre-repair ruling. Every regression test feeds it the
SAME two readings the live verdict gets and asserts the old rule says PASS while
the repaired one says FAIL. A guard that cannot fail is decoration, so the
inverting input is constructed here rather than hoped for.

MEASURED with this file's own builders (n=1100, d=1024, admissible ceiling):
    COLLAPSED  top-1 energy 0.549   effective_rank 769.09   participation  3.31
    HEALTHY    top-1 energy 0.142   effective_rank 659.69   participation 31.27
The collapsed representation has the HIGHER effective_rank. That is the whole
defect in two rows, and it is the same shape the v7f board reports for
``postrain30k_freeze`` (lowest by participation, highest by effective_rank).

ASCII-only strings: this box is cp1252 and a non-ASCII assertion message is
fatal at print time, not at compare time.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.models.v6 import (O6_PARTICIPATION_FLOOR, O6_RANK_FLOOR,
                               o6_rank_verdict, spectrum_report)

# small enough to run in ~0.1 s, with >= 4 jackknife blocks so the retention
# clause can actually rule. `ceiling_min` is lowered to D in the calls below;
# the ruling logic does not depend on the ceiling's absolute value, and the
# full-scale admissible reading is exercised by the premise test.
N, D, BLOCK = 260, 256, 26


def _collapsed(top1_share: float = 0.55, seed: int = 7,
               n: int = N, d: int = D) -> torch.Tensor:
    """Rows with ``top1_share`` of the ENERGY in a single direction, and a long
    flat tail of tiny-but-nonzero directions -- exactly what inflates
    effective_rank while participation collapses."""
    g = torch.Generator().manual_seed(seed)
    var = torch.full((d,), (1.0 - top1_share) / (d - 1), dtype=torch.float64)
    var[0] = top1_share
    return torch.randn(n, d, generator=g, dtype=torch.float64) * var.sqrt()


def _healthy(seed: int = 11, n: int = N, d: int = D) -> torch.Tensor:
    """A mild power-law spectrum: no single dominant axis."""
    g = torch.Generator().manual_seed(seed)
    j = torch.arange(1, d + 1, dtype=torch.float64)
    return torch.randn(n, d, generator=g, dtype=torch.float64) * (1.0 / j).sqrt()


def _OLD_RULE_frozen(cur: dict, ref: dict | None, *, retention: float = 0.8,
                     floor: float = O6_RANK_FLOOR,
                     ceiling_min: int = 1024) -> str:
    """VERBATIM the pre-repair ruling: every clause on ``effective_rank``.

    Frozen on purpose. It is the regression arm, so it must NOT be refactored
    to share code with the thing it is testing.
    """
    ceiling = int(cur.get("rank_ceiling", min(int(cur.get("n", 2)) - 1,
                                              int(cur.get("d", 1)))))
    er = float(cur["effective_rank"])
    if ceiling < ceiling_min:
        return "INCONCLUSIVE"
    if er < floor:
        return "FAIL"
    if ref is None:
        return "INCONCLUSIVE"
    er0 = float(ref["effective_rank"])
    if er0 <= 0:
        return "INCONCLUSIVE"
    ci_c, ci_r = cur.get("effective_rank_ci95"), ref.get("effective_rank_ci95")
    if not (isinstance(ci_c, dict) and "lo" in ci_c
            and isinstance(ci_r, dict) and "lo" in ci_r):
        return "INCONCLUSIVE"
    lo = max(ci_c["lo"], 0.0) / max(ci_r["hi"], 1e-12)
    hi = max(ci_c["hi"], 0.0) / max(ci_r["lo"], 1e-12)
    if hi < retention:
        return "FAIL"
    if lo >= retention:
        return "PASS"
    return "INCONCLUSIVE"


@pytest.fixture(scope="module")
def readings() -> tuple[dict, dict]:
    """One collapsed reading and one healthy reference, both with intervals."""
    cur = spectrum_report(_collapsed(), ci_reps=1, block=BLOCK)
    ref = spectrum_report(_healthy(), ci_reps=1, block=BLOCK)
    return cur, ref


# --------------------------------------------------------------------------
# PREMISE (true on both trees -- it describes spectrum_report, not the verdict)
# --------------------------------------------------------------------------
def test_PREMISE_the_inverting_representation_exists_at_an_admissible_ceiling():
    """55 % of the energy in ONE direction, yet effective_rank clears its floor
    by an order of magnitude AND beats a genuinely healthier representation."""
    zc = _collapsed(n=1100, d=1024)
    col = spectrum_report(zc)
    hea = spectrum_report(_healthy(n=1100, d=1024))
    assert col["rank_ceiling"] >= 1024, "premise needs an ADMISSIBLE ceiling"

    sv = torch.linalg.svdvals(zc - zc.mean(0, keepdim=True))
    top1 = float((sv[0] ** 2) / (sv ** 2).sum())
    assert 0.50 < top1 < 0.60, f"top-1 energy share {top1:.4f} is not ~0.55"

    # the false PASS: an energy-collapsed arm sails over the amplitude floor
    assert col["effective_rank"] >= O6_RANK_FLOOR, (
        f"effective_rank {col['effective_rank']:.2f} should clear the old "
        f"floor {O6_RANK_FLOOR} -- that is what makes it a false pass")
    # the inversion: collapsed reads HIGHER than healthy on the wrong statistic
    assert col["effective_rank"] > hea["effective_rank"], (
        f"collapsed effective_rank {col['effective_rank']:.2f} must exceed "
        f"healthy {hea['effective_rank']:.2f} -- this is the sign flip")
    # and the right statistic orders them correctly
    assert col["participation_ratio"] < hea["participation_ratio"], (
        f"participation must order these correctly: collapsed "
        f"{col['participation_ratio']:.2f} < healthy "
        f"{hea['participation_ratio']:.2f}")
    assert col["participation_ratio"] < O6_PARTICIPATION_FLOOR


# --------------------------------------------------------------------------
# THE REGRESSION TESTS -- these FAIL on the unfixed tree
# --------------------------------------------------------------------------
def test_REGRESSION_old_rule_PASSES_what_the_repaired_gate_FAILS(readings):
    """The decisive one. Same two readings, opposite verdicts."""
    cur, ref = readings
    old = _OLD_RULE_frozen(cur, ref, ceiling_min=D)
    new = o6_rank_verdict(cur, ref, ceiling_min=D)

    assert old == "PASS", (
        "the frozen pre-repair rule must PASS this input, otherwise it is not "
        f"a deliberate regression -- got {old}")
    assert new["status"] == "FAIL" and new["pass"] is False, (
        "the repaired gate must FAIL a representation with 55 % of its energy "
        f"in one direction -- got {new['status']} / {new['pass']}. On the "
        "UNFIXED tree this reads PASS, which is the defect.")
    assert new["ruling_statistic"] == "participation_ratio"


def test_REGRESSION_retention_is_computed_on_participation_not_amplitude(readings):
    cur, ref = readings
    v = o6_rank_verdict(cur, ref, ceiling_min=D)
    assert v["retention_statistic"] == "participation_ratio"
    expect = cur["participation_ratio"] / ref["participation_ratio"]
    assert v["retention"] == pytest.approx(expect), (
        "`retention` must be the participation ratio; on the unfixed tree it "
        "is the effective_rank ratio")
    # the amplitude ratio is > 1 here, so reading it would invert the verdict
    diag = v["effective_rank_retention_DIAGNOSTIC"]
    assert diag > 1.0 > v["retention"], (
        f"the two retentions must straddle 1.0 for this to be a real "
        f"inversion: amplitude {diag:.4f} vs energy {v['retention']:.4f}")
    assert v["retention_ci95"]["statistic"] == "participation_ratio"


def test_REGRESSION_effective_rank_is_reported_but_never_ruling(readings):
    """The PI-visible half: effective_rank stays, demoted and labelled."""
    cur, ref = readings
    v = o6_rank_verdict(cur, ref, ceiling_min=D)
    # still reported -- the banked series stays comparable
    for k in ("effective_rank", "rank_ceiling", "effective_rank_frac",
              "absolute_floor", "participation_ratio", "participation_floor",
              "participation_pass", "statistic_note"):
        assert k in v, f"pre-existing key {k} was REMOVED"
    assert v["effective_rank"] == pytest.approx(cur["effective_rank"])
    # ... and explicitly marked non-ruling, inside the record itself
    assert v["effective_rank_is_ruling"] is False
    assert v["participation_floor_is_ruling"] is False
    assert v["effective_rank_above_floor_DIAGNOSTIC"] is True, (
        "this arm clears the amplitude floor -- that fact is exactly why the "
        "floor must not rule")


def test_REGRESSION_no_silent_fallback_to_effective_rank(readings):
    """A reading without the ruling statistic must be INCONCLUSIVE, never a
    verdict computed from the inverting one."""
    cur, ref = readings
    stripped = {k: v for k, v in cur.items() if k != "participation_ratio"}
    v = o6_rank_verdict(stripped, ref, ceiling_min=D)
    assert v["pass"] is None and v["status"] == "INCONCLUSIVE"
    assert "does NOT fall back to effective_rank" in v["reason"]


def test_REGRESSION_healthy_arm_still_passes(readings):
    """The guard must not simply fail everything -- a control that must read a
    KNOWN value. A healthy arm against itself retains 1.0 and passes."""
    _, ref = readings
    v = o6_rank_verdict(ref, ref, ceiling_min=D)
    assert v["status"] == "PASS" and v["pass"] is True, v["reason"]
    assert v["retention"] == pytest.approx(1.0)


def test_REGRESSION_participation_interval_exists_and_is_labelled(readings):
    cur, _ = readings
    ci = cur["participation_ratio_ci95"]
    assert ci["statistic"] == "participation_ratio"
    assert ci["lo"] < cur["participation_ratio"] < ci["hi"]
    # the coverage stamp must NOT be quoted as if measured for this statistic
    assert "INHERITED" in ci["measured_coverage_scope"]


# --------------------------------------------------------------------------
# THE ABSOLUTE CLAUSE -- surfaced as a PI decision, never silently ruling
# --------------------------------------------------------------------------
def test_absolute_clause_does_not_rule_without_a_named_reference(readings):
    cur, ref = readings
    v = o6_rank_verdict(cur, ref, ceiling_min=D)
    assert v["absolute_clause"]["ruling"] is False
    assert v["absolute_clause"]["status"] == "REPORTED_NOT_RULING"
    assert "pi_decision_pending" in v["absolute_clause"]

    # a floor WITHOUT its corpus/d reference is inadmissible, not a FAIL
    v2 = o6_rank_verdict(cur, ref, ceiling_min=D, participation_floor=8.56)
    assert v2["absolute_clause"]["status"] == "INCONCLUSIVE"
    assert v2["absolute_clause"]["ruling"] is False


def test_absolute_clause_fires_only_when_explicitly_armed(readings):
    """Armed with BOTH a floor and a named reference, clause 3 rules -- and it
    rules on participation, not on effective_rank."""
    cur, ref = readings
    v = o6_rank_verdict(cur, ref, ceiling_min=D, participation_floor=8.56,
                        participation_reference="dinov3_vitl16_meanpooled/"
                                                "physicalai-val-12clips/n1440/d1024")
    assert v["absolute_clause"]["status"] == "ARMED"
    assert v["status"] == "FAIL"
    assert "participation_ratio" in v["reason"]
    # the same arm clears the AMPLITUDE floor, so an er-based clause 3 could
    # never have produced this FAIL
    assert cur["effective_rank"] >= O6_RANK_FLOOR
