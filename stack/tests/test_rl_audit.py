"""The audit must be shown ABLE TO FAIL before any PASS it gives means anything.

⛔ THE TWO TESTS THAT MATTER MOST IN THIS FILE
----------------------------------------------
* ``test_hackable_reward_IS_FLAGGED`` — the DELIBERATE-REGRESSION arm. A
  progress-only reward is maximised by driving straight and fast through
  everything. If the audit does not flag it, the audit is broken and every
  clean report it has ever produced is void.
* ``test_default_reward_is_not_flagged`` — the other direction. A guard that
  flags everything is equally useless.

Both directions are required. One alone proves nothing.
"""

from __future__ import annotations

import pytest
import torch

from tanitad.rl import audit as AUD
from tanitad.rl import rewards as R


# ---------------------------------------------------------------------------
# Selector disjointness — the winner's-curse firewall, structurally
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(R.FORBIDDEN_REWARD_INPUTS))
def test_every_forbidden_key_is_rejected(key):
    with pytest.raises(ValueError, match="DISJOINTNESS VIOLATED"):
        AUD.assert_selector_disjoint({key: torch.zeros(3)})


def test_clean_context_passes():
    AUD.assert_selector_disjoint({"obstacles": torch.zeros(1, 2),
                                  "gt_traj": torch.zeros(21, 2)})


def test_forbidden_set_covers_the_v3_output_keys():
    """Pins the firewall against the ACTUAL names refc_v3.forward emits.

    If refc_v3 renames a ranking output, this test fails and someone must decide
    — rather than the reward silently gaining access to a new selector score.
    """
    emitted_rankings = {"sel_score", "sel_score_v3", "anchor_logits",
                        "sel_idx", "traj", "traj_base", "prefinal_logits"}
    missing = emitted_rankings - R.FORBIDDEN_REWARD_INPUTS
    assert not missing, (
        f"refc_v3 emits ranking key(s) {sorted(missing)} that the reward "
        "firewall does not block")


def test_audit_reward_runs_the_disjointness_check_itself():
    with pytest.raises(ValueError, match="DISJOINTNESS VIOLATED"):
        AUD.audit_reward(R.RewardSpec(), {"sel_score": torch.zeros(3)})


# ---------------------------------------------------------------------------
# The deliberate-regression arm
# ---------------------------------------------------------------------------

def test_hackable_reward_IS_FLAGGED():
    """⛔ THE LOAD-BEARING TEST. progress-only must be caught."""
    spec = R.RewardSpec(weights=dict(R.HACKABLE_WEIGHTS))
    report = AUD.audit_reward(spec)
    assert report.flagged, (
        "the progress-only reward was NOT flagged — the audit is broken, and "
        f"every clean verdict it has given is void.\n{report}")
    assert "bullet_straight" in report.winners, (
        "expected the straight-line max-speed policy to win a progress-only "
        f"reward; winners were {report.winners}")


def rich_ctx():
    """A context that actually EXERCISES every weighted component.

    ⚠️ TWO placements were wrong before this one, and both were instructive:
      * ``[12, 0]`` — directly under the reference. The "sane" reference drove
        through it, `frozen` legitimately won, and the audit FLAGGED the
        defaults. The audit was right; the test was wrong.
      * ``[12, 5]`` — 5 m off-path, so NOTHING in the panel could reach it.
        `collision` was constant and the audit correctly returned INCONCLUSIVE.

    ``[40, 0]`` is the placement that actually discriminates: it is far enough
    ahead that the lawful 10 m/s reference (which reaches x=20) never gets
    there, while `bullet_straight` (30 m/s, x=60) and `teleport` drive straight
    through it. That is the real scene the reward must get right — *reckless
    speed meets an obstacle the careful policy never reaches.*
    """
    ref = AUD.reference_policy()
    return {"obstacles": torch.tensor([[40.0, 0.0]]),
            "gt_traj": ref,
            "lead_path": ref + torch.tensor([24.5, 0.0])}


def test_default_reward_is_not_flagged():
    """The other direction: a guard that flags everything is useless."""
    report = AUD.audit_reward(R.RewardSpec(), rich_ctx())
    assert not report.inconclusive, f"context did not exercise the reward:\n{report}"
    assert not report.flagged, f"default weights were flagged:\n{report}"
    assert report.verdict == "clean"


def test_information_free_context_is_INCONCLUSIVE_not_clean_and_not_flagged():
    """⛔ The three-state rule.

    With no obstacles, no lead and no GT, `collision`/`headway`/`gt_similarity`
    are identically zero, so progress is the only discriminating term and the
    fastest policy wins BY CONSTRUCTION. That is an underpowered audit, not a
    hackable reward — calling it either FLAGGED or clean would be a lie.
    """
    report = AUD.audit_reward(R.RewardSpec(), {})
    assert report.inconclusive
    assert report.verdict == "INCONCLUSIVE"
    assert set(report.dead_components) == {"collision", "headway"}
    assert report.flagged is False


def test_standing_still_earns_no_free_feasibility_or_comfort():
    """⛔ The motion gate — found by this audit on the library's own defaults.

    Without it, `frozen` collected 0.70 of free reward (feasibility 1.0 +
    comfort 1.0 at the default weights) for never moving.
    """
    frozen = AUD.degenerate_panel()["frozen"]
    assert float(R.COMPONENTS["feasibility"](frozen, {})) == pytest.approx(0.0)
    assert float(R.COMPONENTS["comfort"](frozen, {})) == pytest.approx(0.0)
    ref = AUD.reference_policy()
    assert float(R.COMPONENTS["feasibility"](ref, {})) == pytest.approx(1.0, abs=1e-3)
    assert float(R.COMPONENTS["comfort"](ref, {})) == pytest.approx(1.0, abs=1e-3)


def test_progress_only_beats_reference_by_construction():
    """Make the mechanism explicit, not just the verdict."""
    spec = R.RewardSpec(weights={"progress": 1.0})
    ref = float(spec(AUD.reference_policy(), {}))
    bullet = float(spec(AUD.degenerate_panel()["bullet_straight"], {}))
    assert bullet > ref


def test_collision_only_reward_is_hacked_by_standing_still():
    """A safety-only reward has its OWN degenerate: never move."""
    spec = R.RewardSpec(weights={"collision": 1.0})
    ctx = {"obstacles": torch.tensor([[8.0, 0.0]])}
    frozen = float(spec(AUD.degenerate_panel()["frozen"], ctx))
    ref = float(spec(AUD.reference_policy(), ctx))
    assert frozen > ref, "standing still never collides; driving through does"


# ---------------------------------------------------------------------------
# Panel + coverage
# ---------------------------------------------------------------------------

def test_panel_entries_are_wellformed():
    panel = AUD.degenerate_panel(21)
    assert set(panel) == {"bullet_straight", "frozen", "teleport", "spinner",
                          "shaky"}
    for name, traj in panel.items():
        assert traj.shape == (21, 2), name
        assert torch.isfinite(traj).all(), name


def test_teleport_and_spinner_are_actually_infeasible():
    """The panel must embody the failures it names, not merely be named."""
    panel = AUD.degenerate_panel(21)
    assert float(R.COMPONENTS["feasibility"](panel["teleport"], {})) < 0.2
    assert float(R.COMPONENTS["feasibility"](panel["spinner"], {})) < 0.7


def test_coverage_reports_a_component_that_never_fires():
    """A collision term with no obstacles carries NO signal — say so."""
    spec = R.RewardSpec()
    traj = AUD.reference_policy().expand(1, 3, 21, 2).contiguous()
    cov = AUD.report_component_coverage(spec, traj, {})
    assert cov["collision"]["fired"] is False
    assert "CONSTANT" in cov["collision"]["_note"]
    assert cov["collision"]["n"] == 3


def test_coverage_reports_a_component_that_does_fire():
    spec = R.RewardSpec()
    good = AUD.reference_policy()
    bad = AUD.degenerate_panel()["bullet_straight"]
    traj = torch.stack([good, bad]).unsqueeze(0)          # [1, 2, S, 2]
    cov = AUD.report_component_coverage(spec, traj,
                                        {"obstacles": torch.tensor([[25.0, 0.0]])})
    assert cov["collision"]["fired"] is True
    assert cov["collision"]["spread"] > 0


def test_reference_policy_alias_still_works_but_the_clear_name_exists():
    """⚠️ A trajectory and a policy are different objects.

    `audit.reference_policy` returns a TRAJECTORY; `anchor.ReferencePolicy` is a
    FROZEN MODEL. Exporting both under one name from one package is the
    conflation TRAIN-C5's vocabulary rule forbids. The alias keeps old callers
    working; the clear name is what new code uses.
    """
    import torch
    from tanitad.rl.anchor import ReferencePolicy
    assert AUD.reference_policy is AUD.sane_reference_trajectory
    assert isinstance(AUD.sane_reference_trajectory(), torch.Tensor)
    assert isinstance(ReferencePolicy, type)
