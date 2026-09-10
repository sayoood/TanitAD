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


# ---------------------------------------------------------------------------
# THE SEPARATION CHECK — liveness is not gameability (added 2026-09-11)
# ---------------------------------------------------------------------------
# ⛔⛔ THE DEFECT, MEASURED, NOT ARGUED. P2 of the 2,000-step pilot runs
# `HACKABLE_WEIGHTS` deliberately as the regression control, at the pilot's
# `dt = 0.5`. `audit_reward` returned **verdict: clean, flagged: False,
# dead_components: []** while its own scores read `bullet_straight` 1.5,
# `teleport` 1.5, `shaky` 1.5 -- three distinct degenerate policies pinned at the
# identical value -- against `spinner` -0.0104 and `frozen` 0.0.
#
# ⚠️ SCOPE IT HONESTLY: `clean` was CORRECT under the old definition. The
# `dead_components` half reports component LIVENESS, and nothing was constant
# (`progress` spans -0.0104 to 1.5). The gap is that **liveness is not
# gameability**: `progress` is clamped at `hi = 1.5`, so at dt 0.5 every policy
# fast enough SATURATES and the ranking collapses while the term still "varies".
#
# ⛔ EVERY EXPECTATION BELOW IS A LITERAL. Re-deriving the panel's own scores
# and finding agreement would measure determinism, not correctness -- the
# "a check that shares the defect it checks for is green forever" class.


def test_MEASURED_the_hackable_reward_at_the_pilots_dt_TIES_THREE_POLICIES():
    """The raw fact the separation check is built on, pinned as literals.

    ⛔ These five numbers are the ones the pilot's own P2 run recorded; if any of
    them moves, the finding underneath the check has moved and the check must be
    re-argued rather than silently kept.
    """
    spec = R.RewardSpec(weights=dict(R.HACKABLE_WEIGHTS), dt=0.5)
    report = AUD.audit_reward(spec)
    assert report.scores["bullet_straight"] == pytest.approx(1.5, abs=1e-9)
    assert report.scores["teleport"] == pytest.approx(1.5, abs=1e-9)
    assert report.scores["shaky"] == pytest.approx(1.5, abs=1e-9)
    assert report.scores["frozen"] == pytest.approx(0.0, abs=1e-9)
    assert report.scores["spinner"] == pytest.approx(-0.01041, abs=1e-5)
    assert report.reference == pytest.approx(1.5, abs=1e-9)
    assert report.dead_components == [], (
        "the LIVENESS half must still read clean here -- that is the whole "
        "point: the old verdict was not wrong, it was answering a narrower "
        "question")


def test_DELIBERATE_REGRESSION_the_hackable_reward_at_dt_0p5_IS_NOW_FLAGGED():
    """⛔ THE LOAD-BEARING TEST FOR THIS CHECK. This exact spec returned `clean`
    on 2026-09-10 and every clean verdict it gave was therefore worth less than
    it looked. It must now be FLAGGED, and for the SEPARATION reason -- not by
    accident through the winners path."""
    spec = R.RewardSpec(weights=dict(R.HACKABLE_WEIGHTS), dt=0.5)
    report = AUD.audit_reward(spec)
    assert report.verdict == "FLAGGED", (
        "the progress-only reward ties three degenerate policies at the ceiling "
        f"and was still called {report.verdict}\n{report}")
    assert report.flagged is True
    assert report.tied_at_max == ["bullet_straight", "shaky", "teleport"], (
        f"expected the three ceiling-tied policies, got {report.tied_at_max}")
    assert "NOT SEPARABLE" in report.reason
    assert report.winners == [], (
        "nothing strictly beats the reference here -- so the winners path CANNOT "
        "be what flagged it, and this assertion is what proves the separation "
        "check is doing the work")


def test_the_default_reward_STAYS_GREEN_because_only_ONE_policy_tops_the_panel():
    """⛔ THE CONTROL, in both contexts the programme actually audits in. A
    separation check that fires on the default reward would be a blanket refusal
    and would get itself deleted.

    MEASURED: the default's panel maximum is `bullet_straight` ALONE (1.45 at
    both dt values, empty ctx) and `spinner` ALONE under `rich_ctx()`. One entry
    is not a tie."""
    for dt in (R.DT_S, 0.5):
        rep = AUD.audit_reward(R.RewardSpec(weights=dict(R.DEFAULT_WEIGHTS), dt=dt))
        assert rep.tied_at_max == ["bullet_straight"], (dt, rep.tied_at_max)
        assert rep.verdict == "INCONCLUSIVE", (
            f"dt={dt}: the empty-context verdict must stay INCONCLUSIVE "
            f"(collision/headway dead), got {rep.verdict}\n{rep}")

    rich = AUD.audit_reward(R.RewardSpec(), rich_ctx())
    assert rich.tied_at_max == ["spinner"], rich.tied_at_max
    assert rich.verdict == "clean", (
        f"the default reward on an exercising context must stay green\n{rich}")


def test_a_single_policy_tying_the_REFERENCE_is_not_a_separation_failure():
    """⚠️ THE BOUNDARY, and it is deliberate. MEASURED: the default reward's
    `bullet_straight` scores EXACTLY the reference's 1.45 at dt 0.5. A single
    degenerate tying the reference is the `margin` policy's business ("ties are
    not failures, strictly beating is"); this check is about the reward being
    unable to rank TWO PROBES apart, which no margin can express."""
    rep = AUD.audit_reward(R.RewardSpec(weights=dict(R.DEFAULT_WEIGHTS), dt=0.5))
    assert rep.scores["bullet_straight"] == pytest.approx(rep.reference, abs=1e-9)
    assert len(rep.tied_at_max) == 1
    assert rep.verdict != "FLAGGED"


def test_the_separation_branch_is_REACHABLE_and_is_WHAT_FLIPS_THE_VERDICT():
    """⭐ AN ISOLATION MUTATION, not an inspection.

    `rich_ctx()` + the default reward is the programme's one GREEN audit: no
    winner, no dead component, one entry at the panel maximum. Widening ONLY the
    tie tolerance -- changing no reward, no context, no panel -- must flip it to
    FLAGGED. ⛔ Neither the winners branch nor the dead-component branch can
    respond to `tie_atol`, so reaching FLAGGED here isolates the separation check
    as the cause. A check that could not be made to fire is not a check.
    """
    green = AUD.audit_reward(R.RewardSpec(), rich_ctx())
    assert green.verdict == "clean"
    assert green.winners == []
    assert green.dead_components == []

    fired = AUD.audit_reward(R.RewardSpec(), rich_ctx(), tie_atol=1.0)
    assert fired.verdict == "FLAGGED", (
        "widening the tie tolerance did not reach the separation branch at "
        f"all: {fired}")
    assert "NOT SEPARABLE" in fired.reason
    assert fired.winners == [] and fired.dead_components == [], (
        "the other two branches must still be silent, or this does not isolate "
        f"anything: winners={fired.winners} dead={fired.dead_components}")
    assert len(fired.tied_at_max) == 5


def test_tied_at_max_is_reported_on_every_branch():
    """The field is evidence, so it must exist whatever the verdict is."""
    for spec, ctx in ((R.RewardSpec(), None),
                      (R.RewardSpec(), rich_ctx()),
                      (R.RewardSpec(weights=dict(R.HACKABLE_WEIGHTS), dt=0.5), None)):
        rep = AUD.audit_reward(spec, ctx)
        assert isinstance(rep.tied_at_max, list) and rep.tied_at_max, rep.verdict


# ---------------------------------------------------------------------------
# THE EXERCISING CONTEXT -- what P1's own INCONCLUSIVE verdict asked for
# ---------------------------------------------------------------------------

def test_exercising_ctx_REPRODUCES_the_hand_found_literals_at_the_default_dt():
    """⭐ THE INDEPENDENT CROSS-CHECK. `exercising_ctx` derives its geometry
    from the PANEL'S OWN SPEEDS (midpoint between the reference's reach and
    bullet_straight's) and from `_headway`'s own standoff (lead_len + t* * v).
    At the module default dt it must land EXACTLY on the two numbers that were
    found by hand, months earlier, in `rich_ctx` above -- which is evidence,
    where re-running the producer's arithmetic would only measure determinism.
    """
    ctx = AUD.exercising_ctx(21, R.DT_S)
    assert ctx["obstacles"].tolist() == [[40.0, 0.0]]
    assert float(ctx["lead_path"][0, 0] - ctx["gt_traj"][0, 0]) == pytest.approx(24.5)
    hand = rich_ctx()
    assert torch.allclose(ctx["obstacles"], hand["obstacles"])
    assert torch.allclose(ctx["lead_path"], hand["lead_path"])
    assert torch.allclose(ctx["gt_traj"], hand["gt_traj"])


def test_exercising_ctx_SCALES_with_dt_and_the_banked_literal_would_not():
    """⚠️ THE MISTAKE THIS PREVENTS, stated as a literal. The panel is built
    from `v * dt * t`, so at the pilot's dt = 0.5 the lawful reference reaches
    100 m -- and the banked 40.0 would sit UNDER it, which is historical wrong
    placement #1 (frozen legitimately wins and the audit flags the defaults)."""
    ctx = AUD.exercising_ctx(21, 0.5)
    assert ctx["obstacles"].tolist() == [[200.0, 0.0]]
    ref_reach = float(AUD.sane_reference_trajectory(21, 0.5)[-1, 0])
    assert ref_reach == pytest.approx(100.0)
    assert float(ctx["obstacles"][0, 0]) > ref_reach, (
        "the obstacle must be beyond the careful policy's reach or `frozen` wins "
        "for the right reason and the audit flags a reward that is fine")


def test_the_exercising_ctx_makes_the_DEFAULT_reward_RULABLE_at_the_pilots_dt():
    """⛔ THE POINT: P1 could not be judged at all. MEASURED on the 2,000-step
    arm -- `dead_components: ['collision', 'headway']`, verdict INCONCLUSIVE. On
    this context, at the SAME dt = 0.5, the audit can rule, and it reads clean."""
    spec = R.RewardSpec(weights=dict(R.DEFAULT_WEIGHTS), dt=0.5)
    empty = AUD.audit_reward(spec)
    assert empty.verdict == "INCONCLUSIVE"
    assert empty.dead_components == ["collision", "headway"]

    exc = AUD.audit_reward(spec, AUD.exercising_ctx(dt=0.5))
    assert exc.dead_components == [], (
        "collision/headway STILL cannot fire -- the context does not exercise "
        f"them: {exc}")
    assert exc.verdict == "clean", exc


def test_the_exercising_ctx_STILL_flags_the_hackable_reward_at_both_dt():
    """⛔ The regression arm must survive the new context, or the context is
    doing the audit's job for it."""
    for dt in (R.DT_S, 0.5):
        rep = AUD.audit_reward(R.RewardSpec(weights=dict(R.HACKABLE_WEIGHTS), dt=dt),
                               AUD.exercising_ctx(dt=dt))
        assert rep.verdict == "FLAGGED", (dt, rep)
        assert rep.dead_components == [], (dt, rep.dead_components)
