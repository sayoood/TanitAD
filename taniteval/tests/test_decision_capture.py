"""TACTICAL/STRATEGIC decision capture — closing the two-family hole.

⛔ WHY THIS EXISTS. ``four_families.tactical`` / ``strategic`` return
``status: UNAVAILABLE`` unless the window dict carries the model's DECODED
DECISIONS, and ``rollout.collect`` is a world-model FIDELITY pass that never
traverses the decision heads. So an eval built on the standard collect comes
back TWO FAMILIES SHORT — inadmissible under the binding four-family rule
(Sayed, 2026-08-02), and short in a way that is easy to miss because the ADE
columns all look fine.

``decision_fn`` is the seam. A loader that can expose its manoeuvre/route heads
supplies one; everything else is unchanged and still reports UNAVAILABLE **with
its reason and n**, which is the contract — never a silent omission.

⚠️ The GT label is computed from the ego's OWN FUTURE poses. That is allowed:
LABELS MAY USE EGO, INFERENCE IS VISION-ONLY. The guard is on what the model is
FED, not on how the target is derived.
"""
from __future__ import annotations

import torch

from taniteval import four_families as ff
from taniteval import rollout


# --- the label ------------------------------------------------------------- #

def test_gt_label_reuses_the_training_rule_not_a_reimplementation():
    """If the eval re-derived the manoeuvre rule, the eval label and the
    training label could drift apart silently. Pin that it calls the same
    classifier ``refb_labels`` uses."""
    import refb_labels as rl
    poses = torch.zeros(60, 4)
    poses[:, 3] = 12.0
    poses[:, 0] = torch.arange(60, dtype=torch.float32)
    last = torch.tensor([0, 10, 20])
    got = rollout._man_gt(poses, last)
    h = rl.LABEL_HORIZON
    want = rl.classify_maneuver(poses[last][:, 2], poses[last + h][:, 2],
                                poses[last][:, 3], poses[last + h][:, 3])
    assert torch.equal(got, want)


def test_gt_label_clamps_instead_of_indexing_past_the_episode():
    """A window near the tail has no full horizon. Clamping is the same
    degradation ``maneuver_labels`` applies there; an IndexError would take the
    whole eval down for a handful of windows."""
    poses = torch.zeros(30, 4)
    poses[:, 3] = 8.0
    got = rollout._man_gt(poses, torch.tensor([25, 29]))
    assert got.shape == (2,)


def test_gt_label_separates_a_turn_from_lane_keeping():
    """A label that returned one class for everything would satisfy the shape
    tests above while measuring nothing."""
    straight = torch.zeros(60, 4)
    straight[:, 3] = 12.0
    straight[:, 0] = torch.arange(60, dtype=torch.float32)
    turning = straight.clone()
    turning[:, 2] = torch.linspace(0.0, 1.5, 60)      # yaw sweeps left
    last = torch.tensor([0])
    assert not torch.equal(rollout._man_gt(straight, last),
                           rollout._man_gt(turning, last))


# --- the contract when NO decision_fn is supplied --------------------------- #

def test_absent_decisions_report_UNAVAILABLE_WITH_A_REASON():
    """⛔ The failure this rule exists to prevent is a family SILENTLY missing.
    Absence must be loud, and must say what to do about it."""
    fam = ff.tactical({"pred": torch.zeros(4, 4, 2)})
    assert fam["status"] == "UNAVAILABLE"
    assert fam["n"] == 0
    assert "maneuver_pred" in fam["reason"] or "maneuver_gt" in fam["reason"]
    assert "WORK ITEM" in fam["reason"]


# --- the contract when decisions ARE present -------------------------------- #

def test_present_decisions_populate_the_tactical_family():
    win = {"maneuver_pred": torch.tensor([0, 1, 2, 0, 0]),
           "maneuver_gt": torch.tensor([0, 1, 0, 0, 4])}
    fam = ff.tactical(win)
    assert fam["status"] == "OK"
    assert fam["n"] == 5
    assert fam["accuracy"] == 0.6                      # 3 of 5
    assert "lane_keep" in fam["per_class"]


def test_a_never_predicted_class_is_surfaced_not_averaged_away():
    """THE reason this family exists. Our measured defect is a 5-way softmax
    that emitted 0 of 881 'accelerate' decisions — an accuracy scalar hides a
    class the model never chooses; ``never_predicted`` does not."""
    win = {"maneuver_pred": torch.tensor([0, 0, 0, 0]),
           "maneuver_gt": torch.tensor([0, 0, 0, 3])}   # 3 == accelerate
    fam = ff.tactical(win)
    assert fam["accuracy"] == 0.75                      # looks healthy
    assert "accelerate" in fam["never_predicted"]       # ...and it is not


# --- the seam itself -------------------------------------------------------- #

def test_collect_signature_carries_the_seam_and_defaults_to_off():
    import inspect
    sig = inspect.signature(rollout.collect)
    assert "decision_fn" in sig.parameters
    assert sig.parameters["decision_fn"].default is None, (
        "decision capture must be OPT-IN: defaulting it on would change every "
        "existing caller's returned dict")
