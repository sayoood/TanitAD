"""`map_iou_drivable` is a QUALITY read the soft CE cannot give, and it is always logged.

⛔ WHY IT EXISTS. `map` is a soft cross-entropy: it falls monotonically while telling
nobody whether the **drivable mask** is right, and a collision gate needs exactly that
mask. MEASURED 2026-09-18, the no-information floor -- predict drivable everywhere -- is
IoU **0.3412** over **10,068,274** seen cells (`…/2026-09-18-occupancy-floor/`), so:

* an IoU **below 0.3412** is worse than a constant predictor;
* an IoU **near 0.3412** is **prevalence**, not skill;
* only an IoU clearly above it is evidence the head learned anything.

⭐ **ALWAYS LOGGED, never behind a flag.** The D9 DAC term was identically 1 for 600 steps
across 3 arms and left **no trace**, because it was not in the telemetry key list. A
quality metric that only appears when someone remembers to ask for it is the same defect
waiting to happen.

⛔ **Thresholded the SAME way on both sides** -- `>= 0.5` on the GT fraction and on the
softmax probability. Scoring a prediction by argmax against a GT scored by threshold is
two different rules, and the disagreement between them would read as model error.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.data import semantic_map_gt as SMG              # noqa: E402

SRC = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
DCH = SMG.CHANNELS.index("drivable")
H, W = SMG.CART_SHAPE
C = SMG.N_CHANNELS


def iou(logits, frac, seen):
    """The trainer's computation, restated here so the test owns an independent copy."""
    gt = (frac[:, DCH] >= 0.5) & seen
    pr = (logits.softmax(dim=1)[:, DCH] >= 0.5) & seen
    u = float((gt | pr).sum())
    return (float((gt & pr).sum()) / u) if u else 0.0


def _mk(pred_drivable: bool, gt_drivable: bool, seen_all: bool = True):
    logits = torch.full((1, C, H, W), -10.0)
    logits[:, DCH] = 10.0 if pred_drivable else -10.0
    if not pred_drivable:                 # put the mass somewhere else, legally
        logits[:, (DCH + 1) % C] = 10.0
    frac = torch.zeros((1, C, H, W))
    frac[:, DCH] = 1.0 if gt_drivable else 0.0
    seen = torch.ones((1, H, W), dtype=torch.bool) if seen_all else \
        torch.zeros((1, H, W), dtype=torch.bool)
    return logits, frac, seen


def test_perfect_prediction_reads_exactly_1():
    assert iou(*_mk(True, True)) == pytest.approx(1.0)


def test_completely_wrong_prediction_reads_exactly_0():
    assert iou(*_mk(True, False)) == pytest.approx(0.0)
    assert iou(*_mk(False, True)) == pytest.approx(0.0)


def test_an_all_unseen_window_reads_0_and_does_not_divide_by_zero():
    """⛔ An unseen cell carries no evidence, so the union is empty. The value must be
    a counted 0.0, never a NaN and never a crash."""
    v = iou(*_mk(True, True, seen_all=False))
    assert v == 0.0 and v == v          # the second clause rejects NaN


def test_iou_equals_prevalence_when_the_head_predicts_drivable_EVERYWHERE():
    """⭐ The floor, reproduced analytically: predicting drivable everywhere scores
    exactly `P(drivable | seen)` -- which is why 0.3412 on the real corpus is the
    no-information value and not an achievement."""
    logits = torch.full((1, C, H, W), -10.0)
    logits[:, DCH] = 10.0                       # predict drivable everywhere
    frac = torch.zeros((1, C, H, W))
    frac[:, DCH, : H // 3] = 1.0                # GT drivable on exactly one third
    seen = torch.ones((1, H, W), dtype=torch.bool)
    assert iou(logits, frac, seen) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_both_sides_use_THE_SAME_threshold_rule():
    """⛔ GT at 0.49 is NOT drivable; a prediction at 0.49 is NOT drivable either."""
    logits = torch.zeros((1, C, H, W))          # uniform softmax = 1/C < 0.5 -> not drivable
    frac = torch.zeros((1, C, H, W))
    frac[:, DCH] = 0.49                         # just under the GT threshold
    seen = torch.ones((1, H, W), dtype=torch.bool)
    assert iou(logits, frac, seen) == 0.0, (
        "neither side crosses 0.5, so the union is empty and the score is a counted 0")


def test_the_TRAINER_uses_the_same_threshold_rule_as_the_GT():
    """⛔ THE ARM THAT CATCHES A RULE SWAP, and the first version of this file lacked it.

    Every test above restates the computation locally, which pins the DEFINITION and is
    **blind to what the trainer actually does**. MEASURED by mutation 2026-09-18:
    changing the trainer to `argmax(dim=1) == _dch` while the GT stayed `>= 0.5` left
    all six arms green. ⇒ **a test that owns a private copy of the computation tests the
    copy.**

    Argmax and threshold are not the same rule: with 9 channels a cell can be argmax-
    drivable at probability 0.2, which the GT side -- a hard `>= 0.5` on the fraction --
    would never call drivable. The disagreement would read as model error.
    """
    assert ".softmax(dim=1)[:, _dch] >= 0.5" in SRC, (
        "the trainer must threshold the PROBABILITY at 0.5, the same rule the GT "
        "fraction is thresholded with; argmax is a different rule and the mismatch "
        "would be scored as model error")
    assert "[:, _dch]\n                           >= 0.5) & _sn" in SRC or \
           "_dch] >= 0.5) & _sn" in SRC, "the GT side must use the same >= 0.5 rule"


def test_the_trainer_logs_it_unconditionally_next_to_the_loss():
    """⚠️ The lesson the D9 DAC term earned: a sub-score that is not logged cannot be
    seen to be wrong. This metric must sit in the same block as `map`, not behind a flag."""
    assert 'extra["map_iou_drivable"]' in SRC
    assert 'extra["map_gt_drivable_frac"]' in SRC, (
        "the GT prevalence must be logged beside the IoU, or a reader cannot tell "
        "skill from prevalence on that particular batch")
    i = SRC.index('extra["n_map_cells"] = _mrow["n_map_cells"]')
    j = SRC.index('extra["map_iou_drivable"]')
    assert 0 < j - i < 2500, (
        "map_iou_drivable must be computed in the same block as the loss, from the "
        "same tensors -- that adjacency is what makes `map` the instrument check")


def test_a_threshold_free_companion_is_logged_because_IoU_is_blind_to_under_confidence():
    """⚠️ MEASURED 2026-09-18: at ImageNet init the 9-way softmax is ~1/9 everywhere,
    nothing crosses 0.5, and `map_iou_drivable` reads EXACTLY 0.0.

    That is correct -- the head calls no cell majority-drivable -- but it would read
    0.0 just the same for a head that had learned the shape and stayed at 0.45. The
    mean predicted drivable probability on seen cells moves CONTINUOUSLY toward the
    GT's ~0.35, so it shows progress the IoU cannot.
    ⇒ The two answer different questions and must both be logged: the IoU says
    "usable by a gate yet?", the mean says "is it learning at all?".

    ⚠️ MEASURED, not assumed: after 1 step it reads **0.0717** (train) / **0.0728**
    (eval), NOT the 1/9 = 0.1111 a uniform softmax would give. A randomly initialised
    head does not produce equal logits, so 1/9 is the expectation of a uniform draw and
    not the value of any particular one. Quoting 0.111 as "the init value" would have
    been theory standing in for a measurement.
    """
    assert 'extra["map_pred_drivable_prob_mean"]' in SRC
    i = SRC.index('extra["map_iou_drivable"]')
    j = SRC.index('extra["map_pred_drivable_prob_mean"]')
    assert 0 < j - i < 2500, "both must live in the same block, from the same tensors"


def test_the_untrained_floor_is_the_DEGENERATE_one_not_the_prevalence_one():
    """⭐ The reading this metric earned, pinned as arithmetic.

    A uniform 9-way softmax puts 1/9 = 0.111 on every channel, which is below the 0.5
    threshold, so an untrained head predicts NO cell drivable -> IoU 0.0. It therefore
    starts at the DEGENERATE floor (0.0), not the no-information prevalence floor
    (0.3412 corpus-wide). Any training must move it off exactly zero.
    """
    C_ = SMG.N_CHANNELS
    assert 1.0 / C_ < 0.5, (
        f"with {C_} channels a uniform softmax is {1.0/C_:.4f}, below the 0.5 "
        "threshold -- if this ever fails the untrained-floor reasoning changes")
