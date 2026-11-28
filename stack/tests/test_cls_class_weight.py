"""`slot_set_loss(cls_class_weight=…)` — the capability `H-BOXCLS-1` needs, pinned by MUTATION.

⛔ WHY IT EXISTS. `2adadda`/`731ecd7` measured a TOTAL class collapse on BOTH agent heads — one
class of ten on 2,000/2,000 and 320/320 slots, top-1 accuracy exactly the majority baseline, 188
`person` and 50 `rider` emitted as cars — against an UNWEIGHTED `cross_entropy`
(`agent_slots.py`) on a **577.5 : 1** target, while `presence` gets `NO_OBJECT_W = 0.1` for exactly
that reason. `d7fa093`'s pre-registered gate then showed the signal IS linearly readable from the
head's own features, so the collapse is an OBJECTIVE failure. ⇒ the arm needs a class-weighted
`cls` term, and **no such option existed** — so the arm could not have launched whatever the PI
decided.

⛔ WHY NOT THE FOCAL LOSS WE ALREADY HAVE. `refcv6_diffusion.focal_cls_loss` exists and is tempting
under the "already exists, never re-implement" rule. It is the WRONG tool here, and its own
docstring says why: it is a **sigmoid** loss on `[B, N]` mode logits, ported from DD, and *"swapping
in cross_entropy reproduces neither the gradient nor the scale"*. The slot `cls` term is a 10-way
**softmax**. Using it would change the loss FAMILY, not the weighting — a second variable inside a
one-variable arm.

⭐ THE SUBTLE PART IS THE DENOMINATOR, AND IT IS WHAT MOST OF THESE TESTS PIN.
`cross_entropy(reduction="sum", weight=w)` returns `Σ wᵢ·Lᵢ`, so dividing by the item COUNT would
make the term's SCALE move with the weights as well as its per-class emphasis — the `cls` term
would then compete differently against `centre`, `size` and the rest for reasons nobody asked for.
The normaliser is therefore `Σ wᵢ`, which buys the identity control for free: at `w = ones` the
weight sum IS the count, so a uniform weight must be **bit-identical** to passing nothing.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.models.agent_slots import AGENT_CLASSES, slot_set_loss

K = len(AGENT_CLASSES)


def _case():
    """A DETERMINISTIC target. ⚠️ An earlier hand-check used `randint` over 12 slots and a
    non-uniform weight changed nothing — class 0 simply never appeared in the draw, which reads
    exactly like a dead weight path. The class counts are asserted below so this test can never
    pass for that reason."""
    torch.manual_seed(0)
    B, N = 2, 6
    pred = {"cls_logits": torch.randn(B, N, K),
            "presence_logit": torch.randn(B, N),
            "box": torch.randn(B, N, 4).abs(),
            "yaw_vec": torch.nn.functional.normalize(torch.randn(B, N, 2), dim=-1),
            "rates": torch.randn(B, N, 3), "occ_logit": torch.randn(B, N)}
    cls = torch.tensor([[0, 0, 0, 0, 5, 5], [0, 0, 0, 5, 5, 9]])
    tgt = {"box": torch.randn(B, N, 4).abs(), "yaw": torch.randn(B, N), "cls": cls,
           "valid": torch.ones(B, N, dtype=torch.bool), "occ": torch.zeros(B, N),
           "rates": torch.randn(B, N, 3),
           "rates_mask": torch.ones(B, N, dtype=torch.bool)}
    return pred, tgt


def test_the_fixture_actually_contains_the_classes_the_tests_reweight():
    """⛔ THE SAME-BREATH CONTROL. Without it, 'a non-uniform weight changes the loss' can fail to
    fire because the class is ABSENT, and the test would read as coverage of a dead path."""
    _, tgt = _case()
    counts = torch.bincount(tgt["cls"].flatten(), minlength=K)
    assert int(counts[0]) == 7 and int(counts[5]) == 4 and int(counts[9]) == 1, counts.tolist()


def test_uniform_weight_is_BIT_IDENTICAL_to_passing_nothing():
    """The identity control, and the proof that the denominator follows the weights."""
    pred, tgt = _case()
    a = slot_set_loss(pred, tgt)["loss_cls"]
    b = slot_set_loss(pred, tgt, cls_class_weight=torch.ones(K))["loss_cls"]
    assert torch.allclose(a, b, atol=0.0, rtol=0.0), (float(a), float(b))


@pytest.mark.parametrize("c", [0.1, 3.7, 100.0])
def test_a_CONSTANT_weight_changes_nothing_at_any_scale(c):
    """⛔ THE SCALE CONTROL. If the normaliser were the item count, a constant weight would scale
    the whole term by `c` — re-weighting would then be TWO changes, and any arm using it would be
    confounded between 'the classes were re-balanced' and 'the cls term got louder'."""
    pred, tgt = _case()
    a = slot_set_loss(pred, tgt)["loss_cls"]
    b = slot_set_loss(pred, tgt, cls_class_weight=torch.full((K,), c))["loss_cls"]
    assert torch.allclose(a, b, atol=1e-5), (c, float(a), float(b))


@pytest.mark.parametrize("idx, w, expect", [(0, 0.1, 2.025816), (5, 20.0, 2.227341)])
def test_a_NON_uniform_weight_moves_the_loss_to_a_LITERAL(idx, w, expect):
    """⛔ EXPECTATIONS ARE LITERALS, never expressions over the code under test: recomputing the
    weighted mean here with the same formula the loss uses would be green for any formula."""
    pred, tgt = _case()
    base = float(slot_set_loss(pred, tgt)["loss_cls"])
    vec = torch.ones(K)
    vec[idx] = w
    got = float(slot_set_loss(pred, tgt, cls_class_weight=vec)["loss_cls"])
    assert got != pytest.approx(base, abs=1e-5), "the weight vector did not reach the loss"
    assert got == pytest.approx(expect, abs=1e-4), (got, expect)


def test_box3d_set_loss_FORWARDS_it():
    """⛔ DECLARED IS NOT PLUMBED — and the 3-D head is the one `s1_pass` scores."""
    from tanitad.models.box3d_head import box3d_set_loss
    pred, tgt = _case()
    B, N = pred["cls_logits"].shape[:2]
    pred = {**pred, "cz": torch.randn(B, N), "h": torch.randn(B, N).abs()}
    tgt = {**tgt, "cz": torch.randn(B, N), "h": torch.randn(B, N).abs(),
           "zh_mask": torch.ones(B, N, dtype=torch.bool)}
    vec = torch.ones(K)
    vec[0] = 0.1
    a = float(box3d_set_loss(pred, tgt)["loss_cls"])
    b = float(box3d_set_loss(pred, tgt, cls_class_weight=vec)["loss_cls"])
    assert a != pytest.approx(b, abs=1e-5), "box3d_set_loss dropped cls_class_weight on the floor"


def test_agent_losses_FORWARDS_it():
    from tanitad.refs.refc_agents import AgentSeamConfig, agent_losses
    pred, tgt = _case()
    cfg = AgentSeamConfig(enable=True)
    vec = torch.ones(K)
    vec[0] = 0.1
    a = float(agent_losses(pred, tgt, cfg, filter_visible=False)["loss_cls"])
    b = float(agent_losses(pred, tgt, cfg, filter_visible=False,
                           cls_class_weight=vec)["loss_cls"])
    assert a != pytest.approx(b, abs=1e-5), "agent_losses dropped cls_class_weight on the floor"
