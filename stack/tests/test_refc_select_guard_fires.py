"""⛔ THE DEAD-PARAMETER GUARD MUST BE OBSERVED TO **FIRE**, not only to pass.

``tests/test_refc_select.py::test_rank_grafts_are_gated_not_dead`` pins the
HAPPY direction: with the ranked-score CE present, every D-SEL graft receives a
real gradient and ``assert_selection_params_are_alive`` returns. That is half a
control.

The other half is the one this programme keeps paying for. C13's class is *a
guard that is structurally unable to report the failure it is cited for*, and
the H2 "chance comparator" post-mortem is the same shape: a baseline nobody ever
checked against chance, which turned out to be **1.7259x chance** and biased
every null toward the desired verdict. **A guard never observed to raise is
indistinguishable from a guard that cannot raise.**

So this file removes the ranked-score CE and asserts the guard RAISES — the
exact condition ``compute_losses`` was changed to prevent, reproduced on demand.
That is what makes any future S1 result trustworthy: the instrument that would
have caught a dead graft has been shown to catch one.

⚠️ WHY ``REFINED_CLS_WEIGHT = 0`` IS THE RIGHT WAY TO REMOVE IT. ``loss_rcls``
is the ONLY term in REF-C's loss that differentiates w.r.t. ``sel_score``:
``traj`` and ``law_pred`` differentiate w.r.t. the FAN through a DETACHED index,
and ``argmax`` has no gradient. Zeroing its weight therefore reproduces exactly
the pre-D-SEL graph — a graft on the ranked score with nothing reading it —
without editing the model, so what is being tested is the GUARD and not a
hand-built fake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_train                                                   # noqa: E402
from tests.test_refc_select import _build                           # noqa: E402
from tests.test_refc import _batch, _make_cached_root               # noqa: E402


def _grads(model, batch, *, refined_cls_weight: float):
    """One forward + backward at a chosen ``REFINED_CLS_WEIGHT``."""
    old = refc_train.REFINED_CLS_WEIGHT
    refc_train.REFINED_CLS_WEIGHT = refined_cls_weight
    try:
        model.zero_grad(set_to_none=True)
        out = refc_train.compute_losses(model, batch)
        out["loss"].backward()
    finally:
        refc_train.REFINED_CLS_WEIGHT = old
    return out


def test_the_dead_parameter_guard_actually_fires(tmp_path):
    """PROOF BY CONTRAST on ONE model: the guard passes with the ranked-score CE
    and RAISES without it.

    Same weights, same batch, same flags — the ONLY difference is whether the
    ranked score is supervised. Anything less than this contrast leaves open
    that the guard passes for a reason unrelated to gradients.
    """
    root = _make_cached_root(tmp_path)
    m = _build(graft_cons=True, graft_route=True).train()
    batch = _batch(root, m.cfg)

    # --- (1) WITH the ranked-score CE: the guard must PASS, with real gradients
    _grads(m, batch, refined_cls_weight=refc_train.REFINED_CLS_WEIGHT)
    live = refc_train.assert_selection_params_are_alive(m)
    assert set(live) == {"decoder.route_to_anchor.weight", "decoder.cons_gate"}
    assert all(v > 0.0 for v in live.values()), live

    # --- (2) WITHOUT it: the SAME guard on the SAME model must RAISE ---------
    _grads(m, batch, refined_cls_weight=0.0)
    with pytest.raises(RuntimeError, match="received NO gradient") as exc:
        refc_train.assert_selection_params_are_alive(m)
    msg = str(exc.value)
    # the message must NAME the dead parameters and the mechanism, or a future
    # reader gets an alarm with no diagnosis
    assert "cons_gate" in msg and "route_to_anchor" in msg, msg
    assert "argmax" in msg and "loss_rcls" in msg, msg


def test_zero_init_is_gated_not_dead_and_the_weight_cannot_tell(tmp_path):
    """The distinction the guard exists to draw, stated as an assertion.

    ``cons_gate`` and ``route_to_anchor`` are zero-init on purpose, so their
    VALUES are identical in the live and the dead case. Only the gradient
    separates them. This pins that the weight really is uninformative here —
    otherwise someone will eventually "check the gate" by printing it.
    """
    root = _make_cached_root(tmp_path)
    m = _build(graft_cons=True, graft_route=True).train()
    batch = _batch(root, m.cfg)
    val = {n: p.detach().clone() for n, p in m.named_parameters()
           if "cons_gate" in n or "route_to_anchor" in n}
    assert val and all(float(v.abs().sum()) == 0.0 for v in val.values()), val

    _grads(m, batch, refined_cls_weight=0.0)
    dead_vals = {n: p.detach().clone() for n, p in m.named_parameters()
                 if n in val}
    _grads(m, batch, refined_cls_weight=refc_train.REFINED_CLS_WEIGHT)
    live_vals = {n: p.detach().clone() for n, p in m.named_parameters()
                 if n in val}
    # identical weights in BOTH cases (no optimizer step was taken) ...
    for n in val:
        assert torch.equal(dead_vals[n], live_vals[n])
        assert float(live_vals[n].abs().sum()) == 0.0
    # ... and yet one has gradient and the other does not.
    live = refc_train.assert_selection_params_are_alive(m)
    assert all(v > 0.0 for v in live.values()), live


def test_s1_alone_adds_no_parameter_for_the_guard_to_check(tmp_path):
    """⚠️ SCOPE OF THE GUARD, stated so it is not over-read.

    ``assert_selection_params_are_alive`` scans ``route_to_anchor``,
    ``cons_gate``, ``goal_gate``, ``goal_dist_gate`` and ``goal_head``. **S1 adds
    ZERO parameters**, so on a ``--sel-refined``-only arm the guard has nothing
    to check and returns an EMPTY dict — which is correct, and is NOT evidence
    that S1 is alive.

    What keeps S1 honest instead is that ``loss_rcls`` is built whenever
    ``sel_refined`` is set, and it is a CE on ``sel_score``, i.e. on the refined
    readout S1 ranks with. This test pins both halves so nobody later reads an
    empty ``d_sel_gradients`` banner as a failure — or as a pass.
    """
    root = _make_cached_root(tmp_path)
    m = _build(sel_refined=True).train()
    batch = _batch(root, m.cfg)
    out = _grads(m, batch, refined_cls_weight=refc_train.REFINED_CLS_WEIGHT)
    assert refc_train.assert_selection_params_are_alive(m) == {}
    # the S1 supervision term exists and is a real, finite, positive CE
    assert "cls_refined" in out, sorted(out)
    assert float(out["cls_refined"]) > 0.0 and torch.isfinite(out["cls_refined"])
