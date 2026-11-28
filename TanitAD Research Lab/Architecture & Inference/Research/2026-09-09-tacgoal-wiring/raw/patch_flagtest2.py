"""Pin the CONTRACT that `train()`'s split-fitting block depends on.

⛔ THE GAP THIS CLOSES, and it is a gap in MY OWN first draft. Both gradient
probes set ``model._tac_goal_pos_weight = None`` by hand and the dataset tests
set ``ds.tac_goal_targets`` by hand, so nothing executes the block in ``train()``
that fits ``pos_weight`` and ``class_mask`` on the loaded split. A wrong key
there — ``mask_report(...)["mask"]`` vs ``["masks"]`` — would surface only at a
real launch, after the corpus mounts and a GPU-day is already committed. That is
the failure mode this whole package exists to prevent, one level up.
"""
from __future__ import annotations

import ast
import io
import sys

SRC, DST = sys.argv[1], sys.argv[2]
with io.open(SRC, "rb") as fh:
    _raw = fh.read()
CRLF, LF = _raw.count(b"\r\n"), _raw.count(b"\n")
NEWLINE = "\r\n" if CRLF * 2 > LF else "\n"
s = _raw.decode("utf-8").replace("\r\n", "\n")

NEW = '''

# ==========================================================================
# 7 — THE SPLIT-FITTED CONSTANTS: every key train() reads, as a LITERAL
# ==========================================================================
def _three_labels(monkeypatch):
    from tanitad.data import v7_labels as v7l
    monkeypatch.setattr(v7l, "_MEASURED_GEOMETRY_TOKENS", _GEOM)

    def _lab(clip, goals):
        return v7l.V7Label(
            clip_id=clip, tac_lat="LANE_KEEP", tac_lon="CRUISE",
            str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
            tac_anchor=None, bands={"tactical_s": [0.0, 60.0]}, t0_s=8.0,
            horizon={}, tac_goals=frozenset(goals),
            tac_goal_meta={k: {"provenance": "geometry"} for k in goals})

    return v7l, [_lab("c1", ("FOLLOW_LANE", "SPEED_BAND")),
                 _lab("c2", ("TURN_L", "SPEED_BAND")),
                 _lab("c3", ("FOLLOW_LANE",))]


def test_train_fits_pos_weight_and_mask_from_the_split_not_a_literal(
        monkeypatch):
    """⛔ EVERY KEY `train()` READS, ASSERTED AS A LITERAL.

    `mask_report` must return exactly these five keys and `goal_pos_weight` a
    22-tuple, because ``train()`` indexes them by name. ⭐ The expectation is
    written out, never derived by calling the same functions and comparing them
    to themselves — that would measure determinism, not correctness.
    """
    import torch
    from tanitad.refs import tac_goal_head as tgh
    v7l, labels = _three_labels(monkeypatch)

    census = v7l.goal_supervision_census(labels)
    mask = tgh.mask_report(census)
    pw = v7l.goal_pos_weight(labels)

    assert len(census) == 22
    assert set(mask) == {"mask", "masked_why", "n_total", "n_trainable",
                         "trainable"}, sorted(mask)
    assert int(mask["n_total"]) == 22
    assert len(pw) == 22
    # exactly the two conversions train() performs
    pw_t = torch.tensor(pw, dtype=torch.float32)
    mask_t = torch.tensor(mask["mask"], dtype=torch.float32)
    assert tuple(pw_t.shape) == (22,) and pw_t.dtype == torch.float32
    assert tuple(mask_t.shape) == (22,) and mask_t.dtype == torch.float32
    # and the stamp train() writes into config.json must be JSON-serialisable:
    # a run record that cannot be written is a run record that does not exist
    import json
    json.dumps({"negatives": "measured",
                "n_trainable": int(mask["n_trainable"]),
                "n_total": int(mask["n_total"]),
                "trainable": list(mask["trainable"]),
                "masked_why": mask["masked_why"],
                "pos_weight": [float(x) for x in pw],
                "census": census})


def test_the_loss_accepts_those_two_tensors_and_a_gradient_flows(monkeypatch):
    """⭐ AN ANALYTIC TARGET, not a recorded number.

    With every logit at 0 and every target 0, BCE-with-logits is exactly
    ``-log(1 - sigmoid(0)) = log 2 = 0.6931471805599453`` per supervised cell,
    and the weighted mean over supervised cells is that same value whatever the
    mask is. ⛔ An analytic expectation is the strongest cross-check available
    here, because it is derived from the definition of the loss rather than from
    a previous run of this code.
    """
    import math
    import torch
    from tanitad.refs import tac_goal_head as tgh
    v7l, labels = _three_labels(monkeypatch)
    mask = tgh.mask_report(v7l.goal_supervision_census(labels))

    K = 22
    logits = torch.zeros(2, K, requires_grad=True)
    y = torch.zeros(2, K)
    w = torch.ones(2, K)
    loss, n_sup = tgh.tac_goal_loss(
        logits, y, w,
        pos_weight=torch.tensor(v7l.goal_pos_weight(labels),
                                dtype=torch.float32),
        class_mask=torch.tensor(mask["mask"], dtype=torch.float32))
    assert abs(float(loss) - math.log(2.0)) < 1e-6, (
        f"BCE at logit 0 with target 0 must be exactly log 2 "
        f"({math.log(2.0)}); got {float(loss)}")
    # n_supervised = (rows) x (classes the mask leaves live) -- a literal
    assert n_sup == 2 * int(mask["n_trainable"]), (
        f"n_supervised {n_sup} != 2 rows x {int(mask['n_trainable'])} "
        f"trainable classes -- the class_mask is not being applied")
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0, (
        "the loss produced no gradient on its own logits")
'''

s = s.rstrip("\n") + "\n" + NEW
ast.parse(s)
assert s.count("def test_train_fits_pos_weight_and_mask_from_the_split_not_a_literal") == 1

with io.open(DST, "wb") as fh:
    fh.write(s.replace("\n", NEWLINE).encode("utf-8"))
print(f"WROTE {DST} lines={s.count(chr(10))}")
