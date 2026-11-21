"""The tactical-GOAL token head — the 22-token vocabulary nothing trained.

⭐⭐ WHAT THIS CLOSES. The v7 emitter mints a 22-token tactical goal set on every
one of 4,572 training clips — including a ground-truth traffic-light COLOUR —
and until 2026-09-06 **no head was sized on it, no loss referenced it, and no
gradient could reach it** (audit ``AUDIT_RESULT.json``; pinned by
``stack/tests/test_tactical_label_reach.py``). It is a STRUCTURAL absence, not a
zero weight. It is the mechanism behind two PI observations on the refcv4b
video: *the model never brakes for a red light*, and *lane changes never
activate*. The labels existed; nothing taught them.

⛔ IT IS MULTI-LABEL, NEVER A SOFTMAX. MEASURED on the v7.2 train blob: 2–7
tokens per record, mean 2.751. Collapsing a SET into a CHOICE is the 5-way
manoeuvre-softmax defect — the programme's single largest known defect, the one
mechanism behind 0/881 accelerate and the speed-fan — rebuilt one layer up. So
this is 22 independent sigmoids under BCE, with a per-CLASS ``pos_weight`` and a
per-CELL validity weight.

⛔⛔ THE LOSS NEVER GUARDS ITSELF OUT OF EXISTENCE. MEASURED 2026-09-06 across
the programme: **42 of 138 optimizer tensors received no gradient — 52.2 % of a
declared budget** — because objectives were weighted 0.0 *and the loss guarded
those terms behind ``if w > 0``*. A guarded term makes ``p.grad`` **None**, which
is indistinguishable from a head that was never wired. Here the term is ALWAYS
computed and the weight is applied by MULTIPLICATION, so at weight 0.0 the
parameters still receive a **zeros** gradient. ⇒ ``p.grad is None`` remains a
clean discriminator for *"this head is not wired"*, and it can never again be
confused with *"this head is switched off"*.

⚠️ AND CHECK THE MASK BEFORE DECLARING ANYTHING DEAD. A sibling found the route
head's apparent zero gradient was a **validity mask**, not a dead head: forcing
``route_valid=True`` moved the loss 0.0 → 0.687. The same trap lives here —
:func:`tac_goal_loss` returns ``n_supervised`` precisely so a zero loss can be
read as *"nothing was in band"* rather than *"the head is broken"*, and
:func:`mask_report` names every class that carries no supervised negative.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tanitad.data import v7_labels as v7l

__all__ = ["TacGoalTokenHead", "TacGoalVocabMismatch", "mask_report",
           "per_class_scores", "majority_control_scores", "tac_goal_loss",
           "assert_head_matches_vocabulary"]


class TacGoalVocabMismatch(RuntimeError):
    """Head width does not match the label vocabulary."""


def assert_head_matches_vocabulary(logits: torch.Tensor,
                                   n_expect: int | None = None) -> None:
    """⛔ Mirror of the trainer's z_tac refusal, for this head.

    The paired refusal in ``refc_v3_train.py`` exists because *"a v3 build once
    passed the core check and trained 8-wide z_tac heads against 3-class labels
    (2026-09-01)"*. A 22-wide head silently scored against a differently-sized
    target is the same defect, and it would not raise on its own — BCE
    broadcasts.
    """
    n_expect = len(v7l.TAC_GOAL_TOKENS) if n_expect is None else n_expect
    if logits.shape[-1] != n_expect:
        raise TacGoalVocabMismatch(
            f"[tac_goal] ⛔ head width {logits.shape[-1]} != tactical goal "
            f"vocabulary {n_expect}. The head and the labels MUST share one "
            f"source: size the head from "
            f"`v7_labels.TAC_GOAL_TOKENS` (equivalently "
            f"`v6.TACTICAL_GOAL_VOCAB_VERSIONS[tac_vocab_version]`), never "
            f"from a literal. This refusal mirrors the z_tac one, which exists "
            f"because an 8-wide head once trained against 3-class labels.")


class TacGoalTokenHead(nn.Module):
    """``[B, d_in] -> [B, 22]`` multi-label logits over the tactical goal set.

    ⚠️ NAMED ``..._TOKEN_...`` ON PURPOSE. ``RefCV3Model.tac_goal_head`` already
    exists and is the **geometric** goal regressor (x, y, heading, speed at
    tau). Two different objects called "the tactical goal head" is how a
    programme ends up quoting one arm's number for the other.
    """

    def __init__(self, d_in: int, *, hidden: int | None = None,
                 n_tokens: int | None = None):
        super().__init__()
        self.n_tokens = len(v7l.TAC_GOAL_TOKENS) if n_tokens is None else n_tokens
        self.tokens = tuple(v7l.TAC_GOAL_TOKENS[:self.n_tokens])
        if hidden:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, self.n_tokens))
        else:
            self.net = nn.Linear(d_in, self.n_tokens)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def tac_goal_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                  *, pos_weight: torch.Tensor | Sequence[float] | None = None,
                  class_mask: torch.Tensor | Sequence[bool] | None = None,
                  ) -> tuple[torch.Tensor, int]:
    """``(loss, n_supervised_cells)``. Weighted multi-label BCE.

    ``logits`` ``[B, K]``, ``y`` ``[B, K]`` in {0, 1}, ``w`` ``[B, K]`` in
    {0, 1} — 0 marks a cell that carries NO EVIDENCE and must not train either
    way (:func:`v7_labels.tactical_goal_targets` builds both).

    ``pos_weight`` ``[K]`` is the per-class positive re-weighting; build it with
    :func:`v7_labels.goal_pos_weight` FROM THE LOADED SPLIT, never from a
    literal. ``class_mask`` ``[K]`` switches whole classes off — use it for the
    five classes that carry positives but **no supervised negative**, where an
    unmasked logit can only be pushed towards 1.

    ⛔ Returns a REAL zero (``0.0`` with ``n_supervised == 0``) when nothing is
    in band, and the caller must report those two together. A bare ``0.0`` in a
    metrics row reads as *"supervised, and perfect"*, so an unlabelled step and
    a perfect step would print the same character — the defect
    ``test_tac_loss_logging.py`` pins with ``assert row[k] is None``.
    """
    assert_head_matches_vocabulary(logits)
    if logits.shape != y.shape or logits.shape != w.shape:
        raise TacGoalVocabMismatch(
            f"[tac_goal] ⛔ shape mismatch: logits {tuple(logits.shape)}, "
            f"y {tuple(y.shape)}, w {tuple(w.shape)} — all three must be "
            f"[B, {logits.shape[-1]}].")
    y = y.to(logits.dtype)
    w = w.to(logits.dtype)
    if class_mask is not None:
        m = (class_mask if torch.is_tensor(class_mask)
             else torch.tensor(list(class_mask)))
        w = w * m.to(logits.device, logits.dtype).view(1, -1)
    pw = None
    if pos_weight is not None:
        pw = (pos_weight if torch.is_tensor(pos_weight)
              else torch.tensor(list(pos_weight)))
        pw = pw.to(logits.device, logits.dtype)
    per = F.binary_cross_entropy_with_logits(
        logits, y, weight=None, pos_weight=pw, reduction="none")
    denom = w.sum()
    n_sup = int(denom.detach().item())
    # ⛔ NO `if n_sup:` GUARD. Dividing by a clamped denominator keeps the term
    # in the graph, so every parameter still receives a gradient tensor (zeros)
    # and `p.grad is None` stays the discriminator for "never wired".
    loss = (per * w).sum() / denom.clamp_min(1.0)
    return loss, n_sup


@torch.no_grad()
def per_class_scores(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                     *, threshold: float = 0.5,
                     tokens: Sequence[str] | None = None) -> dict[str, Any]:
    """PER-CLASS recall/precision/n. ⛔ NEVER a pooled accuracy.

    ⭐ THE REASON, MEASURED: a head that predicts the majority class scores well
    while doing nothing. ``LANE_CHANGE_R`` is 15 of 4,572 (0.33 %), so a head
    that never fires it is 99.67 % "accurate" on that column. The programme has
    already banked this exact shape once — a ``turn_left`` recall of **exactly
    0.0000 of 11 at BOTH seeds**, replicated, sitting beside an
    apparently-healthy safety number.

    Only cells with ``w > 0`` are scored: an ignored cell is not a miss.
    """
    toks = list(tokens or v7l.TAC_GOAL_TOKENS)
    pred = (torch.sigmoid(logits) >= threshold).to(torch.float32)
    y = y.to(torch.float32)
    m = (w > 0).to(torch.float32)
    out: dict[str, Any] = {}
    for i, t in enumerate(toks):
        mi, yi, pi = m[:, i], y[:, i], pred[:, i]
        tp = float((mi * yi * pi).sum())
        fn = float((mi * yi * (1 - pi)).sum())
        fp = float((mi * (1 - yi) * pi).sum())
        n_pos, n_neg = tp + fn, float((mi * (1 - yi)).sum())
        out[t] = {
            "n_pos": int(n_pos), "n_neg": int(n_neg),
            "recall": (tp / n_pos) if n_pos else None,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "n_fired": int(tp + fp),
        }
    return out


@torch.no_grad()
def majority_control_scores(y: torch.Tensor, w: torch.Tensor,
                            *, tokens: Sequence[str] | None = None
                            ) -> dict[str, Any]:
    """⭐⭐ THE CONTROL THAT MUST READ ITS KNOWN VALUE.

    Scores the **majority-class predictor** — for each token, always predict
    whichever of {present, absent} is more common in the supervised cells — on
    the SAME cells, with the SAME function. Its recall is known in advance:

      * a token whose majority is ABSENT  -> recall **exactly 0.0**;
      * a token whose majority is PRESENT -> recall **exactly 1.0**.

    A trained head that does not beat this has added nothing, and a panel whose
    control does not read those exact values has a scoring bug, not a result.
    This is the rule that caught four separate manufactured results in one
    afternoon (2026-08-22): in three of them the control read the same value as
    the thing being measured.
    """
    toks = list(tokens or v7l.TAC_GOAL_TOKENS)
    m = (w > 0).to(torch.float32)
    y = y.to(torch.float32)
    out: dict[str, Any] = {}
    for i, t in enumerate(toks):
        mi, yi = m[:, i], y[:, i]
        n_pos = float((mi * yi).sum())
        n_neg = float((mi * (1 - yi)).sum())
        majority_is_present = n_pos > n_neg
        recall = 1.0 if majority_is_present else 0.0
        out[t] = {"n_pos": int(n_pos), "n_neg": int(n_neg),
                  "majority": "present" if majority_is_present else "absent",
                  "recall": (recall if (n_pos + n_neg) else None),
                  "precision": ((n_pos / (n_pos + n_neg))
                                if majority_is_present and (n_pos + n_neg)
                                else None)}
    return out


def mask_report(census: dict[str, Any]) -> dict[str, Any]:
    """Which classes may be trained at all, and WHY each is switched off.

    ⛔ A class with positives but ZERO supervised negatives must be MASKED. Its
    logit can only ever be pushed towards 1, so leaving it live does not teach
    it — it degrades the shared trunk and inflates any pooled score. MEASURED on
    the v7.2 train blob under the default policy, five classes are in that
    state: ``SPEED_BAND`` (prevalence 100.0 %, a constant), ``YIELD``,
    ``CORRIDOR_OFFSET``, ``GAP_TARGET``, ``REACT_ON_ONCOMING`` — all CoT-backed
    tokens with no exclusion partner in the frozen table.
    """
    mask, why = [], {}
    for t in v7l.TAC_GOAL_TOKENS:
        d = census.get(t) or {}
        pos, neg = int(d.get("pos", 0)), int(d.get("neg", 0))
        if pos == 0:
            why[t] = "no positives in this split"
        elif neg == 0:
            why[t] = ("no supervised negative — the logit could only be pushed "
                      "towards 1; needs an exclusion partner or a probed "
                      "negative before it is trainable")
        mask.append(pos > 0 and neg > 0)
    return {"mask": tuple(mask),
            "trainable": [t for t, k in zip(v7l.TAC_GOAL_TOKENS, mask) if k],
            "masked_why": why,
            "n_trainable": sum(mask), "n_total": len(mask)}
