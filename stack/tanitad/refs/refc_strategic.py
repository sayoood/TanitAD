"""The STRATEGIC token heads — the 15-token vocabulary nothing trained (P4).

⭐⭐ WHAT THIS CLOSES. ``STRATEGIC_GOAL_TOKENS_V7`` (8) and
``STRATEGIC_ACTION_TOKENS_V7`` (7) are minted on **4,572/4,572** training clips
and, until 2026-09-06, **no REF-C head was sized on them and no REF-C loss
referenced them** — MEASURED, two independent probe mechanisms, ``str_goal`` /
``a_str`` at **0 hits** in the live trainer. It is a STRUCTURAL absence, not a
zero weight. This module is the model side; the label side already exists
(``v7_labels.HEADS``, ``head_mask``, ``class_weights``) and is a sibling's.

⛔ **PLUMBING OR ARCHITECTURE? ARCHITECTURE, and the near-miss matters.**
``refc_v3.py`` HAS a ``str_goal_head`` — but it is ``nn.Linear(d_ctx, 3)``, a
**geometric bearing** head (cos, sin, valid), not a token classifier. ``v6``'s
``goal_head_tac`` has **0 occurrences** in the REF-C line. There is no 8-way or
7-way slot anywhere in REF-C for a reader to fill in. A name collision with a
head that does something else is exactly how a wiring change gets reported as
already done, so the head built here is deliberately named ``str_goal_tok_head``
— mirroring the sibling's ``tac_goal_tok_head`` / ``tac_goal_head`` split, and
for the same reason.

## What is admissible in, and what is not

⛔ **The heads EXTRACT from vision + measured ``v0``. Alpamayo GT SUPERVISES.**
The strategic tokens are ``provenance = "geometry"`` — derived from the ego's
own FUTURE path — which is fine for a LABEL (PI 2026-08-03: *labels may use ego;
inference is vision-only*) and inadmissible as an input.

⛔⛔ **AND THE NAV COMMAND IS NOT ADMISSIBLE INTO THESE HEADS, ON A MEASUREMENT.**
Nav is a legitimate inference input in general (*"NOT a training signal; an INPUT
simulating the nav system"*), which is precisely why this needed checking rather
than assuming. MEASURED 2026-09-06 over 801 banked v7 clips:
``nav -> g_str`` **determinism 0.8302**, mutual information **0.6444 bits of
H = 1.2125**, i.e. **53.1 % of the strategic goal label is already in the nav
command**. That is not the flagship route head's outright bijection (369/369,
scored 1.0000) but it is the same family, and a nav-fed strategic head would
report half its own input back as skill. ⇒ ``forward`` takes ONE tensor, the
vision/ego context, and there is no parameter through which nav can arrive.
A nav-fed arm is a separate, explicitly-named experiment that owes a
**nav-ablated control** before any number of its is quotable.

## Single-label, and why that is NOT the manoeuvre-softmax defect

⚠️ The tactical GOAL head is multi-label (2–7 tokens/record) and collapsing it
into a softmax would rebuild the 5-way manoeuvre defect. **The strategic
tokens are different in kind**: the emitter writes exactly ONE ``g_str.token``
and ONE ``a_str.token`` per clip — MEASURED, 801/801 records, one each. A
softmax is the correct likelihood for a genuine choice; the defect was never
"softmax", it was "softmax over a SET". Stated explicitly because the two heads
sit next to each other and the wrong lesson transfers easily.

⛔ **The two heads stay SEPARATE.** ``str_goal`` x ``str_action`` merged into one
56-way head is the same collapse one level up, and ``v7_labels.HEADS`` already
keeps ``tac_lat`` / ``tac_lon`` apart for exactly this reason.

## The loss never guards itself out of existence

⛔ MEASURED across the programme: **42 of 138 optimizer tensors received no
gradient — 52.2 % of a declared budget** — because terms were weighted 0.0 *and
guarded behind* ``if w > 0``. A guarded term makes ``p.grad`` **None**, which is
indistinguishable from a head that was never wired. Here the term is ALWAYS
computed and the weight is applied by MULTIPLICATION, so at weight 0.0 the
parameters still receive a **zeros** gradient — and ``p.grad is None`` stays a
clean discriminator for *"not wired"*.

⚠️ **AND CHECK THE MASK BEFORE DECLARING ANYTHING DEAD.** A sibling found the
route head's apparent zero gradient was a **validity mask**: forcing
``route_valid=True`` moved the loss **0.0 -> 0.687**. :func:`strategic_loss`
therefore returns ``n_supervised`` so a zero loss reads as *"nothing was in
band"* rather than *"the head is broken"*.

## The ceiling this head cannot exceed, MEASURED

| fact | value | consequence |
|---|---|---|
| supervisable share of the usable horizon | **11.43 %** (one record/clip, ``t0_s = 8.0``, +/- 2.0 s, median 35.0 s available) | 88.57 % of frames carry NO strategic GT |
| ``g_str`` classes with support | **3 of 8** (4 are ``NOT_YET_EXTRACTABLE``; ``STOP_AT_FOLLOW_ROUTE`` is extractable but had 0/801) | 5 logits are dead |
| ``a_str`` classes with support | **3 of 7** (2 ``NOT_YET_EXTRACTABLE``) | 4 logits are dead |
| provenance | **geometry, 801/801** | ⭐ the 175-grounded / 604-disputed traffic-light problem does NOT apply here |
| majority class | ``FOLLOW_ROUTE`` **68.54 %** / ``HOLD_MAIN_ROAD`` **57.30 %** | ⛔ pooled accuracy is inadmissible; macro per-class recall only |

⇒ **"Use the whole strategic vocab" is, on this corpus, 6 of 15 tokens.** The
other 9 are blocked on a corpus that does not carry them, not on this module.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tanitad.data import v7_labels as v7l

__all__ = ["StrategicTokenHead", "StrategicVocabMismatch",
           "OffVocabularyToken", "assert_head_matches_vocabulary",
           "encode_targets", "strategic_loss", "per_class_recall",
           "majority_control_recall", "support_report", "STRATEGIC_HEADS"]

#: The two heads this module owns. Names match ``v7_labels.HEADS`` keys exactly,
#: so a mismatch is a KeyError at import rather than a silent width error.
STRATEGIC_HEADS: tuple[str, ...] = ("str_goal", "str_action")


class StrategicVocabMismatch(RuntimeError):
    """Head width does not match the label vocabulary."""


class OffVocabularyToken(RuntimeError):
    """A label carries a token the vocabulary does not contain.

    ⛔⛔ **THIS IS RAISED, NEVER SILENTLY DROPPED, AND THAT IS THE WHOLE POINT.**
    MEASURED 2026-09-06 on the banked ``s2_labels_v7.jsonl`` sample (801 clips):
    **``REDUCE_TO_FOLLOW_ROUTE`` appears on 90 records = 11.24 % of
    ``a_str``**, and it is **NOT in** ``STRATEGIC_ACTION_TOKENS_V7`` — the
    absence is deliberate and PINNED (``test_vocab_v7_frozen.py`` asserts
    ``"REDUCE_TO_FOLLOW_ROUTE" not in STRATEGIC_ACTION_TOKENS_V7``), so the
    writer and the vocabulary disagree.

    A ``tokens.index(tok)`` would raise here; a ``dict.get(tok, -1)`` would
    return ``-1``, be swallowed by ``ignore_index``, and **silently delete
    11.24 % of the action supervision** while every log still said the head was
    trained. Refusing names the blob and the token, so the operator fixes the
    labels instead of training on a hole.

    ⚠️ **SCOPE.** That measurement is on the **v7** sample; the live arm trains
    on **v7.2**, which this dev box does not hold. The refusal is what makes
    that difference *visible on the first batch* rather than after 40k steps.
    """


def assert_head_matches_vocabulary(logits: torch.Tensor, head: str) -> None:
    """⛔ Mirror of the trainer's ``z_tac`` refusal, for the strategic heads.

    The paired refusal exists because *a v3 build once passed the core check
    and trained 8-wide ``z_tac`` heads against 3-class labels*. An 8-wide
    strategic goal head against a 7-token action vocabulary is the identical
    error one module over, and it converges quietly.
    """
    if head not in v7l.HEADS:
        raise StrategicVocabMismatch(
            f"{head!r} is not a v7_labels head; known: {sorted(v7l.HEADS)}")
    k = len(v7l.HEADS[head])
    if logits.shape[-1] != k:
        raise StrategicVocabMismatch(
            f"{head}: head is {logits.shape[-1]}-wide but the vocabulary has "
            f"{k} tokens ({v7l.HEADS[head]!r}). Refusing, because an 8-wide "
            f"head once trained against 3-class labels and converged.")


class StrategicTokenHead(nn.Module):
    """``ctx -> (str_goal logits [B, 8], str_action logits [B, 7])``.

    ⛔ **ONE input, deliberately.** ``ctx`` is the vision + measured-``v0``
    context. There is no ``nav`` parameter and no ``**kwargs`` through which one
    could arrive — see the module docstring's 53.1 % measurement.

    ⭐ **This head does NOT condition the decoder.** It reads ``ctx`` and emits
    logits; nothing downstream consumes them. That is what lets the strategic
    **VOCABULARY** be supervised while the strategic **LAYER's conditioning
    path** stays switchable via ``--no-strategic`` (commit ``bfcdd758b``) — the
    two PI instructions, reconciled by object rather than by compromise. ⛔ Do
    not add a return path into the decoder here; that would silently re-enable
    the conditioning the ablation exists to measure.
    """

    def __init__(self, d_in: int, *, hidden: int | None = None,
                 zero_init: bool = True):
        super().__init__()
        h = int(hidden or max(d_in, 64))
        self.trunk = nn.Sequential(nn.Linear(int(d_in), h), nn.GELU())
        self.goal = nn.Linear(h, len(v7l.HEADS["str_goal"]))
        self.action = nn.Linear(h, len(v7l.HEADS["str_action"]))
        if zero_init:
            # Zero-init the OUTPUT layers only: the head starts at a uniform
            # posterior, so adding it to a trained arm cannot move any other
            # loss on step 0. The trunk keeps its normal init or the head can
            # never escape the zero (dead-ReLU family).
            for lin in (self.goal, self.action):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(ctx)
        return {"str_goal": self.goal(h), "str_action": self.action(h)}


def encode_targets(tokens: Sequence[str | None], head: str,
                   *, strict: bool = True) -> torch.Tensor:
    """Token strings -> class indices ``[B]``, with ``-1`` for *absent*.

    ⛔ ``-1`` means **"this clip carries no label"** and nothing else. A token
    that is PRESENT but off-vocabulary raises :class:`OffVocabularyToken`; the
    two conditions must never share an encoding, because the first is a band
    limit (expected, 88.57 % of frames) and the second is a corpus defect.

    ``strict=False`` downgrades the refusal to a returned ``-1`` **and is for
    an audit path only** — it is the behaviour that silently deletes 11.24 % of
    the action supervision, so it is opt-in and named.
    """
    if head not in v7l.HEADS:
        raise StrategicVocabMismatch(f"unknown head {head!r}")
    index = {t: i for i, t in enumerate(v7l.HEADS[head])}
    out = []
    for tok in tokens:
        if tok is None:
            out.append(-1)
            continue
        if tok in index:
            out.append(index[tok])
            continue
        if strict:
            raise OffVocabularyToken(
                f"{head}: token {tok!r} is not in the {len(index)}-token "
                f"vocabulary {tuple(v7l.HEADS[head])!r}. This is a LABEL/VOCAB "
                f"disagreement, not a missing label -- refusing rather than "
                f"encoding it as -1, which ignore_index would swallow and "
                f"delete this class's supervision entirely. (MEASURED "
                f"precedent: REDUCE_TO_FOLLOW_ROUTE, 90/801 = 11.24 % of "
                f"a_str on the v7 sample.)")
        out.append(-1)
    return torch.tensor(out, dtype=torch.long)


def strategic_loss(logits: dict[str, torch.Tensor],
                   targets: dict[str, torch.Tensor],
                   *, weights: dict[str, torch.Tensor] | None = None,
                   w: float = 1.0) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-head CE, summed, ALWAYS computed. -> ``(loss, telemetry)``.

    ⛔ ``w`` multiplies; it never guards. At ``w = 0.0`` the head still receives
    a **zeros** gradient, so ``p.grad is None`` keeps meaning *"not wired"*.

    ⛔ ``ignore_index = -1`` is the BAND, not a failure: 88.57 % of frames carry
    no strategic GT. ``n_supervised`` is returned per head precisely so a zero
    loss can be read as *"nothing was in band"* rather than *"the head is
    broken"* — the mask trap that made a route head look dead.

    ``weights[head]`` is a ``[K]`` per-class weight; build it from the LOADED
    SPLIT with ``v7_labels.class_weights`` (masked/absent classes get 0.0),
    never hardcoded — ``HOLD_MAIN_ROAD`` is 57.30 % here and 52.3 % in the other
    blob, and a hardcoded weight is the derived-constant trap.
    """
    total = None
    tele: dict[str, Any] = {}
    for head in STRATEGIC_HEADS:
        lg, tg = logits[head], targets[head]
        assert_head_matches_vocabulary(lg, head)
        cw = None if weights is None else weights.get(head)
        if cw is not None:
            cw = cw.to(device=lg.device, dtype=lg.dtype)
        n_sup = int((tg >= 0).sum())
        # ⚠️ CE over an all-ignored batch returns NaN, not 0.0 -- and a NaN in
        # the sum poisons every other term's gradient. An empty band is the
        # NORMAL case here (88.57 %), so it is handled, not asserted away.
        if n_sup == 0:
            term = lg.sum() * 0.0
        else:
            term = F.cross_entropy(lg, tg.to(lg.device), weight=cw,
                                   ignore_index=-1)
        tele[f"{head}_n_supervised"] = n_sup
        tele[f"{head}_loss"] = float(term.detach())
        total = term if total is None else total + term
    return float(w) * total, tele


def per_class_recall(logits: torch.Tensor, targets: torch.Tensor,
                     head: str) -> dict[str, Any]:
    """⛔ PER-CLASS RECALL, never pooled accuracy.

    MEASURED, and this is why: a constant ``FOLLOW_ROUTE`` predictor scores
    **0.6854 pooled accuracy** on this corpus and its macro per-class recall is
    **exactly 1/8 = 0.1250**, the no-information value. Reporting the first
    number would make a head that learned nothing look like a head that works.
    """
    assert_head_matches_vocabulary(logits, head)
    toks = v7l.HEADS[head]
    pred = logits.argmax(dim=-1)
    keep = targets >= 0
    per: dict[str, Any] = {}
    recalls = []
    for i, tok in enumerate(toks):
        m = keep & (targets == i)
        n = int(m.sum())
        if n == 0:
            per[tok] = {"n": 0, "recall": None}   # ⛔ None, never 0.0
            continue
        r = float((pred[m] == i).to(torch.float64).mean())
        per[tok] = {"n": n, "recall": r}
        recalls.append(r)
    return {"per_class": per, "n_supervised": int(keep.sum()),
            "n_classes_with_support": len(recalls),
            "macro_recall_over_supported": (sum(recalls) / len(recalls)
                                            if recalls else None),
            "pooled_accuracy_DO_NOT_QUOTE": (
                float((pred[keep] == targets[keep]).to(torch.float64).mean())
                if int(keep.sum()) else None)}


def majority_control_recall(targets: torch.Tensor, head: str) -> dict[str, Any]:
    """The CONTROL THAT MUST READ A KNOWN VALUE.

    A constant predictor of the majority class. Its macro recall over the FULL
    vocabulary is **exactly ``1/K``** and its pooled accuracy is the majority
    share. Any head that does not beat ``1/K`` has learned nothing.
    """
    toks = v7l.HEADS[head]
    keep = targets >= 0
    if int(keep.sum()) == 0:
        return {"n_supervised": 0, "macro_recall_full_vocab": None}
    cnt = Counter(int(t) for t in targets[keep].tolist())
    maj, maj_n = cnt.most_common(1)[0]
    n_sup = int(keep.sum())
    supported = [i for i in range(len(toks)) if cnt.get(i, 0) > 0]
    return {"n_supervised": n_sup,
            "majority_class": toks[maj],
            "majority_share": maj_n / n_sup,
            # over the FULL vocabulary the constant predictor gets exactly one
            # class right out of K -> 1/K, the no-information value.
            "macro_recall_full_vocab": 1.0 / len(toks),
            "macro_recall_over_supported": 1.0 / max(len(supported), 1),
            "K": len(toks), "K_supported": len(supported)}


def support_report(labels: Sequence[Any]) -> dict[str, Any]:
    """Per-head support census + the reason each dead class is dead.

    ⛔ A class is dead for one of TWO reasons and they need different fixes:
    ``NOT_YET_EXTRACTABLE`` (the emitter cannot produce it — a corpus/detector
    work item) or merely ABSENT from this split (possibly a sampling artifact).
    Collapsing them into "unpopulated" is how a corpus blocker gets reported as
    a training detail.
    """
    from tanitad.models.vocab_v7 import NOT_YET_EXTRACTABLE
    out: dict[str, Any] = {}
    for head in STRATEGIC_HEADS:
        attr = head            # V7Label field names match the head names
        seen = Counter(getattr(x, attr) for x in labels)
        rows = {}
        for tok in v7l.HEADS[head]:
            n = seen.get(tok, 0)
            rows[tok] = {"n": n, "share": n / max(len(labels), 1),
                         "dead_reason": (None if n else
                                         ("NOT_YET_EXTRACTABLE"
                                          if tok in NOT_YET_EXTRACTABLE
                                          else "ABSENT_FROM_SPLIT"))}
        off = {t: c for t, c in seen.items() if t not in v7l.HEADS[head]}
        out[head] = {"K": len(v7l.HEADS[head]), "n_labels": len(labels),
                     "classes": rows,
                     "n_with_support": sum(1 for r in rows.values() if r["n"]),
                     "off_vocabulary": off}
    return out
