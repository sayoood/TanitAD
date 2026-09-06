"""⛔ THE DECLARATION A MODULE MAKES WHEN NO LOSS CAN EVER REACH IT.

WHAT THIS EXISTS TO PREVENT (MEASURED 2026-09-06, `…/Research/2026-09-06-v7f-budget/`).
A parameter with ``requires_grad=True`` that never receives a gradient is not a
wasted-FLOPs problem — it is a **MEASUREMENT DEFECT**. It is counted in
``param_report()["trainable"]`` and in ``apply_stage_freeze()["n_trainable"]``,
both of which ship in every run's ``config.json``, so the run's own artifact
overstates what it trained. Three measurements, all on the live code:

  * **v7-tiny, its own 30 k config** (`v7tiny_postrain30k/config.json`):
    **42 of 138** optimizer tensors, **5,305,667 params = 52.2 %** of the
    declared trainable budget, received no gradient.
  * **v7f at its own pre-registered launch line** (`PREREG_V7F.md` §9):
    **193 of 443** optimizer tensors, **94,717,187 params = 40.3 %** of a
    234.9 M declared budget.
  * The single largest term in that 94.7 M is **86,138,112 params of
    `ema_o5_enc`** — the O5 EMA TEACHER, whose own class docstring says its
    parameters are *"buffers-in-spirit: requires_grad=False and excluded from
    every optimiser"*. :func:`~tanitad.models.v6.apply_stage_freeze` walks
    EVERY named parameter and sets ``requires_grad`` from the GROUP MAP, and
    ``ema_o5_enc.`` maps to group ``aux`` — which S-W trains. The constructor's
    freeze is therefore UNDONE, and the teacher lands in the optimizer.

⭐ THE MECHANISM IS ALREADY NAMED IN THIS CODEBASE, ONE GUARD OVER.
:func:`~tanitad.models.v6.assert_frozen_external`'s docstring states it exactly
— *"a frozen external encoder installed under the ``encoder`` group is
UN-FROZEN by S-W"* — and quantifies it at **86,580,480** foreign parameters.
This module is that same finding applied to NATIVE modules, which the
frozen-external guard deliberately does not cover (it is about FOREIGN
backbones, and its Direction-B check would change meaning if native dead
tensors were folded into it).

⛔ WHY A SEPARATE FLAG AND NOT ``FROZEN_EXTERNAL_FLAG``. Two different claims:

  ``FROZEN_EXTERNAL``  "this subtree is someone else's trained weights and must
                       not move" — an X3/licensing statement about PROVENANCE.
  ``GRAD_UNREACHABLE`` "no loss in this ladder can reach this subtree, so
                       declaring it trainable is a false statement about the
                       run" — a statement about the LOSS GRAPH.

An EMA teacher is native and grad-unreachable. A DINOv3 backbone is foreign and
frozen. Folding them together would make ``assert_frozen_external``'s audit
("declared_subtrees") answer neither question, and its Direction-B check —
"every trainable group still holds a trainable NATIVE parameter" — would be
satisfied by a group holding nothing but dead natives.

⚠️ THIS IS A DECLARATION, NOT A DELETION. The tensors stay in ``state_dict``
(``requires_grad`` is not serialised), so **every banked checkpoint still loads
strictly**. What changes is that they leave the optimizer and stop being
counted as trained.
"""

from __future__ import annotations

from torch import nn

__all__ = ["GRAD_UNREACHABLE_FLAG", "declare_grad_unreachable",
           "grad_unreachable_prefixes", "in_declared_subtree"]

#: Attribute a submodule carries to declare that NO ladder loss reaches it.
#: Set it with :func:`declare_grad_unreachable`, never by hand — the setter is
#: what makes the declaration greppable and forces a REASON to be written down.
GRAD_UNREACHABLE_FLAG = "_tanitad_grad_unreachable"


def declare_grad_unreachable(module: nn.Module, why: str) -> nn.Module:
    """Mark ``module`` as reachable by no loss, and freeze it now.

    Returns the module so it can wrap a constructor call. ``why`` is stored on
    the module and surfaces in :func:`grad_unreachable_prefixes`, in the freeze
    audit, and therefore in the run's ``config.json`` — a dead tensor that
    cannot say WHY it is dead is the state this declaration exists to end.
    """
    if not str(why).strip():
        raise ValueError("declare_grad_unreachable needs a REASON: the "
                         "declaration ships in config.json and an unexplained "
                         "dead subtree is the defect, not the fix")
    setattr(module, GRAD_UNREACHABLE_FLAG, str(why))
    module.requires_grad_(False)
    return module


def grad_unreachable_prefixes(root: nn.Module) -> dict[str, str]:
    """``{parameter-name prefix: reason}`` for every declared subtree."""
    out: dict[str, str] = {}
    for name, mod in root.named_modules():
        why = getattr(mod, GRAD_UNREACHABLE_FLAG, None)
        if why:
            out[name] = str(why)
    return out


def in_declared_subtree(param_name: str, prefixes) -> str | None:
    """The declared prefix owning ``param_name``, or None.

    ⚠️ Prefix matching is on PATH SEGMENTS, not characters: ``heads.4`` must not
    swallow ``heads.40``. Hence the explicit ``pre + "."`` and equality tests
    rather than a bare ``startswith(pre)``.
    """
    for pre in prefixes:
        if pre == "" or param_name == pre or param_name.startswith(pre + "."):
            return pre
    return None
