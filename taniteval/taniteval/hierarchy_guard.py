"""TanitEval — PC2: *"the hierarchy is in the scored loop"* as a CODE ASSERTION.

WHY THIS MODULE EXISTS
----------------------
The deployed flagship headline **0.4271 m** is produced by
``metric_dynamics.rollout_decode``, whose signature is::

    rollout_decode(predictor, states, actions, future_actions, step_readout, k)
    #   -> predictor(win_s, win_a)      NO intent=, NO ctx, NO nav

and which is additionally fed the **expert's true future actions**
(``rollout.py`` ``fa = ep.actions[t+window : t+window+fwd_k]``). It is therefore
a **world-model fidelity** number: a dynamics decode of a control sequence the
model was handed. No policy of any shape — flat or hierarchical — can differ on
that surface except through the fidelity of its decode. It cannot express a
strategic decision, a tactical decision, or an operative one.

That was discoverable only by reading three files. This module makes it
**impossible to report a "hierarchy" number from a bypassing path again**: any
evaluation that CLAIMS to exercise the hierarchy runs inside
:class:`HierarchyTrace` and calls :func:`assert_hierarchy_traversed`, which
raises :class:`HierarchyBypass` unless the scored forward pass actually
traversed strategic -> tactical -> operative.

Spec: ``01_EXECUTION_PLAN.md`` Part A, PC2 ·
``HPP0_CONFOUND_AUDIT.md`` §2 and HPP-3 item 1.

THE THREE SEAMS, AND WHAT EACH HOOK PROVES
------------------------------------------
======================  ==================================  ====================
seam                    hooked module                       proves
======================  ==================================  ====================
``strategic``           ``model.strategic_policy``          brain ① ran
``tactical``            ``model.tactical_policy``           brain ② ran
``operative_intent``    ``model.predictor.intent_proj``     brain ③ *received*
                                                            the tactical intent
======================  ==================================  ====================

``intent_proj`` is the right hook for the third seam, not the predictor itself:
``OperativePredictor.forward`` calls it **only** when ``intent is not None``
(``predictor.py:104-105``), so a non-zero count is direct evidence that the
tactical->operative seam carried something. The predictor runs on every path,
hierarchical or not, so hooking it would prove nothing.

HONEST LIMITS — read before quoting a PASS
------------------------------------------
1. **This is a TRAVERSAL assertion, not a causality one.** It proves the three
   brains were on the code path. It does NOT prove the strategic command
   *changed* the trajectory — that is HP-3, and it is measured, not asserted
   (``strategic_probes.py``).
2. **It says nothing about whether actions were CHOSEN.** The scored rollout can
   traverse all three seams and still be fed the expert's true future actions.
   :func:`assert_actions_are_chosen` is the separate, equally load-bearing
   check, and ``rollout.collect`` is honest about failing it.
3. **A SKIP IS NOT A PASS.** An arm with no ``strategic_policy`` (REF-B, REF-C)
   cannot traverse the seam; :func:`assert_hierarchy_traversed` reports that as
   ``absent`` and still FAILS a hierarchy claim, because "there is no brain ①"
   is not a reason to accept a hierarchy number from it.
"""
from __future__ import annotations

SEAMS = ("strategic", "tactical", "operative_intent")

#: Where each seam's counter comes from: attribute path from the model root.
SEAM_MODULE = {
    "strategic": ("strategic_policy",),
    "tactical": ("tactical_policy",),
    "operative_intent": ("predictor", "intent_proj"),
}

BLOCK = "taniteval.hierarchy_guard/pc2"
VERSION = "1.0.0"
SPEC = ("Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/"
        "01_EXECUTION_PLAN.md Part A, PC2")

#: Eval surfaces that are KNOWN to bypass the hierarchy, with the reason. A
#: block in here may never be reported as a hierarchy number; the value is the
#: name it should be quoted under instead.
KNOWN_BYPASSING_SURFACES = {
    "taniteval.rollout/collect": (
        "metric_dynamics.rollout_decode takes no intent/ctx/nav AND is fed the "
        "expert's true future actions -> quote as `wm_fidelity_ade_2s`, never "
        "as a driving or hierarchy number"),
    "taniteval.bench/collect_full": (
        "same rollout_decode surface as taniteval.rollout -> WM fidelity"),
    "stack/scripts/eval_grounded_rollout_4b": (
        "its own docstring: 'roll the OPERATIVE predictor fwd_k steps under the "
        "TRUE action sequence (intent-free...)'"),
}


class HierarchyBypass(AssertionError):
    """A path claiming to evaluate the hierarchy did not traverse it."""


def _resolve(model, path):
    obj = model
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


class HierarchyTrace:
    """Count forward calls through the three hierarchy seams.

    Use as a context manager around the *scored* forward pass — not around the
    whole eval, or a warm-up/diagnostic call will mask a bypass in the number
    that ships::

        with HierarchyTrace(model) as tr:
            win = my_scored_collect(model, ...)
        assert_hierarchy_traversed(tr, block="taniteval.myblock")

    ``absent`` seams (the module does not exist on this arch) are distinguished
    from ``bypassed`` ones (it exists and was never called), because the two
    demand different corrections.
    """

    def __init__(self, model, seams=SEAMS):
        self.model = model
        self.seams = tuple(seams)
        self.counts = {s: 0 for s in self.seams}
        self.absent = tuple(s for s in self.seams
                            if _resolve(model, SEAM_MODULE[s]) is None)
        self._handles = []

    def _bump(self, seam):
        def hook(_m, _i, _o):
            self.counts[seam] += 1
        return hook

    def __enter__(self):
        for s in self.seams:
            mod = _resolve(self.model, SEAM_MODULE[s])
            if mod is not None:
                self._handles.append(
                    mod.register_forward_hook(self._bump(s)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def reset(self):
        for s in self.counts:
            self.counts[s] = 0
        return self

    @property
    def traversed(self) -> bool:
        return all(self.counts[s] > 0 for s in self.seams)

    def report(self) -> dict:
        bypassed = [s for s in self.seams
                    if s not in self.absent and self.counts[s] == 0]
        return {
            "block": BLOCK, "version": VERSION, "spec": SPEC,
            "counts": dict(self.counts),
            "absent_modules": list(self.absent),
            "bypassed_seams": bypassed,
            "hierarchy_traversed": bool(self.traversed),
            "_read": ("counts are FORWARD CALLS through each seam during the "
                      "scored pass. `absent` = this architecture has no such "
                      "module (structural); `bypassed` = it exists and the "
                      "scored path never called it (a wiring defect). Traversal "
                      "is necessary, NOT sufficient: it does not show the "
                      "command CHANGED the trajectory (that is HP-3) and does "
                      "not show the actions were CHOSEN (see "
                      "assert_actions_are_chosen)."),
        }


def assert_hierarchy_traversed(trace, *, block, claim="hierarchy",
                               require=SEAMS, strict=True):
    """FAIL LOUD when ``block`` claims a hierarchy its scored pass never used.

    ``strict=False`` returns the report instead of raising — for callers that
    want to *record* the bypass in their artifact (which is what
    ``rollout.collect`` does, since it is a known-bypassing WM-fidelity surface
    and must stay runnable)."""
    rep = trace.report()
    rep["claim"] = claim
    rep["asserted_block"] = block
    missing = [s for s in require if trace.counts.get(s, 0) == 0]
    rep["required_seams"] = list(require)
    rep["missing_seams"] = missing
    rep["pc2_pass"] = not missing
    if missing and strict:
        detail = ", ".join(
            f"{s} ({'no such module on this arch' if s in trace.absent else 'module exists, never called'})"
            for s in missing)
        known = KNOWN_BYPASSING_SURFACES.get(block)
        raise HierarchyBypass(
            f"PC2 VIOLATION in `{block}`: it claims to evaluate `{claim}` but "
            f"the scored forward pass never traversed: {detail}. "
            f"Seam counts: {trace.counts}. "
            + (f"This surface is a KNOWN bypass — {known}. " if known else "")
            + "A number from this path is NOT a hierarchy number. Either score "
              "a path where all three brains run (planner/closed-loop "
              "surfaces), or quote the number under its honest name. See "
              + SPEC)
    return rep


def assert_actions_are_chosen(*, block, actions_source, strict=True):
    """PC2's second half: was the trajectory CHOSEN, or was it handed over?

    ``actions_source`` is a short tag the caller supplies for its own surface:

      ``"expert_future"``  the expert's true future actions were fed in
                           (``rollout_decode(..., future_actions=...)``) —
                           **a WM-fidelity decode, not a decision**
      ``"model_chosen"``   the evaluated policy produced the actions
      ``"planner"``        a planner (CEM/MPC/fan) selected them

    Kept separate from :func:`assert_hierarchy_traversed` on purpose: an arm can
    traverse all three seams and still be told what to do. HPP-0 §2.2 calls this
    "the decision bypass — and this one is not in the review"."""
    ok = actions_source in ("model_chosen", "planner")
    rep = {"block": block, "actions_source": actions_source,
           "actions_are_chosen": bool(ok),
           "_read": ("a `expert_future` surface measures world-model fidelity "
                     "under a known control sequence. No policy of any shape "
                     "can differ on it except through decode fidelity, so it "
                     "may not carry a hierarchy-vs-flat verdict.")}
    if not ok and strict:
        raise HierarchyBypass(
            f"PC2 VIOLATION in `{block}`: actions_source={actions_source!r} — "
            "the scored rollout is fed a control sequence rather than choosing "
            "one, so the number is a world-model fidelity diagnostic "
            "(`wm_fidelity_ade_2s`). HPP-4 comparisons must run on a surface "
            "where actions are chosen (planner_p2 / closed-loop). See " + SPEC)
    return rep


def guarded(model, fn, *, block, claim="hierarchy", require=SEAMS,
            actions_source=None, strict=True):
    """Run ``fn()`` under a trace and assert PC2 on it. Returns ``(out, rep)``.

    The one-call form for a block that claims the hierarchy::

        res, pc2 = guarded(model, lambda: hierarchy_collect(...),
                           block="taniteval.hierarchy", claim="H26 seams",
                           actions_source="model_chosen")
        res["pc2"] = pc2
    """
    with HierarchyTrace(model, require) as tr:
        out = fn()
    rep = assert_hierarchy_traversed(tr, block=block, claim=claim,
                                     require=require, strict=strict)
    if actions_source is not None:
        rep["decision_surface"] = assert_actions_are_chosen(
            block=block, actions_source=actions_source, strict=strict)
        rep["pc2_pass"] = bool(rep["pc2_pass"]
                               and rep["decision_surface"]["actions_are_chosen"])
    return out, rep
