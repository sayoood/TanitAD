"""GOAL PROVENANCE — the PI's admissibility check as a COMPUTED, SYMMETRIC gate.

⚠️ NOT THE SAME FILE as ``stack/scripts/goal_provenance.py``, which shares this
basename and answers a DIFFERENT question: *was the evaluated goal an oracle read
off the ego's own future, or produced from vision?* (goal SOURCE; cited at
``GATE_PROTOCOL.md:186``). This module answers *is the goal path
information-disjoint from the situation classifier's output?* Both are real and
neither replaces the other. Always import this one fully qualified as
``tanitad.eval.goal_provenance``; never put ``stack/scripts`` on ``sys.path``.

WHY THIS EXISTS, AND WHY ``goal_admissibility.py`` DID NOT CLOSE IT
------------------------------------------------------------------
The binding ruling (Sayed, 2026-08-03) has two clauses:

  * ✅ a goal / route signal is admissible at inference — including a predicted
    geometric goal point;
  * ⛔ the **output of the situation classifier** is not, *"in any form — class
    posterior, argmax, embedding, or any feature derived from them."*

and it mandates a check: *"for any goal signal, ask **could this have been
computed from the situation classifier's output?**"*

``tanitad/eval/goal_admissibility.py`` answers most of that ruling — echo,
horizon, incremental information — but its provenance clause,
:func:`~tanitad.eval.goal_admissibility.situation_disjoint`, takes **DECLARED
provenance**: two hand-written lists of symbol NAMES, intersected. Its own
docstring says so. That is the C112/C113 failure class exactly — *a non-overlap
ASSUMED FROM PROVENANCE rather than computed*, which on C113 turned out to be a
**78.21 % contamination**. A name-set intersection cannot see a wire; it can only
see what somebody remembered to write down.

This module replaces the assumption with a measurement, and closes three further
gaps that were open at the time of writing (2026-08-18):

**A. The existing gradient probe is structurally blind to this question.**
:meth:`tanitad.models.v6.V6Stack.assert_isolation` is a real autograd probe, and
a good one — but it measures **BACKWARD** edges (*what trains what*), while
admissibility is a **FORWARD** question (*what is READ at inference*). The two
are independent, and they come apart precisely where this codebase lives:
``V6Stack.forward`` routes every downward goal port through ``self._cut()``,
i.e. ``Tensor.detach()`` (``v6.py:4341-4342``, applied at ``:4698-4700``,
``:4750``, ``:4757``, ``:4766``, ``:4773``). **A detached wire carries the full
signal and zero gradient.** So a situation-classifier output spliced into the
goal path behind a ``detach()`` would leak completely and be certified clean by
the gradient probe. :func:`compare_to_gradient_probe` demonstrates that on a
deliberately wired arm rather than asserting it.

**B. Nobody had audited the REVERSE direction.** The ruling says
*information-disjoint*, which is symmetric, but every check written before this
module asked only *goal ← situation*. :func:`dependency_matrix` computes both
directions in one pass, because with an interventional probe the reverse is
free — it is the same matrix read down the other axis.

**C. It was not a gate.** ``goal_admissibility`` has **zero call sites** outside
its own tests and one study script — flagged as item 5 of the 2026-08-16
stale-blocker sweep, open 12 days. A rule with no mechanism is C108.
:func:`assert_information_disjoint` raises.

THE PROBE, AND WHAT IT CAN AND CANNOT ESTABLISH
-----------------------------------------------
The primitive is an **intervention**, not a correlation. To ask *"does Y read
X?"* we do not measure whether Y and X move together across the corpus — a
shared trunk makes independent signals covary, which is the confound that makes
correlational provenance useless here. We **replace X's value** mid-forward and
ask whether Y changes. That is a do-operator on the live computation graph:

  * it is **detach-transparent** — it sees information, not gradient;
  * it distinguishes a **DIRECT PATH** (intervening on X moves Y) from a
    **COMMON ANCESTOR** (X and Y both move when a shared *input* is perturbed,
    but intervening on X leaves Y bit-identical). That distinction is the whole
    of the shared-trunk disclosure the ruling requires, and a correlational test
    cannot make it.

⚠️ **WHAT IT CANNOT DO.** It certifies the graph built by THIS forward on THESE
inputs — the same evidence class as ``assert_isolation``, and stated in the same
words on purpose. It is not a proof over all inputs. Two consequences are handled
explicitly rather than left to the reader:

  * a path that is dead for the probed batch (a gate that happens to be closed,
    a zero-init projection) reads as INDEPENDENT. :func:`probe_dependency`
    therefore runs several perturbation KINDS and reports ``n_probes``, and
    :func:`audit_arm` refuses to emit a clean verdict without a live positive
    control on the same batch — an all-clean report whose control is also clean
    is ``UNPOWERED``, never ``DISJOINT`` (C109: a cited positive control that was
    **inert by construction** proved nothing for as long as nobody ran it).
  * a nondeterministic forward would make every target look dependent.
    :func:`determinism_check` runs first and downgrades the whole audit to
    ``UNPOWERED`` if two clean passes disagree, rather than reporting the noise
    as a leak.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "ProvenanceViolation", "PERTURBATIONS", "determinism_check",
    "probe_dependency", "dependency_matrix", "classify_edge", "audit_arm",
    "audit_arms", "assert_information_disjoint", "module_runner",
    "compare_to_gradient_probe", "SITUATION_OUTPUT_ROLE", "GOAL_ROLE",
]

#: Role tags. An audit is expressed over ROLES, not over module names, so the
#: gate keeps meaning when a module is renamed.
SITUATION_OUTPUT_ROLE = "situation_output"
GOAL_ROLE = "goal"


class ProvenanceViolation(RuntimeError):
    """Raised by :func:`assert_information_disjoint` when a forbidden FORWARD
    information path is measured — the inference-time sibling of
    :class:`tanitad.models.v6.IsolationViolation`, which covers gradients."""


# --------------------------------------------------------------------------- #
# perturbation kinds                                                          #
# --------------------------------------------------------------------------- #
# ⚠️ SEVERAL KINDS, NOT ONE. A single perturbation can land in a dead zone of a
# live path (a saturated gate, a ReLU's negative side, a zero-init projection
# that is genuinely zero for this batch) and report INDEPENDENT for a path that
# is wired. Each kind fails differently, so their disjunction is a strictly
# stronger probe than any one of them. ``zero`` catches multiplicative reads
# that additive noise leaves invariant; ``shuffle`` catches batch-wise reads
# that a per-element scale leaves invariant; ``sign`` catches even functions.
def _p_noise(t, g):
    import torch
    return t + 3.0 * (t.detach().float().std().clamp(min=1e-3)) * torch.randn(
        t.shape, generator=g, dtype=torch.float32, device=t.device).to(t.dtype)


def _p_zero(t, g):
    return t * 0


def _p_sign(t, g):
    return -t


def _p_shuffle(t, g):
    import torch
    if t.shape[0] < 2:
        return _p_noise(t, g)
    idx = torch.randperm(t.shape[0], generator=g, device=t.device)
    if bool((idx == torch.arange(t.shape[0], device=t.device)).all()):
        idx = idx.flip(0)
    return t[idx]


def _p_large(t, g):
    return t + 7.0


#: name -> f(tensor, generator) -> tensor. Applied to a node's VALUE mid-forward.
PERTURBATIONS: dict[str, Callable] = {
    "noise": _p_noise, "zero": _p_zero, "sign": _p_sign,
    "shuffle": _p_shuffle, "offset": _p_large,
}


# --------------------------------------------------------------------------- #
# the runner contract                                                         #
# --------------------------------------------------------------------------- #
# A ``runner`` is ``run(intervention) -> dict[node_name, Tensor]`` where
# ``intervention`` is ``None`` or ``(node_name, fn)``. The runner executes ONE
# forward, applying ``fn`` to that node's value as it is produced, and returns
# every observable node. Keeping the contract this small is what lets the SAME
# instrument probe a real ``V6Stack`` (via :func:`module_runner`) and a
# hand-wired positive control, so the control exercises the instrument that is
# actually used rather than a parallel toy of it.


def module_runner(model, batch: Mapping, nodes: Mapping[str, str],
                  *, output_nodes: Mapping[str, Callable] | None = None,
                  input_nodes: Mapping[str, str] | None = None):
    """Build a runner over a real ``nn.Module`` using forward hooks.

    ``nodes`` maps ``role_name -> dotted submodule path``; the node's value is
    that submodule's OUTPUT. ``output_nodes`` maps ``role_name -> fn(out_dict)``
    for roles read off the forward's RETURN value instead of a submodule.
    ``input_nodes`` maps ``role_name -> batch key``, which makes the model's own
    INPUTS probeable — that is what lets the same instrument answer the
    vision-only question (*does the goal path read ``v0``?*) with the same
    machinery rather than a second, differently-calibrated one.

    Intervention is a ``forward_hook`` that REPLACES the submodule's output, so
    the perturbation propagates through everything downstream exactly as the real
    value would — including through ``detach()``, which is the entire point.
    """
    import torch

    def _resolve(path: str):
        m = model
        for part in path.split("."):
            m = getattr(m, part)
        return m

    def run(intervention=None):
        tgt_name, fn = (intervention or (None, None))
        seen: dict = {}
        handles = []
        use_batch = dict(batch)
        for role, key in (input_nodes or {}).items():
            val = use_batch[key]
            if role == tgt_name:
                gen = torch.Generator(device="cpu").manual_seed(20260818)
                val = fn(val, gen)
                use_batch[key] = val
            seen[role] = val

        def _mk(role, is_target):
            def hook(_mod, _inp, out):
                val = out
                if is_target:
                    gen = torch.Generator(device="cpu").manual_seed(20260818)
                    if isinstance(out, torch.Tensor):
                        val = fn(out, gen)
                    elif isinstance(out, dict):
                        val = {k: (fn(v, gen) if isinstance(v, torch.Tensor)
                                   else v) for k, v in out.items()}
                seen[role] = val
                return val
            return hook

        try:
            for role, path in nodes.items():
                handles.append(_resolve(path).register_forward_hook(
                    _mk(role, role == tgt_name)))
            with torch.no_grad():
                # ⛔ ``use_batch``, NOT ``batch``. This line read ``batch`` when
                # first written, so an input perturbation was computed, recorded
                # as the node's value, and then NEVER PASSED TO THE MODEL — the
                # input probe was INERT BY CONSTRUCTION and reported every input
                # as unread, including ``frames``. It was caught by an
                # implausible reading (a vision-derived goal path that did not
                # depend on vision), not by the positive control, which covered
                # only the submodule-intervention path. ⇒ A control that powers
                # one code path does NOT power another; see
                # ``test_the_INPUT_probe_has_its_own_positive_control``.
                out = model(**use_batch)
            for role, getter in (output_nodes or {}).items():
                seen[role] = getter(out)
            return seen
        finally:
            for h in handles:
                h.remove()

    return run


# --------------------------------------------------------------------------- #
# numeric core                                                                #
# --------------------------------------------------------------------------- #
def _flat(v) -> list:
    """Flatten a node value (Tensor / dict / sequence) to a list of floats."""
    import torch
    if v is None:
        return []
    if isinstance(v, torch.Tensor):
        return v.detach().reshape(-1).float().tolist()
    if isinstance(v, Mapping):
        out: list = []
        for k in sorted(v):
            out.extend(_flat(v[k]))
        return out
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_flat(x))
        return out
    if isinstance(v, (int, float)):
        return [float(v)]
    return []


def _delta(a, b) -> dict:
    """Max-abs and relative change between two node values."""
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb) or not fa:
        # A shape change IS a dependency (the strongest possible one).
        return {"max_abs": float("inf") if fa or fb else 0.0,
                "rel": float("inf") if fa or fb else 0.0, "n": len(fa),
                "shape_changed": len(fa) != len(fb)}
    m = max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)
    scale = max((abs(x) for x in fa), default=0.0)
    return {"max_abs": float(m), "n": len(fa), "shape_changed": False,
            "rel": float(m / scale) if scale > 0 else (
                float("inf") if m > 0 else 0.0)}


def determinism_check(run: Callable, *, reps: int = 2) -> dict:
    """Two clean passes must agree BIT-FOR-BIT before any verdict is emitted.

    ⚠️ This is not ceremony. A nondeterministic forward (dropout left on, a
    non-deterministic kernel, an uninitialised buffer) makes every target differ
    between passes, which reads as *"everything depends on everything"* — an
    audit that fails loudly in the safe direction but is still WRONG, and would
    burn a day. If this check fails the audit is ``UNPOWERED``, not ``leaky``.

    ⚠️ SCOPE, stated: it covers the OBSERVED nodes — exactly the nodes the audit
    gives verdicts about. Nondeterminism strictly downstream of every observed
    node cannot affect a verdict and is deliberately out of scope.
    """
    base = run(None)
    worst, offender = 0.0, None
    for _ in range(max(1, reps - 1)):
        other = run(None)
        for k in base:
            d = _delta(base[k], other.get(k))["max_abs"]
            if d > worst:
                worst, offender = d, k
    return {"deterministic": worst == 0.0, "max_drift": worst,
            "worst_node": offender, "nodes": sorted(base),
            "_reads": ("a non-deterministic forward makes every node look "
                       "dependent on every other; no verdict is admissible "
                       "until this is exactly 0.0")}


def probe_dependency(run: Callable, source: str, targets: Sequence[str], *,
                     kinds: Iterable[str] = ("noise", "zero", "sign",
                                             "shuffle", "offset"),
                     atol: float = 0.0) -> dict:
    """Does intervening on ``source`` change any of ``targets``?

    Returns per-target evidence. A target is ``depends=True`` if ANY
    perturbation kind moves it by more than ``atol`` — the disjunction, because
    each kind is blind to a different wiring (see :data:`PERTURBATIONS`).

    ``atol=0.0`` by default: this is an interventional probe on a deterministic
    forward, so *any* movement at all is a real information path. A tolerance
    would be a way to lose a small leak, and the C92 leak (ego speed through the
    latent) was small.
    """
    clean = run(None)
    if source not in clean:
        raise KeyError(f"source node {source!r} not observable; "
                       f"have {sorted(clean)}")
    missing = [t for t in targets if t not in clean]
    if missing:
        raise KeyError(f"target nodes not observable: {missing}; "
                       f"have {sorted(clean)}")

    per_target: dict[str, dict] = {
        t: {"depends": False, "max_abs": 0.0, "rel": 0.0,
            "kinds_that_moved": []} for t in targets}
    used = []
    for kind in kinds:
        fn = PERTURBATIONS[kind]
        used.append(kind)
        pert = run((source, fn))
        # ⚠️ SANITY: the perturbation must actually have changed the source. A
        # perturbation that is a no-op on this batch (``zero`` on an
        # already-zero tensor) proves nothing about the targets, so it is not
        # allowed to contribute a clean reading.
        moved_src = _delta(clean[source], pert[source])["max_abs"] > 0.0
        for t in targets:
            d = _delta(clean[t], pert[t])
            rec = per_target[t]
            rec["max_abs"] = max(rec["max_abs"], d["max_abs"])
            rec["rel"] = max(rec["rel"], d["rel"])
            if moved_src and d["max_abs"] > atol:
                rec["depends"] = True
                rec["kinds_that_moved"].append(kind)
        per_target.setdefault("_src", {})
        per_target["_src"].setdefault("effective_kinds", [])
        if moved_src:
            per_target["_src"]["effective_kinds"].append(kind)

    src_meta = per_target.pop("_src", {"effective_kinds": []})
    return {
        "source": source,
        "n_probes": len(used), "kinds": used,
        "effective_kinds": src_meta["effective_kinds"],
        "source_was_perturbable": bool(src_meta["effective_kinds"]),
        "targets": per_target,
        "any_dependence": any(v["depends"] for v in per_target.values()),
        "_reads": ("depends=True means the target's VALUE changed when the "
                   "source's value was replaced mid-forward: a live forward "
                   "information path, regardless of detach()."),
    }


def dependency_matrix(run: Callable, roles: Sequence[str], **kw) -> dict:
    """The full ``source -> target`` matrix over ``roles``. BOTH DIRECTIONS.

    The reverse-direction audit (does the SITUATION path read the goal?) is not
    a second study — it is this matrix read down the other axis, which is why
    it costs nothing and why leaving it unchecked was never justified.
    """
    cells: dict[str, dict] = {}
    for s in roles:
        others = [t for t in roles if t != s]
        if not others:
            continue
        cells[s] = probe_dependency(run, s, others, **kw)
    edges = sorted(
        (s, t) for s, rep in cells.items()
        for t, v in rep["targets"].items() if v["depends"])
    # ⭐ DEAD NODES. A target that NO source moved is constant on this batch, and
    # a constant node reads as independent of everything — including of a signal
    # it is genuinely wired to. MEASURED on the real v6 stack: ``emission``'s
    # final layer is ZERO-INIT for the CV warm start (``v6.py:4534``), so on a
    # freshly built model the vision path into it is live but dead-valued and
    # every input except ``v0`` reads as unread. Reporting that as "disjoint"
    # would be a vacuous pass, so it is surfaced and the audit calls it
    # UNPOWERED for those nodes.
    moved = {t for _s, t in edges}
    all_targets = {t for rep in cells.values() for t in rep["targets"]}
    return {"roles": list(roles), "cells": cells,
            "edges": [list(e) for e in edges],
            "n_edges": len(edges),
            "unmoved_targets": sorted(all_targets - moved),
            "_unmoved_reads": ("a node no probe could move is CONSTANT on this "
                               "batch; disjointness cannot be established for "
                               "it — typically a zero-init head on an "
                               "untrained model")}


def classify_edge(matrix: dict, source: str, target: str,
                  shared_input_probe: dict | None = None) -> dict:
    """⭐ DIRECT PATH vs COMMON ANCESTOR — the shared-trunk disclosure.

    The ruling requires that *"where a shared trunk feeds both paths, say so and
    state why that is not a back door"*. This is that statement, computed:

      * ``DIRECT_PATH`` — intervening on ``source`` moves ``target``. ``target``
        READS ``source``. For (situation_output -> goal) this is the violation.
      * ``COMMON_ANCESTOR`` — intervening on ``source`` leaves ``target``
        bit-identical, but a shared upstream input moves both. They COVARY and
        neither reads the other. **This is not a back door**, and the reason is
        mechanical: a back door requires an information path FROM the
        classifier's output INTO the goal, and the intervention shows there is
        none — the covariation is carried by the ancestor, which both are
        independently entitled to read (vision).
      * ``INDEPENDENT`` — neither.
    """
    cell = matrix["cells"].get(source, {})
    direct = bool(cell.get("targets", {}).get(target, {}).get("depends"))
    shared = None
    if shared_input_probe is not None:
        tg = shared_input_probe.get("targets", {})
        shared = bool(tg.get(source, {}).get("depends")) and \
            bool(tg.get(target, {}).get("depends"))
    kind = ("DIRECT_PATH" if direct else
            "COMMON_ANCESTOR" if shared else "INDEPENDENT")
    return {
        "source": source, "target": target, "relation": kind,
        "direct": direct, "shares_ancestor": shared,
        "_reads": {
            "DIRECT_PATH": "target READS source — a forward information path.",
            "COMMON_ANCESTOR": ("target does NOT read source; both descend "
                                "from a shared input. Covariation without a "
                                "path. NOT a back door — the intervention "
                                "shows no signal flows source->target."),
            "INDEPENDENT": "no path and no measured shared ancestor.",
        }[kind],
    }


# --------------------------------------------------------------------------- #
# the per-arm audit and the gate                                              #
# --------------------------------------------------------------------------- #
def audit_arm(*, arm: str, run: Callable, goal_roles: Sequence[str],
              situation_roles: Sequence[str],
              shared_input_role: str | None = None,
              positive_control: dict | None = None, **kw) -> dict:
    """Audit ONE arm, both directions, with its OWN control.

    ⚠️ ``arm`` and ``positive_control`` are per-arm ON PURPOSE (C107: a control
    run once for a study left **33 of 165 rows** with no control at all). A
    control established on a different arm's graph says nothing about this one's,
    so this function will not inherit one.

    ``positive_control`` is the report from :func:`probe_dependency` (or
    :func:`audit_arm`) on a DELIBERATELY WIRED variant of the same graph. It must
    show a live dependence, or the audit is ``UNPOWERED``: an all-clean matrix
    from a probe that cannot detect anything is not evidence of disjointness.
    """
    det = determinism_check(run)
    roles = list(dict.fromkeys([*goal_roles, *situation_roles]))
    if not det["deterministic"]:
        return {"arm": arm, "status": "UNPOWERED", "verdict": None,
                "determinism": det,
                "reason": (f"forward is non-deterministic (max drift "
                           f"{det['max_drift']} at {det['worst_node']!r}); "
                           f"every node would read as dependent")}

    matrix = dependency_matrix(run, roles, **kw)
    shared_probe = None
    if shared_input_role is not None:
        shared_probe = probe_dependency(run, shared_input_role, roles, **kw)

    forward_edges, reverse_edges = [], []
    for s in situation_roles:
        for t in goal_roles:
            e = classify_edge(matrix, s, t, shared_probe)
            (forward_edges if e["direct"] else forward_edges).append(e)
    for s in goal_roles:
        for t in situation_roles:
            reverse_edges.append(classify_edge(matrix, s, t, shared_probe))

    viol_fwd = [e for e in forward_edges if e["direct"]]
    viol_rev = [e for e in reverse_edges if e["direct"]]

    ctrl_ok = None
    if positive_control is not None:
        ctrl_ok = bool(positive_control.get("any_dependence"))

    # A perturbation that never moved its own source proves nothing.
    inert = [s for s, c in matrix["cells"].items()
             if not c["source_was_perturbable"]]
    # ⛔ A goal node that nothing could move is constant on this batch — its
    # clean reading is an artefact, not a finding. But deadness may ONLY be
    # judged against a source that SHOULD move the node, which is the shared
    # input. Judging it from ``matrix["unmoved_targets"]`` is WRONG and was a
    # measured false positive: in a genuinely DISJOINT pair neither node moves
    # the other by definition, so both read as "dead" and the instrument called
    # its own success an artefact. Pinned by
    # ``test_a_DISJOINT_pair_is_not_mistaken_for_a_DEAD_one``.
    dead = ([r for r in goal_roles
             if not shared_probe["targets"].get(r, {}).get("depends")]
            if shared_probe is not None else [])

    if positive_control is not None and not ctrl_ok:
        status, verdict = "UNPOWERED", None
        reason = ("the positive control did not fire — this probe could not "
                  "have detected a leak on this arm, so its clean matrix is "
                  "not evidence (C109: an inert control proves nothing)")
    elif inert:
        status, verdict = "UNPOWERED", None
        reason = (f"no perturbation moved {inert} on this batch; a node that "
                  f"cannot be perturbed cannot be shown to be unread")
    elif viol_fwd or viol_rev:
        status, verdict = "MEASURED", "INADMISSIBLE"
        reason = ("a forward information path was measured between the goal "
                  "path and the situation classifier's output")
    elif dead:
        status, verdict = "UNPOWERED", None
        reason = (f"goal nodes {dead} are CONSTANT on this batch — nothing "
                  f"moved them, so their clean reading is an artefact "
                  f"(typically a zero-init head on an untrained model), not "
                  f"a measured disjointness")
    else:
        status, verdict = "MEASURED", "DISJOINT"
        reason = ("no intervention on either side moved the other; the paths "
                  "are information-disjoint on this graph and batch")

    shared_note = None
    if shared_probe is not None:
        co = [r for r in roles
              if shared_probe["targets"].get(r, {}).get("depends")]
        if len(co) > 1:
            shared_note = {
                "shared_input": shared_input_role, "roles_it_feeds": co,
                "is_back_door": bool(viol_fwd or viol_rev),
                "_statement": (
                    f"DISCLOSED SHARED TRUNK: {shared_input_role!r} feeds "
                    f"{co}. This is a COMMON ANCESTOR, not a back door: a back "
                    f"door needs a path FROM the situation output INTO the "
                    f"goal, and intervening on the situation output leaves "
                    f"every goal node bit-identical. Both paths are "
                    f"independently entitled to read vision."),
            }

    return {
        "arm": arm, "status": status, "verdict": verdict, "reason": reason,
        "determinism": det,
        "goal_roles": list(goal_roles), "situation_roles": list(situation_roles),
        "forward_violations": viol_fwd, "reverse_violations": viol_rev,
        "forward_edges": forward_edges, "reverse_edges": reverse_edges,
        "dead_goal_nodes": dead,
        "shared_trunk": shared_note,
        "positive_control": (None if positive_control is None else
                             {"fired": ctrl_ok,
                              "source": positive_control.get("source"),
                              "kinds": positive_control.get("effective_kinds")}),
        "matrix": matrix,
        "_note": ("REVERSE DIRECTION IS INCLUDED. The ruling says "
                  "information-disjoint, which is symmetric; every check "
                  "written before 2026-08-18 asked only goal<-situation."),
    }


def audit_arms(specs: Sequence[Mapping]) -> dict:
    """Audit MANY arms. Every arm carries its own verdict and its own control.

    Returns a report whose ``verdict`` is the WEAKEST across arms — one
    inadmissible arm makes the panel inadmissible, and one unpowered arm makes
    the panel unpowered rather than clean.
    """
    rows = [audit_arm(**dict(s)) for s in specs]
    if any(r["verdict"] == "INADMISSIBLE" for r in rows):
        overall = "INADMISSIBLE"
    elif any(r["status"] == "UNPOWERED" for r in rows):
        overall = "UNPOWERED"
    else:
        overall = "DISJOINT"
    return {
        "n_arms": len(rows), "verdict": overall,
        "arms": {r["arm"]: r for r in rows},
        "unpowered": [r["arm"] for r in rows if r["status"] == "UNPOWERED"],
        "inadmissible": [r["arm"] for r in rows
                         if r["verdict"] == "INADMISSIBLE"],
        "_note": ("per-arm, never per-study: a control run once for a study "
                  "left 33 of 165 rows with no control at all (C107)"),
    }


def assert_information_disjoint(report: Mapping, *,
                                allow_unpowered: bool = False) -> dict:
    """⛔ THE GATE. Raise unless every arm measured DISJOINT.

    ``allow_unpowered=False`` by default and that is the load-bearing choice: an
    UNPOWERED audit is *"we did not establish it"*, and letting that pass is how
    a guard becomes decoration. The instrument this one supersedes had **zero
    call sites for 12 days** precisely because nothing ever raised.
    """
    arms = report.get("arms", {}) if "arms" in report else {
        report.get("arm", "?"): report}
    bad, weak = [], []
    for name, r in arms.items():
        if r.get("verdict") == "INADMISSIBLE":
            ev = r.get("forward_violations", []) + r.get("reverse_violations",
                                                         [])
            bad.append(f"{name}: " + "; ".join(
                f"{e['source']} -> {e['target']}" for e in ev))
        elif r.get("status") == "UNPOWERED":
            weak.append(f"{name}: {r.get('reason')}")
    if bad:
        raise ProvenanceViolation(
            "GOAL-PROVENANCE VIOLATION — a forward information path exists "
            "between the goal path and the situation classifier's output, "
            "which the PI ruling of 2026-08-03 forbids in any form:\n  "
            + "\n  ".join(bad)
            + "\n(This is an INFERENCE-time information check. It is NOT the "
              "gradient check: V6Stack.assert_isolation measures backward "
              "edges and a detach()ed wire passes it while leaking fully.)")
    if weak and not allow_unpowered:
        raise ProvenanceViolation(
            "GOAL-PROVENANCE UNPOWERED — disjointness was NOT established:\n  "
            + "\n  ".join(weak)
            + "\n(Absence found by a probe that could not have detected the "
              "thing is not absence. Pass allow_unpowered=True only to record "
              "a known-unpowered state deliberately.)")
    return {"ok": True, "n_arms": len(arms),
            "verdict": report.get("verdict", "DISJOINT")}


# --------------------------------------------------------------------------- #
# the discriminating demonstration                                            #
# --------------------------------------------------------------------------- #
def compare_to_gradient_probe(model, batch: Mapping, run: Callable, *,
                              source: str, source_params: Sequence,
                              target_roles: Sequence[str],
                              target_output_keys: Sequence[str]) -> dict:
    """⭐ Show the two probes DISAGREE on a detached wire — the reason this
    module is not redundant with ``V6Stack.assert_isolation``.

    Both probes are pointed at the SAME question — *does the goal read the
    situation path?* — on the same graph and batch, and differ only in what
    they can see:

      * FORWARD (this module): intervene on the SOURCE's VALUE, ask whether the
        target roles change.
      * BACKWARD (``assert_isolation``'s shape): differentiate the TARGET's
        output w.r.t. the SOURCE MODULE's parameters, ask which received
        gradient.

    ⚠️ ``target_output_keys`` selects the goal outputs ONLY. Summing every
    output instead would give the situation head gradient from its own output
    and report a path that has nothing to do with the goal — a mistake this
    function made until the test caught it, and exactly the *"a probe that
    reports the wrong scope"* class the project's ``df``/Thor rules name.
    """
    import torch
    fwd = probe_dependency(run, source, target_roles)
    pairs = [(n, p) for n, p in source_params if p.requires_grad]
    live: list[str] = []
    if pairs:
        with torch.enable_grad():
            out = model(**batch)
            if not isinstance(out, Mapping):
                raise TypeError("compare_to_gradient_probe needs a dict output "
                                "so the target keys can be selected")
            missing = [k for k in target_output_keys if k not in out]
            if missing:
                raise KeyError(f"target_output_keys not in model output: "
                               f"{missing}; have {sorted(out)}")
            flat = [out[k].float().reshape(-1) for k in target_output_keys]
            grads = torch.autograd.grad(torch.cat(flat).sum(),
                                        [p for _, p in pairs],
                                        allow_unused=True,
                                        materialize_grads=False)
            live = [n for (n, _), g in zip(pairs, grads)
                    if g is not None and float(g.abs().max()) > 0.0]
    return {
        "question": (f"does {target_roles} read {source!r}?"),
        "forward_information_path": bool(fwd["any_dependence"]),
        "backward_gradient_path": bool(live),
        "live_grad_params": live,
        "probes_disagree": bool(fwd["any_dependence"]) != bool(live),
        "forward_detail": fwd,
        "_reads": ("probes_disagree=True on a detached wire is the finding: "
                   "the gradient probe certifies clean while the information "
                   "path is total. Admissibility is a FORWARD question."),
    }
