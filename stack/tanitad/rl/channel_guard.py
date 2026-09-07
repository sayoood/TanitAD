"""REFUSE an RL launch whose adapter has drifted behind the model's forward.

WHY THIS IS A LAUNCH-PREFLIGHT REFUSAL AND NOT ONLY A TEST
==========================================================
``tanitad.rl.refc_adapter.forward_kwargs`` builds its kwargs by iterating
``FORWARD_KEYS``::

    kw = {k: batch.get(k) for k in FORWARD_KEYS}

A conditioning channel that is absent from that tuple is therefore **never
passed at all** -- no error, no warning, no missing-key exception. The model runs
with the channel at its default and the arm is silently blind to it.

MEASURED 2026-09-06, and this is why a passing test suite is not enough:

* the live ``RefCV3Model.forward`` accepted **9** optional conditioning channels;
* the **worktree** ``FORWARD_KEYS`` carried **7** -- missing ``gp_point`` and
  ``gp_valid``;
* ``HEAD``'s blob carried **6** -- ALSO missing ``agent_gt``, so an arm launched
  from HEAD ran agent-blind.

The HEAD regression arrived under a commit whose subject was *"make the suite
carry the FORWARD_KEYS drift"*: the tree it was built from predated the
``agent_gt`` addition, so landing the guard silently reverted the fix. That is
the documented scratch-index-from-a-stale-tree failure, and it means **the state
of the tuple at LAUNCH TIME is not implied by the state of the suite at REVIEW
TIME.** A test asserts what was true when someone ran it. This module asserts
what is true in the process that is about to spend GPU.

⛔⛔ WHAT THIS MODULE GOT WRONG, AND WHAT FIXED IT (2026-09-07)
==============================================================
The version above compared ``FORWARD_KEYS`` against the RAW signature and had **no
exclusion concept at all**. MEASURED on the live pair that day: the forward had
grown to **12** channels, the adapter carried **7**, and
:func:`assert_forward_channels_complete` therefore **refused every RL launch**,
naming ``gp_point, gp_valid, nav_args, v_max_ms, v_max_valid`` as *"silently
dropped"* and telling the operator the arm *"runs blind to"* them.

Two of those five are the goal point. The goal-point stream had already ruled that
plumbing them **manufactures a label leak** -- their only supplier is the ego's own
future pose. So the guard was not merely over-refusing: **its refusal text
instructed the operator to create the leak the programme's vision-only rule exists
to prevent**, and the suite was green on it, because its live-pair test accepted
"refuses" and "passes" as equally correct outcomes.

⇒ A drift guard cannot be built on "the adapter must plumb everything the forward
accepts". Some channels MUST NOT be plumbed. The requirement is
``signature - declared exclusions``, and the exclusions are declared by the seams
that own the channels, each with its reason, its evidence and its unblock
condition (``tanitad.channel_admissibility``). This module reads that union; it
does not hold a list of its own, and must not grow one.

THE THREE DEFECTS THIS GUARD NOW REFUSES
========================================
    MISSING    a required channel absent from ``FORWARD_KEYS`` -- the arm runs
               blind to it, silently. (the original defect)
    LEAKED     a DECLARED-EXCLUDED channel present in ``FORWARD_KEYS`` -- the arm
               would be fed a label at rollout time. (the defect the original
               guard actively recommended)
    STALE      a declared exclusion naming a channel the forward no longer has --
               the exclusion covers nothing, and would keep covering nothing if a
               different channel later took that name.

plus ``extra_in_adapter``, a key the forward does not accept, which is a
``TypeError`` at the first step.

OWNERSHIP: this module belongs to the RL stream and reads ``refc_adapter``,
``refc_v3`` and the seams' declarations WITHOUT importing anything that mutates
them. It never edits any of them.

Tier: none -- ⛔ this is a WIRING-CONTRACT surface. No model produces a trajectory
here, so an eval tier or a four-family metric table would be a category error.
Evidence class: PUBLISHED-CODE (the live signature and the seams' declarations,
read at call time).
"""
from __future__ import annotations

import inspect

__all__ = ["ChannelDriftError", "ChannelLeakError", "NON_CHANNEL_PARAMS",
           "forward_optional_channels", "declared_exclusions",
           "forward_channel_report", "assert_forward_channels_complete"]


class ChannelDriftError(RuntimeError):
    """The adapter's FORWARD_KEYS disagrees with the model's forward signature."""


class ChannelLeakError(ChannelDriftError):
    """FORWARD_KEYS plumbs a channel a seam declared must not reach a rollout.

    ⚠️ A subclass of :class:`ChannelDriftError` so an existing ``except`` still
    catches it, and a distinct type so a caller that wants to tell "the arm is
    blind" from "the arm is being fed the answer" can. They are opposite defects
    and only one of them silently improves the metric.
    """


#: Parameters of ``RefCV3Model.forward`` that are NOT optional conditioning
#: channels: the adapter passes these positionally/explicitly rather than
#: through ``FORWARD_KEYS``.
NON_CHANNEL_PARAMS = frozenset({"self", "frames", "steps"})


def forward_optional_channels(forward=None):
    """Every optional conditioning channel the LIVE forward accepts.

    Derived from the signature at call time, so it cannot go stale the way a
    duplicated tuple can.
    """
    if forward is None:
        from tanitad.refs.refc_v3 import RefCV3Model
        forward = RefCV3Model.forward
    sig = inspect.signature(forward)
    return tuple(
        name for name, p in sig.parameters.items()
        if name not in NON_CHANNEL_PARAMS
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       inspect.Parameter.KEYWORD_ONLY))


def declared_exclusions(exclusions=None):
    """The channels the SEAMS declare must not reach an RL rollout.

    ⛔ Read from ``tanitad.channel_admissibility`` at call time, never held here.
    A copy in this module would be the same hand-maintained mirror the whole guard
    exists to abolish, one level up.

    ``exclusions`` is an injection point for the mutation tests -- pass a tuple of
    :class:`~tanitad.channel_admissibility.ChannelExclusion`, or ``()`` for a
    fixture forward that has no real seams behind it.
    """
    if exclusions is None:
        from tanitad.channel_admissibility import forward_exclusions
        return forward_exclusions()
    return tuple(exclusions)


def forward_channel_report(forward=None, keys=None, exclusions=None):
    """Compare the adapter's tuple against the live signature. Never raises.

    ``required`` is ``signature - declared exclusions``: the channels an RL rollout
    both MAY and MUST be given.
    """
    if keys is None:
        from tanitad.rl import refc_adapter
        keys = tuple(refc_adapter.FORWARD_KEYS)
    sig_channels = forward_optional_channels(forward)
    decls = declared_exclusions(exclusions)
    excluded = tuple(d.channel for d in decls)

    required = tuple(c for c in sig_channels if c not in excluded)
    missing = tuple(c for c in required if c not in keys)
    extra = tuple(k for k in keys if k not in sig_channels)
    leaked = tuple(c for c in excluded if c in keys)
    stale = tuple(c for c in excluded if c not in sig_channels)
    return {
        "signature_channels": list(sig_channels),
        "adapter_forward_keys": list(keys),
        "declared_excluded": list(excluded),
        "required_channels": list(required),
        "n_signature_channels": len(sig_channels),
        "n_adapter_keys": len(keys),
        "n_excluded": len(excluded),
        "n_required": len(required),
        "missing_from_adapter": list(missing),
        "extra_in_adapter": list(extra),
        "leaked_into_adapter": list(leaked),
        "stale_exclusions": list(stale),
        #: banked into the preflight artifact, so a reviewer reading the run record
        #: sees WHY a channel was withheld without opening three modules
        "exclusion_declarations": [d.to_dict() for d in decls],
        "complete": bool(not missing and not extra and not leaked and not stale),
    }


def assert_forward_channels_complete(forward=None, keys=None, exclusions=None):
    """REFUSE a launch whose adapter is blind to a channel, or fed an excluded one.

    Returns the report on success so a caller can bank it; raises
    :class:`ChannelDriftError` (or :class:`ChannelLeakError`) naming the channels
    and quoting the owning seam's reason otherwise.
    """
    from tanitad.channel_admissibility import format_exclusions

    rep = forward_channel_report(forward, keys, exclusions)
    decls = declared_exclusions(exclusions)
    by_chan = {d.channel: d for d in decls}

    # ⛔ THE LEAK DIRECTION FIRST. It is the more dangerous of the two: a blind arm
    # under-performs visibly, while an arm fed the answer over-performs and reads
    # as a win. If both defects are present the operator must see this one.
    if rep["leaked_into_adapter"]:
        lines = []
        for c in rep["leaked_into_adapter"]:
            d = by_chan[c]
            lines.append(f"  {c} -- declared by {d.owner}\n"
                         f"    reason : {d.reason}\n"
                         f"    unblock: {d.unblock}")
        raise ChannelLeakError(
            "refc_adapter.FORWARD_KEYS plumbs %s, which the owning seam declared "
            "MUST NOT reach an RL rollout. forward_kwargs() would feed it from the "
            "batch at every step, and the batch's only supplier for it is the "
            "ego's own future -- i.e. the LABEL. Refusing at preflight; an arm in "
            "this state would report a capability it was handed.\n%s"
            % (", ".join(rep["leaked_into_adapter"]), "\n".join(lines)))

    if rep["stale_exclusions"]:
        raise ChannelDriftError(
            "%s is declared must-not-be-plumbed but RefCV3Model.forward no longer "
            "accepts it. The exclusion is STALE and is now covering nothing -- and "
            "would keep covering nothing if a DIFFERENT channel later took that "
            "name, which is how a guard turns into decoration. Update the "
            "declaration in the owning seam or delete it. (signature: %s)"
            % (", ".join(rep["stale_exclusions"]),
               ", ".join(rep["signature_channels"])))

    if rep["missing_from_adapter"]:
        raise ChannelDriftError(
            "refc_adapter.FORWARD_KEYS has drifted BEHIND RefCV3Model.forward: the "
            "signature accepts %d optional conditioning channels, %d are declared "
            "must-not-be-plumbed, so %d are REQUIRED -- and the adapter plumbs %d. "
            "forward_kwargs() would SILENTLY DROP %s, and an arm launched in this "
            "state runs blind to %s. Refusing at preflight.\n"
            "\n"
            "⛔ BEFORE ADDING A KEY, ask where an RL rollout would GET the value. "
            "If the only supplier is the ego's future path, it is a LABEL: declare "
            "it in the owning seam with its reason and its unblock condition "
            "(tanitad.channel_admissibility) instead of plumbing it.\n"
            "\n"
            "  signature : %s\n"
            "  excluded  : %s\n"
            "  required  : %s\n"
            "  adapter   : %s\n"
            "\n"
            "already declared must-not-be-plumbed:\n%s"
            % (rep["n_signature_channels"], rep["n_excluded"], rep["n_required"],
               rep["n_adapter_keys"],
               ", ".join(rep["missing_from_adapter"]),
               " and ".join(rep["missing_from_adapter"]),
               ", ".join(rep["signature_channels"]),
               ", ".join(rep["declared_excluded"]) or "(none)",
               ", ".join(rep["required_channels"]),
               ", ".join(rep["adapter_forward_keys"]),
               format_exclusions(decls) or "  (none)"))

    if rep["extra_in_adapter"]:
        raise ChannelDriftError(
            "refc_adapter.FORWARD_KEYS names %s, which RefCV3Model.forward does "
            "NOT accept -- forward_kwargs() would raise TypeError at the first "
            "step. Refusing at preflight. (signature: %s | adapter: %s)"
            % (", ".join(rep["extra_in_adapter"]),
               ", ".join(rep["signature_channels"]),
               ", ".join(rep["adapter_forward_keys"])))
    return rep
