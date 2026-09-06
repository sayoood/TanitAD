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

* the live ``RefCV3Model.forward`` accepts **9** optional conditioning channels;
* the **worktree** ``FORWARD_KEYS`` carried **7** -- missing ``gp_point`` and
  ``gp_valid``, so an arm launched from it runs GOAL-BLIND;
* ``HEAD``'s blob carried **6** -- ALSO missing ``agent_gt``, so an arm launched
  from HEAD runs goal-blind **and** agent-blind.

The HEAD regression arrived under a commit whose subject was *"make the suite
carry the FORWARD_KEYS drift"*: the tree it was built from predated the
``agent_gt`` addition, so landing the guard silently reverted the fix. That is
the documented scratch-index-from-a-stale-tree failure, and it means **the state
of the tuple at LAUNCH TIME is not implied by the state of the suite at REVIEW
TIME.** A test asserts what was true when someone ran it. This module asserts
what is true in the process that is about to spend GPU.

THE RULE THIS ENCODES
=====================
Do not hand-maintain a mirror of a signature. DERIVE the requirement from the
signature at call time and refuse when the mirror is short. The refusal names
the missing channels, so the message is actionable rather than an assertion
failure with a tuple in it.

OWNERSHIP: this module belongs to the RL stream and reads ``refc_adapter`` and
``refc_v3`` WITHOUT importing anything that mutates them. It never edits either
file -- ``refc_adapter`` and ``refc.py`` are another stream's.
"""
from __future__ import annotations

import inspect

__all__ = ["ChannelDriftError", "NON_CHANNEL_PARAMS", "forward_optional_channels",
           "forward_channel_report", "assert_forward_channels_complete"]


class ChannelDriftError(RuntimeError):
    """The adapter's FORWARD_KEYS is short of the model's forward signature."""


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


def forward_channel_report(forward=None, keys=None):
    """Compare the adapter's tuple against the live signature. Never raises."""
    if keys is None:
        from tanitad.rl import refc_adapter
        keys = tuple(refc_adapter.FORWARD_KEYS)
    sig_channels = forward_optional_channels(forward)
    missing = tuple(c for c in sig_channels if c not in keys)
    extra = tuple(k for k in keys if k not in sig_channels)
    return {
        "signature_channels": list(sig_channels),
        "adapter_forward_keys": list(keys),
        "n_signature_channels": len(sig_channels),
        "n_adapter_keys": len(keys),
        "missing_from_adapter": list(missing),
        "extra_in_adapter": list(extra),
        "complete": bool(not missing and not extra),
    }


def assert_forward_channels_complete(forward=None, keys=None):
    """REFUSE if the adapter would drop a channel the forward accepts.

    Returns the report on success so a caller can bank it; raises
    :class:`ChannelDriftError` naming the missing channels otherwise.
    """
    rep = forward_channel_report(forward, keys)
    if rep["missing_from_adapter"]:
        raise ChannelDriftError(
            "refc_adapter.FORWARD_KEYS has drifted BEHIND RefCV3Model.forward: "
            "the signature accepts %d optional conditioning channels, the adapter "
            "plumbs %d, and forward_kwargs() would SILENTLY DROP %s. An arm "
            "launched in this state runs blind to %s. Refusing at preflight. "
            "(signature: %s | adapter: %s)"
            % (rep["n_signature_channels"], rep["n_adapter_keys"],
               ", ".join(rep["missing_from_adapter"]),
               " and ".join(rep["missing_from_adapter"]),
               ", ".join(rep["signature_channels"]),
               ", ".join(rep["adapter_forward_keys"])))
    if rep["extra_in_adapter"]:
        raise ChannelDriftError(
            "refc_adapter.FORWARD_KEYS names %s, which RefCV3Model.forward does "
            "NOT accept -- forward_kwargs() would raise TypeError at the first "
            "step. Refusing at preflight. (signature: %s | adapter: %s)"
            % (", ".join(rep["extra_in_adapter"]),
               ", ".join(rep["signature_channels"]),
               ", ".join(rep["adapter_forward_keys"])))
    return rep
