"""FORWARD_KEYS must cover every optional channel RefCV3Model.forward accepts.

WHY THIS TEST EXISTS (and why it is not a style check):

``tanitad.rl.refc_adapter.forward_kwargs`` builds its kwargs by iterating
``FORWARD_KEYS``.  A channel that is absent from that tuple is therefore
**never passed at all** -- no error, no warning, no missing-key exception.
The model simply runs with the channel at its default and the arm is blind
to it.

This has now happened TWICE:

  1. ``agent_gt`` was appended to the forward by the refcv5 WP-6 agent seam
     and not added here.  An ``--agents oracle`` build driven through this
     adapter tripped the model's own "no agent_gt reached the forward" guard
     even though the batch carried it.  ``refc_adapter``'s own module
     docstring records that incident.

  2. ``gp_point`` / ``gp_valid`` were appended by the E15 goal-point seam and
     not added here.  MEASURED 2026-09-06: the signature carries 8 optional
     channels, the tuple carries 7, and the two goal-point channels are the
     difference -- so an RL arm launched in that state runs GOAL-BLIND.

The lesson is not "be more careful when editing the tuple".  A hand-maintained
mirror of a signature drifts, and both drifts were silent.  The durable fix is
to DERIVE the requirement from the signature, which is what this test does.

Same class as ``test_the_TRAINER_ACTUALLY_CALLS_the_preflight_on_BOTH_launch_paths``:
correctness and wiring are different claims, and only the first had a test.
``test_rl_refc_adapter_robust.py`` checks that the adapter behaves correctly
GIVEN its tuple; nothing checked that the tuple still matched reality.
"""

import inspect

import pytest


# Parameters of RefCV3Model.forward that are NOT optional conditioning channels.
# ``frames`` and ``steps`` are passed positionally/explicitly by the adapter
# rather than through FORWARD_KEYS.
NON_CHANNEL_PARAMS = frozenset({"self", "frames", "steps"})


def _optional_channels():
    """Every optional conditioning channel the live forward accepts.

    Derived from the signature at call time, so it cannot go stale the way a
    duplicated tuple can.
    """
    from tanitad.refs.refc_v3 import RefCV3Model

    sig = inspect.signature(RefCV3Model.forward)
    return [
        name
        for name, p in sig.parameters.items()
        if name not in NON_CHANNEL_PARAMS
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def test_forward_keys_covers_every_optional_channel():
    """The load-bearing assertion.

    Fails loudly, naming the missing channels and the one-line fix, rather than
    letting an RL arm run blind to them.
    """
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    channels = _optional_channels()
    assert channels, (
        "no optional channels found on RefCV3Model.forward -- the introspection "
        "in this test is broken, not the adapter"
    )

    missing = [c for c in channels if c not in FORWARD_KEYS]
    assert not missing, (
        "FORWARD_KEYS is missing {n} channel(s) that RefCV3Model.forward accepts: "
        "{missing}.\n"
        "forward_kwargs() iterates FORWARD_KEYS, so these are NEVER PASSED -- no "
        "error is raised and any arm driven through this adapter is blind to them.\n"
        "FIX: add them to FORWARD_KEYS in tanitad/rl/refc_adapter.py, in signature "
        "order.\n"
        "  signature channels: {channels}\n"
        "  FORWARD_KEYS      : {keys}".format(
            n=len(missing),
            missing=missing,
            channels=channels,
            keys=list(FORWARD_KEYS),
        )
    )


def test_forward_keys_has_no_channels_the_forward_rejects():
    """The other direction: a stale key would raise TypeError at call time."""
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    channels = set(_optional_channels())
    stale = [k for k in FORWARD_KEYS if k not in channels]
    assert not stale, (
        "FORWARD_KEYS names {stale} which RefCV3Model.forward does not accept; "
        "forward_kwargs() would raise TypeError. Remove them or restore the "
        "parameter.".format(stale=stale)
    )


# --------------------------------------------------------------------------
# Mutation controls.
#
# A guard nobody has broken on purpose is decoration.  These prove the
# comparison above is capable of FAILING, and that it is discriminating rather
# than a blanket assertion -- without touching the real tuple.
# --------------------------------------------------------------------------


def _missing_against(keys, channels):
    """The exact comparison used by the assertion above, factored for testing."""
    return [c for c in channels if c not in keys]


def test_mutation_a_short_tuple_is_detected():
    """Reintroduce drift: drop a channel and require the check to catch it."""
    channels = _optional_channels()
    assert len(channels) >= 2, "need >=2 channels to run this mutation"

    truncated = tuple(channels[:-1])
    missing = _missing_against(truncated, channels)

    assert missing == [channels[-1]], (
        "the drift check FAILED TO DETECT a deliberately shortened tuple -- it "
        "cannot catch the defect it exists for. got missing={0}".format(missing)
    )


def test_mutation_a_complete_tuple_is_accepted():
    """The discriminating half: a correct tuple must NOT be flagged.

    Without this, a check that always reports 'missing' would pass the mutation
    above while being useless.
    """
    channels = _optional_channels()
    assert _missing_against(tuple(channels), channels) == [], (
        "the drift check flags a COMPLETE tuple -- it is a blanket assertion, "
        "not a discriminating one"
    )


def test_mutation_renaming_a_channel_is_detected():
    """A rename is the drift shape a length check alone would miss."""
    channels = _optional_channels()
    renamed = tuple(("zz_renamed" if c == channels[0] else c) for c in channels)

    assert len(renamed) == len(channels), "mutation must preserve length"
    assert _missing_against(renamed, channels) == [channels[0]], (
        "a RENAMED channel was not detected -- a length-only check would have "
        "passed this, which is why the comparison is by name"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
