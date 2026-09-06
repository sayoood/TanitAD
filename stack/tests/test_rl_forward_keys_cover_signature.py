"""FORWARD_KEYS must cover every optional channel RefCV3Model.forward accepts,
EXCEPT the ones declared diagnostic-only.

WHY THIS TEST EXISTS (and why it is not a style check):

``tanitad.rl.refc_adapter.forward_kwargs`` builds its kwargs by iterating
``FORWARD_KEYS``.  A channel that is absent from that tuple is therefore
**never passed at all** -- no error, no warning, no missing-key exception.
The model simply runs with the channel at its default and the arm is blind
to it.

This has happened TWICE:

  1. ``agent_gt`` was appended to the forward by the refcv5 WP-6 agent seam
     and not added here.  An ``--agents oracle`` build driven through this
     adapter tripped the model's own "no agent_gt reached the forward" guard
     even though the batch carried it.  ``refc_adapter``'s own module
     docstring records that incident.

  2. ``gp_point`` / ``gp_valid`` were appended by the E15 goal-point seam and
     not added here.

The lesson is not "be more careful when editing the tuple".  A hand-maintained
mirror of a signature drifts, and both drifts were silent.  The durable fix is
to DERIVE the requirement from the signature, which is what this test does.

WHY THE EXCEPTION LIST IS NOT A LOOPHOLE:

The first version of this test told the reader to fix case (2) by adding
``gp_point``/``gp_valid`` to ``FORWARD_KEYS``.  **That advice was wrong and the
goal-point stream overruled it**, correctly: the only available source for a
goal point in an RL rollout is the ego's own future path -- which is the LABEL.
Plumbing those two channels through the adapter would have fed the label at
inference and manufactured a leak, which is exactly the class the programme's
vision-only rule exists to prevent.

So the exception is a DECLARATION, not a suppression.  It lives beside the seam
that owns it (``tanitad.refs.goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS``), it
carries its reason in that module, and this test reads it from there rather than
hardcoding a list that would rot the same way the tuple did.  A channel excluded
here is a channel somebody deliberately declared must not reach an RL rollout.

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


def _diagnostic_only():
    """Channels declared as must-not-reach-an-RL-rollout, read from the seam."""
    from tanitad.refs.goal_point import DIAGNOSTIC_ONLY_FORWARD_KWARGS

    return tuple(DIAGNOSTIC_ONLY_FORWARD_KWARGS)


def _required_channels():
    excluded = set(_diagnostic_only())
    return [c for c in _optional_channels() if c not in excluded]


def _missing_against(keys, channels):
    """The exact comparison used by the assertion below, factored for testing."""
    return [c for c in channels if c not in keys]


def test_forward_keys_covers_every_required_channel():
    """The load-bearing assertion.

    Fails loudly, naming the missing channels, rather than letting an arm run
    blind to them.
    """
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    required = _required_channels()
    assert required, (
        "no required channels found on RefCV3Model.forward -- the introspection "
        "in this test is broken, not the adapter"
    )

    missing = _missing_against(FORWARD_KEYS, required)
    assert not missing, (
        "FORWARD_KEYS is missing {n} channel(s) that RefCV3Model.forward accepts "
        "and that are NOT declared diagnostic-only: {missing}.\n"
        "forward_kwargs() iterates FORWARD_KEYS, so these are NEVER PASSED -- no "
        "error is raised and any arm driven through this adapter is blind to them.\n"
        "\n"
        "BEFORE ADDING A CHANNEL HERE, ask where an RL rollout would get its value.\n"
        "If the only source is the ego's future path, it is a LABEL: declare it in\n"
        "tanitad.refs.goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS with its reason\n"
        "instead of plumbing it. Otherwise add it to FORWARD_KEYS in signature order.\n"
        "\n"
        "  signature channels : {channels}\n"
        "  diagnostic-only    : {diag}\n"
        "  FORWARD_KEYS       : {keys}".format(
            n=len(missing),
            missing=missing,
            channels=_optional_channels(),
            diag=list(_diagnostic_only()),
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


def test_diagnostic_only_names_are_real_channels():
    """A stale exclusion is as dangerous as a stale inclusion.

    If a declared-diagnostic channel is renamed or removed from the forward, the
    exclusion silently starts covering nothing -- and would keep covering nothing
    if a DIFFERENT channel later took that name.
    """
    channels = set(_optional_channels())
    diag = _diagnostic_only()
    assert diag, (
        "DIAGNOSTIC_ONLY_FORWARD_KWARGS is empty. If nothing is excluded any more, "
        "delete the exclusion path rather than leaving a no-op guard."
    )
    unknown = [d for d in diag if d not in channels]
    assert not unknown, (
        "DIAGNOSTIC_ONLY_FORWARD_KWARGS names {unknown}, which RefCV3Model.forward "
        "does not accept. The exclusion is stale and is now covering nothing."
        .format(unknown=unknown)
    )


def test_declared_diagnostic_channels_are_absent_from_forward_keys():
    """The exclusion must be honoured, not merely declared.

    A channel declared must-not-reach-an-RL-rollout that is nonetheless plumbed
    is a leak with a comment next to it.
    """
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    leaked = [d for d in _diagnostic_only() if d in FORWARD_KEYS]
    assert not leaked, (
        "{leaked} is declared DIAGNOSTIC-ONLY but appears in FORWARD_KEYS, so it "
        "WOULD be passed in an RL rollout. Its only source is the ego's future "
        "path, i.e. the label -- this is the leak the declaration exists to "
        "prevent.".format(leaked=leaked)
    )


# --------------------------------------------------------------------------
# Mutation controls.
#
# A guard nobody has broken on purpose is decoration.  These prove the
# comparison is capable of FAILING, and that it is discriminating rather than a
# blanket assertion -- without touching the real tuple.
# --------------------------------------------------------------------------


def test_mutation_a_short_tuple_is_detected():
    """Reintroduce drift: drop a required channel and require the check to catch it."""
    required = _required_channels()
    assert len(required) >= 2, "need >=2 required channels to run this mutation"

    truncated = tuple(required[:-1])
    missing = _missing_against(truncated, required)

    assert missing == [required[-1]], (
        "the drift check FAILED TO DETECT a deliberately shortened tuple -- it "
        "cannot catch the defect it exists for. got missing={0}".format(missing)
    )


def test_mutation_a_complete_tuple_is_accepted():
    """The discriminating half: a correct tuple must NOT be flagged.

    Without this, a check that always reports 'missing' would pass the mutation
    above while being useless.
    """
    required = _required_channels()
    assert _missing_against(tuple(required), required) == [], (
        "the drift check flags a COMPLETE tuple -- it is a blanket assertion, "
        "not a discriminating one"
    )


def test_mutation_renaming_a_channel_is_detected():
    """A rename is the drift shape a length check alone would miss."""
    required = _required_channels()
    renamed = tuple(("zz_renamed" if c == required[0] else c) for c in required)

    assert len(renamed) == len(required), "mutation must preserve length"
    assert _missing_against(renamed, required) == [required[0]], (
        "a RENAMED channel was not detected -- a length-only check would have "
        "passed this, which is why the comparison is by name"
    )


def test_mutation_the_exclusion_does_not_hide_a_real_gap():
    """The exclusion must narrow the requirement, never empty it.

    If DIAGNOSTIC_ONLY_FORWARD_KWARGS ever grew to cover every channel, this
    test's headline assertion would pass vacuously.  This is the vacuity gate
    for that path.
    """
    optional = _optional_channels()
    required = _required_channels()
    assert len(required) < len(optional), (
        "nothing is excluded -- either the declaration is empty or the names do "
        "not match, and one of the other tests should have said so first"
    )
    assert len(required) >= len(optional) - len(_diagnostic_only()), (
        "more channels were excluded than are declared diagnostic-only"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
