"""FORWARD_KEYS must cover every optional channel RefCV3Model.forward accepts,
EXCEPT the ones a seam has DECLARED must not reach an RL rollout.

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

⛔⛔ AND THE EXCEPTION LIST IS THE OTHER HALF, BECAUSE CASE (2) WAS NOT A GAP.

The first version of this test told the reader to fix case (2) by adding
``gp_point``/``gp_valid`` to ``FORWARD_KEYS``.  **That advice was wrong and the
goal-point stream overruled it**, correctly: the only available source for a
goal point in an RL rollout is the ego's own future path -- which is the LABEL.
Plumbing those two channels through the adapter would have fed the label at
inference and manufactured a leak, which is exactly the class the programme's
vision-only rule exists to prevent.

⇒ The requirement is ``signature - declared exclusions``, in BOTH directions:
short is an arm running blind, long is an arm being fed the answer, and only the
second one looks like a win in a result table.

⭐⭐ WHY A FLAT TUPLE OF NAMES WAS NOT ENOUGH (2026-09-07)

The exclusion used to be ``goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS``, a tuple
of two names.  When three more channels came up for the same question the answers
were **three different rulings with three different routes back**:

    gp_point / gp_valid       the goal point IS the label                PERMANENT
    nav_args                  ships a ``time_s`` slot that is the ego's
                              future speed profile inverted              as BUILT
    v_max_ms / v_max_valid    the PI ruled the channel legitimate, but
                              this corpus's value is the ego's own
                              realised future speed                      this CORPUS

A five-name tuple states none of that.  After one handover the two TEMPORARY
exclusions are indistinguishable from the permanent one, and nobody lifts them --
**an exclusion with no stated route back reads as permanent.**  So each seam now
declares its own ``FORWARD_EXCLUSIONS`` carrying a reason, an unblock condition
and an evidence class (``tanitad.channel_admissibility``), this test reads the
union, and :func:`test_every_exclusion_carries_a_reason_AND_an_unblock_condition`
below refuses one that cannot say why.

Same class as ``test_the_TRAINER_ACTUALLY_CALLS_the_preflight_on_BOTH_launch_paths``:
correctness and wiring are different claims, and only the first had a test.
``test_rl_refc_adapter_robust.py`` checks that the adapter behaves correctly
GIVEN its tuple; nothing checked that the tuple still matched reality.

⛔ NO EVAL TIER AND NO FOUR-FAMILY TABLE HERE, deliberately: nothing in this
module runs a model or produces a trajectory.  It is a wiring-contract audit, and
stamping a tier on it would be a category error.
"""

import dataclasses
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


def _declarations():
    """The seams' own exclusion records -- reason, unblock condition, evidence."""
    from tanitad.channel_admissibility import forward_exclusions

    return forward_exclusions()


def _diagnostic_only():
    """Channels declared as must-not-reach-an-RL-rollout, read from the seams."""
    return tuple(d.channel for d in _declarations())


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
        "and that NO seam has declared must-not-be-plumbed: {missing}.\n"
        "forward_kwargs() iterates FORWARD_KEYS, so these are NEVER PASSED -- no "
        "error is raised and any arm driven through this adapter is blind to them.\n"
        "\n"
        "BEFORE ADDING A CHANNEL HERE, ask where an RL rollout would GET its value.\n"
        "If the only supplier is the ego's future path, it is a LABEL: declare it in\n"
        "the seam that OWNS the channel, as a ChannelExclusion carrying its reason,\n"
        "its evidence and its UNBLOCK CONDITION, and list that seam in\n"
        "tanitad.channel_admissibility.SEAM_MODULES. Otherwise add it to\n"
        "FORWARD_KEYS in signature order.\n"
        "\n"
        "  signature channels : {channels}\n"
        "  declared excluded  : {diag}\n"
        "  required           : {required}\n"
        "  FORWARD_KEYS       : {keys}".format(
            n=len(missing),
            missing=missing,
            channels=_optional_channels(),
            diag=list(_diagnostic_only()),
            required=required,
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
        "no seam declares an exclusion. If nothing is excluded any more, delete "
        "the exclusion path rather than leaving a no-op guard."
    )
    unknown = [d for d in diag if d not in channels]
    assert not unknown, (
        "the seams declare {unknown}, which RefCV3Model.forward does not accept. "
        "The exclusion is stale and is now covering nothing."
        .format(unknown=unknown)
    )


def test_declared_diagnostic_channels_are_absent_from_forward_keys():
    """The exclusion must be honoured, not merely declared.

    A channel declared must-not-reach-an-RL-rollout that is nonetheless plumbed
    is a leak with a comment next to it.
    """
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    by_chan = {d.channel: d for d in _declarations()}
    leaked = [d for d in _diagnostic_only() if d in FORWARD_KEYS]
    assert not leaked, (
        "{leaked} is declared must-not-be-plumbed but appears in FORWARD_KEYS, so "
        "it WOULD be passed in an RL rollout. Owning seam's reason: {reason}"
        .format(leaked=leaked,
                reason="; ".join(by_chan[c].reason for c in leaked))
    )


# --------------------------------------------------------------------------
# ⭐ THE GUARD THE FLAT-TUPLE VERSION COULD NOT HAVE: an exclusion must say WHY,
# and must say WHAT WOULD LIFT IT.
#
# Two of the three current rulings are TEMPORARY. Without a written route back
# they are indistinguishable from the permanent one, and a temporary exclusion
# nobody can tell is temporary never gets lifted -- the channel is lost to a
# silence rather than to a decision.
# --------------------------------------------------------------------------


def test_every_exclusion_carries_a_reason_AND_an_unblock_condition():
    decls = _declarations()
    assert decls, "nothing declared -- see test_diagnostic_only_names_are_real_channels"

    for d in decls:
        assert d.reason and d.reason.strip(), (
            "{0} is excluded with NO REASON. A reader cannot tell it from a "
            "channel somebody forgot to plumb, which is the exact defect this "
            "whole surface exists to make impossible.".format(d.channel))
        assert d.unblock and d.unblock.strip(), (
            "{0} is excluded with NO UNBLOCK CONDITION. An exclusion with no "
            "stated route back reads as PERMANENT; if it really is permanent, "
            "say so and say why no route exists.".format(d.channel))
        assert d.owner and d.owner.strip(), (
            "{0} is excluded by nobody. Ownership is what stops the reason from "
            "rotting away from the seam.".format(d.channel))
        assert d.evidence and d.evidence.strip(), (
            "{0} is excluded with no evidence class. A claim that decides a "
            "channel's admissibility must be MEASURED or line-cited."
            .format(d.channel))


def test_the_temporary_and_permanent_exclusions_are_DISTINGUISHABLE():
    """The point of the unblock condition is that it separates the two kinds.

    ⚠️ This is not a count check dressed up: if every exclusion were marked
    permanent the guard above would still pass, and the two channels that a map
    source or a build flag would unblock would quietly become forever-channels.
    """
    decls = _declarations()
    permanent = [d.channel for d in decls if d.permanent]
    temporary = [d.channel for d in decls if not d.permanent]
    assert permanent and temporary, (
        "every current exclusion is marked {0} -- but the three rulings on record "
        "are NOT the same kind: the goal point is permanent (it IS the label), "
        "while nav_args (needs a distance-only mode) and v_max_ms (needs a "
        "non-ego-future supplier) are temporary. If that changed, change this "
        "test deliberately. permanent={1} temporary={2}"
        .format("permanent" if not temporary else "temporary",
                permanent, temporary))


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

    If the seams' union ever grew to cover every channel, this test's headline
    assertion would pass vacuously.  This is the vacuity gate for that path.
    """
    optional = _optional_channels()
    required = _required_channels()
    assert len(required) < len(optional), (
        "nothing is excluded -- either the declarations are empty or the names do "
        "not match, and one of the other tests should have said so first"
    )
    assert len(required) >= len(optional) - len(_diagnostic_only()), (
        "more channels were excluded than are declared"
    )


# --------------------------------------------------------------------------
# ⭐ MUTATIONS FOR THE NEW GUARD. Three defects, three reds.
# --------------------------------------------------------------------------


def _live(channel):
    """The real declaration for ``channel``, so a mutation starts from truth."""
    return {d.channel: d for d in _declarations()}[channel]


def test_MUTATION_an_exclusion_with_no_reason_CANNOT_EVEN_BE_CONSTRUCTED():
    """The first line of defence: the type refuses the placeholder."""
    from tanitad.channel_admissibility import (ChannelExclusion,
                                               ExclusionDeclarationError)

    base = dataclasses.asdict(_live("v_max_ms"))
    base.pop("refs", None)

    for field, bad in (("reason", ""), ("reason", "   "), ("reason", "label"),
                       ("reason", "TBD"), ("unblock", ""), ("unblock", "n/a"),
                       ("evidence", "")):
        kw = dict(base)
        kw[field] = bad
        with pytest.raises(ExclusionDeclarationError) as ei:
            ChannelExclusion(**kw)
        assert field in str(ei.value), (field, bad, str(ei.value))

    # ...and the CONTROL: the unmutated record must still construct, or the
    # refusals above are proving nothing but that the constructor is broken.
    assert ChannelExclusion(**base).channel == "v_max_ms"


def test_MUTATION_a_reason_that_slipped_through_is_STILL_caught_by_the_guard():
    """The second line: the guard must not lean entirely on the constructor.

    ⛔ Frozen dataclasses are not tamper-proof -- ``object.__setattr__`` walks
    straight past ``__post_init__``.  If the only check were at construction, a
    record edited after the fact (or built by some future loader that bypasses
    the type) would sail through.  So the guard's own loop is mutated here.
    """
    good = _live("nav_args")
    blanked = dataclasses.replace(good)
    object.__setattr__(blanked, "reason", "")          # bypasses __post_init__

    assert blanked.reason == "", "the mutation did not take -- nothing was proved"
    assert not (blanked.reason and blanked.reason.strip()), (
        "the guard's own reason predicate does not flag an emptied reason; it "
        "would pass a declaration that says nothing")

    # the discriminating half: the UNMUTATED record must pass the same predicate
    assert good.reason and good.reason.strip()
    assert good.unblock and good.unblock.strip()

    # and again for the unblock condition, which is the field this guard added
    no_route = dataclasses.replace(good)
    object.__setattr__(no_route, "unblock", "   ")
    assert not no_route.unblock.strip(), "the unblock mutation did not take"


def test_MUTATION_moving_a_declared_channel_into_FORWARD_KEYS_is_RED():
    """The leak direction: a declared-excluded channel that IS plumbed."""
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    diag = _diagnostic_only()
    assert diag, "need a declared exclusion to run this mutation"
    leaky = tuple(FORWARD_KEYS) + (diag[0],)

    leaked = [d for d in diag if d in leaky]
    assert leaked == [diag[0]], (
        "the honoured-exclusion check FAILED TO DETECT a declared channel moved "
        "into FORWARD_KEYS -- it cannot catch the leak it exists for")

    # discriminating half: the real tuple must NOT be flagged
    assert [d for d in diag if d in tuple(FORWARD_KEYS)] == []


def test_MUTATION_a_channel_in_NEITHER_place_is_RED():
    """The gap that made this test file necessary, reintroduced deliberately.

    A channel the forward accepts, that FORWARD_KEYS does not plumb and no seam
    excludes, must be flagged -- that is the silent-blindness defect, and it is
    exactly the state ``nav_args``/``v_max_ms``/``v_max_valid`` were in before
    the seams ruled on them.
    """
    optional = _optional_channels()
    excluded = set(_diagnostic_only())
    grown = optional + ["brand_new_channel"]

    required = [c for c in grown if c not in excluded]
    from tanitad.rl.refc_adapter import FORWARD_KEYS

    missing = _missing_against(FORWARD_KEYS, required)
    assert missing == ["brand_new_channel"], (
        "a channel in NEITHER the adapter nor any seam's declaration was not "
        "flagged. got missing={0}".format(missing))

    # discriminating half: declare it, and the same comparison must go quiet
    required_if_declared = [c for c in grown
                            if c not in (excluded | {"brand_new_channel"})]
    assert _missing_against(FORWARD_KEYS, required_if_declared) == [], (
        "declaring the channel did not silence the check -- the exclusion path "
        "is not actually what narrows the requirement")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
