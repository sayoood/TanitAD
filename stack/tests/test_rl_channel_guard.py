"""MUTATION proofs of the launch-preflight channel guard.

WHY THESE ARE MUTATION TESTS AND NOT INSPECTION TESTS
=====================================================
A guard that is only *inspected* can be dead. MEASURED earlier in this programme:
an AST census read 0 suspects on BOTH the fixed and the broken trainer, because
inspection cannot tell a guard that fires from a guard that cannot. The rule that
came out of it is that a guard must be proved by REINTRODUCING the defect it
exists to catch and showing the refusal is reachable.

So every test below either (a) hands the guard a deliberately drifted tuple and
requires a REFUSAL that NAMES the dropped channel, or (b) hands it a complete
tuple and requires silence. The live pair is checked by a SECOND, DIFFERENTLY
IMPLEMENTED derivation (`getfullargspec` rather than `signature`), because two
samples through one mechanism are one sample.

⛔⛔ WHAT THIS SUITE MISSED, AND WHY IT MATTERS MORE THAN WHAT IT CAUGHT
======================================================================
The previous version's live-pair test read::

    if missing:
        with pytest.raises(ChannelDriftError):
            assert_forward_channels_complete()
    else:
        assert assert_forward_channels_complete()["complete"] is True

-- i.e. it accepted "the guard refuses the live launch" and "the guard passes the
live launch" as EQUALLY correct outcomes, so it stayed green while the guard
refused every RL launch on the live pair. Worse: the refusal named ``gp_point``
and ``gp_valid`` among the channels the arm "runs blind to", telling the operator
to plumb the two channels whose only supplier is the ego's own future pose. The
suite was green on a preflight that recommended a label leak.

⇒ A test whose expected value is "whatever the code does" cannot fail. The
live-pair test below now asserts the ONE outcome that is correct
(:func:`test_the_LIVE_pair_PASSES_and_the_verdict_is_reproduced_independently`),
and the fixture channels below were renamed off ``gp_point``/``gp_valid`` so that
no test in this file models plumbing them as the desired state.

⛔ These tests do NOT assert that the live adapter's tuple is authored correctly.
That is another stream's file and `test_rl_forward_keys_cover_signature.py` pins
it. What they pin is that THIS guard would refuse a launch -- in both directions:
a channel silently DROPPED, and a declared-excluded channel silently FED.

Tier: none -- a wiring-contract surface, no trajectory is produced anywhere here.
"""

import inspect

import pytest

from tanitad.channel_admissibility import ChannelExclusion
from tanitad.rl.channel_guard import (
    ChannelDriftError,
    ChannelLeakError,
    NON_CHANNEL_PARAMS,
    assert_forward_channels_complete,
    forward_channel_report,
    forward_optional_channels,
)


def _fake_forward(self, frames, nav_cmd=None, v0=None, steps=0, lan=None,
                  chan_seam_a=None, chan_seam_b=None, agent_gt=None):
    """A stand-in forward with a KNOWN channel set, so the tests do not depend
    on the live signature (which other streams move).

    ⚠️ ``chan_seam_a``/``chan_seam_b`` stand for "two channels a seam appended
    after this tuple was written" -- the shape of BOTH real incidents. They are
    deliberately NOT named after real channels: the previous fixture used
    ``gp_point``/``gp_valid`` and therefore asserted, as its control, that the
    label must be plumbed.
    """
    return None


COMPLETE = ("nav_cmd", "v0", "lan", "chan_seam_a", "chan_seam_b", "agent_gt")

#: ⛔ The fixture forward has no seams behind it, so it has no real exclusions.
#: Passing this explicitly (rather than letting the guard read the LIVE seams) is
#: what keeps these mechanism tests independent of the current ruling -- otherwise
#: every real exclusion would read as STALE against a signature that has none.
NO_EXCLUSIONS = ()


def _fake_exclusion(channel, **over):
    """A well-formed declaration for a fixture channel."""
    kw = dict(
        channel=channel,
        owner="tests.fixture (stands in for the seam that owns the channel)",
        reason=("A fixture declaration: this channel's only supplier in a rollout "
                "would be the ego's own future, so it must not be plumbed."),
        unblock="A supplier that is not the ego's own future path.",
        evidence="FIXTURE: constructed by this test module.",
    )
    kw.update(over)
    return ChannelExclusion(**kw)


# --------------------------------------------------------------------------- #
# the CONTROL: a complete tuple must pass, or every refusal below is vacuous     #
# --------------------------------------------------------------------------- #
def test_a_complete_tuple_passes_and_the_channels_are_derived_from_the_signature():
    chans = forward_optional_channels(_fake_forward)
    assert chans == COMPLETE, chans
    assert "frames" not in chans and "steps" not in chans and "self" not in chans
    rep = assert_forward_channels_complete(_fake_forward, COMPLETE, NO_EXCLUSIONS)
    assert rep["complete"] is True
    assert rep["missing_from_adapter"] == []
    assert rep["n_signature_channels"] == 6


# --------------------------------------------------------------------------- #
# MUTATION 1 -- the drift shape measured on 2026-09-06 in the worktree           #
# --------------------------------------------------------------------------- #
def test_two_channels_a_seam_appended_and_nobody_plumbed_are_REFUSED():
    drifted = tuple(k for k in COMPLETE if k not in ("chan_seam_a", "chan_seam_b"))
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, drifted, NO_EXCLUSIONS)
    msg = str(ei.value)
    assert "chan_seam_a" in msg and "chan_seam_b" in msg, msg
    assert "SILENTLY DROP" in msg, msg
    rep = forward_channel_report(_fake_forward, drifted, NO_EXCLUSIONS)
    assert rep["missing_from_adapter"] == ["chan_seam_a", "chan_seam_b"], rep
    assert rep["complete"] is False


# --------------------------------------------------------------------------- #
# MUTATION 2 -- the WORSE drift measured at HEAD the same day                    #
# --------------------------------------------------------------------------- #
def test_the_HEAD_state_dropping_agent_gt_TOO_is_REFUSED():
    drifted = tuple(k for k in COMPLETE
                    if k not in ("chan_seam_a", "chan_seam_b", "agent_gt"))
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, drifted, NO_EXCLUSIONS)
    msg = str(ei.value)
    for chan in ("chan_seam_a", "chan_seam_b", "agent_gt"):
        assert chan in msg, (chan, msg)
    rep = forward_channel_report(_fake_forward, drifted, NO_EXCLUSIONS)
    assert rep["n_adapter_keys"] == 3 and rep["n_signature_channels"] == 6


# --------------------------------------------------------------------------- #
# MUTATION 3 -- ONE dropped channel, the smallest defect that still runs blind   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dropped", COMPLETE)
def test_dropping_ANY_single_channel_is_refused(dropped):
    drifted = tuple(k for k in COMPLETE if k != dropped)
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, drifted, NO_EXCLUSIONS)
    assert dropped in str(ei.value)


# --------------------------------------------------------------------------- #
# MUTATION 4 -- the OPPOSITE defect: a key the forward does not accept           #
# --------------------------------------------------------------------------- #
def test_an_EXTRA_key_the_forward_cannot_accept_is_also_refused():
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward,
                                         COMPLETE + ("no_such_chan",),
                                         NO_EXCLUSIONS)
    msg = str(ei.value)
    assert "no_such_chan" in msg and "TypeError" in msg, msg


# --------------------------------------------------------------------------- #
# the guard must track a signature CHANGE, not a hardcoded list                  #
# --------------------------------------------------------------------------- #
def test_the_requirement_FOLLOWS_the_signature_rather_than_a_frozen_list():
    def grown(self, frames, nav_cmd=None, steps=0, brand_new_channel=None):
        return None

    assert forward_optional_channels(grown) == ("nav_cmd", "brand_new_channel")
    # the tuple that was complete for _fake_forward is now BOTH short and long
    rep = forward_channel_report(grown, COMPLETE, NO_EXCLUSIONS)
    assert rep["missing_from_adapter"] == ["brand_new_channel"], rep
    assert "v0" in rep["extra_in_adapter"], rep
    with pytest.raises(ChannelDriftError):
        assert_forward_channels_complete(grown, COMPLETE, NO_EXCLUSIONS)


# --------------------------------------------------------------------------- #
# ⭐ THE EXCLUSION PATH -- the half the guard did not have, mutated in BOTH       #
#    directions. A declared channel must be ALLOWED to be absent, and REFUSED    #
#    when present: those are different assertions and both have to fire.        #
# --------------------------------------------------------------------------- #
def test_a_DECLARED_channel_may_be_absent_without_the_guard_complaining():
    """The correct state. Without this, the guard refuses every real launch --
    which is exactly what it did on 2026-09-07 before the exclusion path landed."""
    withheld = tuple(k for k in COMPLETE if k != "chan_seam_b")
    decls = (_fake_exclusion("chan_seam_b"),)

    rep = assert_forward_channels_complete(_fake_forward, withheld, decls)
    assert rep["complete"] is True, rep
    assert rep["declared_excluded"] == ["chan_seam_b"], rep
    assert "chan_seam_b" not in rep["required_channels"], rep
    assert rep["n_required"] == 5 and rep["n_signature_channels"] == 6, rep


def test_a_DECLARED_channel_that_IS_plumbed_is_REFUSED_as_a_LEAK():
    """The direction the original guard not only missed but RECOMMENDED."""
    decls = (_fake_exclusion("chan_seam_b"),)
    with pytest.raises(ChannelLeakError) as ei:
        assert_forward_channels_complete(_fake_forward, COMPLETE, decls)
    msg = str(ei.value)
    assert "chan_seam_b" in msg, msg
    # the refusal must QUOTE the owning seam's reason, not merely name the channel:
    # an operator who cannot see why will re-add the key tomorrow
    assert "only supplier in a rollout" in msg, msg
    assert "unblock" in msg, msg

    rep = forward_channel_report(_fake_forward, COMPLETE, decls)
    assert rep["leaked_into_adapter"] == ["chan_seam_b"], rep
    assert rep["complete"] is False


def test_a_LEAK_is_reported_BEFORE_a_simultaneous_drift():
    """Both defects at once: the operator must be shown the leak.

    A blind arm under-performs visibly; an arm fed the answer over-performs and
    reads as a win. If only one message fits, it is this one.
    """
    decls = (_fake_exclusion("chan_seam_b"),)
    both = tuple(k for k in COMPLETE if k != "agent_gt")   # short AND leaking
    with pytest.raises(ChannelLeakError) as ei:
        assert_forward_channels_complete(_fake_forward, both, decls)
    assert "chan_seam_b" in str(ei.value)


def test_a_STALE_exclusion_naming_a_channel_the_forward_LOST_is_REFUSED():
    """An exclusion that covers nothing would keep covering nothing if a
    DIFFERENT channel later took that name."""
    decls = (_fake_exclusion("chan_that_was_removed"),)
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, COMPLETE, decls)
    msg = str(ei.value)
    assert "chan_that_was_removed" in msg and "STALE" in msg, msg


def test_the_report_BANKS_the_reasons_so_a_run_record_carries_them():
    """A conditioning contract that is checked but not written down is not
    evidence -- the same rule ``assert_conditioning`` follows one module over."""
    decls = (_fake_exclusion("chan_seam_b"),)
    withheld = tuple(k for k in COMPLETE if k != "chan_seam_b")
    rep = forward_channel_report(_fake_forward, withheld, decls)
    banked = rep["exclusion_declarations"]
    assert len(banked) == 1 and banked[0]["channel"] == "chan_seam_b"
    for field in ("reason", "unblock", "evidence", "owner", "permanent"):
        assert field in banked[0], (field, banked[0])


# --------------------------------------------------------------------------- #
# the LIVE pair, read by a SECOND mechanism -- two samples through one probe is  #
# one sample, so this derivation deliberately does not use inspect.signature     #
# --------------------------------------------------------------------------- #
def test_the_LIVE_pair_PASSES_and_the_verdict_is_reproduced_independently():
    from tanitad.channel_admissibility import excluded_channels
    from tanitad.refs.refc_v3 import RefCV3Model
    from tanitad.rl import refc_adapter

    spec = inspect.getfullargspec(RefCV3Model.forward)
    named = list(spec.args) + list(spec.kwonlyargs)
    independent = [n for n in named if n not in NON_CHANNEL_PARAMS]

    rep = forward_channel_report()
    assert sorted(rep["signature_channels"]) == sorted(independent), (
        "the two derivations disagree: signature=%s getfullargspec=%s"
        % (rep["signature_channels"], independent))

    excluded = set(excluded_channels())
    keys = tuple(refc_adapter.FORWARD_KEYS)
    independent_required = [c for c in independent if c not in excluded]
    independent_missing = [c for c in independent_required if c not in keys]
    independent_leaked = [c for c in keys if c in excluded]

    assert rep["missing_from_adapter"] == [
        c for c in rep["signature_channels"] if c in independent_missing]
    assert rep["leaked_into_adapter"] == independent_leaked

    # ⛔ THE ONE CORRECT OUTCOME, asserted rather than branched on. The previous
    # version accepted "refuses" too, and stayed green while the preflight blocked
    # every launch and told the operator to plumb the label.
    assert independent_missing == [], (
        "the live adapter is blind to %s -- see "
        "test_rl_forward_keys_cover_signature.py" % independent_missing)
    assert independent_leaked == [], (
        "the live adapter plumbs the declared-excluded %s" % independent_leaked)
    assert assert_forward_channels_complete()["complete"] is True


def test_the_live_exclusions_all_name_channels_the_live_forward_ACTUALLY_HAS():
    """The stale-exclusion check, on the live pair rather than the fixture."""
    from tanitad.channel_admissibility import excluded_channels

    live = set(forward_optional_channels())
    unknown = [c for c in excluded_channels() if c not in live]
    assert not unknown, (
        "%s is declared must-not-be-plumbed but RefCV3Model.forward does not "
        "accept it -- the exclusion covers nothing" % unknown)
