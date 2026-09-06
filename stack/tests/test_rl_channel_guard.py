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

⛔ These tests do NOT assert that the live adapter is complete. It is another
stream's file, `test_rl_forward_keys_cover_signature.py` already pins that, and a
second failing copy of someone else's finding is noise. What they pin is that
THIS guard would refuse the launch.
"""

import inspect

import pytest

from tanitad.rl.channel_guard import (
    ChannelDriftError,
    NON_CHANNEL_PARAMS,
    assert_forward_channels_complete,
    forward_channel_report,
    forward_optional_channels,
)


def _fake_forward(self, frames, nav_cmd=None, v0=None, steps=0, lan=None,
                  gp_point=None, gp_valid=None, agent_gt=None):
    """A stand-in forward with a KNOWN channel set, so the tests do not depend
    on the live signature (which other streams move)."""
    return None


COMPLETE = ("nav_cmd", "v0", "lan", "gp_point", "gp_valid", "agent_gt")


# --------------------------------------------------------------------------- #
# the CONTROL: a complete tuple must pass, or every refusal below is vacuous     #
# --------------------------------------------------------------------------- #
def test_a_complete_tuple_passes_and_the_channels_are_derived_from_the_signature():
    chans = forward_optional_channels(_fake_forward)
    assert chans == COMPLETE, chans
    assert "frames" not in chans and "steps" not in chans and "self" not in chans
    rep = assert_forward_channels_complete(_fake_forward, COMPLETE)
    assert rep["complete"] is True
    assert rep["missing_from_adapter"] == []
    assert rep["n_signature_channels"] == 6


# --------------------------------------------------------------------------- #
# MUTATION 1 -- the EXACT drift measured on 2026-09-06 in the worktree           #
# --------------------------------------------------------------------------- #
def test_the_goal_point_drift_is_REFUSED_and_the_message_names_both_channels():
    drifted = tuple(k for k in COMPLETE if k not in ("gp_point", "gp_valid"))
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, drifted)
    msg = str(ei.value)
    assert "gp_point" in msg and "gp_valid" in msg, msg
    assert "SILENTLY DROP" in msg, msg
    rep = forward_channel_report(_fake_forward, drifted)
    assert rep["missing_from_adapter"] == ["gp_point", "gp_valid"], rep
    assert rep["complete"] is False


# --------------------------------------------------------------------------- #
# MUTATION 2 -- the WORSE drift measured at HEAD the same day                    #
# --------------------------------------------------------------------------- #
def test_the_HEAD_state_dropping_agent_gt_TOO_is_REFUSED():
    drifted = tuple(k for k in COMPLETE
                    if k not in ("gp_point", "gp_valid", "agent_gt"))
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, drifted)
    msg = str(ei.value)
    for chan in ("gp_point", "gp_valid", "agent_gt"):
        assert chan in msg, (chan, msg)
    rep = forward_channel_report(_fake_forward, drifted)
    assert rep["n_adapter_keys"] == 3 and rep["n_signature_channels"] == 6


# --------------------------------------------------------------------------- #
# MUTATION 3 -- ONE dropped channel, the smallest defect that still runs blind   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dropped", COMPLETE)
def test_dropping_ANY_single_channel_is_refused(dropped):
    drifted = tuple(k for k in COMPLETE if k != dropped)
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, drifted)
    assert dropped in str(ei.value)


# --------------------------------------------------------------------------- #
# MUTATION 4 -- the OPPOSITE defect: a key the forward does not accept           #
# --------------------------------------------------------------------------- #
def test_an_EXTRA_key_the_forward_cannot_accept_is_also_refused():
    with pytest.raises(ChannelDriftError) as ei:
        assert_forward_channels_complete(_fake_forward, COMPLETE + ("no_such_chan",))
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
    rep = forward_channel_report(grown, COMPLETE)
    assert rep["missing_from_adapter"] == ["brand_new_channel"], rep
    assert "v0" in rep["extra_in_adapter"], rep
    with pytest.raises(ChannelDriftError):
        assert_forward_channels_complete(grown, COMPLETE)


# --------------------------------------------------------------------------- #
# the LIVE pair, read by a SECOND mechanism -- two samples through one probe is  #
# one sample, so this derivation deliberately does not use inspect.signature     #
# --------------------------------------------------------------------------- #
def test_the_live_verdict_agrees_with_an_INDEPENDENT_derivation():
    from tanitad.refs.refc_v3 import RefCV3Model
    from tanitad.rl import refc_adapter

    spec = inspect.getfullargspec(RefCV3Model.forward)
    named = list(spec.args) + list(spec.kwonlyargs)
    independent = [n for n in named if n not in NON_CHANNEL_PARAMS]

    rep = forward_channel_report()
    assert sorted(rep["signature_channels"]) == sorted(independent), (
        "the two derivations disagree: signature=%s getfullargspec=%s"
        % (rep["signature_channels"], independent))

    missing = [c for c in independent if c not in tuple(refc_adapter.FORWARD_KEYS)]
    assert rep["missing_from_adapter"] == [
        c for c in rep["signature_channels"] if c in missing]

    # and the guard's REFUSAL must be exactly as reachable as the drift is real
    if missing:
        with pytest.raises(ChannelDriftError):
            assert_forward_channels_complete()
    else:
        assert assert_forward_channels_complete()["complete"] is True
