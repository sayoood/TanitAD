"""The VALUE flow of the RL conditioning contract — can a required channel arrive None?

The sibling package (`Research/2026-09-07-rl-channel-admissibility/`) closed the KEY
SET: WHICH channels an RL rollout may receive is now a decision, declared per seam with
a reason, an unblock condition and an owner. It explicitly left this open:

    "`forward_kwargs` does `batch.get(k)`, so a required channel whose batch key is
     absent reaches the forward as `None` with nothing raising."

⛔ AND IT WAS NOT A WIRING GAP. ``assert_conditioning`` is called on the first batch of
every rollout (`refc_adapter.make_refc_sample_fn`) and has been all along. It was a
COVERAGE gap: the requirement map read ``bool(getattr(cfg, flag, False)) if flag else
False``, and SIX of its seven entries had ``flag=None``, so they evaluated False and
were never checked. The map was a list of the channels it did NOT check.

⭐ Of those six, two carried a real reason (``nav_cmd``: asserting it would refuse the
programme's own ``nav_cmd=None`` eval arm; ``agent_gt``: the model refuses both
directions itself). Four carried nothing, or *"no config flag exists — plumbed, not
asserted"* — a mechanism, not a judgement. ⛔ And for three of those four the premise
was FALSE: a declaring field exists and is NESTED, which the flat ``getattr`` could
never have reached.

⚠️ WHAT THIS FILE DELIBERATELY DOES NOT DO. The guard test the sibling found and fixed
read ``if missing: expect refusal; else: expect pass`` — an expected value of *whatever
the code does*, which can never fail. Every pair below is therefore written as TWO
tests with HARD-CODED expectations: one that must refuse, one that must pass. Neither
consults the code to decide what to expect.

⛔ NO EVAL TIER AND NO FOUR-FAMILY TABLE: no model produces a trajectory in this file.
It is a wiring-contract surface.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.rl import refc_adapter as A                              # noqa: E402


def _model(**kw):
    """A model whose ``.cfg`` is the REAL ``RefCV3Config`` — never a stub.

    ⛔ A two-attribute fake config is how the nested predicates stayed invisible to the
    test surface as well as to the guard.
    """
    from tanitad.refs.refc_v3 import RefCV3Config

    class _M:
        pass

    cfg = RefCV3Config()
    cfg.ego_state_inject = kw.get("ego_state_inject", False)
    cfg.core.anchors.v0_conditioned = kw.get("v0_conditioned", False)
    cfg.core.sel_reach_clamp = kw.get("sel_reach_clamp", False)
    cfg.core.graft_lan = kw.get("graft_lan", False)
    cfg.core.nav_known_channel = kw.get("nav_known_channel", False)
    m = _M()
    m.cfg = cfg
    return m


_FRAMES = {"frames": torch.zeros(1)}


# ======================================================================================
# ⭐ v0 — the channel that matters most. THREE directions, three hard-coded answers.
# ======================================================================================

def test_a_v0_conditioned_build_REFUSES_a_batch_with_no_v0():
    """⛔ Dropping v0 does not merely re-condition the policy — it SILENTLY REPLACES THE
    ACTION SPACE.

    `refc.py:3097` -> `v_ms = None`; `refc.py:2128` -> `roll_bank(None, ...)`;
    `refc.py:1722` -> `v = full(ref_speed)`. Every anchor is rolled at the 10 m/s
    reference instead of the window's measured speed and NOTHING raises. MEASURED
    (`refc.py:363-370`): 0.3773 m oracle-in-vocabulary fixed vs 0.2610 m v0-conditioned.
    """
    with pytest.raises(A.ConditioningError) as ei:
        A.assert_conditioning(_model(v0_conditioned=True), dict(_FRAMES))
    assert "v0" in str(ei.value)


def test_the_same_v0_conditioned_build_ACCEPTS_the_batch_once_v0_is_present():
    """⛔ THE CONTROL. A refusal that fires on the good batch too proves nothing, and it
    is how a guard gets deleted rather than fixed."""
    req = A.assert_conditioning(_model(v0_conditioned=True),
                                {**_FRAMES, "v0": torch.zeros(1)})
    assert req["v0"] is True


def test_sel_reach_clamp_alone_also_requires_v0():
    """The SECOND predicate, and it is not redundant: `refc_v3.py:583`/`:751` pin
    ``sel_reach_clamp = True`` as a v3 PRECONDITION, so it covers the fixed-bank builds
    ``anchors.v0_conditioned`` does not. `refc.py:2360` skips the S2 reachability band
    entirely when v_ms is None."""
    with pytest.raises(A.ConditioningError):
        A.assert_conditioning(_model(sel_reach_clamp=True), dict(_FRAMES))


def test_a_build_that_consumes_v0_NEITHER_way_does_not_demand_it():
    """⛔ THE DIRECTION THAT PROTECTS LAUNCHES. Asserting a channel the model was not
    trained with would refuse valid arms, which is how a guard gets deleted."""
    req = A.assert_conditioning(_model(), dict(_FRAMES))
    assert req["v0"] is False


# ======================================================================================
# lan
# ======================================================================================

def test_a_graft_lan_build_REFUSES_a_batch_with_no_lan():
    """`refc.py:3072` is `if self.cfg.graft_lan and lan is not None:` — the route
    encoder is SKIPPED, not fed a zero, and the decoder runs with no corridor."""
    with pytest.raises(A.ConditioningError) as ei:
        A.assert_conditioning(_model(graft_lan=True), dict(_FRAMES))
    assert "lan" in str(ei.value)


def test_the_same_graft_lan_build_ACCEPTS_the_batch_once_lan_is_present():
    req = A.assert_conditioning(_model(graft_lan=True),
                                {**_FRAMES, "lan": torch.zeros(1, 16)})
    assert req["lan"] is True


def test_a_build_without_the_lan_seam_does_not_demand_lan():
    assert A.assert_conditioning(_model(), dict(_FRAMES))["lan"] is False


# ======================================================================================
# ⛔ THE NESTED-FIELD TRAP — the defect class this package closes for everyone
# ======================================================================================

def test_an_unresolvable_predicate_RAISES_instead_of_reading_False():
    """⛔⛔ THE ONE THAT MATTERS.

    ``getattr(cfg, flag, False)`` on a nested or renamed field reads **False forever**:
    a check that is present in the source, passes every batch, and can never fire. That
    is not hypothetical — it is what happened to ``v0``, ``lan`` and ``nav_known``.
    An unresolvable predicate must be LOUD at the first batch.
    """
    from tanitad.refs.refc_v3 import RefCV3Config
    cfg = RefCV3Config()
    with pytest.raises(A.ConditioningError) as ei:
        A._read_predicate(cfg, "core.anchors.no_such_field")
    assert "never" in str(ei.value).lower()
    # ⛔ AND THE FLAT READ OF A NESTED FIELD, which is the exact historical bug:
    with pytest.raises(A.ConditioningError):
        A._read_predicate(cfg, "v0_conditioned")      # the name, un-nested


def test_the_positive_control_a_resolvable_nested_predicate_reads_its_value():
    """⛔ THE CONTROL FOR THE TEST ABOVE. A reader that raised on EVERYTHING would pass
    that test and be useless."""
    from tanitad.refs.refc_v3 import RefCV3Config
    cfg = RefCV3Config()
    cfg.core.anchors.v0_conditioned = True
    assert A._read_predicate(cfg, "core.anchors.v0_conditioned") is True
    cfg.core.anchors.v0_conditioned = False
    assert A._read_predicate(cfg, "core.anchors.v0_conditioned") is False


def test_every_declared_predicate_resolves_on_a_real_config():
    """⭐ THE TEST THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT. Resolution is checked
    against an INSTANCE, because a nested field is invisible to ``hasattr`` on the
    class — which is why the test this replaces passed while nothing was asserted."""
    from tanitad.refs.refc_v3 import RefCV3Config
    cfg = RefCV3Config()
    for r in A.CHANNEL_REQUIREMENTS:
        for p in r.predicates:
            A._read_predicate(cfg, p)


# ======================================================================================
# The declaration discipline — symmetric with ChannelExclusion
# ======================================================================================

def test_every_unasserted_channel_states_a_reason_AND_an_unblock():
    """⛔ 'No config flag exists' is a fact about the code, not a reason. An UNASSERTED
    required channel is a decision, and it earns the same discipline the sibling's
    ``ChannelExclusion`` imposes one level up."""
    unasserted = [r for r in A.CHANNEL_REQUIREMENTS if not r.asserted]
    assert unasserted, "vacuity gate: if nothing is unasserted this test is empty"
    for r in unasserted:
        assert len(r.reason.strip()) >= A._MIN_REASON, r.channel
        assert len(r.unblock.strip()) >= A._MIN_UNBLOCK, r.channel
        assert r.evidence.strip() and r.owner.strip(), r.channel


def test_a_declaration_cannot_be_constructed_without_a_reason():
    with pytest.raises(A.RequirementDeclarationError):
        A.ChannelRequirement(channel="lan", owner="a seam", reason="",
                             evidence="PUBLISHED-CODE x.py:1", unblock="x" * 40)


def test_the_exact_placeholder_this_package_removed_is_refused_BY_NAME():
    """⭐ *"no config flag exists — plumbed, not asserted"* was the old note for
    ``nav_known`` and ``withheld_speed``. For BOTH it was also FALSE or irrelevant. It
    must not be typeable again."""
    with pytest.raises(A.RequirementDeclarationError) as ei:
        A.ChannelRequirement(channel="nav_known", owner="a seam",
                             reason="no config flag exists",
                             evidence="PUBLISHED-CODE x.py:1", unblock="x" * 40)
    assert "not a reason" in str(ei.value)


def test_an_unasserted_channel_cannot_omit_its_unblock():
    with pytest.raises(A.RequirementDeclarationError):
        A.ChannelRequirement(channel="v0", owner="a seam", reason="r" * 60,
                             evidence="PUBLISHED-CODE x.py:1")     # no predicates


def test_a_predicate_that_is_not_an_attribute_path_is_refused():
    with pytest.raises(A.RequirementDeclarationError):
        A.ChannelRequirement(channel="v0", owner="a seam", reason="r" * 60,
                             evidence="PUBLISHED-CODE x.py:1",
                             predicates=("core.anchors.v0 conditioned",))


# ======================================================================================
# Coverage + non-vacuity
# ======================================================================================

def test_every_plumbed_channel_carries_a_requirement_record():
    """⛔ THE COVERAGE GUARD. A channel added to ``FORWARD_KEYS`` with no record would
    be plumbed with nobody having decided whether it may be None — the same defect one
    level down."""
    declared = {r.channel for r in A.CHANNEL_REQUIREMENTS}
    assert declared == set(A.FORWARD_KEYS), (
        f"FORWARD_KEYS - declared = {set(A.FORWARD_KEYS) - declared}; "
        f"declared - FORWARD_KEYS = {declared - set(A.FORWARD_KEYS)}")


def test_the_records_are_not_all_the_same_verdict():
    """⛔ NON-VACUITY, IN BOTH DIRECTIONS. If a later edit asserted everything, the
    'protects launches' tests would start refusing valid arms; if it asserted nothing,
    the whole surface would be back to the map it replaced — and a reason-only guard
    would still pass."""
    asserted = [r.channel for r in A.CHANNEL_REQUIREMENTS if r.asserted]
    unasserted = [r.channel for r in A.CHANNEL_REQUIREMENTS if not r.asserted]
    assert asserted and unasserted, (asserted, unasserted)
    assert "v0" in asserted, "v0 is the PI's one ruled-admissible ego input"


def test_nav_known_stays_unasserted_because_the_model_guards_it_better():
    """⭐ A DECLARING FIELD DOES EXIST (``core.nav_known_channel``) — the old note's
    'no config flag exists' was false. It is still not asserted HERE, and the reason is
    positive: `refc.py:3012-3016` legitimately defaults the bit to 0.0 when nav_cmd is
    None, because *the `follow` fallback IS the sentinel*. REF-C's published arm decodes
    with nav_cmd=None, so requiring nav_known whenever the gate is on would refuse the
    programme's own standard eval arm."""
    rec = {r.channel: r for r in A.CHANNEL_REQUIREMENTS}["nav_known"]
    assert not rec.asserted
    # and the guard must NOT fire on that arm:
    A.assert_conditioning(_model(nav_known_channel=True), dict(_FRAMES))


def test_withheld_speed_stays_unasserted_because_the_model_supplies_it_itself():
    """⭐ ``None`` is the CORRECT rollout value: `refc.py:2944-2946` fills it from the
    hierarchy hook's own ``bank_speed_pred``. Asserting it would refuse the normal
    path."""
    rec = {r.channel: r for r in A.CHANNEL_REQUIREMENTS}["withheld_speed"]
    assert not rec.asserted
    assert A.assert_conditioning(_model(), dict(_FRAMES))["withheld_speed"] is False


def test_the_report_banks_the_reason_for_every_channel():
    """A contract that is checked but not written down is not evidence — and for an
    UNASSERTED channel the reason is the half a reader cannot reconstruct."""
    rep = A.requirement_report()
    assert {r["channel"] for r in rep} == set(A.FORWARD_KEYS)
    assert all(r["reason"] and r["evidence"] for r in rep)
