# -*- coding: utf-8 -*-
"""The conditioning contract is asserted on EVERY batch, not on the first one.

The two sibling packages closed the KEY SET (`Research/2026-09-07-rl-channel-admissibility/`
— which channels an RL rollout may receive is a declared decision per seam) and the VALUE
FLOW's coverage (`Research/2026-09-07-rl-channel-values/` — the requirement map now reads
NESTED declaring fields and raises on a path that does not resolve, instead of reading
False forever). The second ended by naming the one hole it did not close:

    "the guard fires on the FIRST BATCH ONLY (`refc_adapter.py:505`,
     `checked["done"] = True`). A rollout whose *later* batches drop a channel would not
     be caught."

⛔ THAT WAS A COST/BENEFIT CALL THAT HAD INVERTED, and the inversion is measured, not
argued. Checking once is reasonable while a miss costs "a softer policy". A missing ``v0``
does not soften the policy — it SILENTLY REPLACES THE ACTION SPACE:

    `refc.py:3097`  v_ms = v0 if (cfg.sel_reach_clamp and v0 is not None) else None
    `refc.py:2128`  bank = self.roll_bank(v_ms, ego_keep, ...)
    `refc.py:1722`  if v_ms is None or anchor_withheld_bank == "none": v = full(ref_speed)

⇒ every anchor rolled at the 10 m/s reference instead of the window's measured speed
(MEASURED vocabulary gap, `refc.py:363-370`: 0.3773 m oracle-in-vocabulary fixed-path vs
0.2610 m v0-conditioned), plus `refc.py:2360` skipping the S2 reachability band and
`refc.py:2970-2977` setting keep=0 on 100 % of rows where training saw keep=1.

⭐⭐ WHAT THIS FILE IS FOR, AND WHY ITS SHAPE IS THE POINT. The defect is TEMPORAL: a
batch that passes, followed by one that does not. A test that only ever feeds ONE batch
cannot distinguish "checked once" from "checked every time" — it is green under both, and
that is precisely how the hole survived. So every proof below drives the SAME sample_fn
repeatedly and pins WHICH call raises.

⛔ AND NO EXPECTATION HERE IS "WHATEVER THE CODE DOES". The sibling found
`test_rl_channel_guard.py` reading ``if missing: expect refusal; else: expect pass`` — an
expected value computed from the code under test, which can never fail. Every expectation
below is a hard-coded literal: this call must pass, that call must raise, this counter is
1 and that one is 5.

⛔ NO EVAL TIER AND NO FOUR-FAMILY TABLE: no model produces a trajectory in this file. The
model is a stub that returns zeros. It is a wiring-contract surface, and stamping a
longitudinal/lateral/tactical/strategic table on it would be a category error.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.rl import refc_adapter as A                              # noqa: E402
from tanitad.rl.config import PostTrainConfig                         # noqa: E402


# ======================================================================================
# Fixtures — the REAL config, a stub forward that RECORDS what it was given
# ======================================================================================

def _cfg(**kw):
    """⭐ THE REAL ``RefCV3Config``. A flat two-attribute stub is how the nested
    predicates stayed invisible to the test surface as well as to the guard."""
    from tanitad.refs.refc_v3 import RefCV3Config
    cfg = RefCV3Config()
    cfg.ego_state_inject = kw.get("ego_state_inject", False)
    cfg.core.anchors.v0_conditioned = kw.get("v0_conditioned", False)
    cfg.core.sel_reach_clamp = kw.get("sel_reach_clamp", False)
    cfg.core.graft_lan = kw.get("graft_lan", False)
    cfg.core.nav_known_channel = kw.get("nav_known_channel", False)
    return cfg


class _StubModel:
    """Returns the two tensors ``sample_fn`` reads, and RECORDS the kwargs it got.

    ⭐ The recording is not decoration. "The guard refused" and "the batch never reached
    the forward" are different claims, and the second is the one that matters when the
    question is whether a defective batch was SAMPLED FROM.
    """

    N, S = 2, 4

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[dict] = []

    def __call__(self, frames, **kw):
        self.calls.append(kw)
        z = torch.zeros(1, self.N, self.S, 2)
        return {"anchor_traj": z, "offset": z.clone()}


def _good():
    """A batch carrying every channel a fully-conditioned build declares."""
    return {"frames": torch.zeros(1, 4, 1, 8, 8),
            "v0": torch.full((1,), 7.5),
            "ego_state": torch.zeros(1, 5),
            "lan": torch.zeros(1, 2, 4)}


def _no_v0():
    b = _good()
    del b["v0"]
    return b


def _ptc():
    return PostTrainConfig()


# ======================================================================================
# ⭐⭐ THE TEMPORAL PROOF — a good batch, THEN a bad one
# ======================================================================================

def test_a_LATER_batch_that_drops_v0_is_refused_on_that_call():
    """⛔ THE WHOLE POINT. Batches 1 and 2 are good and MUST pass; batch 3 drops ``v0``
    and MUST raise ON THAT CALL.

    Under the once-only guard this sequence completed silently and returned a
    well-formed fan sampled from the 10 m/s fixed-path vocabulary. The expectations are
    hard-coded: nothing here asks the code what it intends to do.
    """
    m = _StubModel(_cfg(v0_conditioned=True, ego_state_inject=True))
    fn = A.make_refc_sample_fn(m, _ptc())

    fn(_good(), _ptc())                       # batch 1 — must not raise
    fn(_good(), _ptc())                       # batch 2 — must not raise
    assert len(m.calls) == 2, "the two GOOD batches must have reached the forward"

    with pytest.raises(A.ConditioningError) as ei:
        fn(_no_v0(), _ptc())                  # batch 3 — MUST raise
    msg = str(ei.value)
    assert "v0" in msg
    assert "#3" in msg, f"the refusal must name WHICH batch failed; got: {msg[:200]}"
    assert "LATER batch" in msg, "an operator must be told this was not the first batch"

    # ⛔ and the defective batch must never have been sampled from
    assert len(m.calls) == 2, (
        f"the forward ran {len(m.calls)} times; the third (v0-less) batch reached it, "
        f"which means the guard refused AFTER the policy had already been sampled")


def test_the_refusal_is_not_a_one_shot_either_batch_5_still_refuses():
    """A guard that fires once and then goes quiet is the same defect wearing the
    opposite sign. Batches 1-4 good, 5 bad, 6 good, 7 bad — hard-coded."""
    m = _StubModel(_cfg(v0_conditioned=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    for _ in range(4):
        fn(_good(), _ptc())
    with pytest.raises(A.ConditioningError):
        fn(_no_v0(), _ptc())
    fn(_good(), _ptc())                       # recovers — a refusal is not a latch
    with pytest.raises(A.ConditioningError):
        fn(_no_v0(), _ptc())
    assert len(m.calls) == 5, (
        f"exactly the 5 GOOD batches may reach the forward, got {len(m.calls)}")


def test_ego_state_is_the_same_story_one_channel_over():
    """The temporal hole was never v0-specific. refcv4b's ``ego_state`` — the channel
    this whole module was created for — must also be caught on a LATER batch."""
    m = _StubModel(_cfg(ego_state_inject=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    fn(_good(), _ptc())
    bad = _good()
    del bad["ego_state"]
    with pytest.raises(A.ConditioningError) as ei:
        fn(bad, _ptc())
    assert "ego_state" in str(ei.value)
    assert len(m.calls) == 1


def test_a_None_VALUE_on_a_later_batch_is_refused_not_only_a_missing_KEY():
    """``forward_kwargs`` does ``batch.get(k)``, so ``{"v0": None}`` and a missing key
    are the same thing downstream. The guard must not distinguish them either."""
    m = _StubModel(_cfg(v0_conditioned=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    fn(_good(), _ptc())
    nulled = _good()
    nulled["v0"] = None
    with pytest.raises(A.ConditioningError):
        fn(nulled, _ptc())


# ======================================================================================
# ⭐ THE VACUITY GATES — a guard that refuses everything proves nothing
# ======================================================================================

def test_a_hundred_good_batches_all_pass_and_all_reach_the_forward():
    """⛔ WITHOUT THIS, EVERY TEST ABOVE IS SATISFIED BY ``raise`` AS THE FIRST LINE OF
    ``sample_fn``. Hard-coded: 100 calls, 0 refusals, 100 forwards."""
    m = _StubModel(_cfg(v0_conditioned=True, ego_state_inject=True, graft_lan=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    for _ in range(100):
        traj, logp, ctx = fn(_good(), _ptc())
    assert len(m.calls) == 100
    assert traj.shape[:2] == (1, 2) and traj.shape[-1] == 2
    assert all(c["v0"] is not None for c in m.calls), (
        "the VALUE must reach the forward, not merely the key")


def test_an_UNCONDITIONED_build_accepts_a_v0_less_batch_on_every_call():
    """⭐ THE OVER-ASSERTION CONTROL, kept alive from the sibling's M4. A build whose
    config says it was NOT trained with ``v0`` must accept a ``v0``-less batch — on
    batch 1 and on batch 20 alike.

    ⛔ Refusing valid launches is how a guard gets deleted rather than fixed, and moving
    a check into the loop multiplies that risk by the number of batches.
    """
    m = _StubModel(_cfg())                    # every predicate False
    fn = A.make_refc_sample_fn(m, _ptc())
    for _ in range(20):
        fn(_no_v0(), _ptc())
    assert len(m.calls) == 20


def test_nav_cmd_is_never_required_on_any_batch_the_C6_eval_arm_must_survive():
    """REF-C's published arm decodes with ``nav_cmd=None`` (the C6 confound /
    ``os_navzero``). A per-batch guard that demanded it would refuse the programme's own
    standard arm on every step instead of once."""
    m = _StubModel(_cfg(nav_known_channel=True, v0_conditioned=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    for _ in range(5):
        fn(_good(), _ptc())                   # no nav_cmd, no nav_known anywhere
    assert len(m.calls) == 5


# ======================================================================================
# ⭐ THE CACHE — resolved once, asserted every time
# ======================================================================================

def test_the_requirement_map_is_resolved_ONCE_and_the_contract_checked_EVERY_batch():
    """⛔ A cache nobody measures is indistinguishable from no cache. Hard-coded: after
    50 batches, ``resolutions == 1`` and ``batches == 50``."""
    m = _StubModel(_cfg(v0_conditioned=True, ego_state_inject=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    for _ in range(50):
        fn(_good(), _ptc())
    c = fn.conditioning_contract
    assert c.resolutions == 1, (
        f"the dotted-path walk ran {c.resolutions} times; it is the only part with any "
        f"cost and it must run once per rollout")
    assert c.batches == 50, f"only {c.batches} of 50 batches were checked"
    assert c.required == ("ego_state", "v0"), c.required


def test_a_SWAPPED_config_re_resolves_so_the_cache_is_not_blind():
    """The map is keyed on the identity of ``model.cfg``. Swapping the config to one
    that was NOT trained with ``v0`` must re-resolve, so the contract describes the
    model actually being sampled. Hard-coded: refuses before the swap, accepts after,
    and ``resolutions`` goes 1 -> 2.

    ⚠️ STATED SCOPE: an IN-PLACE mutation of the same object is deliberately not
    covered — a config mutated mid-rollout changes the policy under the optimiser and is
    a different defect class than a batch that forgot a key.
    """
    m = _StubModel(_cfg(v0_conditioned=True))
    fn = A.make_refc_sample_fn(m, _ptc())
    fn(_good(), _ptc())
    with pytest.raises(A.ConditioningError):
        fn(_no_v0(), _ptc())
    assert fn.conditioning_contract.resolutions == 1

    m.cfg = _cfg()                            # a build that needs nothing
    fn(_no_v0(), _ptc())                      # must now be accepted
    assert fn.conditioning_contract.resolutions == 2


def test_a_declaration_bug_is_loud_on_every_batch_not_swallowed_by_the_cache():
    """An unresolvable predicate must never be cached as "not required". The cfg is
    recorded only AFTER the resolve succeeds, so a broken declaration keeps raising."""
    m = _StubModel(_cfg())
    fn = A.make_refc_sample_fn(m, _ptc())
    fn(_good(), _ptc())
    m.cfg = object()                          # no attributes at all
    for _ in range(3):
        with pytest.raises(A.ConditioningError):
            fn(_good(), _ptc())
    assert fn.conditioning_contract.resolutions == 1, (
        "a failed resolve must not be recorded as a successful one")


def test_a_model_with_no_cfg_is_refused_rather_than_guessed():
    class _NoCfg(_StubModel):
        def __init__(self):
            self.cfg = None
            self.calls = []
    m = _NoCfg()
    fn = A.make_refc_sample_fn(m, _ptc())
    with pytest.raises(A.ConditioningError) as ei:
        fn(_good(), _ptc())
    assert "no .cfg" in str(ei.value)
    assert m.calls == []


# ======================================================================================
# ⭐ THE ESCAPE HATCH — unchanged, and now actually RECORDED
# ======================================================================================

def test_strict_conditioning_False_lets_EVERY_batch_through_including_later_ones():
    """⛔ THE DELIBERATE UN-CONDITIONED ABLATION ARM MUST STILL WORK, AND MUST NOT BE
    SILENTLY RE-ARMED BY THE PER-BATCH CHECK. Hard-coded: a v0-conditioned build, 10
    batches with no ``v0`` at all, zero refusals, 10 forwards — and the forward is
    handed ``v0=None`` each time, which is what an un-conditioned arm asked for."""
    m = _StubModel(_cfg(v0_conditioned=True, ego_state_inject=True))
    fn = A.make_refc_sample_fn(m, _ptc(), strict_conditioning=False)
    for _ in range(10):
        fn(_no_v0(), _ptc())
    assert len(m.calls) == 10
    assert all(c["v0"] is None for c in m.calls)


def test_the_ablation_arm_resolves_nothing_at_all():
    """``strict_conditioning=False`` must not merely swallow the refusal — it must not
    run the contract. A build with a BROKEN declaration is still samplable, because the
    arm never consults the declaration."""
    m = _StubModel(object())                  # a cfg on which no predicate resolves
    fn = A.make_refc_sample_fn(m, _ptc(), strict_conditioning=False)
    for _ in range(3):
        fn(_no_v0(), _ptc())
    assert len(m.calls) == 3
    assert fn.conditioning_contract is None


def test_the_arm_is_RECORDED_on_the_callable_not_only_in_the_docstring():
    """*"it must be typed, and it is recorded"* — a run record reads these instead of
    trusting an argv string."""
    m = _StubModel(_cfg(v0_conditioned=True))
    strict = A.make_refc_sample_fn(m, _ptc())
    loose = A.make_refc_sample_fn(m, _ptc(), strict_conditioning=False)
    assert strict.strict_conditioning is True
    assert loose.strict_conditioning is False
    assert isinstance(strict.conditioning_contract, A.ConditioningContract)
    assert loose.conditioning_contract is None


def test_strict_is_the_DEFAULT_so_the_hatch_cannot_be_taken_by_omission():
    m = _StubModel(_cfg(v0_conditioned=True))
    fn = A.make_refc_sample_fn(m, _ptc())     # nothing typed
    with pytest.raises(A.ConditioningError):
        fn(_no_v0(), _ptc())


# ======================================================================================
# The single-shot form is unchanged
# ======================================================================================

def test_assert_conditioning_still_refuses_and_carries_no_batch_index():
    """The standalone entry point keeps its exact contract; only the rollout path
    counts batches. One refusal text, two callers — they cannot drift."""
    m = _StubModel(_cfg(v0_conditioned=True))
    with pytest.raises(A.ConditioningError) as ei:
        A.assert_conditioning(m, {"frames": torch.zeros(1)})
    msg = str(ei.value)
    assert "v0" in msg and "DIFFERENTLY CONDITIONED" in msg
    assert "REFUSED ON BATCH" not in msg
    assert A.assert_conditioning(m, _good())["v0"] is True
