"""The conditioning contract on the adapter a production RL arm ACTUALLY executes.

⛔⛔ WHY THIS FILE EXISTS — WIRING, NOT CORRECTNESS.
On 2026-09-07 three agents hardened ``tanitad.rl.refc_adapter`` with a channel
contract: exclusions carrying reasons and unblock conditions, per-batch assertions,
mutation proofs. It is good work and it had **zero production callers**. MEASURED
(``os.walk`` over 22,892 ``.py`` files, so not an ``rg``-under-reports artifact):

    make_refcv3_sample_fn under stack/scripts/  ->  rl_pilot_refc21.py   (ONLY)
    make_refc_sample_fn    under stack/scripts/  ->  none
    guard tokens in rl_pilot_refc21.py           ->  0, every one of them

⇒ the guarded adapter had no production caller and the adapter the pilot uses had no
guard. ⭐ *Correctness and wiring are different claims*, at the level of a whole module.

⛔ AND THE OBVIOUS FIX WAS WRONG. Repointing the pilot at ``refc_adapter`` looks like
a one-line change and is a defect: the two adapters bind **different classes**.
``test_the_repoint_would_have_FAILED_not_merely_been_redundant`` runs that fix and
pins both of its failures, so the reasoning is executable rather than narrated.

Tier: T0 training-side. ⚠️ Stated precisely, because "no trajectory" would be the
convenient version and it is false: the smoke models here DO emit fans — that is how
the guard is shown to let a correctly conditioned batch through. But they are
randomly-initialised CPU smoke builds and nothing measures those fans, so there is
**no eval number, no eval tier and no four-family table** to report.
"""
from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
import torch
from torch import nn

from tanitad.refs import refc
from tanitad.refs import refc_v3 as v3
from tanitad.rl import PostTrainConfig
from tanitad.rl import refcv3_adapter as ad


# ============================================================================
# Fixtures — REAL models. ⛔ No hand-written stand-in signature stands in for a
# real forward anywhere in this file: a hand-mirrored signature is the defect
# that started this whole thread, and a test that mirrors one cannot catch it.
# ============================================================================

@pytest.fixture(scope="module")
def refc_model():
    """The PILOT's family: `refc.RefCModel`, defaults ⇒ nothing required."""
    return refc.RefCModel(refc.refc_smoke_config())


@pytest.fixture(scope="module")
def refc_model_v0():
    """The same family with the v0 predicate ON (`sel_reach_clamp`)."""
    cfg = refc.refc_smoke_config()
    cfg.sel_reach_clamp = True
    return refc.RefCModel(cfg)


def _batch(model, *, v0=0.0, drop_v0=False):
    """A minimal real batch for the handed model."""
    cfg = model.cfg.core if hasattr(model.cfg, "core") else model.cfg
    enc = cfg.encoder
    frames = torch.zeros(1, cfg.window, enc.in_channels, enc.image_size,
                         enc.image_width or enc.image_size)
    b = {"frames": frames}
    if not drop_v0:
        b["v0"] = torch.full((1,), float(v0))
    return b


def _cfg():
    return PostTrainConfig(method="grpo", group_size=3, steps=1, lr=1e-4)


# ============================================================================
# 1. THE VERDICT, EXECUTABLE — two families, and the repoint fails
# ============================================================================

def test_the_two_adapters_bind_DIFFERENT_CLASSES():
    """⭐ The fact the whole decision rests on, asserted rather than narrated."""
    assert refc.RefCModel is not v3.RefCV3Model
    assert not issubclass(refc.RefCModel, v3.RefCV3Model)
    assert not issubclass(v3.RefCV3Model, refc.RefCModel)
    assert refc.RefCModel.__module__ == "tanitad.refs.refc"
    assert v3.RefCV3Model.__module__ == "tanitad.refs.refc_v3"

    a = set(inspect.signature(refc.RefCModel.forward).parameters)
    b = set(inspect.signature(v3.RefCV3Model.forward).parameters)
    # ⛔ HARD-CODED, both directions. Not "the sets differ" — WHICH channels, so a
    # signature change has to be read by a human instead of re-blessed by a diff.
    assert a - b == {"maneuver_logits", "target_latent", "hierarchy_hook",
                     "ego_keep"}
    assert b - a == {"ego_state", "nav_args", "v_max_ms", "v_max_valid"}


def test_the_configs_NEST_differently_which_is_what_breaks_a_copied_contract():
    """``RefCV3Config.core`` IS a ``RefCConfig`` (`refc_v3.py:347`)."""
    flat = refc.refc_smoke_config()
    nested = v3.refc_v3_smoke_config(hier=False)
    assert isinstance(nested.core, refc.RefCConfig)
    assert not hasattr(flat, "core"), (
        "a plain RefCConfig must NOT have `.core` — if it grew one, every "
        "`core.`-rooted predicate in refc_adapter would start resolving on the "
        "pilot's model and this file's central claim would need re-deriving")
    assert hasattr(flat, "sel_reach_clamp")
    assert hasattr(nested.core, "sel_reach_clamp")


def test_the_repoint_would_have_FAILED_not_merely_been_redundant(refc_model_v0):
    """⛔⛔ THE PROPOSAL THIS PACKAGE REJECTED, RUN RATHER THAN ARGUED.

    Both failure modes are pinned, because "it would not have worked" is a claim
    that has to be executable or it is just a preference.
    """
    from tanitad.rl import refc_adapter as ra

    # (1) refc_adapter's predicates are rooted at `core.` and _read_predicate
    #     RAISES on a path that does not resolve.
    with pytest.raises(ra.ConditioningError, match=r"core\.|does not resolve"):
        ra.conditioning_requirements(refc_model_v0)

    # (2) and its FORWARD_KEYS carry a channel this model has no parameter for.
    assert "ego_state" in ra.FORWARD_KEYS
    assert "ego_state" not in inspect.signature(refc.RefCModel.forward).parameters
    with pytest.raises(TypeError, match="ego_state"):
        refc_model_v0(torch.zeros(1, 4, 1, 64, 64), ego_state=None)


# ============================================================================
# 2. DERIVATION — from the handed model's own signature, never from a tuple
# ============================================================================

def test_the_channel_set_is_derived_from_the_handed_models_OWN_signature(
        refc_model):
    """⛔ HARD-CODED expectations for BOTH families, from their real signatures."""
    got = ad.forward_conditioning_channels(refc_model)
    assert set(got) == {"nav_cmd", "v0", "maneuver_logits", "target_latent",
                        "lan", "nav_known", "hierarchy_hook", "ego_keep",
                        "withheld_speed", "agent_gt"}
    # frames/steps are not conditioning channels; the goal point and the three
    # E13b/E16 channels are LABELS, excluded in the seams that own them.
    for absent in ("frames", "steps", "self", "gp_point", "gp_valid"):
        assert absent not in got, absent


def test_the_v3_family_derives_a_DIFFERENT_set_through_the_SAME_adapter():
    """⭐ The duck-typing that makes a hard-coded tuple impossible to get right."""
    model = v3.RefCV3Model(v3.refc_v3_smoke_config(hier=False))
    got = ad.forward_conditioning_channels(model)
    assert set(got) == {"nav_cmd", "v0", "lan", "nav_known", "ego_state",
                        "withheld_speed", "agent_gt"}
    assert "ego_state" in got and "ego_keep" not in got


def test_an_UNDECLARED_forward_channel_is_REFUSED(refc_model):
    """A forward that grows a channel must break LOUDLY, not run blind to it.

    ⛔ This is the drift that hit ``FORWARD_KEYS`` twice. Deriving from the
    signature only helps if an underived channel REFUSES instead of vanishing.
    """
    from tanitad.rl import refc_adapter as ra

    class GrewAChannel(nn.Module):
        def forward(self, frames, v0=None, brand_new_channel=None):
            raise AssertionError("must never be reached")

    m = GrewAChannel()
    m.cfg = refc.refc_smoke_config()
    with pytest.raises(ra.RequirementDeclarationError,
                       match="brand_new_channel"):
        ad.forward_conditioning_channels(m)


# ============================================================================
# 3. PREDICATE RESOLUTION — against both REAL config shapes
# ============================================================================

def test_predicates_resolve_on_BOTH_real_config_shapes():
    """⚠️ Closes the hole ``_resolve_any``'s docstring names.

    Alternatives mean a typo in ONE path cannot raise, so the paths are checked
    from the other side: for each config shape, EXACTLY which path resolves.
    """
    flat = refc.refc_smoke_config()
    nested = v3.refc_v3_smoke_config(hier=False)

    def resolves(cfg, dotted):
        obj = cfg
        for part in dotted.split("."):
            obj = getattr(obj, part, ad._MISSING)
            if obj is ad._MISSING:
                return False
        return True

    by = {r.channel: r for r in ad.refc_channel_requirements()}
    # v0: the FLAT pair resolves on RefCConfig, the `core.` pair on RefCV3Config,
    # and each shape resolves EXACTLY ONE of every pair.
    for flat_p, nested_p in (("anchors.v0_conditioned",
                              "core.anchors.v0_conditioned"),
                             ("sel_reach_clamp", "core.sel_reach_clamp")):
        assert (flat_p, nested_p) [0] in by["v0"].predicates
        assert nested_p in by["v0"].predicates
        assert resolves(flat, flat_p) and not resolves(flat, nested_p)
        assert resolves(nested, nested_p) and not resolves(nested, flat_p)
    assert resolves(flat, "graft_lan") and not resolves(flat, "core.graft_lan")
    assert resolves(nested, "core.graft_lan")
    # ego_state's predicate exists only on the v3 config — and the channel is only
    # ever in scope there, so no alternative is needed.
    assert by["ego_state"].predicates == ("ego_state_inject",)
    assert resolves(nested, "ego_state_inject")
    assert not resolves(flat, "ego_state_inject")


def test_unresolvable_predicates_RAISE_rather_than_reading_False():
    """⛔ The false-green this class of guard exists to end."""
    from tanitad.rl import refc_adapter as ra

    class Weird:
        pass

    with pytest.raises(ra.ConditioningError, match="NONE of the conditioning"):
        ad._resolve_any(Weird(), ("nope.not_here", "also.missing"))


def test_a_model_with_no_cfg_is_REFUSED():
    from tanitad.rl import refc_adapter as ra

    class NoCfg(nn.Module):
        def forward(self, frames, v0=None):
            raise AssertionError("must never be reached")

    with pytest.raises(ra.ConditioningError, match="has no .cfg"):
        ad.conditioning_requirements(NoCfg())


# ============================================================================
# 4. ⭐ THE OVER-ASSERTION SIDE. Refusing valid launches is how a guard gets
#    DELETED rather than fixed, so it is pinned as hard as the refusal is.
# ============================================================================

def test_a_build_NOT_trained_with_v0_does_NOT_require_it(refc_model):
    """⛔ HARD-CODED: the required set is EXACTLY empty, and a batch with NO v0
    at all must produce a fan. Any over-assertion turns this RED."""
    req = ad.conditioning_requirements(refc_model)
    assert req["v0"] is False
    assert req["lan"] is False
    assert tuple(k for k, v in req.items() if v) == ()

    sample_fn = ad.make_refcv3_sample_fn(refc_model, _cfg())
    traj, logp, ctx = sample_fn(_batch(refc_model, drop_v0=True), _cfg())
    assert traj.ndim == 5 and traj.shape[2] == 3
    assert logp.shape == traj.shape[:3]
    assert sample_fn.conditioning_contract.batches == 1
    assert sample_fn.conditioning_contract.required == ()


def test_a_measured_zero_speed_is_NOT_a_missing_channel(refc_model_v0):
    """⚠️ 0.0 m/s is a genuinely stationary car, not a dropped key.

    Conflating them is the X15 zero-collision in miniature, and it is why the
    check is ``is None`` and never falsiness.
    """
    assert ad.conditioning_requirements(refc_model_v0)["v0"] is True
    sample_fn = ad.make_refcv3_sample_fn(refc_model_v0, _cfg())
    traj, _, _ = sample_fn(_batch(refc_model_v0, v0=0.0), _cfg())
    assert traj.ndim == 5
    assert sample_fn.conditioning_contract.required == ("v0",)


# ============================================================================
# 5. ⭐⭐ THE TEMPORAL PROOF — refusal on the SECOND batch, forward count 1
# ============================================================================

def test_the_contract_refuses_on_the_SECOND_batch_not_only_the_first(
        refc_model_v0):
    """⛔⛔ ONE ``sample_fn``: good batch, then bad batch.

    A once-only latch passes batch 1 and then runs batch 2 *through the model*
    on the replaced action space. So the assertion is not merely "it raises" —
    it is **WHEN** it raises and that **the model was never called a second
    time**. A latch turns both halves RED.
    """
    forwards = []
    handle = refc_model_v0.register_forward_pre_hook(
        lambda *_a, **_k: forwards.append(1))
    try:
        sample_fn = ad.make_refcv3_sample_fn(refc_model_v0, _cfg())
        contract = sample_fn.conditioning_contract

        # batch 1 — conditioned correctly, and it really runs
        sample_fn(_batch(refc_model_v0, v0=7.5), _cfg())
        assert len(forwards) == 1
        assert contract.batches == 1

        # batch 2 — the SAME rollout, v0 dropped
        from tanitad.rl import refc_adapter as ra
        with pytest.raises(ra.ConditioningError) as exc:
            sample_fn(_batch(refc_model_v0, drop_v0=True), _cfg())

        assert "BATCH #2" in str(exc.value)
        assert "a LATER batch" in str(exc.value)
        # ⛔ THE HALF A LATCH FAILS: the forward never ran on the bad batch.
        assert len(forwards) == 1, (
            f"the model was called {len(forwards)} times — the guard refused "
            f"AFTER sampling, which is not a guard")
        assert contract.batches == 2
    finally:
        handle.remove()


def test_the_map_is_resolved_ONCE_while_batches_grows(refc_model_v0):
    """A cache nobody measures is indistinguishable from no cache at all."""
    sample_fn = ad.make_refcv3_sample_fn(refc_model_v0, _cfg())
    for _ in range(3):
        sample_fn(_batch(refc_model_v0, v0=3.0), _cfg())
    c = sample_fn.conditioning_contract
    assert c.batches == 3
    assert c.resolutions == 1


# ============================================================================
# 6. THE WIRING HALF — a REQUIRED channel this adapter never PLUMBS
# ============================================================================

def test_a_required_channel_this_adapter_never_PLUMBS_is_REFUSED():
    """⛔⛔ The defect this guard would otherwise have reproduced one level along.

    ``ego_state`` is assertable on a ``RefCV3Model`` and ``make_refcv3_sample_fn``
    has NEVER passed it. Without this check the channel would be REQUIRED, PRESENT
    in the batch, and STILL arrive as None — every guard green, the policy
    differently conditioned. *Correctness and wiring are different claims.*
    """
    from tanitad.rl import refc_adapter as ra

    cfg = v3.refc_v3_smoke_config(hier=False)
    cfg.ego_state_inject = True
    model = v3.RefCV3Model(cfg)
    assert "ego_state" not in ad.PLUMBED_CHANNELS

    sample_fn = ad.make_refcv3_sample_fn(model, _cfg())
    batch = _batch(model, v0=5.0)
    batch["ego_state"] = torch.zeros(1, 5)      # PRESENT, and still not plumbed
    with pytest.raises(ra.ConditioningError, match="never passes"):
        sample_fn(batch, _cfg())


def test_the_plumbed_set_IS_the_forward_call_not_a_comment(refc_model):
    """``_forward_kwargs`` iterates PLUMBED_CHANNELS, so the two cannot drift."""
    kw = ad._forward_kwargs({"frames": None, "v0": torch.zeros(1)},
                            PostTrainConfig(decoder_steps=2))
    assert set(kw) == set(ad.PLUMBED_CHANNELS) | {"steps"}
    assert kw["steps"] == 2
    assert kw["nav_cmd"] is None and kw["lan"] is None


# ============================================================================
# 7. THE ESCAPE HATCH AND THE IMPORT CYCLE
# ============================================================================

def test_strict_conditioning_False_is_typed_AND_recorded(refc_model_v0):
    """An ablation arm must be bankable from the OBJECT, not from an argv."""
    sample_fn = ad.make_refcv3_sample_fn(refc_model_v0, _cfg(),
                                         strict_conditioning=False)
    assert sample_fn.strict_conditioning is False
    assert sample_fn.conditioning_contract is None
    # and it really does sample the un-conditioned policy, which is the point
    traj, _, _ = sample_fn(_batch(refc_model_v0, drop_v0=True), _cfg())
    assert traj.ndim == 5

    on = ad.make_refcv3_sample_fn(refc_model_v0, _cfg())
    assert on.strict_conditioning is True
    assert on.conditioning_contract is not None


@pytest.mark.parametrize("first,second", [
    ("tanitad.rl.refc_adapter", "tanitad.rl.refcv3_adapter"),
    ("tanitad.rl.refcv3_adapter", "tanitad.rl.refc_adapter"),
])
def test_the_lazy_import_holds_in_BOTH_orders(first, second):
    """⛔ ``refc_adapter`` imports this module at module scope; this module
    imports it back at CALL time. A regression to a top-level import is a hard
    ImportError only in ONE of the two orders, so both are run."""
    code = (f"import importlib;"
            f"importlib.import_module({first!r});"
            f"m=importlib.import_module({second!r});"
            f"import tanitad.rl.refcv3_adapter as a;"
            f"print(len(a.refc_channel_requirements()))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, f"{first} then {second}:\n{r.stderr[-2000:]}"
    assert r.stdout.strip() == "11"


def test_every_declaration_is_reviewable_and_covers_both_families():
    """The machinery is reused; the VALUES were established for THIS adapter.

    ⛔ So the check is not "they exist" — it is that they are not
    ``refc_adapter``'s, and that a non-asserted channel says what would change
    that (``ChannelRequirement`` refuses construction otherwise, and this forces
    the construction so that refusal cannot go unexercised).
    """
    from tanitad.rl import refc_adapter as ra

    decls = ad.refc_channel_requirements()
    assert len(decls) == 11
    mine = {r.channel: r for r in decls}
    theirs = {r.channel: r for r in ra.CHANNEL_REQUIREMENTS}

    # every channel either family's forward exposes is declared here
    union = (set(inspect.signature(refc.RefCModel.forward).parameters)
             | set(inspect.signature(v3.RefCV3Model.forward).parameters))
    excluded = set(__import__("tanitad.channel_admissibility", fromlist=["x"])
                   .excluded_channels())
    expected = union - ad.NON_CHANNEL_PARAMS - excluded
    assert set(mine) == expected

    for name, r in mine.items():
        assert r.asserted == bool(r.predicates)
        if not r.asserted:
            assert len(r.unblock.strip()) >= 20, name
        # ⛔ not inherited: every shared channel carries a DIFFERENT reason, and
        # a `core.`-rooted predicate here would be the copied-contract bug.
        if name in theirs:
            assert r.reason != theirs[name].reason, (
                f"{name}'s reason is byte-identical to refc_adapter's — the "
                f"values must be established for THIS adapter's models")
        for p in r.predicates:
            assert not p.startswith("core.") or any(
                q == p.removeprefix("core.") for q in r.predicates), (
                f"{name}: `{p}` is v3-rooted with no flat alternative, so it "
                f"could never resolve on the pilot's RefCConfig")
