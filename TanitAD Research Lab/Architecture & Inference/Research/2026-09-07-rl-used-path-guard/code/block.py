# ============================================================================
# THE CONDITIONING CONTRACT FOR *THIS* ADAPTER — derived, never inherited
# ============================================================================
# ⛔⛔ WHY THIS EXISTS SEPARATELY FROM ``refc_adapter``'s CONTRACT.
#
# ``refc_adapter`` was hardened on 2026-09-07 with a per-batch channel contract,
# and it is excellent — for the model it describes. It describes
# ``refc_v3.RefCV3Model``. **This module's only production caller hands it a
# ``refc.RefCModel``**, which is a DIFFERENT CLASS in a DIFFERENT MODULE with a
# DIFFERENT forward signature and a DIFFERENTLY SHAPED config:
#
#     stack/scripts/rl_pilot_refc21.py:169   refc.RefCModel(refc.refc_config())
#     stack/tests/test_rl_refcv3_integration.py:31   v3.RefCV3Model(...)
#
# ⇒ this adapter is **duck-typed and serves both families**, which is why the
# requirement set below is resolved from the model it is HANDED and not written
# down as a tuple. MEASURED symmetric difference of the two signatures
# (`refc.py:2870-2881` vs `refc_v3.py:1361-1372`):
#
#     only on RefCModel    maneuver_logits, target_latent, hierarchy_hook, ego_keep
#     only on RefCV3Model  ego_state, nav_args, v_max_ms, v_max_valid
#
# ⛔ AND THE CONFIGS NEST DIFFERENTLY, which is the part that silently breaks a
# copied contract. ``RefCV3Config.core`` IS a ``RefCConfig`` (`refc_v3.py:347`), so
# ``refc_adapter`` correctly writes ``core.anchors.v0_conditioned``. On a plain
# ``RefCModel`` the very same field is ``anchors.v0_conditioned`` — and
# ``refc_adapter._read_predicate`` RAISES on a path that does not resolve. Copying
# its declarations here would therefore have refused **every** pilot batch with a
# message about a config attribute the pilot's model was never supposed to have.
#
# ⭐ SO: the MACHINERY is reused (``ChannelRequirement`` still refuses construction
# without reason/evidence/unblock/owner); the VALUES were re-established from
# ``refc.py`` on 2026-09-07 and every line cite below was read from source.
#
# ⚠️ WHAT THIS GUARD DOES NOT COVER, STATED RATHER THAN IMPLIED. It asserts what
# the BUILD DECLARES. ``rl_pilot_refc21.py`` rebuilds its config from
# ``refc_config()`` defaults and loads ``ck["model"]`` with ``strict=True``; the
# checkpoint's own config is never consulted, and ``EXPECT_PARAMS`` cannot catch a
# mismatch on these flags because none of them changes the parameter count
# (`refc.py:1391` stores ``v0_conditioned`` as a plain bool). If that checkpoint was
# trained v0-conditioned, the wrong action space is already selected at BUILD time
# and no config-derived guard can see it — that is a checkpoint-provenance defect,
# it is ESCALATED, and it is not what this contract claims to fix.

import inspect
from functools import lru_cache

_MISSING = object()

#: Parameters of a forward that are NOT conditioning channels. ``frames`` is the
#: observation and is always supplied positionally; ``steps`` selects the decoder
#: mode and is set from the RL config, not from the batch.
NON_CHANNEL_PARAMS = frozenset({"self", "frames", "steps"})

#: ⛔⛔ THE CHANNELS THIS ADAPTER ACTUALLY PASSES TO THE FORWARD. Not a comment —
#: :func:`make_refcv3_sample_fn` builds its kwargs BY ITERATING THIS TUPLE, so it
#: cannot drift from the call the way a documented list would.
#:
#: ⭐ AND IT IS ENFORCED AGAINST THE REQUIREMENT MAP. Without that, this guard would
#: have had the exact defect it was built to remove, one level along: a channel that
#: is REQUIRED (the config declares it) and PRESENT (the batch carries it) but NOT
#: PLUMBED (this adapter never forwards it) would pass every check and still reach
#: the model as None. ``ego_state`` is precisely that case — it is assertable on a
#: ``RefCV3Model`` and this adapter has never passed it. ⇒ a required channel outside
#: this tuple is a REFUSAL, not a pass: *correctness and wiring are different claims,*
#: and a contract that only checks the batch is asserting only one of them.
PLUMBED_CHANNELS: tuple[str, ...] = ("nav_cmd", "v0", "lan")


def _machinery():
    """The shared declaration machinery, imported LAZILY.

    ⛔ ``refc_adapter`` does ``from .refcv3_adapter import gt_context,
    sample_offsets`` at module scope, so a top-level import in the other direction
    is a cycle. Deferring it to call time is what lets this module REUSE the
    declaration types rather than fork a second copy of them — and a forked copy is
    how two contracts drift into disagreeing about the same word.
    """
    from . import refc_adapter as _ra
    return _ra


@lru_cache(maxsize=1)
def refc_channel_requirements():
    """The per-channel ruling for the models THIS adapter is handed.

    ⛔ Not ``refc_adapter.CHANNEL_REQUIREMENTS``. Every reason and every predicate
    path below was established against ``refc.py`` (the pilot's model) on
    2026-09-07; the two tables agree in places because the two models share
    mechanisms, and they differ in exactly the places the models differ.

    ⚠️ Built lazily, so a malformed declaration raises at CONTRACT CONSTRUCTION —
    i.e. inside :func:`make_refcv3_sample_fn`, before a single batch — rather than
    at import. ``test_rl_refcv3_used_path_guard.py`` forces the build directly so
    the validation cannot go unexercised.
    """
    CR = _machinery().ChannelRequirement
    return (
        # ---- ASSERTED -----------------------------------------------------
        CR(
            channel="v0",
            owner="REF-C anchor seam (v0-conditioned vocabulary) — arch-inf 2026-09-07",
            predicates=("anchors.v0_conditioned", "core.anchors.v0_conditioned",
                        "sel_reach_clamp", "core.sel_reach_clamp"),
            reason="⛔ a missing v0 REPLACES THE ACTION SPACE, it does not merely "
                   "soften the policy. `refc.py:3097` reads `v_ms = v0 if "
                   "(self.cfg.sel_reach_clamp and v0 is not None) else None`; "
                   "`refc.py:2128` hands that to `roll_bank`; `refc.py:1722` reads "
                   "`if v_ms is None or ...: v = full(ref_speed)` — every anchor is "
                   "rolled at the 10 m/s reference instead of the window's measured "
                   "speed, and nothing raises. Two further silent sites on the SAME "
                   "omission: `refc.py:2360` (`if sel.reach_clamp and v_ms is not "
                   "None`) skips the S2 reachability band entirely, and "
                   "`refc.py:2971-2978` sets v = 0 AND keep = 0 on 100 % of rows, "
                   "which is the X15 zero-collision — a withheld speed made "
                   "indistinguishable from a genuinely stationary car.",
            evidence="PUBLISHED-CODE, read from source 2026-09-07: `refc.py:3097-3098`, "
                     "`:2128-2129`, `:1722`, `:2360`, `:2971-2978`; the flags at "
                     "`refc.py:383` (AnchorConfig.v0_conditioned) and `:778` "
                     "(sel_reach_clamp), both defaulting False."),
        CR(
            channel="lan",
            owner="LAN route-corridor seam (graft_lan) — arch-inf 2026-09-07",
            predicates=("graft_lan", "core.graft_lan"),
            reason="⛔ `refc.py:3072` reads `if self.cfg.graft_lan and lan is not "
                   "None:` — with lan None the route encoder is SKIPPED and the "
                   "decoder runs with no corridor at all; the forward's own docstring "
                   "says it outright (`refc.py:2887`: *None -> the seam is skipped "
                   "entirely*). ⭐ 'the route is missing on most windows anyway' is "
                   "NOT a defence: when graft_lan is on, every window carries a lan "
                   "tensor and 'no route here' is expressed by the per-anchor VALID "
                   "FLAG. lan=None is not that in-distribution state — it bypasses the "
                   "encoder instead of feeding it an honest zero.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:3072-3074`, `:2886-2888` "
                     "(the docstring), `:693` (the flag, default False)."),
        CR(
            channel="ego_state",
            owner="refcv4b ego seam — carried for this adapter's RefCV3Model caller",
            predicates=("ego_state_inject",),
            reason="the v4 [B, 5] ego block, and it is in scope here ONLY because "
                   "`test_rl_refcv3_integration.py` hands this adapter a real "
                   "RefCV3Model. Omitting it on a build trained with it returns a "
                   "well-formed fan from a differently conditioned policy and nothing "
                   "raises; the forward refuses the OTHER direction loudly "
                   "(`refc_v3.py:1410-1414`) but is silent on omission. ⛔ "
                   "`refc.RefCModel` has NO such parameter — its ego channel is "
                   "`ego_keep` — so on the pilot's model it is never even derived.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc_v3.py:1365` (the parameter), "
                     "`:454` (the flag), `:1410-1414` (the converse refusal); ABSENT "
                     "from `refc.py` (0 occurrences of `ego_state` in 197,062 chars)."),

        # ---- NOT ASSERTED. Each says why, and what would change that. ------
        CR(
            channel="nav_cmd",
            owner="nav seam / the C6 confound arm — arch-inf 2026-09-07",
            reason="⛔ asserting it would be WRONG, not merely unnecessary. REF-C's "
                   "published eval arm decodes with nav_cmd=None ON PURPOSE (the C6 "
                   "confound), and `refc.py:2967-2970` makes that a FIRST-CLASS input "
                   "rather than an omission: `nav_cmd_given` is recorded and the "
                   "`follow` index-0 fallback is substituted deliberately. A "
                   "requirement here would refuse the programme's own standard arm.",
            unblock="⛔ nothing should — this is a decision, not a gap. A per-arm 'this "
                    "run intends to supply nav' declaration could carry it, but the "
                    "guard would then be asserting the CALLER'S INTENT rather than the "
                    "checkpoint's training, which is a different instrument.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:2967-2970` (`nav_cmd_given`, "
                     "the follow fallback)."),
        CR(
            channel="nav_known",
            owner="E1 nav companion-bit seam — arch-inf 2026-09-07",
            reason="⭐ GUARDED BY THE MODEL ITSELF, in both silent directions, so there "
                   "is no silent divergence left for this contract to catch: "
                   "`refc.py:3006-3009` refuses a nav_known supplied to a build with "
                   "the gate off, and `refc.py:3018-3022` refuses a missing bit when "
                   "the gate is on AND a nav_cmd was supplied (*Defaulting it to 1.0 "
                   "would assert a judgement the labeller never made*). ⛔ The third "
                   "branch is why a flat assertion would be actively WRONG: "
                   "`refc.py:3012-3016` — gate on, nav_cmd None — legitimately defaults "
                   "the bit to 0.0 because the `follow` fallback IS the sentinel. "
                   "Requiring it whenever the gate is on would refuse the C6 arm.",
            unblock="a BATCH-CONDITIONAL predicate (require nav_known only when the "
                    "batch also carries nav_cmd) rather than a config one. ⚠️ That is "
                    "exactly what `refc.py:3017-3022` already does one layer down and "
                    "with a better message, so duplicating it here would buy nothing "
                    "but a second copy to keep in sync.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:3006-3009`, `:3010-3022`; the "
                     "field `refc.py:836` (nav_known_channel, default False)."),
        CR(
            channel="withheld_speed",
            owner="H-EGO-LIT-4 withheld-bank seam — arch-inf 2026-09-07",
            reason="⭐ NOT AN EXTERNAL INPUT — `None` is the CORRECT rollout value, so "
                   "asserting it would refuse the normal path. It is the model's OWN "
                   "predicted 2 s speed routed back in (`refc.py:1697`: *the model's "
                   "OWN predicted speed*), and when the caller passes None the "
                   "hierarchy hook FILLS IT IN ITSELF: `refc.py:2945-2946`, `if "
                   "withheld_speed is None: withheld_speed = hk.get('bank_speed_pred')`. "
                   "Its one external use is the eval-time SHUFFLE control, which passes "
                   "a deliberately PERMUTED copy — the opposite of a channel you would "
                   "require.",
            unblock="⛔ nothing should. If a future arm ever supplies a MEASURED speed "
                    "here rather than the model's own prediction, that arm needs its "
                    "own declaring field and this record must be REVISITED then rather "
                    "than silently inherited.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:1697-1699`, `:2945-2946`, "
                     "`:1692` (roll_bank's signature)."),
        CR(
            channel="agent_gt",
            owner="agent-oracle seam — arch-inf 2026-09-07",
            reason="⭐ GUARDED BY THE MODEL ITSELF and unconditionally, so there is no "
                   "silent divergence here either: `refc.py:3127-3131` raises when the "
                   "oracle path is built and no `agent_gt` reached the forward. ⚠️ It "
                   "is also read ONLY on that path (`refc.py:3120`), so on every "
                   "non-oracle build a required-agent_gt assertion would refuse a "
                   "batch the model does not want.",
            unblock="nothing needs to — but if the model's refusal were ever relaxed, "
                    "the declaring predicates ('agents.enable', 'agents.oracle', and "
                    "their `core.`-prefixed forms) resolve correctly through "
                    "`_resolve_any` and this record can simply be given them.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:3120`, `:3127-3131`; the "
                     "config slot `refc.py:618` (`agents: object | None = None`)."),
        CR(
            channel="ego_keep",
            owner="REF-C v4 withholding-draw seam — arch-inf 2026-09-07",
            reason="⭐ `None` IS THE DEPLOYED ROLLOUT VALUE and asserting it would "
                   "refuse every correct launch. `refc.py:2994-2996` uses it only to "
                   "OVERRIDE the internal withholding draw; with `ego_keep=None` the "
                   "branch is dead and the forward is byte-identical to the pre-seam "
                   "file (`refc.py:2980-2982` says exactly that). ⛔ And an RL rollout "
                   "runs in eval mode, so the `elif self.training and ego_dropout > 0` "
                   "arm at `refc.py:2997-3000` is dead too: supplying ego_keep would "
                   "WITHHOLD the ego channel on a rollout that means to sample the "
                   "deployed policy.",
            unblock="⛔ nothing should. A withholding ABLATION arm would supply it "
                    "deliberately, and that arm's own record is where the intent "
                    "belongs — a guard that required it would make the ablation the "
                    "default, which inverts the experiment.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:2994-3000`, `:2979-2993` (the "
                     "seam's own rationale), `:624` (ego_dropout, train-time only)."),
        CR(
            channel="maneuver_logits",
            owner="external tactical-brain seam — arch-inf 2026-09-07",
            reason="⭐ `None` is the CORRECT and DEFAULT value: the forward's docstring "
                   "(`refc.py:2883-2886`) states that with the port unfilled *the "
                   "model's own maneuver head drives the H19 reweight*. It is an "
                   "OPTIONAL EXTERNAL OVERRIDE for a hierarchy built outside the class, "
                   "not an observation, and `refc.py:2937-2938` shows the hierarchy "
                   "hook filling it in-forward where the caller passed None. Requiring "
                   "it would refuse every build that has no external tactical brain — "
                   "which is every build this adapter is used on today.",
            unblock="⛔ nothing should. A hierarchy arm that MEANS to drive this port "
                    "declares that in its own run record; a config predicate cannot "
                    "distinguish 'has a tactical brain' from 'intends to use it on "
                    "this rollout', and guessing is what makes a guard get deleted.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:2872` (the parameter), "
                     "`:2883-2886` (the docstring), `:2937-2938` (the hook fill)."),
        CR(
            channel="target_latent",
            owner="external tactical-brain seam (target-latent FiLM) — arch-inf 2026-09-07",
            reason="⭐ `None` is CORRECT and is the documented default — the same port "
                   "pair as `maneuver_logits`: with it unfilled *the target-latent FiLM "
                   "stays inactive* (`refc.py:2884-2886`). It is an external override, "
                   "not an observation, and the hook supplies it in-forward where the "
                   "caller passed None (`refc.py:2939-2940`, `if target_latent is None: "
                   "target_latent = hk.get('target_latent')`). Asserting it would "
                   "refuse every non-hierarchy build.",
            unblock="⛔ nothing should — see `maneuver_logits`. The two ports are one "
                    "decision and must not drift into being two: if a future arm makes "
                    "the FiLM mandatory, BOTH records change together.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:2873` (the parameter), "
                     "`:2884-2886` (the docstring), `:2939-2940` (the hook fill)."),
        CR(
            channel="hierarchy_hook",
            owner="REF-C v3 in-forward supplier — arch-inf 2026-09-07",
            reason="⛔ NOT A TENSOR CHANNEL AT ALL — it is a CALLABLE, and it is in this "
                   "table only because it is a parameter of `RefCModel.forward` and "
                   "this contract derives its scope from the signature rather than from "
                   "a hand-kept tuple. `refc.py:2891-2892` documents `None` as "
                   "*byte-identical to the pre-hook forward*. ⚠️ It is declared rather "
                   "than filtered out by type, because a channel silently dropped from "
                   "the scope is indistinguishable from one nobody thought about — "
                   "which is the exact defect `FORWARD_KEYS` drift produced twice.",
            unblock="⛔ nothing should. If the hook ever became mandatory it would stop "
                    "being an optional seam, and the honest fix would be a positional "
                    "parameter on the forward rather than a guard in an RL adapter.",
            evidence="PUBLISHED-CODE 2026-09-07: `refc.py:2876` (the parameter), "
                     "`:2891-2897` (the docstring), `:2937-2946` (what it fills)."),
    )


def forward_conditioning_channels(model) -> tuple[str, ...]:
    """The conditioning channels **this model's own forward** accepts.

    ⛔⛔ DERIVED FROM ``inspect.signature``, NEVER FROM A TUPLE IN THIS FILE. A
    hand-maintained channel list is what drifted twice in ``refc_adapter``
    (`FORWARD_KEYS` lost `agent_gt`, then the worktree and HEAD disagreed about it),
    and a list that is wrong for the model in front of it is worse than no list: it
    silently narrows what gets checked while looking like a contract.

        channels = signature(model.forward)
                   - {frames, steps}
                   - tanitad.channel_admissibility.excluded_channels()

    ⛔ And every surviving channel MUST be declared in
    :func:`refc_channel_requirements`. A forward that grows a channel therefore
    breaks LOUDLY at the next rollout construction instead of running blind to it —
    that is the whole point of deriving rather than listing.
    """
    ra = _machinery()
    from tanitad.channel_admissibility import excluded_channels

    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError) as exc:
        raise ra.ConditioningError(
            f"cannot read the forward signature of {type(model).__name__} "
            f"({type(exc).__name__}: {exc}). ⛔ REFUSING rather than falling back to a "
            f"default channel list: a fallback list is a contract about a model we "
            f"could not inspect, which is how a contract ends up asserted against the "
            f"wrong object.") from exc

    excluded = frozenset(excluded_channels())
    channels = tuple(
        name for name, p in sig.parameters.items()
        if name not in NON_CHANNEL_PARAMS
        and name not in excluded
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))

    declared = {r.channel for r in refc_channel_requirements()}
    undeclared = [c for c in channels if c not in declared]
    if undeclared:
        raise ra.RequirementDeclarationError(
            f"{type(model).__name__}.forward accepts {undeclared} but "
            f"refc_channel_requirements() does not declare them. ⛔ REFUSING rather "
            f"than ignoring them: an undeclared channel is one this contract silently "
            f"does not check, and 'not checked' and 'checked and fine' are "
            f"indistinguishable from the call site. Declare each one — with what "
            f"silently changes when it is absent, or why nothing here can assert it — "
            f"in tanitad.rl.refcv3_adapter.refc_channel_requirements. If it is a LABEL "
            f"rather than an observation, declare it in its own seam and add that seam "
            f"to tanitad.channel_admissibility.SEAM_MODULES instead.")
    return channels


def _resolve_any(cfg, paths: tuple[str, ...]) -> bool:
    """Read a set of ALTERNATIVE dotted config paths. RAISES when NONE resolve.

    ⛔⛔ WHY ALTERNATIVES, AND WHY THE RAISE SURVIVES THEM. This adapter is handed
    two config SHAPES: ``RefCConfig`` (flat) from the pilot, and ``RefCV3Config``
    (whose ``.core`` IS a ``RefCConfig``, `refc_v3.py:347`) from the integration
    test. The same field is therefore ``anchors.v0_conditioned`` on one and
    ``core.anchors.v0_conditioned`` on the other. A single path would be wrong for
    one of the two callers no matter which was written.

    ⛔ But the property that matters is kept intact: if NOT ONE path resolves the
    channel raises, because a predicate that cannot be read is a check that can
    NEVER FIRE — it is present in the source, passes every batch, and looks exactly
    like a guard. That false green is what happened to `v0`, `lan` and `nav_known`
    behind a flat ``getattr`` on a nested field, and it is the reason this returns a
    refusal rather than ``False``.

    ⚠️ STATED LIMIT: a typo in ONE of several alternatives is tolerated as long as
    another resolves, so the raise cannot catch it.
    ``test_rl_refcv3_used_path_guard.py`` closes that hole from the other side — it
    asserts, for BOTH real config classes, exactly WHICH paths resolve.
    """
    ra = _machinery()
    resolved: list[tuple[str, bool]] = []
    for dotted in paths:
        obj, ok = cfg, True
        for part in dotted.split("."):
            nxt = getattr(obj, part, _MISSING)
            if nxt is _MISSING:
                ok = False
                break
            obj = nxt
        if ok:
            resolved.append((dotted, bool(obj)))
    if not resolved:
        raise ra.ConditioningError(
            f"NONE of the conditioning predicates {list(paths)} resolve on this "
            f"build's config ({type(cfg).__name__}). ⛔ REFUSING rather than reading "
            f"False. A predicate that cannot be read is a check that CAN NEVER FIRE: "
            f"it looks like a guard and passes every batch. Either the config grew a "
            f"new shape and this declaration needs its path, or the requirement no "
            f"longer applies and should be dropped — but it must not read False.")
    return any(value for _, value in resolved)


def conditioning_requirements(model) -> dict[str, bool]:
    """Which of THIS model's forward channels its own config says it was trained with.

    ⛔ Read from ``model.cfg`` — never from a run name, a filename or a caller
    argument. A name is not provenance; a field that DECLARES the fact is the only
    admissible source.
    """
    ra = _machinery()
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ra.ConditioningError(
            f"{type(model).__name__} has no .cfg, so there is nothing that declares "
            f"what it was trained with. ⛔ REFUSING rather than assuming: guessing is "
            f"exactly how a differently conditioned policy gets post-trained silently, "
            f"and the result reads as a lever effect.")
    channels = forward_conditioning_channels(model)
    by_channel = {r.channel: r for r in refc_channel_requirements()}
    return {c: (_resolve_any(cfg, by_channel[c].predicates)
                if by_channel[c].predicates else False)
            for c in channels}


def _missing_channels(required, batch) -> list[str]:
    """The REQUIRED channels this batch drops.

    ⚠️ ``is None``, deliberately — NOT falsiness. A measured ``v0`` of exactly
    0.0 m/s is a legitimate stationary car and must not read as a dropped channel;
    conflating the two is the X15 zero-collision in miniature.
    """
    return [k for k in required if batch.get(k) is None]


def _refusal(missing, batch, *, batch_index=None) -> str:
    """The refusal text. ONE source, so the per-batch path and the single-shot
    :func:`assert_conditioning` cannot drift into saying different things."""
    by_channel = {r.channel: r for r in refc_channel_requirements()}
    why = "\n".join(f"  - {k}: {by_channel[k].reason}"
                    for k in missing if k in by_channel)
    where = ("" if batch_index is None else
             f" ⛔ REFUSED ON BATCH #{batch_index} of this rollout"
             + (" — a LATER batch, which a once-only check would have MISSED."
                if batch_index > 1 else "."))
    return (
        f"this checkpoint was trained WITH {missing} (its own config says so) but the "
        f"batch supplies None.{where} Sampling would return a well-formed fan from a "
        f"DIFFERENTLY CONDITIONED policy and nothing would raise. Supply the channel, "
        f"or retrain the intent. Batch keys present: "
        f"{sorted(k for k in batch if batch.get(k) is not None)}\n"
        f"What each missing channel silently changes:\n{why}")


class RefCConditioningContract:
    """The requirement map resolved ONCE; the contract asserted on EVERY batch.

    ⛔⛔ PER-BATCH, NOT ONCE-ONLY, AND THE COST ARGUMENT IS SETTLED. A sibling
    MEASURED the equivalent check on the ``refc_adapter`` path at **0.181-0.184 µs**
    against a 3.2 ms smoke forward — **1/17,495 of the SMALLEST forward in the
    repo**, and 1/2,365,739 of a deployed-scale one. There is no performance
    argument for a latch, and a latch is not a cheaper guard: it is a guard that
    stops guarding after batch 1. What a missed later batch costs is not "a softer
    policy" — `refc.py:3097` → `:2128` → `:1722` replaces the action space.

    ⭐ WHY A CACHE RATHER THAN A MOVED CHECK. Resolving the map walks dotted config
    paths; the assertion itself is a handful of ``dict.get`` calls. So the map is
    resolved once and the BATCH is checked every time. The cache is kept because it
    is strictly cheaper and because resolving once makes a DECLARATION bug raise at
    one deterministic point — not because it is what makes the guard affordable.

    ⚠️ SCOPE, STATED RATHER THAN IMPLIED. The map is keyed on the IDENTITY of
    ``model.cfg``: a SWAPPED config re-resolves, an IN-PLACE MUTATION of the same
    object does not. That is deliberate — a config mutated mid-rollout changes the
    policy under the optimiser and is a different defect class from a batch that
    forgot a key, which is what this contract is for.
    """

    __slots__ = ("_model", "_cfg", "_req", "_required", "_plumbed",
                 "batches", "resolutions")

    def __init__(self, model, plumbed: tuple[str, ...] = PLUMBED_CHANNELS):
        self._model = model
        self._plumbed = frozenset(plumbed)
        self._cfg = _MISSING          # never a real cfg, so the first check resolves
        self._req: dict[str, bool] = {}
        self._required: tuple[str, ...] = ()
        #: batches this contract has CHECKED (not merely seen)
        self.batches = 0
        #: times the requirement map was resolved. The point of the cache is that
        #: this stays 1 while ``batches`` grows — pinned by a test, because a cache
        #: nobody measures is indistinguishable from no cache at all.
        self.resolutions = 0

    @property
    def requirements(self) -> dict[str, bool]:
        """The resolved map, or ``{}`` before the first batch."""
        return dict(self._req)

    @property
    def required(self) -> tuple[str, ...]:
        """The channels this build's own config says it was TRAINED with."""
        return self._required

    def check(self, batch) -> dict[str, bool]:
        """Assert the contract for ONE batch. Called on EVERY batch of a rollout."""
        ra = _machinery()
        cfg = getattr(self._model, "cfg", None)
        if cfg is not self._cfg:
            # raises on a missing .cfg, an undeclared channel or an unresolvable
            # predicate — and does NOT record the cfg, so a declaration bug stays loud
            self._req = conditioning_requirements(self._model)
            required = tuple(k for k, needed in self._req.items() if needed)
            unplumbed = [k for k in required if k not in self._plumbed]
            if unplumbed:
                raise ra.ConditioningError(
                    f"this build REQUIRES {unplumbed} (its own config says so) but "
                    f"`make_refcv3_sample_fn` never passes {'it' if len(unplumbed) == 1 else 'them'} "
                    f"to the forward — it plumbs only {sorted(self._plumbed)}. ⛔ "
                    f"REFUSING at the first batch. This is the failure a batch-only "
                    f"check cannot see: the channel would be REQUIRED, PRESENT in the "
                    f"batch, and STILL arrive as None, so every guard would read green "
                    f"while the policy was conditioned differently from the one this "
                    f"run means to ship. Plumb the channel here (and add it to "
                    f"PLUMBED_CHANNELS), or use `tanitad.rl.refc_adapter."
                    f"make_refc_sample_fn`, which plumbs the full RefCV3Model set — "
                    f"⚠️ but ONLY on a RefCV3Model: its predicates are rooted at "
                    f"`core.` and it passes `ego_state`, so on a plain `refc.RefCModel` "
                    f"it raises on the first predicate and would TypeError on the "
                    f"forward.")
            self._cfg = cfg
            self._required = required
            self.resolutions += 1
        self.batches += 1
        missing = _missing_channels(self._required, batch)
        if missing:
            raise ra.ConditioningError(
                _refusal(missing, batch, batch_index=self.batches))
        return self._req


def assert_conditioning(model, batch) -> dict[str, bool]:
    """REFUSE a batch that drops a channel THIS model was trained with.

    The single-shot form: it resolves the map on every call. Inside a rollout use
    :class:`RefCConditioningContract`, which resolves once and asserts every batch.

    Returns the requirement map so a caller can bank it in the run record — a
    conditioning contract that is checked but not written down is not evidence.
    """
    ra = _machinery()
    req = conditioning_requirements(model)
    missing = _missing_channels(
        tuple(k for k, needed in req.items() if needed), batch)
    if missing:
        raise ra.ConditioningError(_refusal(missing, batch))
    return req


def requirement_report(model=None) -> list[dict]:
    """The declarations, for a run record or a preflight artifact.

    With a ``model``, only the channels ITS forward actually accepts — which is the
    honest scope, and makes the record say which family the run was on.
    """
    decls = refc_channel_requirements()
    if model is not None:
        in_scope = set(forward_conditioning_channels(model))
        decls = tuple(r for r in decls if r.channel in in_scope)
    return [r.to_dict() for r in decls]
