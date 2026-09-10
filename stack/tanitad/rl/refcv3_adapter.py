"""Connect the RL library to a real ``RefCV3Model`` — importing refc surfaces.

This is the only module that knows what refcv3 IS. Everything else in
``tanitad.rl`` is model-agnostic and talks through the ``sample_fn`` seam, which
is what lets the tests run a synthetic policy on CPU. ⛔ Nothing is copied from
``refc``/``refc_v3``: the model is imported and called.

WHAT IT SAMPLES, AND THE ONE THING IT IS NOT
---------------------------------------------
refcv3's decoder emits, per anchor, a deterministic ``offset``; the fan is
``anchor_traj = anchors + offset`` (``refc.py:1365``, ``:1515``). This adapter
treats the offset as the mean of an explicit **Gaussian exploration policy** and
draws ``G`` samples per anchor::

    offset_g = offset * (1 + sigma * eps),   eps ~ N(0, I)        [multiplicative]
    offset_g = offset + sigma * eps                                [additive]

The multiplicative form is the default because DDv2's ablation measured it above
additive (90.1 vs 89.7 PDMS) — it perturbs proportionally instead of swamping
small offsets. Because this is a reparameterisation with a known density, the
log-probability is analytic and DIFFERENTIABLE w.r.t. the decoder's output, so
the score-function gradient reaches the generator's parameters.

⛔ **STATED LIMIT, NOT HIDDEN.** This is a surrogate Gaussian policy over the
EMITTED offset. It is **not** the true reverse-diffusion density of the
truncated-diffusion decoder. DDv2's exact formulation attaches the policy
gradient to the denoising step distribution itself. The surrogate is:
  * correct as a policy gradient for the sampler *as defined here*, and
  * the cheapest way to get a working, testable loop against the real model,
but any claim of parity with DDv2's estimator is UNVERIFIED and must say so.
Replacing this with the decoder's own step density is a named backlog item.

⚠️ ``offset`` can be exactly 0 for an anchor the decoder does not move. Under
the multiplicative form the sample density then collapses (zero scale), so the
scale is floored at ``min_scale`` — otherwise ``logp`` is -inf and the loss is
NaN, which surfaces as a dead run rather than as the degenerate anchor it is.

⛔⛔ WHICH MODEL THIS BINDS — READ THIS BEFORE THE SENTENCE ABOVE.
This module is DUCK-TYPED: it imports no model at all, and its two callers hand
it DIFFERENT CLASSES.

    stack/scripts/rl_pilot_refc21.py:169   refc.RefCModel        (the only
                                           production RL caller)
    stack/tests/test_rl_refcv3_integration.py:31   refc_v3.RefCV3Model

⚠️ The first line of this docstring said "a real ``RefCV3Model``" and was wrong
for the caller that matters. ``refc.RefCModel`` (`refc.py:2515`, forward
`:2870-2881`) and ``refc_v3.RefCV3Model`` (`refc_v3.py:763`, forward
`:1361-1372`) are separate classes in separate modules with no inheritance
between them; `refc.py` does not import, subclass or alias the other. Their
signatures differ by seven channels and their configs nest differently
(``RefCV3Config.core`` IS a ``RefCConfig``, `refc_v3.py:347`).

⇒ **the conditioning contract below is resolved from the model it is HANDED**,
never from a tuple written here. See :func:`forward_conditioning_channels`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import PostTrainConfig

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def sample_offsets(offset: Tensor, cfg: PostTrainConfig, *,
                   generator: torch.Generator | None = None,
                   min_scale: float = 1e-3) -> tuple[Tensor, Tensor]:
    """``offset [B, N, S, 2]`` -> (``offset_g [B, N, G, S, 2]``, ``logp [B, N, G]``).

    ``logp`` is the summed Gaussian log-density of the drawn sample over the
    (S, 2) axes, differentiable through ``offset``.
    """
    g = int(cfg.group_size)
    if g < 2:
        raise ValueError(f"group_size must be >= 2 to have any advantage, got {g}")
    b, n, s, two = offset.shape
    mean = offset.unsqueeze(2).expand(b, n, g, s, two)

    if cfg.noise_mode in ("multiplicative", "two_scalar"):
        scale = (mean.abs() * cfg.noise_scale).clamp_min(min_scale)
    else:
        scale = torch.full_like(mean, max(cfg.noise_scale, min_scale))

    if cfg.noise_mode == "two_scalar":
        # ⭐ THE V2-FAITHFUL SCALE POLICY (added 2026-09-05, REF-C RL re-scope).
        # DDv2's released sampler draws TWO scalars per trajectory — one
        # longitudinal, one lateral — and scales every waypoint by them
        # (`diffusiondrivev2_model_rl.py:646-654`: `randn([B, N, 1, 1])` per
        # axis, broadcast over the 8 waypoints; the additive DDPM term is
        # multiplied by ZERO at :640/:666). The explored family per anchor is
        # therefore a 2-parameter (stretch-along, stretch-lateral) family, which
        # keeps every sample smooth and is exactly the along-track axis on which
        # our fan's deficit sits (D-REFCV3-AXIS1: 92.2 % of the os-ha gap).
        # ⚠️ STATED LIMIT, as in V2 itself (DDv2 analysis §1.3 #3): the
        # likelihood below is the per-coordinate Gaussian summed over (S, 2),
        # while the sampler has rank 2 — the 2S terms are perfectly correlated.
        # That is the SAME class of estimator the published method trains with
        # (a directional finite difference along the mean), and it is pinned
        # here on purpose so the port is the released mechanism and not a
        # re-derivation of it. `test_rl_v2_faithful.py` pins the constant
        # per-axis ratio and the directional convergence.
        eps = torch.randn((b, n, g, 1, two), generator=generator,
                          device=mean.device, dtype=mean.dtype).expand(b, n, g, s, two)
    else:
        eps = torch.randn(mean.shape, generator=generator, device=mean.device,
                          dtype=mean.dtype)
    sample = mean + scale * eps

    # ⛔ THE SCORE-FUNCTION GRADIENT LIVES IN (sample.detach() - mean) — NOT eps.
    #
    # CORRECTED 2026-08-29 after pilot P-RC21's first P1 arm COLLAPSED THE
    # PLANNER (fan collision 11.1 % -> 0.0 % because R3 exploded 1.97 m ->
    # 347.2 m: every candidate left the road). Root cause, MEASURED by A/B
    # (`code/estimator_ab.py`): the original wrote
    #     logp = -0.5*eps**2 - log(scale)
    # with eps the RAW DRAW — a symbolic substitution that makes the
    # (sample - mean) dependence CANCEL. d(logp)/d(mean) then flows ONLY through
    # log(scale)=log|mean·sigma|, so the update can shrink or inflate |offset|
    # by advantage sign but can NEVER move the mean toward good samples. In the
    # toy A/B the degenerate form leaves the mean at its start (+0.500 after
    # 400 steps, target +3.0) while this form reaches +2.750.
    #
    # The correct REINFORCE logp treats the drawn sample as a CONSTANT and the
    # density parameters as live: eps_eff = (sample.detach() - mean)/scale.
    # ⚠️ The old version PASSED test_sample_offsets_shapes_and_differentiability
    # — a nonzero gradient is not a correct gradient. The pinning test is now
    # DIRECTIONAL (test_score_function_gradient_points_toward_good_samples).
    eps_eff = (sample.detach() - mean) / scale
    logp = (-0.5 * eps_eff.pow(2) - torch.log(scale) - LOG_SQRT_2PI)
    return sample, logp.sum(dim=(-1, -2))


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


def _forward_kwargs(batch, cfg_in: PostTrainConfig) -> dict:
    """The kwargs for ONE forward, built by iterating :data:`PLUMBED_CHANNELS`.

    ⭐ Derived, so the plumbed set the contract ENFORCES and the plumbing the
    forward RECEIVES are one object and cannot drift. (They were previously two:
    a positional call here and nothing enforcing it anywhere.)
    """
    kw = {k: batch.get(k) for k in PLUMBED_CHANNELS}
    kw["steps"] = int(getattr(cfg_in, "decoder_steps", 0))
    return kw


def make_refcv3_sample_fn(model, cfg: PostTrainConfig, *,
                          build_ctx=None, reference=None,
                          gt_bar_fn=None,
                          generator: torch.Generator | None = None,
                          strict_conditioning: bool = True):
    """Return a ``sample_fn(batch, cfg) -> (traj, logp, ctx)`` for the handed model.

    ``batch`` must be a mapping carrying at least ``frames``; the channels in
    :data:`PLUMBED_CHANNELS` (``nav_cmd`` / ``v0`` / ``lan``) are forwarded.
    ``build_ctx(batch, out)`` supplies the REWARD CONTEXT — scene facts only.

    ⛔ The context is never taken from the model's own outputs. The reward may
    not read a model-produced ranking, and the cheapest way to guarantee that is
    to never hand it one: ``build_ctx`` receives the batch, and anything it adds
    is audited by ``assert_selector_disjoint`` on every step.

    ⛔⛔ THE CONDITIONING CONTRACT IS ASSERTED ON **EVERY** BATCH (2026-09-07).
    Before this, the only production RL script in the repo reached a real planner
    through this function with **no conditioning check of any kind** — not this
    adapter's (there was none), not ``channel_guard``'s launch preflight, and not
    ``rl_control_space_preflight`` (MEASURED: 0 occurrences of every guard token
    in ``rl_pilot_refc21.py``). A batch that dropped ``v0`` would have post-trained
    a policy on a DIFFERENT ACTION SPACE — every anchor rolled at the 10 m/s
    reference (`refc.py:3097` → `:2128` → `:1722`) — and returned a well-formed
    fan with nothing raising.

    ⛔ Per-batch, not once-only. The cost is settled: a sibling MEASURED the
    equivalent check at **0.181-0.184 µs**, or **1/17,495 of the smallest forward
    in the repo**. A latch is not a cheaper guard; it is a guard that stops
    guarding after batch 1, and the defect it would miss is the same size.

    ``strict_conditioning=False`` exists only for a deliberate ablation arm that
    means to sample the un-conditioned policy. It must be typed, and it is
    RECORDED on the returned callable (``.strict_conditioning`` /
    ``.conditioning_contract``) so a run record can bank the arm rather than take
    its word for it.

    ⭐⭐ ``gt_bar_fn`` — THE CALLER'S ROUTE TO V2's >=GT TRUNCATION (2026-09-11).
    ``gt_bar_fn(batch, ctx, n_steps) -> [B]`` is scored by the caller under the
    SAME ``RewardSpec`` the candidates are scored under (`posttrain.score_gt_bar`)
    and is handed to `posttrain.rl_objective` as ``extras["gt_bar"]``.

    ⛔ WHY THIS PARAMETER HAD TO EXIST AT ALL — AND IT IS NOT A STYLE POINT.
    `advantage.truncated_inter_anchor_advantage` has carried the >=GT mask since
    2026-09-05; `posttrain.rl_objective` has accepted ``gt_bar`` and
    `PostTrainConfig` has carried ``use_gt_bar``. MEASURED 2026-09-10 on the
    2,000-step pilot: ``use_gt_bar = False``, ``frac_above_bar_mean = None``, and
    the only production RL script in the tree exposed **no flag that could set
    them** — thirteen flags, none reaching the truncation. ⇒ what ran was a
    V1-style baseline and the V2 stage had never been tested. ⭐ Same family as
    `tac_goal_tok_head`: 11,286 parameters, built, tested, rollable, and
    ``grad_abs_sum`` exactly 0 for all 40,284 steps because nothing called them.
    *Rollable and trained are different claims.* This function is where the
    calling ends.

    ⛔ ``None`` (the default) returns the pre-2026-09-11 three-tuple unchanged, so
    every banked run remains reproducible bit-for-bit.
    """
    # ⭐ ONE map for the whole rollout, asserted on every batch. `None` when the
    # ablation escape hatch is typed: nothing is resolved and nothing can raise,
    # which is exactly what an un-conditioned arm asked for.
    contract = RefCConditioningContract(model) if strict_conditioning else None

    def sample_fn(batch, cfg_in: PostTrainConfig):
        if contract is not None:
            contract.check(batch)
        frames = batch["frames"]
        kw = _forward_kwargs(batch, cfg_in)
        out = model(frames, **kw)
        anchor_traj = out["anchor_traj"]                   # [B, N, S, 2]
        offset = out["offset"]                             # [B, N, S, 2]
        base = anchor_traj - offset                        # the anchors alone

        off_g, logp = sample_offsets(offset, cfg_in, generator=generator)
        traj = base.unsqueeze(2) + off_g                   # [B, N, G, S, 2]

        ctx = dict(build_ctx(batch, out)) if build_ctx else {}
        ctx.setdefault("dt", cfg_in.dt)

        extras: dict = {}
        if gt_bar_fn is not None:
            # ⛔ THE HORIZON IS HANDED IN, NOT ASSUMED. `score_gt_bar` refuses a
            # GT whose S differs from the fan's: `_progress` normalises by
            # (S-1)*dt, so a bar scored over a different horizon is a threshold
            # on a DIFFERENT reward function and would look entirely plausible.
            bar = gt_bar_fn(batch, ctx, int(traj.shape[-2]))
            if bar.shape != traj.shape[:1]:
                raise ValueError(
                    f"gt_bar_fn returned {tuple(bar.shape)}, expected per-window "
                    f"{tuple(traj.shape[:1])}. A bar that is not per-window is "
                    "not V2's mask.")
            extras["gt_bar"] = bar
        if reference is not None:
            # The trust-region pair: the LIVE deterministic fan against the FROZEN
            # reference's fan on the SAME inputs. Both are means (pre-exploration),
            # because the anchor constrains the policy, not the noise.
            with torch.no_grad():
                ref_out = reference(frames, **kw)
            extras["anchor_pair"] = (anchor_traj, ref_out["anchor_traj"])
        if not extras:
            # ⛔ BIT-IDENTICAL DEFAULT PATH. With neither a reference nor a bar
            # this returns exactly the three-tuple it always did, so the banked
            # baseline arms reproduce unchanged.
            return traj, logp, ctx
        return traj, logp, ctx, extras

    # ⭐ "it must be typed, and it is RECORDED" — true of the OBJECT, not just of
    # the docstring. A preflight or a run record reads these rather than trusting
    # an argv string that says what the operator meant to type.
    sample_fn.strict_conditioning = bool(strict_conditioning)
    sample_fn.conditioning_contract = contract
    # ⭐ Same doctrine for the bar: a preflight or a run record reads the OBJECT
    # rather than trusting an argv string that says what the operator meant.
    sample_fn.emits_gt_bar = gt_bar_fn is not None
    return sample_fn


def gt_context(batch, out=None) -> dict:
    """A minimal reward context from the BATCH only — the safe default.

    Passes through the scene facts the reward components understand and
    NOTHING the model produced. ``out`` is accepted and ignored on purpose, so
    that a caller who reaches for it has to write that code deliberately.
    """
    ctx: dict = {}
    for key in ("gt_traj", "obstacles", "lead_path", "lead_len_m",
                "target_time_gap_s", "a_max", "kappa_max", "progress_ref_m",
                "v0", "dt"):
        if key in batch:
            ctx[key] = batch[key]
    return ctx
