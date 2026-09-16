"""refcv6 TACTICAL layer — a DETR-style behaviour decoder over the SCENE EMBEDDING.

⭐ THE PI'S INSTRUCTION, 2026-09-16, verbatim: *"All other tactical vocabs must
be learned. The tactical layer must learn to emitt the valid tactical behaviors,
choose the best architecture for it. It should learn them from the scene
embeddings, for the agent and the map."* The architecture below is the Master
Mind's choice under that instruction; every non-obvious decision states its
measured or cited reason.

WHAT IT IS
============================================================================
A 2-layer, d = 256 DETR decoder.

  * **Queries** — 22 behaviour queries (one per tactical goal token) + 8 lateral
    + 8 longitudinal action queries = **38**. One query per OUTPUT, so a class's
    representation is a vector this layer owns rather than a row of a shared
    projection. That is what lets a rare class (``LANE_CHANGE_R``, 15 positives)
    have somewhere to live at all.
  * **Keys / values** — the **agent slots** and the **BEV tokens**. ⛔ **NOT the
    image tokens.** Behaviours are about the scene, not the pixels, and the
    exclusion is STRUCTURAL: :meth:`TacticalBehaviourDecoder.forward` has no
    image-token port to pass one to, and :func:`assert_scene_only` re-checks the
    declaration. The programme's own precedent is
    ``AgentSlotDecoder.forward`` — *"accepts exactly one argument, which is the
    audit in signature form"*.
  * **Conditioning** — FiLM on the queries at EVERY layer, from
    ``[nav one-hot (4), max-speed one-hot (4), v0, a0]`` = 10 channels
    (:func:`build_condition`).
  * **Outputs** — per-behaviour **validity** (multi-label BCE) + per-behaviour
    **confidence**, and lat/lon action posteriors over the v7 8x8 vocabulary.

⛔⛔ WHY VALIDITY IS BCE AND NOT A SOFTMAX — this is the whole point of the
layer. MEASURED on the v7.2/v8 train blob: **2-7 goal tokens per record, mean
2.751** (``tac_goal_head.py`` docstring). Several behaviours are valid at once.
Collapsing a SET into a CHOICE is the 5-way manoeuvre-softmax defect — the
programme's single largest known defect, the one mechanism behind 0/881
accelerate and the speed-fan (``refc_tactical.py`` module docstring, F2) —
rebuilt one layer up. So this is 22 independent sigmoids.

⛔⛔ THE MEASURED LIMITS THIS HEAD MUST BE REPORTED UNDER. Re-measured for
refcv6 through ``v7_labels.goal_supervision_census`` on the locally available
v8 train blob (``s2_labels_v8.0_train.jsonl.gz``, md5
``fa89ea55dfce68403eb30300e57852ab``, n = 4,572), i.e. through the SAME
negative policy the trainer uses (``negatives="measured"``), not through naive
presence/absence:

  * **17 of 22 tokens are trainable.** The 5 masked ones carry positives and
    ZERO supervised negatives — ``YIELD`` (609), ``SPEED_BAND`` (4,572, a
    constant), ``CORRIDOR_OFFSET`` (860), ``GAP_TARGET`` (368),
    ``REACT_ON_ONCOMING`` (333). An unmasked logit there can only be pushed
    towards 1.
  * **10 of 22 sit under the n = 200 scoreability floor** — and they are
    EXACTLY ``vocab_v7.TACTICAL_GOAL_UNDERPOWERED``, reproduced token for token.
  * ⇒ **only 7 tokens are BOTH trainable and scoreable**: ``FOLLOW_LANE``
    (3,629), ``TURN_L`` (275), ``TURN_R`` (259), ``STOP_POINT`` (327),
    ``EVADE_IN_CORRIDOR`` (240), ``TRAFFIC_LIGHT_REACT_RED`` (376),
    ``TRAFFIC_LIGHT_REACT_GREEN`` (363).
  * lat/lon labels exist only within **±2 s of each clip's anchor**
    (``v7_labels.window_in_band``, derived from the record's own bands). Three
    lateral classes (``LANE_CHANGE_L``, ``LANE_CHANGE_R``, ``ABORT_LC``) and one
    longitudinal class (``YIELD_MERGE``) have **ZERO** positives in the train
    blob and cannot be learned at all.

⇒ :func:`per_class_report` is the ONLY admissible readout. A pooled accuracy
over 22 tokens is dominated by ``FOLLOW_LANE`` at 79.4 % prevalence and says
nothing.

⛔ ADMISSIBILITY (PI 2026-08-03, still binding). No situation classifier may
feed a goal input. ``tac_SIT``, the traffic-light tokens and ``YIELD`` may be
auxiliary TARGETS only, never INPUTS to the goal or to selection. This module
declares that in :data:`SITUATION_OUTPUT_TOKENS` and enforces it in
:func:`assert_situation_tokens_are_targets_only`, which is called from
``RefCV3Model.__init__`` and proven by mutation (a build that declares one of
them as an input REFUSES).

⭐⭐ NAV -> TURN IS BY DESIGN, NOT A DEFECT (PI 2026-09-16, verbatim: *"its
totally fine to process the nav command and generate from it the turing command,
we are not to aim in this stage to generate a route"*). refcv5-v2's tactical turn
head is a near-pure function of nav (MEASURED: TURN_L recall 0.475 true / 0.050
shuffled / **0.000 nav-zeroed**, ``REFCV6_CLARIFICATION.md`` §3.2) and under this
ruling that reading is **admissible**, not a failure. ⇒ Nothing here penalises or
gates the turn classes on nav-independence, and :mod:`taniteval
.refcv6_acceptance`'s T-ZERO reports them but EXCLUDES them from its summary. The
place "learned from the scene" has to show is the NON-turn behaviours — the
longitudinal actions (nav explains 5.2 %), the speed bucket (5.5 %), yielding,
gap targets and lane keeping.

⛔ EVERY FEED TO THE PLANNER IS STILL DETACHED, and the reason survives the
ruling. Detaching is not about the turn classes: it is about keeping the PLANNING
loss out of this layer's parameters. With the feed attached, the cheapest way to
lower the trajectory loss is to reshape the behaviour head into whatever
correlates with the trajectory — and then its per-class numbers stop being a
measurement of what the LABELS taught it. Detached, every bit this layer learns
comes from its own losses, on the labels, from the scene.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.models.vocab_v7 import (TACTICAL_GOAL_TOKENS_V7,
                                     TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7,
                                     GOAL_MIN_N_FOR_METRIC)
from tanitad.refs.refcv6_max_speed import N_SPEED_MAX_BINS_V6

__all__ = [
    "N_GOAL_TOKENS", "N_LAT_ACTIONS", "N_LON_ACTIONS", "N_QUERIES",
    "N_NAV_COMMANDS", "COND_DIMS", "SCENE_SOURCES", "SITUATION_OUTPUT_TOKENS",
    "TacticalDecoderConfig", "TacticalBehaviourDecoder",
    "build_condition", "bev_feats_to_tokens", "planner_feeds",
    "per_class_report",
    "tactical_behaviour_losses", "TacticalLossWeights",
    "assert_scene_only", "assert_situation_tokens_are_targets_only",
    "SituationInputRefused", "SceneInputRefused",
]

N_GOAL_TOKENS: int = len(TACTICAL_GOAL_TOKENS_V7)          # 22
N_LAT_ACTIONS: int = len(TACTICAL_LAT_ACTIONS_V7)          # 8
N_LON_ACTIONS: int = len(TACTICAL_LON_ACTIONS_V7)          # 8
N_QUERIES: int = N_GOAL_TOKENS + N_LAT_ACTIONS + N_LON_ACTIONS   # 38

#: ``NAV_COMMANDS`` is 4-wide (follow / left / right / straight). Imported
#: rather than re-listed would pull ``refc.py`` — a heavy module — into every
#: importer, so the WIDTH is mirrored and pinned equal to the original by
#: ``tests/test_refcv6_tactical.py`` (the ``refc_tactical`` mirroring contract).
N_NAV_COMMANDS: int = 4

#: ``[nav(4), max_speed(4), v0, a0]``. ⛔ The ORDER is the contract; a
#: checkpoint's FiLM weights are indexed by it.
COND_DIMS: int = N_NAV_COMMANDS + N_SPEED_MAX_BINS_V6 + 2   # 10

#: The ONLY key/value sources this decoder has. ⛔ ``image`` is deliberately
#: absent and :func:`assert_scene_only` refuses any attempt to add it at a call
#: site; changing this tuple is a design change that needs the PI, not a config.
SCENE_SOURCES: tuple[str, ...] = ("agent", "bev")

#: ⛔⛔ THE PI'S ADMISSIBILITY RULING (2026-08-03), AS DATA. These may be
#: auxiliary TARGETS. They may NEVER be INPUTS to a goal node or to selection.
#: ``tac_SIT`` is the v8 situation-classifier field; the traffic-light tokens
#: and ``YIELD`` are its in-vocabulary siblings — all three are the OUTPUT of a
#: situation judgement, and feeding a judgement back in is the echo the ruling
#: names.
SITUATION_OUTPUT_TOKENS: frozenset[str] = frozenset(
    {"tac_SIT", "YIELD"}
    | {t for t in TACTICAL_GOAL_TOKENS_V7 if t.startswith("TRAFFIC_LIGHT")})

#: ⛔⛔ THE MATCHER IS WORD-BOUNDED, AND THAT IS NOT A DETAIL. The first version
#: of this guard also carried the bare markers ``"sit"`` and ``"situation"`` and
#: matched them as substrings — which made it REFUSE its own correct
#: declaration, because the prose *"the 5 situation columns are structurally
#: dead"* contains the word. A guard that fires on a description of itself is
#: not a stricter guard, it is a broken one: the only way to ship would have
#: been to weaken it. So the identifiers are matched as IDENTIFIERS, and the
#: phrase forms of the classifier are matched explicitly.
#: ⚠️ The real enforcement is STRUCTURAL and lives in
#: ``refcv6_selection.BehaviourSelectionGate`` (five dead columns, proven by
#: mutation). This check is the DECLARATION half; it is deliberately not asked
#: to catch a violation that renamed itself.
_SITUATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(sorted((re.escape(t) for t in SITUATION_OUTPUT_TOKENS),
                      key=len, reverse=True))
    + r")(?![A-Za-z0-9_])"
    r"|situation[ _-]?(?:classifier|output|head)")


class SituationInputRefused(RuntimeError):
    """A situation-classifier output was declared as a GOAL or SELECTION input."""


class SceneInputRefused(RuntimeError):
    """The behaviour decoder was given something other than the scene embedding."""


# ---------------------------------------------------------------------------
# admissibility
# ---------------------------------------------------------------------------

def assert_scene_only(sources: Sequence[str]) -> None:
    """⛔ REFUSE any key/value source that is not the scene embedding.

    The spec's line is *"Keys/values: agent slots and BEV tokens — the scene
    embedding. ⛔ NOT the image tokens: behaviours are about the scene."* The
    signature already makes an image port impossible; this function makes the
    DECLARATION refusable too, so a future caller that adds ``"image"`` to a
    config list fails at build time instead of quietly widening the layer's
    input.
    """
    bad = [s for s in sources if s not in SCENE_SOURCES]
    if bad:
        raise SceneInputRefused(
            f"[refcv6-tac] ⛔ key/value source(s) {bad} are not the scene "
            f"embedding. This decoder attends to {list(SCENE_SOURCES)} and "
            f"nothing else: behaviours are about the scene (agents and the "
            f"map), not the pixels. Image tokens reach the OPERATIVE decoder "
            f"(refc.py), which is where the PI asked for direct trunk access — "
            f"not this layer.")


def assert_situation_tokens_are_targets_only(
        roles: Mapping[str, Any], *, where: str = "provenance_roles") -> None:
    """⛔ REFUSE a role declaration that feeds a situation output into a goal.

    ``roles`` is the dict ``RefCV3Model.provenance_roles()`` returns. This
    checks the three places a situation token could get in:

      1. ``inference_inputs_of_goals`` — the declared inputs of every goal node;
      2. ``selection_inputs`` — the declared inputs of the ranked score
         (refcv6 adds this key; absent on a pre-refcv6 declaration, and its
         absence is NOT treated as a pass — see below);
      3. ``goal`` — a node NAMED for a situation token would be a goal that IS
         the classifier's output.

    ⭐ WHY IT IS NOT JUST A LOOK AT THE STRINGS. A declaration is a claim, and
    the programme's rule is that claims are measured. ``goal_provenance
    .audit_arm`` already MEASURES the edge interventionally on a live forward;
    this guard is the BUILD-TIME half, and it exists because the interventional
    audit only runs where someone runs it, while ``__init__`` runs on every
    build. Both ship. The guard is proven by MUTATION in
    ``tests/test_refcv6_tactical.py``: a declaration that names ``tac_SIT`` as a
    goal input must raise, and one that names it only as an auxiliary target
    must not.

    ⚠️ ``selection_inputs`` MISSING is a refusal under refcv6, not a pass. A
    check that silently accepts an absent key is the "guard proven by
    inspection" failure: it would read green on a model that never declared its
    selection inputs at all.
    """
    def _hits(values) -> list[str]:
        out = []
        for v in (values or ()):
            s = str(v)
            for m in _SITUATION_RE.finditer(s):
                out.append(f"{m.group(0)!r} in {s!r}")
        return out

    if "selection_inputs" not in roles:
        raise SituationInputRefused(
            f"[refcv6-tac] ⛔ {where} declares no `selection_inputs`. refcv6 "
            f"routes the valid-behaviour set and the speed ceiling INTO "
            f"selection, so selection now has declared inputs and the "
            f"admissibility ruling applies to them. An absent key is refused "
            f"rather than assumed empty — a guard that passes on a missing "
            f"declaration is a guard proven by inspection.")

    bad_goal = _hits(roles.get("inference_inputs_of_goals"))
    bad_sel = _hits(roles.get("selection_inputs"))
    bad_node = _hits(roles.get("goal"))
    if bad_goal or bad_sel or bad_node:
        raise SituationInputRefused(
            "[refcv6-tac] ⛔ PI 2026-08-03 (still binding): no situation "
            "classifier may feed a goal input. `tac_SIT`, the traffic-light "
            "tokens and YIELD are auxiliary TARGETS only — never inputs to the "
            "goal or to selection.\n"
            f"  goal inputs      : {bad_goal or '-'}\n"
            f"  selection inputs : {bad_sel or '-'}\n"
            f"  goal nodes       : {bad_node or '-'}\n"
            "If the intent is to SUPERVISE these tokens, declare them under "
            "`situation_output` (targets), which this check permits.")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class TacticalDecoderConfig:
    """⭐ The spec's numbers, as a frozen record that travels with the ckpt."""

    d_model: int = 256              # spec §4 "2 layers, d 256"
    n_layers: int = 2               # spec §4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0            # eval determinism; the seam is small
    #: Feature widths of the two scene sources. ``d_bev = 0`` builds the decoder
    #: WITHOUT a BEV port — the honest state until the perception agent's lift
    #: lands — and the forward then REFUSES a ``bev_tokens`` argument rather
    #: than silently dropping it (the refcv5 WP-6 rule).
    #:
    #: ⛔⛔ THE **COUNT** OF BEV TOKENS IS NEVER CONFIGURED AND NEVER TYPED —
    #: it is read off the tensor at runtime (`forward` concatenates whatever
    #: arrives and derives the pad mask from `t.shape[:2]`). PI 2026-09-16 moved
    #: the input to 256x1024, so the stride-16 map is **16 x 64**, not 16 x 40;
    #: a decoder that had baked in a token count would have silently broken at
    #: integration. Nothing here changes: the queries and the FiLM condition are
    #: shape-independent of the key count.
    #:
    #: ⚠️ THE **WIDTH** IS A PARAMETER SHAPE AND THEREFORE MUST BE DECLARED —
    #: `bev_in` is `Linear(d_bev, d_model)` and a Linear cannot be sized at
    #: runtime without changing the checkpoint under its readers. It depends on
    #: the backbone the PI chose: **resnet101 -> 1024** channels at stride-16,
    #: **resnet34 -> 256** (the comparison run). Set it from the trunk, never
    #: from this default; `forward` REFUSES a mismatch loudly and names both
    #: values, because the alternative is a bare `mat1 and mat2 shapes cannot be
    #: multiplied` a hundred frames into a pod run.
    d_agent: int = 256
    d_bev: int = 128
    sources: tuple[str, ...] = SCENE_SOURCES

    def __post_init__(self) -> None:
        assert_scene_only(self.sources)
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by n_heads "
                f"{self.n_heads}")
        if self.d_agent <= 0 and self.d_bev <= 0:
            raise SceneInputRefused(
                "[refcv6-tac] ⛔ both scene sources are off (d_agent and d_bev "
                "are 0). A behaviour decoder with no scene attends to nothing "
                "and would read as 'behaviours cannot be learned from the "
                "scene' — a refutation manufactured by a wiring gap.")

    def to_dict(self) -> dict:
        return {"d_model": self.d_model, "n_layers": self.n_layers,
                "n_heads": self.n_heads, "ff_mult": self.ff_mult,
                "dropout": self.dropout, "d_agent": self.d_agent,
                "d_bev": self.d_bev, "sources": list(self.sources),
                "n_queries": N_QUERIES, "n_goal_tokens": N_GOAL_TOKENS,
                "n_lat": N_LAT_ACTIONS, "n_lon": N_LON_ACTIONS,
                "cond_dims": COND_DIMS}


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------

class _QueryFiLM(nn.Module):
    """FiLM on the QUERIES from ``[nav, max_speed, v0, a0]``.

    ⛔ ZERO-INIT ON THE DELTA. ``gamma = 1 + dgamma`` with ``dgamma`` and
    ``beta`` from a zero-initialised Linear, so a freshly built decoder is
    EXACTLY the unconditioned decoder at step 0 and every subsequent change is
    attributable to the condition. Same discipline as ``refc.py``'s
    ``ctx_to_cond`` and ``max_speed_input.MaxSpeedConditioner``.
    """

    def __init__(self, d: int, cond_dims: int = COND_DIMS):
        super().__init__()
        self.proj = nn.Linear(cond_dims, 2 * d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, q: Tensor, cond: Tensor) -> Tensor:
        dg, beta = self.proj(cond.to(q.dtype)).chunk(2, dim=-1)   # [B, d] each
        return q * (1.0 + dg).unsqueeze(1) + beta.unsqueeze(1)


class _DecoderLayer(nn.Module):
    """Pre-LN DETR layer: FiLM -> self-attn(queries) -> cross-attn(scene) -> FFN.

    ⭐ SELF-ATTENTION AMONG THE QUERIES IS LOAD-BEARING HERE, and it is why a
    DETR decoder and not 38 independent MLP heads. The outputs are a SET with
    hard structure — ``TURN_L`` excludes ``TURN_R``; ``STOP_POINT`` co-occurs
    with ``BRAKE_TO``; the lateral and longitudinal axes must stay separable
    (``refc_tactical.py``'s F2). Query-to-query attention is where that
    structure can be represented at all. The frozen exclusion table
    (``v7_labels._EXCLUDED_BY``) supervises it only indirectly, through the
    entailed negatives, so the capacity has to exist.
    """

    def __init__(self, cfg: TacticalDecoderConfig):
        super().__init__()
        d = cfg.d_model
        self.film = _QueryFiLM(d)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.n_heads,
                                               dropout=cfg.dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads,
                                                dropout=cfg.dropout,
                                                batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.GELU(),
                                nn.Linear(cfg.ff_mult * d, d))

    def forward(self, q: Tensor, kv: Tensor, kv_pad: Tensor | None,
                cond: Tensor) -> tuple[Tensor, Tensor]:
        q = self.film(q, cond)
        h = self.n1(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(q)
        # ⛔ A FULLY-PADDED ROW IS UN-MASKED AND ZEROED, never passed as an
        # all-True mask: torch's MHA returns NaN for a row with every key
        # masked, and a NaN that reaches the loss looks like divergence rather
        # than like an empty scene. This is the `CrossAttnLayer` convention in
        # `refc.py`, followed rather than re-invented.
        if kv_pad is not None:
            empty = kv_pad.all(dim=-1)                       # [B]
            pad = kv_pad.clone()
            pad[empty] = False
            a, w = self.cross_attn(h, kv, kv, key_padding_mask=pad,
                                   need_weights=True, average_attn_weights=True)
            a = a * (~empty).to(a.dtype).view(-1, 1, 1)
            w = w * (~empty).to(w.dtype).view(-1, 1, 1)
        else:
            a, w = self.cross_attn(h, kv, kv, need_weights=True,
                                   average_attn_weights=True)
        q = q + a
        q = q + self.ff(self.n3(q))
        return q, w


class TacticalBehaviourDecoder(nn.Module):
    """38 queries over ``{agent slots, BEV tokens}`` -> validity + lat/lon.

    ``forward`` returns a dict:

      ``goal_logits``   ``[B, 22]``  validity, for **multi-label BCE**
      ``goal_conf``     ``[B, 22]``  per-behaviour confidence logit
      ``lat_logits``    ``[B, 8]``   lateral action posterior (CE)
      ``lon_logits``    ``[B, 8]``   longitudinal action posterior (CE)
      ``attn``          ``[B, 38, K]`` mean cross-attention of the LAST layer
      ``n_scene``       ``[B]``      attended (non-pad) key count per row
    """

    def __init__(self, cfg: TacticalDecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or TacticalDecoderConfig()
        d = self.cfg.d_model
        assert_scene_only(self.cfg.sources)
        # ⭐ ONE QUERY PER OUTPUT, in a FROZEN order: [22 goals | 8 lat | 8 lon].
        # The slices below are the contract; a checkpoint's rows are indexed by
        # it, so APPEND ONLY (the TACTICAL_LAT_ACTIONS rule).
        self.queries = nn.Embedding(N_QUERIES, d)
        nn.init.normal_(self.queries.weight, std=0.02)
        self.agent_in = (nn.Linear(self.cfg.d_agent, d)
                         if self.cfg.d_agent > 0 else None)
        self.bev_in = (nn.Linear(self.cfg.d_bev, d)
                       if self.cfg.d_bev > 0 else None)
        # A learned per-SOURCE code, so "this key is an agent" and "this key is
        # a map cell" are distinguishable after projection. Two sources sharing
        # one unmarked space is how a decoder learns to ignore the smaller one.
        self.source_code = nn.Embedding(len(SCENE_SOURCES), d)
        nn.init.normal_(self.source_code.weight, std=0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(_DecoderLayer(self.cfg)
                                    for _ in range(self.cfg.n_layers))
        self.out_norm = nn.LayerNorm(d)
        # Per-query scalar readouts. ⛔ Linear(d, 1) applied to EVERY query,
        # not Linear(d, 22) on a pooled vector: the query IS the class, and a
        # shared 22-wide projection would put the rare classes back on rows of
        # one matrix, which is what this architecture exists to avoid.
        self.validity_head = nn.Linear(d, 1)
        self.conf_head = nn.Linear(d, 1)
        self.lat_head = nn.Linear(d, 1)
        self.lon_head = nn.Linear(d, 1)
        self.tokens = tuple(TACTICAL_GOAL_TOKENS_V7)
        self.lat_actions = tuple(TACTICAL_LAT_ACTIONS_V7)
        self.lon_actions = tuple(TACTICAL_LON_ACTIONS_V7)

    # -- introspection ------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def param_breakdown(self) -> dict[str, int]:
        def cnt(m) -> int:
            return 0 if m is None else int(sum(p.numel() for p in m.parameters()))
        out = {
            "queries": cnt(self.queries),
            "agent_in": cnt(self.agent_in),
            "bev_in": cnt(self.bev_in),
            "source_code": cnt(self.source_code),
            "kv_norm": cnt(self.kv_norm),
            "layers": cnt(self.layers),
            "out_norm": cnt(self.out_norm),
            "heads": (cnt(self.validity_head) + cnt(self.conf_head)
                      + cnt(self.lat_head) + cnt(self.lon_head)),
        }
        out["film_within_layers"] = int(sum(
            p.numel() for lyr in self.layers for p in lyr.film.parameters()))
        out["total"] = self.n_params
        return out

    def provenance(self) -> dict[str, Any]:
        """What this layer reads and what it may never read. Goes into the run
        record beside the stamp, so the exclusion is an ARTIFACT, not a
        comment."""
        return {
            "keys_values": list(SCENE_SOURCES),
            "refused_keys_values": ["image tokens (behaviours are about the "
                                    "scene, not the pixels; PI 2026-09-16)"],
            "condition": ["nav one-hot (4)", "max-speed one-hot (4)", "v0", "a0"],
            "outputs_are_targets_only": sorted(SITUATION_OUTPUT_TOKENS),
            "feeds_planner": ["lat/lon posterior -> anchor prior (DETACHED)",
                              "valid behaviour set -> selection (DETACHED)",
                              "speed ceiling -> selection mask (DETACHED)"],
        }

    # -- forward ------------------------------------------------------------
    def forward(self, cond: Tensor, *,
                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None,
                bev_tokens: Tensor | None = None,
                bev_pad: Tensor | None = None) -> dict[str, Tensor]:
        if cond.dim() != 2 or cond.shape[-1] != COND_DIMS:
            raise ValueError(
                f"[refcv6-tac] cond must be [B, {COND_DIMS}] = "
                f"[nav(4), max_speed(4), v0, a0], got {tuple(cond.shape)}. "
                f"Build it with `build_condition`, which refuses a nav or "
                f"max-speed block that is not a one-hot.")
        b = cond.shape[0]
        # ⛔ A SUPPLIED TENSOR THAT THIS BUILD WOULD SILENTLY DROP IS REFUSED,
        # both directions — the refcv5 WP-6 rule. A dropped scene reads as
        # "the scene does not help" in a result table, which is a refutation
        # manufactured by a wiring gap.
        if agent_tokens is not None and self.agent_in is None:
            raise SceneInputRefused(
                "[refcv6-tac] ⛔ agent_tokens supplied but this build has "
                "d_agent = 0, so they would be silently dropped.")
        if bev_tokens is not None and self.bev_in is None:
            raise SceneInputRefused(
                "[refcv6-tac] ⛔ bev_tokens supplied but this build has "
                "d_bev = 0, so they would be silently dropped. Build the "
                "decoder with d_bev > 0 or stop passing them.")
        kv_parts: list[Tensor] = []
        pad_parts: list[Tensor] = []
        if agent_tokens is not None:
            t = self.agent_in(agent_tokens) + self.source_code.weight[0]
            kv_parts.append(t)
            pad_parts.append(agent_pad if agent_pad is not None
                             else torch.zeros(t.shape[:2], dtype=torch.bool,
                                              device=t.device))
        if bev_tokens is not None:
            # ⛔ THE COUNT IS RUNTIME, THE WIDTH IS DECLARED — and the width
            # mismatch is refused HERE, by name, rather than surfacing as a
            # bare `mat1 and mat2 shapes cannot be multiplied` inside a Linear.
            if bev_tokens.dim() != 3:
                raise SceneInputRefused(
                    f"[refcv6-tac] ⛔ bev_tokens must be [B, P, d_bev] — a "
                    f"FLAT token sequence. Got {tuple(bev_tokens.shape)}. A "
                    f"[B, C, X, Y] feature map is flattened by "
                    f"`bev_feats_to_tokens`, which derives P = X*Y from the "
                    f"tensor so no grid size is ever typed.")
            if bev_tokens.shape[-1] != self.cfg.d_bev:
                raise SceneInputRefused(
                    f"[refcv6-tac] ⛔ bev_tokens are {bev_tokens.shape[-1]} "
                    f"wide but this build declared d_bev = {self.cfg.d_bev}. "
                    f"The width is a PARAMETER shape and follows the trunk: "
                    f"resnet101 gives 1024 channels at stride-16, resnet34 "
                    f"gives 256. Set `tac_decoder_cfg.d_bev` from the backbone "
                    f"the arm actually builds. (The token COUNT is free — "
                    f"16x40 and 16x64 both work untouched.)")
            t = self.bev_in(bev_tokens) + self.source_code.weight[1]
            kv_parts.append(t)
            pad_parts.append(bev_pad if bev_pad is not None
                             else torch.zeros(t.shape[:2], dtype=torch.bool,
                                              device=t.device))
        if not kv_parts:
            raise SceneInputRefused(
                "[refcv6-tac] ⛔ no scene reached the behaviour decoder: both "
                "agent_tokens and bev_tokens are None. Behaviours are learned "
                "FROM THE SCENE (PI 2026-09-16); a forward with no scene would "
                "emit the unconditional prior and read as 'the tactical layer "
                "learned nothing', which is a wiring gap, not a result.")
        kv = self.kv_norm(torch.cat(kv_parts, dim=1))            # [B, K, d]
        kv_pad = torch.cat(pad_parts, dim=1)                     # [B, K]
        q = self.queries.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        attn = None
        for lyr in self.layers:
            q, attn = lyr(q, kv, kv_pad, cond)
        q = self.out_norm(q)
        g = q[:, :N_GOAL_TOKENS]
        la = q[:, N_GOAL_TOKENS:N_GOAL_TOKENS + N_LAT_ACTIONS]
        lo = q[:, N_GOAL_TOKENS + N_LAT_ACTIONS:]
        return {
            "goal_logits": self.validity_head(g).squeeze(-1),     # [B, 22]
            "goal_conf": self.conf_head(g).squeeze(-1),           # [B, 22]
            "lat_logits": self.lat_head(la).squeeze(-1),          # [B, 8]
            "lon_logits": self.lon_head(lo).squeeze(-1),          # [B, 8]
            "attn": attn,
            "n_scene": (~kv_pad).sum(dim=-1),
        }


# ---------------------------------------------------------------------------
# the condition
# ---------------------------------------------------------------------------

def bev_feats_to_tokens(feats: Tensor) -> Tensor:
    """``[B, C, X, Y] -> [B, X*Y, C]``. The ONLY place a BEV grid is flattened.

    ⛔⛔ THE GRID SIZE IS DERIVED FROM THE TENSOR, NEVER TYPED. PI 2026-09-16
    moved the input to 256x1024, so the stride-16 map went from **16 x 40** to
    **16 x 64** and the lifted BEV grid moves with it. A call site that had
    written the token count down would have kept running and silently attended
    to the wrong number of cells — the derived-constant trap (``HORIZON =
    round(6.0 * 10 / STRIDE)``) in a new place.

    ⚠️ The CHANNEL width is the backbone's and is a parameter shape, so it is
    declared in ``TacticalDecoderConfig.d_bev`` (resnet101 -> 1024 at stride-16,
    resnet34 -> 256) and checked on every forward. Count free, width declared.
    """
    if feats.dim() != 4:
        raise SceneInputRefused(
            f"[refcv6-tac] bev feats must be [B, C, X, Y], got "
            f"{tuple(feats.shape)}")
    b, c = feats.shape[0], feats.shape[1]
    return feats.reshape(b, c, -1).transpose(1, 2).contiguous()


def build_condition(nav_onehot: Tensor, vmax_onehot: Tensor,
                    v0: Tensor, a0: Tensor,
                    *, v_scale: float = 10.0, a_scale: float = 3.0) -> Tensor:
    """``[B, 10]`` = ``[nav(4), max_speed(4), v0/10, a0/3]``.

    ⛔ THE ONE-HOT BLOCKS ARE CHECKED, NOT TRUSTED. A soft nav distribution or a
    scalar ceiling smuggled into these slots would be a different, richer input
    wearing the same shape — and the whole defence of the 4-value ladder is that
    it is coarse. A row may be ALL-ZERO (the "not known" state, see
    ``refcv6_max_speed.speed_max_onehot``); it may not be anything else.

    ⚠️ ``a0`` is the MEASURED longitudinal acceleration at t0 — the same channel
    ``ego_state_at_t0`` supplies. It is an ego-state read at the last OBSERVED
    frame, which E11' already admits; it is NOT ``(future[0].v - v0)/dt``, the
    wrong implementation ``provenance_roles`` pins an interventional probe
    against.
    """
    for name, oh, k in (("nav", nav_onehot, N_NAV_COMMANDS),
                        ("max_speed", vmax_onehot, N_SPEED_MAX_BINS_V6)):
        if oh.dim() != 2 or oh.shape[-1] != k:
            raise ValueError(
                f"[refcv6-tac] {name} block must be [B, {k}] one-hot, got "
                f"{tuple(oh.shape)}")
        s = oh.sum(dim=-1)
        ok = ((s - 1.0).abs() < 1e-5) | (s.abs() < 1e-5)
        # ⛔ THE ROW SUM IS NOT ENOUGH, AND THE MUTATION TEST CAUGHT THAT: a
        # uniform block (0.25, 0.25, 0.25, 0.25) sums to exactly 1.0 and would
        # pass a sum-only check while being precisely the soft distribution
        # this refusal exists to stop. Every ELEMENT must be 0 or 1.
        binary = (((oh - 0.0).abs() < 1e-5)
                  | ((oh - 1.0).abs() < 1e-5)).all()
        if not bool(ok.all()) or not bool(binary):
            raise ValueError(
                f"[refcv6-tac] ⛔ the {name} block is not a one-hot (row sums "
                f"{s.tolist()[:8]}..., all-binary={bool(binary)}). A soft "
                f"distribution here is a RICHER input than the PI authorised, "
                f"wearing the same shape. An all-zero row (not known) is "
                f"allowed; nothing else is.")
    b = nav_onehot.shape[0]
    return torch.cat([nav_onehot.to(torch.float32),
                      vmax_onehot.to(torch.float32),
                      v0.reshape(b, 1).to(torch.float32) / float(v_scale),
                      a0.reshape(b, 1).to(torch.float32) / float(a_scale)],
                     dim=-1)


# ---------------------------------------------------------------------------
# the feeds to the planner — DETACHED
# ---------------------------------------------------------------------------

def planner_feeds(out: Mapping[str, Tensor], *,
                  valid_threshold: float = 0.5) -> dict[str, Tensor]:
    """The three things this layer hands the planner, **all detached**.

      ``lat_logprob`` ``[B, 8]`` / ``lon_logprob`` ``[B, 8]``
          the anchor PRIOR. It replaces the image-only ``lat3``/``lon3`` through
          NEW zero-init ``8 -> n_anchors`` grafts in ``refc.py`` (delivered as a
          patch; this function is its input).
      ``valid_behaviour`` ``[B, 22]``
          probabilities, for the selection gate.
      ``valid_mask`` ``[B, 22]`` bool
          the thresholded set, for reporting and for a hard gate.

    ⛔⛔ DETACHED, AND THAT IS WHAT MAKES THE FEED SAFE. If the planning loss
    could flow back here, the cheapest way to lower it is to reshape the
    behaviour head into whatever already correlates with the trajectory — and
    then its per-class numbers stop measuring what the LABELS taught it and
    start measuring what the planner found convenient. Detached, every bit this
    layer learns comes from its OWN losses.

    ⚠️ NOT a defence against nav-derived turns: the PI ruled on 2026-09-16 that
    deriving TURN_L/TURN_R from the nav command is BY DESIGN at this stage. The
    detachment is about the planning loss, not about nav.
    """
    return {
        "lat_logprob": torch.log_softmax(out["lat_logits"], dim=-1).detach(),
        "lon_logprob": torch.log_softmax(out["lon_logits"], dim=-1).detach(),
        "valid_behaviour": torch.sigmoid(out["goal_logits"]).detach(),
        "valid_conf": torch.sigmoid(out["goal_conf"]).detach(),
        "valid_mask": (torch.sigmoid(out["goal_logits"]).detach()
                       >= float(valid_threshold)),
    }


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

@dataclass
class TacticalLossWeights:
    """The spec's split of the EXISTING ``MANEUVER_WEIGHT`` budget.

    ⛔ *"inside the existing MANEUVER_WEIGHT budget (do not inflate the total)"*.
    ``refc_train.MANEUVER_WEIGHT`` is **0.1** and today splits as
    ``LAT_WEIGHT = LON_WEIGHT = MANEUVER_WEIGHT / 2 = 0.05``. refcv6 re-splits
    the SAME 0.1 three ways: 0.05 BCE on the goal tokens + 0.025 CE per action
    head. :meth:`assert_within_budget` refuses a re-weighting that inflates it.
    """

    goal_bce: float = 0.05
    lat_ce: float = 0.025
    lon_ce: float = 0.025
    budget: float = 0.10            # == refc_train.MANEUVER_WEIGHT
    #: ⭐⭐ A STATED DEVIATION FROM THE SPEC'S LITERAL WORDING, with its reason.
    #: The spec lists **confidence** as an OUTPUT of this layer but names only
    #: three loss terms. MEASURED the moment the head was built: with three
    #: terms, ``conf_head.weight.grad is None`` — 1,430 parameters that exist,
    #: are emitted, are fed into selection, and receive NO gradient. That is
    #: precisely the "declared budget, no gradient" failure the programme
    #: measured at 42 of 138 tensors on 2026-09-06.
    #: ⇒ the confidence is supervised as a SUB-SPLIT of the goal BCE budget,
    #: NOT as a fourth term: ``goal_bce`` still totals 0.05 and the three-way
    #: total is still exactly ``MANEUVER_WEIGHT``. Nothing is inflated.
    conf_share: float = 0.2         # 0.05 -> validity 0.04 + confidence 0.01

    @property
    def goal_validity(self) -> float:
        return self.goal_bce * (1.0 - self.conf_share)

    @property
    def goal_confidence(self) -> float:
        return self.goal_bce * self.conf_share

    def total(self) -> float:
        return self.goal_bce + self.lat_ce + self.lon_ce

    def assert_within_budget(self) -> None:
        if self.total() > self.budget + 1e-9:
            raise ValueError(
                f"[refcv6-tac] ⛔ tactical weights sum to {self.total():.6g}, "
                f"over the MANEUVER_WEIGHT budget {self.budget:.6g}. The spec "
                f"pins the tactical layer INSIDE the existing budget so a "
                f"refcv6 arm's total objective is comparable with refcv5-v2's; "
                f"raising it silently makes every loss curve incomparable.")

    def to_dict(self) -> dict:
        return {"w_tac_goal_bce": self.goal_bce,
                "w_tac_goal_validity": self.goal_validity,
                "w_tac_goal_confidence": self.goal_confidence,
                "w_tac_lat_ce": self.lat_ce,
                "w_tac_lon_ce": self.lon_ce, "budget": self.budget,
                "total": self.total()}


def tactical_behaviour_losses(
        out: Mapping[str, Tensor],
        *,
        goal_y: Tensor, goal_w: Tensor,
        lat_target: Tensor, lon_target: Tensor,
        weights: TacticalLossWeights | None = None,
        goal_pos_weight: Tensor | Sequence[float] | None = None,
        goal_class_mask: Tensor | Sequence[bool] | None = None,
        lat_class_weight: Tensor | None = None,
        lon_class_weight: Tensor | None = None,
        ignore_index: int = -100) -> tuple[Tensor, dict[str, Any]]:
    """``(weighted_total, telemetry)``.

    ``goal_y`` / ``goal_w`` ``[B, 22]`` come from
    ``v7_labels.tactical_goal_targets`` — ``w = 0`` marks a cell with NO
    EVIDENCE, which must not train either way. ``lat_target`` / ``lon_target``
    ``[B]`` come from ``v7_labels.tactical_class_ids``, which returns
    ``IGNORE_ID = -100`` outside the record's ±2 s band.

    ⛔⛔ NO TERM IS GUARDED BEHIND ``if w > 0``. MEASURED across the programme
    2026-09-06: **42 of 138 optimizer tensors received no gradient — 52.2 % of a
    declared budget** — because objectives were weighted 0.0 *and guarded*. A
    guarded term makes ``p.grad`` **None**, which is indistinguishable from a
    head that was never wired. Here every term is always computed and the weight
    is applied by MULTIPLICATION, so at weight 0.0 the parameters still receive a
    zeros gradient and ``p.grad is None`` stays a clean discriminator for "never
    wired". (Pinned by mutation in the test module.)

    ⛔ The telemetry reports ``n_supervised`` for EVERY term. A bare ``0.0`` in a
    metrics row reads as "supervised, and perfect", so an unlabelled step and a
    perfect step would print the same character.
    """
    from tanitad.refs.tac_goal_head import tac_goal_loss

    w = weights or TacticalLossWeights()
    w.assert_within_budget()
    logits = out["goal_logits"]
    loss_goal, n_goal = tac_goal_loss(
        logits, goal_y, goal_w,
        pos_weight=goal_pos_weight, class_mask=goal_class_mask)

    def _ce(lg: Tensor, tgt: Tensor, cw: Tensor | None) -> tuple[Tensor, int]:
        t = tgt.reshape(-1).long()
        n = int((t != ignore_index).sum().item())
        # ⛔ `reduction="sum"` over a clamped denominator, NOT `"mean"`:
        # torch's mean returns NaN when every element is ignored, and a NaN
        # that reaches the total kills the step. A batch entirely outside the
        # ±2 s band is COMMON here (1,157 of 4,823 eval windows are in band),
        # so this is the normal path, not an edge case.
        per = F.cross_entropy(lg, t.clamp_min(0), weight=cw,
                              reduction="none")
        keep = (t != ignore_index).to(per.dtype)
        return (per * keep).sum() / keep.sum().clamp_min(1.0), n

    loss_lat, n_lat = _ce(out["lat_logits"], lat_target, lat_class_weight)
    loss_lon, n_lon = _ce(out["lon_logits"], lon_target, lon_class_weight)

    # ⭐⭐ THE CONFIDENCE HEAD IS SUPERVISED, AND THIS EXISTS BECAUSE A MUTATION
    # TEST FOUND IT DEAD. `conf_head` emits a per-behaviour confidence that the
    # spec asks for and that selection reads; with only the three named terms
    # its `p.grad` was **None** — 1,430 parameters, emitted and consumed, that
    # no gradient could reach. The target is SELF-ASSESSMENT: "will the
    # validity decision on this token be CORRECT?".
    # ⛔ THE VALIDITY LOGITS ENTER DETACHED (`no_grad` on the target). A
    # confidence head that could reshape the validity head would lower its own
    # loss by making the validity head predictable rather than right — the
    # confidence would then measure the head's own conservatism, not its
    # correctness.
    with torch.no_grad():
        correct = ((torch.sigmoid(logits) >= 0.5).to(logits.dtype)
                   == goal_y.to(logits.dtype)).to(logits.dtype)
    cw = goal_w.to(logits.dtype)
    if goal_class_mask is not None:
        _cm = (goal_class_mask if torch.is_tensor(goal_class_mask)
               else torch.tensor(list(goal_class_mask)))
        cw = cw * _cm.to(logits.device, logits.dtype).view(1, -1)
    per_conf = F.binary_cross_entropy_with_logits(
        out["goal_conf"], correct, reduction="none")
    loss_conf = (per_conf * cw).sum() / cw.sum().clamp_min(1.0)

    total = (w.goal_validity * loss_goal + w.goal_confidence * loss_conf
             + w.lat_ce * loss_lat + w.lon_ce * loss_lon)
    tele = {
        "tac_goal_bce": loss_goal.detach(),
        "tac_goal_conf_bce": loss_conf.detach(),
        "tac_lat_ce": loss_lat.detach(),
        "tac_lon_ce": loss_lon.detach(),
        "tac_total_weighted": total.detach(),
        "n_supervised_goal_cells": n_goal,
        "n_supervised_lat": n_lat,
        "n_supervised_lon": n_lon,
        "weights": w.to_dict(),
    }
    return total, tele


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_class_report(out: Mapping[str, Tensor], *, goal_y: Tensor,
                     goal_w: Tensor, lat_target: Tensor, lon_target: Tensor,
                     threshold: float = 0.5,
                     ignore_index: int = -100) -> dict[str, Any]:
    """⛔ PER CLASS, NEVER POOLED — with the majority-class control beside it.

    The goal block reuses ``tac_goal_head.per_class_scores`` and
    ``majority_control_scores`` (one implementation, and the control's recall is
    known in advance: exactly 0.0 for an absent-majority token, exactly 1.0 for
    a present-majority one — a panel whose control does not read those values
    has a scoring bug, not a result).

    The action blocks report ``n``, recall and precision per class, and name the
    classes with ZERO positives in the blob: ``LANE_CHANGE_L``,
    ``LANE_CHANGE_R``, ``ABORT_LC`` (lateral) and ``YIELD_MERGE``
    (longitudinal). A recall of 0.0 on those is a LABEL fact, not a model fact,
    and reporting them without their n invites the opposite reading.
    """
    from tanitad.refs.tac_goal_head import (per_class_scores,
                                            majority_control_scores)

    rep: dict[str, Any] = {
        "goal": per_class_scores(out["goal_logits"], goal_y, goal_w,
                                 threshold=threshold,
                                 tokens=TACTICAL_GOAL_TOKENS_V7),
        "goal_majority_control": majority_control_scores(
            goal_y, goal_w, tokens=TACTICAL_GOAL_TOKENS_V7),
    }
    for axis, key, toks, tgt in (("lat", "lat_logits", TACTICAL_LAT_ACTIONS_V7,
                                  lat_target),
                                 ("lon", "lon_logits", TACTICAL_LON_ACTIONS_V7,
                                  lon_target)):
        t = tgt.reshape(-1).long()
        m = t != ignore_index
        pred = out[key].argmax(dim=-1)
        blk: dict[str, Any] = {"n_in_band": int(m.sum()),
                               "n_out_of_band": int((~m).sum())}
        for i, name in enumerate(toks):
            n_pos = int(((t == i) & m).sum())
            tp = int(((t == i) & (pred == i) & m).sum())
            n_fired = int(((pred == i) & m).sum())
            blk[name] = {
                "n": n_pos,
                "recall": (tp / n_pos) if n_pos else None,
                "precision": (tp / n_fired) if n_fired else None,
                "n_fired": n_fired,
                "_note": ("ZERO positives in the v8 train blob — a 0.0 here is "
                          "a LABEL fact, not a model fact")
                if n_pos == 0 else None,
            }
        rep[axis] = blk
    rep["_reads"] = (
        "17 of 22 goal tokens are trainable (5 carry no supervised negative); "
        f"10 sit under the n = {GOAL_MIN_N_FOR_METRIC} scoreability floor and "
        "are reported WITH their n, never as a rate; only 7 are both. lat/lon "
        "are supervised only within the record's +-2 s band.")
    return rep
