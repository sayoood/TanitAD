"""ONE hierarchy rung, shared by every arm — the PI's "same and best hierarchy
architecture in all our designs" (2026-08-21).

⛔ WHY THIS EXISTS. MEASURED 2026-08-21 by building all four arms and counting:

    component        v6                    REF-A v1              REF-D        REF-C v3
    tactical         5,767,981             54,587,392            54,587,392   1,980,646
      · predictor    FTac 3,809,792 @512   TokenField 54.59M     TokenField   ⛔ NONE
    strategic        4,152,993             1,911,040             ⛔ NOT BUILT 195
      · predictor    FTac 3,481,856 @256   StratSubspace 1.91M   ⛔ NONE      ⛔ NONE
    shared vocab     vocab_tac/str/a_*     imports the tuples    imports      ⛔ NONE

⛔ **CORRECTED 2026-08-21.** An earlier version of this table said REF-A v1 had
NO per-layer predictor. **It does** — ``.tactical`` is a ``TokenFieldPredictor``
and ``.strategic`` is a ``StrategicSubspacePredictor``, which is literally the
PI's *"strategic gets its OWN predictor on a strategy-only latent subspace"*.
They are simply not NAMED ``predictor_*``. The real gaps are narrower:

  * **REF-C v3** has NEITHER — ``PhiTac`` is a causal-TCN pool, not a predictor,
    and its strategic "layer" is a **195-parameter linear head**;
  * **REF-D's strategic layer is configured but never built**
    (``str_dt``/``str_steps``/``w_future_str``/``strategic_cfg`` exist, yet
    ``named_children()`` has no ``strategic`` and NO state_dict key contains
    "str");
  * **only v6 shares its vocabulary TABLES** — the others import the tuples.

⇒ v6's rung is the reference. This module extracts it so "same hierarchy
everywhere" is ENFORCEABLE rather than aspirational — one implementation, four
consumers, the ``refc_select``/one-implementation precedent.

⛔⛔ ADDITIVE ONLY — v6 IS TRAINING RIGHT NOW UNDER TENSOR-STRICT RESUME.
This module does NOT touch ``V6Stack``. It re-composes the SAME component classes
(:class:`tanitad.models.tactical.FTac`, :class:`~tanitad.models.v6.GoalHead`,
:class:`~tanitad.models.v6.GoalConditioner`) in the SAME order with the SAME
dims, and :func:`assert_matches_v6` PROVES the param count is identical rather
than asserting it. A hierarchy that "looks the same" is not the same hierarchy.

WHAT ONE RUNG IS
============================================================================
    adapter    Linear(d_in, hidden) -> GELU -> Linear(hidden, d_layer) -> LN
    cond       GoalConditioner(vocab_ABOVE)   — the goal handed DOWN (omitted
               at the top rung, which has nothing above it)
    predictor  FTac(d_layer, d_goal=...)      — ⭐ THE LAYER'S OWN IMAGINATION
    goal_head  GoalHead(vocab_own, d_layer, d_cond=...)  — what it emits DOWN
    act_heads  GoalHead(vocab_act, d_layer) x N — factored, never a joint softmax
               (the 5-way lat+lon collapse "provably destroys the longitudinal
               decision" — MEASURED, `refc_tactical`)

⚠️ ONE VOCABULARY SOURCE. Vocabulary objects are PASSED IN, never constructed
here, so the emitting head above and the consuming conditioner below hold the
SAME ``nn.Module`` (§5 "one vocabulary, two views"). Constructing one internally
would silently create a second vocabulary — the failure REF-A v1's source names
and the programme has already paid for once.

⚠️ GOALS TRAVEL DOWNWARD DETACHED. This module does not detach — the CALLER
owns the gradient policy, because v6 uses ``_cut()`` and REF-C v3 uses its own
``cons_detach`` discipline, and hiding a detach in a shared component would make
both callers' policies invisible at their call sites.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from tanitad.models.tactical import FTac
from tanitad.models.v6 import GoalConditioner, GoalHead, GoalVocabulary

__all__ = ["HierarchyRungConfig", "HierarchyRung", "rung_param_count",
           "assert_matches_v6", "RungBlock", "V6_TACTICAL", "V6_STRATEGIC",
           "ALIGNED_TACTICAL", "ALIGNED_STRATEGIC"]


@dataclass(frozen=True)
class HierarchyRungConfig:
    """Geometry of one rung. Defaults are v6's MEASURED values, so a rung built
    with ``HierarchyRungConfig()`` is v6's tactical rung exactly."""

    d_in: int = 2048            # v6 layer T: d_uplink_tac = d_op (shared encoder)
    d_layer: int = 512          # v6 cfg.d_tac
    adapter_hidden: int = 512   # v6 cfg.adapter_hidden
    d_goal_embed: int = 128     # v6 cfg.d_goal_embed
    f_hidden: int = 512         # v6 cfg.f_hidden_tac / f_hidden_str
    f_blocks: int = 3           # v6 cfg.f_blocks
    #: FTac's goal-conditioning width. v6 layer T passes ``2 * d_goal_embed``
    #: (its own goal AND the one from above); layer S passes ``d_goal_embed``.
    d_goal_pred: int = 256
    #: ⭐ OPTIONAL BODY — the REF-A v1 / REF-D capacity tier (PI 2026-08-21:
    #: "align Ref c v3 to Ref av1 and Ref-D tactical size"). REF-A v1's tactical
    #: is a ``TokenFieldPredictor`` whose ``blocks`` MEASURE 50,384,896 at
    #: d_state=1024 / tac_layers=4 (~12.6 M per pre-norm block at d=1024).
    #: ⚠️ ``body_layers=0`` reproduces v6's rung EXACTLY, which is what
    #: :func:`assert_matches_v6` checks — so the body is additive capacity, not
    #: a redefinition of the rung.
    body_layers: int = 0
    body_heads: int = 8


class RungBlock(nn.Module):
    """Pre-norm transformer block — the body's unit of capacity.

    ⚠️ Sized to MATCH REF-A v1's tactical blocks rather than invented: at
    d=1024 this is ~12.6 M, and 4 of them reproduce REF-A v1's measured
    50,384,896. :func:`aligned_body_matches_refa` proves it.
    """

    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Linear(4 * d, d))

    def forward(self, x: Tensor) -> Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class HierarchyRung(nn.Module):
    """One rung: adapter + conditioner + OWN predictor + goal head + act heads.

    ``vocab_above=None`` marks the TOP rung (nothing hands it a goal), which is
    v6's strategic layer — and is why ``cond`` is absent there rather than
    zeroed. A zeroed conditioner would still carry parameters and would make the
    top rung's state_dict differ from v6's.
    """

    def __init__(self, cfg: HierarchyRungConfig, *,
                 vocab_goal: GoalVocabulary,
                 vocab_actions: tuple[GoalVocabulary, ...],
                 vocab_above: GoalVocabulary | None = None):
        super().__init__()
        if not vocab_actions:
            raise ValueError(
                "a rung with no action head cannot express a decision — v6's "
                "layer T has (lat, lon) and layer S has (a_str,)")
        self.cfg = cfg
        self.adapter = nn.Sequential(
            nn.Linear(cfg.d_in, cfg.adapter_hidden), nn.GELU(),
            nn.Linear(cfg.adapter_hidden, cfg.d_layer),
            nn.LayerNorm(cfg.d_layer))
        #: ⭐ REF-A/REF-D capacity tier. Empty at ``body_layers=0`` (v6's rung).
        self.body = nn.ModuleList(
            [RungBlock(cfg.d_layer, cfg.body_heads)
             for _ in range(cfg.body_layers)])
        # ⭐ the layer's OWN predictor — v6's FTac, canonical for every arm
        # (PI: "align Refa av1, refd and Refcv3 to v6 own tactical predictor").
        self.predictor = FTac(cfg.d_layer, d_goal=cfg.d_goal_pred,
                              hidden=cfg.f_hidden, n_blocks=cfg.f_blocks)
        self.cond = (GoalConditioner(vocab_above, cfg.d_goal_embed)
                     if vocab_above is not None else None)
        self.goal_head = GoalHead(
            vocab_goal, cfg.d_layer,
            d_cond=(cfg.d_goal_embed if vocab_above is not None else 0))
        self.act_heads = nn.ModuleList(
            [GoalHead(v, cfg.d_layer) for v in vocab_actions])

    # ---- the three operations a rung performs ---------------------------- #
    def uplink(self, z_below: Tensor) -> Tensor:
        """Lower latent -> this rung's own state, through the adapter and the
        (optional) body. ⚠️ The caller decides whether ``z_below`` arrives
        detached; see the module docstring."""
        z = self.adapter(z_below)
        if not self.body:
            return z
        seq = z.dim() == 2
        if seq:
            z = z[:, None]                      # [B, d] -> [B, 1, d]
        for blk in self.body:
            z = blk(z)
        return z[:, 0] if seq else z

    def condition(self, goal_from_above):
        """Goal handed DOWN -> conditioning vector, or ``None`` at the top."""
        if self.cond is None:
            return None
        return self.cond(goal_from_above)

    def imagine(self, z: Tensor, g_flat: Tensor) -> Tensor:
        """⭐ This rung's OWN forward roll, in its OWN space."""
        return self.predictor(z, g_flat)


def rung_param_count(rung: HierarchyRung) -> dict[str, int]:
    out = {n: sum(p.numel() for p in m.parameters())
           for n, m in rung.named_children()}
    out["total"] = sum(p.numel() for p in rung.parameters())
    return out


def assert_matches_v6(stack, *, tactical: HierarchyRung,
                      strategic: HierarchyRung) -> dict[str, dict[str, int]]:
    """⛔ PROVE the extraction reproduces v6's rungs, do not assert it.

    Compares the SHARED components part-by-part against the live ``V6Stack``.
    ⚠️ Vocabulary tables are EXCLUDED from the comparison: v6 owns them at stack
    level and passes them in, so counting them here would double-count them.
    Raises on any mismatch, naming the component — a guard that cannot fail is
    not a guard.
    """
    def n(m):
        return sum(p.numel() for p in m.parameters())

    pairs = [
        ("tactical.adapter", n(tactical.adapter), n(stack.adapter_tac)),
        ("tactical.predictor", n(tactical.predictor), n(stack.predictor_tac)),
        ("tactical.cond", n(tactical.cond) if tactical.cond else 0,
         n(stack.cond_tac)),
        ("tactical.goal_head", n(tactical.goal_head), n(stack.goal_head_tac)),
        ("tactical.act_head[0]", n(tactical.act_heads[0]), n(stack.act_head_lat)),
        ("tactical.act_head[1]", n(tactical.act_heads[1]), n(stack.act_head_lon)),
        ("strategic.adapter", n(strategic.adapter), n(stack.adapter_str)),
        ("strategic.predictor", n(strategic.predictor), n(stack.predictor_str)),
        ("strategic.goal_head", n(strategic.goal_head), n(stack.goal_head_str)),
        ("strategic.act_head[0]", n(strategic.act_heads[0]), n(stack.act_head_str)),
    ]
    bad = [(k, a, b) for k, a, b in pairs if a != b]
    if bad:
        lines = "\n".join(f"    {k}: rung {a:,} != v6 {b:,}" for k, a, b in bad)
        raise AssertionError(
            f"HierarchyRung does NOT reproduce v6's rungs — {len(bad)} "
            f"component(s) differ:\n{lines}\n"
            f"⇒ this is not 'the same hierarchy'; fix the geometry before any "
            f"arm adopts it.")
    return {"tactical": rung_param_count(tactical),
            "strategic": rung_param_count(strategic),
            "checked": {k: a for k, a, _ in pairs}}


# --------------------------------------------------------------------------- #
# The four canonical rung geometries (PI 2026-08-21)
# --------------------------------------------------------------------------- #
#: v6's rungs EXACTLY — what ``assert_matches_v6`` pins. ``body_layers=0``.
V6_TACTICAL = HierarchyRungConfig(
    d_in=2048, d_layer=512, adapter_hidden=512, d_goal_embed=128,
    f_hidden=512, f_blocks=3, d_goal_pred=256, body_layers=0)
V6_STRATEGIC = HierarchyRungConfig(
    d_in=512, d_layer=256, adapter_hidden=512, d_goal_embed=128,
    f_hidden=512, f_blocks=3, d_goal_pred=128, body_layers=0)

#: ⭐ THE ALIGNED TIER — REF-A v1 / REF-D tactical CAPACITY carrying v6's OWN
#: predictor and heads. d=1024 x 4 blocks reproduces REF-A v1's measured
#: 50,384,896 block stack; the FTac predictor and the goal/action heads are v6's.
ALIGNED_TACTICAL = HierarchyRungConfig(
    d_in=2048, d_layer=1024, adapter_hidden=512, d_goal_embed=128,
    f_hidden=512, f_blocks=3, d_goal_pred=256, body_layers=4, body_heads=8)
#: Strategic stays at v6's module and width — the PI's instruction is to align
#: the OTHER arms TO v6 here, and REF-A v1's own strategic subspace is 256 too.
ALIGNED_STRATEGIC = HierarchyRungConfig(
    d_in=1024, d_layer=256, adapter_hidden=512, d_goal_embed=128,
    f_hidden=512, f_blocks=3, d_goal_pred=128, body_layers=0)


def aligned_body_matches_refa(rung: "HierarchyRung") -> int:
    """-> the body's parameter count, for comparison against REF-A v1's
    MEASURED tactical blocks (50,384,896). Proving the tier rather than
    asserting it is the point."""
    return sum(p.numel() for blk in rung.body for p in blk.parameters())
