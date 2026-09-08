"""refcv5 WP-B / ``E-WP-INDEX-1`` — DiffusionDrive coupling (1): the
**waypoint-indexed** cross-attention into the SPARSE agent tokens.

⭐⭐ **THE TRAJECTORY'S GEOMETRY IS THE ADDRESS.** This is an INDEX, not a
fusion block, and the distinction is the whole work package. A fusion block
concatenates (or gates) two learned latents and lets a network decide how to
mix them; the mixing weights are then a function of *content*. Here the anchor
query's **current waypoint estimate in metres** says *where to look*, and the
attention reads the agent slot that lives at that location. The query's own
latent ``q`` **never enters the address computation** — only its coordinates do
(:func:`waypoint_agent_geometry` takes tensors of metres and nothing else),
which is the audit in signature form.

⭐ **PUBLISHED PRECEDENT FOR EXACTLY OUR DEGRADED SHAPE.** DiffusionDrive ran
this coupling on Transfuser with BEV-only spatial cross-attention and **agent
cross-attention only, no map** — which is the configuration refcv5 is in. This
is not an extrapolation to our setting.

## ⛔ Why the address goes into SPARSE tokens and never into a dense raster

MEASURED by WP-A (``E-READOUT-CEILING-1``): a median of **4 BEV cells (max
313)** share one token cell, and only **242 of 640** token cells receive any
ground-plane cell at all, so **even a perfect front-end caps at AP 0.4713**. A
dense raster inherits a quantisation floor that **no amount of training
removes**. ``refc_agents.slot_features`` already carries **continuous metric
range and bearing** per slot — no grid, no floor — so the sparse route is the
one with a ceiling worth reaching. The dense raster stays a PROBE, never a
component.

## ⛔ The frame, stated once, because a formula in the wrong frame reads exactly like an answer

Both operands are **metres in the ego frame, x forward / y left**:

| tensor | where it comes from | units |
|---|---|---|
| waypoints ``[B, N, S, 2]`` | ``AnchoredDiffusionDecoder``'s current estimate (``x0``/``x``/``x_path``) | **metres**, ego, x fwd / y left (``kinematic.rollout_unicycle``: ``x += v cos(yaw) dt``) |
| agent positions ``[B, M, 2]`` | ``agent_slots["box"][..., :2]`` (``AgentSlotDecoder``'s decode) | **metres**, ego, x fwd / y left (``SlotDecodeRanges``: ``x_fwd_m`` 60.0, ``y_half_m`` 16.0) |

⇒ they are **already in the same frame and the same units**, and the address is
therefore a subtraction, not a projection. ⛔ Nothing in this module computes a
pixel, a bin index, or a camera formula. If a future caller supplies a
normalised trajectory this module will silently address the wrong place, so
:func:`waypoint_agent_geometry` is documented and tested against **analytic
metric literals** (a waypoint at a known range and bearing must resolve to a
computable slot) rather than against a re-run of its own arithmetic.

## The five things this module contains

1. :func:`waypoint_agent_geometry` — the **parameter-free ADDRESS**. Raw metric
   quantities, so a test can assert closed-form literals in metres.
2. :func:`waypoint_agent_relation` — the same address, scaled and stacked into
   the ``[B, N, M, WP_INDEX_FEAT_DIM]`` feature the bias head consumes.
3. :class:`WaypointIndexBias` — the per-layer head: relation -> a **per-head
   additive attention logit bias**. Its output layer is **ZERO-INIT**, so at
   step 0 the coupling contributes exactly nothing and every later movement is
   attributable to the seam (the ``ctx_to_cond`` / ``agent_gate`` discipline).
4. :class:`WaypointIndexConfig` — every default is OFF/inert, and the three
   CONTROLS are first-class run modes, not test-only monkeypatches.
5. :func:`apply_radius_gate` — the optional **deformable/local** variant, with
   the fully-masked-query guard that a naive implementation gets wrong.

## ⛔ The controls are RUN MODES, and each has a KNOWN VALUE it must read

| mode | what it does | the value it must read |
|---|---|---|
| ``detach`` | the address is computed but ``.detach()``-ed | gradient into the waypoints **through the index path** is exactly **0** |
| ``shuffle`` | waypoints permuted **across the batch** | the bias for row ``b`` equals the geometric bias of row ``perm[b]`` — **exactly**, an identity a test can assert. If this arm matches the real one, the coupling is CAPACITY, not CONTENT |
| ``const`` | every waypoint replaced by one fixed point | the bias is **identical across the anchor axis** — bitwise — because every query then asks the same question |

⛔ ``shuffle`` REFUSES ``B < 2``: a batch permutation of one row is the
identity, and a control that silently becomes the treatment is worse than no
control. *(Same family as the deliberate-regression arm that must be shown able
to fail before its PASS means anything.)*

## ⛔⛔ Two NaN traps, both real, both guarded here

1. **A fully-masked attention row returns NaN, not zero.** ``CrossAttnLayer.
   _attend_agents`` already un-masks batch rows with no agent at all; the
   RADIUS gate adds a second way to empty a row — **per QUERY** — and that one
   the existing guard does not see. :func:`apply_radius_gate` therefore drops
   its own mask on any query row it would have emptied.
2. **A non-finite BIAS poisons a row even where the key is padded.** A padded
   slot decodes to ``(0, 0)``; a bearing computed as ``atan2``/normalised dot
   would divide by ``|p| = 0`` and emit NaN, and ``NaN + (-inf)`` is ``NaN``,
   so the softmax over that row is NaN *everywhere* — including the live keys.
   Every denominator here is ``clamp_min``-ed, so the relation is finite by
   construction and not by a downstream ``nan_to_num``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "WP_INDEX_FEAT_DIM", "WP_INDEX_FEATURES", "WP_INDEX_MODES",
    "WaypointIndexConfig", "WaypointIndexBias",
    "waypoint_agent_geometry", "waypoint_agent_relation",
    "apply_radius_gate", "shuffle_waypoints", "const_waypoints",
]

#: numeric floor on every denominator in the address. Chosen so that a padded
#: slot at exactly ``(0, 0)`` and a stopped trajectory both produce FINITE
#: features rather than NaN — see the module docstring's trap 2.
_EPS: float = 1e-6

#: the address features, IN ORDER. Named because a report that says "feature 3"
#: is unreadable and a reordering would silently retrain a different index.
WP_INDEX_FEATURES: tuple[str, ...] = (
    "d_min_over_scale",     # closest approach, trajectory -> agent
    "exp_neg_d_min",        # a bounded locality kernel of the same quantity
    "tau",                  # WHEN on the trajectory the closest approach is
    "lon_over_scale",       # along-track offset at closest approach
    "lat_over_scale",       # cross-track offset at closest approach (left +)
    "cos_dbearing",         # bearing(agent) - bearing(waypoint*), as cos
    "sin_dbearing",         # ... and sin, so there is no wrap discontinuity
    "d_range_over_scale",   # |agent| - |waypoint*|, signed, metres
)
WP_INDEX_FEAT_DIM: int = len(WP_INDEX_FEATURES)

#: run modes. ``geom`` is the treatment; the other three are the CONTROLS.
WP_INDEX_MODES: tuple[str, ...] = ("geom", "shuffle", "const")


@dataclass
class WaypointIndexConfig:
    """Every default is OFF or inert.

    ⛔ ``enable=False`` must mean **NOT CONSTRUCTED**, not constructed-and-idle:
    a disabled-but-present module still draws from the global RNG at
    ``__init__`` and still lands in ``state_dict``, and both break the
    removability proof. The caller (``CrossAttnLayer``) enforces that.
    """

    #: build the bias head and index the agent cross-attention with it.
    enable: bool = False
    #: hidden width of the per-layer bias MLP. 32 is not a tuned value — it is
    #: the smallest width at which the head can represent a non-monotone
    #: function of the 8 address features, and its cost is auditable
    #: (``8*h + h + h*H + H`` per layer; 552 params at h=32, H=8).
    hidden: int = 32
    #: metres. The address features are divided by this so the MLP sees O(1)
    #: inputs. 10 m is the order of a lane-change lateral excursion and of a
    #: 1 s headway at urban speed; it is a NORMALISER, not a cut-off.
    scale_m: float = 10.0
    #: ⛔ THE CONTROL SWITCH. ``geom`` is the treatment.
    mode: str = "geom"
    #: ``const`` mode's fixed address, metres, ego frame.
    const_xy: tuple[float, float] = (10.0, 0.0)
    #: block the gradient that would otherwise flow BACK through the address
    #: into the trunk and the waypoints. The DETACHED control arm.
    detach: bool = False
    #: > 0 turns on the DEFORMABLE/LOCAL variant: an agent further than this
    #: from every waypoint of a query is masked out of that query's attention
    #: entirely. 0.0 = the soft (bias-only) index, which is the primary arm.
    radius_m: float = 0.0
    #: RNG seed for ``shuffle``'s batch permutation, so the control arm is
    #: reproducible and its identity is assertable.
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in WP_INDEX_MODES:
            raise ValueError(
                f"WaypointIndexConfig.mode {self.mode!r} not in "
                f"{WP_INDEX_MODES}. The three non-``geom`` modes are the "
                f"pre-registered CONTROLS and each has a known value it must "
                f"read; a typo here would silently run the treatment.")
        if int(self.hidden) <= 0:
            raise ValueError("WaypointIndexConfig.hidden must be > 0")
        if float(self.scale_m) <= 0.0:
            raise ValueError("WaypointIndexConfig.scale_m must be > 0 metres")

    def as_dict(self) -> dict:
        """Serialised into ``config.json['seams']['wp_index']`` — a run record
        that cannot rebuild its own model config is not a run record."""
        return {
            "enable": bool(self.enable), "hidden": int(self.hidden),
            "scale_m": float(self.scale_m), "mode": str(self.mode),
            "const_xy": [float(self.const_xy[0]), float(self.const_xy[1])],
            "detach": bool(self.detach), "radius_m": float(self.radius_m),
            "shuffle_seed": int(self.shuffle_seed),
            "feat_dim": int(WP_INDEX_FEAT_DIM),
            "features": list(WP_INDEX_FEATURES),
        }


# ---------------------------------------------------------------------------
# THE ADDRESS — parameter-free, metric, and analytically checkable
# ---------------------------------------------------------------------------
def waypoint_agent_geometry(wp: Tensor, pos: Tensor) -> dict:
    """The **raw metric address**: how each anchor trajectory relates to each
    agent slot. ``wp`` ``[B, N, S, 2]`` metres · ``pos`` ``[B, M, 2]`` metres.

    Returns a dict of ``[B, N, M]`` tensors (``s_star`` is ``long``), all in
    **metres / dimensionless**, so a guard can assert closed-form literals:

    ``d_min``   closest approach between the trajectory's waypoints and the
                agent — the primary "does this plan go near this thing" scalar;
    ``s_star``  the waypoint index at which that minimum happens — **WHEN**;
    ``tau``     ``s_star / (S - 1)``, the same in ``[0, 1]``;
    ``lon``     along-track offset of the agent from ``wp[s_star]``, measured
                along the trajectory's own local heading there;
    ``lat``     cross-track offset, LEFT positive (the left-normal of that
                heading is ``(-h_y, h_x)``);
    ``d_range`` ``|agent| - |wp[s_star]|`` in metres, signed;
    ``cos_db`` / ``sin_db``  the bearing of the agent relative to the bearing of
                ``wp[s_star]``, as a (cos, sin) pair so there is no ``atan2``
                wrap discontinuity to learn around.

    ⭐ **The local heading is measured with the EGO ORIGIN PREPENDED.** The
    first waypoint's heading is ``wp[0] - (0, 0)``, i.e. the direction the plan
    sets off in. Dropping that prepend and using ``wp[1:] - wp[:-1]`` shifts
    every heading by one slot and is a real off-by-one — it is one of the
    mutations the guard set is shown to catch.

    ⛔ Every denominator is ``clamp_min``-ed: a padded slot decodes to ``(0, 0)``
    and a stopped plan has coincident waypoints, and a NaN here would poison a
    whole attention row (module docstring, trap 2).
    """
    if wp.dim() != 4 or wp.shape[-1] != 2:
        raise ValueError(f"waypoints must be [B, N, S, 2] metres, got "
                         f"{tuple(wp.shape)}")
    if pos.dim() != 3 or pos.shape[-1] != 2:
        raise ValueError(f"agent positions must be [B, M, 2] metres, got "
                         f"{tuple(pos.shape)}")
    if pos.shape[0] != wp.shape[0]:
        raise ValueError(f"batch mismatch: waypoints {wp.shape[0]} vs agent "
                         f"positions {pos.shape[0]}")
    b, n, s, _ = wp.shape
    m = pos.shape[1]
    pos = pos.to(wp.dtype)

    # displacement waypoint -> agent, for every (anchor, step, slot)
    delta = pos[:, None, None, :, :] - wp[:, :, :, None, :]     # [B,N,S,M,2]
    dist = torch.linalg.vector_norm(delta, dim=-1)              # [B,N,S,M]
    d_min, s_star = dist.min(dim=2)                             # [B,N,M] each

    # local heading of the trajectory, ORIGIN PREPENDED (see the docstring)
    wp0 = torch.cat([wp.new_zeros(b, n, 1, 2), wp], dim=2)      # [B,N,S+1,2]
    step = wp0[:, :, 1:, :] - wp0[:, :, :-1, :]                 # [B,N,S,2]
    head = step / torch.linalg.vector_norm(
        step, dim=-1, keepdim=True).clamp_min(_EPS)             # [B,N,S,2]

    # gather the heading and the waypoint AT the closest approach.
    # `torch.gather(src[B,N,S,2], dim=2, index[B,N,M,2])` reads
    # src[b, n, index[b,n,m,k], k]; with both k-columns holding s_star that is
    # exactly src[b, n, s_star[b,n,m], :].
    idx = s_star[..., None].expand(b, n, m, 2)                  # [B,N,M,2]
    h = torch.gather(head, 2, idx)                              # [B,N,M,2]
    w_star = torch.gather(wp, 2, idx)                           # [B,N,M,2]
    d_star = pos[:, None, :, :] - w_star                        # [B,N,M,2]

    lon = (d_star * h).sum(-1)                                  # [B,N,M]
    lat = -d_star[..., 0] * h[..., 1] + d_star[..., 1] * h[..., 0]

    r_a = torch.linalg.vector_norm(pos, dim=-1)[:, None, :]     # [B,1,M]
    r_w = torch.linalg.vector_norm(w_star, dim=-1)              # [B,N,M]
    den = (r_a * r_w).clamp_min(_EPS)
    a_x = pos[:, None, :, 0]
    a_y = pos[:, None, :, 1]
    cos_db = (w_star[..., 0] * a_x + w_star[..., 1] * a_y) / den
    sin_db = (w_star[..., 0] * a_y - w_star[..., 1] * a_x) / den

    tau = s_star.to(wp.dtype) / float(max(s - 1, 1))
    return {"d_min": d_min, "s_star": s_star, "tau": tau,
            "lon": lon, "lat": lat,
            "d_range": r_a.expand_as(r_w) - r_w,
            "cos_db": cos_db, "sin_db": sin_db}


def waypoint_agent_relation(wp: Tensor, pos: Tensor,
                            scale_m: float = 10.0) -> Tensor:
    """:func:`waypoint_agent_geometry`, scaled and stacked -> ``[B, N, M, F]``
    in :data:`WP_INDEX_FEATURES` order. ``scale_m`` normalises the metric
    features; it is a NORMALISER and never a cut-off."""
    g = waypoint_agent_geometry(wp, pos)
    sc = float(scale_m)
    return torch.stack([
        g["d_min"] / sc,
        torch.exp(-g["d_min"] / sc),
        g["tau"],
        g["lon"] / sc,
        g["lat"] / sc,
        g["cos_db"],
        g["sin_db"],
        g["d_range"] / sc,
    ], dim=-1)


# ---------------------------------------------------------------------------
# The CONTROL transforms — applied to the WAYPOINTS, never to the agents
# ---------------------------------------------------------------------------
def shuffle_waypoints(wp: Tensor, seed: int = 0) -> tuple[Tensor, Tensor]:
    """⛔ CONTROL. Permute the waypoints ACROSS THE BATCH, so each row is
    addressed with **another sample's** geometry against **its own** agents.

    Returns ``(wp[perm], perm)`` — the permutation is returned so a guard can
    assert the exact identity ``bias_shuffled[b] == bias_geom[perm[b]]`` rather
    than merely "it changed".

    ⛔ **Refuses ``B < 2``.** A batch permutation of one row is the identity, so
    the control would silently BE the treatment — the "a guard that cannot go
    red proves nothing" failure, applied to a control instead of a guard.

    ⛔ Permuting the AGENTS instead would be a relabelling of the key axis and,
    with a permutation-equivariant attention, close to a no-op. The address is
    what must be made content-free, so the WAYPOINTS are what move.
    """
    b = wp.shape[0]
    if b < 2:
        raise ValueError(
            "wp_index shuffle control needs batch >= 2: a batch permutation of "
            "one row is the IDENTITY, so the control would be "
            "indistinguishable from the treatment by construction and would "
            "'confirm' the coupling no matter what it does.")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(b, generator=gen).to(wp.device)
    # a derangement is not required, but an identity draw would weaken the
    # control silently, so roll it forward by one if it happens.
    if bool(torch.equal(perm, torch.arange(b, device=wp.device))):
        perm = torch.roll(perm, 1, 0)
    return wp.index_select(0, perm), perm


def const_waypoints(wp: Tensor, xy: tuple[float, float]) -> Tensor:
    """⛔ CONTROL. Every waypoint of every anchor replaced by ONE fixed metric
    point, so every query asks the same question of the agent set. The known
    value: the emitted bias is **identical across the anchor axis**, bitwise."""
    out = wp.new_empty(wp.shape)
    out[..., 0] = float(xy[0])
    out[..., 1] = float(xy[1])
    return out


# ---------------------------------------------------------------------------
# The per-layer bias head
# ---------------------------------------------------------------------------
class WaypointIndexBias(nn.Module):
    """relation ``[B, N, M, F]`` -> per-head additive attention logit bias
    ``[B * n_heads, N, M]``, the shape ``nn.MultiheadAttention`` takes as a
    float ``attn_mask``.

    ⭐ **The output layer is ZERO-INIT** (the ``ctx_to_cond`` / ``agent_gate``
    discipline), so at step 0 the index adds exactly ``0.0`` to every logit and
    a fresh ``+index`` run starts from the same attention the agent seam
    already had — while the gradient is non-zero, so it is GATED, not dead.

    ⛔ The bias is ``[B*H, N, M]`` and NOT ``[B*H, M, N]``. Transposed it is a
    silent mis-index whenever ``N == M`` and a shape error otherwise; the guard
    set carries the mutation.
    """

    def __init__(self, cfg: WaypointIndexConfig, n_heads: int):
        super().__init__()
        self.cfg = cfg
        self.n_heads = int(n_heads)
        h = int(cfg.hidden)
        self.mlp = nn.Sequential(
            nn.Linear(WP_INDEX_FEAT_DIM, h), nn.GELU(), nn.Linear(h, n_heads))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, rel: Tensor) -> Tensor:
        """``rel`` ``[B, N, M, F]`` -> ``[B * n_heads, N, M]``."""
        if rel.shape[-1] != WP_INDEX_FEAT_DIM:
            raise ValueError(
                f"wp_index relation must end in {WP_INDEX_FEAT_DIM} features "
                f"({WP_INDEX_FEATURES}), got {rel.shape[-1]}")
        b, n, m, _ = rel.shape
        bias = self.mlp(rel)                       # [B, N, M, H]
        # -> [B, H, N, M] -> [B*H, N, M]; the head axis must be OUTERMOST after
        # the batch, which is torch's own `attn_mask` convention.
        return bias.permute(0, 3, 1, 2).reshape(b * self.n_heads, n, m)


# ---------------------------------------------------------------------------
# The DEFORMABLE/LOCAL variant and its fully-masked-query guard
# ---------------------------------------------------------------------------
def apply_radius_gate(bias: Tensor, d_min: Tensor, radius_m: float,
                      n_heads: int, pad: Tensor | None = None) -> Tensor:
    """Mask every (query, slot) pair whose closest approach exceeds
    ``radius_m``, giving the index a strictly LOCAL receptive field — the
    sparse-token analogue of deformable attention's bounded sampling offsets.

    ``bias`` ``[B*H, N, M]`` · ``d_min`` ``[B, N, M]`` metres · ``pad``
    ``[B, M]`` bool, torch's convention (``True`` == this slot is PADDING).

    ⛔⛔ **A QUERY ROW THAT THIS WOULD EMPTY GETS ITS GATE DROPPED, AND THE
    EMPTINESS TEST MUST INCLUDE THE PADDING.** ``CrossAttnLayer.
    _attend_agents`` already un-masks *batch* rows with no agent at all, but the
    radius can empty a row **per QUERY** — an anchor that simply plans away from
    every detected agent, which on a real road is the common case, not an edge
    case. Softmax over an all-``-inf`` row is undefined and returns **NaN, not
    zero**, and that NaN flows through the residual and reads as an exploding
    planner rather than as an empty neighbourhood.

    ⚠️ **And testing the radius mask ALONE is not enough.** A padded slot
    decodes to ``(0, 0)``, which is *inside* any sensible radius, so a query
    whose every LIVE neighbour is out of range would look non-empty to a
    radius-only test while every surviving key is padding — the attention mass
    would then land entirely on slots that do not exist. The emptiness test is
    therefore taken over ``radius OR padding``, and the fallback drops only the
    RADIUS half (the padding stays masked, as it must).
    """
    if radius_m <= 0.0:
        return bias
    b, n, m = d_min.shape
    drop = d_min > float(radius_m)                      # [B,N,M] True == drop
    combined = drop if pad is None else (drop | pad[:, None, :])
    empty = combined.all(dim=-1)                        # [B,N] would be NaN
    if bool(empty.any()):
        drop = drop & ~empty[..., None]
    drop = drop[:, None].expand(b, int(n_heads), n, m).reshape(
        b * int(n_heads), n, m)
    return bias.masked_fill(drop, float("-inf"))


def build_relation(wp: Tensor, pos: Tensor, cfg: WaypointIndexConfig
                   ) -> tuple[Tensor, Tensor]:
    """The whole address path for one decoder pass, controls included.

    ``wp`` ``[B, N, S, 2]`` metres · ``pos`` ``[B, M, 2]`` metres ->
    ``(relation [B, N, M, F], d_min [B, N, M])``.

    ⛔ ``detach`` is applied to BOTH operands and BEFORE any arithmetic, so the
    detached arm blocks the gradient the index would otherwise send back into
    the trunk through the waypoints AND through the agent boxes. Detaching only
    one of them would leave a live path and the control would read as a
    treatment.
    """
    if cfg.detach:
        wp, pos = wp.detach(), pos.detach()
    if cfg.mode == "shuffle":
        wp, _ = shuffle_waypoints(wp, cfg.shuffle_seed)
    elif cfg.mode == "const":
        wp = const_waypoints(wp, cfg.const_xy)
    g = waypoint_agent_geometry(wp, pos)
    sc = float(cfg.scale_m)
    rel = torch.stack([
        g["d_min"] / sc, torch.exp(-g["d_min"] / sc), g["tau"],
        g["lon"] / sc, g["lat"] / sc, g["cos_db"], g["sin_db"],
        g["d_range"] / sc,
    ], dim=-1)
    return rel, g["d_min"]
