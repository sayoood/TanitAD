"""refcv5 WP-7 / ``E-DDA-2b`` — the four-family coarse-to-fine sub-metric
selector, scored on the fan the decoder **actually emits**.

## The measured defect this module is the second half of

``D-REFC-DDAUDIT-3``: refcv4b ranks its fan on the **t=0 confidence**, and the
fan leaves the decoder **two refinement passes later**. ``sel_idx_base`` is
unchanged on **201/201** windows for any number of passes — i.e. the object
being ranked is two passes staler than the object being emitted. DiffusionDrive
scores what its last layer emits; we did not.

⛔ **Step 1 of the fix is NOT here and must not be rebuilt.** It already exists
in source as ``SelectionConfig.refined`` / ``score_emitted`` /
``score_emitted_t`` (``refc.py:465-475``) and costs **zero parameters**. This
module is **Step 2**: a *learned* selector, outside the generator, whose heads
are the four metric families rather than one scalar confidence.

## What the heads are, and what their targets may read

Six heads, all reported separately, never pooled into one number before the
composition below (`CLAUDE.md`: "per-family, never pooled into one score").

| head | family | target lives in |
|---|---|---|
| ``nc`` (no contact) | LONGITUDINAL (distance keeping) | ``refc_selector_targets.nc_target`` |
| ``ttc`` (min-TTC / time-gap) | LONGITUDINAL | ``ttc_target`` |
| ``progress`` (along-track vs ``v0``) | LONGITUDINAL (progress) | ``progress_target`` |
| ``comfort`` (jerk, a_lat, curvature, yaw rate) | LATERAL + comfort | ``comfort_target`` |
| ``compliance`` (terminal heading vs nav) | STRATEGIC | ``compliance_target`` |
| ``tactical`` (``(a_lon, a_lat)`` cell agreement) | TACTICAL | ``tactical_target`` ⚠️ **GT-derived, quarantined** |

⛔ The environment heads' targets are **rule-based predicates against the
replayed scene**. The ego's logged future never enters them — that is what
separates this module from ``refc_rescorer.py`` (v1.2, refused as SEL-1's
winner's curse: it was trained toward the fan's own GT-distance winner, which
is the answer wearing a target's clothes). The one exception, ``tactical``, is
derived from an ego-future label and is **quarantined**: it is reported
separately, it is OFF in the score composition by default (``w_tactical =
0.0``), and ``H-DDA-6`` forbids it being both an RL grouping key and a reward.

## The score composition (REFCV5_DESIGN_PLAN §5.2, verbatim)

    sigma(NC) * (5*sig(TTC) + 5*sig(prog) + 2*sig(comf)
                 + w_c*sig(compl) + w_t*sig(tac)) / (12 + w_c + w_t)

PDMS's structure with **the map half removed** and the two hierarchy heads
added. ⛔ ``DAC`` (drivable area) is **not constructible on PhysicalAI** — the
dataset card says verbatim *"we do not include open maps data"*, pinned by
``stack/tests/test_physicalai_feature_readset.py``. It is not omitted for
convenience; it has no source. On an AlpaSim/NuRec arm ``map.xodr`` supplies it
and this composition gains a factor — that is a different module and a
different arm.

## ⛔ Candidate self-attention lives ONLY here. Read this before touching it.

``SelectionConfig.anchor_prefilter`` (S2b) decodes only the reachable subset of
the fan and rests on **one structural fact**: ``refc.CrossAttnLayer``
cross-attends q -> kv with **no interaction along the candidate axis**, so
decoding a subset gives that subset the same answer (the SELECTION INDEX is
identical on 881/881 windows, ``refc.py:483-502``). The 3.46x-3.70x speedup and
that exactness are the same fact.

⇒ **A self-attention over candidates inside the GENERATOR would retire
``anchor_prefilter``.** It is placed here, in a stage-II module that scores an
already-emitted fan, precisely so the generator's independence survives. This
module's ``SelectorConfig.self_attn`` switch exists to make that argument
*testable* rather than merely asserted: with it off, changing candidate *j*
must leave candidate *i*'s score alone (``test_refc_selector.py``).

⚠️ Anyone adding candidate mixing to ``refc.CrossAttnLayer`` must retire
``anchor_prefilter`` in the same commit. This paragraph is the only warning
that will be there.

## What this module is NOT

* Not a re-scorer trained toward the GT-nearest candidate (SEL-1, refused).
* Not a replacement for the generator's own ``anchor_logits`` — it is a
  **second, disjoint** ranking whose disagreement with ``sel_idx`` is itself a
  readout (``sel_idx_selector`` beside ``sel_idx`` / ``sel_idx_base``).
* Not a source of any capability claim on its own. Its readouts are T1 on the
  four families (``EVAL_DOCTRINE.md``); the composed score is an OBJECTIVE, and
  quoting an objective as a result is the ``I5`` error.

## Sizing, and why it is half of V2's

V2 (``diffusiondrivev2_model_sel.py`` @ ``1cd12a1``) uses **d=512 / 16 heads**
over ~800 candidates. We use **d=256 / 8 heads** over ~350-700. That is a
Thor-minded halving and it is a **DECISION, not a default** — the selector runs
on the edge box beside the generator, and its width is the one thing here that
buys nothing on the science axis. Recorded in ``config.json['seams']
['selector']`` via :meth:`SelectorConfig.as_dict` so a run record can rebuild
its own model config.

⚠️ **THE PLAN'S PARAM ESTIMATE IS SHORT BY THE ONE THING THIS MODULE IS FOR.**
``REFCV5_DESIGN_PLAN.md`` §7.1 prices WP-7 at **"≈ 3.5 M, outside the
generator"**. MEASURED by building, ``scene_dim=704``, ``n_steps=8``, defaults::

    SelectorConfig()                        4,530,444    <- the real default
    SelectorConfig(cross_agent=True)        5,585,164
    SelectorConfig(self_attn=False)         3,475,724    <- what "3.5 M" priced

The 1,054,720 difference is exactly the four candidate self-attention blocks
(4 x [MHA(256): 262,144 + 1,024 bias, LayerNorm: 512] = 4 x 263,680). ⇒ the
estimate was arithmetic over the scene/agent path and **did not price the
candidate self-attention** — i.e. it priced the ablation, not the arm. The
launch preflight must assert the **measured** 4,530,444 the way
``REGISTERED_DELTA_KEYS_V4`` pins v4's, not the described 3.5 M. Same family as
every scope error in ``CLAUDE.md``: a true number quoted for a different
object.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "HEAD_NAMES", "ENV_HEADS", "GT_DERIVED_HEADS", "PDMS_WEIGHTS",
    "PDMS_DENOM", "NOT_SCORED", "SelectorConfig", "SubMetricSelector",
    "compose_score", "pdms_nc_binarise", "selector_losses",
    "sinusoidal_xy",
]


# ---------------------------------------------------------------------------
# The head set. Order is the composition's order and is load-bearing for the
# hand-computed control in the test suite.
# ---------------------------------------------------------------------------
HEAD_NAMES: tuple[str, ...] = (
    "nc", "ttc", "progress", "comfort", "compliance", "tactical",
)

#: Heads whose targets are ENVIRONMENT predicates — the ego's logged future is
#: structurally absent from them. Asserted by
#: ``refc_selector_targets.assert_environment_only``.
ENV_HEADS: frozenset[str] = frozenset(
    {"nc", "ttc", "progress", "comfort", "compliance"})

#: ⚠️ Heads whose target is derived from the ego future. Quarantined: reported
#: separately, OFF in the composition by default, and never simultaneously an
#: RL grouping key and a reward term (`H-DDA-6`).
GT_DERIVED_HEADS: frozenset[str] = frozenset({"tactical"})

#: PDMS's own sub-score weights, with the map half (DAC) removed because it has
#: no source on PhysicalAI. These three are NOT tunable knobs — they are the
#: published structure we are porting, and changing one makes the arm no longer
#: a port. `w_compliance` / `w_tactical` are the levers.
PDMS_WEIGHTS: dict[str, float] = {"ttc": 5.0, "progress": 5.0, "comfort": 2.0}
PDMS_DENOM: float = 12.0

#: ⭐ The sentinel for "this candidate was pruned by the coarse stage and the
#: fine scorer never saw it". The composition is a product of a sigmoid and a
#: convex-weighted sum of sigmoids, so it lies in (0, 1) — a NEGATIVE sentinel
#: is therefore unreachable by a real score and an argmax can never pick one.
#: ⛔ NOT ``-inf``: an inf propagates through any downstream mean/softmax as a
#: NaN and the failure surfaces somewhere else entirely.
NOT_SCORED: float = -1.0

#: Metre scale of the lowest sinusoidal band. DECISION, not a default: the 6 s
#: candidate horizon at typical corpus speeds (~10 m/s) spans ~60 m, so the
#: lowest band completes roughly one period over the whole candidate and the
#: higher bands resolve the shape. A scale much smaller than the horizon aliases
#: distant waypoints onto near ones; much larger and the embedding is nearly
#: linear and buys nothing over the raw coordinates.
POS_SCALE_M: float = 60.0

#: Below this the last segment carries no heading (a standstill has no tangent).
#: Mirrors ``taniteval.nav_compliance.STALL_SEGMENT_M``'s role; kept local
#: because that module is numpy and lives in the other package.
STALL_SEGMENT_M: float = 0.05


# ---------------------------------------------------------------------------
# Candidate featurisation
# ---------------------------------------------------------------------------
def sinusoidal_xy(xy: Tensor, n_freq: int,
                  scale_m: float = POS_SCALE_M) -> Tensor:
    """``[..., 2]`` metres -> ``[..., 4 * n_freq]`` sin/cos features.

    Bands are ``2**k * pi / scale_m``, k = 0..n_freq-1, applied to x and y
    independently. Both sin AND cos are kept: sin alone is ambiguous about the
    sign of the phase, which for a lateral offset is the difference between
    turning left and turning right.

    ⚠️ The features are a function of the METRE value, so the scale is part of
    the model. Changing ``scale_m`` between train and eval silently re-maps
    every candidate — it is serialised in :meth:`SelectorConfig.as_dict`.
    """
    if xy.shape[-1] != 2:
        raise ValueError(f"expected [..., 2] positions, got {tuple(xy.shape)}")
    k = torch.arange(n_freq, device=xy.device, dtype=xy.dtype)
    freqs = (2.0 ** k) * (math.pi / float(scale_m))          # [n_freq]
    a = xy.unsqueeze(-1) * freqs                             # [..., 2, n_freq]
    return torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)


def _terminal_heading(traj: Tensor) -> Tensor:
    """``[..., S, 2]`` -> ``[...]`` radians, the tangent of the LAST segment.

    A stalled last segment (< :data:`STALL_SEGMENT_M`) reads exactly 0.0. This
    is the same convention as ``taniteval.nav_compliance.terminal_heading`` and
    the two must not drift: the compliance TARGET is computed with that
    predicate's definition, so a selector that embedded a different heading
    would be asked to predict a quantity it cannot see.
    """
    d = traj[..., -1, :] - traj[..., -2, :]
    seg = d.norm(dim=-1)
    th = torch.atan2(d[..., 1], d[..., 0])
    return torch.where(seg < STALL_SEGMENT_M, torch.zeros_like(th), th)


class CandidateEmbed(nn.Module):
    """Candidate waypoints -> one ``d``-dim token per candidate.

    Features per candidate, concatenated: sinusoids of every waypoint (x, y),
    sin/cos of the terminal heading, and the terminal point divided by
    ``scale_m``. The raw terminal point is kept alongside the sinusoids because
    a sinusoidal code is periodic and a *monotone* along-track feature is what
    the progress head is actually ranking on.

    ⚠️ ``n_steps`` is a GEOMETRY, not a hyper-parameter: it is the emitted fan's
    step count and the input width depends on it. Passing the wrong one raises
    at construction rather than producing a quietly mis-shaped Linear.
    """

    def __init__(self, d: int, n_steps: int, n_freq: int,
                 scale_m: float = POS_SCALE_M):
        super().__init__()
        if n_steps < 2:
            raise ValueError(f"need >=2 waypoints for a heading, got {n_steps}")
        self.n_steps = int(n_steps)
        self.n_freq = int(n_freq)
        self.scale_m = float(scale_m)
        in_dim = n_steps * 4 * n_freq + 2 + 2
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.in_dim = in_dim

    def forward(self, traj: Tensor) -> Tensor:
        """``[B, N, S, 2]`` -> ``[B, N, d]``."""
        if traj.dim() != 4 or traj.shape[-1] != 2:
            raise ValueError(
                f"expected [B, N, S, 2] candidates, got {tuple(traj.shape)}")
        if traj.shape[-2] != self.n_steps:
            raise ValueError(
                f"selector built for n_steps={self.n_steps}, got "
                f"{traj.shape[-2]} — the emitted fan changed geometry")
        b, n = traj.shape[0], traj.shape[1]
        pos = sinusoidal_xy(traj, self.n_freq, self.scale_m)   # [B,N,S,4F]
        th = _terminal_heading(traj)                            # [B, N]
        end = traj[..., -1, :] / self.scale_m                   # [B, N, 2]
        feat = torch.cat(
            [pos.reshape(b, n, -1), th.sin()[..., None], th.cos()[..., None],
             end], dim=-1)
        return self.norm(self.proj(feat))


# ---------------------------------------------------------------------------
# One scorer layer
# ---------------------------------------------------------------------------
class SelectorLayer(nn.Module):
    """scene cross-attn -> [agent cross-attn] -> [candidate self-attn] -> FFN.

    Pre-norm residual throughout, matching ``refc.CrossAttnLayer`` so the two
    blocks are comparable when an ablation moves work between them.

    ⛔ **No FiLM / no ``cond``.** ``CrossAttnLayer`` modulates its MLP with the
    live core conditioning because it sits *inside* the generator. This module
    is stage II with the generator FROZEN: its only view of the scene is the
    conv map it cross-attends, and adding a second conditioning path would make
    "the selector improved" non-attributable between the heads and a new
    conditioning route. One variable per arm.

    ⚠️ ``agent_pad`` is torch's convention: **True where the slot is PADDING**.
    Inverting it attends to nothing and reads exactly like a dead seam (the
    failure ``refc_agents`` documents at its own seam).
    """

    def __init__(self, d: int, n_heads: int, ff_mult: int,
                 self_attn: bool = True, cross_agent: bool = False):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.cross_scene = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.norm_a: nn.LayerNorm | None = None
        self.cross_agent: nn.MultiheadAttention | None = None
        if cross_agent:
            self.norm_a = nn.LayerNorm(d)
            self.cross_agent = nn.MultiheadAttention(
                d, n_heads, batch_first=True)

        self.norm_c: nn.LayerNorm | None = None
        self.self_cand: nn.MultiheadAttention | None = None
        if self_attn:
            self.norm_c = nn.LayerNorm(d)
            self.self_cand = nn.MultiheadAttention(
                d, n_heads, batch_first=True)

        self.norm_f = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))

    def forward(self, q: Tensor, scene_kv: Tensor,
                agents: Tensor | None = None,
                agent_pad: Tensor | None = None,
                cand_pad: Tensor | None = None) -> Tensor:
        h = self.norm_q(q)
        q = q + self.cross_scene(h, scene_kv, scene_kv, need_weights=False)[0]

        if self.cross_agent is not None and agents is not None:
            # ⛔ A FULLY-PADDED ROW IS A NaN FACTORY IN BOTH DIRECTIONS.
            # `key_padding_mask` all-True on a row normalises a softmax over all
            # -inf; `nan_to_num` patches the value and the softmax BACKWARD
            # still emits NaN. A window with no detected agents is an ordinary
            # empty road, not an edge case. Same treatment as
            # `refc.CrossAttnLayer`: un-mask the row and zero its contribution.
            pad, live = agent_pad, None
            if pad is not None:
                empty = pad.all(dim=1, keepdim=True)
                pad = pad & ~empty
                live = (~empty).to(q.dtype).unsqueeze(-1)
            a = self.norm_a(q)
            att = self.cross_agent(a, agents, agents, key_padding_mask=pad,
                                   need_weights=False)[0]
            if live is not None:
                att = att * live
            q = q + att

        if self.self_cand is not None:
            # ⛔ THE ONE PLACE CANDIDATES ARE ALLOWED TO SEE EACH OTHER.
            # See the module docstring: the generator stays candidate-
            # independent so `anchor_prefilter` (S2b) remains sound.
            pad, live = cand_pad, None
            if pad is not None:
                empty = pad.all(dim=1, keepdim=True)
                pad = pad & ~empty
                live = (~empty).to(q.dtype).unsqueeze(-1)
            c = self.norm_c(q)
            att = self.self_cand(c, c, c, key_padding_mask=pad,
                                 need_weights=False)[0]
            if live is not None:
                att = att * live
            q = q + att

        return q + self.mlp(self.norm_f(q))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SelectorConfig:
    """Every field that changes what the selector IS. Serialised whole into
    ``config.json['seams']['selector']`` — a run record that cannot rebuild its
    own model config is not a run record.
    """

    #: ⭐ DECISION (not a default): half of V2's 512/16. Thor-minded; the
    #: selector runs on the edge box beside the generator.
    d: int = 256
    n_heads: int = 8
    ff_mult: int = 4

    #: V2's shape: 1 coarse layer over the whole fan, then 3 fine layers over
    #: the survivors. Copied so the arm is a port rather than a redesign.
    coarse_layers: int = 1
    fine_layers: int = 3
    #: V2's top-32. ⚠️ ``top_k >= N`` disables the prune entirely and every
    #: candidate reaches the fine scorer — the exact-scoring control path.
    top_k: int = 32

    #: ⛔ The candidate self-attention. ON is the module's reason to exist; the
    #: OFF arm is the CONTROL that proves the coupling is real rather than
    #: numerical noise, and it is also the configuration under which this
    #: module would preserve the generator's independence argument.
    self_attn: bool = True

    #: E-AGT-1 agent tokens. ⭐ DEFAULT OFF and that is deliberate: the agent
    #: branch is its own lever (WP-6), and one variable per arm. When OFF the
    #: branch is NOT CONSTRUCTED, so RNG draw order is unchanged and passing
    #: agents to `forward` RAISES rather than silently ignoring them.
    cross_agent: bool = False

    #: sinusoidal bands per coordinate; 8 bands span 1x..128x the base
    #: frequency, i.e. ~0.5 m resolution at a 60 m scale.
    n_freq: int = 8
    pos_scale_m: float = POS_SCALE_M

    #: ⭐ THE TWO ABLATION SWITCHES (I2). Default 0.0 = the pure PDMS-shaped
    #: score with a denominator of exactly 12, so a selector built with
    #: ``SelectorConfig()`` composes the published structure and nothing else.
    #: ⛔ NO NON-ZERO DEFAULT IS SUPPLIED ON PURPOSE. A weight here decides how
    #: much a STRATEGIC and a TACTICAL head move the pick, and we have no
    #: measurement that fixes either value. Inventing one would be a number
    #: with no evidence class deciding a GPU-day. The arm that turns a head on
    #: states its weight in its own pre-registration and records it here.
    #: (For orientation only, NOT a recommendation: PDMS's smallest weight is
    #: comfort at 2.0.)
    w_compliance: float = 0.0
    w_tactical: float = 0.0

    #: V2's ``MarginRankingLoss(0.05)`` on progress pairs, applied twice
    #: ("x 2" in V2's recipe = two independently drawn pair sets per step).
    margin: float = 0.05
    margin_pairs: int = 2

    #: per-head BCE weights. ⚠️ These are LOSS weights and are NOT the score
    #: weights above — a head can be trained hard and contribute nothing to the
    #: pick (that is exactly the state of `compliance`/`tactical` at defaults).
    head_loss_w: float = 1.0
    margin_loss_w: float = 1.0

    def __post_init__(self) -> None:
        if self.d % self.n_heads != 0:
            raise ValueError(
                f"d={self.d} must be divisible by n_heads={self.n_heads}")
        if self.coarse_layers < 1 or self.fine_layers < 1:
            raise ValueError("coarse_layers and fine_layers must be >= 1")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.w_compliance < 0.0 or self.w_tactical < 0.0:
            raise ValueError(
                "a NEGATIVE composition weight inverts a head — the score "
                "would reward NON-compliance. Refused rather than clamped.")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["head_names"] = list(HEAD_NAMES)
        d["env_heads"] = sorted(ENV_HEADS)
        d["gt_derived_heads"] = sorted(GT_DERIVED_HEADS)
        d["pdms_weights"] = dict(PDMS_WEIGHTS)
        d["pdms_denom"] = PDMS_DENOM
        d["score_denominator"] = float(
            PDMS_DENOM + self.w_compliance + self.w_tactical)
        d["dac_head"] = "ABSENT — no map on PhysicalAI (pinned readset)"
        return d


# ---------------------------------------------------------------------------
# The composition — a free function so a test can hand-compute it
# ---------------------------------------------------------------------------
def compose_score(logits: dict[str, Tensor], w_compliance: float = 0.0,
                  w_tactical: float = 0.0) -> Tensor:
    """The §5.2 composition, on LOGITS (sigmoids applied here).

        sigma(NC) * (5*sig(TTC) + 5*sig(prog) + 2*sig(comf)
                     + w_c*sig(compl) + w_t*sig(tac)) / (12 + w_c + w_t)

    ⭐ Kept a free function of a plain dict so the test suite can feed it
    hand-chosen head values and compare against a number computed with a pocket
    calculator. A composition that only exists inside a forward pass can only
    be checked against itself.

    ⚠️ At ``w_c = 0`` the compliance term contributes exactly 0 AND carries no
    gradient from the score. The head is still trained by its own BCE — that is
    the point of the switch: the head exists and is calibrated, and the ablation
    is purely about whether it is allowed to move the pick.
    """
    missing = [h for h in HEAD_NAMES if h not in logits]
    if missing:
        raise KeyError(f"compose_score is missing heads: {missing}")
    nc = torch.sigmoid(logits["nc"])
    num = (PDMS_WEIGHTS["ttc"] * torch.sigmoid(logits["ttc"])
           + PDMS_WEIGHTS["progress"] * torch.sigmoid(logits["progress"])
           + PDMS_WEIGHTS["comfort"] * torch.sigmoid(logits["comfort"]))
    if w_compliance:
        num = num + w_compliance * torch.sigmoid(logits["compliance"])
    if w_tactical:
        num = num + w_tactical * torch.sigmoid(logits["tactical"])
    den = PDMS_DENOM + float(w_compliance) + float(w_tactical)
    return nc * num / den


def pdms_nc_binarise(nc: Tensor) -> Tensor:
    """V2's ``NC 0.5 -> 0`` mapping, kept for parity and honesty.

    In NAVSIM the NC sub-score is ``{0, 0.5, 1}`` (0.5 = an at-fault-less
    collision) and V2 collapses 0.5 to 0. **Our NC target is already binary**
    — we have no fault model on PhysicalAI — so this is a NO-OP on our targets
    and exists so a parity audit against V2 finds the mapping rather than its
    absence. Applying it is free; asserting the parity is not.
    """
    return (nc >= 1.0).to(nc.dtype)


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------
class SubMetricSelector(nn.Module):
    """Coarse (1 layer, whole fan) -> top-K -> fine (3 layers, survivors).

    ``forward`` returns a dict; the keys a caller is expected to consume:

    ==================== =========================================
    ``score``            ``[B, N]`` composed, :data:`NOT_SCORED` where pruned
    ``score_coarse``     ``[B, N]`` the coarse composition (the ranking used
                         for the prune — reported so the prune is auditable)
    ``coarse_logits``    ``{head: [B, N]}``
    ``fine_logits``      ``{head: [B, K]}``
    ``topk_idx``         ``[B, K]`` indices into the fan
    ``sel_idx_selector`` ``[B]`` the pick, named so it can sit beside
                         ``sel_idx`` / ``sel_idx_base`` in a dump
    ``n_candidates``     int, the fan size seen
    ``n_scored``         int, how many reached the fine scorer
    ==================== =========================================

    ⛔ ``n_candidates`` and ``n_scored`` are RETURNED, not logged. A selector
    that reports a loss without its n is the ``n << d`` failure waiting to
    happen (MEASURED 2026-08-22: a ridge probe with 2,050 features on ~700 rows
    correctly chose maximal regularisation and every arm read +0.0000 — an
    underpowered panel that looked exactly like an absence).
    """

    def __init__(self, scene_dim: int, n_steps: int,
                 cfg: SelectorConfig | None = None,
                 agent_dim: int | None = None):
        super().__init__()
        self.cfg = cfg = cfg or SelectorConfig()
        d = cfg.d
        self.embed = CandidateEmbed(d, n_steps, cfg.n_freq, cfg.pos_scale_m)
        self.scene_proj = (nn.Identity() if scene_dim == d
                           else nn.Linear(scene_dim, d))
        self.agent_proj: nn.Module | None = None
        if cfg.cross_agent:
            ad = int(agent_dim if agent_dim is not None else d)
            self.agent_proj = (nn.Identity() if ad == d
                               else nn.Linear(ad, d))

        mk = lambda: SelectorLayer(  # noqa: E731 — one expression, read once
            d, cfg.n_heads, cfg.ff_mult, self_attn=cfg.self_attn,
            cross_agent=cfg.cross_agent)
        self.coarse = nn.ModuleList([mk() for _ in range(cfg.coarse_layers)])
        self.fine = nn.ModuleList([mk() for _ in range(cfg.fine_layers)])

        # ⭐ SEPARATE head sets for coarse and fine, deliberately. V2 has two
        # scorers, and sharing them would make the coarse prune and the final
        # score the same object — the prune could then never be wrong in a way
        # the fine stage could correct, which is the whole reason the two
        # stages exist.
        self.coarse_heads = nn.ModuleDict(
            {h: nn.Linear(d, 1) for h in HEAD_NAMES})
        self.fine_heads = nn.ModuleDict(
            {h: nn.Linear(d, 1) for h in HEAD_NAMES})

    # -- internals ---------------------------------------------------------
    def _heads(self, mod: nn.ModuleDict, q: Tensor) -> dict[str, Tensor]:
        return {h: mod[h](q).squeeze(-1) for h in HEAD_NAMES}

    def _run(self, layers: nn.ModuleList, q: Tensor, scene: Tensor,
             agents: Tensor | None, agent_pad: Tensor | None,
             cand_pad: Tensor | None) -> Tensor:
        for lyr in layers:
            q = lyr(q, scene, agents=agents, agent_pad=agent_pad,
                    cand_pad=cand_pad)
        return q

    # -- forward -----------------------------------------------------------
    def forward(self, cand_traj: Tensor, scene_kv: Tensor,
                agents: Tensor | None = None,
                agent_pad: Tensor | None = None,
                cand_pad: Tensor | None = None) -> dict:
        """``cand_traj [B, N, S, 2]`` — the fan the decoder EMITTED.

        ⛔ Feed ``out["anchor_traj"]`` from ``AnchoredDiffusionDecoder.forward``
        (the emitted fan), never ``anchor_bank`` (the prior) and never the
        pre-refinement pass. Scoring a staler object than the one emitted is
        ``D-REFC-DDAUDIT-3``, which is the defect this module exists to close;
        reintroducing it here would be the same bug one layer out.

        ``scene_kv [B, M, scene_dim]`` — the conv map tokens, reused as KV.
        ``cand_pad [B, N]`` bool, **True where the candidate slot is PADDING**
        (a ragged fan under augmentation). Padded slots are excluded from the
        self-attention, from the top-K and from the pick.
        """
        if cand_traj.dim() != 4:
            raise ValueError(
                f"cand_traj must be [B, N, S, 2], got {tuple(cand_traj.shape)}")
        if agents is not None and self.agent_proj is None:
            # ⛔ A SILENT IGNORE IS THE DEAD-SEAM FAILURE. The branch was not
            # constructed (cfg.cross_agent=False), so these tokens would vanish
            # and the arm would report "agents made no difference".
            raise ValueError(
                "agents were supplied but SelectorConfig.cross_agent is False "
                "— the agent branch was never constructed. Set "
                "cross_agent=True or stop passing agents; a silent drop here "
                "reads exactly like a null result.")
        b, n = cand_traj.shape[0], cand_traj.shape[1]
        scene = self.scene_proj(scene_kv)
        ag = None if agents is None else self.agent_proj(agents)

        q0 = self.embed(cand_traj)
        qc = self._run(self.coarse, q0, scene, ag, agent_pad, cand_pad)
        clog = self._heads(self.coarse_heads, qc)
        cscore = compose_score(clog, self.cfg.w_compliance,
                               self.cfg.w_tactical)               # [B, N]
        if cand_pad is not None:
            cscore = cscore.masked_fill(cand_pad, NOT_SCORED)

        k = min(int(self.cfg.top_k), n)
        pruned = k < n
        if pruned:
            idx = torch.topk(cscore, k, dim=1, sorted=True).indices  # [B, k]
        else:
            # ⚠️ `.contiguous()` is load-bearing, not tidiness: an `expand`ed
            # view is a stride-0 index, and `scatter` on a stride-0 index is
            # the "overlapping writes" case whose behaviour is unspecified.
            idx = torch.arange(n, device=cand_traj.device)
            idx = idx.unsqueeze(0).expand(b, n).contiguous()

        gather = idx.unsqueeze(-1).expand(b, k, q0.shape[-1])
        qs = torch.gather(q0, 1, gather)
        pad_s = (None if cand_pad is None
                 else torch.gather(cand_pad, 1, idx))
        qf = self._run(self.fine, qs, scene, ag, agent_pad, pad_s)
        flog = self._heads(self.fine_heads, qf)
        fscore = compose_score(flog, self.cfg.w_compliance,
                               self.cfg.w_tactical)               # [B, k]
        if pad_s is not None:
            fscore = fscore.masked_fill(pad_s, NOT_SCORED)

        score = cand_traj.new_full((b, n), NOT_SCORED)
        score = score.scatter(1, idx, fscore)
        sel = score.argmax(dim=1)                                  # [B]

        n_live = (int((~cand_pad).sum()) if cand_pad is not None else b * n)
        return {
            "score": score,
            "score_coarse": cscore,
            "score_fine": fscore,
            "coarse_logits": clog,
            "fine_logits": flog,
            "topk_idx": idx,
            "sel_idx_selector": sel,
            "n_candidates": int(n),
            "n_scored": int(k),
            "n_live_candidates": n_live,
            "pruned": bool(pruned),
            "tele": {
                "top_k": int(k), "fan": int(n), "batch": int(b),
                "self_attn": bool(self.cfg.self_attn),
                "cross_agent": bool(self.cfg.cross_agent),
                "agents_fed": bool(agents is not None),
                "w_compliance": float(self.cfg.w_compliance),
                "w_tactical": float(self.cfg.w_tactical),
            },
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _bce(logit: Tensor, tgt: Tensor, mask: Tensor | None
         ) -> tuple[Tensor, int]:
    """BCE-with-logits against a possibly SOFT target, masked.

    Returns ``(loss, n_valid)``. ⛔ ``n_valid`` is returned rather than
    swallowed: a fully-masked head silently contributes 0.0, which is
    indistinguishable from a perfectly-fit head unless the n is reported.
    """
    if mask is None:
        mask = torch.ones_like(tgt, dtype=torch.bool)
    mask = mask & torch.isfinite(tgt)
    n = int(mask.sum())
    if n == 0:
        return logit.sum() * 0.0, 0
    per = F.binary_cross_entropy_with_logits(
        logit, tgt.clamp(0.0, 1.0).to(logit.dtype), reduction="none")
    return (per * mask.to(per.dtype)).sum() / n, n


def _margin_pairs(logit: Tensor, tgt: Tensor, mask: Tensor | None,
                  margin: float, n_draws: int,
                  generator: torch.Generator | None = None
                  ) -> tuple[Tensor, int]:
    """V2's ``MarginRankingLoss(0.05)`` on progress pairs, drawn ``n_draws``x.

    Pairs are drawn by a random permutation of the candidate axis, so each draw
    gives every candidate exactly one partner. Only pairs whose TARGETS
    genuinely differ contribute — a pair with equal targets carries no ordering
    to learn and would just add a constant hinge.

    ⚠️ This is a RANKING term on top of the per-head BCE, not a replacement for
    it. BCE calibrates the head (the selector's ECE is a reported criterion);
    the margin term is what makes the ORDER right, which is the only thing an
    argmax reads.
    """
    if mask is None:
        mask = torch.ones_like(tgt, dtype=torch.bool)
    mask = mask & torch.isfinite(tgt)
    b, n = logit.shape
    total = logit.sum() * 0.0
    used = 0
    for _ in range(max(int(n_draws), 0)):
        perm = torch.randperm(n, device=logit.device, generator=generator)
        j = perm.unsqueeze(0).expand(b, n)
        t_a, t_b = tgt, torch.gather(tgt, 1, j)
        m = mask & torch.gather(mask, 1, j) & (t_a != t_b)
        if not bool(m.any()):
            continue
        s_a, s_b = logit, torch.gather(logit, 1, j)
        sign = torch.where(t_a > t_b, 1.0, -1.0).to(logit.dtype)
        hinge = F.relu(margin - sign * (s_a - s_b))
        cnt = int(m.sum())
        total = total + (hinge * m.to(hinge.dtype)).sum() / cnt
        used += cnt
    return total, used


def selector_losses(out: dict, targets: dict[str, Tensor],
                    masks: dict[str, Tensor] | None = None,
                    cfg: SelectorConfig | None = None,
                    generator: torch.Generator | None = None) -> dict:
    """Per-head BCE (coarse AND fine) + the progress margin term.

    ``targets[head]`` is ``[B, N]`` over the WHOLE fan, in ``[0, 1]``; a
    ``NaN`` entry is treated as "undefined here" and dropped, which is how
    ``comfort`` on a stationary candidate and ``compliance`` on a
    non-informative window are handled (``CLAUDE.md``: say so per family with
    the reason and the n, rather than silently dropping it — hence every
    ``n_*`` key below).

    Returns a flat dict: ``loss`` (the scalar to backprop) plus every component
    and every n. ⛔ **The n keys are part of the contract**, not telemetry.
    """
    cfg = cfg or SelectorConfig()
    masks = masks or {}
    idx = out["topk_idx"]
    res: dict = {}
    total = None

    for h in HEAD_NAMES:
        if h not in targets:
            continue
        t = targets[h]
        m = masks.get(h)
        lc, nc_ = _bce(out["coarse_logits"][h], t, m)
        t_s = torch.gather(t, 1, idx)
        m_s = None if m is None else torch.gather(m, 1, idx)
        lf, nf_ = _bce(out["fine_logits"][h], t_s, m_s)
        res[f"loss_{h}_coarse"], res[f"n_{h}_coarse"] = lc, nc_
        res[f"loss_{h}_fine"], res[f"n_{h}_fine"] = lf, nf_
        term = cfg.head_loss_w * (lc + lf)
        total = term if total is None else total + term

    if "progress" in targets:
        t = targets["progress"]
        m = masks.get("progress")
        lm, nm = _margin_pairs(out["coarse_logits"]["progress"], t, m,
                               cfg.margin, cfg.margin_pairs, generator)
        t_s = torch.gather(t, 1, idx)
        m_s = None if m is None else torch.gather(m, 1, idx)
        lmf, nmf = _margin_pairs(out["fine_logits"]["progress"], t_s, m_s,
                                 cfg.margin, cfg.margin_pairs, generator)
        res["loss_margin_coarse"], res["n_margin_coarse"] = lm, nm
        res["loss_margin_fine"], res["n_margin_fine"] = lmf, nmf
        term = cfg.margin_loss_w * (lm + lmf)
        total = term if total is None else total + term

    if total is None:
        raise ValueError(
            "selector_losses was given no known head targets — "
            f"got {sorted(targets)}, expected any of {list(HEAD_NAMES)}")

    res["loss"] = total
    res["n_candidates"] = int(out["n_candidates"])
    res["n_scored"] = int(out["n_scored"])
    res["n_live_candidates"] = int(out["n_live_candidates"])
    res["heads_supervised"] = sorted(set(targets) & set(HEAD_NAMES))
    res["heads_gt_derived"] = sorted(set(targets) & GT_DERIVED_HEADS)
    return res
