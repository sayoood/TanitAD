"""refcv5 WP-6 / ``E-AGT-HEAD`` — the AGENT-TOKEN SEAM between the environment
head and the planner decoder, plus the monocular supervision that seam needs.

⭐ **The PI's named priority (2026-09-05):** *"Let's follow environment
extraction from the front camera and train it based on GT data."* This module is
the **seam**, not a second detector.

## ⛔ What this module deliberately does NOT contain, and why

Three complete components already exist in this repo. Re-implementing any of
them is how two geometries drift apart (the `advect` precedent, retired
2026-07-27), so every one of them is **imported**:

| already exists | file | what it gives |
|---|---|---|
| DETR slot decoder, exact Hungarian, DETR set loss, join→targets | ``tanitad.models.agent_slots`` | ``AgentSlotDecoder``, ``hungarian``, ``match_slots``, ``slot_set_loss``, ``targets_from_join`` |
| rig ⇄ canonical-cylindrical projection, both directions | ``tanitad.data.rig_projection`` | ``RigCamera``, ``project_rig_to_frame``, ``frame_to_cam_ray``, ``ground_intersection`` |
| the 10-class enum, the BEV grid, the FOV mask | ``tanitad.data.bev_raster`` | ``ALL_CLASSES``, ``GRID_DEFAULT``, ``fov_mask`` |

⚠️ **The class enum in particular is IMPORTED, never re-listed.** A first draft
of this file hand-wrote a plausible-looking 10-class tuple containing
``bicycle``, ``motorcycle`` and ``train_or_tram_car`` — **none of which exist in
the corpus**, which really carries ``other_vehicle``, ``stroller`` and
``animal``. It would have trained a head against three classes that can never
appear and silently dropped three that do. The enum has exactly one spelling,
in ``bev_raster``, and it is the label side's.

## What IS new here — the four gaps the audit named

1. :class:`AgentTokenEmbed` — turn decoded agent slots into the tokens the
   planner decoder cross-attends. **This seam did not exist**: ``AgentSlotDecoder``
   emits a set of boxes and nothing consumes them, and ``V6Config.agent_slots``
   is in ``LADDER_UNTRAINED_GROUPS`` (it trains in no stage).
2. :func:`monocular_projection_loss` — the **image-plane** term. ``slot_set_loss``
   is entirely BEV ``(cx, cy, l, w)``; a MONOCULAR head supervised only in BEV is
   asked to regress range with no term in the space it can actually see.
3. :func:`ground_range_prior` — the free metric anchor. Rig ``z = 0`` **is** the
   road plane (MEASURED over 87,481 cuboids: ground-standing bottom faces
   −0.05 to −0.13 m, ``protruding_object`` +1.68 m), so a pixel plus the plane
   already fixes range. No learned depth needed for the prior.
4. :func:`degrade_boxes` / :class:`OracleAgentEmbed` — the ``E-AGT-ORACLE`` and
   ``E-AGT-BUDGET`` rungs, which must run **before** any detector GPU-day.

## The binding rules this module is written against

| stage | what may be used | here |
|---|---|---|
| label derivation | ego, other agents, maps, future poses | ``obstacle.offline`` cuboids are **TRAIN-TIME TARGETS** |
| **inference** | ⛔ **VISION ONLY** | ``AgentSlotDecoder.forward`` takes ONE argument — the spatial memory. There is no parameter through which a label could arrive. |

⛔ An arm that reads the join at inference is **refused, not fixed**. The
oracle path below exists to price the ceiling and is marked inadmissible as a
capability claim wherever it appears.

⚠️ Frames are **256×640 CYLINDRICAL**, ``f_ref`` 305.577, HFOV **120°**. The
pinhole formula gives 92.6° here and looks entirely plausible. Nothing in this
module computes a pixel itself — every projection goes through
``rig_projection``, whose ``CanonicalFrame`` carries the projection so no call
site chooses a formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
from tanitad.data.rig_projection import RigCamera  # noqa: F401  (re-export)
from tanitad.models.agent_slots import (
    AGENT_CLASSES, N_QUERIES_DEFAULT, AgentSlotDecoder, SlotDecodeRanges,
    match_slots, slot_set_loss,
)

__all__ = [
    "AGENT_CLASSES", "N_AGENT_CLASSES", "AgentSeamConfig", "AgentTokenEmbed",
    "OracleAgentEmbed", "build_agent_head", "degrade_boxes",
    "monocular_projection_loss", "ground_range_prior", "slot_pad_mask",
    "filter_targets_to_visible", "visible_target_filter", "TOKEN_FEAT_DIM",
    "FOV_HALF_ANGLE_RAD", "RigCameraBank", "row_cameras",
]

#: The rig's own horizontal half-field. The camera is `camera_front_wide_120fov`
#: and the canonical frame is CYLINDRICAL, so the half-angle is exactly 60 deg.
#: ⚠️ Do NOT recompute this with the pinhole formula: it gives 92.6 deg total
#: here and looks entirely plausible.
FOV_HALF_ANGLE_RAD: float = math.radians(60.0)
_FOV_HALF = FOV_HALF_ANGLE_RAD

#: 10 — imported, never re-listed. See the module docstring's ⚠️.
N_AGENT_CLASSES = len(ALL_CLASSES)

#: per-slot geometry features handed to the token embedder:
#: (cx/x_range, cy/y_range, l, w, yaw_sin, yaw_cos, range/x_range,
#:  log1p(range)/5, cos bearing, sin bearing, 1/(1+range), v_rel_x, v_rel_y,
#:  yaw_rate_rel, presence)
TOKEN_FEAT_DIM = 15


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AgentSeamConfig:
    """WP-6 ``E-AGT-HEAD``. Every default is OFF — a build that does not ask
    for agent tokens constructs nothing and is bit-identical to refcv4b,
    including RNG draw order."""

    #: build the monocular head and feed its tokens to the decoder
    enable: bool = False
    #: ⛔ E-AGT-ORACLE / E-AGT-BUDGET: feed GROUND-TRUTH boxes instead of the
    #: head's. INADMISSIBLE as a capability claim — it reads a privileged label
    #: at inference. Prices the ceiling before any detector GPU-day is spent.
    oracle: bool = False
    #: E-AGT-BUDGET knobs: degrade the oracle until separation dies. The sigma
    #: that comes out IS the detector specification.
    oracle_sigma_range_m: float = 0.0
    oracle_miss_rate: float = 0.0

    #: ⭐ **100 — RULED BY THE MASTER MIND (M17, 2026-09-05), AND THE CRITERION IS
    #: THE DROP POLICY, NOT THE DISTRIBUTION.**
    #: The previous value 32 was fitted on **val40** and its own note said the
    #: train distribution was UNMEASURED. It has now been measured on the
    #: 2,308-episode train join (433,040 frames / 12,122,129 boxes):
    #:
    #:   corpus       mean   p99    max
    #:   val40        3.16    19     24     <- the basis for 32
    #:   train2400    4.39    30   **94**
    #:
    #: At N = 32 on train: **41,362 boxes (2.18 %) dropped across 3,250 frames,
    #: nearest sacrificed target at 13.1 m**.
    #: ⛔ ``match_slots`` keeps the **NEAREST** N, so **a drop is by construction
    #: the CLOSEST thing we failed to see** — this is not distant clutter being
    #: trimmed, it is the near field going unmodelled once a scene is dense.
    #: 13.1 m is inside the braking envelope at any urban speed, and **N = 64
    #: does not fix it either** (33.9 m, still inside comfortable braking at
    #: 15 m/s). A safety-relevant truncation is not a knob to trade against cost.
    #: ⚠️ **94 is a MAX OVER A SAMPLE, not a bound** — setting N to the observed
    #: maximum guarantees truncation on the first denser frame. 100 carries
    #: headroom and does not drift each time the corpus grows.
    #: ⭐ **100 is the ORDINARY setting for this architecture and 32 was the
    #: anomaly**: DETR uses ~100 queries against ~7 objects per image; ours
    #: averages 4.39. Surplus queries emitting "no object" is the design working.
    #: ⭐ **The cost is MEASURED, not assumed** (each arm run twice, 256x640,
    #: n_pad 94): step **3.346 -> 3.622 s = +8.2 %** with non-overlapping
    #: medians; params **+17,408 = 68 x 256 exactly = +0.077 %**; peak memory
    #: **no detectable difference** (analytic activation delta ~8.1 MiB against a
    #: ~500 MB step). ⚠️ Scope: dev-box CPU; a pod-side re-measure is owed.
    #: ⚠️ **The rule that survives from the old note**: a covering N with **ZERO
    #: DROP**, never "use the p99" — p99 leaves 1 % of frames dropping and that
    #: 1 % is exactly the crowded frames, which is the flattering-on-hard-frames
    #: failure the recipe exists to prevent.
    queries: int = 100
    d_model: int = 256
    depth: int = 3
    n_heads: int = 8
    #: the head's param band is enforced by AgentSlotDecoder (2-4 M). A toy
    #: rig may waive it; a real arm may NOT.
    enforce_band: bool = True

    #: the monocular image-plane auxiliary (gap #2). 0.0 turns it off, which
    #: is a BEV-only head and is a different arm — say which one you ran.
    w_project: float = 0.0
    #: ground-plane range prior (gap #3), weight on |range - plane range|.
    w_ground: float = 0.0
    #: gate the tokens by presence, so a slot the head does not believe in
    #: contributes nothing. Threshold is on sigmoid(presence_logit).
    presence_gate: float = 0.5
    #: ⛔ hard-mask slots below the threshold instead of soft-scaling. Soft is
    #: the default because a hard mask has zero gradient to the presence head
    #: through the planner loss.
    presence_hard: bool = False

    def as_dict(self) -> dict:
        """Serialised into ``config.json['seams']['agents']`` — a run record
        that cannot rebuild its own model config is not a run record."""
        return {
            "enable": bool(self.enable), "oracle": bool(self.oracle),
            "oracle_sigma_range_m": float(self.oracle_sigma_range_m),
            "oracle_miss_rate": float(self.oracle_miss_rate),
            "queries": int(self.queries), "d_model": int(self.d_model),
            "depth": int(self.depth), "n_heads": int(self.n_heads),
            "enforce_band": bool(self.enforce_band),
            "w_project": float(self.w_project),
            "w_ground": float(self.w_ground),
            "presence_gate": float(self.presence_gate),
            "presence_hard": bool(self.presence_hard),
            "n_classes": int(N_AGENT_CLASSES),
            "classes": list(AGENT_CLASSES),
            "n_queries_default_upstream": int(N_QUERIES_DEFAULT),
        }


def build_agent_head(cfg: AgentSeamConfig, d_memory: int, n_memory: int
                     ) -> AgentSlotDecoder:
    """The head, built from the EXISTING ``AgentSlotDecoder``.

    ``n_memory`` is the conv map's token count (``gh * gw``) — the decoder's
    positional table is per-token, so this is a geometry, not a resize.
    """
    return AgentSlotDecoder(
        d_memory=int(d_memory), n_memory=int(n_memory),
        n_queries=int(cfg.queries), d_model=int(cfg.d_model),
        depth=int(cfg.depth), n_heads=int(cfg.n_heads),
        enforce_band=bool(cfg.enforce_band))


# ---------------------------------------------------------------------------
# The seam that did not exist: slots -> planner tokens
# ---------------------------------------------------------------------------
def slot_features(box: Tensor, yaw_vec: Tensor, rates: Tensor,
                  presence: Tensor,
                  ranges: SlotDecodeRanges | None = None) -> Tensor:
    """``-> [B, N, TOKEN_FEAT_DIM]`` — the geometry a PLANNER token must carry.

    ⭐ Why explicit geometry and not just the query feature: the decoder acts on
    a fan of trajectories **in metres**, and every LONGITUDINAL metric the
    programme is bound to report (headway, time-gap, TTC) is a function of
    **range and bearing**. A token that hides those inside a learned embedding
    makes cross-attention re-derive them from scratch.

    ``box`` ``[B, N, 4]`` = (cx, cy, l, w) in METRES (``AgentSlotDecoder``'s
    decode) · ``yaw_vec`` ``[B, N, 2]`` unit (sin, cos) · ``rates``
    ``[B, N, 3]`` = (v_rel_x, v_rel_y, yaw_rate_rel) · ``presence``
    ``[B, N]`` in ``[0, 1]``.
    """
    r = ranges or SlotDecodeRanges()
    cx, cy = box[..., 0], box[..., 1]
    rng = torch.sqrt(cx * cx + cy * cy).clamp_min(1e-3)
    return torch.stack([
        cx / r.x_fwd_m, cy / r.y_half_m,
        box[..., 2], box[..., 3],
        yaw_vec[..., 0], yaw_vec[..., 1],
        rng / r.x_fwd_m,
        torch.log1p(rng) / 5.0,
        cx / rng, cy / rng,                 # bearing (cos, sin)
        1.0 / (1.0 + rng),
        rates[..., 0], rates[..., 1], rates[..., 2],
        presence,
    ], dim=-1)


class AgentTokenEmbed(nn.Module):
    """Decoded agent slots -> the ``[B, N, d_dec]`` tokens the planner decoder
    cross-attends, plus the ``[B, N]`` padding mask it needs.

    ⛔ **The padding convention is torch's**: ``pad[b, n] == True`` means slot
    ``n`` is PADDING and must NOT be attended. Inverting it silently attends to
    nothing and reads exactly like a dead seam.
    """

    def __init__(self, d_out: int, cfg: AgentSeamConfig,
                 ranges: SlotDecodeRanges | None = None):
        super().__init__()
        self.cfg = cfg
        self.ranges = ranges or SlotDecodeRanges()
        d = int(cfg.d_model)
        self.geo = nn.Sequential(
            nn.Linear(TOKEN_FEAT_DIM + N_AGENT_CLASSES, d), nn.GELU(),
            nn.Linear(d, d_out))
        self.norm = nn.LayerNorm(d_out)

    def forward(self, slots: dict) -> tuple[Tensor, Tensor]:
        """``slots`` is ``AgentSlotDecoder.forward``'s output (or the oracle's).

        Returns ``(tokens [B, N, d_out], pad [B, N] bool)``.
        """
        presence = torch.sigmoid(slots["presence_logit"])          # [B, N]
        cls_p = torch.softmax(slots["cls_logits"], dim=-1)         # [B, N, C]
        feats = slot_features(slots["box"], slots["yaw_vec"], slots["rates"],
                              presence, self.ranges)
        tok = self.norm(self.geo(torch.cat([feats, cls_p], dim=-1)))
        thr = float(self.cfg.presence_gate)
        if self.cfg.presence_hard:
            pad = presence < thr
            # a row with NOTHING above threshold is handled by CrossAttnLayer,
            # which un-masks fully-padded rows and zeroes their contribution.
            return tok, pad
        # SOFT (default): scale by presence so the presence head receives
        # gradient THROUGH the planner loss, and mark only structurally absent
        # slots as padding.
        tok = tok * presence.unsqueeze(-1).to(tok.dtype)
        return tok, slot_pad_mask(slots)


def filter_targets_to_visible(tgt: dict, half_angle_rad: float = _FOV_HALF,
                              x_max_m: float | None = None,
                              y_half_m: float | None = None,
                              x_min_m: float = 0.0) -> dict:
    """⛔⛔ **MANDATORY BEFORE ``match_slots`` FOR ANY CAMERA-ONLY HEAD.**
    Narrow ``tgt["valid"]`` to the agents a FRONT CAMERA could actually see.

    ⭐ **This function exists because of a MEASURED defect, not a worry.**
    ``agent_slots.targets_from_join`` sets ``valid = True`` for **every** agent
    in the join record — no azimuth filter, no range filter — and ``match_slots``
    then keeps the ``n_queries`` **NEAREST** of that set. MEASURED on the val40
    join (195,805 boxes / 7,400 frames / 39 clips,
    ``raw/agent_density.json``), at ``n_queries = 16``: of the 82,247 targets
    kept, **50,816 (61.8 %) are ``occ == 1`` — OUTSIDE the 120° field** — and
    **65,884 (80.1 %) fall outside the decode box**. The nearest agents include
    cars **BEHIND the ego**.

    ⛔ **Raising ``n_queries`` makes this WORSE, not better** (61.8 % → 63.1 %
    at N = 32). Without this filter a monocular head is trained to hallucinate
    on ~62 % of its supervision, and the resulting AP would be a measure of how
    well it guesses at the unobservable.

    ``half_angle_rad`` defaults to the rig's own 120° (±60°) — the SAME
    predicate ``bev_raster.fov_mask`` applies, and the same one the join's
    ``occ`` flag records. ⚠️ It is **necessary, not sufficient**: it is
    horizontal only, with no vertical, hood, or inter-agent occlusion.

    ⚠️ **The encoder's field may be NARROWER than the sensor's** (a centred
    sub-frame of the v5f run measured **117°**), so this is an UPPER BOUND on
    what the model can see. Pass the encoder's own half-angle when it differs.

    Returns a NEW dict sharing every tensor except ``valid``; the caller's
    target dict is not mutated.
    """
    box = tgt["box"]
    cx, cy = box[..., 0], box[..., 1]
    keep = tgt["valid"] & (torch.atan2(cy.abs(), cx) <= float(half_angle_rad))
    keep = keep & (cx >= float(x_min_m))
    if x_max_m is not None:
        keep = keep & (cx <= float(x_max_m))
    if y_half_m is not None:
        keep = keep & (cy.abs() <= float(y_half_m))
    out = dict(tgt)
    out["valid"] = keep
    return out


def visible_target_filter(tgt: dict, ranges: SlotDecodeRanges | None = None,
                          half_angle_rad: float = _FOV_HALF) -> dict:
    """:func:`filter_targets_to_visible` with the DECODE BOX as the range cut.

    ⭐ The decode box is the honest bound: ``SlotDecodeRanges`` is what the head
    can EXPRESS (``cx`` in ``[0, 60]``, ``|cy| <= 16``), so a target outside it
    is one the architecture cannot represent at all — training against it adds
    a loss the head can only reduce by being wrong somewhere it can reach.

    MEASURED on that set (in-field ∩ decode box): mean **3.16** agents/frame,
    p99 **19**, **max 24** on VAL40 — but the train corpus reads mean **4.39**,
    p99 **30**, **max 94**, which is why ``AgentSeamConfig.queries`` is **100**
    (M17) and why **32 is REFUTED on train**: it drops 41,362 boxes (2.18 %)
    across 3,250 frames with the nearest sacrificed target at **13.1 m**,
    inside the braking envelope. ``N_QUERIES_DEFAULT = 16`` remains REFUTED.
    ⚠️ A count measured on val40 is not a bound on train — that is the whole
    lesson of this correction.
    """
    r = ranges or SlotDecodeRanges()
    return filter_targets_to_visible(tgt, half_angle_rad=half_angle_rad,
                                     x_max_m=r.x_fwd_m, y_half_m=r.y_half_m)


def slot_pad_mask(slots: dict) -> Tensor:
    """``[B, N]`` bool, True where the slot is structurally absent.

    For a LEARNED head every slot exists (presence is soft), so this is
    all-False; for the ORACLE path it is ``~valid``, because a padded GT row is
    genuinely not an agent.
    """
    v = slots.get("valid")
    if v is None:
        return torch.zeros(slots["box"].shape[:2], dtype=torch.bool,
                           device=slots["box"].device)
    return ~v


# ---------------------------------------------------------------------------
# E-AGT-ORACLE / E-AGT-BUDGET
# ---------------------------------------------------------------------------
def degrade_boxes(box: Tensor, valid: Tensor, sigma_range_m: float = 0.0,
                  miss_rate: float = 0.0,
                  generator: torch.Generator | None = None
                  ) -> tuple[Tensor, Tensor]:
    """``E-AGT-BUDGET`` — degrade ORACLE boxes to a stated accuracy budget.

    Noise is applied **along the RANGE direction only**. That is a modelling
    choice with a reason: on a 120° cylindrical frame the image COLUMN fixes
    bearing almost exactly (``col = (W-1)/2 + f_ref·azimuth``, linear), while
    range is what a monocular detector actually gets wrong. Noising both axes
    isotropically would price a detector nobody would build.

    ``box`` ``[B, N, 4]`` = (cx, cy, l, w) metres · ``valid`` ``[B, N]`` bool.

    ⭐ **The control that must read a known value:** σ = 0 and miss = 0 return
    the inputs **unchanged objects**, so the ORACLE arm is exactly the
    zero-degradation point of the BUDGET sweep and not a separate code path.
    """
    if sigma_range_m <= 0.0 and miss_rate <= 0.0:
        return box, valid
    out = box.clone()
    if sigma_range_m > 0.0:
        cx, cy = out[..., 0], out[..., 1]
        rng = torch.sqrt(cx * cx + cy * cy).clamp_min(1e-3)
        eps = torch.randn(rng.shape, device=box.device, dtype=box.dtype,
                          generator=generator) * float(sigma_range_m)
        scale = (rng + eps).clamp_min(0.1) / rng
        out[..., 0] = cx * scale
        out[..., 1] = cy * scale
    v = valid
    if miss_rate > 0.0:
        keep = torch.rand(valid.shape, device=valid.device,
                          generator=generator) >= float(miss_rate)
        v = valid & keep
    return out, v


class OracleAgentEmbed(nn.Module):
    """``E-AGT-ORACLE`` / ``E-AGT-BUDGET`` — tokens from GT boxes, no detector.

    ⛔ **INADMISSIBLE as a capability claim.** It reads a privileged label at
    inference. Its ONLY job is to answer *"if the detector were perfect, would
    the planner move at all?"* before any GPU-day is spent training one — and
    per the ladder, if it does not separate on **LONGITUDINAL and TACTICAL**,
    the whole mechanism is refused for zero further GPU-days.

    Emits the SAME dict shape ``AgentSlotDecoder`` does, so
    :class:`AgentTokenEmbed` consumes either without a branch.
    """

    def __init__(self, cfg: AgentSeamConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, box: Tensor, yaw: Tensor, cls: Tensor, valid: Tensor,
                rates: Tensor | None = None,
                generator: torch.Generator | None = None) -> dict:
        """``box [B,N,4]`` (cx, cy, l, w) m · ``yaw [B,N]`` rad ·
        ``cls [B,N]`` long (-1 = unknown) · ``valid [B,N]`` bool."""
        c = self.cfg
        box, valid = degrade_boxes(box, valid, c.oracle_sigma_range_m,
                                   c.oracle_miss_rate, generator)
        yaw_vec = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
        # a GT box is present with certainty; a padded one with none.
        big = box.new_tensor(20.0)
        presence_logit = torch.where(valid, big, -big)
        onehot = F.one_hot(cls.clamp_min(0), N_AGENT_CLASSES).to(box.dtype)
        # unknown class (-1) -> uniform, never class 0 (which would invent
        # "automobile" for every unlabelled row).
        unk = (cls < 0).unsqueeze(-1)
        onehot = torch.where(unk, torch.full_like(onehot,
                                                  1.0 / N_AGENT_CLASSES),
                             onehot)
        cls_logits = torch.log(onehot.clamp_min(1e-9))
        if rates is None:
            rates = box.new_zeros(*box.shape[:2], 3)
        return {"box": box, "yaw_vec": yaw_vec, "yaw": yaw,
                "cls_logits": cls_logits, "presence_logit": presence_logit,
                "rates": rates, "valid": valid,
                "occ_logit": box.new_zeros(box.shape[:2])}


# ---------------------------------------------------------------------------
# PER-CLIP CAMERAS -- the corpus has two rigs and a per-CLIP mount pose
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RigCameraBank:
    """episode id -> that clip's OWN :class:`RigCamera`, plus an explicit
    fallback for episodes the table does not cover.

    ⛔⛔ **WHY A BANK AND NOT A CONSTANT**, with each figure's evidence
    class stated because they come from three different samples:

    * **MEASURED (ours, 2026-09-05)** on the repo's own banked render table
      ``taniteval/results/videos/refcv3_five_panel_step{30000,40284}/
      extrinsics_used.json`` -- **3 clips**, mount height **1.2922 / 1.2968 /
      1.5758 m**, forward offset **1.9964-2.1286 m**. ⭐ Small, but decisive
      against a constant: it STRADDLES the shipped guard band ``(1.43, 1.56)``
      at **both** ends.
    * **INHERITED** (``taniteval/tools/pai_extrinsics_table.py``'s docstring,
      citing ``.../pod-rescue-20260802/pod3/workspace/idm3_geom.py``, NOT
      re-verified by this stream): over **40 clips** the height spans
      **1.245-1.607 m**, median 1.306, **37 distinct values in 40**, CV 7.4 %;
      pitch -1.15..+2.34 deg; and **the rig label is not a proxy** -- rig-A vs
      rig-B medians differ 1.5 % while the WITHIN-rig spread is 29 %.
    * **UNMEASURED**: the 2,308-clip train parity corpus. Every band so far
      strictly contained the previous one, so 1.245-1.607 is not a bound.

    ⭐ **MEASURED 2026-09-05 on the PARITY CORPUS ITSELF** (all 2,400
    clips, ``stack/scripts/build_rig_extrinsics_table.py``, clip set
    verified against ``parity_manifest.json``'s
    ``clip_id_sha256_sorted``): mount height **1.2131-1.6672 m**, median
    **1.2993**, **554 DISTINCT VALUES in 2,400 clips**; forward offset
    1.6969-2.1635 m; pitch -2.440..+3.945 deg. This widened BOTH earlier
    bands at BOTH ends, exactly as their monotone growth predicted.

    ⚠️ Three camera-height constants circulate in this repo
    (1.22 / 1.43 / 1.5 m) and **all three are wrong as a constant**. But
    the sharper claim *"1.22 m is below every observed minimum"* held
    only on 40 clips and is **RETRACTED** on parity, whose minimum is
    1.2131 m -- 1.22 is inside the range, and still not a constant. See
    :data:`tanitad.data.rig_projection.CAM_HEIGHT_SAMPLES`.

    ⭐ **Why that binds HERE and not merely in a renderer.**
    :func:`ground_range_prior` back-projects a pixel through the road plane,
    so the range it supervises is directly proportional to the mount height.
    A 1.5 m constant applied to a 1.245 m clip biases every range it teaches
    by **+20 %**, and the head learns that bias as geometry.
    :func:`monocular_projection_loss` is a DIFFERENCE of two projections and
    cancels a common mount error to first order; the ground term does not.
    That asymmetry is exactly why the camera must be per clip, not per run.

    ⛔ A bank is **not** resolved inside the loss: the loss has no episode
    ids, and inventing a lookup there is how a camera silently attaches to the
    wrong clip. The caller resolves it (:meth:`for_episodes`) and passes a
    per-row list.
    """

    #: ``int(episode_id) -> RigCamera``. The key is whatever id the BATCH
    #: carries; ``refc_v3_train`` emits ``stable_episode_id(clip_id)``.
    by_episode: dict
    #: used for an episode the table does not cover. ``None`` means the row
    #: gets NO camera and the monocular terms skip it -- legal only because
    #: both terms COUNT the skipped rows and the trainer refuses an uncovered
    #: corpus unless the gap is accepted by name.
    default: "RigCamera | None" = None

    def __len__(self) -> int:
        return len(self.by_episode)

    def get(self, ep_id) -> "RigCamera | None":
        return self.by_episode.get(int(ep_id), self.default)

    def for_episodes(self, ep_ids) -> list:
        """``[B]`` episode ids (tensor / list) -> ``[B]`` cameras.

        ⭐ The returned list is exactly what :func:`agent_losses` takes, and
        its length IS the batch size, so a mismatch raises rather than
        misaligning rows.
        """
        if torch.is_tensor(ep_ids):
            ep_ids = ep_ids.detach().reshape(-1).tolist()
        return [self.get(e) for e in list(ep_ids)]

    def coverage(self, ep_ids) -> dict:
        """``{n, n_covered, frac, n_missing, missing_sample, has_default}``.

        A run record must state this: a bank covering 60 % of the corpus
        trains the monocular terms on 60 % of it while ``config.json`` says
        the camera source was ``extrinsics``.
        """
        if torch.is_tensor(ep_ids):
            ep_ids = ep_ids.detach().reshape(-1).tolist()
        ids = [int(e) for e in list(ep_ids)]
        miss = [e for e in ids if e not in self.by_episode]
        n = len(ids)
        return {"n": n, "n_covered": n - len(miss),
                "frac": ((n - len(miss)) / n) if n else 0.0,
                "n_missing": len(miss),
                "missing_sample": sorted(set(miss))[:8],
                "has_default": self.default is not None}


def row_cameras(cam, batch_size: int) -> list:
    """Normalise the ``cam`` argument to EXACTLY one entry per batch row.

    Accepts ``None`` (no camera at all), ONE :class:`RigCamera` (the whole
    batch shares a mount pose -- the pre-2026-09-05 behaviour, kept so every
    banked arm is untouched), or a sequence of length ``batch_size`` whose
    entries are ``RigCamera`` or ``None``.

    ⛔ A partial list is REFUSED rather than padded. Padding would attach
    clip *k*'s mount pose to row *k+1*, a label corruption no downstream
    metric could attribute.
    """
    if cam is None:
        return [None] * int(batch_size)
    if isinstance(cam, RigCamera):
        return [cam] * int(batch_size)
    if isinstance(cam, RigCameraBank):
        raise TypeError(
            "a RigCameraBank must be resolved BEFORE the loss: call "
            "bank.for_episodes(batch['agent_ep']). The loss has no episode "
            "ids and must not invent a lookup.")
    cams = list(cam)
    if len(cams) != int(batch_size):
        raise ValueError(
            "cam carries %d entries for a batch of %d. Pass ONE camera for "
            "the whole batch or exactly one per row -- a partial list would "
            "attach one clip's mount pose to another clip's row."
            % (len(cams), int(batch_size)))
    bad = [i for i, c in enumerate(cams)
           if c is not None and not isinstance(c, RigCamera)]
    if bad:
        raise TypeError("cam[%d] is a %s, not a RigCamera or None"
                        % (bad[0], type(cams[bad[0]]).__name__))
    return cams


# ---------------------------------------------------------------------------
# Gap #2 — the IMAGE-PLANE term a monocular head needs
# ---------------------------------------------------------------------------
def monocular_projection_loss(pred_box: Tensor, tgt_box: Tensor,
                              cam, match: dict,
                              z_center_m: float = 0.75) -> dict:
    """L1 between the PROJECTED pixel centres of matched predicted and target
    boxes, in the canonical cylindrical frame.

    ⭐ **Why this term exists.** ``slot_set_loss``'s box terms are entirely BEV
    metres. A MONOCULAR head has no direct observation of range — it sees a
    column (bearing, near-exact) and a row (which, given the road plane, IS the
    range). Supervising only in BEV asks it to regress the hard axis with no
    term in the space it can actually see, and lets a 2 m range error at 50 m
    cost the same as at 5 m although the image evidence differs by an order of
    magnitude. The projection term is naturally range-weighted: it is large
    where the image says the head is wrong.

    ⛔ **Everything geometric goes through ``rig_projection``.** The pinhole FOV
    formula is WRONG on this projection and nothing here re-derives a pixel.

    ``z_center_m`` lifts the BEV centre to a box CENTRE height above the road
    plane, because rig ``z = 0`` is the road (MEASURED) and a box centred on the
    road would project below every real detection. 0.75 m ≈ half a car's height;
    it is a declared constant, not a fitted one, and it cancels to first order
    in the DIFFERENCE of two projections at similar range.

    Returns ``{"loss", "n"}`` — ``n`` is the matched count the loss was
    computed over, RETURNED so a panel can print its own ``n``.
    """
    device = pred_box.device
    cams = row_cameras(cam, int(pred_box.shape[0]))
    losses: list[Tensor] = []
    n = 0
    n_rows = n_rows_no_cam = 0
    for b, (rows, cols) in enumerate(zip(match["rows"], match["cols"])):
        if rows.numel() == 0:
            continue
        n_rows += 1
        cam_b = cams[b]
        # ⛔ A row whose clip has no declared camera is COUNTED, never
        # silently dropped: a term that quietly skips rows while its
        # weight is stamped into config.json is the dead-flag class
        # this module exists to refuse (M18).
        if cam_b is None:
            n_rows_no_cam += 1
            continue
        pb = pred_box[b, rows.to(device)]                  # [M, 4]
        tb = tgt_box[b, cols.to(device)]                   # [M, 4]
        z = pb.new_full((pb.shape[0], 1), float(z_center_m))
        p3 = torch.cat([pb[:, :1], pb[:, 1:2], z], dim=-1)
        t3 = torch.cat([tb[:, :1], tb[:, 1:2], z], dim=-1)
        pc, pr, pv = cam_b.project(p3.to(cam_b.t_cam_in_rig.dtype))
        tc, tr, tv = cam_b.project(t3.to(cam_b.t_cam_in_rig.dtype))
        keep = pv & tv
        if not bool(keep.any()):
            continue
        # normalise by the frame so column and row errors are commensurate and
        # the term is scale-free across the 256x640 / 176x624 frames.
        w = float(cam_b.frame.width)
        h = float(cam_b.frame.height)
        d = ((pc - tc).abs() / w + (pr - tr).abs() / h)[keep]
        losses.append(d.to(pred_box.dtype).mean())
        n += int(keep.sum())
    if not losses:
        return {"loss": pred_box.new_zeros(()), "n": 0,
                "n_rows": n_rows, "n_rows_no_cam": n_rows_no_cam}
    return {"loss": torch.stack(losses).mean(), "n": n,
            "n_rows": n_rows, "n_rows_no_cam": n_rows_no_cam}


# ---------------------------------------------------------------------------
# Gap #3 — the free metric anchor
# ---------------------------------------------------------------------------
def _ground_terms(pred_box: Tensor, cam: RigCamera):
    """``(d, ok)`` UN-REDUCED for ONE camera and any leading shape.

    Split out of :func:`ground_range_prior` so the per-clip path
    reuses the SAME arithmetic rather than a second spelling of it --
    two implementations of one geometry is how they drift apart
    (the ``advect`` precedent, retired 2026-07-27).
    """
    dt = cam.t_cam_in_rig.dtype
    cx, cy = pred_box[..., 0], pred_box[..., 1]
    foot = torch.stack([cx, cy, torch.zeros_like(cx)], dim=-1).to(dt)
    col, row, valid = cam.project(foot)
    p_rig, hits = cam.ground_intersection(col, row)
    ok = valid & hits
    r_pred = torch.sqrt(cx * cx + cy * cy).to(dt)
    r_back = torch.linalg.vector_norm(p_rig[..., :2], dim=-1)
    d = (r_pred - r_back).abs() / r_pred.clamp_min(1.0)
    return d, ok


def ground_range_prior(pred_box: Tensor, cam,
                       z_center_m: float = 0.75) -> dict:
    """Consistency between a slot's predicted range and the range its own
    projected FOOT point implies under the road plane.

    ⭐ Rig ``z = 0`` **is** the road plane — MEASURED over 87,481 cuboids, not
    assumed (ground-standing classes' bottom faces read −0.05 to −0.13 m;
    ``protruding_object``, which by definition does not stand on the ground,
    reads +1.68 m). So a pixel plus the plane already fixes metric range and the
    monocular scale ambiguity does not apply to us. This term says: *project the
    box's foot to a pixel, back-project that pixel to the road, and the range
    you get back must be the range you predicted.*

    It costs no label at all — it is a **self-consistency** term, so it may run
    on windows the join does not cover (the ~NO_LABEL frames past ~20 s, which
    are a large share of every clip).

    Returns ``{"loss", "n", "frac_hit"}``.
    """
    n_rows_no_cam = 0
    if cam is None or isinstance(cam, RigCamera):
        # ⭐ The ONE-camera path is byte-for-byte the arithmetic every
        # banked arm ran: one vectorised call over [B, N]. It is kept
        # as its own branch so adding per-clip cameras cannot perturb
        # an arm that did not ask for them.
        if cam is None:
            return {"loss": pred_box.new_zeros(()), "n": 0,
                    "frac_hit": 0.0,
                    "n_rows_no_cam": int(pred_box.shape[0])}
        d_all, ok = _ground_terms(pred_box, cam)
        d_all, ok = d_all.reshape(-1), ok.reshape(-1)
    else:
        cams = row_cameras(cam, int(pred_box.shape[0]))
        ds, oks = [], []
        for b, cam_b in enumerate(cams):
            if cam_b is None:
                # ⛔ COUNTED, never silently skipped -- see
                # `monocular_projection_loss`.
                n_rows_no_cam += 1
                continue
            d_b, ok_b = _ground_terms(pred_box[b], cam_b)
            ds.append(d_b.reshape(-1))
            oks.append(ok_b.reshape(-1))
        if not ds:
            return {"loss": pred_box.new_zeros(()), "n": 0,
                    "frac_hit": 0.0, "n_rows_no_cam": n_rows_no_cam}
        d_all, ok = torch.cat(ds), torch.cat(oks)
    if not bool(ok.any()):
        return {"loss": pred_box.new_zeros(()), "n": 0,
                "frac_hit": 0.0, "n_rows_no_cam": n_rows_no_cam}
    d = d_all[ok]
    return {"loss": d.to(pred_box.dtype).mean(), "n": int(ok.sum()),
            "frac_hit": float(ok.to(torch.float32).mean()),
            "n_rows_no_cam": n_rows_no_cam}


# ---------------------------------------------------------------------------
# The composed training loss
# ---------------------------------------------------------------------------
def agent_losses(slots: dict, tgt: dict, cfg: AgentSeamConfig,
                 cam=None,
                 weights: dict | None = None,
                 filter_visible: bool = True) -> dict:
    """``slot_set_loss`` + the two monocular terms, per term, with their ``n``.

    ⛔ Per-term, never pooled into one score — the four-families discipline's
    sibling rule: a composite hides exactly the trade-off one wants to see.

    ``cam`` is ``None``, ONE :class:`RigCamera` for the whole batch, or a
    **per-row sequence** of length ``B`` (resolve a :class:`RigCameraBank`
    with :meth:`RigCameraBank.for_episodes` first). ⛔⛔ **The corpus has a
    per-CLIP mount pose** -- height 1.245-1.607 m over 40 clips, 37 distinct
    values, and the rig label explains almost none of it -- so one camera per
    RUN is a measured misconfiguration, not a simplification. The output
    carries ``cam_scope`` and ``n["rows_no_cam_*"]`` so a run record states
    which of the three it actually used and how many rows went unsupervised.
    """
    # ⛔ THE FILTER IS ON BY DEFAULT AND THAT IS THE POINT. Without it 61.8 %
    # of the targets `match_slots` keeps are outside the camera's field
    # (MEASURED, val40 join) and the head is trained to hallucinate. Turning it
    # OFF is the deliberate-regression arm, and it must be asked for by name.
    n_before = int(tgt["valid"].sum())
    if filter_visible:
        tgt = visible_target_filter(tgt)
    n_after = int(tgt["valid"].sum())
    match = match_slots(slots, tgt)
    out = slot_set_loss(slots, tgt, match=match, weights=weights)
    total = out["total"]
    _cams = row_cameras(cam, int(slots["box"].shape[0]))
    _n_cam = sum(1 for c in _cams if c is not None)
    out["cam_scope"] = ("none" if cam is None else
                        ("single" if isinstance(cam, RigCamera)
                         else "per-row"))
    out["n"]["rows_with_cam"] = int(_n_cam)
    out["n"]["rows_no_cam"] = int(len(_cams) - _n_cam)
    if _n_cam and cfg.w_project > 0.0:
        pj = monocular_projection_loss(slots["box"], tgt["box"], cam, match)
        out["loss_project"] = pj["loss"]
        out["n"]["project"] = pj["n"]
        out["n"]["rows_no_cam_project"] = int(pj.get("n_rows_no_cam", 0))
        total = total + cfg.w_project * pj["loss"]
    if _n_cam and cfg.w_ground > 0.0:
        gp = ground_range_prior(slots["box"], cam)
        out["loss_ground"] = gp["loss"]
        out["n"]["ground"] = gp["n"]
        out["n"]["rows_no_cam_ground"] = int(gp.get("n_rows_no_cam", 0))
        out["ground_frac_hit"] = gp["frac_hit"]
        total = total + cfg.w_ground * gp["loss"]
    out["total"] = total
    out["n_dropped"] = int(sum(match["n_dropped"]))
    out["n_target"] = int(sum(match["n_target"]))
    # ⭐ Both counts, always: a panel that reports only the post-filter n cannot
    # say how much supervision the filter removed, and that fraction is the
    # whole finding.
    out["n"]["target_prefilter"] = n_before
    out["n"]["target_visible"] = n_after
    out["filter_visible"] = bool(filter_visible)
    return out
