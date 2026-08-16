"""F-18 — the PERCEPTION agent-slot decoder: a DETR-style set head on frozen
spatial tokens.

⛔ WHY THIS FILE EXISTS. ``DIAGRAM_CONFORMANCE.md`` (2026-08-16) audited the
binding v6 diagram element by element. §4.2's first interpretation-head row —

    perception agent slots (bbox cx,cy,yaw,l,w · state v,yaw-rate,occluded ·
    class & size)                                             ⬜ NOT BUILT

was one of eight ⬜ cells, with the status line *"NEW — design here; DETR-style
slot decoder ~2–4 M params on spatial tokens"* and the note that **the LABEL
side already exists** (``scripts/build_obstacle_join.py``). This module is the
model side of that row (fix F-18). It is the LAST unbuilt PERCEPTION cell.

──────────────────────────────────────────────────────────────────────────────
⛔ THE THREE BINDING RULES THIS MODULE IS BUILT AROUND
──────────────────────────────────────────────────────────────────────────────

1. **VISION-ONLY AT INFERENCE** (PI, 2026-08-03, verbatim: *"for ground truth
   data of scenario classification you can use both ego and other label, for
   inference only vision"*). :meth:`AgentSlotDecoder.forward` takes **exactly
   one tensor** — the spatial memory computed from the camera frames — and
   **nothing else**. No ``v0``, no actions, no goal embedding, no pose, no
   situation channel; there is no keyword argument through which one could
   arrive. The signature IS the audit.
   ⚠️ The labels may and DO use privilege: ``obstacle.offline`` cuboids are
   composed through **egomotion** to reach the ego frame
   (``build_obstacle_join.py`` transform 1), and the per-slot rates below are a
   finite difference of those label positions. That is the admissible half of
   the rule.

2. **NO PERCEPTION LABEL IN ANY TRUNK LOSS** (the diagram's header row, audited
   ✅ CONFORMS in §2.1). This head is trained on ``obstacle.offline`` labels,
   which are PERCEPTION labels — so its gradient must never reach the encoder,
   the readout, or any predictor. The enforcement is structural (the memory is
   ``.detach()``-ed at the seam in :meth:`V6Stack.forward`) **and measured**:
   ``V6Stack.assert_isolation`` grows a fourth edge, ``perception_to_trunk``,
   whenever this head is built, and the mis-wired control arm
   (``isolate_interp_from_encoder=False``) makes that edge FAIL — which is what
   keeps the probe a probe rather than a comment (the C13 "guard that cannot
   fail" family).

3. **THE LABEL PATH ALREADY EXISTS — DO NOT BUILD A SECOND ONE.** Every target
   this module consumes comes from the join file
   ``build_obstacle_join.py`` writes and ``train_p8_occupancy.JoinFileReader``
   reads, through ``tanitad.data.bev_raster.agents_to_array``'s
   ``[A, 6] = (cx, cy, yaw, l, w, occ)`` rows and the reader's per-agent ``cls``
   column. :func:`targets_from_join` is a pure re-shaping of exactly those
   arrays — it opens no file, invents no field, and re-derives no geometry.
   The class vocabulary is ``bev_raster.ALL_CLASSES`` **imported**, never
   re-listed (10 classes, all dynamic agents — MEASURED over 87,481 cuboids).

──────────────────────────────────────────────────────────────────────────────
WHAT A SLOT EMITS, AND THE ONE PLACE THE DIAGRAM CELL IS INTERPRETED
──────────────────────────────────────────────────────────────────────────────
:data:`SLOT_FIELDS` is the channel layout as data. Reading the cell literally:

  * ``bbox cx, cy, yaw, l, w``  -> ``cx``/``cy`` (ego frame, +x fwd, +y LEFT —
    the ``refb_labels.ego_frame`` convention the join already writes),
    ``l = size_x`` / ``w = size_y`` (the join's mapping), and yaw as a
    ``(sin, cos)`` PAIR rather than a scalar: a scalar yaw regression is
    discontinuous at ±π and its L1 is not a metric on the circle, which is the
    same wrap defect ``bev_raster.wrap_to_pi`` exists to remove downstream.
  * ``state v, yaw-rate, occluded`` -> ⚠️ **the rates are EGO-FRAME RELATIVE**:
    ``v_rel_x``/``v_rel_y`` are d/dt of the agent's own ``(cx, cy)`` and
    ``yaw_rate_rel`` is d/dt of its ``yaw``, all in the ego frame.
    THIS IS A DECISION, and here is why it is the right one:
      - ``obstacle.offline`` **carries no velocity column** (MEASURED, join doc
        §1 / ``build_obstacle_join.py``'s header), so ANY rate is derived. The
        ego-frame difference needs the join and its own ``t_s`` and NOTHING
        else; an absolute ground-speed target would additionally need the
        egomotion poses composed per frame — a second derivation, i.e. exactly
        the parallel label path rule 3 forbids.
      - It is the quantity the LONGITUDINAL family actually consumes: closing
        speed to a lead is ``-v_rel_x`` and ``TTC = cx / max(-v_rel_x, ε)``,
        with no ego-speed term to supply. Absolute speed would have to be
        turned back into this by subtracting the ego's own velocity.
      - It is the quantity a monocular sequence SHOWS (looming). An absolute
        speed target asks the head to infer ego speed and add it — a harder
        task whose failure would be unattributable between "cannot see the
        agent" and "cannot see its own speed".
    ``occluded`` is the join's ``occ`` flag. ⚠️ Its stamp travels with it:
    MEASURED 2026-08-16 that ``occ`` IS ``bev_raster.fov_mask``'s predicate
    (0/7,680 cells disagree), so "occluded" here means OUT OF THE FRONT
    CAMERA'S FIELD while the track continues — not object-object occlusion.
    Predicting it is therefore the sharpest available form of the P4 question
    ("does the latent carry agents the camera cannot see"), and it must never
    be reported as generic occlusion reasoning.
  * ``class & size`` -> ``cls`` logits over :data:`AGENT_CLASSES` plus the
    ``l``/``w`` regression above (size IS the box's two extents; there is no
    second size channel to emit).

Plus ``presence`` — the DETR "∅ / no-object" logit. It is not in the diagram
cell because the cell describes an agent, not the set; a set-prediction head
without it cannot express "this slot is empty" and would be forced to place
every query on something.

──────────────────────────────────────────────────────────────────────────────
GEOMETRY, UNITS AND THE DECODE — all declared, none tuned
──────────────────────────────────────────────────────────────────────────────
The decode constants come from ``bev_raster.GRID_DEFAULT`` (60 m forward,
±16 m lateral — the P8 field), so the perception heads share ONE field
definition instead of two that drift. Coordinates are emitted NORMALISED and
multiplied by those extents; sizes go through ``softplus`` (a negative length
is not a value the label set can contain, and softplus is monotone with no
dead zone). ⛔ NOTHING here uses ``tanh``: MEASURED 2026-08-15 in fp32 that
``d/draw tanh(raw)`` is EXACTLY 0.0 from ``raw >= 10``, and this programme has
a gnorm-354,076 spike on the record — the regime that kills a saturating head
(``V6Config.emission_squash``'s docstring, and the reason the emission moved to
``_squash``). A coordinate head that saturates cannot recover a far agent.

⚠️ EVERY LOSS TERM IS IN ITS OWN UNIT and is returned SEPARATELY (metres,
nats, m/s, rad/s, dimensionless). :func:`slot_set_loss` returns the parts and a
weighted ``total``; a caller that logs only the total has thrown away the
attribution, which is the ADE-only failure in a detector's costume.

──────────────────────────────────────────────────────────────────────────────
THE MATCHER
──────────────────────────────────────────────────────────────────────────────
:func:`hungarian` is an exact O(n²m) rectangular assignment written in numpy so
this module adds NO dependency (``scipy`` is not in ``pyproject.toml``'s core
deps and every stack use of it is a lazy import inside a script). It is PINNED
against ``scipy.optimize.linear_sum_assignment`` in
``tests/test_v6_agent_slots.py`` — the duplication is admissible only WITH that
equivalence proof, the same contract ``bev_raster.yaw_from_quaternion`` carries
against ``physicalai.quaternion_yaw``.

⛔ ``n_queries`` must be >= the per-frame agent count. When it is not, the
FARTHEST targets are dropped (by range) and the drop is COUNTED and RETURNED
(``n_target_dropped``) — never silently absorbed, because a head that quietly
stops being scored on crowded frames would report its best numbers exactly
where driving is hardest. ⚠️ The right ``n_queries`` is the join's measured
per-frame agent-count distribution; that distribution is **UNMEASURED on this
box** (no join file is in the repo — the artifacts live pod-side) and must be
measured before the run. :data:`N_QUERIES_DEFAULT` is a declared placeholder,
not a fitted value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT, BEVGrid

__all__ = [
    "AGENT_CLASSES", "SLOT_FIELDS", "SLOT_WIDTH", "SLOT_SLICES",
    "PARAM_BAND", "N_QUERIES_DEFAULT", "MATCH_COST_W", "SLOT_LOSS_W",
    "NO_OBJECT_W", "AgentSlotDecoder", "hungarian", "match_slots",
    "slot_set_loss", "targets_from_join", "track_rates_from_join",
    "SlotDecodeRanges",
]

#: The class vocabulary — IMPORTED from the label side, never re-listed.
#: 10 classes, all dynamic agents (MEASURED over 87,481 cuboids; there is no
#: infrastructure class, which is why `TRAFFIC_LIGHT_REACT`'s agent slot can
#: never come from `obstacle.offline` — V6Config.n_agent_slots' own docstring).
AGENT_CLASSES: tuple[str, ...] = tuple(ALL_CLASSES)

#: ⛔ THE EMITTED CHANNEL LAYOUT, AS DATA. Order is the contract; the slices in
#: :data:`SLOT_SLICES` are derived from it so there is exactly one spelling.
SLOT_FIELDS: tuple[tuple[str, int], ...] = (
    ("presence", 1),                    # logit — DETR's ∅ / no-object
    ("cls", len(AGENT_CLASSES)),        # logits over AGENT_CLASSES
    ("cx", 1), ("cy", 1),               # normalised centre (decoded to metres)
    ("l", 1), ("w", 1),                 # pre-softplus size (decoded to metres)
    ("yaw_sin", 1), ("yaw_cos", 1),     # unnormalised heading vector
    ("v_rel_x", 1), ("v_rel_y", 1),     # ego-frame relative velocity, m/s
    ("yaw_rate_rel", 1),                # ego-frame relative yaw rate, rad/s
    ("occluded", 1),                    # logit — the join's `occ` (== ~fov_mask)
)
SLOT_WIDTH: int = sum(n for _, n in SLOT_FIELDS)


def _slices() -> dict[str, slice]:
    out, off = {}, 0
    for name, n in SLOT_FIELDS:
        out[name] = slice(off, off + n)
        off += n
    return out


SLOT_SLICES: dict[str, slice] = _slices()

#: §6's pre-registered size for this head: *"DETR-style slot decoder ~2–4 M
#: params on spatial tokens"*. Enforced at construction (``enforce_band``) for
#: the same reason ``train_p8_occupancy.BEVOccupancyHead`` enforces its ~1 M
#: band: a bigger head stops measuring what the LATENT carries and starts
#: measuring its own capacity.
PARAM_BAND: tuple[int, int] = (2_000_000, 4_000_000)

#: ⚠️ A DECLARED PLACEHOLDER, NOT A FITTED VALUE. The right number is the
#: per-frame agent-count distribution of the join (its 99th percentile), and
#: that is UNMEASURED here — no join file exists in the repo. Measure it with
#: ``JoinFileReader`` before the run and record the number in the prereg.
N_QUERIES_DEFAULT: int = 16

#: Hungarian matching costs. DETR's shape (class prob + box L1), with the box
#: term in METRES so the cost is interpretable rather than an arbitrary scale.
MATCH_COST_W: dict[str, float] = {
    "presence": 1.0,        # x (-sigmoid(presence)) — prefer confident slots
    "cls": 1.0,             # x (-p[true class])
    "centre_m": 1.0,        # x |Δcx| + |Δcy|, metres
    "size_m": 0.5,          # x |Δl| + |Δw|, metres
}

#: Loss weights. ⚠️ Units differ per term BY CONSTRUCTION (metres · nats ·
#: m/s · rad/s · dimensionless), so these are declared decisions, never
#: defaults — the same rule ``V6LossWeights.w_select`` carries.
SLOT_LOSS_W: dict[str, float] = {
    "presence": 1.0, "cls": 1.0, "centre": 1.0, "size": 1.0,
    "yaw": 1.0, "rates": 0.5, "occ": 0.5,
}

#: DETR's ∅-class down-weight: unmatched slots vastly outnumber matched ones,
#: and an unweighted BCE simply learns "always empty".
NO_OBJECT_W: float = 0.1


@dataclass(frozen=True)
class SlotDecodeRanges:
    """The normalised -> metres decode, from ``bev_raster.GRID_DEFAULT``.

    ⚠️ These are DECODE CONSTANTS, not clamps: a prediction outside the field
    is representable (and is then simply wrong), because a head that CANNOT
    express "there is a car at 70 m" would report a systematic error as a
    modelling success.
    """
    x_fwd_m: float = GRID_DEFAULT.x_fwd_m       # 60.0
    y_half_m: float = GRID_DEFAULT.y_half_m     # 16.0

    @classmethod
    def from_grid(cls, grid: BEVGrid) -> "SlotDecodeRanges":
        return cls(x_fwd_m=float(grid.x_fwd_m), y_half_m=float(grid.y_half_m))


# ============================================================================
# the module
# ============================================================================

class AgentSlotDecoder(nn.Module):
    """DETR-style slot decoder: spatial memory -> a SET of agent slots.

    ``memory`` ``[B, M, d_memory]`` — the spatial tokens, and the ONLY input.
    Returns a dict of decoded per-slot fields (metres / m·s⁻¹ / rad·s⁻¹) plus
    the raw head output, all ``[B, N, ·]``.

    ⛔ The forward signature is the vision-only audit: one tensor, no keyword
    inputs. Whatever privilege exists elsewhere in the stack cannot enter here
    because there is no door.

    Architecture: ``Linear`` memory projection + learned memory positions, N
    learned slot queries, a ``nn.TransformerDecoder`` (self-attention over
    slots, cross-attention into the memory — the DETR pattern that makes the
    output a SET rather than a grid), and one linear head over
    :data:`SLOT_FIELDS`.

    ``presence`` carries a **prior bias** ``logit(0.05)`` so an untrained head
    starts near "every slot empty" instead of near "every slot occupied" — the
    focal/DETR foreground-prior discipline. Nothing downstream in the v6 ladder
    consumes this head's output, so unlike the ``cond_tac_dyn`` port there is no
    loss-continuity-at-introduction requirement and hence no zero-init: a
    zero-init output layer would additionally starve the decoder below it of
    gradient on the first step.
    """

    #: prior probability a slot is occupied, at init.
    PRESENCE_PRIOR: float = 0.05

    def __init__(self, d_memory: int, n_memory: int, *,
                 n_queries: int = N_QUERIES_DEFAULT, d_model: int = 256,
                 depth: int = 3, n_heads: int = 8,
                 ranges: SlotDecodeRanges | None = None,
                 enforce_band: bool = True):
        super().__init__()
        if n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {n_queries}")
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must divide by n_heads "
                             f"{n_heads}")
        self.d_memory, self.n_memory = int(d_memory), int(n_memory)
        self.n_queries, self.d_model = int(n_queries), int(d_model)
        self.ranges = ranges or SlotDecodeRanges()

        self.mem_proj = nn.Linear(self.d_memory, self.d_model)
        self.mem_pos = nn.Parameter(torch.zeros(1, self.n_memory,
                                                self.d_model))
        nn.init.trunc_normal_(self.mem_pos, std=0.02)
        self.queries = nn.Parameter(torch.zeros(1, self.n_queries,
                                                self.d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            self.d_model, n_heads, dim_feedforward=4 * self.d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerDecoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, SLOT_WIDTH)
        with torch.no_grad():
            self.head.bias[SLOT_SLICES["presence"]] = math.log(
                self.PRESENCE_PRIOR / (1.0 - self.PRESENCE_PRIOR))

        n = self.n_params
        if enforce_band and not (PARAM_BAND[0] <= n <= PARAM_BAND[1]):
            raise ValueError(
                f"AgentSlotDecoder has {n:,} params — outside the §6 "
                f"pre-registered band {PARAM_BAND} (d_model={d_model}, "
                f"depth={depth}, n_memory={n_memory}, d_memory={d_memory}). A "
                f"bigger head stops measuring what the LATENT carries and "
                f"starts measuring its own capacity (the BEVOccupancyHead "
                f"precedent). Pass enforce_band=False only for shape tests at "
                f"toy widths.")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, memory: Tensor) -> dict:
        """``memory`` [B, M, d_memory] -> per-slot fields [B, N, ·].

        ⛔ ONE argument. See the class docstring: the signature is the
        vision-only audit.
        """
        if memory.ndim != 3:
            raise ValueError(f"memory must be [B, M, d_memory], got "
                             f"{tuple(memory.shape)}")
        if memory.shape[1] != self.n_memory or \
                memory.shape[2] != self.d_memory:
            raise ValueError(
                f"memory must be [B, {self.n_memory}, {self.d_memory}], got "
                f"{tuple(memory.shape)} — the decoder's positional table is "
                f"per-token, so a memory of a different length is a geometry "
                f"mismatch, not a resize.")
        b = memory.shape[0]
        mem = self.mem_proj(memory) + self.mem_pos.to(memory.dtype)
        q = self.queries.to(memory.dtype).expand(b, -1, -1)
        raw = self.head(self.norm(self.blocks(q, mem)))          # [B, N, W]
        return self.decode(raw)

    def decode(self, raw: Tensor) -> dict:
        """Split :data:`SLOT_FIELDS` out of the head output and apply the
        declared decode. Separated from :meth:`forward` so a test (or a probe
        replaying banked logits) can decode without re-running the network."""
        s = SLOT_SLICES
        r = self.ranges
        yaw_vec = raw[..., s["yaw_sin"].start:s["yaw_cos"].stop]
        yaw_vec = yaw_vec / yaw_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cx = raw[..., s["cx"]].squeeze(-1) * r.x_fwd_m
        cy = raw[..., s["cy"]].squeeze(-1) * r.y_half_m
        lw = nn.functional.softplus(
            raw[..., s["l"].start:s["w"].stop])
        return {
            "raw": raw,
            "presence_logit": raw[..., s["presence"]].squeeze(-1),   # [B,N]
            "cls_logits": raw[..., s["cls"]],                        # [B,N,C]
            "box": torch.stack([cx, cy, lw[..., 0], lw[..., 1]], dim=-1),
            "yaw_vec": yaw_vec,                                      # [B,N,2]
            "yaw": torch.atan2(yaw_vec[..., 0], yaw_vec[..., 1]),    # [B,N]
            "rates": torch.cat(
                [raw[..., s["v_rel_x"]], raw[..., s["v_rel_y"]],
                 raw[..., s["yaw_rate_rel"]]], dim=-1),              # [B,N,3]
            "occ_logit": raw[..., s["occluded"]].squeeze(-1),        # [B,N]
        }


# ============================================================================
# the matcher — exact, dependency-free, and pinned against scipy in tests
# ============================================================================

def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact minimum-cost assignment of ``cost`` ``[n, m]``; returns
    ``(rows, cols)`` with ``len == min(n, m)``.

    The classic O(n²m) shortest-augmenting-path (Kuhn–Munkres / JV) form for a
    RECTANGULAR matrix, transposing when ``n > m`` so the inner loop always
    runs over the longer axis.

    ⚠️ Written out here rather than importing ``scipy.optimize`` because scipy
    is NOT a core dependency of this package (``pyproject.toml``: torch +
    numpy) and every existing stack use of it is a lazy import inside a
    *script*. The duplication is admissible only WITH the equivalence proof:
    ``tests/test_v6_agent_slots.py`` pins this against
    ``scipy.optimize.linear_sum_assignment`` on random matrices and on the
    degenerate/tied cases, skipping honestly when scipy is absent.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"cost must be 2-D, got {c.shape}")
    if c.size == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if not np.isfinite(c).all():
        raise ValueError("cost contains non-finite entries — an assignment "
                         "over NaN/inf is undefined, not merely unstable")
    flip = c.shape[0] > c.shape[1]
    if flip:
        c = c.T
    n, m = c.shape
    inf = float("inf")
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)          # p[j] = row matched to col j
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = inf, -1
            free = ~used[1:]
            if free.any():
                cur = c[i0 - 1] - u[i0] - v[1:]
                better = free & (cur < minv[1:])
                if better.any():
                    minv[1:][better] = cur[better]
                    way[1:][better] = j0
                cand = np.where(free, minv[1:], inf)
                k = int(np.argmin(cand))
                delta, j1 = float(cand[k]), k + 1
            if j1 < 0:                              # defensive: cannot happen
                raise RuntimeError("no free column — the assignment loop lost "
                                   "its invariant")
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][~used[1:]] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows = np.zeros(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] > 0:
            rows[p[j] - 1] = p[j] - 1
            cols[p[j] - 1] = j - 1
    if flip:
        return cols, rows
    return rows, cols


def _match_cost(pred: dict, tgt: dict, b: int, keep: np.ndarray) -> np.ndarray:
    """Cost matrix ``[N, A_kept]`` for one batch element, in the declared
    :data:`MATCH_COST_W` mixture. Built under ``no_grad`` by the caller — the
    ASSIGNMENT is a discrete decision and must not be differentiated (DETR)."""
    w = MATCH_COST_W
    box_p = pred["box"][b]                                   # [N, 4]
    box_t = tgt["box"][b][keep]                              # [A, 4]
    centre = (box_p[:, None, :2] - box_t[None, :, :2]).abs().sum(-1)
    size = (box_p[:, None, 2:] - box_t[None, :, 2:]).abs().sum(-1)
    cls_t = tgt["cls"][b][keep]                              # [A] long, -1 ok
    prob = pred["cls_logits"][b].softmax(-1)                 # [N, C]
    safe = cls_t.clamp_min(0)
    cls_c = -prob[:, safe]                                   # [N, A]
    cls_c = torch.where(
        (cls_t >= 0)[None, :], cls_c, torch.zeros_like(cls_c))
    pres = -pred["presence_logit"][b].sigmoid()[:, None].expand_as(centre)
    c = (w["centre_m"] * centre + w["size_m"] * size
         + w["cls"] * cls_c + w["presence"] * pres)
    return c.detach().to(torch.float64).cpu().numpy()


def match_slots(pred: dict, tgt: dict) -> dict:
    """Hungarian-match slots to targets, per batch element.

    ``tgt`` carries ``box`` [B, A, 4], ``cls`` [B, A] (long, -1 = unknown) and
    ``valid`` [B, A] bool. Returns ``{"rows": [list[LongTensor]], "cols":
    [...], "n_target": [...], "n_dropped": [...]}`` — Python lists because the
    per-element match counts differ and a padded tensor would need a mask that
    is exactly this information again.

    ⛔ When a frame carries MORE valid targets than there are queries, the
    FARTHEST (by range ``√(cx²+cy²)``) are dropped and COUNTED. That policy is
    declared, not incidental: the near field is the one the plan acts on
    (``time_to_reach`` weighting, O2), and a silent drop would flatter the head
    exactly on crowded frames.
    """
    rows, cols, n_t, n_d = [], [], [], []
    n_q = int(pred["box"].shape[1])
    with torch.no_grad():
        for b in range(int(pred["box"].shape[0])):
            valid = tgt["valid"][b].nonzero(as_tuple=False).flatten()
            n_t.append(int(valid.numel()))
            if valid.numel() > n_q:
                rng = tgt["box"][b][valid][:, :2].norm(dim=-1)
                order = torch.argsort(rng)[:n_q]
                valid = valid[order]
            n_d.append(n_t[-1] - int(valid.numel()))
            if valid.numel() == 0:
                rows.append(torch.zeros(0, dtype=torch.long))
                cols.append(torch.zeros(0, dtype=torch.long))
                continue
            keep = valid.cpu().numpy()
            c = _match_cost(pred, tgt, b, valid)
            r, k = hungarian(c)
            rows.append(torch.as_tensor(r, dtype=torch.long))
            cols.append(torch.as_tensor(keep[k], dtype=torch.long))
    return {"rows": rows, "cols": cols, "n_target": n_t, "n_dropped": n_d}


# ============================================================================
# the set-prediction loss
# ============================================================================

def slot_set_loss(pred: dict, tgt: dict, *, match: dict | None = None,
                  weights: dict | None = None) -> dict:
    """DETR set loss over :data:`SLOT_FIELDS`, returned PER TERM.

    Terms and their units, all reported separately (§ the four-families rule's
    sibling discipline — a pooled score hides exactly the trade-off one wants
    to see):

      ``presence`` nats · ``cls`` nats · ``centre`` metres · ``size`` metres ·
      ``yaw`` dimensionless (``1 − cos Δ``) · ``rates`` m·s⁻¹ + rad·s⁻¹ ·
      ``occ`` nats.

    Masking, and what a mask means here:
      * ``tgt["valid"]``     — padding. Absent agents are not "clear road".
      * ``tgt["cls"] < 0``   — class unknown (a join without the ``cls``
        column, which ``JoinFileReader.has_classes`` reports). The class term
        is then computed over ZERO items and ``n_cls`` says so.
      * ``tgt["rates_mask"]`` — the track was not seen in the neighbouring
        frames, so no finite difference exists. ⛔ A missing rate is MASKED,
        never zero-filled: zero is a legitimate value (a stationary car) and
        filling it would teach the head that unseen means still.
      * ``tgt["occ"] < 0``   — the join carried no occlusion flag.

    Returns the parts, the weighted ``total``, and the counts every part was
    computed over — a term with ``n == 0`` is reported as ``0.0`` WITH its
    count, never dropped, so a silently-uncomputed term is visible.
    """
    w = {**SLOT_LOSS_W, **(weights or {})}
    m = match or match_slots(pred, tgt)
    dev = pred["presence_logit"].device
    dt = pred["presence_logit"].dtype
    b_n = int(pred["presence_logit"].shape[0])

    # ---- presence: BCE over ALL slots (matched -> 1, else 0) ---------------
    tgt_pres = torch.zeros_like(pred["presence_logit"])
    wgt_pres = torch.full_like(pred["presence_logit"], NO_OBJECT_W)
    for b in range(b_n):
        r = m["rows"][b].to(dev)
        if r.numel():
            tgt_pres[b, r] = 1.0
            wgt_pres[b, r] = 1.0
    l_pres = nn.functional.binary_cross_entropy_with_logits(
        pred["presence_logit"], tgt_pres, weight=wgt_pres)

    zero = torch.zeros((), device=dev, dtype=dt)
    acc = {"cls": zero.clone(), "centre": zero.clone(), "size": zero.clone(),
           "yaw": zero.clone(), "rates": zero.clone(), "occ": zero.clone()}
    n = {"cls": 0, "centre": 0, "size": 0, "yaw": 0, "rates": 0, "occ": 0}

    for b in range(b_n):
        r, c = m["rows"][b].to(dev), m["cols"][b].to(dev)
        if r.numel() == 0:
            continue
        pb, tb = pred["box"][b][r], tgt["box"][b][c]
        acc["centre"] = acc["centre"] + (pb[:, :2] - tb[:, :2]).abs().sum()
        n["centre"] += int(r.numel())
        acc["size"] = acc["size"] + (pb[:, 2:] - tb[:, 2:]).abs().sum()
        n["size"] += int(r.numel())
        # yaw on the circle: 1 - cos(Δ) via the unit (sin, cos) pair. No wrap,
        # no discontinuity, and the minimum is exact at Δ = 0.
        yv = pred["yaw_vec"][b][r]
        yt = tgt["box"].new_zeros((r.numel(), 2))
        yt[:, 0] = torch.sin(tgt["yaw"][b][c])
        yt[:, 1] = torch.cos(tgt["yaw"][b][c])
        acc["yaw"] = acc["yaw"] + (1.0 - (yv * yt).sum(-1)).sum()
        n["yaw"] += int(r.numel())
        ct = tgt["cls"][b][c]
        ok = ct >= 0
        if bool(ok.any()):
            acc["cls"] = acc["cls"] + nn.functional.cross_entropy(
                pred["cls_logits"][b][r][ok], ct[ok], reduction="sum")
            n["cls"] += int(ok.sum())
        rm = tgt["rates_mask"][b][c]
        if bool(rm.any()):
            acc["rates"] = acc["rates"] + (
                pred["rates"][b][r][rm] - tgt["rates"][b][c][rm]).abs().sum()
            n["rates"] += int(rm.sum())
        ot = tgt["occ"][b][c]
        om = ot >= 0.0
        if bool(om.any()):
            acc["occ"] = acc["occ"] + \
                nn.functional.binary_cross_entropy_with_logits(
                    pred["occ_logit"][b][r][om], ot[om], reduction="sum")
            n["occ"] += int(om.sum())

    parts = {"presence": l_pres}
    for k, v in acc.items():
        parts[k] = v / max(n[k], 1)
    total = sum(w[k] * parts[k] for k in parts)
    out = {f"loss_{k}": v for k, v in parts.items()}
    out["total"] = total
    out["n"] = {"matched": sum(int(x.numel()) for x in m["rows"]),
                "target": int(sum(m["n_target"])),
                "dropped": int(sum(m["n_dropped"])), **n}
    out["_weights"] = dict(w)
    return out


# ============================================================================
# targets — a re-shaping of the EXISTING join arrays, nothing more
# ============================================================================

def track_rates_from_join(prev_rec: dict | None, rec: dict,
                          next_rec: dict | None) -> tuple[np.ndarray,
                                                          np.ndarray]:
    """Ego-frame relative rates for one join record, by track id.

    ``rec`` is a join LINE as ``build_obstacle_join.py`` writes it:
    ``{"clip_id", "frame_idx", "t_s", "agents": [{cx, cy, yaw, l, w, occ,
    track_id, cls}]}``. Returns ``(rates [A, 3], mask [A] bool)`` with
    ``rates = (v_rel_x, v_rel_y, yaw_rate_rel)``.

    A CENTRAL difference when both neighbours carry the track, one-sided when
    only one does, and ``mask=False`` when neither does — because a rate that
    was not observed must be masked out of the loss, not filled with zero (zero
    is a legitimate value: a stationary car).

    ⚠️ ``t_s`` is the join's own frame time and the episode grid's spacing is
    ~0.1007 s, NOT 0.1 (MEASURED, ``build_obstacle_join.py`` header §CLOCK) —
    so the denominator is read from the records, never assumed.

    ⚠️ Yaw differences are wrapped to (−π, π] before dividing, or a wrap at ±π
    would manufacture a ~63 rad/s spike out of a straight-driving car.
    """
    ag = rec.get("agents") or []
    a = len(ag)
    rates = np.zeros((a, 3), dtype=np.float64)
    mask = np.zeros(a, dtype=bool)
    if a == 0:
        return rates, mask

    def _index(r):
        if not r:
            return {}, None
        return ({str(d.get("track_id", "")): d for d in (r.get("agents") or [])
                 if d.get("track_id") is not None},
                float(r["t_s"]))

    prev_i, t_prev = _index(prev_rec)
    next_i, t_next = _index(next_rec)
    t0 = float(rec["t_s"])
    for i, d in enumerate(ag):
        tid = str(d.get("track_id", ""))
        if not tid:
            continue
        lo, t_lo = (prev_i.get(tid), t_prev)
        hi, t_hi = (next_i.get(tid), t_next)
        if lo is not None and hi is not None:
            dt = t_hi - t_lo
            src_lo, src_hi = lo, hi
        elif hi is not None:
            dt, src_lo, src_hi = t_hi - t0, d, hi
        elif lo is not None:
            dt, src_lo, src_hi = t0 - t_lo, lo, d
        else:
            continue
        if not dt or not math.isfinite(dt) or abs(dt) < 1e-6:
            continue
        dyaw = float(src_hi["yaw"]) - float(src_lo["yaw"])
        dyaw -= 2.0 * math.pi * math.floor((dyaw + math.pi) / (2.0 * math.pi))
        rates[i] = ((float(src_hi["cx"]) - float(src_lo["cx"])) / dt,
                    (float(src_hi["cy"]) - float(src_lo["cy"])) / dt,
                    dyaw / dt)
        mask[i] = True
    return rates, mask


def targets_from_join(agents, classes=None, rates=None, rates_mask=None, *,
                      n_pad: int | None = None, device=None,
                      dtype=torch.float32) -> dict:
    """Turn ONE frame's join arrays into the target dict :func:`slot_set_loss`
    consumes. Pure re-shaping — no file is opened and no geometry is re-derived.

    ``agents``: the ``[A, 6] = (cx, cy, yaw, l, w, occ)`` array
    ``tanitad.data.bev_raster.agents_to_array`` produces and
    ``JoinFileReader.lookup`` returns (``occ`` ``-1`` = no flag).
    ``classes``: the per-agent ``label_class`` strings ``JoinFileReader.
    lookup_classes`` returns, or ``None`` when the join predates the column —
    unknown classes become ``-1`` and the class term is then computed over
    zero items and SAYS SO, rather than defaulting to class 0.
    ``rates``/``rates_mask``: from :func:`track_rates_from_join`; omitted =
    all-masked.

    ⛔ ``agents is None`` (a frame ABSENT from the join) is NOT accepted here:
    NO_LABEL is a state of its own and must be skipped+counted by the caller,
    never passed in as an empty set — an empty ``[0, 6]`` array means LABELLED
    CLEAR, which is a different fact (join doc §4).

    Returns a batch of ONE (leading axis 1) so the caller can ``torch.cat``
    frames into a batch after padding to a common ``n_pad``.
    """
    if agents is None:
        raise ValueError(
            "agents is None — that is NO_LABEL, not an empty agent set. Skip "
            "and count the frame; an empty [0, 6] array is 'labelled clear', "
            "which is a different state (obstacle-join doc §4).")
    a = np.asarray(agents, dtype=np.float64).reshape(-1, 6)
    n = a.shape[0]
    pad = int(n_pad if n_pad is not None else n)
    if pad < n:
        raise ValueError(f"n_pad {pad} < n_agents {n}")

    def _z(shape, fill=0.0, dt=dtype):
        return torch.full((1, *shape), fill, dtype=dt, device=device)

    box = _z((pad, 4))
    yaw = _z((pad,))
    occ = _z((pad,), -1.0)
    cls = torch.full((1, pad), -1, dtype=torch.long, device=device)
    rt = _z((pad, 3))
    rm = torch.zeros((1, pad), dtype=torch.bool, device=device)
    valid = torch.zeros((1, pad), dtype=torch.bool, device=device)
    if n:
        t = torch.as_tensor(a, dtype=dtype, device=device)
        box[0, :n] = torch.stack([t[:, 0], t[:, 1], t[:, 3], t[:, 4]], dim=-1)
        yaw[0, :n] = t[:, 2]
        occ[0, :n] = t[:, 5]
        valid[0, :n] = True
        if classes is not None:
            idx = {c: i for i, c in enumerate(AGENT_CLASSES)}
            cls[0, :n] = torch.as_tensor(
                [idx.get(str(c), -1) for c in list(classes)[:n]],
                dtype=torch.long, device=device)
        if rates is not None:
            rt[0, :n] = torch.as_tensor(np.asarray(rates)[:n], dtype=dtype,
                                        device=device)
            rm[0, :n] = torch.as_tensor(
                np.asarray(rates_mask)[:n] if rates_mask is not None
                else np.ones(n, dtype=bool), dtype=torch.bool, device=device)
    return {"box": box, "yaw": yaw, "cls": cls, "occ": occ,
            "rates": rt, "rates_mask": rm, "valid": valid}
