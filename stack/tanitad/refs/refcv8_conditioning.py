"""refcv8 WP-B -- the tactical layer CONDITIONS fan generation and selection (DESIGN.md sec. 3, the WP-B package
`TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/`).

PI 2026-10-04 (R8-4): "We should condition the fan generation and trajectory selection with the tactical goals/actions.
So the most probable actions are influencing the planning process." This module holds every refcv8 conditioning piece
that is NOT a wiring edit, so the decoder / model / trainer edits stay small and the pieces are testable alone:

  * the HYPOTHESIS space h = (lat3, lon6) in the frozen v7 ids (vocab_v7 LAT/LON order; the masked classes listed);
  * :func:`tag_paths` -- the param-free tagger: the hypothesis a candidate path IS (the R1 dense literals at plan level);
  * :func:`constraint_targets` -- the constraint LABELS a window's own future gives (training targets of the constraint
    heads; a probe-level [0, 6] s definition until the v9 constraint fields are wired, see TRAINER notes);
  * :func:`hypothesis_posteriors` -- the tactical decoder's 8-way lat/lon logits -> log p over the hypothesis space;
  * :func:`build_phi` -- the per-candidate feature phi_k (DESIGN sec. 3.1);
  * :class:`PerCandidateModulation` -- zero-init `q <- q * (1 + g(phi)) + b(phi)` after each decoder layer;
  * :func:`allocate` -- M extra candidates over the top-k hypotheses, proportional to p(h) (sec. 3.1.2);
  * :class:`FactorizedSelection` -- zero-init beta * log p(h_k) + gamma_j * sat_j(k) (sec. 3.2);
  * :class:`SubScoreHeads` + :func:`subscore_targets` -- Hydra-style per-candidate critics (X1);
  * :func:`listwise_selection_loss` -- the soft-target listwise CE (R1-H4's definition, verbatim semantics);
  * :func:`constraint_satisfaction_loss` -- L_sat on the conditioned (allocated) candidates;
  * :func:`prior_free_lateral_offset` -- X2b's per-candidate x0 shift that cancels the residual prior's curvature;
  * :class:`R8Generator` -- every refcv8 training-time draw comes from a DEDICATED generator, so the refcv7 RNG stream
    (the sampler's eps) is untouched and a warm start reproduces refcv7 bit for bit.

ADMISSIBILITY (PI 2026-08-03, unchanged): the hypothesis space is built ONLY from the lat / lon ACTION posteriors and
their constraints. The 22 goal tokens -- including the five situation outputs (TRAFFIC_LIGHT_*, YIELD) -- never enter
phi, the allocation or the factorised score; :func:`assert_no_situation_feed` pins it. The route checkpoint is an INPUT
computed outside the model; nothing here writes to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.channel_admissibility import ChannelExclusion
from tanitad.train.config_hygiene import strict_fields

__all__ = [
    "LAT_V7", "LON_V7", "LAT3", "LAT3_V7_IDS", "LON6", "LON6_V7_IDS", "N_LAT3", "N_LON6", "N_HYP",
    "LAT_MASKED_V7_IDS", "LON_MASKED_V7_IDS", "PHI_DIMS", "R8Config", "R8Generator",
    "tag_paths", "constraint_targets", "hypothesis_posteriors", "build_phi", "PerCandidateModulation",
    "allocate", "select_source_anchors", "FactorizedSelection", "SubScoreHeads", "subscore_targets",
    "listwise_selection_loss", "constraint_satisfaction_loss", "constraint_head_loss", "candidate_constraints",
    "prior_free_lateral_offset", "assert_no_situation_feed", "joint_id", "split_joint",
]

# ---- the frozen v7 vocabulary (vocab_v7.py:320-331; pinned against the tree by the tests) ------------------------- #
LAT_V7: tuple[str, ...] = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R",
                           "TURN_L", "TURN_R")
LON_V7: tuple[str, ...] = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
                           "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
#: the lateral hypotheses (v9 SPEC sec. 3.7: LANE_KEEP / TURN_L / TURN_R; LANE_CHANGE only if WP-A's SAM3 rule passes
#: V5 -- added then by appending, never by reordering)
LAT3: tuple[str, ...] = ("LANE_KEEP", "TURN_L", "TURN_R")
LAT3_V7_IDS: tuple[int, ...] = (0, 6, 7)
#: the longitudinal hypotheses (v9 sec. 3.7 mapping: STOP and DECELERATE -> BRAKE_TO; KEEP -> CRUISE)
LON6: tuple[str, ...] = ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")
LON6_V7_IDS: tuple[int, ...] = (0, 1, 3, 4, 5, 7)
LAT_MASKED_V7_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)        # LC_L, LC_R, ABORT_LC, NUDGE_L, NUDGE_R
LON_MASKED_V7_IDS: tuple[int, ...] = (2, 6)                 # YIELD_MERGE, ADAPT_SPEED_FOR_CURVE
N_LAT3, N_LON6 = len(LAT3), len(LON6)
N_HYP = N_LAT3 * N_LON6                                     # 18 joint hypotheses
#: per-hypothesis constraint fields (DESIGN sec. 3.1): lateral (terminal heading psi_term [rad], turn onset [s]);
#: longitudinal (6-s progress P6 [m], speed at 6 s [m/s]). Normalised forms are what the heads emit.
C_LAT, C_LON = 2, 2
#: phi = [lat3 one-hot (3) | lon6 one-hot (6) | log p(lat) | log p(lon) | c_lat (2) | c_lon (2) | c_valid | alloc]
PHI_DIMS = N_LAT3 + N_LON6 + 2 + C_LAT + C_LON + 2           # 17

# ---- tagger literals (the R1 dense rule, `R1 code/r1_lib.py` H2 literals, applied to a PLAN) ---------------------- #
TAG_TURN_DEG = 30.0
TAG_MIN_LEN_M = 5.0
TAG_STALL_M = 0.05
TAG_STOP_MS = 0.5
TAG_CRAWL_MS = 2.0
TAG_DV_MS = 1.5
#: constraint-label literals
ONSET_DEG = 10.0            # turn onset = first time |psi(t) - psi(NOW)| >= 10 deg (only on windows whose max >= 30)
PROG_LOG_SCALE = math.log1p(120.0)
V_SCALE = 30.0
T_SCALE = 6.0
LOGP_FLOOR = math.log(1e-4)


def joint_id(lat3: Tensor, lon6: Tensor) -> Tensor:
    return lat3 * N_LON6 + lon6


def split_joint(h: Tensor) -> tuple[Tensor, Tensor]:
    return torch.div(h, N_LON6, rounding_mode="floor"), h % N_LON6


@strict_fields
@dataclass
class R8Config:
    """Every refcv8 WP-B switch. Defaults = OFF everywhere: a build with the default config is refcv7 bit for bit."""
    enable: bool = False                 # builds the seams (zero-init) -- nothing changes until gates move
    modulate_base: bool = True           # base fan candidates get phi (tag + log p, no constraint unless below)
    base_constraints: bool = False       # base fan candidates also get their tag's predicted constraint
    n_alloc: int = 0                     # M allocated candidates (0 = no allocation)
    alloc_top_k: int = 4
    alloc_min_each: int = 2
    alloc_emit: bool = False             # allocated candidates may be EMITTED (False = trained, never emitted)
    prior_free_group: bool = False       # X2b: + one prior-free duplicate of every anchor
    prior_free_emit: bool = False
    lat_prior_dropout: float = 0.0       # X2a: P(kappa0 -> 0) per training row
    cond_dropout: float = 0.15           # p_uncond on phi (CFG), training only
    tf_start: float = 1.0                # scheduled sampling: teacher-forcing ratio at step 0 ...
    tf_end: float = 0.25                 # ... and at the end of the run (linear, PER WINDOW coin)
    w_cons: float = 0.05                 # constraint-head loss weight (budgeted under X4)
    w_sat: float = 0.0                   # L_sat weight (0 = the T1 arm; > 0 = T2)
    w_alloc_l1: float = 1.0              # matched L1 on the best GT-hypothesis allocated candidate
    w_listwise: float = 0.0              # > 0 replaces the E9 single-winner CE (X1)
    w_subscore: float = 0.0              # > 0 trains the Hydra-style sub-score heads (X1h)
    subscore_hidden: int = 128
    list_t: float = 1.0
    list_speed_scale_m: float = 1.0
    list_dir_scale_deg: float = 15.0
    seed: int = 20261004                 # the DEDICATED generator's seed
    #: X3 (SPEC_REFCV8 sec. 8.3, MM ruling Q5): the SOURCE of the 4-way set-speed channel -- "" = the inherited one,
    #: "n2" / "n3" = the past-only v9 proxy. With a past-only source the speed ceiling is never BELOW the fed value
    #: (refc_v3._scene_hook): a fed 130 km/h is not clamped to the 4-way top step of 120.
    speed_input: str = ""
    #: MM ruling Q2: supervise the v9 CONSTRAINT vectors (lat_c / lon_c / speed_goal) with heads on the behaviour
    #: decoder's action queries (built iff w_v9_cons > 0; they feed the loss, never the planner)
    v9_cons: bool = False
    w_v9_cons: float = 0.0
    #: MM ruling Q1 (SPEC_REFCV8 sec. 3.1, I-2): the training step from which the EXTRA candidates (allocated /
    #: prior-free) may be EMITTED; before it they are generated and trained but never win the argmax, so the
    #: warm-started step 0 is refcv7's emitted plan. 0 = from the first step (the tiny arms, no warm start).
    emit_start: int = 0
    #: MM (B), 2026-10-04: the FULL N2 ladder (8 steps + unknown) through a zero-init FiLM seam beside the inherited
    #: 4-way channel (the 4-way bins merge 70 / 80 into 100 on 20.8 % of rows); the ceiling is then the N2 value
    speed_enc8: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class R8Generator:
    """A dedicated torch.Generator per device. ⛔ Every refcv8 TRAINING-time draw (cond dropout, prior dropout,
    scheduled-sampling coins, allocation choices, allocated-candidate noise) uses THIS, never the global RNG: the global
    stream feeds refcv7's sampler eps, and consuming from it would shift every base candidate's noise -- a warm start
    would then not reproduce refcv7 even with every gate at zero."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._g: dict[str, torch.Generator] = {}

    def get(self, device) -> torch.Generator:
        key = str(torch.device(device))
        if key not in self._g:
            g = torch.Generator(device=torch.device(device))
            g.manual_seed(self.seed)
            self._g[key] = g
        return self._g[key]

    def rand(self, shape, device) -> Tensor:
        return torch.rand(shape, generator=self.get(device), device=device)

    def randn(self, shape, device, dtype=torch.float32) -> Tensor:
        return torch.randn(shape, generator=self.get(device), device=device, dtype=dtype)


# ================================================================================================================= #
# the tagger (what hypothesis a path IS) -- param-free, no grad                                                    #
# ================================================================================================================= #
def _slot_dts(slot_t_s: Sequence[float], device, dtype) -> Tensor:
    t = torch.tensor([0.0] + [float(x) for x in slot_t_s], device=device, dtype=dtype)
    return t[1:] - t[:-1]


@torch.no_grad()
def tag_paths(paths: Tensor, v0: Tensor, slot_t_s: Sequence[float]) -> tuple[Tensor, Tensor]:
    """``paths`` [B, N, S, 2] (ego frame at NOW, x fwd, y left) + ``v0`` [B] -> (lat3 [B, N], lon6 [B, N]) long.

    Lateral (R1 H2 literals): segment headings of origin -> slot 1 -> ... -> slot S (a segment < 0.05 m reads 0);
    TURN_L if max >= +30 deg, TURN_R if min <= -30 deg (both: the larger |.|), else LANE_KEEP; LANE_KEEP if the path
    is shorter than 5 m. Longitudinal: segment speeds v_j = |seg_j| / dt_j; v_end = the last segment's; with
    vmax = max(max_j v_j, v0): CRUISE; ACCELERATE if v_end - v0 >= +1.5; BRAKE_TO if <= -1.5; CREEP if vmax <= 2.0;
    HOLD if vmax < 0.5 (that precedence, R1 `dense_lon`). FOLLOW is never a geometric tag (it needs a lead).
    ⚠ Plan-level [0, 6] s. The v9 band is [NOW+2, NOW+8] s; consistency is scored only for hypotheses observable in
    the plan (DESIGN sec. 4)."""
    p = paths.to(torch.float32)
    b, n, s, _ = p.shape
    q = torch.cat([p.new_zeros(b, n, 1, 2), p], dim=2)
    d = q[:, :, 1:] - q[:, :, :-1]
    seg = torch.linalg.vector_norm(d, dim=-1)                                  # [B, N, S]
    th = torch.where(seg < TAG_STALL_M, torch.zeros_like(seg), torch.atan2(d[..., 1], d[..., 0]))
    mx, mn = th.amax(-1), th.amin(-1)
    t = math.radians(TAG_TURN_DEG)
    left, right = mx >= t, mn <= -t
    lat = torch.zeros(b, n, dtype=torch.long, device=p.device)
    both = left & right
    lat = torch.where(left & ~both, torch.ones_like(lat), lat)
    lat = torch.where(right & ~both, torch.full_like(lat, 2), lat)
    lat = torch.where(both & (mx >= -mn), torch.ones_like(lat), lat)
    lat = torch.where(both & (mx < -mn), torch.full_like(lat, 2), lat)
    lat = torch.where(seg.sum(-1) < TAG_MIN_LEN_M, torch.zeros_like(lat), lat)
    dt = _slot_dts(slot_t_s, p.device, torch.float32)
    v = seg / dt                                                               # [B, N, S]
    vv0 = v0.reshape(-1, 1).to(torch.float32)
    vmax = torch.maximum(v.amax(-1), vv0)
    dv = v[..., -1] - vv0
    lon = torch.full((b, n), LON6.index("CRUISE"), dtype=torch.long, device=p.device)
    lon = torch.where(dv >= TAG_DV_MS, torch.full_like(lon, LON6.index("ACCELERATE")), lon)
    lon = torch.where(dv <= -TAG_DV_MS, torch.full_like(lon, LON6.index("BRAKE_TO")), lon)
    lon = torch.where(vmax <= TAG_CRAWL_MS, torch.full_like(lon, LON6.index("CREEP")), lon)
    lon = torch.where(vmax < TAG_STOP_MS, torch.full_like(lon, LON6.index("HOLD")), lon)
    return lat, lon


# ================================================================================================================= #
# constraint LABELS from the window's own future (training targets only -- labels may use ego, PI 2026-08-03)       #
# ================================================================================================================= #
def _wrap(a: Tensor) -> Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _path_len(p: Tensor) -> Tensor:
    q = torch.cat([p.new_zeros(*p.shape[:-2], 1, 2), p], dim=-2)
    return torch.linalg.vector_norm(q[..., 1:, :] - q[..., :-1, :], dim=-1).sum(-1)


def _terminal_heading(p: Tensor) -> Tensor:
    d = p[..., -1, :] - p[..., -2, :]
    th = torch.atan2(d[..., 1], d[..., 0])
    return torch.where(torch.linalg.vector_norm(d, dim=-1) < TAG_STALL_M, torch.zeros_like(th), th)


@torch.no_grad()
def constraint_targets(gt: Tensor, gt_valid: Tensor, pose_last: Tensor, fut: Tensor, fut_valid: Tensor,
                       tick_s: float = 0.1) -> dict[str, Tensor]:
    """The constraint LABELS of a window (probe-level [0, 6] s; the v9 fields replace them when wired).

    ``gt`` [B, S, 2] slots (ego frame), ``gt_valid`` [B, S]; ``pose_last`` [B, 4] = (x, y, yaw, v) at NOW;
    ``fut`` [B, H, 4] future poses at the native tick, ``fut_valid`` [B, H].
    Returns raw values + validity:
      psi_term  [rad] the terminal (slot S-1 -> S) heading -- the quantity D0c's heading constraint conditions on;
      t_onset   [s]   first tick with |psi(t) - psi(NOW)| >= 10 deg, valid only if max |.| >= 30 deg (a turn);
      prog6     [m]   the 6-s arc of the slot chords -- D0's progress constraint (same definition);
      v_end     [m/s] speed at the last valid future tick of the 6-s horizon.
    """
    g = gt.to(torch.float32)
    gv = gt_valid.to(torch.bool)
    ok_end = gv[:, -1]
    prog6 = _path_len(g)
    psi_term = _terminal_heading(g)
    yaw0 = pose_last[:, 2].to(torch.float32)
    f = fut.to(torch.float32)
    fv = fut_valid.to(torch.bool)
    h = min(f.shape[1], int(round(6.0 / tick_s)))
    dpsi = _wrap(f[:, :h, 2] - yaw0[:, None])
    dpsi = torch.where(fv[:, :h], dpsi, torch.zeros_like(dpsi))
    absd = dpsi.abs()
    is_turn = absd.amax(-1) >= math.radians(TAG_TURN_DEG)
    hit = absd >= math.radians(ONSET_DEG)
    first = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(hit[:, 0], h, dtype=torch.long))
    t_onset = (first.to(torch.float32) + 1.0) * float(tick_s)
    v_end = f[:, h - 1, 3]
    return {"psi_term": psi_term, "psi_term_valid": ok_end & (prog6 > 0.5),
            "t_onset": t_onset, "t_onset_valid": ok_end & is_turn,
            "prog6": prog6, "prog6_valid": ok_end & (prog6 > 0.5),
            "v_end": v_end, "v_end_valid": fv[:, h - 1]}


def normalise_constraints(c: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """-> (lat [B, 2], lat_valid [B, 2], lon [B, 2], lon_valid [B, 2]) in the heads' normalised units:
    lat = (psi_term / pi, t_onset / 6); lon = (log1p(prog6) / log1p(120), v_end / 30)."""
    lat = torch.stack([c["psi_term"] / math.pi, c["t_onset"] / T_SCALE], -1)
    lat_v = torch.stack([c["psi_term_valid"], c["t_onset_valid"]], -1)
    lon = torch.stack([torch.log1p(c["prog6"].clamp_min(0.0)) / PROG_LOG_SCALE, c["v_end"] / V_SCALE], -1)
    lon_v = torch.stack([c["prog6_valid"], c["v_end_valid"]], -1)
    return lat, lat_v, lon, lon_v


def denorm_prog(x: Tensor) -> Tensor:
    return torch.expm1(x * PROG_LOG_SCALE)


# ================================================================================================================= #
# posteriors over the hypothesis space                                                                              #
# ================================================================================================================= #
def hypothesis_posteriors(lat_logits8: Tensor, lon_logits8: Tensor) -> tuple[Tensor, Tensor]:
    """The tactical decoder's 8-way logits -> (log p(lat3) [B, 3], log p(lon6) [B, 6]), the masked classes removed
    BEFORE the softmax (they are never hypotheses). DETACHED: the planning loss must not reshape the tactical head
    (`refcv6_tactical.planner_feeds` docstring)."""
    la = lat_logits8.detach().to(torch.float32)[:, list(LAT3_V7_IDS)]
    lo = lon_logits8.detach().to(torch.float32)[:, list(LON6_V7_IDS)]
    return F.log_softmax(la, -1).clamp_min(LOGP_FLOOR), F.log_softmax(lo, -1).clamp_min(LOGP_FLOOR)


def candidate_constraints(cons_lat8: Tensor, cons_lon8: Tensor, lat3: Tensor, lon6: Tensor) -> Tensor:
    """Per-candidate constraint vector [B, N, 4] = (c_lat of the candidate's lat hypothesis, c_lon of its lon one),
    read from the constraint heads' per-QUERY outputs ``cons_lat8`` [B, 8, 2], ``cons_lon8`` [B, 8, 2] (v7 rows)."""
    lat_rows = torch.as_tensor(LAT3_V7_IDS, device=lat3.device)[lat3]        # [B, N] v7 row
    lon_rows = torch.as_tensor(LON6_V7_IDS, device=lon6.device)[lon6]
    cl = torch.gather(cons_lat8.detach(), 1, lat_rows[..., None].expand(*lat_rows.shape, C_LAT))
    co = torch.gather(cons_lon8.detach(), 1, lon_rows[..., None].expand(*lon_rows.shape, C_LON))
    return torch.cat([cl, co], -1).to(torch.float32)


def build_phi(lat3: Tensor, lon6: Tensor, logp_lat3: Tensor, logp_lon6: Tensor, cons: Tensor | None,
              cons_valid: Tensor | None, alloc: Tensor) -> Tensor:
    """phi_k [B, N, PHI_DIMS]. ``cons`` [B, N, 4] normalised (or None = zeros, valid 0); ``alloc`` [B, N] bool."""
    b, n = lat3.shape
    oh_l = F.one_hot(lat3, N_LAT3).to(torch.float32)
    oh_o = F.one_hot(lon6, N_LON6).to(torch.float32)
    lpl = torch.gather(logp_lat3.to(torch.float32), 1, lat3)[..., None] / -LOGP_FLOOR
    lpo = torch.gather(logp_lon6.to(torch.float32), 1, lon6)[..., None] / -LOGP_FLOOR
    if cons is None:
        c = torch.zeros(b, n, C_LAT + C_LON, device=lat3.device)
        cv = torch.zeros(b, n, 1, device=lat3.device)
    else:
        c = cons.to(torch.float32)
        cv = (torch.ones(b, n, 1, device=lat3.device) if cons_valid is None
              else cons_valid.to(torch.float32).reshape(b, n, 1))
        c = c * cv
    return torch.cat([oh_l, oh_o, lpl, lpo, c, cv, alloc.to(torch.float32)[..., None]], -1)


class PerCandidateModulation(nn.Module):
    """``q <- q * (1 + gamma_i(phi)) + beta_i(phi)`` after decoder layer i. ZERO-INIT, so ``q * (1 + 0) + 0 == q``
    exactly in IEEE arithmetic: a warm start from refcv7 is bit-identical at step 0 (DiT adaLN-Zero, 2212.09748
    sec. 3.2 p5). Applied AFTER the layer and BEFORE F4's time AdaLN -- never by widening the shared ``cond`` to
    [B, N, d], which would route the existing FiLM through a different GEMM shape (not guaranteed bit-identical)."""

    def __init__(self, n_layers: int, d_model: int, d_phi: int = PHI_DIMS):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(d_phi, 2 * d_model) for _ in range(n_layers))
        for p in self.proj:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    def forward(self, i: int, q: Tensor, phi: Tensor) -> Tensor:
        g, b = self.proj[i](phi.to(q.dtype)).chunk(2, dim=-1)
        return q * (1.0 + g) + b


# ================================================================================================================= #
# allocation: the most probable hypotheses get the extra candidates                                                 #
# ================================================================================================================= #
@torch.no_grad()
def allocate(logp_joint: Tensor, m_total: int, top_k: int = 4, min_each: int = 2,
             force: Tensor | None = None, force_min: int = 4) -> Tensor:
    """-> hypothesis ids [B, M] (joint ids), sorted by allocation order. The top-k hypotheses by p receive a floor
    (``min_each``; ``force_min`` for a forced one) and the remaining M - sum(floors) candidates are distributed in
    proportion to p over the selected set by largest remainder (ties: the more probable first). If the floors exceed M,
    the least probable non-forced hypotheses are dropped. ``force`` [B] (joint id or -1): training = the GT hypothesis
    (teacher forcing); eval = a forced condition. Deterministic given its inputs."""
    b, h = logp_joint.shape
    m = int(m_total)
    out = torch.empty(b, m, dtype=torch.long, device=logp_joint.device)
    p = logp_joint.detach().to(torch.float64).exp().cpu()
    k = max(1, min(int(top_k), h))
    for r in range(b):
        order = torch.argsort(p[r], descending=True, stable=True).tolist()
        sel = order[:k]
        f = int(force[r]) if (force is not None and int(force[r]) >= 0) else -1
        if f >= 0:
            if f in sel:
                sel.remove(f)
            sel = [f] + sel[: k - 1]
        floors = [min(int(force_min), m) if hh == f else int(min_each) for hh in sel]
        while sum(floors) > m and len(sel) > 1:
            j = len(sel) - 1 if sel[-1] != f else len(sel) - 2
            del sel[j], floors[j]
        if sum(floors) > m:
            floors = [m]
        rem = m - sum(floors)
        pr = [float(p[r, hh]) for hh in sel]
        tot = sum(pr) or 1.0
        quota = [rem * x / tot for x in pr]
        cnt = [fl + int(math.floor(q)) for fl, q in zip(floors, quota)]
        left = m - sum(cnt)
        frac = sorted(range(len(sel)), key=lambda j: (-(quota[j] - math.floor(quota[j])), -pr[j], j))
        for j in frac[:left]:
            cnt[j] += 1
        ids = [hh for hh, c in zip(sel, cnt) for _ in range(c)]
        out[r] = torch.tensor(ids, dtype=torch.long, device=logp_joint.device)
    return out


@torch.no_grad()
def select_source_anchors(tag_lat: Tensor, tag_lon: Tensor, hyp: Tensor, gen: R8Generator) -> tuple[Tensor, Tensor]:
    """For each allocated hypothesis [B, M] choose a source anchor whose own tag matches it: (lat, lon) exact, else
    lat only, else any. -> (src [B, M] anchor ids, match_level [B, M] 2 exact / 1 lat-only / 0 none). Uniform among the
    matches, drawn from the DEDICATED generator."""
    b, n = tag_lat.shape
    m = hyp.shape[1]
    hl, ho = split_joint(hyp)
    u = gen.rand((b, m, n), tag_lat.device)
    exact = (tag_lat[:, None, :] == hl[..., None]) & (tag_lon[:, None, :] == ho[..., None])
    latm = tag_lat[:, None, :] == hl[..., None]
    lvl = torch.where(exact.any(-1), 2, torch.where(latm.any(-1), 1, 0))
    mask = torch.where((lvl == 2)[..., None], exact, torch.where((lvl == 1)[..., None], latm,
                                                                  torch.ones_like(exact)))
    src = torch.where(mask, u, torch.full_like(u, -1.0)).argmax(-1)
    return src, lvl


def prior_free_lateral_offset(kappa0: Tensor, v: Tensor, n: int, s: int, control_norm_lat: float,
                              alat_v_floor: float) -> Tensor:
    """X2b: the x0 offset [B, n, s, 2] (NORMALISED control units) that cancels the residual prior's curvature on a
    candidate: delta_alat' = delta_alat - kappa0 * max(v, floor)^2. ⚠ Exact only where the vocabulary's +-kappa_cap
    clamp on the residual does not bind (stated, not hidden: kappa0 is capped at 0.3, the residual at 0.12)."""
    sc = v.reshape(-1).to(torch.float32).clamp_min(float(alat_v_floor)) ** 2
    lat = -(kappa0.reshape(-1).to(torch.float32) * sc) / float(control_norm_lat)
    off = torch.zeros(kappa0.shape[0], n, s, 2, device=kappa0.device)
    off[..., 1] = lat[:, None, None]
    return off


# ================================================================================================================= #
# selection                                                                                                         #
# ================================================================================================================= #
class FactorizedSelection(nn.Module):
    """``beta_lat log p(lat_k) + beta_lon log p(lon_k) + gamma_prog sat_prog_k + gamma_head sat_head_k`` -- every
    scalar ZERO-INIT (the term is exactly 0 at a warm start). sat_* are param-free functions of the candidate path and
    the PREDICTED constraint of the candidate's own hypothesis (D0's S-LON / S-HEAD with the real head in place)."""

    def __init__(self):
        super().__init__()
        self.beta_lat = nn.Parameter(torch.zeros(()))
        self.beta_lon = nn.Parameter(torch.zeros(()))
        self.gamma_prog = nn.Parameter(torch.zeros(()))
        self.gamma_head = nn.Parameter(torch.zeros(()))

    @staticmethod
    def sat_terms(paths: Tensor, cons_raw: Tensor, cons_valid: Tensor) -> tuple[Tensor, Tensor]:
        """``cons_raw`` [B, N, 4] = (psi_term rad, t_onset s, prog6 m, v_end m/s) -> (sat_prog, sat_head) [B, N],
        both <= 0, 0 = satisfied exactly; zero where the constraint is not valid."""
        p = paths.to(torch.float32)
        prog = _path_len(p)
        th = _terminal_heading(p)
        sp = -(torch.log1p(prog.clamp_min(0)) - torch.log1p(cons_raw[..., 2].clamp_min(0))).abs()
        sh = -_wrap(th - cons_raw[..., 0]).abs() / math.pi
        cv = cons_valid.to(torch.float32)
        return sp * cv, sh * cv

    def forward(self, lat3, lon6, logp_lat3, logp_lon6, paths, cons_raw, cons_valid) -> Tensor:
        lpl = torch.gather(logp_lat3, 1, lat3)
        lpo = torch.gather(logp_lon6, 1, lon6)
        sp, sh = self.sat_terms(paths, cons_raw, cons_valid)
        return (self.beta_lat * lpl + self.beta_lon * lpo + self.gamma_prog * sp + self.gamma_head * sh)


def denorm_cons(cons_n: Tensor) -> Tensor:
    """normalised [.., 4] (psi/pi, t/6, log1p(P)/log1p(120), v/30) -> raw (rad, s, m, m/s)."""
    return torch.stack([cons_n[..., 0] * math.pi, cons_n[..., 1] * T_SCALE, denorm_prog(cons_n[..., 2]),
                        cons_n[..., 3] * V_SCALE], -1)


SUBSCORES: tuple[str, ...] = ("dir_correct", "progress_10pct", "heading_15deg", "hypothesis_correct")


class SubScoreHeads(nn.Module):
    """X1h: Hydra-style per-candidate critics (2406.06978 Eq 8-11) on [q_k (emitting pass) | phi_k | 8 geometry
    features]. Combined into the score as ``sum_m w_m * logsigmoid(s_m)`` with ``w_m`` ZERO-INIT."""

    def __init__(self, d_q: int, d_phi: int = PHI_DIMS, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_q + d_phi + 8, hidden), nn.GELU(), nn.Linear(hidden, len(SUBSCORES)))
        self.w = nn.Parameter(torch.zeros(len(SUBSCORES)))

    @staticmethod
    def geometry(paths: Tensor) -> Tensor:
        p = paths.to(torch.float32)
        end = p[..., -1, :]
        mid = p[..., p.shape[-2] // 2 - 1, :]
        return torch.cat([end / 50.0, mid / 25.0, _path_len(p)[..., None] / 100.0,
                          _terminal_heading(p)[..., None] / math.pi,
                          torch.atan2(end[..., 1], end[..., 0].clamp_min(1e-3))[..., None] / math.pi,
                          (p[..., 3, :].norm(dim=-1) / 20.0)[..., None]], -1)

    def forward(self, q: Tensor, phi: Tensor, paths: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([q.to(torch.float32), phi.to(torch.float32), self.geometry(paths)], -1)
        logits = self.mlp(x)                                                    # [B, N, 4]
        return logits, (F.logsigmoid(logits) * self.w).sum(-1)


@torch.no_grad()
def subscore_targets(paths: Tensor, gt: Tensor, gt_valid: Tensor, cand_lat3: Tensor, gt_lat3: Tensor,
                     tau_dir: float = 0.18063741505146028) -> tuple[Tensor, Tensor]:
    """-> (y [B, N, 4] in {0, 1}, w [B, N, 4] validity). dir_correct: terminal-heading direction class (route
    definition, tau 10.35 deg) equals GT's; progress_10pct: |P_k / P_gt - 1| <= 0.10; heading_15deg: |wrap(theta_k -
    theta_gt)| <= 15 deg; hypothesis_correct: the candidate's lat3 tag equals the window's lat3 label."""
    p = paths.to(torch.float32)
    g = gt.to(torch.float32)
    ok = gt_valid[:, -1].to(torch.bool)
    th_k, th_g = _terminal_heading(p), _terminal_heading(g)[:, None]

    def dcls(t):
        return torch.where(t >= tau_dir, 1, torch.where(t <= -tau_dir, -1, 0))
    pk, pg = _path_len(p), _path_len(g)[:, None]
    y = torch.stack([(dcls(th_k) == dcls(th_g)).float(),
                     ((pk / pg.clamp_min(1e-6) - 1.0).abs() <= 0.10).float(),
                     (_wrap(th_k - th_g).abs() <= math.radians(15.0)).float(),
                     (cand_lat3 == gt_lat3[:, None]).float()], -1)
    w = torch.stack([ok[:, None].expand_as(pk), (ok[:, None] & (pg > 0.5)).expand_as(pk), ok[:, None].expand_as(pk),
                     (gt_lat3 >= 0)[:, None].expand_as(pk)], -1).float()
    return y, w


def listwise_selection_loss(score: Tensor, fan: Tensor, gt: Tensor, gv: Tensor, keep: Tensor, *, t: float = 1.0,
                            speed_scale_m: float = 1.0, dir_scale_deg: float = 15.0) -> Tensor:
    """X1 (R1-H4's definition, SPEC_R1 sec. 3): cost_k = mean over valid slots |s_k(j) - s_GT(j)| / 1 m (s = cumulative
    arc length: the SPEED PROFILE) + |wrap(theta_k - theta_GT)| / 15 deg (only where slot S is valid);
    p* ∝ exp(-cost / t) over ``keep``; loss = -sum p* log softmax(score) over ``keep``. Rows with no valid slot drop.
    Hydra's soft imitation target (2406.06978 Eq 8) / TNT's psi (2008.08294 Eq 5-6) with this distance."""
    with torch.no_grad():
        v = gv.to(torch.float32)
        n = v.sum(-1)

        def cum(p):
            q = torch.cat([torch.zeros_like(p[..., :1, :]), p], -2)
            return torch.linalg.vector_norm(q[..., 1:, :] - q[..., :-1, :], dim=-1).cumsum(-1)
        sk = cum(fan.to(torch.float32))
        sg = cum(gt.to(torch.float32))[:, None]
        e_sp = ((sk - sg).abs() * v[:, None]).sum(-1) / n.clamp_min(1)[:, None] / float(speed_scale_m)
        dth = _wrap(_terminal_heading(fan.to(torch.float32)) - _terminal_heading(gt.to(torch.float32))[:, None])
        e_dir = dth.abs() / math.radians(dir_scale_deg) * gv[:, -1:].to(torch.float32)
        cost = (e_sp + e_dir).masked_fill(~keep, float("inf"))
        tgt = torch.softmax(-cost / float(t), dim=-1)
        ok = n > 0
    logp = torch.log_softmax(score.to(torch.float32).masked_fill(~keep, float("-inf")), dim=-1)
    row = -(tgt * logp.masked_fill(~keep, 0.0)).sum(-1)
    return row[ok].mean() if bool(ok.any()) else score.sum() * 0.0


def constraint_head_loss(cons_lat8: Tensor, cons_lon8: Tensor, gt_lat_v7: Tensor, gt_lon_v7: Tensor,
                         lat_t: Tensor, lat_v: Tensor, lon_t: Tensor, lon_v: Tensor) -> tuple[Tensor, dict]:
    """Smooth-L1 on the GT-ACTIVE class's query only (the constraint "if this action" of a non-GT class has no label and
    is never taught). ``gt_*_v7`` [B] v7 ids (-100 = no label); targets/validity in the normalised units of
    :func:`normalise_constraints`."""
    b = cons_lat8.shape[0]
    ar = torch.arange(b, device=cons_lat8.device)
    okl = gt_lat_v7 >= 0
    oko = gt_lon_v7 >= 0
    pl = cons_lat8[ar, gt_lat_v7.clamp_min(0)]
    po = cons_lon8[ar, gt_lon_v7.clamp_min(0)]
    ml = lat_v.to(torch.float32) * okl[:, None].to(torch.float32)
    mo = lon_v.to(torch.float32) * oko[:, None].to(torch.float32)
    ll = (F.smooth_l1_loss(pl.float(), torch.nan_to_num(lat_t.float()), reduction="none", beta=0.05) * ml).sum()
    lo = (F.smooth_l1_loss(po.float(), torch.nan_to_num(lon_t.float()), reduction="none", beta=0.05) * mo).sum()
    n = (ml.sum() + mo.sum()).clamp_min(1.0)
    return (ll + lo) / n, {"n_cons_lat": int(ml.sum().item()), "n_cons_lon": int(mo.sum().item())}


# ================================================================================================================= #
# MM (B) 2026-10-04: the 8-step N2 encoding (`--r8-speed-enc8`)                                                     #
# ================================================================================================================= #
#: the road-law ladder N2 is snapped onto (WP-A SPEC_ADDENDUM_S3A1 item 4), LITERAL
SPEED_ENC8_LADDER_KMH: tuple[int, ...] = (20, 30, 50, 70, 80, 100, 120, 130)
SPEED_ENC8_DIMS = len(SPEED_ENC8_LADDER_KMH) + 1          # 8 one-hot + the known bit
#: the tolerance of the step match, m/s (= 0.00036 km/h): the fed value is km/h / 3.6 computed ONCE in float64
SPEED_ENC8_TOL_MS = 1e-4


def speed_enc8(v_ms: Tensor, valid: Tensor | None) -> Tensor:
    """``v_ms`` [B] (the FED value, m/s) + ``valid`` [B] -> [B, 9] = (one-hot over the 8-step ladder, known). The step
    is the LOWEST ladder value >= v (the containing window, as N2 itself is built); above 130 km/h -> the top step. An
    unknown row is exactly zeros next to a 0 (the X15 rule)."""
    v = v_ms.reshape(-1).to(torch.float32)
    ok = torch.ones_like(v) if valid is None else valid.reshape(-1).to(torch.float32)
    steps = torch.tensor([k / 3.6 for k in SPEED_ENC8_LADDER_KMH], device=v.device, dtype=torch.float32)
    ge = v[:, None] <= steps[None, :] + SPEED_ENC8_TOL_MS
    idx = torch.where(ge.any(-1), ge.to(torch.long).argmax(-1), torch.full_like(v, len(steps) - 1, dtype=torch.long))
    oh = F.one_hot(idx, len(steps)).to(torch.float32) * ok[:, None]
    return torch.cat([oh, ok[:, None]], -1)


# ================================================================================================================= #
# the v9 CONSTRAINT vectors (MM ruling Q2, SPEC_REFCV8 sec. 13a): supervised on the GT-active action query            #
# ================================================================================================================= #
#: field order = `tanitad.data.v9_labels.LAT_CONSTRAINTS` / `LON_CONSTRAINTS` / `SPEED_GOAL` (pinned against the reader)
V9_LAT_FIELDS: tuple[str, ...] = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m",
                                  "turn_dyaw_deg", "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m",
                                  "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m")
V9_LON_FIELDS: tuple[str, ...] = ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m",
                                  "stop_y_m", "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")
V9_SPEED_FIELDS: tuple[str, ...] = ("v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
#: WP-A INTEGRATION.md sec. 3.1's recommended scales -- distances / 50 m, times / 8 s, angles / 90 deg, speeds / 15 m/s
#: -- as LITERALS per field; then clamped to +-V9_CLAMP. The tails are real (MEASURED on the train release: turn length
#: to 470 m, |turn dpsi| to 505 deg, lead time gap to 95 s); the clamp bounds them (300 m, 48 s, 540 deg, 90 m/s).
V9_LAT_SCALE: tuple[float, ...] = (90.0, 8.0, 8.0, 50.0, 50.0, 90.0, 50.0, 50.0, 50.0, 90.0, 8.0, 50.0)
V9_LON_SCALE: tuple[float, ...] = (15.0, 8.0, 50.0, 8.0, 50.0, 50.0, 50.0, 8.0, 50.0, 8.0)
V9_SPEED_SCALE: tuple[float, ...] = (15.0, 15.0, 15.0, 15.0)
V9_CLAMP = 6.0
#: INTEGRATION sec. 3.1: "supervise the turn fields only when the class is a TURN" -- columns 1..9, v7 TURN_L / TURN_R
V9_TURN_COLS: tuple[int, ...] = tuple(range(1, 10))
V9_TURN_V7: tuple[int, ...] = (6, 7)
#: Huber knee in normalised units (= 5 m, 0.8 s, 9 deg, 1.5 m/s)
V9_HUBER_BETA = 0.1
V9_LAT_DIMS, V9_LON_DIMS, V9_SPEED_DIMS = len(V9_LAT_FIELDS), len(V9_LON_FIELDS), len(V9_SPEED_FIELDS)


def _v9_norm(x: Tensor, scale) -> tuple[Tensor, Tensor]:
    x = x.to(torch.float32)
    fin = torch.isfinite(x)
    s = torch.tensor(scale, device=x.device, dtype=torch.float32)
    y = torch.nan_to_num((x / s).clamp(-V9_CLAMP, V9_CLAMP), nan=0.0, posinf=0.0, neginf=0.0)
    return y, fin


def v9_constraint_targets(lat_c: Tensor, lon_c: Tensor, speed: Tensor, lat_cls: Tensor,
                          lon_cls: Tensor) -> dict[str, Tensor]:
    """Raw v9 vectors ``lat_c`` [B, 12], ``lon_c`` [B, 10], ``speed`` [B, 4] (NaN = undefined for the row) and the v9
    classes in frozen v7 ids (``lat_cls`` / ``lon_cls`` [B], -100 = PARTIAL or absent) -> normalised targets + bool masks.

    MM ruling Q2: masked where the v9 row is PARTIAL or absent -- the lateral vector needs an EXACT lateral class (it is
    read off that class's query), the longitudinal and speed vectors an EXACT longitudinal class; within a row a field
    is supervised where it is finite (INTEGRATION sec. 3.1), and the turn fields only on a TURN class."""
    lat_cls = lat_cls.reshape(-1).to(torch.long)
    lon_cls = lon_cls.reshape(-1).to(torch.long)
    tl, fl = _v9_norm(lat_c, V9_LAT_SCALE)
    to, fo = _v9_norm(lon_c, V9_LON_SCALE)
    ts, fs = _v9_norm(speed, V9_SPEED_SCALE)
    ml = fl & (lat_cls >= 0)[:, None]
    is_turn = torch.zeros_like(lat_cls, dtype=torch.bool)
    for c in V9_TURN_V7:
        is_turn |= lat_cls == c
    tcols = torch.zeros(V9_LAT_DIMS, dtype=torch.bool, device=ml.device)
    tcols[list(V9_TURN_COLS)] = True
    ml = ml & (~tcols[None, :] | is_turn[:, None])
    mo = fo & (lon_cls >= 0)[:, None]
    ms = fs & (lon_cls >= 0)[:, None]
    return {"lat": tl, "lat_m": ml, "lon": to, "lon_m": mo, "speed": ts, "speed_m": ms,
            "lat_cls": lat_cls, "lon_cls": lon_cls}


def v9_constraint_loss(pred_lat8: Tensor, pred_lon8: Tensor, pred_speed8: Tensor,
                       tg: Mapping[str, Tensor]) -> tuple[Tensor, dict]:
    """Huber (``V9_HUBER_BETA``) on the GT-ACTIVE class's query: ``lat_c`` from the lateral query of the row's v9 lateral
    class, ``lon_c`` and ``speed_goal`` from the longitudinal query of its v9 longitudinal class (the
    ``constraint_head_loss`` rule: the constraint "if this action" of a class that did not happen has no label). Mean
    over the supervised ENTRIES of the three vectors; a batch with none -> an ATTACHED zero (never guarded)."""
    b = pred_lat8.shape[0]
    ar = torch.arange(b, device=pred_lat8.device)
    pl = pred_lat8[ar, tg["lat_cls"].clamp_min(0)].float()
    po = pred_lon8[ar, tg["lon_cls"].clamp_min(0)].float()
    ps = pred_speed8[ar, tg["lon_cls"].clamp_min(0)].float()
    tot = pl.new_zeros(())
    n = 0.0
    tele: dict[str, Any] = {}
    for nm, p, t, m in (("lat_c", pl, tg["lat"], tg["lat_m"]), ("lon_c", po, tg["lon"], tg["lon_m"]),
                        ("speed", ps, tg["speed"], tg["speed_m"])):
        w = m.to(torch.float32)
        tot = tot + (F.smooth_l1_loss(p, t.to(p), reduction="none", beta=V9_HUBER_BETA) * w).sum()
        k = float(w.sum().item())
        n += k
        tele[f"n_v9_{nm}"] = int(k)
        tele[f"r8_v9_mae_{nm}"] = (((p - t.to(p)).abs() * w).sum() / max(k, 1.0)).detach()
    loss = tot / max(n, 1.0) if n > 0 else (pred_lat8.sum() + pred_lon8.sum() + pred_speed8.sum()) * 0.0
    return loss, tele


def constraint_satisfaction_loss(paths: Tensor, cons_raw: Tensor, cons_valid: Tensor, mask: Tensor,
                                 prog_tol: float = math.log(1.10), head_tol_deg: float = 15.0) -> Tensor:
    """L_sat on the CONDITIONED candidates (``mask`` [B, N]: the allocated ones): hinge on the candidate's own path
    obeying the constraint it was generated under -- relu(|log1p P_k - log1p P̂| - log 1.1) + relu(|wrap(theta_k -
    psî)| - 15 deg) / pi. ⛔ Never on the base fan: forcing every anchor of a mode onto one predicted progress would
    collapse the within-mode speed diversity the B3 headroom lives in (DESIGN sec. 3.1.1)."""
    p = paths.to(torch.float32)
    prog = _path_len(p)
    th = _terminal_heading(p)
    ep = (torch.log1p(prog.clamp_min(0)) - torch.log1p(cons_raw[..., 2].detach().clamp_min(0))).abs()
    eh = _wrap(th - cons_raw[..., 0].detach()).abs()
    term = F.relu(ep - prog_tol) + F.relu(eh - math.radians(head_tol_deg)) / math.pi
    w = mask.to(torch.float32) * cons_valid.to(torch.float32)
    return (term * w).sum() / w.sum().clamp_min(1.0)


#: the five situation outputs (refcv6_tactical.SITUATION_OUTPUT_TOKENS) -- never a feed here
SITUATION_TOKENS = frozenset({"YIELD", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW",
                              "TRAFFIC_LIGHT_REACT_GREEN"})


def assert_no_situation_feed(hypothesis_names: Sequence[str] = LAT3 + LON6) -> None:
    """The hypothesis space (the ONLY tactical content phi / allocation / FactorizedSelection read) holds no situation
    output. Called at build; a mutation test appends a TL token and expects a refusal."""
    bad = sorted(set(hypothesis_names) & SITUATION_TOKENS)
    if bad:
        raise ValueError(f"[refcv8] situation-classifier outputs {bad} in the conditioning hypothesis space -- "
                         f"refused (PI 2026-08-03: a goal input must not carry the situation classifier's output)")


# ================================================================================================================= #
# RL channel admissibility (`tanitad.channel_admissibility`): the two refcv8 forward INPUTS an RL adapter does not   #
# plumb yet, declared by their owner with the route back                                                           #
# ================================================================================================================= #
_R8_REFS = ("TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/DESIGN.md",
            "FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/SPEC.md")
FORWARD_EXCLUSIONS: tuple[ChannelExclusion, ...] = (
    ChannelExclusion(
        channel="r8_nav",
        owner="tanitad.refs.refcv8_conditioning (refcv8 WP-B, R8-2 nav arguments)",
        reason=("An RL rollout would have to take the per-frame announced-turn arguments (distance and heading "
                "change to the next announced turn, lookahead, validity) from a ROUTE: on PhysicalAI that route is "
                "the ego's own future path (an ego-future oracle, v9 SPEC sec. 8) and in a closed-loop simulator "
                "it is the scene's route, which no refcv8 RL adapter computes yet. Feeding the logged value to a "
                "rollout that departs from the log would hand the policy the logged driver's route. TEMPORARY."),
        unblock=("An RL arm on refcv8 with a route source that follows the ROLLOUT (the simulator's route "
                 "centreline map-matched to the rolled-out pose -- the EvalFlyWheel NavSim bridge contract, v9 "
                 "SPEC sec. 6.4); then add `r8_nav` to `tanitad.rl.refc_adapter.FORWARD_KEYS` with a "
                 "CHANNEL_REQUIREMENTS record REQUIRED when the build's `r8_enabled` is true."),
        evidence=("DESIGN (refcv8 WP-B DESIGN.md sec. 3.3); v9 SPEC sec. 5.2 and sec. 8 (the field list and its "
                  "ego-future admissibility stamp). MEASURED: `tests/test_refcv8_warm_start.py` pins the "
                  "silent-drop refusal of a supplied r8 input on a build without the seams."),
        permanent=False, refs=_R8_REFS),
    ChannelExclusion(
        channel="r8_rc",
        owner="tanitad.refs.refcv8_conditioning (refcv8 WP-B, R8-3 route checkpoint)",
        reason=("An RL rollout would have to take the route checkpoint (the next reference route point in the NOW "
                "vehicle frame) from a ROUTE: on PhysicalAI it is the smoothed ego future path, so the logged "
                "point is wrong as soon as the rollout leaves the log -- and it is the target-point input whose "
                "shortcut Hidden Biases (2306.07957) documents. No refcv8 RL adapter computes it from the "
                "rolled-out pose yet. TEMPORARY."),
        unblock=("An RL arm on refcv8 that recomputes the checkpoint every step from the rollout's own pose against "
                 "a fixed route (v9 SPEC sec. 6.4 NavSim bridge contract), with the RC-OFF row reported beside it; "
                 "then plumb `r8_rc` in `tanitad.rl.refc_adapter.FORWARD_KEYS` behind `r8_enabled`."),
        evidence=("DESIGN (refcv8 WP-B DESIGN.md sec. 3.3, the echo controls E-1..E-4); v9 SPEC sec. 6 and sec. 8. "
                  "PUBLISHED: 2306.07957 p1 and Tab. 1 p3 (the target-point shortcut)."),
        permanent=False, refs=_R8_REFS),
)
