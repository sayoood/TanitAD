"""refcv7 — the PhysicalAI ORACLE that the disentangled scorer learns to imitate.

DrivoR (arXiv 2601.05083) trains its scorer against NAVSIM's PDM oracle, which needs an HD
map and ground-truth agents. PhysicalAI has no HD map, so refcv7 builds its own oracle from
what the programme DOES hold, all of it LABEL-SIDE:

* ``obstacle.offline`` agent boxes (the 3-D agent join)  -> NC (no collision), TTC
* the SAM3 semantic map raster at the window's t0 frame  -> DAC (drivable-area compliance)
* the GT future ego path                                  -> EP (ego progress, vs the human)
* the candidate's own kinematics                          -> COMF (NAVSIM comfort limits)
* the max-speed INPUT (PI 2026-09-16, a 4-value set speed) -> SPD (set-speed compliance)
* the nav command (MANDATORY, PI 2026-09-16)               -> NAV (terminal heading obeys the
  commanded side; the programme's own predicate ``refc_selector_targets.compliance_target``,
  defined ONLY on LEFT/RIGHT — follow/straight abstain, they never "fail")

⛔ **Everything here is a LABEL.** It reads future agents, a non-causal map (``label_only``
in the map's own meta) and the GT path. The PI's 2026-08-03 ruling allows labels to use
anything and inference to use vision only; this module must therefore never be called on the
inference path. :func:`oracle_subscores` is only ever handed to the scorer's LOSS.

⭐ **Why the oracle exists at all (the TOAD lesson, arXiv 2606.07170):** a scorer that is only
ever shown the planner's own proposals is accurate on them and wrong everywhere else, and it
then fails as a test-time search reward (GTRS's scorer as a CEM reward: 34.7 -> 23.9 EPDMS).
An oracle that can score ANY trajectory lets the trainer show the scorer perturbed,
off-proposal candidates as well. That is what makes the scorer usable by ``refcv7_toad``.

Conventions (all MEASURED from the artifacts, not assumed):
* ego frame at t0: +x forward, +y LEFT, metres (``sam3_map_gt/2`` meta ``frame_convention``);
* map cartesian raster: x in [0, 60) m by row (row 0 = x in [0, 0.5)), y in [-16, 16] m by
  column (column 0 = y = -16 m, i.e. RIGHT), 0.5 m cells, 120 x 64;
* waypoint slots at the V3 horizons (5, 10, 15, 20, 30, 40, 50, 60) x 0.1 s -> NON-uniform dt.

NAVSIM comfort limits: ``navsim/planning/simulation/planner/pdm_planner/scoring/
pdm_comfort_metrics.py`` (read 2026-09-19): MAX_ABS_LAT_ACCEL 4.89, MAX_LON_ACCEL 2.40,
MIN_LON_ACCEL -4.05, MAX_ABS_YAW_RATE 0.95, MAX_ABS_YAW_ACCEL 1.93, MAX_ABS_LON_JERK 4.13,
MAX_ABS_MAG_JERK 8.37.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

#: the sub-score names, in the order every tensor in refcv7 uses
SUBSCORES: tuple[str, ...] = ("nc", "dac", "ttc", "ep", "comf", "spd", "nav")
#: MULTIPLICATIVE gates (any 0 -> the trajectory is unacceptable), as in PDMS.
#: ``nav`` gates ONLY where the command has a side (see :func:`aggregate`).
GATES: tuple[str, ...] = ("nc", "dac", "spd", "nav")
#: ADDITIVE quality terms, as in PDMS (TTC / EP / comfort are weighted and averaged)
QUALITY: tuple[str, ...] = ("ttc", "ep", "comf")

# NAVSIM pdm_comfort_metrics.py (PUBLISHED-CODE, read 2026-09-19)
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL = 2.40
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_RATE = 0.95
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13


@dataclass
class RasterSpec:
    """The SAM3 ``cart`` raster geometry (``sam3_map_gt/2`` meta)."""
    x_max_m: float = 60.0
    y_half_m: float = 16.0
    cell_m: float = 0.5
    drivable_threshold: float = 0.5   # fraction of the cell that is class "drivable"


@dataclass
class OracleConfig:
    ego_length_m: float = 4.9
    ego_width_m: float = 2.0
    n_circles: int = 3                 # circle cover of each box along its length
    ttc_horizon_s: float = 1.0         # PDMS time-to-collision look-ahead
    ttc_substeps: int = 3
    speed_margin_ms: float = 0.5       # SPD tolerance above the set speed
    progress_floor_m: float = 1.0      # EP is 1 when the human barely moved
    raster: RasterSpec = field(default_factory=RasterSpec)


# ---------------------------------------------------------------------------
# kinematics of a waypoint path (non-uniform slots)
# ---------------------------------------------------------------------------
def path_kinematics(xy: Tensor, slot_t: Tensor, v0: Tensor) -> dict[str, Tensor]:
    """Per-slot speed, heading, longitudinal/lateral acceleration, yaw rate and jerk.

    ``xy`` [..., S, 2] waypoints at times ``slot_t`` [S] (seconds, strictly increasing,
    first > 0), starting from the origin with speed ``v0`` [...] along +x.
    Finite differences on the NON-uniform grid; the origin is prepended at t = 0.
    """
    lead = xy.shape[:-2]
    zero = xy.new_zeros(*lead, 1, 2)
    p = torch.cat([zero, xy], dim=-2)                                  # [..., S+1, 2]
    t = torch.cat([slot_t.new_zeros(1), slot_t]).to(xy)              # [S+1]
    dt = (t[1:] - t[:-1]).clamp_min(1e-3)                              # [S]
    d = p[..., 1:, :] - p[..., :-1, :]                                 # [..., S, 2]
    seg_v = d.norm(dim=-1) / dt                                        # [..., S]
    heading = torch.atan2(d[..., 1], d[..., 0])                        # [..., S]
    # a near-zero segment has no heading: carry the previous one (heading 0 at the start)
    moving = d.norm(dim=-1) > 1e-3
    h = heading.clone()
    prev = torch.zeros_like(h[..., 0])
    for k in range(h.shape[-1]):
        h[..., k] = torch.where(moving[..., k], h[..., k], prev)
        prev = h[..., k]
    v_all = torch.cat([v0.to(xy)[..., None].expand(*lead, 1), seg_v], dim=-1)   # [..., S+1]
    h_all = torch.cat([torch.zeros_like(h[..., :1]), h], dim=-1)
    tm = t                                                                      # [S+1]
    dtm = (tm[1:] - tm[:-1]).clamp_min(1e-3)
    a_lon = (v_all[..., 1:] - v_all[..., :-1]) / dtm                           # [..., S]
    dh = torch.atan2(torch.sin(h_all[..., 1:] - h_all[..., :-1]),
                     torch.cos(h_all[..., 1:] - h_all[..., :-1]))
    yaw_rate = dh / dtm                                                         # [..., S]
    a_lat = seg_v * yaw_rate
    jerk = torch.zeros_like(a_lon)
    jerk[..., 1:] = (a_lon[..., 1:] - a_lon[..., :-1]) / dtm[1:]
    yaw_acc = torch.zeros_like(yaw_rate)
    yaw_acc[..., 1:] = (yaw_rate[..., 1:] - yaw_rate[..., :-1]) / dtm[1:]
    return {"v": seg_v, "heading": h, "a_lon": a_lon, "a_lat": a_lat,
            "yaw_rate": yaw_rate, "jerk": jerk, "yaw_acc": yaw_acc}


def comfort_score(kin: dict[str, Tensor]) -> Tensor:
    """1 if every slot is inside NAVSIM's comfort limits, else 0 (``[...]``)."""
    ok = ((kin["a_lat"].abs() <= MAX_ABS_LAT_ACCEL)
          & (kin["a_lon"] <= MAX_LON_ACCEL) & (kin["a_lon"] >= MIN_LON_ACCEL)
          & (kin["yaw_rate"].abs() <= MAX_ABS_YAW_RATE)
          & (kin["yaw_acc"].abs() <= MAX_ABS_YAW_ACCEL)
          & (kin["jerk"].abs() <= MAX_ABS_LON_JERK))
    return ok.all(dim=-1).to(kin["v"].dtype)


def comfort_violation(kin: dict[str, Tensor]) -> Tensor:
    """Squared exceedance of the comfort limits, summed over slots — a SMOOTH
    penalty for the test-time search (TOAD's closed-form ``C_comf``)."""
    def over(x, hi):
        return torch.relu(x - hi) ** 2
    pen = (over(kin["a_lat"].abs(), MAX_ABS_LAT_ACCEL)
           + over(kin["a_lon"], MAX_LON_ACCEL) + over(-kin["a_lon"], -MIN_LON_ACCEL)
           + over(kin["yaw_rate"].abs(), MAX_ABS_YAW_RATE)
           + over(kin["yaw_acc"].abs(), MAX_ABS_YAW_ACCEL)
           + over(kin["jerk"].abs(), MAX_ABS_LON_JERK))
    return pen.sum(dim=-1)


# ---------------------------------------------------------------------------
# collision by circle cover
# ---------------------------------------------------------------------------
def _circles(center: Tensor, yaw: Tensor, length: Tensor, width: Tensor, n: int
             ) -> tuple[Tensor, Tensor]:
    """Cover an oriented box with ``n`` circles along its length.

    Returns centres [..., n, 2] and radius [...]. Radius = half-diagonal of one of the
    ``n`` equal sub-boxes, so the cover is CONSERVATIVE (it contains the box).
    """
    seg = length / n
    offs = (torch.arange(n, device=center.device, dtype=center.dtype) + 0.5) / n - 0.5
    along = offs * length[..., None]                                   # [..., n]
    c = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)          # [..., 2]
    centres = center[..., None, :] + along[..., None] * c[..., None, :]
    radius = 0.5 * torch.sqrt(seg ** 2 + width ** 2)
    return centres, radius


def collision_free(ego_xy: Tensor, ego_yaw: Tensor, agents: Tensor | None,
                   agents_valid: Tensor | None, cfg: OracleConfig,
                   forward_ext_m: Tensor | None = None) -> Tensor:
    """1 if the ego never overlaps any agent at any slot, else 0.

    ``ego_xy`` [B, M, S, 2], ``ego_yaw`` [B, M, S]; ``agents`` [B, A, S, 5] =
    (x, y, length, width, yaw) at the SAME slot times; ``agents_valid`` [B, A, S].
    ``forward_ext_m`` [B, M, S] extends the ego box forward (TTC).
    Returns [B, M].
    """
    B, M, S, _ = ego_xy.shape
    if agents is None or agents.shape[1] == 0:
        return ego_xy.new_ones(B, M)
    L = torch.full_like(ego_yaw, cfg.ego_length_m)
    W = torch.full_like(ego_yaw, cfg.ego_width_m)
    ctr = ego_xy
    if forward_ext_m is not None:
        L = L + forward_ext_m
        head = torch.stack([torch.cos(ego_yaw), torch.sin(ego_yaw)], dim=-1)
        ctr = ego_xy + 0.5 * forward_ext_m[..., None] * head
    n = cfg.n_circles + (0 if forward_ext_m is None else cfg.ttc_substeps)
    ec, er = _circles(ctr, ego_yaw, L, W, n)                           # [B,M,S,n,2], [B,M,S]
    ac, ar = _circles(agents[..., :2], agents[..., 4], agents[..., 2], agents[..., 3],
                      cfg.n_circles)                                    # [B,A,S,m,2], [B,A,S]
    # pairwise over (ego circle, agent circle) per slot
    d = (ec[:, :, None, :, :, None, :] - ac[:, None, :, :, None, :, :]).norm(dim=-1)
    # d: [B, M, A, S, n, m]
    thr = er[:, :, None, :, None, None] + ar[:, None, :, :, None, None]
    hit = (d < thr).any(dim=-1).any(dim=-1)                            # [B, M, A, S]
    if agents_valid is not None:
        hit = hit & agents_valid[:, None, :, :]
    return (~hit.any(dim=-1).any(dim=-1)).to(ego_xy.dtype)


# ---------------------------------------------------------------------------
# drivable-area compliance on the SAM3 raster
# ---------------------------------------------------------------------------
def footprint_points(xy: Tensor, yaw: Tensor, cfg: OracleConfig) -> Tensor:
    """Centre + 4 corners of the ego box at every slot: [..., S, 5, 2]."""
    hl, hw = cfg.ego_length_m / 2, cfg.ego_width_m / 2
    local = xy.new_tensor([[0, 0], [hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    px = xy[..., None, 0] + c * local[:, 0] - s * local[:, 1]
    py = xy[..., None, 1] + s * local[:, 0] + c * local[:, 1]
    return torch.stack([px, py], dim=-1)


def drivable_compliance(xy: Tensor, yaw: Tensor, drivable: Tensor | None,
                        seen: Tensor | None, cfg: OracleConfig
                        ) -> tuple[Tensor, Tensor]:
    """DAC and its validity mask, both [B, M].

    ``drivable`` [B, H, W] = fraction of each cell that is class "drivable";
    ``seen`` [B, H, W] bool. A footprint point that falls outside the raster or on an
    UNSEEN cell is ABSTAINED on (it can neither pass nor fail the trajectory). DAC = 1
    iff every counted point is on drivable ground; the mask is False when no point
    could be counted — never a silent pass.
    """
    B, M = xy.shape[:2]
    if drivable is None:
        return xy.new_ones(B, M), torch.zeros(B, M, dtype=torch.bool, device=xy.device)
    r = cfg.raster
    H, W = drivable.shape[-2:]
    pts = footprint_points(xy, yaw, cfg)                               # [B,M,S,5,2]
    row = torch.floor(pts[..., 0] / r.cell_m).long()
    col = torch.floor((pts[..., 1] + r.y_half_m) / r.cell_m).long()
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    rc = row.clamp(0, H - 1)
    cc = col.clamp(0, W - 1)
    bidx = torch.arange(B, device=xy.device)[:, None, None, None].expand_as(rc)
    frac = drivable[bidx, rc, cc]
    counted = inside
    if seen is not None:
        counted = counted & seen[bidx, rc, cc]
    on = frac >= r.drivable_threshold
    bad = counted & ~on
    n_counted = counted.flatten(2).sum(-1)
    dac = (~bad.flatten(2).any(-1)).to(xy.dtype)
    return dac, n_counted > 0


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------
def oracle_subscores(traj: Tensor, slot_t: Tensor, v0: Tensor, gt: Tensor,
                     gt_valid: Tensor | None = None,
                     agents: Tensor | None = None, agents_valid: Tensor | None = None,
                     drivable: Tensor | None = None, seen: Tensor | None = None,
                     max_speed_ms: Tensor | None = None,
                     nav_cmd: Tensor | None = None, nav_tau_rad: float | None = None,
                     cfg: OracleConfig | None = None) -> dict[str, Tensor]:
    """Oracle sub-scores in [0, 1] and their validity masks for candidates ``traj``.

    ``traj`` [B, M, S, 2] (ego frame at t0), ``slot_t`` [S] seconds, ``v0`` [B],
    ``gt`` [B, S, 2]. Optional label sources as documented above.
    Returns ``{name: [B, M]}`` for every name in :data:`SUBSCORES` plus
    ``{name + "_mask": bool [B, M]}``.
    """
    cfg = cfg or OracleConfig()
    B, M, S, _ = traj.shape
    with torch.no_grad():
        kin = path_kinematics(traj, slot_t, v0[:, None].expand(B, M))
        yaw = kin["heading"]
        out: dict[str, Tensor] = {}
        ones = torch.ones(B, M, dtype=torch.bool, device=traj.device)
        has_agents = agents is not None and agents.shape[1] > 0
        out["nc"] = collision_free(traj, yaw, agents, agents_valid, cfg)
        out["nc_mask"] = ones if has_agents else ~ones
        ext = kin["v"] * cfg.ttc_horizon_s
        out["ttc"] = collision_free(traj, yaw, agents, agents_valid, cfg, forward_ext_m=ext)
        out["ttc_mask"] = out["nc_mask"].clone()
        out["dac"], out["dac_mask"] = drivable_compliance(traj, yaw, drivable, seen, cfg)
        # EP: progress along the path relative to the human's
        def arclen(p):
            z = p.new_zeros(*p.shape[:-2], 1, 2)
            return (torch.cat([z, p], -2).diff(dim=-2).norm(dim=-1)).sum(-1)
        s_c = arclen(traj)
        g = gt if gt_valid is None else torch.where(gt_valid[..., None], gt, gt.new_zeros(()))
        s_g = arclen(g)[:, None].expand(B, M)
        ep = torch.where(s_g < cfg.progress_floor_m, torch.ones_like(s_c),
                         (s_c / s_g.clamp_min(1e-3)).clamp(0.0, 1.0))
        out["ep"] = ep
        out["ep_mask"] = ones
        out["comf"] = comfort_score(kin)
        out["comf_mask"] = ones
        if max_speed_ms is None:
            out["spd"] = traj.new_ones(B, M)
            out["spd_mask"] = ~ones
        else:
            vmax = kin["v"].amax(dim=-1)
            out["spd"] = (vmax <= max_speed_ms[:, None] + cfg.speed_margin_ms).to(traj.dtype)
            out["spd_mask"] = ones
        if nav_cmd is None or not nav_tau_rad:
            out["nav"] = traj.new_ones(B, M)
            out["nav_mask"] = ~ones
        else:
            # ⛔⛔ TIMING, NOT ONLY SIDE. MEASURED 2026-09-19 on eval-139 split A
            # (`refcv7_derive_nav_tau.py`): the nav token is PER CLIP, and on
            # LEFT/RIGHT windows the median |GT terminal heading| at 6 s is only
            # 0.048 rad (Youden J 0.22) — the commanded turn is usually OUTSIDE
            # the 6 s plan. A side-only target ("turns left when told left")
            # would therefore reward turning EARLY and penalise the human's own
            # straight path. The label is instead "the candidate's commanded-
            # side turn matches whether the HUMAN turned within the horizon":
            #   due  = GT turns >= tau on the commanded side  (label-side: GT)
            #   ok   = candidate turns >= tau on the commanded side
            #   nav  = (ok == due)
            # FOLLOW / STRAIGHT carry no side and abstain (never "fail").
            side = _nav_side(nav_cmd).to(traj)                         # [B]
            th_c = terminal_heading_t(traj)                            # [B, M]
            th_g = terminal_heading_t(gt[:, None])[:, 0]               # [B]
            tau = float(nav_tau_rad)
            ok = (side[:, None] * th_c) >= tau
            due = ((side * th_g) >= tau)[:, None]
            out["nav"] = (ok == due).to(traj.dtype)
            out["nav_mask"] = (side != 0)[:, None].expand(B, M).clone()
            out["nav_due"] = due.expand(B, M).clone()
    return out


def terminal_heading_t(xy: Tensor, stall_m: float = 0.05) -> Tensor:
    """``compliance_target``'s rule: heading of the LAST segment, 0 when it is
    shorter than ``stall_m``. ``xy`` [..., S, 2] -> [...]."""
    d = xy[..., -1, :] - xy[..., -2, :]
    th = torch.atan2(d[..., 1], d[..., 0])
    return torch.where(d.norm(dim=-1) < stall_m, torch.zeros_like(th), th)


def _nav_side(nav_cmd: Tensor) -> Tensor:
    """+1 left, -1 right, 0 follow/straight — the SAME map ``compliance_target`` uses."""
    from .refc_selector_targets import NAV_SIDE
    nav = nav_cmd.reshape(-1).long()
    side = torch.zeros(nav.shape, dtype=torch.float32, device=nav.device)
    for k, v in NAV_SIDE.items():
        side = torch.where(nav == k, torch.full_like(side, float(v)), side)
    return side


def aggregate(sub: dict[str, Tensor], weights: dict[str, float] | None = None,
              nav_informative: Tensor | None = None) -> Tensor:
    """PDMS-shaped aggregate: product of the GATES x weighted mean of the QUALITY terms.

    ``sub`` holds probabilities (scorer) or oracle values in [0, 1], each [B, M].
    ``weights`` are the QUALITY weights — the inference-time BEHAVIOUR PROFILE
    (DrivoR §3.4): changing them changes the driving style without retraining.
    Default = PDMS (TTC 5, EP 5, comfort 2).
    """
    w = {"ttc": 5.0, "ep": 5.0, "comf": 2.0}
    if weights:
        w.update(weights)
    gate = torch.ones_like(sub["nc"])
    for k in GATES:
        if k not in sub:
            continue
        g = sub[k]
        if k == "nav":
            # a command without a side (follow / straight) states no preference:
            # the nav gate is 1 there, never an unconditional penalty
            if nav_informative is None:
                continue
            inf = nav_informative.to(torch.bool)
            if inf.dim() == 1:
                inf = inf[:, None]
            g = torch.where(inf, g, torch.ones_like(g))
        gate = gate * g
    num = sum(w[k] * sub[k] for k in QUALITY)
    den = sum(w[k] for k in QUALITY)
    return gate * num / den


# ---------------------------------------------------------------------------
# D4 — off-proposal coverage for the scorer (the TOAD lesson)
# ---------------------------------------------------------------------------
def smooth_perturbations(base: Tensor, slot_t: Tensor, n: int, *, seed: int,
                         progress_sigma: float = 0.15, lat_sigma_m: float = 1.5
                         ) -> Tensor:
    """``n`` smooth variants per row of the candidate set ``base`` [B, M, S, 2].

    Each variant picks a random base candidate, scales its path radially by
    ``1 + N(0, progress_sigma)`` (faster / slower along the same shape; clamped at 0)
    and adds a lateral offset growing as ``(t / T)^2 * N(0, lat_sigma_m)`` (a smooth
    drift that starts tangent to the base, so the variant stays drivable-looking).
    Deterministic in ``seed``. Label-side only: these are candidates the SCORER is
    trained on, never plans the model emits.
    """
    B, M, S, _ = base.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    pick = torch.randint(0, M, (B, n), generator=g).to(base.device)
    sel = base.gather(1, pick[:, :, None, None].expand(B, n, S, 2))
    scale = (1.0 + progress_sigma * torch.randn(B, n, 1, 1, generator=g)).clamp_min(0.0)
    lat = lat_sigma_m * torch.randn(B, n, 1, generator=g)
    tt = (slot_t.to(base) / slot_t.max().to(base)) ** 2                  # [S]
    out = sel * scale.to(base)
    out = out.clone()
    out[..., 1] = out[..., 1] + lat.to(base) * tt
    return out
