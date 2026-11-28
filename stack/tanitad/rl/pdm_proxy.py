"""``tanitad.rl.pdm_proxy`` — a PDMS-SHAPED reward for data with no map, no route, no PDM-Closed.

⭐ WHAT THIS IS. DiffusionDriveV2's RL reward is NAVSIM's PDM score of each candidate
(``SPEC_DDV2_RL_PAPER.md`` §5, §13): ``PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) / 12`` over a
4 s, 10 Hz self-action replay against RECORDED (non-reactive) agents. This module computes the
same formula from what the PhysicalAI-AV B1 corpus actually carries, and states per term
what is the same and what is not. It is a PROXY: no number from it may be quoted as PDMS.

| term | NAVSIM fork (SPEC §13) | here | class |
|---|---|---|---|
| horizon | 40 poses x 0.1 s (``scoring.yaml:4-5``) + t0 | the same 41 ticks | FAITHFUL |
| ego states | LQR-tracked kinematic bicycle of the proposal | the candidate IS a control sequence rolled by the programme's unicycle; no tracker | DIFFERENT |
| NC | at-fault polygon collision with tracks: 0 agent / 0.5 static object | same values; at-fault = ego moving AND (track stopped OR front-edge contact), rear contacts never at fault, lateral contacts not at fault (no lane map to establish "multiple lanes / off-road") | DIFFERENT (lateral clause) |
| DAC | footprint in non-drivable area at any tick | :func:`dac_from_drivable` from SAM3 map GT when a map exists; **1 when it does not** | MISSING on the RL-train split |
| EP | centerline progress, normalised pairwise against **PDM-Closed** | progress along the **human's own future path**, normalised pairwise against **the human** | DIFFERENT (reference) |
| TTC | footprint projected at const. velocity to +0/0.3/0.6/0.9 s; 0 if it meets a track AHEAD (or in lanes/intersection and not behind) | the same projection; 0 if it meets a track inside the +-30 deg forward cone; no intersection clause | DIFFERENT (map clause) |
| C | 6 Savitzky-Golay checks over the 41 ticks, window = n_time | the same 6 thresholds; derivatives of ONE least-squares polynomial over all 41 ticks (what savgol with window = n_time computes) | FAITHFUL-UNBANKED (partial transcription) |
| ego box | nuPlan Pacifica 5.176 x 2.297 m, rear axle 1.461 m behind centre | the same numbers; the PhysicalAI vehicle's are not in the corpus | UNVERIFIED |

⛔ The ego pose origin is treated as the REAR AXLE (nuPlan's convention). For PhysicalAI's rig
frame this is UNVERIFIED; the known-value control (the human's own NC over real windows) is the
instrument that would expose a wrong origin, and the chain driver reports it.

Tier: T0 reward machinery. Evidence class: MEASURED on analytic cases
(``tests/test_pdm_proxy.py``); on real windows, only the controls the driver reports.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

__all__ = ["ProxyConfig", "PROXY", "box_corners", "boxes_overlap", "ego_states_from_controls",
           "ego_states_from_poses", "AgentTracks", "no_at_fault_collision", "ttc_within_bound",
           "comfort", "path_progress", "ego_progress", "dac_from_drivable", "pdms",
           "speed_appropriateness",
           "score_candidates"]


@dataclass(frozen=True)
class ProxyConfig:
    n_ticks: int = 40
    dt: float = 0.1
    ego_length: float = 5.176
    ego_width: float = 2.297
    rear_axle_to_center: float = 1.461
    stopped_speed_ego: float = 5e-2          # nuPlan-style collision typing (UNVERIFIED value)
    stopped_speed_track: float = 5e-2
    ttc_stopped_speed: float = 5e-3          # scoring.yaml:29
    ttc_offsets: tuple = (0, 3, 6, 9)        # pdm_scorer._calculate_ttc
    ahead_cone_deg: float = 30.0             # nuPlan is_agent_ahead tolerance (UNVERIFIED value)
    progress_threshold: float = 5.0          # scoring.yaml:30
    w_ep: float = 5.0
    w_ttc: float = 5.0
    w_c: float = 2.0
    max_abs_mag_jerk: float = 8.37
    max_abs_lat_accel: float = 4.89
    max_lon_accel: float = 2.40
    min_lon_accel: float = -4.05
    max_abs_yaw_accel: float = 1.93
    max_abs_lon_jerk: float = 4.13
    max_abs_yaw_rate: float = 0.95
    static_classes: tuple = ("protruding_object",)
    #: ⭐ `L2-SPD`'s speed-appropriateness term (`PREREG_D9_REWARD_REPAIR.md` §13),
    #: specified BEFORE it ran and DEFAULT OFF. At ``w_spd = 0`` the numerator and the
    #: denominator are both unchanged, so the score is byte-identical to the arm without
    #: the term -- §13.3's first control, and a test asserts it.
    #: ⚠️ `v_tol_mps` is NOT tuned here. §13.2 fixes the candidate ladder
    #: {0.75, 1.5, 3.0} and requires selection on the FIT split only; the default below
    #: is the ladder's midpoint, chosen for being the midpoint and for no other reason.
    w_spd: float = 0.0
    v_tol_mps: float = 1.5
    #: ⛔ WITHOUT THE TERM ABOVE THERE IS NO SPEED TERM AT ALL, AND THAT IS RECORDED.
    #: MEASURED 2026-09-17: ``ego_progress`` saturates at the route end, so candidates at
    #: 10 / 12 / 14 / 20 m/s against a human at 10 all score the SAME progress (38.5390 on
    #: a 4 s route) while 6 and 8 m/s score 24.0 and 32.0. ⇒ EP penalises being SLOW and
    #: is **blind** to being fast; NC and TTC bite only when an agent is near; comfort
    #: bounds ACCELERATIONS, not speed. **Speed above the human's is a FLAT DIRECTION in
    #: this reward** -- neither rewarded nor penalised -- and a flat direction under a
    #: stochastic policy gradient DRIFTS, which is consistent with the two `H-DDV2RL-2`
    #: seeds damaging different families and only one of them over-speeding (+0.372 m/s,
    #: separated). A repair must ADD a penalty here; reshaping EP cannot work, because its
    #: input is already clipped.

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_source"] = ("NAVSIM fork @hustvl/DiffusionDriveV2 1cd12a1 read 2026-09-15 "
                        "(SPEC §13, PUBLISHED-CODE-UNBANKED) + nuPlan conventions (UNVERIFIED)")
        return d


PROXY = ProxyConfig()


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
def box_corners(cx: Tensor, cy: Tensor, yaw: Tensor, length, width) -> Tensor:
    """Oriented box corners ``[..., 4, 2]``: front-left, rear-left, rear-right, front-right."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    hl = torch.as_tensor(length, dtype=cx.dtype, device=cx.device) / 2
    hw = torch.as_tensor(width, dtype=cx.dtype, device=cx.device) / 2
    hl = hl.expand_as(cx) if hl.dim() else hl
    hw = hw.expand_as(cx) if hw.dim() else hw
    sx = torch.stack([hl, -hl, -hl, hl], dim=-1)
    sy = torch.stack([hw, hw, -hw, -hw], dim=-1)
    x = cx[..., None] + sx * c[..., None] - sy * s[..., None]
    y = cy[..., None] + sx * s[..., None] + sy * c[..., None]
    return torch.stack([x, y], dim=-1)


def boxes_overlap(a: Tensor, b: Tensor) -> Tensor:
    """Separating-axis test for two convex quads ``[..., 4, 2]`` (broadcast). -> bool [...].

    Touching (zero-area contact) counts as overlap, like shapely's ``intersects``.
    """
    def axes(p):
        e0 = p[..., 1, :] - p[..., 0, :]
        e1 = p[..., 2, :] - p[..., 1, :]
        return torch.stack([torch.stack([-e0[..., 1], e0[..., 0]], -1),
                            torch.stack([-e1[..., 1], e1[..., 0]], -1)], dim=-2)   # [..., 2, 2]
    ax = torch.cat([axes(a).expand(*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), 2, 2),
                    axes(b).expand(*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), 2, 2)],
                   dim=-2)                                                          # [..., 4, 2]
    pa = torch.einsum("...kd,...cd->...kc", ax, a.expand(*ax.shape[:-2], 4, 2))
    pb = torch.einsum("...kd,...cd->...kc", ax, b.expand(*ax.shape[:-2], 4, 2))
    sep = (pa.amax(-1) < pb.amin(-1)) | (pb.amax(-1) < pa.amin(-1))
    return ~sep.any(-1)


def _wrap(a: Tensor) -> Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def ego_states_from_controls(u: Tensor, v0: Tensor, horizons, *, control_units: str = "alat",
                             alat_v_floor: float = 4.0, kappa_cap: float = 0.12,
                             cfg: ProxyConfig = PROXY) -> Tensor:
    """Control sequence ``u [B, M, S, 2]`` (metric units) -> states ``[B, M, n_ticks+1, 4]``
    ``(x, y, yaw, v)`` incl. the t0 state, through THE programme integrator.

    ⭐ Same expansion and integrator as ``refc_sampler.roll_controls`` (pinned equal at the slot
    ticks in the tests), but keeping every tick and the heading and speed the reward needs.
    """
    from tanitad.models.kinematic import rollout_unicycle
    from tanitad.refs import refc_sampler as rs
    b, m = u.shape[:2]
    u32 = u.to(torch.float32)
    v = torch.as_tensor(v0, dtype=torch.float32, device=u.device).reshape(-1)
    a_lon = u32[..., 0]
    if control_units == "alat":
        kap = rs.alat_to_curvature(u32[..., 1], v[:, None, None], alat_v_floor, kappa_cap)
    else:
        kap = u32[..., 1]
    ticks = rs.controls_to_ticks(torch.stack([a_lon, kap], dim=-1), tuple(horizons))
    need = cfg.n_ticks
    if ticks.shape[-2] < need:
        raise ValueError(f"control horizon {ticks.shape[-2]} ticks < scored {need}")
    ticks = ticks[..., :need, :].reshape(b * m, need, 2)
    state0 = torch.zeros(b * m, 4, device=u.device, dtype=torch.float32)
    state0[:, 3] = v[:, None].expand(b, m).reshape(-1)
    path = rollout_unicycle(state0, ticks, dt=cfg.dt)            # [BM, need, 4]
    full = torch.cat([state0[:, None], path], dim=1)
    return full.reshape(b, m, need + 1, 4)


def ego_states_from_poses(pose_last: Tensor, future_poses: Tensor,
                          cfg: ProxyConfig = PROXY) -> Tensor:
    """The HUMAN's states in the ego frame at t0: ``pose_last [B, 4]``, ``future_poses
    [B, K>=n_ticks, 4]`` (x, y, yaw, v; world) -> ``[B, n_ticks+1, 4]``. Rotation exactly
    ``refb_labels.ego_frame`` (``cos(-yaw)``, ``sin(-yaw)``)."""
    need = cfg.n_ticks
    if future_poses.shape[1] < need:
        raise ValueError(f"{future_poses.shape[1]} future poses < {need}")
    fp = torch.cat([pose_last[:, None], future_poses[:, :need]], dim=1).float()
    yaw0 = pose_last[:, 2:3].float()
    dx = fp[..., 0] - pose_last[:, 0:1].float()
    dy = fp[..., 1] - pose_last[:, 1:2].float()
    c, s = torch.cos(-yaw0), torch.sin(-yaw0)
    x = dx * c - dy * s
    y = dx * s + dy * c
    return torch.stack([x, y, _wrap(fp[..., 2] - yaw0), fp[..., 3]], dim=-1)


def _centers(states: Tensor, cfg: ProxyConfig) -> tuple[Tensor, Tensor]:
    return (states[..., 0] + cfg.rear_axle_to_center * torch.cos(states[..., 2]),
            states[..., 1] + cfg.rear_axle_to_center * torch.sin(states[..., 2]))


def _ego_boxes(states: Tensor, cfg: ProxyConfig) -> Tensor:
    cx, cy = _centers(states, cfg)
    return box_corners(cx, cy, states[..., 2], cfg.ego_length, cfg.ego_width)


# --------------------------------------------------------------------------- #
# agents                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class AgentTracks:
    """Recorded agents in the ego frame AT t0, dense over ticks. ``T = n_ticks + 1 + max(ttc)``.

    ``xy [T, A, 2]``, ``yaw [T, A]``, ``lw [A, 2]``, ``valid [T, A]`` bool, ``static [A]`` bool,
    ``speed [T, A]`` (finite differences along each track).
    """
    xy: Tensor
    yaw: Tensor
    lw: Tensor
    valid: Tensor
    static: Tensor
    speed: Tensor

    @staticmethod
    def from_frames(frames: list[list[dict]], ego_poses: Tensor, cfg: ProxyConfig = PROXY,
                    device=None) -> "AgentTracks":
        """``frames[k]`` = the join's agent dicts at tick k (ego frame of THAT frame);
        ``ego_poses [T, 4]`` world (x, y, yaw, v) of the ego at those frames, ``ego_poses[0]`` = t0.
        """
        ids: dict[str, int] = {}
        for fr in frames:
            for a in fr:
                ids.setdefault(str(a["track_id"]), len(ids))
        T, A = len(frames), max(1, len(ids))
        xy = torch.zeros(T, A, 2)
        yaw = torch.zeros(T, A)
        lw = torch.zeros(A, 2)
        valid = torch.zeros(T, A, dtype=torch.bool)
        static = torch.zeros(A, dtype=torch.bool)
        p = ego_poses.float()
        x0, y0, h0 = p[0, 0], p[0, 1], p[0, 2]
        c0, s0 = torch.cos(-h0), torch.sin(-h0)
        for k, fr in enumerate(frames):
            hk = p[k, 2]
            ck, sk = torch.cos(hk), torch.sin(hk)
            for a in fr:
                j = ids[str(a["track_id"])]
                wx = p[k, 0] + a["cx"] * ck - a["cy"] * sk          # ego@k -> world
                wy = p[k, 1] + a["cx"] * sk + a["cy"] * ck
                dx, dy = wx - x0, wy - y0                             # world -> ego@t0
                xy[k, j, 0] = dx * c0 - dy * s0
                xy[k, j, 1] = dx * s0 + dy * c0
                yaw[k, j] = float(_wrap(torch.tensor(a["yaw"] + float(hk) - float(h0))))
                lw[j, 0], lw[j, 1] = float(a["l"]), float(a["w"])
                valid[k, j] = True
                static[j] = static[j] | (str(a.get("cls", "")) in cfg.static_classes)
        d = torch.zeros(T, A)
        if T > 1:
            step = (xy[1:] - xy[:-1]).norm(dim=-1) / cfg.dt
            both = valid[1:] & valid[:-1]
            d[1:] = torch.where(both, step, torch.zeros_like(step))
            d[0] = torch.where(both[0], step[0], torch.zeros_like(step[0]))
        out = AgentTracks(xy=xy, yaw=yaw, lw=lw, valid=valid, static=static, speed=d)
        return out.to(device) if device is not None else out

    def to(self, device) -> "AgentTracks":
        return AgentTracks(*(t.to(device) for t in (self.xy, self.yaw, self.lw, self.valid,
                                                    self.static, self.speed)))

    def boxes(self) -> Tensor:
        return box_corners(self.xy[..., 0], self.xy[..., 1], self.yaw,
                           self.lw[None, :, 0].expand_as(self.yaw),
                           self.lw[None, :, 1].expand_as(self.yaw))


def _rel_in_ego(states: Tensor, pts: Tensor) -> Tensor:
    """Agent centres ``pts [..., T, A, 2]`` in each ego tick's own rear-axle frame."""
    dx = pts[..., 0] - states[..., None, 0]
    dy = pts[..., 1] - states[..., None, 1]
    c, s = torch.cos(-states[..., None, 2]), torch.sin(-states[..., None, 2])
    return torch.stack([dx * c - dy * s, dx * s + dy * c], dim=-1)


def _segment_hits_box(p0: Tensor, p1: Tensor, box: Tensor) -> Tensor:
    """Does segment p0-p1 ``[..., 2]`` touch quad ``[..., 4, 2]``? (SAT on a degenerate quad.)"""
    seg = torch.stack([p0, p1, p1, p0], dim=-2)
    return boxes_overlap(seg, box)


def no_at_fault_collision(states: Tensor, agents: AgentTracks, cfg: ProxyConfig = PROXY) -> Tensor:
    """``states [M, T0, 4]`` (T0 = n_ticks + 1) -> NC ``[M]`` in {0, 0.5, 1}.

    Per (candidate, track) the FIRST overlapping tick decides (NAVSIM appends a non-at-fault
    track to an ignore list; an at-fault one scores). Tracks already overlapping the ego at
    t0 are ignored (NAVSIM seeds the list from the observation's initial collisions).
    At-fault at that tick = ego moving AND (track stopped OR ego front edge touches the track).
    """
    t0n = states.shape[-2]
    ego = _ego_boxes(states, cfg)                                   # [M, T0, 4, 2]
    ag = agents.boxes()[:t0n]                                        # [T0, A, 4, 2]
    valid = agents.valid[:t0n]                                       # [T0, A]
    over = boxes_overlap(ego[:, :, None], ag[None]) & valid[None]    # [M, T0, A]
    initial = over[:, 0].clone()                                     # [M, A]
    over = over & ~initial[:, None]
    ego_moving = states[..., 3] > cfg.stopped_speed_ego              # [M, T0]
    track_stopped = agents.speed[:t0n] <= cfg.stopped_speed_track    # [T0, A]
    front = _segment_hits_box(ego[:, :, None, 0], ego[:, :, None, 3], ag[None])  # FL..FR edge
    rel = _rel_in_ego(states, agents.xy[:t0n][None].expand(states.shape[0], -1, -1, -1))
    behind = rel[..., 0] < 0
    atfault = ego_moving[..., None] & (track_stopped[None] | (front & ~behind))
    any_over = over.any(dim=1)                                        # [M, A]
    first = torch.where(any_over, over.float().argmax(dim=1), torch.zeros_like(any_over, dtype=torch.long))
    fault_first = atfault.gather(1, first[:, None, :]).squeeze(1) & any_over     # [M, A]
    val = torch.where(agents.static[None], torch.full_like(fault_first, 1, dtype=torch.float32) * 0.5,
                      torch.zeros_like(fault_first, dtype=torch.float32))
    score = torch.where(fault_first, val, torch.ones_like(val))
    return score.amin(dim=1)


def ttc_within_bound(states: Tensor, agents: AgentTracks, cfg: ProxyConfig = PROXY) -> Tensor:
    """TTC ``[M]`` in {0, 1}: the footprint projected at constant velocity to each offset meets
    a valid track inside the forward cone while the ego is moving. Agents are read at
    ``tick + offset`` (``agents`` must cover ``n_ticks + max(offset)`` ticks)."""
    m, t0n = states.shape[0], states.shape[1]
    need = t0n + max(cfg.ttc_offsets)
    if agents.valid.shape[0] < need:
        raise ValueError(f"TTC needs agents over {need} ticks, have {agents.valid.shape[0]}")
    cx, cy = _centers(states, cfg)
    vel = torch.stack([torch.cos(states[..., 2]), torch.sin(states[..., 2])], -1) * states[..., 3:4]
    moving = states[..., 3] >= cfg.ttc_stopped_speed
    ag_all = agents.boxes()
    fail = torch.zeros(m, dtype=torch.bool, device=states.device)
    cone = math.cos(math.radians(cfg.ahead_cone_deg))
    for off in cfg.ttc_offsets:
        dt = off * cfg.dt
        box = box_corners(cx + vel[..., 0] * dt, cy + vel[..., 1] * dt, states[..., 2],
                          cfg.ego_length, cfg.ego_width)            # [M, T0, 4, 2]
        ag = ag_all[off:off + t0n]
        val = agents.valid[off:off + t0n]
        over = boxes_overlap(box[:, :, None], ag[None]) & val[None]
        rel = _rel_in_ego(states, agents.xy[off:off + t0n][None].expand(m, -1, -1, -1))
        dist = rel.norm(dim=-1).clamp_min(1e-6)
        ahead = (rel[..., 0] / dist) >= cone
        fail |= (over & ahead & moving[..., None]).any(dim=(1, 2))
    return (~fail).float()


# --------------------------------------------------------------------------- #
# comfort                                                                      #
# --------------------------------------------------------------------------- #
def _poly_deriv(y: Tensor, dt: float, order: int, deriv: int) -> Tensor:
    """Derivative of ONE least-squares polynomial of ``order`` fit over the whole last axis.

    This is what ``scipy.signal.savgol_filter(y, window_length=n, polyorder=order,
    deriv=deriv, delta=dt)`` returns when the window spans the signal (pinned in tests).
    """
    n = y.shape[-1]
    t = (torch.arange(n, dtype=torch.float64, device=y.device) - (n - 1) / 2) * dt
    V = torch.stack([t ** k for k in range(order + 1)], dim=-1)            # [n, p+1]
    coef = torch.linalg.lstsq(V, y.to(torch.float64).reshape(-1, n).T).solution.T   # [.., p+1]
    out = torch.zeros(coef.shape[0], n, dtype=torch.float64, device=y.device)
    for k in range(deriv, order + 1):
        fac = math.factorial(k) / math.factorial(k - deriv)
        out = out + coef[:, k:k + 1] * fac * t[None] ** (k - deriv)
    return out.reshape(y.shape).to(y.dtype)


def comfort(states: Tensor, cfg: ProxyConfig = PROXY) -> Tensor:
    """``[M, T0, 4] -> [M]`` in {0, 1}: all six NAVSIM comfort checks hold at every tick."""
    dt = cfg.dt
    v = states[..., 3]
    yaw = torch.cumsum(torch.cat([states[..., :1, 2],
                                  _wrap(states[..., 1:, 2] - states[..., :-1, 2])], dim=-1), dim=-1)
    a_lon = _poly_deriv(v, dt, 2, 1)
    yaw_rate = _poly_deriv(yaw, dt, 2, 1)
    a_lat = v * yaw_rate
    yaw_acc = _poly_deriv(yaw, dt, 3, 2)
    j_lon = _poly_deriv(a_lon, dt, 2, 1)
    j_lat = _poly_deriv(a_lat, dt, 2, 1)
    j_mag = torch.sqrt(j_lon ** 2 + j_lat ** 2)
    ok = ((a_lon <= cfg.max_lon_accel) & (a_lon >= cfg.min_lon_accel)
          & (a_lat.abs() <= cfg.max_abs_lat_accel) & (j_mag <= cfg.max_abs_mag_jerk)
          & (j_lon.abs() <= cfg.max_abs_lon_jerk) & (yaw_acc.abs() <= cfg.max_abs_yaw_accel)
          & (yaw_rate.abs() <= cfg.max_abs_yaw_rate))
    return ok.all(dim=-1).float()


# --------------------------------------------------------------------------- #
# progress                                                                     #
# --------------------------------------------------------------------------- #
def path_progress(points: Tensor, polyline: Tensor) -> Tensor:
    """Arc length of the closest point on ``polyline [K, 2]`` to each ``points [..., 2]``."""
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    seg_len = ab.norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=polyline.device, dtype=polyline.dtype),
                     torch.cumsum(seg_len, 0)])
    ap = points[..., None, :] - a
    tt = ((ap * ab).sum(-1) / (seg_len ** 2).clamp_min(1e-12)).clamp(0, 1)
    proj = a + tt[..., None] * ab
    d = (points[..., None, :] - proj).norm(dim=-1)
    j = d.argmin(dim=-1)
    return cum[j] + tt.gather(-1, j[..., None]).squeeze(-1) * seg_len[j]


def ego_progress(states: Tensor, route: Tensor, cfg: ProxyConfig = PROXY) -> Tensor:
    """Raw progress [M] of the footprint CENTRE from t0 to the last tick along ``route [K, 2]``
    (``pdm_scorer._calculate_progress``: project start and end, subtract, clip at 0)."""
    cx, cy = _centers(states, cfg)
    s0 = path_progress(torch.stack([cx[:, 0], cy[:, 0]], -1), route)
    s1 = path_progress(torch.stack([cx[:, -1], cy[:, -1]], -1), route)
    return (s1 - s0).clamp_min(0)


def dac_from_drivable(states: Tensor, drivable_frac: Tensor, seen: Tensor, *,
                      x_max_m: float = 60.0, y_half_m: float = 16.0, cell_m: float = 0.5,
                      threshold: float = 0.5, cfg: ProxyConfig = PROXY) -> Tensor:
    """DAC [M] from a t0 SAM3 map (``semantic_map_gt``: 120 x 64 rig grid, 0.5 m, x in [0, 60),
    y in [-16, 16)). 0 if any footprint corner, at any tick, falls on a SEEN cell whose drivable
    fraction is below ``threshold``. Unseen / off-grid corners carry no evidence (compliant) —
    stated because it makes this an UPPER bound on compliance."""
    corners = _ego_boxes(states, cfg)                                   # [M, T0, 4, 2]
    ix = torch.floor(corners[..., 0] / cell_m).long()
    iy = torch.floor((corners[..., 1] + y_half_m) / cell_m).long()
    h, w = drivable_frac.shape
    inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
    ixc, iyc = ix.clamp(0, h - 1), iy.clamp(0, w - 1)
    frac = drivable_frac[ixc, iyc]
    sn = seen[ixc, iyc]
    off = inside & sn & (frac < threshold)
    return (~off.any(dim=(1, 2))).float()


def speed_appropriateness(states: Tensor, cfg: ProxyConfig = PROXY) -> Tensor:
    """``SPD [M]`` from ``states [M, T, 4]`` whose row 0 is the HUMAN (`PREREG` §13.1).

    ``SPD = clamp(1 - max(0, v_cand - v_human) / v_tol, 0, 1)`` over the mean speed of
    the scored ticks.

    ⛔ **ONE-SIDED, DELIBERATELY.** Being *slower* than the human is ALREADY penalised by
    EP, which discriminates fully below the reference (MEASURED: 6 m/s -> progress 24.0,
    8 -> 32.0 against the human's 38.5). A two-sided form would penalise slowness twice
    and change the meaning of the existing EP result.
    ⛔ **It is an EXCESS-OVER-THE-HUMAN term, never a speed limit.** On a window where the
    human speeds it is silent -- correct for a proxy whose whole reference frame is the
    human's own future, and a stated limitation of this family rather than of this term.
    ⭐ The human's own row scores EXACTLY 1.0 by construction (its excess over itself is
    zero), which §13.3 requires as a control.
    """
    v = states[..., 3].mean(dim=-1)                       # [M] mean speed per candidate
    excess = (v - v[0]).clamp(min=0.0)                    # row 0 IS the human
    tol = max(float(cfg.v_tol_mps), 1e-9)
    return (1.0 - excess / tol).clamp(0.0, 1.0)


def pdms(nc: Tensor, dac: Tensor, ep: Tensor, ttc: Tensor, c: Tensor,
         cfg: ProxyConfig = PROXY, spd: Tensor | None = None) -> Tensor:
    """``NC x DAC x (w_ep EP + w_ttc TTC + w_c C) / (w_ep + w_ttc + w_c)`` (P Eq. 14),
    plus ``w_spd SPD`` in numerator AND denominator when the term is enabled.

    ⭐ **ADDITIVE, NOT MULTIPLICATIVE** (`PREREG` §13.1). NC and DAC multiply because they
    are CONSTRAINTS -- a collision or leaving the road annihilates the score. Driving
    0.4 m/s fast is a QUALITY failure, and the quality terms (EP, TTC, comfort) are
    weighted components. A multiplicative SPD would make a mild over-speed catastrophic,
    reintroducing from the other side the all-or-nothing behaviour that made a constant
    DAC so damaging.
    ⭐ At ``w_spd == 0`` both numerator and denominator are untouched, so the result is
    byte-identical to the pre-term score.
    """
    num = cfg.w_ep * ep + cfg.w_ttc * ttc + cfg.w_c * c
    den = cfg.w_ep + cfg.w_ttc + cfg.w_c
    if spd is not None and float(cfg.w_spd) > 0.0:
        num = num + cfg.w_spd * spd
        den = den + cfg.w_spd
    return nc * dac * num / den


def score_candidates(cand_states: Tensor, human_states: Tensor, agents: AgentTracks,
                     route: Tensor, *, dac_cand: Tensor | None = None,
                     dac_human: Tensor | None = None, cfg: ProxyConfig = PROXY) -> dict:
    """One window. ``cand_states [M, T0, 4]``, ``human_states [T0, 4]``, ``route [K, 2]`` (the
    human's own future centre path). -> per-candidate sub-scores and PDMS, plus the human's.

    EP is re-normalised PAIRWISE exactly like ``_pairwise_scores`` (``rl.py:94-108``) but with
    the HUMAN as proposal 0 (the fork uses PDM-Closed; SPEC §13.1).
    """
    allst = torch.cat([human_states[None], cand_states], dim=0)          # 0 = human
    nc = no_at_fault_collision(allst, agents, cfg)
    ttc = ttc_within_bound(allst, agents, cfg)
    cf = comfort(allst, cfg)
    ones = torch.ones_like(nc)
    dac = ones.clone()
    if dac_cand is not None:
        dac[1:] = dac_cand
    if dac_human is not None:
        dac[0] = dac_human
    multi = nc * dac
    raw = ego_progress(allst, route, cfg) * multi
    ref = raw[0]
    mx = torch.maximum(ref, raw)
    # ⛔ EP IS BLIND TO OVER-SPEEDING, AND NOT BECAUSE OF THIS LINE. `ego_progress` is a
    # PROJECTION ONTO A ROUTE THAT ENDS AT THE HUMAN'S LAST POINT, so it SATURATES:
    # MEASURED 2026-09-17 with a 4 s route, candidates at 10 / 12 / 14 / 20 m/s against a
    # human at 10 all score progress 38.5390 -- IDENTICAL -- while 6 and 8 m/s score 24.0
    # and 32.0. A candidate at TWICE the human's speed is indistinguishable from one that
    # matches it. ⇒ making this ratio two-sided would change nothing: `raw` is already
    # clipped. The reward has NO speed term, and speed above the human's is a FLAT
    # DIRECTION -- neither rewarded nor penalised. See `PREREG_D9_REWARD_REPAIR.md`
    # arm `L2-SPD`, which must therefore ADD a penalty rather than reshape this.
    ep = torch.where(mx > cfg.progress_threshold, raw / (mx + 1e-6),
                     torch.where(multi == 0, torch.zeros_like(raw), torch.ones_like(raw)))
    # ⭐ COMPUTED ALWAYS, WEIGHTED ONLY WHEN ENABLED. `spd` is reported in `out` even at
    # w_spd = 0, so a run can SEE the speed excess it is not yet penalising -- the lesson
    # the D9 DAC term earned, where an unlogged sub-score stayed invisible for 600 steps
    # across 3 arms. At w_spd = 0 `pdms` ignores it and the score is byte-identical.
    spd = speed_appropriateness(allst, cfg)
    score = pdms(nc, dac, ep, ttc, cf, cfg, spd=spd)
    out = {"nc": nc[1:], "dac": dac[1:], "ep": ep[1:], "ttc": ttc[1:], "comfort": cf[1:],
           "spd": spd[1:],
           "pdms": score[1:], "human": {"nc": float(nc[0]), "dac": float(dac[0]),
                                       "ep": float(ep[0]), "ttc": float(ttc[0]),
                                       "comfort": float(cf[0]), "pdms": float(score[0]),
                                       "spd": float(spd[0]),
                                       "progress_m": float(ref)},
           "constraint_fail": (nc[1:] != 1) | (dac[1:] != 1)}
    return out
