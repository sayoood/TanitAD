"""``tanitad.rl.ddv2_refcv7`` -- DiffusionDriveV2's RL stage bound to refcv7 (WP-RL, 2026-10-04).

WHAT THIS MODULE IS
-------------------
The model- and data-specific half of the DDv2 port for **refcv7-r101-s0** (step 50,400). The
released arithmetic (step, log-probability, advantage, loss) is :mod:`tanitad.rl.ddv2_rl`, unchanged
and pinned bitwise to ``hustvl/DiffusionDriveV2@1cd12a1``. The refcv5-v2 binding
(:mod:`tanitad.rl.ddv2_refc_chain`) cannot drive refcv7, and is NOT edited: its capture hook REFUSES
agent tokens and a BEV map by design (refcv5-v2 had neither), and refcv7 has both
(``--agents head``, ``--bev-coupling``) plus NEW-1's residual prior and F3's cascade. This module is
that binding for refcv7, beside the old one:

* :func:`capture_sampler_inputs` records EVERYTHING ``_sample`` is called with -- ``kv, cond, bank,
  v_ms, steps, agents, agent_pad, agent_pos, bev, prior`` -- and the RNG state at entry, during an
  ordinary deployed forward. The chain then sees exactly the conditioning the deployed model computes.
* :func:`make_x0_fn` is ONE sampler pass: ``_roll_state(denorm(x), v, prior)`` -> ``_decode_ctrl``
  with the SAME agents / BEV / prior -> ``x + du``. No input clamp (see SPEC_RL.md D-4).
* :func:`native_sample` re-drives the DEPLOYED ladder (``[10, 0]`` under F2's DD step pairs) through
  that callable: the parity instrument. It must reproduce ``_sample`` bit for bit.
* :func:`tick_states` integrates a residual state at 0.1 s (x, y, yaw, v) through the SAME
  ``compose_ticks`` + ``rollout_unicycle`` the decoder's ``roll_plan`` uses, so the reward scores the
  path the model emits (pinned: at the slot ticks it equals ``_roll_state`` bitwise).
* :func:`dac_fine` -- DAC from the 10 cm SAM3 map the map head is trained on (``map_fine``, the
  window's own NOW frame), and :func:`tracks_from_join` -- the replayed agents for NC / TTC.

Units: the diffusion state is the NORMALISED RESIDUAL ``Delta / control_norm``; ``Delta`` is the
control offset from the ``ha0_ext_pose`` prior (``a_lon`` m/s^2, ``a_lat`` m/s^2 -- the anchor
artifact declares ``control_units: alat``). Metres only after a roll.

Tier: T0 machinery. Evidence class: MEASURED (``stack/tests/test_ddv2_refcv7.py``) on analytic
cases; on the real model only through the driver's parity / control records.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.models import kinematic_prior as _kp
from tanitad.rl import ddv2_rl as D
from tanitad.rl import pdm_proxy as P

__all__ = [
    "SamplerInputsV7", "capture_sampler_inputs", "anchor_state", "make_x0_fn", "roll_fan",
    "native_sample", "tick_states", "ROAD_CODES", "NONROAD_CODES", "NOT_SEEN_CODE",
    "DacRule", "DAC_RULE", "offroad_cells", "offroad_cells_r2", "OFFROAD_RULES", "dac_fine", "tracks_from_join", "filter_tracks",
    "score_window", "microbatch_step_loss", "shuffle_permutation", "stage_x0_list", "trainable_named",
    "TRAINABLE_PREFIXES",
]

#: SAM3 fine-code legend (``semantic_map_gt.CHANNELS``): 0 seen-no-class, 1 drivable, 2 lane / road
#: line, 3 crosswalk, 4 arrow / text, 5 non-drivable edge, 6 hatched area, 7 sidewalk / verge,
#: 255 not seen. "Road-like" = the D3 audit's literal set (``.../D3_raw_plausibility/scripts/
#: c4_sam3.py:27``: drivable + painted road furniture), MEASURED there at 99.92 % of the ego's own
#: next-6-s path (2.98 M points, 4,369 train clips). Not fitted here.
ROAD_CODES: tuple[int, ...] = (1, 2, 3, 4, 6)
NONROAD_CODES: tuple[int, ...] = (0, 5, 7)
NOT_SEEN_CODE: int = 255

#: the decoder modules that ARE the trajectory generator (DDv2 trains ``_trajectory_head.*`` only,
#: SPEC 7.1). ``control_head`` is listed but is grad-UNREACHABLE under F3 (config.json
#: ``declared_vs_built.grad_unreachable``): the cascade's last stage emits ``du``.
TRAINABLE_PREFIXES: tuple[str, ...] = ("traj_proj", "time_mlp", "layers", "adaln", "cascade",
                                       "control_head")


class Ddv2V7Error(D.Ddv2ConfigError):
    """A refcv7-binding contract was broken (wrong build, missing input, shape)."""


# =========================================================================== #
# capture                                                                      #
# =========================================================================== #
@dataclass
class SamplerInputsV7:
    """``_sample``'s inputs VERBATIM (refc.py:2693-2700) + what it derived.

    ``v`` is the speed ``_sample`` rolls from: on a residual build ``prior[2]`` (refc.py:2771-2774).
    """
    kv: Tensor
    cond: Tensor
    bank: Tensor
    v_ms: Tensor | None
    steps: int
    agents: Tensor | None
    agent_pad: Tensor | None
    agent_pos: Tensor | None
    bev: Tensor | None
    prior: tuple
    v: Tensor
    out: tuple | None = None
    rng_cpu: Tensor | None = None
    rng_cuda: Tensor | None = None

    def replay_eps(self, decoder) -> Tensor:
        """Re-draw ``_sample``'s anchored-Gaussian ``eps`` (its FIRST draw, refc.py:2802) from the
        recorded RNG state; the caller's RNG state is restored."""
        x0_n = anchor_state(decoder, self.bank.shape[0], self.bank.dtype)
        saved_cpu = torch.get_rng_state()
        saved_cuda = torch.cuda.get_rng_state(x0_n.device) if x0_n.is_cuda else None
        try:
            torch.set_rng_state(self.rng_cpu)
            if x0_n.is_cuda and self.rng_cuda is not None:
                torch.cuda.set_rng_state(self.rng_cuda, x0_n.device)
            return torch.randn_like(x0_n)
        finally:
            torch.set_rng_state(saved_cpu)
            if saved_cuda is not None:
                torch.cuda.set_rng_state(saved_cuda, x0_n.device)

    def select(self, rows) -> "SamplerInputsV7":
        """The same record restricted to batch ``rows`` (a list / LongTensor)."""
        idx = torch.as_tensor(rows, dtype=torch.long, device=self.kv.device)

        def s(t):
            return None if t is None else t.index_select(0, idx.to(t.device))
        return SamplerInputsV7(kv=s(self.kv), cond=s(self.cond), bank=s(self.bank),
                               v_ms=s(self.v_ms), steps=self.steps, agents=s(self.agents),
                               agent_pad=s(self.agent_pad), agent_pos=s(self.agent_pos),
                               bev=s(self.bev), prior=tuple(s(t) for t in self.prior),
                               v=s(self.v), out=None, rng_cpu=None, rng_cuda=None)


def _check_refcv7_decoder(decoder) -> None:
    if getattr(decoder, "control_head", None) is None or getattr(decoder, "sched", None) is None:
        raise Ddv2V7Error("decoder has no WP-4 sampler (control_head/sched is None); the DDv2 "
                          "chain binds to `--sampler ddim` builds only")
    if str(getattr(decoder.cfg, "sampler_space", "control")) != "control":
        raise Ddv2V7Error("the refcv7 binding is for sampler_space='control'")
    mode = str(getattr(decoder, "residual_prior", _kp.RESIDUAL_PRIOR_OFF))
    if mode == _kp.RESIDUAL_PRIOR_OFF:
        raise Ddv2V7Error("this binding is for a refcv7 RESIDUAL build (NEW-1); the decoder has "
                          "residual_prior='off' -- use tanitad.rl.ddv2_refc_chain")
    if getattr(decoder.rv6, "f8_flat_waypoint_noise", False):
        raise Ddv2V7Error("F8 (metre waypoint state) is not refcv7's sampler; refusing")


@contextlib.contextmanager
def capture_sampler_inputs(decoder):
    """Record ``decoder._sample``'s inputs for every forward inside the block. -> list of
    :class:`SamplerInputsV7`. The original method is restored exactly on exit."""
    _check_refcv7_decoder(decoder)
    records: list[SamplerInputsV7] = []
    orig = decoder._sample

    def hook(kv, cond, bank, v_ms, steps, agents=None, agent_pad=None, agent_pos=None,
             bev=None, prior=None):
        if prior is None:
            raise Ddv2V7Error("a residual build called _sample without `prior` -- the chain "
                              "would roll Delta as if it were the plan")
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(bank.device) if bank.is_cuda else None
        out = orig(kv, cond, bank, v_ms, steps, agents, agent_pad, agent_pos, bev, prior=prior)
        records.append(SamplerInputsV7(
            kv=kv, cond=cond, bank=bank, v_ms=v_ms, steps=int(steps), agents=agents,
            agent_pad=agent_pad, agent_pos=agent_pos, bev=bev, prior=tuple(prior),
            v=prior[2], out=out, rng_cpu=rng_cpu, rng_cuda=rng_cuda))
        return out

    decoder._sample = hook
    try:
        yield records
    finally:
        del decoder._sample
        assert decoder._sample.__func__ is orig.__func__


# =========================================================================== #
# the chain                                                                    #
# =========================================================================== #
def _norm(decoder, like: Tensor) -> Tensor:
    return like.new_tensor(tuple(decoder.cfg.control_norm))


def anchor_state(decoder, batch: int, dtype=torch.float32) -> Tensor:
    """``[B, N, S, 2]`` normalised anchor residual sequence -- ``_sample``'s ``x0_n``
    (refc.py:2745-2746)."""
    seq = decoder.anchor_control_seq(batch, dtype)
    return seq / seq.new_tensor(tuple(decoder.cfg.control_norm))


def roll_fan(decoder, x_n: Tensor, inputs: SamplerInputsV7) -> Tensor:
    """Normalised residual state ``[B, M, S, 2]`` -> the plan in metres, through the decoder's OWN
    roll on the captured prior (``_roll_state``, refc.py:2347-2359)."""
    return decoder._roll_state(x_n * _norm(decoder, x_n), inputs.v, False, inputs.prior)


def make_x0_fn(decoder, inputs: SamplerInputsV7, *, stages: list | None = None):
    """-> ``x0_fn(x_n [B, M, S, 2], t) -> x0_hat_n``: ONE sampler pass of THIS decoder.

    Exactly ``_sample``'s per-pass body (refc.py:2841-2849) with the captured agents / BEV / prior:
    ``x_path = _roll_state(denorm(x_n), v, prior)``; ``conf, du = _decode_ctrl(..., agents, pad,
    pos, bev)``; ``x0_hat = x_n + du``. ``M`` may be any multiple of ``N`` (groups): no layer mixes
    queries (``CrossAttnLayer.forward``: cross-attention to kv / agents, the BEV read at the query's
    own waypoints, a per-token MLP), pinned in the tests.

    ``stages`` (a list, cleared by the caller) receives, after each call, the per-STAGE x0
    predictions of F3's cascade (``decoder._last_layer_du``) in normalised units -- the release's IL
    is averaged over every decoder layer's output (``rl.py:1104-1112``). On a non-cascade build it
    receives the single output.
    """
    kv, cond, v, prior = inputs.kv, inputs.cond, inputs.v, inputs.prior

    def x0_fn(x_n: Tensor, t: int) -> Tensor:
        if x_n.dim() != 4 or x_n.shape[0] != kv.shape[0]:
            raise Ddv2V7Error(f"state {tuple(x_n.shape)} vs captured batch {kv.shape[0]}")
        x_path = decoder._roll_state(x_n * _norm(decoder, x_n), v, False, prior)
        tt = torch.full((x_n.shape[0],), float(t), device=x_n.device, dtype=torch.float32)
        _, du = decoder._decode_ctrl(kv, cond, x_path, tt, inputs.agents, inputs.agent_pad,
                                     inputs.agent_pos, inputs.bev)
        if stages is not None:
            dus = (list(decoder._last_layer_du) if getattr(decoder, "cascade", None) is not None
                   else [du])
            stages.append([x_n + d for d in dus])
        return x_n + du

    return x0_fn


def stage_x0_list(stages: list) -> list:
    """The LAST call's per-stage x0 list (and the buffer is cleared)."""
    if not stages:
        raise Ddv2V7Error("no stage record -- make_x0_fn was built without `stages=` or not called")
    last = stages[-1]
    stages.clear()
    return last


def native_sample(decoder, inputs: SamplerInputsV7, eps: Tensor) -> tuple[Tensor, Tensor]:
    """The DEPLOYED sampler re-driven through :func:`make_x0_fn` -> ``(fan, u0_hat_abs)``.

    ``_sample``'s eval path (refc.py:2720-2869): the anchored Gaussian at ``sampler_infer_t``, the
    ladder ``sched.infer_timesteps``, F2's ``dd_step_pairs``, ``sched.step`` between passes, the
    LAST pass's prediction rolled on the prior and exported ABSOLUTE. A parity instrument only.
    """
    from tanitad.models import refcv6_diffusion as _rv6
    cfg = decoder.cfg
    b = inputs.bank.shape[0]
    x0_n = anchor_state(decoder, b, inputs.bank.dtype)
    dev = x0_n.device
    t0 = int(cfg.sampler_infer_t)
    k = int(inputs.steps) if int(inputs.steps) > 0 else int(cfg.sampler_steps)
    ladder = decoder.sched.infer_timesteps(t0, max(k, 1))
    decoder.sched.to(dev)
    x_n = decoder.sched.add_noise(x0_n, eps, torch.tensor(t0, device=dev))
    pairs = _rv6.dd_step_pairs([int(x) for x in ladder], bool(decoder.rv6.f2_dd_step))
    fn = make_x0_fn(decoder, inputs)
    x0_hat = x_n
    for t, t_prev in pairs:
        x0_hat = fn(x_n, t)
        x_n = decoder.sched.step(x0_hat, x_n, torch.tensor(t, device=dev),
                                 torch.tensor(t_prev, device=dev))
    u0 = x0_hat * _norm(decoder, x0_hat)
    fan = decoder._roll_state(u0, inputs.v, False, inputs.prior)
    return fan, decoder._export_controls(u0, inputs.prior)


# =========================================================================== #
# per-tick states (the reward's ego rollout)                                   #
# =========================================================================== #
def tick_states(decoder, x_n: Tensor, inputs_or_prior, n_ticks: int) -> Tensor:
    """Normalised residual ``[B, M, S, 2]`` -> ego states ``[B, M, n_ticks+1, 4]`` (x, y, yaw, v),
    row 0 = t0, at the decoder's own tick (0.1 s).

    The SAME composition as ``kinematic_prior.roll_plan`` (``compose_ticks`` -> ``rollout_unicycle``
    from ``state0 = (0, 0, 0, v)``), keeping every tick plus heading and speed. Pinned in the tests:
    rows ``h`` for ``h in anchor_horizons`` equal ``_roll_state`` bitwise.
    """
    from tanitad.models.kinematic import rollout_unicycle
    prior = getattr(inputs_or_prior, "prior", inputs_or_prior)
    a0, k0, pv = prior
    h = tuple(int(x) for x in decoder.anchor_horizons)
    if n_ticks > max(h):
        raise Ddv2V7Error(f"n_ticks {n_ticks} > the plan horizon {max(h)} ticks")
    delta = x_n * _norm(decoder, x_n)
    b, m = delta.shape[:2]
    ticks = _kp.compose_ticks(delta, a0, k0, pv, h, control_units=decoder.anchor_control_units,
                              alat_v_floor=decoder.anchor_alat_v_floor,
                              kappa_cap=decoder.anchor_kappa_cap)          # [B, M, Tmax, 2]
    t_max = ticks.shape[-2]
    state0 = torch.zeros(b * m, 4, device=ticks.device, dtype=torch.float32)
    state0[:, 3] = pv.reshape(-1).to(torch.float32)[:, None].expand(b, m).reshape(-1)
    path = rollout_unicycle(state0, ticks.reshape(b * m, t_max, 2), dt=float(decoder.anchor_dt))
    full = torch.cat([state0[:, None], path[:, :n_ticks]], dim=1)
    return full.reshape(b, m, n_ticks + 1, 4)


# =========================================================================== #
# DAC on the 10 cm SAM3 map                                                    #
# =========================================================================== #
@dataclass(frozen=True)
class DacRule:
    """The off-road rule, FIXED here before any number (SPEC_RL.md D-7 and AMENDMENT A-0).

    A footprint corner is OFF-ROAD iff the 0.5 m block (``block x block`` fine cells) centred on its
    cell has at least ``min_seen`` SEEN cells and fewer than ``road_frac`` of them are road-like.
    A corner on unseen / off-grid cells carries no evidence (compliant) -- an UPPER bound on DAC,
    as the 0.5 m rule it replaces (``pdm_proxy.dac_from_drivable``). (!) The 0.5 m block rather
    than the single 10 cm cell: a painted lane line or a one-cell label error under a wheel is not
    "leaving the road"; the block is the old rule's own resolution.
    """
    cell_m: float = 0.1
    x_max_m: float = 100.0
    y_half_m: float = 30.0
    block: int = 5
    min_seen: int = 13
    road_frac: float = 0.5


DAC_RULE = DacRule()


def offroad_cells(codes: Tensor, rule: DacRule = DAC_RULE) -> Tensor:
    """``codes [X, Y]`` uint8 (row 0 = nearest 0.1 m, col 0 = y = -y_half, +y LEFT) -> bool
    ``[X, Y]``: the cell's block says OFF-ROAD."""
    if codes.dim() != 2:
        raise Ddv2V7Error(f"codes must be [X, Y], got {tuple(codes.shape)}")
    c = codes.to(torch.long)
    road = torch.zeros_like(c, dtype=torch.float32)
    for k in ROAD_CODES:
        road = road + (c == k).float()
    seen = (c != NOT_SEEN_CODE).float()
    k = int(rule.block)
    pad = k // 2
    stk = torch.stack([road, seen])[:, None]                                  # [2, 1, X, Y]
    box = F.conv2d(F.pad(stk, (pad, pad, pad, pad)), torch.ones(1, 1, k, k, device=c.device))
    road_n, seen_n = box[0, 0], box[1, 0]
    return (seen_n >= float(rule.min_seen)) & (road_n < float(rule.road_frac) * seen_n)


def offroad_cells_r2(codes: Tensor, rule: DacRule = DAC_RULE) -> Tensor:
    """SPEC_RL A-0 rung **R2**: only the D3 audit's "contradict" codes {5 non-drivable edge,
    7 sidewalk / verge} are off-road evidence; code 0 ("seen, no map class") and 255 carry none.
    A cell is OFF iff its block holds >= ``min_seen`` road-or-contradict cells and more contradict
    than road cells."""
    if codes.dim() != 2:
        raise Ddv2V7Error(f"codes must be [X, Y], got {tuple(codes.shape)}")
    c = codes.to(torch.long)
    road = torch.zeros_like(c, dtype=torch.float32)
    for k in ROAD_CODES:
        road = road + (c == k).float()
    contra = ((c == 5) | (c == 7)).float()
    k = int(rule.block)
    pad = k // 2
    stk = torch.stack([road, contra])[:, None]
    box = F.conv2d(F.pad(stk, (pad, pad, pad, pad)), torch.ones(1, 1, k, k, device=c.device))
    rn, cn = box[0, 0], box[1, 0]
    return ((rn + cn) >= float(rule.min_seen)) & (cn > rn)


#: the pre-listed DAC ladder (SPEC_RL A-0), strict -> lenient
OFFROAD_RULES = {"R1": offroad_cells, "R2": offroad_cells_r2}


def dac_fine(states: Tensor, off: Tensor, rule: DacRule = DAC_RULE,
             cfg: P.ProxyConfig = P.PROXY) -> Tensor:
    """DAC ``[M]`` in {0, 1}: 0 iff any ego-footprint CORNER at any tick lies on an off-road cell
    of ``off = offroad_cells(map_fine)`` (NAVSIM: ``_ego_areas[..., NON_DRIVABLE_AREA].any``).
    ``states [M, T, 4]`` in the window's ego frame at t0 (the map's rig frame at the same raw frame,
    D3's alignment control)."""
    corners = P._ego_boxes(states, cfg)                                       # [M, T, 4, 2]
    ix = torch.floor(corners[..., 0] / rule.cell_m).long()
    iy = torch.floor((corners[..., 1] + rule.y_half_m) / rule.cell_m).long()
    h, w = off.shape
    inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
    o = off[ix.clamp(0, h - 1), iy.clamp(0, w - 1)] & inside
    return (~o.any(dim=(1, 2))).float()


# =========================================================================== #
# agents (NC / TTC)                                                            #
# =========================================================================== #
def tracks_from_join(reader, episode_id: int, t0: int, ego_poses: Tensor, n_frames: int,
                     cfg: P.ProxyConfig = P.PROXY):
    """The replayed agents of frames ``t0 .. t0+n_frames-1`` in the ego frame AT t0, or ``None``
    when any frame is UNLABELLED (no join record = NO_LABEL, never "road clear").

    ``reader`` = the trainer's ``JoinFileReader`` (``train_p8_occupancy.py:240``), built WITH track
    ids; rows ``[A, 6] = (cx, cy, yaw, l, w, occ)`` in the ego frame of THAT frame, frame index in
    the episode's (= pose row) space -- the convention ``V3Dataset._agent_item(ep, t + w - 1)`` uses.
    ``ego_poses [>= n_frames, 4]`` = the episode's world poses from row t0.
    Vectorised equivalent of :meth:`pdm_proxy.AgentTracks.from_frames` (pinned equal in the tests).
    """
    rows, tids, clss = [], [], []
    for k in range(n_frames):
        a = reader.lookup(int(episode_id), int(t0) + k)
        if a is None:
            return None
        ti = reader.lookup_track_ids(int(episode_id), int(t0) + k)
        if a.shape[0] and (ti is None or len(ti) != a.shape[0]):
            raise Ddv2V7Error("the join reader carries no track ids for a labelled frame; build it "
                              "with with_track_ids=True (NC/TTC need tracks, not boxes)")
        cl = reader.lookup_classes(int(episode_id), int(t0) + k)
        rows.append(torch.as_tensor(a, dtype=torch.float32).reshape(-1, 6))
        tids.append([] if ti is None else [str(x) for x in ti])
        clss.append([""] * a.shape[0] if cl is None else [str(x) for x in cl])
    ids: dict[str, int] = {}
    for tl in tids:
        for t in tl:
            ids.setdefault(t, len(ids))
    T, A = n_frames, max(1, len(ids))
    p = ego_poses[:n_frames].to(torch.float32)
    xy = torch.zeros(T, A, 2)
    yaw = torch.zeros(T, A)
    lw = torch.zeros(A, 2)
    valid = torch.zeros(T, A, dtype=torch.bool)
    static = torch.zeros(A, dtype=torch.bool)
    x0, y0, h0 = p[0, 0], p[0, 1], p[0, 2]
    c0, s0 = torch.cos(-h0), torch.sin(-h0)
    for k in range(T):
        r = rows[k]
        if r.shape[0] == 0:
            continue
        j = torch.tensor([ids[t] for t in tids[k]], dtype=torch.long)
        hk = p[k, 2]
        ck, sk = torch.cos(hk), torch.sin(hk)
        wx = p[k, 0] + r[:, 0] * ck - r[:, 1] * sk
        wy = p[k, 1] + r[:, 0] * sk + r[:, 1] * ck
        dx, dy = wx - x0, wy - y0
        xy[k, j, 0] = dx * c0 - dy * s0
        xy[k, j, 1] = dx * s0 + dy * c0
        yaw[k, j] = torch.remainder(r[:, 2] + hk - h0 + math.pi, 2 * math.pi) - math.pi
        lw[j, 0], lw[j, 1] = r[:, 3], r[:, 4]
        valid[k, j] = True
        st = torch.tensor([c in cfg.static_classes for c in clss[k]], dtype=torch.bool)
        static[j] = static[j] | st
    d = torch.zeros(T, A)
    if T > 1:
        step = (xy[1:] - xy[:-1]).norm(dim=-1) / cfg.dt
        both = valid[1:] & valid[:-1]
        d[1:] = torch.where(both, step, torch.zeros_like(step))
        d[0] = torch.where(both[0], step[0], torch.zeros_like(step[0]))
    return P.AgentTracks(xy=xy, yaw=yaw, lw=lw, valid=valid, static=static, speed=d)


def filter_tracks(tracks: P.AgentTracks, states: Tensor, margin_m: float = 60.0) -> P.AgentTracks:
    """Drop tracks that never come within ``margin_m`` of the bounding box of every ego state.

    EXACT for NC and TTC: a footprint overlap needs centres within (ego half-diagonal 2.8 m +
    agent half-diagonal, < 15 m for any class in the join) and TTC projects the ego by at most
    0.9 s x 40.1 m/s (the corpus maximum, D3) = 36 m. 60 m covers both (pinned in the tests:
    filtered == unfiltered on a random scene). A cost measure, never a scoring change.
    """
    xs, ys = states[..., 0], states[..., 1]
    lo_x, hi_x = float(xs.min()) - margin_m, float(xs.max()) + margin_m
    lo_y, hi_y = float(ys.min()) - margin_m, float(ys.max()) + margin_m
    ax, ay = tracks.xy[..., 0], tracks.xy[..., 1]
    near = (tracks.valid & (ax >= lo_x) & (ax <= hi_x) & (ay >= lo_y) & (ay <= hi_y)).any(dim=0)
    keep = near.nonzero().flatten()
    empty = keep.numel() == 0
    if empty:                       # keep ONE slot, marked invalid everywhere: shapes stay legal
        keep = torch.zeros(1, dtype=torch.long, device=near.device)
    valid = tracks.valid.index_select(1, keep)
    if empty:
        valid = torch.zeros_like(valid)
    return P.AgentTracks(xy=tracks.xy.index_select(1, keep), yaw=tracks.yaw.index_select(1, keep),
                         lw=tracks.lw.index_select(0, keep), valid=valid,
                         static=tracks.static.index_select(0, keep),
                         speed=tracks.speed.index_select(1, keep))


# =========================================================================== #
# one window's reward                                                          #
# =========================================================================== #
def score_window(cand_states: Tensor, human_states: Tensor, tracks: P.AgentTracks, route: Tensor,
                 off: Tensor | None, cfg: P.ProxyConfig, rule: DacRule = DAC_RULE,
                 chunk: int = 160) -> dict:
    """``pdm_proxy.score_candidates`` with DAC from the 10 cm map, candidates chunked for memory.

    (!!) The human is scored IN EVERY CHUNK as proposal 0 (``score_candidates``' pairwise EP and the
    >=GT bar need it) and must read the SAME in every chunk -- asserted, so a chunk boundary can
    never change a score. ``off`` None = DAC dead (an ABLATION; the driver refuses it on a
    deployable arm).
    """
    m = cand_states.shape[0]
    tr = filter_tracks(tracks, torch.cat([human_states[None], cand_states], 0))
    dac_h = None if off is None else dac_fine(human_states[None], off, rule, cfg)[0]
    parts, human = [], None
    for s in range(0, m, chunk):
        cs = cand_states[s:s + chunk]
        dk = {} if off is None else {"dac_cand": dac_fine(cs, off, rule, cfg), "dac_human": dac_h}
        o = P.score_candidates(cs, human_states, tr, route, cfg=cfg, **dk)
        if human is None:
            human = o["human"]
        elif any(abs(human[k] - o["human"][k]) > 0 for k in human):
            raise Ddv2V7Error("the human scored differently in two chunks -- a chunk changed a score")
        parts.append(o)
    out = {k: torch.cat([p_[k] for p_ in parts]) for k in
           ("nc", "dac", "ep", "ttc", "comfort", "spd", "pdms", "constraint_fail")}
    out["human"] = human
    out["n_tracks_scored"] = int(tr.valid.any(dim=0).sum())
    return out


def microbatch_step_loss(logp_i: Tensor, coef_rl_rows_i: Tensor, il_coef: Tensor, il_i: Tensor,
                         frac: float) -> tuple[Tensor, Tensor]:
    """The release's per-step loss (``ddv2_rl.per_step_loss``) for ONE micro-batch of a larger step.

    ``coef_rl_rows_i [m, M]`` = the FULL-batch ``step_loss_weights(...)["coef_rl"]`` rows of this
    micro-batch at step ``i`` (its ``1 / (T * B)`` and per-row non-zero counts already use the FULL
    batch ``B``); ``il_i`` = this micro-batch's batch-global mean L1; ``frac = m / B``. Summed over
    the micro-batches this is ``per_step_loss`` of the full batch EXACTLY in real arithmetic (the
    release's IL is a batch mean, so equal-sized micro-batches average to it) -- pinned in the tests.
    -> ``(loss_i, rl_i)``.
    """
    r = torch.exp(logp_i - logp_i.detach())
    rl_i = (-(r * coef_rl_rows_i)).sum()
    return rl_i + il_coef * il_i * float(frac), rl_i


def shuffle_permutation(b: int, gen: torch.Generator) -> Tensor:
    """A DERANGEMENT of ``range(b)`` (no window keeps its own reward) for the RL-SHUF arm.
    ``b == 1`` cannot be deranged and is refused."""
    if b < 2:
        raise Ddv2V7Error("RL-SHUF needs >= 2 windows per step to shuffle rewards ACROSS scenes")
    perm = torch.randperm(b, generator=gen)
    while bool((perm == torch.arange(b)).any()):
        perm = torch.randperm(b, generator=gen)
    return perm


def trainable_named(decoder) -> list[tuple[str, torch.nn.Parameter]]:
    """``(name, param)`` of the trajectory generator (``TRAINABLE_PREFIXES``), in module order."""
    out = []
    for n, p in decoder.named_parameters():
        if n.split(".")[0] in TRAINABLE_PREFIXES:
            out.append((n, p))
    if not out:
        raise Ddv2V7Error("no trainable generator parameter found")
    return out
