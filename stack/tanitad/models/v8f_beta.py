"""v8F rung BETA -- the world model as an EVAL-TIME scorer of the 117-candidate fan. ZERO TRAINING.

Spec: ``Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`` §4.3 / §6 (rung beta) and ``Project Steering/PREREG_V8F_BETA.md``.

WHAT IT DOES. For every eval window, v8F-mini's OWN world model (``V8FMini.wm``, trained only as an auxiliary) rolls level 1 under each
of 117 candidate plans, lifts the rollout through E^2 (``HJEPAStack.score_candidates``) and scores the squared distance of the
level-2 (tactical, 0.4 s) latent at +6 s to a PREDICTED goal latent g2(+6 s). M2's own 6 s plan is a 118th candidate, the INCUMBENT:
it is replaced only by a fan member whose cost is STRICTLY lower, so a constant cost reproduces M2's output bit for bit.

THE FAN (the plan's "117 candidates"). A 13 x 9 product grid of CONSTANT (a_lon, a_lat) controls, a_lon in {-4.0} U {k * 7/12 : k = -6..5}
m/s^2 and a_lat in {-3.0 .. 3.0 step 0.75} m/s^2, rolled for 60 x 0.1 s from the window's MEASURED speed v0 through
``tanitad.models.kinematic.rollout_unicycle`` with kappa = clamp(a_lat / max(v0, 4)^2, +-0.12). That is ``AnchorDecoder.roll_bank``
(``stack/tanitad/refs/refc.py:2246`` ff., the ``anchor_control_units == "alat"`` branch) and the geometry pinned by
``stack/tests/test_anchor_flyability.py:11-22`` ("the shipped refcv4b geometry"). ``test_v8f_beta.py`` checks :func:`fan_paths`
against a REAL ``AnchoredDiffusionDecoder.roll_bank``. The shipped file itself (``anchors_117_alat_declared.pt``) lives on a pod; pass it
as ``--fan-artifact`` and :func:`load_fan_controls` uses ITS controls and records the distance to the pinned grid.

THE GOAL (the binding rule of 2026-08-03). g2 is PREDICTED and information-disjoint from the situation classifier AND from the tactical
layer: it is the world model's level-2 latent at the end of the route the STRATEGIC layer predicted -- S's checkpoints (30 / 50 / 80 m,
ego frame, S reads scene + nav + distance + ego + limit and NOT T), timed by the frozen P10 prior's progress profile (measured ego motion
only). :func:`route_goal_xy` takes exactly ``(ck_now, p30)``; nothing else can reach the goal. Shared trunk: S, T and O share the
frozen-feature adapter, and the goal path reads the same adapted scene through S -- the rule bans carrying the classifier's OUTPUT, not
the trunk (plan §4.3 lists "trunk features" as an allowed input of g2).

THE ACTION CONVENTION IS TRAINING'S. ``WMAux`` pairs the transition j -> j+1 with the action channels MEASURED at cache frame j
(backward differences: ``a`` over 0.2 s, the yaw rate over one 0.1 s step). :func:`wm_action_sequence` derives the same two channels from
ANY 6 s path (a candidate, the goal path, M2's plan) with ONE function, so the three are comparable; transition 0 is the MEASURED
(a, omega) at NOW for every candidate. Scaling goes through ``WMAux.scale_actions`` -- the code training uses. No steering angle appears
anywhere (D-PI-YAWRATE-1): the second action channel is the yaw rate.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tanitad.models import v8f_mini as M
from tanitad.models.kinematic import rollout_unicycle

DT = 0.1                     # the native step: the fan, traj60 and the WM-action derivation
N_STEP = 60                  # 6 s
H_WM = 30                    # level-1 transitions of the WM: the 0.2 s cache grid over 6 s
WM_LEVEL = 2                 # the tactical latent
N_FAN = 117
KAPPA_CAP, ALAT_V_FLOOR = 0.12, 4.0
FAR_M = 1000.0               # the route polyline continues this far past S's last checkpoint, along its heading
EPS_MOVE = 1e-6              # a step shorter than this has no heading (held), as in ``v8f_mini.plan_kin_t``
SCORE_CALLS = {"goal": 0, "candidates": 0}      # evidence for the reachability audit: how often the WM scorer really ran


def reset_counters() -> None:
    SCORE_CALLS["goal"] = SCORE_CALLS["candidates"] = 0


# ------------------------------------------------------------------------------------------------ the fan
def fan_controls() -> Tensor:
    """[117, 2] = (a_lon, a_lat) m/s^2: the 13 x 9 product grid of the shipped refcv4b vocabulary (``test_anchor_flyability.py:11-22``)."""
    a_lat = torch.arange(-3.0, 3.01, 0.75)
    a_lon = torch.cat([torch.tensor([-4.0]), torch.arange(-6, 6) * (7.0 / 12.0)])
    lon, lat = torch.meshgrid(a_lon, a_lat, indexing="ij")
    return torch.stack([lon.reshape(-1), lat.reshape(-1)], -1)


def load_fan_controls(src) -> tuple:
    """``(controls [117, 2], record)`` from a DECLARED anchor artifact (``anchors_117_alat_declared.pt``): refuses undeclared units, a
    non-``alat`` file, a non-constant schedule and any width but 117. ``record`` carries the distance to the pinned grid."""
    from tanitad.refs import anchor_meta as AM
    art = AM.read_anchor_artifact(src)
    if art.controls is None or art.control_units != "alat" or art.control_schedule != AM.CONSTANT_SCHEDULE \
            or tuple(art.controls.shape) != (N_FAN, 2):
        raise ValueError(f"fan artifact must be a 117 x 2 constant-schedule 'alat' file; got units {art.control_units!r}, schedule "
                         f"{art.control_schedule!r}, controls {None if art.controls is None else tuple(art.controls.shape)}")
    ctrl = art.controls.detach().float().cpu()
    rec = {"source": "artifact", "path": art.path, "control_units_source": art.control_units_source,
           "max_abs_diff_vs_pinned_grid": float((ctrl - fan_controls()).abs().max())}
    if art.path:
        h = hashlib.sha256()
        with open(art.path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        rec["sha256"] = h.hexdigest()
    return ctrl, rec


def fan_paths(v0: Tensor, controls: Tensor | None = None) -> Tensor:
    """``v0`` [B] (the MEASURED speed at NOW) -> the fan's ego-frame paths [B, 117, 60, 2] at 0.1 s: ``AnchorDecoder.roll_bank``'s alat branch."""
    ctrl = (fan_controls() if controls is None else controls).to(v0.device, torch.float32)
    n, b = ctrl.shape[0], v0.shape[0]
    v = v0.reshape(-1).float()
    kap = (ctrl[None, :, 1] / (v.clamp_min(ALAT_V_FLOOR) ** 2)[:, None]).clamp(-KAPPA_CAP, KAPPA_CAP)       # [B, N]
    c = torch.stack([ctrl[None, :, 0].expand(b, n), kap], -1)
    c = c[:, :, None, :].expand(b, n, N_STEP, 2).reshape(-1, N_STEP, 2)
    s0 = torch.zeros(b * n, 4, device=v.device, dtype=torch.float32)
    s0[:, 3] = v[:, None].expand(b, n).reshape(-1)
    return rollout_unicycle(s0, c, dt=DT)[..., :2].reshape(b, n, N_STEP, 2)


# ------------------------------------------------------------------------------------------------ path -> the WM's action channels
def chord_kinematics(xy: Tensor) -> tuple:
    """``xy`` [..., K, 2] ego-frame positions at DT spacing from the origin -> (speed [..., K], heading [..., K]) of the K chord steps.
    A step shorter than ``EPS_MOVE`` keeps the previous heading (the ego heading, 0, before the first move)."""
    lead = xy.shape[:-2]
    p = torch.cat([xy.new_zeros(*lead, 1, 2), xy], -2)
    seg = p[..., 1:, :] - p[..., :-1, :]
    ds = torch.linalg.norm(seg, dim=-1)
    ang = torch.atan2(seg[..., 1], seg[..., 0])
    cur = xy.new_zeros(*lead)
    heads = []
    for k in range(xy.shape[-2]):
        cur = torch.where(ds[..., k] > EPS_MOVE, ang[..., k], cur)
        heads.append(cur)
    return ds / DT, torch.stack(heads, -1)


def wm_action_sequence(xy: Tensor, now_act: Tensor) -> Tensor:
    """A 6 s path ``xy`` [..., 60, 2] (0.1 s, ego frame) -> the 30 RAW action pairs (a [m/s^2], omega [rad/s]) of the WM's 0.2 s grid.

    Transition j uses the channels at frame 2j, derived like the cache does: ``a`` = the speed difference over 0.2 s, ``omega`` = the
    one-step (0.1 s) wrapped heading difference. Transition 0 is ``now_act`` [..., 2], the MEASURED (a, omega) at NOW, shared by every
    path. Speeds / headings are the chord quantities of the path (:func:`chord_kinematics`), the same for a candidate, the goal path
    and M2's plan. A constant-acceleration path returns exactly its acceleration at every j >= 1."""
    if xy.shape[-2] != N_STEP:
        raise ValueError(f"expected {N_STEP} path points at {DT} s, got {xy.shape[-2]}")
    if tuple(now_act.shape) != tuple(xy.shape[:-2]) + (2,):
        raise ValueError(f"now_act {tuple(now_act.shape)} must be {tuple(xy.shape[:-2]) + (2,)}")
    speed, head = chord_kinematics(xy)
    dev = xy.device
    j = torch.arange(2, H_WM, device=dev)
    a_rest = (speed[..., 2 * j - 1] - speed[..., 2 * j - 3]) / (2 * DT)
    a_1 = ((speed[..., 1] - speed[..., 0]) / DT).unsqueeze(-1)
    acc = torch.cat([now_act[..., 0:1], a_1, a_rest], -1)
    jj = torch.arange(1, H_WM, device=dev)
    om = M.wrap(head[..., 2 * jj - 1] - head[..., 2 * jj - 2]) / DT
    omega = torch.cat([now_act[..., 1:2], om], -1)
    return torch.stack([acc, omega], -1)


# ------------------------------------------------------------------------------------------------ the predicted goal path
def route_goal_xy(ck_now: Tensor, p30: Tensor) -> Tensor:
    """The route S predicted, travelled at the pace of the frozen P10 prior: ``ck_now`` [B, 3, 3] (x, y, psi of the 30 / 50 / 80 m
    checkpoints, ego frame AT NOW) and ``p30`` [B, 30, 2] (P10, 0.2 s) -> a goal path [B, 60, 2] at 0.1 s.

    The route is the polyline origin -> ck30 -> ck50 -> ck80 -> a far point along ck80's heading; the pace is P10's arc length at each
    0.1 s step. ADMISSIBILITY (binding 2026-08-03): these two arguments are ALL a goal can be computed from -- no tactical code, no goal
    token, no lateral / longitudinal logit, no constraint regression, no label."""
    b = ck_now.shape[0]
    p60 = M.upsample_prior(p30)
    pr = torch.cat([p60.new_zeros(b, 1, 2), p60], 1)
    s = torch.cumsum(torch.linalg.norm(pr[:, 1:] - pr[:, :-1], dim=-1), 1)                  # [B, 60] P10's arc at tau = 0.1 .. 6.0
    far = ck_now[:, 2, :2] + FAR_M * torch.stack([torch.cos(ck_now[:, 2, 2]), torch.sin(ck_now[:, 2, 2])], -1)
    v = torch.cat([ck_now.new_zeros(b, 1, 2), ck_now[..., :2], far[:, None]], 1)             # [B, 5, 2]
    seg = v[:, 1:] - v[:, :-1]
    ell = torch.linalg.norm(seg, dim=-1).clamp_min(1e-6)                                    # [B, 4]
    cum = torch.cat([ell.new_zeros(b, 1), torch.cumsum(ell, 1)], 1)                          # [B, 5]
    idx = (torch.searchsorted(cum.contiguous(), s.contiguous(), right=True) - 1).clamp(0, 3)
    frac = (s - torch.gather(cum, 1, idx)) / torch.gather(ell, 1, idx)
    return torch.gather(v, 1, idx[..., None].expand(-1, -1, 2)) + frac[..., None] * torch.gather(seg, 1, idx[..., None].expand(-1, -1, 2))


# ------------------------------------------------------------------------------------------------ inputs
@dataclass
class BetaInputs:
    """One row per window, CPU float32. Everything the planner may read; ``traj60`` is only a CANDIDATE (the incumbent), never a goal."""
    z1_0: Tensor        # [N, D1]  the WM's level-1 latent at NOW
    v0: Tensor          # [N]      MEASURED speed at NOW (the fan is rolled from it)
    now_act: Tensor     # [N, 2]   MEASURED (a, omega) at NOW
    ck_now: Tensor      # [N, 3, 3] S's checkpoints, re-expressed in the NOW frame with the measured odometry
    p30: Tensor         # [N, 30, 2] the frozen P10 prior (measured ego motion only)
    traj60: Tensor      # [N, 60, 2] M2's own 6 s plan (the incumbent)

    def __len__(self):
        return int(self.z1_0.shape[0])

    def take(self, sl) -> "BetaInputs":
        return BetaInputs(*(getattr(self, f)[sl] for f in _INPUT_FIELDS))


_INPUT_FIELDS = ("z1_0", "v0", "now_act", "ck_now", "p30", "traj60")


def collect_beta_inputs(model, batch: dict, out: dict) -> dict:
    """The planner's per-window inputs from ONE ``forward_infer`` call: ``batch`` is read through a key guard and may touch only
    ``M.INFERENCE_INPUT_KEYS``; ``out`` is the dict ``forward_infer`` returned (only ``adapt``, ``ckpt``, ``p10``, ``traj60`` are read --
    never ``goal_prob`` / ``lat_logits`` / ``lon_logits`` / ``cons_pred``). Returns CPU tensors keyed like :class:`BetaInputs`."""
    if model.wm is None:
        raise ValueError("rung beta needs a checkpoint with a world model (arm M2a / M2b); this one was built with wm='off'")
    g = M._KeyGuard(batch)
    with torch.no_grad():
        z1_0 = model.wm.z1(out["adapt"])[:, -1]
        p10f, act = g["p10f"], g["act"]
        ck_now = M.transform_ckpt(out["ckpt"].float(), g["pose_s"], g["pose_now"])
        row = {"z1_0": z1_0, "v0": p10f[:, -1, 0], "now_act": act[:, -1], "ck_now": ck_now, "p30": out["p10"], "traj60": out["traj60"]}
    extra = g.read - M.INFERENCE_INPUT_KEYS
    assert not extra, f"beta collected inputs outside the admissible set: {sorted(extra)}"
    return {k: v.detach().float().cpu() for k, v in row.items()}


def cat_inputs(rows: list) -> BetaInputs:
    return BetaInputs(*(torch.cat([r[f] for r in rows]) for f in _INPUT_FIELDS))


# ------------------------------------------------------------------------------------------------ the scorer
def _dev(wm) -> torch.device:
    return next(wm.parameters()).device


@torch.no_grad()
def goal_latents(wm, inp: BetaInputs, goal_xy60: Tensor, *, chunk: int = 32) -> Tensor:
    """g2(+6 s) [N, D2]: the WM's level-2 latent at the END of the rollout under ``goal_xy60`` [N, 60, 2] -- the SAME rollout + lift
    (``score_candidates``) that scores a candidate."""
    dev, d2 = _dev(wm), wm.stack.cfg[WM_LEVEL - 1].latent_dim
    out = []
    for i0 in range(0, len(inp), chunk):
        sl = slice(i0, i0 + chunk)
        z1, now = inp.z1_0[sl].to(dev), inp.now_act[sl].to(dev)
        acts = wm.scale_actions(wm_action_sequence(goal_xy60[sl].to(dev)[:, None], now[:, None]))          # [b, 1, 30, 2]
        SCORE_CALLS["goal"] += 1
        _, det = wm.stack.score_candidates(z1, acts, z1.new_zeros(z1.shape[0], d2), level=WM_LEVEL, return_details=True)
        out.append(det["lifted"][:, 0, -1].cpu())
    return torch.cat(out)


@torch.no_grad()
def plan_beta(wm, inp: BetaInputs, goal: Tensor, *, controls: Tensor | None = None, chunk: int = 16) -> dict:
    """Rank the fan against ``goal`` [N, D2] with the world model and return, per window:

    ``traj30`` [N, 30, 2]   BETA's plan: the incumbent (M2's own) unless a fan member's cost is STRICTLY lower;
    ``replaced`` [N] bool   whether it was replaced; ``fan_idx`` [N] the best fan member (valid for every window);
    ``traj30_fan`` [N, 30, 2]  the best fan member itself (diagnostic arms rank the fan WITHOUT the incumbent);
    ``cost_inc`` / ``cost_min`` [N]   the incumbent's cost and the best fan cost;
    ``end_bar`` [N, D2], ``end_sig`` [N]   the fan's level-2 endpoint latents' mean and within-window spread (WM-sensitivity diagnostic).
    A non-finite cost is +inf. Ties go to the lowest fan index and then to the incumbent."""
    dev = _dev(wm)
    res = {k: [] for k in ("traj30", "replaced", "fan_idx", "traj30_fan", "cost_inc", "cost_min", "end_bar", "end_sig")}
    for i0 in range(0, len(inp), chunk):
        sl = slice(i0, i0 + chunk)
        z1, v0, now = inp.z1_0[sl].to(dev), inp.v0[sl].to(dev), inp.now_act[sl].to(dev)
        inc = inp.traj60[sl].to(dev)
        fan = fan_paths(v0, None if controls is None else controls.to(dev))                  # [b, 117, 60, 2]
        xy = torch.cat([inc[:, None], fan], 1)                                              # [b, 118, 60, 2]; column 0 = the incumbent
        acts = wm.scale_actions(wm_action_sequence(xy, now[:, None].expand(-1, xy.shape[1], -1)))
        SCORE_CALLS["candidates"] += 1
        cost, det = wm.stack.score_candidates(z1, acts, goal[sl].to(dev), level=WM_LEVEL, return_details=True)   # BETA-SCORING-LINE
        cost = torch.where(torch.isfinite(cost), cost, torch.full_like(cost, float("inf")))
        best_c, best_i = cost[:, 1:].min(1)
        replaced = best_c < cost[:, 0]                                                       # BETA-STRICT-IMPROVEMENT
        fan30 = fan[:, :, 1::2]
        pick = fan30[torch.arange(fan30.shape[0], device=dev), best_i]
        inc30 = inc[:, 1::2]
        end = det["lifted"][:, 1:, -1]
        bar = end.mean(1)
        res["traj30"].append(torch.where(replaced[:, None, None], pick, inc30).cpu())
        res["replaced"].append(replaced.cpu())
        res["fan_idx"].append(best_i.cpu())
        res["traj30_fan"].append(pick.cpu())
        res["cost_inc"].append(cost[:, 0].cpu())
        res["cost_min"].append(best_c.cpu())
        res["end_bar"].append(bar.cpu())
        res["end_sig"].append((end - bar[:, None]).norm(dim=-1).mean(1).cpu())
    return {k: torch.cat(v) for k, v in res.items()}


@torch.no_grad()
def fan_gt_arms(inp: BetaInputs, gt: Tensor, *, controls: Tensor | None = None, seed: int = 0, chunk: int = 64) -> dict:
    """The two world-model-free references on the SAME fan: a uniformly random member (chance) and the GT-best member (the ceiling the
    vocabulary allows; T0 -- it reads the ground truth). Both are DIAGNOSTICS, never a result."""
    g = torch.Generator().manual_seed(seed)
    rnd, best = [], []
    for i0 in range(0, len(inp), chunk):
        sl = slice(i0, i0 + chunk)
        fan30 = fan_paths(inp.v0[sl], controls)[:, :, 1::2]
        ade = torch.linalg.norm(fan30 - gt[sl].float()[:, None], dim=-1).mean(-1)            # [b, 117]
        ar = torch.arange(fan30.shape[0])
        best.append(fan30[ar, ade.argmin(1)])
        rnd.append(fan30[ar, torch.randint(fan30.shape[1], (fan30.shape[0],), generator=g)])
    return {"fan_random": torch.cat(rnd), "fan_oracle": torch.cat(best)}


def run_beta(wm, inp: BetaInputs, *, gt: Tensor | None = None, eid=None, controls: Tensor | None = None, chunk: int = 16,
             seed: int = 0) -> dict:
    """All arms of the gate for one checkpoint. ``beta`` is the only RESULT; the rest are controls and T0 diagnostics:
    ``beta_shuf`` (the plan's shuffled-goal control: g2 from a window of ANOTHER episode), ``beta_oracle`` (g2 from the GROUND-TRUTH
    path -- the discriminating experiment for 'is z2 a usable ranking space'; T0), ``fan_random`` / ``fan_oracle``."""
    from tanitad.models.hjepa import donor_indices
    g_own = goal_latents(wm, inp, route_goal_xy(inp.ck_now, inp.p30), chunk=chunk)
    out = {"beta": plan_beta(wm, inp, g_own, controls=controls, chunk=chunk)}
    if eid is not None and len(set(int(e) for e in eid)) >= 2:
        perm = donor_indices(len(inp), episode_ids=torch.as_tensor(eid), seed=seed)
        out["beta_shuf"] = plan_beta(wm, inp, g_own[perm], controls=controls, chunk=chunk)
    if gt is not None:
        gtt = torch.as_tensor(gt).float()
        out["beta_oracle"] = plan_beta(wm, inp, goal_latents(wm, inp, M.upsample_prior(gtt), chunk=chunk), controls=controls, chunk=chunk)
        out.update({k: {"traj30": v} for k, v in fan_gt_arms(inp, gtt, controls=controls, seed=seed).items()})
    return out


def endpoint_spread_ratio(plan: dict) -> float:
    """The fan's WITHIN-window spread of level-2 endpoints / the BETWEEN-window spread of their means. ~0 => the WM barely tells the
    candidates apart (ranking = noise); ~1 => the candidates differ as much as different scenes do."""
    bar = plan["end_bar"].double()
    between = (bar - bar.mean(0)).norm(dim=-1).mean()
    return float(plan["end_sig"].double().mean() / between.clamp_min(1e-12))
