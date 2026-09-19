#!/usr/bin/env python3
"""s1_gate.py -- the collision-gate harness of ``PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md``
(S1 AMENDMENT + ERRATUM-1, 2026-09-19): every inference-only S1 arm read POST-HOC from ONE
forward pass over refcv6's own fan.

This file holds the CHECKPOINT-FREE core: the waypoint->state expansion, the three track
builders the gate can see, the selection rule, and the per-window arm table. The forward
pass that feeds it lives in ``s1_pass.py`` (it needs a checkpoint).

⛔ THE CHECKER IS IMPORTED, NEVER RE-IMPLEMENTED: ``tanitad.rl.pdm_proxy`` -- the exact module
behind item 19's 28/493 (``ddv2_rl_refcv5.py:274-311``). ``taniteval/tools/fan_safety.py``'s
lead-only 2 m ``_collision`` is a DIFFERENT model and is not used anywhere here.

⛔ EVERY TRACK SET GOES THROUGH ``AgentTracks.from_frames`` ITSELF. ORACLE uses the recorded
join frames (item 19's own call). PRED, ORACLE-CV and CONST build synthetic per-tick frames in
the t0 ego frame and pass a CONSTANT ego pose, so ``from_frames``' transform is the identity
and its OWN conventions -- ``static`` by CLASS (``pdm_proxy.py:246``), ``speed`` by finite
difference (``:247-252``), validity -- apply unchanged. A parallel constructor would be a
second definition of "an agent", and the gate would be tested against a different world
than the one it is scored in.

⛔ EVERY SELECTED CANDIDATE IS SCORED AGAINST THE RECORDED FUTURE (``score_true``), whichever
tracks gated it. A gate scored against its own tracks certifies anything.

⛔ N IS READ FROM THE FAN'S SHAPE, NEVER WRITTEN AS A LITERAL (ERRATUM-1: the dev-box bank is
128 synthetic anchors, not the 117 of refcv4b's fitted bank).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.rl import pdm_proxy as P  # noqa: E402  -- the item-19 checker, unchanged

#: the arms read from one pass. ORACLE / ORACLE_CV see privileged recorded agents -> T0.
ARMS = ("BASE", "RANDOM", "GATE_ORACLE", "GATE_CONST", "GATE_PRED", "ORACLE_CV")
TIER = {"BASE": "T1*", "RANDOM": "T1*", "GATE_ORACLE": "T0", "GATE_CONST": "T1*",
        "GATE_PRED": "T1*", "ORACLE_CV": "T0"}
#: agent ticks the proxy reads (``ddv2_rl_refcv5.py:77``): n_ticks + 1 + max(ttc offsets)
AGENT_TICKS = P.PROXY.n_ticks + 1 + max(P.PROXY.ttc_offsets)
PRESENCE_THRESH = 0.5                  # S1A.4, fixed before data
LOW_SPEED_YAW = 0.1                    # m/s: below this the tangent carries no heading


# ============================================================================ waypoints
def expand_waypoints(wp, horizons, v0: float, n_ticks: int, dt: float):
    """A candidate's slots -> the proxy's dense grid, in the t0 EGO frame.

    ``wp [M, S, 2]`` metres at ``horizons`` (ticks). Returns numpy ``(x, y, yaw, v)``, each
    ``[M, n_ticks + 1]``, tick 0 = the ego at t0 = ``(0, 0, 0, v0)``.

    ⭐ S1A.2: a C² cubic spline in TIME through the origin and every slot, with the initial
    velocity CLAMPED to ``(v0, 0)`` (the measured, admissible v0 -- PI 2026-09-02) and a
    natural end. Yaw is the tangent and speed is |derivative|. Below ``LOW_SPEED_YAW`` the
    tangent carries no heading, so the last defined yaw is held.
    """
    from scipy.interpolate import CubicSpline      # lazy: not a core dependency
    wp = np.asarray(wp, dtype=np.float64)
    if wp.ndim != 3 or wp.shape[-1] != 2:
        raise ValueError(f"wp must be [M, S, 2], got {wp.shape}")
    h = np.asarray(horizons, dtype=np.float64)
    if h.shape[0] != wp.shape[1] or np.any(np.diff(h) <= 0) or h[0] <= 0:
        raise ValueError(f"horizons {tuple(h)} do not match {wp.shape[1]} increasing slots")
    if h[-1] < n_ticks:
        raise ValueError(f"the fan ends at tick {h[-1]:.0f} < the proxy's {n_ticks}: "
                         f"extrapolating a spline past its last knot is not scoring")
    m = wp.shape[0]
    tk = np.concatenate([[0.0], h * dt])
    pts = np.concatenate([np.zeros((1, m, 2)), np.transpose(wp, (1, 0, 2))], axis=0)
    tq = np.arange(n_ticks + 1) * dt
    csx = CubicSpline(tk, pts[..., 0], axis=0,
                      bc_type=((1, np.full(m, float(v0))), (2, np.zeros(m))))
    csy = CubicSpline(tk, pts[..., 1], axis=0,
                      bc_type=((1, np.zeros(m)), (2, np.zeros(m))))
    x, y = csx(tq).T, csy(tq).T                              # [M, T+1]
    dx, dy = csx(tq, 1).T, csy(tq, 1).T
    v = np.hypot(dx, dy)
    yaw = np.arctan2(dy, dx)
    yaw[:, 0] = 0.0                                          # the ego's own heading at t0
    for k in range(1, yaw.shape[1]):                         # hold heading when stopped
        slow = v[:, k] < LOW_SPEED_YAW
        yaw[slow, k] = yaw[slow, k - 1]
    return x, y, yaw, v


def candidate_states(wp, horizons, pose_last, cfg=P.PROXY) -> torch.Tensor:
    """``[M, n_ticks + 1, 4]`` through ``pdm_proxy.ego_states_from_poses`` -- the constructor
    the HUMAN is scored with (``ddv2_rl_refcv5.py:241``), so candidate and human share one
    construction, rotation convention and yaw wrap."""
    pl = torch.as_tensor(pose_last, dtype=torch.float64).reshape(4)
    x, y, yaw, v = expand_waypoints(wp, horizons, float(pl[3]), cfg.n_ticks, cfg.dt)
    x0, y0, h0 = float(pl[0]), float(pl[1]), float(pl[2])
    c, s = math.cos(h0), math.sin(h0)
    wx, wy = x0 + x * c - y * s, y0 + x * s + y * c         # ego@t0 -> world
    fut = np.stack([wx, wy, yaw + h0, v], axis=-1)[:, 1:]   # ticks 1..n
    m = fut.shape[0]
    return P.ego_states_from_poses(pl[None].expand(m, 4).float(),
                                   torch.as_tensor(fut, dtype=torch.float32), cfg)


# ============================================================================ tracks
def _tracks_from_t0_agents(agents: list[dict], pose_last, n_ticks: int = AGENT_TICKS,
                           dt: float = P.PROXY.dt, cfg=P.PROXY) -> P.AgentTracks:
    """Constant-velocity agents in the t0 ego frame -> ``AgentTracks`` via ``from_frames``.

    Each agent dict: ``track_id, cx, cy, yaw, l, w, cls, vx, vy`` with ``(vx, vy)`` the
    velocity IN THE FIXED t0 FRAME. The ego pose is held CONSTANT at t0, so ``from_frames``'
    ego@k -> world -> ego@t0 transform is the identity (asserted by a test), and its own
    ``static``-by-class and finite-difference ``speed`` apply unchanged.
    """
    frames = []
    for k in range(n_ticks):
        tk = k * dt
        frames.append([{"track_id": str(a["track_id"]), "cx": float(a["cx"] + a["vx"] * tk),
                        "cy": float(a["cy"] + a["vy"] * tk), "yaw": float(a["yaw"]),
                        "l": float(a["l"]), "w": float(a["w"]), "cls": str(a["cls"])}
                       for a in agents])
    poses = torch.as_tensor(pose_last, dtype=torch.float32).reshape(1, 4).repeat(n_ticks, 1)
    return P.AgentTracks.from_frames(frames, poses, cfg)


def empty_tracks(pose_last, n_ticks: int = AGENT_TICKS, cfg=P.PROXY) -> P.AgentTracks:
    """CONST: 'everything is free' -- no agent at all, through the same constructor."""
    return _tracks_from_t0_agents([], pose_last, n_ticks, cfg.dt, cfg)


def cv_tracks_from_recorded(rec: P.AgentTracks, pose_last, cfg=P.PROXY) -> P.AgentTracks:
    """ORACLE-CV (S1A.5): each agent recorded AT t0, moved at its recorded t0 velocity.

    The t0 velocity is ``from_frames``' own tick-0 rule: the finite difference to tick 1 when
    both ticks are valid, else zero. Class is re-expressed as the one string that reproduces
    the recorded ``static`` flag, so the class rule gives back exactly what was recorded.
    """
    agents = []
    stat_cls = cfg.static_classes[0]
    for j in range(rec.xy.shape[1]):
        if not bool(rec.valid[0, j]):
            continue
        if rec.xy.shape[0] > 1 and bool(rec.valid[1, j]):
            vx, vy = ((rec.xy[1, j] - rec.xy[0, j]) / cfg.dt).tolist()
        else:
            vx = vy = 0.0
        agents.append({"track_id": j, "cx": float(rec.xy[0, j, 0]),
                       "cy": float(rec.xy[0, j, 1]), "yaw": float(rec.yaw[0, j]),
                       "l": float(rec.lw[j, 0]), "w": float(rec.lw[j, 1]),
                       "cls": stat_cls if bool(rec.static[j]) else "vehicle",
                       "vx": vx, "vy": vy})
    return _tracks_from_t0_agents(agents, pose_last, rec.xy.shape[0], cfg.dt, cfg)


def pred_agents_from_decoded(dec: dict, v0: float, classes,
                             presence_thresh: float = PRESENCE_THRESH) -> list[dict]:
    """PRED (S1A.4): the box head's decoded slots for ONE window -> t0 agents.

    ``dec`` is ``AgentSlotHead.decode(raw)`` for one window (leading batch dim removed):
    ``presence_logit [N]``, ``cls_logits [N, C]``, ``box [N, 4] = (cx, cy, l, w)``,
    ``yaw [N]``, ``rates [N, 3] = (v_rel_x, v_rel_y, yaw_rate_rel)``.

    ⚠️ ``v_rel`` is d/dt of the agent's position IN THE EGO FRAME (``agent_slots.py:64-66``),
    so its velocity in the FIXED t0 frame is ``v_rel + (v0, 0)`` plus an ego-rotation term
    ``omega x r``. The ego yaw rate is NOT an admissible input (v0 only), so that term is
    dropped, and it is stated here. Yaw is held constant (S1A.4).
    """
    p = torch.sigmoid(dec["presence_logit"].float())
    keep = (p > presence_thresh).nonzero().flatten().tolist()
    cls_idx = dec["cls_logits"].float().argmax(dim=-1)
    out = []
    for j in keep:
        cx, cy, ln, wd = dec["box"][j].float().tolist()
        vrx, vry = dec["rates"][j, 0].item(), dec["rates"][j, 1].item()
        out.append({"track_id": f"pred{j}", "cx": cx, "cy": cy,
                    "yaw": float(dec["yaw"][j]), "l": ln, "w": wd,
                    "cls": str(classes[int(cls_idx[j])]),
                    "vx": vrx + float(v0), "vy": vry})
    return out


def pred_tracks(dec: dict, pose_last, classes, cfg=P.PROXY) -> P.AgentTracks:
    v0 = float(torch.as_tensor(pose_last).reshape(4)[3])
    return _tracks_from_t0_agents(pred_agents_from_decoded(dec, v0, classes), pose_last,
                                  AGENT_TICKS, cfg.dt, cfg)


# ============================================================================ selection
def select(rank: torch.Tensor, model_keep: torch.Tensor | None = None,
           gate_ok: torch.Tensor | None = None) -> tuple[int, bool]:
    """The model's OWN rule (argmax of its ranking score with its own mask as -inf,
    ``refc_v3.py`` ~2069-2079), plus ONE extra -inf mask for candidates the gate's tracks
    say collide. If the gate blocks every candidate, BASE's pick stands (PREREG_S1 §3:
    *never select a colliding candidate WHILE a collision-free one is in the fan*).
    Returns ``(index, fell_back)``."""
    r = rank.detach().double().clone()
    if model_keep is not None and bool(model_keep.any()):
        r = r.masked_fill(~model_keep.bool(), float("-inf"))
    base = int(r.argmax())
    if gate_ok is None:
        return base, False
    rg = r.masked_fill(~gate_ok.bool(), float("-inf"))
    if not bool(torch.isfinite(rg).any()):
        return base, True
    return int(rg.argmax()), False


def gate_ok(states: torch.Tensor, tracks: P.AgentTracks, cfg=P.PROXY) -> torch.Tensor:
    """``[M]`` bool: no at-fault collision under THESE tracks -- the imported checker."""
    return P.no_at_fault_collision(states, tracks, cfg) == 1


def score_true(states, human, tracks_rec, route, dac_cand=None, dac_human=None,
               cfg=P.PROXY) -> dict:
    """Every candidate scored ONCE against the RECORDED future. EP is normalised pairwise
    against the human only (``pdm_proxy.py`` ``score_candidates``), so a candidate's scores do
    not depend on the rest of the fan and any arm's pick is an index into this dict."""
    return P.score_candidates(states, human, tracks_rec, route, dac_cand=dac_cand,
                              dac_human=dac_human, cfg=cfg)


def window_arms(states, rank, model_keep, tracks: dict, cfg=P.PROXY) -> dict:
    """``tracks`` maps ``GATE_ORACLE`` / ``GATE_CONST`` / ``GATE_PRED`` / ``ORACLE_CV`` to the
    AgentTracks that arm's GATE sees (any subset). -> ``{arm: (index, fell_back)}``;
    RANDOM has no index (it is the fan's exact expectation, see ``arm_metrics``)."""
    out = {"BASE": select(rank, model_keep, None)}
    for arm, tr in tracks.items():
        out[arm] = select(rank, model_keep, gate_ok(states, tr, cfg))
    return out


def evaluate_window(states, rank, model_keep, human, route, tracks_rec: P.AgentTracks,
                    gate_tracks: dict, dac_cand=None, dac_human=None, cfg=P.PROXY) -> dict:
    """ONE window, every arm: gate with each arm's OWN tracks, then score every pick against
    the RECORDED future. ``gate_tracks`` = ``{GATE_ORACLE: tracks_rec, GATE_CONST: empty,
    GATE_PRED: pred, ORACLE_CV: cv}`` (any subset)."""
    true = score_true(states, human, tracks_rec, route, dac_cand, dac_human, cfg)
    sel = window_arms(states, rank, model_keep, gate_tracks, cfg)
    m = arm_metrics(sel, true)
    m["_fan"] = {"n": int(states.shape[0]),
                 "collision_free_share": float((true["nc"] == 1).double().mean()),
                 "any_free": bool((true["nc"] == 1).any()),
                 "human": true["human"]}
    return m


def arm_metrics(sel: dict, true: dict, keys=("nc", "dac", "ep", "ttc", "comfort", "pdms")) -> dict:
    """Per arm: the RECORDED-future sub-scores of its pick. RANDOM = the exact uniform
    expectation over ALL N candidates (ERRATUM-1: N from the fan, never a literal)."""
    out = {}
    for arm, (idx, fb) in sel.items():
        out[arm] = {k: float(true[k][idx]) for k in keys}
        out[arm]["collided"] = float(true["nc"][idx] != 1)
        out[arm]["idx"], out[arm]["fell_back"] = int(idx), bool(fb)
    out["RANDOM"] = {k: float(true[k].double().mean()) for k in keys}
    out["RANDOM"]["collided"] = float((true["nc"] != 1).double().mean())
    out["RANDOM"]["n_candidates"] = int(true["nc"].shape[0])
    return out
