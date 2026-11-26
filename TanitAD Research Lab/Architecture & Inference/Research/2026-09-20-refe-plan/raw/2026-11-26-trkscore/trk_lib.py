"""TRK-SCORE (PREREG_TRKSCORE.md section 2.1): tracker-consistent selection for a REFe planner. Parameter-free.

Per token: the planner's 64 planned proposals -> NAVSIM's LQR tracker (the harness route, from the planner's own EgoState) ->
the tracked set in REFe's frame and grid -> the SAME scoring heads (`REFe.score_trajectories`, the complete set) -> the planner's
own navsim_v1 aggregate -> argmax k_trk. The SUBMITTED trajectory stays the PLANNED proposal k_trk (the caller executes it).

Frames and grids (checked by test_trk_score.py):
  * REFe traj [M, 20, 3]: rear axle at t0, x forward, y left, heading (rad), t = 0.2 .. 4.0 s (planner.TRAJ_DT_S 0.2).
  * NAVSIM tracked states [M, 41, StateIndex.size()]: WORLD frame, t = 0.0 .. 4.0 s at 0.1 s; row 0 is the start state.
  * REFe step i (t = 0.2 (i+1)) <-> simulated row 2 (i+1).
Pure numpy below the imports of `tracker_setup`; the tracker itself is navsim @ 3e8291b, imported exactly as
`refe/navsim_pdm_targets.py` / `eval/instruments/transp_score.py` import it (PDM-Closed's row is NOT simulated: it is privileged).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

N_PROP, N_STEP, TRAJ_DIM = 64, 20, 3
SRC_DT, SIM_DT, SIM_N = 0.2, 0.1, 40
SIM_IDX = np.arange(2, SIM_N + 1, 2)          # rows 2, 4, ..., 40  <->  t = 0.2, ..., 4.0
assert len(SIM_IDX) == N_STEP
RED_DY_M = 1.0                                 # CTRL-RED: the start state's rear axle 1.0 m to the LEFT (+y, ego frame)
RED_MOD = 10                                   # CTRL-RED subset of a full run: sha1(token) mod 10 == 0


def wrap(a):
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def is_red_token(tok: str, mod: int = RED_MOD) -> bool:
    return int(hashlib.sha1(str(tok).encode()).hexdigest(), 16) % mod == 0


def world_to_refe20(sim: np.ndarray, x0: float, y0: float, h0: float, ref20: np.ndarray | None = None) -> np.ndarray:
    """[M, 41, >=3] world-frame states at 0.1 s (row 0 = t0) -> [M, 20, 3] REFe frame at t = 0.2 .. 4.0 s.
    Heading: relative to h0, then put on the branch of `ref20`'s heading (the planned proposal) when given:
    h = h_ref + wrap(h - h_ref). Parameter-free: it only picks the 2*pi branch the planned set already uses."""
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 3 or sim.shape[1] != SIM_N + 1 or sim.shape[2] < 3:
        raise ValueError(f"tracked states must be [M, {SIM_N + 1}, >=3], got {sim.shape}")
    s = sim[:, SIM_IDX, :3]
    c, sn = math.cos(h0), math.sin(h0)
    dx, dy = s[..., 0] - x0, s[..., 1] - y0
    out = np.empty(s.shape, dtype=np.float64)
    out[..., 0] = c * dx + sn * dy
    out[..., 1] = -sn * dx + c * dy
    h = wrap(s[..., 2] - h0)
    if ref20 is not None:
        ref = np.asarray(ref20, dtype=np.float64)
        if ref.shape != out.shape:
            raise ValueError(f"ref20 {ref.shape} != tracked {out.shape}")
        h = ref[..., 2] + wrap(h - ref[..., 2])
    out[..., 2] = h
    return out


def refe20_to_world(p20: np.ndarray, x0: float, y0: float, h0: float) -> np.ndarray:
    """The inverse map (tests only): [M, 20, 3] REFe frame -> [M, 41, 3] world states, row 0 = the start pose, the REFe poses at rows 2(i+1),
    odd rows linearly interpolated (never read by world_to_refe20)."""
    p = np.asarray(p20, dtype=np.float64)
    M = p.shape[0]
    c, sn = math.cos(h0), math.sin(h0)
    w = np.zeros((M, SIM_N + 1, 3), dtype=np.float64)
    w[:, 0] = (x0, y0, h0)
    w[:, SIM_IDX, 0] = x0 + c * p[..., 0] - sn * p[..., 1]
    w[:, SIM_IDX, 1] = y0 + sn * p[..., 0] + c * p[..., 1]
    w[:, SIM_IDX, 2] = h0 + p[..., 2]
    for r in range(1, SIM_N + 1, 2):
        w[:, r] = 0.5 * (w[:, r - 1] + w[:, r + 1])
    return w


# ------------------------------------------------------------------ the tracker (navsim @ 3e8291b, the harness's route)
_T: dict = {}


def tracker_setup(nav_root: str | None = None) -> dict:
    """navsim's PDMSimulator / transform_trajectory / get_trajectory_as_array through refe/navsim_pdm_targets._setup() (its two stubs make the
    DriveRL nuplan fork import navsim). `nav_root` is APPENDED to sys.path (lowest priority: it can shadow nothing)."""
    if _T:
        return _T
    import sys
    if nav_root and nav_root not in sys.path:
        sys.path.append(nav_root)
    import navsim_pdm_targets as PT
    S = PT._setup()
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    import navsim
    _T.update(S=S, Trajectory=Trajectory, transform=transform_trajectory, as_array=get_trajectory_as_array,
              state_array=ego_state_to_state_array, navsim_file=navsim.__file__)
    return _T


def proposal_states(props8: np.ndarray, ego) -> np.ndarray:
    """[M, 8, 3] float32 NAVSIM-frame plans -> [M, 41, 11] proposal states, exactly as score_arms builds the tracked arms' input."""
    T = tracker_setup()
    S = T["S"]
    out = []
    for p in np.asarray(props8, dtype=np.float32):
        it = T["transform"](T["Trajectory"](np.asarray(p, np.float32), S["samp8"]), ego)
        out.append(T["as_array"](it, S["samp"], ego.time_point))
    return np.stack(out, 0)


def simulate(states: np.ndarray, start_ego) -> np.ndarray:
    """PDMSimulator.simulate_proposals (LQR + kinematic bicycle) from `start_ego`: [M, 41, 11] world frame."""
    return tracker_setup()["S"]["sim"].simulate_proposals(np.asarray(states), start_ego)


def red_ego(ego, dy: float = RED_DY_M):
    """CTRL-RED's deliberately wrong start state: the rear axle moved `dy` m along the ego's +y (left); all else equal."""
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2
    ra = ego.rear_axle
    d = ego.dynamic_car_state
    pose = StateSE2(ra.x - math.sin(ra.heading) * dy, ra.y + math.cos(ra.heading) * dy, ra.heading)
    return EgoState.build_from_rear_axle(pose, d.rear_axle_velocity_2d, d.rear_axle_acceleration_2d, ego.tire_steering_angle,
                                         ego.time_point, ego.car_footprint.vehicle_parameters, is_in_auto_mode=True,
                                         angular_vel=d.angular_velocity, angular_accel=d.angular_acceleration,
                                         tire_steering_rate=d.tire_steering_rate)


# ------------------------------------------------------------------ the selection
def pick(planner, score) -> int:
    """The planner's own rule on a [M, 6] head output: argmax planner.aggregate in the tensor's dtype and device (float32 on the GPU in a
    seam). Same shape path as REFePlanner._pick: aggregate([1, M, 6])[0].argmax()."""
    if getattr(planner, "select", "best") != "best":
        raise ValueError(f"TRK-SCORE is defined for select='best' only, got {planner.select!r}")
    return int(planner.aggregate(score.unsqueeze(0))[0].argmax())


def check_set(traj, model_cfg=None):
    if traj.dim() != 3 or tuple(traj.shape[1:]) != (N_STEP, TRAJ_DIM) or traj.shape[0] != N_PROP:
        raise ValueError(f"TRK-SCORE needs the COMPLETE set [{N_PROP}, {N_STEP}, {TRAJ_DIM}], got {tuple(traj.shape)}")
    if model_cfg is not None:
        got = (model_cfg.n_proposals, model_cfg.horizon_steps, model_cfg.traj_dim)
        if got != (N_PROP, N_STEP, TRAJ_DIM):
            raise ValueError(f"model config (n_proposals, horizon_steps, traj_dim) = {got}, TRK-SCORE is registered for {(N_PROP, N_STEP, TRAJ_DIM)}")


def trk_step(planner, ego, traj, score, k_ship: int, sctx, to_navsim, red: bool = False, keep_sim: bool = False,
             n_prop: int = N_PROP) -> dict:
    """ONE token. `traj` [M, 20, 3] and `score` [M, 6] are the planner's forward outputs (device tensors), `sctx` the scoring context that
    forward used, `to_navsim` the seam's own converter. Returns the record; rec['k_trk'] is the index to SUBMIT (as a PLANNED proposal)."""
    import time
    import torch
    t0 = time.time()
    model = planner.model
    if n_prop == N_PROP:
        check_set(traj, getattr(model, "cfg", None))
    elif traj.shape[0] != n_prop:
        raise ValueError(f"incomplete set: {traj.shape[0]} != {n_prop}")
    with torch.no_grad():
        # CTRL-ID: the new path on the PLANNED set (the forward's own tensor), same dtype and device
        s_plan = model.score_trajectories(traj.unsqueeze(0), sctx)[0]
        k_id = pick(planner, s_plan)
        id_maxdiff = float((s_plan.float() - score.float()).abs().max())
        # the harness's tracker input: the executed (A7-repaired) plan of every proposal, on NAVSIM's grid, float32 (the seam-row form)
        props8 = np.stack([to_navsim(planner.executed(traj, j).float().cpu().numpy()) for j in range(traj.shape[0])]).astype(np.float32)
        states = proposal_states(props8, ego)
        sim = simulate(states, ego)
        ra = ego.rear_axle
        plan20 = traj.detach().float().cpu().numpy().astype(np.float64)
        trk20 = world_to_refe20(sim, ra.x, ra.y, ra.heading, ref20=plan20)
        tt = torch.from_numpy(trk20.astype(np.float32)).to(device=traj.device, dtype=traj.dtype).unsqueeze(0)
        s_trk = model.score_trajectories(tt, sctx)[0]
        k_trk = pick(planner, s_trk)
        rec = {"k_ship": int(k_ship), "k_id": k_id, "k_trk": k_trk, "id_maxdiff": id_maxdiff,
               "s_plan": s_plan.float().cpu().numpy().astype(np.float32), "s_trk": s_trk.float().cpu().numpy().astype(np.float32),
               "trk20": trk20.astype(np.float32), "ship8": props8[int(k_ship)].copy(),
               "s0": np.asarray(tracker_setup()["state_array"](ego), dtype=np.float64), "k_red": -1, "red_d0": float("nan")}
        if red:
            sim_r = simulate(states, red_ego(ego))
            trk20_r = world_to_refe20(sim_r, ra.x, ra.y, ra.heading, ref20=plan20)
            tr = torch.from_numpy(trk20_r.astype(np.float32)).to(device=traj.device, dtype=traj.dtype).unsqueeze(0)
            rec["k_red"] = pick(planner, model.score_trajectories(tr, sctx)[0])
            rec["red_d0"] = float(np.linalg.norm(trk20_r[:, 0, :2] - trk20[:, 0, :2], axis=-1).min())
        if keep_sim:
            rec["sim"] = sim
            rec["props8"] = props8
    rec["t_trk"] = time.time() - t0
    return rec
