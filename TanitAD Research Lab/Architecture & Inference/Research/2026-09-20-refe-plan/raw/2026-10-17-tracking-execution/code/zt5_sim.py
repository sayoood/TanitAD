"""zt5_sim.py -- NAVSIM v1.1's own pipeline for PREREG_ZT5 (py3.9 navsim venv; navsim @ 3e8291b on PYTHONPATH).

raw states = navsim.evaluate.pdm_score transform_trajectory + get_trajectory_as_array (exactly as the harness and
refe/navsim_pdm_targets.score_plans build them); sim = PDMSimulator.simulate_proposals (LQR + kinematic bicycle);
DAC = pdm_scorer.py:284-286's rule on the 4 footprint corners against ROADBLOCK/INTERSECTION/DRIVABLE_AREA/CARPARK of a
PDMDrivableMap.  Counterfactual hooks: initial ego override (CF-S, CF-V), actuator lags removed (CF-A), dense 20-pose
raw states (CF-D), and P4's compensation (F_a / CF-P).  Nothing here selects or scores a plan.
"""
from __future__ import annotations

import numpy as np

import zt5_lib as L

_S: dict = {}


def setup():
    if _S:
        return _S
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    samp = TrajectorySampling(num_poses=40, interval_length=0.1)
    simA = PDMSimulator(samp)
    simA._motion_model._accel_time_constant = 1e-6                       # CF-A: actuator lags removed
    simA._motion_model._steering_angle_time_constant = 1e-6
    _S.update(samp=samp, samp8=TrajectorySampling(num_poses=8, interval_length=0.5),
              samp20=TrajectorySampling(num_poses=20, interval_length=0.2), sim=PDMSimulator(samp), simA=simA)
    return _S


def raw_states(plans, ego, grid="8"):
    """[M, 8|20, 3] ego-frame plans -> [M, 41, StateIndex] interpolated states (the harness's own path)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    S = setup()
    samp_in = S["samp8"] if grid == "8" else S["samp20"]
    out = []
    for p in np.asarray(plans, dtype=np.float32):
        it = transform_trajectory(Trajectory(p, samp_in), ego)
        out.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    return np.stack(out, 0)


def simulate(raw, ego, actuator_lag=True):
    S = setup()
    return (S["sim"] if actuator_lag else S["simA"]).simulate_proposals(raw, ego)


def ego_override(ego, v=None, a=None, steer=None, yaw_rate=None):
    """a copy of the initial ego state with the named fields replaced (CF-S / CF-V)."""
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateVector2D
    dcs = ego.dynamic_car_state
    vel = dcs.rear_axle_velocity_2d if v is None else StateVector2D(float(v), 0.0)
    acc = dcs.rear_axle_acceleration_2d if a is None else StateVector2D(float(a), 0.0)
    return EgoState.build_from_rear_axle(
        rear_axle_pose=ego.rear_axle, rear_axle_velocity_2d=vel, rear_axle_acceleration_2d=acc,
        tire_steering_angle=float(ego.tire_steering_angle if steer is None else steer), time_point=ego.time_point,
        vehicle_parameters=ego.car_footprint.vehicle_parameters, is_in_auto_mode=True,
        angular_vel=float(dcs.angular_velocity if yaw_rate is None else yaw_rate),
        angular_accel=float(dcs.angular_acceleration))


def drivable_idx(dm):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    return dm.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                       SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])


def dac(states, ego, dm, idcs):
    """[M, 41, S] -> [M] 1.0 = compliant (NAVSIM DAC), pdm_scorer's per-step rule."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    coords = state_array_to_coords_array(states, ego.car_footprint.vehicle_parameters)         # [M, 41, 5, 2]
    inp = dm.points_in_polygons(coords).transpose(1, 2, 0, 3)                                  # [M, 41, n_poly, 5]
    if len(idcs) == 0:
        return np.zeros(states.shape[0])
    cin = inp[:, :, idcs, :4].any(axis=-2)                                                     # [M, 41, 4]
    nd = cin.sum(-1) < 4
    return (~nd.any(-1)).astype(np.float64)


def ego_frame(states, ego):
    """[..., 41, S] global rear-axle states -> (xy [..., 41, 2], psi [..., 41]) in the t0 ego frame."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    ra = ego.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    dx, dy = states[..., StateIndex.X] - ra.x, states[..., StateIndex.Y] - ra.y
    xy = np.stack([dx * c + dy * s, -dx * s + dy * c], -1)
    return xy, L.wrap(states[..., StateIndex.HEADING] - ra.heading)


def tracked8(sim, ego):
    xy, psi = ego_frame(sim, ego)
    return np.concatenate([xy[:, 5::5][:, :8], psi[:, 5::5][:, :8, None]], -1)


def compensate(p8, ego):
    """F_a = P4's eval/p4_tracker_comp.py `compensate` (3 iterations, best-of by max knot-xy deviation)."""
    p = np.asarray(p8, np.float64)
    r = p.copy()
    best_r, best_dev = p.copy(), None
    for k in range(L.COMP_ITERS + 1):
        s = tracked8(simulate(raw_states(r[None], ego), ego), ego)[0]
        dev = float(np.linalg.norm(s[:, :2] - p[:, :2], axis=1).max())
        if best_dev is None or dev < best_dev:
            best_r, best_dev = r.copy(), dev
        if k == L.COMP_ITERS:
            break
        r = L.comp_step(r, p, s)
    return best_r


def admissible_ego_ok(ego) -> bool:
    """AG1: NAVSIM's initial state carries nothing beyond the agent's ego status (steering 0, yaw rate 0)."""
    return float(ego.tire_steering_angle) == 0.0 and float(ego.dynamic_car_state.angular_velocity) == 0.0


def counterfactuals(p8, ego, dm, idcs, p20=None):
    """single-factor CFs of PREREG_ZT5 section 3 for ONE executed plan -> dict of DAC(S_CF) (1.0 = rescued/compliant)."""
    out = {}
    R = raw_states(np.asarray(p8)[None], ego)
    out["base"] = float(dac(simulate(R, ego), ego, dm, idcs)[0])
    out["raw"] = float(dac(R, ego, dm, idcs)[0])
    Rh = raw_states(L.fix_heading_tangent(p8)[None], ego)
    out["H"] = float(dac(simulate(Rh, ego), ego, dm, idcs)[0])
    k0 = L.kappa0(p8)
    v0 = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
    out["S"] = float(dac(simulate(R, ego_override(ego, steer=L.steer_from_kappa(k0), yaw_rate=v0 * k0)), ego, dm, idcs)[0])
    v_plan = float(np.hypot(p8[0][0], p8[0][1]) / 0.5)
    out["V"] = float(dac(simulate(R, ego_override(ego, v=v_plan, a=0.0)), ego, dm, idcs)[0])
    out["A"] = float(dac(simulate(R, ego, actuator_lag=False), ego, dm, idcs)[0])
    if p20 is not None:
        Rd = raw_states(np.asarray(p20)[None], ego, grid="20")
        out["D"] = float(dac(simulate(Rd, ego), ego, dm, idcs)[0])
    pc = compensate(p8, ego)
    out["P"] = float(dac(simulate(raw_states(pc[None], ego), ego), ego, dm, idcs)[0])
    out["P_drawn"] = float(dac(raw_states(pc[None], ego), ego, dm, idcs)[0])
    return out, pc


def descriptors(p8, ego, R, S):
    """PREREG section 3 descriptors for one plan given its raw/sim states ([41, S] or [1, 41, S])."""
    R, S = (R[0] if R.ndim == 3 else R), (S[0] if S.ndim == 3 else S)
    rxy, rpsi = ego_frame(R, ego)
    sxy, _ = ego_frame(S, ego)
    v0 = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
    d = L.decompose(sxy, rxy, rpsi)
    return {"hinc_deg": float(np.degrees(L.heading_inconsistency(p8))), "dv0": float(np.hypot(p8[0][0], p8[0][1]) / 0.5 - v0),
            "kappa0": L.kappa0(p8), "v0": v0, "delta": L.self_deviation(sxy, rxy), "dev_t": d["t"], "along": d["along"],
            "cross": d["cross"], "cross_inside": d["cross_inside"], "turn_deg": d["turn_deg"], "straight": d["straight"],
            "final_deg": float(np.degrees(L.wrap(p8[-1][2]))), "path_len": float(np.linalg.norm(np.diff(
                np.concatenate([np.zeros((1, 2)), np.asarray(p8)[:, :2]], 0), axis=0), axis=-1).sum())}
