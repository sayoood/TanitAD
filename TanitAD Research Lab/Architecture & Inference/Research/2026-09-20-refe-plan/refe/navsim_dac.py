"""NAVSIM-faithful drivable-area compliance for a set of proposals (PI decision 2026-09-26).

WHY. The teacher's drivable-area label (`dac.violation`) is DriveRL's CURB-CROSSING detector
(`OffRoad.detect_cross_line_knn`): MEASURED in the fixed bank at 0-1 % on paths shifted 2-4 m, 12 %
on a path aimed through a curb, and 0 of 1,024 on the model's own proposals -- while NAVSIM fails
33 % of those proposals, and drivable area is the largest single loss of the planner's pick
(SPEC_NAVTEST E-6, 37 % of picks). The scorer cannot learn a violation its labels never contain.

WHAT. NAVSIM v1's rule, from its source (navsim @ 3e8291b):
  * polygons -- `PDMDrivableMap.from_simulation(map_api, ego_state, map_radius)` with the metric
    cache's `map_radius = 100` (metric_cache_processor.py); the NON-DRIVABLE test uses the layers
    ROADBLOCK, INTERSECTION, (DRIVABLE_AREA, never queried), CARPARK_AREA (pdm_scorer.py
    `_calculate_ego_area`), queried around the ego CENTRE
  * footprint -- `state_array_to_coords_array`: the rear-axle pose moved `rear_axle_to_center` along
    the heading, corners at +-half_length / +-half_width (nuPlan Pacifica)
  * a step is off the drivable area when FEWER THAN 4 corners lie in at least one such polygon;
    the proposal fails if ANY step does. The steps are the 0.1 s grid over 4 s including t = 0
    (`get_trajectory_as_array`, TrajectorySampling(40, 0.1)).
  * the proposal is judged in NAVSIM's own representation: REFe's 20 poses at 5 Hz -> NAVSIM's 8 at
    2 Hz by the seam's `to_navsim` (copied below, asserted equal in the self-test) -> 0.1 s by linear
    interpolation of x, y and the WRAPPED heading, from the ego origin at t = 0.
⚠ NOT REPLICATED: NAVSIM tracks the proposal with its LQR controller + kinematic bicycle model before
the check; this checks the proposal's own interpolated poses. The agreement with NAVSIM's verdict is
MEASURED by eval/validate_navsim_dac.py on the E-6 tables before any label is used.
"""
from __future__ import annotations

import math

import numpy as np

MAP_RADIUS = 100.0
NAVSIM_DT, NAVSIM_N, SRC_DT = 0.5, 8, 0.2
# nuPlan's get_pacifica_parameters(): front 4.049, rear 1.127, width 2.297
HALF_LENGTH = (4.049 + 1.127) / 2.0
HALF_WIDTH = 2.297 / 2.0
REAR_AXLE_TO_CENTER = (4.049 - 1.127) / 2.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_navsim(poses20: np.ndarray, dt: float = SRC_DT) -> np.ndarray:
    """The seam's conversion (eval/refe_navtest_seam.py `to_navsim`), verbatim."""
    p = np.asarray(poses20, dtype=np.float64)
    t_src = dt * np.arange(1, p.shape[0] + 1)
    out = np.zeros((NAVSIM_N, 3), dtype=np.float64)
    for k in range(NAVSIM_N):
        t = NAVSIM_DT * (k + 1)
        j = int(np.searchsorted(t_src, t - 1e-9))
        if j < p.shape[0] and abs(t_src[j] - t) < 1e-9:
            out[k] = p[j]
            continue
        w = (t - t_src[j - 1]) / (t_src[j] - t_src[j - 1])
        out[k, :2] = (1 - w) * p[j - 1, :2] + w * p[j, :2]
        out[k, 2] = wrap(p[j - 1, 2] + w * wrap(p[j, 2] - p[j - 1, 2]))
    return out


def at_10hz(poses8: np.ndarray) -> np.ndarray:
    """[..., 8, 3] at 0.5..4.0 s -> [..., 41, 3] at 0.0..4.0 s, from the ego origin at t = 0."""
    p = np.asarray(poses8, dtype=np.float64)
    lead = p.shape[:-2]
    full = np.concatenate([np.zeros(lead + (1, 3)), p], axis=-2)                  # t = 0, 0.5, ..., 4
    hd = np.unwrap(full[..., 2], axis=-1)
    t_k = np.arange(NAVSIM_N + 1) * NAVSIM_DT
    t_q = np.arange(41) * 0.1
    out = np.empty(lead + (41, 3))
    flat_in, flat_out = full.reshape(-1, NAVSIM_N + 1, 3), out.reshape(-1, 41, 3)
    hdf = hd.reshape(-1, NAVSIM_N + 1)
    for i in range(flat_in.shape[0]):
        flat_out[i, :, 0] = np.interp(t_q, t_k, flat_in[i, :, 0])
        flat_out[i, :, 1] = np.interp(t_q, t_k, flat_in[i, :, 1])
        flat_out[i, :, 2] = np.interp(t_q, t_k, hdf[i])
    return out


def corners_global(rel: np.ndarray, x0: float, y0: float, h0: float) -> np.ndarray:
    """Rear-axle poses in the ego frame -> the footprint's 4 corners in the map frame: [..., 4, 2]."""
    c, s = math.cos(h0), math.sin(h0)
    gx = x0 + rel[..., 0] * c - rel[..., 1] * s
    gy = y0 + rel[..., 0] * s + rel[..., 1] * c
    gh = h0 + rel[..., 2]
    ch, sh = np.cos(gh), np.sin(gh)
    cx, cy = gx + REAR_AXLE_TO_CENTER * ch, gy + REAR_AXLE_TO_CENTER * sh
    out = np.empty(rel.shape[:-1] + (4, 2))
    for i, (lo, la) in enumerate(((HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH),
                                  (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH))):
        out[..., i, 0] = cx + lo * ch - la * sh
        out[..., i, 1] = cy + lo * sh + la * ch
    return out


def drivable_polygons(map_api, ego_state, radius: float = MAP_RADIUS) -> list:
    """ROADBLOCK + INTERSECTION + CARPARK_AREA polygons within `radius` of the ego CENTRE."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    layers = [L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION, L.CARPARK_AREA]
    objs = map_api.get_proximal_map_objects(ego_state.center.point, radius, layers)
    return [o.polygon for layer in (L.ROADBLOCK, L.INTERSECTION, L.CARPARK_AREA) for o in objs[layer]]


def violations(polys: list, corners: np.ndarray) -> np.ndarray:
    """[P, T, 4, 2] corners -> [P] bool: some step has fewer than 4 corners inside any polygon."""
    import shapely
    flat = corners.reshape(-1, 2)
    inside = np.zeros(len(flat), dtype=bool)
    for poly in polys:
        inside |= shapely.contains_xy(poly, flat[:, 0], flat[:, 1])
    ok = inside.reshape(corners.shape[:-1]).sum(-1) == 4                         # [P, T]
    return ~ok.all(-1)


def simulated_corners(props8: np.ndarray, ego_state) -> np.ndarray:
    """[P, 41, 4, 2] footprint corners of the SIMULATED proposals (see `simulated_states`), with the scorer's
    Pacifica parameters -- what NAVSIM's drivable-area check reads."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    sim = simulated_states(props8, ego_state)
    coords = state_array_to_coords_array(sim, get_pacifica_parameters())            # [P, 41, 5, 2]
    corner_idx = [i for i in range(coords.shape[2]) if i != int(BBCoordsIndex.CENTER)]
    return coords[:, :, corner_idx]


def simulated_states(props8: np.ndarray, ego_state) -> np.ndarray:
    """`navsim/evaluate/pdm_score.py` for a SET of proposals: `transform_trajectory` (8 poses in the ego
    frame -> an InterpolatedTrajectory from the initial ego state, velocities/accelerations zero -- "ignored
    by LQR + bicycle model") -> `get_trajectory_as_array` (41 states at 0.1 s) -> `PDMSimulator.
    simulate_proposals` (LQR tracker + kinematic bicycle, VENDORED VERBATIM in navsim_pdm/). Returns the
    simulated state array [P, 41, StateIndex.size()] that NAVSIM's scorer reads (`PDMScorer._states`)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    # nuplan's ml_planner helpers, vendored (DriveRL's nuplan fork lacks the module)
    from navsim_pdm.nuplan_transform_utils import _get_fixed_timesteps, _se2_vel_acc_to_ego_state
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim_pdm.pdm_array_representation import ego_states_to_state_array, state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    from navsim_pdm.pdm_simulator import PDMSimulator
    future = TrajectorySampling(num_poses=40, interval_length=0.1)                 # proposal_sampling
    model_sampling = TrajectorySampling(time_horizon=4.0, interval_length=NAVSIM_DT)
    timesteps = _get_fixed_timesteps(ego_state, model_sampling.time_horizon, model_sampling.interval_length)
    vp = ego_state.car_footprint.vehicle_parameters
    times_s = np.arange(0.0, future.time_horizon + future.interval_length, future.interval_length)
    times_s = times_s + ego_state.time_point.time_s
    arrs = []
    for p in np.asarray(props8, dtype=np.float64):
        absolute = relative_to_absolute_poses(ego_state.rear_axle, [StateSE2.deserialize(q) for q in p])
        agents = [_se2_vel_acc_to_ego_state(s, [0.0, 0.0], [0.0, 0.0], t, vp) for s, t in zip(absolute, timesteps)]
        tr = InterpolatedTrajectory([ego_state] + agents)
        t_us = np.clip([int(t * 1e6) for t in times_s], tr.start_time.time_us, tr.end_time.time_us)
        arrs.append(ego_states_to_state_array(tr.get_state_at_times([TimePoint(int(t)) for t in t_us])))
    return PDMSimulator(future).simulate_proposals(np.stack(arrs), ego_state)


def comfortable_from_states(sim: np.ndarray) -> np.ndarray:
    """NAVSIM v1's comfort verdict on simulated states, EXACTLY `PDMScorer._calculate_is_comfortable`:
    `ego_is_comfortable` (nuPlan's six comfort metrics, VENDORED VERBATIM in navsim_pdm/) over the 41
    time points `arange(num_poses + 1) * interval_length` of the proposal sampling (40 x 0.1 s), then ALL
    six must hold. Returns [P] float 1.0 = comfortable, 0.0 = not."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from navsim_pdm.pdm_comfort_metrics import ego_is_comfortable
    time_point_s = np.arange(0, 40 + 1).astype(np.float64) * 0.1
    return np.all(ego_is_comfortable(sim, time_point_s), axis=-1).astype(np.float64)


def navsim_comfortable(props: np.ndarray, ego_state, grid: str = "refe20") -> np.ndarray:
    """props: [P, 20, 3] REFe poses ('refe20') or [P, 8, 3] NAVSIM poses ('navsim8') -> [P] float 1.0 =
    comfortable by NAVSIM's own rule on its own simulation, 0.0 = not (PI decision 2026-09-26, option 2)."""
    p8 = np.stack([to_navsim(p) for p in props]) if grid == "refe20" else np.asarray(props, np.float64)
    return comfortable_from_states(simulated_states(p8, ego_state))


def navsim_dac_and_comfort(props: np.ndarray, ego_state, map_api, grid: str = "refe20") -> tuple:
    """ONE simulation, both NAVSIM labels: ([P] drivable-area violation 1/0, [P] comfortable 1/0) -- the
    labeller's call, so the proposals are simulated once, exactly as NAVSIM's scorer does."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    p8 = np.stack([to_navsim(p) for p in props]) if grid == "refe20" else np.asarray(props, np.float64)
    sim = simulated_states(p8, ego_state)
    coords = state_array_to_coords_array(sim, get_pacifica_parameters())
    corner_idx = [i for i in range(coords.shape[2]) if i != int(BBCoordsIndex.CENTER)]
    viol = violations(drivable_polygons(map_api, ego_state), coords[:, :, corner_idx]).astype(np.float64)
    return viol, comfortable_from_states(sim)


def navsim_dac_violation(props: np.ndarray, ego_state, map_api, grid: str = "refe20",
                         mode: str = "simulated") -> np.ndarray:
    """props: [P, 20, 3] REFe poses (grid 'refe20') or [P, 8, 3] NAVSIM poses ('navsim8') ->
    [P] float 1.0 = off the drivable area at some step (NAVSIM DAC 0), 0.0 = compliant.
    mode 'simulated' (NAVSIM's pipeline, the default) or 'raw' (the plan's own interpolated poses,
    MEASURED 80-84 % agreement with NAVSIM -- kept only as the comparison arm)."""
    p8 = np.stack([to_navsim(p) for p in props]) if grid == "refe20" else np.asarray(props, np.float64)
    if mode == "simulated":
        cr = simulated_corners(p8, ego_state)
    else:
        ra = ego_state.rear_axle
        cr = corners_global(at_10hz(p8), ra.x, ra.y, ra.heading)
    return violations(drivable_polygons(map_api, ego_state), cr).astype(np.float64)
