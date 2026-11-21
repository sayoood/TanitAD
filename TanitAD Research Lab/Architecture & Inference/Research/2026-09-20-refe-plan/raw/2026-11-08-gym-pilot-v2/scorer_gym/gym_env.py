"""M-GYM-1: Stage-B scorer-gym environment at pilot scale (CPU only).

state  = ego + agents + map at a nuPlan log state
action = one of K=24 proposals per 0.5 s cycle (15 PDM-Closed + 9 centreline speed-sweep)
step   = nuPlan LQR tracker + kinematic bicycle at 10 Hz; agents in BOTH modes
         ("log" = log replay, "idm" = nuPlan IDMAgents, reactive)
reward = SCORER_GYM_DESIGN 2.5: multiplicative, hard terminations, every term logged per cycle.
Round-0 continuation policy = PDM-Closed (argmax of its own score), re-planned every 0.5 s.
Pilot definitions of the soft terms are operationalisations (see M_GYM_1_RESULT.md).
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import sqlite3
import sys
import time
from collections import deque
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
MAPS_ROOT = "F:/Archive/devbox-C/navsim/data/maps"
DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
NAVTEST_LOG_DIR = "F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
if NAVSIM_ROOT not in sys.path:
    sys.path.insert(0, NAVSIM_ROOT)

import numpy as np
from scipy.signal import savgol_filter
from shapely import affinity
from shapely.geometry import Point

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from shapely.ops import unary_union
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
    state_array_to_ego_states,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, StateIndex

VEH = get_pacifica_parameters()
DT = 0.1
CYCLE_STEPS = 5          # 0.5 s replanning cycle
SPEED_MULTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
CONSISTENCY_M = 1.0
# nuPlan ego_is_comfortable bounds
COMF = dict(lon_acc=(-4.05, 2.40), lat_acc=4.89, yaw_rate=0.95, yaw_acc=1.93, lon_jerk=4.13, mag_jerk=8.37)
# v2 (M-GYM-2): progress is paid only above a -2 m/s^2 comfortable-braking reference started at the episode's initial speed
BRAKE_BASELINE_DECEL = 2.0
JERK_HIST = 30           # samples (3 s) kept for the nuPlan smoothed-jerk estimate
TERMS = ["q_free", "q_gap", "q_stall", "q_yield", "q_comf", "q_osc", "q_lane", "q_cons"]
HARD = ["collision", "off_drivable", "wrong_way", "ttc"]
MAX_MAP_SLACK_M = 1.0    # POST_HOC (state 8: human log overshoots up to 0.55 m, measured diag2.out); above this NOT grandfathered, flagged map_unreliable
MAP_NAMES = {"las_vegas": "us-nv-las-vegas-strip", "boston": "us-ma-boston",
             "pittsburgh": "us-pa-pittsburgh-hazelwood", "singapore": "sg-one-north"}


def nuplan_smoothed(y, t_us):
    """nuPlan comfort-metric estimators, verbatim recipe. Acceleration: Savitzky-Golay, window 8, poly 2
    (nuplan/planning/metrics/utils/state_extractors.py:187-218, extract_ego_acceleration).
    Jerk: approximate_derivatives(acc, t, deriv_order=1, poly_order=2, window_length=15)
    (same file :221-248 extract_ego_jerk; the derivative is :78-114, itself savgol_filter with delta=mean dt)."""
    y = np.asarray(y, dtype=float)
    acc = np.round(savgol_filter(y, polyorder=2, window_length=min(8, len(y))), 8)
    jerk = approximate_derivatives(acc, np.asarray(t_us, dtype=float) / 1e6, deriv_order=1, poly_order=2,
                                   window_length=min(15, len(acc)))
    return acc, np.round(jerk, 8)


def jerk_channels_nuplan(t_us, ax, ay, n_new):
    """Comfort-limit violations of the last n_new samples under nuPlan's smoothed estimates.
    Returns (set of violated channel names, max |lon_jerk|, max mag_jerk) over those samples."""
    amag = np.hypot(np.asarray(ax), np.asarray(ay))
    _, jx = nuplan_smoothed(ax, t_us)
    _, jm = nuplan_smoothed(amag, t_us)
    v = set()
    mx, mm = float(np.max(np.abs(jx[-n_new:]))), float(np.max(np.abs(jm[-n_new:])))
    if mx > COMF["lon_jerk"]:
        v.add("lon_jerk")
    if mm > COMF["mag_jerk"]:
        v.add("mag_jerk")
    return v, mx, mm


def jerk_channels_raw(ax, ay, dt=DT):
    """The PILOT (M-GYM-1) estimator: single-step finite difference. Kept only so tests can show the difference."""
    v = set()
    jx = np.diff(ax) / dt
    jy = np.diff(ay) / dt
    if np.any(np.abs(jx) > COMF["lon_jerk"]):
        v.add("lon_jerk")
    if np.any(np.hypot(jx, jy) > COMF["mag_jerk"]):
        v.add("mag_jerk")
    return v


class Timer:
    def __init__(self):
        self.t: Dict[str, float] = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt

    def __call__(self, k):
        outer = self

        class _C:
            def __enter__(s):
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                outer.add(k, time.perf_counter() - s.t0)
        return _C()


# ------------------------------------------------------------------ data selection
def navtest_log_names() -> set:
    return {f[:-4] for f in os.listdir(NAVTEST_LOG_DIR) if f.endswith(".pkl")}


def select_states(n_states=20, n_logs=10, seed=20261108, min_lidar=400, exclude_navtest=True):
    """Seeded-hash selection (no hand picking). Returns list of dicts (db, token_hex, ts, row_idx)."""
    nt = navtest_log_names() if exclude_navtest else set()
    names = sorted(f for f in os.listdir(DB_DIR) if f.endswith(".db") and f[:-3] not in nt)
    h = lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
    out, used_logs = [], 0
    per_log = n_states // n_logs
    for name in sorted(names, key=h):
        if used_logs >= n_logs:
            break
        con = sqlite3.connect(f"file:{DB_DIR}/{name}?mode=ro", uri=True)
        try:
            n = con.execute("select count(*) from lidar_pc").fetchone()[0]
            if n < min_lidar:
                continue
            lo, hi = 40, n - 260            # >=2 s history margin, >=13 s of future
            if hi <= lo:
                continue
            picks = []
            for k in range(per_log):
                picks.append(lo + int(h(f"{name}:{k}")[:8], 16) % (hi - lo))
            for r in picks:
                tok, ts = con.execute("select token, timestamp from lidar_pc order by timestamp limit 1 offset ?", (r,)).fetchone()
                out.append(dict(db=f"{DB_DIR}/{name}", log=name[:-3], row=r, token=tok.hex(), ts=int(ts)))
            used_logs += 1
        finally:
            con.close()
    return out


# nuPlan scenario_tag types. Rule-based oversampling of the situations in which a stall is possible.
STOP_AND_GO_TAGS = (
    "stationary_in_traffic", "stationary_at_traffic_light_with_lead", "stationary_at_traffic_light_without_lead",
    "stopping_with_lead", "stopping_at_crosswalk", "stopping_at_stop_sign_with_lead", "stopping_at_stop_sign_without_lead",
    "stopping_at_traffic_light_with_lead", "stopping_at_traffic_light_without_lead",
    "accelerating_at_traffic_light", "accelerating_at_traffic_light_with_lead", "accelerating_at_traffic_light_without_lead",
    "accelerating_at_stop_sign", "accelerating_at_crosswalk", "waiting_for_pedestrian_to_cross",
    "starting_straight_traffic_light_intersection_traversal", "starting_straight_stop_sign_intersection_traversal",
    "starting_left_turn", "starting_right_turn", "starting_unprotected_cross_turn", "starting_protected_cross_turn",
    "starting_protected_noncross_turn", "starting_unprotected_noncross_turn", "starting_low_speed_turn")
JUNCTION_TAGS = ("on_all_way_stop_intersection", "on_stopline_traffic_light", "on_stopline_stop_sign",
                 "on_traffic_light_intersection", "on_stopline_crosswalk")


def select_states_stall(n_states=20, n_logs=10, seed=20261108, min_lidar=400, exclude_navtest=True):
    """Seeded, rule-based sampler that OVERSAMPLES stop-and-go and junction states (M-GYM-2 item 4). No hand picking.
    Per log (logs visited in seeded-hash order, navtest-excluded, same margins as select_states) it takes the hash-first
    lidar_pc row carrying a STOP_AND_GO tag and the hash-first row carrying a JUNCTION tag; logs missing either are skipped."""
    nt = navtest_log_names() if exclude_navtest else set()
    names = sorted(f for f in os.listdir(DB_DIR) if f.endswith(".db") and f[:-3] not in nt)
    h = lambda s: hashlib.sha256(f"{seed}:stall:{s}".encode()).hexdigest()
    out, used = [], 0
    per_log = n_states // n_logs
    assert per_log == 2
    for name in sorted(names, key=h):
        if used >= n_logs:
            break
        con = sqlite3.connect(f"file:{DB_DIR}/{name}?mode=ro", uri=True)
        try:
            n = con.execute("select count(*) from lidar_pc").fetchone()[0]
            if n < min_lidar:
                continue
            lo, hi = 40, n - 260
            picks = []
            for kind, tags in (("stop_and_go", STOP_AND_GO_TAGS), ("junction", JUNCTION_TAGS)):
                q = ",".join("?" * len(tags))
                rows = con.execute(
                    f"select distinct lp.token, lp.timestamp from lidar_pc lp join scenario_tag t on t.lidar_pc_token = lp.token "
                    f"where t.type in ({q})", tags).fetchall()
                cand = []
                for tok, ts in rows:
                    r = con.execute("select count(*) from lidar_pc where timestamp < ?", (ts,)).fetchone()[0]
                    if lo <= r < hi:
                        cand.append((h(f"{name}:{kind}:{tok.hex()}"), tok.hex(), int(ts), r))
                if not cand:
                    break
                _, tok, ts, r = min(cand)
                picks.append(dict(db=f"{DB_DIR}/{name}", log=name[:-3], row=r, token=tok, ts=ts, sampler=kind))
            if len(picks) == per_log:
                out += picks
                used += 1
        finally:
            con.close()
    return out


# ------------------------------------------------------------------ scenario / state
def build_scenario(st: dict, duration_s: float = 10.5) -> NuPlanScenario:
    con = sqlite3.connect(f"file:{st['db']}?mode=ro", uri=True)
    loc = con.execute("select location from log").fetchone()[0]
    mv = con.execute("select map_version from log").fetchone()[0]
    con.close()
    map_name = MAP_NAMES.get(loc, mv)
    info = ScenarioExtractionInfo("gym", duration_s, 0.0, 0.5)
    return NuPlanScenario(
        data_root="E:/Projects/TanitAD/data/nuplan", log_file_load_path=st["db"],
        initial_lidar_token=st["token"], initial_lidar_timestamp=st["ts"], scenario_type="unknown",
        map_root=MAPS_ROOT, map_version="nuplan-maps-v1.0", map_name=map_name,
        scenario_extraction_info=info, ego_vehicle_parameters=VEH)


class GymPDM(PDMClosedPlanner):
    """PDM-Closed that can also return all 15 proposals (the action fan)."""

    def plan_all(self, current_input):
        ego_state, observation = current_input.history.current_state
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)
        self._drivable_area_map = PDMDrivableMap.from_simulation(self._map_api, ego_state, self._map_radius)
        self._observation.update(ego_state, observation, current_input.traffic_light_data, self._route_lane_dict)
        self._update_proposal_manager(ego_state)
        props = self._generator.generate_proposals(ego_state, self._observation, self._proposal_manager)
        sim = self._simulator.simulate_proposals(props, ego_state)
        scores = self._scorer.score_proposals(sim, self._observation, self._centerline,
                                              list(self._route_lane_dict.keys()), self._drivable_area_map)
        trajs = [self._generator.generate_trajectory(i) for i in range(len(scores))]
        self._iteration += 1
        return trajs, np.asarray(scores)


def make_planner(scenario) -> GymPDM:
    ts = TrajectorySampling(num_poses=40, interval_length=DT)
    p = GymPDM(ts, ts, BatchIDMPolicy(speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0], fallback_target_velocity=15.0,
                                      min_gap_to_lead_agent=1.0, headway_time=1.5, accel_max=1.5, decel_max=3.0),
               [-1.0, 1.0], 100)

    class _Init:
        map_api = scenario.map_api
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        mission_goal = scenario.get_mission_goal()
    p.initialize(_Init())
    return p


def make_controller(scenario, ego0):
    c = TwoStageController(scenario,
                           LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0],
                                      r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10,
                                      jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                                      stopping_proportional_gain=0.5, stopping_velocity=0.2),
                           KinematicBicycleModel(VEH))
    c._current_state = ego0
    return c


def speed_sweep_trajectories(ego, centerline, limit, mults=SPEED_MULTS):
    """9 centreline speed-hold proposals (accel<=2, decel<=3 m/s^2) - exposes phantom braking."""
    v0 = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
    s0 = float(centerline.project(Point(ego.rear_axle.x, ego.rear_axle.y)))
    trajs = []
    tps = [TimePoint(ego.time_point.time_us + int(i * DT * 1e6)) for i in range(41)]
    for m in mults:
        vt = min(limit, m * max(v0, 1.0))
        v = np.empty(41)
        v[0] = v0
        for i in range(1, 41):
            v[i] = v[i - 1] + np.clip(vt - v[i - 1], -3.0 * DT, 2.0 * DT)
        s = s0 + np.concatenate([[0.0], np.cumsum(v[1:] * DT)])
        arr = np.zeros((41, StateIndex.size()))
        arr[0] = ego_state_to_state_array(ego)
        arr[1:, StateIndex.STATE_SE2] = centerline.interpolate(s[1:], as_array=True)
        arr[1:, StateIndex.VELOCITY_X] = v[1:]
        arr[1:, StateIndex.ACCELERATION_X] = np.diff(v)[0:] / DT
        trajs.append(InterpolatedTrajectory(state_array_to_ego_states(arr, tps, VEH)))
    return trajs


class Agents:
    """Agent provider. mode 'log' = log replay (non-reactive), 'idm' = nuPlan IDMAgents (reactive)."""

    def __init__(self, scenario, mode):
        self.sc, self.mode = scenario, mode
        if mode == "idm":
            self.idm = IDMAgents(target_velocity=10.0, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0,
                                 decel_max=3.0, scenario=scenario,
                                 open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE",
                                                             "GENERIC_OBJECT"])
            self.idm.initialize()

    def observe(self, i) -> DetectionsTracks:
        if self.mode == "log":
            return self.sc.get_tracked_objects_at_iteration(i)
        return self.idm.get_observation()

    def advance(self, it, nxt, history):
        if self.mode == "idm":
            self.idm.update_observation(it, nxt, history)


# ------------------------------------------------------------------ reward evaluation
def _local(ego, p):
    c, s = math.cos(ego.rear_axle.heading), math.sin(ego.rear_axle.heading)
    dx, dy = p[0] - ego.rear_axle.x, p[1] - ego.rear_axle.y
    return dx * c + dy * s, -dx * s + dy * c


class RewardEvaluator:
    """Per-cycle reward with every term logged. One per state (map objects shared across arms/modes)."""

    def __init__(self, scenario, planner0, ego0, centerline, route_lane_ids, limit,
                 brake_baseline=BRAKE_BASELINE_DECEL, jerk_estimator="nuplan"):
        # brake_baseline: decel (m/s^2) of the reference stop progress is measured above; None = M-GYM-1 reward (legacy).
        # jerk_estimator: "nuplan" (smoothed, v2) or "raw" (M-GYM-1 single-step difference).
        self.brake_baseline, self.jerk_estimator = brake_baseline, jerk_estimator
        self.collision_rule = "navsim"       # "legacy" = M-GYM-1 shortcut, kept only so a test can show the pilot's behaviour
        self.sc, self.cl, self.limit = scenario, centerline, max(float(limit), 2.0)
        self.map_api = scenario.map_api
        self.drivable = PDMDrivableMap.from_simulation(self.map_api, ego0, 150)
        self.route_lane_ids = list(route_lane_ids)
        self.scorer = PDMScorer(TrajectorySampling(num_poses=CYCLE_STEPS, interval_length=DT))
        self.cl_pts = np.asarray(centerline.discrete_path and [[p.x, p.y, p.heading] for p in centerline.discrete_path])
        self.cl_s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(self.cl_pts[:, :2], axis=0).T))])
        self._u = None
        self.map_slack_m, self.map_unreliable, self.t0_overshoot_m = self._human_map_slack(ego0)

    # ---- v2 (item 1): the logged start pose can sit a few cm outside the map's drivable polygons
    def _drivable_union(self):
        if self._u is None:
            idc = self.drivable.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                                         SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
            self._u = unary_union([self.drivable._geometries[i] for i in idc])
        return self._u

    def footprint_overshoot(self, ego):
        """Largest distance (m) of a footprint corner outside the union of the polygons the PDM scorer calls drivable."""
        u = self._drivable_union()
        return max(float(u.distance(Point(c))) for c in list(ego.car_footprint.geometry.exterior.coords)[:4])

    def _human_map_slack(self, ego0):
        """If the LOGGED start pose already violates the strict all-corners-inside test (PDM's own scorer then gives every
        proposal 0), the violation is map/pose slop, not driving. Allowance = the human log's largest overshoot over the
        episode horizon, accepted only up to MAX_MAP_SLACK_M. Clean starts get allowance 0 = M-GYM-1 behaviour exactly."""
        o0 = self.footprint_overshoot(ego0)
        if o0 <= 0.0:
            return 0.0, False, 0.0
        n_it = min(self.sc.get_number_of_iterations(), 100)
        worst = max(self.footprint_overshoot(self.sc.get_ego_state_at_iteration(i)) for i in range(0, n_it, 2))
        if worst > MAX_MAP_SLACK_M:
            return 0.0, True, o0
        return worst, False, o0

    def curve_speed(self, s0, ahead=30.0, a_lat=2.0):
        m = (self.cl_s >= s0) & (self.cl_s <= s0 + ahead)
        h = self.cl_pts[m, 2]
        if len(h) < 3:
            return 99.0
        dh = np.abs(np.diff(np.unwrap(h)))
        ds = np.maximum(np.diff(self.cl_s[m]), 1e-3)
        k = float(np.max(dh / ds))
        return 99.0 if k < 1e-4 else math.sqrt(a_lat / k)

    def ego_areas(self, ego_states):
        arr = np.stack([ego_state_to_state_array(e) for e in ego_states])[None]
        sc = self.scorer
        sc._reset(arr, None, self.cl, self.route_lane_ids, self.drivable)
        sc._calculate_ego_area()
        return sc._ego_areas[0]

    def red_lane_ids(self, tl_data):
        return {str(d.lane_connector_id) for d in tl_data if d.status == TrafficLightStatusType.RED}

    @staticmethod
    def brake_ref_dist(v0, t_start, t_len, decel):
        """Distance a constant-decel stop from speed v0 (started at episode t=0) covers in [t_start, t_start+t_len]."""
        v0 = max(float(v0), 0.0)
        def s(t):
            tt = min(t, v0 / decel)
            return v0 * tt - 0.5 * decel * tt * tt
        return s(t_start + t_len) - s(t_start)

    def new_context(self, ego0, a_lon0):
        acc0 = ego0.dynamic_car_state.center_acceleration_2d
        t0 = ego0.time_point.time_us
        return dict(acc_t=deque([t0 - int(k * DT * 1e6) for k in range(JERK_HIST - 1, -1, -1)], maxlen=JERK_HIST),
                    acc_x=deque([acc0.x] * JERK_HIST, maxlen=JERK_HIST), acc_y=deque([acc0.y] * JERK_HIST, maxlen=JERK_HIST),
                    v0_ep=float(ego0.dynamic_car_state.rear_axle_velocity_2d.x), elapsed=0.0,v_hist=deque([ego0.dynamic_car_state.rear_axle_velocity_2d.x] * 20, maxlen=20),
                    a_hist=deque([a_lon0] * 40, maxlen=40), dev_hist=deque([False] * 20, maxlen=20),
                    onc_hist=deque([(False, 0.0, 0.0)] * 10, maxlen=10), prev_plan=None, prev_ego=ego0,
                    prev_acc=None, prev_yaw=None, prev_jerk_state=None)

    def cycle(self, ctx, egos, objs_per_step, tl_data, new_plan):
        """egos: 5 EgoStates after the cycle's steps; objs_per_step: list of tracked-object lists (same steps)."""
        out = dict(hard=None, hard_step=None)
        s_start = float(self.cl.project(Point(ctx["prev_ego"].rear_axle.x, ctx["prev_ego"].rear_axle.y)))
        areas = self.ego_areas([ctx["prev_ego"]] + egos)[1:].copy()
        if self.map_slack_m > 0.0:       # grandfather map slop already present in the human log
            for jj, eg in enumerate(egos):
                if areas[jj, EgoAreaIndex.NON_DRIVABLE_AREA] and self.footprint_overshoot(eg) <= self.map_slack_m:
                    areas[jj, EgoAreaIndex.NON_DRIVABLE_AREA] = False
        red = self.red_lane_ids(tl_data)
        lead_gap = None
        n_lead_steps = n_conf_steps = 0
        term_acc = {k: [] for k in TERMS}
        comf_viol = set()
        n_acc_new = 0
        for j, (ego, objs) in enumerate(zip(egos, objs_per_step)):
            v = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
            a = float(ego.dynamic_car_state.rear_axle_acceleration_2d.x)
            ctx["v_hist"].append(v)
            ctx["a_hist"].append(a)
            epoly = ego.car_footprint.geometry
            ec = (ego.center.x, ego.center.y)
            near = [o for o in objs if (o.center.x - ec[0]) ** 2 + (o.center.y - ec[1]) ** 2 < 60.0 ** 2]
            # ---- hard events
            if areas[j, EgoAreaIndex.NON_DRIVABLE_AREA] and out["hard"] is None:
                out["hard"], out["hard_step"] = "off_drivable", j
            ctx["onc_hist"].append((bool(areas[j, EgoAreaIndex.ONCOMING_TRAFFIC]), ego.rear_axle.x, ego.rear_axle.y))
            oh = ctx["onc_hist"]
            if all(f for f, _, _ in oh) and math.hypot(oh[-1][1] - oh[0][1], oh[-1][2] - oh[0][2]) > 6.0 and out["hard"] is None:
                out["hard"], out["hard_step"] = "wrong_way", j
            colliding = set()
            for o in near:
                if epoly.intersects(o.box.geometry):
                    colliding.add(o.track_token)
                    # v2: the PDM scorer's own at-fault rule (pdm_scorer.py:330-345 with pdm_scorer_utils.get_collision_type),
                    # replacing M-GYM-1's "rear within 1.2 m of the axis" shortcut
                    if self.collision_rule == "legacy":
                        lx, ly = _local(ego, (o.center.x, o.center.y))
                        at_fault = not (lx < -1.0 and abs(ly) < 1.2)
                    else:
                        ctype = get_collision_type(ego_state_to_state_array(ego), epoly, o, o.box.geometry)
                        at_fault = ctype in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION) or (
                            ctype == CollisionType.ACTIVE_LATERAL_COLLISION and
                            bool(areas[j, EgoAreaIndex.MULTIPLE_LANES] or areas[j, EgoAreaIndex.NON_DRIVABLE_AREA]))
                    if at_fault and v > 0.05 and out["hard"] is None:
                        out["hard"], out["hard_step"] = "collision", j
            if v > 0.05 and out["hard"] is None:
                hd = ego.rear_axle.heading
                ux, uy = math.cos(hd), math.sin(hd)
                for o in near:
                    if o.track_token in colliding:
                        continue
                    lx, _ = _local(ego, (o.center.x, o.center.y))
                    ov = getattr(o, "velocity", None)
                    ovx, ovy = (ov.x, ov.y) if ov is not None else (0.0, 0.0)
                    if lx < 0 or (v - (ovx * ux + ovy * uy)) < 0.3:     # behind / not closing
                        continue
                    for dt in (0.2, 0.4, 0.6, 0.8, 1.0):
                        ep = affinity.translate(epoly, ux * v * dt, uy * v * dt)
                        op = affinity.translate(o.box.geometry, ovx * dt, ovy * dt)
                        if ep.intersects(op):
                            out["hard"], out["hard_step"] = "ttc", j
                            break
                    if out["hard"]:
                        break
            if out["hard"] is not None:
                break
            # ---- soft context at last step of the cycle (and steps for hist-based terms)
            # jerk / yaw channels
            acc = ego.dynamic_car_state.rear_axle_acceleration_2d
            yaw = ego.dynamic_car_state.angular_velocity
            cacc = ego.dynamic_car_state.center_acceleration_2d
            ctx["acc_t"].append(ego.time_point.time_us)
            ctx["acc_x"].append(cacc.x)
            ctx["acc_y"].append(cacc.y)
            n_acc_new += 1
            if ctx["prev_acc"] is not None:
                if abs((yaw - ctx["prev_yaw"]) / DT) > COMF["yaw_acc"]:
                    comf_viol.add("yaw_acc")
            ctx["prev_acc"], ctx["prev_yaw"] = (acc.x, acc.y), yaw
            if not COMF["lon_acc"][0] <= acc.x <= COMF["lon_acc"][1]:
                comf_viol.add("lon_acc")
            if abs(acc.y) > COMF["lat_acc"]:
                comf_viol.add("lat_acc")
            if abs(yaw) > COMF["yaw_rate"]:
                comf_viol.add("yaw_rate")
            dev = self.cl.linestring.distance(Point(ec)) > 0.5
            in_x = self.drivable.is_in_layer(Point(ec), SemanticMapLayer.INTERSECTION) if j == len(egos) - 1 else False
            ctx["dev_hist"].append(bool(dev))
            if j == len(egos) - 1:
                ctx["in_intersection"] = in_x
        # ---- jerk: v2 = nuPlan's smoothed estimate over the rolling history; raw kept for the before/after row
        raw_v = jerk_channels_raw(list(ctx["acc_x"])[-(n_acc_new + 1):], list(ctx["acc_y"])[-(n_acc_new + 1):]) if n_acc_new else set()
        if n_acc_new:
            sm_v, out["lon_jerk_max"], out["mag_jerk_max"] = jerk_channels_nuplan(
                list(ctx["acc_t"]), list(ctx["acc_x"]), list(ctx["acc_y"]), n_acc_new)
        else:
            sm_v, out["lon_jerk_max"], out["mag_jerk_max"] = set(), 0.0, 0.0
        out["comf_raw_jerk"] = sorted(raw_v)
        comf_viol |= (sm_v if self.jerk_estimator == "nuplan" else raw_v)
        ego = egos[-1] if out["hard"] is None else egos[min(out["hard_step"], len(egos) - 1)]
        objs = objs_per_step[-1] if out["hard"] is None else objs_per_step[min(out["hard_step"], len(egos) - 1)]
        v = float(ego.dynamic_car_state.rear_axle_velocity_2d.x)
        a = float(ego.dynamic_car_state.rear_axle_acceleration_2d.x)
        s_end = float(self.cl.project(Point(ego.rear_axle.x, ego.rear_axle.y)))
        vbar = float(np.mean([e.dynamic_car_state.rear_axle_velocity_2d.x for e in egos[: (out["hard_step"] or CYCLE_STEPS - 1) + 1]]))
        norm = self.limit * DT * CYCLE_STEPS
        out["progress_norm_legacy"] = float(np.clip((s_end - s_start) / norm, 0.0, 1.5))
        if self.brake_baseline is None:
            out["progress_norm"] = out["progress_norm_legacy"]
        else:
            ref = self.brake_ref_dist(ctx["v0_ep"], ctx["elapsed"], DT * CYCLE_STEPS, self.brake_baseline)
            out["progress_norm"] = float(np.clip((s_end - s_start - ref) / norm, 0.0, 1.5))
        ctx["elapsed"] += DT * CYCLE_STEPS
        out["v"], out["a_lon"], out["s"] = v, a, s_end
        # ---- lead / conflict geometry
        L = max(2.5 * v, 20.0)
        ahead = self.cl.substring(s_end, min(s_end + L, self.cl.length)).buffer(1.5, cap_style=2)
        near = [o for o in objs if (o.center.x - ego.center.x) ** 2 + (o.center.y - ego.center.y) ** 2 < 60.0 ** 2]
        lead, lead_gap = None, None
        for o in near:
            lx, _ = _local(ego, (o.center.x, o.center.y))
            if lx > 0 and ahead.intersects(o.box.geometry):
                g = ego.car_footprint.geometry.distance(o.box.geometry)
                if lead_gap is None or g < lead_gap:
                    lead, lead_gap = o, g
        conf_ttc = None
        for o in near:
            ov = getattr(o, "velocity", None)
            if ov is None or math.hypot(ov.x, ov.y) < 0.3 or ahead.intersects(o.box.geometry):
                continue
            for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                if ahead.intersects(affinity.translate(o.box.geometry, ov.x * t, ov.y * t)):
                    if conf_ttc is None or t < conf_ttc:
                        conf_ttc = t
                    break
        vstar = min(self.limit, self.curve_speed(s_end))
        # ---- soft factors
        q = {k: 1.0 for k in TERMS}
        act = {k: False for k in TERMS}
        if lead is None and conf_ttc is None and vstar >= 1.0 and not ctx.get("stalled_by_red"):
            act["q_free"] = True          # term is APPLICABLE
            if vbar < 0.9 * vstar:
                q["q_free"] = float(np.clip(1.0 - 0.7 * (0.9 * vstar - vbar) / (0.4 * vstar), 0.3, 1.0))
        if lead is not None:
            act["q_gap"] = True
            if v < 1.0:
                q["q_gap"] = 1.0 if lead_gap >= 1.0 else 0.2
            else:
                tg = lead_gap / v
                if tg < 0.8:
                    q["q_gap"] = 0.2
                elif tg < 1.0:
                    q["q_gap"] = 0.2 + 0.3 * (tg - 0.8) / 0.2
                elif tg < 1.5:
                    q["q_gap"] = 0.5 + 0.5 * (tg - 1.0) / 0.5
                elif tg <= 2.5:
                    q["q_gap"] = 1.0
                elif tg <= 3.5:
                    q["q_gap"] = 1.0 - 0.5 * (tg - 2.5)
                else:
                    q["q_gap"] = 0.5
        # red light on the corridor (do not call a red-light stop a stall)
        red_here = False
        if red:
            for lc in self.map_api.get_proximal_map_objects(Point2D_from(ahead.centroid), 30.0, [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]:
                if lc.id in red and ahead.intersects(lc.polygon):
                    red_here = True
                    break
        stalled = all(x < 0.5 for x in list(ctx["v_hist"])[-20:])
        if stalled and conf_ttc is None and not red_here and (lead_gap is None or lead_gap > 10.0):
            act["q_stall"] = True
            q["q_stall"] = 0.5
        if conf_ttc is not None and conf_ttc < 3.0 and v > 1.0 and a > -0.5:
            act["q_yield"] = True
            q["q_yield"] = 0.2
        q["q_comf"] = 0.7 ** len(comf_viol)
        act["q_comf"] = len(comf_viol) > 0
        ah = np.asarray(ctx["a_hist"])
        big = ah[np.abs(ah) > 0.5]
        flips = int(np.sum(np.sign(big[1:]) != np.sign(big[:-1]))) if len(big) > 1 else 0
        if flips > 2:
            q["q_osc"], act["q_osc"] = 0.8, True
        if len(ctx["dev_hist"]) == 20 and all(ctx["dev_hist"]) and not ctx.get("in_intersection", False):
            q["q_lane"], act["q_lane"] = 0.7, True
        if new_plan is not None and ctx["prev_plan"] is not None:
            t_now = egos[-1].time_point
            try:
                p_old = ctx["prev_plan"].get_state_at_time(TimePoint(t_now.time_us + 1_000_000)).center
                p_new = new_plan.get_state_at_time(TimePoint(t_now.time_us + 1_000_000)).center
                if math.hypot(p_old.x - p_new.x, p_old.y - p_new.y) > CONSISTENCY_M:
                    q["q_cons"], act["q_cons"] = 0.9, True
            except Exception:
                pass
        if new_plan is not None:
            ctx["prev_plan"] = new_plan
        qprod = float(np.prod([q[k] for k in TERMS]))
        out.update(q=q, active=act, qprod=qprod, lead_gap=lead_gap, conf_ttc=conf_ttc, comf_viol=sorted(comf_viol))
        out["reward"] = 0.0 if out["hard"] else out["progress_norm"] * qprod
        out["reward_legacy"] = 0.0 if out["hard"] else out["progress_norm_legacy"] * qprod
        ctx["prev_ego"] = egos[-1]
        return out


def Point2D_from(pt):
    from nuplan.common.actor_state.state_representation import Point2D
    return Point2D(pt.x, pt.y)


# ------------------------------------------------------------------ environment
class GymState:
    def __init__(self, st, timer: Optional[Timer] = None, brake_baseline=BRAKE_BASELINE_DECEL, jerk_estimator="nuplan"):
        self.timer = timer or Timer()
        with self.timer("scene_map_build"):
            self.st = st
            self.sc = build_scenario(st)
            self.ego0 = self.sc.initial_ego_state
            self.planner0 = make_planner(self.sc)
            self.obs0 = self.sc.get_tracked_objects_at_iteration(0)
            self.tl0 = list(self.sc.get_traffic_light_status_at_iteration(0))
            it0 = SimulationIteration(self.ego0.time_point, 0)
            hist = SimulationHistoryBuffer.initialize_from_list(2, [self.ego0], [self.obs0], 0.1)
            self.input0 = PlannerInput(it0, hist, self.tl0)
        with self.timer("proposals"):
            pdm_trajs, scores = self.planner0.plan_all(self.input0)
            lane = self.planner0._get_starting_lane(self.ego0)
            self.limit = float(lane.speed_limit_mps or 15.0)
            self.sweep = speed_sweep_trajectories(self.ego0, self.planner0._centerline, self.limit)
            self.proposals = list(pdm_trajs) + self.sweep
            self.names = [f"pdm{i}" for i in range(len(pdm_trajs))] + [f"sweep{m}" for m in SPEED_MULTS]
            self.pdm_scores = scores
        with self.timer("scene_map_build"):
            self.evaluator = RewardEvaluator(self.sc, self.planner0, self.ego0, self.planner0._centerline,
                                             self.planner0._route_lane_dict.keys(), self.limit,
                                             brake_baseline=brake_baseline, jerk_estimator=jerk_estimator)

    @staticmethod
    def _stop_traj(ego):
        arr = np.tile(ego_state_to_state_array(ego), (41, 1))
        arr[:, StateIndex.VELOCITY_X] = 0.0
        arr[:, StateIndex.VELOCITY_Y] = 0.0
        arr[:, StateIndex.ACCELERATION_X] = 0.0
        arr[:, StateIndex.ACCELERATION_Y] = 0.0
        arr[:, StateIndex.ANGULAR_VELOCITY] = 0.0
        tps = [TimePoint(ego.time_point.time_us + int(j * DT * 1e6)) for j in range(41)]
        return InterpolatedTrajectory(state_array_to_ego_states(arr, tps, VEH))

    def first_traj_for(self, policy):
        if policy == 'stop':
            return self._stop_traj(self.ego0)
        mv = [np.mean([e.dynamic_car_state.rear_axle_velocity_2d.x for e in t.get_sampled_trajectory()]) for t in self.proposals]
        return self.proposals[int(np.argmin(mv) if policy == 'slowest' else np.argmax(mv))]

    def _policy_pick(self, planner, pin, ego, policy):
        """Regression policies over the full 24-fan (15 PDM + 9 sweep) re-built each cycle."""
        if policy == 'stop':
            return self._stop_traj(ego)
        trajs, _ = planner.plan_all(pin)
        cur, _o = pin.history.current_state
        trajs = list(trajs) + speed_sweep_trajectories(cur, planner._centerline, self.limit)
        mv = [np.mean([e.dynamic_car_state.rear_axle_velocity_2d.x for e in t.get_sampled_trajectory()]) for t in trajs]
        return trajs[int(np.argmin(mv) if policy == 'slowest' else np.argmax(mv))]

    def rollout(self, k: int, mode: str, horizon_s: float = 8.0, timer: Optional[Timer] = None, policy: str = 'pdm', first_traj=None, hold_cycles: int = 1):
        """Execute proposal k for 0.5 s, then PDM-Closed for horizon_s. Returns per-cycle term logs."""
        tm = timer or self.timer
        n_cycles = 1 + int(round(horizon_s / (CYCLE_STEPS * DT)))
        gc.disable()
        try:
            with tm(f"agents_{mode}"):
                agents = Agents(self.sc, mode)
            ctrl = make_controller(self.sc, self.ego0)
            ego = self.ego0
            obs = agents.observe(0)
            hist = SimulationHistoryBuffer.initialize_from_list(2, [ego], [obs], 0.1)
            ctx = self.evaluator.new_context(ego, float(ego.dynamic_car_state.rear_axle_acceleration_2d.x))
            planner = None
            traj = first_traj if first_traj is not None else self.proposals[k]
            logs, i = [], 0
            replanned = False
            for c in range(n_cycles):
                replanned = c > 0 and (c - 1) % hold_cycles == 0      # hold_cycles=2 -> a plan is held 1.0 s
                if replanned:
                    with tm(f"plan_{mode}"):
                        if planner is None:
                            planner = make_planner(self.sc)
                            planner._route_roadblock_dict = dict(self.planner0._route_roadblock_dict)
                            planner._route_lane_dict = dict(self.planner0._route_lane_dict)
                        it = SimulationIteration(ego.time_point, i)
                        pin = PlannerInput(it, hist, list(self.sc.get_traffic_light_status_at_iteration(i)))
                        if policy == 'pdm':
                            traj = planner.compute_planner_trajectory(pin)
                        else:
                            traj = self._policy_pick(planner, pin, ego, policy)
                egos, objs_steps = [], []
                for j in range(CYCLE_STEPS):
                    it = SimulationIteration(ego.time_point, i)
                    nxt = SimulationIteration(TimePoint(ego.time_point.time_us + int(DT * 1e6)), i + 1)
                    with tm(f"ctrl_{mode}"):
                        ctrl.update_state(it, nxt, ego, traj)
                        ego = ctrl.get_state()
                    with tm(f"agents_{mode}"):
                        agents.advance(it, nxt, hist)
                        obs = agents.observe(i + 1)
                    hist.append(ego, obs)
                    i += 1
                    egos.append(ego)
                    objs_steps.append(list(obs.tracked_objects.tracked_objects))
                with tm(f"score_{mode}"):
                    tl = list(self.sc.get_traffic_light_status_at_iteration(i))
                    rec = self.evaluator.cycle(ctx, egos, objs_steps, tl, traj if replanned else None)
                rec["cycle"] = c
                logs.append(rec)
                if rec["hard"]:
                    break
            return logs
        finally:
            gc.enable()
            gc.collect()


def episode_return(logs):
    return float(sum(r["reward"] for r in logs))
