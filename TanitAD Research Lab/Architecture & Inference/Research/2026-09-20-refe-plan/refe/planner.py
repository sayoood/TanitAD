"""REFePlanner -- run REFe closed-loop in the SAME nuPlan harness the DriveRL teacher already runs in.

Without this there is no comparable number. A trained REFe that cannot be scored by their metric
engine, on their scenarios, under the same protocol, is not a reproduction of anything.

INTERFACE, matched to theirs (src/driverl/nuplan/planner.py:50 DriveRLNuPlanPlanner):
    name() · observation_type() -> DetectionsTracks · initialize(PlannerInitialization)
    compute_planner_trajectory(PlannerInput) -> AbstractTrajectory

WHAT REFe CONSUMES, and the one thing that makes it harder than the teacher. The teacher is
PRIVILEGED: it reads ego, agents and the vector map straight from the simulation. REFe is
CAMERA-ONLY, so at every step it needs the front-camera frame for the CURRENT timestamp. nuPlan's
DBs carry only `image.filename_jpg` plus a timestamp; the pixels live in the sensor blobs. This
wrapper therefore resolves frames from a local CAM_F0 root by nearest timestamp, exactly as the
Stage-2 target builder does, and REFUSES rather than guessing when no frame is close enough.

GOAL. Built with THEIR `route_goal_positions` from the route polyline, so the goal REFe receives is
the same quantity the teacher received. num_goal_positions=2, horizon 12.0 s, min speed 5.0 m/s are
the released config's values (MEASURED), not defaults I chose.

SELECTION. REFe emits 64 proposals and a 6-component score each. The executed trajectory is the
proposal with the highest summed score. ⚠️ That mirrors their scorer's role, but our scorer has no
PDM targets yet (Stage 2 scope), so with an untrained scorer this degenerates to an arbitrary pick.
`--select best|first|mean` makes that explicit rather than hiding it behind a default.
"""
from __future__ import annotations

import glob
import math
import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import torch

try:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
    from nuplan.planning.simulation.planner.abstract_planner import (
        AbstractPlanner, PlannerInitialization, PlannerInput)
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    _HAVE_NUPLAN = True
except ImportError:                                    # importable without nuPlan for unit checks
    AbstractPlanner = object
    PlannerInitialization = PlannerInput = Observation = DetectionsTracks = object
    InterpolatedTrajectory = None
    _HAVE_NUPLAN = False

from model import REFe, REFeConfig


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


class FrameResolver:
    """Map a simulation timestamp to one local jpg PER CAMERA. Refuses a mispairing.

    ⛔ WAS SINGLE-CAMERA UNTIL 2026-09-21 (Review 5, D5). It queried `channel='CAM_F0'` and returned
    one path, so the NAVSIM-facing arm would have raised REFe's camera guard on step 1 -- and the
    `max_consec_holds` refusal machinery, built precisely to fail loudly, sits DOWNSTREAM of an
    exception it would never have reached.

    ⚠️ The cameras are NOT synchronised to a common clock (MEASURED: up to ~17 ms apart on the same
    trigger), so each channel is paired to the simulation timestamp independently and the WORST
    residual decides admission -- the same rule `build_targets.py` uses, so training and inference
    accept the same frames.
    """

    def __init__(self, db_dir: str, images_root: str, max_dt_ms: float = 120.0,
                 cameras: tuple[str, ...] = CAMERAS):
        self.db_dir, self.images_root, self.max_dt_ms = db_dir, images_root, max_dt_ms
        self.cameras = tuple(cameras)
        self._idx: dict[str, dict[str, tuple[np.ndarray, list[str]]]] = {}
        self.misses = 0
        self.miss_reason: str = ""
        self._file_index: dict | None = None   # NOT `_index`: that name is the per-log DB method

    def _find(self, rel: str) -> str | None:
        """`<root>/**/<rel>`, answered from a basename index built ONCE.

        ⛔ R21's defect on the eval side: this was `glob(root/**/<log>/<CAM>/<hash>.jpg,
        recursive=True)` per camera per simulation step -- a walk of the whole image tree four
        times a step. The index is by the per-image hash (unique), and a hit must END WITH the
        full relative path, which is exactly what the glob required.
        """
        if self._file_index is None:
            self._file_index = {}
            if self.images_root and os.path.isdir(self.images_root):
                for dp, _dn, fns in os.walk(self.images_root):
                    for fn in fns:
                        if fn.endswith((".jpg", ".jpeg", ".png")):
                            self._file_index.setdefault(fn, os.path.join(dp, fn))
        p = self._file_index.get(os.path.basename(rel))
        if p is not None and p.replace("\\", "/").endswith("/" + rel.lstrip("/")):
            return p
        return None

    def _index(self, log_name: str):
        if log_name in self._idx:
            return self._idx[log_name]
        p = os.path.join(self.db_dir, f"{log_name}.db")
        per: dict[str, tuple[np.ndarray, list[str]]] = {
            ch: (np.zeros(0, np.int64), []) for ch in self.cameras}
        if os.path.exists(p):
            c = sqlite3.connect(p)
            for ch in self.cameras:
                row = c.execute("SELECT token FROM camera WHERE channel=?", (ch,)).fetchone()
                if not row:
                    continue
                rs = c.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token=? "
                               "ORDER BY timestamp", (row[0],)).fetchall()
                per[ch] = (np.array([r[0] for r in rs], dtype=np.int64), [r[1] for r in rs])
            c.close()
        self._idx[log_name] = per
        return per

    def resolve(self, log_name: str, timestamp_us: int) -> list[str] | None:
        """One path per camera, in `self.cameras` order, or None with `miss_reason` set."""
        per = self._index(log_name)
        out: list[str] = []
        for ch in self.cameras:
            ts, names = per[ch]
            if ts.size == 0:
                self.misses += 1
                self.miss_reason = f"{ch}: no rows in {log_name}.db"
                return None
            j = int(np.argmin(np.abs(ts - timestamp_us)))
            dt = abs(int(ts[j]) - int(timestamp_us)) / 1000.0
            if dt > self.max_dt_ms:
                self.misses += 1
                self.miss_reason = f"{ch}: nearest frame {dt:.1f} ms away (> {self.max_dt_ms})"
                return None
            hit = self._find(names[j])
            hits = [hit] if hit else []
            if not hits:
                # containerised layout: "<log>_<cam>.zip" holding members named by image hash
                zp = os.path.join(self.images_root, f"{log_name}_{ch}.zip")
                base = os.path.basename(names[j])
                if os.path.exists(zp):
                    import zipfile
                    try:
                        if base in zipfile.ZipFile(zp).namelist():
                            out.append(f"{zp}::{base}")
                            continue
                    except Exception:
                        pass
                self.misses += 1
                self.miss_reason = f"{ch}: no image file for {names[j]}"
                return None
            out.append(hits[0])
        return out


TRAJ_DT_S = 0.2   # the policy's own query cadence; matches score_proposals.TRAJ_DT


def repair_last_heading(traj):
    """SPEC_NAVTEST Amendment 7 (ADOPTED 2026-09-27, eval/RESULT_A7_a7confirm_ep015.md): the model's OWN heading at
    the last native pose (index -1, t = 4.0 s) is corrupted in every evaluated snapshot (median ~1.1 rad off the
    path, |heading| > pi on ~60 %; GOALS_AND_CLAIMS D-REFE-LASTYAW-1), so the executed plan takes the previous pose's
    heading there. Positions, speed and every other heading are untouched. [..., T, 3] numpy or torch in -> a COPY of
    the same type and dtype out; confirmed +16.20 [+12.83, +19.52] PDMS on 923 tokens disjoint from W3's subset."""
    out = traj.clone() if isinstance(traj, torch.Tensor) else np.array(traj, copy=True)
    out[..., -1, 2] = out[..., -2, 2]
    return out


# ---- SPEC_NAVTEST Amendment 8 (registered 2026-09-28 06:32 Berlin, SPEC blob 8ace80cc): test-time GOAL SANITISATION.
# DEFAULT OFF (REFePlanner.SANITIZE_GOAL) until the amendment reads ADOPT; OFF, `_goal_for` is bit-identical to before.
# The defect (raw/2026-09-28-goal-clamp/goal_trace.json): when a scenario's route_roadblock_ids do not contain the ego's
# roadblock, `_route_start_index` takes the nearest route roadblock at ANY distance and `route_goal_positions` clamps
# the ego's projection to the route's FIRST point at any distance -- the goal lands 349-480 m ahead. ON: if the ego is
# farther than GOAL_SANITIZE_D_M from the route polyline `_goal_for` builds, the goal is re-derived by the SAME
# `route_goal_positions` on a fallback route that starts on the ego's OWN lane and follows the driving command at forks;
# with no lane within GOAL_FALLBACK_LANE_RADIUS_M, the straight-route goal along the ego heading. Inputs: the map, the
# ego pose, v0 at t0 and NAVSIM's driving command -- never GT, never future ego. D, the radius and the fork rule were
# fixed in the registered amendment BEFORE any confirmation token was read; they are NOT tuned.
GOAL_SANITIZE_D_M = 20.0
GOAL_FALLBACK_LANE_RADIUS_M = 10.0
GOAL_FALLBACK_MIN_LEN_M = 150.0
GOAL_FALLBACK_MAX_EDGES = 64          # a loop guard only
# NAVSIM's driving_command one-hot order; MEASURED on the 1,123 selection tokens (2026-09-28): index 0 -> mean GT heading
# at 4 s +0.656 rad (left), 1 -> -0.001, 2 -> -0.602 (right), 3 never set
DRIVING_COMMANDS = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")


def ego_to_polyline_m(poly) -> float:
    """distance from the ego -- the origin of the route's anchor frame -- to the polyline's SEGMENTS (Amendment 8's
    trigger quantity; `raw/2026-09-28-goal-clamp/route_cover_census.py` computes the same)"""
    P = np.asarray(poly, dtype=np.float64)
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab, axis=1).min())


def _polyline_len(coords) -> float:
    c = np.asarray(coords, dtype=np.float64)
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0


def _exit_heading(coords) -> float:
    (x0, y0), (x1, y1) = coords[-2], coords[-1]
    return math.atan2(y1 - y0, x1 - x0)


def choose_successor(prev_coords, succs, command: str, coords_of):
    """Amendment 8's fork rule: among `succs`, the one whose exit heading change (its own exit heading minus the
    previous edge's exit heading, wrapped) is the MOST counter-clockwise for LEFT, the MOST clockwise for RIGHT, and
    the smallest in magnitude for STRAIGHT / UNKNOWN. Ties break on the edge id, so the choice is deterministic."""
    h0 = _exit_heading(prev_coords)

    def dh(e):
        return float((_exit_heading(coords_of(e)) - h0 + math.pi) % (2.0 * math.pi) - math.pi)
    if command == "LEFT":
        return max(succs, key=lambda e: (dh(e), str(getattr(e, "id", ""))))
    if command == "RIGHT":
        return min(succs, key=lambda e: (dh(e), str(getattr(e, "id", ""))))
    return min(succs, key=lambda e: (abs(dh(e)), str(getattr(e, "id", ""))))


def fallback_route(map_api, anchor, command: str, M):
    """(polyline in the anchor frame | None, diag). Candidates: lanes + lane connectors within the radius; the start is
    the best `_route_edge_anchor_score` (distance + 5 x heading error); extended through `outgoing_edges` by
    `choose_successor` until the stitched route is >= GOAL_FALLBACK_MIN_LEN_M; resampled by `_fit_route_polyline`."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    near = map_api.get_proximal_map_objects(Point2D(anchor.x, anchor.y), GOAL_FALLBACK_LANE_RADIUS_M,
                                            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    cands = [o for lay in near.values() for o in lay if len(M._edge_centerline_coords(o)) >= 2]
    if not cands:
        return None, {"fallback": "straight", "n_candidates": 0}
    start = min(cands, key=lambda e: (M._route_edge_anchor_score(e, anchor), str(getattr(e, "id", ""))))
    edges = [start]
    length = _polyline_len(M._edge_centerline_coords(start))
    while length < GOAL_FALLBACK_MIN_LEN_M and len(edges) < GOAL_FALLBACK_MAX_EDGES:
        succ = [e for e in (getattr(edges[-1], "outgoing_edges", []) or []) if len(M._edge_centerline_coords(e)) >= 2]
        if not succ:
            break
        nxt = choose_successor(M._edge_centerline_coords(edges[-1]), succ, command, M._edge_centerline_coords)
        edges.append(nxt)
        length += _polyline_len(M._edge_centerline_coords(nxt))
    coords = M._stitch_route_edge_coords(edges)
    local = np.asarray([M._global_to_anchor_xy(x, y, anchor) for x, y in coords], dtype=np.float32)
    return M._fit_route_polyline(local, M.ROUTE_POINTS), {
        "fallback": "lane", "n_candidates": len(cands), "start_edge": str(getattr(start, "id", "")),
        "edges": [str(getattr(e, "id", "")) for e in edges], "route_len_m": round(length, 2)}


# ---- 2026-10-04 GOAL FIX candidates (SPEC_NAVTEST Amendment 9 draft; DEFAULT None = `_goal_for` bit-identical).
# The PI (2026-10-01): "no lane geometry for the model driving; nav commands and the related nav goals can of course be
# used" and "use the output of the privileged planner". Three candidates, each named and recorded:
#   "pdm_route"        the ROUTE is corrected exactly as NAVSIM's privileged planner corrects it at iteration 0
#                      (pdm_route_correction.route_roadblock_correction, vendored verbatim from navsim 3e8291b) and the
#                      goal is then derived by the UNCHANGED path. It is the route NAVSIM's scorer measures EP and DDC
#                      against (the metric cache's `centerline`). Nothing new reaches the model: only the nav goal moves.
#   "navgoal_straight" / "navgoal_arc"   MAP-FREE: a trigger that reads only the nav goal and v0 -- the goal is
#                      impossible when |p2| > NAVGOAL_KAPPA * max(|v0|, v_min) * horizon (a route goal is an arc-length
#                      point at exactly max(|v0|, v_min) * horizon, so its chord cannot exceed it) -- and on a trigger the
#                      goal is re-placed at the same arc lengths along the ego heading, or along the constant-curvature
#                      arc omega / max(|v0|, 1 m/s) (omega = the ego's yaw rate).
# NAVGOAL_KAPPA was chosen on the 1,123 selection tokens ONLY and frozen before any held-out token was read
# (raw/2026-10-01-goal-trigger/goal_trigger_analysis.json: selection F1 0.951; held-out F1 0.914 vs Amendment 8's trigger).
GOAL_FIXES = ("pdm_route", "navgoal_straight", "navgoal_arc")
NAVGOAL_KAPPA = 1.2


def pdm_corrected_route_ids(map_api, rear_axle, ids):
    """(ids after NAVSIM's PDM route correction, diag). Builds the route dict the way PDM-Closed's `_load_route_dicts`
    does (ROADBLOCK, else ROADBLOCK_CONNECTOR); an id that resolves to neither is dropped and counted."""
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    import pdm_route_correction as PRC
    rd, n_unres = {}, 0
    for i in ids:
        b = map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK) or \
            map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if b is None:
            n_unres += 1
            continue
        rd[b.id] = b
    if not rd:
        return list(ids), {"route_fix": "pdm_route", "changed": False, "n_unresolved": n_unres, "empty": True}
    new = list(PRC.route_roadblock_correction(rear_axle, map_api, rd))
    return new, {"route_fix": "pdm_route", "changed": new != list(ids), "n_before": len(ids), "n_after": len(new),
                 "n_prepended": next((k for k, x in enumerate(new) if x in rd), len(new)), "n_unresolved": n_unres}


def navgoal_fallback(g, v_xy, omega, mode, horizon_s, min_speed_mps, kappa=NAVGOAL_KAPPA):
    """(goal, diag): the map-free trigger + fallback. `g` is the [.., 2n] goal (ego frame), returned UNCHANGED (the same
    tensor) below the trigger."""
    flat = g.reshape(-1)
    n = flat.numel() // 2
    s = max(math.hypot(*v_xy), min_speed_mps) * horizon_s
    ratio = float(torch.linalg.norm(flat[2 * n - 2:2 * n])) / s
    diag = {"goal_fix": mode, "navgoal_ratio": round(ratio, 4), "triggered": ratio > kappa}
    if ratio <= kappa:
        return g, diag
    k = (float(omega) / max(math.hypot(*v_xy), 1.0)) if mode == "navgoal_arc" else 0.0
    pts = []
    for j in range(1, n + 1):
        L = s * j / n
        pts += [L, 0.0] if abs(k) < 1e-6 else [math.sin(k * L) / k, (1.0 - math.cos(k * L)) / k]
    diag["curvature"] = round(k, 6)
    return torch.tensor(pts, dtype=g.dtype).reshape(g.shape), diag


class REFePlanner(AbstractPlanner):
    # ⛔ WAS False, WHICH MADE THE PLANNER STRUCTURALLY UNABLE TO SEE A LOG NAME. MEASURED by the
    # 2026-09-20 conformance review: `PlannerInitialization` carries only
    # ['route_roadblock_ids', 'mission_goal', 'map_api'] and `mission_goal` is a StateSE2 with
    # ['x','y','heading'] -- neither has a `log_name`. `_log_hint` was a class attribute that was
    # never assigned. So the frame lookup could never resolve, and
    # `compute_planner_trajectory` returned `_hold(ego)` on EVERY step: a closed-loop run would
    # have scored a STATIONARY CAR and reported it as a REFe result.
    # ⇒ ask the runner for the scenario, which is the only object that carries `log_name`.
    requires_scenario = True

    def __init__(self, checkpoint: str | None = None, images_root: str = "",
                 db_dir: str = "D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14",
                 backbone: str = "vitl16", device: str = "cuda", select: str = "best",
                 horizon_s: float = 12.0, min_speed_mps: float = 5.0, scenario=None, rule: str | None = None,
                 repair_last_heading: bool | None = None, sanitize_goal: bool | None = None,
                 goal_fix: str | None = None):
        # the selection RULE is named and recorded, never implicit (SPEC_NAVTEST Amendment 5, RETRACTION_LOG R25)
        self.rule = rule or self.DEFAULT_RULE
        if self.rule not in self.RULES:
            raise ValueError(f"unknown selection rule {self.rule!r}; one of {self.RULES}")
        # the last-pose heading repair is named and recorded too (SPEC_NAVTEST Amendment 7); False reproduces every
        # evaluation before 2026-09-27 ~14:30 Berlin
        self.repair_last_heading = (self.REPAIR_LAST_HEADING if repair_last_heading is None
                                    else bool(repair_last_heading))
        # the goal sanitisation is named and recorded too (SPEC_NAVTEST Amendment 8); OFF (the default until ADOPT)
        # reproduces every evaluation before it bit for bit. `driving_command` is NAVSIM's one-hot, set by the runner
        # per scenario (read only when the sanitisation is ON); `goal_diag` records what the last goal did.
        self.sanitize_goal = self.SANITIZE_GOAL if sanitize_goal is None else bool(sanitize_goal)
        # the 2026-10-04 goal-fix candidates, named and recorded; None (the default) is bit-identical to before
        self.goal_fix = self.GOAL_FIX if goal_fix is None else goal_fix
        if self.goal_fix is not None and self.goal_fix not in GOAL_FIXES:
            raise ValueError(f"unknown goal_fix {self.goal_fix!r}; one of {GOAL_FIXES}")
        if self.goal_fix is not None and self.sanitize_goal:
            raise ValueError("goal_fix and sanitize_goal (Amendment 8) are exclusive")
        self.driving_command = None
        self.goal_diag = None
        self.cfg = REFeConfig.for_backbone(backbone)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = REFe(self.cfg).to(self.device).eval()
        if checkpoint:
            # FULL (`model_final.pt`: strict load, as before) or PARTIAL (`snap_epochNNN.pt`:
            # trunk from its DINOv3 weights + trained tensors, admitted only on a matching
            # frozen-trunk fingerprint). One loader for trainer and planner -- see ckpt_io.
            import ckpt_io
            meta: dict = {}
            self.ckpt_format = ckpt_io.load_for_inference(self.model, checkpoint,
                                                          map_location=self.device,
                                                          backbone=backbone, meta_out=meta)
            # ⭐ R22: a model trained on each sample's OWN camera rig must be fed each scenario's
            # own rig at inference -- read from the checkpoint, never guessed
            self.per_sample_calib = bool(meta.get("per_sample_calib", False))
            self.trained = True
        else:
            self.trained = False
            self.per_sample_calib = False
        self._calib_cache: dict = {}
        self.frames = FrameResolver(db_dir, images_root)
        self.select, self.horizon_s, self.min_speed_mps = select, horizon_s, min_speed_mps
        self._init = None
        self._route_poly = None
        self._route_fix_diag = None
        self._scenario = scenario
        self._log_hint = None
        self._log_hint_source = "unset"
        self.n_steps = self.n_no_frame = self._consec_holds = 0
        # ⛔ A SILENT HOLD IS THE FAILURE MODE THAT MATTERS. Holding is the right response to ONE
        # unresolvable frame; holding for the whole scenario means the planner never perceived
        # anything and the run is scoring a parked car. Refuse loudly after this many in a row.
        self.max_consec_holds = 20

    # ---- AbstractPlanner interface -------------------------------------------------
    def name(self) -> str:
        return "REFePlanner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization) -> None:
        self._init = initialization
        self._route_poly = None
        self._route_fix_diag = None
        # the scenario arrives either on the initialization or via the constructor, depending on
        # how the runner is wired; take whichever is present and RECORD which, so a later reader
        # does not have to guess where the name came from.
        sc = getattr(initialization, "scenario", None) or self._scenario
        if sc is not None and getattr(sc, "log_name", None):
            self._log_hint = sc.log_name
            self._log_hint_source = "scenario.log_name"
        else:
            self._log_hint_source = "NONE -- the frame lookup cannot resolve"
        self.n_steps = self.n_no_frame = self._consec_holds = 0

    def compute_planner_trajectory(self, current_input):
        ego: EgoState = current_input.history.ego_states[-1]
        self.n_steps += 1
        img = self._image_for(ego)
        if img is None:
            self.n_no_frame += 1
            self._consec_holds += 1
            # REFUSE to invent a frame. Holding the current state is a visible, scoreable failure;
            # feeding noise would produce a plausible trajectory from no perception at all.
            # ⛔ BUT A HOLD THAT NEVER ENDS IS NOT VISIBLE AT ALL -- it scores as a stationary car
            # and looks like a poor policy rather than a broken one. Fail loud instead.
            if self._consec_holds >= self.max_consec_holds:
                raise RuntimeError(
                    f"REFePlanner: {self._consec_holds} consecutive steps with no resolvable "
                    f"camera frame (log hint: {self._log_hint!r} from {self._log_hint_source}). "
                    f"This is NOT a driving failure -- the planner never perceived anything, and "
                    f"any score from this run would describe a parked car. Wire the scenario in, "
                    f"or point --images at the fetched camera root. "
                    f"Last miss: {self.frames.miss_reason or 'unknown'}")
            return self._hold(ego)
        self._consec_holds = 0
        traj, _score, k = self.infer(ego, img)
        return self._to_trajectory(ego, self.executed(traj, k).cpu().numpy())

    def executed(self, traj, k: int):
        """The plan this planner EXECUTES: proposal `k` of `traj` [M, T, 3], with the Amendment-7 repair when it is
        on. Every consumer (the nuPlan loop above, the NAVSIM seam) takes the executed plan from here, so the repair
        cannot be applied in one path and forgotten in another."""
        p = traj[k]
        return repair_last_heading(p) if self.repair_last_heading else p

    # ---- internals -----------------------------------------------------------------
    def _ego_vec(self, ego) -> torch.Tensor:
        """The 7-D ego kinematics at t0, in the order the bank stores them."""
        ego_v = ego.dynamic_car_state.rear_axle_velocity_2d
        ego_a = ego.dynamic_car_state.rear_axle_acceleration_2d
        # ⛔ THIS ENDED IN A LITERAL 0.0 UNTIL 2026-09-21 -- a pad, and the planner was the one
        # consumer nobody updated when `ego_dim` moved. Seven real kinematic scalars now, in the
        # SAME ORDER `build_targets.py` banks them; `diag_consumer_conformance.py` asserts that
        # this vector, the bank's rows and `REFeConfig.ego_dim` are all the same width.
        ego_vec = torch.tensor([[float(ego_v.x), float(ego_v.y), float(ego_a.x), float(ego_a.y),
                                 float(ego.dynamic_car_state.angular_velocity),
                                 float(ego.tire_steering_angle),
                                 float(math.hypot(ego_v.x, ego_v.y))]],
                               dtype=torch.float32, device=self.device)
        if ego_vec.shape[1] != self.model.cfg.ego_dim:
            raise RuntimeError(
                f"REFePlanner builds a {ego_vec.shape[1]}-D ego vector but the model expects "
                f"{self.model.cfg.ego_dim}-D. These two must move together.")
        return ego_vec

    def infer(self, ego, img: torch.Tensor):
        """ONE model call from the planner's own inputs -> (traj [M, T, 3] ego frame, score [M, 6], k).

        Factored out 2026-09-24 so the NAVSIM navtest eval (`eval/refe_navtest_seam.py`) and the
        nuPlan closed-loop planner run the SAME input construction and the SAME selection -- two
        implementations of one quantity is how two checks agree on a wrong answer.
        """
        ego_vec = self._ego_vec(ego)
        goal = self._goal_for(ego).to(self.device)
        self.last_inputs = (ego_vec, goal)             # recorded only (Amendment 8's gate (e)); changes nothing
        calib = self._calib_for(self._log_hint) if self.per_sample_calib else None
        with torch.no_grad():
            traj, score = self.model(img.to(self.device), ego_vec, goal, calib=calib)
        k = self._pick(score)
        return traj[0], score[0], k
    def _calib_for(self, log_name):
        """[1, n_cam, 16] float64: THIS log's camera rig, read from its own DB (R22).

        ⛔ Refuses rather than falling back to the baked rig: a per-sample-trained model fed the
        baked rig is exactly the mismatch R22 removed, and it would not error -- it would drive
        with every lifted feature displaced by up to ~2.7 deg.
        """
        if log_name not in self._calib_cache:
            import calib_table as CT
            db = os.path.join(self.frames.db_dir, f"{log_name}.db")
            cams = CT.read_db(db, self.model.cfg.cameras)
            missing = [ch for ch in self.model.cfg.cameras if ch not in cams]
            if missing:
                raise RuntimeError(f"REFePlanner: no calibration for {missing} in {db}; the model "
                                   f"was trained per-sample and must not be fed the baked rig")
            self._calib_cache[log_name] = torch.tensor(
                [[cams[ch] for ch in self.model.cfg.cameras]], dtype=torch.float64)
        return self._calib_cache[log_name]

    # NAVSIM's weights for the averaged half. The multiplicative half needs none: a violation
    # zeroes the product.
    #
    # ⚠️ HOW THIS DIFFERS FROM NAVSIM'S OWN RULE, MEASURED against `autonomousvision/navsim`
    # `pdm_scorer.py` by the 2026-09-20 fix-verification review, and stated here rather than left
    # as a silent approximation:
    #     reference   multiplicative  NC · DAC · DDC · TLC            (FOUR terms)
    #                 weighted        EP(5) · TTC(5) · LK(2) · HC(2)  (denominator 14)
    #     ours        multiplicative  NC · DAC · DDC                  (THREE)
    #                 weighted        EP(5) · TTC(5) · comfort(2)     (denominator 12)
    # ⇒ TWO deviations, both forced by the COMPONENT SET rather than chosen here:
    #   * TLC (traffic-light compliance) is absent. DriveZero's paper names SIX components and TLC
    #     is not among them, so the scorer head has no output to put there. Adding it needs a
    #     seventh component and a calculator that produces it -- see D-REFE-TLC below.
    #   * one `comfort` stands in for NAVSIM's separate lane-keeping and history-comfort terms,
    #     which is why the denominator is 12 and not 14.
    # ⛔ A REFe number aggregated this way is NOT an EPDMS and must not be reported as one. It is
    # the paper's six components combined with the benchmark's STRUCTURE, which is the most faithful
    # thing available while the head emits six.
    PDM_W = (5.0, 5.0, 4.0)          # progress, time-to-collision, comfort(=LK+HC), sum 14 (rule "v2_shape")
    # ⭐ SPEC_NAVTEST AMENDMENT 5 (PI option 3, 2026-09-26): the harness REFe is scored on is NAVSIM **v1** PDMS --
    # NC x DAC x (5 EP + 5 TTC + 2 C) / 12, driving direction at weight 0 (navsim @ 3e8291b `pdm_scorer.py:38-42`).
    # "v2_shape" above is NAVSIM v2's EPDMS shape, which every point before Amendment 5 was selected with
    # (RETRACTION_LOG R25). Both stay selectable BY NAME so every earlier point remains reproducible.
    V1_W = (5.0, 5.0, 2.0)           # progress, time-to-collision, comfort, sum 12 (rule "navsim_v1")
    RULES = ("v2_shape", "navsim_v1")
    # ⭐ FLIPPED 2026-09-26 21:23 Berlin by SPEC Amendment 5's verdict (eval/RESULT_A5_a5confirm_ep013.md): NO MEASURABLE
    # DIFFERENCE, +0.78 [-0.30, +1.82] PDMS on 923 tokens from the 43 logs outside W3's subset -- not harmful, so the
    # benchmark's own formula becomes the selection rule for every evaluation from then on (the pre-registered branch).
    DEFAULT_RULE = "navsim_v1"
    # ⭐ ON from 2026-09-27 ~14:30 Berlin by SPEC Amendment 7's verdict (eval/RESULT_A7_a7confirm_ep015.md): ADOPT,
    # 60.76 -> 76.96 PDMS, +16.20 [+12.83, +19.52] on the 923 confirmation tokens (43 logs outside W3's subset).
    REPAIR_LAST_HEADING = True
    # SPEC_NAVTEST Amendment 8 (registered 2026-09-28): the goal sanitisation. OFF until the amendment reads ADOPT.
    SANITIZE_GOAL = False
    # 2026-10-04 goal-fix candidates (GOAL_FIXES above). None until a pre-registered confirmation reads ADOPT.
    GOAL_FIX = None

    def aggregate(self, score: torch.Tensor) -> torch.Tensor:
        """The BENCHMARK SCORING RULE, over probabilities, not a sum of logits.

        ⛔ THE OLD LINE WAS `score[0].sum(-1).argmax()` -- an UNWEIGHTED SUM OF SIX RAW LOGITS,
        with no sigmoid and no aggregation. A sum of logits is the log-odds PRODUCT, which is
        neither the PDM rule nor a monotone transform of it: the PDM rule multiplies the
        violation terms and takes a WEIGHTED AVERAGE of the graded ones, so a candidate that fails
        one violation term must score ZERO however good the rest are, and a sum can never express
        that. The paper: "aggregated according to the benchmark scoring rule ... the candidate with
        the highest predicted aggregate score".
        Component order is `train.ScorerBank.COMPONENTS`:
            0 collision (NC) · 1 drivable area (DAC) · 2 progress (EP)
            3 time to collision (TTC) · 4 comfort (C) · 5 driving direction (DDC)
        """
        p = score.sigmoid()
        if self.rule == "navsim_v1":
            # NAVSIM v1 PDMS: NC x DAC x (5 EP + 5 TTC + 2 C) / 12 -- driving direction does not enter
            w1 = torch.tensor(self.V1_W, device=p.device, dtype=p.dtype)
            return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()
        # rule "v2_shape" (every point before Amendment 5):
        # multiplicative group: NAVSIM has FOUR -- NC, DAC, DDC, TLC. We emit three; traffic-light
        # compliance has no component in the paper's six, so it enters as an EXPLICIT constant 1.0
        # rather than by being quietly omitted. Replace `tlc` the day a seventh component exists.
        tlc = torch.ones_like(p[..., 0])
        mult = p[..., 0] * p[..., 1] * p[..., 5] * tlc            # NC x DAC x DDC x TLC(=1)
        # weighted group: NAVSIM is EP(5) + TTC(5) + LANE_KEEPING(2) + HISTORY_COMFORT(2) over 14.
        # Our single comfort component stands for BOTH 2-weight terms, so it carries weight 4 and
        # the denominator is 14 -- which keeps the aggregate on the benchmark's scale instead of a
        # 12-denominator of our own invention.
        w = torch.tensor(self.PDM_W, device=p.device, dtype=p.dtype)
        avg = (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()
        return mult * avg

    def _pick(self, score: torch.Tensor) -> int:
        if self.select == "first":
            return 0
        if self.select == "mean":
            return int(score.shape[1] // 2)
        if self.select == "logitsum":            # the OLD rule, kept only as a named control
            return int(score[0].sum(-1).argmax())
        return int(self.aggregate(score)[0].argmax())

    def _image_for(self, ego) -> torch.Tensor | None:
        # ⛔ The two lookups that used to lead this expression -- initialization.mission_goal
        # .log_name and initialization.log_name -- are DEAD and were removed 2026-09-20.
        # MEASURED: PlannerInitialization carries only [route_roadblock_ids, mission_goal,
        # map_api], and mission_goal is a StateSE2 with [x, y, heading]. Neither has a log_name,
        # so the chain ALWAYS fell through to _log_hint -- which was never assigned. Leaving them
        # in would keep suggesting the name arrives from the initialization; it does not and it
        # never could. It comes from the SCENARIO, which is why requires_scenario is now True.
        log_name = self._log_hint
        paths = self.frames.resolve(log_name, int(ego.time_point.time_us)) if log_name else None
        if not paths:
            return None
        import cv2
        frames = []
        for p in paths:
            if "::" in p:                       # containerised: "<log>_<cam>.zip::<member>"
                import zipfile
                zp, member = p.split("::", 1)
                try:
                    buf = np.frombuffer(zipfile.ZipFile(zp).read(member), dtype=np.uint8)
                    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                except Exception:
                    return None
            else:
                im = cv2.imread(p)
            if im is None:
                return None
            # ⚠️ BGR -> RGB is the `[:, :, ::-1]` here and it is load-bearing: cv2 decodes BGR and
            # the frozen trunk's ImageNet normalisation is defined on RGB. Same class-A defect the
            # package advisory names; kept explicit so a refactor cannot drop it silently.
            im = cv2.resize(im, (self.cfg.img_w, self.cfg.img_h))[:, :, ::-1]
            frames.append(np.ascontiguousarray(im.astype(np.float32).transpose(2, 0, 1)) / 255.0)
        if len(frames) != self.cfg.n_cameras:
            raise RuntimeError(
                f"REFePlanner resolved {len(frames)} frames but the model wants "
                f"{self.cfg.n_cameras} ({', '.join(self.cfg.cameras)}).")
        # [1, N_cam, 3, H, W] -- the shape REFe.forward expects for a multi-camera rig
        return torch.from_numpy(np.stack(frames, axis=0))[None]

    _log_hint: str | None = None

    def _goal_for(self, ego) -> torch.Tensor:
        """THEIR route_goal_positions on the route polyline, so REFe sees the teacher's goal."""
        from driverl.datatypes.goal_position_utils import route_goal_positions
        if self._route_poly is None:
            # ⛔ `augment_routes` lives in the package's `code/`, not beside this file: it imported only
            # when a runner happened to put `code/` on sys.path. MEASURED 2026-09-24 by the navtest
            # eval's E-2 arm: ModuleNotFoundError on the first goal. Resolve the sibling dir here.
            _code = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
            if _code not in sys.path:
                sys.path.insert(0, _code)
            import augment_routes as A
            from nuplan.planning.script import driverl_runtime_map_features as M  # noqa: F401
            ids = list(getattr(self._init, "route_roadblock_ids", []) or [])
            if self.goal_fix == "pdm_route" and ids:   # the privileged planner's route, once per scenario (as PDM)
                ids, self._route_fix_diag = pdm_corrected_route_ids(self._init.map_api, ego.rear_axle, ids)
            poly, _ = A._route_with_lane_rank(self._init.map_api, ids, A._anchor_from_ego(ego), 0)
            self._route_poly = poly
        if self._route_poly is None:
            return torch.zeros(1, 2 * self.cfg.n_goal_points)
        rp = torch.as_tensor(self._route_poly[None], dtype=torch.float32)
        mask = torch.as_tensor((np.abs(self._route_poly).sum(-1) > 0)[None])
        v = ego.dynamic_car_state.rear_axle_velocity_2d
        g = route_goal_positions(torch.zeros(1, 1, 2),
                                 torch.tensor([[[float(v.x), float(v.y)]]], dtype=torch.float32),
                                 rp, mask, horizon_s=self.horizon_s,
                                 min_speed_mps=self.min_speed_mps,
                                 num_goal_positions=self.cfg.n_goal_points)
        if self.sanitize_goal:                         # SPEC_NAVTEST Amendment 8 -- OFF by default (nothing changes)
            g = self._sanitized_goal(ego, g, v, A, M, route_goal_positions)
        if self.goal_fix == "pdm_route":
            self.goal_diag = dict(self._route_fix_diag or {}, goal=[round(float(x), 4) for x in g.reshape(-1).tolist()])
        elif self.goal_fix in ("navgoal_straight", "navgoal_arc"):
            g, diag = navgoal_fallback(g, (float(v.x), float(v.y)), ego.dynamic_car_state.angular_velocity,
                                       self.goal_fix, self.horizon_s, self.min_speed_mps)
            diag["goal"] = [round(float(x), 4) for x in g.reshape(-1).tolist()]
            self.goal_diag = diag
        return g.reshape(1, -1)

    def _command_name(self) -> str:
        c = self.driving_command
        if c is None:
            return "UNKNOWN"
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        return DRIVING_COMMANDS[int(np.argmax(c))] if c.size == len(DRIVING_COMMANDS) and c.max() > 0 else "UNKNOWN"

    def _sanitized_goal(self, ego, g, v, A, M, route_goal_positions):
        """Amendment 8: below the trigger the goal is returned UNCHANGED (the same tensor); above it, re-derived."""
        d = ego_to_polyline_m(self._route_poly)
        triggered = d > GOAL_SANITIZE_D_M
        diag = {"ego_to_route_m": d, "triggered": bool(triggered), "command": self._command_name(), "fallback": None}
        if triggered:
            anchor = A._anchor_from_ego(ego)
            poly, fd = fallback_route(self._init.map_api, anchor, diag["command"], M)
            diag.update(fd)
            n = self.cfg.n_goal_points
            if poly is None:
                s = max(math.hypot(float(v.x), float(v.y)), self.min_speed_mps) * self.horizon_s
                g = torch.tensor([[[c for k in range(1, n + 1) for c in (s * k / n, 0.0)]]], dtype=g.dtype)
            else:
                rp = torch.as_tensor(poly[None], dtype=torch.float32)
                mask = torch.as_tensor((np.abs(poly).sum(-1) > 0)[None])
                g = route_goal_positions(torch.zeros(1, 1, 2),
                                         torch.tensor([[[float(v.x), float(v.y)]]], dtype=torch.float32),
                                         rp, mask, horizon_s=self.horizon_s, min_speed_mps=self.min_speed_mps,
                                         num_goal_positions=n)
            diag["goal"] = [round(float(x), 4) for x in g.reshape(-1).tolist()]
        self.goal_diag = diag
        return g

    def _to_trajectory(self, ego, xyyaw: np.ndarray):
        """Ego-frame (x, y, yaw) samples at 5 Hz -> a nuPlan InterpolatedTrajectory."""
        from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
        from nuplan.common.actor_state.ego_state import EgoState as ES
        c, s = math.cos(ego.rear_axle.heading), math.sin(ego.rear_axle.heading)
        states = [ego]
        dt_us = int(0.2 * 1e6)
        for i, (x, y, yaw) in enumerate(xyyaw, start=1):
            gx = ego.rear_axle.x + float(x) * c - float(y) * s
            gy = ego.rear_axle.y + float(x) * s + float(y) * c
            states.append(ES.build_from_rear_axle(
                StateSE2(gx, gy, ego.rear_axle.heading + float(yaw)),
                ego.dynamic_car_state.rear_axle_velocity_2d,
                ego.dynamic_car_state.rear_axle_acceleration_2d,
                ego.tire_steering_angle, ego.time_point + TimePoint(dt_us * i),
                ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)

    def _hold(self, ego):
        """A stationary trajectory whose states carry DISTINCT, ADVANCING timestamps.

        ⛔ THE OLD VERSION REPEATED THE SAME EgoState OBJECT. That gives 21 states with ONE distinct
        timestamp, and `InterpolatedTrajectory.get_state_at_time` RAISES on it -- so the "safe"
        fallback was itself broken, and every hold would have crashed the simulation rather than
        producing the visible, scoreable failure it was written to produce.
        ⚠️ `diag_planner_holds` did not catch this because it STUBS `_hold`. A guard that replaces
        the thing it is guarding tests the harness, not the code.
        """
        if InterpolatedTrajectory is None:
            return None
        from nuplan.common.actor_state.state_representation import TimePoint
        dt_us = int(TRAJ_DT_S * 1e6)
        t0 = int(ego.time_point.time_us)
        states = [ego]
        for i in range(1, self.cfg.horizon_steps + 1):
            states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=ego.rear_axle,
                rear_axle_velocity_2d=ego.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=ego.dynamic_car_state.rear_axle_acceleration_2d,
                tire_steering_angle=ego.tire_steering_angle,
                time_point=TimePoint(t0 + i * dt_us),
                vehicle_parameters=ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)

    def report(self) -> dict:
        return {"steps": self.n_steps, "steps_without_a_frame": self.n_no_frame,
                "frame_resolver_misses": self.frames.misses, "trained": self.trained,
                "select": self.select, "backbone": self.cfg.backbone, "rule": self.rule,
                "repair_last_heading": self.repair_last_heading}
