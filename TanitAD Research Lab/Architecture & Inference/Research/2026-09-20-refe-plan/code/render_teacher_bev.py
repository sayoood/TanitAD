"""Render a DriveRL nuPlan closed-loop SimulationLog (*.msgpack.xz) to a BEV mp4.

Per the TanitEval viz standard: metric BEV + a text overlay of what the policy is doing.
This is the TEACHER on nuPlan MINI, so there is no camera panel (mini ships no camera blobs)
and the DriveRL reward components are NOT in a nuPlan SimulationLog (they live in the DriveRL
engine, not in nuPlan's metrics) -- the overlay therefore shows what the log actually holds:
ego speed/heading, the teacher's planned trajectory, the expert (human log) trajectory for the
same scenario, the mission goal, agents, lanes, and a nearest-agent-gap proxy. When a DriveRL
sidecar (per-step JSONL written by our planner wrapper) is present next to the log, its fields
(jerk, steering rate, Beta params, value channels) are overlaid too.

Untested until the first real log exists (written 2026-09-20 against the devkit schema).
Usage: python render_teacher_bev.py <log.msgpack.xz> <out.mp4> [--radius 60] [--fps 5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from nuplan.planning.simulation.simulation_log import SimulationLog  # noqa: E402

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

AGENT_COLORS = {"VEHICLE": "#e07a5f", "PEDESTRIAN": "#f2cc8f", "BICYCLE": "#81b29a"}


def _f(v, fmt="{:+.3f}"):
    try:
        return fmt.format(float(v))
    except Exception:
        return "n/a"


def _box_corners(cx, cy, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    pts = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts]


def _ego_xyh(ego_state):
    st = ego_state.center if hasattr(ego_state, "center") else ego_state.rear_axle
    return float(st.x), float(st.y), float(st.heading)


def _ego_speed(ego_state):
    try:
        v = ego_state.dynamic_car_state.rear_axle_velocity_2d
        return math.hypot(float(v.x), float(v.y))
    except Exception:
        return float("nan")


def _planned_xy(trajectory):
    try:
        pts = trajectory.get_sampled_trajectory()
        return [(float(p.center.x), float(p.center.y)) if hasattr(p, "center") else (float(p.x), float(p.y)) for p in pts]
    except Exception:
        return []


def _agents(observation):
    out = []
    try:
        for obj in observation.tracked_objects.tracked_objects:
            b = obj.box
            t = getattr(getattr(obj, "tracked_object_type", None), "name", "VEHICLE")
            out.append((float(b.center.x), float(b.center.y), float(b.center.heading), float(b.length), float(b.width), t))
    except Exception:
        pass
    return out


def _lanes(map_api, x, y, radius):
    """Lane centerlines + boundaries within radius, as polylines. Defensive: any failure -> []"""
    polys = []
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        objs = map_api.get_proximal_map_objects(Point2D(x, y), radius, layers)
        for layer in layers:
            for lane in objs.get(layer, []):
                try:
                    pts = [(p.x, p.y) for p in lane.baseline_path.discrete_path]
                    polys.append(("center", pts))
                except Exception:
                    pass
                for side in ("left_boundary", "right_boundary"):
                    try:
                        pts = [(p.x, p.y) for p in getattr(lane, side).discrete_path]
                        polys.append(("bound", pts))
                    except Exception:
                        pass
    except Exception:
        pass
    return polys


def _expert_xy(scenario, n):
    out = []
    for i in range(n):
        try:
            st = scenario.get_ego_state_at_iteration(i)
            out.append(_ego_xyh(st)[:2])
        except Exception:
            break
    return out


def _load_sidecar(log_path: Path):
    for cand in (log_path.with_suffix("").with_suffix(".driverl.jsonl"), log_path.parent / "driverl_steps.jsonl"):
        if cand.exists():
            rows = [json.loads(l) for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
            return {int(r.get("iteration", i)): r for i, r in enumerate(rows)}
    return {}


def render(log_path: Path, out_path: Path, radius: float = 60.0, fps: int = 5, dpi: int = 110):
    log = SimulationLog.load_data(file_path=log_path)
    hist = log.simulation_history
    samples = hist.data
    scenario = log.scenario
    map_api = getattr(hist, "map_api", None) or getattr(scenario, "map_api", None)
    goal = getattr(hist, "mission_goal", None)
    expert = _expert_xy(scenario, len(samples))
    sidecar = _load_sidecar(log_path)
    name = getattr(scenario, "scenario_name", "?")
    stype = getattr(scenario, "scenario_type", "?")
    token = getattr(scenario, "token", "?")
    log_name = getattr(scenario, "log_name", "?")  # the source nuPlan DB log; scenario_name == token, so print it ONCE

    fig = plt.figure(figsize=(9, 9), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    writer = None
    frames = []
    for i, smp in enumerate(samples):
        ax.clear()
        ax.set_facecolor("#1b1b1f")
        ex, ey, eh = _ego_xyh(smp.ego_state)
        for kind, pts in _lanes(map_api, ex, ey, radius):
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#7a7a85" if kind == "center" else "#3d3d45", lw=1.0 if kind == "center" else 0.6, zorder=1)
        # agents
        min_gap = float("inf")
        for cx, cy, ch, L, W, t in _agents(smp.observation):
            ax.add_patch(Polygon(_box_corners(cx, cy, ch, L, W), closed=True, fc=AGENT_COLORS.get(t, "#e07a5f"), ec="none", alpha=0.85, zorder=3))
            min_gap = min(min_gap, math.hypot(cx - ex, cy - ey) - 0.5 * L)
        # expert (human log) trajectory so far and ahead
        if expert:
            xs, ys = zip(*expert)
            ax.plot(xs, ys, color="white", lw=1.2, ls="--", alpha=0.7, zorder=4, label="expert log")
        # ego footprint + heading
        try:
            L, W = smp.ego_state.car_footprint.length, smp.ego_state.car_footprint.width
        except Exception:
            L, W = 4.6, 1.9
        ax.add_patch(Polygon(_box_corners(ex, ey, eh, L, W), closed=True, fc="#3a86ff", ec="white", lw=1.0, zorder=5))
        ax.plot([ex, ex + 3 * math.cos(eh)], [ey, ey + 3 * math.sin(eh)], color="white", lw=1.5, zorder=6)
        # teacher's planned trajectory
        plan = _planned_xy(smp.trajectory)
        if len(plan) > 1:
            xs, ys = zip(*plan)
            ax.plot(xs, ys, color="#2ee59d", lw=2.2, zorder=6, label="DriveRL plan")
        if goal is not None:
            ax.plot([goal.x], [goal.y], marker="*", ms=16, color="#ffd166", zorder=7, label="mission goal")
        # the policy's OWN two goal anchors (near/far), persisted in ego (rear-axle) frame -> global
        gp = getattr(smp.trajectory, "goal_points", None)
        if gp:
            try:
                ra = smp.ego_state.rear_axle
                c, s = math.cos(ra.heading), math.sin(ra.heading)
                gx = [ra.x + c * float(px) - s * float(py) for px, py in gp]
                gy = [ra.y + s * float(px) + c * float(py) for px, py in gp]
                ax.plot(gx, gy, marker="D", ms=9, ls="none", color="#ff70a6", zorder=7, label="policy goal anchors")
            except Exception:
                pass
        # the road graph AS THE POLICY SAW IT (debug_info.model_input.road_graph, global frame) -> thin cyan
        try:
            dbg0 = getattr(smp.trajectory, "debug_info", None) or {}
            rg = (dbg0.get("model_input") or {}).get("road_graph") or {}
            for lane in rg.get("lanes", []):
                pts = lane.get("points") or []
                if len(pts) > 1:
                    xs, ys = zip(*[(float(p[0]), float(p[1])) for p in pts])
                    ax.plot(xs, ys, color="#4cc9f0", lw=0.7, alpha=0.9, zorder=2)
        except Exception:
            pass
        ax.set_xlim(ex - radius, ex + radius)
        ax.set_ylim(ey - radius, ey + radius)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        # overlay
        v = _ego_speed(smp.ego_state)
        t_s = i / fps
        lines = [
            f"DriveRL teacher (u2400)  nuPlan mini  |  {stype}",
            f"token {name}   log {log_name}",
            f"iter {i:3d}  t={t_s:5.1f}s   v={v:5.2f} m/s ({v*3.6:5.1f} km/h)   heading={math.degrees(eh):6.1f} deg",
            f"nearest agent gap ~ {min_gap:5.1f} m" if math.isfinite(min_gap) else "no agents in view",
        ]
        # What DriveRL's planner PERSISTS in every sample (DriveRLActionTrajectory, verified 2026-09-20):
        # jerk_long, lat_command, raw_action (Beta sample/mode), acceleration_control, steering_control,
        # goal_points (ego frame), debug_info{model_input{ego, other_agents, road_graph}, test_time_scaling}.
        traj = smp.trajectory
        jl, lc = getattr(traj, "jerk_long", None), getattr(traj, "lat_command", None)
        ac, stc = getattr(traj, "acceleration_control", None), getattr(traj, "steering_control", None)
        if jl is not None:
            lines.append(f"DriveRL action: jerk_long={jl:+.2f} m/s3  lat_cmd={lc:+.3f}  ->  accel_ctrl={_f(ac)}  steer_ctrl={_f(stc)}")
        raw = getattr(traj, "raw_action", None)
        if raw is not None:
            try:
                lines.append("raw Beta action: " + " ".join(f"{float(v):+.3f}" for v in np.asarray(raw).ravel()[:4]))
            except Exception:
                pass
        dbg = getattr(traj, "debug_info", None) or {}
        mi = dbg.get("model_input") if isinstance(dbg.get("model_input"), dict) else {}
        tts = mi.get("test_time_scaling") if isinstance(mi, dict) else None
        if tts:
            sel, n = tts.get("selected_candidate"), tts.get("num_candidates")
            try:
                scs = np.asarray(tts.get("candidate_scores")).ravel()
                lines.append(f"TTS N={n}: selected={sel}  score={scs[int(sel)]:+.3f} vs mode={scs[0]:+.3f}  "
                             f"(switch margin {tts.get('total_return_switch_margin')}, gamma {tts.get('gamma')})")
            except Exception:
                lines.append(f"TTS N={n}: selected={sel}")
        sc = sidecar.get(i)
        if sc:
            lines.append("DriveRL: jerk={:+.2f} m/s3  steer_rate={:+.3f} rad/s  value={}".format(
                float(sc.get("jerk_long", float("nan"))), float(sc.get("steer_rate", float("nan"))), sc.get("value_total", "?")))
            comps = sc.get("value_components")
            if comps:
                lines.append("V: " + "  ".join(f"{k[:9]}={float(v):+.2f}" for k, v in comps.items()))
        else:
            lines.append("(no DriveRL sidecar: reward/value components not in a nuPlan log)")
        ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", color="white", fontsize=9, family="monospace",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"))
        ax.legend(loc="lower right", fontsize=8, facecolor="black", labelcolor="white", framealpha=0.5)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
    plt.close(fig)
    if not frames:
        raise SystemExit(f"no samples in {log_path}")
    h, w = frames[0].shape[:2]
    if cv2 is None:
        raise SystemExit("opencv-python-headless is required to write mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"RENDERED {out_path}  frames={len(frames)}  {w}x{h}@{fps}fps  scenario={stype}/{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("out"); ap.add_argument("--radius", type=float, default=60.0); ap.add_argument("--fps", type=int, default=5)
    a = ap.parse_args()
    render(Path(a.log), Path(a.out), a.radius, a.fps)
