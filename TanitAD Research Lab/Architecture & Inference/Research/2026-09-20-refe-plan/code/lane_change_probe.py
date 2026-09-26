"""Why does a scenario lose route progress? Measure the MECHANISM, do not eyeball it.

Reports per step: cumulative PATH LENGTH for ego and for the human expert (frame-independent), ego
speed, and the gap to the nearest agent IN FRONT in the ego's INSTANTANEOUS frame (|lateral| < 2.0 m,
longitudinal > 0). Discriminates the two stall causes a progress term cannot tell apart:
  blocked  -> speed -> 0 AND the front gap stays small
  stopped  -> speed -> 0 AND the front gap GROWS (nothing is in the way)

⛔ It deliberately does NOT report cross-track in the initial-heading frame. MEASURED 2026-09-20:
on `changing_lane_to_left` both ego and expert turn >60 deg, so that number is dominated by the TURN
and reads as a 19 m "lane offset" that does not exist. A correct formula in the wrong frame reads
exactly like an answer.

Usage: python lane_change_probe.py <simulation_log.msgpack.xz>
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
from nuplan.planning.simulation.simulation_log import SimulationLog

LAT_HALF_WIDTH, MAX_LOOK = 2.0, 60.0


def _front_gap(ego_state, tracked) -> float:
    ex, ey = ego_state.rear_axle.x, ego_state.rear_axle.y
    h = ego_state.rear_axle.heading
    c, s = math.cos(-h), math.sin(-h)
    best = float("inf")
    for b in tracked:
        dx, dy = b.center.x - ex, b.center.y - ey
        lon, lat = dx * c - dy * s, dx * s + dy * c
        if lon > 0.0 and abs(lat) < LAT_HALF_WIDTH and lon < MAX_LOOK:
            best = min(best, lon)
    return best


def main(path: str) -> None:
    log = SimulationLog.load_data(file_path=Path(path))
    smps = log.simulation_history.data
    sc = log.scenario
    n = len(smps)
    print(f"scenario {sc.scenario_type}/{sc.scenario_name}  log {sc.log_name}  steps {n}")

    P = np.array([[sm.ego_state.rear_axle.x, sm.ego_state.rear_axle.y] for sm in smps])
    ego_len = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    spd = np.array([math.hypot(sm.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                               sm.ego_state.dynamic_car_state.rear_axle_velocity_2d.y) for sm in smps])
    E = []
    for i in range(n):
        try:
            st = sc.get_ego_state_at_iteration(i); E.append([st.rear_axle.x, st.rear_axle.y])
        except Exception:
            break
    E = np.asarray(E)
    exp_len = (np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))])
               if len(E) > 1 else np.zeros(0))

    gaps = []
    for i, sm in enumerate(smps):
        try:
            obs = sc.get_tracked_objects_at_iteration(i).tracked_objects
            gaps.append(_front_gap(sm.ego_state, obs))
        except Exception:
            gaps.append(float("nan"))
    gaps = np.asarray(gaps)

    print("\n  step | ego path  speed | front gap | expert path")
    for i in range(0, n, max(1, n // 12)):
        g = "    none" if not np.isfinite(gaps[i]) else f"{gaps[i]:8.1f}"
        e = f"{exp_len[i]:11.1f}" if i < len(exp_len) else "          -"
        print(f"  {i:4d} | {ego_len[i]:8.1f} {spd[i]:6.2f} | {g}  |{e}")

    slow = spd < 1.0
    print(f"\nEGO    path {ego_len[-1]:7.1f} m   final speed {spd[-1]:.2f}   min {spd.min():.2f}")
    if len(exp_len):
        print(f"EXPERT path {exp_len[-1]:7.1f} m")
        print(f"PATH-LENGTH RATIO ego/expert = {ego_len[-1]/max(exp_len[-1],1e-6):.3f}")
    print(f"STALLED {slow.sum()}/{n} steps below 1.0 m/s ({100*slow.mean():.0f}% of the episode)")
    if slow.any():
        first = int(np.argmax(slow))
        gs = gaps[slow]; gs = gs[np.isfinite(gs)]
        print(f"first stall at step {first} (t={first*0.2:.1f}s), ego had travelled {ego_len[first]:.1f} m")
        if gs.size:
            print(f"front gap WHILE STALLED: min {gs.min():.1f} m  median {np.median(gs):.1f} m  "
                  f"max {gs.max():.1f} m   ({gs.size}/{slow.sum()} stalled steps had an agent in front)")
            print("VERDICT: BLOCKED by a lead vehicle" if np.median(gs) < 12.0
                  else "VERDICT: stopped with NO close lead -- look elsewhere (light/goal/policy)")
        else:
            print("front gap WHILE STALLED: no agent in the front cone at any stalled step")
            print("VERDICT: stopped with NOTHING in front -- not a blocking agent")


if __name__ == "__main__":
    main(sys.argv[1])
