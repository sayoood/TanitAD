"""Is the `changing_lane_to_left` stall unique, or does the released teacher stall elsewhere too?

Runs the same measurement over EVERY simulation log of a run and tabulates, per scenario:
path length (ego vs human expert), the ratio, stalled fraction, and -- for stalled episodes -- the
front-cone gap at the LAST stalled step, which discriminates "still blocked" from "did not resume".

⚠️ This is a census within one 8-scenario sample, not a characterisation of the teacher. It turns
an n=1 instance into k-of-8, which is the cheapest thing that can sharpen the claim.

Usage: python stall_census.py <run_root>        e.g. C:/dzo/m-nr-n
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import numpy as np
from nuplan.planning.simulation.simulation_log import SimulationLog

SLOW, LAT, LOOK = 1.0, 2.0, 60.0


def front_gap(ego, tracked) -> float:
    c, s = math.cos(-ego.rear_axle.heading), math.sin(-ego.rear_axle.heading)
    best = float("inf")
    for b in tracked:
        dx, dy = b.center.x - ego.rear_axle.x, b.center.y - ego.rear_axle.y
        lon, lat = dx * c - dy * s, dx * s + dy * c
        if 0 < lon < LOOK and abs(lat) < LAT:
            best = min(best, lon)
    return best


def one(path: str) -> dict:
    log = SimulationLog.load_data(file_path=Path(path)); sc = log.scenario
    smps = log.simulation_history.data; n = len(smps)
    P = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y] for s in smps])
    ego_len = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    spd = np.array([math.hypot(s.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                               s.ego_state.dynamic_car_state.rear_axle_velocity_2d.y) for s in smps])
    E = []
    for i in range(n):
        try:
            st = sc.get_ego_state_at_iteration(i); E.append([st.rear_axle.x, st.rear_axle.y])
        except Exception:
            break
    E = np.asarray(E)
    exp_len = float(np.linalg.norm(np.diff(E, axis=0), axis=1).sum()) if len(E) > 1 else float("nan")
    slow = spd < SLOW
    r = {"type": sc.scenario_type, "token": sc.scenario_name, "n": n,
         "ego_m": round(ego_len, 1), "expert_m": round(exp_len, 1),
         "ratio": round(ego_len / exp_len, 3) if exp_len == exp_len and exp_len > 0 else None,
         "stall_frac": round(float(slow.mean()), 3), "final_speed": round(float(spd[-1]), 2)}
    if slow.any():
        last = int(np.max(np.flatnonzero(slow)))
        g = front_gap(smps[last].ego_state, sc.get_tracked_objects_at_iteration(last).tracked_objects)
        r["last_stall_step"] = last
        r["gap_at_last_stall_m"] = None if not np.isfinite(g) else round(float(g), 1)
    return r


def main(root: str) -> None:
    logs = sorted(glob.glob(f"{root}/**/*.msgpack.xz", recursive=True))
    rows = [one(p) for p in logs]
    rows.sort(key=lambda r: (r["ratio"] is None, r["ratio"]))
    print(f"{'scenario type':43s} {'ratio':>6s} {'ego m':>7s} {'exp m':>7s} {'stall':>6s} {'vend':>5s} {'gap@laststall':>14s}")
    for r in rows:
        g = "-" if r.get("gap_at_last_stall_m") is None else f"{r['gap_at_last_stall_m']:.1f} m"
        st = "-" if r["stall_frac"] == 0 else f"{100*r['stall_frac']:.0f}%"
        print(f"{r['type'][:43]:43s} {str(r['ratio']):>6s} {r['ego_m']:7.1f} {r['expert_m']:7.1f} "
              f"{st:>6s} {r['final_speed']:5.2f} {g:>14s}")
    k = sum(1 for r in rows if r["stall_frac"] > 0.05)
    print(f"\nSTALLED (>5% of steps below {SLOW} m/s): {k} of {len(rows)} scenarios")
    for r in rows:
        if r["stall_frac"] > 0.05:
            print(f"  {r['type']}: {100*r['stall_frac']:.0f}% stalled, ends at {r['final_speed']:.2f} m/s, "
                  f"gap at last stalled step {r.get('gap_at_last_stall_m')} m")
    print(json.dumps(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/dzo/m-nr-n")
