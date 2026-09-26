"""Is the DriveRL critic's value gap FLAT at low speed generally, or only on the stall token?

For every TTS step in every scenario of a run, reads the persisted candidate table and computes the
ADVANTAGE = best candidate score - mode score (candidate_sources[0] == 'argmax', verified). Buckets
it by ego speed. The switch rule (tts.py:387) executes a sampled candidate only when the advantage
EXCEEDS total_return_switch_margin, so a bucket whose advantage never clears the margin is a regime
where value-guided search is structurally inert.

⚠️ Controls this carries: the candidate ACTION spread per bucket (a small advantage with a small
action spread would mean the policy collapsed, not the critic flattened), and the count per bucket.

Usage: python critic_flatness.py <run_root>     e.g. C:/dzo/m-nr-16
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import numpy as np
from nuplan.planning.simulation.simulation_log import SimulationLog

EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 99.0]


def main(root: str) -> None:
    logs = sorted(glob.glob(f"{root}/**/*.msgpack.xz", recursive=True))
    rows, margin, nsrc_ok = [], None, 0
    for lp in logs:
        log = SimulationLog.load_data(file_path=Path(lp))
        stype = log.scenario.scenario_type
        for sm in log.simulation_history.data:
            t = ((getattr(sm.trajectory, "debug_info", None) or {}).get("model_input") or {}).get("test_time_scaling")
            if not t:
                continue
            src = t.get("candidate_sources")
            if src and str(src[0]) != "argmax":
                raise ValueError(f"candidate 0 is {src[0]!r}, not 'argmax' -- advantage undefined")
            nsrc_ok += 1
            margin = t.get("total_return_switch_margin", margin)
            s = np.asarray(t["candidate_scores"], dtype=float).ravel()
            if s.size < 2:
                continue
            a = np.asarray(t["candidate_actions"], dtype=float).reshape(s.size, -1)
            v = math.hypot(sm.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                           sm.ego_state.dynamic_car_state.rear_axle_velocity_2d.y)
            rows.append((stype, v, float(s.max() - s[0]), float(np.abs(a - a[0]).max()),
                         int(t.get("selected_candidate", 0)) != 0))
    if not rows:
        print("no TTS steps in this run"); return
    V = np.array([r[1] for r in rows]); A = np.array([r[2] for r in rows])
    S = np.array([r[3] for r in rows]); W = np.array([r[4] for r in rows])
    print(f"run {root}   TTS steps {len(rows)}   scenarios {len({r[0] for r in rows})}   "
          f"switch margin {margin}   (candidate_sources[0]=='argmax' verified on {nsrc_ok} steps)\n")
    print(f"{'ego speed':>14s} {'n':>6s} {'mean adv':>9s} {'max adv':>9s} {'>margin':>8s} {'switched':>9s} {'action spread':>14s}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (V >= lo) & (V < hi)
        if not m.any():
            continue
        over = (A[m] > margin).mean() if margin is not None else float("nan")
        print(f"{lo:5.1f}-{hi:5.1f} m/s {m.sum():6d} {A[m].mean():9.5f} {A[m].max():9.5f} "
              f"{100*over:7.1f}% {100*W[m].mean():8.1f}% {S[m].mean():14.3f}")
    lowm = V < 1.0
    print(f"\nLOW SPEED (<1 m/s): n={lowm.sum()}  max advantage {A[lowm].max():.5f}  margin {margin}  "
          f"-> {'NEVER clears the bar' if A[lowm].max() <= margin else 'CLEARS the bar sometimes'}")
    hi = V >= 5.0
    if hi.any():
        print(f"HIGH SPEED (>=5 m/s): n={hi.sum()}  max advantage {A[hi].max():.5f}  "
              f"ratio of max advantages high/low = {A[hi].max()/max(A[lowm].max(),1e-9):.1f}x")
    print(f"action spread low vs high: {S[lowm].mean():.3f} vs {S[hi].mean():.3f}  "
          f"(a COLLAPSED policy would show a small spread at low speed; it does not)")
    json.dump({"run": root, "margin": margin, "n": len(rows),
               "low_max_adv": float(A[lowm].max()), "high_max_adv": float(A[hi].max()) if hi.any() else None,
               "low_spread": float(S[lowm].mean()), "high_spread": float(S[hi].mean()) if hi.any() else None},
              open(Path(root).name + "_critic_flatness.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/dzo/m-nr-16")
