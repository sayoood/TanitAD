"""When value-guided search DOES override the policy mode, which way does it move the car?

The action is (jerk_long, lat_command); jerk_long is bounded [-8, +5] (driverl_teacher.yaml).
For every step where selected_candidate != 0, compares the SELECTED candidate's jerk to the MODE's.

⭐ The control that makes this interpretable: the same statistic over the NON-selected candidates at
those same steps. If the available pool is itself skewed toward braking, a braking-skewed SELECTION
is not evidence of a selection bias -- it is evidence about the sampler. Both are printed.

Usage: python switch_direction.py <run_root> [<run_root> ...]
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
from nuplan.planning.simulation.simulation_log import SimulationLog


def main(roots: list[str]) -> None:
    print(f"{'run':>10s} {'switches':>9s} {'d_jerk SELECTED':>16s} {'d_jerk pool mean':>17s} "
          f"{'% selected slower':>18s} {'% pool slower':>14s}")
    for root in roots:
        dsel, dpool, n = [], [], 0
        for lp in sorted(glob.glob(f"{root}/**/*.msgpack.xz", recursive=True)):
            for sm in SimulationLog.load_data(file_path=Path(lp)).simulation_history.data:
                t = ((getattr(sm.trajectory, "debug_info", None) or {}).get("model_input") or {}).get("test_time_scaling")
                if not t:
                    continue
                k = int(t.get("selected_candidate", 0))
                if k == 0:
                    continue
                a = np.asarray(t["candidate_actions"], dtype=float)
                a = a.reshape(a.shape[0], -1)
                mode = a[0, 0]
                dsel.append(float(a[k, 0] - mode))
                others = np.delete(a[:, 0], 0)
                dpool.append(float(np.mean(others - mode)))
                n += 1
        if not n:
            print(f"{Path(root).name:>10s} {0:9d}   (no switches)"); continue
        S, P = np.array(dsel), np.array(dpool)
        print(f"{Path(root).name:>10s} {n:9d} {S.mean():+16.4f} {P.mean():+17.4f} "
              f"{100*(S < 0).mean():17.1f}% {100*(P < 0).mean():13.1f}%")
    print("\nd_jerk = selected jerk_long - mode jerk_long, in m/s^3 (action bound [-8, +5]).")
    print("NEGATIVE = the override brakes harder / accelerates less than the policy's own mode.")
    print("The POOL column is the control: it is the mean over the non-selected sampled candidates at")
    print("the SAME steps, so a pool near 0 with a negative selection isolates the SELECTION.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["C:/dzo/m-nr-8", "C:/dzo/m-nr-16", "C:/dzo/m-nr-32", "C:/dzo/m-nr-64"])
