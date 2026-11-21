"""Count M1-like ticks in the banked T2-A box-2 rollouts (2026-10-17).  CPU, reads the git tip only.

M1 tick (A3/A6 card reading, HANDOFF_LFREAD item 2): logged route speed >= 5.2 m/s AND own speed < 5.2 m/s.
Logged route speed here = first-second path length of the gt track ahead of the tick (10 Hz gt, 10 frames),
the same operationalisation as lf_read.logged_speed.  Own speed = the rollout step's `v` (executed state).
Run:  python -I m1_ticks_in_t2a.py  > m1_ticks_in_t2a.json
"""
import json
import math
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
REPO = "C:/Users/Admin/tanitad-push"
TIP = "c59e7b85"
B = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-box2/raw/results"
SCENES = {"00040136": "00040136", "7c72937c": "7c72937c"}
ARMS = {"refcv7_vmax_off_s0": "rollouts_refcv7_empty.json", "refcv7_vmax_off_s1": "rollouts_refcv7_empty.json",
        "cl_p10": "rollouts_cl_p10_empty.json", "cl_p10_t0": "rollouts_cl_p10_t0_empty.json"}
V_THR = 5.2


def git_show(path):
    out = subprocess.run(["git", "-C", REPO, "show", f"{TIP}:{path}"], capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8"))


def gt_speed_ahead(gt, i, n=10):
    j = min(i + n, len(gt) - 1)
    if j <= i:
        return float("nan")
    d = math.hypot(gt[j]["x"] - gt[i]["x"], gt[j]["y"] - gt[i]["y"])
    return d / ((j - i) * 0.1)


res = {"tip": TIP, "v_threshold_ms": V_THR, "definition": "logged first-second path speed >= 5.2 and own v < 5.2", "scenes": {}}
for sc in SCENES:
    base = f"{B}/{sc}"
    d0 = git_show(f"{base}/cl_p10/rollouts_cl_p10_empty.json")
    gt = d0["gt"]
    sp = [gt_speed_ahead(gt, i) for i in range(len(gt))]
    frac_ge = sum(1 for s in sp if s >= V_THR) / len(sp)
    sc_out = {"n_gt_frames": len(gt), "gt_speed_ahead_min": min(sp), "gt_speed_ahead_max": max(sp),
              "frac_gt_frames_logged_speed_ge_5p2": frac_ge, "arms": {}}
    for arm, fn in ARMS.items():
        try:
            d = git_show(f"{base}/{arm}/{fn}")
        except Exception as e:  # noqa: BLE001
            sc_out["arms"][arm] = {"error": repr(e)[:120]}
            continue
        n = m1 = lag = 0
        per_start = {}
        for r in d["rollouts"]:
            c = 0
            for s in r["steps"]:
                i = s["i_gt"]
                lv = gt_speed_ahead(gt, i)
                n += 1
                if lv >= V_THR:
                    lag += 1
                    if s["v"] < V_THR:
                        m1 += 1
                        c += 1
            per_start[r["start_frame"]] = c
        sc_out["arms"][arm] = {"ticks": n, "ticks_logged_ge_5p2": lag, "m1_ticks": m1, "m1_share_of_ticks": m1 / n if n else None,
                               "m1_per_start": per_start}
    res["scenes"][sc] = sc_out
json.dump(res, sys.stdout, indent=1)
