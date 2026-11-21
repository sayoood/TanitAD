"""Per-rollout wall time, render_ms and plan_ms in the banked T2-A box-2 rollouts (RTX 4060).  CPU, git tip only."""
import json, subprocess, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
REPO, TIP = "C:/Users/Admin/tanitad-push", "c59e7b85"
B = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-box2/raw/results"
def gs(p):
    return json.loads(subprocess.run(["git", "-C", REPO, "show", f"{TIP}:{p}"], capture_output=True, check=True).stdout.decode("utf-8"))
out = {}
for sc in ("00040136", "7c72937c"):
    for arm, fn in (("cl_p10", "rollouts_cl_p10_empty.json"), ("refcv7_vmax_off_s0", "rollouts_refcv7_empty.json")):
        d = gs(f"{B}/{sc}/{arm}/{fn}")
        rm = [r["render_ms"] for r in d["rollouts"]]
        ws = [r["wall_s"] for r in d["rollouts"]]
        pm = [s["plan_ms"] for r in d["rollouts"] for s in r["steps"]]
        out[f"{sc}/{arm}"] = {"rollouts": len(ws), "wall_s_per_rollout_median": float(np.median(ws)), "wall_s_per_rollout_sum": float(np.sum(ws)),
                              "render_ms_mean_of_rollouts": float(np.mean(rm)), "plan_ms_median": float(np.median(pm)), "plan_ms_p95": float(np.percentile(pm, 95)),
                              "ticks": len(pm), "wall_ms_per_tick_median": float(np.median(ws)) * 1000 / 50}
print(json.dumps(out, indent=1))
