"""For the M1-like ticks of the banked scene-7c72937c refcv7 rollout: what does the harness record about a lead? CPU, git tip only."""
import json, math, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
REPO, TIP = "C:/Users/Admin/tanitad-push", "c59e7b85"
B = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-box2/raw/results/7c72937c"
def gs(p):
    return json.loads(subprocess.run(["git", "-C", REPO, "show", f"{TIP}:{p}"], capture_output=True, check=True).stdout.decode("utf-8"))
d = gs(f"{B}/refcv7_vmax_off_s0/rollouts_refcv7_empty.json")
gt = d["gt"]
def lsp(i, n=10):
    j = min(i + n, len(gt) - 1)
    return math.hypot(gt[j]["x"] - gt[i]["x"], gt[j]["y"] - gt[i]["y"]) / ((j - i) * 0.1) if j > i else float("nan")
rows = []
for r in d["rollouts"]:
    for s in r["steps"]:
        i = s["i_gt"]; lv = lsp(i)
        if lv >= 5.2 and s["v"] < 5.2:
            L = s.get("lead", {})
            l25 = L.get("lead25", {})
            rows.append({"start": r["start_frame"], "k": s["k"], "i_gt": i, "v_own": round(s["v"], 2), "v_log": round(lv, 2),
                         "lead25_headway": l25.get("headway_m"), "lead25_v": l25.get("v_lead"), "lead25_tg": l25.get("time_gap_s")})
print(json.dumps({"n_m1_ticks": len(rows), "condition": d["condition"], "lead_path_extrapolated_calls": d.get("lead_path_extrapolated_calls"),
                  "headway_finite_positive": sum(1 for r in rows if r["lead25_headway"] is not None and r["lead25_headway"] > 0),
                  "first_rows": rows[:6], "last_rows": rows[-4:]}, indent=1))
