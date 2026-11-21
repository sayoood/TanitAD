"""Share of banked T2-A ticks (refcv7_vmax_off_s0) with a logged lead inside 25 m / 15 m, per scene.  CPU, git tip only."""
import json, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
REPO, TIP = "C:/Users/Admin/tanitad-push", "c59e7b85"
B = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-box2/raw/results"
def gs(p):
    return json.loads(subprocess.run(["git", "-C", REPO, "show", f"{TIP}:{p}"], capture_output=True, check=True).stdout.decode("utf-8"))
out = {}
for sc in ("00040136", "7c72937c"):
    d = gs(f"{B}/{sc}/refcv7_vmax_off_s0/rollouts_refcv7_empty.json")
    n = a25 = a15 = 0
    hw = []
    for r in d["rollouts"]:
        for s in r["steps"]:
            n += 1
            l = s.get("lead", {}).get("lead25", {})
            h = l.get("headway_m")
            if h is not None and h > 0:
                a25 += 1
                hw.append(h)
                if h <= 15:
                    a15 += 1
    out[sc] = {"ticks": n, "ticks_lead_headway_pos_in_lead25_slot": a25, "of_which_le_15m": a15,
               "headway_min": min(hw) if hw else None, "headway_median": sorted(hw)[len(hw)//2] if hw else None}
print(json.dumps(out, indent=1))
