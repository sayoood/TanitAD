"""Pilot timing from the trainer's own log: `sec` is CUMULATIVE since the loop start (not per step), so s/step is a difference of two rows over the step difference. Std lib only."""
import json
import os
import sys

runs = sys.argv[1]
out = {}
for kind in ("C0", "N1", "S1"):
    tag = f"pilot200_{kind}_s0"
    lp = os.path.join(runs, tag, "train", "train_log.jsonl")
    rows = [json.loads(x) for x in open(lp, encoding="utf-8") if x.strip().startswith("{") and '"sec"' in x]
    rows = [r for r in rows if "step" in r and "sec" in r]
    rows.sort(key=lambda r: r["step"])
    d = {"n_rows": len(rows), "steps": [r["step"] for r in rows], "sec_cum": [round(r["sec"], 2) for r in rows]}
    if len(rows) >= 2:
        d["s_per_step_overall"] = rows[-1]["sec"] / rows[-1]["step"]
        d["s_per_step_steady"] = (rows[-1]["sec"] - rows[0]["sec"]) / (rows[-1]["step"] - rows[0]["step"])
        d["first_interval_s_per_step"] = rows[0]["sec"] / rows[0]["step"]
    gp = os.path.join(runs, tag + ".gpu_mib")
    if os.path.isfile(gp):
        v = [int(x) for x in open(gp).read().split() if x.strip().isdigit()]
        d["gpu_mib_peak_sampled_2s"] = max(v) if v else None
        d["gpu_mib_baseline"] = v[0] if v else None
    out[kind] = d
ss = [out[k].get("s_per_step_steady") for k in ("C0", "N1", "S1")]
out["note"] = "s_per_step_steady = (sec_last - sec_first_row)/(step_last - step_first_row), from cumulative `sec` of train_log.jsonl; peak GPU = nvidia-smi memory.used sampled every 2 s (includes other tenants)"
if all(ss):
    steady = {"C0": ss[0], "N1": ss[1], "S1": ss[2]}
    # chain: C0 x3 (s0, s1, rep), N1 x2, S1 x2; 3000 steps each; first-interval overhead added once per arm
    tot = 0.0
    for k, n in (("C0", 3), ("N1", 2), ("S1", 2)):
        tot += n * (out[k]["first_interval_s_per_step"] * 50 + steady[k] * 2950)
    out["chain_projected_h_ESTIMATED"] = tot / 3600.0
print(json.dumps(out, indent=1))
