"""Runs collapse_monitor over the banked ladder arms (R35H s0/s1 first, then C0, R35, S35, R35b) and writes RESULT_COLLAPSE_R35H.json.  Env: see README.md."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collapse_monitor as CM  # noqa: E402

PULLS = "C:/Users/Admin/t2v8f_pulls"
FRAMES = "F:/TanitAD-scratch/ladder_read/cache"
ARMS = ["R35H_s0", "R35H_s1", "C0_s0", "C0_s1", "R35_s0", "R35_s1", "S35_s0", "S35_s1", "R35b_s0"]

if __name__ == "__main__":
    sc = CM.selfcheck()
    assert sc["ok"], sc
    out = {"tool": "collapse_monitor.py", "evidence_class": "MEASURED", "tier": "T0 latent diagnostic", "planted_selfcheck": sc, "arms": {}, "errors": {}}
    for a in ARMS:
        seed = int(a.rsplit("_s", 1)[1])
        run = f"{PULLS}/ladder_{a}"
        if not os.path.exists(f"{run}/ckpt_last.pt"):
            out["errors"][a] = "no ckpt_last.pt"
            continue
        try:
            out["arms"][a] = CM.monitor(run, f"{FRAMES}/frames_s{seed}.npz")
            for l, s in out["arms"][a]["levels"].items():
                print(a, "L" + l, f"{s['step_seconds']}s dim={s['dim']} PR={s['pr']:.2f} PR/dim={s['pr_over_dim']:.4f} 90%={s['dims_for_90pct']} 99%={s['dims_for_99pct']} dead={s['dead_dims']} {s['reading']}", flush=True)
        except Exception as e:  # recorded, not hidden
            out["errors"][a] = f"{type(e).__name__}: {e}"
            print(a, "ERROR", out["errors"][a], flush=True)
        with open(f"{HERE}/RESULT_COLLAPSE_R35H.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
