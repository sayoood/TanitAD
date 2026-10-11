"""G3 grid: every 3rd S6-eligible window per episode (ws order). Selected from the banked dump_s0 ws + gt_future_valid_ext
only (no outcome data). usage: python make_g3_windows.py <out.json> [n_episodes_limit]"""
import json, sys, numpy as np
BANK = "D:/refcv7_eval_kit/battery/step50400_a6text/dump_s0"
man = json.load(open(BANK + "/manifest.json", encoding="utf-8"))
out, n_all, n_el = {}, 0, 0
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
for e in man["episodes"][:lim]:
    fi = e["file_index"]
    z = np.load(f"{BANK}/ep{fi:03d}.npz"); dz = np.load(f"{BANK}/decisions/ep{fi:03d}.npz")
    keep = dz["gt_future_valid_ext"][:, 59] > 0.5
    ws = sorted(int(x) for x in z["ws"][keep]); n_all += len(z["ws"]); n_el += len(ws)
    if ws: out[e["clip_id"]] = ws[::int(__import__("os").environ.get("STRIDE", "3"))]
json.dump(out, open(sys.argv[1], "w"))
print("S2 windows", n_all, "S6-eligible", n_el, "G3 windows", sum(map(len, out.values())), "episodes", len(out))
