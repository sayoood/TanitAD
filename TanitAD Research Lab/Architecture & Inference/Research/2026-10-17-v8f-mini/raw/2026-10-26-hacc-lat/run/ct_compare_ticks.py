"""C-T / C-T2 comparison for ct_smoke.sh: argv[1] = the smoke dir. The --dump-ticks run's dumps and JSON equal the original tool's (but for the header's extra ticks_file key), and the tick-0 plans equal the dump's `ol`."""
import json
import os
import sys

import numpy as np

W = sys.argv[1]
da, db = W + "/orig/dump_CT", W + "/new_ticks/dump_CT"
bad = []
for f in sorted(os.listdir(da)):
    a, b = np.load(da + "/" + f), np.load(db + "/" + f)
    if not (set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a.files)):
        bad.append(f)
ja, jb = json.load(open(W + "/orig/t1v8f_CT.json")), json.load(open(W + "/new_ticks/t1v8f_CT.json"))
tf = jb["t1v8f"].pop("ticks_file")
z = np.load(W + "/ticks/ticks_CT.npz")
key = {(int(c), int(t)): i for i, (c, t, k) in enumerate(zip(z["ci"], z["t0"], z["tick"])) if k == 0}
worst, n = 0.0, 0
for f in sorted(os.listdir(da)):
    a = np.load(da + "/" + f)
    for ci, ws, ol in zip(a["clip_index"], a["ws"], a["ol"]):
        i = key[(int(ci), int(ws))]
        worst = max(worst, float(np.abs(z["plan"][i] - ol).max()))
        n += 1
res = {"label": "C-T (dump-ticks run) + C-T2 on eval windows", "dumps_identical_to_original": not bad, "differing_dump_files": bad, "json_identical_except_ticks_file": ja == jb,
       "tick0_plan_vs_dump_ol_max_abs_m": worst, "n_windows_compared": n, "n_tick_samples": int(len(z["plan"])), "n_gt_ok": int(z["gt_ok"].sum()), "ticks_file": tf,
       "pass": bool(not bad and ja == jb and worst == 0.0 and n == int((z["tick"] == 0).sum()))}
json.dump(res, open(W + "/C-T_ticks.json", "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if res["pass"] else 2)
