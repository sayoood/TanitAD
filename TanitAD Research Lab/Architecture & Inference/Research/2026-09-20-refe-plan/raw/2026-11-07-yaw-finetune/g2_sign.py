"""G2 sign gate (PREREG_YAWFT section 4): on the TRAIN rows (v > 2) corr(table omega, ay/v) >= 0.85 and corr(omega, target yaw @ 4 s) > 0. Reads any cache's rows.jsonl (ego / traj come from the bank,
identical across caches of the same frames) + the manifest's TRAIN split + the pinned table. CPU, seconds.
    python g2_sign.py --cache <dir> [--out g2_sign.json]      # rc 0 = green, 3 = RED;  --negate = the deliberate-regression arm (must be RED)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_common as YC

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--manifest", default=str(HERE / "manifest_yaw_4k.json"))
ap.add_argument("--table", default=str(YC.TABLE_NAVTRAIN))
ap.add_argument("--negate", action="store_true", help="deliberate-regression arm: negate omega")
ap.add_argument("--out", default=None)
a = ap.parse_args()
table, _, tsha = YC.load_yaw_table(a.table)
man = json.load(open(a.manifest, encoding="utf-8"))
tr = {(f["log_name"], f["token"], int(f["step"])) for f in man["frames"] if f["yf2_split"] == "TRAIN"}
w, ayv, y4, seen = [], [], [], set()
for l in open(Path(a.cache) / "rows.jsonl", encoding="utf-8"):
    r = json.loads(l)
    k = (r["key"][0], r["key"][1], int(r["key"][2]))
    if k not in tr or k in seen:
        continue
    seen.add(k)
    v = r["ego"][0]
    if v > 2.0:
        w.append(table[k[1]] * (-1.0 if a.negate else 1.0))
        ayv.append(r["ego"][3] / v)
        y4.append(r["traj"][19][2])
w, ayv, y4 = map(np.array, (w, ayv, y4))
c1, c2 = float(np.corrcoef(w, ayv)[0, 1]), float(np.corrcoef(w, y4)[0, 1])
res = {"n_frames_v_gt_2": int(len(w)), "n_train_frames": len(seen), "corr_omega_ay_over_v": c1, "corr_omega_target_yaw_4s": c2, "table_sha256": tsha, "negated": a.negate,
       "pass": bool(c1 >= 0.85 and c2 > 0)}
print(json.dumps(res, indent=1))
if a.out:
    YC.jdump(res, a.out)
sys.exit(0 if res["pass"] else 3)
