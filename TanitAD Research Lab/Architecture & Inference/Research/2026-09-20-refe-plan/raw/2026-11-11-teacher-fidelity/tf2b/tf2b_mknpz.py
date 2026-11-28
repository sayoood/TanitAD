#!/usr/bin/env python3
"""TF-2b step 4 (PREREG_TF2B.md): rolled t20 rows -> a p5s-format npz (logs, tokens, labels20_t20_tc20_tw, v0), with the
Tc20s column derived exactly as P5-S does (eval/p5_target_labels.iterate20, cubic, from the scenario's t0 ego).
Column 2 (Tw) is not used by tf2_score and is filled with Tc20s.  Also the derivation control: iterate20 on the BANKED t20
of the C0 control keys must reproduce the banked Tc20 label within 1e-9 m (written into the C0 json as derivation_max_dev_m).
python tf2b_mknpz.py --w <dir holding refe/ eval/ p5s/> --rows <rolled jsonl ...> --out <npz> [--c0 <c0 json>]"""
import argparse, json, os, sys
os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--w", required=True); ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--c0", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.w, "refe")); sys.path.insert(0, os.path.join(a.w, "eval"))
import navsim_dac as ND
import navtrain_scenarios as NS
import navsim_pdm_targets as PT
from p5_target_labels import iterate20
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
dbs = NS.index_dbs(db_root=DB_ROOT)


def derive(items):
    """items: {(log, tok): t20} -> {(log, tok): (tc20s, v0)}"""
    by = {}
    for (lg, tok) in items:
        by.setdefault(lg, []).append(tok)
    out = {}
    for lg in sorted(by):
        for sc in NS.build_scenarios_for_log(dbs[lg], by[lg]):
            tok = sc._initial_lidar_token
            parts = PT.metric_cache_parts(sc)
            ego = parts[1]
            ctl = {"knot_exact": True, "knot_checks": 0}
            t20 = np.asarray(items[(lg, tok)], np.float64)
            tc = iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")[0]
            out[(lg, tok)] = (tc, float(ego.dynamic_car_state.speed))
    return out


rows = {}
for p in a.rows:
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            if "traj20" in r:
                rows[(r["log"], r["token"])] = r["traj20"]
d = derive(rows)
keys = sorted(rows)
miss = [k for k in keys if k not in d]
assert not miss, miss[:3]
t20 = np.stack([np.asarray(rows[k], np.float64) for k in keys]); tc = np.stack([d[k][0] for k in keys])
np.savez(a.out, logs=np.array([k[0] for k in keys]), tokens=np.array([k[1] for k in keys]),
         labels20_t20_tc20_tw=np.stack([t20, tc, tc], 1), v0=np.array([d[k][1] for k in keys]))
print("ZZTF2B_NPZ", len(keys), a.out)
if a.c0:
    z = np.load(os.path.join(a.w, "p5s", "p5s_cubic.npz"))
    lab = {(l, t): z["labels20_t20_tc20_tw"][i] for i, (l, t) in enumerate(zip(z["logs"].tolist(), z["tokens"].tolist()))}
    rep = json.load(open(a.c0, encoding="utf-8"))
    ck = [(r["log"], r["token"]) for r in rep["rows"]]
    dd = derive({k: lab[k][0] for k in ck})
    rep["derivation_max_dev_m"] = max(float(np.abs(dd[k][0] - lab[k][1]).max()) for k in ck)
    rep["pass"] = bool(rep["pass"] and rep["derivation_max_dev_m"] <= 1e-9)
    json.dump(rep, open(a.c0, "w", encoding="utf-8"), indent=1)
    print("ZZTF2B_C0DERIV", rep["derivation_max_dev_m"], rep["pass"])
