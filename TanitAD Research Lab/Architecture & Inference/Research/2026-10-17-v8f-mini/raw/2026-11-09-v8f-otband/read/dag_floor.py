"""dag_floor.py -- the run-to-run floor of the DAG-OACC recipe at T1-kin, per band row (beside the OTBAND card, decides nothing in it).
Usage: python dag_floor.py <out.json> [n_boot]

Pair per seed s: PARITY_DAGOACCREP_s{s} (DAG-OACC's argv at training seed 10 / 11, the DAG-ORC card's REP) minus PARITY_DAGOACC_s{s},
both dumped in this read's environment (run_dumps.py, amendment 4).  Rows are read_lib's band rows (the same as read_otband.py), the
estimator taniteval.ci.paired_episode_cluster_bootstrap, B as given, seed 0.  Window identity and model-free arrays are asserted equal to
the banked DAG-OACC dump.  The floor of a row is |delta| at that seed; both seeds are reported, never pooled.
Beside: the ladder's DAgger-rows step (DAG - DAG0) and acceleration step (DAG-OACC - DAG) from out/READ_FULL.json, set against this floor.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import read_lib as BL  # noqa: E402
from read_lib import CI  # noqa: E402

OUT = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
NEW = "F:/TanitAD-scratch/otband_read/t1"
DAG = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s{s}_m10_causal_heads"
SAME = ("eid", "clip_index", "ws", "g", "fl", "v0")
t0 = time.time()


def load(p, ref=None):
    d = BL.load_dir(p)
    if ref is not None:
        for k in SAME:
            assert np.array_equal(d[k], ref[k]), f"window or model-free array differs: {k} in {p}"
    return d


def paired(a, b, eid):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-APPLICABLE"}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], eid[m], n_boot=NB, seed=0)
    return {"status": "OK", "delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def labels(X):
    """read_otband.py's labels(), unchanged: the plan's own 0-6 s manoeuvre label (lat, lon)"""
    import torch
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = BL.FF.maneuver_kinematics(t, BL.DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


full = json.load(open(os.path.join(HERE, "out/READ_FULL.json"), encoding="utf-8"))
res = {"tier": "T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene)", "evidence_class": "MEASURED (dumps of banked checkpoints, zero training)",
       "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={NB}, seed 0, 141 episodes; answers 'another draw of EPISODES'; the pair itself is one draw of 'another TRAINING run'",
       "pair": "PARITY_DAGOACCREP_s{s} (DAG-OACC argv, --seed 10 / 11) minus PARITY_DAGOACC_s{s}", "seeds": {}}
for s in (0, 1):
    banked = load(DAG.format(s=s))
    dg = load(f"{NEW}/dump_PARITY_DAGOACC_s{s}_m10_causal_heads", banked)
    dr = load(f"{NEW}/dump_PARITY_DAGOACCREP_s{s}_m10_causal_heads", banked)
    eid = dg["eid"]
    sd = {}
    for band, (a, b) in BL.BANDS.items():
        ra, rb = BL.band_rows(dr["cl"], dg["g"], a, b), BL.band_rows(dg["cl"], dg["g"], a, b)
        rows = {}
        for m in BL.METRICS:
            p = paired(ra[m], rb[m], eid)
            row = {"REP_minus_DAGOACC": p, "floor": abs(p["delta"]) if p.get("status") == "OK" else None}
            if band in full["seeds"][f"s{s}"]["bands"]:
                L = full["seeds"][f"s{s}"]["bands"][band]["rows"].get(m, {}).get("ladder_post_hoc")
                if L and row["floor"] is not None:
                    for k, nm in (("step2_dagger_rows_DAG_minus_DAG0", "step2_dagger_rows"), ("step3_w_acc_DAGOACC_minus_DAG", "step3_w_acc")):
                        x = L[k]
                        row[nm] = {"delta": x["delta"], "separated": x["separated"], "clears_dag_floor": bool(x["separated"] and abs(x["delta"]) > row["floor"])}
            rows[m] = row
        sd[band] = rows
        print(f"s{s} {band} {time.time() - t0:.0f}s", flush=True)
    gl, go = labels(dg["g"])
    (rl, ro), (bl_, bo) = labels(dr["cl"]), labels(dg["cl"])
    sd["TACTICAL_plan_reads"] = {}
    for nm, g, a, b in (("TAC-LAT", gl, rl, bl_), ("TAC-LON", go, ro, bo)):
        p = paired((a == g).astype(np.float64), (b == g).astype(np.float64), eid)
        sd["TACTICAL_plan_reads"][nm] = {"REP_minus_DAGOACC": p, "floor": abs(p["delta"]) if p.get("status") == "OK" else None,
                                         "statistic": "per-window indicator: the plan's own 0-6 s manoeuvre label equals the GT's (accuracy)"}
    res["seeds"][f"s{s}"] = sd
res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
for s in ("s0", "s1"):
    for band in ("operative_0_2s", "tactical_2_6s"):
        r = res["seeds"][s][band]
        print(s, band, {m: (round(r[m]["REP_minus_DAGOACC"]["delta"], 4), r[m]["REP_minus_DAGOACC"]["separated"]) for m in ("ade_m", "fde_m", "speed_mae_mps", "accel_mae_mps2", "yaw_rate_mae_degps")})
print("WROTE", OUT, res["wall_s"], "s")
