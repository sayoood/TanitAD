#!/usr/bin/env python3
"""A1 instrument-fix test (WTRANSP_DIAG.md): W-TRANSP's tracked score is harness-identical iff it is handed the harness's OWN metric-cache
inputs (trajectory, ego, observation, centerline, route lanes, drivable map) instead of the ones rebuilt from the nuPlan DB.

Scores a BANKED NON-N1 seam (the human seam of eval/raw/e0_human_sub200, W3's exported human future) through
transp_score.score_arms in two ways and compares both with the banked harness CSV refe_e0b_human_sub200.csv:
   cache  : parts = the lzma pickle in metric_cache_navtest/          -> must be within 1e-6 on every metric of every token (PASS bar)
   db     : parts = navsim_pdm_targets.metric_cache_parts(NuPlan scn) -> deliberate-regression arm: must exceed 1e-6 somewhere (else the test
                                                                         could not tell the two apart and proves nothing)
No N1 arm is scored. CPU only. Usage (navsim venv, PYTHONPATH = devkit;refe):
   python wtransp_cache_parts_check.py --pk <w/pk> --seam human_sub200.npz --banked refe_e0b_human_sub200.csv --n 60 --out result.json
"""
import argparse, csv, json, lzma, os, pickle, sys
os.environ.setdefault("REFE_SIM_HZ", "10"); os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

CACHE = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance", "ep": "ego_progress",
       "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}


def cache_parts(log, tok):
    """what navsim.evaluate.pdm_score.pdm_score reads from MetricCache, in score_arms' order"""
    mc = pickle.loads(lzma.open(f"{CACHE}/{log}/unknown/{tok}/metric_cache.pkl", "rb").read())
    return (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk", required=True); ap.add_argument("--seam", required=True); ap.add_argument("--banked", required=True)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--db-root", default="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
    a = ap.parse_args()
    sys.path[:0] = [os.path.join(a.pk, "refe"), os.path.join(a.pk, "eval", "instruments")]
    import navsim_pdm_targets as PT, navtrain_scenarios as NS, transp_score as TS
    PT._setup()
    z = np.load(a.seam, allow_pickle=False)
    toks, poses = [str(t) for t in z["token"]], z["poses"]
    banked = {r["token"]: r for r in csv.DictReader(open(a.banked, encoding="utf-8")) if r["token"] != "average"}
    import gzip
    tl = json.load(gzip.open("F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
    dbs = NS.index_dbs(db_root=a.db_root)
    sel = sorted(toks)[: a.n]
    res = {"cache": {m: 0.0 for m in COL}, "db": {m: 0.0 for m in COL}}
    over = {"cache": 0, "db": 0}; n = 0
    for tok in sel:
        log = tl[tok]["log_name"] if isinstance(tl[tok], dict) else tl[tok]
        p8 = poses[toks.index(tok)]
        sc = next(NS.build_scenarios_for_log(dbs[log], [tok], history_rows=1, future_rows=80))
        for nm, parts in (("cache", cache_parts(log, tok)), ("db", PT.metric_cache_parts(sc))):
            out, _ = TS.score_arms(parts, [p8], [])
            d = {m: abs(out[0][m] - float(banked[tok][COL[m]])) for m in COL}
            for m in COL: res[nm][m] = max(res[nm][m], d[m])
            over[nm] += max(d.values()) > 1e-6
        n += 1
    rep = {"n_tokens": n, "tol": 1e-6, "max_abs_diff_vs_banked_harness": res, "tokens_over_tol": over,
           "PASS": over["cache"] == 0, "regression_arm_discriminates": over["db"] > 0}
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep, indent=1))
    return 0 if rep["PASS"] and rep["regression_arm_discriminates"] else 1


if __name__ == "__main__":
    sys.exit(main())
