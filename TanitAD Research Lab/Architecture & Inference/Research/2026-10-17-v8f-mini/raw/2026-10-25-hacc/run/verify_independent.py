#!/usr/bin/env python3
"""verify_independent.py -- a post-hoc INDEPENDENT recomputation of the HACC read's headline numbers (descriptive; it changes no row, grid, rule or bar).
Checks, per seed:  (1) w* is reproduced from the TRAIN windows alone at lambda* (fit_w again) and equals the banked weights file;  (2) the banked HACC plan equals P + w*(tau - P)
recomputed from HBLEND's saved eval arrays;  (3) per-window accel MAE of HACC / P10 from the float32 instrument equals the banked per-window arrays, and the paired mean difference with an
OWN episode-cluster bootstrap (numpy only, B = 2000) agrees with the banked interval;  (4) the same for curvature through the eval's family function.
Writes verify_independent.json.  Importing the eval world is read-only."""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import v8f_hacc as A  # noqa: E402

H = A.H


def boot(diff, eid, B=2000, seed=1):
    eids = np.unique(eid)
    idx = {e: np.where(eid == e)[0] for e in eids}
    s = np.array([diff[idx[e]].sum() for e in eids])
    n = np.array([len(idx[e]) for e in eids])
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        k = rng.integers(0, len(eids), len(eids))
        out[b] = s[k].sum() / n[k].sum()
    return float(diff.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    res = {}
    grid, J, X = H.eval_world()
    gt, eid = X["gt"].astype(np.float64), X["eid"]
    for seed in A.SEEDS:
        F = json.load(open(f"{HERE}/fit_{seed}.json"))
        Wf = json.load(open(f"{HERE}/weights_HACC_{seed}.json"))
        T = A.train_bundle(seed)
        f = A.fit_w(T.tau, T.P, T.y, F["lam_star"])
        w_banked = np.stack([Wf["w_along_by_step"], Wf["w_cross_by_step"]], 1)
        sv = np.load(f"{H.OUT}/score_{seed}_perwindow.npz")
        sc = np.load(f"{HERE}/score_{seed}_perwindow.npz")
        plan = H.apply_w(sv["P"].astype(np.float64), sv["tau"].astype(np.float64), w_banked)
        ag = A.AP.accel(gt)
        e_h = np.abs(A.AP.accel(sc["HACC"]) - ag).mean(1).astype(np.float64)
        e_p = np.abs(A.AP.accel(sv["P_file"]) - ag).mean(1).astype(np.float64)
        BE = A._be()
        fh, fp = BE.family_windows(sc["HACC"].astype(np.float64), gt, 0.2), BE.family_windows(sv["P_file"].astype(np.float64), gt, 0.2)
        ok = np.isfinite(fh["lat_curvature_mae_1pm"]) & np.isfinite(fp["lat_curvature_mae_1pm"])
        sj = json.load(open(f"{HERE}/score_{seed}.json"))["paired"]["HACC_minus_P10"]
        res[seed] = {
            "w_refit_on_TRAIN_only_vs_banked_weights_maxabs": float(np.abs(f["w"] - w_banked).max()),
            "banked_HACC_plan_vs_P_plus_w_tau_minus_P_maxabs_m": float(np.abs(plan - sc["HACC"]).max()),
            "accel_instrument_vs_banked_perwindow_HACC_maxabs": float(np.abs(e_h - sc["HACC__fw_lon_accel_mae"]).max()),
            "own_bootstrap_accel_HACC_minus_P10": dict(zip(("mean", "lo", "hi"), boot(e_h - e_p, eid))),
            "banked_accel_HACC_minus_P10": {k: sj["fw_lon_accel_mae"][k] for k in ("mean", "lo", "hi")},
            "own_bootstrap_curvature_HACC_minus_P10": dict(zip(("mean", "lo", "hi"), boot((fh["lat_curvature_mae_1pm"] - fp["lat_curvature_mae_1pm"])[ok], eid[ok]))),
            "banked_curvature_HACC_minus_P10": {k: sj["fw_lat_curvature_mae_1pm"][k] for k in ("mean", "lo", "hi")},
            "n_curvature_windows": int(ok.sum()),
            "w_star_equals_fit_json": float(np.abs(np.asarray(F["w_star"]) - w_banked).max()),
        }
        print(seed, json.dumps(res[seed], default=float), flush=True)
    H.jdump(res, f"{HERE}/verify_independent.json")


if __name__ == "__main__":
    main()
