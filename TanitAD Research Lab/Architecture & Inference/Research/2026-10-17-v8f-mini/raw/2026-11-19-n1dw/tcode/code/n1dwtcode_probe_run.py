#!/usr/bin/env python3
"""n1dwtcode_probe_run.py -- PREREG_N1D_TCODE (sha256 4cec3917...): probes + controls + paired bootstrap on F-TEXT, F-CODE, F-TTRUNK, F-PLAN (+ F-CODE-LAT beside).
Reuses the parent n1dwprobe_probe.py unchanged in everything that decides (fit_one = tree HGB rbdist_zt settings / standardised logistic C 1; 5-fold episode-grouped folds, fold seed 0; kin_labels;
paired/level = taniteval.ci paired episode-cluster bootstrap B 2000 seed 0). Probe seed 0 only: the parent MEASURED (TREE-SEED-IDENTITY) identical predictions between probe seeds 0 and 1.
Units = the two N1D seeds."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import n1dwprobe_probe as P  # noqa: E402
import n1dwtcode_outcome as OC  # noqa: E402

ARMS = P.ARMS
FS_MAIN = ("F-TEXT", "F-CODE", "F-TTRUNK", "F-PLAN")
FS_BESIDE = ("F-CODE-LAT",)
FS_REG = "F-CODE-ZERO"
FS_SHUF = FS_MAIN
CONTRASTS = {"CODE-TEXT": ("F-CODE", "F-TEXT"), "TTRUNK-TEXT": ("F-TTRUNK", "F-TEXT"), "PLAN-TEXT": ("F-PLAN", "F-TEXT"), "PLAN-CODE": ("F-PLAN", "F-CODE"),
             "PLAN-TTRUNK": ("F-PLAN", "F-TTRUNK"), "CODE-TTRUNK": ("F-CODE", "F-TTRUNK"), "CODELAT-TEXT": ("F-CODE-LAT", "F-TEXT"), "CODELAT-CODE": ("F-CODE-LAT", "F-CODE")}
SEED = 0
KEYS_ALL = ("F_TEXT", "F_NAV", "F_PLAN", "F_VIS", "F_CODE", "F_TTRUNK", "F_CODE_LAT", "plan0", "g", "ci", "t0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--det-feats", required=True, help="second extraction (K-DET)")
    ap.add_argument("--banked-feats", required=True, help="PROBE-N1D-FEAT feat1 dir (K-PARITY)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    CI = P.load_ci()
    paths = {arm: f"{a.feats}/feat_{arm}.npz" for arm in ARMS}
    F = {arm: P.load_feat(p) for arm, p in paths.items()}
    ref = F[ARMS[0]]
    for arm in ARMS:
        assert np.array_equal(F[arm]["ci"], ref["ci"]) and np.array_equal(F[arm]["t0"], ref["t0"]) and np.array_equal(F[arm]["g"], ref["g"]), "window lists differ between arms"
    eid, g = ref["ci"].astype(np.int64), ref["g"]
    y = P.kin_labels(g)
    fold = P.fold_ids(eid)
    N = len(y)
    shares = np.bincount(y, minlength=3) / N
    res = {"card": "PREREG_N1D_TCODE.md (sha256 4cec3917...)", "tier": "T0 decodability of frozen features, never driving performance", "smoke": bool(a.smoke),
           "evidence_class": "MEASURED (ours; banked N1D checkpoints, D1-HO grid, CPU, zero training)",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over the held-out episodes; answers another draw of EPISODES only (training variance: the two N1D seeds)",
           "n_windows": int(N), "n_episodes": int(len(np.unique(eid))), "class_counts": np.bincount(y, minlength=3).tolist(), "majority_share": float(shares.max()), "bar": OC.BAR,
           "probe_seed": SEED, "probe_seed_note": "probe seed 1 omitted: the parent measured identical predictions between probe seeds 0 and 1 (TREE-SEED-IDENTITY); units are the two N1D seeds",
           "feat_dims": {k: list(ref[k].shape[1:]) for k in ("F_TEXT", "F_PLAN", "F_CODE", "F_TTRUNK", "F_CODE_LAT")}, "gt_ok_false": int((~ref["gt_ok"]).sum()),
           "feat_sha256": {arm: P.sha256_file(p) for arm, p in paths.items()}, "folds": {"windows_per_fold": np.bincount(fold, minlength=P.NFOLD).tolist()},
           "code_lat_cols": json.load(open(paths[ARMS[0]] + ".json")).get("code_lat_names")}
    tasks = []
    for arm in ARMS:
        for fs in FS_MAIN + FS_BESIDE:
            for probe in P.PROBES:
                for k in range(P.NFOLD):
                    tasks.append((arm, fs, probe, SEED, False, k))
        for k in range(P.NFOLD):
            tasks.append((arm, FS_REG, "tree", SEED, False, k))
        for fs in FS_SHUF:
            for k in range(P.NFOLD):
                tasks.append((arm, fs, "tree", SEED, True, k))
    w = {"F-PLAN": 0, "F-CODE": 1, "F-TTRUNK": 2, "F-CODE-LAT": 3, "F-TEXT": 4, FS_REG: 5}
    tasks.sort(key=lambda t: (w[t[1]], t[2] != "tree", t[4]))
    print(f"{len(tasks)} fits, {a.workers} workers", flush=True)
    pred, done_n = {}, 0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=P._init, initargs=(paths, y, fold)) as ex:
        for task, idx, p, dt in ex.map(P.fit_one, tasks, chunksize=1):
            pred.setdefault(task[:5], np.full(N, -1, np.int64))[idx] = p
            done_n += 1
            if done_n % 10 == 0 or done_n == len(tasks):
                print(f"fits {done_n}/{len(tasks)}  {time.time() - t_all:.0f}s", flush=True)
    assert all((v >= 0).all() for v in pred.values()), "a window was never out-of-fold"
    corr = {k: (v == y).astype(np.float64) for k, v in pred.items()}
    np.savez_compressed(f"{a.out}/oof_correct.npz", **{"|".join(map(str, k)): v.astype(np.int8) for k, v in corr.items()}, y=y, eid=eid, fold=fold)
    acc = {}
    for (arm, fs, probe, seed, shuf), c in corr.items():
        acc.setdefault(arm, {}).setdefault(probe, {})[("SHUF:" if shuf else "") + fs] = {**P.level(CI, c, eid), "pred_counts": np.bincount(pred[(arm, fs, probe, seed, shuf)], minlength=3).tolist()}
    res["accuracy"] = acc
    cons = {}
    for arm in ARMS:
        for probe in P.PROBES:
            cons[f"{arm}|{probe}"] = {n: P.paired(CI, corr[(arm, A, probe, SEED, False)], corr[(arm, B, probe, SEED, False)], eid) for n, (A, B) in CONTRASTS.items()}
    res["contrasts"] = cons
    for probe in P.PROBES:
        units = {u: c for u, c in cons.items() if u.split("|")[1] == probe}
        cb = OC.combine(units)
        res["outcome_" + probe] = {"combine": cb, "headline": OC.headline(cb), "deciding": probe == "tree"}
    ctrl = {}
    ctrl["K-CONST"] = {"majority_share_live": float(shares.max()), "majority_share_card": P.MAJ_SHARE, "pass": bool(round(float(shares.max()), 6) == P.MAJ_SHARE) if not a.smoke else None}
    reg = {}
    for arm in ARMS:
        unit = f"{arm}|tree"
        code_clears = OC.above(cons[unit]["CODE-TEXT"])
        vs_code = P.paired(CI, corr[(arm, FS_REG, "tree", SEED, False)], corr[(arm, "F-CODE", "tree", SEED, False)], eid)
        vs_text = P.paired(CI, corr[(arm, FS_REG, "tree", SEED, False)], corr[(arm, "F-TEXT", "tree", SEED, False)], eid)
        used = "F-CODE" if code_clears else "F-TEXT"
        c = vs_code if code_clears else vs_text
        reg[arm] = {"F-CODE-clears-F-TEXT": bool(code_clears), "reference_used": used, "ZERO_minus_F-CODE": vs_code, "ZERO_minus_F-TEXT": vs_text, "separated_below": bool(c["separated"] and c["delta"] < 0)}
    ctrl["REGRESSION-ARM"] = {"definition": "F-CODE with all 60 columns zeroed (tree) must read separated below F-CODE, or below F-TEXT where F-CODE does not clear F-TEXT (clear = separated above 0 by >= bar)",
                              "cells": reg, "pass": bool(all(v["separated_below"] for v in reg.values()))}
    sh = {}
    for arm in ARMS:
        for fs in FS_SHUF:
            v = float(corr[(arm, fs, "tree", SEED, True)].mean())
            sh[f"{arm}|{fs}"] = {"accuracy": v, "abs_diff_from_majority_share": abs(v - P.MAJ_SHARE), "within_0.02": bool(abs(v - P.MAJ_SHARE) <= 0.02)}
    ctrl["SHUF (beside; moves no outcome)"] = {"cells": sh, "max_abs_diff": max(v["abs_diff_from_majority_share"] for v in sh.values()),
                                               "note": "parent band 0.02 FAILED there (reported in PROBE-N1D-FEAT); no SHUF-null run exists here"}
    det, par = {}, {}
    for arm in ARMS:
        Z = P.load_feat(f"{a.det_feats}/feat_{arm}.npz")
        det[arm] = {k: bool(np.array_equal(Z[k], F[arm][k])) for k in KEYS_ALL}
        B = P.load_feat(f"{a.banked_feats}/feat_{arm}.npz")
        key_b = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(B["ci"], B["t0"]))}
        ii = np.array([key_b.get((int(c), int(t)), -1) for c, t in zip(F[arm]["ci"], F[arm]["t0"])])
        ok = ii >= 0
        par[arm] = {"windows_matched": int(ok.sum()), "windows_mine": int(len(ii))}
        for k in ("F_TEXT", "F_PLAN"):
            par[arm][k + "_max_abs_diff"] = float(np.abs(F[arm][k][ok].astype(np.float64) - B[k][ii[ok]].astype(np.float64)).max()) if ok.any() else None
            par[arm][k + "_equal"] = bool(ok.any() and np.array_equal(F[arm][k][ok], B[k][ii[ok]]))
    ctrl["K-DET"] = {"cells": det, "n_windows": int(N), "pass": bool(all(all(v.values()) for v in det.values())), "second_extraction": a.det_feats}
    ctrl["K-PARITY"] = {"cells": par, "pass": bool(all(v["F_TEXT_equal"] and v["F_PLAN_equal"] and v["windows_matched"] == v["windows_mine"] for v in par.values())), "banked": a.banked_feats}
    res["controls"] = ctrl
    res["elapsed_s"] = float(time.time() - t_all)
    json.dump(res, open(f"{a.out}/RESULT_TCODE.json", "w"), indent=1)
    print("wrote", f"{a.out}/RESULT_TCODE.json", f"{res['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
