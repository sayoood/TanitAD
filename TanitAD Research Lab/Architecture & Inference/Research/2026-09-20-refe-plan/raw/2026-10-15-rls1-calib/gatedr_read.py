#!/usr/bin/env python3
"""RL + a cross-fitted TTC gate on the R side vs the control AS DEPLOYED, exactly as PREREG_O2_NEXT_GATEDR.md (sha256 pinned; written before any pick of this read).

R_k = R's raw logits under navsim_v1_lane x p_TTC^k, k in {0, 0.5, 1, 2, 4}; control = navsim_v1_lane.  Per pair and per held-out log fold (section 11's folds, seed
20261015): k = argmax PDMS delta on the other four folds among k with TTC and NC deltas >= -0.25, else the k with the largest TTC delta (ties -> smaller k).
Bar = PREREG_RL_S1.md 11.3.  Exploratory: a PASS names a lever only.

    python gatedr_read.py --out gatedr_read.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_common as CC  # noqa: E402
import calib_read as CR  # noqa: E402

RV = CC.RV
PREREG = "PREREG_O2_NEXT_GATEDR.md"
PREREG_SHA256 = "e403b92e20dcdccfd4e284676d7b12defa6345c5faadf603fa882936638380ce"
PREREG_WRITTEN_UTC = "2026-10-14T23:40:58Z"
GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
HALF = -0.25


def picks_k(z, k):
    p = RV.sigmoid(z)
    return (CC.sel_v1_lane(p) * p[..., 3] ** k).argmax(-1)


def pt(s, pick, m):
    a = CC.sub_at(s, pick)
    return {q: 100.0 * float(a[q][m].mean()) for q in ("PDMS", "TTC", "NC", "EP")}


def choose_k(s, R_by_k, pc, m, grid):
    c = pt(s, pc, m)
    rows = []
    for k in grid:
        r = pt(s, R_by_k[k], m)
        rows.append((k, r["PDMS"] - c["PDMS"], r["TTC"] - c["TTC"], r["NC"] - c["NC"]))
    ok = [x for x in rows if x[2] >= HALF and x[3] >= HALF]
    if ok:
        return max(ok, key=lambda x: (x[1], -x[0]))[0], rows, True
    return max(rows, key=lambda x: (x[2], -x[0]))[0], rows, False


def crossfit(s, keys, R_by_k, pc, fold, grid):
    out = np.empty(len(keys), np.int64)
    chosen = {}
    for f in range(CR.K_FOLDS):
        fit_m, sc_m = fold != f, fold == f
        assert not ({keys[i][0] for i in np.flatnonzero(fit_m)} & {keys[i][0] for i in np.flatnonzero(sc_m)})
        k, rows, qualified = choose_k(s, R_by_k, pc, fit_m, grid)
        out[sc_m] = R_by_k[k][sc_m]
        chosen[f"fold{f}"] = {"k": k, "qualified": qualified, "fit_rows_k_dPDMS_dTTC_dNC": rows}
    return out, chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default=CC.DL_DEFAULT)
    ap.add_argument("--x", default=CC.X_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    got = CC.sha(os.path.join(CC.HERE, PREREG))
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG} changed since registration: {got}")
    res = {"tool": "gatedr_read/1", "prereg": {"file": PREREG, "sha256": got, "written_utc": PREREG_WRITTEN_UTC}, "grid_k": GRID,
           "tier": "RL-val (selection tier; never an endpoint)", "status": "EXPLORATORY: a PASS names a lever only",
           "estimator": "paired log-cluster bootstrap, proxy_eval.boot, 10,000, seed 20260927 -- another draw of LOGS only; per-fold k held fixed"}
    res["inputs_sha256"] = CC.verify(a.dl, sorted(set(CR.DUMPS.values())) + ["data/data/rlfan_pdm_val.tar.gz", "data/data/rlfan_terms_val.tar.gz"])
    keys, tg, ck, s, st = CC.load_val(a.x)
    res["rl_val_join"], res["n_sets"] = st, len(keys)
    L = {d: CC.load_scores_aligned(os.path.join(a.dl, rel), keys, ck) for d, rel in CR.DUMPS.items()}
    raw = {d: CC.picks_from_logits(z) for d, z in L.items()}
    fold, _ = CR.folds_of(keys, CR.FOLD_SEED)
    allm = np.ones(len(keys), bool)

    # control: grid {0} reproduces the raw picks
    ctrl = {}
    for _arm, rd, t, ctl in CR.DECIDING:
        pk, _ = crossfit(s, keys, {0.0: picks_k(L[(rd, t)], 0.0)}, raw[(ctl, t)], fold, (0.0,))
        ctrl[f"{rd}_{t}_vs_{ctl}"] = int((pk != raw[(rd, t)]).sum())
    res["control_grid0_changed"] = ctrl
    if any(v != 0 for v in ctrl.values()):
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        raise SystemExit("VOID: control failed")

    dec, chosen, frontier = {}, {}, {}
    for arm, rd, t, ctl in CR.DECIDING:
        Rk = {k: picks_k(L[(rd, t)], k) for k in GRID}
        pk, ch = crossfit(s, keys, Rk, raw[(ctl, t)], fold, GRID)
        tab = CC.delta_table(s, pk, raw[(ctl, t)], keys)
        name = f"{rd}_{t}_gated_vs_{ctl}_deployed"
        dec[name] = {"arm": arm, "table": tab, "bar": CR.bar(tab)}
        chosen[name] = ch
        Ck = {k: picks_k(L[(ctl, t)], k) for k in GRID}
        frontier[f"{rd}_{t}"] = {str(k): pt(s, Rk[k], allm) for k in GRID}
        frontier[f"{ctl}_{t}"] = {str(k): pt(s, Ck[k], allm) for k in GRID}
    res["deciding"] = dec
    res["k_chosen"] = chosen
    res["frontier_points_x100_all_sets"] = frontier
    res["verdict"] = CR.verdict(dec)
    res["runtime_s"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZGATEDR_OK", res["verdict"], a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
