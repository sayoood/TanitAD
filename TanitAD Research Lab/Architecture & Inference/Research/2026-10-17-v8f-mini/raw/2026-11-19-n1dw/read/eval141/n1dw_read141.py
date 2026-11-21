"""n1dw_read141.py -- STAGED, NOT RUN. The card PREREG_V8F_N1DW (sha256 59f2b7f9) section 5.2 GUARD: the four families on eval-141 for N1DW - N1R (deciding the guard) and N1DW - N1D (beside),
24 cells per seed (LONGITUDINAL, LATERAL, TACTICAL, ADE / FDE) under the STRAT1 cost / gain rule with floor F2, Z-EVAL, Z-USE beside, K-RE.  Dev box CPU, 2 threads, zero training.
Reuses navrbd_read.py's eval-141 block verbatim in logic (navrb2_read.load_arm / floor2 / d1_row / z_use, strat1_read.pair_rows / slim); only the arm list changes.
Tier T1-kin (m=10 causal bucket, dead-reckoned nav, nav_rb held from t0).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0 = 'another draw of EPISODES' only.
The guard holds at a weight iff NO cell is a cost at EITHER seed and there is no ROUTE-COST (--contrast: n1dw_strategic.py output, route_cost['N1DW<w>-N1R'] a list of RC-A cells that are a cost at BOTH seeds).
Without --contrast STRATEGIC is NOT-YET and the guard prints PENDING-STRATEGIC, never a guessed label.
N1D eval-141 dumps (for the beside rows) were consumed by the NAVRBD read and are banked in navrbd_read_outputs.tar; give --n1d-141 DIR (dump_NAVRBD-N1D_s{s}_m10_causal_heads inside) to read them, else beside rows print NOT-YET.
Usage: python n1dw_read141.py --out OUT.json [--contrast J] [--n1d-141 DIR] [--d141 DIR] [--hmax 16]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N2R = "F:/TanitAD-scratch/strat1/navrb2/read"
SR = "F:/TanitAD-scratch/strat1/read"
for _p in (HERE, N2R, SR, SR + "/theory/obd/read", SR + "/theory/otband/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WEIGHTS = (2, 4)
SEEDS = (0, 1)
CARD_SHA256_PREFIX = "59f2b7f9"
NB = 2000


def arm(w, s):
    return f"N1DW{w}_s{s}"


def guard_of(cost_cells_by_seed, route_cost):
    """pure: guard holds iff no cost cell at either seed and route_cost is False; None route_cost (no contrast) -> PENDING-STRATEGIC"""
    if any(c is None for c in cost_cells_by_seed.values()):
        return "NOT-YET"
    if any(cost_cells_by_seed.values()):
        return "GUARD-COST"
    if route_cost is None:
        return "PENDING-STRATEGIC"
    return "GUARD-HOLDS" if route_cost is False else "ROUTE-COST"


def z_eval(real, zero, mask, cells):
    """Z-EVAL (navrbd_read.z_eval verbatim): real - zero on the mask windows must be EXACTLY 0.0 on every cell, window by window (NaN equal to NaN); an empty mask fails"""
    m = np.asarray(mask, bool)
    out, ok = {}, bool(m.any())
    for c in cells:
        x, y = np.asarray(real[c], np.float64)[m], np.asarray(zero[c], np.float64)[m]
        nm = int((np.isnan(x) != np.isnan(y)).sum())
        both = np.isfinite(x) & np.isfinite(y)
        mx = float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0
        out[c] = {"max_abs_window_diff": mx, "nan_mismatch": nm}
        ok = ok and mx == 0.0 and nm == 0
    return {"pass": bool(ok), "n_windows": int(m.sum()), "must_read": "exactly 0.0 on every cell", "cells": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--d141", default=os.environ.get("N1DW_D_141", HERE + "/dumps141"))
    ap.add_argument("--n1d-141", default=os.environ.get("N1D_141", ""))
    ap.add_argument("--contrast", default=None)
    ap.add_argument("--hmax", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    t0 = time.time()
    import navrb2_lib as NL
    import navrb2_read as N2
    import strat1_lib as L
    import strat1_read as S
    import obd_read as OR
    import obd_geom as G
    CI = L.load_ci()
    rep = {"tool": "n1dw_read141.py", "card": "PREREG_V8F_N1DW.md sha256 59f2b7f9 section 5.2", "tier": "T1-kin; STRATEGIC T0", "evidence_class": "MEASURED (ours; dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B={a.n_boot} seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = floor F2", "controls": {"problems": []}, "not_yet": []}
    prob, ny = rep["controls"]["problems"], rep["not_yet"]
    geom = G.build()
    d141 = lambda x: f"{a.d141}/dump_N1DW-{x}_m10_causal_heads"  # noqa: E731
    n1r = lambda x: f"{N2R}/t1/dump_NAVRB2-{x}_m10_causal_heads"  # noqa: E731
    n1d = lambda x: f"{a.n1d_141}/dump_NAVRBD-{x}_m10_causal_heads"  # noqa: E731
    bk = lambda x: f"{SR}/t1/dump_STRAT1-{x}_m10_causal_heads"  # noqa: E731
    c0 = ["C0_s0", "C0_rep", "C0_s10", "C0_s10r", "C0_s11", "C0_s11r"]
    need = [bk(x) for x in c0 + ["N1_s0", "N1_s1"]] + [n1r("N1R_s0"), n1r("N1R_s1")]
    if any(not os.path.isdir(p) for p in need):
        ny.append("eval-141 reference dumps missing: " + str([os.path.basename(p) for p in need if not os.path.isdir(p)]))
        json.dump(rep, open(a.out, "w"), indent=1)
        print("NOT-YET reference dumps")
        sys.exit(1)
    dg = S.load_dump_n(bk("C0_s0"), 141)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    e141 = dg["eid"]
    cells, predl = {}, {}
    todo = [(x, bk(x)) for x in c0 + ["N1_s0", "N1_s1"]] + [(f"N1R_s{s}", n1r(f"N1R_s{s}")) for s in SEEDS]
    for w in WEIGHTS:
        for s in SEEDS:
            for z in ("", "Z"):
                todo.append((arm(w, s) + z, d141(arm(w, s) + z)))
    if a.n1d_141:
        for s in SEEDS:
            todo.append((f"N1D_s{s}", n1d(f"N1D_s{s}")))
    for name, d in todo:
        if os.path.isdir(d):
            cells[name], predl[name] = N2.load_arm(geom, dg, J, glat, glon, d, 141)
        else:
            ny.append(f"dump missing: {name} ({d})")
    c0rep = json.load(open(NL.C0REP_RESULT, encoding="utf-8"))
    F2 = N2.floor2(c0rep["floors"]["F"], c0rep["floors"]["F_rep"])
    rep["floors_eval141"] = {"F": c0rep["floors"]["F"], "F_rep": c0rep["floors"]["F_rep"], "F2": F2}
    b2 = json.load(open(N2R + "/out/RESULT_NAVRB2.json", encoding="utf-8"))
    bad, nre = [], 0
    for s in SEEDS:                                                     # K-RE: N1R - N1 recomputed equals RESULT_NAVRB2.json
        mine = S.slim(S.pair_rows(CI, cells[f"N1R_s{s}"], cells[f"N1_s{s}"], e141, F2, a.n_boot))
        for c, row in b2["pairs"]["N1R-N1"]["all_window_24_cells_under_F2"][str(s)].items():
            nre += 1
            for k, v in row.items():
                if k in mine.get(c, {}) and mine[c][k] != v:
                    bad.append(f"s{s} {c} {k}: {mine[c][k]!r} vs {v!r}")
    rep["controls"]["K-RE"] = {"pass": bool(nre > 0 and not bad), "n_rows_compared": nre, "mismatches": bad[:20]}
    if not rep["controls"]["K-RE"]["pass"]:
        prob.append("K-RE failed")
    w_ = NL.rb_windows()
    hmax = int(a.hmax or b2["strata"]["hmax"])
    st = NL.strata(w_, hmax)
    rb, zh = st["rb"], st["zero_hist"]
    rep["strata_141"] = {"RB": NL.count_stratum(rb, e141), "zero_hist_Z-EVAL": NL.count_stratum(zh, e141), "hmax": hmax}
    guard_cost, allw, allw_d, zev, zu = {}, {}, {}, {}, {}
    route = {}
    if a.contrast and os.path.exists(a.contrast):
        rc = json.load(open(a.contrast, encoding="utf-8")).get("route_cost", {})
        route = {w: bool(rc.get(f"N1DW{w}-N1R")) for w in WEIGHTS if f"N1DW{w}-N1R" in rc}
    else:
        ny.append("STRATEGIC contrast (ROUTE-COST); fog stays unscored")
    for w in WEIGHTS:
        for s in SEEDS:
            D, R = arm(w, s), f"N1R_s{s}"
            if D not in cells:
                guard_cost[(w, s)] = None
                continue
            allw[(w, s)] = S.pair_rows(CI, cells[D], cells[R], e141, F2, a.n_boot)
            guard_cost[(w, s)] = {c for c, v in allw[(w, s)].items() if v.get("cost")}
            if f"N1D_s{s}" in cells:
                allw_d[(w, s)] = S.pair_rows(CI, cells[D], cells[f"N1D_s{s}"], e141, F2, a.n_boot)
            rep.setdefault("D1_eval141_RB_beside", {})[f"{D}-{R}"] = N2.d1_row(CI, cells[D], cells[R], e141, rb, None, a.n_boot)
            if D + "Z" in cells:
                zev[D] = z_eval(cells[D], cells[D + "Z"], zh, list(OR.CELLS))
                zu[D] = N2.z_use(CI, cells[D], cells[D + "Z"], e141, rb, a.n_boot)
                if not zev[D]["pass"]:
                    prob.append(f"Z-EVAL {D} not exactly 0.0 on the zero-history windows")
            else:
                ny.append(f"Z-EVAL / Z-USE {D} (zero dump missing)")
    rep["controls"]["Z-EVAL"] = zev
    rep["Z_USE_eval141_RB_beside"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in zu.items()}
    rep["guard_eval141_N1DW-N1R"] = {f"{arm(w, s)}-N1R_s{s}": S.slim(v) for (w, s), v in allw.items()}
    rep["beside_eval141_N1DW-N1D"] = {f"{arm(w, s)}-N1D_s{s}": S.slim(v) for (w, s), v in allw_d.items()} if allw_d else "NOT-YET (no --n1d-141 dumps)"
    rep["cost_cells"] = {f"{arm(w, s)}": (sorted(c) if c is not None else None) for (w, s), c in guard_cost.items()}
    rep["gain_cells"] = {f"{arm(w, s)}": sorted(c for c, v in allw[(w, s)].items() if v.get("gain")) for (w, s) in allw}
    rep["guard_by_weight"] = {}
    for w in WEIGHTS:
        by_seed = {s: guard_cost.get((w, s)) for s in SEEDS}
        g = guard_of(by_seed, route.get(w))
        both = sorted(by_seed[0] & by_seed[1]) if all(x is not None for x in by_seed.values()) else None
        lon = sorted({c for s in SEEDS for c in (by_seed[s] or set()) if OR.FAMILY.get(c.split(":")[-1]) == "LONGITUDINAL"})
        rep["guard_by_weight"][f"w{w}"] = {"guard": g, "cost_cells_by_seed": {str(s): (sorted(by_seed[s]) if by_seed[s] is not None else None) for s in SEEDS}, "cost_cells_both_seeds": both,
                                          "longitudinal_cost_cells_any_seed": lon, "route_cost": route.get(w), "rule": "no cell a cost at either seed and no ROUTE-COST; RBW-COSTS needs a cost cell at BOTH seeds"}
    if prob:
        rep["status"] = "INVALID (control problems; labels unvalidated)"
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, {k: v["guard"] for k, v in rep["guard_by_weight"].items()}, "problems", len(prob), "not-yet", len(ny), rep["wall_s"], "s")
    for p in prob + ny:
        print("  ", p)
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
