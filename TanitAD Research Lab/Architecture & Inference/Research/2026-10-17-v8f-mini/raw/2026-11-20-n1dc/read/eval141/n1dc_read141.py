"""n1dc_read141.py -- STAGED. The card PREREG_V8F_N1DC (sha256 2eae7964) section 3 GUARD: the four families on eval-141 for N1DC_s{s} - N1D_s{s}, per seed, 24 cells (LONGITUDINAL incl. distance-keeping, LATERAL, TACTICAL, ADE / FDE)
under the STRAT1 cost / gain rule with floor F2 ("no cost at either seed", as N1DW 5.2).  Adapted from n1dw_read141.py; the instrument functions (navrb2_read.load_arm / floor2 / d1_row / z_use, strat1_read.pair_rows / slim) are the same.
EXECUTOR OF RECORD: T1-kin/hb0.2 (dumps141_hb02: N1DC and N1D both re-dumped under v8f_t1_eval_hb.py --kappa-max 0.2).  T1-kin/old is beside: N1DC_s{s} old (dumps141_old) against the BANKED old N1D eval-141 dumps
(banked_n1d/x/dumps141, NAVRBD read outputs tar sha256 143f3b2a) and against N1R (the N1DW guard's comparator).  FLOOR NOTE: F2 is the banked old-executor C0 floor (c0rep); no hb0.2 replicate exists, so under hb0.2 it is
the floor of record by default and is reported as such, not re-derived.
Controls: Z-EVAL (zero twin equals follow exactly on the zero-history windows, hb0.2 N1DC and N1D), K1 (self-pair reads exactly 0 on every cell), K-RE (instrument: N1R - N1 recomputed equals RESULT_NAVRB2.json) and K-HB0-141
(read from K-HB0-141.json when present: wrapper with --kappa-max none reproduces the banked old N1D_s0 dump array for array).
The guard holds at a seed iff NO cell is a cost; GUARD-HOLDS needs no cost at EITHER seed and no ROUTE-COST (--contrast: n1dc_strategic.py output, route_cost['N1DC-N1D'] = RC-A cells that are a cost at BOTH seeds).
Without --contrast STRATEGIC is NOT-YET and the guard prints PENDING-STRATEGIC, never a guessed label.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0 = 'another draw of EPISODES' only.  Tier T1-kin (m=10 causal bucket, dead-reckoned nav, nav_rb held from t0).
Usage: python n1dc_read141.py --out OUT.json [--contrast J] [--hb DIR] [--old DIR] [--n1d-old DIR] [--hmax 16]"""
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

SEEDS = (0, 1)
CARD_SHA256_PREFIX = "2eae7964"
NB = 2000


def arm(s):
    return f"N1DC_s{s}"


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
    ap.add_argument("--hb", default=os.environ.get("N1DC_D_141_HB", HERE + "/dumps141_hb02"))
    ap.add_argument("--old", default=os.environ.get("N1DC_D_141_OLD", HERE + "/dumps141_old"))
    ap.add_argument("--n1d-old", default=os.environ.get("N1DC_N1D_OLD", HERE + "/banked_n1d/x/dumps141"))
    ap.add_argument("--khb0", default=HERE + "/K-HB0-141.json")
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
    rep = {"tool": "n1dc_read141.py", "card": "PREREG_V8F_N1DC.md sha256 2eae7964 section 3", "tier": "T1-kin/hb0.2 (executor of record); T1-kin/old beside; STRATEGIC T0", "evidence_class": "MEASURED (ours; dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B={a.n_boot} seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = the two seeds (floor F2 from the banked old-executor C0 replicate; no hb0.2 replicate exists)",
           "controls": {"problems": []}, "not_yet": []}
    prob, ny = rep["controls"]["problems"], rep["not_yet"]
    geom = G.build()
    hb = lambda x: f"{a.hb}/dump_N1DC-{x}_m10_causal_heads"  # noqa: E731
    old = lambda x: f"{a.old}/dump_N1DC-{x}_m10_causal_heads"  # noqa: E731
    n1d_old = lambda x: f"{a.n1d_old}/dump_NAVRBD-{x}_m10_causal_heads"  # noqa: E731
    n1r = lambda x: f"{N2R}/t1/dump_NAVRB2-{x}_m10_causal_heads"  # noqa: E731
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
    cells = {}
    todo = [(x, bk(x)) for x in c0 + ["N1_s0", "N1_s1"]] + [(f"N1R_s{s}", n1r(f"N1R_s{s}")) for s in SEEDS]
    for s in SEEDS:
        for z in ("", "Z"):
            todo.append((f"N1DC_s{s}{z}", hb(f"N1DC_s{s}{z}")))
            todo.append((f"N1D_s{s}{z}", hb(f"N1D_s{s}{z}")))
        todo.append((f"N1DC_s{s}~old", old(f"N1DC_s{s}")))
        todo.append((f"N1D_s{s}~old", n1d_old(f"N1D_s{s}")))
    # FIRST-RUN FIX (MM, 2026-11-20 ~19:30Z; the first run stopped here): 'fl' is the FLOOR planner's executed path, and under T1-kin/hb0.2 the
    # floor is rolled through the bounded executor too, so it legitimately differs from the old-executor reference C0_s0 (fix/check_fl.json:
    # 324 / 2,399 windows).  hb0.2 dumps therefore check 'fl' against the hb0.2 floor (taken from N1D_s0 hb0.2) and every other SAME key against
    # C0_s0 as before; old-executor dumps keep N2.load_arm unchanged.  'fl' enters no cell (cells come from 'cl' only).
    fl_hb_path = hb("N1D_s0")
    fl_hb = OR.load_dump(fl_hb_path)["fl"] if os.path.isdir(fl_hb_path) else None
    n_fl_checked = 0
    for name, d in todo:
        if os.path.isdir(d):
            if d.startswith(a.hb + "/") and fl_hb is not None:
                x = OR.load_dump(d)
                for k in OR.SAME:
                    assert np.array_equal(x[k], fl_hb if k == "fl" else dg[k]), f"window or model-free array differs from the reference dump: {k} in {d} (hb0.2 floor reference for fl)"
                A = geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"])
                cells[name] = OR.to_cells(A, glat, glon)
                n_fl_checked += 1
            else:
                cells[name], _ = N2.load_arm(geom, dg, J, glat, glon, d, 141)
        else:
            ny.append(f"dump missing: {name} ({d})")
    if fl_hb is not None:
        dfl = np.abs(fl_hb.astype(np.float64) - dg["fl"].astype(np.float64)).reshape(len(fl_hb), -1).max(1)
        rep["controls"]["K-FL-HB"] = {"pass": bool(n_fl_checked == 8), "n_hb_dumps_fl_equal_to_hb_floor": n_fl_checked, "hb_floor_source": fl_hb_path,
                                      "windows_where_hb_floor_differs_from_old_floor": int((dfl > 0).sum()), "n_windows": int(len(dfl)),
                                      "note": "first-run fix: under hb0.2 the floor path is checked against the hb0.2 floor; eid, clip_index, ws, g, v0 still against C0_s0"}
        if n_fl_checked != 8:
            prob.append(f"K-FL-HB: {n_fl_checked} of 8 hb0.2 dumps checked")
    else:
        ny.append("K-FL-HB (hb0.2 N1D_s0 dump missing)")
    c0rep = json.load(open(NL.C0REP_RESULT, encoding="utf-8"))
    F2 = N2.floor2(c0rep["floors"]["F"], c0rep["floors"]["F_rep"])
    rep["floors_eval141"] = {"F": c0rep["floors"]["F"], "F_rep": c0rep["floors"]["F_rep"], "F2": F2, "note": "banked old-executor C0 floor, used for every pair here, hb0.2 included"}
    b2 = json.load(open(N2R + "/out/RESULT_NAVRB2.json", encoding="utf-8"))
    bad, nre = [], 0
    for s in SEEDS:                                                     # K-RE: N1R - N1 recomputed equals RESULT_NAVRB2.json (the instrument reproduces the banked read)
        mine = S.slim(S.pair_rows(CI, cells[f"N1R_s{s}"], cells[f"N1_s{s}"], e141, F2, a.n_boot))
        for c, row in b2["pairs"]["N1R-N1"]["all_window_24_cells_under_F2"][str(s)].items():
            nre += 1
            for k, v in row.items():
                if k in mine.get(c, {}) and mine[c][k] != v:
                    bad.append(f"s{s} {c} {k}: {mine[c][k]!r} vs {v!r}")
    rep["controls"]["K-RE"] = {"pass": bool(nre > 0 and not bad), "n_rows_compared": nre, "mismatches": bad[:20]}
    if not rep["controls"]["K-RE"]["pass"]:
        prob.append("K-RE failed")
    if os.path.exists(a.khb0):                                         # K-HB0-141 (the smoke): wrapper with --kappa-max none equals the banked old N1D_s0 dump
        kh = json.load(open(a.khb0, encoding="utf-8"))
        rep["controls"]["K-HB0-141"] = {"pass": bool(kh.get("pass")), "source": a.khb0, "n_files": kh.get("n_files"), "n_arrays": kh.get("n_arrays")}
        if not kh.get("pass"):
            prob.append("K-HB0-141 failed")
    else:
        ny.append("K-HB0-141 (K-HB0-141.json absent)")
    w_ = NL.rb_windows()
    hmax = int(a.hmax or b2["strata"]["hmax"])
    st = NL.strata(w_, hmax)
    rb, zh = st["rb"], st["zero_hist"]
    rep["strata_141"] = {"RB": NL.count_stratum(rb, e141), "zero_hist_Z-EVAL": NL.count_stratum(zh, e141), "hmax": hmax}
    route = None
    if a.contrast and os.path.exists(a.contrast):
        rc = json.load(open(a.contrast, encoding="utf-8")).get("route_cost", {})
        if "N1DC-N1D" in rc:
            route = bool(rc["N1DC-N1D"])
    if route is None:
        ny.append("STRATEGIC contrast (ROUTE-COST); fog stays unscored")
    guard_cost, allw, beside, zev, zu = {}, {}, {}, {}, {}
    for s in SEEDS:
        D, B = arm(s), f"N1D_s{s}"
        if D not in cells or B not in cells:
            guard_cost[s] = None
            continue
        allw[s] = S.pair_rows(CI, cells[D], cells[B], e141, F2, a.n_boot)                 # DECIDING: hb0.2, N1DC - N1D
        guard_cost[s] = {c for c, v in allw[s].items() if v.get("cost")}
        if f"{D}~old" in cells and f"{B}~old" in cells:
            beside.setdefault("old_N1DC-N1D", {})[s] = S.slim(S.pair_rows(CI, cells[f"{D}~old"], cells[f"{B}~old"], e141, F2, a.n_boot))
        if f"{D}~old" in cells and f"N1R_s{s}" in cells:
            beside.setdefault("old_N1DC-N1R", {})[s] = S.slim(S.pair_rows(CI, cells[f"{D}~old"], cells[f"N1R_s{s}"], e141, F2, a.n_boot))
        if f"{D}~old" in cells:
            beside.setdefault("N1DC_hb02_minus_old", {})[s] = S.slim(S.pair_rows(CI, cells[D], cells[f"{D}~old"], e141, F2, a.n_boot))
        rep.setdefault("D1_eval141_RB_beside", {})[f"{D}-{B}"] = N2.d1_row(CI, cells[D], cells[B], e141, rb, None, a.n_boot)
        for X in (D, B):
            if X + "Z" in cells:
                zev[X] = z_eval(cells[X], cells[X + "Z"], zh, list(OR.CELLS))
                zu[X] = N2.z_use(CI, cells[X], cells[X + "Z"], e141, rb, a.n_boot)
                if not zev[X]["pass"]:
                    prob.append(f"Z-EVAL {X} not exactly 0.0 on the zero-history windows")
            else:
                ny.append(f"Z-EVAL / Z-USE {X} (zero dump missing)")
    # K1: an arm paired with itself must read exactly 0 on every cell
    k1_bad, k1_arms = {}, []
    for s in SEEDS:
        X = arm(s)
        if X in cells:
            k1_arms.append(X)
            for c, v in S.pair_rows(CI, cells[X], cells[X], e141, F2, 200).items():
                d = v.get("delta", v.get("diff", v.get("mean_diff")))
                if d is None:
                    k1_bad[f"{X}:{c}"] = "no delta key in the pair row"
                elif d != 0.0 and d == d:
                    k1_bad[f"{X}:{c}"] = d
    rep["controls"]["K1_self_pair_exactly_0"] = {"pass": bool(k1_arms) and not k1_bad, "arms": k1_arms, "nonzero": k1_bad}
    if k1_bad or not k1_arms:
        prob.append("K1 self-pair not exactly 0")
    rep["controls"]["Z-EVAL"] = zev
    rep["Z_USE_eval141_RB_beside"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in zu.items()}
    rep["guard_eval141_hb02_N1DC-N1D"] = {f"{arm(s)}-N1D_s{s}": S.slim(v) for s, v in allw.items()}
    rep["beside_eval141_old_executor"] = {k: {f"s{s}": v for s, v in d.items()} for k, d in beside.items()}
    rep["cost_cells"] = {arm(s): (sorted(c) if c is not None else None) for s, c in guard_cost.items()}
    rep["gain_cells"] = {arm(s): sorted(c for c, v in allw[s].items() if v.get("gain")) for s in allw}
    by_seed = {s: guard_cost.get(s) for s in SEEDS}
    g = guard_of(by_seed, route)
    both = sorted(by_seed[0] & by_seed[1]) if all(x is not None for x in by_seed.values()) else None
    lon = sorted({c for s in SEEDS for c in (by_seed[s] or set()) if OR.FAMILY.get(c.split(":")[-1]) == "LONGITUDINAL"})
    rep["guard"] = {"guard": g, "cost_cells_by_seed": {str(s): (sorted(by_seed[s]) if by_seed[s] is not None else None) for s in SEEDS}, "cost_cells_both_seeds": both,
                    "longitudinal_cost_cells_any_seed": lon, "route_cost": route, "rule": "no cell a cost at either seed and no ROUTE-COST (card 3 'no cost at either seed')",
                    "executor": "T1-kin/hb0.2 (record)"}
    if prob:
        rep["status"] = "INVALID (control problems; labels unvalidated)"
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, rep["guard"]["guard"], "problems", len(prob), "not-yet", len(ny), rep["wall_s"], "s")
    for p in prob + ny:
        print("  ", p)
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
