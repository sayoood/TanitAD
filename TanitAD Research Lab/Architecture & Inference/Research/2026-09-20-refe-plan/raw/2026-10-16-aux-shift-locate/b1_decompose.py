#!/usr/bin/env python3
"""H-b / H-c: DECOMPOSE the navtrain-held-out -> navtest gap of the aux head's P1 IoU [0, 40) m.  CPU, read-only, DESCRIPTIVE.

Inputs : held-out head_pred dumps (trainer, 24 whole logs, 1,705 frames) and the REGENERATED navtest head_pred (1,058 C3 tokens) of H1-s0/s1/s2.
Method : every number is a ratio of sums over frames, exactly RO.frame_stats' P1 definition.  Re-weighting = frame weights in BOTH numerator and denominator.
         A re-weighting moves the point estimate toward what the OTHER domain's composition would read; it is a direct standardisation, not a causal model,
         and the strata are not independent of one another (so the explained shares do NOT add up).
Sections: R0 reproduction | S1 bands | S2 confusion | S3 spatial shift | S4 maps | S5 natural-population + map standardisation | S6 navtest strata |
          S7 speed | S8 label-difficulty strata | S9 error depth | S10 independent-bootstrap gap | S11 layout-prior baseline
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C  # noqa: E402
import shf_arrays as SA  # noqa: E402

ARMS = tuple(os.environ.get("SHF_ARMS", "H1-s0,H1-s1,H1-s2").split(","))
BANKED_NAVTEST = {"H1-s0": 0.4957035913034702, "H1-s1": 0.5210982139151368, "H1-s2": 0.4863984780737079}
RES = {}
MAPS = ["LV", "BOS", "PIT", "SG"]


def part(D, a, lo=0.0, hi=40.0):
    i, u, _ = SA.frame_iou_parts(D, a, lo, hi)
    return i, u


def iou(i, u, w=None, m=None):
    m = np.ones(len(i), bool) if m is None else m
    w = np.ones(len(i)) if w is None else w
    d = (w[m] * u[m]).sum()
    return float((w[m] * i[m]).sum() / d) if d > 0 else float("nan")


def rake(maps, strata, margins_map, margins_stratum, iters=200):
    """iterative proportional fitting of frame weights to a map margin and a stratum margin (shares sum to 1 each)."""
    w = np.ones(len(maps))
    for _ in range(iters):
        for lab, marg in ((maps, margins_map), (strata, margins_stratum)):
            if marg is None:
                continue
            tot = w.sum()
            for k, share in marg.items():
                m = lab == k
                s = w[m].sum()
                if s > 0:
                    w[m] *= share * tot / s
    return w / w.mean()


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    Dh, Dn = SA.domain("heldout", ARMS), SA.domain("navtest", ARMS)
    # ---------------- R0 reproduction of the banked navtest numbers
    R0 = {}
    for a in ARMS:
        i, u = part(Dn, a)
        R0[a] = dict(mine=iou(i, u), banked=BANKED_NAVTEST[a], abs_diff=abs(iou(i, u) - BANKED_NAVTEST[a]))
    RES["R0_navtest_repro"] = R0
    print("R0", R0, flush=True)
    P = {"ho": {a: part(Dh, a) for a in ARMS}, "nt": {a: part(Dn, a) for a in ARMS}}

    # ---------------- S1 bands (the gap is uniform across range?)
    S1 = {}
    for (lo, hi, nm) in ((0, 20, "0-20"), (20, 40, "20-40"), (40, 72, "40-72"), (0, 40, "0-40")):
        S1[nm] = {}
        for a in ARMS:
            ih, uh = part(Dh, a, lo, hi)
            inn, un = part(Dn, a, lo, hi)
            S1[nm][a] = dict(heldout=iou(ih, uh), navtest=iou(inn, un), gap=iou(ih, uh) - iou(inn, un))
    RES["S1_bands"] = S1

    # ---------------- S2 confusion (bias vs. error)
    b = SA.band()[None]
    S2 = {}
    for nm, D in (("heldout", Dh), ("navtest", Dn)):
        S2[nm] = {}
        for a in ARMS:
            tm, pm = D["TM"] & b, D["PM"][a] & b
            tp, fp, fn = (tm & pm).sum(), (~D["TM"] & pm & b).sum(), (tm & ~pm).sum()
            tot = b.sum() * D["n"]
            S2[nm][a] = dict(true_drivable_share=float(tm.sum() / tot), pred_drivable_share=float(pm.sum() / tot), precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                             fp_share_of_cells=float(fp / tot), fn_share_of_cells=float(fn / tot), accuracy=float(1 - (fp + fn) / tot))
    RES["S2_confusion_0_40"] = S2

    # ---------------- S3 spatial shift of the PREDICTION relative to the label (cells of 2 m); constant cell set across shifts
    S3 = {}
    x0, x1 = 4, 24                       # band [0, 40) m
    y0, y1 = 4, 28                       # lateral margin of 4 cells so that every shift in [-4, 4] stays inside the grid
    for nm, D in (("heldout", Dh), ("navtest", Dn)):
        S3[nm] = {}
        for a in ARMS:
            TMc = D["TM"][:, x0:x1, y0:y1]
            grid = {}
            for sx in range(-4, 5):
                for sy in range(-4, 5):
                    PMs = D["PM"][a][:, x0 + sx:x1 + sx, y0 + sy:y1 + sy]
                    grid[f"{sx},{sy}"] = float((TMc & PMs).sum() / (TMc | PMs).sum())
            best = max(grid, key=grid.get)
            # mirror / transposition controls that must be much worse (they are the deliberate-regression arms of the label gate)
            PMf = D["PM"][a][:, x0:x1, y0:y1][:, :, ::-1]
            S3[nm][a] = dict(zero_shift=grid["0,0"], best_shift=best, best=grid[best], x_only=[grid[f"{s},0"] for s in range(-4, 5)], y_only=[grid[f"0,{s}"] for s in range(-4, 5)],
                             y_flipped=float((TMc & PMf).sum() / (TMc | PMf).sum()))
    RES["S3_shift"] = S3

    # ---------------- S4 maps
    S4 = {}
    for nm, D in (("heldout", Dh), ("navtest", Dn)):
        S4[nm] = {}
        for m in MAPS:
            k = D["maps"] == m
            S4[nm][m] = dict(n=int(k.sum()), logs=int(len(set(D["logs"][k]))), share=float(k.mean()),
                             **{a: iou(*P["ho" if nm == "heldout" else "nt"][a], m=k) for a in ARMS},
                             true_drivable_share=float((D["TM"] & b)[k].sum() / (b.sum() * max(k.sum(), 1))))
    RES["S4_maps"] = S4

    # shares used as targets
    feats = np.load(os.path.join(C.SCRATCH, "label_feats.npz"), allow_pickle=False)
    tr_maps = feats["train_maps"]
    nt_maps_all = feats["navtest_maps"]
    short = {"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"}
    share_train = {short[m]: float((tr_maps == m).mean()) for m in short}
    share_navtest_pop = {short[m]: float((nt_maps_all == m).mean()) for m in short}
    held = set(C.heldout_logs())
    tr_keys = feats["train_keys"]
    ho_mask = np.array([k.split("|")[0] in held for k in tr_keys])
    share_ho_pop = {short[m]: float((tr_maps[ho_mask] == m).mean()) for m in short}
    c3 = C.c3()
    n_zero_pop, n_pop = c3["provenance"]["n_zero_NCxDAC"], c3["provenance"]["n_csv_tokens"]
    strat_pop = {"zero": n_zero_pop / n_pop, "random": 1 - n_zero_pop / n_pop}
    RES["targets"] = dict(share_train_all=share_train, share_navtest_population=share_navtest_pop, share_heldout_logs_population=share_ho_pop, stratum_population=strat_pop,
                          sample_map_share_heldout={m: float((Dh["maps"] == m).mean()) for m in MAPS}, sample_map_share_navtest_c3={m: float((Dn["maps"] == m).mean()) for m in MAPS})

    # ---------------- S5 standardisation
    S5 = {}
    for a in ARMS:
        ih, uh = P["ho"][a]
        inn, un = P["nt"][a]
        base_h, base_n = iou(ih, uh), iou(inn, un)
        row = dict(heldout_raw=base_h, navtest_raw=base_n, gap_raw=base_h - base_n)
        # (a) navtest -> its own natural population (stratum + map raking)
        w_nat = rake(Dn["maps"], Dn["stratum"], share_navtest_pop, strat_pop)
        row["navtest_natural_population"] = iou(inn, un, w_nat)
        row["ess_natural"] = ess(w_nat)
        # (b) navtest sample: map mix only -> held-out map mix / train map mix
        for tag, tgt in (("heldout_map_mix", share_ho_pop), ("train_map_mix", share_train)):
            w = rake(Dn["maps"], Dn["stratum"], tgt, None)
            row[f"navtest_sample_to_{tag}"] = iou(inn, un, w)
            # (c) natural stratum AND the target map mix
            w2 = rake(Dn["maps"], Dn["stratum"], tgt, strat_pop)
            row[f"navtest_natural_stratum_and_{tag}"] = iou(inn, un, w2)
            row[f"ess_natural_stratum_and_{tag}"] = ess(w2)
        # (d) stratum only (failure enrichment alone)
        w3 = rake(Dn["maps"], Dn["stratum"], None, strat_pop)
        row["navtest_sample_to_natural_stratum_only"] = iou(inn, un, w3)
        # (e) the reverse direction: held-out -> navtest-population map mix / C3 sample map mix
        for tag, tgt in (("navtest_population_map_mix", share_navtest_pop), ("navtest_c3_sample_map_mix", RES["targets"]["sample_map_share_navtest_c3"])):
            w = rake(Dh["maps"], Dh["stratum"], tgt, None)
            row[f"heldout_to_{tag}"] = iou(ih, uh, w)
            row[f"ess_heldout_to_{tag}"] = ess(w)
        S5[a] = row
    RES["S5_standardisation"] = S5

    # ---------------- S6 navtest strata (zero / random; DAC=0 from the u400 csv)
    prov = c3["provenance"]
    dac0 = set()
    nc0 = set()
    with open(prov["csv"], encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["token"] in ("average", ""):
                continue
            if float(r["drivable_area_compliance"]) == 0.0:
                dac0.add(r["token"])
            if float(r["no_at_fault_collisions"]) == 0.0:
                nc0.add(r["token"])
    d0 = np.array([t in dac0 for t in Dn["tokens"]])
    n0 = np.array([t in nc0 for t in Dn["tokens"]])
    S6 = {}
    for a in ARMS:
        inn, un = P["nt"][a]
        S6[a] = {"all": iou(inn, un), "stratum_zero": iou(inn, un, m=Dn["stratum"] == "zero"), "stratum_random": iou(inn, un, m=Dn["stratum"] == "random"),
                 "dac0": iou(inn, un, m=d0), "dac_ok": iou(inn, un, m=~d0), "nc0": iou(inn, un, m=n0), "nc_ok": iou(inn, un, m=~n0),
                 "n_zero": int((Dn["stratum"] == "zero").sum()), "n_random": int((Dn["stratum"] == "random").sum()), "n_dac0": int(d0.sum()), "n_nc0": int(n0.sum())}
    RES["S6_navtest_strata"] = S6
    # per-map within stratum (is the zero/random difference a city effect?)
    S6b = {}
    for s in ("zero", "random"):
        S6b[s] = {m: dict(n=int(((Dn["stratum"] == s) & (Dn["maps"] == m)).sum()), **{a: iou(*P["nt"][a], m=(Dn["stratum"] == s) & (Dn["maps"] == m)) for a in ARMS[:1]}) for m in MAPS}
    RES["S6b_stratum_by_map_H1s0"] = S6b

    # ---------------- S7 speed
    edges = [0, 0.5, 3.0, 8.0, 14.0, 100.0]
    S7 = {}
    for nm, D, pk in (("heldout", Dh, "ho"), ("navtest", Dn, "nt")):
        sp = D["speed"]
        S7[nm] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (sp >= lo) & (sp < hi)
            S7[nm][f"{lo}-{hi}"] = dict(n=int(k.sum()), share=float(k.mean()), **{a: iou(*P[pk][a], m=k) for a in ARMS[:1]})
    RES["S7_speed_H1s0"] = S7
    # standardise navtest to the held-out speed mix and vice versa (frame weights by speed bin)
    def bins(sp):
        return np.digitize(sp, edges[1:-1])
    for a in ARMS[:1]:
        bh, bn = bins(Dh["speed"]), bins(Dn["speed"])
        sh = {k: float((bh == k).mean()) for k in range(5)}
        w = np.ones(len(bn))
        for k in range(5):
            m = bn == k
            if m.any():
                w[m] = sh[k] / max((bn == k).mean(), 1e-9)
        RES["S7_navtest_to_heldout_speed_mix_H1s0"] = iou(*P["nt"][a], w=w)

    # ---------------- S8 label-difficulty strata: share of drivable cells in the band (from the label alone)
    tm_share = lambda D: ((D["TM"] & b).sum((1, 2)) / b.sum()).astype(np.float64)
    fh, fn_ = tm_share(Dh), tm_share(Dn)
    qedges = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.01]
    S8 = {}
    for nm, D, f, pk in (("heldout", Dh, fh, "ho"), ("navtest", Dn, fn_, "nt")):
        S8[nm] = {}
        for lo, hi in zip(qedges[:-1], qedges[1:]):
            k = (f >= lo) & (f < hi)
            S8[nm][f"{lo}-{hi}"] = dict(n=int(k.sum()), share=float(k.mean()), **{a: iou(*P[pk][a], m=k) for a in ARMS[:1]})
    RES["S8_true_drivable_share_strata_H1s0"] = S8
    def bins8(f):
        return np.digitize(f, qedges[1:-1])
    a = ARMS[0]
    bh, bn = bins8(fh), bins8(fn_)
    w = np.ones(len(bn))
    for k in range(6):
        m = bn == k
        if m.any():
            w[m] = float((bh == k).mean()) / max(float(m.mean()), 1e-9)
    RES["S8_navtest_to_heldout_drivable_share_mix_H1s0"] = iou(*P["nt"][a], w=w)
    # joint: map x drivable-share bin
    keyh = np.array([f"{m}|{k}" for m, k in zip(Dh["maps"], bh)])
    keyn = np.array([f"{m}|{k}" for m, k in zip(Dn["maps"], bn)])
    wj = np.ones(len(keyn))
    for kk in set(keyn):
        m = keyn == kk
        ph = float((keyh == kk).mean())
        wj[m] = ph / m.mean()
    RES["S8_navtest_to_heldout_map_x_drivable_share_mix_H1s0"] = dict(iou=iou(*P["nt"][a], w=wj), ess=ess(wj), cells_missing_in_heldout=int(sum(1 for kk in set(keyn) if not (keyh == kk).any())))

    # ---------------- S9 error depth (true block-mean SDF, clipped at +-6 m): gross errors vs boundary errors
    S9 = {}
    depth_edges = [-6.01, -3.0, 0.0, 3.0, 6.01]
    for nm, D in (("heldout", Dh), ("navtest", Dn)):
        sd = D["SDFT"]
        S9[nm] = {}
        for a in ARMS[:1]:
            wrong = (D["PM"][a] != D["TM"]) & b
            S9[nm][a] = {}
            for lo, hi in zip(depth_edges[:-1], depth_edges[1:]):
                k = (sd >= lo) & (sd < hi) & b
                S9[nm][a][f"sdf[{lo:+.0f},{hi:+.0f})"] = dict(cell_share=float(k.sum() / (b.sum() * D["n"])), error_rate=float((wrong & k).sum() / max(k.sum(), 1)),
                                                              error_share_of_all_errors=float((wrong & k).sum() / max(wrong.sum(), 1)))
    RES["S9_error_depth"] = S9

    # ---------------- S10 independent log-cluster bootstrap of the gap (different log sets in the two domains)
    rng = np.random.default_rng(20261016)
    BOOT = 5000
    def by_log(D, i, u):
        ul = sorted(set(D["logs"]))
        pos = {l: j for j, l in enumerate(ul)}
        inv = np.array([pos[l] for l in D["logs"]])
        return len(ul), np.bincount(inv, weights=i, minlength=len(ul)), np.bincount(inv, weights=u, minlength=len(ul))
    S10 = {}
    for a in ARMS:
        gh, ih_l, uh_l = by_log(Dh, *P["ho"][a])
        gn, in_l, un_l = by_log(Dn, *P["nt"][a])
        reps = np.empty(BOOT)
        for r in range(BOOT):
            sh = rng.integers(0, gh, gh)
            sn = rng.integers(0, gn, gn)
            reps[r] = ih_l[sh].sum() / uh_l[sh].sum() - in_l[sn].sum() / un_l[sn].sum()
        est = ih_l.sum() / uh_l.sum() - in_l.sum() / un_l.sum()
        S10[a] = dict(gap=float(est), ci95=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], n_logs_heldout=gh, n_logs_navtest=gn, B=BOOT,
                      estimator="independent log-cluster bootstrap (the two domains have disjoint log sets); answers 'another draw of LOGS' only")
    RES["S10_gap_ci"] = S10

    # per-log IoU spread
    S10b = {}
    for nm, D, pk in (("heldout", Dh, "ho"), ("navtest", Dn, "nt")):
        g, il, ul_ = by_log(D, *P[pk][ARMS[0]])
        v = il / np.maximum(ul_, 1)
        S10b[nm] = dict(n_logs=g, per_log_iou_mean=float(v.mean()), p10=float(np.percentile(v, 10)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), min=float(v.min()), max=float(v.max()))
    RES["S10b_per_log_iou_H1s0"] = S10b

    # ---------------- S11 layout-prior baseline (no image): per-map, per-cell drivable frequency from the TRAIN store's NON-held-out keys, thresholded at 0.5
    src = C.open_train_labels()
    keys_all = [tuple(k.split("|")) for k in tr_keys]
    rng2 = np.random.default_rng(7)
    idx = rng2.choice(len(keys_all), 12000, replace=False)                          # a 12k subsample of the training frames is plenty for a per-cell frequency
    sel = [keys_all[j] for j in idx if keys_all[j][0] not in held]
    lab, valid = src.labels(sel)
    t = C.AUX.build_targets(lab, valid)
    prior_m = t.mask_frac.numpy() >= 0.5
    sel_maps = np.array([short[m] for m in tr_maps[idx][[keys_all[j][0] not in held for j in idx]]])
    S11 = {"n_prior_frames": len(sel)}
    for tag, pooled in (("single_global_prior", True), ("per_map_prior", False)):
        pri = {}
        if pooled:
            f = prior_m.mean(0)
            for m in MAPS:
                pri[m] = f >= 0.5
        else:
            for m in MAPS:
                k = sel_maps == m
                pri[m] = (prior_m[k].mean(0) >= 0.5) if k.sum() else (prior_m.mean(0) >= 0.5)
        res_ = {}
        for nm, D in (("heldout", Dh), ("navtest", Dn)):
            pm = np.stack([pri[m] for m in D["maps"]])
            i_ = (D["TM"] & pm & b).sum((1, 2)).astype(float)
            u_ = ((D["TM"] | pm) & b).sum((1, 2)).astype(float)
            res_[nm] = iou(i_, u_)
        S11[tag] = res_
    RES["S11_layout_prior_baseline"] = S11

    C.jdump(RES, os.path.join(C.HERE, os.environ.get("SHF_OUT", "b1_decompose.json")))
    print("ZZB1_DONE")


if __name__ == "__main__":
    main()
