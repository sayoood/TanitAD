#!/usr/bin/env python3
"""agreement.py -- Task 2 of the aux-raster veto feasibility package: does geoDAC on the GT raster reproduce NAVSIM's DAC sub-score?

UNIT: the PROPOSAL, pooled over ALL 64 proposals of ALL 1,058 C3 tokens (67,712). Never restricted to picks. No pick, no PDMS delta, no veto outcome.
Positive class = VIOLATION (NAVSIM DAC = 0).  Estimator: log-cluster bootstrap over the 123 logs, 2,000 draws, seed 20260927 (answers "another draw of LOGS";
the raster/tracker are deterministic, so there is no training or inference seed in this number).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I agreement.py [--tracked C:/Users/Admin/tmpauxv/tracked]

Primary instrument (declared before the number was seen): GT raster, RAW proposal poses densified to the 41-step 0.1 s grid, bilinear signed-field sign,
unknown = "drivable".  Secondary: nearest cell.  Diagnostics (tracked states / exact polygons) are labelled DIAGNOSTIC and decide nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agreement_core as AC  # noqa: E402
import aux_label_store as A  # noqa: E402
import geodac as G  # noqa: E402

PKTOP = HERE.parent.parent
B_BOOT, SEED = 2000, 20260927


def cj(o):
    if isinstance(o, dict):
        return {k: cj(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [cj(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return cj(o.tolist())
    return o


def boot(geo_fail, nav_fail, log_ids, B=B_BOOT, seed=SEED):
    """Log-cluster bootstrap of agreement and kappa.  geo_fail/nav_fail [N tokens, 64]; log_ids [N]."""
    logs = sorted(set(log_ids))
    idx_by = {lg: np.array([i for i, l in enumerate(log_ids) if l == lg]) for lg in logs}
    rng = np.random.default_rng(seed)
    g, n = geo_fail.astype(bool), nav_fail.astype(bool)
    tp_l = {lg: np.array([(g[i] & n[i]).sum(), (g[i] & ~n[i]).sum(), (~g[i] & n[i]).sum(), (~g[i] & ~n[i]).sum()]) for lg, i in idx_by.items()}
    M = np.stack([tp_l[lg] for lg in logs])                                   # [L, 4]
    ag, kp = [], []
    for _ in range(B):
        s = M[rng.integers(0, len(logs), len(logs))].sum(0)
        tp, fp, fn, tn = s
        N = s.sum()
        po = (tp + tn) / N
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N ** 2
        ag.append(po)
        kp.append((po - pe) / (1 - pe))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"agreement_ci95": q(ag), "kappa_ci95": q(kp), "B": B, "seed": seed, "clusters": len(logs)}


def rate_by(mask_dis, key, edges, name):
    """Disagreement rate by bins of `key` (same shape as mask_dis)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        out.append({"bin": f"[{lo:g},{hi:g})", "n": int(m.sum()), "disagree_rate": float(mask_dis[m].mean()) if m.any() else None})
    return {"by": name, "bins": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", default="C:/Users/Admin/tmpauxv/tracked")
    a = ap.parse_args()
    d = AC.load_c3()
    tok, logs = d["token"], d["log"]
    N = len(tok)
    nav_fail = d["dac"] == 0
    tj = json.load(open(AC.C3 / "tokens.json"))
    stratum = np.array([tj["stratum"][t] for t in tok])
    store_man = json.load(open(AC.STORE / "manifest.json"))
    fields = {t: AC.gt_field(l, t) for t, l in zip(tok, logs)}
    res = {"tool": "agreement/1", "unit": "proposal, pooled over all 64 proposals of all tokens (never restricted to picks)", "n_tokens": N, "n_proposals": int(nav_fail.size),
           "positive_class": "violation (NAVSIM DAC = 0)", "navsim_dac_zero_share": float(nav_fail.mean()),
           "gt_store_sha256": store_man["store_sha256"], "table": "raw/2026-10-11-c3-selection-headroom/table.npz sub[:,:,1]",
           "primary_instrument": "GT raster, raw proposal poses -> 41-step 0.1 s grid, bilinear signed field, unknown=drivable", "estimator": None}

    # ---------------------------------------------------------------- A. primary + secondary, full set
    runs = {}
    for mode in ("bilinear", "nearest"):
        _, r = AC.run(mode=mode, fields=fields)
        runs[mode] = r
        c = AC.confusion(~r["passed"], nav_fail)
        c.update(boot(~r["passed"], nav_fail, logs))
        c["n_corner_samples_outside_raster_extent"] = int(r["n_unknown"].sum())
        res[f"A_{mode}"] = c
    r = runs["bilinear"]
    geo_fail = ~r["passed"]
    dis = geo_fail != nav_fail
    props = d["proposals"]
    final_x, final_y, final_h = props[:, :, -1, 0], props[:, :, -1, 1], np.abs(props[:, :, -1, 2])
    steps = np.linalg.norm(np.diff(np.concatenate([np.zeros(props.shape[:2] + (1, 2)), props[..., :2]], -2), axis=-2), axis=-1)
    plen = steps.sum(-1)
    margin = np.where(np.isfinite(r["min_margin_m"]), r["min_margin_m"], 99.0)
    splits = []
    splits.append(rate_by(dis, margin, [-99, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 99], "geoDAC min corner margin (m; negative = a corner is outside)"))
    splits.append(rate_by(dis, final_x, [-1, 5, 10, 20, 30, 40, 70], "final x (m) at 4 s"))
    splits.append(rate_by(dis, np.abs(final_y), [0, 1, 2, 4, 8, 30], "|final y| (m) at 4 s"))
    splits.append(rate_by(dis, final_h, [0, 0.1, 0.3, 0.6, 1.0, 4.0], "|final heading| (rad) at 4 s"))
    splits.append(rate_by(dis, plen, [0, 5, 10, 20, 30, 70], "path length (m)"))
    fb = r["first_bad_step"]
    splits.append({"by": "geoDAC first off-road step (0.1 s steps; -1 = none)", "bins": [
        {"bin": nm, "n_geo_fail": int(((fb >= lo) & (fb <= hi) & geo_fail).sum()), "navsim_also_fails": float(nav_fail[(fb >= lo) & (fb <= hi) & geo_fail].mean())
         if ((fb >= lo) & (fb <= hi) & geo_fail).any() else None}
        for nm, lo, hi in (("t=0", 0, 0), ("0.1-1.0 s", 1, 10), ("1.1-2.0 s", 11, 20), ("2.1-3.0 s", 21, 30), ("3.1-4.0 s", 31, 40))]})
    res["A_splits_bilinear"] = splits
    per = {}
    for nm in ("zero", "random"):
        m = stratum == nm
        c = AC.confusion(geo_fail[m], nav_fail[m])
        per[f"stratum_{nm}"] = {k: c[k] for k in ("N", "agreement", "kappa", "base_rate_navsim_fail", "sensitivity_navsim_fail_found", "specificity")}
    heads = {}
    for lg in sorted(set(logs)):
        if lg not in heads:
            heads[lg] = A.read_shard(AC.STORE / "shards" / f"{lg}.aux1", with_blobs=False)[0]["map"]
    mp = np.array([heads[l] for l in logs])
    for m_ in sorted(set(mp)):
        m = mp == m_
        c = AC.confusion(geo_fail[m], nav_fail[m])
        per[f"map_{m_}"] = {k: c[k] for k in ("N", "agreement", "kappa", "base_rate_navsim_fail", "sensitivity_navsim_fail_found", "specificity")}
    res["A_by_stratum_and_map_bilinear"] = per

    # ---------------------------------------------------------------- B. diagnostic decomposition on NAVSIM's own code
    files = sorted(glob.glob(str(Path(a.tracked) / "tracked_*of*.npz")))
    if files:
        Z = [np.load(f) for f in files]
        idx = np.concatenate([z["idx"] for z in Z])
        order = np.argsort(idx)
        idx = idx[order]
        cat = lambda k: np.concatenate([z[k] for z in Z])[order]  # noqa: E731
        d0, d1 = cat("d0_sim_poly"), cat("d1_raw_poly")
        c_sim, c_raw, origin = cat("corners_sim_ego").astype(np.float64), cat("corners_raw_ego").astype(np.float64), cat("origin")
        tab_dac = d["dac"][idx]
        B = {"n_tokens": int(len(idx)), "n_proposals": int(tab_dac.size), "note": "DIAGNOSTIC: re-derivation from NAVSIM v1.1's own code (tracked_dump.py); decides nothing"}
        B["R0_navsim_code_reproduces_table_dac"] = {"equal": int((d0 == tab_dac).sum()), "of": int(tab_dac.size)}
        # origin: metric cache rear axle vs the store entry
        dxy, dh = [], []
        for n, i in enumerate(idx):
            ent = {e["t"]: e for e in A.read_shard(AC.STORE / "shards" / f"{logs[i]}.aux1", with_blobs=False)[0]["entries"]}[tok[i]]
            p = ent["p"]
            dxy.append(float(np.hypot(origin[n, 0] - p[0], origin[n, 1] - p[1])))
            dh.append(float(abs(np.angle(np.exp(1j * (origin[n, 2] - p[2]))))))
        B["R1_origin_metric_cache_vs_store"] = {"max_xy_m": max(dxy), "max_heading_rad": max(dh)}
        # my densify+corners vs NAVSIM's raw interpolated corners (both ego frame)
        mine = np.stack([G.corners(G.densify(d["proposals"][i])) for i in idx])
        worst, p99 = 0.0, []
        for n in range(len(idx)):                                   # NAVSIM orders corners (FL, RL, RR, FR); mine is (FL, FR, RL, RR): match as SETS
            D = np.linalg.norm(mine[n][..., :, None, :] - c_raw[n][..., None, :, :], axis=-1)      # [64, 41, 4(mine), 4(navsim)]
            m = D.min(axis=-1)
            worst = max(worst, float(m.max()))
            p99.append(float(np.percentile(m, 99)))
        B["R2_my_raw_corners_vs_navsim_raw_corners_as_sets_max_abs_m"] = worst
        B["R2_per_token_p99_max_over_tokens_m"] = float(max(p99))
        B["R3_tracker_displacement_of_corners_m"] = {"median": float(np.median(np.linalg.norm(c_sim - c_raw, axis=-1).max(axis=(-1, -2)))),
                                                      "p90": float(np.percentile(np.linalg.norm(c_sim - c_raw, axis=-1).max(axis=(-1, -2)), 90)),
                                                      "max": float(np.linalg.norm(c_sim - c_raw, axis=-1).max())}
        lg_sub = [logs[i] for i in idx]
        nf = tab_dac == 0
        variants = {}
        variants["V1_exact_polygons_on_NAVSIM_raw_states (tracker effect alone)"] = ~(d1 == 1)
        v2, v3, v2n = [], [], []
        gd2 = []
        for n, i in enumerate(idx):
            g2 = G.geodac_corners(fields[tok[i]], G.GT_GEOM, c_sim[n], mode="bilinear")
            gd2.append(g2)
            v2.append(~g2.passed)
            v2n.append(~G.geodac_corners(fields[tok[i]], G.GT_GEOM, c_sim[n], mode="nearest").passed)
            v3.append(~G.geodac_corners(fields[tok[i]], G.GT_GEOM, c_raw[n], mode="bilinear").passed)
        variants["V2_GT_raster_on_TRACKED_states (raster effect alone)"] = np.stack(v2)
        variants["V2n_same_with_nearest_cell"] = np.stack(v2n)
        variants["V3_GT_raster_on_NAVSIM_raw_states"] = np.stack(v3)
        variants["V4_PRIMARY_GT_raster_on_my_raw_poses"] = ~runs["bilinear"]["passed"][idx]
        # ---- DIAGNOSTIC detail of the raster-only effect (V2): where do its false alarms sit?
        mg2 = np.stack([g.min_margin_m for g in gd2])
        fb2 = np.stack([g.first_bad_step for g in gd2])
        V2 = variants["V2_GT_raster_on_TRACKED_states (raster effect alone)"]
        fp2, fn2 = V2 & ~nf, ~V2 & nf
        pc = lambda x: {"p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))} if len(x) else None  # noqa: E731
        t0_tokens = int((fb2 == 0).any(axis=1).sum())
        B["V2_detail"] = {
            "FP_n": int(fp2.sum()), "FN_n": int(fn2.sum()),
            "FP_min_margin_m_percentiles": pc(mg2[fp2]), "FN_min_margin_m_percentiles": pc(mg2[fn2]),
            "FP_share_margin_in_[-0.25,0)": float(((mg2[fp2] >= -0.25) & (mg2[fp2] < 0)).mean()), "FP_share_margin_in_[-0.5,-0.25)": float(((mg2[fp2] >= -0.5) & (mg2[fp2] < -0.25)).mean()),
            "FP_share_margin_below_-0.5": float((mg2[fp2] < -0.5).mean()),
            "FP_with_first_bad_step_0 (initial footprint flagged)": int((fp2 & (fb2 == 0)).sum()),
            "tokens_whose_initial_footprint_is_flagged_by_the_raster": t0_tokens,
            "tokens_where_raster_flags_ALL_64_but_NAVSIM_passes_at_least_one": int((V2.all(axis=1) & (~nf).any(axis=1)).sum()),
            "descriptive_margin_threshold_sweep (fail iff min margin < t; NOT a tuned setting)": [
                {"t_m": float(t), **{k: AC.confusion(mg2 < t, nf)[k] for k in ("agreement", "kappa", "sensitivity_navsim_fail_found", "specificity")}} for t in (-0.5, -0.35, -0.25, -0.15, 0.0, 0.1, 0.25)]}
        B["variants_vs_table_dac"] = {}
        for k, v in variants.items():
            c = AC.confusion(v, nf)
            c.update(boot(v, nf, lg_sub, B=1000))
            B["variants_vs_table_dac"][k] = c
        # attribution of the PRIMARY disagreements
        prim = variants["V4_PRIMARY_GT_raster_on_my_raw_poses"]
        poly_raw = variants["V1_exact_polygons_on_NAVSIM_raw_states (tracker effect alone)"]
        dis_p = prim != nf
        fp, fn = prim & ~nf, ~prim & nf
        B["attribution_of_primary_disagreements"] = {
            "n_disagree": int(dis_p.sum()), "FP_geo_fails_navsim_ok": int(fp.sum()), "FN_geo_ok_navsim_fails": int(fn.sum()),
            "disagreements_where_exact_polygons_on_raw_states_ALSO_disagree_with_table (tracker-caused)": int((dis_p & (poly_raw != nf)).sum()),
            "disagreements_where_exact_polygons_on_raw_states_AGREE_with_table (raster-caused)": int((dis_p & (poly_raw == nf)).sum()),
            "FP_tracker_caused": int((fp & (poly_raw != nf)).sum()), "FP_raster_caused": int((fp & (poly_raw == nf)).sum()),
            "FN_tracker_caused": int((fn & (poly_raw != nf)).sum()), "FN_raster_caused": int((fn & (poly_raw == nf)).sum())}
        B["tracker_direction_exact_polygons_raw_vs_table"] = {"raw_fails_navsim_ok(tracker repairs)": int((poly_raw & ~nf).sum()), "raw_ok_navsim_fails(tracker breaks)": int((~poly_raw & nf).sum())}
        res["B_decomposition"] = B
    else:
        res["B_decomposition"] = "NOT RUN: no tracked shards at " + a.tracked
    json.dump(cj(res), open(HERE / "AGREEMENT.json", "w"), indent=1)
    print(json.dumps(cj({k: res[k] for k in res if k.startswith(("A_bil", "A_near"))}), indent=1)[:3000])
    if isinstance(res["B_decomposition"], dict):
        print(json.dumps(cj(res["B_decomposition"]), indent=1)[:6000])


if __name__ == "__main__":
    main()
