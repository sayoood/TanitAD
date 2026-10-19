"""S1b identity analysis: navtrain's OWN filter (CV PDMS > 0.8 out; human PDMS < 0.8 out) with OUR PDM scorer, against the known navtrain tokens.
Reads raw/identity/<log>.jsonl (s1b_worker.py --mode all) + the S1 token table.  numpy only (tanitad venv); sqlite read-only; no label of any endpoint.

    python analyze_identity.py [--in raw/identity] [--out raw/identity_s1b.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "C:/Users/Admin/par18/unseen_s1/code")
import unseen_token_rule as R  # noqa: E402  (frozen S1)

PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
ROOT = os.path.dirname(HERE)
SHUF_SEED, N_SHUF, N_BOOT, BOOT_SEED = 20261019, 2000, 4000, 20261019


def load_known():
    import gzip
    by = collections.defaultdict(set)
    with gzip.open(PK + "/raw/rl_pool/navtrain_token_table.tsv.gz", "rt") as f:
        h = f.readline().rstrip("\n").split("\t")
        for l in f:
            r = dict(zip(h, l.rstrip("\n").split("\t")))
            if r["log"]:
                by[r["log"]].add(r["token"])
    return by


def features(path, lp_idx):
    """VERBATIM from S1 bias_measure.py:features (label-free covariates of the logged ego; population stage only)."""
    c = R._ro(path)
    try:
        rows = c.execute("SELECT e.x, e.y, e.vx, e.vy, e.qw, e.qx, e.qy, e.qz, e.acceleration_x FROM lidar_pc l JOIN ego_pose e ON e.token = l.ego_pose_token ORDER BY l.timestamp").fetchall()
    finally:
        c.close()
    a = np.array(rows, dtype=float)
    yaw = np.arctan2(2 * (a[:, 4] * a[:, 7] + a[:, 5] * a[:, 6]), 1 - 2 * (a[:, 6] ** 2 + a[:, 7] ** 2))
    i = np.asarray(lp_idx)
    j = np.minimum(i + 80, len(a) - 1)
    v = np.hypot(a[i, 2], a[i, 3])
    disp = np.hypot(a[j, 0] - a[i, 0], a[j, 1] - a[i, 1])
    dy = np.abs((yaw[j] - yaw[i] + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    return v, disp, dy


def prf(keep, known):
    k, n = keep & known, known.sum()
    rec = k.sum() / n if n else float("nan")
    pre = k.sum() / keep.sum() if keep.sum() else float("nan")
    return float(rec), float(pre), int(k.sum()), int(keep.sum()), int(n)


def q(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), **{k: round(float(v), 4) for k, v in zip(("min", "p5", "p25", "p50", "p75", "p95", "max"), np.percentile(x, [0, 5, 25, 50, 75, 95, 100]))}, "mean": round(float(x.mean()), 4)} if len(x) else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=f"{ROOT}/raw/identity")
    ap.add_argument("--out", default=f"{ROOT}/raw/identity_s1b.json")
    a = ap.parse_args()
    known = load_known()
    files = sorted(glob.glob(f"{a.inp}/*.jsonl"))
    logs = [os.path.basename(f)[:-6] for f in files if os.path.exists(f[:-6] + ".done")]
    rows = collections.defaultdict(list)       # per field arrays over ALL candidates of the analysed logs
    LG, KN, KEEP, CVP, HP, X0, X2, V, CPU, ERR, GUARD = ([] for _ in range(11))
    FV, FD, FY = [], [], []
    miss_unscored = 0
    for n, lg in enumerate(logs):
        rec = R.read_log(f"{DB}/{lg}.db")
        c = R.candidates(rec)
        sc = {}
        for line in open(f"{a.inp}/{lg}.jsonl", encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    sc[r["t"]] = r
                except Exception:  # noqa: BLE001
                    pass
        fv, fd, fy = features(f"{DB}/{lg}.db", c["lp_idx"])
        for j, t in enumerate(c["tokens"]):
            r = sc.get(t)
            if r is None:
                miss_unscored += 1
                continue
            LG.append(n)
            KN.append(t in known[lg])
            ok = bool(r.get("res_ok")) and r.get("cv") is not None and r.get("h") is not None
            KEEP.append(bool(r.get("keep")))
            CVP.append(r["cv"]["pdms"] if ok else np.nan)
            HP.append(r["h"]["pdms"] if ok else np.nan)
            X0.append(r["x0"]["pdms"] if ok and "x0" in r else np.nan)
            X2.append(r["x2"]["pdms"] if ok and "x2" in r else np.nan)
            V.append(r.get("v", np.nan))
            CPU.append(r.get("cpu", np.nan))
            ERR.append("err" in r)
            GUARD.append(bool(r.get("skipped_guard")))
            FV.append(fv[j]); FD.append(fd[j]); FY.append(fy[j])
    LG, KN, KEEP = np.array(LG), np.array(KN, bool), np.array(KEEP, bool)
    CVP, HP, X0, X2, V, CPU = map(lambda z: np.array(z, float), (CVP, HP, X0, X2, V, CPU))
    ERR, GUARD = np.array(ERR, bool), np.array(GUARD, bool)
    FV, FD, FY = map(np.array, (FV, FD, FY))
    nlog = len(logs)
    res = {"what": "S1b identity: navtrain's own filter (CV agent PDMS > 0.8 removed; human PDMS < 0.8 removed) with OUR PDM scorer, vs the known navtrain tokens, on a seeded subset of the 214 navtrain-in-val logs",
           "evidence": "MEASURED (ours; DB + map only, CPU; no frame, no model, no head)", "n_logs": nlog, "logs": logs, "n_candidates": int(len(KN)), "n_known": int(KN.sum()),
           "n_candidates_not_scored": miss_unscored, "n_scorer_errors": int(ERR.sum()), "n_guard_skipped": int(GUARD.sum())}
    ok = np.isfinite(CVP) & np.isfinite(HP)
    res["n_scored_both"] = int(ok.sum())

    # ---- the frozen rule ---------------------------------------------------------------------------------------------------------
    rec, pre, hit, nk, nknown = prf(KEEP, KN)
    base_prec = float(KN.sum() / len(KN))
    per = []
    for n in range(nlog):
        m = LG == n
        r_, p_, h_, k_, kn_ = prf(KEEP[m], KN[m])
        per.append({"log": logs[n], "n_cand": int(m.sum()), "n_known": int(KN[m].sum()), "n_kept": int(KEEP[m].sum()), "hit": h_, "recall": r_, "precision": p_, "baseline_precision": float(KN[m].mean())})
    res["frozen_filter"] = {"recall_micro": rec, "precision_micro": pre, "hits": hit, "n_kept": nk, "n_known": nknown, "S1_window_rule_only_precision_on_these_logs": base_prec,
                            "kept_share_of_candidates": float(KEEP.mean()), "known_share_of_candidates": base_prec,
                            "precision_per_log": q([p["precision"] for p in per if p["precision"] == p["precision"]]), "recall_per_log": q([p["recall"] for p in per]),
                            "n_logs_recall_1": int(sum(1 for p in per if p["recall"] >= 1 - 1e-12))}
    res["per_log"] = per
    rng = np.random.default_rng(BOOT_SEED)
    bs_r, bs_p = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, nlog, nlog)
        h = sum(per[i]["hit"] for i in pick); nn = sum(per[i]["n_known"] for i in pick); kk = sum(per[i]["n_kept"] for i in pick)
        bs_r.append(h / nn); bs_p.append(h / kk if kk else np.nan)
    res["frozen_filter"]["recall_ci95_log_cluster_bootstrap"] = [float(np.nanpercentile(bs_r, 2.5)), float(np.nanpercentile(bs_r, 97.5))]
    res["frozen_filter"]["precision_ci95_log_cluster_bootstrap"] = [float(np.nanpercentile(bs_p, 2.5)), float(np.nanpercentile(bs_p, 97.5))]
    res["frozen_filter"]["bootstrap"] = {"n": N_BOOT, "seed": BOOT_SEED, "unit": "log (episode-cluster)", "estimator": "micro recall / precision over resampled logs"}

    # ---- decomposition of the clauses and the residual ---------------------------------------------------------------------------
    cv_out, h_out = ok & (CVP > 0.8), ok & (HP < 0.8)
    res["clauses"] = {"candidates_removed_by_CV_gt_0.8": int(cv_out.sum()), "removed_by_human_lt_0.8": int(h_out.sum()), "removed_by_both": int((cv_out & h_out).sum()),
                      "removed_by_neither_but_unscored": int((~ok).sum()), "kept": int(KEEP.sum()),
                      "known_navtrain_with_our_CV_gt_0.8": int((KN & cv_out).sum()), "known_navtrain_with_our_human_lt_0.8": int((KN & h_out).sum()),
                      "known_navtrain_unscored_(error/guard)": int((KN & ~ok).sum())}
    miss = KN & ~KEEP
    reasons = collections.Counter()
    for i in np.where(miss)[0]:
        if not ok[i]:
            reasons["unscored (error / guard)"] += 1
        elif cv_out[i] and h_out[i]:
            reasons["CV > 0.8 AND human < 0.8"] += 1
        elif cv_out[i]:
            reasons["CV > 0.8 only"] += 1
        else:
            reasons["human < 0.8 only"] += 1
    res["residual_false_removals_(known navtrain tokens our filter removes)"] = {"n": int(miss.sum()), "reasons": dict(reasons)}
    marg = {}
    for name, x, thr, sign in (("CV_pdms_minus_0.8_of_removed_known", CVP[miss & cv_out], 0.8, 1), ("0.8_minus_human_pdms_of_removed_known", HP[miss & h_out], 0.8, -1)):
        d = (x - thr) * sign
        marg[name] = {"n": int(len(d)), "within_0.01": int((d <= 0.01).sum()), "within_0.05": int((d <= 0.05).sum()), "within_0.10": int((d <= 0.10).sum()), "quantiles": q(d) if len(d) else None}
    res["residual_false_removals_margins"] = marg
    fk = KEEP & ~KN
    res["residual_false_keeps_(kept, not in navtrain)"] = {"n": int(fk.sum()), "share_of_kept": float(fk.sum() / max(KEEP.sum(), 1)),
                                                           "cv_pdms_quantiles": q(CVP[fk]) if fk.any() else None, "human_pdms_quantiles": q(HP[fk]) if fk.any() else None,
                                                           "kept_known_cv_pdms_quantiles": q(CVP[KEEP & KN]) if (KEEP & KN).any() else None, "kept_known_human_pdms_quantiles": q(HP[KEEP & KN]) if (KEEP & KN).any() else None,
                                                           "stationary_share_(speed<0.5)": {"false_keeps": float((FV[fk] < 0.5).mean()) if fk.any() else None, "kept_known": float((FV[KEEP & KN] < 0.5).mean()) if (KEEP & KN).any() else None},
                                                           "within_0.05_of_a_threshold": int((fk & ((np.abs(CVP - 0.8) <= 0.05) | (np.abs(HP - 0.8) <= 0.05))).sum())}
    # distribution of the known tokens' scores (what navtrain's own scorer implies): known tokens must have CV <= 0.8 < ... human >= 0.8 under NAVSIM's scorer
    res["known_tokens_our_scores"] = {"cv_pdms": q(CVP[KN & ok]), "human_pdms": q(HP[KN & ok]), "share_cv_le_0.8": float((CVP[KN & ok] <= 0.8).mean()), "share_human_ge_0.8": float((HP[KN & ok] >= 0.8).mean())}
    res["non_known_candidates_our_scores"] = {"share_cv_le_0.8": float((CVP[~KN & ok] <= 0.8).mean()), "share_human_ge_0.8": float((HP[~KN & ok] >= 0.8).mean())}

    # ---- inclusivity variants (the threshold equality) ----------------------------------------------------------------------------
    eq_cv, eq_h = int((np.abs(CVP[ok] - 0.8) < 1e-9).sum()), int((np.abs(HP[ok] - 0.8) < 1e-9).sum())
    var = {}
    for name, kp in {"keep CV<=0.8 & human>=0.8 (frozen: 'exceeds' / 'less than')": ok & (CVP <= 0.8) & (HP >= 0.8), "keep CV<0.8 & human>=0.8": ok & (CVP < 0.8) & (HP >= 0.8),
                     "keep CV<=0.8 & human>0.8": ok & (CVP <= 0.8) & (HP > 0.8), "keep CV<0.8 & human>0.8": ok & (CVP < 0.8) & (HP > 0.8)}.items():
        r_, p_, h_, k_, _ = prf(kp, KN)
        var[name] = {"recall": r_, "precision": p_, "n_kept": k_}
    res["inclusivity"] = {"n_cv_pdms_equal_0.8_(1e-9)": eq_cv, "n_human_pdms_equal_0.8_(1e-9)": eq_h, "variants": var}

    # ---- RED arms (all offline, from the stored scores) ---------------------------------------------------------------------------
    red = {}
    sw = ok & (CVP > 0.8) & (HP < 0.8)           # thresholds' directions swapped
    r_, p_, h_, k_, _ = prf(sw, KN); red["RED_swapped_directions_(keep CV>0.8 & human<0.8)"] = {"recall": r_, "precision": p_, "n_kept": k_, "hits": h_}
    sw2 = ok & (CVP > 0.8) & (HP >= 0.8)
    r_, p_, h_, k_, _ = prf(sw2, KN); red["RED_CV_direction_swapped_only_(keep CV>0.8 & human>=0.8)"] = {"recall": r_, "precision": p_, "n_kept": k_, "hits": h_}
    sw3 = ok & (CVP <= 0.8) & (HP < 0.8)
    r_, p_, h_, k_, _ = prf(sw3, KN); red["RED_human_direction_swapped_only_(keep CV<=0.8 & human<0.8)"] = {"recall": r_, "precision": p_, "n_kept": k_, "hits": h_}
    for name, kp in {"DIAG_CV_clause_only": ok & (CVP <= 0.8), "DIAG_human_clause_only": ok & (HP >= 0.8)}.items():
        r_, p_, h_, k_, _ = prf(kp, KN); red[name] = {"recall": r_, "precision": p_, "n_kept": k_, "hits": h_}
    if np.isfinite(X0).any():
        for name, X in (("CVDEF_stationary_agent_(v=0)_instead_of_CV", X0), ("CVDEF_CV_at_2x_ego_speed", X2)):
            kp = np.isfinite(X) & ok & (X <= 0.8) & (HP >= 0.8)
            r_, p_, h_, k_, _ = prf(kp, KN); red[name] = {"recall": r_, "precision": p_, "n_kept": k_, "hits": h_}
    res["red_arms"] = red

    # ---- control: shuffle the filter decisions within each log ----------------------------------------------------------------------
    rng = np.random.default_rng(SHUF_SEED)
    idx_by_log = [np.where(LG == n)[0] for n in range(nlog)]
    sp, sr = [], []
    for _ in range(N_SHUF):
        kp = np.zeros(len(KEEP), bool)
        for ix in idx_by_log:
            kp[ix] = rng.permutation(KEEP[ix])
        r_, p_, _, _, _ = prf(kp, KN)
        sp.append(p_); sr.append(r_)
    res["control_shuffle_within_log"] = {"n_shuffles": N_SHUF, "seed": SHUF_SEED, "precision_mean": float(np.mean(sp)), "precision_p2.5_p97.5": [float(np.percentile(sp, 2.5)), float(np.percentile(sp, 97.5))],
                                         "recall_mean": float(np.mean(sr)), "expected_precision_if_random_(= known share of candidates, weighted by kept per log)":
                                         float(sum(p["n_kept"] * p["baseline_precision"] for p in per) / max(sum(p["n_kept"] for p in per), 1))}

    # ---- threshold sweep (DIAGNOSTIC ONLY: the thresholds stay 0.8, committed by the paper) -----------------------------------------
    sweep = []
    for tc in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        for th in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            kp = ok & (CVP <= tc) & (HP >= th)
            r_, p_, _, k_, _ = prf(kp, KN)
            sweep.append({"cv_max": tc, "human_min": th, "recall": r_, "precision": p_, "f1": (2 * r_ * p_ / (r_ + p_)) if (r_ + p_) else 0.0, "n_kept": k_})
    res["threshold_sweep_diagnostic"] = sweep
    res["threshold_sweep_diagnostic_best_f1"] = max(sweep, key=lambda s: s["f1"])

    # ---- population shares: navtrain vs ours vs all candidates (S1's label-free features) --------------------------------------------
    def shares(m, label):
        out = {"n": int(m.sum())}
        for nm, mm in {"stationary_speed_lt_0.5": FV < 0.5, "progress_4s_lt_2m": FD < 2.0, "turning_dyaw_gt_15deg": FY > 15.0}.items():
            pooled = float(mm[m].mean()) if m.any() else None
            dw = float(np.mean([mm[m & (LG == n)].mean() for n in range(nlog) if (m & (LG == n)).any()])) if m.any() else None
            out[nm] = {"pooled": pooled, "design_weighted_equal_log": dw}
        return out
    res["population_shares"] = {"navtrain_known": shares(KN, "navtrain"), "our_filter_kept": shares(KEEP, "ours"), "all_candidates": shares(np.ones(len(KN), bool), "all")}
    # log-cluster bootstrap of (ours - navtrain) design-weighted shares
    dd = {k: [] for k in ("stationary_speed_lt_0.5", "turning_dyaw_gt_15deg")}
    rngb = np.random.default_rng(BOOT_SEED + 1)
    mm = {"stationary_speed_lt_0.5": FV < 0.5, "turning_dyaw_gt_15deg": FY > 15.0}
    perlog = {}
    for n in range(nlog):
        for nm, m in mm.items():
            km, ko = (KN & (LG == n)), (KEEP & (LG == n))
            perlog[(n, nm)] = (m[km].mean() if km.any() else np.nan, m[ko].mean() if ko.any() else np.nan)
    for _ in range(N_BOOT):
        pick = rngb.integers(0, nlog, nlog)
        for nm in dd:
            a_ = np.array([perlog[(i, nm)] for i in pick])
            dd[nm].append(np.nanmean(a_[:, 1]) - np.nanmean(a_[:, 0]))
    res["population_share_diff_ours_minus_navtrain_ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in dd.items()}

    # ---- cost -----------------------------------------------------------------------------------------------------------------------
    done = [json.load(open(f"{a.inp}/{lg}.done")) for lg in logs]
    cpu_tok = CPU[np.isfinite(CPU)]
    tot_cpu = sum(d["cpu_s"] for d in done); tot_wall = sum(d["wall_s"] for d in done); tot_n = sum(d["n_scored"] for d in done)
    res["cost"] = {"n_tokens": tot_n, "cpu_s_total": tot_cpu, "wall_s_total": tot_wall, "cpu_s_per_token_incl_all_overheads": tot_cpu / tot_n, "wall_s_per_token": tot_wall / tot_n,
                   "cpu_s_per_token_median_warm": float(np.median(cpu_tok)), "cpu_s_per_token_mean_excl_top_1pct": float(np.mean(np.sort(cpu_tok)[: int(0.99 * len(cpu_tok))])),
                   "cpu_s_per_token_quantiles": q(cpu_tok), "cpu_s_per_token_by_ego_state": {"stationary": float(np.nanmedian(CPU[V < 0.5])) if (V < 0.5).any() else None, "moving": float(np.nanmedian(CPU[V >= 0.5])) if (V >= 0.5).any() else None}}
    U = json.load(open("C:/Users/Admin/par18/unseen_s1/raw/U_candidates_per_log.json"))["per_log"]
    s3 = json.load(open(f"{ROOT}/raw/s3_provisional_logs.json"))["logs"]
    n_s3 = sum(U[l]["n_candidates"] for l in s3)
    n_u = sum(v["n_candidates"] for v in U.values())
    cpt = res["cost"]["cpu_s_per_token_incl_all_overheads"]
    res["cost"]["projection"] = {"S3_sample_123_logs_all_candidates": {"n_tokens": n_s3, "cpu_hours": n_s3 * cpt / 3600, "bar_cpu_hours": 2.0, "within_bar": bool(n_s3 * cpt / 3600 <= 2.0)},
                                 "all_1129_U_logs_all_candidates": {"n_tokens": n_u, "cpu_hours": n_u * cpt / 3600, "bar_cpu_hours": 4.0, "within_bar": bool(n_u * cpt / 3600 <= 4.0)}}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    f = res["frozen_filter"]
    print("IDENTITY", json.dumps({"n_logs": nlog, "recall": rec, "precision": pre, "hits": hit, "n_kept": nk, "n_known": nknown, "baseline_precision": base_prec}), flush=True)
    print("RED", json.dumps(red.get("RED_swapped_directions_(keep CV>0.8 & human<0.8)")), flush=True)
    print("CONTROL", json.dumps(res["control_shuffle_within_log"]), flush=True)
    print("COST", json.dumps(res["cost"]["projection"]), cpt, flush=True)
    print("ZZANALYSE_OK", flush=True)


if __name__ == "__main__":
    main()
