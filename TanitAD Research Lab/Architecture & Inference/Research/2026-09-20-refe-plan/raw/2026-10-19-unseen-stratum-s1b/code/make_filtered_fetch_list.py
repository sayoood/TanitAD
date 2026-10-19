"""S1b step 5: the S2 / S4 fetch list from the FILTERED candidates (CV PDMS <= 0.8 and human PDMS >= 0.8), PROVISIONAL seed.
Per log: the first 4 tokens of S1's own seeded permutation that PASS the filter and resolve their 4 cameras (== a uniform draw of 4 from the filtered
candidates, because a uniformly random permutation's accepted subsequence is a uniform order).  Reads raw/seq/<log>.jsonl (s1b_worker.py --mode seq).
Also reports the label-free population features of the accepted tokens (stationary / turning) vs navtrain, and the acceptance-rate estimates per log.
No frame, no model, no head.  numpy + sqlite only.

    python make_filtered_fetch_list.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, "C:/Users/Admin/par18/unseen_s1/code")
sys.path.insert(0, HERE)
import unseen_token_rule as R  # noqa: E402
from analyze_identity import features  # noqa: E402

DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
SEED, N_TOK, MB_PER_JPG = 20261020, 4, 0.2055          # S1: 1,968 x 0.2055 MB (mean of 3,000 local navtrain jpgs, MEASURED sizes)
N_BOOT, BOOT_SEED = 4000, 20261019


def main():
    logs = json.load(open(f"{ROOT}/raw/s3_provisional_logs.json"))["logs"]
    S1_FL = json.load(open("C:/Users/Admin/par18/unseen_s1/raw/fetch_list_s2_s4.json"))
    assert S1_FL["logs"] == logs and S1_FL["seed"] == SEED
    SEQ = os.environ.get("S1B_SEQ_DIR", f"{ROOT}/raw/seq")            # test hooks only; the chain uses the defaults
    OUT = os.environ.get("S1B_OUT_PREFIX", f"{ROOT}/raw/fetch_list_s2_s4_filtered")
    if os.environ.get("S1B_NLOGS"):
        logs = logs[: int(os.environ["S1B_NLOGS"])]
    items, per, unfilled = [], [], []
    feat = []
    for lg in logs:
        rec = R.read_log(f"{DB}/{lg}.db")
        c = R.candidates(rec)
        ts = {t: int(rec["ts"][i]) for t, i in zip(c["tokens"], c["lp_idx"])}
        idx = {t: int(i) for t, i in zip(c["tokens"], c["lp_idx"])}
        rows = []
        for line in open(f"{SEQ}/{lg}.jsonl", encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
        scored = [r for r in rows if not r.get("skipped_guard")]
        acc = [r for r in rows if r.get("keep") and r.get("keep_counted", True)]
        n_s, n_a = len(rows), len(acc)
        n_err = sum(1 for r in rows if "err" in r)
        # the estimate of the log's acceptance rate: stopping at the r-th success => unbiased (r-1)/(n-1); a capped log has no such stopping
        if n_a >= N_TOK and n_s > 1:
            rate, est = (n_a - 1) / (n_s - 1), "inverse-binomial (r-1)/(n-1)"
        else:
            rate, est = (n_a / n_s if n_s else float("nan")), "r/n at the cap (censored: a LOWER bound region, the log is unfilled)"
        per.append({"log": lg, "n_candidates": len(c["tokens"]), "n_scored": n_s, "n_accepted": n_a, "n_errors": n_err, "acceptance_rate_estimate": rate, "estimator": est,
                    "estimated_filtered_candidates_ESTIMATED": rate * len(c["tokens"])})
        take = acc[:N_TOK]
        if len(take) < N_TOK:
            unfilled.append({"log": lg, "n_accepted": len(take), "n_scored": n_s, "n_candidates": len(c["tokens"])})
        for r in take:
            t = r["t"]
            fr = R.nearest_jpgs(f"{DB}/{lg}.db", ts[t])
            items.append({"log": lg, "token": t, "ts_us": ts[t], "pdms_cv": r["cv"]["pdms"], "pdms_human": r["h"]["pdms"], "jpgs": {ch: {"path": fr[ch][0], "dt_ms": round(fr[ch][1], 2)} for ch in R.CAMERAS}})
        if take:
            fv, fd, fy = features(f"{DB}/{lg}.db", [idx[r["t"]] for r in take])
            feat.append((lg, fv, fd, fy))
    paths = [it["jpgs"][ch]["path"] for it in items for ch in R.CAMERAS]
    lst = sorted(set(paths))
    # population features of the accepted tokens vs navtrain (design-weighted equal log weight, the S1 convention)
    def dw(sel):
        return float(np.mean([np.mean(sel(fv, fd, fy)) for _, fv, fd, fy in feat]))
    sh = {"stationary_speed_lt_0.5": dw(lambda v, d, y: v < 0.5), "progress_4s_lt_2m": dw(lambda v, d, y: d < 2.0), "turning_dyaw_gt_15deg": dw(lambda v, d, y: y > 15.0)}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {k: [] for k in sh}
    fns = {"stationary_speed_lt_0.5": lambda v, d, y: v < 0.5, "progress_4s_lt_2m": lambda v, d, y: d < 2.0, "turning_dyaw_gt_15deg": lambda v, d, y: y > 15.0}
    per_log_share = {k: np.array([np.mean(fns[k](fv, fd, fy)) for _, fv, fd, fy in feat]) for k in sh}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(feat), len(feat))
        for k in sh:
            boot[k].append(float(per_log_share[k][pick].mean()))
    # the filtered share of the candidates, ratio estimator over logs (the 123 logs were drawn uniformly from the 1,129 U logs with >= 4 candidates); ESTIMATED, not a count
    n_i = np.array([p["n_candidates"] for p in per], float)
    e_i = np.array([p["estimated_filtered_candidates_ESTIMATED"] for p in per], float)
    ratio = float(e_i.sum() / n_i.sum())
    rb = np.random.default_rng(BOOT_SEED + 2)
    bratio = []
    for _ in range(N_BOOT):
        pk = rb.integers(0, len(per), len(per))
        bratio.append(float(e_i[pk].sum() / n_i[pk].sum()))
    U_TOTAL, S3_TOTAL = 548812, int(n_i.sum())
    u_extrap = {"estimator": "ratio over logs of the per-log inverse-binomial filtered-candidate estimates / candidates; log-cluster bootstrap CI (n=%d, seed %d)" % (N_BOOT, BOOT_SEED + 2),
                "filtered_share_of_candidates": ratio, "ci95": [float(np.percentile(bratio, 2.5)), float(np.percentile(bratio, 97.5))],
                "S3_123_logs_candidates": S3_TOTAL, "S3_123_logs_filtered_ESTIMATED": ratio * S3_TOTAL, "S3_123_logs_filtered_ESTIMATED_ci95": [float(np.percentile(bratio, 2.5)) * S3_TOTAL, float(np.percentile(bratio, 97.5)) * S3_TOTAL],
                "U_1129_logs_candidates": U_TOTAL, "U_1129_logs_filtered_ESTIMATED": ratio * U_TOTAL, "U_1129_logs_filtered_ESTIMATED_ci95": [float(np.percentile(bratio, 2.5)) * U_TOTAL, float(np.percentile(bratio, 97.5)) * U_TOTAL],
                "caveat": "ESTIMATED from a 123-log seeded sample, not counted; unfilled/capped logs contribute r/n (censored); the navtrain-logs reference share is 0.195 (214 logs, S1)"}
    ident = json.load(open(f"{ROOT}/raw/identity_s1b.json")) if os.path.exists(f"{ROOT}/raw/identity_s1b.json") else {}
    res = {"what": "S2/S4 fetch list from the FILTERED candidates (our PDM scorer: CV PDMS <= 0.8, human PDMS >= 0.8); 123 S1-provisional logs x up to 4 tokens x 4 cameras",
           "evidence": "MEASURED names / dt (DB) and PDMS (ours); size ESTIMATED", "seed": SEED, "seed_status": "PROVISIONAL (S3's sealed prereg fixes the real one)",
           "sampler": "per log: S1's seeded permutation (default_rng([seed, sha256(log)[:8]])), first 4 tokens that pass the filter and resolve all 4 cameras; cap on scored tokens per log in the driver",
           "n_logs": len(logs), "tokens_per_log": N_TOK, "n_tokens": len(items), "n_jpgs": len(paths), "n_unique_paths": len(lst),
           "size_MB_ESTIMATED": round(len(lst) * MB_PER_JPG, 1), "mb_per_jpg_basis": "S1: mean of 3,000 local navtrain jpgs (raw/jpg_size_sample.json)",
           "S1_unfiltered_list": {"n_jpgs": S1_FL["n_jpgs"], "n_tokens": S1_FL["n_tokens"], "size_MB_ESTIMATED": 404},
           "n_unfilled_logs": len(unfilled), "unfilled": unfilled, "max_dt_ms": max([it["jpgs"][ch]["dt_ms"] for it in items for ch in R.CAMERAS] or [0]),
           "list_sha256": hashlib.sha256("\n".join(lst).encode()).hexdigest(),
           "tokens_scored_total": int(sum(p["n_scored"] for p in per)), "tokens_accepted_total": int(sum(p["n_accepted"] for p in per)),
           "scored_per_accepted_overall": float(sum(p["n_scored"] for p in per) / max(sum(p["n_accepted"] for p in per), 1)),
           "acceptance_rate_across_logs": {"estimator": "per-log inverse-binomial; the sum below is ESTIMATED", "estimated_total_filtered_candidates_ESTIMATED": float(sum(p["estimated_filtered_candidates_ESTIMATED"] for p in per)),
                                           "total_candidates_in_these_logs": int(sum(p["n_candidates"] for p in per)),
                                           "median_per_log_rate": float(np.nanmedian([p["acceptance_rate_estimate"] for p in per]))},
           "U_extrapolation": u_extrap,
           "accepted_token_population_shares_design_weighted_equal_log": {k: {"value": sh[k], "ci95_log_bootstrap": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]} for k in sh},
           "navtrain_reference_shares": {"S1_design_weighted_214_logs": {"stationary": 0.077, "turning": 0.321, "progress<2m": 0.026}, "S1_design_weighted_248_logs": {"stationary": 0.096, "turning": 0.298, "progress<2m": 0.034},
                                         "S1b_20_identity_logs_navtrain": ident.get("population_shares", {}).get("navtrain_known"), "S1b_20_identity_logs_our_filter": ident.get("population_shares", {}).get("our_filter_kept")},
           "per_log": per, "logs": logs, "items": items}
    json.dump(res, open(f"{OUT}.json", "w"))
    open(f"{OUT}.txt", "w", newline="\n").write("\n".join(lst) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_log", "logs", "items", "unfilled")}, indent=1, default=float), flush=True)
    print("ZZFETCH_OK", len(lst), flush=True)


if __name__ == "__main__":
    main()
