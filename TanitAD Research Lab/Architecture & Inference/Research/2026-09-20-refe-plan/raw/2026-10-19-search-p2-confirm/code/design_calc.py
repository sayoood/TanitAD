#!/usr/bin/env python3
"""SEARCH-P2-CONFIRM deliverable 3: the power of the confirmation reads, simulated from carve B's ALREADY-READ P2HEAD per-token deltas.  It reads NO B-REST label (none exists) and no navtest label.

INPUTS (all already read by Step B / L2-FRESH): carve B's bundle (the pick's BANKED labels), `search_P2HEAD.npz`, `labels_P2HEAD.npz` (400 tokens, 24 logs, 3 search seeds).  B-REST's and navtest's STRUCTURE (tokens per log) comes from token / log
lists only: `raw/b_rest.json` and `navtest_full_tokens.json`.  No label of either is read.

THE ESTIMATOR (same as Step B's read): `stepA.ratio_ci` over `stepA.boot_indices` (paired log-cluster bootstrap, ratio of sums, WHOLE LOGS resampled, percentile 95 %).  Identity: the real read is reproduced at B = 10,000 / seed 20260927
(+0.3273 [+0.0565, +0.7216], per-seed 0.398 / 0.184 / 0.400, range 0.216).  In the SIMULATION the bootstrap uses B_SIM resamples (default 2,000) per simulated read, not 10,000 (say so wherever a number is quoted);
`--selfcheck-b10000` re-runs one condition at 10,000 to show the difference is inside the Monte-Carlo error.

THE MODEL (ESTIMATED, assumptions named).  Per token a vector v = [mean-of-seeds dPDMS, seed 0, seed 1, seed 2, dNC, dDAC, dTTC, dC] in PDMS points.  Null-centred per series (each series minus its own pooled mean), so a seed-to-seed offset in
carve B is treated as NOISE, not a true effect.  A true effect mu is added to the four PDMS series; the safety series are set to their carve-B observed means scaled by mu / 0.3273 (so they are 0 at mu = 0).
  scheme R (PRIMARY, random effects):  v_il = c_i(resid, within-log) + a_l,  a_l ~ N(0, tau^2) added to the four PDMS series, tau^2 by the unbalanced one-way ANOVA moment estimator on the mean-of-seeds series, within-log residual VECTORS
      drawn jointly with replacement from the pooled carve-B residuals (keeps the heavy tails and the seed / family correlation).  Target structure = (tokens per log) of the read.
  scheme S (B-REST and the carve-B replica ONLY, the LOG-SHARED realism): the target logs ARE carve B's 24 logs, so their true effects are shared with carve B: a_l = lambda_l x (carve B centred log mean),
      lambda_l the empirical-Bayes shrinkage tau^2 / (tau^2 + sigma^2 / n_l), re-centred so the true pooled effect on the target structure is mu; within-log residuals as in R.  The bootstrap interval is the SAME (it resamples logs).
  scheme E (SENSITIVITY, conservative two-stage): for each target log pick a SOURCE log among carve B's 24 uniformly at random and draw its n_l tokens with replacement from that log's null-centred pool (its log mean is kept, so the
      between-log variance carries carve B's own sampling noise as well: this OVERSTATES the variance, a conservative power).
Both assume that a different population (navtest) has carve B's heterogeneity (tau^2, within-log spread): an ASSUMPTION, not a measurement.  The effect on navtest may differ in size and mix.

OUTCOMES per simulated read (the same rules the real read uses): O1 verbatim = the sealed Step B bar (point >= +0.30 AND lower CI > 0 AND point > the 3-seed range AND dNC, dDAC, dTTC, dC points each >= -0.25), the 2-term bar
(point >= +0.30 AND lower CI > 0), the SIGN bar (lower CI > 0), NEGATIVE (upper CI < 0) and NULL (the interval contains 0).

    CUDA_VISIBLE_DEVICES=-1 python design_calc.py --out ../raw/design_calc.json [--reads 2000 --boot 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
sys.path.insert(0, L1_CODE)
import stepA as A  # noqa: E402

STEPB_RUN = "C:/Users/Admin/par18/l2stepb/run"
BUNDLE_B = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle/bundle.npz"
B_REST_JSON = "C:/Users/Admin/par18/spc/raw/b_rest.json"
NAVTEST_TOKENS = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
OBSERVED_P2HEAD = 0.3272945259675774            # RESULT_STEPB.json arms.P2HEAD.endpoint_dPDMS_points.point
BAR_DPDMS, BAR_SAFETY = 0.30, -0.25
SERIES = ("y", "s0", "s1", "s2", "nc", "dac", "ttc", "c")
SIM_SEED = 20261020                              # fixed before any simulated read; the bootstrap draws inside a read come from the same stream
EFFECTS = (0.0, 0.08, 0.16, 0.24, 0.327, 0.40, 0.50)
HEADLINE = (0.327, 0.16, 0.0)


# ------------------------------------------------------------------------------------------------------------------------------ inputs
def load_carve_b():
    b = A.load_npz(BUNDLE_B)
    so = {k: v for k, v in np.load(f"{STEPB_RUN}/search_P2HEAD.npz", allow_pickle=False).items()}
    lab = {k: v for k, v in np.load(f"{STEPB_RUN}/labels_P2HEAD.npz", allow_pickle=False).items()}
    idx = np.asarray(so["bundle_index"], np.int64)
    assert [str(t) for t in lab["token"]] == [str(t) for t in so["token"]]
    pick = np.asarray(b["pick"], np.int64)[idx]
    pick_sub = np.asarray(b["sub"], np.float64)[idx, pick]
    pick_pdms = np.asarray(b["pdms"], np.float64)[idx, pick]
    d_pdms = 100.0 * (np.asarray(lab["pdms"], np.float64) - pick_pdms[:, None])                  # [n,3]
    d_sub = 100.0 * (np.asarray(lab["sub"], np.float64) - pick_sub[:, None, :])                  # [n,3,6]
    names = sorted(set(str(x) for x in b["log"]))
    li = {l: i for i, l in enumerate(names)}
    log_idx = np.array([li[str(b["log"][i])] for i in idx], np.int64)
    sub_ix = {k: j for j, k in enumerate(A.SUBS)}
    V = np.column_stack([d_pdms.mean(1), d_pdms[:, 0], d_pdms[:, 1], d_pdms[:, 2], *[d_sub[:, :, sub_ix[k]].mean(1) for k in ("nc", "dac", "ttc", "c")]])
    return V, log_idx, len(names), names


def identity_check(V, log_idx, n_logs):
    """the real Step B read, reproduced with L1's own estimator at B = 10,000 / seed 20260927"""
    boot = A.boot_indices(n_logs, A.N_BOOT, A.SEED)
    num = np.bincount(log_idx, weights=V[:, 0], minlength=n_logs)
    den = np.bincount(log_idx, minlength=n_logs).astype(np.float64)
    ci = A.ratio_ci(num, den, boot)
    seeds = [float(np.bincount(log_idx, weights=V[:, k], minlength=n_logs).sum() / den.sum()) for k in (1, 2, 3)]
    safety = {s: float(V[:, SERIES.index(s)].sum() / len(V)) for s in ("nc", "dac", "ttc", "c")}
    ref = json.load(open(f"{STEPB_RUN}/RESULT_STEPB.json"))["arms"]["P2HEAD"]
    r = ref["endpoint_dPDMS_points"]
    ok = (abs(ci["point"] - r["point"]) < 1e-9 and abs(ci["lo"] - r["lo"]) < 1e-9 and abs(ci["hi"] - r["hi"]) < 1e-9
          and max(abs(a - b) for a, b in zip(seeds, ref["per_seed_points"]["points"])) < 1e-9
          and all(abs(safety[k] - ref["sub_score_deltas_points"][k]["point"]) < 1e-9 for k in safety))
    return {"reproduces_RESULT_STEPB_P2HEAD": bool(ok), "point": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "per_seed": seeds, "seed_range": max(seeds) - min(seeds),
            "safety_points": safety, "banked": {"point": r["point"], "lo": r["lo"], "hi": r["hi"], "per_seed": ref["per_seed_points"]["points"]}}


def variance_components(V, log_idx, n_logs):
    y = V[:, 0]
    n = len(y)
    nl = np.bincount(log_idx, minlength=n_logs).astype(np.float64)
    ybar_l = np.bincount(log_idx, weights=y, minlength=n_logs) / nl
    m = y.mean()
    ssw = float(((y - ybar_l[log_idx]) ** 2).sum())
    ssb = float((nl * (ybar_l - m) ** 2).sum())
    sig2 = ssw / (n - n_logs)
    n0 = (n - (nl ** 2).sum() / n) / (n_logs - 1)
    tau2 = max(0.0, (ssb / (n_logs - 1) - sig2) / n0)
    # the share of the variance of the pooled estimator that is between-log (at this allocation): Var(sum_l n_l a_l)/n^2 vs sum_l n_l sig2/n^2
    var_b, var_w = tau2 * (nl ** 2).sum() / n ** 2, sig2 / n
    icc = tau2 / (tau2 + sig2) if (tau2 + sig2) > 0 else 0.0
    return {"n_tokens": int(n), "n_logs": int(n_logs), "grand_mean_points": float(m), "sigma2_within": float(sig2), "tau2_between": float(tau2), "icc": float(icc),
            "between_share_of_estimator_variance_at_carveB_allocation": float(var_b / (var_b + var_w)), "effective_clusters_(sum n)^2/sum n^2": float(n ** 2 / (nl ** 2).sum()),
            "se_of_pooled_mean_model": float(np.sqrt(var_b + var_w)), "tokens_with_nonzero_mean_delta": int((np.abs(y) > 1e-12).sum()), "share_nonzero": float((np.abs(y) > 1e-12).mean()),
            "per_log_mean": [float(x) for x in ybar_l], "per_log_n": [int(x) for x in nl]}


# ------------------------------------------------------------------------------------------------------------------------------ the simulation
def make_pools(V, log_idx, n_logs):
    nl = np.bincount(log_idx, minlength=n_logs).astype(np.float64)
    lm_raw = np.stack([np.bincount(log_idx, weights=V[:, k], minlength=n_logs) for k in range(V.shape[1])], 1) / nl[:, None]      # raw per-log means [L, 8]
    ubar = lm_raw.mean(0)                              # UNWEIGHTED mean of log means: scheme E picks source logs uniformly, so this is the null for it
    C = V - ubar                                       # null-centred pools for scheme E
    R = V - lm_raw[log_idx]                            # within-log residual vectors (joint across the 8 series): schemes R and S
    gm = V.mean(0)                                     # pooled (ratio-of-sums) mean
    by_log = [np.where(log_idx == l)[0] for l in range(n_logs)]
    return {"mean": gm, "C": C, "R": R, "by_log": by_log, "obs_safety": gm[4:8].copy(), "log_mean_centered": lm_raw[:, 0] - gm[0], "n_carveB": nl}


def effect_vector(mu, pools):
    e = np.zeros(8)
    e[:4] = mu
    e[4:] = pools["obs_safety"] * (mu / OBSERVED_P2HEAD)
    return e


def shared_log_effects(n_l, pools, tau2, sigma2):
    """scheme S: the target logs ARE carve B's 24 logs, so their true effects are shared with carve B.  a_l = lambda_l x (carve B's centred log mean), lambda_l = tau2 / (tau2 + sigma2 / n_l(carve B))
    (empirical-Bayes shrinkage: carve B's log mean is a noisy estimate of the log's true effect), re-centred so the TOKEN-WEIGHTED mean over the target structure is exactly 0 (the true pooled effect is then mu)."""
    n_l = np.asarray(n_l, np.float64)
    lam = tau2 / (tau2 + sigma2 / pools["n_carveB"]) if tau2 > 0 else np.zeros_like(n_l)
    a = lam * pools["log_mean_centered"]
    return a - float((n_l * a).sum() / n_l.sum())


def one_condition(n_l, mu, scheme, pools, tau2, n_reads, B, rng, boot_idx, sigma2=None):
    """simulate n_reads reads of the structure n_l (tokens per log, in log order) at true effect mu.  Returns outcome probabilities and summary stats."""
    L = len(n_l)
    n_l = np.asarray(n_l, np.int64)
    N = int(n_l.sum())
    starts = np.concatenate([[0], np.cumsum(n_l)[:-1]])
    eff = effect_vector(mu, pools)
    R, C, by_log = pools["R"], pools["C"], pools["by_log"]
    NR, NL = len(R), len(by_log)
    cnt = {k: 0 for k in ("O1_verbatim", "O1_2term", "sign_pos", "negative", "null", "T1", "T2", "T3", "T4", "pdms_clears", "trade")}
    pts, los, his, rngs = [], [], [], []
    den = n_l.astype(np.float64)
    a_shared = shared_log_effects(n_l, pools, tau2, sigma2) if scheme == "S" else None
    for _ in range(n_reads):
        if scheme in ("R", "S"):
            V = R[rng.integers(0, NR, size=N)].copy()
            a = rng.normal(0.0, np.sqrt(tau2), size=L) if scheme == "R" else a_shared
            V[:, :4] += np.repeat(a, n_l)[:, None]
        else:                                                                         # scheme E: source log per target log, tokens from that log's null-centred pool
            src = rng.integers(0, NL, size=L)
            ix = np.concatenate([by_log[s][rng.integers(0, len(by_log[s]), size=int(k))] for s, k in zip(src, n_l)])
            V = C[ix].copy()
        V += eff
        num = np.add.reduceat(V, starts, axis=0)                                     # [L, 8] cluster sums
        pt = num.sum(0) / N
        bi = boot_idx
        r = num[:, 0][bi].sum(1) / den[bi].sum(1)
        lo, hi = float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
        rg = float(pt[1:4].max() - pt[1:4].min())
        t1, t2, t3 = bool(pt[0] >= BAR_DPDMS), bool(lo > 0.0), bool(pt[0] > rg)
        t4 = bool((pt[4:8] >= BAR_SAFETY).all())
        clears = t1 and t2 and t3
        cnt["T1"] += t1; cnt["T2"] += t2; cnt["T3"] += t3; cnt["T4"] += t4
        cnt["pdms_clears"] += clears
        cnt["O1_verbatim"] += clears and t4
        cnt["trade"] += clears and not t4
        cnt["O1_2term"] += t1 and t2
        cnt["sign_pos"] += t2
        cnt["negative"] += bool(hi < 0.0)
        cnt["null"] += bool(lo <= 0.0 <= hi)
        pts.append(float(pt[0])); los.append(lo); his.append(hi); rngs.append(rg)
    p = {k: v / n_reads for k, v in cnt.items()}
    p["not_cleared_sealed_rule"] = 1.0 - p["pdms_clears"]
    se_mc = {k: float(np.sqrt(max(v * (1 - v), 1e-12) / n_reads)) for k, v in p.items()}
    pts, los, his = np.array(pts), np.array(los), np.array(his)
    return {"prob": p, "mc_se_prob_max": float(max(se_mc.values())), "n_reads": n_reads, "boot_per_read": int(len(boot_idx)),
            "point_mean": float(pts.mean()), "point_sd": float(pts.std(ddof=1)), "point_q05_q50_q95": [float(np.quantile(pts, q)) for q in (0.05, 0.5, 0.95)],
            "lower_ci_mean": float(los.mean()), "ci_half_width_mean": float(((his - los) / 2).mean()), "seed_range_mean": float(np.mean(rngs)), "mu": mu}


def structure_from_counts(counts_in_log_order):
    return [int(x) for x in counts_in_log_order]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reads", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--effects", type=float, nargs="*", default=list(EFFECTS))
    ap.add_argument("--selfcheck-b10000", action="store_true")
    ap.add_argument("--structures", nargs="*", default=["B_REST", "carveB_replica", "navtest"])
    a = ap.parse_args(argv)
    t0 = time.time()
    V, log_idx, n_logs, names = load_carve_b()
    ident = identity_check(V, log_idx, n_logs)
    if not ident["reproduces_RESULT_STEPB_P2HEAD"]:
        print("ZZDESIGN_FAIL identity: the simulation inputs do not reproduce the banked Step B P2HEAD read", flush=True)
        return 3
    vc = variance_components(V, log_idx, n_logs)
    pools = make_pools(V, log_idx, n_logs)
    br = json.load(open(B_REST_JSON))
    cb_counts = br["per_log_counts"]["carveB"]
    structures = {}
    # B-REST: the 24 logs, token counts known WITHOUT any label (b_rest.json)
    structures["B_REST"] = structure_from_counts([br["per_log_counts"]["B-REST"][l] for l in names])
    structures["carveB_replica"] = structure_from_counts([cb_counts[l] for l in names])               # control: the structure that produced the observed read
    nav = json.load(open(NAVTEST_TOKENS))
    cnt = {}
    for t in nav["tokens"]:
        cnt[nav["token_log"][t]] = cnt.get(nav["token_log"][t], 0) + 1
    structures["navtest"] = structure_from_counts([cnt[k] for k in sorted(cnt)])
    structures = {k: v for k, v in structures.items() if k in a.structures}
    assert structures["B_REST"] and sum(structures["B_REST"]) == 905 and len(structures["B_REST"]) == 24
    if "navtest" in structures:
        assert sum(structures["navtest"]) == 12146 and len(structures["navtest"]) == 136
    out = {"evidence": "ESTIMATED (a simulation from carve B's already-read P2HEAD per-token deltas; assumptions in the module docstring); inputs MEASURED; no B-REST or navtest label read",
           "sim": {"seed": SIM_SEED, "reads_per_condition": a.reads, "boot_per_read": a.boot, "bootstrap": "stepA.boot_indices-style draws of WHOLE LOGS, ratio of sums, percentile 95 % (the real read uses B = 10,000, seed 20260927)",
                   "stamps": "SELECTION tier, label-only, one training seed, 3 search seeds, 'another draw of LOGS' only (H-ESTIM-SEED-1)"},
           "identity": ident, "variance_components_carveB": {k: v for k, v in vc.items() if not k.startswith("per_log")}, "per_log_carveB": {"mean": vc["per_log_mean"], "n": vc["per_log_n"], "logs": names},
           "structures": {k: {"n_logs": len(v), "n_tokens": int(sum(v)), "tokens_per_log_min_median_max": [int(min(v)), float(np.median(v)), int(max(v))],
                              "effective_clusters": float(sum(v) ** 2 / sum(x * x for x in v))} for k, v in structures.items()},
           "conditions": {}}
    rng_master = np.random.SeedSequence(SIM_SEED)
    kids = rng_master.spawn(len(structures) * 3 * len(a.effects))
    ki = 0
    for sname, nl in structures.items():
        for scheme in (("R", "E", "S") if sname in ("B_REST", "carveB_replica") else ("R", "E")):
            for mu in a.effects:
                rng = np.random.default_rng(kids[ki]); ki += 1
                boot_idx = rng.integers(0, len(nl), size=(a.boot, len(nl)))
                res = one_condition(nl, mu, scheme, pools, vc["tau2_between"], a.reads, a.boot, rng, boot_idx, vc["sigma2_within"])
                out["conditions"][f"{sname}|{scheme}|mu={mu}"] = res
                print("ZZCOND %-15s %s mu=%.3f O1=%.3f 2term=%.3f sign=%.3f null=%.3f neg=%.3f pt=%.3f hw=%.3f  (%.0fs)" % (
                    sname, scheme, mu, res["prob"]["O1_verbatim"], res["prob"]["O1_2term"], res["prob"]["sign_pos"], res["prob"]["null"], res["prob"]["negative"],
                    res["point_mean"], res["ci_half_width_mean"], time.time() - t0), flush=True)
    if a.selfcheck_b10000:
        rng = np.random.default_rng(np.random.SeedSequence(SIM_SEED + 1))
        nl = structures["B_REST"]
        bi = rng.integers(0, len(nl), size=(10000, len(nl)))
        r10 = one_condition(nl, 0.327, "R", pools, vc["tau2_between"], 300, 10000, rng, bi, vc["sigma2_within"])
        out["selfcheck_b10000_B_REST_R_mu0.327_300reads"] = r10
    out["seconds"] = time.time() - t0
    out["inputs_sha256"] = {"search_P2HEAD.npz": hashlib.sha256(open(f"{STEPB_RUN}/search_P2HEAD.npz", "rb").read()).hexdigest(), "labels_P2HEAD.npz": hashlib.sha256(open(f"{STEPB_RUN}/labels_P2HEAD.npz", "rb").read()).hexdigest(),
                            "bundle.npz(carve B)": hashlib.sha256(open(BUNDLE_B, "rb").read()).hexdigest(), "b_rest.json": hashlib.sha256(open(B_REST_JSON, "rb").read()).hexdigest()}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZDESIGN_OK", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
