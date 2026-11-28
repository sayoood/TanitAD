#!/usr/bin/env python3
"""REVIEW 7 -- gap-to-paper decomposition (CPU only, streaming, < 1 GB RAM).

Inputs (all read-only, all MEASURED artifacts of the final REFe navtest run):
  CSV    data/refe_navtest/score/refe_navtest_final/refe_navtest_final.csv   (official NAVSIM v1.1 run_pdm_score output)
  HOOKS  data/refe_navtest/score/refe_navtest_final/refe_navtest_final_hooks.json  (per-token raw progress [pdm, agent],
         multiplier products, v0, executed agent poses)
  CENSUS raw/2026-10-04-lane-discipline/lane_census.jsonl  (per proposal: 0 = PDM-Closed, 1..64 = hypotheses, 65 = human;
         GT nc/dac/ttc/c/ddc/pdms through NAVSIM's own scorer, and the planner scorer's `agg` for the 64 hypotheses)

Outputs: gap_decompose.json next to this file.

Every block carries its n. Intervals: log-cluster bootstrap of the per-token mean (10,000, seed 20260927), the
estimator the programme already uses for navtest; it answers "another draw of LOGS" only (one checkpoint,
deterministic inference).
"""
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = r"E:/Projects/TanitAD"
PKG = ROOT + r"/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
CSV = ROOT + r"/data/refe_navtest/score/refe_navtest_final/refe_navtest_final.csv"
HOOKS = ROOT + r"/data/refe_navtest/score/refe_navtest_final/refe_navtest_final_hooks.json"
CENSUS = PKG + r"/raw/2026-10-04-lane-discipline/lane_census.jsonl"
OUT = PKG + r"/raw/2026-10-04-gap-review/gap_decompose.json"

B = 10000
SEED = 20260927


def cluster_boot(vals_by_log, B=B, seed=SEED):
    """Log-cluster bootstrap of a per-token mean. vals_by_log: dict log -> list of floats."""
    logs = list(vals_by_log)
    sums = [sum(vals_by_log[l]) for l in logs]
    cnts = [len(vals_by_log[l]) for l in logs]
    tot = sum(sums) / sum(cnts)
    rng = random.Random(seed)
    n = len(logs)
    est = []
    for _ in range(B):
        s = c = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += sums[i]
            c += cnts[i]
        est.append(s / c)
    est.sort()
    return {"est": tot, "lo": est[int(0.025 * B)], "hi": est[int(0.975 * B) - 1], "n": sum(cnts), "logs": n}


def pdms_formula(nc, dac, ep, ttc, c):
    return nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12.0


def main():
    out = {"what": "REVIEW_7 gap decomposition", "inputs": {"csv": CSV, "hooks": HOOKS, "census": CENSUS}}

    # ---------------- A. official CSV ----------------
    rows = {}
    with open(CSV, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            if d["token"] == "average":
                continue
            rows[d["token"]] = {k: float(d[k]) for k in
                                ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                                 "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]}
    n = len(rows)
    max_d_v1 = max_d_ddcmult = 0.0
    for t, d in rows.items():
        s1 = pdms_formula(d["no_at_fault_collisions"], d["drivable_area_compliance"], d["ego_progress"],
                          d["time_to_collision_within_bound"], d["comfort"])
        s2 = s1 * d["driving_direction_compliance"]
        max_d_v1 = max(max_d_v1, abs(s1 - d["score"]))
        max_d_ddcmult = max(max_d_ddcmult, abs(s2 - d["score"]))
    out["A_formula_check"] = {
        "n": n,
        "max_abs_delta_score_vs_NC*DAC*(5EP+5TTC+2C)/12": max_d_v1,
        "max_abs_delta_score_vs_same_times_DDC": max_d_ddcmult,
        "verdict": "DDC is NOT in the PDMS (weight 0, NAVSIM v1.1 PDMScorerConfig.driving_direction_weight = 0.0)"
        if max_d_v1 < 1e-9 and max_d_ddcmult > 1e-6 else "CHECK",
    }

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    NC = [d["no_at_fault_collisions"] for d in rows.values()]
    DAC = [d["drivable_area_compliance"] for d in rows.values()]
    EP = [d["ego_progress"] for d in rows.values()]
    TTC = [d["time_to_collision_within_bound"] for d in rows.values()]
    C = [d["comfort"] for d in rows.values()]
    S = [d["score"] for d in rows.values()]
    M = [a * b for a, b in zip(NC, DAC)]
    W = [(5 * e + 5 * t + 2 * c) / 12 for e, t, c in zip(EP, TTC, C)]
    passing = [i for i, m in enumerate(M) if m > 0]
    full_pass = [i for i, m in enumerate(M) if m == 1.0]
    out["A_summary_x100"] = {
        "PDMS": 100 * mean(S), "NC": 100 * mean(NC), "DAC": 100 * mean(DAC), "EP": 100 * mean(EP),
        "TTC": 100 * mean(TTC), "C": 100 * mean(C),
        "mean_M=NC*DAC": 100 * mean(M),
        "n_M0": n - len(passing), "n_M_half": sum(1 for m in M if 0 < m < 1),
        "n_NC_half": sum(1 for x in NC if x == 0.5), "n_NC_zero": sum(1 for x in NC if x == 0.0),
        "n_DAC_zero": sum(1 for x in DAC if x == 0.0),
        "EP_among_M>0": 100 * mean([EP[i] for i in passing]),
        "TTC_among_M>0": 100 * mean([TTC[i] for i in passing]),
        "C_among_M>0": 100 * mean([C[i] for i in passing]),
        "W_among_M>0": 100 * mean([W[i] for i in passing]),
        "M-weighted_W = PDMS/mean(M)": 100 * mean(S) / mean(M),
        "n_EP_eq_1_among_M>0": sum(1 for i in passing if EP[i] >= 0.9999),
        "n_TTC_zero_among_M>0": sum(1 for i in passing if TTC[i] == 0.0),
        "n_C_zero": sum(1 for x in C if x == 0.0),
    }

    # ---------------- B. paper-side estimate (ESTIMATED) ----------------
    # Paper rows have only means. Assumptions (stated): (i) EP = 0 on every M = 0 token (true by construction in
    # NAVSIM: raw progress is multiplied by NC*DAC before normalisation); (ii) the paper's M = 0 rate is
    # 1 - NC - DAC + 1 with our own NC=0.5 share and overlap ratio applied; (iii) TTC/C among passing ~ their means.
    nc_half_share = out["A_summary_x100"]["n_NC_half"] / max(1, out["A_summary_x100"]["n_NC_half"] + out["A_summary_x100"]["n_NC_zero"])
    def paper_side(nc, dac, ep, ttc, c, pdms):
        # NC deficit d_nc = share_zero*1 + share_half*0.5 -> fraction of zero-NC tokens
        d_nc = 1 - nc / 100
        f_nc0 = d_nc / ((1 - nc_half_share) + 0.5 * nc_half_share) * (1 - nc_half_share)
        f_nchalf = d_nc / ((1 - nc_half_share) + 0.5 * nc_half_share) * nc_half_share
        f_dac0 = 1 - dac / 100
        meanM = 1 - f_nc0 - 0.5 * f_nchalf - f_dac0  # overlap ignored (ours: 42/983 zero tokens fail both)
        f_zero = f_nc0 + f_dac0
        ep_pass = ep / 100 / (1 - f_zero)
        w_pass_from_means = (5 * ep_pass + 5 * ttc / 100 + 2 * c / 100) / 12
        return {"PDMS": pdms, "est_zero_rate": f_zero, "est_mean_M": meanM,
                "implied_M-weighted_W = PDMS/mean(M)": pdms / meanM,
                "est_EP_among_passing": 100 * ep_pass,
                "W_from_means_among_passing": 100 * w_pass_from_means,
                "mean_field_product": nc / 100 * dac / 100 * (5 * ep + 5 * ttc + 2 * c) / 12}

    paper = {
        "T7_DINOv3_ViT-S_93.88": paper_side(98.93, 99.01, 91.31, 95.77, 99.97, 93.88),
        "TA13_DINOv3_ViT-L_94.55": paper_side(98.80, 99.25, 92.72, 95.74, 99.94, 94.55),
        "T4_DriveZero_DriveVFM-L_94.8": paper_side(99.0, 99.2, 93.1, 96.0, 100.0, 94.8),
        "T4_Human_94.8": paper_side(100.0, 100.0, 87.5, 100.0, 99.9, 94.8),
        "T4_DriveRL_GT_95.8": paper_side(99.8, 99.9, 91.5, 99.1, 99.0, 95.8),
    }
    ours_mf = mean(NC) * mean(DAC) * (5 * mean(EP) + 5 * mean(TTC) + 2 * mean(C)) / 12
    out["B_paper_side_ESTIMATED"] = {"assumptions": "EP=0 on M=0 tokens (NAVSIM construction); NC=0.5 share and "
                                                    "overlap taken from OUR run; TTC/C among passing ~ their means",
                                     "rows": paper, "ours_mean_field_product_x100": 100 * ours_mf,
                                     "ours_actual": 100 * mean(S)}

    # Log-ratio split of the gap: PDMS = mean(M) * Wbar_M.
    ours_M = mean(M)
    ours_W = 100.0 * mean(S) / ours_M   # x100, same units as the paper-side implied W (bug fixed 2026-10-04)
    split = {}
    for k, p in paper.items():
        if "Human" in k or "GT" in k:
            continue
        lr_M = math.log(p["est_mean_M"] / ours_M)
        lr_W = math.log(p["implied_M-weighted_W = PDMS/mean(M)"] / ours_W)
        gap = p["PDMS"] - 100 * mean(S)
        tot = lr_M + lr_W
        # split the weighted part further by EP / TTC / C using passing-token means (linear in W)
        ours_ep, ours_ttc, ours_c = (mean([EP[i] for i in passing]), mean([TTC[i] for i in passing]),
                                     mean([C[i] for i in passing]))
        p_ep = p["est_EP_among_passing"] / 100
        # TTC/C among passing for the paper are unknown -> use means (conservative: TTC fails concentrate on M=0)
        dW_ep = 5 * (p_ep - ours_ep) / 12
        rowp ={"T7_DINOv3_ViT-S_93.88": (95.77, 99.97), "TA13_DINOv3_ViT-L_94.55": (95.74, 99.94),
                "T4_DriveZero_DriveVFM-L_94.8": (96.0, 100.0)}[k]
        dW_ttc = 5 * (rowp[0] / 100 - ours_ttc) / 12
        dW_c = 2 * (rowp[1] / 100 - ours_c) / 12
        dW_sum = dW_ep + dW_ttc + dW_c
        W_part = gap * lr_W / tot
        split[k] = {
            "gap_points": gap,
            "multiplicative_part_points (NC,DAC zeros)": gap * lr_M / tot,
            "weighted_part_points (EP,TTC,C on surviving tokens)": W_part,
            "weighted_part_split_by_linear_share": {
                "EP": W_part * dW_ep / dW_sum, "TTC": W_part * dW_ttc / dW_sum, "C": W_part * dW_c / dW_sum},
            "ours_passing_EP_TTC_C_x100": [100 * ours_ep, 100 * ours_ttc, 100 * ours_c],
        }
    out["B_gap_split_ESTIMATED"] = split

    # ---------------- C. census: harness validation + proposal/selection attribution ----------------
    acc = defaultdict(list)          # name -> list of (log, value)
    nzero_safe = 0
    n_tok = 0
    pick_fail_but_safe_exists = 0
    pick_fail = 0
    auc_num = auc_den = 0.0
    auc_sets = 0
    ep_fan_pass = []
    rank_corr_sum = 0.0
    rank_corr_n = 0
    human_ep_sum = 0.0
    human_ep_n = 0
    pdmc_ep_sum = 0.0
    for line in open(CENSUS):
        r = json.loads(line)
        log = r["log"]
        nc, dac, ttc, c, pdms = r["nc"], r["dac"], r["ttc"], r["c"], r["pdms"]
        agg = r["agg"]
        pick = r["pick"]
        hyp = list(range(1, 65))
        n_tok += 1
        acc["pdm_closed"].append((log, pdms[0]))
        acc["human"].append((log, pdms[65]))
        acc["pick"].append((log, pdms[1 + pick]))
        acc["mean_of_64"].append((log, sum(pdms[i] for i in hyp) / 64))
        acc["oracle_best_of_64"].append((log, max(pdms[i] for i in hyp)))
        for k_name, arr in [("NC", nc), ("DAC", dac), ("TTC", ttc), ("C", c)]:
            acc["human_" + k_name].append((log, arr[65]))
            acc["pdmclosed_" + k_name].append((log, arr[0]))
        # derive EP for human / pdm-closed / proposals where M>0
        def ep_of(i):
            m = nc[i] * dac[i]
            if m <= 0:
                return 0.0
            return max(0.0, min(1.0, (12 * pdms[i] / m - 5 * ttc[i] - 2 * c[i]) / 5))
        acc["human_EP"].append((log, ep_of(65)))
        acc["pdmclosed_EP"].append((log, ep_of(0)))
        safe = [i for i in hyp if nc[i] * dac[i] == 1.0]
        if not safe:
            nzero_safe += 1
            acc["oracle_safe_gate_then_scorer"].append((log, pdms[1 + pick]))
        else:
            best_agg = max(safe, key=lambda i: agg[i - 1])
            acc["oracle_safe_gate_then_scorer"].append((log, pdms[best_agg]))
        # EP oracle: among the scorer's own top-5, take the best GT
        top5 = sorted(hyp, key=lambda i: -agg[i - 1])[:5]
        acc["oracle_best_of_scorer_top5"].append((log, max(pdms[i] for i in top5)))
        top1 = 1 + pick
        if pdms[top1] == 0:
            pick_fail += 1
            if safe:
                pick_fail_but_safe_exists += 1
        # within-set AUC of agg for safe (M==1) vs unsafe
        pos = [agg[i - 1] for i in hyp if nc[i] * dac[i] == 1.0]
        neg = [agg[i - 1] for i in hyp if nc[i] * dac[i] < 1.0]
        if pos and neg:
            cnt = 0.0
            for a in pos:
                for b in neg:
                    cnt += 1.0 if a > b else (0.5 if a == b else 0.0)
            auc_num += cnt / (len(pos) * len(neg))
            auc_sets += 1
        # Spearman within set between agg and GT pdms
        g = [pdms[i] for i in hyp]
        a = agg
        def ranks(x):
            idx = sorted(range(len(x)), key=lambda j: x[j])
            rk = [0.0] * len(x)
            j = 0
            while j < len(x):
                k = j
                while k + 1 < len(x) and x[idx[k + 1]] == x[idx[j]]:
                    k += 1
                for q in range(j, k + 1):
                    rk[idx[q]] = (j + k) / 2.0
                j = k + 1
            return rk
        rg, ra = ranks(g), ranks(a)
        mg, ma = sum(rg) / 64, sum(ra) / 64
        num = sum((x - mg) * (y - ma) for x, y in zip(rg, ra))
        den = math.sqrt(sum((x - mg) ** 2 for x in rg) * sum((y - ma) ** 2 for y in ra))
        if den > 0:
            rank_corr_sum += num / den
            rank_corr_n += 1
        # fan EP among safe proposals
        eps = [ep_of(i) for i in safe]
        if eps:
            acc["fan_EP_mean_among_safe"].append((log, sum(eps) / len(eps)))
            acc["fan_EP_max_among_safe"].append((log, max(eps)))
        acc["fan_safe_share"].append((log, len(safe) / 64))
        acc["pick_EP"].append((log, ep_of(top1)))
        acc["pick_minus_human_EP"].append((log, ep_of(top1) - ep_of(65)))

    def boot(name, scale=100.0):
        by = defaultdict(list)
        for lg, v in acc[name]:
            by[lg].append(scale * v)
        return cluster_boot(by)

    keys = ["pdm_closed", "human", "pick", "mean_of_64", "oracle_best_of_64", "oracle_safe_gate_then_scorer",
            "oracle_best_of_scorer_top5", "human_NC", "human_DAC", "human_EP", "human_TTC", "human_C",
            "pdmclosed_NC", "pdmclosed_DAC", "pdmclosed_EP", "pdmclosed_TTC", "pdmclosed_C",
            "fan_EP_mean_among_safe", "fan_EP_max_among_safe", "fan_safe_share", "pick_EP", "pick_minus_human_EP"]
    out["C_census"] = {
        "n_tokens": n_tok,
        "estimator": "log-cluster bootstrap, 10,000, seed 20260927; x100",
        "means": {k: boot(k) for k in keys},
        "tokens_with_no_safe_proposal (NC*DAC<1 for all 64)": nzero_safe,
        "pick_PDMS_zero": pick_fail,
        "pick_zero_but_a_safe_proposal_existed": pick_fail_but_safe_exists,
        "agg_within_set_AUC_safe_vs_unsafe": {"mean": auc_num / max(1, auc_sets), "mixed_sets": auc_sets},
        "agg_vs_GT_pdms_within_set_spearman_mean": rank_corr_sum / max(1, rank_corr_n),
    }

    # ---------------- D. EP mechanics on the executed pick (hooks) ----------------
    hk = json.load(open(HOOKS))["pdm_score_calls"]
    cats = defaultdict(lambda: {"n": 0, "ep_loss": 0.0})
    vbins = defaultdict(lambda: {"n": 0, "ep_sum": 0.0, "pdms_sum": 0.0, "agent_prog": 0.0, "pdm_prog": 0.0,
                                 "path_len": 0.0, "zero": 0})
    tot_loss = 0.0
    n_pass = 0
    stops = 0
    ratio_hist = defaultdict(int)
    for h in hk:
        rowd = h["row"]
        m = rowd["no_at_fault_collisions"] * rowd["drivable_area_compliance"]
        v0 = h["v0_mps"]
        vb = ("0-1" if v0 < 1 else "1-5" if v0 < 5 else "5-10" if v0 < 10 else "10-15" if v0 < 15 else "15+")
        pp, ap = h["progress_raw_m"]
        poses = h["agent_poses"]
        L = 0.0
        px, py = 0.0, 0.0
        for x, y, _ in poses:
            L += math.hypot(x - px, y - py)
            px, py = x, y
        vbins[vb]["n"] += 1
        vbins[vb]["pdms_sum"] += rowd["score"]
        vbins[vb]["zero"] += 1 if rowd["score"] == 0 else 0
        if m <= 0:
            continue
        n_pass += 1
        ep = rowd["ego_progress"]
        loss = 1.0 - ep
        tot_loss += loss
        vbins[vb]["ep_sum"] += ep
        vbins[vb]["agent_prog"] += ap
        vbins[vb]["pdm_prog"] += pp
        vbins[vb]["path_len"] += L
        mx = h["max_compliant_progress_m"]
        if mx <= 5.0:
            cat = "max_progress<=5m (EP forced to 1)"
        elif ap >= pp:
            cat = "agent >= pdm-closed (EP=1)"
        else:
            # how was progress lost?  path long but projection short -> geometric (route/lateral);
            # path short -> longitudinal (slow / stop)
            if L < 1.0:
                cat = "stop (path < 1 m) while PDM-Closed > 5 m"
                stops += 1
            elif ap < 0.7 * L:
                cat = "geometric: projected progress < 70% of path length (wrong way / off centreline)"
            elif L < 0.9 * pp:
                cat = "longitudinal: path shorter than PDM-Closed progress (slower)"
            else:
                cat = "other"
        cats[cat]["n"] += 1
        cats[cat]["ep_loss"] += loss
        if mx > 5.0 and pp > 0:
            rb = min(10, int(10 * ap / pp))
            ratio_hist[rb] += 1
    out["D_EP_mechanics_on_pick"] = {
        "n_tokens_M>0": n_pass,
        "mean_EP_loss_x100_among_M>0": 100 * tot_loss / n_pass,
        "categories": {k: {"n": v["n"], "share_of_EP_loss": v["ep_loss"] / tot_loss,
                           "mean_EP_loss_x100_in_cat": 100 * v["ep_loss"] / max(1, v["n"])}
                       for k, v in sorted(cats.items(), key=lambda kv: -kv[1]["ep_loss"])},
        "agent_over_pdm_progress_ratio_hist_decile": {str(k / 10): v for k, v in sorted(ratio_hist.items())},
        "by_v0_bin": {k: {"n": v["n"], "PDMS_x100": 100 * v["pdms_sum"] / v["n"], "zero_share": v["zero"] / v["n"],
                          "EP_x100_among_M>0": 100 * v["ep_sum"] / max(1, (v["n"] - v["zero"])),
                          "mean_agent_progress_m": v["agent_prog"] / max(1, v["n"] - v["zero"]),
                          "mean_pdm_progress_m": v["pdm_prog"] / max(1, v["n"] - v["zero"]),
                          "mean_agent_path_m": v["path_len"] / max(1, v["n"] - v["zero"])}
                      for k, v in vbins.items()},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
