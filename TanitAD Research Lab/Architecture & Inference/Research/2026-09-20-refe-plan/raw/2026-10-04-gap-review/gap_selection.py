#!/usr/bin/env python3
"""REVIEW 7 -- which scorer head costs the most, and what a deployable selection fix could buy (CPU only).

Run with the navsim venv python (needs numpy):
    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe gap_selection.py

Inputs (read-only, MEASURED):
  PROPS  data/refe_navtest/proptable/navtest_final/proposals.npz  -- the final model's 64 proposals (8 poses @ 0.5 s)
         and its stored per-head logits (order = train.ScorerBank.COMPONENTS: nc, dac, ep, ttc, c, ddc) and pick
  CENSUS raw/2026-10-04-lane-discipline/lane_census.jsonl         -- NAVSIM v1.1 GT nc/dac/ttc/c/pdms per proposal
         (0 = PDM-Closed, 1..64 = hypotheses, 65 = human), pdms reproduced pairwise (EP vs PDM-Closed only)

What it computes (all on the 12,146 navtest tokens; GT EP derived as (12*pdms/(nc*dac) - 5ttc - 2c)/5 when nc*dac>0):
  * reproduction control: navsim_v1 aggregate of the stored logits must reproduce the stored pick on every token
  * PARTIAL ORACLES (diagnostic upper bounds, not deployable): replace ONE predicted head by its NAVSIM GT and re-select
  * DEPLOYABLE RULES (EXPLORATORY, no GT at selection): use the proposal's own geometry (path length) as a progress
    signal. Any rule with a tuned knob is CROSS-FITTED: logs split in two halves by a fixed hash, the knob chosen on
    one half and scored on the other, both directions pooled -- the scored half never tunes its own knob.
  * descriptive: pick-slot concentration, pick path length vs the fan and vs the human

Intervals: log-cluster bootstrap of the per-token mean (10,000, seed 20260927), paired deltas vs the shipped pick.
They answer "another draw of LOGS" only. EVERYTHING here is EXPLORATORY: navtest has been read many times.
"""
import hashlib
import json
import math
import os

import numpy as np

ROOT = "E:/Projects/TanitAD"
PKG = ROOT + "/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PROPS = ROOT + "/data/refe_navtest/proptable/navtest_final/proposals.npz"
CENSUS = PKG + "/raw/2026-10-04-lane-discipline/lane_census.jsonl"
OUT = PKG + "/raw/2026-10-04-gap-review/gap_selection.json"
B, SEED = 10000, 20260927


def sig(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def agg(nc, dac, ep, ttc, c):
    return nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12.0


def boot_paired(vals, base, logs, B=B, seed=SEED):
    """log-cluster bootstrap of mean(vals) and of mean(vals - base); returns x100."""
    ul, inv = np.unique(logs, return_inverse=True)
    n_l = len(ul)
    s_v = np.bincount(inv, weights=vals, minlength=n_l)
    s_d = np.bincount(inv, weights=vals - base, minlength=n_l)
    cnt = np.bincount(inv, minlength=n_l).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_l, size=(B, n_l))
    c_b = cnt[idx].sum(1)
    mv = s_v[idx].sum(1) / c_b
    md = s_d[idx].sum(1) / c_b
    q = lambda a: [float(np.quantile(a, 0.025) * 100), float(np.quantile(a, 0.975) * 100)]  # noqa: E731
    return {"mean_x100": float(vals.mean() * 100), "ci": q(mv),
            "delta_vs_pick_x100": float((vals - base).mean() * 100), "delta_ci": q(md),
            "n": int(len(vals)), "logs": int(n_l)}


def main():
    z = np.load(PROPS, allow_pickle=True)
    tok = z["token"]
    props = z["proposals"].astype(np.float64)          # [N, 64, 8, 3]
    logit = z["logits"]
    pick_st = z["pick"]
    pos = {t: i for i, t in enumerate(tok)}
    N = len(tok)
    GT = {k: np.full((N, 64), np.nan) for k in ("nc", "dac", "ttc", "c", "pdms", "ep")}
    H = {k: np.full(N, np.nan) for k in ("pdms", "ep")}
    logs = np.empty(N, dtype=object)
    found = 0
    for line in open(CENSUS):
        r = json.loads(line)
        i = pos.get(r["token"])
        if i is None:
            continue
        found += 1
        logs[i] = r["log"]
        for k in ("nc", "dac", "ttc", "c", "pdms"):
            GT[k][i] = np.asarray(r[k][1:65], float)
        m = GT["nc"][i] * GT["dac"][i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ep = np.where(m > 0, (12 * GT["pdms"][i] / np.where(m > 0, m, 1) - 5 * GT["ttc"][i] - 2 * GT["c"][i]) / 5, 0.0)
        GT["ep"][i] = np.clip(ep, 0, 1)
        hm = r["nc"][65] * r["dac"][65]
        H["pdms"][i] = r["pdms"][65]
        H["ep"][i] = (np.clip((12 * r["pdms"][65] / hm - 5 * r["ttc"][65] - 2 * r["c"][65]) / 5, 0, 1) if hm > 0 else 0.0)
    assert found == N, (found, N)
    logs = logs.astype(str)
    P = sig(logit)                                      # [N, 64, 6]
    pnc, pdac, pep, pttc, pc = (P[..., k] for k in range(5))
    rows = np.arange(N)

    def score_of(sel):
        return GT["pdms"][rows, sel]

    # ---- reproduction control
    a0 = agg(pnc, pdac, pep, pttc, pc)
    sel0 = a0.argmax(1)
    repro = int((sel0 == pick_st).sum())
    base = score_of(pick_st)
    out = {"n_tokens": N, "control_navsim_v1_on_stored_logits_reproduces_pick": f"{repro}/{N}",
           "estimator": "log-cluster bootstrap, 10,000, seed 20260927; paired vs the shipped pick; EXPLORATORY",
           "pick": boot_paired(base, base, logs)}

    # ---- partial oracles (diagnostic, NOT deployable)
    g = GT
    orc = {
        "oracle_NC": agg(g["nc"], pdac, pep, pttc, pc),
        "oracle_DAC": agg(pnc, g["dac"], pep, pttc, pc),
        "oracle_NC+DAC": agg(g["nc"], g["dac"], pep, pttc, pc),
        "oracle_EP": agg(pnc, pdac, g["ep"], pttc, pc),
        "oracle_TTC": agg(pnc, pdac, pep, g["ttc"], pc),
        "oracle_C": agg(pnc, pdac, pep, pttc, g["c"]),
        "oracle_EP+TTC": agg(pnc, pdac, g["ep"], g["ttc"], pc),
        "oracle_NC+DAC+TTC": agg(g["nc"], g["dac"], pep, g["ttc"], pc),
        "oracle_ALL (= best of 64)": g["pdms"],
    }
    out["partial_oracles"] = {k: boot_paired(score_of(v.argmax(1)), base, logs) for k, v in orc.items()}

    # ---- geometry: path length of each proposal (the model's OWN output; vision-only admissible)
    xy = props[..., :2]
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 64, 1, 2)), xy], axis=2), axis=2), axis=-1)
    L = seg.sum(-1)                                     # [N, 64] metres over 4 s
    Lmax = L.max(1, keepdims=True)
    ep_geo = np.where(Lmax > 5.0, L / np.maximum(Lmax, 1e-6), 1.0)   # NAVSIM's own <=5 m rule, self-normalised

    # ---- descriptive
    pick_L = L[rows, pick_st]
    slot_counts = np.bincount(pick_st, minlength=64)
    top = np.argsort(-slot_counts)[:5]
    # human path length approximated by GT EP relation is not available; use fan medians instead
    rank_of_pick = (L > pick_L[:, None]).sum(1)          # how many proposals are LONGER than the pick
    out["descriptive"] = {
        "pick_slot_top5": {int(s): int(slot_counts[s]) for s in top},
        "distinct_slots_ever_picked": int((slot_counts > 0).sum()),
        "pick_path_over_fan_mean_path": float(np.mean(pick_L / np.maximum(L.mean(1), 1e-6))),
        "pick_path_over_fan_median_path": float(np.median(pick_L / np.maximum(np.median(L, 1), 1e-6))),
        "median_rank_of_pick_by_length_(0=longest)": float(np.median(rank_of_pick)),
        "share_tokens_pick_shorter_than_fan_median": float(np.mean(pick_L < np.median(L, 1))),
        "mean_GT_EP_pick_among_M>0": float(np.nanmean(np.where(g["nc"][rows, pick_st] * g["dac"][rows, pick_st] > 0,
                                                                g["ep"][rows, pick_st], np.nan))),
        "mean_GT_EP_human": float(np.mean(H["ep"])),
        "human_PDMS_x100": float(np.mean(H["pdms"]) * 100),
        "corr_within_set_pred_EP_vs_GT_EP_mean": None,
        "corr_within_set_pred_EP_vs_path_length_mean": None,
    }
    # within-set correlations of the predicted EP head with GT EP and with path length
    def wcorr(a, b):
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        ok = den > 1e-12
        return float(np.mean((a * b).sum(1)[ok] / den[ok])), int(ok.sum())
    out["descriptive"]["corr_within_set_pred_EP_vs_GT_EP_mean"] = wcorr(pep, g["ep"])
    out["descriptive"]["corr_within_set_pred_EP_vs_path_length_mean"] = wcorr(pep, L)
    out["descriptive"]["corr_within_set_GT_EP_vs_path_length_mean"] = wcorr(g["ep"], L)
    out["descriptive"]["corr_within_set_pred_agg_vs_path_length_mean"] = wcorr(a0, L)
    out["descriptive"]["corr_within_set_GT_pdms_vs_path_length_mean"] = wcorr(g["pdms"], L)
    out["descriptive"]["mean_pred_EP_prob_on_pick"] = float(pep[rows, pick_st].mean())
    out["descriptive"]["mean_pred_EP_prob_all"] = float(pep.mean())

    # ---- deployable rules, cross-fitted over two log halves
    half = np.array([int(hashlib.sha256(l.encode()).hexdigest(), 16) % 2 for l in logs])

    def rule_family(name, fn, knobs):
        """fn(knob) -> aggregate [N,64]; choose knob on one half, score on the other."""
        sel_by_knob = {k: fn(k).argmax(1) for k in knobs}
        sc_by_knob = {k: score_of(s) for k, s in sel_by_knob.items()}
        chosen = {}
        final = np.empty(N)
        for h in (0, 1):
            fit = half == (1 - h)
            best = max(knobs, key=lambda k: sc_by_knob[k][fit].mean())
            chosen[str(h)] = best
            final[half == h] = sc_by_knob[best][half == h]
        res = {"crossfit": boot_paired(final, base, logs), "knob_chosen_per_scored_half": chosen,
               "per_knob_full_set_x100_NOT_crossfit": {str(k): float(sc_by_knob[k].mean() * 100) for k in knobs}}
        return res

    rules = {}
    rules["D1_replace_pred_EP_by_self_normalised_path_length"] = {
        "full_set": boot_paired(score_of(agg(pnc, pdac, ep_geo, pttc, pc).argmax(1)), base, logs)}
    rules["D2_agg_times_lengthratio_pow_alpha"] = rule_family(
        "D2", lambda a: a0 * np.power(ep_geo, a), [0.0, 0.25, 0.5, 1.0, 2.0])
    rules["D3_blend_pred_EP_with_geo_EP_w"] = rule_family(
        "D3", lambda w: agg(pnc, pdac, (1 - w) * pep + w * ep_geo, pttc, pc), [0.0, 0.25, 0.5, 0.75, 1.0])
    # D4: safety gate on predicted multiplicative probability, then maximise the weighted part with geo EP
    def d4(tau):
        safe = (pnc * pdac) >= tau
        w = (5 * ep_geo + 5 * pttc + 2 * pc) / 12.0
        # tokens where nothing passes the gate fall back to the shipped aggregate
        none = ~safe.any(1, keepdims=True)
        return np.where(none, a0, np.where(safe, w + 1.0, w - 1.0))
    rules["D4_gate_pnc_pdac_ge_tau_then_max_weighted_with_geo_EP"] = rule_family(
        "D4", d4, [0.5, 0.7, 0.8, 0.9, 0.95])
    # D5: re-normalise EP the way NAVSIM does: path length over the longest proposal the scorer believes safe
    def d5(tau):
        safe = (pnc * pdac) >= tau
        Ls = np.where(safe, L, 0.0).max(1, keepdims=True)
        ep5 = np.where(Ls > 5.0, np.minimum(L / np.maximum(Ls, 1e-6), 1.0), 1.0)
        return agg(pnc, pdac, ep5, pttc, pc)
    rules["D5_EP_as_length_over_longest_believed_safe"] = rule_family("D5", d5, [0.5, 0.7, 0.8, 0.9])
    out["deployable_rules_EXPLORATORY"] = rules

    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
