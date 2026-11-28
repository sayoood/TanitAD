#!/usr/bin/env python3
"""LANE-1 analysis of lane_census.jsonl, exactly as PREREG_LANE1.md registers it. CPU, seconds.

Order inside every row: index 0 = PDM-Closed, 1..64 = the hypotheses (hypothesis k at k+1), 65 = the human future.
Every rate carries a log-cluster bootstrap CI (10,000 resamples, seed 20260927); paired deltas resample the same logs.
    python lane_analyze.py [--partial]      -> lane_result.json (+ prints a summary)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHANGED = os.path.join(HERE, "..", "2026-10-01-final-video", "routefix_changed_tokens.json")
B, SEED = 10000, 20260927
H, PDM = 65, 0


def boot(num, den, logs, rng_seed=SEED):
    """ratio-of-sums estimate with a log-cluster bootstrap; num/den are per-token arrays"""
    ul, inv = np.unique(logs, return_inverse=True)
    n_l = np.bincount(inv, weights=num, minlength=len(ul))
    d_l = np.bincount(inv, weights=den, minlength=len(ul))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(ul), size=(B, len(ul)))
    bs = n_l[idx].sum(1) / np.maximum(d_l[idx].sum(1), 1e-12)
    est = n_l.sum() / max(d_l.sum(), 1e-12)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": round(float(est), 5), "lo": round(float(lo), 5), "hi": round(float(hi), 5),
            "num": round(float(num.sum()), 3), "den": round(float(den.sum()), 3), "logs": int(len(ul))}


def auc_within(scores, labels_bad):
    """P(score of a clean hypothesis > score of a violating one), ties 1/2; None if one class is empty"""
    s_ok, s_bad = scores[~labels_bad], scores[labels_bad]
    if len(s_ok) == 0 or len(s_bad) == 0:
        return None
    gt = (s_ok[:, None] > s_bad[None, :]).mean()
    eq = (s_ok[:, None] == s_bad[None, :]).mean()
    return float(gt + 0.5 * eq)


def load(path):
    rows, errs = [], []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (errs if "error" in r else rows).append(r)
    return rows, errs


def arr(rows, key, fill=np.nan):
    return np.array([[fill if v is None else v for v in r[key]] for r in rows], dtype=np.float64)


def analyze(rows, label):
    logs = np.array([r["log"] for r in rows])
    n = len(rows)
    pick = np.array([r["pick"] for r in rows]) + 1
    I = np.arange(n)
    pdms, ddc, dac, nc = arr(rows, "pdms"), arr(rows, "ddc"), arr(rows, "dac"), arr(rows, "nc")
    ttc, c, strad = arr(rows, "ttc"), arr(rows, "c"), arr(rows, "strad_s")
    agg, pddc = arr(rows, "agg"), arr(rows, "p_ddc")
    V = {"onc": ddc < 1.0, "lane": strad >= 1.0, "lane2s": strad >= 2.0, "dac": dac < 1.0}
    one = np.ones(n)
    out = {"label": label, "n_tokens": n}
    # rates per trajectory class
    rates = {}
    for vk, M in V.items():
        rates[vk] = {
            "human": boot(M[:, H].astype(float), one, logs),
            "pdm_closed": boot(M[:, PDM].astype(float), one, logs),
            "pick": boot(M[I, pick].astype(float), one, logs),
            "fan_mean": boot(M[:, 1:65].mean(1), one, logs),
            "fan_any": boot(M[:, 1:65].any(1).astype(float), one, logs),
            "fan_all": boot(M[:, 1:65].all(1).astype(float), one, logs),
        }
    out["violation_rates"] = rates
    # the PI's two effects
    eff = {}
    for vk in ("onc", "lane", "lane2s"):
        M = V[vk]
        hum_ok = ~M[:, H]
        frac = M[:, 1:65].mean(1)
        p_bad = M[I, pick]
        clean_alt = np.array([bool(np.any(~M[i, 1:65] & (pdms[i, 1:65] >= pdms[i, pick[i]] - 1e-9))) for i in range(n)])
        fan_eff = hum_ok & (frac >= 0.5)
        sel_eff = hum_ok & p_bad & clean_alt
        sel_fan_ok = hum_ok & p_bad & (frac < 0.5)
        eff[vk] = {
            "human_clean_tokens": int(hum_ok.sum()),
            "FAN_effect": boot(fan_eff.astype(float), hum_ok.astype(float), logs),
            "SELECTION_effect_free_clean_alt": boot(sel_eff.astype(float), hum_ok.astype(float), logs),
            "pick_violates_while_fan_mostly_clean": boot(sel_fan_ok.astype(float), hum_ok.astype(float), logs),
            "pick_violates": boot((hum_ok & p_bad).astype(float), hum_ok.astype(float), logs),
            "of_violating_picks_share_with_free_clean_alt": boot(sel_eff.astype(float), (hum_ok & p_bad).astype(float), logs),
        }
    out["effects_human_clean"] = eff
    # does the scorer already see it? within-set AUC
    heads = {}
    for name, S, vk in (("p_ddc_vs_onc", pddc, "onc"), ("agg_vs_onc", agg, "onc"), ("agg_vs_lane", agg, "lane"),
                        ("p_ddc_vs_lane", pddc, "lane")):
        a = [auc_within(S[i], V[vk][i, 1:65]) for i in range(n)]
        a = np.array([x for x in a if x is not None])
        heads[name] = {"mixed_sets": int(len(a)), "mean_auc": round(float(a.mean()), 4) if len(a) else None,
                       "share_auc_gt_0.5": round(float((a > 0.5).mean()), 4) if len(a) else None}
    out["scorer_within_set_auc"] = heads
    # selection rules (L1 registered; L2 reported only; oracles = upper bounds, not deployable)
    rules = {"navsim_v1": pick - 1, "L1_x_pddc": np.argmax(agg * pddc, 1)}
    g = np.where(pddc >= 0.5, agg, -1.0)
    rules["L2_gate_pddc_0.5"] = np.where(g.max(1) >= 0, np.argmax(g, 1), pick - 1)
    for vk in ("onc", "lane"):
        m = np.where(~V[vk][:, 1:65], agg, -1.0)
        rules[f"ORACLE_mask_{vk}"] = np.where(m.max(1) >= 0, np.argmax(m, 1), pick - 1)
    rules["ORACLE_best_pdms"] = np.argmax(pdms[:, 1:65], 1)
    sel = {}
    base = pdms[I, pick]
    for rk, rp in rules.items():
        q = rp + 1
        d = pdms[I, q] - base
        sel[rk] = {"pdms_x100": boot(100 * pdms[I, q], one, logs),
                   "paired_delta_x100": boot(100 * d, one, logs),
                   "changed_picks": int((q != pick).sum())}
        for vk in ("onc", "lane", "dac"):
            sel[rk][f"{vk}_rate"] = boot(V[vk][I, q].astype(float), one, logs)
            sel[rk][f"{vk}_rate_delta"] = boot(V[vk][I, q].astype(float) - V[vk][I, pick].astype(float), one, logs)
        for nm, A in (("nc0", nc < 1), ("ttc0", ttc < 1), ("c0", c < 1)):
            sel[rk][f"{nm}_rate"] = boot(A[I, q].astype(float), one, logs)
    out["selection_rules"] = sel
    l1 = sel["L1_x_pddc"]
    out["L1_verdict"] = {
        "pdms_delta_lo_gt_-0.10": l1["paired_delta_x100"]["lo"] > -0.10,
        "onc_rate_delta_hi_lt_0": l1["onc_rate_delta"]["hi"] < 0,
    }
    out["L1_verdict"]["ADOPT"] = all(out["L1_verdict"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    rows, errs = load(os.path.join(HERE, "lane_census.jsonl"))
    changed = set(json.load(open(CHANGED, encoding="utf-8")))
    if isinstance(next(iter(changed)), dict):
        changed = {d["token"] for d in changed}
    ctrl = np.array([abs(r["pdms"][r["pick"] + 1] - r["csv_pdms"]) < 1e-3 if r.get("csv_pdms") is not None else False
                     for r in rows])
    res = {"what": "LANE-1 (PREREG_LANE1.md, landed e983685)", "partial": a.partial, "rows": len(rows),
           "errors": len(errs), "error_samples": [e["error"] for e in errs[:5]],
           "control_repro_eq_csv": {"agree": int(ctrl.sum()), "of": len(rows),
                                    "disagree_tokens": [r["token"] for r, ok in zip(rows, ctrl) if not ok][:60]},
           "control_pick_ok": int(sum(r["pick_ok"] for r in rows))}
    res["unchanged_by_routefix"] = analyze([r for r in rows if r["token"] not in changed], "routefix_unchanged (PRIMARY)")
    res["all_tokens"] = analyze(rows, "all navtest tokens (pre-fix table)")
    json.dump(res, open(os.path.join(HERE, "lane_result_partial.json" if a.partial else "lane_result.json"), "w"), indent=1)
    p = res["unchanged_by_routefix"]
    print(json.dumps({k: res[k] for k in ("rows", "errors", "control_repro_eq_csv", "control_pick_ok")})[:600])
    for vk in ("onc", "lane"):
        r = p["violation_rates"][vk]
        print(vk, {k: (v["est"], v["lo"], v["hi"]) for k, v in r.items()})
        print("  effects", {k: (v["est"], v["lo"], v["hi"]) if isinstance(v, dict) else v for k, v in p["effects_human_clean"][vk].items()})
    print("heads", p["scorer_within_set_auc"])
    for rk, v in p["selection_rules"].items():
        print(rk, "pdms", v["pdms_x100"]["est"], "d", (v["paired_delta_x100"]["est"], v["paired_delta_x100"]["lo"], v["paired_delta_x100"]["hi"]),
              "onc", v["onc_rate"]["est"], "lane", v["lane_rate"]["est"], "dac", v["dac_rate"]["est"], "chg", v["changed_picks"])
    print("L1", p["L1_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
