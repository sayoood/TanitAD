#!/usr/bin/env python3
"""LANE-1, the lane-centreline measure (lane_census2.py, the declared instrument fix) joined to lane_census.py's scores.

Same estimators and effect definitions as lane_analyze.py, with V = off the containing lane's centreline by > 1.0 m
(lk10) or > 1.5 m (lk15) for >= 1.0 s on non-junction lane steps. Adds ORACLE_mask_lk10 (drop hypotheses that violate;
map-based, an upper bound -- not deployable) to quantify what a lane-keeping label + selection could buy at most.
    python lane_analyze2.py [--partial]      -> lane2_result.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import lane_analyze as LA

HERE = os.path.dirname(os.path.abspath(__file__))


def analyze(rows1, rows2, label):
    b = {r["token"]: r for r in rows2}
    pairs = [(r, b[r["token"]]) for r in rows1 if r["token"] in b]
    n = len(pairs)
    logs = np.array([r["log"] for r, _ in pairs])
    pick = np.array([r["pick"] for r, _ in pairs]) + 1
    I = np.arange(n)
    pdms = np.array([r["pdms"] for r, _ in pairs], dtype=np.float64)
    agg = np.array([r["agg"] for r, _ in pairs], dtype=np.float64)
    ddc = np.array([r["ddc"] for r, _ in pairs], dtype=np.float64)
    lk10 = np.array([q["lk10_s"] for _, q in pairs], dtype=np.float64) >= 1.0
    lk15 = np.array([q["lk15_s"] for _, q in pairs], dtype=np.float64) >= 1.0
    one = np.ones(n)
    out = {"label": label, "n_tokens": n, "violation_rates": {}, "effects_human_clean": {}}
    for vk, M in (("lk10", lk10), ("lk15", lk15)):
        out["violation_rates"][vk] = {
            "human": LA.boot(M[:, LA.H].astype(float), one, logs),
            "pdm_closed": LA.boot(M[:, LA.PDM].astype(float), one, logs),
            "pick": LA.boot(M[I, pick].astype(float), one, logs),
            "fan_mean": LA.boot(M[:, 1:65].mean(1), one, logs),
            "fan_any": LA.boot(M[:, 1:65].any(1).astype(float), one, logs),
            "fan_all": LA.boot(M[:, 1:65].all(1).astype(float), one, logs)}
        hum_ok = ~M[:, LA.H]
        frac = M[:, 1:65].mean(1)
        p_bad = M[I, pick]
        alt = np.array([bool(np.any(~M[i, 1:65] & (pdms[i, 1:65] >= pdms[i, pick[i]] - 1e-9))) for i in range(n)])
        out["effects_human_clean"][vk] = {
            "human_clean_tokens": int(hum_ok.sum()),
            "FAN_effect": LA.boot((hum_ok & (frac >= 0.5)).astype(float), hum_ok.astype(float), logs),
            "SELECTION_effect_free_clean_alt": LA.boot((hum_ok & p_bad & alt).astype(float), hum_ok.astype(float), logs),
            "pick_violates_while_fan_mostly_clean": LA.boot((hum_ok & p_bad & (frac < 0.5)).astype(float), hum_ok.astype(float), logs),
            "pick_violates": LA.boot((hum_ok & p_bad).astype(float), hum_ok.astype(float), logs),
            "of_violating_picks_share_with_free_clean_alt": LA.boot((hum_ok & p_bad & alt).astype(float),
                                                                     (hum_ok & p_bad).astype(float), logs)}
    a = [LA.auc_within(agg[i], lk10[i, 1:65]) for i in range(n)]
    a = np.array([x for x in a if x is not None])
    out["scorer_within_set_auc"] = {"agg_vs_lk10": {"mixed_sets": int(len(a)), "mean_auc": round(float(a.mean()), 4) if len(a) else None}}
    m = np.where(~lk10[:, 1:65], agg, -1.0)
    rules = {"navsim_v1": pick - 1, "ORACLE_mask_lk10": np.where(m.max(1) >= 0, np.argmax(m, 1), pick - 1)}
    base = pdms[I, pick]
    out["selection_rules"] = {}
    for rk, rp in rules.items():
        q = rp + 1
        out["selection_rules"][rk] = {
            "pdms_x100": LA.boot(100 * pdms[I, q], one, logs),
            "paired_delta_x100": LA.boot(100 * (pdms[I, q] - base), one, logs),
            "changed_picks": int((q != pick).sum()),
            "lk10_rate": LA.boot(lk10[I, q].astype(float), one, logs),
            "onc_rate": LA.boot((ddc[I, q] < 1).astype(float), one, logs)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    r1, _ = LA.load(os.path.join(HERE, "lane_census.jsonl"))
    r2, e2 = LA.load(os.path.join(HERE, "lane_census2.jsonl"))
    changed = set(json.load(open(LA.CHANGED, encoding="utf-8")))
    res = {"what": "LANE-1 lane-centreline measure (lane_census2.py; declared instrument fix)", "partial": a.partial,
           "rows_census2": len(r2), "errors_census2": len(e2), "error_samples": [e["error"] for e in e2[:5]]}
    res["unchanged_by_routefix"] = analyze([r for r in r1 if r["token"] not in changed], r2, "routefix_unchanged (PRIMARY)")
    res["all_tokens"] = analyze(r1, r2, "all navtest tokens (pre-fix table)")
    json.dump(res, open(os.path.join(HERE, "lane2_result_partial.json" if a.partial else "lane2_result.json"), "w"), indent=1)
    p = res["unchanged_by_routefix"]
    print("n", p["n_tokens"], "errors", len(e2))
    for vk in ("lk10", "lk15"):
        print(vk, {k: (v["est"], v["lo"], v["hi"]) for k, v in p["violation_rates"][vk].items()})
        print("  effects", {k: (v["est"], v["lo"], v["hi"]) if isinstance(v, dict) else v for k, v in p["effects_human_clean"][vk].items()})
    print(p["scorer_within_set_auc"])
    for rk, v in p["selection_rules"].items():
        print(rk, v["pdms_x100"]["est"], (v["paired_delta_x100"]["est"], v["paired_delta_x100"]["lo"], v["paired_delta_x100"]["hi"]),
              "lk10", v["lk10_rate"]["est"], "onc", v["onc_rate"]["est"], "chg", v["changed_picks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
