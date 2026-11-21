#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The RESULT tables, computed (never typed): attribution ladder deltas at b1 / b2, per-sample slope,
linear b16 projection on the dev box, shares of the R7 - R6 delta, and the Thor mapping.

R7 REFERENCE, stated: R7 was run 3x plain at b2 (2.110 / 1.986 / 2.039 s marginal) -- the first
run sat in the tail of another session's GPU/CPU work (IQR 1.87-2.03 vs <=0.04 for every other
config) -- and the five in-process A/B runs carry R7 in their OFF blocks. The reference is the
MEDIAN of the steady R7 steps pooled over rep, rep2 and the A/B OFF halves, plus the MEASURED
amortised conflict extra (median over the same runs / 10). The perturbed first run is reported,
never silently dropped, and the deltas against it are printed beside the reference ones."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RAW = Path(sys.argv[1])          # C:/lgt/rc7cost_runs/gpu
OUT = Path(sys.argv[2])


def steps(job):
    p = RAW / job / "steps.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def prof(job):
    ps = list((RAW / job).glob("profile_*.json"))
    return json.loads(ps[0].read_text(encoding="utf-8")) if ps else None


def split(job, want_ab=None):
    pr, rows = prof(job), steps(job)
    if not pr or not rows or pr.get("error"):
        return None
    a = pr["args"]
    warm, n, le = int(a["warm"]), int(a["steps"]), int(a["log_every"])
    cd_off = "--conflict-detector" in pr["spec"]["argv"] and \
        pr["spec"]["argv"][pr["spec"]["argv"].index("--conflict-detector") + 1] == "off"
    st, cf = [], []
    for r in rows:
        k = int(r["step_index"])
        if k < warm or k >= n - 1 or (k + 1) % le == 0 or "wall" not in r:
            continue
        if want_ab is not None and int(r.get("ab_on", 0)) != want_ab:
            continue
        c = r["wall"] - r.get("data_wait", 0.0)
        (cf if (not cd_off and k % 10 == 0) else st).append(c)
    return {"steady": st, "conflict": cf, "peak": pr.get("peak_mem_gb"), "batch": int(a["batch"])}


def marginal(sp):
    s = statistics.median(sp["steady"])
    if sp["conflict"]:
        return s + (statistics.median(sp["conflict"]) - s) / 10.0, s, statistics.median(sp["conflict"]) - s
    return s, s, 0.0


def main():
    res = {}
    # ---- R7 reference at b2 and b1
    pool_st, pool_cf = [], []
    for j, ab in (("R7_plain_b2_rep", None), ("R7_plain_b2_rep2", None), ("ab_percls_census_b2", 0),
                  ("ab_percls_b2", 0), ("ab_census_b2", 0), ("ab_cudnn_b2", 0), ("ab_scipy_b2", 0)):
        sp = split(j, ab)
        if sp:
            pool_st += sp["steady"]
            pool_cf += sp["conflict"] if ab is None else []
    s7 = statistics.median(pool_st)
    cx7 = statistics.median(pool_cf) - s7
    r7b2 = s7 + cx7 / 10
    first = marginal(split("R7_plain_b2"))[0]
    p1s, p1c = [], []
    for j, ab in (("R7_plain_b1", None), ("R7_plain_b1_rep", None), ("ab_percls_census_b1", 0),
                  ("ab_ckpt_off_b1", 0)):
        sp = split(j, ab)
        if sp:
            p1s += sp["steady"]
            p1c += sp["conflict"] if ab is None else []
    s71 = statistics.median(p1s)
    r7b1 = s71 + (statistics.median(p1c) - s71) / 10
    res["R7_reference"] = {"b2_marginal_s": round(r7b2, 4), "b2_steady_s": round(s7, 4),
                           "b2_conflict_extra_s": round(cx7, 4), "n_pooled_steady": len(pool_st),
                           "b2_first_run_marginal_s_perturbed": round(first, 4),
                           "b1_marginal_s": round(r7b1, 4), "b1_steady_s": round(s71, 4),
                           "n_pooled_steady_b1": len(p1s)}
    rows = {}
    for rung in ("R6", "map_v6", "box_v6", "no_priors", "no_near", "q100", "no_deepsup", "bce",
                 "no_vis1", "learned", "conflict_off"):
        row = {}
        vals = []
        for j in (f"{rung}_plain_b2", f"{rung}_plain_b2_rep"):
            sp = split(j)
            if sp:
                vals.append(marginal(sp)[0])
        if vals:
            m2 = statistics.median(vals)
            row["b2_marginal_s"] = [round(v, 4) for v in vals]
            row["delta_b2_s"] = round(r7b2 - m2, 4)
            row["delta_b2_vs_perturbed_first_R7_s"] = round(first - m2, 4)
        v1 = [marginal(x)[0] for x in (split(f"{rung}_plain_b1"), split(f"{rung}_plain_b1_rep")) if x]
        if v1:
            m1 = statistics.median(v1)
            row["b1_runs_marginal_s"] = [round(v, 4) for v in v1]
            row["b1_marginal_s"] = round(m1, 4)
            row["delta_b1_s"] = round(r7b1 - m1, 4)
            if "delta_b2_s" in row:
                sl = row["delta_b2_s"] - row["delta_b1_s"]
                row["slope_per_sample_s"] = round(sl, 4)
                row["fixed_s"] = round(row["delta_b1_s"] - sl, 4)
                row["linear_b16_devbox_s"] = round(row["fixed_s"] + 16 * sl, 3)
        rows[rung] = row
    res["ladder"] = rows
    tot = rows.get("R6", {}).get("linear_b16_devbox_s")
    if tot:
        res["shares_of_R7_minus_R6_at_b16_devbox"] = {
            k: round(rows[k]["linear_b16_devbox_s"] / tot, 3)
            for k in ("map_v6", "box_v6", "no_priors") if "linear_b16_devbox_s" in rows.get(k, {})}
    tot2 = rows.get("R6", {}).get("delta_b2_s")
    if tot2:
        res["shares_of_R7_minus_R6_at_b2_devbox"] = {
            k: round(rows[k]["delta_b2_s"] / tot2, 3)
            for k in ("map_v6", "box_v6", "no_priors", "no_near", "q100", "no_deepsup", "bce",
                      "no_vis1", "learned") if "delta_b2_s" in rows.get(k, {})}
    # ---- A/B levers
    ab = {}
    for j in ("ab_percls_census_b2", "ab_percls_census_b1", "ab_percls_b2", "ab_census_b2",
              "ab_cudnn_b2", "ab_scipy_b2", "ab_ckpt_off_b1"):
        p = RAW / j / "ab_summary.json"
        if p.exists():
            ab[j] = json.loads(p.read_text(encoding="utf-8"))
    res["ab_levers"] = ab
    # ---- conflict probe
    cfx = {}
    for j in ("R7_plain_b2_rep", "R7_plain_b2_rep2", "R7_plain_b1", "R6_plain_b2", "R6_plain_b2_rep",
              "R6_plain_b1"):
        sp = split(j)
        if sp and sp["conflict"]:
            m, s, x = marginal(sp)
            cfx[j] = {"steady_s": round(s, 4), "conflict_extra_s": round(x, 4),
                      "extra_over_steady": round(x / s, 3), "n_conflict_steps": len(sp["conflict"])}
    res["conflict_probe"] = cfx
    # ---- memory
    res["peak_mem_gb"] = {j: (prof(j) or {}).get("peak_mem_gb") for j in
                          ("R7_plain_b1", "ab_ckpt_off_b1", "R7_plain_b2_rep", "R6_plain_b2",
                           "map_v6_plain_b2", "ckpt_off_plain_b2", "conflict_off_plain_b2")}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
