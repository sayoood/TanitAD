"""BAR-M7-1..4 and BAR-B7-1/2 (SPEC.md §3.4 / §3.5) from `perc_pass_r7.py`'s banked outputs. CPU only.

    python perc_score.py --prefix <perc prefix> --out <json> [--milestone] [--n-boot 2000]

MAP: `taniteval.map_hires_metrics.summarize` + `evaluate_bars_m7` (landed, unchanged) on the WindowTable.
BOX: the trainer's own `detection_metrics` on the per-window packs. Paired clip-cluster bootstrap
(`taniteval.ci.paired_episode_cluster_bootstrap`, callable reducer over window indices -- the same path
`map_hires_metrics.paired_delta` uses). Per window the greedy rows are computed ONCE (they depend on the
window alone), and each draw re-pools them:
  * BAR-B7-1: `eval_box3d_ap2m` (= `det_ap2_all_all`, AP@2 m over pooled classes) refcv7 - refcv6@38k;
  * BAR-B7-2: F1 @ the declared gate (sigma >= 0.5) for refcv7 vs refcv6@38k's F1 @ ITS best-F1 gate,
    that gate chosen on the same scored windows (favours the baseline) and held fixed in the bootstrap.
⛔ Controls that must read known values (else the box block is VOID): the per-window census summed over
windows equals the pooled census exactly (additivity), and the pooled AP computed here equals
`detection_metrics.summarise`'s `ap2m` on the full set exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--milestone", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    from taniteval import ci
    from taniteval import map_hires_metrics as MH
    from tanitad.eval import detection_metrics as dm
    t0 = time.time()
    meta = json.load(open(a.prefix + ".json", encoding="utf-8"))
    out = {"tool": "perc_score.py", "perc_record": meta, "is_milestone": bool(a.milestone),
           "surface": "M = S2 windows (SPEC AMENDMENT A1: departs from SPEC_REFCV7 §6.2's "
                      "'every eval window'; the full 23,772-window index is the registered upgrade)",
           "estimator": "paired_episode_cluster_bootstrap over clips, n_boot %d, seed 0" % a.n_boot}
    # ------------------------------------------------------------------ MAP ------ #
    table = MH.WindowTable.load(a.prefix + "_map_table.npz")
    out["map"] = {"summary": MH.summarize(table, n_boot=a.n_boot, seed=0)}
    if {"refcv7", "refcv6_38k", "prior"} <= set(table.arms):
        bars = MH.evaluate_bars_m7(table, arm="refcv7", base="refcv6_38k", prior="prior",
                                   n_boot=a.n_boot, seed=0)
        if not a.milestone:
            for k in ("BAR-M7-1", "BAR-M7-2", "BAR-M7-3", "BAR-M7-4"):
                bars[k]["verdict_if_milestone"] = bars[k]["verdict"]
                bars[k]["verdict"] = "NOT EVALUATED (pipeline validation checkpoint; SPEC §6)"
        for k in ("BAR-M7-1", "BAR-M7-2", "BAR-M7-3", "BAR-M7-4"):
            bars[k]["single_seed_note"] = ("no replicate floor is measured for the map; the "
                                           "2x-floor NOT PROVEN rule cannot be applied (SPEC §3.5)")
            bars[k]["surface_departure"] = "S2 windows, not every eval window (SPEC A1)"
        out["map"]["bars"] = bars
    else:
        out["map"]["bars"] = {"status": "NOT EVALUABLE", "arms_present": list(table.arms),
                              "reason": "the refcv6@38k dump and/or the positional prior were not "
                                        "passed to the perception pass"}
    # ------------------------------------------------------------------ BOX ------ #
    det = pickle.load(open(a.prefix + "_det.pkl", "rb"))
    rows7, rows6, eid = [], [], []
    for r in det:
        if r.get("box3d") and r.get("box3d_refcv6_38k"):
            rows7.append(r["box3d"][0])
            rows6.append(r["box3d_refcv6_38k"][0])
            eid.append(r["sha12"])
    box = {"n_windows": len(eid), "n_clips": len(set(eid))}
    packs7 = [r["box3d"][0] for r in det if r.get("box3d")]
    box["refcv7_all_windows"] = {k: v for k, v in dm.summarise(packs7, "box3d").items()
                                 if not k.startswith("eval_box3d_det_ap") or k.endswith("_all_all")} \
        if packs7 else None
    if not eid:
        box["bars"] = {"status": "NOT EVALUABLE", "reason": "no window carries both models' box3d packs"}
        out["box"] = box
    else:
        g7 = float(dm._gate())
        W = len(eid)
        full7 = dm.summarise(rows7, "box3d")
        full6 = dm.summarise(rows6, "box3d")
        box["refcv7"] = full7
        box["refcv6_38k"] = full6
        box["conf_ratio_watch"] = {
            "band": list(dm.CONF_RATIO_BAND), "refcv7": full7.get("eval_box3d_conf_ratio"),
            "alarm": full7.get("eval_box3d_conf_ratio_alarm"),
            "note": "A10 §15.3: a Watch ALARM outside [0.5, 1.5], never a bar"}
        # per-window greedy rows at 2 m (AP) and the census at a gate, computed ONCE per window
        ap_rows = [dm.greedy_rows(pk, 2.0, None) for pk in rows7] + \
                  [dm.greedy_rows(pk, 2.0, None) for pk in rows6]
        npos = [int(pk["pos"].sum()) for pk in rows7] + [int(pk["pos"].sum()) for pk in rows6]

        def ap_red(idx):
            rr = [x for i in np.asarray(idx, np.int64) for x in ap_rows[i]]
            return dm._ap(rr, sum(npos[i] for i in np.asarray(idx, np.int64)))
        ap_red.__name__ = "pooled_ap2m"
        # control: our pooled AP == summarise's ap2m, exactly
        ap7_here, ap6_here = ap_red(np.arange(W)), ap_red(np.arange(W, 2 * W))
        ctl_ap = {"refcv7": [ap7_here, full7["eval_box3d_ap2m"]],
                  "refcv6_38k": [ap6_here, full6["eval_box3d_ap2m"]]}
        ctl_ap_ok = all((math.isnan(x) and math.isnan(y)) or x == y for x, y in ctl_ap.values())

        def census(packs, gate):
            per = []
            for pk in packs:
                c, _ = dm._census([pk], "box3d", "", gate)
                per.append((c["box3d_tp@gate"], c["box3d_n_conf"], c["box3d_n_pos"]))
            return np.asarray(per, np.float64)
        # refcv6's best-F1 gate on the SAME windows (favours the baseline)
        grid = [round(0.05 + 0.01 * i, 2) for i in range(91)]
        best = None
        for g in grid:
            c = census(rows6, g).sum(0)
            p, r = (c[0] / c[1] if c[1] else math.nan), (c[0] / c[2] if c[2] else math.nan)
            f = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else math.nan
            if f == f and (best is None or f > best[1]):
                best = (g, f)
        g6 = best[0] if best else 0.5
        cen = np.concatenate([census(rows7, g7), census(rows6, g6)])
        pooled7 = dm._census(rows7, "box3d", "", g7)[0]
        ctl_add_ok = (float(cen[:W, 0].sum()) == pooled7["box3d_tp@gate"]
                      and float(cen[:W, 1].sum()) == pooled7["box3d_n_conf"]
                      and float(cen[:W, 2].sum()) == pooled7["box3d_n_pos"])

        def f1_red(idx):
            c = cen[np.asarray(idx, np.int64)].sum(0)
            p, r = (c[0] / c[1] if c[1] else math.nan), (c[0] / c[2] if c[2] else math.nan)
            return 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else math.nan
        f1_red.__name__ = "pooled_f1"
        a_idx = np.arange(W, dtype=np.float64)
        b1 = ci.paired_episode_cluster_bootstrap(a_idx, a_idx + W, eid, n_boot=a.n_boot, seed=0,
                                                 reduce=ap_red)
        b2 = ci.paired_episode_cluster_bootstrap(a_idx, a_idx + W, eid, n_boot=a.n_boot, seed=0,
                                                 reduce=f1_red)
        controls = {"pooled_ap_equals_summarise": {"pass": bool(ctl_ap_ok), "values": ctl_ap},
                    "census_additive": {"pass": bool(ctl_add_ok)}}
        void = not (ctl_ap_ok and ctl_add_ok)

        def verdict(c):
            if void:
                return "VOID (a control did not read its known value)"
            if not a.milestone:
                return "NOT EVALUATED (pipeline validation checkpoint; SPEC §6)"
            if c.get("delta") is None:
                return ("NOT EVALUABLE (the metric is undefined on an arm -- e.g. no confident slot at "
                        "the gate makes the trainer's precision and F1 undefined; see the census)")
            return ("PASS (single training seed; no replicate floor measured for boxes)"
                    if (c["separated"] and c["delta"] > 0) else "FAILED")
        cen7, cen6 = cen[:W].sum(0), cen[W:].sum(0)
        box["bars"] = {
            "BAR-B7-1": {"statement": "box3d mAP@2 m (VIS-1, pooled classes): refcv7 - refcv6@38k > 0, "
                                      "separated", "metric": "eval_box3d_ap2m (= det_ap2_all_all)",
                         "cell": b1, "verdict": verdict(b1),
                         "class_mean_map2_all": {"refcv7": full7.get("eval_box3d_det_map2_all"),
                                                 "refcv6_38k": full6.get("eval_box3d_det_map2_all"),
                                                 "note": "reported beside, no verdict"}},
            "BAR-B7-2": {"statement": "F1 at the declared rule (focal, gate 0.5): refcv7 - refcv6@38k at "
                                      "refcv6's best-F1 gate > 0, separated",
                         "refcv7_gate": g7, "refcv6_best_f1_gate": g6,
                         "refcv6_gate_rule": "argmax pooled F1 over sigma gates 0.05..0.95 step 0.01 on the "
                                             "SAME scored windows (favours the baseline), then fixed",
                         "census_tp_nconf_npos": {"refcv7@0.5": [float(x) for x in cen7],
                                                  "refcv6_38k@best": [float(x) for x in cen6]},
                         "cell": b2, "verdict": verdict(b2)},
            "controls": controls,
            "surface_departure": "S2 windows, not every eval window (SPEC A1)",
            "refcv6_presence_cost": "sigmoid (refcv6 trained its presence with BCE); the matching cost "
                                    "affects only the Hungarian set (AUROC_matched, centre error), never "
                                    "the greedy AP / P / R"}
        out["box"] = box
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(MH.to_json(out))
    brief = {"map": {k: (v.get("verdict") if isinstance(v, dict) else v)
                     for k, v in (out["map"].get("bars") or {}).items() if k.startswith("BAR")},
             "box": {k: v.get("verdict") for k, v in (out["box"].get("bars") or {}).items()
                     if k.startswith("BAR")}}
    print(json.dumps(brief))


if __name__ == "__main__":
    main()
