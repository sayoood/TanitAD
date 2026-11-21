"""detres_pack_analysis.py -- CPU-only companion P of PREREG_DETECTION_RESIDUAL.md (EvalFlyWheel 2026-10-16).

Reads banked detection packs (`g0.packs.npz`: arms s0 and fp32_s0, head box3d) and answers, with NO model and NO GPU:
  P1  does the pack path reproduce the banked term (s0 0.31453, fp32_s0 0.31912)?
  P2  the support of `eval_box3d_det_ap2_automobile_0_20`
  P3  near-threshold census: AP / TP count at 1.9 / 2.0 / 2.1 m, and GT / detections within +-0.1 m of the 20 m band edge
  P4  Monte-Carlo dose-response: add the EMPIRICAL s0-vs-fp32_s0 per-slot (dx, dy, dlogit) displacement, scaled by f, to the s0 packs, recompute the term

    python detres_pack_analysis.py --packs E:/refcv8_eval_kit/battery/step5000/g0.packs.npz --out <json> [--draws 200] [--smoke]

Evidence class of the numbers it writes: MEASURED on the banked packs (P1-P3); SIMULATED (P4: a perturbation model, not a GPU run).
The matching / AP code is the repo's own (`detection_metrics.greedy_rows`, `box3d_head.ap_from_rows`), never re-implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path[:0] = [p for p in ("C:/lgs/Lfb47a58376/stack", "C:/lgs/Lfb47a58376/taniteval") if p not in sys.path]
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
from tanitad.eval import detection_metrics as dm  # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES  # noqa: E402

CLS = "automobile"
BAND = "0_20"
THR = 2.0


def load_arm(path, arm, head="box3d"):
    with np.load(str(path), allow_pickle=False) as z:
        pre = f"{arm}__{head}"
        off = z[f"{pre}__gt_off"]
        out = []
        for i in range(len(off) - 1):
            lo, hi = int(off[i]), int(off[i + 1])
            out.append({"logit": z[f"{pre}__logit"][i].astype(np.float32), "xy": z[f"{pre}__xy"][i].astype(np.float32),
                        "cls": z[f"{pre}__cls"][i].astype(np.int16), "gt_xy": z[f"{pre}__gt_xy"][lo:hi].astype(np.float32),
                        "gt_cls": z[f"{pre}__gt_cls"][lo:hi].astype(np.int16),
                        "pos": z[f"{pre}__gt_pos"][lo:hi].astype(bool), "ign": z[f"{pre}__gt_ign"][lo:hi].astype(bool)})
        return out


def term(packs, thr=THR, band=BAND, ci=None):
    """The AP of one (class, band) cell exactly as `detection_metrics.summarise` builds it (rows pooled over windows,
    TP counted in its GT's band, FP in its own cx's band, n_gt = positives of the class in the band)."""
    ci = list(AGENT_CLASSES).index(CLS) if ci is None else ci
    rows = [r for pk in packs for r in dm.greedy_rows(pk, thr, ci)]
    rb = [r for r in rows if r[2] == band]
    n_gt = 0
    for pk in packs:
        for gxy, gc, p in zip(pk["gt_xy"], pk["gt_cls"], pk["pos"]):
            if p and int(gc) == ci and dm._band_of(float(gxy[0])) == band:
                n_gt += 1
    ap = dm._ap(rb, n_gt)
    tp = sum(1 for r in rb if r[1] == 1)
    return {"ap": ap, "ap5": round(float(ap), 5) if ap == ap else None, "n_gt": n_gt, "n_rows": len(rb), "n_tp": tp}


def support(packs):
    ci = list(AGENT_CLASSES).index(CLS)
    n_pos = n_pos_band = n_ign_auto = n_ign = 0
    for pk in packs:
        for gxy, gc, p, ig in zip(pk["gt_xy"], pk["gt_cls"], pk["pos"], pk["ign"]):
            if p and int(gc) == ci:
                n_pos += 1
                n_pos_band += int(dm._band_of(float(gxy[0])) == BAND)
            n_ign += int(ig)
    n_det_gate = 0
    n_det_auto = 0
    for pk in packs:
        p = 1.0 / (1.0 + np.exp(-pk["logit"].astype(np.float64)))
        n_det_gate += int(((p >= 0.5) & (pk["cls"] == ci)).sum())
        n_det_auto += int((pk["cls"] == ci).sum())
    return {"n_windows": len(packs), "n_gt_automobile_all_bands": n_pos, "n_gt_automobile_0_20": n_pos_band,
            "n_ignore_rows": n_ign, "n_slots_argmax_automobile": n_det_auto, "n_slots_automobile_at_gate_0p5": n_det_gate,
            "slots_per_window": int(packs[0]["logit"].shape[0]) if packs else None}


def edge_census(packs, half=0.1):
    """GT automobiles / automobile detections within +-half m of the band edges (0 and 20 m) -- a band flip moves a TP or FP
    between cells; and GT that are POSITIVE vs the 2 m threshold sweep."""
    ci = list(AGENT_CLASSES).index(CLS)
    gt_edge = det_edge = 0
    for pk in packs:
        for gxy, gc, p in zip(pk["gt_xy"], pk["gt_cls"], pk["pos"]):
            if p and int(gc) == ci and abs(float(gxy[0]) - 20.0) <= half:
                gt_edge += 1
        for x, c, lg in zip(pk["xy"][:, 0], pk["cls"], pk["logit"]):
            if int(c) == ci and abs(float(x) - 20.0) <= half and lg > -6.0:
                det_edge += 1
    return {"half_m": half, "gt_automobile_within_edge_20m": gt_edge, "det_automobile_logit_gt_-6_within_edge_20m": det_edge}


def displacement_pool(a, b):
    """Per-slot (dx, dy, dlogit) between two arms over the SAME windows, slots with argmax automobile in `a` and p >= 0.01."""
    ci = list(AGENT_CLASSES).index(CLS)
    pool = []
    for pa, pb in zip(a, b):
        p = 1.0 / (1.0 + np.exp(-pa["logit"].astype(np.float64)))
        m = (pa["cls"] == ci) & (p >= 0.01)
        d = np.concatenate([(pb["xy"] - pa["xy"])[m], (pb["logit"] - pa["logit"])[m][:, None]], 1)
        pool.append(d)
    pool = np.concatenate(pool) if pool else np.zeros((0, 3), np.float32)
    return pool


def perturbed(packs, pool, f, rng):
    ci = list(AGENT_CLASSES).index(CLS)
    out = []
    for pk in packs:
        q = dict(pk)
        n = pk["logit"].shape[0]
        idx = rng.integers(0, len(pool), size=n)
        d = pool[idx].astype(np.float64)
        sgn = rng.choice([-1.0, 1.0], size=(n, 3))
        d = d * sgn * f
        q["xy"] = (pk["xy"] + d[:, :2]).astype(np.float32)
        q["logit"] = (pk["logit"] + d[:, 2]).astype(np.float32)
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--scales", default="0.5,1,2,4,8")
    ap.add_argument("--smoke", action="store_true", help="3 windows, 3 draws: code-path check only")
    a = ap.parse_args()
    t0 = time.time()
    s0, fp = load_arm(a.packs, "s0"), load_arm(a.packs, "fp32_s0")
    if a.smoke:
        s0, fp, a.draws = s0[:3], fp[:3], 3
    rec = {"tool": "detres_pack_analysis.py", "packs": a.packs, "smoke": a.smoke, "class": CLS, "band": BAND, "thr_m": THR,
           "evidence": "MEASURED on banked packs (P1-P3); SIMULATED perturbation (P4)", "started": time.strftime("%FT%T")}
    # P1 -----------------------------------------------------------------------------------------------------------
    rec["P1"] = {"s0": term(s0), "fp32_s0": term(fp), "banked_expected": {"s0": 0.31453, "fp32_s0": 0.31912}}
    rec["P1"]["reproduces"] = (not a.smoke) and rec["P1"]["s0"]["ap5"] == 0.31453 and rec["P1"]["fp32_s0"]["ap5"] == 0.31912
    # P2 -----------------------------------------------------------------------------------------------------------
    rec["P2"] = {"s0": support(s0), "fp32_s0": support(fp)}
    # P3 -----------------------------------------------------------------------------------------------------------
    sweep = {}
    for thr in (1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.2):
        sweep[str(thr)] = {"s0": term(s0, thr), "fp32_s0": term(fp, thr)}
    rec["P3"] = {"ap_tp_vs_threshold": sweep, "edge20": edge_census(s0)}
    s_ap = {k: v["s0"]["ap"] for k, v in sweep.items()}
    rec["P3"]["summary"] = {
        "tp_gained_between_1.9_and_2.1_s0": sweep["2.1"]["s0"]["n_tp"] - sweep["1.9"]["s0"]["n_tp"],
        "ap_2.1_minus_1.9_s0": round(float(s_ap["2.1"] - s_ap["1.9"]), 5),
        "tp_gained_between_1.9_and_2.1_fp32": sweep["2.1"]["fp32_s0"]["n_tp"] - sweep["1.9"]["fp32_s0"]["n_tp"],
        "note": "a centre shift of <= 0.1 m moves AP by about the 2.1-1.9 spread; the gap to explain is 0.02382"}
    # P4 -----------------------------------------------------------------------------------------------------------
    pool = displacement_pool(s0, fp)
    ad = np.abs(pool)
    rec["P4_pool"] = {"n_slots": int(len(pool)), "xy_norm_m": {q: float(np.quantile(np.linalg.norm(pool[:, :2], axis=1), p))
                                                              for q, p in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0))},
                      "dlogit_abs": {q: float(np.quantile(ad[:, 2], p)) for q, p in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0))}}
    base = rec["P1"]["s0"]["ap"]
    rng = np.random.default_rng(20261016)
    dose = {}
    for f in [float(x) for x in a.scales.split(",")]:
        vals = []
        for _ in range(a.draws):
            vals.append(term(perturbed(s0, pool, f, rng))["ap"])
        v = np.asarray(vals, dtype=np.float64)
        d = v - base
        dose[str(f)] = {"n_draws": int(len(v)), "mean": round(float(v.mean()), 5), "sd": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 5),
                        "min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
                        "mean_abs_delta": round(float(np.abs(d).mean()), 5),
                        "frac_abs_delta_ge_0.02382": round(float((np.abs(d) >= 0.02382).mean()), 4),
                        "frac_delta_ge_+0.02382": round(float((d >= 0.02382).mean()), 4)}
    rec["P4"] = {"base_s0": base, "dose_response": dose,
                 "model": "per slot (automobile argmax, p>=0.01 pool) draw one empirical (dx,dy,dlogit) of s0->fp32_s0, random sign per component, scale f; all slots of every window",
                 "caveat": "SIMULATION: the pool is bf16-vs-fp32 displacement, not Thor-compiled-vs-eager displacement, and independent per-slot draws ignore the correlation real numerics have"}
    rec["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("DETRES_PACK_DONE", rec["P1"]["s0"]["ap5"], rec["P1"]["fp32_s0"]["ap5"], rec["wall_s"], flush=True)


if __name__ == "__main__":
    main()
