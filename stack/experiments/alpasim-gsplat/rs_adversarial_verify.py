#!/usr/bin/env python3
"""ADVERSARIAL RE-DERIVATION of the rolling-shutter stream, from the banked artifacts only.

Runs on the dev box, no GPU, no pod. Everything below is recomputed from
`results/2026-08-03-rolling-shutter/*.json`; nothing is copied from the prose.

What it checks
--------------
1. every `grad_ncc_mean` reproduces from its own `grad_ncc_per_frame`
2. an INDEPENDENT paired percentile bootstrap over frames, per arm
3. the ARM-vs-ARM paired deltas the report never computed (it compared two CIs
   that share a baseline, which is not a comparison of the two arms)
4. the +6 reference offset: argmax histogram, mean curve, and a paired CI the
   report did not give
5. whether the phase/pose "effect" is anything more than the +6 misalignment,
   by predicting each phase arm's gain from the offset curve alone
6. cost-ratio reproducibility across runs

Writes results/2026-08-03-rolling-shutter-adversarial/ADVERSARIAL_VERIFY.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "2026-08-03-rolling-shutter"
OUT = HERE / "results" / "2026-08-03-rolling-shutter-adversarial"
SEED = 20260803
N_BOOT = 40000


def paired_boot(delta: np.ndarray, rng: np.random.Generator, n: int = N_BOOT):
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    bs = delta[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min(float((bs <= 0).mean()), float((bs >= 0).mean()))
    return float(delta.mean()), float(lo), float(hi), float(p)


def main():
    rng = np.random.default_rng(SEED)
    sw = json.load(open(SRC / "rs_sweep_chosen.report.json"))
    k10 = json.load(open(SRC / "rs_frame_offset_k10.json"))
    k3 = json.load(open(SRC / "rs_frame_offset.json"))
    seam = json.load(open(SRC / "rs_seam_control.json"))
    batch = json.load(open(SRC / "rs_batch_chosen.report.json"))
    cost = json.load(open(SRC / "rs_cost_probe.json"))

    arms = {a["arm"]: a for a in sw["arms"]}
    base = np.array(arms["g_p1.00"]["grad_ncc_per_frame"])
    rep = {"source_dir": str(SRC), "seed": SEED, "n_boot": N_BOOT}

    # 1 + 2 --------------------------------------------------------------------
    rep["mean_field_reproduces"] = {}
    rep["arm_vs_production"] = {}
    for name, a in arms.items():
        v = np.array(a["grad_ncc_per_frame"])
        rep["mean_field_reproduces"][name] = bool(abs(a["grad_ncc_mean"] - float(v.mean())) < 5e-5)
        m, lo, hi, p = paired_boot(v - base, rng)
        rep["arm_vs_production"][name] = {
            "grad_ncc": a["grad_ncc_mean"], "delta": round(m, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "boot_p": round(p, 4),
            "neg_ctl": f"{a['neg_control_pass_frames']}/{a['n_frames']}",
            "mean_alpha": a["mean_alpha"]}

    # 3 -- the comparisons the report asserted but never made ------------------
    rep["arm_vs_arm_paired"] = {}
    for x, y in [("s4", "g_p0.00"), ("s4", "g_p0.50"), ("g_p0.00", "g_p0.50"),
                 ("native", "native_swapped"), ("native", "g_p0.00"),
                 ("s16_rev", "s16"), ("s16_fixedactor", "s16")]:
        d = np.array(arms[x]["grad_ncc_per_frame"]) - np.array(arms[y]["grad_ncc_per_frame"])
        m, lo, hi, p = paired_boot(d, rng)
        rep["arm_vs_arm_paired"][f"{x}_minus_{y}"] = {
            "delta": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "boot_p": round(p, 4), "separated": bool(lo > 0 or hi < 0)}

    # 4 -- the +6 offset -------------------------------------------------------
    pf = k10["per_frame"]
    am = [int(max(r["grad_ncc_by_offset"], key=lambda k: r["grad_ncc_by_offset"][k])) for r in pf.values()]
    offs = [str(i) for i in range(-k10["k"], k10["k"] + 1)]
    mean_by = {o: float(np.mean([pf[f]["grad_ncc_by_offset"][o] for f in pf])) for o in offs}
    d6 = np.array([pf[f]["grad_ncc_by_offset"]["6"] - pf[f]["grad_ncc_by_offset"]["0"] for f in pf])
    m, lo, hi, p = paired_boot(d6, rng)
    rep["offset_scan"] = {
        "argmax_histogram_recomputed": dict(sorted(Counter(am).items())),
        "matches_reported": dict(sorted(Counter(am).items())) == {int(k): v for k, v in k10["argmax_histogram"].items()},
        "mean_gain_plus6": round(mean_by["6"] - mean_by["0"], 4),
        "paired_frame_ci95_NOT_IN_REPORT": [round(lo, 4), round(hi, 4)],
        "second_best_offset_every_frame": sorted({
            sorted(r["grad_ncc_by_offset"].items(), key=lambda kv: -kv[1])[1][0] for r in pf.values()}),
        "FRAME_SETS_ARE_NOT_THE_SAME": {
            "sweep": sw["frames"], "k3": k3["frames"], "k10": k10["frames"],
            "overlap_sweep_k10": sorted(set(sw["frames"]) & set(k10["frames"])),
            "note": ("ROLLING_SHUTTER.md section 11 says 'over the SAME 12 frames'. "
                     "The overlap between the sweep set and the k=10 set is EMPTY, which is "
                     "why offset-0 reads 0.3114 there and production reads 0.3241 in section 1.")},
        "cross_instrument_on_shared_frames": {
            str(f): {"sweep_g_p1.00": arms["g_p1.00"]["grad_ncc_per_frame"][sw["frames"].index(f)],
                     "k3_offset0": k3["per_frame"][str(f)]["grad_ncc_by_offset"]["0"]}
            for f in sorted(set(sw["frames"]) & set(k3["frames"]))}}

    # 5 -- is the phase effect anything but the misalignment? -------------------
    FP, RO = k10["frame_period_ms"], k10["readout_ms"]
    o = np.array(sorted(int(x) for x in mean_by))
    v = np.array([mean_by[str(x)] for x in o])

    def predict(dt_ms):
        return float(np.interp(dt_ms / FP, o, v)) - float(np.interp(0.0, o, v))

    rep["phase_effect_is_the_misalignment"] = {
        "method": ("moving the RENDER earlier by dt is, for a static scene, the same displacement "
                   "as moving the REFERENCE later by dt; so predict each phase arm's gain from the "
                   "offset curve alone. ESTIMATED: the two curves are on different frame sets."),
        "readout_ms": RO, "frame_period_ms": FP, "arms": {}}
    for name, n_read in [("g_p0.75", 0.25), ("g_p0.50", 0.50), ("g_p0.25", 0.75), ("g_p0.00", 1.00)]:
        meas = arms[name]["grad_ncc_mean"] - arms["g_p1.00"]["grad_ncc_mean"]
        pred = predict(n_read * RO)
        rep["phase_effect_is_the_misalignment"]["arms"][name] = {
            "earlier_ms": round(n_read * RO, 1), "measured": round(meas, 4),
            "predicted_from_offset_curve": round(pred, 4), "ratio": round(meas / pred, 2)}
    for name, n_read, g in [("g_pm0.50", 1.5, 0.3332), ("g_pm1.00", 2.0, 0.3388), ("g_pm2.00", 3.0, 0.3499)]:
        meas = g - arms["g_p1.00"]["grad_ncc_mean"]
        pred = predict(n_read * RO)
        rep["phase_effect_is_the_misalignment"]["arms"][name] = {
            "earlier_ms": round(n_read * RO, 1), "measured": round(meas, 4),
            "predicted_from_offset_curve": round(pred, 4), "ratio": round(meas / pred, 2),
            "PROVENANCE": "utgate PARTIAL LOG ONLY -- no per-frame data, no CI, no report.json"}
    rep["phase_effect_is_the_misalignment"]["sliced_arms"] = {
        "mean_phase": 0.5, "equivalent_shift_ms": round(0.5 * RO, 1),
        "predicted": round(predict(0.5 * RO), 4),
        "measured_seam_masked": [a["masked_delta"] for a in seam["arms"]],
        "measured_mean": round(float(np.mean([a["masked_delta"] for a in seam["arms"]])), 4),
        "seam_control_has_per_frame_data": False}

    # 6 -- cost ratios ---------------------------------------------------------
    ratios = {"sweep_within_run": round(arms["native"]["raster_ms_median"] / arms["g_p1.00"]["raster_ms_median"], 1),
              "doc_headline_spliced_3258.6_over_36.4": 89.5,
              "utgate_log_within_run_6844.1_over_36.4": 188.0}
    for f in cost["per_frame"]:
        ratios[f"cost_probe_f{f}_within_run"] = round(
            cost["per_frame"][f]["rs_sweep"]["median_ms"] / cost["per_frame"][f]["global_end"]["median_ms"], 1)
    rep["native_rs_cost_ratio_across_runs"] = ratios

    # batching self-consistency is stronger than "4 decimals": it is bit-exact
    rep["batch_vs_sequential_bit_exact"] = {}
    bm = {a["arm"]: a for a in batch["arms"]}
    for n in (2, 4, 8, 16):
        rep["batch_vs_sequential_bit_exact"][n] = bool(
            bm[f"b{n}"]["grad_ncc_per_frame"] == bm[f"s{n}"]["grad_ncc_per_frame"])
    rep["sliced_arms_differ_between_the_two_runs"] = {
        f"s{n}": {"sweep_run": arms[f"s{n}"]["grad_ncc_mean"], "batch_run": bm[f"s{n}"]["grad_ncc_mean"],
                  "diff": round(arms[f"s{n}"]["grad_ncc_mean"] - bm[f"s{n}"]["grad_ncc_mean"], 4)}
        for n in (2, 4, 8, 16)}
    rep["global_arm_reproduces_between_runs"] = {
        "g_p1.00_sweep": arms["g_p1.00"]["grad_ncc_mean"], "g_p1.00_batch": bm["g_p1.00"]["grad_ncc_mean"]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ADVERSARIAL_VERIFY.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT / 'ADVERSARIAL_VERIFY.json'}")


if __name__ == "__main__":
    main()
