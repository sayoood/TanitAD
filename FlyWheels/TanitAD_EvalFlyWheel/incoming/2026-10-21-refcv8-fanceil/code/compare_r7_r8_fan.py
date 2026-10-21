#!/usr/bin/env python
"""compare_r7_r8_fan.py -- D-REFCV8-FANCEIL, PAIRED refcv8-vs-refcv7 fan ceiling on the IDENTICAL 4,754 windows.

refcv7 @ 50,400: its banked X8 fan (117 candidates, 2 s grid) + its battery GT dump. refcv8 EMA-48,852: this run's fan (149
candidates). Pairs by (clip_index, ws); CONTROL: the two GT dumps must agree on every window (same windows, same standalone pose
clock) -- the max abs GT difference is reported and the paired numbers are refused above 1e-4 m.
Also computes, for refcv7, the two refcv8 additions that its X8 json lacks (selected ADE rank; LON/LAT split), with the
fan_control_r8.py definitions, so the comparison is matched. Same estimator: FULL-SET mean, (paired) episode-cluster bootstrap,
taniteval ci, n_boot 2000 seed 0. T0 DIAGNOSTIC for anything using GT; CI = another draw of EPISODES only.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r7-fan", required=True)
    ap.add_argument("--r7-dump", required=True)
    ap.add_argument("--r8-fan", required=True)
    ap.add_argument("--r8-dump", required=True)
    ap.add_argument("--x8", required=True)
    ap.add_argument("--taniteval-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--name-a", default="refcv7_50400", help="name of the --r7-* arm (A)")
    ap.add_argument("--name-b", default="refcv8_ema48852", help="name of the --r8-* arm (B); paired = B - A")
    a = ap.parse_args()
    sys.path.insert(0, a.taniteval_root)
    os.environ.setdefault("TANITEVAL_ROOT", a.taniteval_root)
    X8 = _load(a.x8, "fan_control_x8")
    from taniteval import ci as C
    b = lambda v, e, t: dict(C.episode_cluster_bootstrap(np.asarray(v, np.float64), e, n_boot=2000, seed=0), metric=t)
    p = lambda x, y, e, t: dict(C.paired_episode_cluster_bootstrap(np.asarray(x, np.float64), np.asarray(y, np.float64), e,
                                                                   n_boot=2000, seed=0), metric=t)
    t7, _ = X8.load_gt_dump(a.r7_dump)
    t8, _ = X8.load_gt_dump(a.r8_dump)
    z7, z8 = np.load(a.r7_fan, allow_pickle=True), np.load(a.r8_fan, allow_pickle=True)
    k7 = [(int(c), int(w)) for c, w in zip(z7["clip_index"], z7["ws"])]
    k8 = [(int(c), int(w)) for c, w in zip(z8["clip_index"], z8["ws"])]
    assert set(k7) == set(k8), "the two fans are not on the same windows"
    i7 = {k: i for i, k in enumerate(k7)}
    keys = k8
    j7 = np.asarray([i7[k] for k in keys])
    g7 = np.stack([t7[k][0] for k in keys])
    g8 = np.stack([t8[k][0] for k in keys])
    gdiff = float(np.abs(g7 - g8).max())
    eid = [t8[k][1] for k in keys]
    assert [t7[k][1] for k in keys] == eid, "episode ids differ between the two dumps"
    out = {"label": a.label, "n_windows": len(keys), "control_gt_max_abs_diff_m": gdiff, "control_gt_pass": gdiff <= 1e-4}
    if gdiff > 1e-4:
        out["refused"] = "GT differs between the two dumps by > 1e-4 m: not the same windows/clock"
        json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
        raise SystemExit(out["refused"])
    gt = g8
    B = len(keys)
    ar = np.arange(B)
    res = {}
    for name, fan, sel in ((a.name_a, np.asarray(z7["anchor_traj"], np.float64)[j7][:, :, :4],
                            np.asarray(z7["sel_idx"]).astype(np.int64).ravel()[j7]),
                           (a.name_b, np.asarray(z8["anchor_traj"], np.float64)[:, :, :4],
                            np.asarray(z8["sel_idx"]).astype(np.int64).ravel())):
        de = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
        ds, do, dr = de[ar, sel], de.min(1), de.mean(1)
        rank = (de < ds[:, None]).sum(1)
        prev = np.concatenate([np.zeros((B, 1, 2)), gt[:, :-1]], 1)
        tan = gt - prev
        nrm = np.linalg.norm(tan, axis=-1, keepdims=True)
        tan = np.where(nrm < 0.05, np.array([1.0, 0.0]), tan / np.maximum(nrm, 1e-9))
        nor = np.stack([-tan[..., 1], tan[..., 0]], -1)
        es, eo = fan[ar, sel] - gt, fan[ar, de.argmin(1)] - gt
        res[name] = {"n_candidates": int(fan.shape[1]), "_de": (ds, do, dr),
                     "selected": round(float(ds.mean()), 4), "oracle_in_fan": round(float(do.mean()), 4),
                     "random_expectation": round(float(dr.mean()), 4),
                     "skill_fraction": round(float((dr - ds).mean() / (dr - do).mean()), 4),
                     "selected_rank": {"mean": round(float(rank.mean()), 3), "median": float(np.median(rank)),
                                       "frac_rank0": round(float((rank == 0).mean()), 4),
                                       "frac_top5": round(float((rank < 5).mean()), 4),
                                       "frac_top10": round(float((rank < 10).mean()), 4),
                                       "rank_as_fraction_of_N_mean": round(float((rank / (fan.shape[1] - 1)).mean()), 4)},
                     "lon_lat": {"selected_along": round(float(np.abs((es * tan).sum(-1)).mean()), 4),
                                 "selected_cross": round(float(np.abs((es * nor).sum(-1)).mean()), 4),
                                 "oracle_along": round(float(np.abs((eo * tan).sum(-1)).mean()), 4),
                                 "oracle_cross": round(float(np.abs((eo * nor).sum(-1)).mean()), 4)}}
    s7, o7, r7 = res[a.name_a].pop("_de")
    s8, o8, r8 = res[a.name_b].pop("_de")
    out["per_model"] = res
    out["paired_r8_minus_r7"] = {
        "selected": p(s8, s7, eid, "refcv8 selected - refcv7 selected"),
        "oracle_in_fan": p(o8, o7, eid, "refcv8 oracle - refcv7 oracle (T0)"),
        "random_expectation": p(r8, r7, eid, "refcv8 random - refcv7 random (T0)"),
        "sel_gap": p(s8 - o8, s7 - o7, eid, "refcv8 (sel-oracle) - refcv7 (sel-oracle) (T0)")}
    out["single"] = {"refcv7_oracle": b(o7, eid, "refcv7 oracle"), "refcv8_oracle": b(o8, eid, "refcv8 oracle")}
    out["stamp"] = {"tier": "T0 DIAGNOSTIC (oracle/random use GT)", "estimator": "FULL-SET mean; paired episode-cluster bootstrap "
                    "taniteval ci n_boot 2000 seed 0", "ci_answers": "another draw of EPISODES only; one training seed per model; "
                    "one inference seed per fan", "r7_fan": a.r7_fan, "r8_fan": a.r8_fan, "script": os.path.abspath(__file__)}
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print("ZZR7R8ZZ gt_diff=%.2e or8-or7=%.4f sel8-sel7=%.4f" % (gdiff, out["paired_r8_minus_r7"]["oracle_in_fan"]["delta"],
                                                              out["paired_r8_minus_r7"]["selected"]["delta"]))


if __name__ == "__main__":
    main()
