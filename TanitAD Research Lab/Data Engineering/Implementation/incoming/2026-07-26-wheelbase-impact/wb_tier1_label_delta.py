#!/usr/bin/env python3
"""TIER 1 — the LABEL delta from `WHEELBASE = 2.9` (physicalai.py:51).

    steer_shipped = atan(2.9   * curvature)      <- every action label in every arm
    steer_true    = atan(L_clip * curvature)     <- L_clip from vehicle_dimensions
    delta         = steer_true - steer_shipped

Computed on the canonical parity corpus (`physicalai-train-e438721ae894`, order
lines 0..2399 minus the 24 skips = 2376 built episodes) and on the canonical val
order (`physicalai-val-0c5f7dac3b11`, 600 lines; the eval-pod deployment is the
first 40). Curvature is read from the dataset's own `labels/egomotion` and
resampled to the pipeline's 10 Hz with the SAME `np.interp` the pipeline uses
(`physicalai.signals_at`).

⚠️ THE QUERY GRID — a trap that cost the first run of this script.
`labels/egomotion` is NOT a 20 s file. MEASURED: it spans ~140 s per clip — a
dense ~100 Hz stretch covering the 20 s camera clip, then a ~3.5 Hz tail over
the rest of the parent session. Resampling the FULL egomotion span (the naive
read) samples ~7x too much time and gave a per-episode mean|steer| that
correlates only 0.68 with the built episodes' own labels. The pipeline queries
`linspace(t_frames[0], t_frames[-1], int(span_s*10))` from the CAMERA
timestamps, which MEASURED over 500 local clips is 20.10-20.13 s starting
0.03-0.21 s after the egomotion t0 (n_target = 201 for 454/500).

So this script uses, per clip:
  * EXACT camera timestamps when the clip's `.timestamps.parquet` is held
    locally (95 of our 3000 corpus clips are in the local R0 mirror);
  * otherwise `linspace(t_ego0, t_ego0 + CLIP_SPAN_US, N_TARGET)`.
The `--validate` mode quantifies the second against the first on those 95.
`k = n_stack - 1 = 2` leading steps are dropped, as `build_episode` does.

GATED-CONFIDENTIAL: no clip UUID is written to --out. Per-clip rows go to
--scratch only; --out receives aggregates and per-population summaries.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SHIPPED_WB = 2.9
TARGET_HZ = 10.0
# MEASURED over the 500 locally-mirrored R0 camera timestamp files:
# span_s min 19.933 / p05 20.100 / med 20.133 / p95 20.133 / max 20.767;
# int(span_s*10) == 201 for 454/500 clips.
CLIP_SPAN_US = 20.133e6
N_TARGET = 201
N_STACK_DROP = 2          # build_episode: actions[k:n], k = n_stack - 1 = 2


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    ax = np.abs(x)
    return {
        "n": int(x.size),
        "mean": float(x.mean()), "std": float(x.std()),
        "mean_abs": float(ax.mean()), "median_abs": float(np.median(ax)),
        "p95_abs": q(ax, 0.95), "p99_abs": q(ax, 0.99), "max_abs": float(ax.max()),
        "p01": q(x, 0.01), "p50": q(x, 0.50), "p99": q(x, 0.99),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego-root",
                    default=r"C:\Users\Admin\tanitad-data\physicalai\labels\egomotion")
    ap.add_argument("--cam-root",
                    default=r"C:\Users\Admin\tanitad-data\physicalai\r0\camera_front_wide")
    a = ap.parse_args()
    scratch, out = Path(a.scratch), Path(a.out)
    ego_root, cam_root = Path(a.ego_root), Path(a.cam_root)
    cam_have = {p.name.split(".")[0]: p for p in
                cam_root.glob("*.timestamps.parquet")} if cam_root.exists() else {}

    rows = json.load(open(scratch / "corpus_dims.json", encoding="utf-8"))
    by_chunk = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk"]].append(r)

    per_clip = []
    grid_val = []          # exact-vs-rule validation on clips with local camera ts
    all_s, all_d, all_wb, all_split, all_ep = [], [], [], [], []
    for ci, (chunk, rs) in enumerate(sorted(by_chunk.items())):
        zp = ego_root / f"egomotion.chunk_{chunk:04d}.zip"
        with zipfile.ZipFile(zp) as zf:
            names = {n.split(".")[0]: n for n in zf.namelist()}
            for r in rs:
                nm = names.get(r["clip_id"])
                if nm is None:
                    per_clip.append({**{k: r[k] for k in
                                        ("split", "idx", "clip_id", "chunk",
                                         "wheelbase", "country", "platform_class")},
                                     "status": "ego_missing"})
                    continue
                with zf.open(nm) as fh:
                    ego = pd.read_parquet(io.BytesIO(fh.read()))
                t = ego["timestamp"].to_numpy(np.float64)
                o = np.argsort(t)
                t = t[o]
                curv_raw = ego["curvature"].to_numpy(np.float64)[o]
                vx = ego["vx"].to_numpy(np.float64)[o]
                vy = ego["vy"].to_numpy(np.float64)[o]

                # --- the query grid (see module docstring) -----------------
                tq_rule = np.linspace(t[0], t[0] + CLIP_SPAN_US,
                                      N_TARGET)[N_STACK_DROP:]
                cp = cam_have.get(r["clip_id"])
                grid = "rule"
                if cp is not None:
                    ts = pd.read_parquet(cp)
                    tcol = next(c for c in ts.columns if "time" in c.lower())
                    tf = ts[tcol].to_numpy(np.float64)
                    span = tf[-1] - tf[0]
                    nt = max(int(span / 1e6 * TARGET_HZ), 4)
                    tq = np.linspace(tf[0], tf[-1], nt)[N_STACK_DROP:]
                    grid = "camera_exact"
                    s_rule = np.arctan(SHIPPED_WB * np.interp(tq_rule, t, curv_raw))
                    s_ex = np.arctan(SHIPPED_WB * np.interp(tq, t, curv_raw))
                    grid_val.append({
                        "idx": r["idx"], "split": r["split"],
                        "exact_mean_abs": float(np.abs(s_ex).mean()),
                        "rule_mean_abs": float(np.abs(s_rule).mean()),
                        "exact_p99_abs": q(np.abs(s_ex), 0.99),
                        "rule_p99_abs": q(np.abs(s_rule), 0.99)})
                else:
                    tq = tq_rule
                n_target = len(tq)
                curv = np.interp(tq, t, curv_raw)
                v = np.hypot(np.interp(tq, t, vx), np.interp(tq, t, vy))
                L = float(r["wheelbase"])
                s0 = np.arctan(SHIPPED_WB * curv)          # what every arm trained on
                s1 = np.arctan(L * curv)                   # what it should have been
                d = s1 - s0
                per_clip.append({
                    "split": r["split"], "idx": r["idx"], "clip_id": r["clip_id"],
                    "chunk": chunk, "wheelbase": L, "country": r["country"],
                    "platform_class": r["platform_class"], "status": "ok",
                    "grid": grid, "n": int(n_target),
                    "steer_mean_abs": float(np.abs(s0).mean()),
                    "steer_p99_abs": q(np.abs(s0), 0.99),
                    "steer_max_abs": float(np.abs(s0).max()),
                    "delta_mean_abs": float(np.abs(d).mean()),
                    "delta_p95_abs": q(np.abs(d), 0.95),
                    "delta_max_abs": float(np.abs(d).max()),
                    "v_mean": float(v.mean()),
                })
                all_s.append(s0.astype(np.float32))
                all_d.append(d.astype(np.float32))
                all_wb.append(np.full(n_target, L, np.float32))
                all_split.append(np.full(n_target, 0 if r["split"] == "train" else 1,
                                         np.int8))
                all_ep.append(np.full(n_target, r["idx"], np.int32))
        if (ci + 1) % 40 == 0:
            print(f"[t1] {ci+1}/{len(by_chunk)} chunks", flush=True)

    S = np.concatenate(all_s)
    D = np.concatenate(all_d)
    W = np.concatenate(all_wb)
    SP = np.concatenate(all_split)
    EP = np.concatenate(all_ep)
    np.savez_compressed(scratch / "tier1_samples.npz", steer=S, delta=D,
                        wheelbase=W, split=SP, ep=EP)
    pd.DataFrame(per_clip).to_csv(scratch / "tier1_per_clip.csv", index=False)

    gv = pd.DataFrame(grid_val)
    grid_check = None
    if len(gv):
        rel = (gv.rule_mean_abs - gv.exact_mean_abs).abs() / gv.exact_mean_abs.clip(1e-6)
        grid_check = {
            "n_clips_with_exact_camera_grid": int(len(gv)),
            "corr_mean_abs_exact_vs_rule": float(np.corrcoef(gv.exact_mean_abs,
                                                             gv.rule_mean_abs)[0, 1]),
            "median_rel_diff_pct": float(np.median(rel) * 100),
            "p90_rel_diff_pct": float(np.quantile(rel, 0.90) * 100),
            "max_rel_diff_pct": float(rel.max() * 100),
            "pooled_mean_abs_exact": float(gv.exact_mean_abs.mean()),
            "pooled_mean_abs_rule": float(gv.rule_mean_abs.mean()),
        }

    res = {"_meta": {
        "evidence_class": "MEASURED",
        "shipped_WHEELBASE": SHIPPED_WB,
        "source_curvature": "PhysicalAI-AV labels/egomotion (local mirror), 10 Hz np.interp",
        "source_wheelbase": "PhysicalAI-AV calibration/vehicle_dimensions (HF, 197 chunks)",
        "query_grid": f"camera timestamps where mirrored locally, else "
                      f"linspace(t_ego0, +{CLIP_SPAN_US/1e6:.3f}s, {N_TARGET}) "
                      f"minus {N_STACK_DROP} stack steps",
        "grid_rule_validation": grid_check,
        "n_clips_total": len(per_clip),
        "n_clips_ok": int(sum(1 for r in per_clip if r["status"] == "ok")),
    }}

    for name, mask in (("train", SP == 0), ("val600", SP == 1),
                       ("all", np.ones_like(SP, bool))):
        s, d, w = S[mask], D[mask], W[mask]
        # steer scale references (the denominators the delta is quoted against)
        rng = float(np.quantile(np.abs(s), 0.99) * 2)      # symmetric p99 range
        full_rng = float(np.abs(s).max() * 2)
        blk = {
            "n_samples": int(mask.sum()),
            "n_clips": int(len({(int(e)) for e in EP[mask]})) if name != "all" else None,
            "steer_shipped": describe(s),
            "steer_p99_symmetric_range_rad": rng,
            "steer_full_range_rad": full_rng,
            "delta_rad": describe(d),
            "delta_as_frac_of_p99_range": {
                "mean_abs": float(np.abs(d).mean() / rng),
                "median_abs": float(np.median(np.abs(d)) / rng),
                "p95_abs": float(np.quantile(np.abs(d), 0.95) / rng),
                "max_abs": float(np.abs(d).max() / rng),
            },
            "delta_over_steer_std": float(d.std() / s.std()),
            "corr_delta_steer": float(np.corrcoef(d, s)[0, 1]),
            "by_wheelbase": {},
        }
        for L in sorted(set(np.round(w, 4).tolist())):
            m2 = np.isclose(w, L)
            s2, d2 = s[m2], d[m2]
            gain = L / SHIPPED_WB
            blk["by_wheelbase"][f"{L:.3f}"] = {
                "n_samples": int(m2.sum()),
                "n_clips": int(len({int(e) for e in EP[mask][m2]})),
                "frac_of_clips": None,
                "wheelbase_rel_error_of_2.9_pct": float((SHIPPED_WB - L) / L * 100),
                "implied_steer_gain_true_over_shipped": float(gain),
                "steer_shipped": describe(s2),
                "delta_rad": describe(d2),
                "delta_as_frac_of_p99_range": {
                    "mean_abs": float(np.abs(d2).mean() / rng),
                    "p95_abs": float(np.quantile(np.abs(d2), 0.95) / rng),
                    "max_abs": float(np.abs(d2).max() / rng),
                },
                "delta_over_steer_std_within_pop": float(d2.std() / s2.std()),
                "median_rel_error_pct_where_steer_gt_0p01":
                    float(np.median(np.abs(d2[np.abs(s2) > 0.01]
                                           / s2[np.abs(s2) > 0.01])) * 100)
                    if (np.abs(s2) > 0.01).any() else None,
            }
        res[name] = blk

    # per-clip population fractions
    pc = pd.DataFrame([r for r in per_clip if r["status"] == "ok"])
    for name, sub in (("train", pc[pc.split == "train"]), ("val600", pc[pc.split == "val"])):
        vc = sub.wheelbase.round(4).value_counts()
        for L, k in vc.items():
            res[name]["by_wheelbase"][f"{L:.3f}"]["frac_of_clips"] = float(k / len(sub))
        res[name]["n_clips"] = int(len(sub))

    out.mkdir(parents=True, exist_ok=True)
    (out / "tier1_label_delta.json").write_text(json.dumps(res, indent=2),
                                                encoding="utf-8")
    print("[t1] wrote", out / "tier1_label_delta.json")


if __name__ == "__main__":
    main()
