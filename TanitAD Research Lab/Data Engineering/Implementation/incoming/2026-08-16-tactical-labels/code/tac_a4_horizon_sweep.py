#!/usr/bin/env python3
"""A4 — IS THE LONGITUDINAL DISAGREEMENT A *HORIZON* ARTIFACT, OR A REAL CONFLICT?

⭐ THE CHEAPEST DISCRIMINATING EXPERIMENT, pre-registered with BOTH outcomes
committed in advance (operating standard rule 5).

A3 MEASURED: Alpamayo's longitudinal meta-action and our engine-A longitudinal
band agree on only **45.3 %** (kappa 0.187, n=201), and the single largest
confusion cell is **alp=decelerate / ego=accelerate, n=43** — the two legs are
OPPOSITE on 21.4 % of clips. That is far too large to be noise.

**TWO HYPOTHESES, and they have different consequences:**

  H-HORIZON  The legs are answering DIFFERENT QUESTIONS. Alpamayo's meta-action
             is the decision AT t0 ("what should I do now"); engine-A's
             `net_dv_ms` is the net speed change over the WHOLE 11.9 s future.
             A car that brakes for a lead vehicle and then accelerates away is
             `decelerate` at t0 and `accelerate` net — both correct.
             ⇒ PREDICTION: agreement RISES sharply as the horizon shortens and
             peaks at a short horizon. The label IS buildable; we were reading
             the wrong window.

  H-CONFLICT The legs genuinely disagree about the same question.
             ⇒ PREDICTION: agreement stays near chance at EVERY horizon and
             every threshold. The label is NOT buildable from this pair, and
             saying otherwise would repeat the S2 failure.

⛔ THE OUTCOME IS NOT CHOSEN BY WHICH IS CONVENIENT. Both branches are written
above before the run; the sweep decides.

⚠️ A sweep over horizons AND thresholds is a multiple-comparison surface. The
peak is reported WITH the full sweep table so a reader can see whether it is a
ridge (a real horizon effect) or a spike (a fitted artefact), and the null
(chance agreement) is carried in every row.

Inputs: the 201 local ego pose tracks (poses = [x, y, yaw, v] at 10 Hz, t0 at
index 80 per engine-A) + A1's Alpamayo per-clip axes.

Usage:
  python tac_a4_horizon_sweep.py --ego-dir <npz dir> --alpamayo-per-clip <jsonl> \
      --engine-a <jsonl> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

POSE_HZ = 10.0

ALP_LON3 = {
    "Maintain Speed": "maintain",
    "Gentle Deceleration": "decelerate", "Strong Deceleration": "decelerate",
    "Stop": "decelerate",
    "Gentle Acceleration": "accelerate", "Strong Acceleration": "accelerate",
    "Reverse": None,
}
ALP_LAT3 = {
    "Go Straight": "straight",
    "Steer Left": "left", "Sharp Steer Left": "left", "Slight Steer Left": "left",
    "Steer Right": "right", "Sharp Steer Right": "right",
    "Slight Steer Right": "right",
    "Reverse Left": None, "Reverse Right": None,
}
#: ⚠️ Alpamayo's longitudinal axis has SEVERITY (Gentle vs Strong). The 3-way
#: projection discards it; the severity contrast is measured separately below.
ALP_LON_SEVERE = {"Strong Deceleration": "decelerate", "Stop": "decelerate",
                  "Strong Acceleration": "accelerate",
                  "Maintain Speed": "maintain"}

HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 11.8)
DV_THRESH = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
DYAW_THRESH = (0.05, 0.10, 0.15, 0.25, 0.40)


def agreement(pairs) -> dict:
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(both)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None, "chance": None}
    agree = sum(1 for a, b in both if a == b)
    labs = sorted({a for a, _ in both} | {b for _, b in both})
    po = agree / n
    pe = sum((sum(1 for a, _ in both if a == c) / n) *
             (sum(1 for _, b in both if b == c) / n) for c in labs)
    return {"n": n, "agreement": round(po, 4),
            "kappa": (round((po - pe) / (1 - pe), 4)
                      if abs(1 - pe) > 1e-9 else None),
            "chance": round(pe, 4),
            "n_distinct_b": len({b for _, b in both})}


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--ego-dir", required=True)
    ap.add_argument("--alpamayo-per-clip", required=True)
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    alp = {}
    with open(a.alpamayo_per_clip, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                alp[r["clip_id"]] = r
    t0_by = {}
    with open(a.engine_a, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("clip_id"):
                    t0_by[r["clip_id"]] = (r.get("engine_a") or {}).get("t0_idx", 80)

    # --- load poses, compute per-horizon ego quantities --------------------
    per: dict[str, dict] = {}
    n_short = 0
    for p in sorted(glob.glob(os.path.join(a.ego_dir, "*.npz"))):
        cid = os.path.splitext(os.path.basename(p))[0]
        if cid not in alp:
            continue
        try:
            poses = np.load(p)["poses"].astype(float)
        except Exception:                                        # noqa: BLE001
            continue
        t0 = int(t0_by.get(cid, 80))
        if poses.shape[0] <= t0 + 5:
            n_short += 1
            continue
        v = poses[:, 3]
        yaw = poses[:, 2]
        row = {"clip_id": cid, "v_t0": float(v[t0]), "dv": {}, "dyaw": {},
               "amean": {}}
        for H in HORIZONS_S:
            k = min(int(round(t0 + H * POSE_HZ)), poses.shape[0] - 1)
            if k <= t0:
                continue
            row["dv"][str(H)] = float(v[k] - v[t0])
            # mean signed acceleration over the window — less end-point
            # sensitive than a single difference
            row["amean"][str(H)] = float((v[k] - v[t0]) / ((k - t0) / POSE_HZ))
            d = float(yaw[k] - yaw[t0])
            row["dyaw"][str(H)] = (d + math.pi) % (2 * math.pi) - math.pi
        per[cid] = row
    res: dict = {
        "_evidence_class": "MEASURED (ours; 201 local ego pose tracks, 10 Hz)",
        "_pre_registered_hypotheses": {
            "H_HORIZON": ("legs answer different questions; agreement RISES as "
                          "the horizon shortens and peaks short."),
            "H_CONFLICT": ("legs genuinely disagree; agreement stays near "
                           "chance at EVERY horizon and threshold."),
        },
        "n_clips_with_poses_and_alpamayo": len(per),
        "n_clips_dropped_too_short": n_short,
    }

    # ---------------- LONGITUDINAL SWEEP ----------------------------------
    lon_rows = []
    for H in HORIZONS_S:
        for th in DV_THRESH:
            pairs = []
            for cid, r in per.items():
                dv = r["dv"].get(str(H))
                if dv is None:
                    continue
                ego = ("accelerate" if dv > th else
                       "decelerate" if dv < -th else "maintain")
                pairs.append((ALP_LON3.get(alp[cid]["longitudinal"]), ego))
            st = agreement(pairs)
            lon_rows.append({"horizon_s": H, "dv_thresh_ms": th, **st})
    res["longitudinal_sweep"] = lon_rows
    valid = [r for r in lon_rows if r["kappa"] is not None and r["n_distinct_b"] > 1]
    best = max(valid, key=lambda r: r["kappa"]) if valid else None
    res["longitudinal_best"] = best
    base = next((r for r in lon_rows
                 if r["horizon_s"] == 11.8 and r["dv_thresh_ms"] == 1.0), None)
    res["longitudinal_baseline_full_window"] = base

    # ---------------- LATERAL SWEEP ---------------------------------------
    lat_rows = []
    for H in HORIZONS_S:
        for th in DYAW_THRESH:
            pairs = []
            for cid, r in per.items():
                dy = r["dyaw"].get(str(H))
                if dy is None:
                    continue
                ego = ("left" if dy > th else "right" if dy < -th else "straight")
                pairs.append((ALP_LAT3.get(alp[cid]["lateral"]), ego))
            st = agreement(pairs)
            lat_rows.append({"horizon_s": H, "dyaw_thresh_rad": th, **st})
    res["lateral_sweep"] = lat_rows
    validl = [r for r in lat_rows if r["kappa"] is not None and r["n_distinct_b"] > 1]
    res["lateral_best"] = max(validl, key=lambda r: r["kappa"]) if validl else None

    # ---------------- does SEVERITY help? ---------------------------------
    # Alpamayo distinguishes Gentle vs Strong. If the STRONG subset agrees much
    # better, the disagreement is concentrated in the ambiguous middle — which
    # is a usable finding (label only the confident ones).
    if best:
        H, th = best["horizon_s"], best["dv_thresh_ms"]
        sub = {"all": [], "strong_only": []}
        for cid, r in per.items():
            dv = r["dv"].get(str(H))
            if dv is None:
                continue
            ego = ("accelerate" if dv > th else
                   "decelerate" if dv < -th else "maintain")
            raw = alp[cid]["longitudinal"]
            sub["all"].append((ALP_LON3.get(raw), ego))
            if raw in ALP_LON_SEVERE:
                sub["strong_only"].append((ALP_LON_SEVERE[raw], ego))
        res["longitudinal_severity_contrast"] = {
            "_at": {"horizon_s": H, "dv_thresh_ms": th},
            "_definition": ("strong_only keeps Strong Deceleration / Stop / "
                            "Strong Acceleration / Maintain Speed and DROPS "
                            "the two 'Gentle' classes — the ambiguous middle."),
            "all": agreement(sub["all"]),
            "strong_only": agreement(sub["strong_only"]),
        }

    # ---------------- the VERDICT, mechanically ---------------------------
    if best and base and base.get("kappa") is not None:
        lift = best["kappa"] - base["kappa"]
        res["_verdict"] = {
            "best_kappa": best["kappa"],
            "best_at": {"horizon_s": best["horizon_s"],
                        "dv_thresh_ms": best["dv_thresh_ms"]},
            "full_window_kappa": base["kappa"],
            "kappa_lift": round(lift, 4),
            "reading": (
                "H_HORIZON SUPPORTED — a short horizon recovers substantial "
                "agreement" if best["kappa"] >= 0.40 else
                "H_HORIZON PARTIALLY SUPPORTED — agreement improves but stays "
                "below the kappa 0.40 'fair-to-moderate' line" if lift >= 0.10
                else "H_CONFLICT SUPPORTED — no horizon or threshold recovers "
                     "agreement; the legs disagree about the same question"),
        }

    # is the peak a RIDGE or a SPIKE? (the multiple-comparison honesty check)
    if best:
        near = [r for r in valid
                if r["kappa"] is not None and r["kappa"] >= best["kappa"] - 0.05]
        res["_peak_shape"] = {
            "n_cells_within_0.05_kappa_of_peak": len(near),
            "n_cells_swept": len(lon_rows),
            "horizons_in_that_band": sorted({r["horizon_s"] for r in near}),
            "_reading": ("a RIDGE (many neighbouring cells) is a real effect; "
                         "a SPIKE (one isolated cell) is a fitted artefact."),
        }

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    slim = {k: v for k, v in res.items()
            if k not in ("longitudinal_sweep", "lateral_sweep")}
    print(json.dumps(slim, indent=1, ensure_ascii=False)[:4500])
    print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
