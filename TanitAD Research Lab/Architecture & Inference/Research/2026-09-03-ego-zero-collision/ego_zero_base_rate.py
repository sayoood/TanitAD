"""E-ARCH-EGOZERO-1 — the base rate of a near-zero fed ego speed on refcv3's live TRAIN corpus.

CONSUMER-EXACT by construction. The fed value is `v0 = pose_last[:, 3]`
(`scripts/refc_v3_train.py:445`), `pose_last = ep.poses[t + window - 1]`
(`tanitad/data/_contract.py:137`), enumeration `range(T - window - max_horizon)`
(`scripts/refb_train.py:117`, window=8 `refc.py:419`, max_horizon=20
`refc_v3_train.py:798`). `ep.poses` IS `man["poses"][i]` — the manifest this
script reads is the same object `build_v2_providers` hands the dataset
(`tanitad/data/v2_dataset.py:543`). Speed provenance: `v = ||(vx, vy)||` from the
egomotion parquet (`tanitad/data/physicalai.py:618-619`).

Input : train_manifest.pt  (scp of tanitad-refcv3:/root/data/train/_v2manifest.pt,
                            4,572 clips, 24,881,873 B, pulled 2026-09-03)
Output: raw/ego_zero_base_rate.json
"""
import json
import os

import numpy as np
import torch

WINDOW = 8          # refc.py:419  RefCConfig.window
MAX_HORIZON = 20    # refc_v3_train.py:798  V3Dataset(..., max_horizon=20)
MAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_manifest.pt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ego_zero_base_rate.json")
# Thresholds asked for by the brief, plus the fp32 exact-zero cell.
THRESH = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
EGO_DROPOUT = 0.5   # refc.py:429


def main():
    man = torch.load(MAN, map_location="cpu", weights_only=False)
    assert man["version"] == 3, man["version"]
    poses = man["poses"]
    n_clips = len(poses)
    fed = []           # the v0 actually fed, one entry per TRAINER WINDOW
    per_clip_T = []
    for p in poses:
        a = np.asarray(p, dtype=np.float64)
        T = a.shape[0]
        per_clip_T.append(int(T))
        n_win = T - WINDOW - MAX_HORIZON
        assert n_win > 0, (T, n_win)
        # pose_last index = t + WINDOW - 1 for t in range(n_win)
        fed.append(a[WINDOW - 1: WINDOW - 1 + n_win, 3])
    v = np.concatenate(fed)
    n = int(v.size)
    assert n == sum(t - WINDOW - MAX_HORIZON for t in per_clip_T)

    # --- 1. base rate -----------------------------------------------------
    cum = {}
    for th in THRESH:
        c = int((v <= th).sum()) if th > 0 else int((v == 0.0).sum())
        cum[f"{th:g}"] = {"count": c, "frac": c / n}
    exact_zero = int((v == 0.0).sum())
    # negative or non-finite would be a data defect — assert loudly
    n_nonfinite = int((~np.isfinite(v)).sum())
    n_negative = int((v < 0).sum())

    # --- 2. the collision arithmetic -------------------------------------
    # At the model input, `v` is 0.0 iff (dropout withheld it) OR (v0 was 0.0).
    # keep=1 always here (the trainer ALWAYS passes v0, refc_train.py:38), so
    #   P(input==0) = p + (1-p)*P(v0==0)
    # and the share of zero-input samples that are GENUINE stationary is
    #   (1-p)*P(v0==0) / P(input==0).
    p = EGO_DROPOUT
    coll = {}
    for th in THRESH:
        key = f"{th:g}"
        q = cum[key]["frac"]                       # P(v0 in the zero-ish band)
        p_zero_in = p + (1 - p) * q
        genuine = (1 - p) * q
        coll[key] = {
            "P_v0_in_band": q,
            "P_input_reads_zero": p_zero_in,
            "share_withheld": p / p_zero_in,
            "share_genuine": genuine / p_zero_in,
            "odds_withheld_to_genuine": p / genuine if genuine > 0 else None,
        }

    # --- extra: the exact-zero band is the only BYTE-IDENTICAL collision --
    q0 = exact_zero / n
    exact = {
        "P_v0_exactly_0": q0,
        "P_input_exactly_0": p + (1 - p) * q0,
        "share_withheld": p / (p + (1 - p) * q0),
        "share_genuine": ((1 - p) * q0) / (p + (1 - p) * q0),
    }

    res = {
        "instrument": "ego_zero_base_rate.py",
        "manifest": {"path": MAN, "bytes": os.path.getsize(MAN),
                     "n_clips": n_clips, "version": int(man["version"])},
        "enumeration": {"window": WINDOW, "max_horizon": MAX_HORIZON,
                        "pose_last_index": "t + window - 1",
                        "t_range": "range(T - window - max_horizon)"},
        "n_windows": n,
        "T_out": {"min": int(min(per_clip_T)), "max": int(max(per_clip_T)),
                  "median": float(np.median(per_clip_T))},
        "speed_stats": {"mean": float(v.mean()), "median": float(np.median(v)),
                        "std": float(v.std()), "min": float(v.min()),
                        "max": float(v.max()),
                        "p01": float(np.percentile(v, 1)),
                        "p05": float(np.percentile(v, 5)),
                        "p10": float(np.percentile(v, 10)),
                        "p25": float(np.percentile(v, 25)),
                        "p75": float(np.percentile(v, 75)),
                        "p99": float(np.percentile(v, 99))},
        "data_health": {"n_nonfinite": n_nonfinite, "n_negative": n_negative},
        "exact_zero_windows": exact_zero,
        "cumulative_at_or_below": cum,
        "collision": coll,
        "collision_exact_zero": exact,
        "ego_dropout": p,
        "histogram_0_to_2_ms_bin_0p1": [
            int(c) for c in np.histogram(v, bins=20, range=(0.0, 2.0))[0]],
        # per-clip: how many clips are stationary for a MEANINGFUL share
        "clips_with_any_window_below_0p5": int(sum(
            1 for f in fed if (f <= 0.5).any())),
        "clips_with_over_half_windows_below_0p5": int(sum(
            1 for f in fed if (f <= 0.5).mean() > 0.5)),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"n_windows={n} over {n_clips} clips")
    print(f"exact 0.0 : {exact_zero} ({100 * q0:.4f} %)")
    for th in THRESH:
        k = f"{th:g}"
        print(f"<= {k:>4} m/s: {cum[k]['count']:>7} ({100 * cum[k]['frac']:6.3f} %)"
              f"  -> P(input==0)={100 * coll[k]['P_input_reads_zero']:.3f} %"
              f", genuine share {100 * coll[k]['share_genuine']:.2f} %")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
