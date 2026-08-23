"""Measured run: nuScenes-style open-loop L2 (metric-BEV ego frame) for the
trivial nulls and the learned no-vision ego-status shortcut, on our real val
corpora (comma2k19 highway + Cosmos-Drive-Dreams urban).

DATA-ONLY, $0, CPU. No TanitAD model output here -- this establishes the
community-comparable *denominators* (shortcut ceiling + kinematic floor) that a
future TanitAD L2 is divided by (skill_score). Clip-level train/val split (I3):
the learned shortcut is fit on train clips and every predictor is scored on the
held-out val clips, so the shortcut number is honest.

Usage:
  python run_openloop_l2.py \
      --comma-root  C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
      --cosmos-root C:/Users/Admin/tanitad-data/cosmos_bench3 \
      --out results_openloop_l2.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

import baseline_predictors as bp
import openloop_l2 as ol

DT = 0.1
KMAX = 30                       # 3.0 s at 10 Hz
HORIZONS = (1.0, 2.0, 3.0)
SRC_FPS, TARGET_HZ = 30.0, 10.0
T_START = 6                     # need ego-status history + backward diffs
PREDICTORS = ("stop", "go_straight", "cv", "ctrv", "floor", "ego_status_mlp")


def _ensure_stack_on_path() -> None:
    for c in [Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack"),
              Path(r"C:/Users/Admin/tanitad-wt/bench-20260717/stack")]:
        if (c / "tanitad").is_dir():
            sys.path.insert(0, str(c))
            return


def load_comma(root: Path) -> list[np.ndarray]:
    import torch
    out = []
    for f in sorted(glob.glob(str(root / "ep_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out.append(d["poses"].numpy().astype(np.float64))
    return out


def load_cosmos(root: Path) -> list[np.ndarray]:
    _ensure_stack_on_path()
    from tanitad.data.cosmos_drive import poses_to_signals
    stride = int(round(SRC_FPS / TARGET_HZ))
    out = []
    for tar in sorted(glob.glob(str(root / "vehicle_pose" / "*.tar"))):
        frames = {}
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".vehicle_pose.npy"):
                    idx = int(m.name.split(".")[-3])
                    frames[idx] = np.load(io.BytesIO(tf.extractfile(m).read())).reshape(4, 4)
        ego4 = np.stack([frames[i] for i in sorted(frames)]).astype(np.float64)[::stride]
        _, poses = poses_to_signals(ego4, DT)
        out.append(poses.astype(np.float64))
    return out


def anchors_of(poses: np.ndarray):
    """Yield (t, gt_full[KMAX,2], feats, curv_stratum, speed) for valid anchors."""
    x = poses[:, 0].astype(np.float64)
    y = poses[:, 1].astype(np.float64)
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    T = len(x)
    ks = np.arange(1, KMAX + 1)
    h1 = ol.horizon_step(DT, 1.0)
    for t in range(T_START, T - KMAX):
        gt = bp.gt_future_ego(x, y, yaw, t, ks)              # [KMAX,2]
        vx = (x[t] - x[t - 1]) / DT
        vy = (y[t] - y[t - 1]) / DT
        speed = float(np.hypot(vx, vy))
        turn = float(np.degrees(yaw[t + h1] - yaw[t]))
        curv = "standstill" if speed < 2.0 else bp.curvature_stratum(turn)
        feats = ol.ego_status_features(poses, t, DT)
        yield t, gt, feats, curv, speed


def build_corpus(name: str, sequences: list[np.ndarray]) -> dict:
    # clip-level split: every 3rd sequence -> val (deterministic, I3)
    train_idx = [i for i in range(len(sequences)) if i % 3 != 0]
    val_idx = [i for i in range(len(sequences)) if i % 3 == 0]

    Xtr, Ytr = [], []
    for i in train_idx:
        for _t, gt, feats, _c, _s in anchors_of(sequences[i]):
            Xtr.append(feats); Ytr.append(gt.reshape(-1))
    head = ol.RidgeTrajectoryHead(lam=10.0).fit(np.asarray(Xtr), np.asarray(Ytr))

    # score every predictor on held-out val anchors
    recs = []
    for i in val_idx:
        poses = sequences[i]
        for t, gt, feats, curv, speed in anchors_of(poses):
            preds = ol.kinematic_preds_full(poses, t, DT, KMAX)
            preds["ego_status_mlp"] = head.predict(feats[None, :])[0]
            # per-anchor best-of-3 kinematic floor (min avg_pointwise)
            floor_name = min(("cv", "go_straight", "ctrv"),
                             key=lambda n: ol.l2_metrics(preds[n], gt, DT)["avg_pointwise"])
            preds["floor"] = preds[floor_name]
            rec = {"curv": curv, "speed": speed}
            for p in PREDICTORS:
                m = ol.l2_metrics(preds[p], gt, DT, HORIZONS)
                for h in HORIZONS:
                    rec[f"{p}_point_{h:g}s"] = m["pointwise"][h]
                    rec[f"{p}_cumul_{h:g}s"] = m["cumulative"][h]
                rec[f"{p}_avg_point"] = m["avg_pointwise"]
            recs.append(rec)

    def agg(subset, key):
        a = np.asarray([r[key] for r in subset], dtype=float)
        return {"median": round(float(np.median(a)), 3),
                "mean": round(float(np.mean(a)), 3), "n": int(a.size)}

    out = {"corpus": name, "n_seq": len(sequences),
           "n_train_anchors": len(Xtr), "n_val_anchors": len(recs),
           "speed_mps": agg(recs, "cv_avg_point") and
                        {"median": round(float(np.median([r["speed"] for r in recs])), 2)}}
    # overall L2 table (pointwise = UniAD convention, the leaderboard one)
    out["L2_pointwise"] = {p: {f"{h:g}s": agg(recs, f"{p}_point_{h:g}s")["median"]
                               for h in HORIZONS} for p in PREDICTORS}
    out["L2_cumulative"] = {p: {f"{h:g}s": agg(recs, f"{p}_cumul_{h:g}s")["median"]
                                for h in HORIZONS} for p in PREDICTORS}
    out["avg_pointwise"] = {p: agg(recs, f"{p}_avg_point")["median"] for p in PREDICTORS}
    # curvature population (our-corpus analogue of nuScenes' 74% straight)
    pop = {}
    for c in ("straight", "gentle", "sharp", "standstill"):
        sub = [r for r in recs if r["curv"] == c]
        pop[c] = {"frac": round(len(sub) / max(1, len(recs)), 3),
                  "shortcut_point_1s": (agg(sub, "ego_status_mlp_point_1s")["median"]
                                        if sub else None),
                  "floor_point_1s": (agg(sub, "floor_point_1s")["median"] if sub else None)}
    out["by_curvature"] = pop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comma-root", required=True)
    ap.add_argument("--cosmos-root", required=True)
    ap.add_argument("--out", default="results_openloop_l2.json")
    args = ap.parse_args()

    result = {"protocol": {"frame": "metric-BEV ego (FLU)", "unit": "m", "hz": TARGET_HZ,
                           "horizons_s": list(HORIZONS), "split": "clip-level 2/3 train, 1/3 val",
                           "conventions": ["pointwise=UniAD", "cumulative=ST-P3/VAD"]}}
    result["comma_highway"] = build_corpus("comma_highway", load_comma(Path(args.comma_root)))
    result["cosmos_urban"] = build_corpus("cosmos_urban", load_cosmos(Path(args.cosmos_root)))
    Path(args.out).write_text(json.dumps(result, indent=1))

    for corpus in ("comma_highway", "cosmos_urban"):
        r = result[corpus]
        print(f"\n=== {corpus}  ({r['n_seq']} seq, {r['n_val_anchors']} val anchors, "
              f"v~{r['speed_mps']['median']} m/s) ===")
        print(f"  {'predictor':<16} L2@1s  L2@2s  L2@3s   avg(pointwise, metric-BEV m)")
        for p in PREDICTORS:
            lp = r["L2_pointwise"][p]
            print(f"  {p:<16} {lp['1s']:>5}  {lp['2s']:>5}  {lp['3s']:>5}   {r['avg_pointwise'][p]}")
        print(f"  curvature pop: "
              + " ".join(f"{c}={r['by_curvature'][c]['frac']}" for c in
                         ("straight", "gentle", "sharp", "standstill")))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
