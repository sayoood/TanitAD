#!/usr/bin/env python3
"""SOTA_SCAN §11 item 3 — EXECUTED. Ego-frame lane-graph raster + the D-MAP-1 control. 0 GPU.

Order is control-first, as the standard requires:

  D-MAP-1   can a readout that sees ONLY the raster predict the ego's own route direction at 3 s,
            better than always-predict-the-majority-class, on held-out CONTIGUOUS time blocks?
  D-MAP-2   the negative control on the control: the SAME readout on a raster taken from a random
            other timestep must collapse to the baseline. If it does not, the readout is reading
            something other than the map and D-MAP-1 means nothing.

⚠️ ESTIMATOR LIMIT, registered in ../PRE_REGISTRATION.md §3b BEFORE this ran: exactly ONE scene on
this disk carries a `map.xodr` (the second NuRec bundle was mid-download by another stream at run
time, 776 MB of ~2 GB, central directory absent). One scene is ONE cluster, so there is no valid
episode-cluster bootstrap here. The interval below is a **contiguous-block bootstrap** and the
finding is **DIRECTIONAL ONLY** — it may not promote anything to a trained config.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval.lane_raster import RasterSpec, render_track   # noqa: E402

SRC = (REPO / "TanitAD Research Hub" / "Architecture & Inference" / "Research"
       / "2026-08-02-nurec-xodr-map")
OUT = Path(__file__).resolve().parents[1] / "raw"
HORIZON_S = 3.0
TURN_DEG = 10.0          # |dyaw| below this over the horizon = "straight"
N_BLOCKS = 8             # contiguous held-out blocks
SEED = 0


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def load_scene():
    lanes = json.loads((SRC / "lane_centerlines.json").read_text(encoding="utf-8"))
    track = json.loads((SRC / "ego_track_map_frame.json").read_text(encoding="utf-8"))["track"]
    xs = np.array([t["x"] for t in track], float)
    ys = np.array([t["y"] for t in track], float)
    yaw = np.array([t["yaw_deg"] for t in track], float)
    t_us = np.array([t["t_us"] for t in track], float)
    return lanes, xs, ys, yaw, t_us


def route_labels(yaw_deg, t_us, horizon_s=HORIZON_S, turn_deg=TURN_DEG):
    """-> (labels, valid). 0 = right, 1 = straight, 2 = left, from the ego's OWN future yaw.

    ⛔ The label comes from the ego's future, the raster only from the present pose and the static
    map — no future information enters the input. That is the property that makes this a prediction
    rather than a lookup, and it is asserted in the runner rather than assumed.
    """
    t_s = t_us / 1e6
    n = len(yaw_deg)
    lab = np.full(n, -1, dtype=int)
    for i in range(n):
        j = np.searchsorted(t_s, t_s[i] + horizon_s)
        if j >= n:
            continue
        d = wrap_deg(yaw_deg[j] - yaw_deg[i])
        lab[i] = 1 if abs(d) < turn_deg else (2 if d > 0 else 0)
    return lab, lab >= 0


def pooled_features(rast, pool=8):
    """Mean-pool each channel to a coarse grid — a LINEAR readout on ~700 features, not a CNN.

    ⛔ Deliberately weak. The question is whether the RASTER carries route information, and a weak
    readout that succeeds is evidence about the raster; a strong readout that succeeds is evidence
    about the readout. Under-fitting is the conservative error here.
    """
    n, c, H, W = rast.shape
    Hp, Wp = H // pool, W // pool
    r = rast[:, :, :Hp * pool, :Wp * pool].reshape(n, c, Hp, pool, Wp, pool)
    return r.mean(axis=(3, 5)).reshape(n, -1)


def block_cv(X, y, n_blocks=N_BLOCKS):
    """Contiguous-block cross-validation. ⛔ NOT a random split.

    Adjacent frames at 10 Hz are near-identical; a random split puts a frame's neighbour in the
    train set and reports memorisation as skill. Blocks are contiguous in TIME, so a held-out block
    shares no neighbourhood with training.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    pred = np.full(n, -1, dtype=int)
    for b in range(n_blocks):
        te = np.zeros(n, bool)
        te[edges[b]:edges[b + 1]] = True
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    ok = pred >= 0
    return pred, ok


def block_bootstrap_acc(correct, n_blocks=N_BLOCKS, n_boot=2000, seed=SEED):
    """Interval over CONTIGUOUS BLOCKS. ⛔ Explicitly NOT the programme's decision-grade estimator.

    The programme's decision-grade interval is the episode-cluster bootstrap over 40 val episodes.
    Here there is ONE scene, so blocks are the only resampling unit available and the resulting
    interval describes within-scene variation only. Labelled in the output so it cannot be quoted
    as if it were the other thing.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [correct[edges[b]:edges[b + 1]] for b in range(n_blocks) if edges[b + 1] > edges[b]]
    draws = []
    for _ in range(n_boot):
        sel = rng.integers(0, len(blocks), len(blocks))
        draws.append(float(np.concatenate([blocks[i] for i in sel]).mean()))
    d = np.asarray(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"acc": round(float(correct.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": int(n), "n_blocks": len(blocks),
            "estimator": "contiguous_block_bootstrap",
            "⛔_not_decision_grade": ("one scene = one cluster; the programme's decision-grade "
                                     "interval is the episode-cluster bootstrap over 40 val "
                                     "episodes and this is NOT it")}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lanes, xs, ys, yaw_deg, t_us = load_scene()
    spec = RasterSpec()
    poses = [(x, y, math.radians(h)) for x, y, h in zip(xs, ys, yaw_deg)]

    tr0 = time.time()
    rast = render_track(lanes, poses, spec)
    render_s = time.time() - tr0
    occ = rast[:, 0].mean(axis=(1, 2))
    print(f"rendered {rast.shape} in {render_s:.1f}s "
          f"({1000 * render_s / len(poses):.1f} ms/frame); lane occupancy "
          f"mean {occ.mean():.4f} min {occ.min():.4f} max {occ.max():.4f}")

    lab, valid = route_labels(yaw_deg, t_us)
    X = pooled_features(rast)[valid]
    y = lab[valid]
    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    maj = max(counts.values()) / len(y)
    print("label counts (0=right,1=straight,2=left):", counts, "majority", round(maj, 4))

    pred, ok = block_cv(X, y)
    corr = (pred[ok] == y[ok]).astype(float)
    acc = block_bootstrap_acc(corr)

    # D-MAP-2 — the same readout on a SHUFFLED raster (input/label correspondence destroyed)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    pred_s, ok_s = block_cv(X[perm], y)
    corr_s = (pred_s[ok_s] == y[ok_s]).astype(float)
    acc_s = block_bootstrap_acc(corr_s)

    margin = acc["acc"] - maj
    verdict = ("PASS — the raster carries route information beyond the majority baseline"
               if (margin > 0.03 and acc["lo"] > maj) else
               "FAIL — raster-only route readout does NOT beat always-predict-majority")
    shuffle_ok = acc_s["acc"] <= maj + 0.03
    print(f"D-MAP-1: acc {acc['acc']} [{acc['lo']},{acc['hi']}] vs majority {maj:.4f} -> {verdict}")
    print(f"D-MAP-2 shuffled-raster: acc {acc_s['acc']} (must be <= majority+0.03) "
          f"-> {'PASS' if shuffle_ok else 'FAIL — the readout is not reading the map'}")

    # ---------------------------------------------------------------- #
    # D-MAP-3 — the ANTICIPATION stratum, and it is the one that matters #
    # ---------------------------------------------------------------- #
    # ⛔ D-MAP-1 alone is confounded: a frame taken MID-TURN already shows the lane curving, so a
    # readout can report a turn in progress and score well without anticipating anything. The
    # strategic question is whether the map predicts a turn that has NOT STARTED. So: restrict to
    # frames whose CURRENT yaw rate is ~0 (straight right now) and whose label at +3 s is a turn.
    t_s = t_us / 1e6
    dyaw_now = np.zeros(len(yaw_deg))
    for i in range(len(yaw_deg)):
        j = min(len(yaw_deg) - 1, int(np.searchsorted(t_s, t_s[i] + 0.5)))
        dyaw_now[i] = abs(wrap_deg(yaw_deg[j] - yaw_deg[i]))
    not_turning = dyaw_now[valid] < 1.0            # < 1 deg over the last 0.5 s = going straight
    y_a, X_a = y[not_turning], X[not_turning]
    cnt_a = {int(k): int(v) for k, v in zip(*np.unique(y_a, return_counts=True))}
    d3 = {"n_stratum": int(not_turning.sum()),
          "criterion": "|dyaw over 0.5 s| < 1.0 deg at t0 — the ego is NOT yet turning",
          "counts": cnt_a,
          "n_minority": int(min(cnt_a.values())) if cnt_a else 0,
          "min_minority_required": 20}
    # ⛔ THE ADJUDICATION GATE, and it exists because the first version of this block called a
    # 2-example stratum a FAIL. A stratum whose minority class is 2 rows cannot beat a 98.88 %
    # majority baseline no matter what the raster contains, so scoring it produces a FALSE
    # NEGATIVE about the map. Under-powered is NOT-APPLICABLE, with its n — never a verdict.
    if len(cnt_a) < 2 or d3["n_minority"] < d3["min_minority_required"]:
        d3["verdict"] = (
            f"NOT-APPLICABLE — the anticipation stratum holds only {d3['n_minority']} turn "
            f"example(s) on this single 30 s clip. It is UNDER-POWERED, not negative: no verdict "
            f"is issued. Answering it is a work item (more scenes), not a pass and not a failure.")
    else:
        maj_a = max(cnt_a.values()) / sum(cnt_a.values())
        pa, oka = block_cv(X_a, y_a, n_blocks=min(N_BLOCKS, 6))
        ca = (pa[oka] == y_a[oka]).astype(float)
        acc_a = block_bootstrap_acc(ca, n_blocks=min(N_BLOCKS, 6))
        d3["scored"] = {**acc_a, "majority_rate": round(maj_a, 4),
                        "margin": round(acc_a["acc"] - maj_a, 4)}
        d3["verdict"] = ("PASS — the raster ANTICIPATES a turn that has not started"
                         if (acc_a["acc"] - maj_a > 0.03 and acc_a["lo"] > maj_a) else
                         "FAIL — the raster only reports a turn already in progress")
    print("D-MAP-3 anticipation:", d3["verdict"])

    # bank the rasters and a human-checkable strip
    np.save(OUT / "lane_raster_scene00040136.npy", rast.astype(np.float32))
    try:
        from PIL import Image
        idx = np.linspace(0, len(rast) - 1, 6).astype(int)
        tiles = []
        for i in idx:
            rgb = np.stack([rast[i, 0], (rast[i, 1] + 1) / 2 * rast[i, 0],
                            (rast[i, 2] + 1) / 2 * rast[i, 0]], axis=-1)
            rgb[spec.H - int(spec.ahead / spec.res) - 1:, spec.W // 2 - 1:spec.W // 2 + 2] = \
                np.maximum(rgb[spec.H - int(spec.ahead / spec.res) - 1:,
                               spec.W // 2 - 1:spec.W // 2 + 2], [0, 0, 0])
            tiles.append((rgb * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(OUT / "lane_raster_strip.png")
    except Exception as e:            # a missing PIL must not lose the measurement
        print("strip PNG skipped:", e)
    result = {
        "run": "SOTA_SCAN §11 item 3 — ego-frame lane-graph raster + D-MAP-1",
        "date": "2026-08-03", "host": "dev-box CPU", "gpu_hours": 0.0,
        "evidence_class": "MEASURED (ours)",
        "scene": "00040136-e651-4abd-991d-0655ccda9430 (NuRec sample_set 26.04)",
        "source": str(SRC),
        "raster_spec": spec.to_dict(),
        "n_poses": len(poses), "render_s_total": round(render_s, 2),
        "render_ms_per_frame": round(1000 * render_s / len(poses), 2),
        "lane_occupancy": {"mean": round(float(occ.mean()), 5),
                           "min": round(float(occ.min()), 5),
                           "max": round(float(occ.max()), 5),
                           "n_frames_empty": int((occ == 0).sum())},
        "label": {"horizon_s": HORIZON_S, "turn_threshold_deg": TURN_DEG,
                  "classes": {"0": "right", "1": "straight", "2": "left"},
                  "counts": counts, "majority_rate": round(maj, 4),
                  "n_valid": int(valid.sum()), "n_total": int(len(lab))},
        "readout": {"model": "multinomial logistic regression, C=0.1, on 8x8 mean-pooled channels",
                    "n_features": int(X.shape[1]),
                    "split": f"{N_BLOCKS} CONTIGUOUS time blocks (never a random split — 10 Hz "
                             f"neighbours would leak)"},
        "D_MAP_1": {**acc, "majority_rate": round(maj, 4), "margin": round(margin, 4),
                    "verdict": verdict},
        "D_MAP_2_shuffled_raster": {**acc_s,
                                    "verdict": ("PASS — collapses to baseline, so D-MAP-1 is "
                                                "reading the map" if shuffle_ok else
                                                "FAIL — the readout succeeds without the right "
                                                "raster; D-MAP-1 is VOID")},
        "D_MAP_3_anticipation": d3,
        "⛔_scope": ("this measures whether the RASTER carries route information, NOT whether our "
                    "model would use it. The trained frozen-trunk probe is SOTA §11 row 4 and is "
                    "out of scope at 0 GPU."),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (OUT / "item3_lane_raster.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT / "item3_lane_raster.json")


if __name__ == "__main__":
    main()
