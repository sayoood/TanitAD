"""T2 prerequisite — PROVE the ``egomotion`` <-> ``obstacle.offline`` join.

⚠️ THE CLOCK TRAP (brief): ``egomotion`` spans ~140 s while ``obstacle.offline``
spans ~20 s "with different origins". A mis-stated join has already moved a
headline rate by x5.7 elsewhere in the program, so this is proved, not assumed.

THE PROOF (falsifiable, and it does not assume the answer)
----------------------------------------------------------
``obstacle.offline`` boxes are in ``reference_frame="rig"`` — i.e. EGO-relative.
A world-static object (parked car, sign, pole) therefore has a rig-frame track
that is exactly the ego's own motion, negated. So:

    world_xy(track, t) = ego_pose(t + delta) (+) rig_xy(track, t)

is CONSTANT for a static track **iff** ``delta`` is the true clock offset. We
sweep ``delta`` and measure, per delta, the 10th percentile of per-track
world-position dispersion across all tracks (a delta-symmetric statistic — no
track is pre-selected at any particular delta, so the test cannot be rigged
toward delta = 0). The minimising delta is the offset.

Self-test in BOTH directions, per the standing rule: a deliberately wrong
offset (+1.0 s) must be reported as WORSE, or the instrument may not adjudicate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcl_common import (EGO_ZIP, OBST_ZIP, ego_at, from_ego_frame,  # noqa: E402
                        load_scenes)

MIN_OBS = 8            # a track needs this many samples to have a dispersion


def dispersion_curve(scene, deltas_s: np.ndarray) -> np.ndarray:
    """[len(deltas)] p10 of per-track world-xy dispersion [m] at each delta."""
    o = scene.obst
    t_us = o["timestamp_us"].to_numpy(np.float64)
    rig = np.stack([o["center_x"].to_numpy(np.float64),
                    o["center_y"].to_numpy(np.float64)], axis=1)
    tid = o["track_id"].to_numpy()
    uniq, inv = np.unique(tid, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    keep = counts >= MIN_OBS
    out = np.full(len(deltas_s), np.nan)
    if keep.sum() < 3:
        return out
    for i, d in enumerate(deltas_s):
        pose = ego_at(scene.ego, t_us + d * 1e6)              # [n, 4]
        w = from_ego_frame(rig, pose.T[:, None].T) if False else None
        # vectorised world transform (from_ego_frame is per-origin)
        c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        wx = rig[:, 0] * c - rig[:, 1] * s + pose[:, 0]
        wy = rig[:, 0] * s + rig[:, 1] * c + pose[:, 1]
        disp = []
        for k in np.where(keep)[0]:
            m = inv == k
            disp.append(0.5 * (wx[m].std() + wy[m].std()))
        out[i] = float(np.percentile(disp, 10))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    deltas = np.round(np.arange(-2.0, 2.01, 0.1), 2)
    scenes = load_scenes(limit=a.clips)
    curves, spans = [], []
    for sc in scenes:
        curves.append(dispersion_curve(sc, deltas))
        et = sc.ego["timestamp"].to_numpy(np.float64)
        ot = sc.obst["timestamp_us"].to_numpy(np.float64)
        spans.append(dict(
            alias=sc.alias,
            ego_n=int(len(et)), ego_t0_s=float(et.min() / 1e6),
            ego_t1_s=float(et.max() / 1e6),
            ego_n_in_0_20s=int(((et >= 0) & (et <= 20e6)).sum()),
            ego_dt_med_us=float(np.median(np.diff(np.sort(et)))),
            obst_n=int(len(ot)), obst_t0_s=float(ot.min() / 1e6),
            obst_t1_s=float(ot.max() / 1e6),
            n_tracks=int(sc.obst["track_id"].nunique())))
    C = np.vstack(curves)                                   # [clips, deltas]
    med = np.nanmedian(C, axis=0)
    best = float(deltas[int(np.nanargmin(med))])
    per_clip_best = deltas[np.nanargmin(C, axis=1)]
    z = int(np.where(deltas == 0.0)[0][0])
    p1 = int(np.where(deltas == 1.0)[0][0])

    res = dict(
        what="egomotion <-> obstacle.offline clock join proof",
        method=("world-frame dispersion of rig-frame tracks vs a swept clock "
                "offset; statistic = p10 of per-track xy dispersion, taken "
                "fresh at every delta so no track is pre-selected"),
        source=dict(obstacle=str(OBST_ZIP), egomotion=str(EGO_ZIP),
                    chunk="0000", n_clips=len(scenes)),
        deltas_s=deltas.tolist(),
        median_dispersion_m=[None if np.isnan(v) else round(float(v), 4)
                             for v in med],
        best_delta_s=best,
        per_clip_best_delta_s=per_clip_best.tolist(),
        frac_clips_best_at_zero=float((per_clip_best == 0.0).mean()),
        dispersion_at_0_m=round(float(med[z]), 4),
        dispersion_at_plus1s_m=round(float(med[p1]), 4),
        selftest_failing_input=dict(
            note="a deliberately wrong +1.0 s offset must be reported WORSE",
            ratio_plus1s_over_0=round(float(med[p1] / med[z]), 3),
            passes=bool(med[p1] > med[z])),
        clip_spans=spans)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("best_delta_s", "frac_clips_best_at_zero",
                       "dispersion_at_0_m", "dispersion_at_plus1s_m",
                       "selftest_failing_input")}, indent=1))


if __name__ == "__main__":
    main()
