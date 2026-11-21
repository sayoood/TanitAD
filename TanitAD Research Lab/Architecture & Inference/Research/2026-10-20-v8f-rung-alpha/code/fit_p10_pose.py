#!/usr/bin/env python3
"""Fit the POSE-ONLY P10 (mode `p10_pose`) with the v8f_mini_fit_p10 recipe, on the FULL train4572 view.

The recipe is `stack/scripts/v8f_mini_fit_p10.py` unchanged where it can be: MLP 8 -> 256 -> 60 (`fk.fit_p10`: wd in {1e-4, 1e-2}
and epochs by 4-fold grouped inner CV), the 0.2 s operative grid, 30 knots = 6 s, stride 8, `t_first` 3, rows ordered by clip
sha12, v2-manifest frame i + 2. What differs (and ONLY this): the three kinematic inputs. `v8f_mini_fit_p10` reads
(v0, a0 = 2-frame difference, recorded STEER) and integrates the steer as tan(k)/L; this reads the POSE TRACK only,
`p10pose_np.pose_kin` = `ha0_ext_pose`'s own derivation (ONE-frame a0, kappa0 = clip(omega0 / max(v0, 2.0), +-0.3)), and
integrates kappa0 as a curvature. No steer, no action channel, no frame and no label enters the fit.

TRAIN clips only: the manifest is the train4572 view; the fit asserts it disjoint from eval139 and eval141 by clip id AND by
stable episode id before it fits anything.

Outputs (--out-dir): p10_pose_prior.pt (`v8f_mini.save_p10` format; meta['units'] == 'pose_kappa'), p10_fit.json,
p10_pose_eval_pred6.npy [2399, 30, 2], fit_clip_ids.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                   # noqa: E402
import torch                                                                         # noqa: E402
import p10pose_np as L                                                               # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
from tanitad.data import v8f_mini_data as D                                          # noqa: E402
from tanitad.models import v8f_mini as M                                             # noqa: E402
from tanitad.models.p10_prior import assert_fit_disjoint                             # noqa: E402

PKG_MINI = os.path.join(L.ROOT, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw")


def window_kin_pose(P, ts):
    """(kin [n, 3] = (v0, a0, kappa0) from the pose track, gt [n, 30, 2], grid) for operative steps ``ts`` -- the frame is
    ``k = 2 t`` exactly as `v8f_mini_data.window_kin`; GT from the 0.2 s pose grid ``P[0::2]``."""
    ts = np.asarray(ts, np.int64)
    k = 2 * ts
    kin = L.pose_kin(P[:, 3], P[:, 2], k)
    grid = P[0::2]
    gt = fk.gt_waypoints(grid, ts, np.arange(1, D.HORIZON + 1))
    return kin, gt, grid


def _base(kin):
    ha0 = fk.ha0_paths(kin[:, 0], D.DT, D.HORIZON)
    he = fk.const_control_paths(kin[:, 0], kin[:, 1], kin[:, 2], D.DT, D.HORIZON, action_units="kappa")
    return fk.kd_x(ha0, he).astype(np.float64), he


def build_rows(poses_by_clip, clips, stride, t_first=3):
    kf, y, base, g, grp = [], [], [], [], []
    for gi, c in enumerate(clips):
        P = poses_by_clip[c]
        Tg = (len(P) + 1) // 2
        ts = np.arange(t_first, Tg - D.HORIZON, stride)
        ts = ts[2 * ts < len(P)]
        if len(ts) == 0:
            continue
        kin, gt, _ = window_kin_pose(P, ts)
        ok = np.isfinite(kin).all(1) & np.isfinite(gt).all((1, 2))
        kin, gt = kin[ok], gt[ok]
        b, _ = _base(kin)
        kf.append(fk.kin_features(kin))
        base.append(b)
        g.append(gt)
        grp.append(np.full(len(kin), gi))
        y.append((gt - b).reshape(len(gt), -1))
    cat = np.concatenate
    return cat(kf), cat(y), cat(base), cat(g), cat(grp)


def eval_rows(eval_kin, grid):
    z = np.load(eval_kin, allow_pickle=True)
    off = np.concatenate([[0], np.cumsum(z["n"])])
    kf, base, g, he = [], [], [], []
    for i, c in enumerate(z["clip"].tolist()):
        P = z["poses"][off[i]:off[i + 1]].astype(np.float64)
        kin, gt, _ = window_kin_pose(P, grid["windows"][c])
        b, e = _base(kin)
        kf.append(fk.kin_features(kin))
        base.append(b)
        g.append(gt)
        he.append(e)
    return np.concatenate(kf), np.concatenate(base), np.concatenate(g), np.concatenate(he)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=L.TRAIN_MAN)
    ap.add_argument("--eval139-manifest", default=L.EVAL_MAN)
    ap.add_argument("--eval-kin", default=os.path.join(PKG_MINI, "eval141_kin.npz"))
    ap.add_argument("--eval-grid", default=os.path.join(PKG_MINI, "eval_grid_windows.json"))
    ap.add_argument("--clips", type=int, default=0, help="0 = every train clip (the full 4,572-clip view)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "8")))
    t0 = time.time()
    m = torch.load(a.manifest, weights_only=False)
    byc = {str(c): D.full_from_manifest(p.numpy(), ac.numpy())[0] for c, p, ac in zip(m["clip_id"], m["poses"], m["actions"])}
    clips = sorted(byc, key=D.sha12)
    if a.clips:
        clips = clips[:a.clips]
    # ---- disjointness BEFORE any fit: train clips vs eval139 and vs eval141 (clip id AND stable episode id) ------------
    e139 = [str(c) for c in torch.load(a.eval139_manifest, weights_only=False)["clip_id"]]
    z = np.load(a.eval_kin, allow_pickle=True)
    e141 = [str(c) for c in z["clip"].tolist()]
    dis = {"eval139": assert_fit_disjoint(clips, e139), "eval141": assert_fit_disjoint(clips, e141)}
    print("[fit_p10_pose] disjoint:", dis, flush=True)
    grid = json.load(open(a.eval_grid))
    kf, y, base, g, grp = build_rows(byc, clips, a.stride)
    print(f"[fit_p10_pose] clips={len(clips)} rows={len(kf)} base ADE6 (train rows)={float(fk.ade(base, g).mean()):.4f} "
          f"prep {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    net, st, best = fk.fit_p10(kf, y, base, g, grp, seed=a.seed)
    fit_s = time.time() - t1
    kfe, be, ge, hee = eval_rows(a.eval_kin, grid)
    pred = fk.predict_p10((net, st, best), kfe, be)
    ade6, fde6 = fk.ade(pred, ge), fk.fde(pred, ge)
    ade2 = fk.ade(pred[:, :10], ge[:, :10])
    res = {"variant": "pose", "units": "pose_kappa", "manifest": a.manifest,
           "manifest_md5": hashlib.md5(open(a.manifest, "rb").read()).hexdigest(), "n_manifest_clips": len(byc),
           "fit_clips": len(clips), "stride": a.stride, "seed": a.seed, "n_fit_rows": int(len(kf)),
           "t_first": 3, "grid_dt_s": D.DT, "knots": D.HORIZON,
           "best": [int(best[0]), float(best[1]), int(best[2])], "fit_seconds": round(fit_s, 1),
           "features": "kin_features((v0, a0 = ONE-frame backward diff, kappa0 = clip(omega0 / max(v0, 2.0), +-0.3))) -- the "
                       "past POSE TRACK only (ha0_ext_pose's own derivation); no steer, no action channel",
           "base": "kd_x_pose = retime(damp50(ha0, ha0_ext_pose), ha0_ext_pose), 30 x 0.2 s, kappa integrated as a curvature",
           "disjoint_from_eval": dis, "train_eval_overlap": 0,
           "eval141": {"n": int(len(ge)), "ade6_p10pose_m": float(ade6.mean()), "fde6_p10pose_m": float(fde6.mean()),
                       "ade2_first10knots_m": float(ade2.mean()),
                       "kd_x_pose_ade6_m": float(fk.ade(be, ge).mean()),
                       "ha0_ext_pose_ade6_m": float(fk.ade(hee, ge).mean()),
                       "acts": fk.acts_check(pred, D.DT),
                       "banked_steer_p10_ade6_m": 2.5743599929068472,
                       "note": "the steer P10 (v8F-mini, 600 clips) read 2.5744 m on this eval141 grid"},
           "seconds": round(time.time() - t0, 1)}
    prior = M.P10Prior.from_fit((net, st, best))
    M.save_p10(prior, os.path.join(a.out_dir, "p10_pose_prior.pt"),
               {"units": "pose_kappa", "variant": "pose", "fit_clips": len(clips), "stride": a.stride, "best": res["best"],
                "eval_ade6_m": res["eval141"]["ade6_p10pose_m"]})
    np.save(os.path.join(a.out_dir, "p10_pose_eval_pred6.npy"), pred.astype(np.float32))
    open(os.path.join(a.out_dir, "fit_clip_ids.txt"), "w", encoding="utf-8").write("\n".join(clips) + "\n")
    res["artifact_sha256"] = hashlib.sha256(open(os.path.join(a.out_dir, "p10_pose_prior.pt"), "rb").read()).hexdigest()
    json.dump(res, open(os.path.join(a.out_dir, "p10_fit.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("fit_clips", "n_fit_rows", "best", "fit_seconds", "eval141", "artifact_sha256")},
                     indent=1, default=str))
    print("ZZFITPOSE-DONEZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
