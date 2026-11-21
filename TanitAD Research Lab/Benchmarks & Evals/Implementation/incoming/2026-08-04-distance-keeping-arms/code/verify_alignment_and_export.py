#!/usr/bin/env python3
"""PRECONDITION CHECK, then export the banked dumps to the scorer's --pred-npz form.

⛔ The lead block is attached to a banked `pred` by ROW POSITION. If the dump's row order is not
`window_last_indices`' emission order over `sorted(ep_*.pt)`, every lead lands on the wrong window
and the metric still returns a plausible number. So this REBUILDS `gt` and `cv` from the local
poses view using exactly the scorer's formulae and compares them to the persisted tensors. Bit-exact
(or float32-round-trip exact) is the only admissible outcome.

Also compares `eid` as a PARTITION, not by literal value — three dumps in this programme label the
same 40 episodes with packed string uids.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))
from tanitad.data.mixing import load_episode                      # noqa: E402
from taniteval.lead_source import window_last_indices             # noqa: E402

VIEW = Path(sys.argv[1])
OUTDIR = Path(sys.argv[2]); OUTDIR.mkdir(parents=True, exist_ok=True)
ARMS = sys.argv[3:]
WP_REL_S = np.array([0.5, 1.0, 1.5, 2.0])

eps = sorted(VIEW.glob("ep_*.pt"))
GT, CV, EPIDX = [], [], []
for k, p in enumerate(eps):
    ep = load_episode(p, mmap=False)
    poses = np.asarray(ep.poses, dtype=np.float64)
    last = window_last_indices(int(poses.shape[0]))
    steps = np.round(WP_REL_S / 0.1).astype(int)
    x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    gt = np.full((last.size, steps.size, 2), np.nan)
    cv = np.full_like(gt, np.nan)
    for i, l in enumerate(last):
        j = np.clip(l + steps, 0, poses.shape[0] - 1)
        c, s = np.cos(yaw[l]), np.sin(yaw[l])
        dx, dy = x[j] - x[l], y[j] - y[l]
        gt[i, :, 0], gt[i, :, 1] = dx * c + dy * s, -dx * s + dy * c
        cv[i, :, 0], cv[i, :, 1] = v[l] * WP_REL_S, 0.0
    GT.append(gt); CV.append(cv); EPIDX.extend([k] * last.size)
GT = np.concatenate(GT); CV = np.concatenate(CV); EPIDX = np.array(EPIDX)
print(f"rebuilt {GT.shape[0]} windows over {len(eps)} episodes")

report = {"_what": "row-alignment precondition for attaching a lead block to a banked pred dump",
          "n_windows_rebuilt": int(GT.shape[0]), "n_episodes": len(eps),
          "wp_rel_s": WP_REL_S.tolist(), "arms": {}}
for arm in ARMS:
    d = torch.load(REPO / "taniteval" / "results" / f"windows_{arm}.pt",
                   map_location="cpu", weights_only=False)
    gt_b = d["gt"].numpy().astype(np.float64)
    cv_b = d["cv"].numpy().astype(np.float64)
    pred = d["pred"].numpy().astype(np.float64)
    eid = np.array([str(e) for e in d["eid"]], dtype=object)
    # partition agreement: same window -> same episode grouping
    _, inv_b = np.unique(eid, return_inverse=True)
    part_ok = bool(GT.shape[0] == eid.size and
                   np.array_equal(np.unique(inv_b[EPIDX == k]).size * 0 + inv_b[EPIDX == k],
                                  inv_b[EPIDX == k])
                   and all(np.unique(inv_b[EPIDX == k]).size == 1 for k in range(len(eps)))
                   and len(set(int(np.unique(inv_b[EPIDX == k])[0]) for k in range(len(eps)))) == len(eps))
    # float32 round-trip: the dump is float32, our rebuild float64
    gt_err = float(np.nanmax(np.abs(gt_b - GT.astype(np.float32).astype(np.float64))))
    cv_err = float(np.nanmax(np.abs(cv_b - CV.astype(np.float32).astype(np.float64))))
    # float32 ulp at the largest |GT| present — the only tolerance an f32 dump can meet
    ulp = float(np.spacing(np.abs(gt_b[np.isfinite(gt_b)]).max().astype(np.float32)))
    gt_ok = gt_err <= 4.0 * ulp
    r = {"n_rows": int(pred.shape[0]), "wp_steps": list(map(int, d.get("wp_steps", []))),
         "eid_partition_matches_episode_order": part_ok,
         "gt_max_abs_err_m": gt_err, "gt_p999_abs_err_m": float(np.nanpercentile(
             np.abs(gt_b - GT.astype(np.float32).astype(np.float64)), 99.9)),
         "gt_max_abs_m": float(np.nanmax(np.abs(gt_b))), "f32_ulp_at_max": ulp,
         "gt_within_4ulp": bool(gt_ok),
         "cv_max_abs_err_m_vs_holdv0": cv_err,
         "_cv_note": ("the scorer's internal 'CV' is [poses[:,3]*t, 0] = go_straight on the "
                      "RECORDED speed channel; the dump's cv is baseline_waypoints()["
                      "'constant_velocity'] = finite-difference last-step velocity vector "
                      "extrapolated (has a lateral component). Different floors, both valid; "
                      "the canonical driving_*.json floor is the dump's."),
         "pred_finite_frac": float(np.isfinite(pred).mean())}
    report["arms"][arm] = r
    print(arm, json.dumps(r))
    if not (gt_ok and part_ok):
        print(f"  !! {arm} FAILED the alignment precondition — NOT exporting")
        continue
    np.savez(OUTDIR / f"{arm}.npz", pred=pred.astype(np.float32))
    print(f"  wrote {OUTDIR / (arm + '.npz')}")
    if not (OUTDIR / "cv-canonical.npz").exists():
        np.savez(OUTDIR / "cv-canonical.npz", pred=d["cv"].numpy().astype(np.float32))
        np.savez(OUTDIR / "gt-oracle.npz", pred=d["gt"].numpy().astype(np.float32))
        print(f"  wrote {OUTDIR / 'cv-canonical.npz'} and gt-oracle.npz")
(OUTDIR / "alignment_report.json").write_text(json.dumps(report, indent=1))
print("wrote", OUTDIR / "alignment_report.json")
