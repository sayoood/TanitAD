"""EXECUTED-manoeuvre tactical metrics for v1arch vs v1.6 vs GT — from the banked dump.

The declared tactical head is FROZEN and identical in both arms (dwell 0.55 s, toggle
0.1759 — the measured defect). What the readout swap CAN change is the manoeuvre the
trajectory actually EXECUTES. This scores that, with the label rule the programme's own
tactical labels use (`refb_labels.classify_maneuver`: dyaw + dv over the 2 s horizon —
frame-invariant, so it applies to each window's origin-frame path directly).

Per window (stride-1, 6,834 rows): yaw0 = 0 (origin frame), yaw1 from the path's yaw
channel (or last-segment heading if the dump stores only xy), v0 = |first GT segment|/dt
(entry speed, shared by construction across arms), v1 = |arm's last segment|/dt.

Reported per arm: executed-class distribution, agreement with GT-executed (acc + per-class),
toggle rate and mean dwell over consecutive stride-1 windows (the executed twin of the
declared-head numbers), episode-cluster bootstrap CIs on toggle rate via per-episode rates.
"""
import glob
import json

import numpy as np
import torch

import sys
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import refb_labels as rl

DT = 0.1
CLASSES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")


def yaw1_of(P):
    """P [n,20,C] origin-frame path -> yaw at the 2 s horizon."""
    if P.shape[2] >= 3:
        return torch.as_tensor(P[:, -1, 2], dtype=torch.float64)
    d = P[:, -1, :2] - P[:, -2, :2]
    return torch.atan2(torch.as_tensor(d[:, 1], dtype=torch.float64),
                       torch.as_tensor(d[:, 0], dtype=torch.float64))


def speeds_of(P, G):
    v0 = np.linalg.norm(G[:, 0, :2], axis=1) / DT          # entry speed, GT segment
    v1 = np.linalg.norm(P[:, -1, :2] - P[:, -2, :2], axis=1) / DT
    return (torch.as_tensor(v0, dtype=torch.float64),
            torch.as_tensor(v1, dtype=torch.float64))


def exec_classes(P, G):
    y0 = torch.zeros(P.shape[0], dtype=torch.float64)
    v0, v1 = speeds_of(P, G)
    return rl.classify_maneuver(y0, yaw1_of(P), v0, v1).numpy()


def runs(seq):
    r, n = [], 0
    for i, s in enumerate(seq):
        n += 1
        if i + 1 == len(seq) or seq[i + 1] != s:
            r.append(n)
            n = 0
    return r


dumps = sorted(glob.glob("/workspace/v16_eval/dump/ep*.npz"))
assert len(dumps) == 40
per_ep = {k: [] for k in ("v1arch", "v16", "gt")}      # per-episode class sequences
for f in dumps:
    d = np.load(f)
    G = d["g"].astype(np.float64)
    for key, arr in (("v1arch", d["a"]), ("v16", d["b"]), ("gt", G)):
        per_ep[key].append(exec_classes(arr.astype(np.float64), G))
n_win = sum(len(s) for s in per_ep["gt"])
print(f"[windows] {n_win} across 40 episodes  path_dims={np.load(dumps[0])['a'].shape}",
      flush=True)

rng = np.random.default_rng(0)
out = {"_n_windows": int(n_win), "_grid": "stride-1, 6,834 windows, 40 OOD-val q90 episodes",
       "_label_rule": "refb_labels.classify_maneuver over the 2 s window horizon "
                      "(executed, from the trajectory; the DECLARED head is frozen and "
                      "identical in both arms)"}
gt_all = np.concatenate(per_ep["gt"])
for key in ("v1arch", "v16", "gt"):
    allc = np.concatenate(per_ep[key])
    dist = {CLASSES[c]: int((allc == c).sum()) for c in range(5)}
    agree = float((allc == gt_all).mean())
    conf = [[int(((gt_all == g) & (allc == p)).sum()) for p in range(5)] for g in range(5)]
    ep_toggle = np.array([float((np.diff(s) != 0).mean()) for s in per_ep[key]])
    ep_dwell = np.array([float(np.mean(runs(list(s)))) for s in per_ep[key]])
    draws = [float(ep_toggle[rng.integers(0, 40, 40)].mean()) for _ in range(2000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out[key] = {"dist": dist, "agreement_with_gt_executed": agree,
                "confusion_gt_rows_pred_cols": conf,
                "toggle_rate": float(ep_toggle.mean()),
                "toggle_rate_ci95": [float(lo), float(hi)],
                "mean_dwell_windows": float(ep_dwell.mean()),
                "mean_dwell_s": float(ep_dwell.mean() * DT),
                "estimator": "episode-cluster bootstrap over 40 episodes, 2000 draws"}
    print(f"[{key}] agree={agree:.4f}  toggle={ep_toggle.mean():.4f} "
          f"[{lo:.4f},{hi:.4f}]  dwell={ep_dwell.mean()*DT:.3f}s  dist={dist}", flush=True)

# paired toggle-rate deltas on the same episodes
for a, b, name in (("v1arch", "v16", "v16_minus_v1arch"), ("gt", "v16", "v16_minus_gt")):
    ta = np.array([float((np.diff(s) != 0).mean()) for s in per_ep[a]])
    tb = np.array([float((np.diff(s) != 0).mean()) for s in per_ep[b]])
    d = tb - ta
    draws = [float(d[rng.integers(0, 40, 40)].mean()) for _ in range(2000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out[f"paired_toggle_{name}"] = {
        "delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
        "separated": bool(lo > 0 or hi < 0),
        "estimator": "paired episode-cluster bootstrap, 2000 draws"}
    print(f"[paired {name}] delta={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

json.dump(out, open("/workspace/v16_eval/v16_tactical_executed.json", "w"), indent=1)
print("TACT_DONE", flush=True)
