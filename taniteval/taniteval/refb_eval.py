"""TanitEval — REF-B planner-native trajectory eval.

REF-B has no grounded step-readout (it is a hierarchical planner), so its
trajectory surface is the tactical head's DIRECT per-horizon waypoints
(5/10/15/20 steps = the shared WP_STEPS, ego frame of the last window pose —
the refb_labels convention, i.e. the same frame as gt_ego_waypoints).

Method differs from the world-model arms (direct regression vs grounded
rollout) — recorded in the output; windows/GT/CV/strata/metrics identical, so
the leaderboard row is comparable with an explicit method flag."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import refb_labels as rl  # noqa: E402  (wrap_to_pi; scripts on sys.path)
from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)

K_MAX = max(WP_STEPS)
DT = 0.1                       # 10 Hz (matches rollout.py / refb_train.py)


@torch.no_grad()
def collect(model, episodes, device, window=8, stride=8, batch=8,
            speed_input=True, yaw_input=False):
    S_wp, GT, CV, EID, SPD, HDG = [], [], [], [], [], []
    for ep in episodes:
        fr = ep.feats                                  # raw frames [T,9,S,S]
        T = fr.shape[0]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + window])
                              for t in ch]).to(device).float().div_(255.0)
            v0 = ep.poses[last, 3].to(device) if speed_input else None
            # arch-v2 (B2): yr0 = BACKWARD-diff yaw-rate at t0, RAW rad/s —
            # exactly refb_train.compute_losses (pose_last vs pose_prev, /0.1,
            # no YAW_SCALE). Passed as kwarg ONLY for yaw_input arms so pre-v2
            # RefBModel forwards (no yr0 param) keep working.
            kw = {}
            if yaw_input:
                kw["yr0"] = (rl.wrap_to_pi(ep.poses[last, 2]
                                           - ep.poses[last - 1, 2])
                             / DT).to(device)
            out = model(fw, nav_cmd=None, v0=v0, **kw)  # follow-command eval
            wp = torch.stack([out["waypoints"][k] for k in WP_STEPS],
                             dim=1).cpu().float()      # [b, 4, 2]
            S_wp.append(wp)
            GT.append(gt_ego_waypoints(ep.poses, last))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
    return {"pred": torch.cat(S_wp), "gt": torch.cat(GT).float(),
            "cv": torch.cat(CV).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(),
            "wp_steps": list(WP_STEPS),
            "method": "refb tactical waypoint heads (direct, nav=follow)"}
