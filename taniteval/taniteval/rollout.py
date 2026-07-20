"""TanitEval — trajectory rollout engine.

One collect() for every arch: window -> encode -> operative-predictor rollout
under TRUE actions -> per-step Δpose via the arch's grounded step-readout ->
SE(2) accumulate. Ports the proven gate protocol (eval_refa4b_grounded /
eval_grounded_rollout_4b) verbatim so numbers are apples-to-apples with every
gate run to date. Model differences are fully described by (episode view,
encode_window, step_readout, speed_input) — all supplied by loaders.load().
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import refb_labels as rl  # noqa: E402  (wrap_to_pi; scripts on sys.path)
from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402

K_MAX = max(WP_STEPS)          # 20 steps = 2 s @ 10 Hz
SPEED_SCALE = 10.0             # matches every trainer
DT = 0.1                       # 10 Hz
YAW_SCALE = 1.0                # yaw-rate normalizer (refa_train_plus)


def ego_action_channels(poses, last, speed_input, yaw_input, dyn_input, device):
    """Canonical [v0(,yr0)] ego action-channels for speed/dyn-input arms —
    matches refa_train_plus._append_ego / hierarchy._ego_channels EXACTLY.
    Order [v0, yr0]:
      v0  = pose_last.v / SPEED_SCALE                         (--speed-input)
      yr0 = wrap(yaw_last - yaw_{last-1}) / DT / YAW_SCALE    (--yaw/--dyn-input,
            OBSERVED-only, leakage-safe). Returns [b, n_ego] or None."""
    feed_yaw = yaw_input or dyn_input
    if not (speed_input or feed_yaw):
        return None
    chans = []
    if speed_input:
        chans.append(poses[last, 3:4].float() / SPEED_SCALE)
    if feed_yaw:
        yr0 = (rl.wrap_to_pi(poses[last, 2] - poses[last - 1, 2]) / DT
               / YAW_SCALE).reshape(-1, 1)
        chans.append(yr0)
    return torch.cat(chans, dim=-1).to(device)


def append_ego(aw, fa, poses, last, speed_input, yaw_input, dyn_input, device):
    """Broadcast the ego channels across the action window/future and concat.
    No-op (returns aw, fa unchanged) for base action_dim=2 arms."""
    ego = ego_action_channels(poses, last, speed_input, yaw_input, dyn_input,
                              device)
    if ego is None:
        return aw, fa
    aw = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)], dim=-1)
    fa = torch.cat([fa, ego[:, None].expand(-1, fa.shape[1], -1)], dim=-1)
    return aw, fa


@torch.no_grad()
def collect(model, step_readout, episodes, device, window=8, fwd_k=K_MAX,
            stride=8, batch=8, speed_input=False, yaw_input=False,
            dyn_input=False):
    """Predict WP_STEPS waypoints for every window of every episode.

    Returns dict of tensors: pred/gt/cv [N, 4, 2] + eid/speed/head_deg [N].
    speed/yaw/dyn_input append the canonical ego action-channels (v0, yr0) so
    the fed action matches the checkpoint's action_dim (dyn-in arm = 4)."""
    S_wp, GT, CV, EID, SPD, HDG = [], [], [], [], [], []
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + window])
                              for t in ch]).to(device)
            if fw.dtype == torch.uint8:                      # raw frames path
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:                  # frozen features
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + fwd_k]
                              for t in ch]).to(device)
            aw, fa = append_ego(aw, fa, ep.poses, last, speed_input,
                                yaw_input, dyn_input, device)
            states = model.encode_window(fw)                       # [b, W, S]
            wp_full, _ = rollout_decode(model.predictor, states, aw, fa,
                                        step_readout, fwd_k)       # [b, k, 2]
            S_wp.append(wp_full.index_select(1, wp_idx.to(device)).cpu().float())
            GT.append(gt_ego_waypoints(ep.poses, last))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
    return {"pred": torch.cat(S_wp), "gt": torch.cat(GT).float(),
            "cv": torch.cat(CV).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(),
            "wp_steps": list(WP_STEPS)}


def save_windows(data, path):
    torch.save({k: v for k, v in data.items()}, path)


def load_windows(path):
    return torch.load(path, map_location="cpu", weights_only=False)
