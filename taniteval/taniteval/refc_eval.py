"""TanitEval — REF-C anchored-diffusion trajectory eval.

REF-C (Anchored-Diffusion-C) decodes trajectories with its OWN DiffusionDrive-
style anchored decoder, NOT a grounded operative step-readout: a fixed anchor
vocabulary cross-attends the conv feature map -> per-anchor confidence + offset;
optional truncated-diffusion steps refine the winning modes; the trajectory is
the argmax-confidence anchor trajectory (deterministic at eval — model.eval()
zeroes the denoise noise, so the decode is reproducible).

Trajectory surface = the selected anchor trajectory read at the shared WP_STEPS
(5/10/15/20 steps = 0.5/1/1.5/2 s, ego frame of the LAST window pose). REF-C is
trained on refb_labels.waypoint_targets, whose frame is IDENTICAL to
gt_ego_waypoints (the d1_probe `_ego` convention), so the row is directly
comparable to every other arm — same windows, same GT, same CV, same strata,
same metric — with only the decode MECHANISM differing (recorded in `method`).

Only refc1=False checkpoints are time-waypoint comparable: refc1 reads the same
step slots as fixed-DISTANCE path checkpoints (2/5/10/20 m), which are NOT the
time waypoints gt_ego_waypoints scores — such a ckpt is refused here.

v0 = pose_last.v is fed to the model (measurement encoder, /10 scaling applied
internally); it is NOT an action channel (REF-C has no action-conditioned
rollout). nav_cmd=None -> the `follow` command, matching the other arms' eval.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)

K_MAX = max(WP_STEPS)
DT = 0.1                       # 10 Hz (matches rollout.py / refc_train.py)


@torch.no_grad()
def collect(model, episodes, device, window=None, stride=8, batch=8,
            speed_input=True, mode="diffusion", steps=None):
    """Predict WP_STEPS waypoints for every window of every episode via REF-C's
    anchored-diffusion decoder. Returns the SAME dict shape as rollout.collect /
    refb_eval.collect (pred/gt/cv/eid/speed/head_deg/wp_steps) so bench.run()
    consumes it unchanged.

    ``mode`` picks the decoder inference mode ("diffusion" == the trained
    truncated-denoise refinement over cfg.decoder.diffusion_steps; "classifier"
    == the 0-step anchor-selection floor). ``steps`` overrides the resolved step
    count when given.
    """
    assert not getattr(model.cfg, "refc1", False), (
        "REF-C.1 ckpt: horizons are fixed-DISTANCE path checkpoints (2/5/10/20 "
        "m), not time waypoints — not comparable to gt_ego_waypoints. Eval it "
        "with a path/speed metric, not this time-ADE path.")
    horizons = tuple(model.cfg.trajectory.horizons)
    assert horizons == tuple(WP_STEPS), (
        f"REF-C horizons {horizons} != eval WP_STEPS {tuple(WP_STEPS)}; the "
        "anchor trajectory must be read at the shared 5/10/15/20-step slots")
    if steps is None:
        steps = model.cfg.decoder.diffusion_steps if mode == "diffusion" else 0
    if window is None:
        window = int(model.cfg.window)             # trained state window (=8)

    S_wp, GT, CV, EID, SPD, HDG = [], [], [], [], [], []
    for ep in episodes:
        fr = ep.feats                                  # raw frames [T,9,S,S] u8
        T = fr.shape[0]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + window])
                              for t in ch]).to(device).float().div_(255.0)
            v0 = ep.poses[last, 3].to(device) if speed_input else None
            out = model(fw, nav_cmd=None, v0=v0, steps=steps)   # follow-command
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
            "method": (f"refc anchored-diffusion decode (mode={mode}, "
                       f"steps={steps}, {model.cfg.anchors.n_anchors} anchors, "
                       f"argmax-conf anchor trajectory, nav=follow)")}
