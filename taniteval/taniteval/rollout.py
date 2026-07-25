"""TanitEval — trajectory rollout engine.

One collect() for every arch: window -> encode -> operative-predictor rollout
under TRUE actions -> per-step Δpose via the arch's grounded step-readout ->
SE(2) accumulate. Ports the proven gate protocol (eval_refa4b_grounded /
eval_grounded_rollout_4b) verbatim so numbers are apples-to-apples with every
gate run to date. Model differences are fully described by (episode view,
encode_window, step_readout, speed_input) — all supplied by loaders.load().

THE DENSE PATH (added 2026-07-25 — the residual open since 2026-07-09)
---------------------------------------------------------------------
``rollout_decode`` computes the FULL ``[b, fwd_k, 2]`` 10 Hz path and this
module used to keep only the 4 waypoints at ``WP_STEPS`` (5/10/15/20), throwing
away 16 of 20 steps at what ``driving.py`` called "rollout.py:94". That single
discard blocked the entire comfort / behavioural axis — jerk, the curvature
*profile*, decel-onset lead time (the LAL-v2 metric in
``stack/tanitad/eval/metrics.py``), plan stability — because every one of them
needs 10 Hz derivatives, not 4 samples 0.5 s apart.

``collect`` now ALSO returns, **additively** (the sparse keys keep their exact
meaning, so every existing consumer is untouched):

  ``pred_dense`` [N, fwd_k, 2]  the model's full 10 Hz ego-frame path
  ``gt_dense``   [N, fwd_k, 2]  the matching ground-truth path
  ``dense_steps`` list(1..fwd_k) · ``dt_s`` 0.1 — the sampling contract

``pred``/``gt`` remain the 4-waypoint sparse view: ``pred == pred_dense[:,
[4,9,14,19]]`` by construction. A **CV dense floor is deliberately NOT stored**
— constant velocity is exactly linear in the step index, so
``cv_dense[:, k-1] == cv[:, 0] * k / 5`` reconstructs it for free; likewise
hold-v0 from ``speed``. Only ``gt_dense`` is irrecoverable (the val poses are
not persisted), so only it is paid for.

Cost, MEASURED 2026-07-25 on the committed 881-window / 40-episode
flagship-30k dump (re-saved both ways through ``save_windows``): two
``[881, 20, 2]`` float32 tensors, so ``results/windows_<arm>.pt`` grows
**95 950 B -> 378 359 B = +282 409 B (+275.8 KiB, 3.94x)** — 0.378 MB, inside
the ~1 MB/arm budget the behavioural axis was costed at. No extra compute: the
dense tensor was already being produced and thrown away.

**Optional, not guaranteed.** ``refb_eval.collect`` and ``refc_eval.collect``
build their own window dicts for direct-trajectory arms and do not emit the
dense keys; anything downstream must treat them as ``win.get("pred_dense")``.
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

    Returns dict of tensors: pred/gt/cv [N, 4, 2] + eid/speed/head_deg [N],
    PLUS the dense 10 Hz path pred_dense/gt_dense [N, fwd_k, 2] (+ dense_steps,
    dt_s) — see the module docstring for why the dense keys exist and what is
    deliberately not stored. speed/yaw/dyn_input append the canonical ego
    action-channels (v0, yr0) so the fed action matches the checkpoint's
    action_dim (dyn-in arm = 4)."""
    S_wp, GT, CV, EID, SPD, HDG = [], [], [], [], [], []
    S_dense, GT_dense = [], []
    dense_steps = tuple(range(1, fwd_k + 1))     # every 0.1 s tick, 1..fwd_k
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
            # DENSE PATH: keep all fwd_k steps, not just the 4 at WP_STEPS. The
            # tensor is already computed above — this is a persistence fix, not
            # extra compute (no second rollout, no extra GPU work).
            S_dense.append(wp_full.cpu().float())
            GT.append(gt_ego_waypoints(ep.poses, last))
            GT_dense.append(gt_ego_waypoints(ep.poses, last,
                                             wp_steps=dense_steps))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
    return {"pred": torch.cat(S_wp), "gt": torch.cat(GT).float(),
            "cv": torch.cat(CV).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(),
            "wp_steps": list(WP_STEPS),
            # --- dense 10 Hz path (tier-1 behavioural metrics) -------------- #
            "pred_dense": torch.cat(S_dense).float(),
            "gt_dense": torch.cat(GT_dense).float(),
            "dense_steps": list(dense_steps), "dt_s": DT}


def dense_speed_profile(path_dense, dt: float = DT):
    """Dense ego-frame path ``[N, K, 2]`` -> per-step speed ``[N, K]`` (m/s).

    THE convention, defined once. The dense path starts at step 1 *relative to
    the ego pose at the window's last frame*, which is the origin (0, 0) — so
    the first step's displacement is ``p[:, 0]`` itself, not ``p[:,1]-p[:,0]``.
    Prepending the origin is what makes ``speed[:, 0]`` the realised speed over
    the first tick rather than silently dropping it; getting this wrong shifts
    every derived jerk / decel-onset index by one sample.

    This is the input ``tanitad.eval.metrics.decel_onset_index`` /
    ``compute_lal_v2`` want (their ``ego_v``), and the base for jerk
    (``diff(speed)/dt`` twice) and the curvature profile.
    """
    p = path_dense if isinstance(path_dense, torch.Tensor) \
        else torch.as_tensor(path_dense)
    zero = torch.zeros(p.shape[0], 1, 2, dtype=p.dtype, device=p.device)
    d = torch.cat([zero, p], dim=1).diff(dim=1)        # [N, K, 2] per-tick Δp
    return torch.linalg.norm(d, dim=-1) / dt


def save_windows(data, path):
    """Persist a collect() window dict (dense keys included when present)."""
    torch.save({k: v for k, v in data.items()}, path)


def load_windows(path):
    """Load a window dump. Dumps written before 2026-07-25, and those from
    ``refb_eval`` / ``refc_eval``, carry NO dense keys — read them with
    ``win.get("pred_dense")`` and degrade, never assume."""
    return torch.load(path, map_location="cpu", weights_only=False)
