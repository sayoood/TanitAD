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
rollout).

⚠️ THE ROUTE INPUT — the 07-21 C6 confound, closed 2026-07-26
--------------------------------------------------------------
This module used to call ``model(fw, nav_cmd=None, ...)`` unconditionally.
``refc.py:785-786`` turns ``None`` into ``zeros(b)`` = the ``follow`` command,
so **REF-C's route input was never exercised in any comparison** — a decoder
that only ever saw `follow` learns the marginal over trajectories, and was then
compared against a hierarchy. ``RETRACTION_LOG`` 07-21 (C6) records that this
confound *"nearly designed the hierarchy away"*; ``V4_FLAGSHIP_DESIGN.md:806``
registers removing it as an unlanded precondition.

:func:`collect` now takes an explicit ``nav_mode`` and **stamps it into every
window dict** (``nav_provenance``) and into ``method``:

``"produced"`` (**default**)
    Two passes. Pass 1 reads the model's OWN ``route_head`` logits (they depend
    only on ``pooled``, i.e. on the image — ``nav`` enters solely through
    ``measurement``, so pass 1's route logits are identical whatever nav it was
    given). Pass 2 feeds ``argmax`` of those logits as ``nav_cmd``. This is the
    *produced goal* the v4 design demands: no future, no label, no oracle.
``"follow_constant"``
    The historical path, kept **only** as an explicitly-named control. Every
    published REF-C number (base 0.4728, XL 0.4714, small) was produced this
    way and is NOT comparable to a ``produced`` run.
``"oracle"``
    GT route from ``refb_labels`` — future-derived. Reported ONLY as an upper
    bound, never as a leaderboard number.
"""
from __future__ import annotations

import sys

import torch

# ⛔ was: sys.path.insert(0, "/root/TanitAD/stack"[/scripts]) — that put a
# possibly PRE-v5 tree IN FRONT of the caller's PYTHONPATH and published a
# plausible wrong number instead of an error (STALE_IMPORT_GUARD.md).
from taniteval.stack_guard import ensure_stack_on_path as _ensure_stack  # noqa: E402
_ensure_stack()

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)

K_MAX = max(WP_STEPS)
DT = 0.1                       # 10 Hz (matches rollout.py / refc_train.py)

NAV_MODES = ("produced", "follow_constant", "oracle")
#: REF-C route class -> nav command index. ``refc.NAV_COMMANDS`` is
#: ``("follow", "left", "right", "straight")``; ``refb.ROUTE_CLASSES`` is
#: ``("route_left", "route_straight", "route_right")``.
ROUTE_TO_NAV = {0: 1, 1: 0, 2: 2}                       # L->left S->follow R->right


def resolve_nav(model, fw, v0, steps, nav_mode, poses=None, last=None):
    """The route command actually fed to REF-C, plus its provenance.

    Returns ``(nav_cmd [b] long | None, note)``. ``"produced"`` costs one extra
    forward; the route logits are image-only (``route_head(pooled)``) so the
    extra pass is used solely to READ them, never to select a trajectory."""
    b = fw.shape[0]
    if nav_mode == "follow_constant":
        return None, "constant follow (nav_cmd=None -> zeros): THE C6 CONFOUND"
    if nav_mode == "oracle":
        assert poses is not None and last is not None, \
            "nav_mode='oracle' needs poses+last to mint the GT route"
        import refb_labels as rl
        cmds = [rl.nav_command(poses, int(t))[0] for t in last]
        return (torch.tensor(cmds, dtype=torch.long, device=fw.device),
                "GT route from the ego's OWN FUTURE poses — AN ORACLE, upper "
                "bound only, never a leaderboard number")
    if nav_mode == "produced":
        probe = model(fw, nav_cmd=None, v0=v0, steps=0)
        rl_logits = probe.get("route_logits")
        if rl_logits is None:                    # arch without a route head
            return None, ("no route_head on this checkpoint -> fell back to "
                          "constant follow; NOT a produced-goal number")
        cls = rl_logits.float().argmax(-1)                       # [b] in 0..2
        nav = torch.tensor([ROUTE_TO_NAV[int(c)] for c in cls.cpu()],
                           dtype=torch.long, device=fw.device)
        return nav, ("PRODUCED by the model's own route_head (image-only, no "
                     "future, no label)")
    raise ValueError(f"nav_mode must be one of {NAV_MODES}, got {nav_mode!r}")


@torch.no_grad()
def collect(model, episodes, device, window=None, stride=8, batch=8,
            speed_input=True, mode="diffusion", steps=None,
            nav_mode="produced"):
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
    assert nav_mode in NAV_MODES, (
        f"nav_mode must be one of {NAV_MODES}, got {nav_mode!r}")
    if nav_mode == "follow_constant":
        print("[refc] ⚠️  nav_mode='follow_constant' — the route input is NOT "
              "exercised (the 07-21 C6 confound). This number may not be "
              "compared against a route-conditional arm.", flush=True)
    nav_hist = torch.zeros(4, dtype=torch.long)
    nav_note = None

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
            nav_cmd, nav_note = resolve_nav(model, fw, v0, steps, nav_mode,
                                            poses=ep.poses, last=last)
            if nav_cmd is not None:
                nav_hist += torch.bincount(nav_cmd.cpu(), minlength=4)
            else:
                nav_hist[0] += len(ch)
            out = model(fw, nav_cmd=nav_cmd, v0=v0, steps=steps)
            wp = torch.stack([out["waypoints"][k] for k in WP_STEPS],
                             dim=1).cpu().float()      # [b, 4, 2]
            S_wp.append(wp)
            GT.append(gt_ego_waypoints(ep.poses, last))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
    n_win = int(nav_hist.sum())
    names = ("follow", "left", "right", "straight")
    hist = {n: int(c) for n, c in zip(names, nav_hist.tolist()) if c}
    return {"pred": torch.cat(S_wp), "gt": torch.cat(GT).float(),
            "cv": torch.cat(CV).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(),
            "wp_steps": list(WP_STEPS),
            # ROUTE PROVENANCE — stamped so no artifact can be read without it.
            "nav_provenance": {
                "nav_mode": nav_mode, "note": nav_note,
                "fed_command_hist": hist,
                "route_input_exercised": bool(nav_mode != "follow_constant"
                                              and len(hist) > 1),
                "is_oracle": bool(nav_mode == "oracle"),
                "n_windows": n_win,
                "_read": ("`route_input_exercised` False means the decoder saw "
                          "ONE constant command for every window, so it was "
                          "compared on its marginal — the 07-21 C6 confound. "
                          "Every REF-C number published before 2026-07-26 "
                          "(base 0.4728, XL 0.4714) was collected that way."),
            },
            "method": (f"refc anchored-diffusion decode (mode={mode}, "
                       f"steps={steps}, {model.cfg.anchors.n_anchors} anchors, "
                       f"argmax-conf anchor trajectory, nav_mode={nav_mode}"
                       f"{' [ORACLE]' if nav_mode == 'oracle' else ''})")}
