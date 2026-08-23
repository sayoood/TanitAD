"""TEMPORAL STABILITY — the two defects the four families cannot see.

⛔ WHY THIS EXISTS. Sayed, 2026-08-06, from watching flagship v1 drive:
  * *"its trajectory is jumping sometimes between the frames"*
  * *"the tactical manoeuvre is toggling the whole time"*

Neither is visible in ANY metric the programme publishes. Every one of the four binding
families scores a SINGLE window against its ground truth. A model can be perfect on all
four at every window and still emit a plan that disagrees violently with the plan it
emitted 0.8 s earlier — because nothing has ever compared two of its own outputs to each
other. That is a whole failure mode with no instrument.

⭐ THE HUMAN'S FLOOR IS EXACTLY ZERO, AND THAT IS WHAT MAKES THIS CLEAN. The ground truth
is ONE trajectory: the future from `t + dt` is literally a suffix of the future from `t`,
re-expressed. So GT replan disagreement is 0 by construction (up to the frame transform),
and any arm's value is pure self-inconsistency — no baseline to argue about, no estimator
choice to get wrong.

Three readings, kept separate because they have different fixes:
  1. ``replan_shift`` — do two consecutive plans agree where they overlap? (the jumping)
  2. ``replan_control_jump`` — the same question in CONTROL space, which is where a fix
     would act, and which is where a small position shift can hide a large accel change.
  3. ``maneuver_toggle`` — how often the declared manoeuvre changes, and how long it
     dwells. (the toggling)

⚠️ INTRA-plan jerk is reported too, because it is the same defect at a different
timescale and separating them decides the fix. MEASURED 2026-08-06 on 39 OOD-val clips:
the human's jerk RMS is 1.7975 m/s^3, Alpamayo's 1.7908 (0.92x), and flagship v1's
**64.2966 — 35.8x the human**. If the roughness is already inside one plan, smoothing
across plans treats a symptom.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np


def to_frame(path_a: np.ndarray, pose_a, pose_b) -> np.ndarray:
    """Re-express an ego-frame path from window A's origin into window B's frame.

    ``pose_*`` are ``(x, y, yaw)`` in the episode's world frame. Without this the two
    plans are in different frames and any comparison measures the ego's own motion
    rather than the model's disagreement with itself."""
    ca, sa = math.cos(pose_a[2]), math.sin(pose_a[2])
    wx = pose_a[0] + path_a[:, 0] * ca - path_a[:, 1] * sa
    wy = pose_a[1] + path_a[:, 0] * sa + path_a[:, 1] * ca
    dx, dy = wx - pose_b[0], wy - pose_b[1]
    cb, sb = math.cos(pose_b[2]), math.sin(pose_b[2])
    return np.stack([dx * cb + dy * sb, -dx * sb + dy * cb], axis=1)


def replan_pair(p_prev, p_cur, pose_prev, pose_cur, shift_steps):
    """Compare the OVERLAPPING portion of two consecutive plans, in the later frame.

    ``shift_steps`` is how many 0.1 s ticks separate the two window origins. Plan A's
    waypoint ``k + shift`` and plan B's waypoint ``k`` refer to the SAME absolute time —
    aligning on that is the whole comparison, and getting it wrong would measure the
    ego's travel instead."""
    n = p_cur.shape[0] - shift_steps
    if n <= 1:
        return None
    a = to_frame(p_prev[shift_steps:shift_steps + n], pose_prev, pose_cur)
    b = p_cur[:n]
    return a, b


def controls(path, dt=0.1):
    """(accel, curvature) implied by a path — the same convention as
    ``tanitad.models.kinematic.unicycle_controls_from_path``, reimplemented in numpy
    so this instrument does not import the training stack."""
    p = np.concatenate([np.zeros((1, 2)), path], 0)
    d = p[1:] - p[:-1]
    ds = np.linalg.norm(d, axis=-1)
    sp = ds / dt
    acc = np.concatenate([(sp[1:] - sp[:-1]) / dt, [(sp[-1] - sp[-2]) / dt]])
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = (h[1:] - h[:-1] + math.pi) % (2 * math.pi) - math.pi
    moving = ds[:-1] > 0.05
    cur = np.where(moving, dh / np.maximum(ds[:-1], 1e-8), 0.0)
    return acc, np.concatenate([cur, cur[-1:]])


def jerk_rms(path, dt=0.1):
    acc, _ = controls(path, dt)
    return float(np.sqrt((((acc[1:-1] - acc[:-2]) / dt) ** 2).mean()))


def summarise(name, shifts, cjumps, jerks, toggles=None, dwell=None):
    out = {"arm": name, "n_pairs": len(shifts)}
    if shifts:
        out["replan_shift_m_mean"] = round(float(np.mean(shifts)), 4)
        out["replan_shift_m_p90"] = round(float(np.percentile(shifts, 90)), 4)
        out["replan_shift_m_max"] = round(float(np.max(shifts)), 4)
    if cjumps:
        out["replan_accel_jump_mps2_mean"] = round(float(np.mean(cjumps)), 4)
        out["replan_accel_jump_mps2_p90"] = round(float(np.percentile(cjumps, 90)), 4)
    if jerks:
        out["intra_plan_jerk_rms_mps3"] = round(float(np.sqrt(np.mean(np.square(jerks)))), 4)
    if toggles is not None:
        out["maneuver_toggle_rate"] = round(float(np.mean(toggles)), 4)
        out["maneuver_mean_dwell_windows"] = round(float(np.mean(dwell)), 4) if dwell else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch", default="flagship-worldmodel-v2")
    a = ap.parse_args()

    import torch

    from taniteval import loaders
    from taniteval.cam_overlay import ego_future_path
    from taniteval.corpus_overlay import episode_rollouts
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import K

    files = sorted(glob.glob(os.path.join(a.corpus, "ep_*.pt")))[:a.episodes]
    L = loaders.load({"arch": a.arch, "ckpt": a.ckpt, "run_config": a.run_config,
                      "speed_input": True}, device=a.device)
    model, sr = L["model"].eval(), L["step_readout"]
    print(f"[temporal] step={L.get('step')} · {len(files)} episodes", flush=True)

    m_shift, m_cjump, m_jerk, m_tog, m_dwell = [], [], [], [], []
    g_shift, g_cjump, g_jerk = [], [], []
    n_win = 0
    for fi, f in enumerate(files):
        ep = load_frames([f])[0]
        poses = ep.poses.float()
        preds = episode_rollouts(model, sr, ep.feats, poses, ep.actions.float(),
                                 "frames", True, False, a.device)
        ws = sorted(preds)
        n_win += len(ws)
        mans = [preds[w].get("man") for w in ws]
        # --- manoeuvre toggling: change between CONSECUTIVE windows -------------
        run = 1
        for i in range(1, len(mans)):
            if mans[i] is None or mans[i - 1] is None:
                continue
            changed = int(mans[i] != mans[i - 1])
            m_tog.append(changed)
            if changed:
                m_dwell.append(run)
                run = 1
            else:
                run += 1
        if len(mans) > 1:
            m_dwell.append(run)
        # --- replan agreement, and the GT control which is 0 by construction ----
        for i in range(1, len(ws)):
            wp, wc = ws[i - 1], ws[i]
            shift = wc - wp
            if shift <= 0 or shift >= K:
                continue
            pp = preds[wp]["wp"].numpy()
            pc = preds[wc]["wp"].numpy()
            gp = ego_future_path(poses, wp, K).numpy()
            gc = ego_future_path(poses, wc, K).numpy()
            pose_p = (float(poses[wp, 0]), float(poses[wp, 1]), float(poses[wp, 2]))
            pose_c = (float(poses[wc, 0]), float(poses[wc, 1]), float(poses[wc, 2]))
            for tag, a_path, b_path, sh, cj, jk in (
                    ("model", pp, pc, m_shift, m_cjump, m_jerk),
                    ("gt", gp, gc, g_shift, g_cjump, g_jerk)):
                r = replan_pair(a_path, b_path, pose_p, pose_c, shift)
                if r is None:
                    continue
                A_, B_ = r
                sh.append(float(np.linalg.norm(A_ - B_, axis=-1).mean()))
                ca, _ = controls(A_)
                cb, _ = controls(B_)
                cj.append(float(np.abs(ca[:-1] - cb[:-1]).mean()))
                jk.append(jerk_rms(b_path))
        if (fi + 1) % 10 == 0:
            print(f"  [{fi+1}/{len(files)}] windows so far {n_win}", flush=True)

    res = {
        "_what": ("temporal self-consistency of the arm's own successive plans. NOT a "
                  "single-window accuracy metric — none of the four binding families "
                  "compares two of the model's outputs to each other."),
        "_gt_floor": ("the GT rows are the INSTRUMENT'S FLOOR, not a competing arm. The "
                      "human's future from t+dt is a suffix of its future from t, so a "
                      "perfect replanner scores ~0 shift. Any residual is the frame "
                      "transform and pose noise."),
        "_evidence_class": "MEASURED (ours)",
        "n_windows": n_win, "n_episodes": len(files), "ckpt_step": L.get("step"),
        "model": summarise("flagship", m_shift, m_cjump, m_jerk, m_tog, m_dwell),
        "gt_floor": summarise("ground truth (floor)", g_shift, g_cjump, g_jerk),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    M, G = res["model"], res["gt_floor"]
    print(f"\n{'':34s} {'FLAGSHIP':>12s} {'GT floor':>12s}")
    for lab, k in (("replan shift mean (m)", "replan_shift_m_mean"),
                   ("replan shift p90 (m)", "replan_shift_m_p90"),
                   ("replan shift max (m)", "replan_shift_m_max"),
                   ("replan accel jump mean (m/s^2)", "replan_accel_jump_mps2_mean"),
                   ("intra-plan jerk RMS (m/s^3)", "intra_plan_jerk_rms_mps3")):
        print(f"{lab:34s} {str(M.get(k)):>12s} {str(G.get(k)):>12s}")
    print(f"{'manoeuvre toggle rate':34s} {str(M.get('maneuver_toggle_rate')):>12s}")
    print(f"{'manoeuvre mean dwell (windows)':34s} "
          f"{str(M.get('maneuver_mean_dwell_windows')):>12s}")
    print(f"\n[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
