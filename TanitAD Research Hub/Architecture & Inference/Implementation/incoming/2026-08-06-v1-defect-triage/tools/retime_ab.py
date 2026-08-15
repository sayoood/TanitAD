"""RETIME A/B — does the retrain-free projection actually improve flagship v1?

⛔ THE QUESTION, asked because everything before this MEASURED v1 without CHANGING it.
`retime_path` keeps the curve the frozen head drew and re-times the schedule along it
under the ego's TRUE v0 and bounded accel/jerk. On 39 single windows that moved ADE
0.3303 -> 0.2675 and jerk 64.30 -> 4.53. This runs the same A/B over ALL 6,834 windows,
and adds the two things 39 clips could not answer: whether the LATERAL channel survives
at scale, and whether re-timing each frame INDEPENDENTLY also reduces the frame-to-frame
control jump (1.1021 m/s^2 measured) -- which it might not, since nothing couples the
frames.

⭐ THE LIMITS ARE A RULE, NOT A TUNING. accel/jerk bounds are the HUMAN's own p99 on
this corpus, computed in this same pass. Hand-picking them on the eval set would be
selection on the test set; MEASURED sensitivity over a 6.5x range of the accel limit
moves ADE only 0.2675-0.2916, all better than the 0.3303 baseline.

⛔ v0 IS NOT PRIVILEGED INFORMATION HERE. The arm is already trained and evaluated with
`speed_input=True`, i.e. the ego speed at the window origin is an EXISTING input channel.
Using it in post-processing adds nothing the model was not already given. That is the
leak test from CLAUDE.md applied to a fix rather than to a head, and it passes.

ORIGINAL DOCSTRING (temporal stability instrument):
TEMPORAL STABILITY — the two defects the four families cannot see.

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



def four_family_row(P, G, v0):
    """LONGITUDINAL + LATERAL scalars for one batch of paths, numpy-pure."""
    out = {}
    ca, cg = np.stack([controls(p)[0] for p in P]), np.stack([controls(g)[0] for g in G])
    ka, kg = np.stack([controls(p)[1] for p in P]), np.stack([controls(g)[1] for g in G])
    sp = lambda A: np.linalg.norm(np.diff(np.concatenate(
        [np.zeros((A.shape[0], 1, 2)), A], 1), axis=1), axis=-1) / 0.1
    ja = (ca[:, 1:-1] - ca[:, :-2]) / 0.1
    out["ade_2s_m"] = float(np.linalg.norm(P - G, axis=-1).mean())
    out["speed_bias_mps"] = float((sp(P) - sp(G)).mean())
    out["speed_mae_mps"] = float(np.abs(sp(P) - sp(G)).mean())
    out["along_final_bias_m"] = float((P[:, -1, 0] - G[:, -1, 0]).mean())
    out["accel_rms_mps2"] = float(np.sqrt((ca ** 2).mean()))
    out["accel_bias_mps2"] = float((ca - cg).mean())
    out["jerk_rms_mps3"] = float(np.sqrt((ja ** 2).mean()))
    out["curvature_mae_1pm"] = float(np.abs(ka - kg).mean())
    out["entry_transient_mps2"] = float(np.abs(
        np.linalg.norm(P[:, 0], axis=-1) / 0.1 - v0).mean() / 0.1)
    return {k: round(v, 6) for k, v in out.items()}


def main():
    import argparse, glob, json, os, sys
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--run-config", required=True)
    ap.add_argument("--corpus", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40); ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch", default="flagship-worldmodel-v2")
    a = ap.parse_args()

    sys.path.insert(0, "/workspace/TanitAD/stack")
    from tanitad.models.kinematic import retime_path
    from taniteval import loaders
    from taniteval.cam_overlay import ego_future_path
    from taniteval.corpus_overlay import episode_rollouts
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import K

    files = sorted(glob.glob(os.path.join(a.corpus, "ep_*.pt")))[:a.episodes]
    L = loaders.load({"arch": a.arch, "ckpt": a.ckpt, "run_config": a.run_config,
                      "speed_input": True}, device=a.device)
    model, sr = L["model"].eval(), L["step_readout"]
    print(f"[retime-ab] step={L.get('step')} · {len(files)} episodes", flush=True)

    # ---- PASS 1: collect every window's prediction, GT and v0 ------------------
    EP = []
    for fi, f in enumerate(files):
        ep = load_frames([f])[0]
        poses = ep.poses.float()
        preds = episode_rollouts(model, sr, ep.feats, poses, ep.actions.float(),
                                 "frames", True, False, a.device)
        ws = sorted(preds)
        EP.append({
            "ws": ws,
            "P": np.stack([preds[w]["wp"].numpy() for w in ws]),
            "G": np.stack([ego_future_path(poses, w, K).numpy() for w in ws]),
            "v0": np.array([float(poses[w, 3]) for w in ws]),
            "pose": np.array([[float(poses[w, 0]), float(poses[w, 1]),
                               float(poses[w, 2])] for w in ws]),
        })
        if (fi + 1) % 10 == 0:
            print(f"  [{fi+1}/{len(files)}]", flush=True)

    Pall = np.concatenate([e["P"] for e in EP]); Gall = np.concatenate([e["G"] for e in EP])
    v0all = np.concatenate([e["v0"] for e in EP])

    # ---- the LIMITS ARE A RULE: the human's own p99 on this corpus -------------
    cg = np.stack([controls(g)[0] for g in Gall])
    jg = (cg[:, 1:-1] - cg[:, :-2]) / 0.1
    A_LIM = float(np.percentile(np.abs(cg), 99))
    J_LIM = float(np.percentile(np.abs(jg), 99))
    print(f"[limits] human p99 accel {A_LIM:.4f} m/s^2 · jerk {J_LIM:.4f} m/s^3", flush=True)

    def retime_ep(e):
        return retime_path(torch.tensor(e["P"], dtype=torch.float64),
                           torch.tensor(e["v0"], dtype=torch.float64),
                           accel_limit=A_LIM, jerk_limit=J_LIM).numpy()

    Rall = np.concatenate([retime_ep(e) for e in EP])

    res = {"_evidence_class": "MEASURED (ours)", "ckpt_step": L.get("step"),
           "n_windows": int(Pall.shape[0]), "n_episodes": len(files),
           "limits": {"accel_mps2": round(A_LIM, 4), "jerk_mps3": round(J_LIM, 4),
                      "_rule": "the HUMAN's own p99 on this corpus, computed in this "
                               "pass -- a rule, not a value tuned on the metric"},
           "_v0_is_not_privileged": ("the arm trains and evaluates with speed_input=True, "
                                     "so v0 is an EXISTING input channel; using it in "
                                     "post-processing adds no information at inference"),
           "_what_this_is_not": ("a projection applied AFTER a frozen head, NOT a fix to "
                                 "the model. The model still plans infeasibly; this "
                                 "corrects its output and gives the retrain a baseline."),
           "before": four_family_row(Pall, Gall, v0all),
           "after": four_family_row(Rall, Gall, v0all),
           "human": four_family_row(Gall, Gall, v0all)}

    # ⛔ BANK PER-EPISODE MEANS, or no interval is ever computable from this artifact.
    # Windows inside an episode are strongly dependent -- an i.i.d. CI over 6,834 of them
    # would be badly optimistic -- so the decision-grade estimator is the EPISODE-CLUSTER
    # bootstrap over the 40 episodes (taniteval/ci.py). Pooling to a single scalar throws
    # away exactly the structure that estimator needs, and the loss is irreversible
    # without another GPU pass. Banking the per-episode PAIRED deltas costs nothing and
    # makes the interval a post-hoc computation.
    off, per_ep = 0, []
    for e in EP:
        n = e["P"].shape[0]
        b = four_family_row(Pall[off:off + n], Gall[off:off + n], v0all[off:off + n])
        a_ = four_family_row(Rall[off:off + n], Gall[off:off + n], v0all[off:off + n])
        per_ep.append({"n_windows": int(n),
                       "before": b, "after": a_,
                       "delta": {k: round(a_[k] - b[k], 6) for k in b}})
        off += n
    res["per_episode"] = per_ep
    res["_estimator_note"] = (
        "headline rows are unweighted means over ALL windows and carry NO interval. "
        "Windows within an episode are strongly dependent, so an i.i.d. CI would be "
        "badly optimistic. `per_episode[].delta` is the PAIRED per-cluster delta the "
        "episode-cluster bootstrap (taniteval/ci.py) consumes -- compute the interval "
        "from it rather than re-running the model.")

    # ---- temporal stability, BEFORE and AFTER ---------------------------------
    for tag, getter in (("before", lambda e: e["P"]), ("after", retime_ep)):
        sh, cj, jk = [], [], []
        for e in EP:
            X, ws, pose = getter(e), e["ws"], e["pose"]
            for i in range(1, len(ws)):
                shift = ws[i] - ws[i - 1]
                if shift <= 0 or shift >= K:
                    continue
                r = replan_pair(X[i - 1], X[i], pose[i - 1], pose[i], shift)
                if r is None:
                    continue
                A_, B_ = r
                sh.append(float(np.linalg.norm(A_ - B_, axis=-1).mean()))
                cj.append(float(np.abs(controls(A_)[0][:-1] - controls(B_)[0][:-1]).mean()))
                jk.append(jerk_rms(X[i]))
        res[f"temporal_{tag}"] = summarise(tag, sh, cj, jk)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    B, Aa, H = res["before"], res["after"], res["human"]
    print(f"\n{'':26s}{'BEFORE':>12s}{'AFTER':>12s}{'HUMAN':>12s}")
    for k in ("ade_2s_m", "speed_bias_mps", "speed_mae_mps", "along_final_bias_m",
              "accel_rms_mps2", "accel_bias_mps2", "jerk_rms_mps3",
              "curvature_mae_1pm", "entry_transient_mps2"):
        print(f"{k:26s}{B[k]:>12.4f}{Aa[k]:>12.4f}{H[k]:>12.4f}")
    print(f"\n{'':26s}{'BEFORE':>12s}{'AFTER':>12s}")
    for k in ("replan_shift_m_mean", "replan_accel_jump_mps2_mean", "intra_plan_jerk_rms_mps3"):
        print(f"{k:26s}{res['temporal_before'].get(k):>12}{res['temporal_after'].get(k):>12}")
    print(f"\n[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
