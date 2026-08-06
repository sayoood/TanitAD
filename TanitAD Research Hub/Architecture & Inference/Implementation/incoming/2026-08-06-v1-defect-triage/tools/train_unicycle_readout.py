"""Option 2 trainer — the unicycle trajectory decoder on the FROZEN v1arch trunk.

⛔ THE CONTRACT (Sayed, 2026-08-06): *"assure that the WM and its representation and
prediction are used instead of predicting future ego state based on current one."*
Three mechanisms enforce it, none of them optional in this script:

  1. **The readout's only trajectory-shaped inputs are the WM's rolled latent
     transitions.** Frames never reach it; poses reach it only as v0 (an existing
     model input) plus its own integrated feedback, both behind `shortcut_dropout`
     and `detach_feedback`.
  2. **The reliance CANARY runs during training** (every `--reliance-every` steps):
     the real-vs-MEAN latent ablation from `wm_reliance.py`, on a fixed val batch.
     A head drifting toward the v0/feedback shortcut shows up as a falling
     `wm_reliance` DURING the run, not in the post-mortem.
  3. **The final gate is pre-registered**: `wm_reliance_gate(min_reliance=0.5)`.
     A FAIL is written into the summary — the run cannot quietly ship a bypassed head.

⛔ FROZEN MEANS PROVED FROZEN. Encoder + predictor are `requires_grad_(False)` AND the
optimiser is built over the readout's parameters only AND the trunk's weights are
checksummed before/after training — three independent locks, because "I called eval()"
is not evidence (the supervise-manifest trap: verify the RUNNING state, not the config).

⚠️ Trains on the v2bal TRAIN cache; validates on the OOD-val q90 corpus the four-family
numbers were measured on, so before/after is apples-to-apples with every banked number.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch


def trunk_md5(model) -> str:
    h = hashlib.md5()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def kin_metrics(wp: torch.Tensor, gt: torch.Tensor, dt: float = 0.1) -> dict:
    """The report-card metrics on the DENSE 0.1 s grid — matching the banked
    instruments, so 'before' is the published v1arch row, not a re-derivation."""
    def geom(X):
        p = torch.cat([torch.zeros_like(X[:, :1]), X], 1)
        d = p[:, 1:] - p[:, :-1]
        ds = torch.linalg.norm(d, dim=-1)
        sp = ds / dt
        acc = (sp[:, 1:] - sp[:, :-1]) / dt
        h = torch.atan2(d[..., 1], d[..., 0])
        dh = (h[:, 1:] - h[:, :-1] + math.pi) % (2 * math.pi) - math.pi
        ok = (ds[:, 1:] > 0.05) & (ds[:, :-1] > 0.05)
        return sp, acc, (dh * ok).sum(1)
    sp_p, ac_p, ny_p = geom(wp)
    sp_g, ac_g, ny_g = geom(gt)
    jerk = (ac_p[:, 1:] - ac_p[:, :-1]) / dt
    return {
        "ade_m": round(float(torch.linalg.norm(wp - gt, dim=-1).mean()), 4),
        "speed_bias_mps": round(float((sp_p - sp_g).mean()), 4),
        "accel_rms_mps2": round(float(ac_p.pow(2).mean().sqrt()), 4),
        "jerk_rms_mps3": round(float(jerk.pow(2).mean().sqrt()), 4),
        "net_yaw_err_rad": round(float((ny_p - ny_g).abs().mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-episodes", type=int, default=1500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--reliance-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    # loss weights: position stays primary; net_yaw is THE heading target
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-heading", type=float, default=0.3)
    ap.add_argument("--w-net-yaw", type=float, default=0.5)
    ap.add_argument("--w-accel", type=float, default=0.05)
    ap.add_argument("--w-jerk", type=float, default=0.05)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = a.device

    from taniteval import loaders
    from taniteval.data import load_frames
    from taniteval.rollout import SPEED_SCALE
    from tanitad.data.v2_dataset import build_v2_providers
    from tanitad.models.kinematic import kinematic_losses
    from tanitad.models.metric_dynamics import (UnicycleStepReadout,
                                                accumulate_se2,
                                                gt_ego_waypoints,
                                                rollout_transitions)
    from tanitad.models.wm_reliance import wm_reliance, wm_reliance_gate

    # ---- frozen trunk, three locks --------------------------------------------
    L = loaders.load({"arch": "flagship-worldmodel-v2", "ckpt": a.ckpt,
                      "run_config": a.run_config, "speed_input": True}, device=dev)
    model, sr_old = L["model"].eval(), L["step_readout"]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sr_old.parameters():
        p.requires_grad_(False)
    md5_before = trunk_md5(model)
    print(f"[trainer] trunk frozen · step {L.get('step')} · md5 {md5_before[:12]}",
          flush=True)

    state_dim = sr_old.net[1].weight.shape[1] // 2
    head = UnicycleStepReadout.warm_start_from(sr_old, state_dim, hidden=512).to(dev)
    n_par = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    print(f"[trainer] UnicycleStepReadout {n_par/1e6:.2f} M trainable "
          f"(warm-started trunk, zero output)", flush=True)

    # ---- data ------------------------------------------------------------------
    eps = build_v2_providers(a.train_cache, lru_size=96, verbose=True)
    rng = random.Random(a.seed)
    rng.shuffle(eps)
    eps = eps[:a.train_episodes]
    W = 8

    def sample_batch(b):
        fw, aw_l, fa_l, pl, fp = [], [], [], [], []
        while len(fw) < b:
            ep = eps[rng.randrange(len(eps))]
            T = ep.poses.shape[0]
            if T < W + a.k + 1:
                continue
            s = rng.randrange(0, T - W - a.k)
            last = s + W - 1
            fw.append(torch.as_tensor(ep.frames[s:s + W]))
            aw_l.append(ep.actions[s:s + W])
            fa_l.append(ep.actions[s + W:s + W + a.k])
            pl.append(ep.poses[last])
            fp.append(ep.poses[last + 1:last + 1 + a.k])
        frames = torch.stack(fw).to(dev).float().div_(255.0)
        aw = torch.stack(aw_l).to(dev).float()
        fa = torch.stack(fa_l).to(dev).float()
        pose_last = torch.stack(pl).to(dev).float()
        fut_poses = torch.stack(fp).to(dev).float()
        # ⛔ the SAME [v0/SPEED_SCALE] channel append_ego builds, constructed directly:
        # append_ego indexes an EPISODE's [T,4] poses with time indices, which a
        # per-batch pose stack would silently mis-index (advanced indexing on dim 0).
        ego = (pose_last[:, 3:4] / SPEED_SCALE)
        aw = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)], -1)
        fa = torch.cat([fa, ego[:, None].expand(-1, fa.shape[1], -1)], -1)
        return frames, aw, fa, pose_last, fut_poses

    # ---- fixed VAL batch, same corpus + grid as every banked number ------------
    from taniteval.lead_source import window_last_indices
    val_files = sorted(glob.glob(os.path.join(a.val_corpus, "ep_*.pt")))[:40]
    val_eps = load_frames(val_files)
    vf, vaw, vfa, vpl, vfp = [], [], [], [], []
    vrng = random.Random(123)
    while len(vf) < 128:
        ep = val_eps[vrng.randrange(len(val_eps))]
        poses = ep.poses.float()
        cand = window_last_indices(poses.shape[0])
        last = int(cand[vrng.randrange(len(cand))])
        s = last - W + 1
        if s < 0 or last + a.k >= poses.shape[0]:
            continue
        vf.append(torch.as_tensor(ep.feats[s:s + W]))
        vaw.append(ep.actions[s:s + W].float())
        vfa.append(ep.actions[s + W:s + W + a.k].float())
        vpl.append(poses[last])
        vfp.append(poses[last + 1:last + 1 + a.k])
    # ⛔ 128 windows of [8,9,256,256] float frames is ~10 GB — encode in CHUNKS and
    # keep only the [128, W, S] states; the frames never live on the GPU all at once.
    v_frames_u8 = torch.stack(vf)
    v_aw = torch.stack(vaw).to(dev)
    v_fa = torch.stack(vfa).to(dev)
    v_pl = torch.stack(vpl).to(dev)
    v_fp = torch.stack(vfp).to(dev)
    v_ego = (v_pl[:, 3:4] / SPEED_SCALE)
    v_aw = torch.cat([v_aw, v_ego[:, None].expand(-1, v_aw.shape[1], -1)], -1)
    v_fa = torch.cat([v_fa, v_ego[:, None].expand(-1, v_fa.shape[1], -1)], -1)
    v_gt = gt_ego_waypoints(v_pl, v_fp, list(range(1, a.k + 1)))
    v_v0 = v_pl[:, 3]
    with torch.no_grad():
        chunks = []
        for i0 in range(0, v_frames_u8.shape[0], 16):
            fb = v_frames_u8[i0:i0 + 16].to(dev).float().div_(255.0)
            chunks.append(model.encode_window(fb))
            del fb
        v_states = torch.cat(chunks, 0)
        del chunks, v_frames_u8
        v_trans = rollout_transitions(model.predictor, v_states, v_aw, v_fa, a.k)

    def decode(trans, v0, training):
        """Unicycle decode over rolled transitions, carrying (v, a_prev, yr_prev)."""
        head.train(training)
        b = trans[0][0].shape[0]
        v = v0.clone()
        a_prev = torch.zeros_like(v)
        yr_prev = torch.zeros_like(v)
        rows = []
        for z_prev, z_hat in trans:
            aj, yrj = head(z_prev, z_hat, v, a_prev, yr_prev)
            rows.append(torch.stack([v * 0.1, torch.zeros_like(v), yrj * 0.1], -1))
            v = (v + aj * 0.1).clamp_min(0.0)
            a_prev, yr_prev = aj, yrj
        return accumulate_se2(torch.stack(rows, 1))

    def reliance_now():
        """The canary: real-vs-mean over FULL re-rolls, so both the representation
        AND the prediction pathways are exercised by the ablation."""
        def rollout_fn(states, aw, fa, v0):
            with torch.no_grad():
                trans = rollout_transitions(model.predictor, states, aw, fa, a.k)
                return decode(trans, v0, training=False)
        return wm_reliance(rollout_fn, v_states, v_aw, v_fa, v_gt, v_v0, k=a.k)

    # ---- baseline row: the OLD displacement readout on the same val batch ------
    from tanitad.models.metric_dynamics import decode_transitions
    with torch.no_grad():
        wp_old, _ = decode_transitions(sr_old, v_trans, a.k)
        base = kin_metrics(wp_old.float(), v_gt)
    rel0 = reliance_now()
    print(f"[baseline] displacement readout: {base}", flush=True)
    print(f"[reliance @init] {rel0['verdict']}", flush=True)

    log_path = os.path.join(a.out_dir, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({"baseline_displacement_readout": base,
                         "reliance_init": rel0["verdict"],
                         "args": vars(a), "trunk_md5": md5_before,
                         "n_trainable": n_par}) + "\n")
    fh.flush()

    t0 = time.time()
    for step in range(1, a.steps + 1):
        frames, aw, fa, pose_last, fut_poses = sample_batch(a.batch)
        with torch.no_grad():
            states = model.encode_window(frames)
            trans = rollout_transitions(model.predictor, states, aw, fa, a.k)
        wp = decode(trans, pose_last[:, 3], training=True)
        gt = gt_ego_waypoints(pose_last, fut_poses, list(range(1, a.k + 1)))
        pos_l1 = (wp - gt).abs().mean()
        kin = kinematic_losses(wp, gt, dt=0.1)
        loss = (a.w_pos * pos_l1 + a.w_heading * kin["heading"]
                + a.w_net_yaw * kin["net_yaw"] + a.w_accel * kin["accel"]
                + a.w_jerk * kin["jerk"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(float(loss), 5),
                   "pos_l1": round(float(pos_l1), 5),
                   "kin_heading": round(float(kin["heading"]), 5),
                   "kin_net_yaw": round(float(kin["net_yaw"]), 5),
                   "kin_accel": round(float(kin["accel"]), 5),
                   "kin_jerk": round(float(kin["jerk"]), 5),
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.val_every == 0:
            with torch.no_grad():
                wp_v = decode(v_trans, v_v0, training=False)
                vm = kin_metrics(wp_v.float(), v_gt)
            fh.write(json.dumps({"step": step, "val": vm, "baseline": base}) + "\n")
            fh.flush()
            print(f"[val @{step}] {vm}  (baseline {base})", flush=True)

        if step % a.reliance_every == 0:
            rel = reliance_now()
            fh.write(json.dumps({"step": step,
                                 "reliance": rel["verdict"],
                                 "reliance_arms": {k: rel[k] for k in
                                                   ("real", "mean", "shuffled",
                                                    "frozen", "cv")}}) + "\n")
            fh.flush()
            print(f"[reliance @{step}] wm_reliance="
                  f"{rel['verdict'].get('wm_reliance')} "
                  f"(real {rel['real']['ade_m']} mean {rel['mean']['ade_m']} "
                  f"cv {rel['cv']['ade_m']})", flush=True)

    # ---- final: gate, frozen-trunk proof, checkpoint ---------------------------
    md5_after = trunk_md5(model)
    rel = reliance_now()
    gate = wm_reliance_gate(rel, min_reliance=0.5)
    with torch.no_grad():
        wp_v = decode(v_trans, v_v0, training=False)
        final = kin_metrics(wp_v.float(), v_gt)
    summary = {
        "final_val": final, "baseline_displacement_readout": base,
        "reliance_final": rel["verdict"], "reliance_gate": gate,
        "trunk_frozen_proof": {"md5_before": md5_before, "md5_after": md5_after,
                               "identical": md5_before == md5_after},
        "steps": a.steps, "wall_s": round(time.time() - t0, 1),
        "_evidence_class": "MEASURED (ours)",
    }
    torch.save({"head": head.state_dict(), "summary": summary,
                "args": vars(a), "warm_started_from": "grounding.step.op",
                "base_ckpt": a.ckpt, "base_step": L.get("step")},
               os.path.join(a.out_dir, "unicycle_readout.pt"))
    fh.write(json.dumps({"summary": summary}) + "\n")
    fh.close()
    print(f"\n[SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    if not summary["trunk_frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK CHANGED DURING TRAINING — the run is invalid")
    if gate["status"] == "FAIL":
        print("⛔ RELIANCE GATE FAILED — this head bypasses the world model and "
              "must not be presented as a WM decoder", flush=True)


if __name__ == "__main__":
    main()
