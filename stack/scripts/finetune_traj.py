"""fix-B1: fine-tune the 27k world-model with METRIC-DYNAMICS grounding.

THE EXPERIMENT (dynamics mode, primary)
---------------------------------------
The diagnostic proved the latent does not hold metric ego-trajectory (oracle
ceiling 1.65 m vs constant-velocity 0.28 m; held-out probe 3.89 m). We fine-tune
the frozen 27k checkpoint ~8k steps, KEEPING the SSL world-model objective intact
(JEPA latent prediction + SigReg + imagination + action inverse-dynamics — that is
why D2 passes) and ADDING two proprioceptive, label-free grounding losses
(:mod:`tanitad.models.metric_dynamics`):

  loss = KEEP(pred + tactical + SigReg + action-invdyn + H15)          [unchanged]
       + λ_invdyn · metric-inverse-dynamics                            [--invdyn-weight, def 2.0]
       + λ_fwd    · forward metric consistency (rollout accumulation)  [--traj-weight,  def 1.0]

- metric-inverse-dynamics: from REAL latent pairs (x_t, x_{t+k}) regress the
  odometry metric relative ego-pose (Δx, Δy, Δyaw). Grounds the ENCODER.
- forward metric consistency: roll the operative predictor forward under the TRUE
  action sequence, decode each PREDICTED latent's per-step Δpose with the step
  readout, accumulate SE(2), and L2 the accumulated trajectory against the true
  odometry ego-trajectory. The trajectory EMERGES from grounded dynamics.

Both grounding signals are proprioceptive (CAN actions + IMU/GNSS ego-pose) — no
human labels. The metric heads are saved SEPARATELY from ``WorldModel.state_dict``
so ``driving_diagnostic`` / ``evaluate_checkpoint`` still load a vanilla model.

``--mode head`` swaps the dynamics grounding for a direct one-shot waypoint head
(:class:`tanitad.models.traj_head.TrajectoryHead`) — the imitation-flavored
ABLATION, kept for contrast only.

OOM guard: an inline v1/v2-cgroup posix_fadvise sweeper thread is PRE-ARMED before
the loop (mirrors /workspace/cache_guard.py) so a page-cache spike drops the
episode caches' clean pages instead of OOM-killing the trainer (2026-07-12 memcg
incident, 62 GB cap).

Usage (pod1):
  python scripts/finetune_traj.py --ckpt /workspace/ckpt27k_flagship.pt \
     --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
     --episodes 200 --steps 8000 --traj-weight 1.0 --invdyn-weight 2.0 \
     --grad-checkpoint --out /workspace/experiments/finetune_traj
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tanitad.config import base250cam_config, smoke_config
from tanitad.data._contract import EpisodeWindowDataset
from tanitad.data.mixing import load_episode
from tanitad.models.fourbrain import WorldModel
from tanitad.models.imagination import imagination_nll, sector_mask
from tanitad.models.metric_dynamics import (MetricInverseDynamics,
                                            StepDisplacementReadout,
                                            gt_ego_waypoints, gt_step_dposes,
                                            relative_ego_pose, rollout_decode,
                                            wrap_angle)
from tanitad.models.traj_head import TrajectoryHead
from tanitad.train.train_worldmodel import (_max_horizon,
                                            _needed_future_indices,
                                            _pred_losses, cosine_lr)

WP_STEPS = (5, 10, 15, 20)               # 0.5/1/1.5/2 s @ 10 Hz (eval waypoints)
K_MAX = max(WP_STEPS)


# --------------------------------------------------------------------------- #
# OOM guard — PRE-ARMED before the loop (mirrors /workspace/cache_guard.py)    #
# --------------------------------------------------------------------------- #
def start_cache_guard(cache_dirs, limit_gb: float = 60.0, period: float = 20.0
                      ) -> threading.Thread | None:
    """Launch a daemon thread that drops the episode caches' clean pages via
    posix_fadvise(DONTNEED) whenever cgroup memory usage crosses ``limit_gb``.

    Userspace page-cache relief (no privileges): the trainer survives a page-cache
    spike at the cost of periodic cold re-reads. Supports cgroup v1
    (memory.usage_in_bytes) and v2 (memory.current); a no-op if neither exists.
    """
    v1 = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    v2 = "/sys/fs/cgroup/memory.current"
    usage = v1 if os.path.exists(v1) else (v2 if os.path.exists(v2) else None)
    if usage is None:
        print("[guard] no v1/v2 cgroup usage file — guard is a no-op", flush=True)
        return None
    patterns = [str(Path(cd) / "*" / "ep_*.pt") for cd in cache_dirs]
    limit = int(limit_gb * 1024 ** 3)

    def _sweep() -> int:
        n = 0
        for pat in patterns:
            for p in glob.glob(pat):
                try:
                    fd = os.open(p, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                    n += 1
                except OSError:
                    pass
        return n

    def _loop():
        while True:
            try:
                u = int(open(usage).read())
            except OSError:
                return
            if u > limit:
                n = _sweep()
                try:
                    u2 = int(open(usage).read())
                    print(f"[guard] {u/1e9:.1f} -> {u2/1e9:.1f} GB ({n} files)",
                          flush=True)
                except OSError:
                    pass
            time.sleep(period)

    t = threading.Thread(target=_loop, daemon=True, name="cache_guard")
    t.start()
    print(f"[guard] pre-armed: sweep >{limit_gb:.0f} GB over {len(patterns)} "
          f"cache pattern(s), every {period:.0f}s ({usage})", flush=True)
    return t


# --------------------------------------------------------------------------- #
# Data — combine cached corpora (comma2k19 + PhysicalAI) train splits          #
# --------------------------------------------------------------------------- #
def load_cached_episodes(cache_dirs, n_episodes: int, split: str = "train"):
    """Load up to ``n_episodes`` episodes per cache dir from its ``*{split}*``
    subdir (mmap). Fail-loud if a cache dir has no matching split dir/episodes."""
    eps = []
    for cd in cache_dirs:
        dirs = sorted(Path(cd).glob(f"*{split}*"))
        assert dirs, f"no *{split}* dir under {cd}"
        files = sorted(dirs[-1].glob("ep_*.pt"))[:n_episodes]
        assert files, f"no ep_*.pt in {dirs[-1]}"
        loaded = [load_episode(str(p), mmap=True) for p in files]
        eps.extend(loaded)
        print(f"[data] {cd}: {len(loaded)} {split} episodes from {dirs[-1].name}",
              flush=True)
    assert eps, "no episodes loaded"
    return eps


def build_heads(mode: str, state_dim: int, cfg, device, hidden: int = 512,
                traj_use_pred: bool = False):
    """Instantiate the fine-tune heads for ``mode`` (kept separate from the
    WorldModel). Returns a dict of nn.Modules on ``device``."""
    if mode == "dynamics":
        return {
            "metric_invdyn": MetricInverseDynamics(state_dim, hidden=hidden).to(device),
            "step_readout": StepDisplacementReadout(state_dim, hidden=hidden).to(device),
        }
    if mode == "head":
        n_extra = len(cfg.predictor.horizons) if traj_use_pred else 0
        return {"traj_head": TrajectoryHead(state_dim, horizons=WP_STEPS,
                                            n_extra_states=n_extra).to(device)}
    raise ValueError(f"unknown mode {mode}")


# --------------------------------------------------------------------------- #
# Loss assembly — KEEP the SSL core, ADD metric grounding                      #
# --------------------------------------------------------------------------- #
def compute_losses(model, heads, batch, cfg, needed_fut, idx_of, *,
                   mode: str, invdyn_weight: float, traj_weight: float,
                   fwd_k: int, fwd_step_weight: float, mid_horizons,
                   pose_scale: float, device):
    """Total loss + per-component log dict + raw loss-tensor parts (for tests).

    SSL terms are byte-for-byte the ``train_worldmodel`` assembly (weights from
    ``cfg``), so the JEPA/SigReg/imagination core — and D2 — are preserved. The
    metric losses are conditioned by dividing metre errors by ``pose_scale`` so
    the λ weights operate on O(1) quantities and gradient clipping does not starve
    the SSL gradients; the heads still output raw metres (eval reads them direct).
    """
    frames = batch["frames"].to(device)                  # [B, W, C, H, W]
    actions = batch["actions"].to(device)                # [B, W, 2]
    fut = batch["future_frames"].to(device)              # [B, Hmax, C, H, W]
    fut_actions = batch.get("future_actions")
    fut_actions = fut_actions.to(device) if fut_actions is not None else None
    future_poses = batch["future_poses"].to(device).float()   # [B, Hmax, 4]
    pose_last = batch["pose_last"].to(device).float()         # [B, 4]

    states = model.encode_window(frames)                 # [B, W, S] (grad -> encoder)
    fut_states = model.encode_window(fut[:, needed_fut])  # [B, |needed|, S]
    preds = model.imagine(states, actions)
    z_t = states[:, -1]

    # ---- KEEP: SSL world-model losses (identical to train_worldmodel) --------
    loss_pred = _pred_losses(preds, cfg.predictor.horizons, z_t, fut_states,
                             idx_of, cfg.predictor.change_weighted)
    loss_tac = torch.zeros((), device=device)
    if model.tactical_pred is not None:
        tac_preds = model.tactical_pred(states, actions)
        loss_tac = _pred_losses(tac_preds, cfg.tactical_pred.horizons, z_t,
                                fut_states, idx_of, cfg.predictor.change_weighted)
    z_all = torch.cat([states.reshape(-1, states.shape[-1]),
                       fut_states.reshape(-1, states.shape[-1])])
    z_pred_all = torch.cat([preds[k] for k in cfg.predictor.horizons])
    loss_sig = model.sigreg(z_all) + model.sigreg(z_pred_all)
    a_hat = model.inv_dyn(states[:, -2], states[:, -1])
    loss_inv = (a_hat - actions[:, -2]).pow(2).mean()
    loss_h15 = torch.zeros((), device=device)
    if model.imagination is not None and torch.rand(()) < cfg.h15.mask_prob:
        masked, vis = sector_mask(frames[:, -1], model.encoder.grid_hw)
        tok_belief = model.encode_tokens(masked)
        tok_true = model.encode_tokens(fut[:, 0])
        imag_pred, logvar = model.imagination(tok_belief, vis)
        loss_h15 = imagination_nll(imag_pred, tok_true, logvar, vis,
                                   cfg.h15.observed_weight)

    ssl = (cfg.loss.pred_weight * loss_pred
           + cfg.loss.pred_weight * 0.5 * loss_tac
           + cfg.loss.sigreg.weight * loss_sig
           + cfg.loss.inv_dyn_weight * loss_inv
           + cfg.h15.weight * loss_h15)

    parts = {"pred": loss_pred, "tac": loss_tac, "sigreg": loss_sig,
             "inv": loss_inv, "h15": loss_h15}
    log = {"pred": loss_pred.item(), "tac": loss_tac.item(),
           "sigreg": loss_sig.item(), "inv": loss_inv.item(),
           "h15": loss_h15.item()}
    ps = pose_scale

    if mode == "dynamics":
        mid, step_ro = heads["metric_invdyn"], heads["step_readout"]
        # metric inverse dynamics on REAL latent pairs -> grounds the encoder
        loss_mid = torch.zeros((), device=device)
        mid_de = 0.0
        for kh in mid_horizons:
            dpose = mid(z_t, fut_states[:, idx_of[kh - 1]])
            tgt = relative_ego_pose(pose_last, future_poses[:, kh - 1])
            loss_mid = loss_mid + ((dpose[..., :2] - tgt[..., :2]) / ps).pow(2).mean() \
                + wrap_angle(dpose[..., 2] - tgt[..., 2]).pow(2).mean()
            mid_de += float((dpose[..., :2] - tgt[..., :2]).detach()
                            .norm(dim=-1).mean())
        loss_mid = loss_mid / len(mid_horizons)
        mid_de /= len(mid_horizons)
        # forward metric consistency: rollout w/ TRUE actions, decode PREDICTED
        # latents, accumulate SE(2), match odometry ego-trajectory
        pred_wp, pred_step = rollout_decode(model.predictor, states, actions,
                                            fut_actions, step_ro, fwd_k)
        gt_wp = gt_ego_waypoints(pose_last, future_poses, range(1, fwd_k + 1))
        gt_step = gt_step_dposes(pose_last, future_poses, fwd_k)
        loss_acc = ((pred_wp - gt_wp) / ps).pow(2).mean()
        loss_step = ((pred_step[..., :2] - gt_step[..., :2]) / ps).pow(2).mean() \
            + wrap_angle(pred_step[..., 2] - gt_step[..., 2]).pow(2).mean()
        loss_fwd = loss_acc + fwd_step_weight * loss_step
        fwd_ade = float((pred_wp.detach() - gt_wp).norm(dim=-1).mean())
        total = ssl + invdyn_weight * loss_mid + traj_weight * loss_fwd
        parts.update({"mid": loss_mid, "fwd": loss_fwd,
                      "fwd_acc": loss_acc, "fwd_step": loss_step})
        log.update({"mid": loss_mid.item(), "fwd": loss_fwd.item(),
                    "mid_de_m": round(mid_de, 4), "fwd_ade_m": round(fwd_ade, 4)})
    elif mode == "head":
        head = heads["traj_head"]
        extra = ([preds[k] for k in cfg.predictor.horizons]
                 if head.n_extra_states else None)
        pred_wp = head(z_t, extra)
        gt_wp = gt_ego_waypoints(pose_last, future_poses, head.horizons)
        loss_traj = ((pred_wp - gt_wp) / ps).pow(2).mean()
        traj_ade = float((pred_wp.detach() - gt_wp).norm(dim=-1).mean())
        total = ssl + traj_weight * loss_traj
        parts.update({"traj": loss_traj})
        log.update({"traj": loss_traj.item(), "traj_ade_m": round(traj_ade, 4)})
    else:
        raise ValueError(f"unknown mode {mode}")

    return total, log, parts


def _health_rows(states) -> dict:
    """erank / dim-std / step-ratio on the last micro-batch states (collapse
    watchdog, A9 applied to training — same rows as train_worldmodel)."""
    with torch.no_grad():
        flat = states.detach().float().reshape(-1, states.shape[-1])
        s = torch.linalg.svdvals(flat - flat.mean(0))
        p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
        erank = float(torch.exp(-(p * p.log()).sum()))
        step_ratio = float((states[:, 1:] - states[:, :-1]).norm(dim=-1).mean()
                           / flat.norm(dim=-1).mean().clamp_min(1e-8))
        dim_std = float(flat.std(0).mean())
    return {"erank": round(erank, 1), "dim_std": round(dim_std, 5),
            "step_ratio": round(step_ratio, 5)}


# --------------------------------------------------------------------------- #
# Fine-tune loop                                                               #
# --------------------------------------------------------------------------- #
def finetune(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = smoke_config() if args.config == "smoke" else base250cam_config()
    if args.grad_checkpoint:
        cfg.encoder.grad_checkpoint = True
    torch.manual_seed(args.seed)
    use_amp = (not args.no_amp) and device == "cuda"

    max_h = max(_max_horizon(cfg), K_MAX, args.fwd_k)
    needed_fut, idx_of = _needed_future_indices(cfg)
    mid_horizons = ([int(x) for x in args.mid_horizons.split(",")]
                    if args.mid_horizons else list(cfg.predictor.horizons))
    for kh in mid_horizons:
        assert (kh - 1) in idx_of, (
            f"mid-horizon {kh}: future index {kh-1} not encoded "
            f"(available {sorted(idx_of)}); pick from horizons that are encoded")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PRE-ARM the OOM guard BEFORE any heavy allocation / the loop.
    start_cache_guard(args.cache_dirs, limit_gb=args.guard_limit_gb)

    episodes = load_cached_episodes(args.cache_dirs, args.episodes, split="train")
    ds = EpisodeWindowDataset(episodes, window=cfg.predictor.window,
                              max_horizon=max_h)
    assert len(ds) > 0, "no training windows (episodes too short for window+max_h)"
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers,
                    persistent_workers=bool(args.workers))
    print(f"[data] {len(ds)} train windows, {len(episodes)} episodes, "
          f"window {cfg.predictor.window}, max_h {max_h}, batch {args.batch_size}, "
          f"mode {args.mode}, fwd_k {args.fwd_k}", flush=True)

    model = WorldModel(cfg).to(device)
    heads = build_heads(args.mode, model.state_dim, cfg, device,
                        traj_use_pred=args.traj_use_pred)
    params = list(model.parameters())
    for h in heads.values():
        params += list(h.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=cfg.train.betas,
                            weight_decay=args.weight_decay)

    # Resume own fine-tune ckpt if present; else load the base 27k checkpoint.
    ckpt_path = out_dir / "ckpt.pt"
    step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        for name, h in heads.items():
            if name in ck:
                h.load_state_dict(ck[name])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] fine-tune ckpt found — resuming at step {step}", flush=True)
    else:
        base = torch.load(args.ckpt, map_location=device, weights_only=True)
        state = base["model"] if isinstance(base, dict) and "model" in base else base
        missing, unexpected = model.load_state_dict(state, strict=False)
        base_step = int(base.get("step", -1)) if isinstance(base, dict) else -1
        assert not missing, f"base ckpt missing WorldModel params: {missing[:6]}..."
        print(f"[init] loaded base ckpt {args.ckpt} (step {base_step}); "
              f"unexpected keys ignored: {len(unexpected)}; heads fresh", flush=True)

    n_params = sum(p.numel() for p in params)
    print(f"[init] optimizing {n_params/1e6:.1f}M params "
          f"(model + {list(heads)}) lr {args.lr}", flush=True)

    (out_dir / "finetune_config.json").write_text(json.dumps({
        "mode": args.mode, "ckpt": args.ckpt, "cache_dirs": args.cache_dirs,
        "episodes": args.episodes, "steps": args.steps, "lr": args.lr,
        "invdyn_weight": args.invdyn_weight, "traj_weight": args.traj_weight,
        "fwd_k": args.fwd_k, "fwd_step_weight": args.fwd_step_weight,
        "pose_scale": args.pose_scale, "mid_horizons": mid_horizons,
        "batch_size": args.batch_size, "accum": args.accum,
        "grad_checkpoint": cfg.encoder.grad_checkpoint, "amp": use_amp,
        "config": args.config, "n_params": n_params, "wp_steps": list(WP_STEPS),
    }, indent=2), encoding="utf-8")

    model.train()
    for h in heads.values():
        h.train()
    data_iter = iter(dl)
    accum = max(1, args.accum)
    t_data = t_step = 0.0
    logf = (out_dir / "finetune_log.jsonl").open("a")
    t0 = time.time()

    def save_ckpt(s):
        tmp = ckpt_path.with_suffix(".tmp")
        blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": s,
                "mode": args.mode}
        for name, h in heads.items():
            blob[name] = h.state_dict()
        torch.save(blob, tmp)
        tmp.replace(ckpt_path)
        print(f"[ckpt] saved at step {s} -> {ckpt_path}", flush=True)

    while step < args.steps:
        lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        t_s0 = time.perf_counter()
        log = {}
        for _micro in range(accum):
            t_d0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            t_data += time.perf_counter() - t_d0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                total, log, parts = compute_losses(
                    model, heads, batch, cfg, needed_fut, idx_of,
                    mode=args.mode, invdyn_weight=args.invdyn_weight,
                    traj_weight=args.traj_weight, fwd_k=args.fwd_k,
                    fwd_step_weight=args.fwd_step_weight,
                    mid_horizons=mid_horizons, pose_scale=args.pose_scale,
                    device=device)
            (total / accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % args.ckpt_every == 0:
            save_ckpt(step)

        if step % args.log_every == 0 or step == args.steps - 1:
            # health rows recomputed on a fresh no-grad encode of this batch
            with torch.no_grad():
                hs = model.encode_window(batch["frames"].to(device))
            row = {"step": step, "loss": float(total.item()), "lr": round(lr, 8),
                   "data_s": round(t_data, 1), "step_s": round(t_step, 1)}
            row.update(log)
            row.update(_health_rows(hs))
            t_data = t_step = 0.0
            line = json.dumps(row)
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()
        step += 1

    save_ckpt(step - 1)
    logf.close()
    summary = {"done": True, "final_step": step - 1, "mode": args.mode,
               "wallclock_s": round(time.time() - t0, 1), "out": str(ckpt_path)}
    (out_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    print("FINETUNE_TRAJ_DONE", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="base 27k checkpoint")
    ap.add_argument("--cache-dirs", nargs="+", required=True,
                    help="epcache roots (each has *train*/*val* subdirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["dynamics", "head"], default="dynamics",
                    help="dynamics = metric-grounding (PRIMARY); head = direct "
                         "waypoint regression (ablation)")
    ap.add_argument("--episodes", type=int, default=200,
                    help="max episodes per cache dir (cgroup-bounded)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--invdyn-weight", type=float, default=2.0,
                    help="λ_invdyn: metric-inverse-dynamics weight")
    ap.add_argument("--traj-weight", type=float, default=1.0,
                    help="λ_fwd: forward-metric-consistency (rollout) weight; in "
                         "--mode head this is the direct-head weight")
    ap.add_argument("--fwd-k", type=int, default=20,
                    help="forward rollout steps (20 = 2 s @ 10 Hz)")
    ap.add_argument("--fwd-step-weight", type=float, default=0.5,
                    help="weight of the per-step Δpose anchor within λ_fwd")
    ap.add_argument("--mid-horizons", type=str, default=None,
                    help="comma list of metric-invdyn horizons (default: "
                         "predictor horizons)")
    ap.add_argument("--pose-scale", type=float, default=10.0,
                    help="metre normalizer for metric losses (heads emit metres)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute encoder activations (F-5 memory lever)")
    ap.add_argument("--traj-use-pred", action="store_true",
                    help="(--mode head) also feed predictor states to the head")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--guard-limit-gb", type=float, default=60.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--config", choices=["base250cam", "smoke"],
                    default="base250cam")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
