"""REF-C trainer: behavior-clone TCP-C (tanitad/refs/refc.py) from cached
episodes.

Mirrors scripts/refb_train.py: the SAME fail-loud dataset (imported, not
copied — FailLoudWindowDataset with the derived nav_cmd/nav_valid/
route_target fields), the same cached-mode ep_*.pt episode dirs, the same
atomic-ckpt/resume/jsonl-step-log/--workers machinery. Divergences are the
POINT of the arm and are deliberate:

    optimizer   Adam, lr 1e-4 (the TCP paper's operating point — NOT the
                main run's AdamW/3e-4; batch/warmup/save/log cadence still
                read programmatically from base250cam_config().train)
    losses      wp L1 (1.0)           ego-frame waypoints at (5,10,15,20)
                                      [refc1: fixed-distance path checkpoints
                                      at (2,5,10,20) m via
                                      refb_labels.path_targets]
                control L1 (1.0)      K=4 future (steer, accel) pairs vs
                                      future_actions[:, :4]
                speed L1 (0.05)       v at t+5 (0.5 s) / 10.0, image-branch
                                      head (TCP's anti-shortcut placement)
                LAW MSE (1.0)         predicted next pooled latent vs the
                                      no_grad-encoded frames at t+5 (LAW
                                      bolt-on — replaces Roach distillation)
                [refc1] speed CE (1.0) target-speed class (4 bins, [0,30] m/s)

v0 = pose_last[:, 3] is ALWAYS fed (the model applies /10 scaling and the
per-sample ego-dropout p=0.5 internally, training-gated).

Usage (only AFTER Sayed's GO — implementation ships untrained):
  python scripts/refc_train.py --data-root /workspace/data \
      --out /workspace/experiments/refc-30k --steps 30000
Smoke (CPU):
  python scripts/refc_train.py --data-root <cache> --out <dir> --steps 10 \
      --batch 8 --smoke --log-every 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import refb_labels
from refb_train import FailLoudWindowDataset, load_cached_episodes
from tanitad.config import base250cam_config
from tanitad.refs.refc import (RefCModel, param_breakdown, refc_config,
                               refc_smoke_config)
from tanitad.train.train_worldmodel import cosine_lr

# Loss weights (module docstring). wp/control are the co-equal primaries
# (this stack IS the action decoder), speed is TCP's 0.05, LAW rides at 1.0
# (the imagination-error signal this reference is allowed to have).
WP_WEIGHT = 1.0
CTRL_WEIGHT = 1.0
SPEED_WEIGHT = 0.05
LAW_WEIGHT = 1.0
SPEED_CLS_WEIGHT = 1.0        # refc1 only

TCP_LR = 1e-4                 # Adam lr — the TCP paper's operating point
LAW_AHEAD = 5                 # LAW target: pooled latent 0.5 s (5 steps) ahead
SPEED_AHEAD = 5               # speed target: v at t+5 (same 0.5 s horizon)


# ---- losses ------------------------------------------------------------------

def compute_losses(model: RefCModel, batch: dict, device: str = "cpu") -> dict:
    """One forward pass -> all loss components (tensors, differentiable).

    The LAW target is the pooled latent of the frame stack ``LAW_AHEAD``
    steps past the window, encoded under no_grad through the SAME encoder
    (spec item 5); the prediction path keeps gradients THROUGH the predicted
    waypoints — that is the point of the aux."""
    frames = batch["frames"].to(device)            # [B, W, C, H, W']
    fut_frames = batch["future_frames"].to(device)  # [B, Hmax, C, H, W']
    fut_actions = batch["future_actions"].to(device)   # [B, Hmax, 2]
    fut_poses = batch["future_poses"].to(device)   # [B, Hmax, 4]
    pose_last = batch["pose_last"].to(device)      # [B, 4]
    nav_cmd = batch["nav_cmd"].to(device)          # [B] long (derived)
    v0 = pose_last[:, 3]                           # [B] current ego speed (t0)

    out = model(frames, nav_cmd=nav_cmd, v0=v0)
    cfg = model.cfg
    k_ctrl = cfg.control.k

    # Waypoint L1: time-indexed ego-frame targets, or (refc1) fixed-distance
    # path checkpoints via the arc-length resample.
    if cfg.refc1:
        wp_tgt = refb_labels.path_targets(pose_last, fut_poses, cfg.path_dists)
    else:
        wp_tgt = refb_labels.waypoint_targets(pose_last, fut_poses,
                                              cfg.trajectory.horizons)
    loss_wp = (out["wp_seq"] - wp_tgt).abs().mean()

    # Control L1: K future action pairs vs the recorded expert actions.
    loss_ctrl = (out["actions"] - fut_actions[:, :k_ctrl]).abs().mean()

    # Speed L1: v at t+SPEED_AHEAD / 10, image-branch head.
    speed_tgt = fut_poses[:, SPEED_AHEAD - 1, 3] / 10.0
    loss_speed = (out["speed_pred"] - speed_tgt).abs().mean()

    # LAW latent MSE: no_grad target through the same encoder.
    with torch.no_grad():
        law_tgt = model.encode_pooled(fut_frames[:, LAW_AHEAD - 1])
    loss_law = (out["law_pred"] - law_tgt).pow(2).mean()

    # refc1: target-speed classification (bins over [0, speed_max]).
    if cfg.refc1:
        v_tgt = fut_poses[:, SPEED_AHEAD - 1, 3].clamp(0.0, cfg.speed_max)
        edges = torch.linspace(0.0, cfg.speed_max, cfg.speed_bins + 1,
                               device=v_tgt.device)[1:-1]
        cls_tgt = torch.bucketize(v_tgt, edges)
        loss_speed_cls = F.cross_entropy(out["speed_logits"], cls_tgt)
        speed_mae = (out["target_speed"].detach() - v_tgt).abs().mean()
    else:
        loss_speed_cls = torch.zeros((), device=out["pooled"].device)
        speed_mae = torch.zeros((), device=out["pooled"].device)

    loss = (WP_WEIGHT * loss_wp + CTRL_WEIGHT * loss_ctrl
            + SPEED_WEIGHT * loss_speed + LAW_WEIGHT * loss_law
            + SPEED_CLS_WEIGHT * loss_speed_cls)
    return {"loss": loss, "wp": loss_wp, "ctrl": loss_ctrl,
            "speed": loss_speed, "law": loss_law,
            "speed_cls": loss_speed_cls, "speed_mae": speed_mae,
            "nav_follow_frac": (nav_cmd == 0).float().mean(),
            "pooled": out["pooled"]}


def _save_ckpt(path: Path, model, opt, step: int) -> None:
    # atomic write: a kill mid-save must not corrupt the resume point
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)
    print(f"[ckpt] saved at step {step}", flush=True)


def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    # Cadence/batch from the main run's config object (mirrors refb_train);
    # the OPTIMIZER is deliberately TCP's (Adam, lr 1e-4) — the arm's point.
    main_tr = base250cam_config().train
    lr = args.lr if args.lr is not None else TCP_LR
    batch = args.batch if args.batch is not None else main_tr.batch_size
    warmup = args.warmup if args.warmup is not None else main_tr.warmup_steps
    save_every = args.save_every if args.save_every is not None \
        else main_tr.save_every
    log_every = args.log_every if args.log_every is not None \
        else main_tr.log_every

    cfg = refc_smoke_config() if args.smoke else refc_config()
    cfg.refc1 = bool(args.refc1)       # gated BEFORE build (module presence)
    model = RefCModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    max_h = max(max(cfg.trajectory.horizons), cfg.control.k,
                LAW_AHEAD, SPEED_AHEAD)
    train_eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                                args.episodes)
    ds = FailLoudWindowDataset(train_eps, window=cfg.window, max_horizon=max_h,
                               channels=cfg.encoder.in_channels)
    assert len(ds) >= batch, \
        f"only {len(ds)} windows for batch {batch} — add episodes"
    dl_kw = dict(batch_size=batch, shuffle=True, drop_last=True)
    if getattr(args, "workers", 0) > 0:
        dl_kw.update(num_workers=args.workers, persistent_workers=True,
                     prefetch_factor=4, pin_memory=True)
    dl = DataLoader(ds, **dl_kw)
    print(f"[refc] train: {len(train_eps)} episodes / {len(ds)} windows "
          f"from {train_dir}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"arch": "REF-C (TCP-C)", "cfg": dataclasses.asdict(cfg),
         "args": vars(args),
         "optimizer": {"kind": "Adam (TCP paper)", "lr": lr,
                       "warmup": warmup, "schedule": "cosine (main run's)"},
         "loss_weights": {"wp": WP_WEIGHT, "ctrl": CTRL_WEIGHT,
                          "speed": SPEED_WEIGHT, "law": LAW_WEIGHT,
                          "speed_cls": SPEED_CLS_WEIGHT},
         "param_breakdown": param_breakdown(model)},
        indent=2, default=str), encoding="utf-8")

    # Interruptible-pod resume (refb_train convention).
    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] checkpoint found — resuming at step {step}",
              flush=True)

    data_iter = iter(dl)
    t_data = t_step = 0.0
    last_log: dict = {}
    while step < args.steps:
        cur_lr = cosine_lr(step, args.steps, warmup, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        t_s0 = time.perf_counter()
        t_d0 = time.perf_counter()
        try:
            batch_d = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch_d = next(data_iter)
        t_data += time.perf_counter() - t_d0

        opt.zero_grad(set_to_none=True)
        out = compute_losses(model, batch_d, device)
        out["loss"].backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % save_every == 0:
            _save_ckpt(ckpt_path, model, opt, step)

        if step % log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)  # noqa: E731
            last_log = {
                "step": step, "loss": sc(out["loss"]),
                "wp": sc(out["wp"]), "ctrl": sc(out["ctrl"]),
                "speed": sc(out["speed"]), "law": sc(out["law"]),
                "speed_cls": sc(out["speed_cls"]),
                "speed_mae": sc(out["speed_mae"]),
                "nav_follow_frac": sc(out["nav_follow_frac"]),
                "gnorm": round(gnorm, 4), "lr": cur_lr,
                "data_s": round(t_data, 1), "step_s": round(t_step, 1),
            }
            t_data = t_step = 0.0
            print(json.dumps(last_log), flush=True)
        step += 1

    _save_ckpt(ckpt_path, model, opt, step - 1)     # final resume point
    metrics = {"final": last_log, "steps": step, "device": device,
               "param_breakdown": param_breakdown(model),
               "n_params_trainable": sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)}
    # Light val row (REAL-only val dir), if present.
    try:
        val_eps, _ = load_cached_episodes(args.data_root, "*val*",
                                          min(args.episodes or 8, 8))
        vds = FailLoudWindowDataset(val_eps, window=cfg.window,
                                    max_horizon=max_h,
                                    channels=cfg.encoder.in_channels)
        model.eval()
        with torch.no_grad():
            vb = torch.utils.data.default_collate(
                [vds[i] for i in range(min(16, len(vds)))])
            vout = compute_losses(model, vb, device)
        metrics["val"] = {k: round(float(vout[k]), 5)
                          for k in ("wp", "ctrl", "speed", "law",
                                    "speed_cls", "speed_mae",
                                    "nav_follow_frac")}
    except AssertionError:
        pass                                        # no val cache dir
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"done": True, "steps": step, "out": str(out_dir)}),
          flush=True)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="epcache root containing *train*/*val* dirs of "
                         "ep_*.pt (the train_worldmodel --data cached layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=None,
                    help="default: the main run's batch (base250cam)")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"default: the TCP paper's Adam lr ({TCP_LR})")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: the main run's warmup (base250cam, 2000)")
    ap.add_argument("--refc1", action="store_true",
                    help="REF-C.1: fixed-distance path checkpoints at "
                         "(2,5,10,20) m + target-speed classification head")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (0 = in-loop decode, old behavior)")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config (CI/CPU smoke; 1-channel 64 px episodes)")
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
