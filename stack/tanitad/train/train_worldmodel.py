"""Phase 0 self-supervised world-model training (Stage A rehearsal).

Loss = residual multi-horizon prediction (change-weighted, A4)
     + SIGReg on embeddings AND predictions (A1, LeJEPA)
     + inverse-dynamics regression (A5).

Every run writes an experiment record (config + metrics.json with I2/I4
instrument rows + REPORT stub) per CONTINUATION_PROTOCOL §6.

Usage:
    python -m tanitad.train.train_worldmodel --smoke
    python -m tanitad.train.train_worldmodel --steps 2000 --episodes 200
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tanitad.config import StackConfig, smoke_config
from tanitad.data.toy_driving import ToyDrivingDataset
from tanitad.instruments.checks import (i2_batch_consistency, i3_episode_split,
                                        i4_imag_relative, instrument_rows)
from tanitad.models.fourbrain import WorldModel
from tanitad.models.predictor import change_weighted_mse


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def train(cfg: StackConfig, n_episodes: int = 40) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.train.device == "auto" else cfg.train.device
    torch.manual_seed(cfg.train.seed)

    train_ids, val_ids = i3_episode_split(list(range(n_episodes)), val_frac=0.2,
                                          seed=cfg.train.seed)          # I3
    max_h = max(cfg.predictor.horizons)
    ds_train = ToyDrivingDataset(train_ids, window=cfg.predictor.window,
                                 max_horizon=max_h, size=cfg.encoder.image_size)
    ds_val = ToyDrivingDataset(val_ids, window=cfg.predictor.window,
                               max_horizon=max_h, size=cfg.encoder.image_size)
    dl = DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True,
                    drop_last=True)

    model = WorldModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            betas=cfg.train.betas,
                            weight_decay=cfg.train.weight_decay)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    step, t0 = 0, time.time()
    log: dict[str, float] = {}
    data_iter = iter(dl)
    while step < cfg.train.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        frames = batch["frames"].to(device)          # [B, W, 1, H, W]
        actions = batch["actions"].to(device)        # [B, W, 2]
        fut = batch["future_frames"].to(device)      # [B, Hmax, 1, H, W]

        states = model.encode_window(frames)         # [B, W, S]
        # Future targets are encoded WITH gradients: LeJEPA needs SIGReg on all
        # embeddings and there is no stop-gradient/EMA crutch anywhere (A1).
        fut_states = model.encode_window(fut)        # [B, Hmax, S]

        preds = model.imagine(states, actions)
        z_t = states[:, -1]
        loss_pred = torch.zeros((), device=device)
        for k in cfg.predictor.horizons:
            target = fut_states[:, k - 1]
            prev = z_t if k == 1 else fut_states[:, k - 2]
            if cfg.predictor.change_weighted:
                loss_pred = loss_pred + change_weighted_mse(preds[k], target, prev)
            else:
                loss_pred = loss_pred + (preds[k] - target).pow(2).mean()
        loss_pred = loss_pred / len(cfg.predictor.horizons)

        # SIGReg on embeddings AND predictions (A1).
        z_all = torch.cat([states.reshape(-1, states.shape[-1]),
                           fut_states.reshape(-1, states.shape[-1])])
        z_pred_all = torch.cat([preds[k] for k in cfg.predictor.horizons])
        loss_sig = model.sigreg(z_all) + model.sigreg(z_pred_all)

        # Inverse dynamics on consecutive window states (A5).
        a_hat = model.inv_dyn(states[:, -2], states[:, -1])
        loss_inv = (a_hat - actions[:, -2]).pow(2).mean()

        loss = (cfg.loss.pred_weight * loss_pred
                + cfg.loss.sigreg.weight * loss_sig
                + cfg.loss.inv_dyn_weight * loss_inv)

        lr = cosine_lr(step, cfg.train.steps, cfg.train.warmup_steps, cfg.train.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.train.log_every == 0 or step == cfg.train.steps - 1:
            log = {"step": step, "loss": loss.item(), "pred": loss_pred.item(),
                   "sigreg": loss_sig.item(), "inv": loss_inv.item(), "lr": lr}
            print(json.dumps(log))
        step += 1

    # ---- Instrument rows (D-004) on validation data ----
    model.eval()
    vb = torch.utils.data.default_collate([ds_val[i] for i in range(
        min(64, len(ds_val)))])
    frames = vb["frames"].to(device)
    actions = vb["actions"].to(device)
    fut = vb["future_frames"].to(device)
    with torch.no_grad():
        states = model.encode_window(frames)
        fut_states = model.encode_window(fut)
        preds = model.imagine(states, actions)
    i2_pass, i2_dev = i2_batch_consistency(model.encode, frames[:, -1])   # I2
    i4 = i4_imag_relative(preds[1], fut_states[:, 0], states[:, -1])      # I4

    metrics = {
        "final": log,
        "wallclock_s": time.time() - t0,
        "device": device,
        "n_params": sum(p.numel() for p in model.parameters()),
        **instrument_rows(
            I2_batch_consistency_pass=i2_pass,
            I2_max_rel_dev=i2_dev,
            I3_split="episode-level",
            I4_imag_relative=i4,
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(json.dumps(metrics["instruments"], indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CI run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else StackConfig()
    if args.steps:
        cfg.train.steps = args.steps
    if args.out:
        cfg.train.out_dir = args.out
    train(cfg, n_episodes=args.episodes if not args.smoke else 8)


if __name__ == "__main__":
    main()
