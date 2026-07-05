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

from tanitad.config import (StackConfig, base250_config, base250cam_config,
                            smoke_config)
from tanitad.data.toy_driving import ToyDrivingDataset
from tanitad.instruments.checks import (i2_batch_consistency, i3_episode_split,
                                        i4_imag_relative, instrument_rows)
from tanitad.models.fourbrain import WorldModel
from tanitad.models.imagination import d9_rows, imagination_nll, sector_mask
from tanitad.models.predictor import change_weighted_mse


def _max_horizon(cfg: StackConfig) -> int:
    h = max(cfg.predictor.horizons)
    if cfg.tactical_pred is not None:
        h = max(h, max(cfg.tactical_pred.horizons))
    return h


def _pred_losses(preds: dict[int, torch.Tensor], horizons, z_t, fut_states,
                 change_weighted: bool) -> torch.Tensor:
    loss = torch.zeros((), device=z_t.device)
    for k in horizons:
        target = fut_states[:, k - 1]
        prev = z_t if k == 1 else fut_states[:, k - 2]
        if change_weighted:
            loss = loss + change_weighted_mse(preds[k], target, prev)
        else:
            loss = loss + (preds[k] - target).pow(2).mean()
    return loss / len(horizons)


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def _build_datasets(cfg: StackConfig, n_episodes: int, data: str,
                    data_root: str | None):
    max_h = _max_horizon(cfg)
    if data == "comma2k19":
        from tanitad.data.comma2k19 import (Comma2k19Dataset,
                                            discover_segments, split_by_route)
        assert data_root, "--data-root required for comma2k19"
        assert cfg.encoder.in_channels == 6, \
            "comma2k19 emits 6-channel frames — use --config base250cam"
        segs = discover_segments(data_root)[:n_episodes]
        assert segs, f"no comma2k19 segments under {data_root}"
        train_segs, val_segs = split_by_route(segs, val_frac=0.2,
                                              seed=cfg.train.seed)       # I3
        print(f"[data] comma2k19: {len(train_segs)} train / "
              f"{len(val_segs)} val segments (route-level split)")
        mk = lambda s: Comma2k19Dataset(s, window=cfg.predictor.window,
                                        max_horizon=max_h,
                                        size=cfg.encoder.image_size)
        return mk(train_segs), mk(val_segs)
    # default: procedural toy (CI fixture / pipeline checks only, D-009)
    train_ids, val_ids = i3_episode_split(list(range(n_episodes)), val_frac=0.2,
                                          seed=cfg.train.seed)           # I3
    ep_steps = max(80, cfg.predictor.window + max_h + 40)
    mk = lambda ids: ToyDrivingDataset(ids, window=cfg.predictor.window,
                                       max_horizon=max_h,
                                       size=cfg.encoder.image_size,
                                       steps=ep_steps)
    return mk(train_ids), mk(val_ids)


def train(cfg: StackConfig, n_episodes: int = 40, data: str = "toy",
          data_root: str | None = None) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.train.device == "auto" else cfg.train.device
    torch.manual_seed(cfg.train.seed)

    ds_train, ds_val = _build_datasets(cfg, n_episodes, data, data_root)
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
        loss_pred = _pred_losses(preds, cfg.predictor.horizons, z_t, fut_states,
                                 cfg.predictor.change_weighted)

        # Tactical brain: same SSL objective at maneuver horizons (8/16).
        loss_tac = torch.zeros((), device=device)
        if model.tactical_pred is not None:
            tac_preds = model.tactical_pred(states, actions)
            loss_tac = _pred_losses(tac_preds, cfg.tactical_pred.horizons, z_t,
                                    fut_states, cfg.predictor.change_weighted)

        # SIGReg on embeddings AND predictions (A1).
        z_all = torch.cat([states.reshape(-1, states.shape[-1]),
                           fut_states.reshape(-1, states.shape[-1])])
        z_pred_all = torch.cat([preds[k] for k in cfg.predictor.horizons])
        loss_sig = model.sigreg(z_all) + model.sigreg(z_pred_all)

        # Inverse dynamics on consecutive window states (A5).
        a_hat = model.inv_dyn(states[:, -2], states[:, -1])
        loss_inv = (a_hat - actions[:, -2]).pow(2).mean()

        # H15: sector-masked imagination on the token grid (D-008).
        loss_h15 = torch.zeros((), device=device)
        if model.imagination is not None and torch.rand(()) < cfg.h15.mask_prob:
            f_t, f_next = frames[:, -1], fut[:, 0]
            masked, vis = sector_mask(f_t, model.encoder.grid_hw)
            tok_belief = model.encode_tokens(masked)
            tok_true = model.encode_tokens(f_next)
            imag_pred, logvar = model.imagination(tok_belief, vis)
            loss_h15 = imagination_nll(imag_pred, tok_true, logvar, vis,
                                       cfg.h15.observed_weight)

        loss = (cfg.loss.pred_weight * loss_pred
                + cfg.loss.pred_weight * 0.5 * loss_tac
                + cfg.loss.sigreg.weight * loss_sig
                + cfg.loss.inv_dyn_weight * loss_inv
                + cfg.h15.weight * loss_h15)

        lr = cosine_lr(step, cfg.train.steps, cfg.train.warmup_steps, cfg.train.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.train.log_every == 0 or step == cfg.train.steps - 1:
            log = {"step": step, "loss": loss.item(), "pred": loss_pred.item(),
                   "tac": loss_tac.item(), "sigreg": loss_sig.item(),
                   "inv": loss_inv.item(), "h15": loss_h15.item(), "lr": lr}
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

    # D9 evidence rows (H15): imagination quality in hidden sectors.
    d9 = {}
    if model.imagination is not None:
        with torch.no_grad():
            masked, vis = sector_mask(frames[:, -1], model.encoder.grid_hw)
            imag_pred, logvar = model.imagination(model.encode_tokens(masked), vis)
            d9 = d9_rows(imag_pred, model.encode_tokens(fut[:, 0]), logvar, vis)

    metrics = {
        "d9": d9,
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
    ap.add_argument("--config", choices=["smoke", "base", "base250", "base250cam"],
                    default="base", help="base250cam = TanitAD-4B-M on real camera "
                                         "data (PRIMARY, D-009)")
    ap.add_argument("--smoke", action="store_true", help="alias for --config smoke")
    ap.add_argument("--data", choices=["toy", "comma2k19"], default="toy",
                    help="toy is a CI fixture only (D-009: real data first)")
    ap.add_argument("--data-root", type=str, default=None,
                    help="comma2k19 extracted root (contains Chunk_*/...)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=40,
                    help="toy: #episodes; comma2k19: max #segments")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke or args.config == "smoke":
        cfg = smoke_config()
    elif args.config == "base250":
        cfg = base250_config()
    elif args.config == "base250cam":
        cfg = base250cam_config()
    else:
        cfg = StackConfig()
    if args.steps:
        cfg.train.steps = args.steps
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.out:
        cfg.train.out_dir = args.out
    n_eps = 8 if (args.smoke or args.config == "smoke") else args.episodes
    train(cfg, n_episodes=n_eps, data=args.data, data_root=args.data_root)


if __name__ == "__main__":
    main()
