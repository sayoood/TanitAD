"""REF-B trainer: behavior-clone the 4-layer E2E stack from cached episodes.

Matched-everything principle (REFERENCE_ARCHITECTURES REF-B): the optimizer,
schedule and data path are IDENTICAL to the main run — AdamW betas/weight-decay
/lr/warmup/batch/save cadence are read PROGRAMMATICALLY from
base250cam_config().train (no EMA, no augmentation: the main trainer has
neither), data is the same cached-mode ep_*.pt episode dirs
(*train*/*val* under --data-root, episode-subset bound, mmap) that
train_worldmodel --data cached consumes. Only the architecture axis differs.

Losses (weights documented below):
    action BC            (steer, accel) at t vs actions[:, -1]
    action-sequence BC   direct heads a_{t+1..t+4} vs future_actions (0.5 s)
    waypoint L2          ego-frame 2 s waypoints vs pose-derived targets
    maneuver CE          vs kinematic pseudo-labels (refb_labels.py)
    route-heading CE     strategic aux (rev2) vs the SAME nav derivation,
                         valid-masked, inverse-class-frequency weighted
                         (clamped <= 10 — comma2k19 is highway-dominated, so
                         `follow`/route_straight dominates heavily and would
                         otherwise drown the rare left/right windows)
    inverse dynamics     same aux + weight (0.5) as the main run (fair aux)
    confidence           (conf_pred - realized waypoint error)^2, DETACHED —
                         optimizes ONLY the confidence head (fallback (a))
Nav commands are DERIVED PER WINDOW (rev2 defect fix — the layer previously
trained on a constant `follow`): FailLoudWindowDataset computes
refb_labels.nav_command from 15-25 s of future episode poses and emits
nav_cmd / nav_valid / route_target with every item; the model consumes
nav_cmd, the aux CE consumes route_target where nav_valid.
Fallback signals conf_mae + ood_score are computed and logged every step
(emitted on the main run's log cadence); the feature-OOD stats freeze after
--ood-warmup steps (fallback (b)).

Windowing is FAIL-LOUD (2026-07-10 contract-windowing review, tightened per
the REF-B spec): ANY episode too short for window+max_horizon+1, any
frames/actions/poses length misalignment, and any channel mismatch RAISE at
dataset build — no silent skips.

Usage (pod3, only AFTER Sayed's GO — implementation ships untrained):
  python scripts/refb_train.py --data-root /workspace/data \
      --out /workspace/experiments/refb-30k --steps 30000
Smoke (CPU):
  python scripts/refb_train.py --data-root <cache> --out <dir> --steps 10 \
      --batch 8 --smoke --log-every 1 --ood-warmup 5
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
from tanitad.config import base250cam_config
from tanitad.data._contract import EpisodeWindowDataset
from tanitad.data.mixing import load_episode
from tanitad.refs.refb import (RefBModel, param_breakdown, refb_config,
                               refb_smoke_config)
from tanitad.train.train_worldmodel import cosine_lr

# Loss weights. inv-dyn matches the main run's operating point
# (LossConfig.inv_dyn_weight = 0.5 — the fair aux); the BC/waypoint terms are
# co-equal primaries (this stack IS the action decoder); maneuver CE and the
# rev2 route-heading aux CE are 0.5 shaping terms (mirroring the main
# trainer's 0.5 on the tactical loss); the confidence weight only scales the
# fallback head's own learning signal (its gradient cannot reach anything
# else — detached by construction).
ACTION_WEIGHT = 1.0
SEQ_WEIGHT = 1.0
WP_WEIGHT = 1.0
MANEUVER_WEIGHT = 0.5
ROUTE_WEIGHT = 0.5
INV_WEIGHT = 0.5
CONF_WEIGHT = 1.0
ROUTE_CE_CLAMP = 10.0     # cap on the inverse-class-frequency CE weights


# ---- fail-loud windowing (2026-07-10 review, REF-B strict variant) ----------

def build_window_index(episode_lengths: list[int], window: int,
                       max_horizon: int) -> list[tuple[int, int]]:
    """(episode, start) index that RAISES on any too-short episode.

    Derived from the 2026-07-10 contract-windowing-failloud review; the REF-B
    spec tightens warn-and-drop to raise-on-any (no silent skips): a silently
    shrunk, long-episode-biased train set is exactly the defect class this
    guards. Valid-episode indexing is byte-identical to the shared convention
    (`range(T - window - max_horizon)`)."""
    if window < 1 or max_horizon < 0:
        raise ValueError(f"invalid window/max_horizon: window={window} "
                         f"(need >=1), max_horizon={max_horizon} (need >=0)")
    min_len = window + max_horizon + 1
    short = [(e_i, int(T)) for e_i, T in enumerate(episode_lengths)
             if int(T) < min_len]
    if short:
        raise ValueError(
            f"REF-B windowing: {len(short)} of {len(episode_lengths)} "
            f"episode(s) too short for window+max_horizon+1={min_len} "
            f"(window={window}, max_horizon={max_horizon}); offending "
            f"(episode_idx, T): {short[:8]}"
            + (" ..." if len(short) > 8 else "")
            + ". No silent skips — fix the data source or the window/horizon.")
    index: list[tuple[int, int]] = []
    for e_i, T in enumerate(episode_lengths):
        index.extend((e_i, t) for t in range(int(T) - window - max_horizon))
    if not index:                                  # only when episodes == []
        raise ValueError("REF-B windowing: no episodes given")
    return index


class FailLoudWindowDataset(EpisodeWindowDataset):
    """EpisodeWindowDataset with fail-loud construction (same item contract).

    Raises at build time on: any too-short episode, frames/actions/poses
    length misalignment, and frame-channel mismatch vs the encoder config.
    __len__ is inherited; __getitem__ returns the byte-identical window dict
    EXTENDED (rev2) with the derived strategic fields:
        nav_cmd      [] long — refb_labels.nav_command at the last window pose
        nav_valid    [] bool — False when < NAV_MIN_STEPS of future exist
        route_target [] long — the same 3-way derivation, aux-CE target
    """

    def __init__(self, episodes: list, window: int, max_horizon: int,
                 channels: int | None = None):
        for e_i, ep in enumerate(episodes):
            T = ep.frames.shape[0]
            if ep.actions.shape[0] != T or ep.poses.shape[0] != T:
                raise ValueError(
                    f"REF-B windowing: episode {e_i} (id {ep.episode_id}) is "
                    f"misaligned: frames T={T}, actions "
                    f"T={ep.actions.shape[0]}, poses T={ep.poses.shape[0]}")
            if channels is not None and ep.frames.shape[1] != channels:
                raise ValueError(
                    f"REF-B windowing: episode {e_i} (id {ep.episode_id}) has "
                    f"{ep.frames.shape[1]} frame channels; encoder expects "
                    f"{channels}")
        # NOTE: deliberately not calling super().__init__ — it builds the
        # silent index; we install the fail-loud one instead.
        self.window, self.max_horizon = window, max_horizon
        self.episodes = episodes
        self.index = build_window_index([ep.frames.shape[0]
                                         for ep in episodes],
                                        window, max_horizon)

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        cmd, valid = refb_labels.nav_command(self.episodes[e_i].poses,
                                             t + self.window - 1)
        item["nav_cmd"] = torch.tensor(cmd, dtype=torch.long)
        item["nav_valid"] = torch.tensor(valid)
        item["route_target"] = torch.tensor(refb_labels.route_target(cmd),
                                            dtype=torch.long)
        return item


def load_cached_episodes(data_root: str, pattern: str, n: int = 0):
    """Newest cache dir matching ``pattern`` under data_root -> mmap episodes
    (episode-subset bound, the train_worldmodel --data cached convention)."""
    root = Path(data_root)
    dirs = sorted(d for d in root.glob(pattern) if d.is_dir())
    assert dirs, f"no cache dir matching {pattern} under {root}"
    files = sorted(dirs[-1].glob("ep_*.pt"))
    if n:
        files = files[:n]
    assert files, f"no ep_*.pt files in {dirs[-1]}"
    eps = [load_episode(str(p), mmap=True) for p in files]
    return eps, dirs[-1]


# ---- losses ------------------------------------------------------------------

def compute_losses(model: RefBModel, batch: dict, device: str = "cpu") -> dict:
    """One forward pass -> all loss components (tensors, differentiable).

    The confidence target is the REALIZED per-sample waypoint error with a
    fully detached graph (detached predictions AND detached head inputs
    inside the model), so `conf` trains only the confidence head. nav_cmd is
    the per-window DERIVED command (rev2) — never a constant."""
    frames = batch["frames"].to(device)            # [B, W, C, H, W']
    actions = batch["actions"].to(device)          # [B, W, 2]
    fut_actions = batch["future_actions"].to(device)   # [B, Hmax, 2]
    fut_poses = batch["future_poses"].to(device)   # [B, Hmax, 4]
    pose_last = batch["pose_last"].to(device)      # [B, 4]
    nav_cmd = batch["nav_cmd"].to(device)          # [B] long (derived)
    nav_valid = batch["nav_valid"].to(device)      # [B] bool
    route_tgt = batch["route_target"].to(device)   # [B] long

    out = model(frames, nav_cmd=nav_cmd)
    horizons = model.cfg.tactical.waypoint_horizons
    k_seq = model.cfg.operative.action_seq

    # Operative: direct BC at t + direct multi-horizon sequence heads.
    action_seq = out["action_seq"]                 # [B, K, 2]
    loss_action = (action_seq[:, 0] - actions[:, -1]).pow(2).mean()
    loss_seq = (action_seq[:, 1:] - fut_actions[:, :k_seq - 1]).pow(2).mean()

    # Tactical: ego-frame waypoint L2 + maneuver CE (kinematic pseudo-labels).
    wp_tgt = refb_labels.waypoint_targets(pose_last, fut_poses, horizons)
    wp_pred = torch.stack([out["waypoints"][k] for k in horizons], dim=1)
    loss_wp = (wp_pred - wp_tgt).pow(2).mean()
    man_tgt = refb_labels.window_maneuver_labels(
        pose_last, fut_poses, horizon=max(horizons))
    loss_man = F.cross_entropy(out["maneuver_logits"], man_tgt)

    # Strategic aux (rev2): route-heading CE on the SAME nav derivation,
    # valid-masked, inverse-class-frequency weighted (clamped) — comma2k19
    # is highway-dominated, so route_straight would otherwise drown the
    # rare left/right windows.
    n_route = model.cfg.strategic.n_route
    if bool(nav_valid.any()):
        tgt_v = route_tgt[nav_valid]
        counts = torch.bincount(tgt_v, minlength=n_route).float()
        w = (tgt_v.numel() / (n_route * counts.clamp_min(1.0))).clamp(
            max=ROUTE_CE_CLAMP)
        loss_route = F.cross_entropy(out["route_logits"][nav_valid], tgt_v,
                                     weight=w)
        route_acc = (out["route_logits"][nav_valid].argmax(-1)
                     == tgt_v).float().mean()
    else:                       # no route-scale future in this batch
        loss_route = torch.zeros((), device=out["route_logits"].device)
        route_acc = torch.zeros(())

    # Inverse dynamics on consecutive window states (A5 — same as main run).
    states = out["states"]
    a_hat = model.inv_dyn(states[:, -2], states[:, -1])
    loss_inv = (a_hat - actions[:, -2]).pow(2).mean()

    # Fallback (a): regress the realized waypoint error, DETACHED everywhere.
    err = (wp_pred.detach() - wp_tgt).norm(dim=-1).mean(dim=1)   # [B]
    loss_conf = (out["conf_pred"] - err).pow(2).mean()
    conf_mae = (out["conf_pred"] - err).abs().mean()

    loss = (ACTION_WEIGHT * loss_action + SEQ_WEIGHT * loss_seq
            + WP_WEIGHT * loss_wp + MANEUVER_WEIGHT * loss_man
            + ROUTE_WEIGHT * loss_route
            + INV_WEIGHT * loss_inv + CONF_WEIGHT * loss_conf)
    man_acc = (out["maneuver_logits"].argmax(-1) == man_tgt).float().mean()
    return {"loss": loss, "action": loss_action, "seq": loss_seq,
            "wp": loss_wp, "man": loss_man, "route": loss_route,
            "inv": loss_inv, "conf": loss_conf, "conf_mae": conf_mae,
            "man_acc": man_acc, "route_acc": route_acc,
            "nav_valid_frac": nav_valid.float().mean(),
            "nav_follow_frac": (nav_cmd == 0).float().mean(),
            "states": states}


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

    # Matched-everything: optimizer/schedule hyperparameters come from the
    # main run's config object, not from copies.
    main_tr = base250cam_config().train
    lr = args.lr if args.lr is not None else main_tr.lr
    batch = args.batch if args.batch is not None else main_tr.batch_size
    warmup = args.warmup if args.warmup is not None else main_tr.warmup_steps
    save_every = args.save_every if args.save_every is not None \
        else main_tr.save_every
    log_every = args.log_every if args.log_every is not None \
        else main_tr.log_every

    cfg = refb_smoke_config() if args.smoke else refb_config()
    if args.grad_checkpoint:
        cfg.encoder.grad_checkpoint = True
    model = RefBModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=main_tr.betas,
                            weight_decay=main_tr.weight_decay)

    max_h = max(max(cfg.tactical.waypoint_horizons),
                cfg.operative.action_seq - 1)
    train_eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                                args.episodes)
    ds = FailLoudWindowDataset(train_eps, window=cfg.window, max_horizon=max_h,
                               channels=cfg.encoder.in_channels)
    assert len(ds) >= batch, \
        f"only {len(ds)} windows for batch {batch} — add episodes"
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    print(f"[refb] train: {len(train_eps)} episodes / {len(ds)} windows "
          f"from {train_dir}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"arch": "REF-B", "cfg": dataclasses.asdict(cfg), "args": vars(args),
         "optimizer": {"lr": lr, "betas": main_tr.betas,
                       "weight_decay": main_tr.weight_decay,
                       "warmup": warmup, "schedule": "cosine (main run's)"},
         "loss_weights": {"action": ACTION_WEIGHT, "seq": SEQ_WEIGHT,
                          "wp": WP_WEIGHT, "man": MANEUVER_WEIGHT,
                          "route": ROUTE_WEIGHT, "inv": INV_WEIGHT,
                          "conf": CONF_WEIGHT,
                          "route_ce_clamp": ROUTE_CE_CLAMP},
         "param_breakdown": param_breakdown(model)},
        indent=2, default=str), encoding="utf-8")

    # Interruptible-pod resume (OOD stats travel inside model.state_dict).
    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] checkpoint found — resuming at step {step} "
              f"(OOD stats restored, frozen={bool(model.ood.frozen)})",
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

        # Fallback (b): accumulate feature stats during warmup, then freeze.
        z_last = out["states"][:, -1].detach()
        if step < args.ood_warmup:
            model.ood.update(z_last)
        elif not bool(model.ood.frozen):
            model.ood.freeze()
            print(f"[ood] feature stats frozen at step {step} "
                  f"(n={int(model.ood.count)})", flush=True)
        ood_mean = float(model.ood.score(z_last).mean())
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % save_every == 0:
            _save_ckpt(ckpt_path, model, opt, step)

        if step % log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)  # noqa: E731
            last_log = {
                "step": step, "loss": sc(out["loss"]),
                "action": sc(out["action"]), "seq": sc(out["seq"]),
                "wp": sc(out["wp"]), "man": sc(out["man"]),
                "route": sc(out["route"]),
                "inv": sc(out["inv"]), "conf": sc(out["conf"]),
                # fallback signals (spec item 5: log both per step)
                "conf_mae": sc(out["conf_mae"]),
                "ood_score": round(ood_mean, 5),
                "ood_frozen": bool(model.ood.frozen),
                "man_acc": sc(out["man_acc"]),
                "route_acc": sc(out["route_acc"]),
                # nav feed health (rev2): non-constant commands, valid share
                "nav_valid_frac": sc(out["nav_valid_frac"]),
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
                          for k in ("action", "seq", "wp", "man", "route",
                                    "conf_mae", "man_acc", "route_acc",
                                    "nav_valid_frac", "nav_follow_frac")}
        metrics["val"]["ood_score"] = round(
            float(model.ood.score(vout["states"][:, -1]).mean()), 5)
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
                    help="default: the main run's lr (base250cam, 3e-4)")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: the main run's warmup (base250cam, 2000)")
    ap.add_argument("--ood-warmup", type=int, default=2000,
                    help="steps of feature-stat accumulation before the OOD "
                         "monitor freezes (fallback (b))")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute encoder activations (F-5 memory lever)")
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
