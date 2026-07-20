"""REF-C trainer: behavior-clone Anchored-Diffusion-C (tanitad/refs/refc.py)
from cached episodes.

Mirrors scripts/refb_train.py: the SAME fail-loud dataset (imported, not copied
— FailLoudWindowDataset with the derived nav_cmd/nav_valid/route_target fields),
the same cached-mode ep_*.pt episode dirs, the same atomic-ckpt/resume/jsonl-
step-log/--workers machinery. Divergences are the POINT of the arm:

    optimizer   Adam, lr 1e-4 (the DiffusionDrive/TCP operating point — NOT the
                main run's AdamW/3e-4; batch/warmup/save/log cadence still read
                programmatically from base250cam_config().train)
    decoder     --mode {classifier, diffusion}: classifier is the 0-step anchor-
                selection floor; diffusion refines the winning modes with the
                truncated-denoise steps (cfg.decoder.diffusion_steps). Both train
                the SAME weight set (classifier == diffusion at 0 steps).
    anchors     --anchors <file.pt>: install the FPS anchor vocabulary built by
                scripts/build_refc_anchors.py (else the model's built-in default
                synthetic-FPS anchors are used).
    labels      --labels {v1, v21}: the ROUTE AUX target derivation. ``v1`` is
                what REF-C-XL trained with (refb_labels.route_target(nav_cmd) —
                circular with the fed command AND straight-by-default). ``v21``
                re-derives the route target from refb_labels.route_from_future_v21
                (adaptive horizon, never-straight-by-default, ROUTE_UNKNOWN=3 as
                an out-of-CE-range sentinel that the route CE MASKS OUT — never
                clamped to `straight`). nav_cmd (a model INPUT) keeps the v1
                derivation under both settings, so v21 changes ONE thing.
    losses      traj-recon L1   (1.0)  the GT-assigned anchor's reconstructed
                                       ego-frame trajectory vs the target
                                       [refc1: fixed-distance path checkpoints
                                       via refb_labels.path_targets]
                anchor-cls CE   (1.0)  classify the GT-nearest anchor
                LAW MSE         (0.5)  predicted next pooled latent vs the
                                       no_grad-encoded frames at t+5
                route CE        (0.1)  strategic route-heading aux
                maneuver CE     (0.1)  kinematic maneuver pseudo-label aux
                [refc1] speed CE (0.2) target-speed class (4 bins, [0,30] m/s)

v0 = pose_last[:, 3] is ALWAYS fed (the model applies /10 scaling and the per-
sample ego-dropout p=0.5 internally, training-gated).

Usage (only AFTER Sayed's GO — implementation ships untrained):
  python scripts/refc_train.py --data-root /workspace/data \
      --out /workspace/experiments/refc-30k --steps 30000 --mode diffusion \
      --anchors /workspace/experiments/refc_anchors.pt
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
from tanitad.refs.refc import (N_ROUTE, RefCModel, param_breakdown, refc_config,
                               refc_small_config, refc_smoke_config,
                               refc_xl_config)
from tanitad.train.train_worldmodel import cosine_lr

# Loss weights (module docstring). traj/anchor-cls are the co-equal primaries
# (this stack IS the trajectory decoder), LAW rides at 0.5, route/maneuver are
# 0.1 strategic/tactical aux shaping, speed-class (refc1) at 0.2.
TRAJ_WEIGHT = 1.0
ANCHOR_CLS_WEIGHT = 1.0
LAW_WEIGHT = 0.5
ROUTE_WEIGHT = 0.1
MANEUVER_WEIGHT = 0.1
SPEED_CLS_WEIGHT = 0.2        # refc1 only

TCP_LR = 1e-4                 # Adam lr — the DiffusionDrive/TCP operating point
LAW_AHEAD = 5                 # LAW target: pooled latent 0.5 s (5 steps) ahead
SPEED_AHEAD = 5              # refc1 speed target: v at t+5 (same 0.5 s horizon)

MILESTONES = (5000, 15000, 20000, 30000)   # scaling-study gate protocol (D-030)


# ---- v2.1 route labels (opt-in; --labels v21) --------------------------------

class RouteV21Dataset(FailLoudWindowDataset):
    """FailLoudWindowDataset whose ROUTE AUX TARGET is re-derived by the v2.1
    labeler (``refb_labels.route_from_future_v21``).

    ONLY the route target and its validity mask change. ``nav_cmd`` is a model
    INPUT and keeps the exact v1 derivation REF-C-XL trained with, so a v21 run
    differs from an XL-style run in the route LABEL SET alone — not in what the
    measurement encoder is fed (this mirrors flagship v1.5's `route_v21` mint,
    which likewise left the fed command on the v1 path).

    ``route_target`` carries ``refb_labels.ROUTE_UNKNOWN`` (= 3), deliberately
    OUTSIDE the 3-wide CE class range, whenever ``route_valid`` is False. The
    trainer masks those windows out of the route CE. It is NEVER clamped to
    `straight` — an unmasked cross-entropy raising an index error is the
    intended fail-loud behaviour (the silent straight-clamp is the exact bug the
    v2.1 labeler exists to remove).

    ``use_net_dyaw`` defaults to False per Sayed's 2026-07-20 ruling: a wide
    sweep is ROAD FOLLOWING, not a route event.
    """

    def __init__(self, *a, use_net_dyaw: bool = False, **kw):
        super().__init__(*a, **kw)
        self.use_net_dyaw = bool(use_net_dyaw)

    def __getitem__(self, i: int):
        item = super().__getitem__(i)          # v1 nav_cmd/nav_valid + clones
        e_i, t = self.index[i]
        r = refb_labels.route_from_future_v21(
            self.episodes[e_i].poses, t + self.window - 1,
            use_net_dyaw=self.use_net_dyaw)
        item["route_target"] = torch.tensor(int(r["route"]), dtype=torch.long)
        item["route_valid"] = torch.tensor(bool(r["valid"]))
        return item

    def label_stats(self, n: int = 4000, seed: int = 0) -> dict:
        """Route-label provenance row over ``n`` sampled windows (config.json)."""
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self.index), generator=g)[:min(n, len(self))]
        counts = [0, 0, 0, 0]
        reasons: dict[str, int] = {}
        for i in idx.tolist():
            e_i, t = self.index[i]
            r = refb_labels.route_from_future_v21(
                self.episodes[e_i].poses, t + self.window - 1,
                use_net_dyaw=self.use_net_dyaw)
            counts[int(r["route"])] += 1
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        tot = max(int(idx.numel()), 1)
        return {"n_sampled": tot,
                "route_counts": {"left": counts[0], "straight": counts[1],
                                 "right": counts[2], "UNKNOWN": counts[3]},
                "route_frac": [round(c / tot, 4) for c in counts],
                "valid_frac": round((tot - counts[3]) / tot, 4),
                "reasons": {k: round(v / tot, 4) for k, v in sorted(reasons.items())}}


# ---- losses ------------------------------------------------------------------

def compute_losses(model: RefCModel, batch: dict, device: str = "cpu",
                   mode: str = "diffusion") -> dict:
    """One forward pass -> all loss components (tensors, differentiable).

    Anchor assignment: the GT trajectory target is assigned to its NEAREST anchor
    (flattened L2); anchor-cls CE classifies that index and traj-recon L1
    regresses the reconstructed trajectory FROM the assigned anchor. The LAW
    target is the pooled latent LAW_AHEAD steps past the window, encoded under
    no_grad through the SAME encoder; the prediction path keeps gradients THROUGH
    the decoded trajectory — the point of the aux. ``mode`` picks the decoder's
    inference mode (classifier == 0 steps, diffusion == cfg.diffusion_steps)."""
    frames = batch["frames"].to(device)            # [B, W, C, H, W']
    fut_frames = batch["future_frames"].to(device)  # [B, Hmax, C, H, W']
    fut_poses = batch["future_poses"].to(device)   # [B, Hmax, 4]
    pose_last = batch["pose_last"].to(device)      # [B, 4]
    nav_cmd = batch["nav_cmd"].to(device)          # [B] long (derived)
    nav_valid = batch["nav_valid"].to(device)      # [B] bool
    route_tgt = batch["route_target"].to(device)   # [B] long
    v0 = pose_last[:, 3]                            # [B] current ego speed (t0)

    steps = model.cfg.decoder.diffusion_steps if mode == "diffusion" else 0
    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps)
    cfg = model.cfg
    b = frames.shape[0]

    # Trajectory target: time-indexed ego-frame waypoints, or (refc1) fixed-
    # distance path checkpoints via the arc-length resample.
    if cfg.refc1:
        traj_tgt = refb_labels.path_targets(pose_last, fut_poses, cfg.path_dists)
    else:
        traj_tgt = refb_labels.waypoint_targets(pose_last, fut_poses,
                                                cfg.trajectory.horizons)

    # Anchor assignment: nearest anchor to the GT trajectory (flattened L2).
    anchors = model.decoder.anchors.to(traj_tgt.dtype)     # [N, S, 2]
    dist = ((traj_tgt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))  # [B, N]
    a_star = dist.argmin(dim=1)                            # [B]

    # anchor-cls CE + traj-recon L1 (reconstruction FROM the assigned anchor).
    loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    recon = out["anchor_traj"][torch.arange(b, device=device), a_star]
    loss_traj = (recon - traj_tgt).abs().mean()

    # LAW latent MSE: no_grad target through the same encoder.
    with torch.no_grad():
        law_tgt = model.encode_pooled(fut_frames[:, LAW_AHEAD - 1])
    loss_law = (out["law_pred"] - law_tgt).pow(2).mean()

    # Maneuver CE (kinematic pseudo-labels) + route-heading CE.
    man_tgt = refb_labels.window_maneuver_labels(
        pose_last, fut_poses, horizon=max(cfg.trajectory.horizons))
    loss_man = F.cross_entropy(out["maneuver_logits"], man_tgt)
    if "route_valid" in batch:
        # v2.1 labels: mask on the ROUTE validity. route_target is
        # ROUTE_UNKNOWN (=3, out of CE range) wherever invalid — masked out, and
        # NEVER clamped to `straight`. An UNKNOWN surviving the mask is a
        # labeler contract violation: raise, do not train a wrong class.
        mask = batch["route_valid"].to(device)
        if bool(mask.any()):
            tgt_v = route_tgt[mask]
            if int(tgt_v.max()) >= N_ROUTE:
                raise ValueError(
                    f"ROUTE_UNKNOWN survived the valid mask (max target "
                    f"{int(tgt_v.max())} >= n_route {N_ROUTE}) — the v2.1 "
                    f"contract is route<3 <=> valid=True")
            loss_route = F.cross_entropy(out["route_logits"][mask], tgt_v)
        else:                       # no judgeable window in this batch
            loss_route = torch.zeros((), device=out["pooled"].device)
    else:
        # v1 labels (what REF-C-XL trained with) — byte-identical path,
        # including the fall-back-to-all-windows behaviour.
        mask = nav_valid if bool(nav_valid.any()) else torch.ones_like(nav_valid)
        loss_route = F.cross_entropy(out["route_logits"][mask], route_tgt[mask])
    route_acc = ((out["route_logits"][mask].argmax(-1) == route_tgt[mask])
                 .float().mean() if bool(mask.any())
                 else torch.zeros((), device=out["pooled"].device))

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

    loss = (TRAJ_WEIGHT * loss_traj + ANCHOR_CLS_WEIGHT * loss_cls
            + LAW_WEIGHT * loss_law + ROUTE_WEIGHT * loss_route
            + MANEUVER_WEIGHT * loss_man + SPEED_CLS_WEIGHT * loss_speed_cls)
    anchor_acc = (out["anchor_logits"].argmax(dim=1) == a_star).float().mean()
    man_acc = (out["maneuver_logits"].argmax(dim=1) == man_tgt).float().mean()
    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
            "route": loss_route, "man": loss_man,
            "speed_cls": loss_speed_cls, "speed_mae": speed_mae,
            "anchor_acc": anchor_acc, "man_acc": man_acc,
            "route_acc": route_acc, "route_valid_frac": mask.float().mean(),
            "nav_follow_frac": (nav_cmd == 0).float().mean(),
            "pooled": out["pooled"]}


def _save_ckpt(path: Path, model, opt, step: int) -> None:
    # atomic write: a kill mid-save must not corrupt the resume point
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)
    print(f"[ckpt] saved at step {step}", flush=True)
    # Milestone archive (D-030 scaling study): ckpt.pt is overwritten every
    # save_every, so preserve 5k/15k/20k/30k for the gate protocol. Atomic —
    # a bare copy can leave a truncated file that exists() calls done forever.
    for m in MILESTONES:
        if step >= m:
            arch = path.with_name(f"ckpt_step{m}.pt")
            if not arch.exists():
                from tanitad.train.ckpt_io import atomic_archive
                atomic_archive(path, arch)
                print(f"[ckpt] milestone archived: {arch.name}", flush=True)


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

    # Scale preset: --smoke (tiny CPU) overrides; else --config small/base/xl
    # selects the size-vs-data-scaling study arm (all three share refc.py's
    # anchored-diffusion algorithm; only widths/depths/anchor-vocab differ).
    _presets = {"small": refc_small_config, "base": refc_config,
                "xl": refc_xl_config}
    cfg = refc_smoke_config() if args.smoke else _presets[args.config]()
    cfg.refc1 = bool(args.refc1)       # gated BEFORE build (module presence)
    model = RefCModel(cfg).to(device)
    # Install the FPS anchor vocabulary (else the built-in default anchors).
    if args.anchors:
        anc = torch.load(args.anchors, map_location=device, weights_only=True)
        anc = anc["anchors"] if isinstance(anc, dict) else anc
        model.decoder.load_anchors(anc.to(device))
        print(f"[refc] loaded {tuple(anc.shape)} anchors from {args.anchors}",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    max_h = max(max(cfg.trajectory.horizons), LAW_AHEAD, SPEED_AHEAD)
    train_eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                                args.episodes)
    ds_kw = dict(window=cfg.window, max_horizon=max_h,
                 channels=cfg.encoder.in_channels)
    label_stats: dict | None = None
    if args.labels == "v21":
        ds = RouteV21Dataset(train_eps, use_net_dyaw=args.use_net_dyaw, **ds_kw)
        label_stats = ds.label_stats()
        print(f"[labels] v21 route (use_net_dyaw={args.use_net_dyaw}): "
              f"{json.dumps(label_stats)}", flush=True)
    else:
        ds = FailLoudWindowDataset(train_eps, **ds_kw)
    assert len(ds) >= batch, \
        f"only {len(ds)} windows for batch {batch} — add episodes"
    dl_kw = dict(batch_size=batch, shuffle=True, drop_last=True)
    if getattr(args, "workers", 0) > 0:
        dl_kw.update(num_workers=args.workers, persistent_workers=True,
                     prefetch_factor=2, pin_memory=True)
    dl = DataLoader(ds, **dl_kw)
    print(f"[refc] train: {len(train_eps)} episodes / {len(ds)} windows "
          f"from {train_dir} (mode={args.mode})", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"arch": "REF-C (Anchored-Diffusion-C)",
         "cfg": dataclasses.asdict(cfg), "args": vars(args),
         "optimizer": {"kind": "Adam (DiffusionDrive/TCP)", "lr": lr,
                       "warmup": warmup, "schedule": "cosine (main run's)"},
         "loss_weights": {"traj": TRAJ_WEIGHT, "cls": ANCHOR_CLS_WEIGHT,
                          "law": LAW_WEIGHT, "route": ROUTE_WEIGHT,
                          "man": MANEUVER_WEIGHT, "speed_cls": SPEED_CLS_WEIGHT},
         # Label provenance — the artifact must describe its own labels.
         "labels": {
             "label_set": args.labels,
             "route_derivation": ("refb_labels.route_from_future_v21"
                                  if args.labels == "v21"
                                  else "refb_labels.route_target(nav_command)"),
             "use_net_dyaw": (bool(args.use_net_dyaw)
                              if args.labels == "v21" else None),
             "nav_cmd_derivation": "refb_labels.nav_command (v1, unchanged)",
             "route_unknown_handling": ("masked out of the route CE "
                                        "(ROUTE_UNKNOWN=3, never clamped)"
                                        if args.labels == "v21" else "n/a"),
             "train_label_stats": label_stats},
         "data": {"cache_dir": str(train_dir), "n_episodes": len(train_eps),
                  "n_windows": len(ds)},
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
        out = compute_losses(model, batch_d, device, mode=args.mode)
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
                "traj": sc(out["traj"]), "cls": sc(out["cls"]),
                "law": sc(out["law"]), "route": sc(out["route"]),
                "man": sc(out["man"]), "speed_cls": sc(out["speed_cls"]),
                "speed_mae": sc(out["speed_mae"]),
                "anchor_acc": sc(out["anchor_acc"]), "man_acc": sc(out["man_acc"]),
                "route_acc": sc(out["route_acc"]),
                "route_valid_frac": sc(out["route_valid_frac"]),
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
        vkw = dict(window=cfg.window, max_horizon=max_h,
                   channels=cfg.encoder.in_channels)
        vds = (RouteV21Dataset(val_eps, use_net_dyaw=args.use_net_dyaw, **vkw)
               if args.labels == "v21" else FailLoudWindowDataset(val_eps, **vkw))
        model.eval()
        with torch.no_grad():
            vb = torch.utils.data.default_collate(
                [vds[i] for i in range(min(16, len(vds)))])
            vout = compute_losses(model, vb, device, mode=args.mode)
        metrics["val"] = {k: round(float(vout[k]), 5)
                          for k in ("traj", "cls", "law", "route", "man",
                                    "speed_cls", "speed_mae", "anchor_acc",
                                    "man_acc", "route_acc", "route_valid_frac",
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
    ap.add_argument("--mode", choices=("classifier", "diffusion"),
                    default="diffusion",
                    help="decoder mode: classifier = 0-step anchor selection "
                         "floor; diffusion = truncated-denoise refinement")
    ap.add_argument("--config", choices=("small", "base", "xl"), default="base",
                    help="scale preset (ignored under --smoke): small ~55M "
                         "(64 anchors) / base ~110M (128) / xl ~260M (256)")
    ap.add_argument("--anchors", default=None,
                    help="FPS anchor vocabulary .pt (build_refc_anchors.py); "
                         "default = the model's built-in synthetic-FPS anchors")
    ap.add_argument("--labels", choices=("v1", "v21"), default="v1",
                    help="route AUX target derivation: v1 = what REF-C-XL "
                         "trained with (route_target(nav_cmd), straight-by-"
                         "default); v21 = refb_labels.route_from_future_v21 "
                         "(adaptive horizon, ROUTE_UNKNOWN masked out of the CE)")
    ap.add_argument("--use-net-dyaw", action="store_true",
                    help="v21 only: count a >=45 deg net heading change as a "
                         "route turn. OFF per Sayed 2026-07-20 (a wide sweep is "
                         "ROAD FOLLOWING)")
    ap.add_argument("--batch", type=int, default=None,
                    help="default: the main run's batch (base250cam)")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"default: the TCP/DiffusionDrive Adam lr ({TCP_LR})")
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
