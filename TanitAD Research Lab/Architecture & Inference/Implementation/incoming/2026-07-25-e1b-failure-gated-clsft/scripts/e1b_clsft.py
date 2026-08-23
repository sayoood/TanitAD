"""E1b step 2 — failure-gated closed-loop SFT of REF-C base (R2LPL-shaped).

Fine-tunes the deployed REF-C base anchored-diffusion planner so its SCORE HEAD
ranks the *return* anchor above the departing ones at failure-adjacent states —
forcing the planner to USE the lateral offset E2a proved it already perceives
(oracle R^2 0.72; 91% of the recovery loss is downstream / objective, not the
encoder). Two interleaved objectives per step:

  CL-SFT (mined failure states, e1b_mine.py buffer): at each recoverable
    pre-failure state the R2LPL target is the logged corridor path in the current
    offset ego frame; supervise anchor-cls CE toward the nearest ("return")
    anchor + traj-recon L1 on that anchor. This is the open-loop anchor block of
    refc_train.compute_losses with the RECOVERY target substituted for GT.

  REPLAY (parity-train open-loop pairs, RouteV21Dataset — the label set base
    trained with): the FULL refc_train.compute_losses (traj+cls+law+route+man),
    interleaved to prevent catastrophic forgetting of open-loop ADE.

  loss = lam_cl * cl_loss + lam_replay * replay_loss

ENCODER IS FROZEN (E2a: perception is not the bottleneck; freezing directly
targets the identified downstream lever, protects open-loop ADE + the perceived
offset, and halves compute). BatchNorm running stats frozen (encoder.eval()).

NO LEAK: the CL buffer is mined ONLY from physicalai-train-e438721ae894 (proved
disjoint from the held-out eval set). --assert-disjoint-heldout re-checks the
buffer's episode-ids against the eval cache at startup and REFUSES to train on
any overlap. Renderer-free. Starts from base weights, fresh Adam, low lr.

Usage (real):
  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python e1b_clsft.py \
    --base-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
    --buffer /workspace/e1b/mined_buffer.pt \
    --parity-dir /workspace/pai_epcache/physicalai-train-e438721ae894 \
    --out /workspace/e1b/refc-base-e1b-clsft --steps 4000 --lr 2e-5 \
    --cl-batch 32 --replay-batch 32 --replay-episodes 0 --workers 4 \
    --assert-disjoint-heldout /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6
Smoke (pod GPU, tiny):
  ... --buffer mined_smoke.pt --steps 30 --cl-batch 8 --replay-batch 8 \
      --replay-episodes 8 --log-every 5 --smoke
"""
from __future__ import annotations
import argparse, dataclasses, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e1a_horizon import (W, WP_STEPS, sampling_homography, warp_batch, load_refc)
from tanitad.data.mixing import load_episode
from tanitad.train.train_worldmodel import cosine_lr
import refc_train
from refc_train import RouteV21Dataset, compute_losses

TRAJ_W = 1.0            # R2LPL recon (matches refc_train TRAJ_WEIGHT)
CLS_W = 1.0             # R2LPL score CE (matches refc_train ANCHOR_CLS_WEIGHT)
LAW_AHEAD = 5
MAX_H = max(max(WP_STEPS), LAW_AHEAD)   # 20


# --------------------------------------------------------------------------- #
# CL branch — the mined failure buffer as a dataset (reconstructs the warp)     #
# --------------------------------------------------------------------------- #
class MinedFailureDataset(Dataset):
    """Each mined record -> the EXACT warped model input the rollout saw + v0 +
    the R2LPL recovery target [4,2]. Episodes are mmap-loaded and cached per
    worker; the warp output is an owned tensor (safe across the worker boundary)."""

    def __init__(self, records: list):
        self.records = records
        self._cache: dict = {}

    def __len__(self):
        return len(self.records)

    def _frames(self, path):
        ep = self._cache.get(path)
        if ep is None:
            ep = load_episode(path, mmap=True)
            self._cache[path] = ep
        return ep.frames

    def __getitem__(self, i):
        r = self.records[i]
        fr = self._frames(r["ep_path"])
        s = r["slice_start"]
        win = fr[s:s + W]
        win = win.float().div(255.0) if win.dtype == torch.uint8 else win.float()
        H = sampling_homography(r["dlat"], math.degrees(r["dpsi"]), 1.5, 0.0)
        warped = warp_batch(win[None], H[None])[0]         # [W,C,H,W'] owned
        return {"frames": warped.clone(),
                "v0": torch.tensor(float(r["v0"])),
                "recovery_target": torch.tensor(r["recovery_target"],
                                                dtype=torch.float32)}


def cl_loss(model, batch, device, steps):
    """R2LPL anchor block with the recovery target (mirrors refc_train's anchor
    assignment + CE + recon exactly, GT -> recovery demonstration)."""
    frames = batch["frames"].to(device)
    v0 = batch["v0"].to(device)
    tgt = batch["recovery_target"].to(device)              # [B,4,2]
    out = model(frames, nav_cmd=None, v0=v0, steps=steps)
    anchors = model.decoder.anchors.to(tgt.dtype)          # [N,4,2]
    dist = ((tgt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))   # [B,N]
    a_star = dist.argmin(dim=1)                            # return anchor
    loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    b = frames.shape[0]
    recon = out["anchor_traj"][torch.arange(b, device=device), a_star]
    loss_traj = (recon - tgt).abs().mean()
    loss = TRAJ_W * loss_traj + CLS_W * loss_cls
    acc = (out["anchor_logits"].argmax(dim=1) == a_star).float().mean()
    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "anchor_acc": acc}


def _save_ckpt(path: Path, model, opt, step):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)
    print(f"[ckpt] saved at step {step}", flush=True)


def _heldout_ids(d):
    return {str(load_episode(str(p), mmap=True).episode_id)
            for p in sorted(Path(d).glob("ep_*.pt"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--buffer", default="/workspace/e1b/mined_buffer.pt")
    ap.add_argument("--parity-dir",
                    default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--out", default="/workspace/e1b/refc-base-e1b-clsft")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--cl-batch", type=int, default=32)
    ap.add_argument("--replay-batch", type=int, default=32)
    ap.add_argument("--replay-episodes", type=int, default=0,
                    help="parity-train episodes for the replay branch (0 = all)")
    ap.add_argument("--lam-cl", type=float, default=1.0)
    ap.add_argument("--lam-replay", type=float, default=1.0)
    ap.add_argument("--freeze-encoder", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assert-disjoint-heldout", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CL buffer + leak guard ----
    buf = torch.load(args.buffer, weights_only=False)
    records = buf["records"]
    assert records, f"empty mined buffer {args.buffer}"
    buf_ids = {str(r["episode_id"]) for r in records}
    if args.assert_disjoint_heldout:
        ho = _heldout_ids(args.assert_disjoint_heldout)
        inter = sorted(buf_ids & ho)
        assert not inter, (f"LEAK: {len(inter)} mined episode-ids are in the "
                           f"held-out eval set {args.assert_disjoint_heldout}: "
                           f"{inter[:10]} — refusing to train")
        print(f"[leak-guard] buffer ids={len(buf_ids)} heldout ids={len(ho)} "
              f"intersection=0 -> DISJOINT ok", flush=True)

    # ---- model: base weights, encoder frozen ----
    model, base_step, cfg = load_refc(args.base_ckpt, args.preset, device)
    for p in model.parameters():
        p.requires_grad_(True)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    model.train()
    if args.freeze_encoder:
        model.encoder.eval()                    # freeze BN running stats
    diff_steps = cfg.decoder.diffusion_steps
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr)
    print(f"[e1b] base step {base_step} | trainable {n_train:,} frozen "
          f"{n_frozen:,} | encoder_frozen={bool(args.freeze_encoder)} | "
          f"diff_steps={diff_steps} | dev {device}", flush=True)

    # ---- CL loader ----
    cl_ds = MinedFailureDataset(records)
    cl_kw = dict(batch_size=min(args.cl_batch, len(cl_ds)), shuffle=True,
                 drop_last=True)
    if args.workers > 0:
        cl_kw.update(num_workers=args.workers, persistent_workers=True,
                     prefetch_factor=2)
    cl_dl = DataLoader(cl_ds, **cl_kw)

    # ---- replay loader (parity-train, v21 labels — base's label set) ----
    pfiles = sorted(Path(args.parity_dir).glob("ep_*.pt"))
    if args.replay_episodes:
        pfiles = pfiles[:args.replay_episodes]
    parity_eps = [load_episode(str(p), mmap=True) for p in pfiles]
    rp_ds = RouteV21Dataset(parity_eps, use_net_dyaw=False, window=cfg.window,
                            max_horizon=MAX_H, channels=cfg.encoder.in_channels)
    rp_kw = dict(batch_size=args.replay_batch, shuffle=True, drop_last=True)
    if args.workers > 0:
        rp_kw.update(num_workers=args.workers, persistent_workers=True,
                     prefetch_factor=2)
    rp_dl = DataLoader(rp_ds, **rp_kw)
    print(f"[e1b] CL states {len(cl_ds)} | replay {len(parity_eps)} eps / "
          f"{len(rp_ds)} windows (v21)", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "experiment": "E1b failure-gated closed-loop SFT (R2LPL-shaped)",
        "base_ckpt": args.base_ckpt, "base_step": base_step,
        "objective": "lam_cl*cl_loss + lam_replay*replay_loss",
        "cl_loss": "anchor-cls CE + traj L1 toward the nearest anchor of the "
                   "logged-corridor recovery demonstration (offset ego frame)",
        "replay_loss": "refc_train.compute_losses (v21 labels) — open-loop "
                       "forgetting guard",
        "encoder_frozen": bool(args.freeze_encoder),
        "n_trainable": n_train, "n_frozen": n_frozen,
        "lr": args.lr, "steps": args.steps, "warmup": args.warmup,
        "cl_batch": args.cl_batch, "replay_batch": args.replay_batch,
        "lam_cl": args.lam_cl, "lam_replay": args.lam_replay,
        "diffusion_steps": diff_steps,
        "cl_buffer": args.buffer, "cl_buffer_meta": buf.get("meta"),
        "parity_dir": args.parity_dir,
        "cfg": dataclasses.asdict(cfg),
    }, indent=2, default=str), encoding="utf-8")

    # ---- resume ----
    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] at step {step}", flush=True)

    cl_it, rp_it = iter(cl_dl), iter(rp_dl)
    logf = (out_dir / "train_log.jsonl").open("a")
    t_step = 0.0
    while step < args.steps:
        cur_lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        t0 = time.perf_counter()
        try:
            clb = next(cl_it)
        except StopIteration:
            cl_it = iter(cl_dl); clb = next(cl_it)
        try:
            rpb = next(rp_it)
        except StopIteration:
            rp_it = iter(rp_dl); rpb = next(rp_it)

        opt.zero_grad(set_to_none=True)
        cl = cl_loss(model, clb, device, diff_steps)
        rp = compute_losses(model, rpb, device, mode="diffusion")
        loss = args.lam_cl * cl["loss"] + args.lam_replay * rp["loss"]
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0))
        opt.step()
        t_step += time.perf_counter() - t0

        if step > 0 and step % args.save_every == 0:
            _save_ckpt(ckpt_path, model, opt, step)
        if step % args.log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)   # noqa: E731
            row = {"step": step, "loss": sc(loss),
                   "cl_loss": sc(cl["loss"]), "cl_traj": sc(cl["traj"]),
                   "cl_cls": sc(cl["cls"]), "cl_anchor_acc": sc(cl["anchor_acc"]),
                   "rp_loss": sc(rp["loss"]), "rp_traj": sc(rp["traj"]),
                   "rp_cls": sc(rp["cls"]), "rp_law": sc(rp["law"]),
                   "rp_anchor_acc": sc(rp["anchor_acc"]),
                   "gnorm": round(gnorm, 4), "lr": round(cur_lr, 8),
                   "step_s": round(t_step, 2)}
            t_step = 0.0
            print(json.dumps(row), flush=True)
            logf.write(json.dumps(row) + "\n"); logf.flush()
        step += 1

    _save_ckpt(ckpt_path, model, opt, step - 1)
    logf.close()
    (out_dir / "metrics.json").write_text(json.dumps(
        {"done": True, "steps": step, "base_step": base_step,
         "n_trainable": n_train, "encoder_frozen": bool(args.freeze_encoder)},
        indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "steps": step, "out": str(out_dir)}),
          flush=True)
    print("E1B_CLSFT_DONE", flush=True)


if __name__ == "__main__":
    main()
