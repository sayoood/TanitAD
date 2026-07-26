"""E1c — failure-gated closed-loop SFT with a CORRECTED (held-out) forgetting guard.

This is `e1b_clsft.py` with TWO changes and nothing else:

  1. FRONTIER CHECKPOINTING. The trainable (non-encoder) parameters are written
     at 17 pre-registered steps, so the run yields a TRAJECTORY instead of an
     endpoint. MEASURED on this model: the only non-encoder buffer is the
     constant `decoder.anchors` and the encoder is frozen INCLUDING its BN
     running stats, so `base (+) delta` reconstructs the full model exactly —
     asserted at the end against a full state dict, not assumed. 55 MB/ckpt
     instead of 417 MB.

  2. THE GUARD MOVED TO HELD-OUT DATA. E1b monitored forgetting with the REPLAY
     BRANCH'S OWN LOSS — on the very corpus it replays. That loss FELL
     (1.826 -> 1.613) while held-out open-loop ADE ROSE 41 %. A training-set
     loss is not a generalisation guard. Here, at every checkpoint step, the
     open-loop probe runs on the 44 HELD-OUT episodes (byte-level disjoint from
     mining and replay) and is compared to the base arm on IDENTICAL windows
     with the paired episode-cluster bootstrap.

EVERYTHING ELSE IS BYTE-FOR-BYTE E1b: same base ckpt, same mined buffer (reused,
NOT regenerated), same seed, lr, schedule, batch sizes, lam_cl = lam_replay =
1.0, encoder frozen. lam_replay is deliberately NOT a lever here —
attributability matters more than squeezing the best number out of this run.

The probe SAVES AND RESTORES every RNG state (python / numpy / torch CPU / all
CUDA devices), so the instrumentation cannot perturb the training trajectory:
E1c's step-4000 endpoint stays a reproduction test of E1b's.

Usage:
  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python e1c_clsft.py \
    --base-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
    --buffer /workspace/e1b/mined_buffer.pt \
    --buffer-md5 a32cfe9bfea4b1b5c196d3bb7f71fa5f \
    --parity-dir /workspace/pai_epcache/physicalai-train-e438721ae894 \
    --heldout-dir /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
    --out /workspace/e1c/refc-base-e1c-clsft --steps 4000 --lr 2e-5 \
    --cl-batch 16 --replay-batch 16 --replay-episodes 0 --workers 4 \
    --assert-disjoint-heldout /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a", "/workspace/TanitAD/taniteval", "/workspace/e1c"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ORDER MATTERS (E1b §8 documents this trap): `e1a_horizon` PREPENDS
# /root/taniteval — a stale checkout whose package has NO `ci` module — to
# sys.path at import time.  Bind `taniteval` to the real checkout FIRST and
# assert it, or the estimator import dies (measured: it did).
import taniteval  # noqa: E402
assert taniteval.__file__.startswith("/workspace/TanitAD/taniteval/"), \
    f"taniteval bound to the wrong checkout: {taniteval.__file__}"
from taniteval import ci as CI  # noqa: E402

from e1a_horizon import (W, WP_STEPS, sampling_homography, warp_batch, load_refc)  # noqa: E402
from tanitad.data.mixing import load_episode                    # noqa: E402
from tanitad.train.train_worldmodel import cosine_lr            # noqa: E402
import refc_train                                               # noqa: E402,F401
from refc_train import RouteV21Dataset, compute_losses          # noqa: E402
from driving_diagnostic import gt_ego_waypoints                 # noqa: E402
import e1c_common as C                                          # noqa: E402

TRAJ_W = 1.0            # R2LPL recon (matches refc_train TRAJ_WEIGHT)
CLS_W = 1.0             # R2LPL score CE (matches refc_train ANCHOR_CLS_WEIGHT)
LAW_AHEAD = 5
MAX_H = max(max(WP_STEPS), LAW_AHEAD)   # 20
BOOT, PAIRED = C.make_stat(CI)


# --------------------------------------------------------------------------- #
# CL branch — the mined failure buffer as a dataset (VERBATIM from e1b_clsft)   #
# --------------------------------------------------------------------------- #
class MinedFailureDataset(Dataset):
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
        warped = warp_batch(win[None], H[None])[0]
        return {"frames": warped.clone(),
                "v0": torch.tensor(float(r["v0"])),
                "recovery_target": torch.tensor(r["recovery_target"],
                                                dtype=torch.float32)}


def cl_loss(model, batch, device, steps):
    frames = batch["frames"].to(device)
    v0 = batch["v0"].to(device)
    tgt = batch["recovery_target"].to(device)              # [B,4,2]
    out = model(frames, nav_cmd=None, v0=v0, steps=steps)
    anchors = model.decoder.anchors.to(tgt.dtype)          # [N,4,2]
    dist = ((tgt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))
    a_star = dist.argmin(dim=1)
    loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    b = frames.shape[0]
    recon = out["anchor_traj"][torch.arange(b, device=device), a_star]
    loss_traj = (recon - tgt).abs().mean()
    loss = TRAJ_W * loss_traj + CLS_W * loss_cls
    acc = (out["anchor_logits"].argmax(dim=1) == a_star).float().mean()
    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "anchor_acc": acc}


# --------------------------------------------------------------------------- #
# frontier checkpointing (trainable-only delta)                                 #
# --------------------------------------------------------------------------- #
def _delta_state(model):
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items() if not k.startswith("encoder.")}


def save_delta(out_dir: Path, model, step, base_ckpt):
    p = out_dir / f"delta_step{step:05d}.pt"
    tmp = p.with_suffix(".tmp")
    torch.save({"step": int(step), "base_ckpt": str(base_ckpt),
                "trainable": _delta_state(model)}, tmp)
    tmp.replace(p)
    return p


def _save_resume(path: Path, model, opt, step):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)


def _heldout_ids(d):
    return {str(load_episode(str(p), mmap=True).episode_id)
            for p in sorted(Path(d).glob("ep_*.pt"))}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# THE CORRECTED GUARD — held-out open-loop probe, RNG-isolated                  #
# --------------------------------------------------------------------------- #
def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def rng_restore(s):
    random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2])
    if s[3] is not None:
        torch.cuda.set_rng_state_all(s[3])


def heldout_probe(model, episodes, device, freeze_encoder, stride=8, batch=16):
    """Open loop on the HELD-OUT 44 — never on the replay corpus. RNG-isolated."""
    snap = rng_snapshot()
    was_training = model.training
    model.eval()
    try:
        ol = C.openloop_full(model, episodes, device, W, WP_STEPS,
                             gt_ego_waypoints, stride=stride, batch=batch)
    finally:
        if was_training:
            model.train()
            if freeze_encoder:
                model.encoder.eval()
        rng_restore(snap)
        torch.cuda.empty_cache()
    return ol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--buffer", default="/workspace/e1b/mined_buffer.pt")
    ap.add_argument("--buffer-md5", default="")
    ap.add_argument("--parity-dir",
                    default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--heldout-dir",
                    default="/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6")
    ap.add_argument("--out", default="/workspace/e1c/refc-base-e1c-clsft")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--cl-batch", type=int, default=16)
    ap.add_argument("--replay-batch", type=int, default=16)
    ap.add_argument("--replay-episodes", type=int, default=0)
    ap.add_argument("--lam-cl", type=float, default=1.0)
    ap.add_argument("--lam-replay", type=float, default=1.0)
    ap.add_argument("--freeze-encoder", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume-every", type=int, default=500)
    ap.add_argument("--probe-stride", type=int, default=8)
    ap.add_argument("--probe-batch", type=int, default=16)
    ap.add_argument("--probe-episodes", type=int, default=999,
                    help="cap the held-out guard set (SMOKE ONLY — the real run "
                         "must use all 44, E1a/E1b's exact eval set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assert-disjoint-heldout", default="")
    ap.add_argument("--ckpt-steps", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ck_steps = ([int(x) for x in args.ckpt_steps.split(",") if x]
                if args.ckpt_steps else list(C.CHECKPOINT_STEPS))
    ck_set = set(ck_steps)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CL buffer + md5 + leak guard (E1b's, re-asserted) ------------------
    if args.buffer_md5:
        got = md5_of(args.buffer)
        assert got == args.buffer_md5, \
            f"mined buffer md5 {got} != expected {args.buffer_md5} — refusing"
        print(f"[e1c] mined buffer md5 {got} VERIFIED (reused from E1b, "
              f"not regenerated)", flush=True)
    buf = torch.load(args.buffer, weights_only=False)
    records = buf["records"]
    assert records, f"empty mined buffer {args.buffer}"
    buf_ids = {str(r["episode_id"]) for r in records}
    if args.assert_disjoint_heldout:
        ho = _heldout_ids(args.assert_disjoint_heldout)
        inter = sorted(buf_ids & ho)
        assert not inter, (f"LEAK: {len(inter)} mined episode-ids are in the "
                           f"held-out eval set: {inter[:10]} — refusing to train")
        print(f"[leak-guard] buffer ids={len(buf_ids)} heldout ids={len(ho)} "
              f"intersection=0 -> DISJOINT ok (byte level)", flush=True)

    # ---- held-out episodes for the CORRECTED guard --------------------------
    ho_files = sorted(Path(args.heldout_dir).glob("ep_*.pt"))[:args.probe_episodes]
    ho_eps = [load_episode(str(p), mmap=True) for p in ho_files]
    ho_id_list = [str(e.episode_id) for e in ho_eps]
    assert not (set(ho_id_list) & buf_ids), "held-out probe set overlaps the buffer"
    print(f"[e1c] held-out guard set: {len(ho_eps)} episodes from "
          f"{args.heldout_dir}", flush=True)

    # ---- model: base weights, encoder frozen (identical to E1b) -------------
    model, base_step, cfg = load_refc(args.base_ckpt, args.preset, device)
    for p in model.parameters():
        p.requires_grad_(True)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    model.train()
    if args.freeze_encoder:
        model.encoder.eval()
    diff_steps = cfg.decoder.diffusion_steps
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_nonenc = sum(p.numel() for n, p in model.named_parameters()
                   if not n.startswith("encoder."))
    assert n_nonenc == n_train, (f"trainable-only checkpointing assumes trainable "
                                 f"== non-encoder ({n_train} vs {n_nonenc})")
    nonenc_bufs = [n for n, _ in model.named_buffers()
                   if not n.startswith("encoder.")]
    assert nonenc_bufs == ["decoder.anchors"], \
        f"unexpected non-encoder buffers {nonenc_bufs} — delta save would lose state"
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr)
    print(f"[e1c] base step {base_step} | trainable {n_train:,} frozen "
          f"{n_frozen:,} | encoder_frozen={bool(args.freeze_encoder)} | "
          f"diff_steps={diff_steps} | dev {device}", flush=True)
    print(f"[e1c] frontier checkpoints at steps {ck_steps}", flush=True)

    # ---- loaders (identical to E1b) ----------------------------------------
    cl_ds = MinedFailureDataset(records)
    cl_kw = dict(batch_size=min(args.cl_batch, len(cl_ds)), shuffle=True,
                 drop_last=True)
    if args.workers > 0:
        cl_kw.update(num_workers=args.workers, persistent_workers=True,
                     prefetch_factor=2)
    cl_dl = DataLoader(cl_ds, **cl_kw)

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
    print(f"[e1c] CL states {len(cl_ds)} | replay {len(parity_eps)} eps / "
          f"{len(rp_ds)} windows (v21)", flush=True)

    (out_dir / "config.json").write_text(json.dumps({
        "experiment": "E1c held-out-gated closed-loop SFT (E1b + corrected guard)",
        "changed_vs_e1b": ["frontier checkpointing (trainable-only deltas)",
                           "forgetting guard moved to HELD-OUT open loop, paired "
                           "vs base, at every checkpoint"],
        "unchanged_vs_e1b": ["base ckpt", "mined buffer (reused, md5-verified)",
                             "seed", "lr/warmup/schedule", "batch sizes",
                             "lam_cl", "lam_replay", "encoder frozen",
                             "replay corpus"],
        "base_ckpt": args.base_ckpt, "base_step": base_step,
        "objective": "lam_cl*cl_loss + lam_replay*replay_loss",
        "encoder_frozen": bool(args.freeze_encoder),
        "n_trainable": n_train, "n_frozen": n_frozen,
        "lr": args.lr, "steps": args.steps, "warmup": args.warmup,
        "cl_batch": args.cl_batch, "replay_batch": args.replay_batch,
        "lam_cl": args.lam_cl, "lam_replay": args.lam_replay,
        "diffusion_steps": diff_steps, "seed": args.seed,
        "checkpoint_steps": ck_steps,
        "cl_buffer": args.buffer, "cl_buffer_md5": args.buffer_md5,
        "cl_buffer_meta": buf.get("meta"),
        "parity_dir": args.parity_dir,
        "heldout_guard_dir": args.heldout_dir,
        "heldout_guard_n_episodes": len(ho_eps),
        "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py, B=%d)"
                     % C.B_BOOT,
        "cfg": dataclasses.asdict(cfg),
    }, indent=2, default=str), encoding="utf-8")

    # ---- BASE arrays for the guard, cached ONCE at step 0 -------------------
    t_b = time.time()
    base_ol = heldout_probe(model, ho_eps, device, args.freeze_encoder,
                            args.probe_stride, args.probe_batch)
    print(f"[e1c] BASE held-out open loop: n={len(base_ol['eid'])} "
          f"ADE@2s={base_ol['ade2s'].mean():.4f} "
          f"anchor_acc={base_ol['anchor_acc'].mean():.4f} "
          f"anchor_traj_l1={base_ol['anchor_traj_l1'].mean():.4f} "
          f"({time.time() - t_b:.0f}s)", flush=True)
    np.savez_compressed(out_dir / "base_heldout_openloop.npz",
                        ade2s=base_ol["ade2s"], anchor_acc=base_ol["anchor_acc"],
                        anchor_ce=base_ol["anchor_ce"],
                        anchor_traj_l1=base_ol["anchor_traj_l1"],
                        eid=np.array(base_ol["eid"]),
                        key=np.array(base_ol["key"]))

    gate_f = (out_dir / "heldout_gate.jsonl").open("a")
    logf = (out_dir / "train_log.jsonl").open("a")
    gate_rows, stop_point = [], None

    def gate(step):
        nonlocal stop_point
        t0 = time.time()
        ol = heldout_probe(model, ho_eps, device, args.freeze_encoder,
                           args.probe_stride, args.probe_batch)
        row = C.heldout_gate_row(step, ol, base_ol, BOOT, PAIRED)
        row["probe_s"] = round(time.time() - t0, 1)
        if stop_point is None and step > 0 and not row["gate_ok"]:
            stop_point = step
            row["IS_STOPPING_POINT"] = True
            print(f"[e1c][GATE] *** STOPPING POINT at step {step}: held-out "
                  f"open-loop ADE@2s paired delta "
                  f"{row['ade2s']['paired_delta']['delta']} "
                  f"[{row['ade2s']['paired_delta']['lo']}, "
                  f"{row['ade2s']['paired_delta']['hi']}] SEPARATED WORSE. "
                  f"Run CONTINUES by pre-registration (§4.3) to complete the "
                  f"frontier.", flush=True)
        gate_rows.append(row)
        gate_f.write(json.dumps(row, default=str) + "\n"); gate_f.flush()
        print(f"[e1c][gate] step {step}: ADE@2s {row['ade2s']['base_mean']:.4f} "
              f"-> {row['ade2s']['ft_mean']:.4f} (d="
              f"{row['ade2s']['paired_delta']['delta']}, sep="
              f"{row['ade2s']['paired_delta']['separated']}) | acc "
              f"{row['anchor_acc']['base_mean']:.4f} -> "
              f"{row['anchor_acc']['ft_mean']:.4f} | l1 "
              f"{row['anchor_traj_l1']['base_mean']:.4f} -> "
              f"{row['anchor_traj_l1']['ft_mean']:.4f} | gate_ok="
              f"{row['gate_ok']} ({row['probe_s']}s)", flush=True)

    gate(0)                                        # base row (delta == 0)

    cl_it, rp_it = iter(cl_dl), iter(rp_dl)
    resume_path = out_dir / "ckpt.pt"
    step, t_step = 0, 0.0
    t_train0 = time.time()
    while step < args.steps:
        if step in ck_set:
            save_delta(out_dir, model, step, args.base_ckpt)
            gate(step)
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

        if step > 0 and step % args.resume_every == 0:
            _save_resume(resume_path, model, opt, step)
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

    # ---- final: step == args.steps -----------------------------------------
    save_delta(out_dir, model, args.steps, args.base_ckpt)
    gate(args.steps)
    _save_resume(resume_path, model, opt, args.steps - 1)

    # ---- PROVE the trainable-only delta is lossless -------------------------
    full = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"step": args.steps, "model": full},
               out_dir / "full_state_final.pt")
    base_model, _, _ = load_refc(args.base_ckpt, args.preset, "cpu")
    rec = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    rec.update(torch.load(out_dir / f"delta_step{args.steps:05d}.pt",
                          map_location="cpu")["trainable"])
    bad = {k: float((rec[k].float() - full[k].float()).abs().max())
           for k in full if rec[k].shape == full[k].shape
           and float((rec[k].float() - full[k].float()).abs().max()) > 0}
    missing = sorted(set(full) ^ set(rec))
    overlay_ok = (not bad) and (not missing)
    print(f"[e1c] OVERLAY CHECK base(+)delta == full : {overlay_ok} "
          f"(nonzero-diff keys {len(bad)}, key-set diff {len(missing)})",
          flush=True)
    del base_model

    gate_f.close(); logf.close()
    (out_dir / "metrics.json").write_text(json.dumps(
        {"done": True, "steps": args.steps, "base_step": base_step,
         "n_trainable": n_train, "encoder_frozen": bool(args.freeze_encoder),
         "checkpoint_steps": ck_steps,
         "heldout_gate_stopping_point": stop_point,
         "overlay_lossless": bool(overlay_ok),
         "overlay_bad_keys": sorted(bad)[:20],
         "train_wall_s": round(time.time() - t_train0, 1)},
        indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "steps": args.steps,
                      "stopping_point": stop_point, "out": str(out_dir)}),
          flush=True)
    print("E1C_CLSFT_DONE", flush=True)


if __name__ == "__main__":
    main()
