"""P2b — decoder-only IN-ENVELOPE RECOVERY-AUGMENTATION fine-tune of REF-C base.

Changes EXACTLY ONE thing vs the base REF-C objective (refc_train.compute_losses):
the input window is warped by a per-sample in-envelope (dlat, dpsi) and the traj/
anchor targets become the RECOVERY trajectory from that perturbed pose (perturb.py).
Everything else — the frozen 90 M encoder, the anchored-diffusion decoder algorithm,
the anchor vocabulary (loaded strict from the base ckpt buffer), the diffusion noise,
ego-dropout, Adam/lr — is base REF-C. Trainable surface = the decoder only (~Gate-1's
8.6 M), so a frozen-encoder forward is cheap and the encoder can never be degraded.

WHY THIS IS THE DATA-EFFICIENT LEVER (not the ruled-out free floor, not the held
data-bound Gate-1 FT): recovery examples are synthesised ANALYTICALLY from EVERY
window via geometry — not collected from ~15 scarce real junction scenes (Gate-1,
memorised: held-out Δ≈0) and not from a self-referential WM rollout (DAgger, HURT).
The perturbation magnitude is bounded to the P1-MEASURED low-OOD envelope so the
frozen encoder still sees in-distribution input (unlike NuRec's 3.2× OOD).

OUTPUT: a REF-C-shaped ckpt {model, step} + config.json (base cfg copied from the
base run) so the abe82f1f instrument loads it verbatim via
  lowood_lanekeep.py --refc-ckpt <this out>/ckpt.pt --refc-preset base
No instrument code is modified; the proof reuses it.

Run (eval pod, gpu_lock refc-cl-improve, AFTER abe82f1f frees eval):
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    python3 recovery_aug_ft.py \
      --refc-ckpt /root/models/refc-base-30k/ckpt.pt \
      --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 \
      --ft-slice 0:28 --steps 1500 --out /workspace/refc-recovery-ft
CPU smoke (no ckpt; tiny model, verifies wiring + frozen-encoder-forward parity):
  python3 recovery_aug_ft.py --smoke --out /tmp/refc_ft_smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_ROOTS = ["/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
          "/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts"]
for _up in (5, 6):                       # local repo layout depth (crash-safe)
    try:
        _r = _HERE.parents[_up]
        _ROOTS += [str(_r / "stack"), str(_r / "stack" / "scripts")]
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import perturb  # noqa: E402  (same dir)
from refb_labels import waypoint_targets  # noqa: E402
from tanitad.refs.refc import (NAV_COMMANDS, RefCModel, param_breakdown,  # noqa: E402
                               refc_config, refc_small_config, refc_smoke_config,
                               refc_xl_config)

_PRESETS = {"base": refc_config, "small": refc_small_config, "xl": refc_xl_config}


# --------------------------------------------------------------------------- #
# Data — minimal windowed dataset over the SAME load_episode objects the        #
# instrument consumes (so FT and eval read identical episodes/poses).           #
# --------------------------------------------------------------------------- #
def _load_corpus(val_dir, sl):
    """Keep frames RAW (uint8/mmap) — float ONLY the W-frame window in _batch
    (O(W), never O(T)); pre-float the small pose tensors once."""
    from tanitad.data.mixing import load_episode
    eps = sorted(Path(val_dir).glob("ep_*.pt"))
    a, b = (int(x) if x else None for x in sl.split(":"))
    eps = eps[a:b]
    frames, poses, names = [], [], []
    for p in eps:
        e = load_episode(str(p), mmap=True)
        frames.append(e.frames)                            # raw (uint8/mmap)
        poses.append(e.poses.float())
        names.append(p.name)
    return frames, poses, names


def _windows(poses_list, W, max_h, stride):
    """Yield (ep_i, t0) window starts with W history + max_h future available."""
    idx = []
    for ei, poses in enumerate(poses_list):
        T = poses.shape[0]
        idx += [(ei, t0) for t0 in range(0, T - W - max_h, stride)]
    return idx


def _batch(frames_list, poses_list, idx_chunk, W, max_h):
    fr_out, pose_last, fut = [], [], []
    for ei, t0 in idx_chunk:
        last = t0 + W - 1
        fw = frames_list[ei][t0:t0 + W]                    # slice RAW [W,C,H,W']
        fw = fw.float().div(255.0) if fw.dtype == torch.uint8 else fw.float()
        fr_out.append(fw)
        pose_last.append(poses_list[ei][last])             # [4]
        fut.append(poses_list[ei][last + 1:last + 1 + max_h])   # [max_h,4]
    return (torch.stack(fr_out), torch.stack(pose_last), torch.stack(fut))


# --------------------------------------------------------------------------- #
# Frozen-encoder forward (mirrors RefCModel.forward hierarchy path; encoder +   #
# all non-decoder modules under no_grad; ONLY the decoder carries gradients).    #
# Parity vs model.forward is asserted in the smoke (must match in eval mode).    #
# --------------------------------------------------------------------------- #
def _encode(model, frames, encoder_grad=False):
    """Encoder forward. encoder_grad=False (decoder-only): no_grad + detach (cheap,
    byte-identical to the frozen path). encoder_grad=True (RefcCL encoder-in-loop):
    grads flow to the unfrozen encoder blocks. BN stays in eval mode (frozen
    running stats) either way -- set by model.encoder.eval() in the caller."""
    import contextlib
    b, w = frames.shape[:2]
    cm = contextlib.nullcontext() if encoder_grad else torch.no_grad()
    with cm:
        fmap_all, pooled_all = model.encoder(frames.reshape(b * w, *frames.shape[2:]))
        pooled_seq = pooled_all.reshape(b, w, -1)
        pooled = pooled_seq[:, -1]
        fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
        ctx = model.strategic(pooled_seq) if model.cfg.hierarchy else None
        if model.cfg.graft_imagination:
            fmap, _ = model.imagination(fmap)
        man = model.maneuver_head(pooled)
    if not encoder_grad:
        fmap, pooled = fmap.detach(), pooled.detach()
        ctx = ctx.detach() if ctx is not None else None
        man = man.detach()
    return fmap, pooled, ctx, man


def _decode(model, fmap, pooled, ctx, man, v0, nav, steps, training):
    b = fmap.shape[0]
    nav_oh = F.one_hot(nav, len(NAV_COMMANDS)).to(pooled.dtype)
    v = (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
    if training and model.cfg.ego_dropout > 0:
        keep = (torch.rand(b, 1, device=v.device) >= model.cfg.ego_dropout).to(v.dtype)
        v = v * keep
    with torch.no_grad():
        m = model.measurement(torch.cat([v, nav_oh], dim=-1))   # measurement frozen
    return model.decoder(fmap, m, ctx=ctx, maneuver_logits=man, steps=steps)


def _plan(model, frames, v0, nav, steps, training, encoder_grad=False):
    fmap, pooled, ctx, man = _encode(model, frames, encoder_grad)
    return _decode(model, fmap, pooled, ctx, man, v0, nav, steps, training)


# --------------------------------------------------------------------------- #
# Loss — base REF-C traj L1 + anchor CE on the RECOVERY target, + λ_dev trust    #
# region on CLEAN windows toward the base plan (the Gate-1 high-deviation fix).  #
# --------------------------------------------------------------------------- #
def ft_step(model, base_model, batch, envcfg, horizons, steps, lambda_dev,
            device, gen, lambda_prog=0.0, encoder_grad=False):
    frames, pose_last, fut = (t.to(device) for t in batch)
    b = frames.shape[0]
    dlat, dpsi = perturb.sample_perturbation(b, envcfg, gen)
    dlat, dpsi = dlat.to(device), dpsi.to(device)
    warped = perturb.warp_windows(frames, dlat, dpsi)
    tgt = perturb.recovery_targets(pose_last, fut, horizons, dlat, dpsi,
                                   waypoint_targets)                 # [B,S,2]
    nav = torch.zeros(b, dtype=torch.long, device=device)           # follow
    v0 = pose_last[:, 3]

    dec = _plan(model, warped, v0, nav, steps, training=True, encoder_grad=encoder_grad)
    anchors = model.decoder.anchors.to(tgt.dtype)
    dist = ((tgt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))   # [B,N]
    a_star = dist.argmin(dim=1)
    loss_cls = F.cross_entropy(dec["anchor_logits"], a_star)
    recon = dec["anchor_traj"][torch.arange(b, device=device), a_star]
    loss_traj = (recon - tgt).abs().mean()

    # λ_dev: on CLEAN windows (dlat==dpsi==0) keep the FT plan == the base plan on
    # the ORIGINAL frame -> on-path behaviour cannot drift (prevents the Gate-1
    # "scenes that PASSED go newly off-road" high-deviation side-effect).
    loss_dev = torch.zeros((), device=device)
    clean = (dlat == 0) & (dpsi == 0)
    if lambda_dev > 0 and bool(clean.any()):
        with torch.no_grad():
            base_dec = _plan(base_model, frames, v0, nav, steps, training=False)
        loss_dev = (dec["traj"][clean] - base_dec["traj"][clean]).abs().mean()

    # progress / return-to-GT-speed preservation: extra L1 weight on the FORWARD
    # (x) component of the recovery traj -> protects longitudinal tracking (the
    # MEASURED locus of the ADE cost) WITHOUT resisting the lateral (y) recovery.
    loss_prog = (recon[..., 0] - tgt[..., 0]).abs().mean()
    loss = loss_traj + loss_cls + lambda_dev * loss_dev + lambda_prog * loss_prog
    return loss, {"traj": float(loss_traj.detach()), "cls": float(loss_cls.detach()),
                  "dev": float(loss_dev.detach()), "prog": float(loss_prog.detach()),
                  "frac_clean": float(clean.float().mean()),
                  "mean_recovery_m": float(tgt[~clean, 0].abs().mean())
                  if bool((~clean).any()) else 0.0}


def _setup_trainable(model, scope, unfreeze_encoder_stages=0):
    """Freeze all, then unfreeze the decoder (+scope) and the LAST-k encoder
    ResNet stages (RefcCL). Returns (enc_params, head_params) so the optimizer can
    use a low lr_encoder for the unfrozen encoder blocks (the warm-trunk lesson).
    unfreeze_encoder_stages=0 => decoder-only (byte-identical to the frozen path)."""
    for p in model.parameters():
        p.requires_grad_(False)
    train_mods = [model.decoder]
    if scope == "decoder+meas+strat":
        train_mods += [model.measurement]
        if model.cfg.hierarchy:
            train_mods.append(model.strategic)
    for mod in train_mods:
        for p in mod.parameters():
            p.requires_grad_(True)
    head_params = [p for mod in train_mods for p in mod.parameters()]
    enc_params = []
    if unfreeze_encoder_stages > 0:
        stages = list(model.encoder.stages)               # 4 ResNet stages
        k = min(unfreeze_encoder_stages, len(stages))
        for st in stages[-k:]:                             # deepest k stages
            for p in st.parameters():
                p.requires_grad_(True)
                enc_params.append(p)
    return enc_params, head_params


def _load_base(ckpt, preset, device):
    cfg = _PRESETS[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        d = json.loads(cj.read_text()).get("cfg", {})
        perturb_cfg_apply(cfg, d)
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])                     # STRICT
    return model.to(device), cfg, int(ck.get("step", -1))


def perturb_cfg_apply(cfg, d):
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            perturb_cfg_apply(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refc-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base", choices=("base", "small", "xl"))
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--ft-slice", default="0:28",
                    help="python slice over sorted ep_*.pt used for the FT "
                         "(episode-DISJOINT from the held-out eval slice)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)       # base REF-C Adam lr
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lambda-dev", type=float, default=0.5)
    ap.add_argument("--lambda-prog", type=float, default=0.0,
                    help="return-to-GT-speed/progress term: extra L1 on the "
                         "forward (x) recovery component (protects longitudinal)")
    ap.add_argument("--lat-max", type=float, default=1.75)
    ap.add_argument("--yaw-max-deg", type=float, default=5.0)
    ap.add_argument("--clean-frac", type=float, default=0.30)
    ap.add_argument("--train-scope", default="decoder",
                    choices=("decoder", "decoder+meas+strat"))
    ap.add_argument("--unfreeze-encoder-stages", type=int, default=0,
                    help="RefcCL: unfreeze the last-k ResNet encoder stages "
                         "(0 = decoder-only frozen encoder). BN stays frozen.")
    ap.add_argument("--lr-encoder", type=float, default=5e-6,
                    help="RefcCL: low lr for the unfrozen encoder blocks "
                         "(decoder trains at --lr)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # geometry self-check (MEASURED discipline) before spending any step
    vid = perturb.validate_identity(waypoint_targets)
    assert vid["ok"], f"perturb geometry identity FAILED: {vid}"

    if args.smoke:
        cfg = refc_smoke_config()
        model = RefCModel(cfg).to(device)
        base_model = RefCModel(cfg).to(device)
        base_model.load_state_dict(model.state_dict())
        base_model.eval()
        horizons = cfg.trajectory.horizons
        steps = cfg.decoder.diffusion_steps
        # parity: _plan(eval) must equal model.forward(eval) exactly
        model.eval()
        fr = torch.rand(3, cfg.window, cfg.encoder.in_channels,
                        cfg.encoder.image_size, cfg.encoder.image_size, device=device)
        v0 = torch.rand(3, device=device) * 10
        nav = torch.zeros(3, dtype=torch.long, device=device)
        with torch.no_grad():
            ref = model(fr, nav_cmd=nav, v0=v0, steps=steps)["traj"]
            got = _plan(model, fr, v0, nav, steps, training=False)["traj"]
        parity = float((ref - got).abs().max())
        print(f"[smoke] frozen-forward vs model.forward parity maxabs={parity:.2e}")
        assert parity < 1e-4, f"frozen-encoder forward diverges: {parity}"
        enc_params, head_params = _setup_trainable(
            model, args.train_scope, args.unfreeze_encoder_stages)
        params = head_params + enc_params
        encoder_grad = len(enc_params) > 0
        groups = [{"params": head_params, "lr": args.lr}]
        if enc_params:
            groups.append({"params": enc_params, "lr": args.lr_encoder})
        opt = torch.optim.Adam(groups)
        print(f"[smoke] trainable: head {sum(p.numel() for p in head_params):,} + "
              f"enc {sum(p.numel() for p in enc_params):,} "
              f"(unfreeze_stages={args.unfreeze_encoder_stages}, "
              f"encoder_grad={encoder_grad})")
        envcfg = perturb.EnvelopeCfg(args.lat_max, args.yaw_max_deg, args.clean_frac)
        poses_list = [torch.randn(60, 4) for _ in range(3)]
        for p in poses_list:
            p[:, 3] = p[:, 3].abs() * 10
        frames_list = [torch.rand(60, cfg.encoder.in_channels,
                                  cfg.encoder.image_size, cfg.encoder.image_size)
                       for _ in range(3)]
        model.train(); model.encoder.eval()
        max_h = max(horizons)
        wins = _windows(poses_list, cfg.window, max_h, args.stride)
        for step in range(5):
            ch = [wins[(step * args.batch + i) % len(wins)] for i in range(args.batch)]
            batch = _batch(frames_list, poses_list, ch, cfg.window, max_h)
            opt.zero_grad(set_to_none=True)
            loss, comp = ft_step(model, base_model, batch, envcfg, horizons,
                                 steps, args.lambda_dev, device, gen,
                                 lambda_prog=args.lambda_prog,
                                 encoder_grad=encoder_grad)
            loss.backward(); opt.step()
            print(f"[smoke] step {step} loss={float(loss):.4f} {comp}")
        assert model.decoder.offset_head.weight.grad is not None, "decoder no grad"
        if args.unfreeze_encoder_stages > 0:
            last = list(model.encoder.stages)[-1]
            first = list(model.encoder.stages)[0]
            assert any(p.grad is not None for p in last.parameters()), \
                "RefcCL: last encoder stage got NO grad"
            assert all(p.grad is None for p in first.parameters()), \
                "RefcCL: a FROZEN (early) encoder stage got grad"
            print("[smoke] RefcCL: last-stage grads present, early stages frozen OK")
        else:
            assert all(p.grad is None for p in model.encoder.parameters()), \
                "encoder received grad — not frozen"
        print("RECOVERY_AUG_FT_SMOKE_OK")
        return

    model, cfg, base_step = _load_base(args.refc_ckpt, args.refc_preset, device)
    base_model, _, _ = _load_base(args.refc_ckpt, args.refc_preset, device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    horizons = cfg.trajectory.horizons
    steps = cfg.decoder.diffusion_steps
    enc_params, head_params = _setup_trainable(
        model, args.train_scope, args.unfreeze_encoder_stages)
    params = head_params + enc_params
    encoder_grad = len(enc_params) > 0
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_encoder})
    opt = torch.optim.Adam(groups)
    n_tr = sum(p.numel() for p in params)
    n_enc = sum(p.numel() for p in enc_params)
    envcfg = perturb.EnvelopeCfg(args.lat_max, args.yaw_max_deg, args.clean_frac)

    frames_list, poses_list, ep_names = _load_corpus(args.val_dir, args.ft_slice)
    max_h = max(horizons)
    wins = _windows(poses_list, cfg.window, max_h, args.stride)
    g_sh = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(wins), generator=g_sh).tolist()
    arm = "RefcCL (encoder-in-loop)" if encoder_grad else "recovery-aug (decoder-only)"
    print(f"[ft] {arm} | base step {base_step} | FT eps {args.ft_slice} "
          f"({len(poses_list)}) | {len(wins)} windows | head {n_tr - n_enc:,} + "
          f"enc {n_enc:,} (unfreeze_stages={args.unfreeze_encoder_stages}, "
          f"lr_enc={args.lr_encoder}) | envelope lat<={args.lat_max}m "
          f"yaw<={args.yaw_max_deg}deg clean={args.clean_frac} "
          f"lambda_dev={args.lambda_dev} lambda_prog={args.lambda_prog}", flush=True)

    # config.json so the instrument builds the right shapes + strict-loads
    src_cfg = Path(args.refc_ckpt).parent / "config.json"
    if src_cfg.exists():
        shutil.copy(src_cfg, out / "config.json")
    else:
        (out / "config.json").write_text(json.dumps(
            {"cfg": dataclasses.asdict(cfg)}, indent=2, default=str))

    model.train(); model.encoder.eval()
    t0 = time.time(); ptr = 0
    for step in range(args.steps):
        ch = []
        for _ in range(args.batch):
            ch.append(wins[order[ptr % len(order)]]); ptr += 1
        batch = _batch(frames_list, poses_list, ch, cfg.window, max_h)
        opt.zero_grad(set_to_none=True)
        loss, comp = ft_step(model, base_model, batch, envcfg, horizons, steps,
                             args.lambda_dev, device, gen,
                             lambda_prog=args.lambda_prog,
                             encoder_grad=encoder_grad)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4),
                              **{k: round(v, 4) for k, v in comp.items()},
                              "gnorm": round(gnorm, 3),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    torch.save({"model": model.state_dict(), "step": args.steps,
                "base_step": base_step}, out / "ckpt.pt")
    (out / "ft_provenance.json").write_text(json.dumps({
        "arm": arm,
        "lever": "in-envelope geometric recovery augmentation" + (
            " + encoder-in-the-loop (RefcCL)" if encoder_grad else " (decoder-only)"),
        "base_ckpt": args.refc_ckpt, "base_step": base_step,
        "ft_episode_slice": args.ft_slice, "ft_episode_names": ep_names,
        "n_windows": len(wins), "steps": args.steps, "lr_head": args.lr,
        "unfreeze_encoder_stages": args.unfreeze_encoder_stages,
        "lr_encoder": args.lr_encoder, "n_encoder_trainable": n_enc,
        "batch": args.batch, "train_scope": args.train_scope,
        "trainable_params": n_tr, "param_breakdown": param_breakdown(model),
        "envelope": {"lat_max_m": args.lat_max, "yaw_max_deg": args.yaw_max_deg,
                     "clean_frac": args.clean_frac,
                     "bound_source": "P1 lower-ood-closedloop-source "
                                     "P1_DECISION_GRADE_FINDINGS.md §1.2/1.3"},
        "lambda_dev": args.lambda_dev, "lambda_prog": args.lambda_prog,
        "geometry_selfcheck": vid,
        "evidence_class": "MEASURED (this FT run)"},
        indent=2, default=str))
    print(f"[ft] wrote {out}/ckpt.pt", flush=True)
    print("RECOVERY_AUG_FT_DONE", flush=True)


if __name__ == "__main__":
    main()
