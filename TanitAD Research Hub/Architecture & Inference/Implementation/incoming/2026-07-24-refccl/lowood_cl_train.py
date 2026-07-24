"""LOWOOD-CL-TRAIN — closed-loop-CONSISTENT recovery on the real-footage low-OOD source.

The decisive renderer-free test (Research/2026-07-24-low-ood-closedloop-renderer.md §7).
D2 + RefcCL proved the departure<->ADE trade is intrinsic to the SINGLE-STEP SYNTHETIC
recovery objective (random i.i.d. perturbations). This trains on the policy's OWN
ACCUMULATED ON-POLICY DRIFT instead — RoaD/CAT-K:

  COLLECT (no grad): roll the current REF-C forward on the low-OOD harness (real frame
    arc-length re-indexed + homography-warped by the ON-POLICY residual (dlat,dpsi) —
    the SAME 1.05x-OOD operator the instrument scores on). Record, at each visited
    (compounding-drift) tick: the shown window, the on-policy (dlat,dpsi), and the RoaD
    target = the recorded path 0.5-2s ahead expressed in the drifted ego frame.
  TRAIN: fine-tune the decoder on those on-policy states toward the RoaD target
    (traj-L1 + anchor-CE), CAT-K-filtered to the MEASURED <=1.16x envelope (feasible
    recovery only), + lambda_dev on low-drift states (preserve on-path).
  DAgger rounds: re-collect with the updated policy each round.

Non-self-referential (real frames, not the WM's imagination -> avoids the MEASURED
DAGGER_HURTS trap). Decoder-only first (WM-safe); RefcCL encoder-in-loop optional.
Reuses recovery_aug_ft.py (model load / forward / optimizer / canary-safe unfreeze).

Run (eval, gpu_lock lowood-cl-train, after checking eval free):
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts:/root/taniteval \
    python3 lowood_cl_train.py --refc-ckpt /root/models/refc-base-30k/ckpt.pt \
      --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 --ft-slice 0:28 \
      --dagger-rounds 3 --steps-per-round 250 --out /workspace/refc-lowood-cl
CPU smoke: python3 lowood_cl_train.py --smoke --out /tmp/lct_smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_ROOTS = ["/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
          "/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
          "/root/taniteval", "/root/TanitAD/taniteval",
          str(_HERE.parent), str(_HERE.parents[1] / "2026-07-23-refc-planner-closedloop")]
for _up in (5, 6):
    try:
        r = _HERE.parents[_up]
        _ROOTS += [str(r / "stack"), str(r / "stack" / "scripts"), str(r / "taniteval"),
                   str(r / "TanitAD Research Hub" / "Architecture & Inference" /
                       "Implementation" / "incoming" / "2026-07-23-refc-planner-closedloop")]
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import perturb  # noqa: E402
import recovery_aug_ft as raf  # noqa: E402  (reuse model/forward/optimizer/unfreeze)
from refb_labels import ego_frame, waypoint_targets  # noqa: E402
from tanitad.refs.refc import NAV_COMMANDS, RefCModel, param_breakdown  # noqa: E402

# harness rollout constants (lowood_lanekeep.py verbatim)
W = 8
K = 20                       # rollout length (2 s @ 10 Hz)
DT = 0.1
WHEELBASE = 2.7
LOOKAHEAD_STEP = 5
LD2_FLOOR = 0.25
STEER_CLAMP = 0.05
ACCEL_CLAMP = 3.0
SPEED_TC = 0.5
HORIZONS = (5, 10, 15, 20)
ENV_LAT = 3.0                # CAT-K feasibility: drop on-policy states beyond the
ENV_YAW_DEG = 12.0           # MEASURED <=1.16x envelope (P1 §1.3); flag the fraction


def wp_to_control(w_look, v):
    x, y = w_look[:, 0], w_look[:, 1]
    ld2 = (x * x + y * y).clamp_min(LD2_FLOOR)
    kappa = 2.0 * y / ld2
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    v_target = x / (LOOKAHEAD_STEP * DT)
    accel = ((v_target - v) / SPEED_TC).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    return steer, accel


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@torch.no_grad()
def collect_onpolicy(model, frames_list, poses_list, steps, device, stride=8,
                     batch=16, max_windows=None):
    """Roll the CURRENT REF-C forward on the low-OOD harness; record per visited tick
    (recorded-window index s, on-policy dlat/dpsi, RoaD recovery target, v0). Returns a
    dict of stacked tensors + the envelope-inside fraction (CAT-K feasibility)."""
    rec = {k: [] for k in ("ep", "s", "dlat", "dpsi", "tgt", "v0")}
    n_done = 0
    for ei, (fr, poses) in enumerate(zip(frames_list, poses_list)):
        T = poses.shape[0]
        starts = list(range(0, T - W - K - max(HORIZONS), stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            t0 = torch.tensor(ch)
            last = t0 + W - 1
            # reference path = recorded future, long enough for mstar+horizon lookahead
            HL = K + max(HORIZONS)
            idx = last[:, None] + torch.arange(0, HL + 1)[None]
            Pxy = poses[idx][..., :2]                       # [b, HL+1, 2]
            Pyaw = poses[idx][..., 2]
            ex = poses[last, 0].clone().to(device); ey = poses[last, 1].clone().to(device)
            eyaw = poses[last, 2].clone().to(device); ev = poses[last, 3].clone().to(device)
            Pxy = Pxy.to(device); Pyaw = Pyaw.to(device)
            ar = torch.arange(b, device=device)
            hz = torch.tensor(HORIZONS, device=device)
            for k in range(K):
                d = (Pxy - torch.stack([ex, ey], -1)[:, None]).norm(dim=-1)
                mstar = d.argmin(dim=1)                     # [b] nearest ref index
                pref = Pxy[ar, mstar]; yref = Pyaw[ar, mstar]
                dx = ex - pref[:, 0]; dy = ey - pref[:, 1]
                dlat = -torch.sin(yref) * dx + torch.cos(yref) * dy      # left +
                dpsi = _wrap(eyaw - yref)
                # RoaD target: recorded path at (mstar+horizon) in the DRIFTED ego frame
                mh = (mstar[:, None] + hz[None]).clamp(max=HL)           # [b, S]
                ref_fut = Pxy[ar[:, None], mh]                           # [b, S, 2]
                d_world = ref_fut - torch.stack([ex, ey], -1)[:, None]   # [b, S, 2]
                tgt = ego_frame(d_world, eyaw[:, None])                  # [b, S, 2]
                # record the UNWARPED window (re-warp identically at train time)
                for i in range(b):
                    s = int(t0[i] + mstar[i])
                    rec["ep"].append(ei); rec["s"].append(s)
                    rec["dlat"].append(float(dlat[i])); rec["dpsi"].append(float(dpsi[i]))
                    rec["tgt"].append(tgt[i].cpu()); rec["v0"].append(float(ev[i]))
                # advance the ego with the CURRENT policy's 0.5 s lookahead
                wins = torch.stack([fr[int(t0[i] + mstar[i]):int(t0[i] + mstar[i]) + W]
                                    for i in range(b)]).to(device)
                wins = wins.float().div(255.0) if wins.dtype == torch.uint8 else wins.float()
                Hs = torch.stack([perturb.sampling_homography(
                    float(dlat[i]), float(math.degrees(dpsi[i])), 1.5, 0.0)
                    for i in range(b)])
                fw = perturb.warp_batch(wins, Hs)
                out = model(fw, nav_cmd=None, v0=ev, steps=steps)
                w_look = out["traj"][:, 0]
                steer, accel = wp_to_control(w_look, ev)
                ex = ex + ev * torch.cos(eyaw) * DT
                ey = ey + ev * torch.sin(eyaw) * DT
                eyaw = eyaw + ev / WHEELBASE * torch.tan(steer) * DT
                ev = (ev + accel * DT).clamp_min(0.0)
            n_done += b
            if max_windows and n_done >= max_windows:
                break
        if max_windows and n_done >= max_windows:
            break
    out = {"ep": torch.tensor(rec["ep"]), "s": torch.tensor(rec["s"]),
           "dlat": torch.tensor(rec["dlat"]), "dpsi": torch.tensor(rec["dpsi"]),
           "tgt": torch.stack(rec["tgt"]), "v0": torch.tensor(rec["v0"])}
    inside = ((out["dlat"].abs() <= ENV_LAT) &
              (out["dpsi"].abs() * 180 / math.pi <= ENV_YAW_DEG))
    out["_frac_inside_envelope"] = float(inside.float().mean())
    out["_keep"] = inside                                 # CAT-K feasibility mask
    return out


def cl_train_step(model, base_model, records, frames_list, idx, horizons, steps,
                  lambda_dev, device, encoder_grad):
    """One supervised step on collected ON-POLICY states toward the RoaD target."""
    b = len(idx)
    frames, tgt, v0, dlat, dpsi = [], [], [], [], []
    for j in idx:
        ei = int(records["ep"][j]); s = int(records["s"][j])
        fw = frames_list[ei][s:s + W]
        fw = fw.float().div(255.0) if fw.dtype == torch.uint8 else fw.float()
        frames.append(fw); tgt.append(records["tgt"][j])
        v0.append(records["v0"][j]); dlat.append(records["dlat"][j]); dpsi.append(records["dpsi"][j])
    frames = torch.stack(frames).to(device); tgt = torch.stack(tgt).to(device)
    v0 = torch.stack(v0).to(device); dlat = torch.stack(dlat).to(device); dpsi = torch.stack(dpsi).to(device)
    warped = perturb.warp_windows(frames, dlat, dpsi)     # SAME op as the harness
    nav = torch.zeros(b, dtype=torch.long, device=device)
    dec = raf._plan(model, warped, v0, nav, steps, training=True, encoder_grad=encoder_grad)
    anchors = model.decoder.anchors.to(tgt.dtype)
    dist = ((tgt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))
    a_star = dist.argmin(dim=1)
    loss_cls = F.cross_entropy(dec["anchor_logits"], a_star)
    recon = dec["anchor_traj"][torch.arange(b, device=device), a_star]
    loss_traj = (recon - tgt).abs().mean()
    # lambda_dev on the LOW-drift states (near on-path): keep the plan == base plan
    loss_dev = torch.zeros((), device=device)
    low = (dlat.abs() < 0.25) & (dpsi.abs() < math.radians(2))
    if lambda_dev > 0 and bool(low.any()):
        with torch.no_grad():
            base_dec = raf._plan(base_model, warped, v0, nav, steps, training=False)
        loss_dev = (dec["traj"][low] - base_dec["traj"][low]).abs().mean()
    loss = loss_traj + loss_cls + lambda_dev * loss_dev
    return loss, {"traj": float(loss_traj.detach()), "cls": float(loss_cls.detach()),
                  "dev": float(loss_dev.detach()), "mean_drift_m": float(dlat.abs().mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refc-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--ft-slice", default="0:28")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dagger-rounds", type=int, default=3)
    ap.add_argument("--steps-per-round", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lambda-dev", type=float, default=1.0)
    ap.add_argument("--unfreeze-encoder-stages", type=int, default=0)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--collect-windows", type=int, default=0, help="0=all FT windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        from tanitad.refs.refc import refc_smoke_config
        cfg = refc_smoke_config()
        global W, K
        W = cfg.window
        model = RefCModel(cfg).to(device); base = RefCModel(cfg).to(device)
        base.load_state_dict(model.state_dict()); base.eval()
        steps = cfg.decoder.diffusion_steps
        fl = [torch.rand(80, cfg.encoder.in_channels, cfg.encoder.image_size,
                         cfg.encoder.image_size) for _ in range(2)]
        pl = [torch.randn(80, 4) for _ in range(2)]
        for p in pl:
            p[:, 3] = p[:, 3].abs() * 8
        model.eval()
        recs = collect_onpolicy(model, fl, pl, steps, device, stride=8, batch=8,
                                max_windows=32)
        print(f"[smoke] collected {len(recs['s'])} on-policy states, "
              f"frac_inside_envelope={recs['_frac_inside_envelope']:.3f}")
        enc_params, head_params = raf._setup_trainable(model, "decoder", 0)
        opt = torch.optim.Adam(head_params, lr=args.lr)
        model.train(); model.encoder.eval()
        keep = torch.nonzero(recs["_keep"]).flatten().tolist()
        for st in range(3):
            idx = [keep[(st * 8 + i) % len(keep)] for i in range(8)]
            opt.zero_grad(set_to_none=True)
            loss, comp = cl_train_step(model, base, recs, fl, idx, HORIZONS, steps,
                                       args.lambda_dev, device, False)
            loss.backward(); opt.step()
            print(f"[smoke] step {st} loss={float(loss):.4f} {comp}")
        assert model.decoder.offset_head.weight.grad is not None
        print("LOWOOD_CL_TRAIN_SMOKE_OK")
        return

    model, cfg, base_step = raf._load_base(args.refc_ckpt, args.refc_preset, device)
    base_model, _, _ = raf._load_base(args.refc_ckpt, args.refc_preset, device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    steps = cfg.decoder.diffusion_steps
    enc_params, head_params = raf._setup_trainable(model, "decoder",
                                                   args.unfreeze_encoder_stages)
    params = head_params + enc_params
    encoder_grad = len(enc_params) > 0
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_encoder})
    opt = torch.optim.Adam(groups)
    frames_list, poses_list, ep_names = raf._load_corpus(args.val_dir, args.ft_slice)
    src_cfg = Path(args.refc_ckpt).parent / "config.json"
    if src_cfg.exists():
        shutil.copy(src_cfg, out / "config.json")
    else:
        (out / "config.json").write_text(json.dumps({"cfg": dataclasses.asdict(cfg)},
                                                     indent=2, default=str))
    arm = ("LOWOOD-CL-TRAIN + RefcCL-enc" if encoder_grad else "LOWOOD-CL-TRAIN (decoder-only)")
    print(f"[cl] {arm} | base step {base_step} | FT eps {args.ft_slice} "
          f"({len(poses_list)}) | dagger_rounds {args.dagger_rounds} x "
          f"{args.steps_per_round} | head {sum(p.numel() for p in head_params):,} + "
          f"enc {sum(p.numel() for p in enc_params):,} | lr {args.lr} lambda_dev "
          f"{args.lambda_dev}", flush=True)

    t0 = time.time(); gen = torch.Generator().manual_seed(args.seed)
    fracs = []
    for rd in range(args.dagger_rounds):
        model.eval()
        recs = collect_onpolicy(model, frames_list, poses_list, steps, device,
                                stride=args.stride, batch=args.batch,
                                max_windows=(args.collect_windows or None))
        keep = torch.nonzero(recs["_keep"]).flatten().tolist()
        fracs.append(recs["_frac_inside_envelope"])
        print(f"[cl] round {rd}: collected {len(recs['s'])} states, kept {len(keep)} "
              f"(frac_inside_envelope {recs['_frac_inside_envelope']:.3f}, "
              f"mean|drift| {float(recs['dlat'].abs().mean()):.3f}m)", flush=True)
        model.train(); model.encoder.eval()
        order = torch.randperm(len(keep), generator=gen).tolist()
        ptr = 0
        for st in range(args.steps_per_round):
            idx = [keep[order[(ptr + i) % len(order)]] for i in range(args.batch)]
            ptr += args.batch
            opt.zero_grad(set_to_none=True)
            loss, comp = cl_train_step(model, base_model, recs, frames_list, idx,
                                       HORIZONS, steps, args.lambda_dev, device, encoder_grad)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if st % 50 == 0 or st == args.steps_per_round - 1:
                print(json.dumps({"round": rd, "step": st,
                                  "loss": round(float(loss.detach()), 4),
                                  **{k: round(v, 4) for k, v in comp.items()},
                                  "gnorm": round(gnorm, 3),
                                  "elapsed_s": round(time.time() - t0, 1)}), flush=True)
            opt.step()

    torch.save({"model": model.state_dict(), "step": args.dagger_rounds * args.steps_per_round,
                "base_step": base_step}, out / "ckpt.pt")
    (out / "cl_provenance.json").write_text(json.dumps({
        "arm": arm, "objective": "closed-loop-consistent on-policy recovery "
        "(RoaD/CAT-K) on the real-footage low-OOD source",
        "base_ckpt": args.refc_ckpt, "base_step": base_step, "ft_slice": args.ft_slice,
        "dagger_rounds": args.dagger_rounds, "steps_per_round": args.steps_per_round,
        "lr_head": args.lr, "unfreeze_encoder_stages": args.unfreeze_encoder_stages,
        "lr_encoder": args.lr_encoder if encoder_grad else None,
        "lambda_dev": args.lambda_dev, "frac_inside_envelope_per_round": fracs,
        "n_encoder_trainable": sum(p.numel() for p in enc_params),
        "param_breakdown": param_breakdown(model),
        "evidence_class": "MEASURED (this run)"}, indent=2, default=str))
    print(f"[cl] wrote {out}/ckpt.pt", flush=True)
    print("LOWOOD_CL_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
