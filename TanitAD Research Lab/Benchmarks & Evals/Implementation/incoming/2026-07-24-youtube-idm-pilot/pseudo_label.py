"""P3 — pseudo-label the harvested YouTube pilot clips.

  frozen flagship-v1 encoder  ->  latent z [T,2048] per clip  (durable artifact;
                                   the downstream P4 read consumes these)
  multi-domain IDM head L (trained on parity real labels rigA+rigB+comma)  ->
                                   per-window pseudo ego-motion.

Primary channels: SPEED + LONGITUDINAL TRAJECTORY (zero-shot R2 0.60-0.66 MEASURED
on cross-domain). YAW is CAVEATED (cross-class R2~=0). ACCEL is DROPPED (unusable).

Distributional sanity: pseudo speeds must land in a physically plausible band and
resemble the parity/comma speed distribution — the only GT-free check available on
YouTube unless a clip carries a speedometer overlay (see spot_check hook).

Frames are DELETED after encoding (privacy + disk): only latents (non-imagery),
pseudo-labels, and pointers persist.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                               # noqa: E402
import run_idm_proof as R                                           # noqa: E402

PARITY_LAT = "/workspace/tmp/branchb_eval/lat_flagshipv1"           # cached v1 latents


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def clip_windows_from_latfiles(tags, latdir, k=4, stride=2):
    out = []
    for t in tags:
        p = Path(latdir) / f"{t}.pt"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]:
            out.append((zw, sc, tj))
    return out


def cat(lst):
    return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
            torch.cat([x[2] for x in lst]))


def build_labeler(device, epochs=50, seed=0):
    """Multi-domain IDM head = v1 + head trained on parity real labels
    {rigA[:60]+rigB[:60]+comma[:40]} — identical recipe to the parity/proxy runs."""
    tags = ([f"tr_a_{i:05d}" for i in range(60)] +
            [f"tr_b_{i:05d}" for i in range(60)] +
            [f"cm_{i:05d}" for i in range(40)])
    clips = clip_windows_from_latfiles(tags, PARITY_LAT)
    if not clips:
        raise RuntimeError(f"no parity latents in {PARITY_LAT} to build labeler")
    Z, S, T = cat(clips)
    torch.manual_seed(seed)
    std = ih.Standardizer.fit(S)
    head = ih.IDMHead(state_dim=2048, horizons=ih.DEFAULT_HORIZONS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    n = Z.shape[0]; Z, S, T = Z.to(device), S.to(device), T.to(device)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // 256))
    g = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(epochs):
        head.train(); perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            ld = ih.idm_loss(head(Z[idx]), S[idx], T[idx], std)
            opt.zero_grad(set_to_none=True); ld["loss"].backward(); opt.step(); sched.step()
    head.eval()
    log(f"labeler built on {n} parity windows ({len(clips)} clips)")
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--clips-dir", default="/workspace/tmp/yt_pilot/clips")
    ap.add_argument("--latents-dir", default="/workspace/tmp/yt_pilot/latents")
    ap.add_argument("--out", default="/workspace/tmp/yt_pilot/results/pseudo_labels.json")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--encode-batch", type=int, default=48)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.latents_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    enc, ro, meta = R.load_encoder(args.ckpt, device)
    labeler = build_labeler(device)

    clip_files = sorted(Path(args.clips_dir).glob("clip_*.pt"))
    log(f"pseudo-labeling {len(clip_files)} clips")
    per_clip = []
    all_speed = []
    for cf in clip_files:
        d = torch.load(cf, weights_only=False)
        cid = d.get("clip_id")
        latf = Path(args.latents_dir) / f"yt_{cid:05d}.pt"
        if latf.exists():
            z = torch.load(latf, weights_only=False)["z"]
        else:
            z = R.encode_frames(enc, ro, d["frames_u8"], device,
                                batch=args.encode_batch)
            torch.save({"z": z, "poses": torch.zeros(z.shape[0], 4),
                        "actions": torch.zeros(z.shape[0], 2),
                        "video_id": d.get("video_id"), "clip_id": cid}, latf)
        # windows + pseudo labels
        zw, _sc, _tj = ih.build_windows(z.float(), torch.zeros(z.shape[0], 4),
                                        torch.zeros(z.shape[0], 2), k=4, stride=2)
        if zw.shape[0] == 0:
            continue
        with torch.no_grad():
            o = labeler(zw.to(device))
        speed = o["scalars"][:, 0].cpu()                 # SCALAR_NAMES[0]
        yaw = o["scalars"][:, 1].cpu()
        traj = o["traj"].cpu()                            # [N,H,2]
        long_2s = traj[:, -1, 0]                          # +x forward @ 2 s (primary)
        all_speed.append(speed)
        per_clip.append({
            "clip_id": cid, "video_id": d.get("video_id"), "n_windows": int(zw.shape[0]),
            "speed_mean": round(float(speed.mean()), 3),
            "speed_p05": round(float(speed.quantile(0.05)), 3),
            "speed_p95": round(float(speed.quantile(0.95)), 3),
            "yaw_rate_abs_mean_CAVEAT": round(float(yaw.abs().mean()), 4),
            "long_disp_2s_mean": round(float(long_2s.mean()), 3),
        })
        if not args.keep_frames:
            try: os.remove(cf)                            # free imagery + disk
            except OSError: pass
    speeds = torch.cat(all_speed) if all_speed else torch.zeros(0)
    # distributional sanity vs a physical band + parity reference
    plausible = (speeds >= -1.0) & (speeds <= 45.0)      # m/s (0..162 km/h band)
    sanity = {
        "n_windows_total": int(speeds.numel()),
        "speed_mean_mps": round(float(speeds.mean()), 3) if speeds.numel() else None,
        "speed_std_mps": round(float(speeds.std()), 3) if speeds.numel() else None,
        "speed_min_mps": round(float(speeds.min()), 3) if speeds.numel() else None,
        "speed_max_mps": round(float(speeds.max()), 3) if speeds.numel() else None,
        "frac_in_plausible_0_45_mps": round(float(plausible.float().mean()), 3)
        if speeds.numel() else None,
        "note": "GT-free check. speeds should sit in a road-plausible band and "
                "not collapse to a constant; a real speedometer-overlay clip is "
                "the only direct GT (see spot_check_speed.py).",
    }
    out = {"meta": {"experiment": "youtube_idm_pilot_pseudo_label",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "ckpt": args.ckpt, "ckpt_md5": R.md5_of(args.ckpt),
                    "ckpt_step": meta.get("ckpt_step"), "state_dim": meta.get("state_dim"),
                    "labeler": "v1 + IDMHead{parity rigA[:60]+rigB[:60]+comma[:40]}",
                    "primary_channels": ["speed", "long_traj"],
                    "caveated": ["yaw_rate"], "dropped": ["long_accel", "steer"],
                    "n_clips_labeled": len(per_clip)},
           "speed_sanity": sanity, "per_clip": per_clip}
    Path(args.out).write_text(json.dumps(out, indent=2))
    log(f"speed sanity: {sanity}")
    log(f"WROTE {args.out}")
    log("YT_PSEUDO_LABEL_DONE")


if __name__ == "__main__":
    main()
