"""Train and PERSIST `idm_head_v1` — the reusable multi-domain IDM labeler.

WHY THIS EXISTS: every IDM experiment to date (pilot pseudo-label, parity
validation, downstream ablation, the YouTube scale-up) trained this head
IN-PROCESS, measured it, and threw it away. The 80 pilot pseudo-labels were
produced by a head that no longer exists, so nothing that consumed them can be
re-derived or audited. This script makes the labeler a first-class artifact:
weights + the exact config that produced them + honest held-out val metrics.

RECIPE (identical TRAIN set to `pseudo_label.build_labeler`, so this head is the
same object the pilot used — reproduced, not redefined):
    encoder : frozen flagship-v1 (ckpt md5 recorded), state_dim 2048
    train   : parity latents  tr_a_[:60] + tr_b_[:60] + cm_[:40]
    head    : IDMHead(state_dim=2048, d_model=256, depth=3, heads=4, window=9)
    loss    : Huber on standardised scalars + smooth-L1 on the 2 s trajectory
    epochs  : 50, AdamW lr 3e-4 wd 0.01, cosine, batch 256, seed 0

ADDED (the pilot had none): two EPISODE-DISJOINT val sets that were never in
train, so the persisted card carries measured generalisation instead of a
training loss:
    val_heldout_traindomain : tr_a_[60:90] + tr_b_[60:90] + cm_[40:70]
    val_parityval           : va_a_[:20] + va_b_[:20]   (PhysicalAI val eps)

OUTPUT: a single self-describing `.pt`:
    {"state_dict", "config", "standardizer", "val", "provenance"}
so a consumer only needs:
    d = torch.load("idm_head_v1.pt"); h = IDMHead(**d["config"]["head_kwargs"])
    h.load_state_dict(d["state_dict"])

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python idm_head_v1_train.py \
      --out /workspace/tmp/yt_val/results/idm_head_v1.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                              # noqa: E402

PARITY_LAT = "/workspace/tmp/branchb_eval/lat_flagshipv1"
ENC_CKPT = "/workspace/tmp/idm/ckpt.pt"

TRAIN_TAGS = ([f"tr_a_{i:05d}" for i in range(60)] +
              [f"tr_b_{i:05d}" for i in range(60)] +
              [f"cm_{i:05d}" for i in range(40)])
VAL_HELDOUT_TAGS = ([f"tr_a_{i:05d}" for i in range(60, 90)] +
                    [f"tr_b_{i:05d}" for i in range(60, 90)] +
                    [f"cm_{i:05d}" for i in range(40, 70)])
VAL_PARITY_TAGS = ([f"va_a_{i:05d}" for i in range(20)] +
                   [f"va_b_{i:05d}" for i in range(20)])


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_windows(tags, latdir, k=4, stride=2):
    """tags -> (Z, S, T) concatenated windows; also returns the tags actually used."""
    parts, used = [], []
    for t in tags:
        p = Path(latdir) / f"{t}.pt"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]:
            parts.append((zw, sc, tj))
            used.append(t)
    if not parts:
        raise RuntimeError(f"no latents found for {len(tags)} tags in {latdir}")
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
            torch.cat([p[2] for p in parts]), used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=PARITY_LAT)
    ap.add_argument("--enc-ckpt", default=ENC_CKPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--skip-enc-md5", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    Ztr, Str, Ttr, tr_used = load_windows(TRAIN_TAGS, args.latents,
                                          k=args.k, stride=args.stride)
    log(f"train windows {tuple(Ztr.shape)} from {len(tr_used)} clips")
    vals = {}
    for name, tags in (("val_heldout_traindomain", VAL_HELDOUT_TAGS),
                       ("val_parityval", VAL_PARITY_TAGS)):
        Zv, Sv, Tv, used = load_windows(tags, args.latents, k=args.k,
                                        stride=args.stride)
        vals[name] = (Zv, Sv, Tv, used)
        log(f"{name}: {tuple(Zv.shape)} from {len(used)} clips")

    state_dim = Ztr.shape[-1]
    head_kwargs = dict(state_dim=state_dim, d_model=256, depth=3, n_heads=4,
                       window=2 * args.k + 1, n_scalars=len(ih.SCALAR_NAMES),
                       horizons=ih.DEFAULT_HORIZONS)

    # ---- train (byte-for-byte the pilot's build_labeler loop) ------------ #
    torch.manual_seed(args.seed)
    std = ih.Standardizer.fit(Str)
    head = ih.IDMHead(**head_kwargs).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    n = Ztr.shape[0]
    Zd, Sd, Td = Ztr.to(device), Str.to(device), Ttr.to(device)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs * max(1, n // args.batch))
    g = torch.Generator(device=device).manual_seed(args.seed + 1)
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n, generator=g, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            ld = ih.idm_loss(head(Zd[idx]), Sd[idx], Td[idx], std)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            opt.step()
            sched.step()
            tot += float(ld["loss"].detach()) * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"epoch {ep+1}/{args.epochs} train_loss {tot/n:.4f}")
    head.eval()
    log(f"trained in {time.time()-t0:.0f}s on {n} windows")

    # ---- honest held-out metrics ---------------------------------------- #
    val_metrics = {}
    for name, (Zv, Sv, Tv, used) in vals.items():
        m = ih.evaluate(head, Zv, Sv, Tv, device=device)
        m["n_clips"] = len(used)
        val_metrics[name] = m
        log(f"{name}: speed_r2 {m['r2']['speed']:.4f} yaw_r2 "
            f"{m['r2']['yaw_rate']:.4f} ade_2s {m['ade_2s']:.3f} n={m['n']}")

    enc_md5 = None if args.skip_enc_md5 else md5_of(args.enc_ckpt)
    ck = torch.load(args.enc_ckpt, map_location="cpu", weights_only=False)
    enc_step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    del ck

    config = {
        "name": "idm_head_v1",
        "role": ("multi-domain inverse-dynamics LABELER: frozen flagship-v1 "
                 "latents -> per-window ego-motion (speed, yaw_rate, steer, "
                 "long_accel) + 2 s ego-frame trajectory. NON-CAUSAL (sees "
                 "past AND future frames) — offline labeling only."),
        "head_kwargs": head_kwargs,
        "head_class": "scripts/idm_head.py::IDMHead",
        "state_dim": int(state_dim),
        "window_k": args.k,
        "window_frames": 2 * args.k + 1,
        "horizons_steps": list(ih.DEFAULT_HORIZONS),
        "horizons_s": [h * ih.DT for h in ih.DEFAULT_HORIZONS],
        "dt_s": ih.DT,
        "scalar_names": list(ih.SCALAR_NAMES),
        "build_windows_stride": args.stride,
        "encoder": {
            "id": "flagship4b-speedjerk-30k (v1) encoder+readout, FROZEN",
            "ckpt_path_at_train": args.enc_ckpt,
            "ckpt_md5": enc_md5,
            "ckpt_step": enc_step,
            "note": ("encoder is purely visual: encode = readout(encoder(x/255)); "
                     "no action/speed channel enters the latent."),
        },
        "train": {
            "corpora": ("PhysicalAI parity rig-A + rig-B + comma2k19 "
                        "(the multi-domain mix)"),
            "tags": {"rig_a": "tr_a_[0:60]", "rig_b": "tr_b_[0:60]",
                     "comma": "cm_[0:40]"},
            "n_clips": len(tr_used), "n_windows": int(n),
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "weight_decay": args.wd, "seed": args.seed,
            "optimizer": "AdamW + CosineAnnealingLR",
            "loss": ("idm_head.idm_loss = Huber(standardised scalars) + "
                     "smooth_L1(traj/10 m)"),
            "latents_dir": args.latents,
        },
        "target_normalisation": {
            "kind": "idm_head.Standardizer (fit on TRAIN scalars only)",
            "mean": [float(x) for x in std.mean],
            "std": [float(x) for x in std.std],
            "note": ("used INSIDE the loss only; the head's outputs are already "
                     "in RAW PHYSICAL UNITS (m/s, rad/s, rad, m/s^2, metres)."),
        },
        "usage_caveats": [
            "speed + longitudinal trajectory are the trustworthy channels "
            "(MEASURED cross-domain speed R2 0.60-0.66).",
            "yaw_rate is CAVEATED cross-domain; steer and long_accel are "
            "DROPPED as unusable (pilot NOTE 2026-07-24).",
            "NON-CAUSAL by design — never use online/closed-loop.",
        ],
    }
    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_by": "idm-youtube-validation agent (2026-07-25)",
        "script": "idm_head_v1_train.py",
        "host": platform.node(),
        "torch": torch.__version__,
        "reason": ("first PERSISTED IDM head. Prior runs (pilot pseudo-label, "
                   "parity validation, downstream ablation, scale-up) trained "
                   "an identical head in-process and discarded it."),
        "evidence_class": "MEASURED (this run; val metrics below are held-out)",
    }
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "config": config,
                "standardizer": {"mean": std.mean.cpu(), "std": std.std.cpu()},
                "val": val_metrics,
                "provenance": provenance,
                "params": ih.count_params(head)}, args.out)
    log(f"WROTE {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB, "
        f"{ih.count_params(head)/1e6:.2f}M params)")
    card = Path(args.out).with_suffix(".json")
    card.write_text(json.dumps({"config": config, "val": val_metrics,
                                "provenance": provenance,
                                "params": ih.count_params(head),
                                "weights_md5": md5_of(args.out)}, indent=2))
    log(f"WROTE {card}")
    print("IDM_HEAD_V1_DONE", flush=True)


if __name__ == "__main__":
    main()
