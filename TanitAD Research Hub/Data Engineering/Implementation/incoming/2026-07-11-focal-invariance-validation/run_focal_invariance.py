"""Measured focal-invariance validation of D-016 on the trained encoder (G-H).

Controlled single-scene experiment (see focal_invariance.py): a real 120-deg
clip frame at focal f0 is re-imaged as the SAME scene at f1 = z*f0 (a narrower
camera). Both are canonicalized to F_REF with the CORRECT vs the WRONG (=f0)
intrinsics, encoded by the trained WorldModel, and the relative latent drift is
compared. D-016 works iff drift_correct << drift_wrong.

Base corpus = Cosmos-Drive-Dreams (front_wide_120fov, local extracted) — it has
the FOV headroom comma2k19 lacks. Encoder = the local partial checkpoint
(decision-relevant: it is the encoder we deploy).

Usage (local RTX 4060):
  python run_focal_invariance.py \
      --ckpt C:/Users/Admin/tanitad-data/eval/ckpt_full.pt \
      --clips-glob "C:/Users/Admin/tanitad-data/eval/extracted/generation/*.mp4" \
      --out result.json --n-clips 10 --triples-per-clip 4 --zooms 1.25 1.5
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import focal_invariance as fi  # noqa: E402

from tanitad.data.calib import focal_crop_resize, nominal_focal_from_hfov  # noqa: E402

HFOV_DEG = 120.0            # front_wide_120fov (== PhysicalAI front-wide)
SIZE = 256


def decode_raw(mp4: str, n_frames: int, stride: int = 3) -> torch.Tensor:
    """video.mp4 -> uint8 [T, 3, H, W] at native resolution (NO focal crop)."""
    import av
    out = []
    with av.open(mp4) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i % stride:
                continue
            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24"))  # H W 3
            out.append(rgb.permute(2, 0, 1))
            if len(out) >= n_frames:
                break
    return torch.stack(out)                                          # T 3 H W u8


def stack9(canon3: torch.Tensor) -> torch.Tensor:
    """[3, 3, S, S] uint8 -> [1, 9, S, S] float in [0,1] (D-015 3-frame stack)."""
    return canon3.reshape(1, 9, canon3.shape[-2], canon3.shape[-1]).float().div(255.0)


@torch.no_grad()
def encode(world, batch: torch.Tensor, device: str, bs: int = 16) -> torch.Tensor:
    outs = []
    for i in range(0, batch.shape[0], bs):
        outs.append(world.encode(batch[i:i + bs].to(device)).cpu())
    return torch.cat(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=10)
    ap.add_argument("--triples-per-clip", type=int, default=4)
    ap.add_argument("--zooms", type=float, nargs="+", default=[1.25, 1.5])
    ap.add_argument("--frame-stride", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tanitad.config import base250cam_config
    from tanitad.models.fourbrain import WorldModel
    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    world = world.to(device).eval()

    clips = sorted(glob.glob(args.clips_glob))[:args.n_clips]
    assert clips, f"no clips matched {args.clips_glob}"
    zooms = args.zooms

    # frames needed: triples of 3 consecutive strided frames
    n_frames = args.triples_per_clip + 2

    A, Bc, Bw = [], {z: [] for z in zooms}, {z: [] for z in zooms}
    f_eff_base, f_eff_wrong = [], {z: [] for z in zooms}
    used, hw = 0, None
    for mp4 in clips:
        try:
            raw = decode_raw(mp4, n_frames, args.frame_stride)     # [T,3,H,W] u8
        except Exception as e:                                     # corrupt clip
            print(f"skip {Path(mp4).name}: {type(e).__name__}: {e}")
            continue
        if raw.shape[0] < 3:
            continue
        h, w = raw.shape[-2:]
        hw = (h, w)
        f0 = nominal_focal_from_hfov(w, HFOV_DEG)
        fi.assert_effective_focal(f0, h, w, SIZE)                  # base headroom
        T = raw.shape[0]
        for t in range(T - 2):
            raw3 = raw[t:t + 3]                                    # [3,3,H,W] u8
            a = focal_crop_resize(raw3, f0, SIZE)                 # canon @ f0
            f_eff_base.append(focal_crop_resize.last_f_eff)
            A.append(stack9(a))
            for z in zooms:
                rawz = fi.virtual_focal_zoom(raw3, z).clamp(0, 255).to(torch.uint8)
                bc = focal_crop_resize(rawz, z * f0, SIZE)        # correct intr.
                Bc[z].append(stack9(bc))
                bw = focal_crop_resize(rawz, f0, SIZE)            # wrong intr.
                f_eff_wrong[z].append(focal_crop_resize.last_f_eff)
                Bw[z].append(stack9(bw))
            used += 1

    A = torch.cat(A)                                              # [N,9,S,S]
    za = encode(world, A, device)
    report = {
        "meta": {
            "hardware": ("RTX 4060" if device == "cuda" else "cpu"),
            "ckpt": Path(args.ckpt).name,
            "base_corpus": "cosmos-drive-dreams (front_wide_120fov)",
            "n_scenes": used, "n_clips": len(clips), "native_hw": hw,
            "hfov_deg": HFOV_DEG, "size": SIZE, "F_REF": fi.F_REF,
            "f_eff_base_mean": round(sum(f_eff_base) / len(f_eff_base), 2),
            "wall_clock_s": None, "cost_usd": 0.0,
        },
        "zooms": {},
    }
    for z in zooms:
        zbc = encode(world, torch.cat(Bc[z]), device)
        zbw = encode(world, torch.cat(Bw[z]), device)
        d_correct = fi.relative_latent_drift(za, zbc)
        d_wrong = fi.relative_latent_drift(za, zbw)
        cos_correct = fi.cosine_sim(za, zbc)
        cos_wrong = fi.cosine_sim(za, zbw)
        ratio = float(d_wrong.mean() / d_correct.mean().clamp_min(1e-8))
        report["zooms"][str(z)] = {
            "f_eff_wrong_mean": round(sum(f_eff_wrong[z]) / len(f_eff_wrong[z]), 2),
            "drift_correct_mean": round(float(d_correct.mean()), 4),
            "drift_correct_median": round(float(d_correct.median()), 4),
            "drift_wrong_mean": round(float(d_wrong.mean()), 4),
            "drift_wrong_median": round(float(d_wrong.median()), 4),
            "drift_reduction_x": round(ratio, 2),
            "cos_correct_mean": round(float(cos_correct.mean()), 4),
            "cos_wrong_mean": round(float(cos_wrong.mean()), 4),
            "falsifier_pass": bool(d_correct.mean() < d_wrong.mean()),
        }
        print(f"z={z}: drift correct {d_correct.mean():.4f} vs wrong "
              f"{d_wrong.mean():.4f}  ({ratio:.2f}x)  "
              f"cos {cos_correct.mean():.3f}/{cos_wrong.mean():.3f}")

    report["meta"]["wall_clock_s"] = round(time.time() - t0, 1)
    report["verdict"] = {
        "d016_validated": all(v["falsifier_pass"] for v in report["zooms"].values()),
        "note": ("drift_correct << drift_wrong across zooms => using per-camera "
                 "intrinsics (D-016) removes focal nuisance in the trained "
                 "encoder's latent space."),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out, "in", report["meta"]["wall_clock_s"], "s")


if __name__ == "__main__":
    main()
