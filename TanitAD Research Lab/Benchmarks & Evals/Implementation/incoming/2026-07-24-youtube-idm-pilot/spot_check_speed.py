"""P3 spot-check — the ONLY direct GT available on YouTube: when a harvested clip
carries a speedometer / GPS-speed overlay (or a stated constant speed), compare
the IDM's pseudo speed against it.

This needs a HUMAN-supplied GT because reading a burned-in overlay is a per-video
task (no OCR dependency is installed, and clip frames are deleted after encoding
for privacy). Provide GT one of two ways:
  --gt-mps V            a single stated/constant speed for the whole clip (m/s)
  --gt-file path.csv    lines 't_seconds,speed_mps' (piecewise; nearest-window match)

Reports pseudo-vs-GT bias, MAE and (for a series) R2 on that clip's windows.
Run AFTER pseudo_label has written latents (yt_<clip>.pt). If no CC clip carries a
readable speed, report that plainly — the distributional sanity + the MEASURED
comma2k19 cross-domain speed R2 0.62-0.66 remain the quality evidence.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                               # noqa: E402
import pseudo_label as PL                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", type=int, required=True)
    ap.add_argument("--latents-dir", default="/workspace/tmp/yt_pilot/latents")
    ap.add_argument("--gt-mps", type=float, default=None)
    ap.add_argument("--gt-file", default=None)
    ap.add_argument("--target-hz", type=float, default=10.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latf = Path(args.latents_dir) / f"yt_{args.clip_id:05d}.pt"
    if not latf.exists():
        print(json.dumps({"error": f"missing {latf}"})); return
    z = torch.load(latf, weights_only=False)["z"].float()
    zw, _sc, _tj = ih.build_windows(z, torch.zeros(z.shape[0], 4),
                                    torch.zeros(z.shape[0], 2), k=4, stride=2)
    lab = PL.build_labeler(device)
    with torch.no_grad():
        pred = lab(zw.to(device))["scalars"][:, 0].cpu()           # speed
    # window center time: valid_centers start at k=4, stride 2 -> t = 4 + 2*i
    idx = torch.arange(pred.shape[0])
    center_frame = 4 + 2 * idx
    center_t = center_frame.float() / args.target_hz

    if args.gt_mps is not None:
        gt = torch.full_like(pred, args.gt_mps)
    elif args.gt_file:
        rows = [l.split(",") for l in Path(args.gt_file).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        gt_t = torch.tensor([float(r[0]) for r in rows])
        gt_v = torch.tensor([float(r[1]) for r in rows])
        gt = torch.stack([gt_v[(gt_t - t).abs().argmin()] for t in center_t])
    else:
        print(json.dumps({"error": "provide --gt-mps or --gt-file"})); return

    err = (pred - gt)
    out = {"clip_id": args.clip_id, "n_windows": int(pred.shape[0]),
           "pseudo_speed_mean": round(float(pred.mean()), 3),
           "gt_speed_mean": round(float(gt.mean()), 3),
           "bias_mps": round(float(err.mean()), 3),
           "mae_mps": round(float(err.abs().mean()), 3)}
    if args.gt_file:
        out["r2"] = round(ih.r2_score(pred, gt), 3)
    print("SPOT_CHECK_JSON_START")
    print(json.dumps(out, indent=2))
    print("SPOT_CHECK_JSON_END")


if __name__ == "__main__":
    main()
