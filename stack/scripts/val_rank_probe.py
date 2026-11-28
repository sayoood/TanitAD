#!/usr/bin/env python3
"""VAL-side rank/participation probe for v7-tiny trunk arms.

WHY THIS FILE EXISTS
--------------------
This instrument produced `~/v7tiny/val_rank_3way.json` on Thor (lewm 3.500,
sub32c 3.329, sub64c 3.564, k1 4.052, cap 3.548 — all @ step 6000, n=1440) but
lived ONLY at `thor:/tmp/vp2.py` (md5 dd5d898c40ee5458a8839c0141abb129). A
`/tmp` sweep or a reboot would have destroyed the only copy of the instrument
behind five registered claims. Recovered into the repo 2026-08-23 by the
TrainingFlyWheel and generalised to take arm names on the command line.

WHY IT IS NOT THE GATE NUMBER
-----------------------------
H-RANK-9 (SUPPORTED): the in-trainer stage gate pools the O4-BIASED TRAIN
stream, which OVERSTATES participation (~5.5 gate vs ~3.4 val on the SAME
model). Gate numbers are cross-arm comparable but are NOT a val-side measure.
This probe is the val-side measure. Quote gate numbers against gate numbers and
val numbers against val numbers — never mix the two streams in one comparison.

COMPARABILITY CONTRACT (do not "improve" these without renaming the output)
--------------------------------------------------------------------------
Every number already in `val_rank_3way.json` was produced with:
  * the FIRST 12 val clips of the val cache, sorted (`[:12]`)
  * at most 120 frames per clip  ->  n = 1440 rows
  * 3-frame channel stacking with lags (2, 1, 0), clamped at the clip start
  * `world.encode_window(x)[:, 0]`, fp32 on CPU before the spectrum
  * `spectrum_report` participation_ratio (p is proportional to sigma^2 --
    ENERGY, the collapse statistic; NOT effective_rank, which uses sigma and is
    tail-sensitive -- see H-GATE-2 / C132)
Changing the clip count, the frame cap, the stacking lags or the statistic
makes new numbers INCOMPARABLE with the banked ones. If you must change them,
write to a NEW json path so the old numbers stay quotable.

USAGE
-----
    python scripts/val_rank_probe.py champ30k
    python scripts/val_rank_probe.py k1 cap --steps-note "6k arms"

Evidence class of what it emits: MEASURED (ours).
"""

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# --- comparability constants (see the contract above) -----------------------
N_CLIPS = 12
MAX_FRAMES_PER_CLIP = 120
STACK_LAGS = (2, 1, 0)
CHUNK = 16

DEFAULT_ROOT = os.path.expanduser("~/v7tiny")
DEFAULT_VAL = os.path.expanduser(
    "~/valdata/physicalai-val-0c5f7dac3b11-w120-256x640cyl"
)
DEFAULT_STACK = os.path.expanduser("~/TanitAD/stack")
DEFAULT_OUT = os.path.join(DEFAULT_ROOT, "val_rank_3way.json")


def encode_clip(world, clip_path, dev):
    """Encode one .v2ep.pt clip into trunk latents. Mirrors vp2.py exactly."""
    d = torch.load(clip_path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
    n = min(MAX_FRAMES_PER_CLIP, len(off) - 1)

    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(
            torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        )

    out = []
    with torch.no_grad():
        for s in range(0, n, CHUNK):
            chunk = [
                torch.cat([imgs[max(i - j, 0)] for j in STACK_LAGS], 0)
                for i in range(s, min(s + CHUNK, n))
            ]
            x = torch.stack(chunk)[:, None].to(dev)
            out.append(world.encode_window(x)[:, 0].float().cpu())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="arm directory names under --root")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--val-cache", default=DEFAULT_VAL)
    ap.add_argument("--stack", default=DEFAULT_STACK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-clips", type=int, default=N_CLIPS,
                    help="CHANGING THIS BREAKS COMPARABILITY -- see the contract")
    args = ap.parse_args()

    sys.path.insert(0, args.stack)
    from tanitad.eval.v6_probe_trunk import load_trunk_auto  # noqa: E402
    from tanitad.models.v6 import spectrum_report  # noqa: E402

    if args.n_clips != N_CLIPS:
        print(f"[WARN] n_clips={args.n_clips} != banked {N_CLIPS} -- results are "
              f"NOT comparable with existing rows in {args.out}", flush=True)

    dev = torch.device("cuda")
    clips = sorted(glob.glob(os.path.join(args.val_cache, "*.v2ep.pt")))[: args.n_clips]
    if not clips:
        raise SystemExit(f"[FAIL] no val clips found under {args.val_cache}")
    print(f"[probe] {len(clips)} val clips from {args.val_cache}", flush=True)

    res = {}
    if os.path.isfile(args.out):
        try:
            res = json.load(open(args.out))
        except Exception as e:  # keep going, but say so loudly
            print(f"[WARN] could not parse existing {args.out}: {e}", flush=True)

    for arm in args.arms:
        ckpt_path = os.path.join(args.root, arm, "ckpt.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[FAIL] missing {ckpt_path} -- skipping {arm}", flush=True)
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        world, _g, step = load_trunk_auto(ck, dev, ckpt_path=ckpt_path)
        world.eval()

        Z = []
        for cp in clips:
            Z.extend(encode_clip(world, cp, dev))
        A = torch.cat(Z)
        r = spectrum_report(A)

        res[arm] = {
            "step": int(step),
            "n": int(A.shape[0]),
            "participation": round(r["participation_ratio"], 3),
            "top8": round(r["top_k_share"], 4),
            "_stream": "VAL (unbiased) -- NOT the O4-biased train-pooled gate stream",
            "_evidence_class": "MEASURED (ours)",
        }
        print(f"  VAL {arm}: step={step} n={A.shape[0]} "
              f"partic={r['participation_ratio']:.3f} "
              f"top8={r['top_k_share']:.4f}", flush=True)

        del world
        torch.cuda.empty_cache()

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[probe] wrote {args.out}", flush=True)
    print("ZZVALPROBE-DONEZZ", flush=True)


if __name__ == "__main__":
    main()
