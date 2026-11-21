#!/usr/bin/env python3
"""P6a - can the dev box rebuild the refcv5-v2 trunk's INPUT FRAMES from raw mp4, for clips
that have no `*.v2ep.pt` payload? MEASURED against clips that DO have one.

Why it matters: the frozen BEV head is data-limited (82 train clips; val AP peaks at step
1,000 and declines). The only way to add training clips is to build their frames locally.
A rebuilt frame that is not the trunk's frame would make those tokens a different input
distribution -- so equivalence is measured, not assumed.

Path replicated (`stack/scripts/v2_compressed.py::_resampled` + `_decode_cropped_selected`):
  t_query = linspace(t_cam[0], t_cam[-1], int(span_s * 10)); frame_idx = searchsorted
  decode those mp4 frames (PyAV rgb24) -> `calib.cylindrical_rectify(batch, intrinsics_for_clip,
  CanonicalFrame(256, 640, 305.5774907364391, 'cylindrical'))`, batches of 16
compared to the payload's PNG frames (lossless), pixel by pixel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_build_corpus as B  # noqa: E402
from lidar_fetch import sha12  # noqa: E402


def rebuild_frames(clip_id: str, frame_idx: np.ndarray) -> torch.Tensor:
    import av
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
    from tanitad.data.physicalai import _physicalai_root_of, intrinsics_for_clip
    mp4 = Path(B.CAM_DIR) / f"{clip_id}.mp4"
    intr = intrinsics_for_clip(clip_id, _physicalai_root_of(mp4))
    if not getattr(intr, "per_clip", False):
        raise RuntimeError("no per-clip intrinsics -- refusing the corpus-median fallback")
    fr = CanonicalFrame(height=256, width=640, f_ref=305.5774907364391, projection="cylindrical")
    need = set(int(i) for i in frame_idx.tolist())
    crops, bidx, bfr = {}, [], []

    def flush():
        if bfr:
            out = cylindrical_rectify(torch.stack(bfr), intr, fr)
            for j, idx in enumerate(bidx):
                crops[idx] = out[j]

    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        fi = 0
        for vf in c.decode(st):
            if fi in need:
                bfr.append(torch.from_numpy(vf.to_ndarray(format="rgb24")).permute(2, 0, 1))
                bidx.append(fi)
                if len(bfr) >= 16:
                    flush()
                    bidx, bfr = [], []
            fi += 1
        flush()
    return torch.stack([crops[int(i)] for i in frame_idx.tolist()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(2)
    import torchvision.io as tvio
    res = {"schema": "tanitad.frames_equivalence/1", "evidence_class": "MEASURED (ours, dev-box CPU)",
           "torch": torch.__version__, "clips": []}
    for cid in B.join_clips()[: args.n_clips]:
        g = B.episode_grid(cid)
        t0 = time.time()
        loc = rebuild_frames(cid, g["cam_frame_idx"])
        t_build = time.time() - t0
        d = torch.load(Path(B.V2EP_DIR) / f"{cid}.v2ep.pt", map_location="cpu", weights_only=False)
        lens = d["jpeg_len"].to(torch.int64)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
        T = len(lens)
        ref = torch.stack([tvio.decode_png(d["jpeg_buf"][int(offs[i]):int(offs[i + 1])],
                                           mode=tvio.ImageReadMode.RGB) for i in range(T)])
        if loc.shape != ref.shape:
            res["clips"].append({"clip": sha12(cid), "shape_mismatch": [list(loc.shape), list(ref.shape)]})
            continue
        diff = (loc.to(torch.int16) - ref.to(torch.int16)).abs()
        rec = {"clip": sha12(cid), "T": T, "rebuild_s": round(t_build, 1),
               "exact_equal_frac": float((diff == 0).float().mean()),
               "le1_frac": float((diff <= 1).float().mean()),
               "max_abs": int(diff.max()), "mean_abs": float(diff.float().mean()),
               "p999_abs": float(torch.quantile(diff.flatten()[::97].float(), 0.999)),
               "ref_nonzero_frac": float((ref > 0).float().mean())}
        res["clips"].append(rec)
        print(json.dumps(rec), flush=True)
    (HERE.parent / "raw" / "p6a_frames_equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
