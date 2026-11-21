#!/usr/bin/env python3
"""P6c - gate G5 of `PREREG_L4_MORE_TRAINING_CLIPS.md`: do frames REBUILT from mp4 (P6a) give the
frozen trunk the same TOKENS as the pod-built v2ep frames it trained on?

For eval clips that have both: rebuild the 201 frames locally, D-015-stack them exactly as the
trainer does, encode with the frozen refcv5-v2 encoder (fp32), and compare to the cached
`tokens/tokens_s32_fp16.npy` rows of the same (clip, stacked row).
  * metric: relative mean-abs = mean|tok_rebuilt - tok_cached| / mean|tok_cached|
  * control that must read LARGE: the same rebuilt tokens against a DIFFERENT row of the cache
    (row + 50 within the clip) -- a comparison that cannot distinguish right from wrong rows
    has measured nothing.
  * bar (pre-registered): relative mean-abs <= 0.01 on every clip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_build_corpus as B  # noqa: E402
import p4a_extract_tokens as X  # noqa: E402
from lidar_fetch import sha12  # noqa: E402
from p6a_frames_equivalence import rebuild_frames  # noqa: E402


def main() -> int:
    from tanitad.data.comma2k19 import stack_frames
    dev = torch.device("cuda")
    enc, info = X.build_encoder(dev)
    div = torch.full((), 255.0, device=dev, dtype=torch.float32)
    tok = np.load(X.OUT / "tokens_s32_fp16.npy", mmap_mode="r")
    idx = np.load(X.OUT / "index.npz")
    shas = [str(s) for s in idx["clip_sha12"]]
    res = {"schema": "tanitad.token_equivalence/1", "evidence_class": "MEASURED (ours, RTX 4060)",
           "bar_rel_mean_abs": 0.01, "clips": []}
    for cid in B.join_clips()[:2]:
        s = sha12(cid)
        ci = shas.index(s)
        rows = np.nonzero(idx["clip_ordinal"] == ci)[0]
        g = B.episode_grid(cid)
        vid = rebuild_frames(cid, g["cam_frame_idx"])                     # [T,3,256,640] u8
        stacked = stack_frames(vid, 3)                                    # [T-2, 9, 256, 640]
        out = []
        with torch.inference_mode():
            for a in range(0, stacked.shape[0], 24):
                x = stacked[a:a + 24].to(dev).float().div_(div)
                out.append(enc(x)[0].float().cpu())
        rebuilt = torch.cat(out).numpy()
        cached = np.asarray(tok[rows]).astype(np.float32)
        assert rebuilt.shape == cached.shape, (rebuilt.shape, cached.shape)
        denom = float(np.abs(cached).mean())
        rel = float(np.abs(rebuilt - cached).mean()) / denom
        shifted = np.roll(np.arange(len(rows)), -50)
        rel_ctrl = float(np.abs(rebuilt - cached[shifted]).mean()) / denom
        rec = {"clip": s, "rows": int(len(rows)), "rel_mean_abs": rel, "control_rel_mean_abs_shift50": rel_ctrl,
               "pass": rel <= 0.01}
        res["clips"].append(rec)
        print(json.dumps(rec), flush=True)
    res["G5_pass"] = all(c["pass"] for c in res["clips"])
    (HERE.parent / "raw" / "p6c_token_equivalence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("G5_pass", res["G5_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
