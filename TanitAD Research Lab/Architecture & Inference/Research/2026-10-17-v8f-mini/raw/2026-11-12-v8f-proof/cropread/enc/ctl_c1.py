#!/usr/bin/env python3
"""Control c1 (card section 7): same frames, same encoder. Pod only (GPU).

For the 6 evaluation clips that come first in the order of the sha256 hex of the clip id (or --clips): the FULL frame is built from the
staged video by the cache's own build route (scripts/v2_compressed._decode_cropped_selected with the 256x640 cylindrical frame and the
per-clip intrinsics: the very call build_compressed makes), encoded by the pilot's encoder path (enc_common.Encoder, bs 8) and pooled as the
cache pools (8x20, fp32), then compared with the cached tokens of the clip (pool8x20, fp8 decoded to float).
Clause 1: the median over all frames (all clips pooled) of the cosine between the flattened tokens of a frame and the cached tokens of the
same index is >= 0.98.
Clause 2: on the frames with an ego speed >= 2 m/s (the cache's own `v`), the cached frame of the clip that matches best (largest
cosine) is the frame of the same index on >= 95 percent of them.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enc_common import Encoder, FULL_H, FULL_HFOV, FULL_W, Layout, frame_plan, read_ids   # noqa: E402


def first_by_sha(clips, k):
    return sorted(clips, key=lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest())[:k]


def compare_tokens(mine, cached, v, speed_min=2.0):
    """mine, cached: [n, tokens, width] (torch float); v: ego speed per frame (m/s). -> (cos per frame [n], match flags on the frames with
    speed >= speed_min, similarity matrix S[i, m] of mine frame i against cached frame m)."""
    import torch
    A = mine.reshape(mine.shape[0], -1).double()
    B = cached.reshape(cached.shape[0], -1).double()
    An, Bn = A / A.norm(dim=1, keepdim=True), B / B.norm(dim=1, keepdim=True)
    S = An @ Bn.T
    cos = torch.diagonal(S).numpy()
    best = S.argmax(dim=1).numpy()
    moving = np.asarray(v) >= speed_min
    return cos, (best == np.arange(len(best)))[moving], S, A, B


def window_route_full_frames(L, clip, plan, frame):
    """The window builder's OWN route (enc_common.decode_selected + window_grid + window_sample, the calls make_windows makes) with the
    full 256x640 frame's parameters: uint8 [n,3,256,640]. Equal to the builder's frames <=> the window route picks the right video frames
    and samples them as the build does (a frame shift of one anywhere in that route makes it differ)."""
    import torch
    from tanitad.data.physicalai import intrinsics_for_clip
    from enc_common import decode_selected, window_grid, window_sample
    intr = intrinsics_for_clip(clip, L.root)
    cidx = plan["cache_idx"]
    by = {}
    for j, vi in enumerate(cidx):
        by.setdefault(int(vi), []).append(j)
    out = [None] * len(cidx)
    grid = mask = None
    pend = []

    def flush():
        nonlocal grid, mask, pend
        if not pend:
            return
        vid = torch.stack([f for _, f in pend])
        if grid is None:
            grid, mask = window_grid(intr, vid.shape[-2], vid.shape[-1], frame, 0.0, 0.0)
        res = window_sample(vid, grid.expand(len(pend), -1, -1, -1), mask.expand(len(pend), -1, -1))
        for (j, _), r in zip(pend, res):
            out[j] = r
        pend = []
    for vi, fr in decode_selected(L.mp4(clip), sorted(by)):
        for j in by[vi]:
            pend.append((j, fr))
            if len(pend) >= 8:
                flush()
    flush()
    return torch.stack(out)


def clause_values(cos_all, match_flags, cos_bar=0.98, match_bar=0.95):
    """cos_all: concatenated per-frame cosines (all clips); match_flags: bool array on the frames with speed >= 2 m/s."""
    med = float(np.median(cos_all)) if len(cos_all) else float("nan")
    rate = float(np.mean(match_flags)) if len(match_flags) else float("nan")
    return {"median_cos_all_frames": med, "n_frames": int(len(cos_all)), "clause1_pass": bool(len(cos_all) and med >= cos_bar),
            "match_rate_speed_ge_2": rate, "n_frames_speed_ge_2": int(len(match_flags)),
            "clause2_pass": bool(len(match_flags) and rate >= match_bar)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-list", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache-dir", default="/workspace/v8proof/data/cache_pod")
    ap.add_argument("--clips", default="", help="explicit clip ids (comma separated) instead of the first 6 by sha256")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    import v2_compressed as V
    from tanitad.data.calib import CanonicalFrame
    L = Layout(a.root)
    ev = read_ids(a.eval_list)
    clips = [c for c in a.clips.split(",") if c] if a.clips else first_by_sha(ev, a.n)
    assert torch.cuda.is_available(), "no CUDA"
    enc = Encoder("cuda", 8)
    frame = CanonicalFrame.from_hfov(FULL_HFOV, FULL_H, FULL_W, "cylindrical")
    per, cos_all, match_all = {}, [], []
    t0 = time.time()
    for c in clips:
        tc = time.time()
        plan = frame_plan(L.ts(c))
        cidx = plan["cache_idx"]
        vid = V._decode_cropped_selected(L.mp4(c), FULL_H, cidx, frame, "cylindrical")        # [n,3,256,640] u8 (the builder's own call)
        route_equal = bool(torch.equal(vid, window_route_full_frames(L, c, plan, frame)))
        mine = enc.full_pooled(vid)                                                            # [n,160,1024] fp32
        blob = torch.load(os.path.join(a.cache_dir, "eval", "clips", f"{c}.pt"), map_location="cpu", weights_only=True)
        cached = blob["pool8x20"].float()
        v = blob["v"].float().numpy()
        row = {"clip": c, "n_mine": int(mine.shape[0]), "n_cached": int(cached.shape[0]), "secs": None, "window_route_frames_identical": route_equal}
        if mine.shape != cached.shape:
            row["error"] = f"shape {tuple(mine.shape)} vs cached {tuple(cached.shape)}"
            per[c] = row
            continue
        cos, flags, S, A, B = compare_tokens(mine, cached, v)                                    # S: [n, n] cosine, mine i vs cached m
        moving = v >= 2.0
        q = mine.to(torch.float8_e4m3fn)
        bit_eq = float((q.view(torch.uint8) == blob["pool8x20"].view(torch.uint8)).double().mean())
        row.update(median_cos=float(np.median(cos)), min_cos=float(cos.min()), mean_cos=float(cos.mean()),
                   n_moving=int(moving.sum()), match_rate_moving=float(flags.mean()) if flags.size else None,
                   rel_l2=float(((A - B).norm() / B.norm())), frac_fp8_bytes_equal=bit_eq, secs=round(time.time() - tc, 1))
        per[c] = row
        cos_all.append(cos)
        match_all.append(flags)
        print(f"[c1] {c[:8]} n={row['n_mine']} median_cos={row['median_cos']:.5f} match_moving={row['match_rate_moving']} secs={row['secs']}", flush=True)
    ok_clips = [c for c in clips if "error" not in per[c]]
    res = clause_values(np.concatenate(cos_all) if cos_all else np.zeros(0), np.concatenate(match_all) if match_all else np.zeros(0, bool))
    res.update({"control": "c1", "clips": clips, "per_clip": per, "n_clips_ok": len(ok_clips), "n_clips_error": len(clips) - len(ok_clips),
                "pass": bool(res["clause1_pass"] and res["clause2_pass"] and len(ok_clips) == len(clips)),
                "kit_route_pass": bool(len(ok_clips) == len(clips) and all(per[c].get("window_route_frames_identical") for c in ok_clips)), "secs": round(time.time() - t0, 1),
                "cache_dir": a.cache_dir, "encoder": "enc_common.Encoder (cache builder's build_model, bf16, bs 8, no autocast)"})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    res["gate_pass"] = bool(res["pass"] and res["kit_route_pass"])
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("median_cos_all_frames", "clause1_pass", "match_rate_speed_ge_2", "clause2_pass", "pass", "kit_route_pass", "gate_pass")}))
    print("ZZ C1 DONE ZZ")
    return 0 if res["gate_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
