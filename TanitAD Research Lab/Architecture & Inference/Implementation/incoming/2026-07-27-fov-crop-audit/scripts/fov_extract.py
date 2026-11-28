"""FOV crop audit — PART 2 step 2: the FOV x RESOLUTION x ASPECT sweep, feature extraction.

Every arm sees **the same clips, the same native frames, the same labels and the same split**; only
the crop geometry and the input shape change. Each clip's mp4 is decoded ONCE and every arm's crop
is taken from those same decoded pixels, so the sweep is paired at the pixel level.

⚠️ C-FID (fidelity, one half of the required bi-directional check) is enforced per clip: the
`A_51_256` arm rebuilt here from the raw mp4 must reproduce `calib.ftheta_crop_resize`'s output on
the same frames. A mismatch is fatal for the clip, not a warning — the whole sweep is a contrast
against that arm.

Host: dev box. ⛔ No pod is touched.

usage:
  python fov_extract.py <labels_dir> <out_dir> [--stride 3] [--limit N] [--arms a,b,c]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get(
    "TANITAD_STACK", r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack"))

from fov_geom import blur_like, crop_geometry, crop_resize, pad_like        # noqa: E402
from shape_shim import build_trunk                                          # noqa: E402
from tanitad.data.calib import (F_REF, ftheta_crop_box,                     # noqa: E402
                                ftheta_crop_resize)
from tanitad.data.physicalai import (discover_r0_clips,                     # noqa: E402
                                     intrinsics_for_clip)

ROOT = os.environ.get("TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")
CANON_HFOV = 2 * math.degrees(math.atan(128.0 / F_REF))       # 51.394

# name -> (out_h, out_w, hfov_deg, degrade)   degrade in {None, ("match", ref), ("blur", ref)}
ARMS = {
    #   the FOV axis, at TODAY's input shape (all cost EXACTLY what v1 costs: 256 tokens)
    "A_51_256":      (256, 256, CANON_HFOV, None),            # TODAY — the baseline
    "F70_256":       (256, 256, 70.0, None),
    "F90_256":       (256, 256, 90.0, None),
    "F100_256":      (256, 256, 100.0, None),                 # the PI's ask, free
    "F120_256":      (256, 256, 120.5, None),                 # the whole sensor
    #   the two confounds, isolated at the CANONICAL field
    "M_match100":    (256, 256, CANON_HFOV, ("match", "F100_256")),
    "R_blur100":     (256, 256, CANON_HFOV, ("blur", "F100_256")),   # INSTRUMENT-BLIND check
    #   the RESOLUTION axis
    "A_51_384":      (384, 384, CANON_HFOV, None),            # 1.5x resolution, today's field
    "F100_384":      (384, 384, 100.0, None),
    "F100_640sq":    (640, 640, 100.0, None),                 # 100 deg at ~today's angular res
    #   the ASPECT axis — same width, same HFOV, same angular scale, 2.5x fewer tokens
    "F100_640x256":  (256, 640, 100.0, None),
}
DEFAULT_ARMS = list(ARMS)
CROP_CHUNK = 16          # native frames pushed to the GPU per crop batch (peak ~1 chunk)


def decode_needed(mp4, want_idx, max_idx):
    """Decode only the native frames in `want_idx` (a sorted unique array). -> dict idx -> [3,H,W]."""
    import av
    out = {}
    want = set(int(i) for i in want_idx)
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = int(os.environ.get("PAI_DECODE_THREADS", "6"))
        for i, fr in enumerate(c.decode(st)):
            if i in want:
                out[i] = torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1)
            if i >= max_idx:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--batch", type=int, default=8)
    # `--shard i/n` splits the clip list across n concurrent workers writing into the SAME output
    # directory (each clip is one self-contained npz, so there is no cross-shard state). Sharding by
    # `k % n` keeps every shard's chunk mix comparable.
    # ⚠️ Set OMP_NUM_THREADS=6 before launching concurrent shards: torch spawns ~113 threads per
    # process by default and concurrent workers then thrash to GPU sm 0-6 %, which looks exactly
    # like a hang (MEASURED on this pod class; it cost a sibling stream 50 minutes).
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    sh_i, sh_n = (int(x) for x in args.shard.split("/"))
    arms = [a for a in args.arms.split(",") if a]
    os.makedirs(args.out_dir, exist_ok=True)

    L = np.load(os.path.join(args.labels_dir, "fov_labels.npz"))
    meta = json.load(open(os.path.join(args.labels_dir, "fov_meta.json")))
    k2clip = json.load(open(os.path.join(args.labels_dir, "_LOCAL_ONLY_k2clip.json")))
    clips = {c["clip_id"]: c for c in discover_r0_clips(ROOT)}
    if args.limit:
        meta = meta[:args.limit]
    if sh_n > 1:
        meta = [m for m in meta if m["k"] % sh_n == sh_i]
        print(f"[extract] shard {sh_i}/{sh_n}: {len(meta)} clips", flush=True)

    trunks = {}
    for a in arms:
        oh, ow, _hf, _dg = ARMS[a]
        if (oh, ow) not in trunks:
            trunks[(oh, ow)] = build_trunk(oh, ow, "cuda")
    print(f"[extract] {len(arms)} arms, {len(trunks)} distinct input shapes", flush=True)

    geom_rows, fid_rows, t0 = [], [], time.time()
    done = 0
    for m in meta:
        k = m["k"]
        dst = os.path.join(args.out_dir, f"clip_{k:05d}.npz")
        if os.path.exists(dst):
            done += 1
            continue
        clip_id = k2clip[str(k)]
        c = clips.get(clip_id)
        if c is None:
            continue
        intr = intrinsics_for_clip(clip_id, ROOT)
        if not intr.per_clip:
            print(f"  [k={k}] REFUSED: no per-clip intrinsics (rig-B fallback cy)", flush=True)
            continue
        fidx = L[f"c{k}_frame_idx"]
        T = int(m["T"])
        # output index t (0..T-1) uses native frames fidx[t], fidx[t+1], fidx[t+2] (n_stack=3)
        ts = np.arange(0, T, args.stride)
        need = np.unique(np.concatenate([fidx[ts + j] for j in range(3)]))
        try:
            frames = decode_needed(c["mp4"], need, int(need.max()))
        except Exception as exc:                                     # noqa: BLE001
            print(f"  [k={k}] decode FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if any(int(i) not in frames for i in need):
            print(f"  [k={k}] REFUSED: decode short ({len(frames)}/{len(need)} frames)", flush=True)
            continue
        native = torch.stack([frames[int(i)] for i in need])          # [Nn,3,H,W]
        pos = {int(i): j for j, i in enumerate(need)}
        h, w = native.shape[-2:]
        sx, sy = w / float(intr.width), h / float(intr.height)
        cxp, cyp = intr.cx * sx, intr.cy * sy

        # ---- per-arm geometry ledger (per clip: the rig split makes padding clip-specific) ----
        G = {a: crop_geometry(intr.poly, cxp, cyp, h, w, *ARMS[a][:3])
             for a in arms if ARMS[a][3] is None}
        for a in arms:
            if ARMS[a][3] is not None:
                G[a] = crop_geometry(intr.poly, cxp, cyp, h, w, *ARMS[a][:3])

        # ---- crop + encode, ARM BY ARM, on the GPU (the crop is the CPU bottleneck otherwise) ----
        # `native` stays on the host (a 1080p clip is ~1.2 GB); each arm streams it through the GPU
        # in CROP_CHUNK-frame batches, so peak device memory is one chunk of one arm, never a clip.
        feats, fid_done = {}, False
        for a in arms:
            oh, ow, _hf, dg = ARMS[a]
            tr = trunks[(oh, ow)]
            img = torch.empty((native.shape[0], 3, oh, ow), dtype=torch.uint8, device="cuda")
            for b in range(0, native.shape[0], CROP_CHUNK):
                img[b:b + CROP_CHUNK] = crop_resize(
                    native[b:b + CROP_CHUNK].to("cuda", non_blocking=True), G[a])
            if dg is not None:
                ref = G[dg[1]]
                img = blur_like(img, ref.native_px_per_out_px, G[a].native_px_per_out_px)
                if dg[0] == "match":
                    bh = 2 * ref.half_h_px
                    img = pad_like(img, ref.pad_top / bh, ref.pad_bot / bh)
            # ---- C-FID: A_51_256 must equal the repo's OWN canonical crop ----
            # Two legs. (1) the crop BOX must equal `calib.ftheta_crop_box` exactly, in integers —
            # this is the leg that has teeth and it already caught a 1-px offset bug. (2) the
            # PIXELS are compared too, but with a tolerance: leg 1's crop runs on the GPU here and
            # on the CPU in `calib`, and `uint8` cast TRUNCATES, so a 1-ulp bilinear difference can
            # flip a whole level. A box mismatch, or a pixel diff > 2, refuses the clip.
            if a == "A_51_256":
                c_ref, t_ref, l_ref = ftheta_crop_box(intr, h, w, 256)
                box_ok = (int(round(2 * G[a].half_w_px)) == c_ref
                          and G[a].top == t_ref and G[a].left == l_ref)
                ref_img = ftheta_crop_resize(native, intr, 256)
                diff = (ref_img.int() - img.cpu().int()).abs()
                d = int(diff.max())
                fid_rows.append({"k": k, "max_abs_px_diff": d,
                                 "frac_px_gt1": float((diff > 1).float().mean()),
                                 "box_exact": bool(box_ok)})
                fid_done = True
                if (not box_ok) or d > 2:
                    print(f"  [k={k}] C-FID FAILED: box_exact={box_ok} |diff|={d}", flush=True)
                    break
            gather = [torch.as_tensor([pos[int(i)] for i in fidx[ts + j]], device="cuda")
                      for j in range(3)]
            f = np.empty((len(ts), tr.state_dim), dtype=np.float16)
            with torch.no_grad():
                for b in range(0, len(ts), args.batch):
                    sl = slice(b, b + args.batch)
                    x = torch.cat([img[g[sl]] for g in gather], 1).float().div_(255.0)
                    f[sl] = tr.encode(x).to(torch.float16).cpu().numpy()
            feats[a] = f
            del img
        if fid_done and len(feats) != len(arms):
            continue
        np.savez(dst, t=ts.astype(np.int32), **feats)
        for a in arms:
            g = G[a]
            geom_rows.append({"k": k, "arm": a, "rig": "B" if intr.cy >= 650 else "A",
                              "hfov_deg": round(g.hfov_deg, 3), "vfov_deg": round(g.vfov_deg, 3),
                              "px_per_deg": round(g.px_per_deg, 4),
                              "native_px_per_out_px": round(g.native_px_per_out_px, 4),
                              "pad_frac_rows": round(g.pad_frac_rows, 5),
                              "pad_frac_cols": round(g.pad_frac_cols, 5),
                              "crop_w_px": round(2 * g.half_w_px, 1),
                              "crop_h_px": round(2 * g.half_h_px, 1)})
        done += 1
        if done % 20 == 0:
            el = time.time() - t0
            print(f"[extract] {done}/{len(meta)} clips ({el:.0f}s, {el/done:.2f}s/clip)", flush=True)

    sfx = "" if sh_n == 1 else f".shard{sh_i}"
    pd.DataFrame(geom_rows).to_parquet(os.path.join(args.out_dir, f"geom{sfx}.parquet"))
    fid = pd.DataFrame(fid_rows)
    summ = {"arms": arms, "stride": args.stride, "n_clips": done, "shard": args.shard,
            "C_FID": {"n": int(len(fid)),
                      "max_abs_px_diff_over_all_clips": int(fid.max_abs_px_diff.max())
                      if len(fid) else None,
                      "n_px_exact": int((fid.max_abs_px_diff == 0).sum()) if len(fid) else 0,
                      "n_box_exact": int(fid.box_exact.sum()) if len(fid) else 0,
                      "max_frac_px_gt1": round(float(fid.frac_px_gt1.max()), 8) if len(fid) else None,
                      "rule": "leg1 crop BOX == calib.ftheta_crop_box exactly (integers); "
                              "leg2 pixels vs calib.ftheta_crop_resize, |diff| > 2 refuses "
                              "(GPU-vs-CPU bilinear + uint8 truncation costs up to 1 level)"},
            "wallclock_s": round(time.time() - t0, 1)}
    json.dump(summ, open(os.path.join(args.out_dir, f"extract_summary{sfx}.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
