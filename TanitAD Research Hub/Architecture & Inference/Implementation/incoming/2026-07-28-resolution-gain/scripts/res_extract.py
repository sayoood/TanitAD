"""RESOLUTION-GAIN — step 2: the DOWNWARD angular-resolution ladder, feature extraction.

The question is *"what does finer angular resolution buy on our tasks?"*, asked the cheap way:
**run the ladder DOWNWARD on frames we can already build, and measure what LOSING resolution costs.**
If the model is insensitive to losing it, it will almost certainly not benefit from gaining it.
(That is a BOUND, not a proof — see `RESOLUTION_GAIN.md` Sec 1.3.)

Design, and why each choice is forced:

* **C34 — capacity is matched by construction.** Every rung of the primary ladder is rendered onto
  the SAME 256x640 raster, encoded by the SAME frozen deployed-v1 trunk with the SAME 640 tokens
  and the SAME resampled positional embedding. **Only the pixel content's angular bandwidth
  changes.** Nothing else can be the cause of a difference between rungs.
* **The low-pass is real, not asserted.** A rung at factor `k` is produced by resampling the
  rendered frame to `1/k` size with `antialias=True` (a true pre-filter) and back. Downsampling
  WITHOUT a low-pass measures ALIASING, not resolution — so one rung is deliberately built the
  wrong way (`A_1p5_alias`) to show the two are not the same thing.
* **The frozen-trunk handicap is CONSTANT across the ladder.** v1's weights were trained at
  256x256 / 51.4 deg, so a 256x640 input is evaluated under a train/test shape shift. That shift is
  IDENTICAL for every rung of ladder A, so it cannot bias a rung-vs-rung contrast. It DOES bias the
  `U_960` upward arm (a third shape), and that arm is therefore declared weak-if-null.

Arms
----
Ladder A (PRIMARY) — 256x640 cylindrical @ 120.00 deg, f_ref 305.5775, 640 tokens:
    V5_640      k=1.0000   5.3333 px/deg   <- the chosen v5 frame, the BASELINE
    D_today     k=1.1488   4.6426 px/deg   <- calibration: today's DEPLOYED on-axis density
    D_1p5       k=1.5000   3.5556 px/deg   <- the exact MIRROR of the 384x960 step
    D_2         k=2.0000   2.6667 px/deg
    D_3         k=3.0000   1.7778 px/deg
    D_6         k=6.0000   0.8889 px/deg   <- the C13 sensitivity demonstration (extreme)
    A_1p5_alias k=1.5, NO antialias        <- the aliasing control
Ladder B (REPLICATION) — 256x256 f-theta crop @ 51.39 deg, f_ref 266, 256 tokens:
    B_today     k=1.0      4.6426 px/deg   <- TODAY'S DEPLOYED INPUT, C-FID'd against `calib`
    B_D2        k=2.0      2.3213 px/deg
    B_D6        k=6.0      0.7738 px/deg
Upward (SECONDARY, weak-if-null) — 384x960 cylindrical @ 120 deg, 1440 tokens:
    U_960                  8.0000 px/deg   <- the candidate the PI is deciding against

Bi-directional validation built in here (the other half lives in `res_eval.py`):
  * **V-FID-A** — the wide frame's per-clip *observed* fraction must reproduce the independent
    n=3,000 rig census in `…/2026-07-28-wide-fov-build/WIDE_FOV_BUILD.md` Sec 5
    (rig A 0.0017 %, rig B 8.897 % masked). A frame built wrong will not land there.
  * **V-FID-B** — `B_today` must reproduce `calib.ftheta_crop_box` EXACTLY (integers) and
    `calib.ftheta_crop_resize` to within 2 levels. A failure refuses the clip, it does not warn.
  * **the spectral ledger** — per-rung high-frequency energy, so "information was removed
    monotonically" is MEASURED rather than assumed.

Host: dev box (RTX 4060). No pod is touched: pod1 trains, pod2 builds the 120 deg corpus.
⚠️ **No timing claim is made anywhere in this stream**, so GPU contention costs wall-clock only and
cannot invalidate a number. (That is why this experiment is safe to run on a shared desktop GPU.)

usage:
  python res_extract.py <labels_dir> <out_dir> [--stride 6] [--limit N] [--shard i/n]
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.environ.get("TANITAD_STACK",
                       r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
FOV_SCRIPTS = os.environ.get(
    "FOV_SCRIPTS",
    r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
    r"\Architecture & Inference\Implementation\incoming\2026-07-27-fov-crop-audit\scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, FOV_SCRIPTS)
sys.path.insert(0, STACK)

from shape_shim import build_trunk                                          # noqa: E402
from tanitad.data.calib import (F_REF, CanonicalFrame, cylindrical_rectify,  # noqa: E402
                                ftheta_crop_box, ftheta_crop_resize)
from tanitad.data.physicalai import (discover_r0_clips,                     # noqa: E402
                                     intrinsics_for_clip)

ROOT = os.environ.get("TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")

HFOV_WIDE = 120.0
TODAY_HFOV = 2 * math.degrees(math.atan(128.0 / F_REF))          # 51.394
TODAY_PXDEG = F_REF * math.pi / 180.0                            # 4.642576 on-axis


def cyl_frame(h: int, w: int, hfov_deg: float) -> CanonicalFrame:
    """A cylindrical (equidistant-azimuth) frame delivering EXACTLY `hfov_deg`."""
    return CanonicalFrame(height=h, width=w,
                          f_ref=(w / 2.0) / math.radians(hfov_deg / 2.0),
                          projection="cylindrical")


FRAME_V5 = cyl_frame(256, 640, HFOV_WIDE)          # f_ref 305.5775, 5.3333 px/deg, 640 tok
FRAME_UP = cyl_frame(384, 960, HFOV_WIDE)          # f_ref 458.3662, 8.0000 px/deg, 1440 tok
K_TODAY = (FRAME_V5.f_ref * math.pi / 180.0) / TODAY_PXDEG        # 1.148788

# name -> (base, k, antialias)   base in {"V5", "UP", "SQ"}
ARMS: dict[str, tuple[str, float, bool]] = {
    # ---- Ladder A (PRIMARY): identical raster, identical trunk, identical token count ----
    "V5_640":      ("V5", 1.0,     True),
    "D_today":     ("V5", K_TODAY, True),
    "D_1p5":       ("V5", 1.5,     True),
    "D_2":         ("V5", 2.0,     True),
    "D_3":         ("V5", 3.0,     True),
    "D_6":         ("V5", 6.0,     True),
    "A_1p5_alias": ("V5", 1.5,     False),          # the aliasing control
    # ---- Ladder B (REPLICATION at today's deployed frame) ----
    "B_today":     ("SQ", 1.0,     True),
    "B_D2":        ("SQ", 2.0,     True),
    "B_D6":        ("SQ", 6.0,     True),
    # ---- the direct UPWARD arm (secondary, weak-if-null) ----
    "U_960":       ("UP", 1.0,     True),
}
BASE_SHAPE = {"V5": (256, 640), "UP": (384, 960), "SQ": (256, 256)}
BASE_PXDEG = {"V5": FRAME_V5.f_ref * math.pi / 180.0,
              "UP": FRAME_UP.f_ref * math.pi / 180.0,
              "SQ": TODAY_PXDEG}
BASE_BATCH = {"V5": 8, "UP": 2, "SQ": 16}
CROP_CHUNK = 12


def degrade(img: torch.Tensor, k: float, antialias: bool,
            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Remove angular resolution by factor `k`: render at 1/k and resample back.

    `antialias=True` applies torch's true pre-filter, so information is REMOVED. With
    `antialias=False` the same nominal factor instead FOLDS high frequencies back into the band —
    that is aliasing, not a resolution change, and it is why one arm is built this way on purpose.

    `mask` (the cylindrical frame's observed mask) is re-applied afterwards so that blur does not
    bleed the honest-black unobserved periphery into observed pixels — a genuinely coarser render
    would have a sharp mask at its own scale, not a smeared one.
    """
    if k <= 1.0 + 1e-9:
        return img
    h, w = img.shape[-2:]
    sh, sw = max(8, int(round(h / k))), max(8, int(round(w / k)))
    x = img.float()
    x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False,
                      antialias=bool(antialias))
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    x = x.clamp(0, 255)
    if mask is not None:
        x = x * mask.to(x.dtype)
    return x.to(torch.uint8)


def hf_energy(img: torch.Tensor) -> float:
    """Fraction of spectral power above half-Nyquist, averaged over frames — the MEASUREMENT that
    says information really was removed (and how much), rather than the assertion that it was."""
    x = img.float().mean(1)                                     # [T,H,W] luma-ish
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    Fx = torch.fft.rfft2(x)
    p = (Fx.real ** 2 + Fx.imag ** 2)
    h, w = p.shape[-2:]
    fy = torch.fft.fftfreq(x.shape[-2], device=p.device).abs()[:, None]
    fx = torch.fft.rfftfreq(x.shape[-1], device=p.device)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2)                            # cycles/px, Nyquist = 0.5
    hi = (r > 0.25).to(p.dtype)
    tot = p.sum(dim=(-2, -1)).clamp_min(1e-12)
    return float(((p * hi).sum(dim=(-2, -1)) / tot).mean())


def decode_needed(mp4, want_idx, max_idx):
    """Decode only the native frames in `want_idx` -> {idx: uint8 [3,H,W]}."""
    import av
    out = {}
    want = {int(i) for i in want_idx}
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
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
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
    print(f"[extract] shard {sh_i}/{sh_n}: {len(meta)} clips, {len(arms)} arms", flush=True)

    trunks = {b: build_trunk(*BASE_SHAPE[b], "cuda") for b in {ARMS[a][0] for a in arms}}
    print("[extract] trunks: " + ", ".join(f"{b}{BASE_SHAPE[b]}->{t.encoder.n_tokens}tok"
                                           for b, t in trunks.items()), flush=True)

    geom_rows, fid_rows, spec_rows, t0, done = [], [], [], time.time(), 0
    for m in meta:
        k = m["k"]
        dst = os.path.join(args.out_dir, f"clip_{k:05d}.npz")
        if os.path.exists(dst):
            done += 1
            continue
        c = clips.get(k2clip[str(k)])
        if c is None:
            continue
        intr = intrinsics_for_clip(k2clip[str(k)], ROOT)
        if not intr.per_clip:
            # `cylindrical_rectify` REFUSES a corpus-median principal point and it is right to:
            # its cy is a rig-B value and a ray fan centred on the wrong rig is ~215 px off.
            print(f"  [k={k}] REFUSED: no per-clip intrinsics", flush=True)
            continue
        fidx = L[f"c{k}_frame_idx"]
        T = int(m["T"])
        ts = np.arange(0, T, args.stride)
        need = np.unique(np.concatenate([fidx[ts + j] for j in range(3)]))
        try:
            frames = decode_needed(c["mp4"], need, int(need.max()))
        except Exception as exc:                                        # noqa: BLE001
            print(f"  [k={k}] decode FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if any(int(i) not in frames for i in need):
            print(f"  [k={k}] REFUSED: decode short ({len(frames)}/{len(need)})", flush=True)
            continue
        native = torch.stack([frames[int(i)] for i in need])              # [N,3,1080,1920] host
        del frames
        pos = {int(i): j for j, i in enumerate(need)}
        gather = [torch.as_tensor([pos[int(i)] for i in fidx[ts + j]], device="cuda")
                  for j in range(3)]

        # ---- render the three BASE frames once each (the ladder degrades these) ----
        bases, masks = {}, {}
        need_bases = {ARMS[a][0] for a in arms}
        for b in need_bases:
            oh, ow = BASE_SHAPE[b]
            buf = torch.empty((native.shape[0], 3, oh, ow), dtype=torch.uint8, device="cuda")
            for s in range(0, native.shape[0], CROP_CHUNK):
                blk = native[s:s + CROP_CHUNK].to("cuda", non_blocking=True)
                if b == "SQ":
                    buf[s:s + CROP_CHUNK] = ftheta_crop_resize(blk, intr, 256)
                else:
                    fr = FRAME_V5 if b == "V5" else FRAME_UP
                    buf[s:s + CROP_CHUNK] = cylindrical_rectify(blk, intr, fr)
                    if b not in masks:
                        masks[b] = cylindrical_rectify.last_mask.to("cuda")
            bases[b] = buf
        del native

        # ---- V-FID-B: ladder B's baseline IS the deployed crop, by CALL not by re-implementation.
        # The FOV-audit sibling needed a numerical C-FID because it re-implemented the crop; this
        # stream calls `calib.ftheta_crop_resize` itself, which is the stronger guarantee. What is
        # still worth recording is the crop BOX `calib` chose per clip, because it is rig-dependent
        # and it is the quantity a wrong principal point would move.
        if "SQ" in bases:
            cref, tref, lref = ftheta_crop_box(intr, 1080, 1920, 256)
            fid_rows.append({"k": k, "box_c": int(cref), "box_top": int(tref),
                             "box_left": int(lref), "cy": round(float(intr.cy), 2),
                             "source": "calib.ftheta_crop_box"})

        # ---- the ladder ----
        feats = {}
        for a in arms:
            b, kk, aa = ARMS[a]
            img = degrade(bases[b], kk, aa, masks.get(b))
            if done < 10:
                spec_rows.append({"k": k, "arm": a, "hf_frac": hf_energy(img[:8])})
            tr = trunks[b]
            bs = BASE_BATCH[b]
            f = np.empty((len(ts), tr.state_dim), dtype=np.float16)
            with torch.no_grad():
                for s in range(0, len(ts), bs):
                    sl = slice(s, s + bs)
                    x = torch.cat([img[g[sl]] for g in gather], 1).float().div_(255.0)
                    f[sl] = tr.encode(x).to(torch.float16).cpu().numpy()
            feats[a] = f
            if img is not bases[b]:
                del img
        np.savez(dst, t=ts.astype(np.int32), **feats)
        rig = "B" if intr.cy >= 650 else "A"
        for b in need_bases:
            if b in masks:
                geom_rows.append({"k": k, "base": b, "rig": rig,
                                  "observed_frac": round(float(masks[b].float().mean()), 6),
                                  "masked_frac": round(1 - float(masks[b].float().mean()), 6),
                                  "px_per_deg": round(BASE_PXDEG[b], 4)})
            else:
                geom_rows.append({"k": k, "base": b, "rig": rig, "observed_frac": None,
                                  "masked_frac": None, "px_per_deg": round(BASE_PXDEG[b], 4)})
        del bases, masks
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"[extract] {done}/{len(meta)} ({el:.0f}s, {el/max(done,1):.2f}s/clip)", flush=True)

    sfx = "" if sh_n == 1 else f".shard{sh_i}"
    pd.DataFrame(geom_rows).to_csv(os.path.join(args.out_dir, f"geom{sfx}.csv.gz"), index=False)
    pd.DataFrame(spec_rows).to_csv(os.path.join(args.out_dir, f"spectral{sfx}.csv.gz"), index=False)
    summ = {"arms": arms, "stride": args.stride, "n_clips_done": done, "shard": args.shard,
            "ladder": {a: {"base": ARMS[a][0], "k": round(ARMS[a][1], 6),
                           "antialias": ARMS[a][2],
                           "px_per_deg": round(BASE_PXDEG[ARMS[a][0]] / ARMS[a][1], 4),
                           "equiv_width_px": round(BASE_SHAPE[ARMS[a][0]][1] / ARMS[a][1], 1),
                           "tokens": (BASE_SHAPE[ARMS[a][0]][0] // 16)
                                     * (BASE_SHAPE[ARMS[a][0]][1] // 16)}
                       for a in arms},
            "frames": {"V5": FRAME_V5.to_dict(), "UP": FRAME_UP.to_dict(),
                       "SQ": {"height": 256, "width": 256, "f_ref": F_REF,
                              "projection": "ftheta_crop (deployed)"}},
            "V_FID_B_box": {"n": len(fid_rows),
                            "rule": "the deployed square arm's crop box comes from "
                                    "calib.ftheta_crop_box itself"},
            "wallclock_s": round(time.time() - t0, 1)}
    json.dump(summ, open(os.path.join(args.out_dir, f"extract_summary{sfx}.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "ladder"}, indent=2))


if __name__ == "__main__":
    main()
