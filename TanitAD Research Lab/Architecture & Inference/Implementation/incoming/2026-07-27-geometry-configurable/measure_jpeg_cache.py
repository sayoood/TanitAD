"""Does the compressed cache make a wide geometry SCHEDULABLE? — MEASURED.

Storage is the binding constraint on v5 (349 GB deployed -> 873 GB at every
640-wide candidate). The v2 compressed cache is the named mitigation. This
measures whether it actually pays, on REAL PhysicalAI clips, and decomposes the
win so it cannot be over-claimed:

  raw epcache      [T, 9, H, W] uint8   (D-015 STACKED: each frame stored 3x)
  v2 unstacked     [T+2, 3, H, W]       lossless -> ~2.97x on its own
  v2 + codec       JPEG q / PNG         the compression on top

⚠️ THE COST LANDS ON A DIFFERENT AXIS THAN THE BUILD. The rebuild is mp4-decode
bound; the codec's cost is paid by the TRAINING DATALOADER, once per window
served, forever. Both are measured separately here — conflating them is how a
"decode-bound anyway" argument would be wrong.

Fidelity: PSNR / max-abs / mean-abs of the codec round-trip against the exact
pixels the encoder would otherwise see, per quality setting, plus a LOSSLESS
option costed alongside.

⚠️ Local clips are the NON-PARITY selection (`14231cd29c74`) — used to measure
per-clip cost only, never to build a corpus. No clip UUID is emitted.

Usage: python measure_jpeg_cache.py --root <physicalai root> --clips 3 --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

# Codec backend. torchvision.io is what `v2_compressed.py` uses in production;
# it is absent from the dev-box venv, so PIL is the fallback. Both sit on the
# SAME libjpeg-turbo / libpng, so SIZE and FIDELITY are backend-independent —
# only wall-clock carries PIL's extra Python overhead, which is why the timing
# is cross-checked against torchvision on a pod and labelled either way.
try:
    import torchvision.io as tvio
    BACKEND = "torchvision"
except ModuleNotFoundError:
    tvio = None
    BACKEND = "PIL"
import io as _io

from PIL import Image

STACK = Path(__file__).resolve().parents[5] / "stack"
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

from tanitad.data import calib as C                              # noqa: E402
from tanitad.data.physicalai import (discover_r0_clips,          # noqa: E402
                                     intrinsics_for_clip)

RAW_EPISODE_MB = 117.384          # MEASURED: a real local ep_*.pt, T=199, 256px
STORED_FRAMES = 199
DECODED_FRAMES = 605


def candidates():
    F = C.CanonicalFrame
    return [
        ("deployed_256sq_51deg", C.CANONICAL_256, "ftheta_crop"),
        ("100deg_256x640_cyl", F.from_hfov(100.0, 256, 640, "cylindrical"),
         "cylindrical"),
        ("120deg_256x640_cyl", F.from_hfov(120.0, 256, 640, "cylindrical"),
         "cylindrical"),
        ("120deg_384x960_cyl", F.from_hfov(120.0, 384, 960, "cylindrical"),
         "cylindrical"),
    ]


def codecs():
    return [("jpeg_q75", "jpeg", 75), ("jpeg_q85", "jpeg", 85),
            ("jpeg_q90", "jpeg", 90), ("jpeg_q95", "jpeg", 95),
            ("png_lossless", "png", None),
            ("webp_lossless", "webp", None)]


def encode(frame_u8: torch.Tensor, kind: str, q):
    if BACKEND == "torchvision":
        if kind == "png":
            return tvio.encode_png(frame_u8)
        if kind == "webp":
            return tvio.encode_webp(frame_u8, quality=100)
        return tvio.encode_jpeg(frame_u8, quality=int(q))
    im = Image.fromarray(frame_u8.permute(1, 2, 0).numpy())
    b = _io.BytesIO()
    if kind == "png":
        im.save(b, format="PNG", optimize=False, compress_level=6)
    elif kind == "webp":
        im.save(b, format="WEBP", lossless=True)
    else:
        im.save(b, format="JPEG", quality=int(q), subsampling="4:2:0")
    return torch.frombuffer(bytearray(b.getvalue()), dtype=torch.uint8)


def decode(buf: torch.Tensor, kind: str):
    if BACKEND == "torchvision":
        if kind == "png":
            return tvio.decode_png(buf, mode=tvio.ImageReadMode.RGB)
        if kind == "webp":
            return tvio.decode_image(buf, mode=tvio.ImageReadMode.RGB)
        return tvio.decode_jpeg(buf, mode=tvio.ImageReadMode.RGB)
    im = Image.open(_io.BytesIO(bytes(buf.numpy()))).convert("RGB")
    import numpy as _np
    return torch.from_numpy(_np.asarray(im)).permute(2, 0, 1).contiguous()


def decode_some(mp4: Path, n: int) -> torch.Tensor:
    import av
    out = []
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = 4
        for fr in c.decode(st):
            out.append(torch.from_numpy(fr.to_ndarray(format="rgb24")
                                        ).permute(2, 0, 1))
            if len(out) >= n:
                break
    return torch.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--window", type=int, default=8, help="trainer window")
    ap.add_argument("--n-stack", type=int, default=3)
    ap.add_argument("--episodes-train", type=int, default=2376)
    ap.add_argument("--episodes-val", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clips = discover_r0_clips(a.root)[:a.clips]
    n_ep = a.episodes_train + a.episodes_val
    rows: list[dict] = []
    clip_meta = []

    for ci, clip in enumerate(clips):
        cid = clip["clip_id"]
        intr = intrinsics_for_clip(cid, a.root)
        native = decode_some(Path(clip["mp4"]), a.frames)
        clip_meta.append({
            "index": ci,
            "clip_sha1_8": hashlib.sha1(cid.encode()).hexdigest()[:8],
            "rig": "B" if intr.cy > 650 else "A",
            "per_clip_calib": bool(intr.per_clip),
            "frames_used": int(native.shape[0])})

        for gname, frame, mode in candidates():
            if mode == "cylindrical":
                canon = C.cylindrical_rectify(native, intr, frame,
                                              require_per_clip=intr.per_clip)
            else:
                canon = C.ftheta_crop_resize(native, intr, frame=frame)
            H, W = int(canon.shape[-2]), int(canon.shape[-1])
            raw_frame_bytes = 3 * H * W

            for cname, kind, q in codecs():
                t0 = time.perf_counter()
                bufs = [encode(canon[i].contiguous(), kind, q)
                        for i in range(canon.shape[0])]
                enc_s = (time.perf_counter() - t0) / canon.shape[0]
                nbytes = sum(int(b.numel()) for b in bufs)

                t0 = time.perf_counter()
                dec = [decode(b, kind) for b in bufs]
                dec_s = (time.perf_counter() - t0) / len(bufs)

                back = torch.stack(dec)
                diff = (back.int() - canon.int()).abs()
                mse = float((diff.float() ** 2).mean())
                psnr = float("inf") if mse == 0 else 10.0 * torch.log10(
                    torch.tensor(255.0 ** 2 / mse)).item()
                rows.append({
                    "clip_index": ci, "geometry": gname, "codec": cname,
                    "frame_hw": [H, W],
                    "bytes_per_frame": nbytes / len(bufs),
                    "raw_frame_bytes": raw_frame_bytes,
                    "codec_ratio_vs_unstacked_raw": raw_frame_bytes
                    / (nbytes / len(bufs)),
                    "encode_s_per_frame": enc_s,
                    "decode_s_per_frame": dec_s,
                    "lossless": bool(kind in ("png", "webp")),
                    "psnr_db": psnr, "max_abs_err": int(diff.max()),
                    "mean_abs_err": float(diff.float().mean()),
                    "exact": bool(int(diff.max()) == 0),
                })

    # ---- aggregate ------------------------------------------------------- #
    def mean(xs):
        return sum(xs) / max(1, len(xs))

    agg = {}
    for r in rows:
        k = (r["geometry"], r["codec"])
        agg.setdefault(k, []).append(r)
    summary = []
    for (g, c), rs in agg.items():
        H, W = rs[0]["frame_hw"]
        bpf = mean([r["bytes_per_frame"] for r in rs])
        # storage per EPISODE: v2 stores UNSTACKED frames (T + n_stack - 1)
        ep_mb = bpf * (STORED_FRAMES + a.n_stack - 1) / 1e6
        raw_stacked_mb = (3 * H * W) * 3 * STORED_FRAMES / 1e6   # 9-ch stacked
        raw_unstacked_mb = (3 * H * W) * (STORED_FRAMES + a.n_stack - 1) / 1e6
        # window-serve cost: a window of `window` stacked rows needs
        # window + n_stack - 1 raw frames decoded
        n_dec = a.window + a.n_stack - 1
        summary.append({
            "geometry": g, "codec": c, "frame_hw": [H, W],
            "lossless": rs[0]["lossless"],
            "psnr_db": round(mean([r["psnr_db"] for r in rs]), 2)
            if not rs[0]["lossless"] else None,
            "max_abs_err": max(r["max_abs_err"] for r in rs),
            "mean_abs_err": round(mean([r["mean_abs_err"] for r in rs]), 4),
            "bit_exact": all(r["exact"] for r in rs),
            "bytes_per_frame": round(bpf, 1),
            "codec_ratio_vs_unstacked_raw": round(
                mean([r["codec_ratio_vs_unstacked_raw"] for r in rs]), 2),
            "episode_mb": round(ep_mb, 2),
            "raw_stacked_episode_mb": round(raw_stacked_mb, 2),
            "unstack_only_saving_x": round(raw_stacked_mb / raw_unstacked_mb, 3),
            "total_saving_vs_raw_stacked_x": round(raw_stacked_mb / ep_mb, 2),
            "corpus_gb": round(ep_mb * n_ep / 1000.0, 1),
            "encode_s_per_frame": round(mean(
                [r["encode_s_per_frame"] for r in rs]), 5),
            "decode_s_per_frame": round(mean(
                [r["decode_s_per_frame"] for r in rs]), 5),
            "window_serve_ms": round(1000 * n_dec * mean(
                [r["decode_s_per_frame"] for r in rs]), 2),
            "build_encode_min_per_1000ep": round(
                mean([r["encode_s_per_frame"] for r in rs])
                * (STORED_FRAMES + a.n_stack - 1) * 1000 / 60.0, 1),
        })
    summary.sort(key=lambda r: (r["geometry"], r["corpus_gb"]))

    out = {
        "artifact": "compressed-cache viability for a wide-FOV v5",
        "date": "2026-07-27",
        "machine": "dev box, CPU only (no GPU -> the WDDM host-RAM spill "
                   "artefact cannot affect any timing here)",
        "codec_backend": BACKEND,
        "backend_note": "sizes and fidelity are backend-independent (same "
                        "libjpeg-turbo/libpng); PIL wall-clock carries extra "
                        "Python overhead and is an UPPER BOUND on the "
                        "torchvision path production uses.",
        "confidentiality": "no clip UUID (index + sha1 prefix only)",
        "anchors": {
            "raw_episode_mb_MEASURED": RAW_EPISODE_MB,
            "stored_frames_per_episode": STORED_FRAMES,
            "decoded_frames_per_episode": DECODED_FRAMES,
            "episodes_train": a.episodes_train, "episodes_val": a.episodes_val,
            "trainer_window": a.window, "n_stack": a.n_stack,
            "note": "v2 stores UNSTACKED frames (T + n_stack - 1) and restacks "
                    "at load, so the ~2.97x from not triplicating each frame is "
                    "LOSSLESS and separate from the codec's contribution.",
        },
        "clips": clip_meta,
        "summary": summary,
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'geometry':22s} {'codec':14s} {'GB':>8s} {'x vs raw':>9s} "
          f"{'PSNR':>7s} {'maxerr':>7s} {'win ms':>7s} {'enc/f ms':>9s}")
    for r in summary:
        print(f"{r['geometry']:22s} {r['codec']:14s} {r['corpus_gb']:8.1f} "
              f"{r['total_saving_vs_raw_stacked_x']:9.2f} "
              f"{(r['psnr_db'] if r['psnr_db'] else 0):7.2f} "
              f"{r['max_abs_err']:7d} {r['window_serve_ms']:7.2f} "
              f"{1000*r['encode_s_per_frame']:9.2f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
