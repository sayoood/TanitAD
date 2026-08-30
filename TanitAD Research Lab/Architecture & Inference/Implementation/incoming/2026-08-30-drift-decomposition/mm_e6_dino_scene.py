"""MM-E6 step 1 — FROZEN DINOv3 scene features for the HELD-OUT drift corpus.

⭐ WHY AN INDEPENDENT ENCODER IS THE WHOLE POINT (PREREG_DRIFT_DECOMPOSITION.md).
``P`` must project z_t onto "what the scene explains". If the scene representation
came from one of OUR trained encoders it could inherit the very pathology the
probe is trying to isolate, and a high ``r_env`` would then be unreadable. DINOv3
is frozen, externally trained, and never saw our objective.

⚠️ THE BANKED BANK IS THE WRONG CORPUS — this is not a duplicate of
``build_dino_targets.py``. That one covers ``slotprobe-lead130``, which
``v7tiny_g2.py`` documents as **IN-SAMPLE for every v7-tiny arm**. The drift
instrument reads ``physicalai-val130-heldout``, so the scene bank must too, or
the probe would decompose held-out latents against in-sample scene features.

GEOMETRY — the corpus is **256x640 CYLINDRICAL** (HFOV 120 deg, f_ref 305.58).
Frames are already at that size, so the resize is a no-op guard, not a rescale.

⛔ bf16 NEVER fp16 for the FORWARD (a DINOv3 fp16 forward returns ALL-NaN
silently and prints DONE on garbage); fp16 is STORAGE ONLY, after a finite-check.
⛔ Content-asserted per clip before writing: finite AND non-zero mean. The
E-DETECT-1 all-zero-floor failure is the reason a bank is never trusted by
presence.
⚠️ The frame buffer is named ``jpeg_buf`` but its ``codec`` field says ``png`` on
this corpus. PIL sniffs the magic bytes, so decoding is codec-agnostic here — but
the decoded frames are asserted non-zero anyway.

TIER: T0-DIAGNOSTIC · MEASURED (ours; dev-box RTX 4060).
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OLD = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
           r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
           r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
WORK = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("MME6_CORPUS",
                            str(OLD / "sp2/cache/physicalai-val130-heldout")))
OUTDIR = Path(os.environ.get("MME6_DINO", str(WORK / "dino_heldout_4x8")))
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# ⭐ GRID. 4x8 was the readout-cell granularity of the programme's banked bank; MM-E6
# v2 uses the FULL 16x40 TOKEN GRID, approved 2026-08-30. Why: the 4x8 pooling is
# 4 azimuth bins over 120 deg = 30 deg/bin, and MEASURED, a 4x8 DINOv3 scene rep is
# BEATEN BY A RAW 32x80 PIXEL FLOOR at predicting drift (+0.0387 vs +0.0722) while
# still beating it on speed — drift is a spatial question and the pooling destroyed
# the spatial detail. At GH=16, GW=40 the pooling reshape below becomes the identity
# (rows//GH = cols//GW = 1), so this is a parameter change, not a code path change.
GH = int(os.environ.get("MME6_GH", "4"))
GW = int(os.environ.get("MME6_GW", "8"))
H, W, PATCH = 256, 640, 16
F = int(os.environ.get("MME6_FRAMES", "100"))   # matches the drift instrument
BATCH = int(os.environ.get("MME6_DINO_BATCH", "4"))
LIMIT = int(os.environ.get("MME6_NCLIPS_ALL", "0"))   # 0 = every clip


def main() -> int:
    import truststore
    truststore.inject_into_ssl()
    from transformers import AutoImageProcessor, DINOv3ViTModel

    # ⛔⛔ `CUDA_VISIBLE_DEVICES=""` DOES NOT DISABLE CUDA. MEASURED 2026-08-30:
    # `is_available()` True, `device_count()` 0, and `.to("cuda")` STILL lands on
    # cuda:0 — so the usual availability guard put this extraction on a GPU another
    # session was holding while the operator believed it was on CPU. Only
    # `CUDA_VISIBLE_DEVICES=-1` returns False. The device is decided from an
    # EXPLICIT flag here, never from availability alone.
    if os.environ.get("MME6_CPU", "0") == "1":
        # ⛔ VERIFY ISOLATION BY ATTEMPTING AN ALLOCATION THAT MUST RAISE — never by
        # reading device_count(), which reads 0 under the very setting that still
        # lets .to("cuda") succeed (MM-C7).
        try:
            torch.zeros(1).to("cuda")
            print("  [FATAL] CPU isolation requested but a CUDA allocation "
                  "SUCCEEDED — set CUDA_VISIBLE_DEVICES=-1; refusing to run")
            raise SystemExit(2)
        except RuntimeError:
            pass
        dev = torch.device("cpu")
    elif torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    # ⛔ bf16 on CUDA; fp32 (NEVER fp16) on CPU, where bf16 kernels are absent.
    fdt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    proc = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    model = DINOv3ViTModel.from_pretrained(
        MODEL, dtype=fdt, local_files_only=True).to(dev).eval()
    rows, cols = H // PATCH, W // PATCH
    n_patch = rows * cols
    OUTDIR.mkdir(parents=True, exist_ok=True)
    clips = sorted(CACHE.glob("*.v2ep.pt"))
    if LIMIT:
        clips = clips[:LIMIT]
    if not clips:
        print(f"  [FATAL] no clips in {CACHE}")
        return 2

    meta = {"_evidence_class": "MEASURED (ours; frozen DINOv3 ViT-L/16, bf16 fwd)",
            "eval_tier": "T0-DIAGNOSTIC", "model": MODEL, "device": str(dev),
            "corpus": str(CACHE), "corpus_split": "HELD-OUT",
            "grid": [GH, GW], "token_grid": [rows, cols], "d_model": 1024,
            "frames_per_clip": F, "dtype_stored": "float16",
            "forward_dtype": str(fdt), "n_clips": len(clips), "clips": {}}
    print(f"\n  MM-E6 · frozen DINOv3 scene bank -> {OUTDIR}")
    print(f"  {len(clips)} clips from {CACHE.name} · {F} frames each · "
          f"{rows}x{cols} tokens -> {GH}x{GW} cells x 1024\n", flush=True)

    for ci, c in enumerate(clips, 1):
        dst = OUTDIR / f"{c.stem}.npy"
        d = torch.load(c, map_location="cpu", weights_only=False)
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
        m = min(len(off) - 1, F)
        # ⚠️ A CACHE HIT IS CHECKED BY ROW COUNT, NOT BY PRESENCE. A bank written
        # by an earlier smoke with a smaller F is a SHORT bank that would be
        # silently reused and would truncate every clip's rows. MEASURED here:
        # two clips banked at 8 frames were reported as valid for an F=24 run.
        if dst.is_file():
            A = np.load(dst, mmap_mode="r")
            if int(A.shape[0]) >= m:
                meta["clips"][c.stem] = {"frames": int(A.shape[0]), "cached": True}
                continue
            print(f"    [{ci}] {c.stem[:12]} re-extract: banked {A.shape[0]} "
                  f"rows < {m} needed", flush=True)
            del A
        out = []
        with torch.no_grad():
            for s in range(0, m, BATCH):
                ims = [Image.open(io.BytesIO(raw[off[j]:off[j + 1]])).convert("RGB")
                       for j in range(s, min(s + BATCH, m))]
                if s == 0 and float(np.asarray(ims[0]).mean()) == 0.0:
                    raise SystemExit(f"[FATAL] {c.stem} frame 0 decoded all-zero")
                ims = [im if im.size == (W, H) else im.resize((W, H)) for im in ims]
                inp = proc(images=ims, return_tensors="pt", do_resize=False)
                inp = {k: (v.to(dev, fdt) if v.dtype.is_floating_point
                           else v.to(dev)) for k, v in inp.items()}
                tok = model(**inp).last_hidden_state[:, -n_patch:].float()
                b, _, dd = tok.shape
                t = tok.reshape(b, GH, rows // GH, GW, cols // GW, dd).mean(dim=(2, 4))
                out.append(t.reshape(b, GH * GW, dd).cpu().numpy())
        A = np.concatenate(out)
        # ⛔ CONTENT ASSERTION — presence is not evidence.
        if not np.isfinite(A).all() or float(np.abs(A).mean()) == 0.0:
            raise SystemExit(f"[FATAL] non-finite / all-zero DINOv3 on {c.stem} "
                             f"-- the silent bf16/fp16 NaN mode; bank NOT written")
        np.save(dst, A.astype(np.float16))
        meta["clips"][c.stem] = {"frames": int(A.shape[0]),
                                 "abs_mean": round(float(np.abs(A).mean()), 5)}
        if ci % 5 == 0 or ci == len(clips):
            print(f"    [{ci}/{len(clips)}] {c.stem[:12]} {A.shape} "
                  f"|mean| {float(np.abs(A).mean()):.4f}", flush=True)
        del A, out

    (OUTDIR / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tot = sum(v.get("frames", 0) for v in meta["clips"].values())
    short = [k for k, v in meta["clips"].items() if v.get("frames", 0) < F]
    print(f"\n  banked {len(meta['clips'])} clips, {tot} frames -> {OUTDIR}")
    if short:
        print(f"  ⚠️ {len(short)} clips shorter than {F} frames: {short[:5]}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
