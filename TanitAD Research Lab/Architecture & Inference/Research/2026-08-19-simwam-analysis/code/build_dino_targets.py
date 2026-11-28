"""E-DEC-8(a) step 1: precompute FROZEN DINOv3 targets for the training corpus.

WHY (E-DEC-7): every term in our objective has a SELF-GENERATED target, so
"ego motion + noise" satisfies the whole objective with zero scene content. The
fix is a target the model CANNOT choose. Frozen DINOv3 is that target, it is
DINO-WM's actual recipe (banked 2411.04983: frozen patch features + latent L2),
and we have PAIRED evidence it carries what we lack (n_agents +0.2754 against our
best -0.81; and it beats us on speed, t -2.72).

Stored at the READOUT-CELL granularity that the distillation head will predict:
patch tokens 16x40 mean-pooled to 4x8 = 32 cells x 1024 dims, fp16, per clip.
~25.8k frames over 130 clips => ~1.7 GB.

⛔ bf16 NEVER fp16 for the FORWARD (a DINOv3 fp16 forward returns ALL-NaN
silently and a full extraction prints DONE on garbage) -- fp16 is used only for
STORAGE, after a finite-check.
⛔ Content-asserted per clip before writing: finite, non-zero mean. A poisoned
target bank would be worse than no bank, because a distillation loss against
zeros trains the encoder toward zero (the E-DETECT-1 all-zero-floor failure).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
CACHE = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
OUTDIR = SP / "dino_targets_4x8"
GH, GW = 4, 8
H, W, PATCH = 256, 640, 16


def main() -> int:
    import truststore
    truststore.inject_into_ssl()
    from transformers import AutoImageProcessor, DINOv3ViTModel

    M = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dev = torch.device("cuda")
    proc = AutoImageProcessor.from_pretrained(M, local_files_only=True)
    model = DINOv3ViTModel.from_pretrained(M, dtype=torch.bfloat16,
                                           local_files_only=True).to(dev).eval()
    rows, cols = H // PATCH, W // PATCH
    n_patch = rows * cols
    OUTDIR.mkdir(exist_ok=True)
    clips = sorted(CACHE.glob("*.v2ep.pt"))
    meta = {"_evidence_class": "MEASURED (ours; frozen DINOv3 ViT-L/16, bf16 forward)",
            "model": M, "grid": [GH, GW], "token_grid": [rows, cols],
            "dtype_stored": "float16", "n_clips": len(clips), "clips": {}}
    print(f"\n  DINOv3 targets -> {OUTDIR}  ({rows}x{cols} tokens -> {GH}x{GW} cells)\n", flush=True)

    for ci, c in enumerate(clips, 1):
        cid = torch.load(c, map_location="cpu", weights_only=False)["clip_id"]
        dst = OUTDIR / f"{cid}.npy"
        if dst.is_file():
            meta["clips"][cid] = {"cached": True}
            continue
        d = torch.load(c, map_location="cpu", weights_only=False)
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
        m = len(off) - 1
        out = []
        with torch.no_grad():
            for s in range(0, m, 8):
                ims = [Image.open(io.BytesIO(raw[off[j]:off[j + 1]])).convert("RGB").resize((W, H))
                       for j in range(s, min(s + 8, m))]
                inp = proc(images=ims, return_tensors="pt", do_resize=False)
                inp = {k: (v.to(dev, torch.bfloat16) if v.dtype.is_floating_point else v.to(dev))
                       for k, v in inp.items()}
                tok = model(**inp).last_hidden_state[:, -n_patch:].float()   # [b, 640, 1024]
                b, _, dd = tok.shape
                t = tok.reshape(b, GH, rows // GH, GW, cols // GW, dd).mean(dim=(2, 4))
                out.append(t.reshape(b, GH * GW, dd).cpu().numpy())
        A = np.concatenate(out)
        if not np.isfinite(A).all() or float(np.abs(A).mean()) == 0.0:
            raise SystemExit(f"[FATAL] non-finite / all-zero DINOv3 on {cid} "
                             f"-- the silent fp16 NaN mode; bank NOT written")
        np.save(dst, A.astype(np.float16))
        meta["clips"][cid] = {"frames": int(A.shape[0]), "abs_mean": round(float(np.abs(A).mean()), 5)}
        if ci % 10 == 0 or ci == len(clips):
            print(f"    [{ci}/{len(clips)}] {cid[:12]} {A.shape} |mean| "
                  f"{float(np.abs(A).mean()):.4f}", flush=True)
        del A, out
    (OUTDIR / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tot = sum(v.get("frames", 0) for v in meta["clips"].values())
    print(f"\n  banked {len(meta['clips'])} clips, {tot} frames -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
