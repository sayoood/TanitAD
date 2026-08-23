"""Extract frozen DINOv3 ViT-L/16 patch fields for the 130-clip probe cache.

⭐ WHY THIS EXISTS. E-ACTSTREAM-1 measured action-as-token vs action-as-broadcast
on v6 CELL fields — **16 tokens x 128 d**. REF-A v1's real geometry is
**640 tokens x 1024 d**. The transfer is NOT obvious and could plausibly flip:

    at 16 tokens,  2 action tokens are 11 %   of the stream
    at 640 tokens, 2 action tokens are 0.3 %  of the stream

The broadcast scheme reaches EVERY token by construction; the token scheme
relies on attention finding 2 tokens among 642. A result that holds at 11 %
saying nothing about 0.3 % is exactly the scope error this programme keeps
retracting, so it is measured rather than assumed.

⛔ PATCH TOKENS ONLY, NEVER CLS/POOLED — change #5 of the v1 design, from
DINO-WM's own ablation (global R3M / ResNet18 / DINOv2 CLS "significantly
degrades"). Taking the pooled output here would silently test a different
representation from the one v1 uses.

⚠️ FRAME GEOMETRY IS INHERITED, NOT CHOSEN: the same clips, the same stride 4,
and the same frame indices as `cache_s16000`, so the DINOv3 run and the v6-cell
run sit on identical windows and the two results are comparable.

⚠️ The token is read in place from the environment and never written to a file,
an argument or a log.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
OUT = SP / "dinov3_fields"
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
H, W = 256, 640
STRIDE = 4


def main() -> int:
    import truststore
    truststore.inject_into_ssl()
    from PIL import Image
    from transformers import AutoImageProcessor, DINOv3ViTModel

    tok = os.environ.get("HF_TOKEN") or None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(MODEL, token=tok)
    # ⛔ bf16, NOT fp16. MEASURED 2026-08-19: a DINOv3 ViT-L/16 forward in
    # float16 returns ALL-NaN — and it does so SILENTLY, so a full 130-clip /
    # 7.4 GB extraction completed, printed "DONE", and was entirely garbage.
    # bf16 absmax 15.75 against fp32's 15.64 on the same frames, so bf16 is
    # both safe and faithful here.
    model = DINOv3ViTModel.from_pretrained(
        MODEL, token=tok, dtype=torch.bfloat16).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[dino] loaded {MODEL} in {time.time()-t0:.0f}s  "
          f"{n_par/1e6:.1f}M params  dev={dev}", flush=True)

    # the frame indices the v6 cache used, so the windows match exactly
    lat = torch.load(SP / "sp2/cache_s16000/latents.pt", map_location="cpu",
                     weights_only=False)
    want: dict[str, list[int]] = {}
    for r in lat["rows"]:
        want.setdefault(r["clip_id"], []).append(int(r["frame_idx"]))
    for k in want:
        want[k] = sorted(want[k])
    total = sum(len(v) for v in want.values())
    print(f"[dino] {len(want)} clips, {total} frames (stride {STRIDE})", flush=True)

    OUT.mkdir(exist_ok=True)
    meta = {"model": MODEL, "n_params": n_par, "h": H, "w": W,
            "stride": STRIDE, "patch_tokens_only": True,
            "source_cache": "cache_s16000", "clips": {}}
    done = 0
    for ci, (cid, idxs) in enumerate(sorted(want.items())):
        dst = OUT / f"{cid}.npy"
        if dst.exists():
            # ⚠️ the skip path MUST still record the clip. It did not, and
            # meta.json came out with 69 of 130 clips — so the experiment
            # silently ran on half the corpus and reported no error.
            import numpy as _np
            meta["clips"][cid] = {"frames": idxs,
                                  "shape": list(_np.load(dst, mmap_mode="r").shape)}
            done += len(idxs)
            continue
        pt = EPS / f"{cid}.v2ep.pt"
        if not pt.exists():
            continue
        o = torch.load(pt, map_location="cpu", weights_only=False)
        buf = o["jpeg_buf"].numpy().tobytes()
        lens = o["jpeg_len"].tolist()
        offs = [0]
        for L in lens:
            offs.append(offs[-1] + int(L))
        fields = []
        B = 8
        for i in range(0, len(idxs), B):
            grp = idxs[i:i + B]
            ims = [Image.open(io.BytesIO(buf[offs[j]:offs[j + 1]])).convert("RGB")
                   .resize((W, H)) for j in grp]
            inp = proc(images=ims, return_tensors="pt", do_resize=False)
            inp = {k: (v.to(dev, torch.bfloat16) if v.dtype.is_floating_point
                       else v.to(dev)) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp).last_hidden_state          # [B, 1+reg+N, d]
            # ⛔ patch tokens only: drop CLS and any register tokens by taking
            # the LAST (H/16 * W/16) positions, which is where the patch grid
            # lives regardless of how many registers the checkpoint carries.
            n_patch = (H // 16) * (W // 16)
            out = out[:, -n_patch:]
            f32 = out.float()
            # ⛔ VERIFY BY CONTENT, NEVER BY EXISTENCE. The fp16 failure above
            # produced a full-size, correctly-shaped, entirely-NaN artifact that
            # every existence check passed. This is the check that would have
            # caught it in the first batch instead of after 7.4 GB.
            if not torch.isfinite(f32).all():
                raise RuntimeError(
                    f"NON-FINITE DINOv3 output on clip {cid} batch {i}: "
                    f"nan={torch.isnan(f32).any().item()} "
                    f"inf={torch.isinf(f32).any().item()} — refusing to bank")
            fields.append(f32.cpu().numpy().astype(np.float16))
        arr = np.concatenate(fields, 0)
        np.save(dst, arr)
        meta["clips"][cid] = {"frames": idxs, "shape": list(arr.shape)}
        done += len(idxs)
        if ci % 10 == 0:
            el = time.time() - t0
            print(f"[dino] {ci+1}/{len(want)} clips  {done}/{total} frames  "
                  f"{el/60:.1f} min  {done/max(el,1):.1f} fr/s  "
                  f"shape {arr.shape}", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    gb = sum(f.stat().st_size for f in OUT.glob("*.npy")) / 1e9
    print(f"[dino] DONE {done} frames, {gb:.1f} GB -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
