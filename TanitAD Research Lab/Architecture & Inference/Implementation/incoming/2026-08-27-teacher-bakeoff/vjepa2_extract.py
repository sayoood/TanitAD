"""Extract frozen V-JEPA2 ViT-L patch fields for the 130-clip probe cache.

P2 of the parallel plan (PI, 2026-08-27): the fallback-teacher bake-off needs
V-JEPA2 measured on the SAME rig as the banked DINOv3 columns (E-DEC-29 family).
This mirrors `dinov3_extract.py` byte-for-byte in its discipline:

  * SAME clips, SAME frame indices (from `cache_s16000`'s rows) — identical
    windows, so the two teachers are paired, not merely compared;
  * PATCH TOKENS ONLY (the DINO-WM lesson: pooled/CLS silently tests a
    different representation);
  * bf16, never fp16 (the DINOv3 fp16 forward returned ALL-NaN silently —
    7.4 GB of garbage passed every existence check);
  * VERIFY BY CONTENT on every batch (isfinite), and the skip path still
    records the clip in meta (the 69-of-130 silent-half-corpus bug).

V-JEPA2 DIFFERENCE, stated: it is a VIDEO model (tubelet 2 frames -> 1 temporal
slice). Each sample is the frame PAIRED WITH ITS PREDECESSOR (t-1, t), giving
one temporal slice whose spatial grid is the per-frame field. Provenance:
official facebook/vjepa2-vitl-fpc64-256 (an official 2.1 repo does NOT exist on
HF as of 2026-08-27 — two probes; third-party re-uploads rejected).

⚠️ GEOMETRY IS PROBED, NOT ASSUMED: the first batch asserts the token count
equals 16x40 (256x640 / 16) x 1 slice; anything else refuses before banking.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
OLDSP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
             r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
             r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
EPS = OLDSP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
OUT = SP / "vjepa2_fields"
MODEL_DIR = SP / "vjepa2_vitl"
H, W = 256, 640
N_PATCH = (H // 16) * (W // 16)          # 640 spatial positions


def main() -> int:
    import truststore
    truststore.inject_into_ssl()
    from PIL import Image
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[vjepa2] loaded in {time.time()-t0:.0f}s  {n_par/1e6:.1f}M  dev={dev}",
          flush=True)

    lat = torch.load(OLDSP / "sp2/cache_s16000/latents.pt", map_location="cpu",
                     weights_only=False)
    want: dict[str, list[int]] = {}
    for r in lat["rows"]:
        want.setdefault(r["clip_id"], []).append(int(r["frame_idx"]))
    for k in want:
        want[k] = sorted(want[k])
    total = sum(len(v) for v in want.values())
    print(f"[vjepa2] {len(want)} clips, {total} frames", flush=True)

    OUT.mkdir(exist_ok=True)
    meta = {"model": "facebook/vjepa2-vitl-fpc64-256 (local selective snapshot)",
            "n_params": n_par, "h": H, "w": W, "patch_tokens_only": True,
            "temporal": "tubelet pair (t-1, t) -> 1 slice; spatial grid banked",
            "source_cache": "cache_s16000", "clips": {}}
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    done, checked_geom = 0, False
    for ci, (cid, idxs) in enumerate(sorted(want.items())):
        dst = OUT / f"{cid}.npy"
        if dst.exists():
            meta["clips"][cid] = {"frames": idxs,
                                  "shape": list(np.load(dst, mmap_mode="r").shape)}
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

        def img(j):
            return (torch.from_numpy(np.asarray(
                Image.open(io.BytesIO(buf[offs[j]:offs[j + 1]]))
                .convert("RGB").resize((W, H)))).float() / 255.0)

        fields = []
        B = 4
        for i in range(0, len(idxs), B):
            grp = idxs[i:i + B]
            vids = torch.stack([
                torch.stack([img(max(j - 1, 0)), img(j)])       # [T=2, H, W, 3]
                for j in grp])                                   # [b, 2, H, W, 3]
            vids = vids.permute(0, 1, 4, 2, 3)                   # [b, T, C, H, W]
            vids = ((vids - mean) / std).to(dev, torch.bfloat16)
            with torch.no_grad():
                out = model(pixel_values_videos=vids).last_hidden_state
            if not checked_geom:
                print(f"[vjepa2] first-batch token grid: {tuple(out.shape)} "
                      f"(want [b, {N_PATCH}, d] for one temporal slice)",
                      flush=True)
                if out.shape[1] != N_PATCH:
                    raise RuntimeError(
                        f"GEOMETRY REFUSED: {out.shape[1]} tokens != {N_PATCH} "
                        f"— the model did not keep the 256x640 grid; do NOT "
                        f"bank a mismatched field")
                checked_geom = True
            f32 = out.float()
            if not torch.isfinite(f32).all():
                raise RuntimeError(f"NON-FINITE output on {cid} batch {i} — "
                                   f"refusing to bank (the DINOv3 fp16 lesson)")
            fields.append(f32.cpu().numpy().astype(np.float16))
        arr = np.concatenate(fields, 0)
        np.save(dst, arr)
        meta["clips"][cid] = {"frames": idxs, "shape": list(arr.shape)}
        done += len(idxs)
        if ci % 10 == 0:
            el = time.time() - t0
            print(f"[vjepa2] {ci+1}/{len(want)}  {done}/{total} frames  "
                  f"{el/60:.1f} min  {done/max(el,1):.1f} fr/s", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    gb = sum(f.stat().st_size for f in OUT.glob("*.npy")) / 1e9
    print(f"[vjepa2] DONE {done} frames, {gb:.1f} GB -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
