"""E-DEC-8(a): does an EXTERNAL target make the encoder carry the SCENE?

THE QUESTION (E-DEC-7). Every term in our objective has a SELF-GENERATED target,
so "ego motion + noise" satisfies all of it with zero scene content -- which
reproduces every number measured this campaign. The proposed remedy is a target
the model cannot choose. Before paying for trainer integration, run the CHEAPEST
DISCRIMINATING EXPERIMENT: train the SAME encoder+readout architecture on nothing
but distillation into frozen DINOv3 cells, then probe it for ENVIRONMENT content.

    encoder(9ch stack) -> tokens 16x40 -> pool 4x8 -> head -> 1024
    loss = MSE against frozen DINOv3's own 4x8 cells for that frame

  environment decodability RISES above the constant control
      => an external target is what was missing; integrating O7 into the trainer
         is justified, and E-DEC-7 is the right root cause.
  it does NOT rise
      => distillation is not the lever either, and the gap is elsewhere. Cheaply.

⚠️ This arm is DELIBERATELY NOT A WORLD MODEL -- no O5, no O6, no actions. It
answers "can an external target put the scene into THIS encoder", nothing else,
and must never be quoted as a world-model result.

⛔ Targets are content-asserted on load (finite, non-zero): a distillation loss
against an all-zero bank trains the encoder TOWARD ZERO and would look like
convergence (the E-DETECT-1 all-zero-floor failure).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
CACHE = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
TGT = SP / "dino_targets_4x8"
OUTD = SP / "v7tiny_distill"
GH, GW, H, W, PATCH = 4, 8, 256, 640, 16


def main() -> int:
    from tanitad.models.v6 import V6Config, V6Stack, EncoderConfig, ReadoutConfig, PredictorConfig

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    dev = torch.device("cuda")
    OUTD.mkdir(exist_ok=True)

    clips = sorted(CACHE.glob("*.v2ep.pt"))
    ready = {p.stem for p in TGT.glob("*.npy")}
    usable = []
    for c in clips:
        d = torch.load(c, map_location="cpu", weights_only=False)
        if d["clip_id"] in ready:
            usable.append((c, d["clip_id"]))
    print(f"\n  E-DEC-8(a) distillation · {len(usable)} clips with banked DINOv3 targets", flush=True)
    assert len(usable) >= 8, f"only {len(usable)} clips banked — wait for the target bank"

    cfg = V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=H, image_width=W, patch_size=PATCH,
                              d_model=128, depth=3, n_heads=4),
        readout=ReadoutConfig(grid=GH, grid_w=GW, d_readout=64),
        predictor=PredictorConfig(d_model=256, depth=3, n_heads=4, window=6,
                                  horizons=(1, 2, 4), action_dim=3),
        d_tac=128, d_str=64, sigreg_slices=512)
    stack = V6Stack(cfg).to(dev)
    # The ENCODER is what is under test, so the head reads its TOKEN dim (128)
    # directly. Routing through the readout's own projection would confound
    # "did the encoder learn the scene" with "did the readout learn to expose it".
    head = torch.nn.Sequential(torch.nn.Linear(128, 512), torch.nn.GELU(),
                               torch.nn.Linear(512, 1024)).to(dev)
    params = list(stack.encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    n_enc = sum(p.numel() for p in stack.encoder.parameters())
    print(f"  encoder {n_enc / 1e6:.2f}M params · cells {GH}x{GW} · head 128->1024\n", flush=True)

    cacheA: dict[str, np.ndarray] = {}
    cacheI: dict[str, list] = {}

    def load_clip(path, cid):
        if cid not in cacheA:
            A = np.load(TGT / f"{cid}.npy")
            assert np.isfinite(A).all() and float(np.abs(A).mean()) > 0, \
                f"[FATAL] poisoned target bank for {cid}"
            cacheA[cid] = A
            d = torch.load(path, map_location="cpu", weights_only=False)
            raw = d["jpeg_buf"].numpy().tobytes()
            off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
            cacheI[cid] = [(raw, off)]
            if len(cacheA) > 10:
                k = next(iter(cacheA))
                cacheA.pop(k), cacheI.pop(k)
        return cacheA[cid], cacheI[cid][0]

    rng = np.random.default_rng(0)
    log = (OUTD / "train_log.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    B = 6
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(usable), B)
        ims, tg = [], []
        for k in idx:
            path, cid = usable[k]
            A, (raw, off) = load_clip(path, cid)
            m = min(len(A), len(off) - 1)
            j = int(rng.integers(2, m))
            fr = []
            for q in (j - 2, j - 1, j):
                im = Image.open(io.BytesIO(raw[off[q]:off[q + 1]])).convert("RGB")
                fr.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0)
            ims.append(torch.cat(fr, 0))
            tg.append(torch.from_numpy(A[j].astype(np.float32)))
        x = torch.stack(ims)[:, None].to(dev)
        y = torch.stack(tg).to(dev)                      # [B, 32, 1024]
        _z, tok = stack.encode_window(x, return_tokens=True)
        t = tok[:, 0]                                     # [B, 640, 128]
        b, n, dd = t.shape
        rows, cols = H // PATCH, W // PATCH
        cells = t.reshape(b, GH, rows // GH, GW, cols // GW, dd).mean(dim=(2, 4))
        cells = cells.reshape(b, GH * GW, dd)
        pred = head(cells)                                # [B, 32, 1024]
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "elapsed_s": round(time.time() - t0, 1),
                   "target_abs_mean": round(float(y.abs().mean()), 4)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"    [{step:>5}] loss {float(loss):.5f}  "
                  f"({(time.time() - t0) / step:.2f} s/step)", flush=True)
    log.close()
    torch.save({"model": stack.state_dict(), "head": head.state_dict(),
                "cfg": cfg, "step": steps}, OUTD / "ckpt.pt")
    print(f"\n  -> {OUTD / 'ckpt.pt'}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
