"""H5b — is refcv5-v2's low participation ratio a TRAINED collapse, or generic to this CNN?

Same frames, same architecture, three encoders, participation ratio (sigma^2) of the
stride-32 tokens and of the frame-pooled vector:
  * trained   : refcv5-v2 ckpt_40284 `core.encoder.*`, strict load, eval()
  * rand_bnEval: random init (PyTorch defaults), eval() -> BN with init stats (identity)
  * rand_bnBatch: random init, BN on batch statistics (train mode, no grad)
Controls that must read known values: isotropic Gaussian (-> ~d), rank-1 (-> 1.0).
Frames: 5 stacked rows (evenly spaced) from each of the first 40 eval clips (sorted),
decoded by the trainer's own `_decode_stacked`, uint8 -> float32 / 255 (CPU).
CPU only. No clip ids are written: only aggregates.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:/Users/Admin/tanitad-snap-20260915/stack")
from tanitad.refs import refc  # noqa: E402
from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets  # noqa: E402

torch.set_num_threads(6)
torch.manual_seed(0)
V2EP = Path(r"C:/Users/Admin/refcv5cmp/data/eval")
CKPT = Path(r"C:/Users/Admin/hf-refcv5v2/ckpt_40284.pt")
N_CLIPS, PER_CLIP, BATCH = 40, 5, 10


def enc_cfg():
    return refc.CNNEncoderConfig(in_channels=9, image_size=256, image_width=640,
                                 base_width=88, blocks=(3, 6, 16, 6))


def pr(f):
    f = f.astype(np.float64)
    f = f - f.mean(0)
    lam = np.clip(np.linalg.eigvalsh(f.T @ f / (f.shape[0] - 1)), 0, None)[::-1]
    return {"participation_ratio": round(float(lam.sum() ** 2 / (lam ** 2).sum()), 3),
            "top1_energy_share": round(float(lam[0] / lam.sum()), 4)}


# ---- frames ------------------------------------------------------------------
frames = []
for p in sorted(V2EP.glob("*.v2ep.pt"))[:N_CLIPS]:
    d = torch.load(p, map_location="cpu", weights_only=False)
    offs = _jpeg_offsets(d["jpeg_len"])
    ns = int(d["n_stack"])
    n_rows = len(d["jpeg_len"]) - (ns - 1)
    for j in np.linspace(0, n_rows - 1, PER_CLIP).round().astype(int):
        frames.append(_decode_stacked(d["jpeg_buf"], offs, ns, int(j), int(j) + 1, str(d["codec"])))
x_u8 = torch.cat(frames, 0)
assert x_u8.shape[1:] == (9, 256, 640), x_u8.shape
div = torch.full((), 255.0, dtype=torch.float32)

# ---- encoders ----------------------------------------------------------------
trained = refc.ResNetEncoder(enc_cfg())
ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
sd = {k[len("core.encoder."):]: v for k, v in ck["model"].items() if k.startswith("core.encoder.")}
trained.load_state_dict(sd, strict=True)
del ck
rand = refc.ResNetEncoder(enc_cfg())


def run(enc, bn_batch):
    enc.eval()
    if bn_batch:
        for m in enc.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    toks, pooled = [], []
    with torch.no_grad():
        for a in range(0, x_u8.shape[0], BATCH):
            fmap, pool = enc(x_u8[a:a + BATCH].float().div_(div))
            toks.append(fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).numpy())
            pooled.append(pool.numpy())
    t, pl = np.concatenate(toks), np.concatenate(pooled)
    return {"token_level": pr(t), "frame_pooled": pr(pl),
            "finite": bool(np.isfinite(t).all()), "max_abs": round(float(np.abs(t).max()), 3)}


out = {"n_frames": int(x_u8.shape[0]), "n_tokens": int(x_u8.shape[0] * 160), "d": 704,
       "trained": run(trained, False),
       "rand_bnEval": run(rand, False),
       "rand_bnBatch": run(refc.ResNetEncoder(enc_cfg()), True)}
rng = np.random.default_rng(0)
out["ctrl_isotropic_token_shape"] = pr(rng.standard_normal((out["n_tokens"], 704)).astype(np.float32))
u = rng.standard_normal((1, 704))
out["ctrl_rank1"] = pr(rng.standard_normal((out["n_frames"], 1)) @ u
                       + 1e-3 * rng.standard_normal((out["n_frames"], 704)))
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
