"""RENC library (PREREG_T2_RENC.md, frozen prefix 5,544 bytes, sha256 425d3496...): LoRA on the in-loop DINOv3-L, the loss, the pair data, the merged-weight save. CPU or CUDA, torch + numpy.

Reuses, imports and never edits: t2a_v8f_encoder (InLoopEncoder: rectify -> encode_cyl -> pool), ralign_lib.fold_of (the fold rule), t2v8f_lora_merge (parse_lora / fold_lora / model_digest: the stage-5 merge path).
Decisions the card leaves open (each recorded in TRAIN_RESULT): P = pool4x10 THEN fp8 e4m3 quantisation with a straight-through gradient (the tokens the planner reads; the L arm's own choice);
batch 8 = 8 (render, real) PAIRS = 16 encoder images per optimiser step, run as micro-batches of 4 pairs; constant lr 1e-4, no schedule, no clipping; a partial last batch is kept; A init U(+-1/sqrt(in)) from a generator seeded 0, B = 0.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
for _p in (HERE, f"{T2}/code", f"{T2}/code_lc", f"{T2}/ralign", f"{T2}/gap48", os.environ.get("RENC_PKG_CODE", ""), os.environ.get("RENC_GAP_CODE", "")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import ralign_lib as RL  # noqa: E402
import t2a_v8f_encoder as E  # noqa: E402
import t2v8f_lora_merge as LM  # noqa: E402

CARD_PREFIX_BYTES, CARD_PREFIX_SHA256 = 5544, "425d3496eb52e30025c11c2609409ea1478aa2fe70ba6609c9f9477c128fd722"
R, ALPHA, TARGETS, BLOCKS = 32, 32.0, ("q", "v"), 24
LR, WD, BATCH, EPOCHS, SEED, LAMBDA_KEEP, MICRO_PAIRS = 1e-4, 0.0, 8, 10, 0, 1.0, 4
DATA_STRIDE, GATE_STEP, TOKEN_SHAPE = 2, 10, (40, 1024)
FOLDS = ("A", "B")
_ATTN = re.compile(r"(?:^|.*\.)layer\.(\d+)\.attention$")


def sha256_file(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def jdump(o, p):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, p)


def fold_of(sid):
    return RL.fold_of(sid, 0)


def other_fold(f):
    return RL.other_fold(f)


def data_ticks(n):
    """Every 2nd tick of a scene with n ticks."""
    return list(range(0, int(n), DATA_STRIDE))


def gate_ticks(n):
    """GAP48's ticks (t2v8f_gap48.tick_list): every 10th, or every tick below 20. Must be a subset of data_ticks(n): the caller asserts it."""
    n = int(n)
    return list(range(n)) if n < 20 else list(range(0, n, GATE_STEP))


# ------------------------------------------------------------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B (A x). The adapter branch runs in fp32 (the master copy) and is added in the base output's dtype. B = 0 at init: the branch adds an exact 0."""

    def __init__(self, base, r, alpha, gen):
        super().__init__()
        self.base, self.r, self.scaling = base, int(r), float(alpha) / int(r)
        for p in base.parameters():
            p.requires_grad_(False)
        bound = 1.0 / math.sqrt(base.in_features)
        a = torch.empty(self.r, base.in_features, dtype=torch.float32).uniform_(-bound, bound, generator=gen)
        dev = base.weight.device
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32, device=dev))

    def forward(self, x):
        y = self.base(x)
        return y + (F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scaling).to(y.dtype)


def inject_lora(model, r=R, alpha=ALPHA, targets=TARGETS, seed=SEED):
    """Freeze the model and wrap q_proj / v_proj of every block. -> {(layer, 'q'|'v'): LoRALinear}. Refuses to wrap nothing."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(int(seed))
    out = {}
    for name, mod in list(model.named_modules()):
        m = _ATTN.fullmatch(name)
        if not m:
            continue
        for t in targets:
            lin = getattr(mod, f"{t}_proj")
            assert isinstance(lin, nn.Linear), f"{name}.{t}_proj is not a Linear"
            w = LoRALinear(lin, r, alpha, gen)
            setattr(mod, f"{t}_proj", w)
            out[(int(m.group(1)), t)] = w
    if not out:
        raise ValueError("inject_lora: no attention block found, nothing adapted")
    return out


def lora_params(mods):
    return [p for k in sorted(mods) for p in (mods[k].lora_A, mods[k].lora_B)]


def lora_state(mods, r=R, alpha=ALPHA, targets=TARGETS, blocks=BLOCKS):
    """The format of t2v8f_lora_merge.parse_lora: {meta, tensors}, tensors fp32 named layer.<i>.attention.<q|v>_proj.lora_<A|B>."""
    ten = {}
    for (i, t), w in sorted(mods.items()):
        ten[f"layer.{i}.attention.{t}_proj.lora_A"] = w.lora_A.detach().cpu().float().clone()
        ten[f"layer.{i}.attention.{t}_proj.lora_B"] = w.lora_B.detach().cpu().float().clone()
    return {"meta": {"r": int(r), "alpha": float(alpha), "targets": list(targets), "blocks": int(blocks)}, "tensors": ten}


def set_lora_state(mods, sd):
    for (i, t), w in mods.items():
        w.lora_A.data.copy_(sd["tensors"][f"layer.{i}.attention.{t}_proj.lora_A"])
        w.lora_B.data.copy_(sd["tensors"][f"layer.{i}.attention.{t}_proj.lora_B"])


# ------------------------------------------------------------------------------------------------------------------------ encoders
def make_encoder(snapshot, device="cpu", dtype="bf16"):
    return E.InLoopEncoder(snapshot=snapshot, dtype=dtype, device=device)


def merged_encoder(blob_path, snapshot, device="cpu"):
    """The stage-5 path: the frozen CPU bf16 model, the LoRA folded in by t2v8f_lora_merge.fold_lora BEFORE .to(device). -> (encoder, merged q/v digest dict)."""
    enc = E.InLoopEncoder(snapshot=snapshot, dtype="bf16", device="cpu")
    blob = LM.load_blob(blob_path)
    mt = blob["lora"]["meta"]
    parsed = LM.parse_lora(blob["lora"], r=int(mt["r"]), alpha=float(mt["alpha"]), targets=tuple(mt["targets"]), blocks=int(mt["blocks"]))
    LM.fold_lora(enc.model, parsed, r=int(mt["r"]), alpha=float(mt["alpha"]))
    d = LM.model_digest(enc.model, int(mt["blocks"]))
    dev = torch.device(device)
    enc.model.to(dev)
    enc.mean, enc.std, enc.device = enc.mean.to(dev), enc.std.to(dev), dev
    return enc, d


class _Fp8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.to(torch.float8_e4m3fn).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


def pooled_train(enc, cyl_u8):
    """uint8 [B,3,256,640] -> P(E'(x)) fp32 [B,40,1024] WITH grad: encode_cyl's maths on the device, the same 4x10 pool, fp8 round trip with a straight-through gradient."""
    x = cyl_u8.to(enc.device).to(enc.mean.dtype) / 255.0
    h = enc.model(pixel_values=(x - enc.mean) / enc.std).last_hidden_state
    return _Fp8STE.apply(E.pool_field(h[:, enc.n_special:, :].float()))


def renc_loss(pred_render, pred_real, target, lam=LAMBDA_KEEP, denom=None):
    """MSE(P(E'(render)), P(E(real))) + lam * MSE(P(E'(real)), P(E(real))). `denom` = elements of the WHOLE batch (micro-batches then sum to the batch mean)."""
    t = target.float().to(pred_render.device)
    d = float(denom if denom is not None else t.numel())
    lr_, lk_ = ((pred_render - t) ** 2).sum() / d, ((pred_real - t) ** 2).sum() / d
    return lr_ + lam * lk_, lr_.detach(), lk_.detach()


def cosine(a, b):
    return E.cosine(a, b)


# ------------------------------------------------------------------------------------------------------------------------ data
class PairData:
    """The pairs of the scenes it is GIVEN (the trainer gives it its own fold only). `touched` records every scene whose arrays were opened."""

    def __init__(self, data_dir, scenes):
        self.dir, self.scenes, self.touched, self.items, self.meta, self._mm = data_dir, list(scenes), set(), [], {}, {}
        for sid in self.scenes:
            m = torch.load(f"{data_dir}/{sid[:8]}.meta.pt", map_location="cpu", weights_only=True)
            assert m["scene"] == sid, f"meta of {sid[:8]} is for {m['scene']}"
            self.meta[sid] = m
            self.items += [(sid, j) for j in range(len(m["ticks"]))]

    def __len__(self):
        return len(self.items)

    def _arr(self, sid, kind):
        k = (sid, kind)
        if k not in self._mm:
            self.touched.add(sid)
            self._mm[k] = np.load(f"{self.dir}/{sid[:8]}.{kind}.npy", mmap_mode="r")
        return self._mm[k]

    def batch(self, idxs):
        rs, qs, ts = [], [], []
        for i in idxs:
            sid, j = self.items[i]
            rs.append(torch.from_numpy(np.array(self._arr(sid, "render")[j])))
            qs.append(torch.from_numpy(np.array(self._arr(sid, "real")[j])))
            ts.append(self.meta[sid]["target"][j])
        return torch.stack(rs), torch.stack(qs), torch.stack(ts)


def list_scenes(data_dir):
    """Scene ids that have a meta file, sorted."""
    out = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".meta.pt"):
            out.append(torch.load(f"{data_dir}/{f}", map_location="cpu", weights_only=True)["scene"])
    return sorted(out)
