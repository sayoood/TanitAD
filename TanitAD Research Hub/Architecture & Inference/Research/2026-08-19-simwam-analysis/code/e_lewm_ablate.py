"""E-LEWM-1 — which v6 deviation from LeWorldModel costs the representation?

⛔ PRE-REGISTERED in `PREREG_E_LEWM_1.md` (committed e4d58be) BEFORE this file
produced any number. Arms, decision rule, falsifiers and limits are fixed there.

⭐ WHY AT THIS SCALE. LeWM (2603.19312) IS v6's architecture — encoder +
action-conditioned next-latent predictor, jointly trained, SIGReg against
collapse, no teacher — and reaches probe r 0.90-0.99. v6 decodes NOTHING and its
SIGReg has stalled at 7.83 since step 11k. v6 costs 9+ days per experiment at
336M params; LeWM does it at ~15M in hours. So the question is answered HERE and
the winner applied to v6 once.

⚠️ DEV BOX ONLY. Thor is training v6F S-W and is not touched.

ARMS — one-factor-at-a-time from the published LeWM configuration:
    lewm    d_latent 192 · SIGReg on z AND z_hat · no detach · 2 loss terms
    d2048   d_latent -> 2048          (v6's d_op = 16 x 128)
    sigop   SIGReg on z_hat ONLY      (v6 applies it to the operative latent alone)
    detach  target detached           (v6: "detached by the caller")
    terms7  + rollout + scene terms   (v6 runs 7 loss terms; LeWM's headline is 2)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
CACHE = SP / "sp2/lewm_frames"
H, W = 128, 320                      # 2.5 aspect, quarter of v6's 256x640
PATCH = 16                           # -> 8 x 20 = 160 tokens
GRID_H, GRID_W = H // PATCH, W // PATCH
N_TOK = GRID_H * GRID_W

#: v6's readout shape, mirrored: d_op = n_cells * d_readout, n_cells = 4x4.
N_CELLS = 16
ARMS = ("lewm", "d2048", "sigop", "detach", "terms7")

#: ⛔ THE TICK, AND WHY IT IS NOT 1. MEASURED 2026-08-20 on the first gate run:
#: at k=1 (0.1 s) the latent moves only **1.12 %** of its own magnitude, so the
#: IDENTITY MAP explains 98.9 % of the target and next-frame prediction is very
#: nearly trivial. The gate arm duly failed to decode anything (lead_gap
#: -0.0130) while SIGReg converged to 0.551 — a model that had learned isotropy
#: and no dynamics. Measured ||dz||^2/||z||^2 by horizon:
#:     k=1 0.0112 · k=10 0.1573 (x14) · k=20 0.2725 (x24) · k=60 0.5415 (x48)
#: LeWM's Push-T/Reacher move materially per step; driving at 10 Hz does not.
#: ⇒ the predictor ticks at STRIDE frames and is rolled ROLL times, which is
#: still LeWM's AUTOREGRESSIVE next-latent formulation ("collected over the
#: history length N") — only at a rate where the world actually changes.
STRIDE = 10          # 1.0 s per predictor tick
ROLL = 3             # rolled 3 ticks -> 3.0 s supervised horizon


# --------------------------------------------------------------------------- #
# frame cache — decoded ONCE to a uint8 memmap
# --------------------------------------------------------------------------- #
def build_frames() -> None:
    from PIL import Image
    CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted(EPS.glob("*.v2ep.pt"))
    metas, total = [], 0
    for p in files:
        o = torch.load(p, map_location="cpu", weights_only=False)
        total += len(o["jpeg_len"])
    print(f"decoding {total} frames from {len(files)} clips -> {H}x{W}", flush=True)
    mm = np.lib.format.open_memmap(CACHE / "frames.npy", mode="w+",
                                   dtype=np.uint8, shape=(total, 3, H, W))
    acts = np.zeros((total, 2), dtype=np.float32)
    i = 0
    for ci, p in enumerate(files):
        cid = p.name.split(".v2ep.pt")[0]
        o = torch.load(p, map_location="cpu", weights_only=False)
        buf, lens = o["jpeg_buf"].numpy(), o["jpeg_len"].tolist()
        a = o["actions"].float().numpy()
        offs = np.concatenate([[0], np.cumsum(lens)])
        start = i
        for f in range(len(lens)):
            im = Image.open(io.BytesIO(buf[offs[f]:offs[f] + lens[f]].tobytes()))
            im = im.convert("RGB").resize((W, H), Image.BILINEAR)
            mm[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
            acts[i] = a[f] if f < len(a) else a[-1]
            i += 1
        metas.append({"clip_id": cid, "start": start, "n": len(lens)})
        if (ci + 1) % 20 == 0:
            print(f"  {ci+1}/{len(files)} clips, {i} frames", flush=True)
    mm.flush()
    np.save(CACHE / "actions.npy", acts)
    (CACHE / "clips.json").write_text(json.dumps(metas), encoding="utf-8")
    print(f"done: {i} frames -> {CACHE}", flush=True)


# --------------------------------------------------------------------------- #
# model — LeWM's shape, with v6's readout geometry so d_latent is ablatable
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.at(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class Encoder(nn.Module):
    """Patch-embed ViT -> v6-shaped readout (pool to 4x4 cells, project)."""

    def __init__(self, d_latent: int, d=256, layers=6):
        super().__init__()
        self.patch = nn.Conv2d(3, d, PATCH, PATCH)
        self.pos = nn.Parameter(torch.zeros(1, N_TOK, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        assert d_latent % N_CELLS == 0, f"d_latent must be divisible by {N_CELLS}"
        self.proj = nn.Linear(d, d_latent // N_CELLS)   # v6's readout.proj
        self.d_latent = d_latent

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        # v6's PARAMETER-FREE pool: token grid -> 4x4 cells, then project
        g = t.transpose(1, 2).reshape(-1, t.shape[-1], GRID_H, GRID_W)
        g = F.adaptive_avg_pool2d(g, (4, 4))
        g = g.flatten(2).transpose(1, 2)                # [B, 16, d]
        return self.proj(g).flatten(1)                  # [B, d_latent]


class Predictor(nn.Module):
    """z_{t+1} = z_t + f(z_t, a_t) — LeWM's action-conditioned next-latent map."""

    def __init__(self, d_latent: int, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_latent))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))


# --------------------------------------------------------------------------- #
def arm_config(arm: str) -> dict:
    """The published LeWM configuration, with exactly ONE axis flipped."""
    cfg = {"d_latent": 192, "sigreg_on_z": True, "sigreg_on_zhat": True,
           "detach": False, "extra_terms": False}
    if arm == "d2048":
        cfg["d_latent"] = 2048
    elif arm == "sigop":
        cfg["sigreg_on_z"] = False          # v6: not on the encoder embedding
    elif arm == "detach":
        cfg["detach"] = True
    elif arm == "terms7":
        cfg["extra_terms"] = True
    elif arm != "lewm":
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    return cfg


def train_arm(arm: str, seed: int, *, steps=3000, batch=32, lam=0.1,
              lr=3e-4, dev="cuda", log_every=500) -> dict:
    from tanitad.models.sigreg import SigReg
    cfg = arm_config(arm)
    torch.manual_seed(seed)

    frames = np.load(CACHE / "frames.npy", mmap_mode="r")
    actions = torch.from_numpy(np.load(CACHE / "actions.npy"))
    clips = json.loads((CACHE / "clips.json").read_text(encoding="utf-8"))

    # valid starts with ROLL ticks of STRIDE frames inside ONE clip
    span = STRIDE * ROLL
    pairs = np.concatenate([np.arange(c["start"], c["start"] + c["n"] - span)
                            for c in clips if c["n"] > span])
    # mean action over each tick — the control summary for a STRIDE-long step
    A = actions.numpy()
    cum = np.concatenate([np.zeros((1, 2), np.float64), np.cumsum(A, 0)])
    enc = Encoder(cfg["d_latent"]).to(dev)
    pred = Predictor(cfg["d_latent"]).to(dev)
    n_par = sum(p.numel() for p in enc.parameters()) + \
        sum(p.numel() for p in pred.parameters())
    sig = SigReg(n_slices=512)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    g = torch.Generator().manual_seed(seed)
    hist, t0 = [], time.time()
    for step in range(steps):
        idx = pairs[torch.randint(0, len(pairs), (batch,), generator=g).numpy()]
        # encode t, t+STRIDE, ..., t+ROLL*STRIDE in ONE forward
        offs = [j * STRIDE for j in range(ROLL + 1)]
        flat = np.concatenate([idx + o for o in offs])
        obs = torch.from_numpy(np.asarray(frames[flat])).to(dev).float() / 255.0
        zs = enc(obs).reshape(ROLL + 1, len(idx), -1)     # [ROLL+1, B, d]
        z_t = zs[0]

        # LeWM's AUTOREGRESSIVE roll, supervised at EVERY tick
        l_pred = z_t.new_zeros(())
        zh, zhats = z_t, []
        for j in range(ROLL):
            a_j = torch.from_numpy(
                ((cum[idx + offs[j + 1]] - cum[idx + offs[j]]) / STRIDE)
            ).to(dev).float()
            zh = pred(zh, a_j)
            zhats.append(zh)
            tg = zs[j + 1]
            l_pred = l_pred + F.mse_loss(zh, tg.detach() if cfg["detach"] else tg)
        l_pred = l_pred / ROLL

        l_sig = z_t.new_zeros(())
        n_sig = 0
        if cfg["sigreg_on_z"]:
            # every ENCODED embedding, per LeWM Fig. 1's SIGReg on z
            l_sig = l_sig + sig(zs.reshape(-1, zs.shape[-1])); n_sig += 1
        if cfg["sigreg_on_zhat"]:
            l_sig = l_sig + sig(torch.cat(zhats, 0)); n_sig += 1
        l_sig = l_sig / max(n_sig, 1)

        loss = (1 - lam) * l_pred + lam * l_sig
        extra = {}
        if cfg["extra_terms"]:
            # v6-shaped auxiliaries. Their POINT is that they dilute SIGReg's
            # share — LeWM's "six hyperparameters -> one" complaint made concrete.
            l_roll = F.mse_loss(zhats[-1], zs[-1].detach())
            l_scene = (zhats[0] - z_t).pow(2).mean()
            loss = loss + 1.0 * l_roll + 0.3 * l_scene
            extra = {"roll": float(l_roll), "scene": float(l_scene)}

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "loss": float(loss), "pred": float(l_pred),
                   "sigreg": float(l_sig), **extra,
                   "s": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"    [{arm}/s{seed}] {rec}", flush=True)

    # encode EVERY frame for the probe
    enc.eval()
    Z = np.zeros((len(frames), cfg["d_latent"]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            b = torch.from_numpy(np.asarray(frames[i:i + 256])).to(dev).float() / 255.0
            Z[i:i + 256] = enc(b).cpu().numpy()
    out = CACHE / f"z_{arm}_s{seed}.npy"
    np.save(out, Z)
    return {"arm": arm, "seed": seed, "n_params": n_par, "config": cfg,
            "history": hist, "latents": str(out.name),
            "wall_s": round(time.time() - t0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "train"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "frames":
        build_frames()
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        res = []
        for arm in a.arms.split(","):
            for s in (int(x) for x in a.seeds.split(",")):
                print(f"\n=== {arm} seed {s} ===", flush=True)
                res.append(train_arm(arm, s, steps=a.steps, dev=dev))
                (SP / "e_lewm_train.json").write_text(
                    json.dumps(res, indent=1), encoding="utf-8")
        print(f"\n-> {SP / 'e_lewm_train.json'}")
