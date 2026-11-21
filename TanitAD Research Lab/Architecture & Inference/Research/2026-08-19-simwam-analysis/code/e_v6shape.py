"""E-V6SHAPE — a v6-SHAPED LeWM implementation, and the explanation test.

⛔ WHY THIS REPLACES `e_lewm_ablate.py`. That harness was retracted as evidence
about LeWM: reading `github.com/lucas-maes/le-wm` found ~14 deviations. The four
structural ones were **history_size 3 vs my 1**, an **AR TRANSFORMER predictor**
vs my 3-layer MLP, **projectors with BatchNorm**, and a **CLS readout**.

⭐ AND THE DECISIVE REALISATION: v6 ALREADY MATCHES the reference on every one of
those axes except the readout. `V6Stack.encode_window` encodes per frame and
returns [B, W, d_op]; `OperativePredictor` consumes "a causal window of (state,
action)" with positional embeddings and a residual `z_hat = z_t + delta`; and the
reference CODE detaches the goal embedding, exactly as v6 does — contradicting
the paper's own "no stop-gradient" claim, which I had quoted against us twice.

⇒ MY HARNESS WAS THE OUTLIER, NOT v6. This one is built to v6's structure, so
that (a) it is a faithful proxy for v6 and (b) restoring history + a transformer
predictor TESTS whether their absence explains the previous null.

TWO AXES, and they are the only genuine v6-vs-reference differences left:

    readout   'pool2048' — v6: PARAMETER-FREE 4x4 spatial pool -> 16*128 = 2048
              'cls192'   — LeWM: CLS token -> MLP projector 192->2048->192 (BN)
              ⭐ LeWM MEASURES saturation at ~184 dims with "diminishing returns"
                 above; v6 runs 10x past that, through a lossy pool.

    terms     2 — LeWM: pred + lambda*sigreg (their headline: "six -> one")
              7 — v6: + rollout, scene-stability, near-field, masked-cell

FAITHFUL TO THE REFERENCE unless noted: detach on the target, SIGReg on the
per-frame latents, ADDITIVE loss (pred + lambda*sigreg, NOT the convex
combination I used before), lr 5e-5, weight_decay 1e-3, grad-clip 1.0,
num_proj 1024, num_preds 1.

⚠️ DELIBERATE DEVIATION, measured: the tick is STRIDE frames, not 1. At k=1 a
driving latent moves 1.12 % of its magnitude so the identity map explains 98.9 %
(MEASURED 2026-08-20). LeWM's Push-T moves materially per step; a car at 0.1 s
does not. STRIDE=5 (0.5 s) gives 6.9x the k=1 signal.

⚠️ DEV BOX ONLY. Thor is training v6F S-W and is not touched.
"""
from __future__ import annotations

import argparse
import json
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

import e_lewm_ablate as B  # frame cache, geometry constants  # noqa: E402

CACHE = B.CACHE
H, W_IMG, PATCH = B.H, B.W, B.PATCH
GRID_H, GRID_W, N_TOK = B.GRID_H, B.GRID_W, B.N_TOK

#: v6's window. `OperativePredictor` takes states [B, W, D]; the live run's
#: config carries window=6.
WINDOW = 6
#: 0.5 s per tick — see the module docstring's measured justification.
STRIDE = 5
ARMS = ("v6shape", "cls192", "terms7", "cls192_terms7")


# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal=False):
        h = self.n1(x)
        m = None
        if causal:
            n = x.shape[1]
            m = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), 1)
        x = x + self.at(h, h, h, need_weights=False, attn_mask=m)[0]
        return x + self.ff(self.n2(x))


class MLPProj(nn.Module):
    """LeWM's projector: 192 -> 2048 -> 192 with BatchNorm1d."""

    def __init__(self, d, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                                 nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """Per-frame ViT, then ONE of the two readouts under test."""

    def __init__(self, readout: str, d=192, layers=6, heads=3):
        super().__init__()
        self.readout = readout
        self.patch = nn.Conv2d(3, d, PATCH, PATCH)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, N_TOK + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        if readout == "pool2048":
            self.proj = nn.Linear(d, 128)          # v6: 16 cells x 128 = 2048
            self.d_latent = 16 * 128
        elif readout == "cls192":
            self.proj = MLPProj(d)                 # LeWM: CLS -> MLP projector
            self.d_latent = d
        else:
            raise ValueError(readout)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        if self.readout == "cls192":
            return self.proj(t[:, 0])
        g = t[:, 1:].transpose(1, 2).reshape(-1, t.shape[-1], GRID_H, GRID_W)
        g = F.adaptive_avg_pool2d(g, (4, 4)).flatten(2).transpose(1, 2)
        return self.proj(g).flatten(1)


class ARPredictor(nn.Module):
    """v6's `OperativePredictor` shape: causal window of (state, action) ->
    residual delta on the LAST position. Mirrors LeWM's ARPredictor."""

    def __init__(self, d_latent, d=192, layers=6, heads=3, action_dim=2):
        super().__init__()
        self.inp = nn.Linear(d_latent, d)
        self.act = nn.Sequential(nn.Linear(action_dim, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_latent))

    def forward(self, states, actions):
        x = self.inp(states) + self.act(actions) + self.pos[:, :states.shape[1]]
        for b in self.blocks:
            x = b(x, causal=True)
        return states[:, -1] + self.out(x[:, -1])       # residual on last


def arm_config(arm: str) -> dict:
    cfg = {"readout": "pool2048", "terms": 2}
    if arm == "cls192":
        cfg["readout"] = "cls192"
    elif arm == "terms7":
        cfg["terms"] = 7
    elif arm == "cls192_terms7":
        cfg = {"readout": "cls192", "terms": 7}
    elif arm != "v6shape":
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    return cfg


def train_arm(arm: str, seed: int, *, steps=6000, batch=32, lam=0.09,
              lr=5e-5, wd=1e-3, clip=1.0, dev="cuda", log_every=1000) -> dict:
    from tanitad.models.sigreg import SigReg
    cfg = arm_config(arm)
    torch.manual_seed(seed)
    frames = np.load(CACHE / "frames.npy", mmap_mode="r")
    actions = torch.from_numpy(np.load(CACHE / "actions.npy"))
    clips = json.loads((CACHE / "clips.json").read_text(encoding="utf-8"))

    span = STRIDE * WINDOW                       # W history ticks + 1 target
    starts = np.concatenate([np.arange(c["start"], c["start"] + c["n"] - span)
                             for c in clips if c["n"] > span])
    enc = Encoder(cfg["readout"]).to(dev)
    pred = ARPredictor(enc.d_latent).to(dev)
    n_par = sum(p.numel() for p in enc.parameters()) + \
        sum(p.numel() for p in pred.parameters())
    # ⚠️ reference uses num_proj 1024 at batch 128 x history 3 = 512 embeddings.
    # Epps-Pulley is O(n^2) PER SLICE, so at our 32 x 7 = 224 embeddings 1024
    # slices cost 2.81 s/step = 11.7 h/arm. 512 keeps the estimator's character
    # at a quarter of the cost; the deviation is declared, not hidden.
    sig = SigReg(n_slices=512)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    A = actions.numpy()
    cum = np.concatenate([np.zeros((1, 2), np.float64), np.cumsum(A, 0)])
    hist, t0 = [], time.time()
    print(f"    [{arm}/s{seed}] d_latent={enc.d_latent} params={n_par/1e6:.2f}M "
          f"window={WINDOW} stride={STRIDE}", flush=True)

    for step in range(steps):
        idx = starts[torch.randint(0, len(starts), (batch,), generator=g).numpy()]
        offs = [j * STRIDE for j in range(WINDOW + 1)]      # W history + target
        flat = np.concatenate([idx + o for o in offs])
        obs = torch.from_numpy(np.asarray(frames[flat])).to(dev).float() / 255.0
        zs = enc(obs).reshape(WINDOW + 1, len(idx), -1).transpose(0, 1)  # [B,W+1,d]
        acts = torch.from_numpy(np.stack(
            [(cum[idx + offs[j + 1]] - cum[idx + offs[j]]) / STRIDE
             for j in range(WINDOW)], 1)).to(dev).float()               # [B,W,2]

        z_hat = pred(zs[:, :WINDOW], acts)
        # ⭐ FAITHFUL: the reference detaches the goal embedding.
        l_pred = F.mse_loss(z_hat, zs[:, WINDOW].detach())
        # ⭐ FAITHFUL: SIGReg on the ENCODER's per-frame latents, ADDITIVE.
        l_sig = sig(zs.reshape(-1, zs.shape[-1]))
        loss = l_pred + lam * l_sig

        extra = {}
        if cfg["terms"] == 7:
            # v6-shaped auxiliaries. Their point is that LeWM ships TWO terms and
            # calls "six hyperparameters -> one" its headline; v6 runs seven.
            z2 = pred(torch.cat([zs[:, 1:WINDOW], z_hat[:, None]], 1), acts)
            l_roll = F.mse_loss(z2, zs[:, WINDOW].detach())
            l_scene = (z_hat - zs[:, WINDOW - 1]).pow(2).mean()
            l_near = (z_hat[:, :z_hat.shape[-1] // 4]
                      - zs[:, WINDOW, :z_hat.shape[-1] // 4].detach()).pow(2).mean()
            m = torch.rand_like(zs[:, :WINDOW]) > 0.25
            l_mask = F.mse_loss(pred(zs[:, :WINDOW] * m, acts),
                                zs[:, WINDOW].detach())
            loss = loss + l_roll + 0.3 * l_scene + l_near + l_mask
            extra = {"roll": float(l_roll.detach()), "scene": float(l_scene.detach()),
                     "near": float(l_near.detach()), "mask": float(l_mask.detach())}

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(pred.parameters()), clip)
        opt.step()
        sch.step()
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "loss": float(loss.detach()),
                   "pred": float(l_pred.detach()), "sigreg": float(l_sig.detach()),
                   **extra, "s": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"    [{arm}/s{seed}] {rec}", flush=True)

    enc.eval()
    Z = np.zeros((len(frames), enc.d_latent), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            b = torch.from_numpy(np.asarray(frames[i:i + 256])).to(dev).float() / 255.0
            Z[i:i + 256] = enc(b).cpu().numpy()
    out = CACHE / f"z_{arm}_s{seed}.npy"
    np.save(out, Z)
    return {"arm": arm, "seed": seed, "n_params": n_par,
            "config": dict(cfg, window=WINDOW, stride=STRIDE,
                           d_latent=enc.d_latent, lam=lam, lr=lr, wd=wd),
            "history": hist, "latents": out.name,
            "wall_s": round(time.time() - t0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="v6shape")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = SP / "e_lewm_train.json"
    res = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else []
    for arm in a.arms.split(","):
        for s in (int(x) for x in a.seeds.split(",")):
            print(f"\n=== {arm} seed {s} ===", flush=True)
            rec = train_arm(arm, s, steps=a.steps, dev=dev)
            res = [r for r in res
                   if not (r["arm"] == arm and r["seed"] == s)] + [rec]
            tp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {tp}")
