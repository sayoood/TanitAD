"""E-DENSE-1 — does a DENSE token-level prediction target make the latent decodable?

Pre-registered in `PREREG_E_DENSE_1.md`. See §1b there: the STRONG form of the
granularity hypothesis was already refuted by `pixel_pooled` before any arm ran,
so what is under test here is the NATURE of the training pressure, not its
resolution.

⭐ THE SINGLE VARIABLE. v6 — and the E-V6SHAPE harness that reproduces its
signature — encodes each frame, POOLS to a 4x4 cell latent, and applies BOTH the
predictor and the loss to that pooled vector. Nothing ever asks an individual
patch token to be informative. The arms differ in exactly one thing:

  A `pooled`      predict the target frame's POOLED latent   (v6's design)
  B `dense`       predict the target frame's 160 PATCH TOKENS, ALL of them
  C `dense_deep`  B, plus the same dense target from an INTERMEDIATE encoder
                  layer (V-JEPA 2.1's Deep Self-Supervision)
  D `distill`     ⭐ POSITIVE CONTROL — regress frozen DINOv3 patch tokens

⛔⛔ WHY D IS NON-NEGOTIABLE. Without it a negative result is uninterpretable:
equally consistent with "the dense target does not help" and with "6.4 M params
on 130 clips cannot carry patch content whatever the objective". V-JEPA 2.1's
gains were at ViT-G on web-scale video. D is the role `oracle` played in
E-DETECT-1 — the arm that makes every other row readable.

⚠️ D TRAINS ON A SUBSET AND THAT IS DECLARED. DINOv3 features are banked for the
5,617 probe rows, not all 26,108 harness frames, so D sees 21.5% of the frames
the other arms see. It is a CAPABILITY control, never a matched arm, and its
number is not comparable to A/B/C on equal-data grounds.

⚠️ RESOLUTION. This harness runs 128x320 -> 8x20 = 160 tokens at d=192, while
E-DETECT-1's banked `dino_tokens` is 640x1024 at 256x640. Cross-table comparison
is therefore INVALID; `pixel160` is built here as the resolution-matched floor so
every claim stays inside this experiment.

TIER: T0-DIAGNOSTIC. Dev-box only; Thor untouched.
"""
from __future__ import annotations

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
import e_v6shape as V   # noqa: E402  (Block, MLPProj, WINDOW, STRIDE, CACHE)

CACHE = V.CACHE
WINDOW, STRIDE, PATCH = V.WINDOW, V.STRIDE, 16
GRID_H, GRID_W = 8, 20
N_TOK = GRID_H * GRID_W                      # 160
D_MODEL, N_LAYERS, N_HEADS = 192, 6, 3
ARMS = ("pooled", "dense", "dense_deep", "distill")
#: which encoder layer feeds the deep-supervision target in arm C
DEEP_LAYER = N_LAYERS // 2 - 1               # 2 of 0..5


class DenseEncoder(nn.Module):
    """v6's readout, but the patch tokens are RETURNED rather than discarded.

    ⚠️ Identical trunk across every arm — patch embed, CLS, 6 blocks, LayerNorm,
    and the same parameter-free 4x4 pool + 192->128 projection that produces
    v6's 2048-d operative latent. Only what the LOSS is applied to changes.
    """

    def __init__(self, d=D_MODEL, layers=N_LAYERS, heads=N_HEADS):
        super().__init__()
        self.patch = nn.Conv2d(3, d, PATCH, PATCH)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, N_TOK + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([V.Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 128)
        self.d_latent = 16 * 128
        self.d_tok = d

    def forward(self, x, want_deep: bool = False):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        deep = None
        for i, b in enumerate(self.blocks):
            t = b(t)
            if want_deep and i == DEEP_LAYER:
                deep = t[:, 1:]
        t = self.norm(t)
        tok = t[:, 1:]                                    # [B, 160, d]
        g = tok.transpose(1, 2).reshape(-1, tok.shape[-1], GRID_H, GRID_W)
        g = F.adaptive_avg_pool2d(g, (4, 4)).flatten(2).transpose(1, 2)
        return self.proj(g).flatten(1), tok, deep


class TokenDecoder(nn.Module):
    """Context vector -> the target frame's FULL token field.

    ⭐ THIS IS THE DENSE PREDICTIVE LOSS'S MACHINERY. 160 learned positional
    queries cross-attend to the predictor's context, so EVERY token position
    carries a training signal — including the ones a masked objective would have
    dropped. That is V-JEPA 2.1's "all tokens ... contribute to the training
    loss", and it is exactly what v6's `o3_masked_cell_loss` excludes.
    """

    def __init__(self, d_ctx: int, d_out: int, d=D_MODEL, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_ctx), nn.Linear(d_ctx, d))
        self.q = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads,
                                                         batch_first=True)
                                   for _ in range(layers)])
        self.lq = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                           nn.Linear(2 * d, d)) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_out))

    def forward(self, ctx):
        kv = self.inp(ctx)[:, None]                       # [B, 1, d]
        q = self.q.expand(ctx.shape[0], -1, -1)
        for at, ln, ff in zip(self.attn, self.lq, self.ff):
            q = q + at(ln(q), kv, kv, need_weights=False)[0]
            q = q + ff(q)
        return self.out(q)


def build(arm: str, dev: str, d_dino: int = 0):
    enc = DenseEncoder().to(dev)
    pred = V.ARPredictor(enc.d_latent).to(dev)
    dec = deep_dec = None
    if arm in ("dense", "dense_deep"):
        dec = TokenDecoder(enc.d_latent, enc.d_tok).to(dev)
    if arm == "dense_deep":
        deep_dec = TokenDecoder(enc.d_latent, enc.d_tok).to(dev)
    if arm == "distill":
        dec = nn.Sequential(nn.LayerNorm(enc.d_tok),
                            nn.Linear(enc.d_tok, d_dino)).to(dev)
    mods = [m for m in (enc, pred, dec, deep_dec) if m is not None]
    n_par = sum(p.numel() for m in mods for p in m.parameters())
    return enc, pred, dec, deep_dec, mods, n_par


def train_arm(arm: str, seed: int = 0, *, steps=6000, batch=32, lam=0.09,
              lr=5e-5, wd=1e-3, clip=1.0, dev="cuda", log_every=500) -> dict:
    from tanitad.models.sigreg import SigReg
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    torch.manual_seed(seed)
    frames = np.load(CACHE / "frames.npy", mmap_mode="r")
    actions = np.load(CACHE / "actions.npy")
    clips = json.loads((CACHE / "clips.json").read_text(encoding="utf-8"))
    span = STRIDE * WINDOW

    dino = d_dino = None
    if arm == "distill":
        # ⚠️ SUBSET, DECLARED: DINOv3 tokens exist only for the 5,617 probe rows.
        import e_detect_prep as P
        keys = [tuple(k) for k in
                json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
        fmap = {(c["clip_id"], f): c["start"] + f for c in clips
                for f in range(c["n"])}
        # ⚠️ TWO PARALLEL INDEX SPACES, and conflating them is the bug this
        # comment exists to stop recurring: `rows` are FRAME indices into the
        # 26,108-frame cache, while `dino` is indexed by PROBE-ROW POSITION
        # (0..5,616). We therefore sample a probe-row position and look the
        # frame up, never the other way round.
        keep = [(i, fmap[k]) for i, k in enumerate(keys) if k in fmap]
        dino_pos = np.array([i for i, _ in keep])
        rows = np.array([r for _, r in keep])
        dino = np.load(P.FEAT / "dino_tokens.npy", mmap_mode="r")
        d_dino = dino.shape[1] // 640
        starts = np.arange(len(rows))            # positions, not frame indices
        print(f"    [distill] {len(rows):,} of {len(frames):,} frames carry a "
              f"DINOv3 target ({100 * len(rows) / len(frames):.1f}%), d={d_dino}",
              flush=True)
    else:
        starts = np.concatenate([np.arange(c["start"], c["start"] + c["n"] - span)
                                 for c in clips if c["n"] > span])

    enc, pred, dec, deep_dec, mods, n_par = build(arm, dev, d_dino or 0)
    sig = SigReg(n_slices=512)
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    cum = np.concatenate([np.zeros((1, 2), np.float64), np.cumsum(actions, 0)])
    hist, t0 = [], time.time()
    print(f"    [{arm}/s{seed}] params={n_par / 1e6:.2f}M  d_latent="
          f"{enc.d_latent}  n_tok={N_TOK}  window={WINDOW} stride={STRIDE}",
          flush=True)

    for step in range(steps):
        pick = starts[torch.randint(0, len(starts), (batch,),
                                    generator=g).numpy()]
        if arm == "distill":
            pos = np.sort(pick)                  # probe-row positions
            obs = torch.from_numpy(np.asarray(frames[rows[pos]])).to(dev)
            obs = obs.float() / 255.0
            _, tok, _ = enc(obs)
            tgt = torch.from_numpy(
                np.asarray(dino[dino_pos[pos]], dtype=np.float32)
            ).to(dev).reshape(len(pos), 640, d_dino)
            # 640 (16x40) -> 160 (8x20) so the target matches this grid
            tgt = F.adaptive_avg_pool2d(
                tgt.transpose(1, 2).reshape(-1, d_dino, 16, 40),
                (GRID_H, GRID_W)).flatten(2).transpose(1, 2)
            l_pred = F.mse_loss(dec(tok), tgt.detach())
            l_sig = sig(enc(obs)[0])   # SIGReg on the pooled latent, as in A/B/C
            loss = l_pred + lam * l_sig
            extra = {}
        else:
            offs = [j * STRIDE for j in range(WINDOW + 1)]
            flat = np.concatenate([pick + o for o in offs])
            obs = torch.from_numpy(np.asarray(frames[flat])).to(dev).float() / 255.
            z, tok, deep = enc(obs, want_deep=(arm == "dense_deep"))
            z = z.reshape(WINDOW + 1, len(pick), -1).transpose(0, 1)
            acts = torch.from_numpy(np.stack(
                [(cum[pick + offs[j + 1]] - cum[pick + offs[j]]) / STRIDE
                 for j in range(WINDOW)], 1)).to(dev).float()
            z_hat = pred(z[:, :WINDOW], acts)
            l_sig = sig(z.reshape(-1, z.shape[-1]))
            tgt_slice = slice(WINDOW * len(pick), (WINDOW + 1) * len(pick))
            if arm == "pooled":
                # ⛔ v6's design: the loss sees ONLY the pooled latent.
                l_pred = F.mse_loss(z_hat, z[:, WINDOW].detach())
                extra = {}
            else:
                # ⭐ DENSE: every one of the 160 target tokens carries signal.
                l_pred = F.mse_loss(dec(z_hat), tok[tgt_slice].detach())
                extra = {}
                if arm == "dense_deep":
                    l_deep = F.mse_loss(deep_dec(z_hat),
                                        deep[tgt_slice].detach())
                    l_pred = l_pred + l_deep
                    extra = {"deep": float(l_deep.detach())}
            loss = l_pred + lam * l_sig

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sch.step()
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "loss": float(loss.detach()),
                   "pred": float(l_pred.detach()),
                   "sigreg": float(l_sig.detach()), **extra,
                   "s": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"    [{arm}/s{seed}] {rec}", flush=True)

    # ---- bank BOTH readouts: pooled latent AND the token field -------------
    enc.eval()
    Zp = np.zeros((len(frames), enc.d_latent), dtype=np.float32)
    Zt = np.lib.format.open_memmap(
        CACHE / f"tok_{arm}_s{seed}.npy", mode="w+", dtype=np.float16,
        shape=(len(frames), N_TOK * enc.d_tok))
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            b = torch.from_numpy(np.asarray(frames[i:i + 256])).to(dev)
            p, t, _ = enc(b.float() / 255.0)
            Zp[i:i + 256] = p.cpu().numpy()
            Zt[i:i + 256] = t.reshape(len(t), -1).half().cpu().numpy()
    np.save(CACHE / f"z_{arm}_s{seed}.npy", Zp)
    Zt.flush()
    s = np.asarray(Zt[::400])
    if (np.abs(s).reshape(len(s), -1).max(1) == 0).any():
        raise SystemExit(f"[FATAL] {arm}: banked token field has all-zero rows")
    return {"arm": arm, "seed": seed, "n_params": n_par,
            "d_latent": enc.d_latent, "n_tok": N_TOK, "d_tok": enc.d_tok,
            "steps": steps, "history": hist,
            "latents": f"z_{arm}_s{seed}.npy",
            "tokens": f"tok_{arm}_s{seed}.npy",
            "wall_s": round(time.time() - t0)}


def main() -> None:
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(ARMS)
    steps = 6000
    for a in sys.argv[1:]:
        if a.startswith("--steps="):
            steps = int(a.split("=")[1])
    out_p = SP / "e_dense.json"
    out = (json.loads(out_p.read_text(encoding="utf-8"))
           if out_p.exists() and "--fresh" not in sys.argv else {
        "_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
        "eval_tier": "T0-DIAGNOSTIC",
        "prereg": "…/simwam-analysis/PREREG_E_DENSE_1.md",
        "question": "does a DENSE token-level prediction target make the latent "
                    "decodable, where v6's pooled target does not?",
        "grid": [GRID_H, GRID_W], "n_tok": N_TOK, "d_model": D_MODEL,
        "arms": {}})
    for arm in want:
        print(f"  == {arm} ==", flush=True)
        rec = train_arm(arm, steps=steps)
        out["arms"][arm] = rec
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  {arm:<12} {rec['n_params'] / 1e6:.2f}M  "
              f"wall {rec['wall_s']}s  final {rec['history'][-1]}", flush=True)
    print(f"\n-> {out_p}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
