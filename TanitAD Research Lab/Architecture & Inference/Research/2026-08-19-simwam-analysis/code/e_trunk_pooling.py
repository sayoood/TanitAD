"""E-TRUNK-1 — is it the v6 TRUNK or the v6 READOUT that carries no dynamics?

⛔ THE CONFOUND THIS REMOVES. E-ACTSTREAM-2 compared DINOv3 patch fields
(640 x 1024) against v6 cell fields (16 x 128) and found the DINOv3 arms beat
C-PERSIST while the v6 arms never did. That comparison changes TWO things at
once — the ENCODER (DINOv3 vs ours) and the GRANULARITY (640 tokens vs 16
pooled cells) — so it cannot say which one is responsible. Quoting it as "the v6
trunk carries no dynamics" would be exactly the scope error this programme keeps
retracting.

⭐ `cache_tok11250` banks BOTH representations for the SAME frames from the SAME
checkpoint: `cells` [16, 128] and `tokens` [640, 768]. So the encoder is held
FIXED and only the granularity moves.

    arm `cells`   the OPERATIVE LATENT itself — 16 x 128 = 2048 = d_op exactly,
                  not a lossy summary of it
    arm `tokens`  the pre-readout encoder field, 640 x 768

Each arm is scored against ITS OWN C-PERSIST, because the representations have
different scales and only skill-over-persistence is comparable across them.

READING:
  tokens beat persistence, cells do not  -> the POOLING destroys the dynamics;
                                            the trunk has them and the readout
                                            is the defect (fixable)
  neither beats persistence              -> the v6 ENCODER does not carry them;
                                            a readout change will not help
  both beat persistence                  -> E-ACTSTREAM-2's v6 arms were limited
                                            by something else entirely

TIER: T0-DIAGNOSTIC. A future-field prediction error is a world-model fidelity
number, never driving performance.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_actstream as E  # noqa: E402  (ego_actions, Block)

import os
CACHE = SP / os.environ.get("TRUNK_CACHE", "sp2/cache_tok11250/latents.pt")
# ⚠️ MEASURED from the cache, not assumed: cache_tok11250 was dumped at
# stride 8 (2,809 frames, 22 per clip), NOT the stride 4 of cache_s16000. My
# first run built ZERO windows because it assumed 4 — the indices simply never
# aligned to the grid. horizon 7 x 8 frames = 5.6 s, the closest reachable point
# to the 6.0 s design horizon at this stride.
# ⚠️ MEASURED from the cache, never assumed — the first run built ZERO windows
# because it assumed stride 4 against a stride-8 cache and the indices never
# aligned. STRIDE is now read from the cache's own meta.
import json as _json
_META = _json.loads((CACHE.parent / "sp1_meta.json").read_text(encoding="utf-8"))
STRIDE = int(_META["stride"])
WINDOW = 4
HORIZON = int(round(6.0 * 10.0 / STRIDE))   # 6.0 s at 10 Hz, in cache slots


def build(kind: str):
    obj = torch.load(CACHE, map_location="cpu", weights_only=False)
    by = {}
    for r in obj["rows"]:
        v = r["cells"] if kind == "cells" else r["tokens"]
        if v is None:
            continue
        by.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = v
    X, A, Y, EP = [], [], [], []
    for cid, frames in by.items():
        acts = E.ego_actions(cid)
        if acts is None:
            continue
        idx = sorted(frames)
        pos = set(idx)
        for f in idx:
            need = [f - (WINDOW - 1 - k) * STRIDE for k in range(WINDOW)]
            tgt = f + HORIZON * STRIDE
            if any(n not in pos for n in need) or tgt not in pos:
                continue
            if tgt >= len(acts):
                continue
            X.append(torch.stack([frames[n] for n in need]))
            a = acts[f:tgt:STRIDE][:HORIZON]
            if a.shape[0] < HORIZON:
                a = torch.cat([a, a[-1:].repeat(HORIZON - a.shape[0], 1)])
            A.append(a)
            Y.append(frames[tgt])
            EP.append(cid)
    return (torch.stack(X).float(), torch.stack(A).float(),
            torch.stack(Y).float(), EP)


class Pred(nn.Module):
    """Identical trunk for both arms; only the input width differs."""

    def __init__(self, d_in, n_tok, d=192, layers=4, heads=6):
        super().__init__()
        self.n_tok = n_tok
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW * n_tok, d))
        self.act = nn.Sequential(nn.Linear(2 * HORIZON, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.mix = nn.Linear(2 * d, d)                 # broadcast+project
        self.blocks = nn.ModuleList([E.Block(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_in))

    def forward(self, x, a):
        B, W, C, _ = x.shape
        t = self.inp(x).reshape(B, W * C, -1) + self.pos
        e = self.act(a.reshape(B, -1))
        t = self.mix(torch.cat([t, e[:, None].expand(-1, t.shape[1], -1)], -1))
        for b in self.blocks:
            t = b(t)
        return x[:, -1] + self.out(t[:, (W - 1) * C:(W - 1) * C + C])


def run(kind, epochs=30, seeds=(0, 1, 2), batch=16):
    X, A, Y, EP = build(kind)
    sel = json.loads((SP / "sp2/p3_selection.json").read_text(encoding="utf-8"))
    ev = set(sel.get("eval") or sel.get("eval_clips") or [])
    if not ev:
        u = sorted(set(EP)); ev = set(u[int(0.7 * len(u)):])
    te = torch.tensor([e in ev for e in EP]); tr = ~te
    mu = Y[tr].mean(0, keepdim=True)
    X, Y = X - mu[:, None], Y - mu            # centre on the TRAIN mean field
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_tok, d_in = X.shape[2], X.shape[3]
    persist = ((X[te][:, -1] - Y[te]) ** 2).mean(dim=(1, 2))
    cvar = float((Y[te] ** 2).mean())
    print(f"\n=== {kind}: {tuple(X.shape)}  train {int(tr.sum())} eval {int(te.sum())} ===")
    print(f"    centred var {cvar:.6f}   C-PERSIST {persist.mean():.6f}")
    errs = []
    for s in seeds:
        torch.manual_seed(s)
        m = Pred(d_in, n_tok).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xtr, Atr, Ytr = X[tr], A[tr], Y[tr]
        for _ in range(epochs):
            m.train()
            perm = torch.randperm(Xtr.shape[0])
            for i in range(0, Xtr.shape[0], batch):
                j = perm[i:i + batch]
                loss = nn.functional.mse_loss(
                    m(Xtr[j].to(dev), Atr[j].to(dev)), Ytr[j].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            pw = torch.cat([((m(X[te][i:i + batch].to(dev),
                               A[te][i:i + batch].to(dev)) -
                              Y[te][i:i + batch].to(dev)) ** 2
                             ).mean(dim=(1, 2)).cpu()
                            for i in range(0, int(te.sum()), batch)])
        errs.append(pw)
        print(f"    seed {s}: {float(pw.mean()):.6f}  "
              f"({'BEATS' if pw.mean() < persist.mean() else 'loses to'} persistence)")
    mean_pw = torch.stack(errs).mean(0)
    d = mean_pw - persist
    ep_te = [e for e, k in zip(EP, te.tolist()) if k]
    by = collections.defaultdict(list)
    for e, v in zip(ep_te, d.tolist()):
        by[e].append(v)
    em = torch.tensor([sum(v) / len(v) for v in by.values()])
    g = torch.Generator().manual_seed(0)
    b = torch.stack([em[torch.randint(len(em), (len(em),), generator=g)].mean()
                     for _ in range(2000)])
    lo, hi = b.quantile(0.025).item(), b.quantile(0.975).item()
    verdict = ("BEATS persistence" if hi < 0 else
               "LOSES to persistence" if lo > 0 else "not separated")
    print(f"    paired (arm - C-PERSIST) {em.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] "
          f"=> {verdict}")
    return {"kind": kind, "n_tok": n_tok, "d_in": d_in,
            "mse": float(mean_pw.mean()), "persist": float(persist.mean()),
            "centred_var": cvar, "delta": float(em.mean()), "ci95": [lo, hi],
            "verdict": verdict}


if __name__ == "__main__":
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "cache": str(CACHE.parent.name),
           "run_stamp": _META.get("run_stamp"), "step": _META.get("step"),
           "stride": STRIDE, "window": WINDOW, "horizon_slots": HORIZON,
           "horizon_s": HORIZON * STRIDE / 10.0,
           "arms": [run(k) for k in ("cells", "tokens")]}
    (SP / "e_trunk_pooling.json").write_text(json.dumps(out, indent=1),
                                             encoding="utf-8")
    print(f"\n-> {SP / 'e_trunk_pooling.json'}")
