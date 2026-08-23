"""E-TRUNK-1 (memory-corrected runner) — v6 TRUNK or v6 READOUT?

⛔ WHY THIS FILE EXISTS. `e_trunk_pooling.py` materialised the windowed tensor
as [N, W, n_tok, d] FLOAT32, which DUPLICATES every frame W=4 times because
consecutive windows overlap by 3 frames. At stride 4 that is ~3,250 x 4 x 640 x
768 x 4 B = 25.6 GB, and the centring line `X - mu[:, None]` allocated a SECOND
25.6 GB. On a 34.2 GB box the run swapped to disk for 2.5 h at 41 % GPU without
finishing. The GPU was never the bottleneck; host RAM was.

⭐ THE FIX, and it changes NO mathematics:
  * unique frames are banked ONCE as fp16  [F, n_tok, d]   (5.52 GB, not 25.6)
  * windows are INDICES into that bank      [N, W] int64
  * centring happens per batch on the GPU, so no second full copy exists
  * the bank lives on the GPU when it fits, else pinned host memory

⚠️ Identical model, identical objective, identical split, identical estimator.
The ONLY differences are storage layout and where the mean subtraction happens.
Run against the stride-8 cache first: it reproduces a banked result, so the
refactor is CHECKED against a known number before it is trusted at scale.

TIER: T0-DIAGNOSTIC. A future-field prediction error is a world-model fidelity
number, never driving performance.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_actstream as E  # noqa: E402  (ego_actions, Block)

CACHE = SP / os.environ.get("TRUNK_CACHE", "sp2/cache_tok20000_s4/latents.pt")
_META = json.loads((CACHE.parent / "sp1_meta.json").read_text(encoding="utf-8"))
STRIDE = int(_META["stride"])
WINDOW = 4
HORIZON = int(round(6.0 * 10.0 / STRIDE))   # 6.0 s at 10 Hz, in cache slots
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(kind: str):
    """Bank unique frames once; return INDICES, never duplicated fields."""
    obj = torch.load(CACHE, map_location="cpu", weights_only=False)
    slot, vecs, seen = {}, [], collections.defaultdict(set)
    for r in obj["rows"]:
        v = r["cells"] if kind == "cells" else r["tokens"]
        if v is None:
            continue
        cid, f = r["clip_id"], int(r["frame_idx"])
        slot[(cid, f)] = len(vecs)
        vecs.append(v.half())
        seen[cid].add(f)
    bank = torch.stack(vecs)                      # [F, n_tok, d] fp16
    del vecs, obj

    XI, A, YI, EP = [], [], [], []
    for cid, fs in seen.items():
        acts = E.ego_actions(cid)
        if acts is None:
            continue
        for f in sorted(fs):
            need = [f - (WINDOW - 1 - k) * STRIDE for k in range(WINDOW)]
            tgt = f + HORIZON * STRIDE
            if any(n not in fs for n in need) or tgt not in fs:
                continue
            if tgt >= len(acts):
                continue
            XI.append([slot[(cid, n)] for n in need])
            a = acts[f:tgt:STRIDE][:HORIZON]
            if a.shape[0] < HORIZON:
                a = torch.cat([a, a[-1:].repeat(HORIZON - a.shape[0], 1)])
            A.append(a)
            YI.append(slot[(cid, tgt)])
            EP.append(cid)
    return (bank, torch.tensor(XI, dtype=torch.long), torch.stack(A).float(),
            torch.tensor(YI, dtype=torch.long), EP)


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
    t_start = time.time()
    bank, XI, A, YI, EP = build(kind)
    n_tok, d_in = bank.shape[1], bank.shape[2]

    sel = json.loads((SP / "sp2/p3_selection.json").read_text(encoding="utf-8"))
    ev = set(sel.get("eval") or sel.get("eval_clips") or [])
    if not ev:
        u = sorted(set(EP))
        ev = set(u[int(0.7 * len(u)):])
    te = torch.tensor([e in ev for e in EP])
    tr = ~te

    gb = bank.numel() * 2 / 1e9
    resident = False
    if DEV == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1e9
        if gb < free - 1.6:                      # leave headroom for the model
            bank = bank.to(DEV)
            resident = True
        else:
            bank = bank.pin_memory()
    A = A.to(DEV)
    print(f"\n=== {kind}: bank {tuple(bank.shape)} {gb:.2f} GB "
          f"{'on GPU' if resident else 'pinned host'} | "
          f"{len(EP)} windows, train {int(tr.sum())} eval {int(te.sum())} ===",
          flush=True)

    def fetch(idx):
        """[B, W, n_tok, d] float32 on device. No duplicated host copy exists."""
        x = bank[XI[idx].reshape(-1)]
        if not resident:
            x = x.to(DEV, non_blocking=True)
        return x.float().reshape(len(idx), WINDOW, n_tok, d_in)

    def fetch_y(idx):
        y = bank[YI[idx]]
        if not resident:
            y = y.to(DEV, non_blocking=True)
        return y.float()

    # Centre on the TRAIN target mean. The subtraction CANCELS in (x - y), so
    # C-PERSIST is identical centred or not; the model still needs centred input.
    tr_i = torch.nonzero(tr).squeeze(1)
    te_i = torch.nonzero(te).squeeze(1)
    acc = torch.zeros(n_tok, d_in, device=DEV, dtype=torch.float64)
    for i in range(0, len(tr_i), 256):
        acc += fetch_y(tr_i[i:i + 256]).double().sum(0)
    mu = (acc / len(tr_i)).float()

    pw_persist, cvar_acc = [], 0.0
    with torch.no_grad():
        for i in range(0, len(te_i), 64):
            j = te_i[i:i + 64]
            x, y = fetch(j), fetch_y(j)
            pw_persist.append(((x[:, -1] - y) ** 2).mean(dim=(1, 2)).cpu())
            cvar_acc += float(((y - mu) ** 2).mean()) * len(j)
    persist = torch.cat(pw_persist)
    cvar = cvar_acc / len(te_i)
    print(f"    centred var {cvar:.6f}   C-PERSIST {persist.mean():.6f}",
          flush=True)

    errs = []
    for s in seeds:
        torch.manual_seed(s)
        m = Pred(d_in, n_tok).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            m.train()
            perm = tr_i[torch.randperm(len(tr_i))]
            for i in range(0, len(perm), batch):
                j = perm[i:i + batch]
                loss = nn.functional.mse_loss(
                    m(fetch(j) - mu, A[j]), fetch_y(j) - mu)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            pw = torch.cat([
                ((m(fetch(te_i[i:i + batch]) - mu, A[te_i[i:i + batch]])
                  - (fetch_y(te_i[i:i + batch]) - mu)) ** 2
                 ).mean(dim=(1, 2)).cpu()
                for i in range(0, len(te_i), batch)])
        errs.append(pw)
        print(f"    seed {s}: {float(pw.mean()):.6f}  "
              f"({'BEATS' if pw.mean() < persist.mean() else 'loses to'} "
              f"persistence)   [{time.time() - t_start:.0f}s]", flush=True)

    mean_pw = torch.stack(errs).mean(0)
    d = mean_pw - persist
    ep_te = [EP[i] for i in te_i.tolist()]
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
          f"=> {verdict}", flush=True)
    del bank
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return {"kind": kind, "n_tok": n_tok, "d_in": d_in, "n_windows": len(EP),
            "n_eval_episodes": len(em), "bank_gb": round(gb, 3),
            "mse": float(mean_pw.mean()), "persist": float(persist.mean()),
            "centred_var": cvar, "delta": float(em.mean()), "ci95": [lo, hi],
            "verdict": verdict, "wall_s": round(time.time() - t_start)}


if __name__ == "__main__":
    arms = os.environ.get("TRUNK_ARMS", "cells,tokens").split(",")
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "runner": "e_trunk_pooling2.py (frame-bank, index-windowed)",
           "cache": str(CACHE.parent.name),
           "run_stamp": _META.get("run_stamp"), "step": _META.get("step"),
           "stride": STRIDE, "window": WINDOW, "horizon_slots": HORIZON,
           "horizon_s": HORIZON * STRIDE / 10.0,
           "arms": [run(k) for k in arms]}
    dest = SP / os.environ.get("TRUNK_OUT", "e_trunk_pooling2.json")
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {dest}", flush=True)
