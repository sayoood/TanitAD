"""E-ACTSTREAM-2 — the same contrast at REF-A v1's REAL geometry (DINOv3).

⭐ THE TRANSFER QUESTION, and it is not rhetorical. E-ACTSTREAM-1 measured
action-as-token vs action-as-broadcast on v6 CELL fields — **16 tokens x 128 d** —
and found tokenisation ahead by 6-10x. v1's real field is **640 x 1024**:

    at  16 vision tokens, 2 action tokens are 11 %  of the stream
    at 640 vision tokens, 2 action tokens are 0.3 % of the stream

Broadcast reaches EVERY token by construction; tokenisation relies on attention
finding 2 tokens among 642. **A 6-10x advantage at 11 % says nothing about
0.3 %**, and quoting it there would be the scope error this programme keeps
retracting. So the transfer is measured.

⚠️ IDENTICAL WINDOWS. Same clips, same stride, same frame indices and the same
episode-disjoint split as the v6-cell run, so the two results are comparable.

⛔ LAZY INDEXING IS A REQUIREMENT, NOT AN OPTIMISATION. Materialising the windows
would be 3,277 x 5 x 640 x 1024 x 2 B = **21 GB**; the per-clip fp16 arrays are
7.4 GB and stay in CPU RAM, gathered per batch.

TIER: **T0-DIAGNOSTIC** — future-field prediction error, never driving performance.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
FIELDS = SP / "dinov3_fields"
sys.path.insert(0, str(SP))
import e_actstream as E  # noqa: E402  (ego_actions, Block)

STRIDE = 4


def build_index(window: int, horizon: int):
    """-> (clips {cid: fp16 array}, index list, actions {cid: [T,2]})."""
    meta = json.loads((FIELDS / "meta.json").read_text(encoding="utf-8"))
    clips, acts, idx = {}, {}, []
    for cid, info in meta["clips"].items():
        arr = np.load(FIELDS / f"{cid}.npy", mmap_mode="r")
        a = E.ego_actions(cid)
        if a is None:
            continue
        clips[cid] = arr
        acts[cid] = a
        frames = info["frames"]
        pos = {f: i for i, f in enumerate(frames)}
        for f in frames:
            need = [f - (window - 1 - k) * STRIDE for k in range(window)]
            tgt = f + horizon * STRIDE
            if any(n not in pos for n in need) or tgt not in pos:
                continue
            if tgt >= len(a):
                continue
            idx.append((cid, [pos[n] for n in need], pos[tgt], f))
    return clips, acts, idx


def gather(clips, acts, idx, sel, horizon, dev):
    xs, ys, aa = [], [], []
    for j in sel:
        cid, need, tgt, f = idx[j]
        arr = clips[cid]
        xs.append(torch.from_numpy(np.asarray(arr[need])))
        ys.append(torch.from_numpy(np.asarray(arr[tgt])))
        a = acts[cid][f:f + horizon * STRIDE:STRIDE][:horizon]
        if a.shape[0] < horizon:
            a = torch.cat([a, a[-1:].repeat(horizon - a.shape[0], 1)])
        aa.append(a)
    return (torch.stack(xs).to(dev).float(), torch.stack(aa).to(dev).float(),
            torch.stack(ys).to(dev).float())


class Pred(nn.Module):
    """Identical trunk; the ONE difference is how the action arrives."""

    def __init__(self, mode, d_in=1024, d=256, layers=4, heads=4,
                 window=4, horizon=15, n_tok=640, n_act_tok=2):
        super().__init__()
        self.mode, self.window, self.n_tok = mode, window, n_tok
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, window * n_tok, d))
        self.act = nn.Sequential(nn.Linear(2 * horizon, d), nn.GELU(),
                                 nn.Linear(d, d))
        if mode == "concat":
            self.mix = nn.Linear(2 * d, d)
        elif mode == "add":
            pass
        elif mode == "token":
            self.n_act_tok = n_act_tok
            self.act_split = nn.Linear(d, n_act_tok * d)
            self.act_pos = nn.Parameter(torch.zeros(1, n_act_tok, d))
        else:
            raise ValueError(mode)
        self.blocks = nn.ModuleList([E.Block(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_in))

    def forward(self, x, a):
        B, W, C, _ = x.shape
        t = self.inp(x).reshape(B, W * C, -1) + self.pos
        e = self.act(a.reshape(B, -1))
        if self.mode == "concat":
            t = self.mix(torch.cat([t, e[:, None].expand(-1, t.shape[1], -1)], -1))
        elif self.mode == "add":
            t = t + e[:, None]
        else:
            at = self.act_split(e).reshape(B, self.n_act_tok, -1) + self.act_pos
            t = torch.cat([t, at], dim=1)
        for b in self.blocks:
            t = b(t)
        r = self.out(t[:, (W - 1) * C:(W - 1) * C + C])
        return x[:, -1] + r                      # residual, as in v1


def run(mode, seed, tr, te, clips, acts, idx, args, dev, mu):
    torch.manual_seed(seed)
    m = Pred(mode, d=args.d, layers=args.layers, heads=args.heads,
             window=args.window, horizon=args.horizon,
             n_act_tok=args.act_tokens).to(dev)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        m.train()
        perm = np.random.RandomState(seed).permutation(len(tr))
        for i in range(0, len(tr), args.batch):
            sel = [tr[k] for k in perm[i:i + args.batch]]
            x, a, y = gather(clips, acts, idx, sel, args.horizon, dev)
            x, y = x - mu, y - mu
            loss = nn.functional.mse_loss(m(x, a), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    m.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(te), args.batch):
            sel = te[i:i + args.batch]
            x, a, y = gather(clips, acts, idx, sel, args.horizon, dev)
            x, y = x - mu, y - mu
            errs.append(((m(x, a) - y) ** 2).mean(dim=(1, 2)).cpu())
    return torch.cat(errs), npar


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--act-tokens", type=int, default=2)
    ap.add_argument("--out", default=str(SP / "e_actstream_dino.json"))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clips, acts, idx = build_index(a.window, a.horizon)
    sel = json.loads((SP / "sp2/p3_selection.json").read_text(encoding="utf-8"))
    ev = set(sel.get("eval") or sel.get("eval_clips") or [])
    if not ev:
        u = sorted({c for c, *_ in idx})
        ev = set(u[int(0.7 * len(u)):])
    tr = [i for i, (c, *_) in enumerate(idx) if c not in ev]
    te = [i for i, (c, *_) in enumerate(idx) if c in ev]
    print(f"windows {len(idx)}  train {len(tr)}  eval {len(te)}  "
          f"clips {len(clips)}  eval-clips {len(ev & set(clips))}  dev {dev}")

    # train-mean field, streamed so we never hold 21 GB
    acc, n = None, 0
    for i in range(0, len(tr), 64):
        _, _, y = gather(clips, acts, idx, [tr[k] for k in range(i, min(i + 64, len(tr)))],
                         a.horizon, dev)
        acc = y.sum(0) if acc is None else acc + y.sum(0)
        n += y.shape[0]
    mu = (acc / n)[None]
    print(f"centred on the TRAIN mean field ({n} windows)")

    # floors on the SAME windows
    pe = []
    for i in range(0, len(te), a.batch):
        x, _, y = gather(clips, acts, idx, te[i:i + a.batch], a.horizon, dev)
        pe.append((((x[:, -1] - mu) - (y - mu)) ** 2).mean(dim=(1, 2)).cpu())
    persist = torch.cat(pe)
    print(f"C-PERSIST {persist.mean():.8f}")

    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "geometry": "DINOv3 ViT-L/16, 640x1024",
           "n_windows": len(idx), "n_train": len(tr), "n_eval": len(te),
           "epochs": a.epochs, "seeds": a.seeds, "d_model": a.d,
           "n_act_tok": a.act_tokens,
           "C_PERSIST": float(persist.mean()), "arms": {}}
    per = {}
    for mode in ("concat", "add", "token"):
        ms, npar = [], None
        for s in a.seeds:
            pw, npar = run(mode, s, tr, te, clips, acts, idx, a, dev, mu)
            per.setdefault(mode, []).append(pw)
            ms.append(float(pw.mean()))
            print(f"  {mode:7s} seed {s}: {ms[-1]:.8f}  params {npar:,}")
        res["arms"][mode] = {"params": npar, "mse_per_seed": ms,
                             "mse_mean": sum(ms) / len(ms)}
    ep_te = [idx[j][0] for j in te]

    def paired(x, y, tag):
        d = (torch.stack(per[x]).mean(0) - (y if torch.is_tensor(y)
                                            else torch.stack(per[y]).mean(0)))
        by = collections.defaultdict(list)
        for e, v in zip(ep_te, d.tolist()):
            by[e].append(v)
        em = torch.tensor([sum(v) / len(v) for v in by.values()])
        g = torch.Generator().manual_seed(0)
        b = torch.stack([em[torch.randint(len(em), (len(em),), generator=g)].mean()
                         for _ in range(2000)])
        lo, hi = b.quantile(0.025).item(), b.quantile(0.975).item()
        out = {"mean": float(em.mean()), "ci95": [lo, hi],
               "separated": bool(lo > 0 or hi < 0)}
        res[tag] = out
        print(f"{tag}: {out['mean']:+.8f} [{lo:+.8f}, {hi:+.8f}] "
              f"{'SEPARATED' if out['separated'] else 'not separated'}")
        return out

    paired("token", "concat", "delta_token_minus_concat")
    paired("token", "add", "delta_token_minus_add")
    paired("token", persist, "delta_token_minus_PERSIST")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
