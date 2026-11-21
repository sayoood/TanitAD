#!/usr/bin/env python3
"""H-REFAV1-MOTION — three arms on real DINOv3 features (SPEC in the Lab dir).

Arms differ in ONE thing:
  base     motion_inject=False, window [z_{t-1}, z_t]   (prev unused by design)
  motion   motion_inject=True,  window [z_{t-1}, z_t]   (the TRUE difference)
  shuffle  motion_inject=True,  window [z_{(t+37)%T}, z_t]  (WRONG difference —
           identical parameters and marginal stats; a gain that survives this
           is regularisation, not motion)

⭐ All arms train and evaluate with target_space="frozen": the metric space is
std(DINOv3), FIXED and shared, so cross-arm MSE is comparable by construction
and the adapter collapse minimum is absent.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config

CACHE = Path("C:/Users/Admin/refav1_probe/dinov3cache")
K = 30                      # 6.0 s at 0.2 s
W = 2                       # obs window (motion needs 2)
BUCKETS = ((1, 1), (2, 5), (6, 15), (16, 30))


def load_eps():
    fs = sorted(CACHE.glob("*.pt"))
    assert len(fs) == 24, len(fs)
    eps = [torch.load(f, weights_only=True) for f in fs]
    return eps[:19], eps[19:], [f.stem for f in fs[19:]]


def windows_of(ep: torch.Tensor):
    T = ep.shape[0]
    return list(range(1, T - K))          # t index of the LAST obs frame


def batch_for(eps, picks, arm: str):
    """picks: list of (ep_idx, t). Returns feats [B,W,N,d], future [B,K,N,d]."""
    fs, futs = [], []
    for ei, t in picks:
        ep = eps[ei]
        prev = (t + 37) % ep.shape[0] if arm == "shuffle" else t - 1
        fs.append(torch.stack([ep[prev], ep[t]]))
        futs.append(ep[t + 1:t + 1 + K])
    return (torch.stack(fs).float(), torch.stack(futs).float())


def run_arm(arm: str, train_eps, val_eps, a) -> dict:
    torch.manual_seed(0)
    cfg = RefAV1Config(
        d_enc=1024, n_tokens=640, d_state=1024,
        op_dt=0.2, op_steps=K, op_layers=2, op_heads=8, op_window=W,
        tac_dt=0.6, tac_steps=10, tac_queries=16, tac_layers=1,
        str_dt=3.0, str_steps=2, str_dim=64, str_layers=1,
        motion_inject=(arm != "base"), target_space="frozen")
    cfg.sanity()
    m = RefAV1(cfg).cuda()

    # ONE standardizer for every arm: fit on the same fixed sample.
    fit = torch.cat([train_eps[i][::10] for i in range(4)]).float()
    m.std.fit(fit.reshape(-1, 1024).cuda())

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)     # SAME window sequence every arm
    all_w = [(ei, t) for ei, ep in enumerate(train_eps)
             for t in windows_of(ep)]
    order = torch.randperm(len(all_w), generator=g)

    t0, losses = time.time(), []
    for step in range(a.steps):
        picks = [all_w[order[(step * a.bs + j) % len(all_w)]]
                 for j in range(a.bs)]
        feats, fut = batch_for(train_eps, picks, arm)
        out = m(feats.cuda(), torch.zeros(a.bs, K, 2, device="cuda"),
                future_feats=fut.cuda())
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(float(out["loss_feat_op"].detach()))
        if step % 100 == 0:
            print(f"  [{arm}] step {step:4d}  op {losses[-1]:.4f}  "
                  f"{time.time()-t0:5.1f}s", flush=True)

    # ---- held-out read, frozen space, per-k ------------------------------ #
    m.eval()
    per_ep = []
    with torch.no_grad():
        for ep in val_eps:
            mses = torch.zeros(K)
            copy = torch.zeros(K)
            n = 0
            for t in windows_of(ep)[::2]:                 # every 2nd window
                feats, fut = batch_for([ep], [(0, t)], arm)
                pred = m.to_enc(m(feats.cuda(),
                                  torch.zeros(1, K, 2, device="cuda"))
                                ["op_pred"])              # [1,K,N,d_enc]
                tgt = m.std(fut.cuda())
                z0 = m.std(feats[:, -1].cuda())
                mses += (pred - tgt).pow(2).mean(dim=(0, 2, 3)).cpu()
                copy += (z0[:, None] - tgt).pow(2).mean(dim=(0, 2, 3)).cpu()
                n += 1
            per_ep.append({"mse_k": (mses / n).tolist(),
                           "copy_k": (copy / n).tolist(), "n_windows": n})
    return {"arm": arm, "train_loss_last50": sum(losses[-50:]) / 50,
            "per_ep": per_ep, "train_s": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in m.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path,
                    default=Path("C:/Users/Admin/refav1_probe/result.json"))
    a = ap.parse_args()
    train_eps, val_eps, val_names = load_eps()
    print(f"train {len(train_eps)} eps / val {len(val_eps)} eps "
          f"({[n[:8] for n in val_names]})")
    res = {"spec": "H-REFAV1-MOTION", "steps": a.steps, "bs": a.bs,
           "lr": a.lr, "val_eps": val_names, "buckets": BUCKETS,
           "arms": [run_arm(arm, train_eps, val_eps, a)
                    for arm in ("base", "motion", "shuffle")]}
    a.out.write_text(json.dumps(res, indent=1))

    # ---- the read, printed ----------------------------------------------- #
    def bucket(m, lo, hi):
        return sum(m[lo - 1:hi]) / (hi - lo + 1)
    print("\nheld-out feature MSE (frozen space), mean over 5 val episodes")
    print(f"{'arm':>8} " + " ".join(f"k{lo}-{hi:>2}" for lo, hi in BUCKETS)
          + "   (copy-last floor in brackets)")
    for arm in res["arms"]:
        row, crow = [], []
        for lo, hi in BUCKETS:
            row.append(sum(bucket(e["mse_k"], lo, hi)
                           for e in arm["per_ep"]) / len(arm["per_ep"]))
            crow.append(sum(bucket(e["copy_k"], lo, hi)
                            for e in arm["per_ep"]) / len(arm["per_ep"]))
        print(f"{arm['arm']:>8} " + " ".join(f"{v:.4f}" for v in row)
              + "   [" + " ".join(f"{v:.4f}" for v in crow) + "]")
    print("\nper-episode sign of (base - motion) at k6-30, motion wins if >0:")
    b, mo = res["arms"][0], res["arms"][1]
    for i, name in enumerate(val_names):
        d = (bucket(b["per_ep"][i]["mse_k"], 6, 30)
             - bucket(mo["per_ep"][i]["mse_k"], 6, 30))
        print(f"  {name[:12]}  {d:+.5f}")


if __name__ == "__main__":
    main()
