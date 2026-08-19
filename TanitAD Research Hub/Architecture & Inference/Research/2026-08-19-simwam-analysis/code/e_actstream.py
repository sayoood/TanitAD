"""E-ACTSTREAM-1 — action-as-CONDITIONING vs action-as-TOKEN.

⭐ THE QUESTION, and it is the PI's: SimWAM puts action tokens in the SAME
self-attention stream as the vision tokens. Both of our arms condition instead —
v6 feeds actions into P_O as a conditioning vector, and REF-A v1's
``TokenFieldPredictor`` broadcasts the action over tokens and
concatenates-then-projects (DINO-WM's exact scheme). **We have never tested the
joint stream.**

⛔ WHY THIS AND NOT E-SIMWAM-1. The isolated-vs-imagination experiment I proposed
is NOT RUNNABLE and I should have checked before proposing it:
  1. no v6 T1 val cache exists locally (the 4.70 GB download is a pending PI
     decision), and
  2. the imagination arm cannot be launched at all — ``mpc_refine`` requires
     ``selector="goal"`` and ``assert_selector_admissible`` REFUSES every
     selector launch while SEL-1 stands refused (E-WC2, sigma/ADE 9.9915
     against the 3.0 line).
⇒ v6 already runs SimWAM's isolated design by default (``mpc_w_consist=0.0``),
so that comparison is settled by a refusal rather than open. The joint stream is
the question that is actually open.

TASK (a pure world-model diagnostic). Given a causal window of cell fields and
the ego actions applied over the horizon, predict the FUTURE cell field.

  arm A  ``concat``  — action embedding broadcast over the 16 tokens,
                        concatenated, projected. Ours / DINO-WM.
  arm B  ``token``   — action embedding appended AS TOKENS to the vision tokens
                        in the self-attention stream; the vision tokens are read
                        out. SimWAM's layout.

⚠️ MATCHED BY CONSTRUCTION, or the result is a capacity comparison wearing an
architecture costume: same width, depth, heads, optimiser, schedule, data,
windows and seeds. Parameter counts are REPORTED, not assumed equal.

⚠️ EPISODE-DISJOINT SPLIT taken from ``p3_selection.json`` — the same split every
other instrument in this programme uses, so the numbers sit on the same windows.

TIER: **T0-DIAGNOSTIC.** A future-field prediction error is a world-model
fidelity number and is NEVER driving performance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
DT = 0.1                      # v2ep is 10 Hz
STRIDE = 4                    # the latents cache was dumped at stride 4


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def ego_actions(clip_id: str) -> torch.Tensor | None:
    """-> [T, 2] = (longitudinal accel m/s^2, yaw rate rad/s), from v2ep poses.

    ⚠️ Derived from poses, not from a label: this is the ACTUAL control the ego
    applied, which is what an action-conditioned world model must be given.
    """
    p = EPS / f"{clip_id}.v2ep.pt"
    if not p.exists():
        return None
    o = torch.load(p, map_location="cpu", weights_only=False)
    poses = o["poses"]                       # [T,4] = x, y, heading, speed
    v = poses[:, 3].float()
    h = poses[:, 2].float()
    acc = torch.zeros_like(v)
    acc[1:] = (v[1:] - v[:-1]) / DT
    dh = torch.zeros_like(h)
    d = h[1:] - h[:-1]
    d = torch.atan2(torch.sin(d), torch.cos(d))      # wrap to (-pi, pi]
    dh[1:] = d / DT
    return torch.stack([acc, dh], dim=-1)


def build(window: int, horizon: int):
    """-> (X [N,W,16,128], A [N,H,2], Y [N,16,128], ep [N])."""
    obj = torch.load(SP / "sp2/cache_s16000/latents.pt", map_location="cpu",
                     weights_only=False)
    by_clip: dict[str, dict[int, torch.Tensor]] = {}
    for r in obj["rows"]:
        by_clip.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r["cells"]
    X, A, Y, EP = [], [], [], []
    for cid, frames in by_clip.items():
        acts = ego_actions(cid)
        if acts is None:
            continue
        idx = sorted(frames)
        pos = {f: i for i, f in enumerate(idx)}
        for f in idx:
            need = [f - (window - 1 - k) * STRIDE for k in range(window)]
            tgt = f + horizon * STRIDE
            if any(n not in pos for n in need) or tgt not in pos:
                continue
            if tgt >= len(acts):
                continue
            X.append(torch.stack([frames[n] for n in need]))
            A.append(acts[f:tgt:STRIDE][:horizon])
            Y.append(frames[tgt])
            EP.append(cid)
    if not X:
        raise RuntimeError("no windows built")
    A = torch.stack([a if a.shape[0] == horizon else
                     torch.cat([a, a[-1:].repeat(horizon - a.shape[0], 1)])
                     for a in A])
    # ⚠️ the cache stores cells in fp16; every arm trains in fp32 so the
    # comparison is not a precision comparison wearing an architecture costume
    return (torch.stack(X).float(), A.float(), torch.stack(Y).float(), EP)


# --------------------------------------------------------------------------- #
# models — identical trunk, one difference
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.at(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class Predictor(nn.Module):
    """``mode='concat'`` = ours/DINO-WM · ``mode='token'`` = SimWAM's stream."""

    def __init__(self, mode: str, d_cell=128, d=192, layers=4, heads=6,
                 window=4, horizon=6, n_cells=16, n_act_tok=2):
        super().__init__()
        self.mode, self.window, self.n_cells = mode, window, n_cells
        self.inp = nn.Linear(d_cell, d)
        self.pos = nn.Parameter(torch.zeros(1, window * n_cells, d))
        self.act = nn.Sequential(nn.Linear(2 * horizon, d), nn.GELU(),
                                 nn.Linear(d, d))
        if mode == "concat":
            # broadcast over tokens, concatenate, project  (DINO-WM)
            self.mix = nn.Linear(2 * d, d)
            self.n_act_tok = 0
        elif mode == "token":
            # the action becomes TOKENS in the shared stream  (SimWAM)
            # ⚠️ n_act_tok=2 makes the arms PARAMETER-MATCHED to ~600 params
            # (act_split 74,496 vs the concat arm's mix 73,920). At 4 the token
            # arm carries +5.5 % capacity and any win is confounded.
            self.n_act_tok = int(n_act_tok)
            self.act_split = nn.Linear(d, self.n_act_tok * d)
            self.act_pos = nn.Parameter(torch.zeros(1, self.n_act_tok, d))
        else:
            raise ValueError(mode)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_cell))

    def forward(self, x, a):
        B, W, C, _ = x.shape
        t = self.inp(x).reshape(B, W * C, -1) + self.pos
        e = self.act(a.reshape(B, -1))
        if self.mode == "concat":
            t = self.mix(torch.cat([t, e[:, None].expand(-1, t.shape[1], -1)], -1))
        else:
            at = self.act_split(e).reshape(B, self.n_act_tok, -1) + self.act_pos
            t = torch.cat([t, at], dim=1)
        for b in self.blocks:
            t = b(t)
        r = self.out(t[:, (W - 1) * C:(W - 1) * C + C])
        # ⛔ RESIDUAL PREDICTION, and it is not a tweak. MEASURED 2026-08-19:
        # with the ABSOLUTE field as target, BOTH arms lost to C-PERSIST (copy
        # the last frame) — concat by 20.8x, token by 5.8x — so the delta
        # between them compared two failures. Predicting the residual makes
        # persistence exactly the zero-prediction, so any MSE below mean(r^2)
        # is skill the model added over "nothing changes".
        return x[:, -1] + r


# --------------------------------------------------------------------------- #
def run(mode, seed, tr, te, epochs, dev, d, layers, heads, window, horizon,
        n_act_tok=2):
    torch.manual_seed(seed)
    m = Predictor(mode, d=d, layers=layers, heads=heads,
                  window=window, horizon=horizon, n_act_tok=n_act_tok).to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    Xtr, Atr, Ytr = (t.to(dev) for t in tr)
    Xte, Ate, Yte = (t.to(dev) for t in te)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, bs = Xtr.shape[0], 256
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = nn.functional.mse_loss(m(Xtr[j], Atr[j]), Ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xte[i:i + 512], Ate[i:i + 512])
                          for i in range(0, Xte.shape[0], 512)])
        per_win = ((pred - Yte) ** 2).mean(dim=(1, 2))       # [N_test]
    return per_win.cpu(), n_par


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--act-tokens", type=int, default=2,
                    help="2 = parameter-matched to the concat arm; 4 = +5.5%")
    ap.add_argument("--out", default=str(SP / "e_actstream.json"))
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, A, Y, EP = build(a.window, a.horizon)
    sel = json.loads((SP / "sp2/p3_selection.json").read_text(encoding="utf-8"))
    eval_ids = set(sel.get("eval") or sel.get("eval_clips") or [])
    if not eval_ids:                       # fall back: last 30 % of episodes
        uniq = sorted(set(EP))
        eval_ids = set(uniq[int(0.7 * len(uniq)):])
    te_m = torch.tensor([e in eval_ids for e in EP])
    tr_m = ~te_m
    print(f"windows {X.shape[0]}  train {int(tr_m.sum())}  eval {int(te_m.sum())}  "
          f"episodes {len(set(EP))}  eval-episodes {len(eval_ids & set(EP))}")
    print(f"X {tuple(X.shape)}  A {tuple(A.shape)}  Y {tuple(Y.shape)}  dev {dev}")
    tr = (X[tr_m], A[tr_m], Y[tr_m])
    te = (X[te_m], A[te_m], Y[te_m])
    ep_te = [e for e, k in zip(EP, te_m.tolist()) if k]

    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC — future-field prediction error is a "
                        "world-model fidelity number, NEVER driving performance",
           "task": f"predict cell field at t+{a.horizon} from a {a.window}-frame "
                   f"causal window + ego actions (accel, yaw-rate)",
           "cache": "cache_s16000 (v6F-SW-30k@16000)",
           "n_windows": int(X.shape[0]), "n_train": int(tr_m.sum()),
           "n_eval": int(te_m.sum()), "n_eval_episodes": len(set(ep_te)),
           "epochs": a.epochs, "seeds": a.seeds,
           "n_act_tok": a.act_tokens, "arms": {}}
    per_seed = {}
    for mode in ("concat", "token"):
        mses, npar = [], None
        for s in a.seeds:
            pw, npar = run(mode, s, tr, te, a.epochs, dev, a.d, a.layers,
                           a.heads, a.window, a.horizon, a.act_tokens)
            per_seed.setdefault(mode, []).append(pw)
            mses.append(float(pw.mean()))
            print(f"  {mode:7s} seed {s}: eval MSE {mses[-1]:.6f}  params {npar:,}")
        res["arms"][mode] = {"params": npar, "mse_per_seed": mses,
                             "mse_mean": sum(mses) / len(mses)}

    # ---- paired, episode-clustered ---------------------------------------- #
    import collections
    ca = torch.stack(per_seed["concat"]).mean(0)
    cb = torch.stack(per_seed["token"]).mean(0)
    delta = cb - ca                                   # token − concat
    byep = collections.defaultdict(list)
    for e, d_ in zip(ep_te, delta.tolist()):
        byep[e].append(d_)
    ep_means = torch.tensor([sum(v) / len(v) for v in byep.values()])
    g = torch.Generator().manual_seed(0)
    boots = torch.stack([ep_means[torch.randint(len(ep_means), (len(ep_means),),
                                                generator=g)].mean()
                         for _ in range(2000)])
    lo, hi = boots.quantile(0.025).item(), boots.quantile(0.975).item()
    res["paired_delta_token_minus_concat"] = {
        "mean": float(ep_means.mean()), "ci95": [lo, hi],
        "estimator": "episode-cluster bootstrap, n_boot 2000, "
                     f"{len(ep_means)} episode clusters",
        "separated": bool(lo > 0 or hi < 0),
        "_read": "NEGATIVE means the SimWAM-style action-TOKEN stream predicts "
                 "the future field BETTER than our broadcast-concat",
    }
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    d0 = res["paired_delta_token_minus_concat"]
    print(f"\nconcat {res['arms']['concat']['mse_mean']:.6f} "
          f"({res['arms']['concat']['params']:,} par)")
    print(f"token  {res['arms']['token']['mse_mean']:.6f} "
          f"({res['arms']['token']['params']:,} par)")
    print(f"paired delta (token - concat) {d0['mean']:+.6f} "
          f"[{d0['ci95'][0]:+.6f}, {d0['ci95'][1]:+.6f}]  "
          f"{'SEPARATED' if d0['separated'] else 'not separated'}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
