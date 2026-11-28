#!/usr/bin/env python3
"""E-LAB-DEPLOY-0901 -- at EQUAL rollout budget, is it cheaper to search BROAD or DEEP?

WHY THIS EXISTS. `LAB_BACKLOG` injected row **I-1** asks whether a cheap learned terminal
value over composed h=1 rollouts beats fan-scoring *at equal rollout budget*. That question
is only decidable if we know how latency splits between the two ways of spending a budget:

    budget B = N candidates x K sequential steps

`2026-08-31-rollout-depth-latency` MEASURED that latency is LINEAR in K (ms/step flat across
K=1..120) and noted -- but did NOT measure -- that "batching helps across *candidates*, never
along it". **The whole I-1 argument rests on that unmeasured half.** This script measures it.

THE CLAIM UNDER TEST. If depth is linear and breadth is sub-linear, then at fixed B the
(large N, small K) corner is strictly cheaper in wall-clock, and a terminal value function --
which is what buys you a small K -- is a LATENCY lever, not merely a quality lever.

⚠️ If breadth turns out to be LINEAR in N too, the claim collapses and I-1's premise is wrong.
Both outcomes are committed in SPEC.md before this ran.

CONTROLS (per CLAUDE.md: a probe without controls that read known values manufactures results)
  C0 NO-WORK  : K iterations of a no-op. MUST be orders below the real block, and MUST scale
                with K (else the harness is not measuring the loop at all).
  C1 DEPTH    : at N=1, ms/step must be ~flat in K. This REPRODUCES the 2026-08-31 result on
                the identical config -- an independent re-measurement, not a citation. If it
                disagrees with that package, one of the two is wrong and neither is quotable.
  C2 BREADTH  : the load-bearing control. At fixed K, sweep N. Report ms/candidate.
  C3 IDENTITY : one arm run twice; run-to-run spread reported, never hidden.
  C4 VALUE    : cost of the terminal value head itself (MLP over N terminal latents). If this
                is not negligible against the rollout it replaces, I-1 is not "cheap".

CONFIG IS DELIBERATELY IDENTICAL to 2026-08-31 (dim 384 / depth 4 / heads 6 / tokens 16 /
fp16) so C1's N=1 row is directly comparable to that package's published K-sweep.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time

import torch
import torch.nn as nn

DEV = "cuda"
import os
REPS = int(os.environ.get('BVD_REPS', 40))
WARMUP = int(os.environ.get('BVD_WARMUP', 12))
CFG = dict(dim=384, depth=4, heads=6, tokens=16)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Predictor(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])

    def forward(self, z):
        for b in self.blocks:
            z = b(z)
        return z


class ValueHead(nn.Module):
    """The 'cheap learned terminal value' of I-1: pool the latent, score it."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, 1))

    def forward(self, z):
        return self.net(z.mean(dim=1))


def timed(fn, reps: int = REPS, warmup: int = WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(p50_ms=round(statistics.median(ts), 4),
                p95_ms=round(ts[min(len(ts) - 1, int(0.95 * len(ts)))], 4),
                min_ms=round(ts[0], 4), n=reps)


@torch.no_grad()
def rollout(model, z, K):
    for _ in range(K):
        z = model(z)
    return z


def main():
    torch.manual_seed(0)
    model = Predictor(CFG["dim"], CFG["depth"], CFG["heads"]).to(DEV).half().eval()
    vhead = ValueHead(CFG["dim"]).to(DEV).half().eval()
    n_par = sum(p.numel() for p in model.parameters())
    n_val = sum(p.numel() for p in vhead.parameters())

    def mk(N):
        return torch.randn(N, CFG["tokens"], CFG["dim"], device=DEV, dtype=torch.float16)

    out = {
        "meta": {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "dtype": "float16", "reps": REPS, "warmup": WARMUP, "config": CFG,
            "predictor_params_M": round(n_par / 1e6, 3),
            "value_head_params_M": round(n_val / 1e6, 4),
            "what_this_is": ("PROXY predictor, not TanitAD's. Measures how latency splits "
                             "between BREADTH (N candidates) and DEPTH (K steps) at equal "
                             "budget B = N*K. Config identical to 2026-08-31 package."),
        },
        "equal_budget": {}, "controls": {},
    }

    # ---- MAIN ARMS: equal budget B = N*K -------------------------------------------------
    BUDGET = 960
    for N, K in [(16, 60), (64, 15), (320, 3), (960, 1)]:
        assert N * K == BUDGET
        z = mk(N)
        r = timed(lambda: rollout(model, z, K))
        r["N"], r["K"], r["budget"] = N, K, BUDGET
        r["ms_per_candidate"] = round(r["p50_ms"] / N, 5)
        out["equal_budget"][f"N{N}_K{K}"] = r
        print(f"[budget {BUDGET}] N={N:4d} K={K:3d} -> p50 {r['p50_ms']:8.3f} ms", flush=True)

    # ---- C0: no-work loop ----------------------------------------------------------------
    c0 = {}
    for K in (1, 15, 60):
        t = torch.zeros(1, device=DEV)
        c0[str(K)] = timed(lambda: [t for _ in range(K)] and None)
    out["controls"]["C0_nowork_by_K"] = c0

    # ---- C1: DEPTH at N=1 (reproduces 2026-08-31) ----------------------------------------
    c1 = {}
    z1 = mk(1)
    for K in (1, 3, 8, 15, 30, 60):
        r = timed(lambda: rollout(model, z1, K))
        r["ms_per_step"] = round(r["p50_ms"] / K, 4)
        c1[str(K)] = r
    mps = [v["ms_per_step"] for v in c1.values()]
    out["controls"]["C1_depth_N1"] = {
        "by_K": c1, "ms_per_step_min": min(mps), "ms_per_step_max": max(mps),
        "ratio_max_over_min": round(max(mps) / min(mps), 3),
        "note": ("MUST be ~flat (device-bound chain) AND must agree with "
                 "2026-08-31 rollout_depth_4060.json tiny_d4_dim384. Disagreement => "
                 "one of the two packages is wrong and neither number is quotable."),
    }

    # ---- C2: BREADTH at fixed K (the load-bearing control) -------------------------------
    c2 = {}
    for K in (1, 3):
        row = {}
        for N in (1, 8, 32, 128, 320, 960):
            z = mk(N)
            r = timed(lambda: rollout(model, z, K))
            r["ms_per_candidate"] = round(r["p50_ms"] / N, 5)
            row[str(N)] = r
        base = row["1"]["p50_ms"]
        row["_summary"] = {
            "p50_at_N1_ms": base, "p50_at_N960_ms": row["960"]["p50_ms"],
            "observed_growth_x": round(row["960"]["p50_ms"] / base, 2),
            "linear_would_be_x": 960.0,
            "note": "sub-linear growth => candidates batch; ~960x => breadth is as serial as depth",
        }
        c2[f"K{K}"] = row
    out["controls"]["C2_breadth_fixed_K"] = c2

    # ---- C3: identity --------------------------------------------------------------------
    z = mk(320)
    a = timed(lambda: rollout(model, z, 3))
    b = timed(lambda: rollout(model, z, 3))
    out["controls"]["C3_identity_N320_K3"] = {
        "run_a_p50_ms": a["p50_ms"], "run_b_p50_ms": b["p50_ms"],
        "rel_spread": round(abs(a["p50_ms"] - b["p50_ms"]) / a["p50_ms"], 4),
    }

    # ---- C4: the value head's own cost ---------------------------------------------------
    c4 = {}
    for N in (16, 320, 960):
        zt = mk(N)
        c4[str(N)] = timed(lambda: vhead(zt))
    out["controls"]["C4_value_head_by_N"] = c4

    dst = sys.argv[1] if len(sys.argv) > 1 else "breadth_vs_depth_4060.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
