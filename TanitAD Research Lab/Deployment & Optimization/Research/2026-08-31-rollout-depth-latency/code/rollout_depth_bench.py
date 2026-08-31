#!/usr/bin/env python3
"""E-LAB-DEPLOY-0831 — how does single-stream rollout latency scale with SEQUENTIAL DEPTH K?

WHY THIS EXISTS. `V7_LAUNCH_GATE.md` P4 requires reaching a 6 s horizon. The recipe reaches it
by a FLAT autoregressive rollout: at dt=0.1 that is K=60 sequential predictor steps per plan.
A rollout step cannot be batched with the step that follows it -- it consumes its own output --
so the cost is LINEAR IN K and no accelerator parallelises it away. Nobody has priced that.

WHAT THIS MEASURES. A transformer-block predictor PROXY at two scales, batch 1 (the deployment
regime), chained K times, timed against the wall. It does NOT measure our predictor: it
measures the STRUCTURE (is latency linear in K? what is the per-step floor?) so the break-even
arithmetic in RESULT.md rests on a measured curve instead of an assumption.

CONTROLS (a measurement without them is not admissible here):
  C0 NO-WORK   : K iterations of a no-op on the same harness. Isolates loop + launch overhead.
                 MUST be far below the real block and MUST still scale with K.
  C1 LINEARITY : latency(K) / K must be ~constant for a device-bound chain. If latency(K) is
                 much less than K*latency(1), the harness failed to synchronise and every
                 number is wrong -- this is the control that catches an async-timing bug.
  C2 IDENTITY  : two identical runs must agree; run-to-run spread is reported, not hidden.
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
REPS = 60
WARMUP = 15


class Block(nn.Module):
    """One pre-norm transformer block -- the unit a latent predictor step is built from."""

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


def timed(fn, reps: int = REPS, warmup: int = WARMUP):
    """p50/p95 wall-clock ms. torch.cuda.synchronize() on BOTH sides -- C1 exists to catch
    the case where this is wrong."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    return {
        "p50_ms": round(statistics.median(xs), 4),
        "p95_ms": round(xs[int(0.95 * (len(xs) - 1))], 4),
        "min_ms": round(xs[0], 4),
        "n": reps,
    }


def main() -> int:
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    ks = [1, 2, 4, 8, 16, 30, 60, 120]
    # Two scales bracketing a small latent predictor. tokens=16 -> a compact latent grid.
    arms = {
        "tiny_d4_dim384": dict(dim=384, depth=4, heads=6, tokens=16),
        "mid_d12_dim512": dict(dim=512, depth=12, heads=8, tokens=16),
    }

    out = {
        "meta": {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "dtype": "float16",
            "batch": 1,
            "reps": REPS,
            "warmup": WARMUP,
            "what_this_is": "a PROXY predictor, not TanitAD's. Measures latency-vs-sequential-depth structure.",
        },
        "arms": {},
        "controls": {},
    }

    # ---- C0: no-work chain. Pure loop + launch overhead on the identical harness. ----
    z0 = torch.zeros(1, 16, 384, device=DEV, dtype=torch.float16)
    for k in [1, 60]:
        def noop(k=k):
            z = z0
            for _ in range(k):
                z = z  # no kernel at all
            return z
        out["controls"][f"C0_nowork_K{k}"] = timed(noop)

    # A second, kernel-bearing floor: the cheapest possible per-step kernel.
    for k in [1, 60]:
        def addk(k=k):
            z = z0
            for _ in range(k):
                z = z + 1
            return z
        out["controls"][f"C0b_add1_K{k}"] = timed(addk)

    for name, cfg in arms.items():
        model = Predictor(cfg["dim"], cfg["depth"], cfg["heads"]).to(DEV).half().eval()
        n_params = sum(p.numel() for p in model.parameters())
        z = torch.randn(1, cfg["tokens"], cfg["dim"], device=DEV, dtype=torch.float16)
        rows = {}
        with torch.inference_mode():
            for k in ks:
                def chain(k=k, model=model, z=z):
                    h = z
                    for _ in range(k):
                        h = model(h)
                    return h
                r = timed(chain)
                r["ms_per_step"] = round(r["p50_ms"] / k, 4)
                rows[str(k)] = r
        out["arms"][name] = {
            "config": cfg,
            "params_M": round(n_params / 1e6, 3),
            "by_K": rows,
        }
        # ---- C1 LINEARITY: p50(K)/K must stay ~flat. ----
        per = [rows[str(k)]["ms_per_step"] for k in ks]
        out["arms"][name]["C1_linearity"] = {
            "ms_per_step_min": min(per),
            "ms_per_step_max": max(per),
            "ratio_max_over_min": round(max(per) / min(per), 3),
            "verdict_note": "device-bound chain => ratio near 1. A ratio far BELOW 1 at high K "
                            "would mean the timer never synchronised.",
        }
        # ---- C2 IDENTITY: repeat K=60 and report the spread. ----
        with torch.inference_mode():
            def chain60(model=model, z=z):
                h = z
                for _ in range(60):
                    h = model(h)
                return h
            a = timed(chain60)
            b = timed(chain60)
        out["arms"][name]["C2_identity_K60"] = {
            "run_a_p50_ms": a["p50_ms"],
            "run_b_p50_ms": b["p50_ms"],
            "rel_spread": round(abs(a["p50_ms"] - b["p50_ms"]) / a["p50_ms"], 4),
        }
        del model
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=1))
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
