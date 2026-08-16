"""Does self-distillation collapse actually DEVELOP on this build — and how fast?

⛔ WHY THIS RUNS BEFORE THE ABLATION. An ablation that finds "no separated
difference" is only informative if the thing being defended against was
PRESENT. If the representation does not collapse in the step budget, then
``w_o6 = 0.1`` vs ``0.0`` measures nothing, and reporting that as "no effect"
would be a power failure dressed up as a result — exactly the outcome-2 trap
the PI's reframing warns about.

So: run the CONTROL arm (``w_o6 = 0``, i.e. SigReg OFF, maximum collapse
pressure) under the SELF (collapse-capable) targets and watch the pooled
effective rank as a function of step, at the live lr and at a faster one.

``ci_reps=0`` here: this is a TRAJECTORY diagnostic, and the point estimate is
what it needs. Every decision-grade reading in the ablation carries its
jackknife.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from w_o6_ablation import (  # noqa: E402
    LIVE, POOL_STEPS, PROBE_BATCH, V6LossWeights, build, frames_for,
    pooled_spectrum, step_batch, v6_loss_step)


def trajectory(seed: int, slices: int, w_o6: float, lr: float, steps: int,
               every: int, k: int, condition: str) -> dict:
    stack = build(seed, slices)
    probe = frames_for(stack, POOL_STEPS * PROBE_BATCH, seed + 9_000)
    weights = V6LossWeights(o6_sigreg=w_o6)
    params = [p for p in stack.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=LIVE["wd"])
    pts = [{"step": 0,
            "effective_rank": pooled_spectrum(stack, probe, 0)["effective_rank"],
            "loss": None}]
    t0 = time.time()
    for step in range(1, steps + 1):
        b = step_batch(stack, step, seed, k, condition)
        out = v6_loss_step(
            stack, b, stage="S-W", o1_k=2, o5_k=k, weights=weights,
            o3_mode=LIVE["o3_mode"], o3_blocks=LIVE["o3_blocks"],
            o3_block_hw=LIVE["o3_block_hw"], o5_mode=LIVE["o5_mode"],
            generator=torch.Generator().manual_seed(seed * 1000 + step),
            sigreg_generator=torch.Generator().manual_seed(
                seed * 7000 + step + 11))
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, LIVE["clip"])
        opt.step()
        if step % every == 0 or step == steps:
            r = pooled_spectrum(stack, probe, 0)
            pts.append({"step": step, "effective_rank": r["effective_rank"],
                        "loss": float(out["loss"].detach())})
            print(f"    lr={lr} w_o6={w_o6} step {step:5d}: "
                  f"ER={r['effective_rank']:9.3f}  "
                  f"loss={float(out['loss'].detach()):.4f}", flush=True)
    er0, er1 = pts[0]["effective_rank"], pts[-1]["effective_rank"]
    return {"seed": seed, "sigreg_slices": slices, "w_o6": w_o6, "lr": lr,
            "condition": condition, "steps": steps,
            "wall_s": round(time.time() - t0, 1),
            "er_start": er0, "er_end": er1,
            "retention": er1 / er0, "collapse_factor": er0 / max(er1, 1e-9),
            "points": pts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parents[1] / "raw"
                                         / "collapse_trajectory.json"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()

    runs = []
    # the CONTROL arm (SigReg OFF) at the live lr and at 10x, under
    # collapse-capable targets — plus the same at the live lr under the
    # NULL targets, which must NOT collapse.
    for lr in (LIVE["lr"], 1e-3):
        print(f"  --- SELF, w_o6=0.0 (SigReg OFF), lr={lr} ---", flush=True)
        runs.append(trajectory(0, 512, 0.0, lr, a.steps, a.every, a.k, "SELF"))
    print("  --- FIXED (null targets), w_o6=0.0, lr=1e-3 ---", flush=True)
    runs.append(trajectory(0, 512, 0.0, 1e-3, a.steps, a.every, a.k, "FIXED"))

    out = {"meta": {
        "what": "does self-distillation collapse develop on this build, and "
                "how fast — the POWER precondition for the w_o6 ablation",
        "date": "2026-08-16", "device": "cpu",
        "evidence_class": "MEASURED (ours) on a SYNTHETIC CPU build",
        "interval": "NONE — point estimates only; this is a trajectory "
                    "diagnostic, not a decision-grade reading",
        "torch": torch.__version__}, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    for r in runs:
        print(f"  {r['condition']:5s} lr={r['lr']:<7} w_o6={r['w_o6']}: "
              f"ER {r['er_start']:.2f} -> {r['er_end']:.2f}  "
              f"(collapse factor {r['collapse_factor']:.3f})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
