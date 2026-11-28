"""Does the learning rate actually REACH ZERO? The paper says "decayed to zero via cosine annealing".

⛔ WHY THIS EXISTS RATHER THAN A TRAINING RUN. Verifying the schedule by training is expensive and
answers the wrong question: a 400-step run under load took 108 s for its FIRST step, and a training
curve cannot distinguish "the scheduler is configured correctly" from "the model happened to
improve". The schedule is a property of the optimiser, so test the optimiser.

⭐ THE CONTROL IS THE `--no-cosine` ARM. A check that only asserts "the LR went down" passes on any
decaying schedule and on several wrong ones. The ablation arm must hold the LR FLAT at 2e-4; if
both arms decay, the flag does nothing and the test proves nothing.

Usage:  python diag_schedule.py [--steps 62]
"""
from __future__ import annotations

import argparse

import torch


def lr_trace(steps: int, lr: float, cosine: bool):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=1e-4)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=0.0)
             if cosine else None)
    trace = []
    for _ in range(steps):
        trace.append(opt.param_groups[0]["lr"])
        p.grad = torch.zeros_like(p)
        opt.step()
        if sched is not None:
            sched.step()
    trace.append(opt.param_groups[0]["lr"])       # the value AFTER the last step
    return trace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=62)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args(argv)

    print("The paper: AdamW, initial LR 2e-4, DECAYED TO ZERO via cosine annealing.\n")
    cos = lr_trace(a.steps, a.lr, True)
    flat = lr_trace(a.steps, a.lr, False)
    idx = [0, a.steps // 4, a.steps // 2, (3 * a.steps) // 4, a.steps]
    print(f"  {'step':>6s} {'cosine':>12s} {'--no-cosine':>14s}")
    for i in idx:
        print(f"  {i:6d} {cos[i]:12.3e} {flat[i]:14.3e}")

    checks = {
        "starts at the paper's 2e-4": abs(cos[0] - a.lr) < 1e-12,
        "ends at EXACTLY zero": cos[-1] == 0.0,
        "is monotonically non-increasing": all(cos[i] >= cos[i + 1] for i in range(len(cos) - 1)),
        "halfway value is near half (cosine, not linear)":
            abs(cos[a.steps // 2] - a.lr / 2) < a.lr * 0.1,
        "CONTROL: --no-cosine stays FLAT at 2e-4": all(abs(v - a.lr) < 1e-12 for v in flat),
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(checks.values())
    print("\n" + ("SCHEDULE_OK" if good else "SCHEDULE_FAILED"))
    if not checks["CONTROL: --no-cosine stays FLAT at 2e-4"]:
        print("  *** the ablation arm decayed too -- the flag does nothing and this test is inert")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
