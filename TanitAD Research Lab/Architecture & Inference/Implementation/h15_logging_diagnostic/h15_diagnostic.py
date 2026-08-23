"""H15 imagination edge — 'is it dark, or is the log lying?' diagnostic.

Program report 2026-07-14 §8 flagged a WATCH: the flagship4b train log shows
``h15=0.0`` and asked whether the H15 imagination loss is actually active/weighted.

This script answers it with MEASURED numbers on the exact code path
(``scripts/train_flagship4b.h15_loss`` + ``flagship4b_smoke_config``), on the
RTX 4060 / CPU. Four questions:

  Q1  Is the imagination module even built in the flagship config?          (structural)
  Q2  When h15_loss fires, does its gradient reach the imagination params?   (learning)
  Q3  What is the per-call FIRE RATE and loss magnitude?                     (stochastic gate)
  Q4  How often does the CURRENT logger (last-micro sample) read exactly 0.0
      even though >=1 micro in the accumulation window actually fired?       (the artifact)

Q4 is the crux: if the logger reads 0.0 on a large fraction of steps *while
imagination is training every window*, then ``h15=0.0`` is a LOGGING ARTIFACT,
not a dark edge — and the fix is a per-accum aggregate, not a training change.

Run:  <venv>/python h15_diagnostic.py --device cuda --steps 400 --accum 4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

# stack root + scripts dir on the path (mirror train_flagship4b's own bootstrap)
STACK = Path(__file__).resolve().parents[4] / "stack"
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

from train_flagship4b import FlagshipWindowDataset, h15_loss  # noqa: E402

from tanitad.config import (flagship4b_config,  # noqa: E402
                            flagship4b_smoke_config)
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402


# --- synthetic contract episodes (unicycle), same recipe as test_flagship4b --- #
def _poses(T, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T, eid, ch, size, yaw_rate=0.0, accel=0.0):
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


def make_batch(cfg, plan, n, T=260):
    ch = cfg.encoder.in_channels
    sz = cfg.encoder.image_size
    eps = [_episode(T, 0, ch, sz, yaw_rate=0.06),
           _episode(T, 1, ch, sz, yaw_rate=-0.06),
           _episode(T, 2, ch, sz, yaw_rate=0.0),
           _episode(T, 3, ch, sz, accel=-1.2)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h, channels=ch)
    return default_collate([ds[i % len(ds)] for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=400, help="fire-rate samples (Q3)")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per opt step (Q4)")
    ap.add_argument("--opt-steps", type=int, default=200, help="opt steps for Q4")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    from tanitad.train.flagship_losses import horizon_plan
    FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)
    cfg = flagship4b_smoke_config()
    plan = horizon_plan(cfg, **FAST)
    out = {"device": dev, "config": "flagship4b_smoke",
           "mask_prob": cfg.h15.mask_prob, "h15_weight": cfg.h15.weight,
           "accum": args.accum}

    # ---- Q1: is the imagination module built in the *full* flagship config? --- #
    with torch.device("meta"):
        full = WorldModel(flagship4b_config())
    out["Q1_full_flagship_imagination_is_none"] = full.imagination is None
    out["Q1_full_flagship_h15_enabled"] = flagship4b_config().h15.enabled
    out["Q1_full_flagship_imag_params_M"] = round(
        sum(p.numel() for p in full.imagination.parameters()) / 1e6, 2)

    model = WorldModel(cfg).to(dev)
    out["Q1_smoke_imagination_is_none"] = model.imagination is None
    batch = make_batch(cfg, plan, args.batch)
    frames = batch["frames"].to(dev)
    fut = batch["future_frames"].to(dev)

    # ---- Q2: gradient reach — force a fire, backprop, check imagination grads -- #
    #   h15_loss gates on torch.rand()<mask_prob; loop until it fires, then grad.
    model.zero_grad(set_to_none=True)
    fired = None
    for _ in range(50):
        lo = h15_loss(model, frames, fut, cfg, dev)
        if float(lo.item()) != 0.0:
            fired = lo
            break
    assert fired is not None, "h15_loss never fired in 50 tries (mask_prob bug?)"
    (cfg.h15.weight * fired).backward()
    imag_grad = sum(float(p.grad.abs().sum()) for p in
                    model.imagination.parameters() if p.grad is not None)
    enc_grad = float(model.encoder.patch.weight.grad.abs().sum()) \
        if model.encoder.patch.weight.grad is not None else 0.0
    out["Q2_imagination_grad_L1"] = round(imag_grad, 4)
    out["Q2_encoder_grad_through_h15_L1"] = round(enc_grad, 4)
    out["Q2_gradient_reaches_imagination"] = imag_grad > 0.0

    # ---- Q3: fire rate + magnitude over independent calls (no_grad) ----------- #
    vals = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.steps):
            vals.append(float(h15_loss(model, frames, fut, cfg, dev).item()))
    vals_t = torch.tensor(vals)
    fired_mask = vals_t != 0.0
    out["Q3_n_calls"] = args.steps
    out["Q3_fire_rate"] = round(float(fired_mask.float().mean()), 4)
    out["Q3_mean_loss_when_fired"] = round(float(vals_t[fired_mask].mean()), 4) \
        if fired_mask.any() else 0.0
    out["Q3_wallclock_s"] = round(time.time() - t0, 1)

    # ---- Q4: the artifact — current logger (last micro) vs a faithful aggregate  #
    #   Replicate the trainer's accumulation loop and record BOTH what the current
    #   line `log["h15"]=last-micro` shows and a per-accum aggregate (sum/fired).
    torch.manual_seed(1)
    current_logger_zero = 0          # steps where last-micro reads 0.0
    but_some_micro_fired = 0         # ... of those, how many had >=1 fire
    agg_zero = 0                     # steps where the aggregate is truly 0
    with torch.no_grad():
        for _ in range(args.opt_steps):
            micro_vals = [float(h15_loss(model, frames, fut, cfg, dev).item())
                          for _ in range(args.accum)]
            last = micro_vals[-1]
            n_fired = sum(1 for v in micro_vals if v != 0.0)
            if last == 0.0:
                current_logger_zero += 1
                if n_fired > 0:
                    but_some_micro_fired += 1
            if n_fired == 0:
                agg_zero += 1
    out["Q4_opt_steps"] = args.opt_steps
    out["Q4_current_logger_reads_zero_frac"] = round(current_logger_zero / args.opt_steps, 4)
    out["Q4_of_those_some_micro_fired_frac"] = round(
        but_some_micro_fired / max(1, current_logger_zero), 4)
    out["Q4_false_dark_edge_frac"] = round(but_some_micro_fired / args.opt_steps, 4)
    out["Q4_true_all_micro_masked_frac"] = round(agg_zero / args.opt_steps, 4)
    # theory: P(last micro masked) = 1-mask_prob; P(all accum masked)=(1-mask_prob)^accum
    mp = cfg.h15.mask_prob
    out["Q4_theory_last_micro_zero"] = round(1 - mp, 4)
    out["Q4_theory_all_masked"] = round((1 - mp) ** args.accum, 4)

    res_dir = Path(__file__).parent / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "2026-07-15-h15_diagnostic.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
