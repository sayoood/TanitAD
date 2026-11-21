"""Time REFe's REAL training step on THIS GPU, per configuration -- the speed-optimisation loop.

PI instruction 2026-09-23: training must fit in <= 15 days on the pod. The FP32 estimate was
1.26-1.36 s/sample (ESTIMATED A40 multiplier), i.e. ~37-41 days for 25 scene-epochs. This file
MEASURES the step instead of estimating it, so every optimisation is a number, not an argument.

What is timed = what train.py does per micro-batch: REFe forward with per-sample camera rigs
(R22), WTA trajectory loss + a scorer BCE term, backward, grad-clip, AdamW step. Inputs are
synthetic (compute is independent of pixel values); data loading is timed separately by the
trainer itself. Each config: 3 warm-up steps, then the median of N timed steps, with
torch.cuda.synchronize() around the timer, peak memory from max_memory_allocated (resident only:
a config whose peak exceeds the card is reported, never quoted).

  python bench_speed.py --backbone vitl16 --configs fp32:4 tf32:4 bf16:4 bf16:8 --steps 6
  config = <precision>[+compile][+fusedadam]:<batch>, precision in fp32 | tf32 | bf16

Prints one line per config and ZZBENCH_<json>ZZ with every result (machine-readable).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import REFe, REFeConfig, wta_loss  # noqa: E402


def parse(cfg: str):
    left, b = cfg.split(":")
    parts = left.split("+")
    return {"prec": parts[0], "compile": "compile" in parts[1:],
            "fused": "fusedadam" in parts[1:], "batch": int(b), "name": cfg}


def set_precision(prec: str):
    tf = prec in ("tf32", "bf16")          # bf16 autocast still runs its FP32 leftovers as TF32
    torch.backends.cuda.matmul.allow_tf32 = tf
    torch.backends.cudnn.allow_tf32 = tf


def run_one(backbone: str, c: dict, steps: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    set_precision(c["prec"])
    cfg = REFeConfig.for_backbone(backbone)
    net = REFe(cfg).cuda().train()
    if c["compile"]:
        net.backbone = torch.compile(net.backbone)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01, fused=c["fused"])
    B = c["batch"]
    img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w, device="cuda")
    ego = torch.randn(B, cfg.ego_dim, device="cuda")
    goal = torch.randn(B, 2 * cfg.n_goal_points, device="cuda")
    tgt = torch.randn(B, cfg.horizon_steps, cfg.traj_dim, device="cuda")
    rig = [list(map(list, net.baked_calib())) for _ in range(B)]
    for i in range(B):
        rig[i][0][9] += 1e-6 * i            # distinct rigs: the per-sample embedding's worst case
    calib = torch.tensor(rig, dtype=torch.float64)
    n_score = cfg.n_score_components
    s_tgt = torch.rand(B, cfg.n_proposals, n_score, device="cuda")
    use_amp = c["prec"] == "bf16"

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            traj, score = net(img, ego, goal, calib=calib)
        l_traj, _ = wta_loss(traj.float(), tgt)
        l_score = F.binary_cross_entropy_with_logits(score.float(), s_tgt) * 0.1
        (l_traj + l_score).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    t_first = time.time()
    for _ in range(3):                      # warm-up (compile happens here)
        step()
    torch.cuda.synchronize()
    warm_s = time.time() - t_first
    ts = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    med = statistics.median(ts)
    out = {"config": c["name"], "batch": B, "s_per_step": round(med, 4),
           "s_per_sample": round(med / B, 4), "spread": round((max(ts) - min(ts)) / med, 3),
           "peak_gib": round(peak, 2), "card_gib": round(total, 2), "warmup_s": round(warm_s, 1),
           "resident": peak < total - 1.0}
    del net, opt, img, ego, goal, tgt, calib, s_tgt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--configs", nargs="+", default=["fp32:1", "fp32:4", "tf32:4", "bf16:4"])
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    dev = torch.cuda.get_device_name(0)
    print(f"  device {dev}  torch {torch.__version__}  backbone {a.backbone}", flush=True)
    res = []
    for cs in a.configs:
        c = parse(cs)
        try:
            r = run_one(a.backbone, c, a.steps)
        except torch.cuda.OutOfMemoryError:
            r = {"config": cs, "batch": c["batch"], "oom": True}
            torch.cuda.empty_cache()
        res.append(r)
        if r.get("oom"):
            print(f"  {cs:24s} OOM", flush=True)
        else:
            flag = "" if r["resident"] else "   *** NOT RESIDENT -- do not quote"
            print(f"  {cs:24s} {r['s_per_sample']:.4f} s/sample  ({r['s_per_step']:.3f} s/step, "
                  f"spread {100*r['spread']:.0f} %)  peak {r['peak_gib']:.1f}/{r['card_gib']:.0f} GiB  "
                  f"warm-up {r['warmup_s']:.0f} s{flag}", flush=True)
    print("ZZBENCH_" + json.dumps({"device": dev, "backbone": a.backbone, "results": res}) + "ZZ")


if __name__ == "__main__":
    main()
