"""How does REFe's step time scale with the number of cameras? MEASURE it, do not assume x4.

⛔ WHY THIS EXISTS. Review 5 refuted the banked 4-camera seconds: at ViT-L the 4-camera config
requests ~10 GB against ~7.4 GB free on this RTX 4060, so those timings were HOST-MEMORY FALLBACK
THRASH, not compute. It replaced them with an ANALYTIC x3.999 (the frozen trunk is 99.19 % of the
work). Analytic is better than thrash, but it is still not a measurement.

⭐ THE CHEAP DISCRIMINATING EXPERIMENT: measure the camera scaling at a backbone that FITS, where
nothing pages, and check whether it lands on xN. If it does, applying it to the ViT-L one-camera
number is an extrapolation along a VERIFIED law rather than an assumption. If it does not, the
A40 estimate is wrong and we learn that before buying 25 days of compute.

Guards, because a timing probe is easy to fool:
  * warm up, then torch.cuda.synchronize() around the timer, and take the MEDIAN of several reps;
  * assert the config is RESIDENT -- max_memory_allocated() well under the free pool -- and SKIP,
    loudly, anything that is not. A number produced while paging is worse than no number.
  * report s/sample, not s/step, so batch and camera count cannot be conflated.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(
    "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-20-refe-plan/refe")))
import model as M  # noqa: E402


def time_cfg(backbone: str, n_cam: int, batch: int, reps: int = 5, dev: str = "cuda"):
    cfg = M.REFeConfig.for_backbone(backbone)
    cfg.img_h, cfg.img_w = 512, 960
    cfg.n_cameras = n_cam
    cfg.cameras = tuple(("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")[:n_cam])
    net = M.REFe(cfg).to(dev)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    img = torch.randn(batch, n_cam, 3, cfg.img_h, cfg.img_w, device=dev)
    ego = torch.randn(batch, cfg.ego_dim, device=dev)
    goal = torch.randn(batch, 2 * cfg.n_goal_points, device=dev)
    tgt = torch.randn(batch, cfg.horizon_steps, cfg.traj_dim, device=dev)

    def one():
        opt.zero_grad(set_to_none=True)
        traj, score = net(img, ego, goal)
        loss, _ = M.wta_loss(traj, tgt)
        (loss + score.sum() * 0.0).backward()
        opt.step()

    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):                       # warm up: allocator, cuDNN algo choice, autotune
        one()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        one()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    del net, opt, img, ego, goal, tgt
    torch.cuda.empty_cache()
    return statistics.median(ts), peak, total / 2**30


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA -- this probe is about device time"); return 2
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device: {name}  {total:.2f} GiB\n")
    print(f"  {'backbone':9s} {'cams':>5s} {'batch':>6s} {'s/step':>9s} {'s/sample':>10s} "
          f"{'peak GiB':>9s}  note")
    rows = {}
    for backbone in ("vits16", "vitb16", "vitl16"):
        for n_cam in (1, 2, 4):
            batch = 1
            try:
                s, peak, _ = time_cfg(backbone, n_cam, batch)
            except torch.cuda.OutOfMemoryError:
                print(f"  {backbone:9s} {n_cam:5d} {batch:6d} {'OOM':>9s} {'-':>10s} {'-':>9s}  "
                      f"does not fit -- SKIPPED rather than timed while paging")
                torch.cuda.empty_cache()
                continue
            # ⛔ a config that fits only by spilling to host RAM produces a plausible number that
            # measures the PCIe bus. Refuse it explicitly instead of publishing it.
            headroom = total - peak
            note = "ok" if headroom > 1.0 else "TOO CLOSE TO THE CARD -- not quotable"
            rows[(backbone, n_cam)] = (s, peak, note)
            print(f"  {backbone:9s} {n_cam:5d} {batch:6d} {s:9.3f} {s/batch:10.3f} {peak:9.2f}  {note}")

    print("\n== the question: is the step time LINEAR in the camera count? ==")
    print(f"  {'backbone':9s} {'t(2)/t(1)':>11s} {'t(4)/t(1)':>11s}   (analytic prediction: 2.00 / 4.00)")
    verdict = []
    for backbone in ("vits16", "vitb16", "vitl16"):
        r1 = rows.get((backbone, 1))
        r2 = rows.get((backbone, 2))
        r4 = rows.get((backbone, 4))
        if not r1 or "ok" not in r1[2]:
            continue
        a = f"{r2[0]/r1[0]:.3f}" if r2 and "ok" in r2[2] else "  --"
        b = f"{r4[0]/r1[0]:.3f}" if r4 and "ok" in r4[2] else "  --"
        print(f"  {backbone:9s} {a:>11s} {b:>11s}")
        if r4 and "ok" in r4[2]:
            verdict.append((backbone, r4[0] / r1[0]))
    if verdict:
        print("\n  MEASURED 4-camera multipliers: " +
              ", ".join(f"{b} {v:.3f}" for b, v in verdict))
        lo, hi = min(v for _, v in verdict), max(v for _, v in verdict)
        print(f"  range {lo:.3f}-{hi:.3f} against the analytic 4.00 "
              f"({'CONSISTENT' if 3.5 <= lo and hi <= 4.5 else 'NOT consistent -- investigate'})")
    else:
        print("\n  no backbone gave a resident 4-camera timing on this card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
