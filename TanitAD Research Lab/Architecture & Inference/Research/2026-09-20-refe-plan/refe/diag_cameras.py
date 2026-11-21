"""Cost of the 4-camera design, MEASURED -- compute, memory and parameters.

The PI moved REFe from ONE front camera to FOUR, exactly as the paper and the reference config, so
REFe now differs from DriveZero in ONE variable: the backbone. This prices that decision.

⛔ ON THIS BOX ONLY `torch.cuda.max_memory_allocated()` IS ADMISSIBLE for device memory. Read it
in-process, after a real forward AND backward -- a forward-only figure understates training by the
activation graph, which is the part that actually decides whether a batch fits.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import REFeConfig, REFe, param_report, wta_loss     # noqa: E402


def run(ncam: int, batch: int, dev: str, backbone: str = "vitl16", train: bool = True):
    c = REFeConfig.for_backbone(backbone); c.n_cameras = ncam
    m = REFe(c).to(dev)
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    img = torch.randn(batch, ncam, 3, c.img_h, c.img_w, device=dev)
    ego = torch.randn(batch, c.ego_dim, device=dev)
    goal = torch.randn(batch, 2 * c.n_goal_points, device=dev)
    tgt = torch.randn(batch, c.horizon_steps, 3, device=dev)
    t0 = time.time()
    if train:
        traj, score = m(img, ego, goal)
        loss, _ = wta_loss(traj, tgt)
        loss.backward()
    else:
        with torch.no_grad():
            m(img, ego, goal)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    pk = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
    r = param_report(m)
    P = (c.img_h // c.patch) * (c.img_w // c.patch)
    del m, img, ego, goal, tgt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return dict(tokens=ncam * P, regs=ncam * c.n_registers, trainable=r["trainable"],
                total=r["total"], sec=dt, peak_gb=pk)


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}   (peak memory is only meaningful on cuda)\n")
    print(f"  {'cams':>4s} {'batch':>5s} {'tokens':>7s} {'regs':>5s} {'trainable':>12s} "
          f"{'total':>12s} {'peak GB':>8s} {'sec':>7s}")
    rows = {}
    for ncam in (1, 4):
        for b in (1, 2):
            try:
                r = run(ncam, b, dev)
            except RuntimeError as e:
                print(f"  {ncam:4d} {b:5d}   OOM/ERROR: {str(e)[:60]}")
                continue
            rows[(ncam, b)] = r
            print(f"  {ncam:4d} {b:5d} {r['tokens']:7d} {r['regs']:5d} {r['trainable']:12,d} "
                  f"{r['total']:12,d} {r['peak_gb']:8.2f} {r['sec']:7.2f}")
    if (1, 1) in rows and (4, 1) in rows:
        a, b = rows[(1, 1)], rows[(4, 1)]
        print(f"\n  4 cameras vs 1, at batch 1:")
        print(f"    tokens     x{b['tokens']/a['tokens']:.2f}")
        print(f"    peak mem   x{b['peak_gb']/max(a['peak_gb'],1e-9):.2f}  "
              f"({a['peak_gb']:.2f} -> {b['peak_gb']:.2f} GB)")
        print(f"    time       x{b['sec']/max(a['sec'],1e-9):.2f}")
        print(f"    trainable  +{b['trainable']-a['trainable']:,} "
              f"(registers only -- the frozen trunk is SHARED, not replicated)")
    print("\n  paper Table A12, FOUR cameras: 338,460,000 total / 18,580,000 trainable (5.49 %)")
    if (4, 1) in rows:
        r = rows[(4, 1)]
        print(f"  REFe         FOUR cameras: {r['total']:,} total / {r['trainable']:,} trainable "
              f"({100*r['trainable']/r['total']:.2f} %)")
        print(f"    -> trainable {100*(r['trainable']/18_580_000-1):+.1f} %, "
              f"total {100*(r['total']/338_460_000-1):+.1f} % against the published figures")
    # ⛔⛔ REFUSE TO CALL A NON-RESIDENT TIMING A MEASUREMENT. Until 2026-09-21 this file printed
    # 10.09 GB / 23.56 s on an 8 GiB card and exited 0 with `CAMERA_COST_MEASURED`, while
    # `raw/2026-09-21-camera-scaling/cam_scaling.py` refuses that exact configuration. Two
    # instruments, **24x apart on the same row**, and the one that said OK was the wrong one --
    # those seconds are the PCIe bus, not the GPU. An operator who runs this file first gets a
    # confident number with a green verdict.
    import torch as _t
    bad = []
    if _t.cuda.is_available():
        total_gb = _t.cuda.get_device_properties(0).total_memory / 1e9
        for k, r in rows.items():
            if r["peak_gb"] == r["peak_gb"] and r["peak_gb"] > total_gb - 1.0:
                bad.append((k, r["peak_gb"], r["sec"]))
        if bad:
            print(f"\n  ⛔ {len(bad)} configuration(s) are NOT RESIDENT on this "
                  f"{total_gb:.2f} GB card -- their SECONDS measure the host bus, not compute:")
            for k, pk, sec in bad:
                print(f"     cameras={k[0]} batch={k[1]}: peak {pk:.2f} GB, {sec:.2f} s "
                      f"-- NOT QUOTABLE")
            print("  The parameter and memory rows above are still valid; the timings are not.")
            print("\nCAMERA_COST_NOT_RESIDENT")
            return 1
    print("\nCAMERA_COST_MEASURED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
