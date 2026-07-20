"""gpu_tripwire — the CUDA half of the TanitAD test gate.

Why this exists (measured 2026-07-20): the whole 396-test `stack/tests` suite is
**CPU-only** — `grep -rl cuda stack/tests` returns nothing. Every trainer, every
eval and every deploy tick runs on a GPU, so the class of breakage that costs us
the most (a device/dtype mismatch, a CPU-only op sneaking into the inference
path, a NaN that only the CUDA kernel produces, a batch-statistic layer that is
consistent on CPU but not on device) is **invisible to `ci_gate`**. This module
closes that hole with four device-parity assertions on the real model:

  P1  encode parity      — CPU vs CUDA `WorldModel.encode` on identical inputs.
  P2  imagine parity     — CPU vs CUDA operative predictor, every horizon.
  P3  I2 on device       — batch-1 vs batch-B encoder consistency, run on CUDA
                           (the instrument doctrine tripwire, but where it is
                           actually deployed).
  P4  backward finite    — one loss.backward() on CUDA; every gradient finite
                           (guards the exp/log/div/sqrt NaN class).

It also reports the batch-1 `encode` latency on the local card, which doubles as
the cheap Orin latency proxy (I8).

Stdlib + torch only, ASCII-clean stdout (the Windows cp1252 console lesson).
No CUDA present => `available=False` and the caller decides whether that is a
skip (dev laptop) or a gate failure (`ci_gate --gpu-smoke require`).

Usage::

    python tools/gpu_tripwire.py                 # human-readable report
    python tools/gpu_tripwire.py --json out.json # machine-readable
    python tools/gpu_tripwire.py --tol 1e-3      # override the parity tolerance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Measured headroom, 2026-07-20 on an RTX 4060 (fp32, torch 2.11+cu128): the
# worst observed CPU-vs-CUDA deviation was 9.54e-07 (P1 encode) / 7.15e-07 (P2
# imagine) on activations of O(1). 1e-3 is ~1000x that -- loose enough never to
# flake on a kernel/driver change, tight enough that a real device bug (wrong
# layer, wrong dtype, a silently-truncated bf16 path) is orders of magnitude past
# it. P3 keeps the stricter 1e-4 the I2 doctrine already mandates.
DEFAULT_TOL = 1e-3


@dataclass
class Probe:
    name: str
    ok: bool
    detail: str
    value: float | None = None


@dataclass
class GpuReport:
    available: bool
    device_name: str = ""
    torch_version: str = ""
    probes: list[Probe] = field(default_factory=list)
    encode_batch1_ms: float | None = None
    wall_s: float = 0.0
    error: str | None = None

    @property
    def failures(self) -> list[Probe]:
        return [p for p in self.probes if not p.ok]

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "encode_batch1_ms": self.encode_batch1_ms,
            "wall_s": round(self.wall_s, 2),
            "error": self.error,
            "probes": [{"name": p.name, "ok": p.ok, "detail": p.detail,
                        "value": p.value} for p in self.probes],
        }


def _max_dev(a, b) -> float:
    return float((a.detach().cpu().double() - b.detach().cpu().double())
                 .abs().max().item())


def run(tol: float = DEFAULT_TOL, batch: int = 8,
        latency_iters: int = 30) -> GpuReport:
    """Run the four device-parity probes. Never raises: failures become rows."""
    t0 = time.monotonic()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dep of stack
        return GpuReport(available=False, error=f"torch not importable: {exc}")

    rep = GpuReport(available=bool(torch.cuda.is_available()),
                    torch_version=torch.__version__)
    if not rep.available:
        rep.wall_s = time.monotonic() - t0
        rep.error = "no CUDA device visible"
        return rep
    rep.device_name = torch.cuda.get_device_name(0)

    try:
        from tanitad.config import smoke_config
        from tanitad.instruments.checks import i2_batch_consistency
        from tanitad.models.fourbrain import WorldModel
    except ImportError as exc:
        rep.error = (f"cannot import the stack ({exc}) -- run from the repo root "
                     f"with PYTHONPATH=stack")
        rep.wall_s = time.monotonic() - t0
        return rep

    cfg = smoke_config()
    torch.manual_seed(0)
    model = WorldModel(cfg).eval()

    ch, px = cfg.encoder.in_channels, cfg.encoder.image_size
    win, adim = cfg.predictor.window, cfg.predictor.action_dim
    g = torch.Generator().manual_seed(1234)
    frames = torch.rand(batch, ch, px, px, generator=g)
    states = torch.randn(2, win, model.state_dim, generator=g)
    actions = torch.randn(2, win, adim, generator=g)

    with torch.no_grad():
        cpu_state = model.encode(frames)
        cpu_imag = model.imagine(states, actions)

    gmodel = model.to("cuda")
    dev_frames = frames.to("cuda")
    dev_states, dev_actions = states.to("cuda"), actions.to("cuda")

    # --- P1: encode parity -------------------------------------------------
    try:
        with torch.no_grad():
            gpu_state = gmodel.encode(dev_frames)
        d = _max_dev(cpu_state, gpu_state)
        rep.probes.append(Probe(
            "P1_encode_parity", d <= tol,
            f"max|cpu-cuda| on encode = {d:.3e} (tol {tol:g})", d))
    except Exception as exc:                      # noqa: BLE001 - report, don't crash
        rep.probes.append(Probe("P1_encode_parity", False, f"raised: {exc!r}"))

    # --- P2: imagine parity, every horizon ---------------------------------
    try:
        with torch.no_grad():
            gpu_imag = gmodel.imagine(dev_states, dev_actions)
        worst, worst_h = 0.0, None
        for h, ref in cpu_imag.items():
            d = _max_dev(ref, gpu_imag[h])
            if d > worst:
                worst, worst_h = d, h
        rep.probes.append(Probe(
            "P2_imagine_parity", worst <= tol,
            f"worst max|cpu-cuda| over horizons {sorted(cpu_imag)} = "
            f"{worst:.3e} at h={worst_h} (tol {tol:g})", worst))
    except Exception as exc:                      # noqa: BLE001
        rep.probes.append(Probe("P2_imagine_parity", False, f"raised: {exc!r}"))

    # --- P3: I2 batch-consistency, ON DEVICE -------------------------------
    try:
        with torch.no_grad():
            ok, dev = i2_batch_consistency(gmodel.encode, dev_frames, tol=1e-4)
        rep.probes.append(Probe(
            "P3_i2_on_device", bool(ok),
            f"batch-1 vs batch-{batch} deviation on CUDA = {float(dev):.3e} "
            f"(tol 1e-4)", float(dev)))
    except Exception as exc:                      # noqa: BLE001
        rep.probes.append(Probe("P3_i2_on_device", False, f"raised: {exc!r}"))

    # --- P4: one backward on device, all grads finite ----------------------
    try:
        gmodel.zero_grad(set_to_none=True)
        s = gmodel.encode(dev_frames)
        out = gmodel.imagine(dev_states.requires_grad_(False), dev_actions)
        loss = s.pow(2).mean() + sum(v.pow(2).mean() for v in out.values())
        loss.backward()
        bad = [n for n, p in gmodel.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        rep.probes.append(Probe(
            "P4_backward_finite", not bad,
            "all gradients finite" if not bad
            else f"{len(bad)} non-finite grad tensors, first: {bad[:3]}",
            float(len(bad))))
        gmodel.zero_grad(set_to_none=True)
    except Exception as exc:                      # noqa: BLE001
        rep.probes.append(Probe("P4_backward_finite", False, f"raised: {exc!r}"))

    # --- I8 proxy: batch-1 encode latency on this card ---------------------
    try:
        one = dev_frames[:1].contiguous()
        with torch.no_grad():
            for _ in range(5):                    # warm the kernels
                gmodel.encode(one)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(latency_iters):
                gmodel.encode(one)
            torch.cuda.synchronize()
        rep.encode_batch1_ms = (time.perf_counter() - t) / latency_iters * 1e3
    except Exception:                             # noqa: BLE001 - latency is informational
        rep.encode_batch1_ms = None

    rep.wall_s = time.monotonic() - t0
    return rep


def format_report(rep: GpuReport) -> str:
    if not rep.available:
        return f"[gpu_tripwire] NO CUDA -- {rep.error} (torch {rep.torch_version})"
    lines = [f"[gpu_tripwire] {rep.device_name}, torch {rep.torch_version}, "
             f"{rep.wall_s:.1f}s"]
    for p in rep.probes:
        lines.append(f"    [{'PASS' if p.ok else 'FAIL'}] {p.name}: {p.detail}")
    if rep.encode_batch1_ms is not None:
        lines.append(f"    [info] batch-1 encode latency (I8 proxy): "
                     f"{rep.encode_batch1_ms:.2f} ms")
    if rep.error:
        lines.append(f"    [info] {rep.error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TanitAD CUDA device-parity tripwire")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    ap.add_argument("--require-cuda", action="store_true",
                    help="exit 1 when no CUDA device is visible "
                         "(default: exit 0 with a loud SKIP)")
    args = ap.parse_args(argv)

    rep = run(tol=args.tol, batch=args.batch)
    print(format_report(rep), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_dict(), indent=2),
                                   encoding="utf-8")
    if not rep.available:
        return 1 if args.require_cuda else 0
    if rep.error or rep.failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
