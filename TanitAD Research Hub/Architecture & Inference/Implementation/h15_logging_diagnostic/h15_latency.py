"""What does the H15 imagination edge COST per decision tick? (CNCE / Efficiency)

Efficiency is a declared moat — every architectural component is judged
quality-per-FLOP. H15 runs at INFERENCE too (the A9 imagination-error self-monitor,
brain 4), so its per-tick latency is a real deployment question, not just a
training one. This measures, batch-1 on the RTX 4060 (the Orin latency proxy), at
the REAL flagship4b scale (~260 M):

  - encode_tokens  : the shared perception forward (ViT+patch) — the dominant cost
  - ImaginationField: the MARGINAL H15 edge cost given already-encoded tokens
                      (what A9 adds per tick — tokens are already computed for the
                      operative brain)
  - operative predictor: one action-conditioned forward (the core decision)

and reports the imagination edge as a % of the (encode + predictor) core tick.

Latency is weight-VALUE-invariant (only tensor shapes/dtypes matter), so an
untrained instantiation is valid for TIMING (unlike accuracy). fp32 + fp16 both,
since the deploy target is TRT-fp16 (Prod-Opt: fp16 decision-safe, bf16 not).

Run:  <venv>/python h15_latency.py --device cuda --iters 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

STACK = Path(__file__).resolve().parents[4] / "stack"
sys.path.insert(0, str(STACK))

from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.imagination import sector_mask  # noqa: E402


def _time(fn, iters, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    dev = args.device
    cuda = dev == "cuda"
    torch.manual_seed(0)

    cfg = flagship4b_config()
    model = WorldModel(cfg).to(dev).eval()
    C = cfg.encoder.in_channels
    S = cfg.encoder.image_size
    W = cfg.predictor.window
    gh = model.encoder.grid_hw
    out = {"device": dev, "config": "flagship4b", "iters": args.iters,
           "in_channels": C, "image_size": S, "grid_hw": gh,
           "state_dim": model.state_dim,
           "imagination_params_M": round(
               sum(p.numel() for p in model.imagination.parameters()) / 1e6, 2),
           "total_params_M": round(
               sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    if cuda:
        out["gpu"] = torch.cuda.get_device_name(0)

    for dt_name, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        if not cuda and dt is torch.float16:
            continue
        m = model.to(dt) if cuda else model
        with torch.no_grad():
            frame = torch.rand(1, C, S, S, device=dev, dtype=dt)
            win = torch.rand(1, W, C, S, S, device=dev, dtype=dt)
            masked, vis = sector_mask(frame, gh)

            def enc():
                return m.encode_tokens(frame)

            toks = m.encode_tokens(masked)

            def imag():
                return m.imagination(toks, vis)

            states = m.encode_window(win)
            actions = torch.randn(1, W, cfg.predictor.action_dim,
                                  device=dev, dtype=dt)

            def pred():
                return m.imagine(states, actions)

            enc_ms = _time(enc, args.iters, args.warmup, cuda)
            imag_ms = _time(imag, args.iters, args.warmup, cuda)
            pred_ms = _time(pred, args.iters, args.warmup, cuda)

        core = enc_ms + pred_ms
        out[dt_name] = {
            "encode_tokens_ms": round(enc_ms, 3),
            "imagination_ms": round(imag_ms, 3),
            "operative_predictor_ms": round(pred_ms, 3),
            "core_tick_ms_enc_plus_pred": round(core, 3),
            "imagination_pct_of_core": round(100 * imag_ms / core, 1),
        }
        # restore fp32 for the next loop
        if cuda:
            model.to(torch.float32)

    res_dir = Path(__file__).parent / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "2026-07-15-h15_latency.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
