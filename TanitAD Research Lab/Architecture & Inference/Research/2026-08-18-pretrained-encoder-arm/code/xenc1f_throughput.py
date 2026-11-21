"""E-XENC-1F — THE STEP-COST RATIO, MEASURED, because it decides GPU-DAYS.

⛔ `CLAUDE.md`: *"A claim that decides a GPU-day must be MEASURED or PUBLISHED,
never INHERITED."* So the arm's cost is not asserted from parameter counts — the
encoder path is actually run forward AND backward and timed.

WHAT IS MEASURED HERE, AND WHAT IS NOT
  MEASURED  : encoder-path fwd+bwd wall time per window on the dev-box RTX 4060,
              for the incumbent ViT5Encoder, the FROZEN external swap, and the
              TRAINABLE external swap — and therefore the RATIOS between them.
  ⛔ NOT MEASURED : the arm's absolute s/step on the Thor. A ratio measured on
              one device is not a rate on another. The ratio is the transferable
              quantity and it is the only thing this file licenses.
  ⛔ NOT MEASURED : the rest of the stack (predictor, losses). This times the
              ENCODER PATH ONLY, so the whole-step ratio is STRICTLY SMALLER
              than the ratio reported here — the encoder is one term of a step
              that also runs a 190 M-parameter predictor. Reporting the encoder
              ratio as the step ratio would be the wrong-scope family.

Also identifies the backbone tensors that receive NO gradient in the trainable
arm — 222 of 223 was measured and the exception must be named, not rounded away.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[5]
for _p in (_REPO / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xenc1f_build_and_count import ExternalTokenEncoder, cfg_from_ckpt  # noqa: E402

from tanitad.models.encoder import ViT5Encoder  # noqa: E402
from tanitad.models.v6 import V6Stack  # noqa: E402


def time_path(enc, readout, cfg, dev, batch, iters, warmup, train_enc):
    """fwd+bwd over encoder->readout, timed per ITERATION (not accumulated)."""
    enc = enc.to(dev).train()
    readout = readout.to(dev).train()
    for p in enc.parameters():
        p.requires_grad_(train_enc)
    x = torch.randn(batch, cfg.encoder.in_channels, *cfg.encoder.image_hw(),
                    device=dev)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ts: list[float] = []
    for i in range(warmup + iters):
        enc.zero_grad(set_to_none=True)
        readout.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = readout(enc(x))
        z.float().pow(2).mean().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else None)
    return {
        "batch": batch, "iters": iters, "warmup": warmup,
        "secs_per_iter_median": round(statistics.median(ts), 5),
        "secs_per_iter_min": round(min(ts), 5),
        "secs_per_iter_max": round(max(ts), 5),
        "secs_per_window_median": round(statistics.median(ts) / batch, 5),
        "windows_per_s": round(batch / statistics.median(ts), 3),
        "peak_mem_GiB": round(peak, 4) if peak is not None else None,
        "_probe": "torch.cuda.max_memory_allocated() only",
        "_scope": "ENCODER PATH ONLY (encoder -> readout), not a full train step",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="facebook/dinov2-base")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(ck["_meta"]["config"]["v6_config"])
    del ck
    ro_src = V6Stack(cfg).readout

    out: dict = {
        "_evidence_class": "MEASURED (ours; dev-box RTX 4060, wall clock)",
        "eval_tier": "N/A - a cost measurement, not an eval",
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "batch": a.batch,
        "_scope_warning": (
            "ENCODER PATH ONLY. The whole-step ratio is STRICTLY SMALLER than "
            "these ratios, because a real S-W step also runs a 189,960,707-param "
            "predictor and every loss. Quoting an encoder ratio as a step ratio "
            "is the wrong-scope family."),
        "runs": {},
    }
    outp = Path(a.out)

    def bank():
        outp.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # 1. incumbent
    try:
        out["runs"]["incumbent_ViT5Encoder"] = time_path(
            ViT5Encoder(cfg.encoder), ro_src, cfg, dev, a.batch, a.iters,
            a.warmup, train_enc=True)
    except Exception as exc:
        out["runs"]["incumbent_ViT5Encoder"] = {"ERROR": f"{type(exc).__name__}: {exc}"}
    bank()

    # 2/3. the two swaps
    for tag, train_bb in (("E-XENC-1_frozen", False),
                          ("E-XENC-1F_trainable", True)):
        try:
            enc = ExternalTokenEncoder(cfg.encoder, repo=a.repo,
                                       trainable_backbone=train_bb)
            r = time_path(enc, ro_src, cfg, dev, a.batch, a.iters, a.warmup,
                          train_enc=train_bb)
            if train_bb:                      # name the no-gradient exceptions
                miss = [n for n, p in enc.backbone.named_parameters()
                        if p.grad is None]
                r["backbone_tensors_without_grad"] = miss
                r["n_backbone_tensors_without_grad"] = len(miss)
            out["runs"][tag] = r
        except Exception as exc:
            out["runs"][tag] = {"ERROR": f"{type(exc).__name__}: {exc}"}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        bank()

    base = out["runs"].get("incumbent_ViT5Encoder", {}).get("secs_per_iter_median")
    if base:
        out["ratios_vs_incumbent_ENCODER_PATH_ONLY"] = {
            k: round(v["secs_per_iter_median"] / base, 4)
            for k, v in out["runs"].items()
            if isinstance(v, dict) and "secs_per_iter_median" in v
        }
    bank()
    print(json.dumps(out, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
