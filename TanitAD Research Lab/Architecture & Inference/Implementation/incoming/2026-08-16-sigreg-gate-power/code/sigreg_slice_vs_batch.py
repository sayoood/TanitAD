#!/usr/bin/env python3
"""Does SigReg's slice resampling contribute ANY of the effective_rank spread?

⚠️ THIS SCRIPT EXISTS TO CHECK A PREMISE BEFORE ADOPTING IT. The loss-determinism
stream landed an opt-in ``sigreg_generator`` (``v6_loss_step``,
``train_v6_staged.py``) and the suggestion followed that holding the slice
directions fixed would remove part of the ``effective_rank`` spread the O6
monitor shows (3.37 -> 30.06 over 38 banked records), i.e. that some of that
spread is the *instrument* resampling its directions rather than the
representation moving.

**That premise does not hold, and the reason is structural, not statistical.**
``spectrum_report`` consumes ONE argument — the latent tensor — and computes the
SVD of its centred rows (``v6.py``). SigReg's random slice directions live
entirely inside ``o6_sigreg_loss`` / ``SigReg._forward_fp32`` and never enter the
statistic. Two calls with different ``sigreg_generator`` values produce different
LOSS values and the SAME spectrum reading on the same tensor.

So the script MEASURES the decomposition rather than asserting it:

* ``slice_component``  — vary ONLY the SigReg generator, hold the batch fixed.
                         Expected exactly 0 by the structural argument above;
                         measured so the claim is falsifiable.
* ``batch_component``  — vary the batch, hold weights and generator fixed. This
                         is where the spread actually comes from.
* ``o6_loss_component``— the same two contrasts on the O6 LOSS, to show the
                         generator genuinely moves something (otherwise a zero
                         slice component would be vacuous — the negative control
                         for the negative control).

Evidence class: **MEASURED (ours)**, CPU-only, seeded, on a small real
``V6Stack`` (not a mock), so the tensors travel the real encode -> readout path.

Usage:  python sigreg_slice_vs_batch.py --out raw/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "stack" / "tanitad").is_dir():
        sys.path.insert(0, str(_p / "stack"))
        sys.path.insert(0, str(_p / "stack" / "scripts"))
        break

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models.v6 import V6Config, V6Stack, spectrum_report  # noqa: E402
from train_v6_staged import o6_sigreg_loss  # noqa: E402


def build(seed: int = 0) -> V6Stack:
    """A small but REAL v6 stack — same classes, same readout geometry logic."""
    cfg = V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=160,
                              patch_size=16, d_model=64, depth=2, n_heads=4),
        readout=ReadoutConfig(grid=4, d_readout=32),
        predictor=PredictorConfig(d_model=64, depth=2, n_heads=4, window=6,
                                  horizons=(1, 2), action_dim=3,
                                  residual=True),
        d_tac=64, d_str=32, adapter_hidden=64, f_hidden_tac=64,
        f_hidden_str=64, f_blocks=1, aux_hidden=32, sigreg_slices=512,
        plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6),
        d_plan_feat=16, emission_hidden=16, n_candidates=8)
    torch.manual_seed(seed)
    return V6Stack(cfg).eval()


@torch.no_grad()
def latents(stack: V6Stack, gen: torch.Generator, b: int = 8,
            w: int = 6) -> torch.Tensor:
    """``[b*w, d_op]`` from the REAL encode_window path, one 'batch' of frames.

    b=8, w=6 mirrors the live run's ``--batch 8 --window 6`` = 48 rows.
    """
    c = stack.cfg.encoder
    frames = torch.randn(b, w, c.in_channels, c.image_size, c.image_width,
                         generator=gen)
    z = stack.encode_window(frames)
    return z.reshape(-1, z.shape[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    stack = build(a.seed)
    d_op = stack.cfg.d_op
    data_gen = torch.Generator().manual_seed(a.seed + 100)

    # ---- contrast A: SAME batch, DIFFERENT SigReg slice directions ---------
    z_fixed = latents(stack, data_gen)
    er_slice, loss_slice = [], []
    for r in range(a.reps):
        g = torch.Generator().manual_seed(1000 + r)
        rep = spectrum_report(z_fixed)
        er_slice.append(rep["effective_rank"])
        with torch.no_grad():
            loss_slice.append(float(o6_sigreg_loss(
                stack.sigreg, z_fixed, 0, generator=g)))

    # ---- contrast B: DIFFERENT batch, SAME (fixed) slice directions --------
    er_batch, loss_batch = [], []
    for _ in range(a.reps):
        z = latents(stack, data_gen)
        er_batch.append(spectrum_report(z)["effective_rank"])
        with torch.no_grad():
            loss_batch.append(float(o6_sigreg_loss(
                stack.sigreg, z, 0,
                generator=torch.Generator().manual_seed(1000))))

    def stats(x):
        return {"n": len(x), "mean": round(statistics.mean(x), 6),
                "sd": round(statistics.pstdev(x), 6),
                "min": round(min(x), 6), "max": round(max(x), 6),
                "range": round(max(x) - min(x), 6)}

    res = {
        "meta": {"d_op": d_op, "rows_per_reading": 48,
                 "rank_ceiling": 47, "reps": a.reps, "seed": a.seed,
                 "torch": torch.__version__,
                 "sigreg_slices": stack.cfg.sigreg_slices},
        "premise_under_test":
            "that holding SigReg's slice directions fixed removes part of the "
            "effective_rank spread the O6 monitor shows",
        "slice_component_effective_rank": stats(er_slice),
        "batch_component_effective_rank": stats(er_batch),
        "slice_component_o6_loss": stats(loss_slice),
        "batch_component_o6_loss": stats(loss_batch),
    }
    res["verdict"] = {
        "slice_moves_effective_rank": res["slice_component_effective_rank"]["range"] > 0,
        "slice_moves_o6_loss": res["slice_component_o6_loss"]["range"] > 0,
        "batch_moves_effective_rank": res["batch_component_effective_rank"]["range"] > 0,
        "share_of_ER_variance_attributable_to_slices":
            (0.0 if res["batch_component_effective_rank"]["sd"] == 0 else
             round((res["slice_component_effective_rank"]["sd"] ** 2)
                   / (res["batch_component_effective_rank"]["sd"] ** 2 +
                      res["slice_component_effective_rank"]["sd"] ** 2), 6)),
        "reading": "spectrum_report takes the latent tensor and nothing else; "
                   "SigReg's directions are internal to o6_sigreg_loss. A "
                   "fixed sigreg_generator makes the LOSS reproducible and "
                   "leaves the SPECTRUM reading untouched.",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sigreg_slice_vs_batch.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
