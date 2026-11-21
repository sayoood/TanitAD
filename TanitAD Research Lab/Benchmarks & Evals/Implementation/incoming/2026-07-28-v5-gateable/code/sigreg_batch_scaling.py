"""Does ``--batch 8 --accum 8`` really preserve the OBJECTIVE, not just the count?

THE QUESTION
------------
The staged v5 command is ``--batch 16 --accum 4``; ``--batch 16`` OOMs on a
44 GB A40 at both candidate frames, and the no-code-change fix is
``--batch 8 --accum 8`` because ``batch * accum`` stays 64. The trainer prints
``"eff_batch": batch * accum`` and its preflight refuses anything whose product
is not 64 — but BOTH of those are the same arithmetic, not a measurement of the
gradient.

The accumulation body is ``(total / accum).backward()`` per micro-batch. That
reproduces the ``batch*accum`` gradient **iff every term of ``total`` is a
per-example MEAN**. One is not:

    tanitad/models/sigreg.py::SigReg._forward_fp32

is the Epps-Pulley normality statistic over the batch. It is an **O(n^2)
pairwise** statistic — ``diff = proj[None] - proj[:, None]`` over the batch
axis — and its own source says, in as many words:

    "Do NOT normalize by n: the statistic's built-in batch-scale is part of the
     validated (lambda=0.1, slices=512) operating point."

It enters the flagship loss at ``flagship_losses.py`` as
``+ weights.sigreg * loss_sig`` with ``sigreg = 0.1``. So halving the MICRO-batch
changes what the regularizer is worth, at an unchanged ``batch * accum``.

WHAT IS MEASURED
----------------
``S(n)`` on real latents, at fixed total sample budget, for the micro-batch
sizes the two launch configurations actually use. The ratio ``S(16)/S(8)`` is
the factor by which the effective SIGReg weight moves when the launch changes
from ``16 x 4`` to ``8 x 8``.

⚠️ ``SigReg`` draws FRESH random slice directions on every call, so every value
here is an average over ``--repeats`` draws with its own spread reported. A
single call is not a measurement of this quantity.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def _sigreg(n_slices=512, beta=None):
    from tanitad.config import flagship4b_config
    from tanitad.models.sigreg import SigReg
    c = flagship4b_config()
    b = c.loss.sigreg.beta if beta is None else beta
    return (SigReg(n_slices, b), float(c.loss.sigreg.weight),
            int(c.loss.sigreg.n_slices), float(b))


def measure(z, sizes, repeats, n_slices, beta, seed=0):
    """``S(n)`` averaged over ``repeats`` slice draws, on the SAME latents.

    Every size reads the FIRST ``n`` rows of the same pool, so the comparison is
    not confounded by which samples each size happened to see."""
    sr, weight, cfg_slices, cfg_beta = _sigreg(n_slices, beta)
    out = []
    for n in sizes:
        vals = []
        for r in range(repeats):
            torch.manual_seed(seed * 1000 + r)
            vals.append(float(sr(z[:n].contiguous())))
        out.append({"n": int(n), "mean": statistics.fmean(vals),
                    "stdev": (statistics.stdev(vals) if len(vals) > 1 else 0.0),
                    "min": min(vals), "max": max(vals), "repeats": repeats})
    return out, weight, cfg_slices, cfg_beta


def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--stack", default="", help="path to <repo>/stack")
    ap.add_argument("--sizes", default="4,8,12,16,32,64")
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--n-slices", type=int, default=0,
                    help="0 = the config's own value")
    ap.add_argument("--latents", default="",
                    help="optional .pt of REAL latents [N, D]; without it a "
                         "standard-normal pool is used and that is STATED")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.stack:
        sys.path.insert(0, a.stack)

    sizes = [int(x) for x in a.sizes.split(",")]
    if a.latents:
        z = torch.load(a.latents, map_location="cpu").float()
        z = z.reshape(-1, z.shape[-1])
        source = f"REAL latents from {a.latents}, shape {tuple(z.shape)}"
    else:
        torch.manual_seed(7)
        z = torch.randn(a.pool, a.dim)
        source = (f"SYNTHETIC standard-normal pool [{a.pool}, {a.dim}] — the "
                  f"batch-size scaling of the statistic is a property of the "
                  f"ESTIMATOR, not of the data, but this is stated rather than "
                  f"hidden")
    assert z.shape[0] >= max(sizes), f"pool {z.shape[0]} < max size {max(sizes)}"

    ns = a.n_slices or None
    rows, weight, cfg_slices, cfg_beta = measure(
        z, sizes, a.repeats, ns or _sigreg()[2], None)
    by = {r["n"]: r["mean"] for r in rows}
    out = {
        "what": "SigReg (Epps-Pulley) value vs BATCH SIZE — the term that is "
                "NOT a per-example mean",
        "source_of_latents": source,
        "sigreg_weight_in_flagship_loss": weight,
        "n_slices": cfg_slices, "beta": cfg_beta,
        "repeats_per_size": a.repeats,
        "by_size": rows,
        "ratios": {
            "S(16)/S(8)  = 16x4 launch vs 8x8 launch": (
                round(by[16] / by[8], 6) if 8 in by and 16 in by else None),
            "S(64)/S(8)": (round(by[64] / by[8], 6)
                           if 8 in by and 64 in by else None),
            "S(16)/S(4)": (round(by[16] / by[4], 6)
                           if 4 in by and 16 in by else None),
        },
        "reading": (
            "With accumulation, the SIGReg contribution to one optimizer step is "
            "(1/accum) * sum_j S(micro) = S(micro), because S does not average "
            "over j. So the term's EFFECTIVE weight moves with the MICRO-batch, "
            "not with batch*accum. Every OTHER term of the flagship loss is a "
            "per-example mean and is preserved exactly."),
    }
    txt = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"[sigreg-scaling] wrote {a.out}")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
