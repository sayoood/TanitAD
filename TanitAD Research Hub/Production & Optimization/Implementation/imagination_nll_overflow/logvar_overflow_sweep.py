"""Measured experiment: the imagination-NLL `exp(-logvar)` overflow — where it
goes non-finite, how far the gradient explodes, and whether plain gradient
descent actually REACHES the overflow region.

Production & Optimization backlog **P1.7** (numerics hardening). This is a
compliance-review *measured* experiment (D-020 G-H): it produces the exact
logvar thresholds and a reachability verdict that justify the clamp shipped in
`Implementation/incoming/2026-07-11-imagination-nll-logvar-clamp/`.

Context — the live loss (`stack/tanitad/models/imagination.py:135`, wired into
`stack/tanitad/train/train_worldmodel.py:338`):

    err2 = (pred - target).pow(2).mean(dim=-1)          # [B, N]
    nll  = 0.5 * (torch.exp(-logvar) * err2 + logvar)   # <-- exp(-logvar) UNCLAMPED
    ...  = (w * nll).sum() / w.sum().clamp_min(1e-8)

`logvar` comes from an UNBOUNDED Linear head (`ImaginationField.logvar_head`,
imagination.py:110, not output-limited). `torch.exp(-logvar)` overflows to +inf
once `-logvar > ln(FLOAT_MAX)`; the product with `err2` is then +inf (or NaN if
`err2==0`), so `loss` is non-finite. There is NO nan/inf guard between the loss
and `opt.step()` (verified train_worldmodel.py:330-358): a single non-finite
cell -> non-finite `loss` -> `backward()` writes NaN into every gradient ->
`clip_grad_norm_` cannot recover (NaN in -> NaN out) -> `opt.step()` writes NaN
into every parameter -> the atomic checkpoint save then PERSISTS a corrupted
resume point. One bad cell kills the run silently.

This is deterministic, CPU-only, ~1 s, $0 — GPU contention (this run's 4060 was
100 % util) does not touch it. G-P2: the "efficiency" here is a safety envelope,
not speed, so we report the accuracy/parity delta of the fix next to it.

Run:
  python logvar_overflow_sweep.py --out logvar_overflow.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  the two NLL cell-term formulas (redefined locally so the experiment needs
#  no `tanitad` install — identical math to imagination.py::imagination_nll)
# --------------------------------------------------------------------------- #
def nll_term_original(logvar: torch.Tensor, err2: torch.Tensor) -> torch.Tensor:
    """Current stack term: 0.5*(exp(-logvar)*err2 + logvar). UNCLAMPED."""
    return 0.5 * (torch.exp(-logvar) * err2 + logvar)


def nll_term_clamped(logvar: torch.Tensor, err2: torch.Tensor,
                     clamp: float = 8.0) -> torch.Tensor:
    """Hardened term: logvar clamped to [-clamp, clamp] before exp."""
    lv = logvar.clamp(min=-clamp, max=clamp)
    return 0.5 * (torch.exp(-lv) * err2 + lv)


# --------------------------------------------------------------------------- #
def first_nonfinite_logvar(err2: float, dtype, lo=-160.0, hi=0.0, iters=200):
    """Bisection: the largest (least-negative) logvar at which the ORIGINAL
    term becomes non-finite in `dtype`. Returns None if finite over [lo, hi]."""
    e = torch.tensor(err2, dtype=dtype)
    fin_hi = torch.isfinite(nll_term_original(torch.tensor(hi, dtype=dtype), e))
    non_lo = ~torch.isfinite(nll_term_original(torch.tensor(lo, dtype=dtype), e))
    if not (fin_hi and non_lo):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if torch.isfinite(nll_term_original(torch.tensor(mid, dtype=dtype), e)):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def grad_dnll_dlogvar(logvar: float, err2: float, dtype) -> float:
    """|d(term)/d(logvar)| of the ORIGINAL term at a point (autograd)."""
    lv = torch.tensor(logvar, dtype=dtype, requires_grad=True)
    y = nll_term_original(lv, torch.tensor(err2, dtype=dtype))
    y.backward()
    g = lv.grad.item()
    return g


def gd_reachability(err2: float, dtype, lr=0.5, max_steps=2000, clamp=8.0):
    """Plain gradient descent on a single cell's logvar to MINIMISE the NLL
    term (the heteroscedastic incentive). The analytic optimum is
    logvar* = ln(err2). Report: does the ORIGINAL path reach a non-finite loss,
    and does the CLAMPED path stay finite and converge?"""
    def run(term_fn):
        lv = torch.tensor(0.0, dtype=dtype, requires_grad=True)
        opt = torch.optim.SGD([lv], lr=lr)
        for s in range(max_steps):
            opt.zero_grad()
            loss = term_fn(lv, torch.tensor(err2, dtype=dtype))
            if not torch.isfinite(loss):
                return {"nonfinite": True, "step": s,
                        "logvar_at_fail": round(float(lv.detach()), 3)}
            loss.backward()
            if not torch.isfinite(lv.grad):
                return {"nonfinite": True, "step": s, "grad_nonfinite": True,
                        "logvar_at_fail": round(float(lv.detach()), 3)}
            opt.step()
        return {"nonfinite": False, "final_logvar": round(float(lv.detach()), 3),
                "final_loss": round(float(loss.detach()), 4)}

    return {
        "logvar_star_analytic": round(math.log(err2), 3),
        "original": run(nll_term_original),
        "clamped": run(lambda lv, e: nll_term_clamped(lv, e, clamp)),
    }


def parity_in_band(clamp=8.0, n=4001):
    """Max |clamped - original| for logvar in the in-band [-clamp, clamp] over a
    realistic err2 grid — must be ~0 (clamp is identity in-band => no behaviour
    change where logvar is well-conditioned)."""
    lv = torch.linspace(-clamp, clamp, n, dtype=torch.float64)
    worst = 0.0
    for e in (1e-6, 1e-3, 0.1, 1.0, 4.0):
        a = nll_term_original(lv, torch.tensor(e, dtype=torch.float64))
        b = nll_term_clamped(lv, torch.tensor(e, dtype=torch.float64), clamp)
        worst = max(worst, float((a - b).abs().max()))
    return worst


def clamp_finiteness(clamp=8.0):
    """The clamped term must be finite for ALL logvar in [-160, 160] across a
    wide err2 range, in BOTH fp32 and fp16 (fp16 = the project's deployment
    precision and a common autocast-training mode)."""
    out = {}
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        lv = torch.linspace(-160, 160, 3201, dtype=dt)
        allfin = True
        worst_err2 = None
        for e in (0.0, 1e-6, 1e-3, 0.1, 1.0, 4.0, 20.0):
            y = nll_term_clamped(lv, torch.tensor(e, dtype=dt), clamp)
            if not bool(torch.isfinite(y).all()):
                allfin = False
                worst_err2 = e
                break
        out[name] = {"all_finite": allfin, "first_bad_err2": worst_err2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logvar_overflow.json")
    ap.add_argument("--clamp", type=float, default=8.0)
    args = ap.parse_args()
    torch.manual_seed(0)

    C = args.clamp
    report = {
        "exp": "imagination-nll logvar exp(-logvar) overflow threshold + reachability",
        "backlog": "Production&Optimization P1.7 (numerics hardening)",
        "target": "stack/tanitad/models/imagination.py:135 (imagination_nll)",
        "wired_into": "stack/tanitad/train/train_worldmodel.py:338 (loss_h15 -> loss)",
        "hardware": "CPU, deterministic (seed 0), ~1 s, $0 (GPU-contention-immune)",
        "clamp_chosen": C,
        "method": [
            "exp(-logvar) is computed BEFORE the *err2 multiply, so it overflows "
            "at -logvar > ln(FLOAT_MAX) independent of err2; the product is then "
            "+inf (or NaN if err2==0) -> non-finite loss.",
            "Thresholds found by bisection on the ORIGINAL term in each dtype.",
            "Reachability = plain SGD minimising a single cell's NLL term (the "
            "heteroscedastic incentive; analytic optimum logvar*=ln(err2)).",
            "Clamp verified: finite over logvar in [-160,160] x err2 in "
            "[0,20], fp32 AND fp16; and parity (identity) inside [-clamp,clamp].",
        ],
    }

    # 1. exact overflow thresholds (theory: -ln(FLOAT_MAX)) --------------------
    fmax32 = float(torch.finfo(torch.float32).max)
    fmax16 = float(torch.finfo(torch.float16).max)
    report["overflow_threshold_logvar"] = {
        "fp32": {
            "measured": round(first_nonfinite_logvar(1.0, torch.float32), 3),
            "theory_-ln(fp32_max)": round(-math.log(fmax32), 3),
        },
        "fp16": {
            "measured": round(first_nonfinite_logvar(1.0, torch.float16), 3),
            "theory_-ln(fp16_max)": round(-math.log(fmax16), 3),
        },
        "note": "logvar below this -> exp(-logvar)=+inf -> non-finite NLL. "
                "fp16 boundary is ~8x closer to 0 than fp32 -> far more reachable "
                "under the fp16 deployment/autocast precision.",
    }

    # err2 below which the NLL's OWN optimum logvar*=ln(err2) is already past
    # the fp16 overflow boundary (so plain convergence overflows fp16)
    lv_of16 = -math.log(fmax16)                       # ~ -11.09
    report["fp16_optimum_overflow"] = {
        "err2_below_which_optimum_overflows_fp16": round(math.exp(lv_of16), 8),
        "reading": "cells predicted better than this err2 have an NLL optimum "
                   "logvar*=ln(err2) < the fp16 overflow boundary -> gradient "
                   "descent toward the optimum crosses into +inf in fp16.",
    }

    # 2. gradient explosion of the original term (fp32) -----------------------
    report["gradient_dnll_dlogvar_fp32"] = {
        f"logvar={lv}": round(grad_dnll_dlogvar(lv, 1.0, torch.float32), 3)
        for lv in (0.0, -5.0, -20.0, -50.0, -80.0)
    }
    report["gradient_note"] = ("d(term)/d(logvar) = 0.5*(1 - exp(-logvar)*err2); "
                               "magnitude blows up as exp(-logvar) well before the "
                               "hard overflow -> clip_grad_norm then collapses the "
                               "whole step onto this one exploding direction.")

    # 3. reachability under plain SGD (fp16 and fp32) -------------------------
    report["reachability_sgd"] = {
        "fp16_err2_1e-7": gd_reachability(1e-7, torch.float16, clamp=C),
        "fp32_err2_1e-40": gd_reachability(1e-40, torch.float32, clamp=C),
    }

    # 4. the fix: finiteness + in-band parity ---------------------------------
    report["clamp_finiteness"] = clamp_finiteness(C)
    report["clamp_parity_in_band_maxabs"] = parity_in_band(C)

    # verdict -----------------------------------------------------------------
    r16 = report["reachability_sgd"]["fp16_err2_1e-7"]
    report["VERDICT"] = {
        "overflow_is_real": True,
        "fp16_reached_by_plain_sgd": r16["original"]["nonfinite"],
        "clamp_stays_finite_and_converges":
            (not r16["clamped"]["nonfinite"])
            and report["clamp_finiteness"]["fp16"]["all_finite"]
            and report["clamp_finiteness"]["fp32"]["all_finite"],
        "parity_preserved_in_band": report["clamp_parity_in_band_maxabs"] < 1e-9,
        "falsifier": "If plain SGD on a well-predicted cell did NOT reach a "
                     "non-finite loss in fp16, the overflow would be unreachable "
                     "in practice and the clamp unnecessary.",
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
