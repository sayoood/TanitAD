"""⭐ IS OUR SigReg CONFIGURATION EXERTING ANTI-COLLAPSE PRESSURE AT OUR d_op?

⛔ WHY THIS EXISTS INSTEAD OF A TRAINING ABLATION. ``collapse_trajectory.py``
MEASURED that on a synthetic CPU build self-distillation collapse does NOT
develop: with SigReg fully OFF and the loss driven down 8.6x, the pooled
effective rank moved 446.49 -> 453.10 at the live lr (it went UP) and
446.49 -> 432.72 at 10x the live lr. An ablation run in that regime compares
two arms in a world where the thing being defended against is absent, and its
null would be a statement about the fixture, not about SigReg.

So this asks the question DIRECTLY, with no training and no waiting for collapse
to develop: take a latent whose rank is collapsed BY CONSTRUCTION, and ask

  1. does the SigReg statistic RESPOND to it at all at our d_op and slice count?
     (a term that cannot see collapse cannot prevent it)
  2. does its GRADIENT point back towards higher rank, and how strongly?
  3. does ``--sigreg-slices`` (live: 512) change either answer?

⛔ THE NULL-DETECTION CONTROL IS BUILT IN AND IS NOT OPTIONAL. Perturbing a
low-rank matrix in ANY direction raises its effective rank a little, so a bare
"rank went up after a SigReg step" proves nothing. Every SigReg step here is
matched against a RANDOM step OF THE SAME L2 NORM on the same latent. The
quantity that means something is the DIFFERENCE, and on an already-isotropic
latent (nothing to fix) the difference must vanish.

Geometry is the live run's, MEASURED from its argv: d_op = 4*4*128 = 2048,
n = batch 8 x window 6 = 48 rows per call, pooled 32 steps = 1536 rows,
``--sigreg-slices 512``, ``--sigreg-free-dims 0``, ``--w-o6 0.1``.

Run:  PYTHONUTF8=1 python sigreg_response.py --out ../raw/sigreg_response.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch

_HERE = Path(__file__).resolve()
_STACK = _HERE.parents[6] / "stack"
sys.path.insert(0, str(_STACK))

from tanitad.eval.spectral import effective_rank  # noqa: E402
from tanitad.models.sigreg import SigReg, position_relaxed  # noqa: E402

D_OP = 2048            # live: --readout-grid 4 * --readout-dim 128
N_CALL = 48            # live: --batch 8 * --window 6 -- THE OPERATIVE GEOMETRY
N_POOL = 192           # 4 pooled steps; see the memory bound below
SLICES = (8, 64, 512, 2048)     # 512 is LIVE; 8 is the stack/tests fixture
W_O6 = 0.1                       # live --w-o6
FREE_DIMS = 0                    # live --sigreg-free-dims

#: ⛔ SigReg's COST IS QUADRATIC IN THE ROW COUNT, and that is a hard bound on
#: this experiment and on any "just pool more rows" recommendation.
#: ``sigreg.py:118`` materialises ``diff = proj[None] - proj[:, None]`` of shape
#: **[n, n, M]** in fp32:
#:
#:      n=48   M=2048  ->      4.7 M elements  =   19 MB   (the live call)
#:      n=192  M=2048  ->     75.5 M elements  =  302 MB
#:      n=1536 M=2048  ->   4 831 M elements   = 19.3 GB   <- INFEASIBLE
#:
#: MEASURED 2026-08-16: an n=1536 sweep was launched, reached a 7.3 GB working
#: set, made no progress for 37 minutes and was killed. It was thrashing, not
#: computing. ⇒ **SigReg is applied to the 48 rows of the current batch and
#: CANNOT simply be evaluated on the pooled set** — the pooled spectrum is a
#: MONITOR-side construct only. Recorded because it constrains what any
#: configuration recommendation is allowed to say.
SIGREG_MEM_BOUND = {
    "shape": "[n, n, M] fp32 at sigreg.py:118",
    "bytes_n48_M2048": 48 * 48 * 2048 * 4,
    "bytes_n192_M2048": 192 * 192 * 2048 * 4,
    "bytes_n1536_M2048": 1536 * 1536 * 2048 * 4,
    "measured": "an n=1536 sweep hit a 7.3 GB working set and made no progress "
                "in 37 min on the dev box; killed by explicit PID",
    "consequence": "SigReg acts on the CURRENT BATCH's 48 rows (4 episodes at "
                   "--eps-per-batch 4). Pooling is available to the MONITOR, "
                   "not to the loss.",
}


def er_of(z: torch.Tensor) -> float:
    """Effective rank of the CENTRED rows — the same statistic the O6 monitor
    reports (``spectrum_report`` centres before the SVD)."""
    zc = z.detach().double()
    zc = zc - zc.mean(0, keepdim=True)
    return effective_rank(torch.linalg.svdvals(zc))


def make_z(n: int, d: int, keep: int, floor: float, seed: int) -> torch.Tensor:
    """A latent collapsed onto ``keep`` directions by a SQUEEZE, not a hard
    truncation. A hard truncation is the easiest possible thing to detect and
    would flatter the instrument (the same choice ``test_o6_spectrum_power``
    made)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.ones(d)
    s[keep:] = floor
    return torch.randn(n, d, generator=g) * s


def cell(n: int, keep: int, floor: float, slices: int, seed: int,
         step_frac: float) -> dict:
    """One (geometry, collapse level, slice count) cell.

    The SigReg step and the random control step have the SAME L2 norm, so the
    comparison is about DIRECTION.
    """
    z0 = make_z(n, D_OP, keep, floor, seed)
    er0 = er_of(z0)

    z = z0.clone().requires_grad_(True)
    sig = SigReg(slices, 1.0)
    g = torch.Generator().manual_seed(seed + 777)
    loss = position_relaxed(sig, z, FREE_DIMS, generator=g)
    (grad,) = torch.autograd.grad(W_O6 * loss, z)

    gn = float(grad.norm())
    if gn < 1e-30:
        return {"n": n, "keep": keep, "slices": slices, "seed": seed,
                "er_before": er0, "sigreg_loss": float(loss.detach()),
                "grad_norm": gn, "status": "ZERO GRADIENT — inert"}

    # step size as a fraction of ||z||, so it is comparable across cells
    step = step_frac * float(z0.norm())
    z_sig = (z0 - step * grad / gn).detach()

    gr = torch.randn(z0.shape, generator=torch.Generator().manual_seed(seed + 5))
    z_rnd = (z0 + step * gr / gr.norm()).detach()

    er_sig, er_rnd = er_of(z_sig), er_of(z_rnd)
    return {
        "n": n, "keep": keep, "slices": slices, "seed": seed,
        "er_before": er0,
        "sigreg_loss": float(loss.detach()),
        "weighted_loss_w_o6_0.1": float((W_O6 * loss).detach()),
        "grad_norm": gn,
        "er_after_sigreg_step": er_sig,
        "er_after_RANDOM_step": er_rnd,
        "d_er_sigreg": er_sig - er0,
        "d_er_random": er_rnd - er0,
        # ⚠️ DIAGNOSTIC ONLY — DO NOT READ AS "SigReg HARMS". MEASURED
        # 2026-08-16: this is negative nearly everywhere, and the decomposition
        # shows WHY — `d_er_sigreg` is ~0 while `d_er_random` grows to +0.61.
        # The comparison is STACKED: effective rank is MAXIMISED by isotropy,
        # so an isotropic random perturbation is BY CONSTRUCTION the optimal
        # one-step rank-raiser and ANY structured direction loses to it. That
        # makes "loses to noise in one step" close to tautological, and it
        # cannot distinguish "SigReg does not push towards rank" from "one step
        # of anything structured loses to isotropic noise".
        # ⇒ A single normalised gradient step is the WRONG probe for a
        # regulariser whose effect is a FIXED POINT reached over many steps.
        # The admissible readings from this file are the LOSS RESPONSE CURVE
        # (does the term SEE collapse) and `slice_variance` (what --sigreg-
        # slices buys). This key is retained so the negative result is visible
        # rather than quietly dropped.
        "anti_collapse_gain_DIAGNOSTIC_do_not_quote_as_harm":
            (er_sig - er0) - (er_rnd - er0),
        "status": "ok",
    }


def slice_variance(n: int, keep: int, floor: float, seed: int,
                   reps: int) -> dict:
    """⭐ WHAT DOES ``--sigreg-slices 512`` ACTUALLY BUY?

    The slice count does not change what the statistic MEASURES — it changes
    how noisily it is ESTIMATED, because the M directions are redrawn every
    call. So the decision-relevant quantity is the SPREAD of the statistic (and
    of its gradient) across independent direction draws AT A FIXED z, not its
    mean. Holding z fixed is what isolates the knob; varying the seed would
    confound the draw with the data.
    """
    z0 = make_z(n, D_OP, keep, floor, seed)
    out = {}
    for slices in SLICES:
        vals, gnorms = [], []
        for r in range(reps):
            z = z0.clone().requires_grad_(True)
            g = torch.Generator().manual_seed(10_000 + r)
            loss = position_relaxed(SigReg(slices, 1.0), z, FREE_DIMS,
                                    generator=g)
            (grad,) = torch.autograd.grad(loss, z)
            vals.append(float(loss.detach()))
            gnorms.append(float(grad.norm()))
        m, sd = st.mean(vals), st.stdev(vals)
        gm, gsd = st.mean(gnorms), st.stdev(gnorms)
        out[str(slices)] = {
            "slices": slices, "reps": reps,
            "loss_mean": m, "loss_sd": sd, "loss_cv": sd / m if m else None,
            "grad_norm_mean": gm, "grad_norm_sd": gsd,
            "grad_norm_cv": gsd / gm if gm else None}
    base = out[str(SLICES[0])]["loss_cv"]
    for v in out.values():
        v["cv_vs_slices8"] = (v["loss_cv"] / base) if base else None
    return {"n": n, "keep": keep, "z_fixed_seed": seed,
            "estimator": "spread over `reps` independent DIRECTION DRAWS at a "
                         "FIXED z — isolates the slice knob from the data",
            "by_slices": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parents[1] / "raw"
                                         / "sigreg_response.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--step-frac", type=float, default=0.05)
    ap.add_argument("--var-reps", type=int, default=32)
    a = ap.parse_args()

    t0 = time.time()
    seeds = list(range(a.seeds))
    #: ``keep = D_OP`` is the ISOTROPIC control: nothing is collapsed, so the
    #: anti-collapse gain must vanish. That is the null-detection arm.
    keeps = (D_OP, 1024, 512, 256, 64, 16)
    rows = []
    for n in (N_CALL, N_POOL):
        for keep in keeps:
            for slices in SLICES:
                for seed in seeds:
                    r = cell(n, keep, a.floor, slices, seed, a.step_frac)
                    rows.append(r)
        print(f"  n={n} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- summary: per (n, keep, slices), across seeds, never pooled --------
    summ = {}
    for n in (N_CALL, N_POOL):
        for keep in keeps:
            for slices in SLICES:
                sel = [r for r in rows if r["n"] == n and r["keep"] == keep
                       and r["slices"] == slices and r["status"] == "ok"]
                if not sel:
                    continue
                gain = [r["anti_collapse_gain_DIAGNOSTIC_do_not_quote_as_harm"]
                        for r in sel]
                summ[f"n{n}_keep{keep}_slices{slices}"] = {
                    "n_rows": n, "keep": keep, "slices": slices,
                    "er_before_mean": st.mean(r["er_before"] for r in sel),
                    "sigreg_loss_mean": st.mean(r["sigreg_loss"] for r in sel),
                    "grad_norm_mean": st.mean(r["grad_norm"] for r in sel),
                    "d_er_sigreg_mean": st.mean(r["d_er_sigreg"] for r in sel),
                    "d_er_random_mean": st.mean(r["d_er_random"] for r in sel),
                    "collapse_sensitivity_note":
                        "sigreg_loss_mean is the ADMISSIBLE reading here: it "
                        "answers whether the term SEES collapse at this "
                        "geometry",
                    "anti_collapse_gain_DIAGNOSTIC_mean": st.mean(gain),
                    "anti_collapse_gain_DIAGNOSTIC_min": min(gain),
                    "anti_collapse_gain_DIAGNOSTIC_max": max(gain),
                    "sign_consistent": (all(x > 0 for x in gain)
                                        or all(x < 0 for x in gain)),
                    "estimator": f"per-seed paired difference vs a "
                                 f"matched-norm RANDOM step, n_seeds={len(sel)}"}

    out = {"meta": {
        "what": "does SigReg RESPOND to collapse, and does its gradient push "
                "back — measured directly, no training required",
        "date": "2026-08-16", "device": "cpu",
        "evidence_class": "MEASURED (ours)",
        "why_not_a_training_ablation":
            "collapse_trajectory.py MEASURED that self-distillation collapse "
            "does not develop on a synthetic CPU build (ER 446.49 -> 453.10 at "
            "the live lr with SigReg OFF), so a training ablation there would "
            "measure the fixture, not SigReg",
        "live_geometry": {"d_op": D_OP, "n_per_call": N_CALL,
                          "n_second": N_POOL, "sigreg_slices_live": 512,
                          "sigreg_free_dims": FREE_DIMS, "w_o6": W_O6},
        "sigreg_memory_bound": SIGREG_MEM_BOUND,
        "null_control": "every SigReg step is matched against a RANDOM step of "
                        "IDENTICAL L2 norm; keep=2048 is the isotropic arm "
                        "where the gain must vanish",
        "squeeze_floor": a.floor, "step_frac_of_z_norm": a.step_frac,
        "seeds": seeds, "torch": torch.__version__,
        "wall_s": None},
        "summary": summ,
        "slice_variance": [slice_variance(N_CALL, k, a.floor, 0, a.var_reps)
                           for k in (D_OP, 256, 16)],
        "cells": rows}
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    for n in (N_CALL, N_POOL):
        print(f"\n=== n={n} rows (d_op={D_OP}) ===")
        print(f"{'keep':>6} {'ER':>9} | " + " | ".join(
            f"sl={s}: {'loss':>8} {'gain':>8}" for s in SLICES))
        for keep in keeps:
            k = summ.get(f"n{n}_keep{keep}_slices{SLICES[0]}")
            if not k:
                continue
            line = f"{keep:>6} {k['er_before_mean']:>9.2f} | "
            parts = []
            for s in SLICES:
                v = summ[f"n{n}_keep{keep}_slices{s}"]
                parts.append(f"sl={s}: {v['sigreg_loss_mean']:>8.4f} "
                             f"{v['anti_collapse_gain_DIAGNOSTIC_mean']:>+8.3f}")
            print(line + " | ".join(parts))
    print(f"\n=== ⭐ WHAT --sigreg-slices BUYS (n={N_CALL}, fixed z, "
          f"{a.var_reps} direction draws) ===")
    for blk in out["slice_variance"]:
        print(f"  keep={blk['keep']}:")
        for s in SLICES:
            v = blk["by_slices"][str(s)]
            print(f"    slices={s:>5}: loss {v['loss_mean']:>8.4f} "
                  f"+-{v['loss_sd']:<8.5f} CV={v['loss_cv']:.5f} "
                  f"({v['cv_vs_slices8']:.3f}x the CV at 8)  "
                  f"grad_CV={v['grad_norm_cv']:.5f}")
    print(f"\nwrote {a.out}  ({out['meta']['wall_s']} s)")


if __name__ == "__main__":
    main()
