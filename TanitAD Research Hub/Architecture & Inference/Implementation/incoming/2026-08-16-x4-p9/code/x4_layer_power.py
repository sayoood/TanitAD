#!/usr/bin/env python3
"""X4 — size the PER-LAYER spectrum monitor's constants by MEASUREMENT.

⛔ WHY THIS EXISTS. The diagram's X4 row is "per-layer SIGReg + per-layer
spectrum monitors (O6 pattern at T/S scale)" (V6_TRAINING_MEASURES.md). The O6
pattern just went through the SIGREG_GATE_POWER.md re-derivation: the reading
must carry its own rank ceiling, the verdict must be INCONCLUSIVE-capable, and
every threshold must ship with the false-positive rate it achieves on the
estimator that will actually be used (RETRACTION_LOG C76: "a gate threshold
nobody chose, on an estimator nobody sized").

z_tac (d=512) and z_str (d=256) CANNOT inherit z_op's constants:

* their per-step row count is B (=8 on the live geometry), not B*W (=48) —
  the hierarchy uplinks ONLY the window's last frame (`V6Stack.forward`:
  `z_op = z_op_win[:, -1]`), so one step contributes 8 rows, and the
  per-batch rank ceiling is **7**, not 47;
* their d is 4x / 8x smaller, so z_op's ceiling_min=1024 EXCEEDS what a
  centred covariance over z_str (d=256) can ever reach — copying it would
  make the strategic layer INCONCLUSIVE FOREVER by construction;
* their clocks are slower (stride_tac=5, stride_str=20), so consecutive-step
  pooling spans fewer of the layer's own ticks — stated, and irrelevant to
  the estimator: rows are per-step encodings either way.

This script MEASURES, per layer, under the same generative model as
`sigreg_gate_power.py` (power-law population spectrum alpha=2, the sampler's
nested episode correlation, rho_ep=0.5 from the calibrated regime):

  1. the reading of a HEALTHY population vs one COLLAPSED to 16 retained
     directions (squeeze x0.01 — a squeeze, not a truncation, which would
     flatter the instrument), across pool sizes;
  2. the pair false-positive rate of the point `>= 0.8x` criterion;
  3. the constants by two PRE-COMMITTED rules (stated before any number):

     ceiling_min = the d/2 principle z_op's 1024 instantiates
                   (largest power of two <= d/2, capped at 1024),
                   RAISED to the next power of two if the measurement shows
                   pair-FP > 0 or healthy/collapsed separation < 3x at that
                   ceiling; if no ceiling <= d satisfies both, the layer's
                   verdict stays INCONCLUSIVE-only and the constant says so.
     floor       = the smallest power of two >= geomean(healthy, collapsed)
                   at the recommended pool. ANCHOR CHECK: this rule must
                   reproduce z_op's shipped 64 from the SIGREG_GATE_POWER
                   readings (geomean(121.6, 19.4) = 48.6 -> 64) — if it does
                   not, the rule is wrong, not the anchor.

  4. a verdict round-trip per layer at the chosen constants: collapsed must
     FAIL, healthy must NOT (o6_rank_verdict + the cluster jackknife), at the
     pooling the recommendation makes available.

Evidence class: MEASURED (ours) — simulation under a stated generative model,
CPU-only, seeded, through the REAL `spectrum_report` / `o6_rank_verdict`
(imported, never reimplemented).

Usage:
    python x4_layer_power.py --out ../raw [--reps 24]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "stack" / "tanitad").is_dir():
        sys.path.insert(0, str(_p / "stack"))
        break
from tanitad.eval.spectral import effective_rank              # noqa: E402
from tanitad.models.v6 import o6_rank_verdict, spectrum_report  # noqa: E402

# ---------------------------------------------------------------------------
# Live geometry (read off RESTART_v6F_SW.sh's argv, same source as
# sigreg_gate_power.py) and the calibrated correlation regime from that study.
# ---------------------------------------------------------------------------
LIVE_B, LIVE_W, LIVE_EPS = 8, 6, 4
RHO_EP, RHO_WIN, ALPHA = 0.5, 0.4, 2.0     # calibrated regime, inherited
COLLAPSE_KEEP, COLLAPSE_FLOOR = 16, 0.01   # the z_op floor-derivation fixture

#: layer -> (d, rows per step, frames per row-cluster w)
#: op:  z_op_win [B, W, d]  -> 48 rows of 6-frame window clusters
#: tac: z_tac    [B, d]     ->  8 rows, one frame each (last-frame uplink)
#: str: z_str    [B, d]     ->  8 rows, one frame each
LAYERS = {
    "op": {"d": 2048, "rows_per_step": LIVE_B * LIVE_W, "w": LIVE_W},
    "tac": {"d": 512, "rows_per_step": LIVE_B, "w": 1},
    "str": {"d": 256, "rows_per_step": LIVE_B, "w": 1},
}

POOLS = (1, 8, 16, 32, 33, 64)
RECOMMENDED_ACCUM = 33          # 33*8-1 = 263 >= 256 = d_tac/2 — the one-row
                                # -short-of-256 fix for the 8-rows/step layers


def powerlaw_eigs(d: int, alpha: float = ALPHA) -> torch.Tensor:
    i = torch.arange(1, d + 1, dtype=torch.float64)
    e = i ** (-float(alpha))
    return e / e.sum()


def collapsed_eigs(d: int, keep: int = COLLAPSE_KEEP,
                   floor: float = COLLAPSE_FLOOR) -> torch.Tensor:
    e = powerlaw_eigs(d)
    s = e.sqrt()
    s[keep:] = s[keep:] * floor
    e2 = s ** 2
    return e2 / e2.sum()


def true_er(eigs: torch.Tensor) -> float:
    return effective_rank(eigs.sqrt())


def draw_steps(eigs: torch.Tensor, gen: torch.Generator, *, steps: int,
               rows_per_step: int, w: int) -> torch.Tensor:
    """``[steps*rows_per_step, d]`` with the sampler's nested correlation.

    Each step draws LIVE_EPS fresh episode factors (the InteractionSampler
    redraws its episode group per call). For w>1 the row-cluster (window)
    factor carries RHO_WIN of the variance; for w=1 the window and the row are
    the same draw, so the window share merges into the idiosyncratic part —
    the per-row variance split is stated either way.
    """
    d = eigs.numel()
    n_win = rows_per_step // w
    a_ep = math.sqrt(RHO_EP)
    a_win = math.sqrt(RHO_WIN) if w > 1 else 0.0
    a_row = math.sqrt(max(1.0 - RHO_EP - (RHO_WIN if w > 1 else 0.0), 0.0))
    out = []
    for _ in range(steps):
        ep_of_win = torch.arange(n_win) % LIVE_EPS
        g_ep = torch.randn(LIVE_EPS, d, generator=gen, dtype=torch.float64)
        g_win = torch.randn(n_win, d, generator=gen, dtype=torch.float64)
        g_row = torch.randn(n_win * w, d, generator=gen, dtype=torch.float64)
        g = (a_ep * g_ep[ep_of_win].repeat_interleave(w, dim=0)
             + a_win * g_win.repeat_interleave(w, dim=0) + a_row * g_row)
        out.append((g * eigs.sqrt()).float())
    return torch.cat(out, dim=0)


def readings(eigs, gen, *, steps, rows_per_step, w, reps) -> torch.Tensor:
    return torch.tensor([
        spectrum_report(draw_steps(eigs, gen, steps=steps,
                                   rows_per_step=rows_per_step, w=w)
                        )["effective_rank"]
        for _ in range(reps)])


def largest_pow2_leq(x: float) -> int:
    return 2 ** int(math.floor(math.log2(max(x, 1.0))))


def smallest_pow2_geq(x: float) -> int:
    return 2 ** int(math.ceil(math.log2(max(x, 1.0))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parent.parent / "raw"))
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    gen = torch.Generator().manual_seed(a.seed)
    result: dict = {
        "_what": "X4 per-layer spectrum monitor sizing (see module docstring)",
        "_evidence_class": "MEASURED (ours; CPU simulation, seeded, through "
                           "the real spectrum_report/o6_rank_verdict)",
        "generative_model": {"alpha": ALPHA, "rho_ep": RHO_EP,
                             "rho_win_for_w>1": RHO_WIN,
                             "collapse": {"keep": COLLAPSE_KEEP,
                                          "floor": COLLAPSE_FLOOR},
                             "note": "inherited from sigreg_gate_power.py's "
                                     "calibrated regime"},
        "live_geometry": {"batch": LIVE_B, "window": LIVE_W,
                          "eps_per_batch": LIVE_EPS},
        "recommended_accum": RECOMMENDED_ACCUM,
        "layers": {},
    }
    for name, geo in LAYERS.items():
        d, rps, w = geo["d"], geo["rows_per_step"], geo["w"]
        healthy_eigs, coll_eigs = powerlaw_eigs(d), collapsed_eigs(d)
        rows = {}
        for pool in POOLS:
            n = pool * rps
            ceiling = min(n - 1, d)
            h = readings(healthy_eigs, gen, steps=pool, rows_per_step=rps,
                         w=w, reps=a.reps)
            c = readings(coll_eigs, gen, steps=pool, rows_per_step=rps,
                         w=w, reps=a.reps)
            # pair false-positive rate of the POINT >=0.8x criterion on an
            # UNCHANGED population: all ordered pairs (i != j)
            ratio = h[:, None] / h[None, :]
            off = ~torch.eye(len(h), dtype=torch.bool)
            fp = float((ratio[off] < 0.8).double().mean())
            rows[pool] = {
                "n_rows": n, "rank_ceiling": ceiling,
                "healthy_mean": round(float(h.mean()), 3),
                "healthy_sd": round(float(h.std()), 4),
                "collapsed_mean": round(float(c.mean()), 3),
                "collapsed_sd": round(float(c.std()), 4),
                "separation": round(float(h.mean() / c.mean()), 3),
                "pair_fp_at_0p8": round(fp, 4),
                "healthy_frac_of_ceiling": round(float(h.mean()) / ceiling, 4),
            }
            print(f"[{name}] pool={pool:3d} n={n:5d} ceil={ceiling:5d} "
                  f"healthy={rows[pool]['healthy_mean']:8.2f} "
                  f"collapsed={rows[pool]['collapsed_mean']:7.2f} "
                  f"sep={rows[pool]['separation']:5.2f}x "
                  f"FP@0.8={fp:6.4f}", flush=True)
        # ---- the two pre-committed selection rules --------------------------
        base = min(largest_pow2_leq(d / 2), 1024)
        # the measurement row whose ceiling first reaches a candidate:
        def row_at(ceil_min):
            for pool in POOLS:
                if rows[pool]["rank_ceiling"] >= ceil_min:
                    return pool, rows[pool]
            return None, None
        ceiling_min, raises = base, []
        while True:
            pool, r = row_at(ceiling_min)
            if r is None:
                ceiling_min = None      # not reachable in the scanned pools
                break
            if r["pair_fp_at_0p8"] == 0.0 and r["separation"] >= 3.0:
                break
            raises.append({"rejected": ceiling_min, "at_pool": pool,
                           "fp": r["pair_fp_at_0p8"],
                           "separation": r["separation"]})
            ceiling_min *= 2
            if ceiling_min > d:
                ceiling_min = None
                break
        # floor at the recommended pool (33 for 8-row layers; 32-33 for op)
        pool_rec = RECOMMENDED_ACCUM if rps == LIVE_B else 32
        hr, cr = rows[pool_rec]["healthy_mean"], rows[pool_rec]["collapsed_mean"]
        floor = smallest_pow2_geq(math.sqrt(hr * cr))
        result["layers"][name] = {
            "d": d, "rows_per_step": rps, "cluster_rows_w": w,
            "true_effective_rank": {"healthy": round(true_er(healthy_eigs), 2),
                                    "collapsed": round(true_er(coll_eigs), 2)},
            "per_pool": rows,
            "selection": {
                "rule_ceiling": "largest pow2 <= d/2 (cap 1024), raised while "
                                "pair-FP>0 or separation<3x at first reachable "
                                "pool; None if no ceiling <= d qualifies",
                "rule_floor": "smallest pow2 >= geomean(healthy, collapsed) "
                              "at the recommended pool",
                "base_candidate": base, "raised": raises,
                "ceiling_min": ceiling_min,
                "floor": floor,
                "floor_margin_healthy": round(hr / floor, 2),
                "floor_margin_collapsed": round(floor / cr, 2),
                "at_pool": pool_rec,
            },
        }
        # ---- verdict round-trip at the chosen constants ---------------------
        if ceiling_min is not None:
            block = w if w > 1 else 1
            def _rep(eigs):
                z = draw_steps(eigs, gen, steps=pool_rec, rows_per_step=rps,
                               w=w)
                return spectrum_report(z, ci_reps=8, block=block,
                                       generator=gen)
            ref, cur_h, cur_c = (_rep(healthy_eigs), _rep(healthy_eigs),
                                 _rep(coll_eigs))
            v_h = o6_rank_verdict(cur_h, ref, floor=float(floor),
                                  ceiling_min=ceiling_min)
            v_c = o6_rank_verdict(cur_c, ref, floor=float(floor),
                                  ceiling_min=ceiling_min)
            result["layers"][name]["verdict_roundtrip"] = {
                "healthy_vs_healthy": v_h["status"],
                "collapsed_vs_healthy": v_c["status"],
                "collapsed_reason": v_c.get("reason", "")[:120],
            }
            print(f"[{name}] verdict roundtrip: healthy={v_h['status']} "
                  f"collapsed={v_c['status']}", flush=True)
    result["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "x4_layer_power.json").write_text(json.dumps(result, indent=1))
    print(f"[x4] wrote {out / 'x4_layer_power.json'} "
          f"({result['elapsed_s']} s)", flush=True)


if __name__ == "__main__":
    main()
