"""Build a TINY v0-conditioned anchor artifact so the control-space sampler can
be smoke-tested end to end through the real trainer.

⛔ **CI / SMOKE ONLY. This vocabulary is NOT a model artifact and must never be
used for a registered arm.** It is a coarse ``(a_lon, a_lat)`` grid, not the
data-driven k-means bank; its oracle-in-vocabulary ADE is unmeasured and
certainly poor. Its ONLY job is to give `--sampler ddim` a bank whose
``anchor_controls`` are non-zero, because the sampler refuses a fixed-path
vocabulary (there, ``anchor_controls`` is all zeros and the anchored Gaussian
would be centred on "do nothing").

The artifact is written through ``anchor_meta.build_anchor_artifact``, so it
declares its own units, horizon, dt, reference speed and clamps — the durable
fix for the units trap that once read a lateral acceleration as a curvature and
produced "396 g at 36 m/s" from a correct formula.

Usage::

    python make_smoke_anchors.py <out.pt> [--n-lon 5] [--n-lat 5]
"""

from __future__ import annotations

import argparse
import sys

import torch

from tanitad.models.kinematic import rollout_unicycle
from tanitad.refs.anchor_meta import build_anchor_artifact

#: the v3 6 s slot grid, in FRAMES at dt = 0.1 s
V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)

#: the grid's own ranges — the same ones `DecoderConfig.control_norm` divides by
A_LON_RANGE = (-4.0, 2.9167)
A_LAT_RANGE = (-3.0, 3.0)

REF_SPEED_MS = 10.0
KAPPA_CAP = 0.12
ALAT_V_FLOOR = 4.0


def grid_through_zero(lo: float, hi: float, n: int) -> torch.Tensor:
    """``n`` nodes spanning ``[lo, hi]`` with **0.0 guaranteed to be a node**.

    ⛔ **This function exists because the trainer REFUSED my first version, and
    it was right to.** A plain ``linspace`` omits 0.0 whenever the count is even
    on a symmetric range, or for ANY count on an asymmetric one — and
    ``A_LON_RANGE`` is asymmetric (−4.0 … +2.9167). A control grid without
    ``{a = 0, kappa = 0}`` cannot express "hold speed, go straight", and the
    trainer's guard states the cost: the set reads **1.2768 m** oracle-in-
    vocabulary against **0.2610 m** — a **4.9x artifact that looks exactly like
    a resolution finding**.

    ⭐ Worth recording as a class: the guard turned a silent 4.9x measurement
    artifact into a loud refusal at wiring time, which is the only point at
    which it is cheap. This is the value of a refusal over a warning.
    """
    if n < 3:
        raise ValueError(f"need >= 3 nodes to straddle zero, got {n}")
    if not (lo < 0.0 < hi):
        raise ValueError(f"range [{lo}, {hi}] does not straddle zero")
    # split the remaining nodes between the two sides in proportion to span
    span = (hi - lo)
    k_neg = max(1, round((0.0 - lo) / span * (n - 1)))
    k_pos = (n - 1) - k_neg
    if k_pos < 1:
        k_neg, k_pos = n - 2, 1
    out = torch.cat([torch.linspace(lo, 0.0, k_neg + 1)[:-1],
                     torch.tensor([0.0]),
                     torch.linspace(0.0, hi, k_pos + 1)[1:]])
    assert bool((out == 0.0).any()), "zero is not a node — the guard's case"
    assert out.numel() == n, (out.numel(), n)
    return out


def build(n_lon: int = 5, n_lat: int = 5, horizons=V3_HORIZONS,
          ref_speed_ms: float = REF_SPEED_MS) -> dict:
    lon = grid_through_zero(*A_LON_RANGE, n_lon)
    lat = grid_through_zero(*A_LAT_RANGE, n_lat)
    controls = torch.stack(torch.meshgrid(lon, lat, indexing="ij"),
                           dim=-1).reshape(-1, 2)                  # [N, 2]
    n = controls.shape[0]
    # roll at the REFERENCE speed, exactly as `roll_bank` does for a fixed
    # build: `anchors` stays the checkpoint-visible artifact a content check
    # can compare, and `controls` is what the decoder re-rolls per window.
    kappa = (controls[:, 1] / max(ref_speed_ms, ALAT_V_FLOOR) ** 2).clamp(
        -KAPPA_CAP, KAPPA_CAP)
    h = int(max(horizons))
    ctrl_ticks = torch.stack([controls[:, 0], kappa], dim=-1)
    ctrl_ticks = ctrl_ticks[:, None, :].expand(n, h, 2).contiguous()
    state0 = torch.zeros(n, 4)
    state0[:, 3] = ref_speed_ms
    path = rollout_unicycle(state0, ctrl_ticks, dt=0.1)[..., :2]
    idx = torch.tensor([k - 1 for k in horizons], dtype=torch.long)
    anchors = path.index_select(1, idx)                            # [N, S, 2]
    return build_anchor_artifact(
        anchors, controls, control_units="alat", horizons=horizons, dt=0.1,
        ref_speed_ms=ref_speed_ms, kappa_cap=KAPPA_CAP,
        alat_v_floor=ALAT_V_FLOOR, builder=__file__,
        extra={"ci_only": True,
               "warning": "SMOKE-ONLY coarse grid; never a registered arm"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--n-lon", type=int, default=5)
    ap.add_argument("--n-lat", type=int, default=5)
    a = ap.parse_args(argv)
    art = build(a.n_lon, a.n_lat)
    torch.save(art, a.out)
    print(f"[smoke-anchors] {art['anchors'].shape[0]} anchors "
          f"{tuple(art['anchors'].shape)} controls "
          f"{tuple(art['controls'].shape)} units={art['control_units']} "
          f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
