"""TWO-SEGMENT anchor candidates: the vocabulary can express a lane change.

⛔ WHY THIS MODULE EXISTS — MEASURED 2026-09-06 (commit ``56dc078``, 24,114
windows / 141 eval clips at stride 1):

* ``a_tac.lat == LANE_CHANGE_L`` is **0 / 4,572** train clips and **0 / 147** eval
  clips (same for ``LANE_CHANGE_R`` and ``ABORT_LC``), against a read-control where
  the other five classes read 259-2,958. The label emitter simply cannot emit one.
* ⭐ **But the behaviour is in the DATA**: model-free from ego poses, **1,297 of
  18,615** windows over **79 of 141** clips execute a lane-change shape.
* ⛔ **And the 117-candidate bank cannot represent it.** Every candidate is a
  constant-curvature arc, so **yaw is monotone on 1.000000 of 564,291 (window,
  candidate) pairs** — no candidate is an S-shape. A shallow arc can reach a lane
  width, but the **minimum residual heading among all pairs reaching >= 2.5 m is
  4.80 deg**: it arrives pointing the wrong way.

⭐ **A vocabulary is a CEILING.** No selector, however good, can pick a trajectory
the fan does not contain. This module adds the cheapest family that contains one.

⭐⭐ **AND IT NEEDS NO LABEL CHANGE.** The anchor classifier's target is
``a_star = dist.argmin(1)`` where ``dist`` is the squared distance from
``traj_tgt`` (``refb_labels.waypoint_targets`` — the RECORDED future ego path) to
``out["anchor_bank"]`` (the EMITTED fan), in ``refc_v3_train.py``. It reads no
tactical label at all, so adding a candidate changes what the classifier is
supervised toward **without touching a label set** (PI, 2026-09-06: *"don't
generate labels again"*). ⚠️ This does NOT fix the tactical head's three dead
lane-change classes — that IS a label change, and it is a different head.

THE FAMILY
----------
A two-segment candidate holds ``+a_lat`` for ``t_split`` seconds and ``-a_lat``
for the rest of the horizon, with ``a_lon`` constant throughout: the incumbent's
own control alphabet, applied twice. Same integrator
(:func:`tanitad.models.kinematic.rollout_unicycle`), same
``kappa = clamp(a_lat / max(v0, alat_v_floor)^2, +-kappa_cap)`` map, same slots.

``controls`` gains a THIRD column:

======  ==============  =====================================================
col     name            units
======  ==============  =====================================================
0       ``a_lon_ms2``   m/s^2, longitudinal acceleration, constant
1       ``a_lat_ms2``   m/s^2, lateral acceleration, sign flips at ``t_split``
2       ``t_split_s``   **s**, the time the lateral sign flips
======  ==============  =====================================================

⭐ **THE INCUMBENT IS THE ``t_split_s = horizon_s`` SPECIAL CASE.** With the split
at (or beyond) the horizon the sign is ``+1`` at every tick, so the two-segment
integrator **is** the constant integrator — bit-identically, asserted by
``tests/test_anchor_twoseg.py::test_single_segment_limit_is_bit_identical``.
That is what makes the default-OFF path provable rather than asserted, and it is
what makes the degenerate-split regression arm an EXACT zero rather than a small
number.

⛔ **THE SPLIT IS SNAPPED TO A TICK, NOT COMPARED IN FLOATING POINT.** ``k * dt``
for ``dt = 0.1`` is not exact in binary, so ``k * dt < t_split`` is a coin-flip at
the boundary. :func:`split_ticks` rounds ``t_split / dt`` to an integer and
REFUSES a ``t_split`` that is not within ``1e-6`` of a tick, so the emitted
geometry cannot depend on a rounding mode.

⛔ **CONSUMER STATUS — a named hand-off, not an omission.**
``refc.py::AnchoredDiffusionDecoder.roll_bank`` rolls ``anchor_controls`` as a
constant and ``anchor_control_seq`` expands to ``[B, N, S, 2]``; a ``[N, 3]`` bank
is **refused** by those shape checks (loudly, which is correct). :func:`roll_bank`
below is the drop-in reference the decoder needs: on a 2-column bank it is
bit-identical to the incumbent, and on a 3-column bank it honours the split.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor

from tanitad.models.kinematic import rollout_unicycle

#: what column 2 of a 3-column ``controls`` MEANS, written into the artifact.
TWO_SEGMENT = "two_segment_alat_flip"
#: the incumbent schedule: one control pair held for the whole horizon.
CONSTANT = "constant"
#: the legal values of an artifact's ``control_schedule`` field.
CONTROL_SCHEDULES = (CONSTANT, TWO_SEGMENT)

#: the third column's name and unit, written into ``controls_columns``.
T_SPLIT_COLUMN = "t_split_s"

#: ⭐ THE DEFAULT FAMILY — 6 candidates, and the count is MEASURED, not chosen.
#: `.../Research/2026-09-06-selection-quality/raw/out_p1h_two_segment.json`
#: (commit 56dc078; 5,000 windows, all 1,297 strict lane-change windows kept):
#:
#:   8 a_lat x 3 splits  ->  24 new, LC supply gain 0.1654 m
#:   4 a_lat x 3 splits  ->  12 new, LC supply gain 0.1651 m
#:   2 a_lat x 3 splits  ->   6 new, LC supply gain 0.1645 m   <- THIS
#:   8 a_lat x 1 split   ->   8 new, LC supply gain 0.0304 m
#:
#: ⇒ going from 6 to 24 buys 0.0009 m for +18 candidates, while dropping from 3
#: split points to 1 loses 81.5 % of the gain EVEN WITH ALL EIGHT MAGNITUDES.
#: **The split point is the lever; the magnitude is not.** Bank cost +5.13 %.
DEFAULT_A_LAT_MS2: tuple[float, ...] = (-0.75, 0.75)
DEFAULT_T_SPLIT_S: tuple[float, ...] = (2.0, 3.0, 4.0)
DEFAULT_A_LON_MS2: tuple[float, ...] = (0.0,)

#: ⚠️ The ``a_lon``-varying rows of that JSON are NOT supersets of the
#: ``a_lon = 0`` rows — the probe's ``a_lon_grid[::4]`` skips 0.0 — and their
#: smaller gains are that artefact, not a finding. The default holds a_lon = 0.


class SplitOffTick(ValueError):
    """A ``t_split_s`` that does not land on an integration tick."""


# --------------------------------------------------------------------- family
def two_segment_controls(a_lat: Sequence[float] = DEFAULT_A_LAT_MS2,
                         t_split_s: Sequence[float] = DEFAULT_T_SPLIT_S,
                         a_lon: Sequence[float] = DEFAULT_A_LON_MS2,
                         *, dtype: torch.dtype = torch.float32) -> Tensor:
    """``[len(a_lat) * len(t_split_s) * len(a_lon), 3]`` new candidates.

    Ordering is ``(t_split outer, a_lat, a_lon inner)`` and is FIXED: the index of
    a candidate is a class label the classifier learns, so a reordering between
    builds would silently permute a trained head's logits.
    """
    rows = []
    for t in t_split_s:
        for al in a_lat:
            for lo in a_lon:
                rows.append((float(lo), float(al), float(t)))
    if not rows:
        raise ValueError("two_segment_controls produced an EMPTY family; "
                         "every one of a_lat / t_split_s / a_lon must be "
                         "non-empty")
    return torch.tensor(rows, dtype=dtype)


def as_three_column(controls: Tensor, horizon_s: float,
                    *, dtype: torch.dtype | None = None) -> Tensor:
    """``[N, 2]`` constant controls -> ``[N, 3]`` with ``t_split_s = horizon_s``.

    ⭐ ``t_split_s = horizon_s`` means "the sign never flips", which is exactly
    the incumbent. This is the widening that lets a mixed bank hold both families
    in one tensor with no sentinel and no branch.
    """
    c = controls if dtype is None else controls.to(dtype)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"expected [N, 2] controls, got {tuple(c.shape)}")
    t = c.new_full((c.shape[0], 1), float(horizon_s))
    return torch.cat([c, t], dim=1)


def extend_controls(base: Tensor, horizon_s: float, *,
                    a_lat: Sequence[float] = DEFAULT_A_LAT_MS2,
                    t_split_s: Sequence[float] = DEFAULT_T_SPLIT_S,
                    a_lon: Sequence[float] = DEFAULT_A_LON_MS2) -> Tensor:
    """``[N, 2 or 3]`` incumbent -> ``[N + M, 3]`` with the new family APPENDED.

    ⛔ **APPENDED, never interleaved.** The first ``N`` rows keep their indices, so
    a checkpoint trained on the incumbent bank still names the same candidate by
    the same integer; only ``anchor_logits``' width changes.
    """
    b = base if base.shape[1] == 3 else as_three_column(base, horizon_s)
    new = two_segment_controls(a_lat, t_split_s, a_lon).to(b.dtype)
    return torch.cat([b, new], dim=0)


# ------------------------------------------------------------------ integrate
def split_ticks(t_split_s: Tensor, dt: float, steps: int,
                *, tol: float = 1e-6) -> Tensor:
    """``[N]`` integer tick index at which the lateral sign flips.

    ⛔ REFUSES a split that is not within ``tol`` of a tick. ``k * dt`` is not
    exact in binary for ``dt = 0.1``, so a float comparison at the boundary is a
    coin-flip; snapping to an integer makes the geometry independent of it.
    A split at or beyond ``steps`` means "never flips" (the incumbent).
    """
    q = t_split_s.to(torch.float64) / float(dt)
    r = torch.round(q)
    bad = (q - r).abs() > tol
    if bool(bad.any()):
        off = t_split_s[bad].tolist()
        raise SplitOffTick(
            f"t_split_s {off} do not land on a dt={dt} tick (within {tol}); "
            f"the emitted geometry would depend on a floating-point rounding "
            f"mode at the boundary. Snap them to a multiple of dt.")
    return r.clamp(0, float(steps)).to(torch.long)


def roll_bank(controls: Tensor, v: Tensor, *, control_units: str,
              steps: int, slots: Sequence[int] | Tensor, dt: float = 0.1,
              alat_v_floor: float = 4.0, kappa_cap: float = 0.12,
              dtype: torch.dtype = torch.float32) -> Tensor:
    """``[B, N, S, 2]`` — the emitted fan, for a 2- OR 3-column bank.

    ⛔ **THE 2-COLUMN PATH IS THE INCUMBENT'S ARITHMETIC, LINE FOR LINE**
    (``refc.py::AnchoredDiffusionDecoder.roll_bank``): float32 throughout,
    ``kappa = clamp(a_lat / clamp_min(v, alat_v_floor)^2, +-kappa_cap)`` under
    ``"alat"``, the control expanded over ``steps`` and integrated by
    ``rollout_unicycle`` from ``state0 = (0, 0, 0, v)``.

    The 3-column path differs in exactly one place: the curvature carries a
    per-tick sign, ``+1`` before ``t_split`` and ``-1`` after. A row whose
    ``t_split_s >= steps * dt`` gets an all-``+1`` sign and is therefore
    BIT-IDENTICAL to the 2-column path.

    ``v`` is ``[B]`` in m/s. ``slots`` are 0-based TICK indices into the rolled
    path (i.e. ``horizon_ticks - 1``), exactly as the decoder's ``anchor_slots``.
    """
    if control_units not in ("alat", "kappa"):
        raise ValueError(f"control_units {control_units!r} must be 'alat' or "
                         f"'kappa'")
    ctrl = controls.to(torch.float32)
    if ctrl.ndim != 2 or ctrl.shape[1] not in (2, 3):
        raise ValueError(f"controls must be [N, 2] or [N, 3]; got "
                         f"{tuple(ctrl.shape)}")
    v = v.reshape(-1).to(torch.float32)
    b, n, h = v.shape[0], ctrl.shape[0], int(steps)
    dev = ctrl.device
    v = v.to(dev)

    if control_units == "alat":
        vv = v.clamp_min(float(alat_v_floor)) ** 2                      # [B]
        kap = (ctrl[None, :, 1] / vv[:, None]).clamp(
            -float(kappa_cap), float(kappa_cap))                        # [B, N]
        a_lon = ctrl[None, :, 0].expand(b, n)                           # [B, N]
    else:
        kap = ctrl[None, :, 1].expand(b, n)
        a_lon = ctrl[None, :, 0].expand(b, n)

    if ctrl.shape[1] == 2:
        seq = torch.stack([a_lon, kap], dim=-1)                         # [B,N,2]
        seq = seq[:, :, None, :].expand(b, n, h, 2).reshape(-1, h, 2)
    else:
        st = split_ticks(ctrl[:, 2], dt, h).to(dev)                     # [N]
        k = torch.arange(h, device=dev)                                 # [h]
        sgn = torch.where(k[None, :] < st[:, None],
                          kap.new_ones(()), -kap.new_ones(()))          # [N, h]
        kap_t = kap[:, :, None] * sgn[None, :, :]                       # [B,N,h]
        lon_t = a_lon[:, :, None].expand(b, n, h)
        seq = torch.stack([lon_t, kap_t], dim=-1).reshape(-1, h, 2)

    state0 = torch.zeros(b * n, 4, device=dev, dtype=torch.float32)
    state0[:, 3] = v[:, None].expand(b, n).reshape(-1)
    path = rollout_unicycle(state0, seq, dt=float(dt))[..., :2]
    idx = (slots if isinstance(slots, Tensor)
           else torch.as_tensor(list(slots), dtype=torch.long))
    return path[:, idx.to(dev)].reshape(b, n, idx.numel(), 2).to(dtype)


# ----------------------------------------------------------------- kinematics
def kamm_report(controls: Tensor, speeds: Iterable[float], *,
                control_units: str = "alat", mu: float = 0.7, g: float = 9.81,
                alat_v_floor: float = 4.0, kappa_cap: float = 0.12) -> dict:
    """Friction-circle admissibility of every candidate over a speed sweep.

    ⛔ **The REALISED lateral acceleration is reported, not the declared one.**
    Under ``"alat"`` the curvature is clamped to ``kappa_cap``, so above
    ``v = sqrt(a_lat / kappa_cap)`` the candidate no longer delivers the
    ``a_lat`` its column claims. Scoring the declared value would report a
    vehicle that is not the one the integrator drives — the units trap in a
    kinematics costume.

    Returns per-candidate peaks and the corpus-level violation count against
    ``mu * g``. ⚠️ A zero here is only meaningful beside the MANOEUVRE RATE:
    MEASURED precedent, one zero-violation result was bought by a ``turn_left``
    recall of exactly 0.0000.
    """
    c = controls.to(torch.float32)
    vs = torch.as_tensor(list(speeds), dtype=torch.float32)             # [V]
    a_lon = c[:, 0]                                                     # [N]
    if control_units == "alat":
        vv = vs.clamp_min(float(alat_v_floor)) ** 2                     # [V]
        kap = (c[None, :, 1] / vv[:, None]).clamp(
            -float(kappa_cap), float(kappa_cap))                        # [V, N]
    else:
        kap = c[None, :, 1].expand(vs.shape[0], c.shape[0])
    a_lat_real = kap * (vs[:, None] ** 2)                               # [V, N]
    tot = (a_lon[None, :] ** 2 + a_lat_real ** 2).sqrt()                # [V, N]
    lim = float(mu) * float(g)
    clamped = (kap.abs() >= float(kappa_cap) - 1e-9)
    return {
        "mu": float(mu), "limit_ms2": lim,
        "speeds_ms": [float(x) for x in vs],
        "n_candidates": int(c.shape[0]),
        "peak_total_ms2": float(tot.max()),
        "peak_total_g": float(tot.max() / g),
        "peak_abs_a_lat_realised_ms2": float(a_lat_real.abs().max()),
        "peak_abs_kappa_inv_m": float(kap.abs().max()),
        "kappa_cap_reached": bool(clamped.any()),
        "n_candidate_speed_pairs": int(tot.numel()),
        "n_violations": int((tot > lim).sum()),
        "violating_candidates": sorted(
            int(i) for i in torch.nonzero((tot > lim).any(0)).reshape(-1)),
    }


def describe_family(controls: Tensor, n_base: int) -> str:
    """One line for a launch log: how many candidates, of which kind."""
    n = int(controls.shape[0])
    m = n - int(n_base)
    if controls.shape[1] == 3:
        splits = sorted({round(float(x), 6) for x in controls[n_base:, 2]})
        mags = sorted({round(float(x), 6) for x in controls[n_base:, 1]})
    else:
        splits, mags = [], []
    return (f"anchor bank {n} = {n_base} constant + {m} two-segment "
            f"({'+' if n_base else ''}{100.0 * m / max(n_base, 1):.2f} %), "
            f"splits {splits} s, a_lat {mags} m/s^2, "
            f"schedule={TWO_SEGMENT if m else CONSTANT}")


def net_yaw_zero_split_s(horizon_s: float) -> float:
    """The split that makes NET yaw exactly zero at constant speed.

    Under ``a_lon = 0`` the yaw integral is ``kappa * v * t``, so the ``+``
    segment cancels the ``-`` segment iff ``t_split = horizon_s / 2``. ⚠️ MEASURED
    that this is NOT where most of the gain lives (a single split at 3.0 s
    recovers 0.0304 m of the 0.1645 m that three splits recover): a real lane
    change is executed on a CURVING road, so the net-zero-heading S-curve is the
    minority case. Provided so the asymmetry is explicit, not so it is preferred.
    """
    return float(horizon_s) / 2.0


__all__ = [
    "TWO_SEGMENT", "CONSTANT", "CONTROL_SCHEDULES", "T_SPLIT_COLUMN",
    "DEFAULT_A_LAT_MS2", "DEFAULT_T_SPLIT_S", "DEFAULT_A_LON_MS2",
    "SplitOffTick", "two_segment_controls", "as_three_column",
    "extend_controls", "split_ticks", "roll_bank", "kamm_report",
    "describe_family", "net_yaw_zero_split_s",
]
