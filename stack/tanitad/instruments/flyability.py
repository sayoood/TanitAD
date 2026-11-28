"""Exact friction-circle (Kamm) load of an anchor vocabulary.

⛔ WHY THIS MODULE EXISTS, AND WHY THE OBVIOUS IMPLEMENTATION IS WRONG.

The 2026-09-04 refcv4b vocabulary gate measured Kamm load by taking
``np.gradient`` TWICE over the anchor's own eight SLOT waypoints. The slots are
spaced 0.5-1.0 s apart while the integrator steps at ``dt = 0.1`` s, so the
second difference smooths the very peak it is supposed to find. MEASURED, same
vocabulary, same windows: at v0 = 10.09 m/s the slot-gradient reads
**12/117 anchors over mu=0.7, peak 1.23 g** where the exact per-step load is
**18/117, peak 2.26 g** -- an under-report of **1.21-1.85x** across the
speed range.

⚠️ It under-reports, which is the dangerous direction: a gate that under-reads
friction PASSES a vocabulary no car can drive. refcv3 shipped exactly such a
fan (14.15 m/s^2, 1.07 g lateral, unable to turn past 74.5 degrees) and trained
40,284 steps on it. Root-cause class: the ``df`` / ``free`` / ``step_s`` family
-- a probe that reports the wrong SCOPE, read as an answer.

⭐ THE FIX IS NOT A FINER FINITE DIFFERENCE. No difference is needed at all,
because the controls ARE the accelerations. Under the decoder's roll
(``refc.py::_anchor_bank``) ``kappa`` is computed once from v0 and held CONSTANT
over the rollout while ``v(t)`` moves under ``a_lon``, so the realised lateral
load ``a_lat(t) = v(t)^2 * kappa`` GROWS along the path. Reading it off the
endpoint geometry cannot see that.

Two details are load-bearing and are the reason this reads the rolled state
rather than a closed form ``v(t) = v0 + a*t``:

* ``rollout_unicycle`` CLAMPS speed at zero (``kinematic.py:265``). Past the
  stopping point the commanded deceleration is not realised, so the closed form
  over-reports longitudinal load on every braking anchor that reaches a stop.
* The yaw update uses the speed at the START of the step
  (``yaw += v * kappa * dt``, before ``v`` is updated). So the lateral load of
  step k is ``v_start(k)^2 * kappa``, not the post-update speed. Using the wrong
  one is a one-step shift -- the same off-by-one the integrator's own docstring
  warns about for the inverse map.

⇒ This module imports the SAME ``rollout_unicycle`` the decoder uses. It does
not re-implement the integrator, because a second implementation of a shared
geometry is how two "independent" checks come to agree on a wrong answer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tanitad.models.kinematic import rollout_unicycle

G = 9.81  #: standard gravity, m/s^2. Loads are reported in g via this constant.


def derive_kappa(controls: Tensor, v0: Tensor, *, units: str,
                 kappa_cap: float, alat_v_floor: float) -> Tensor:
    """``[N, 2]`` controls + ``[B]`` speeds -> ``[B, N]`` curvature.

    Mirrors ``refc.py::_anchor_bank`` exactly. ``units="alat"`` treats column 1
    as LATERAL ACCELERATION (m/s^2) and divides by a floored ``v^2`` before
    clamping; ``units="kappa"`` takes column 1 as curvature already.

    ⚠️ The units are NOT recoverable from the numbers. The shipped refcv4b
    ``anchors.pt`` carries no units field, and its column 1 range of [-3, 3]
    reads as a plausible lateral-acceleration grid AND as a (physically absurd)
    curvature grid. Read as curvature it yields 396 g at 36 m/s and 104/117
    anchors over mu=0.7; read correctly, 0.31 g and 0/117. Both tables look
    like answers. Pass the units explicitly; never infer them.
    """
    if units not in ("alat", "kappa"):
        raise ValueError(f"units must be 'alat' or 'kappa', got {units!r}")
    ctrl = controls.to(torch.float32)
    v = v0.reshape(-1).to(torch.float32)
    if units == "kappa":
        return ctrl[None, :, 1].expand(v.shape[0], ctrl.shape[0]).clone()
    vv = v.clamp_min(alat_v_floor) ** 2
    return (ctrl[None, :, 1] / vv[:, None]).clamp(-kappa_cap, kappa_cap)


def friction_load(controls: Tensor, v0: Tensor, *, units: str, steps: int,
                  dt: float, kappa_cap: float = 0.12,
                  alat_v_floor: float = 4.0) -> dict[str, Tensor]:
    """Per-step friction load of every anchor at every speed.

    Returns a dict of ``[B, N, steps]`` tensors ``a_lon`` / ``a_lat`` /
    ``a_tot`` in m/s^2 (REALISED, read off the rolled state) plus ``peak_g``
    ``[B, N]``, the per-anchor maximum of ``a_tot`` divided by ``G``.

    ``a_lon`` is the realised ``dv/dt`` across the step, so an anchor that
    brakes to a standstill reports the deceleration it actually achieved rather
    than the one it commanded.
    """
    ctrl = controls.to(torch.float32)
    n = ctrl.shape[0]
    v = v0.reshape(-1).to(torch.float32)
    b = v.shape[0]

    kap = derive_kappa(ctrl, v, units=units, kappa_cap=kappa_cap,
                       alat_v_floor=alat_v_floor)                  # [B, N]
    a_cmd = ctrl[None, :, 0].expand(b, n)                          # [B, N]

    seq = torch.stack([a_cmd, kap], dim=-1)[:, :, None, :]
    seq = seq.expand(b, n, steps, 2).reshape(-1, steps, 2)
    state0 = torch.zeros(b * n, 4, dtype=torch.float32)
    state0[:, 3] = v[:, None].expand(b, n).reshape(-1)

    states = rollout_unicycle(state0, seq, dt=dt)                  # [B*N, K, 4]
    v_after = states[..., 3].reshape(b, n, steps)
    # speed at the START of each step: v0 for step 0, then the previous state's
    # speed. This is the speed the yaw update actually used.
    v_start = torch.cat(
        [v[:, None, None].expand(b, n, 1), v_after[:, :, :-1]], dim=2)

    a_lat = v_start.pow(2) * kap[:, :, None]
    a_lon = (v_after - v_start) / dt
    a_tot = torch.sqrt(a_lon.pow(2) + a_lat.pow(2))
    return {"a_lon": a_lon, "a_lat": a_lat, "a_tot": a_tot,
            "peak_g": a_tot.amax(dim=2) / G, "kappa": kap,
            "v_start": v_start}


def kamm_report(controls: Tensor, v0: Tensor, *, units: str, steps: int,
                dt: float, kappa_cap: float = 0.12,
                alat_v_floor: float = 4.0,
                mu: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)) -> dict:
    """Gate-facing summary: how many anchors break each friction circle.

    ⚠️ The counts are per (speed, anchor) pair collapsed over anchors: an anchor
    that is legal at 10 m/s and illegal at 27 m/s is counted at 27. Report the
    speed alongside the count or the number is not interpretable -- the whole
    point of a v0-conditioned vocabulary is that its load is speed-dependent.
    """
    load = friction_load(controls, v0, units=units, steps=steps, dt=dt,
                         kappa_cap=kappa_cap, alat_v_floor=alat_v_floor)
    peak = load["peak_g"]
    out = {
        "n_anchors": int(controls.shape[0]),
        "speeds_ms": [float(x) for x in v0.reshape(-1)],
        "peak_g_overall": float(peak.max()),
        "max_a_lat_ms2": float(load["a_lat"].abs().max()),
        "max_a_lon_ms2": float(load["a_lon"].abs().max()),
        "per_speed": [],
    }
    for i, speed in enumerate(v0.reshape(-1).tolist()):
        row = {"v0_ms": float(speed), "peak_g": float(peak[i].max())}
        for m in mu:
            row[f"over_mu_{m:.1f}"] = int((peak[i] > m).sum())
        out["per_speed"].append(row)
    return out


def slot_gradient_load_DEPRECATED(paths: Tensor, slot_times: Tensor) -> Tensor:
    """⛔ THE DEFECTIVE INSTRUMENT, KEPT ONLY AS A DELIBERATE-REGRESSION ARM.

    ``paths`` ``[N, S, 2]`` at ``slot_times`` ``[S]`` -> ``[N]`` peak load in g,
    by differencing the slot waypoints twice. This is what the 2026-09-04 gate
    used and it UNDER-REPORTS by 1.21-1.85x. Never call it to judge a
    vocabulary; ``test_anchor_flyability.py`` calls it to prove the gate still
    catches what it is supposed to catch. A gate that has never been shown to
    FAIL a known-bad input has not been tested.
    """
    t = slot_times.to(torch.float64)
    p = paths.to(torch.float64)
    v = torch.gradient(p, spacing=(t,), dim=1)[0]
    a = torch.gradient(v, spacing=(t,), dim=1)[0]
    sp = v.norm(dim=-1).clamp_min(1e-6)
    lon = ((v * a).sum(-1) / sp).abs()
    lat = ((v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]) / sp).abs()
    return (torch.sqrt(lon ** 2 + lat ** 2) / G).amax(dim=1)
