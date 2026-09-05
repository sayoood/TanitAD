"""A FEASIBILITY-AWARE DECODE: the friction-feasible set, made unreachable to violate.

⛔ THE ONE THING THIS MODULE EXISTS TO MAKE IMPOSSIBLE
------------------------------------------------------
refcv3's decode emits a fan whose mean friction load is **4.11 g** while the frozen
anchor vocabulary it started from is drivable at **0.48 g** — an **8.56x** blow-up
manufactured entirely *downstream of the vocabulary*
(``…/2026-09-05-veto-only-fan-safety/raw/bank_vs_fan_feasibility.json``, 240 windows
x 128 candidates). Two mitigations have already been priced and refused:

* **a feasibility REWARD**: ``feasibility x4`` moves rho(reward, envelope) by **0.001**,
  a **270x** worse lever than the term that actually rewards violation;
* **a post-train VETO**: it moved ``fan_peak_g_mean`` by **-0.098 g** against a
  **+3.63 g** gap = **~2.7 %**, and regressed T1 ``ade_m`` **+0.0362 m, separated**.

Both are *soft*: they penalise an infeasible path that the decode can still emit. This
module removes the ability. ⭐ **After ``project_feasible``, an envelope-violating or
Kamm-violating path is not merely penalised, it is UNREPRESENTABLE — a structural zero,
not an estimate**, and :func:`assert_feasible` asserts it rather than reporting it.

WHY IT IS EXACT, AND WHY THAT IS THE WHOLE DESIGN
--------------------------------------------------
``fan_safety.score_paths`` derives its verdict from ``rewards.kinematics`` on the
**5-point, 0.5 s** prefix. This module inverts *that exact finite-difference map*,
clamps the recovered controls, and re-integrates with the exact forward map::

    speed[k]   = |p[k+1] - p[k]| / dt          heading[k] = atan2(p[k+1] - p[k])
    accel[k]   = (speed[k+1] - speed[k]) / dt  yaw[k]     = wrap(dheading) / dt
    v_mid[k]   = max(speed[k], MIN_SPEED)      kappa[k]   = yaw[k] / v_mid[k]
    lat_acc[k] = v_mid[k] * yaw[k]

⇒ the re-derived kinematics of the output ARE the clamped controls, so
``envelope`` and ``kamm_over`` read **0.0000 exactly**. A projection built on any other
finite-difference convention would be *approximately* feasible and would report a small
non-zero rate that looks like noise and is actually a units/convention error — the
``df`` / Thor ``free`` / cylindrical-FOV family, in a geometry costume.

⛔ THE CLAMP IS SEQUENTIAL, NOT VECTORISED OVER k, AND THAT IS LOAD-BEARING.
``v_mid[k]`` depends on ``speed[k]``, which depends on ``accel[0..k-1]``. Clamping every
step against the *original* speeds would bound the wrong quantity and leave residual
violations. The loop is over the (3) control steps of a 2 s prefix, not over candidates,
so it costs nothing.

WHAT IT DOES NOT DO
-------------------
* It does **not** shrink the displacement. MEASURED (the sibling's λ-frontier,
  ``…/2026-09-05-kinematic-gate/raw/displacement_frontier.json``): an isotropic shrink
  ``bank + λ(fan - bank)`` buys feasibility at a near-LINEAR ADE cost. A projection
  removes only the *infeasible component* of the control profile and keeps the rest, so
  the two are different families and their frontiers must be compared at matched
  ``fan_peak_g`` — never by quoting one's verdict at the other.
* It does **not** touch ``off_reach`` by design (the 2 s mean-speed band), which is why
  every arm using it MUST report ``off_reach`` beside ``envelope``: trading one failure
  mode for the other and calling it a win is the failure this module's own scoping
  document named in advance.
* It is **not** a closed-loop claim. T0/T1 only (PI ruling 2026-09-02).

Evidence class of anything computed here: MEASURED (ours; rule-based, no learned parts).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["project_feasible", "recover_controls", "assert_feasible",
           "A_MAX_MPS2", "KAPPA_MAX_1PM", "MU_KAMM", "G_MPS2", "MIN_SPEED_MPS", "DT_S"]

#: The scorer's own constants. ⛔ ONE spelling: they are imported by the tools rather
#: than re-typed, because a projection clamping to 4.0 while the scorer flags at 3.9
#: would report a "small residual violation rate" that is a constant mismatch.
A_MAX_MPS2 = 4.0             # rewards.A_MAX_MPS2   -> fan_safety `envelope`
KAPPA_MAX_1PM = 0.2          # rewards.KAPPA_MAX_1PM-> fan_safety `envelope`
MU_KAMM = 0.7                # fan_safety.MU_KAMM   -> fan_safety `kamm_over`
G_MPS2 = 9.80665             # instruments.flyability.G
MIN_SPEED_MPS = 0.5          # rewards.kinematics' curvature denominator floor
DT_S = 0.5                   # fan_safety.DT_S -- the 5-point prefix grid

#: ⛔ THE SAFETY MARGIN, AND WHY IT IS NOT COSMETIC. `fan_safety.score_paths` flags with a
#: STRICT `>`, and a projection that clamps EXACTLY to `a_max` saturates: a large share of
#: steps land precisely on the boundary. Re-deriving that boundary through an fp32
#: round-trip lands above it about half the time, so a projection that is exactly feasible
#: in float64 reports an `envelope` rate near 0.5 in float32.
#: MEASURED 2026-09-05 on a 64-candidate fixture: **0.609375** -- which would have been
#: read as "the projection leaks", i.e. a numerical artifact wearing a mechanism's costume
#: (the `df` / cgroup / `step_s` family). Clamping to `(1 - MARGIN)` of each limit puts the
#: output strictly inside by 4e-4 m/s^2 -- physically nothing, ~100x the fp32 noise.
MARGIN = 1e-2

#: ⛔ THE STOPPED-STEP EPSILON, AND IT IS A CORRECTNESS FIX, NOT A TOLERANCE.
#: A projected path that DECELERATES TO A STOP inside the window emits a zero-length step,
#: and ``atan2(0, 0) == 0`` -- so every recovery reads that step's heading as **0 rad**
#: rather than as undefined, and the step BEFORE it then appears to have turned through the
#: whole previous heading. MEASURED 2026-09-05 on the 400 w x 128 fan: 2 candidates (both in
#: v0 = 0.000 windows, speeds ``[2.00, 4.00, 2.00, 0.00]``) read ``|kappa| = 0.3396`` against
#: a 0.2 cap -- a "hard turn" performed by a stationary vehicle.
#: ⚠️ It is NOT a scorer artifact: ``models.kinematic.unicycle_controls_from_path``, the
#: programme's OTHER and explicitly guarded recovery, reads the same 0.3396. Both substitute
#: a DIRECTION for a non-moving step instead of propagating the previous one. The
#: discriminating probe (a second implementation, not the same one re-run) is what settled
#: it, and it settled it AGAINST the comfortable answer.
#: ⭐ The fix is exact because ``atan2`` is SCALE-INVARIANT: emitting the stopped step with
#: a length of ``STOP_EPS_M`` along the HELD heading makes the recovered heading equal the
#: previous one exactly, so ``dtheta == 0`` and the curvature reads 0 -- while the geometric
#: error against a true stop is one micron. It is applied ONLY after the path has actually
#: moved; a path that never moves (the ``ha0`` floor at v0 = 0) keeps its exact zeros,
#: because there its headings are consistently 0 and no spurious turn exists to remove.
#:
#: ⛔⛔ AND THE SIZE OF THE EPSILON IS SET BY THE CONSUMER'S DTYPE, NOT BY float64.
#: A 1e-6 m held step is EXACT in float64 and MEANINGLESS in float32: at a path offset of
#: ~4 m, float32's resolution is ~4.8e-7 m, so a 1e-6 m displacement carries ~50 % relative
#: error and its `atan2` direction is noise. MEASURED: the 1e-6 version still read
#: `envelope = 1.0` on the regression fixture when scored in float32 -- while reading a
#: clean 0 in float64. ⇒ **The contract is what the fp32 CONSUMER recovers, never what our
#: float64 arithmetic believes** -- the same lesson as MARGIN above, one dtype further down.
#: ⚠️ AND THE THRESHOLD IS RELATIVE, NOT ABSOLUTE -- the first attempt used a 1 mm
#: absolute floor and it INFLATED SLOW-BUT-MOVING PATHS: a `ha0` hold at v0 = 0.0005 m/s
#: takes 0.25 mm steps, which are perfectly well-conditioned (float32's resolution at that
#: magnitude is ~1e-13 m) and were nonetheless rewritten to 1 mm. MEASURED: C-ROUNDTRIP,
#: the control that says an already-feasible path is a fixed point, went from 0.00e+00 to
#: 2.63e-03 m. ⇒ "degenerate" is a statement about a step RELATIVE TO ITS OWN PATH, and a
#: fixed constant that is right at one scale and wrong at another is the `df` / `step_s`
#: family again -- this time introduced by the fix for a different instance of it.
#: Sizing: fp32 carries ~1.2e-7 relative, so a step of `|p| * 1e-3` recovers its heading to
#: about `1.2e-7 / 1e-3 = 1.2e-4 rad`. A stopped step reads `v_mid = MIN_SPEED = 0.5`, so
#: that is `kappa <= 1.2e-4 / (dt * v_mid) = 4.8e-4` -- and MARGIN (1 %) covers it with two
#: orders of room. ⭐ A stopped step ALSO FREEZES THE HEADING (`lat := 0`), because
#: `yaw_rate = v * kappa` and a stationary vehicle does not turn: without that the step
#: before a stop can sit exactly ON the curvature cap and the heading error tips it over.
STOP_EPS_M = 1e-9            # absolute floor, only for a path sitting at the origin
STOP_EPS_REL = 1e-3          # the real threshold: RELATIVE to the path's own magnitude


def recover_controls(paths: Tensor, dt: float = DT_S,
                     min_speed: float = MIN_SPEED_MPS
                     ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``paths [..., S, 2]`` (index 0 = the ego origin) -> ``(speed, heading, accel,
    lat_acc)`` in ``rewards.kinematics``' own convention.

    ``speed``/``heading`` are ``[..., S-1]``; ``accel``/``lat_acc`` are ``[..., S-2]``.
    ⚠️ This is the INVERSE the scorer implies, not the unicycle integrator's inverse
    (``models.kinematic.unicycle_controls_from_path``): that one is shifted by one step
    because ``rollout_unicycle`` updates ``v`` last. Two conventions is how a projection
    and its scorer silently disagree.
    """
    if paths.shape[-1] != 2:
        raise ValueError(f"expected [..., S, 2] waypoints, got {tuple(paths.shape)}")
    if paths.shape[-2] < 4:
        raise ValueError(f"need >=4 points (origin + 3) to derive a control, got "
                         f"{paths.shape[-2]}")
    d = paths[..., 1:, :] - paths[..., :-1, :]
    speed = d.norm(dim=-1) / dt
    heading = torch.atan2(d[..., 1], d[..., 0])
    dtheta = heading[..., 1:] - heading[..., :-1]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    yaw_rate = dtheta / dt
    v_mid = speed[..., :-1].clamp_min(min_speed)
    accel = (speed[..., 1:] - speed[..., :-1]) / dt
    lat_acc = v_mid * yaw_rate
    return speed, heading, accel, lat_acc


def project_feasible(paths: Tensor, v0: Tensor | None = None, *,
                     dt: float = DT_S, a_max: float = A_MAX_MPS2,
                     kappa_max: float = KAPPA_MAX_1PM,
                     mu: float | None = MU_KAMM,
                     clamp_entry: bool = False,
                     enabled: bool = True,
                     margin: float = MARGIN,
                     stop_eps_m: float = STOP_EPS_M,
                     stop_eps_rel: float = STOP_EPS_REL,
                     min_speed: float = MIN_SPEED_MPS) -> Tensor:
    """``paths [..., S, 2]`` -> the nearest path whose implied controls lie in the
    feasible set. Index 0 must be the ego origin and is returned unchanged.

    ``mu=None`` applies the BOX only (``|a| <= a_max``, ``|kappa| <= kappa_max``), which
    zeroes ``envelope`` but NOT ``kamm_over``; ``mu`` given additionally projects
    ``(accel, lat_acc)`` radially onto the friction disc of radius ``mu * g``, which
    zeroes both. ⭐ The radial projection is the minimum-distance point of the disc, and
    it preserves the box because it only ever scales the pair DOWN.

    ``clamp_entry`` additionally binds the first step's speed to
    ``v0 +- a_max * dt``. ⚠️ ``envelope``/``kamm_over`` CANNOT see the entry speed (they
    differentiate the path, and the first speed is a level, not a difference) — but a
    real vehicle can, and a path that starts at a speed the ego is not travelling at is a
    launch transient no control produces. It is a separate flag because it changes a
    different metric (``off_reach``) and must be reported separately, never merged.

    ⛔ ``enabled=False`` is the DISABLED LEVER: it SHORT-CIRCUITS and returns the input
    object, so it is bit-identical by construction. ⚠️ **That is deliberate, and it is
    also exactly why it is a WEAK control on its own.** It proves the surrounding
    pipeline does not perturb a path; it proves nothing about the projection arithmetic,
    because the arithmetic never ran. The arithmetic's control is the ROUND-TRIP (an
    already-feasible path must come back to <1e-5 m), and the two are reported
    separately, never as one green tick. *(RETRACTION #30: a control that checks only
    that an operation was self-consistent cannot see that it ran on the wrong object —
    or, here, that it did not run at all.)*

    ⚠️ A numerically-round-tripped identity is NOT bit-identical and must not be claimed
    as one: ``|d| * cos(atan2(dy, dx))`` does not reproduce ``dx`` in floating point. The
    residual is ~1e-9 m in float64 and is reported as a residual.
    """
    if not enabled:
        return paths
    if paths.shape[-1] != 2:
        raise ValueError(f"expected [..., S, 2] waypoints, got {tuple(paths.shape)}")
    S = paths.shape[-2]
    if S < 4:
        raise ValueError(f"need >=4 points (origin + 3), got {S}")
    orig_dtype = paths.dtype
    p = paths.to(torch.float64)
    if bool((p[..., 0, :].abs() > 1e-6).any()):
        raise ValueError("paths[..., 0, :] must be the ego origin (0, 0); pass the "
                         "origin-prepended prefix (`fan_safety.with_origin`)")

    speed, heading, accel, lat_acc = recover_controls(p, dt=dt, min_speed=min_speed)
    # the margin: strictly INSIDE every limit, so an fp32 round-trip cannot cross it.
    m = 1.0 - float(margin)
    a_max = float(a_max) * m
    kappa_max = float(kappa_max) * m
    R = None if mu is None else float(mu) * G_MPS2 * m

    s = speed[..., 0]
    if clamp_entry:
        if v0 is None:
            raise ValueError("clamp_entry=True needs v0")
        v0b = torch.as_tensor(v0, dtype=p.dtype, device=p.device)
        while v0b.dim() < s.dim():
            v0b = v0b.unsqueeze(-1)
        s = torch.minimum(torch.maximum(s, (v0b - a_max * dt).clamp_min(0.0)),
                          v0b + a_max * dt)
    h = heading[..., 0]

    out = [torch.zeros_like(p[..., 0, :])]
    cur = out[0]
    moved = torch.zeros_like(s, dtype=torch.bool)
    for k in range(S - 1):
        # ⛔ the stopped-step epsilon: hold the heading through a stop the path REACHED,
        # never fabricate motion in a path that never moved. See STOP_EPS_M.
        L = s * dt
        eps = torch.clamp(cur.abs().amax(dim=-1) * float(stop_eps_rel),
                          min=float(stop_eps_m))
        halted = moved & (L < eps)
        L = torch.where(halted, eps, L)
        moved = moved | (L > 0)
        step = torch.stack([L * torch.cos(h), L * torch.sin(h)], dim=-1)
        cur = cur + step
        out.append(cur)
        if k >= S - 2:
            break
        v_mid = s.clamp_min(min_speed)
        a = accel[..., k].clamp(-a_max, a_max)
        lat_cap = v_mid.pow(2) * kappa_max
        lat = torch.minimum(torch.maximum(lat_acc[..., k], -lat_cap), lat_cap)
        if R is not None:
            n = torch.sqrt(a.pow(2) + lat.pow(2)).clamp_min(1e-12)
            scale = torch.clamp(R / n, max=1.0)
            a, lat = a * scale, lat * scale
        s = (s + a * dt).clamp_min(0.0)
        # ⛔ a stationary vehicle does not turn (yaw_rate = v * kappa). Freezing the
        # heading across a halted step is the physics AND the thing that keeps the
        # recovered curvature inside the cap when the step before sits on it.
        h = h + torch.where(halted, torch.zeros_like(lat), lat / v_mid) * dt
    return torch.stack(out, dim=-2).to(orig_dtype)


def assert_feasible(paths: Tensor, *, dt: float = DT_S, a_max: float = A_MAX_MPS2,
                    kappa_max: float = KAPPA_MAX_1PM, mu: float | None = MU_KAMM,
                    atol: float = 1e-6) -> dict:
    """Re-derive the scorer's own flags on ``paths`` and return the observed rates.

    ⭐ **It re-derives rather than trusting the projection's own bookkeeping.** A control
    that checks the arithmetic it just performed cannot see that it performed it on the
    wrong object — RETRACTION #30 — so this reads the OUTPUT tensor back through the
    same inverse the scorer uses and reports what IT says.
    """
    speed, heading, accel, lat_acc = recover_controls(paths.to(torch.float64), dt=dt)
    v_mid = speed[..., :-1].clamp_min(MIN_SPEED_MPS)
    kappa = lat_acc / v_mid.pow(2)
    peak_g = torch.sqrt(accel.pow(2) + lat_acc.pow(2)).amax(dim=-1) / G_MPS2
    env = ((accel.abs().amax(dim=-1) > a_max + atol)
           | (kappa.abs().amax(dim=-1) > kappa_max + atol))
    out = {"envelope_rate": float(env.double().mean()),
           "peak_g_mean": float(peak_g.mean()),
           "peak_g_max": float(peak_g.max()),
           "max_abs_accel": float(accel.abs().max()),
           "max_abs_kappa": float(kappa.abs().max())}
    if mu is not None:
        out["kamm_over_rate"] = float((peak_g > float(mu) + atol).double().mean())
    return out


def max_heading_step(paths: Tensor, dt: float = DT_S,
                     kappa_max: float = KAPPA_MAX_1PM) -> float:
    """The largest per-step heading change implied by ``paths``.

    ⚠️ THE ONE ALIASING EDGE THIS MODULE HAS. ``recover_controls`` wraps ``dtheta`` to
    ``+-pi``, so a projected step whose heading turns by more than ``pi`` would be read
    back as its wrapped complement and the feasibility identity would silently fail. The
    bound is ``v_mid * kappa_max * dt``, i.e. ``pi`` at ``v_mid = pi/(0.2*0.5) = 31.4
    m/s`` (113 km/h). Our corpus tops out well below it; this function is what a caller
    at higher speed must check rather than assume.
    """
    _, heading, _, _ = recover_controls(paths.to(torch.float64), dt=dt)
    d = heading[..., 1:] - heading[..., :-1]
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return float(d.abs().max())


ALIAS_SPEED_LIMIT_MPS = math.pi / (KAPPA_MAX_1PM * DT_S)
