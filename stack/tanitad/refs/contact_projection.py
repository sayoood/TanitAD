"""A CONTACT-AWARE DECODE: the collision-free set, made unreachable to violate.

⛔ THE ONE THING THIS MODULE EXISTS TO MAKE IMPOSSIBLE
------------------------------------------------------
refcv3's decode **emits colliding candidates**. MEASURED on the banked fan
(``…/2026-09-05-veto-only-fan-safety/raw/fan_bank_base_240w.npz``, md5
``ccecf0ade75631e315e6b9ae78d83478``, 240 windows x 128 candidates):
``f_contact`` fires on **764 / 30,720 = 2.4870 %** of candidates, in **19 / 240**
windows, and on **104 of 128 (81 %)** in the worst one — while ``sel_contact``
reads **0.0000**, because the *selector* filters them.
⭐ **The generator is the defect; the filter hid it** (RETRACTION #31).

Two mitigations have already been priced and refused:

* a **feasibility/collision REWARD**: ``rho(reward, contact) = -0.6326`` already —
  the reward does not prefer collision, and strengthening it moved nothing;
* a **post-train VETO**: ~2.7 % of the safety gap at T1 ``ade_m`` **+0.0362 m,
  separated worse** — and on the ``W_KAPPA`` ladder a penalty strong enough to
  suppress a bad turn suppresses *turning* (both turn recalls -> 0.0).

⭐ **A penalty cannot separate "do not turn badly" from "do not turn". A projection
can: it deletes the infeasible candidate and leaves the feasible one untouched.**
This module removes the ability to emit a colliding path, the same way
:mod:`tanitad.refs.feasible_decode` removed the ability to emit an over-friction one.

WHY THE RETRACTION IS LONGITUDINAL FIRST, AND WHY THAT IS THE WHOLE DESIGN
--------------------------------------------------------------------------
The contact predicate (``rewards._collision``, moving-lead branch) is
**time-aligned and swept in the RELATIVE frame**: contact iff the polyline
``rel[s] = lead[s] - ego[s]`` (``s = 1..4``) comes within ``r = ego_r + obs_r``
of the origin, segments included. ⇒ the ego controls its own side of ``rel``
through **where it is at each time**, so the natural control-space knob is the
**time law**: drive the same geometric line, slower.

``sigma`` scales the recovered speed profile and scales ``lat_acc`` by ``sigma^2``,
which holds ``kappa = lat_acc / v_mid^2`` FIXED — a pure time reparametrisation of
the candidate's own line. ⭐ **Three properties follow, and they are what make this
a projection rather than a shrink:**

1. ``sigma = 1`` is the identity (up to the fp round-trip residual);
2. the ORIENTATION and CURVATURE of the plan are untouched — the lateral decision
   the model made survives; only the progress along it is retracted;
3. ⭐ **it can never make a friction-feasible path infeasible.** ``|a| -> sigma|a|``
   (non-increasing), ``kappa -> kappa`` when ``v_mid > MIN_SPEED`` and strictly
   *smaller* when the scaled speed falls under the floor, and
   ``peak_g = |(a, lat)| / g`` is non-increasing in ``sigma``. This is asserted on
   the OUTPUT (:func:`assert_no_contact`), never inferred from the algebra.

⛔ **AND IT IS SEARCHED, NOT SOLVED.** ``d_min(sigma)`` is not guaranteed monotone
for an arbitrary agent track (a lead that is *reversing* toward the ego makes
slowing down worse), so this module **scans** ``sigma`` on a grid and takes the
LARGEST clear value. That is exact on the grid regardless of monotonicity, and it
costs nothing: the scan is over 3 control steps of a 2 s prefix, vectorised over
candidates. Assuming monotonicity and bisecting would be faster and would silently
return a colliding path in exactly the case that matters.

⛔ THE SECOND AXIS, AND WHY IT IS SECOND
----------------------------------------
Some windows have **no collision-free member in the longitudinal family** — if the
lead's own track enters the ego's disc while the ego sits at the origin
(``sigma = 0``), then *not moving* is already contact. That is **physics, not an
instrument defect**, and no amount of longitudinal retraction fixes it. For those
candidates only, a **lateral** axis is searched: a constant curvature bias
``dkappa`` added to the recovered profile (``lat -> (kappa + dkappa) * v_mid^2``),
box- and Kamm-clamped. It is searched SECOND, and only on the residual, because it
changes the plan's *decision* rather than its *timing* — minimal intervention is
the point.

⇒ the taxonomy the caller gets back is three-way and load-bearing:
``LONGITUDINAL`` (the plan, slower) · ``LATERAL`` (the plan, steered) ·
``UNAVOIDABLE`` (no member of either family clears — the residual that a
projection provably cannot constrain, and therefore the honest RL objective).

WHAT IT DOES NOT DO
-------------------
* It does **not** touch ``off_reach``'s definition, and it is EXPECTED to move its
  rate: retracting progress lowers the 2 s mean speed and can push a candidate
  under the S2 band's lower edge. ⛔ **Every arm using this module MUST report
  ``off_reach`` beside ``contact``** — trading one failure mode for the other and
  calling it a win is the failure :mod:`feasible_decode`'s scoping document named
  in advance.
* It does **not** re-rank. The model's ``sel_idx`` is unchanged; a projected fan is
  the same fan with colliding members retracted, so ``sel_*`` after projection is
  "what the deployed selector would have picked, after projection" — never a claim
  that the selector improved.
* It is **not** a closed-loop claim. T0 / T1 only (PI ruling 2026-09-02).
* ⚠️ It is only as good as the **agent track it is given**. A projection against a
  PREDICTED agent inherits the prediction's error, which is what ``margin_m`` is
  for — and ``margin_m`` is a parameter with a measured cost curve, never a
  constant to be guessed.

Evidence class of anything computed here: MEASURED (ours; rule-based, no learned
parts, 0 GPU).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .feasible_decode import (A_MAX_MPS2, DT_S, G_MPS2, KAPPA_MAX_1PM, MARGIN,
                              MIN_SPEED_MPS, MU_KAMM, recover_controls)

__all__ = ["project_contact_free", "min_rel_distance", "assert_no_contact",
           "integrate_controls", "EGO_RADIUS_M", "OBS_RADIUS_M", "CONTACT_R_M",
           "REASON_CLEAR", "REASON_LONGITUDINAL", "REASON_LATERAL",
           "REASON_UNAVOIDABLE", "REASON_NO_AGENT"]

#: ⛔ ONE spelling of the contact geometry, imported by the tools rather than
#: re-typed. ``rewards._collision`` defaults to ``ego_radius_m`` 1.0 +
#: ``obs_radius_m`` 1.0; a projection clearing 2.0 while the scorer flags at 2.1
#: would report a "small residual contact rate" that is a constant mismatch.
EGO_RADIUS_M = 1.0
OBS_RADIUS_M = 1.0
CONTACT_R_M = EGO_RADIUS_M + OBS_RADIUS_M

#: The three-way taxonomy, returned per candidate. ``NO_AGENT`` is distinct from
#: ``CLEAR``: a window with no agent track carries **no information** about contact
#: and must never be counted as a safety success (the ``audit`` doctrine — a
#: component that never fires is reported, not silently trusted).
REASON_NO_AGENT = 0
REASON_CLEAR = 1
REASON_LONGITUDINAL = 2
REASON_LATERAL = 3
REASON_UNAVOIDABLE = 4
REASON_NAMES = {REASON_NO_AGENT: "no_agent", REASON_CLEAR: "clear",
                REASON_LONGITUDINAL: "longitudinal", REASON_LATERAL: "lateral",
                REASON_UNAVOIDABLE: "unavoidable"}


# --------------------------------------------------------------------------- #
# the geometry: the scorer's own predicate, as a DISTANCE rather than a flag    #
# --------------------------------------------------------------------------- #
def min_rel_distance(paths: Tensor, lead: Tensor, *, skip_first: bool = True
                     ) -> Tensor:
    """``[..., S, 2]`` ego paths and ``[..., S, 2]`` lead track -> ``[...]`` the
    minimum distance from the ORIGIN to the swept relative polyline.

    ⭐ This is ``rewards._swept_hit(rel, origin, r)`` with the threshold removed:
    ``_collision`` fires exactly when this value is ``< r``. Returning the distance
    rather than the flag is what lets a projection *search*; the flag is recovered
    by comparing, and :func:`assert_no_contact` checks the recovered flag against
    ``rewards._collision`` itself rather than against this function.

    ``skip_first`` reproduces the scorer's ``lead[..., 1:, :] - traj[..., 1:, :]``
    (step 0 is the ego's own origin and the lead is ahead by construction).
    """
    if paths.shape[-1] != 2 or lead.shape[-1] != 2:
        raise ValueError(f"expected [..., S, 2], got {tuple(paths.shape)} / "
                         f"{tuple(lead.shape)}")
    rel = lead - paths
    if skip_first:
        rel = rel[..., 1:, :]
    d_pt = rel.norm(dim=-1).amin(dim=-1)
    if rel.shape[-2] < 2:
        return d_pt
    # min distance from the origin to each segment p0 -> p1, the same clamped
    # projection ``rewards.segment_point_distance`` uses (with q = 0).
    p0, p1 = rel[..., :-1, :], rel[..., 1:, :]
    d = p1 - p0
    denom = (d * d).sum(-1).clamp_min(1e-12)
    t = ((-p0 * d).sum(-1) / denom).clamp(0.0, 1.0).unsqueeze(-1)
    d_seg = (p0 + t * d).norm(dim=-1).amin(dim=-1)
    return torch.minimum(d_pt, d_seg)


# --------------------------------------------------------------------------- #
# the forward map — the SAME integrator ``feasible_decode`` re-integrates with   #
# --------------------------------------------------------------------------- #
def integrate_controls(s0: Tensor, h0: Tensor, accel: Tensor, lat: Tensor, *,
                       dt: float = DT_S, clamp: bool = False,
                       a_max: float = A_MAX_MPS2,
                       kappa_max: float = KAPPA_MAX_1PM,
                       mu: float | None = MU_KAMM, margin: float = MARGIN,
                       min_speed: float = MIN_SPEED_MPS) -> Tensor:
    """``(s0, h0)`` ``[...]`` and ``(accel, lat)`` ``[..., S-2]`` -> ``[..., S, 2]``.

    ⛔ Byte-for-byte the forward map inside ``feasible_decode.project_feasible``:
    ``L = s*dt``; ``step = L*(cos h, sin h)``; ``v_mid = max(s, MIN_SPEED)``;
    ``s += a*dt`` (floored at 0); ``h += (lat / v_mid) * dt``. A second convention
    is how a projection and its scorer silently disagree.

    ``clamp=False`` (the default here, and the opposite of ``project_feasible``'s)
    integrates the controls AS GIVEN. That is deliberate: the contact projection
    must be attributable to CONTACT alone, so the friction clamp is a separate,
    composable stage rather than a hidden passenger. ``clamp=True`` applies the
    box + Kamm projection in the loop, giving the composed decode.

    ⚠️ The stopped-step epsilon of ``feasible_decode`` is NOT reproduced here, and
    that is a scoped decision, not an oversight: this integrator is used to build a
    RETRACTED path whose speeds are ``sigma`` times the original's, so a
    ``sigma -> 0`` member legitimately halts, and the caller re-derives every flag
    from the OUTPUT with the scorer rather than trusting a heading convention. The
    curvature a halted step implies is reported by the caller's ``assert``, so a
    convention error surfaces as a failed control instead of a silent pass.
    """
    S = accel.shape[-1] + 2
    m = 1.0 - float(margin)
    a_cap = float(a_max) * m
    k_cap = float(kappa_max) * m
    R = None if mu is None else float(mu) * G_MPS2 * m
    s, h = s0, h0
    cur = torch.zeros(*s.shape, 2, dtype=s.dtype, device=s.device)
    out = [cur]
    for k in range(S - 1):
        L = s * dt
        cur = cur + torch.stack([L * torch.cos(h), L * torch.sin(h)], dim=-1)
        out.append(cur)
        if k >= S - 2:
            break
        v_mid = s.clamp_min(min_speed)
        a = accel[..., k]
        y = lat[..., k]
        if clamp:
            a = a.clamp(-a_cap, a_cap)
            cap = v_mid.pow(2) * k_cap
            y = torch.minimum(torch.maximum(y, -cap), cap)
            if R is not None:
                n = torch.sqrt(a.pow(2) + y.pow(2)).clamp_min(1e-12)
                sc = torch.clamp(R / n, max=1.0)
                a, y = a * sc, y * sc
        s = (s + a * dt).clamp_min(0.0)
        h = h + (y / v_mid) * dt
    return torch.stack(out, dim=-2)


# --------------------------------------------------------------------------- #
# the projection                                                                #
# --------------------------------------------------------------------------- #
def project_contact_free(paths: Tensor, lead: Tensor,
                         has_agent: Tensor | None = None, *,
                         dt: float = DT_S,
                         r_m: float = CONTACT_R_M,
                         margin_m: float = 0.0,
                         margin: float = MARGIN,
                         n_sigma: int = 257,
                         n_dkappa: int = 41,
                         skip_first: bool = True,
                         lateral: bool = True,
                         margin_fallback: bool = True,
                         n_margin_rungs: int = 5,
                         clamp_friction: bool = False,
                         enabled: bool = True,
                         min_speed: float = MIN_SPEED_MPS,
                         a_max: float = A_MAX_MPS2,
                         kappa_max: float = KAPPA_MAX_1PM,
                         mu: float | None = MU_KAMM,
                         ) -> tuple[Tensor, dict[str, Tensor]]:
    """``paths [B, N, S, 2]`` (index 0 = the ego origin) and ``lead [B, 1, S, 2]``
    (broadcastable) -> ``(projected [B, N, S, 2], info)``.

    ⛔ ``enabled=False`` SHORT-CIRCUITS and returns the input OBJECT, so the
    disabled lever is bit-identical by construction. ⚠️ **That is exactly why it is
    a WEAK control on its own** — the arithmetic never ran (RETRACTION #30). The
    arithmetic's control is ``info["roundtrip_max_m"]``: the search is executed for
    EVERY candidate, and a candidate the search leaves at ``sigma = 1`` must come
    back through the integrator to within the round-trip residual. Both are
    reported; neither substitutes for the other.

    ⭐ **THE SHIPPED FORM RETURNS THE INPUT SLICE UNCHANGED WHERE ``sigma* == 1``.**
    A path that is already clear is already a member of the constraint set, so the
    projection of it is itself — returning the round-tripped copy instead would
    inject a ~1e-9 m perturbation into 97.5 % of the fan for no reason, and would
    make the "only collider windows move" control unreadable. ``info`` carries the
    residual of the uniform (search-everywhere) form so that decision is MEASURED
    rather than assumed.

    ``has_agent [B]`` marks the windows that carry an agent track. A window without
    one is ``REASON_NO_AGENT`` and is returned untouched: no track means **no
    information about contact**, never a free pass.
    """
    if paths.dim() != 4 or paths.shape[-1] != 2:
        raise ValueError(f"expected [B, N, S, 2] paths, got {tuple(paths.shape)}")
    if not enabled:
        B, N = paths.shape[:2]
        z = torch.zeros(B, N, dtype=torch.long, device=paths.device)
        return paths, {"reason": z + REASON_CLEAR,
                       "sigma": torch.ones(B, N, dtype=paths.dtype, device=paths.device),
                       "dkappa": torch.zeros(B, N, dtype=paths.dtype, device=paths.device),
                       "roundtrip_max_m": torch.zeros((), dtype=paths.dtype),
                       "enabled": torch.zeros((), dtype=torch.bool)}
    B, N, S, _ = paths.shape
    dev = paths.device
    p = paths.to(torch.float64)
    if bool((p[:, :, 0, :].abs() > 1e-6).any()):
        raise ValueError("paths[..., 0, :] must be the ego origin (0, 0)")
    ld = lead.to(torch.float64).expand(B, N, S, 2)
    if has_agent is None:
        has = torch.ones(B, dtype=torch.bool, device=dev)
    else:
        has = has_agent.to(dev).bool().reshape(B)
    has_bn = has.reshape(B, 1).expand(B, N)

    #: the clearance the SEARCH must reach. The scorer flags with a strict ``<``
    #: at ``r_m``; clearing exactly ``r_m`` in float64 lands on the wrong side of
    #: an fp32 re-derivation about half the time (the MARGIN lesson of
    #: ``feasible_decode``, measured there at 0.609375). ``margin`` puts the output
    #: strictly outside by ~2 cm at m = 0 — physically nothing, ~1e5x the fp32 noise.
    #
    # ⛔⛔ THE MARGIN LADDER, AND WHY THE FIRST VERSION WITHOUT IT WAS WRONG.
    # A requested margin can be UNACHIEVABLE for a given candidate while a smaller
    # one is perfectly achievable. The first version searched a single ``r_need``
    # and, failing, returned the INPUT unchanged and labelled it ``UNAVOIDABLE`` —
    # so raising the margin from 4 m to 6 m took ``fan_contact`` at the scorer's own
    # 2 m radius from **0.000000 back up to 0.007389 (227 candidates)**. MEASURED
    # 2026-09-05 on the 240 x 128 bank: asking for MORE safety delivered LESS,
    # because the guarantee was "clear at r_need OR untouched" rather than "as clear
    # as this candidate can be made". ⇒ the ladder retries at decreasing margins and
    # reports ``margin_achieved`` per candidate, so the guarantee becomes the one a
    # caller actually wants: **clear at the scorer's radius unless NO member of the
    # family clears it, and clear at the requested margin wherever that is possible.**
    rungs = [float(margin_m)]
    if margin_fallback and float(margin_m) > 0.0:
        k = max(2, int(n_margin_rungs))
        rungs = [float(margin_m) * (1.0 - i / (k - 1.0)) for i in range(k)]
        rungs = sorted({round(x, 9) for x in rungs}, reverse=True)
    r_needs = [(float(r_m) + g) * (1.0 + float(margin)) for g in rungs]

    speed, heading, accel, lat = recover_controls(p, dt=dt, min_speed=min_speed)
    s0 = speed[..., 0]                                          # [B, N]
    h0 = heading[..., 0]                                        # [B, N]
    a0 = accel                                                  # [B, N, S-2]
    y0 = lat                                                    # [B, N, S-2]
    v_mid0 = speed[..., :-1].clamp_min(min_speed)
    kappa0 = y0 / v_mid0.pow(2)                                 # [B, N, S-2]

    sig_grid = torch.linspace(0.0, 1.0, int(n_sigma), dtype=torch.float64, device=dev)
    G = sig_grid.numel()
    dk_grid = torch.linspace(-float(kappa_max), float(kappa_max), int(n_dkappa),
                             dtype=torch.float64, device=dev)
    K = dk_grid.numel()

    best_sigma = torch.ones(B, N, dtype=torch.float64, device=dev)
    best_dkappa = torch.zeros(B, N, dtype=torch.float64, device=dev)
    solved = torch.zeros(B, N, dtype=torch.bool, device=dev)
    via_lateral = torch.zeros(B, N, dtype=torch.bool, device=dev)
    margin_achieved = torch.full((B, N), float("nan"), dtype=torch.float64, device=dev)

    for rung, r_need in zip(rungs, r_needs):
        # ---- axis 1: the LONGITUDINAL retraction, SCANNED (never bisected) --- #
        todo = has_bn & (~solved)
        if not bool(todo.any()):
            break
        chunk = max(1, int(2_000_000 // max(1, N * G * S)))
        for b0 in range(0, B, chunk):
            b1 = min(B, b0 + chunk)
            if not bool(todo[b0:b1].any()):
                continue
            sg = sig_grid.reshape(1, 1, G)
            ss = s0[b0:b1, :, None] * sg                            # [b, N, G]
            hh = h0[b0:b1, :, None].expand_as(ss)
            aa = a0[b0:b1, :, None, :] * sg[..., None]              # [b, N, G, S-2]
            # ⛔⛔ ONE FORMULA, HERE AND IN THE FINAL RE-INTEGRATION. Writing the
            # lateral control as ``lat * sigma^2`` is algebraically identical to
            # ``kappa * v_mid^2`` ONLY while ``sigma * v_mid > MIN_SPEED``; under
            # the floor the two DIVERGE, so a scan written one way and a
            # re-integration written the other SEARCHES ONE PATH AND EMITS A
            # DIFFERENT ONE. MEASURED 2026-09-05: the scan reported clear and
            # ``assert_no_contact`` on the OUTPUT read a 1.719 m clearance against
            # a 2.0 m disc — the module would have shipped a colliding path while
            # its own bookkeeping said zero. ⭐ Caught only because the control
            # re-derives from the OUTPUT with the scorer instead of trusting the
            # search (RETRACTION #30, verbatim).
            vmg = (v_mid0[b0:b1, :, None, :] * sg[..., None]).clamp_min(min_speed)
            yy = kappa0[b0:b1, :, None, :] * vmg.pow(2)
            q = integrate_controls(ss, hh, aa, yy, dt=dt, clamp=clamp_friction,
                                   a_max=a_max, kappa_max=kappa_max, mu=mu,
                                   margin=margin, min_speed=min_speed)
            d = min_rel_distance(q, ld[b0:b1, :, None, :, :], skip_first=skip_first)
            ok = (d >= r_need) & todo[b0:b1].unsqueeze(-1)
            # the LARGEST clear sigma on the grid — exact on the grid, and no
            # monotonicity is assumed, which is the point of scanning: a lead that
            # REVERSES toward the ego makes slowing down worse, and a bisection
            # would return a colliding path in exactly that case.
            idx = torch.where(ok, sig_grid.reshape(1, 1, G),
                              torch.full_like(d, -1.0)).amax(dim=-1)
            got = idx >= 0.0
            best_sigma[b0:b1] = torch.where(got, idx.clamp_min(0.0),
                                            best_sigma[b0:b1])
            margin_achieved[b0:b1] = torch.where(
                got, torch.full_like(idx, float(rung)), margin_achieved[b0:b1])
            solved[b0:b1] = solved[b0:b1] | got

        # ---- axis 2: the LATERAL evade, on this rung's residual only -------- #
        need_lat = has_bn & (~solved)
        if lateral and bool(need_lat.any()):
            bi, ni = torch.nonzero(need_lat, as_tuple=True)
            M = bi.numel()
            sg = sig_grid.reshape(1, G, 1)
            dk = dk_grid.reshape(1, 1, K)
            step = max(1, int(400_000 // max(1, G * K * S)))
            for m0 in range(0, M, step):
                m1 = min(M, m0 + step)
                b_, n_ = bi[m0:m1], ni[m0:m1]
                mm = b_.numel()
                # ⛔ EVERY control axis is materialised to the FULL [m, G, K] grid.
                # A [m, G, 1] speed broadcasts correctly inside the arithmetic and
                # then sizes the integrator's accumulator wrongly — a shape bug a
                # looser integrator would have turned into a silent wrong answer.
                ss = (s0[b_, n_][:, None, None] * sg).expand(mm, G, K).contiguous()
                hh = h0[b_, n_][:, None, None].expand(mm, G, K).contiguous()
                aa = (a0[b_, n_][:, None, None, :]
                      * sg[..., None]).expand(mm, G, K, S - 2).contiguous()
                vm = (v_mid0[b_, n_][:, None, None, :]
                      * sg[..., None]).clamp_min(min_speed)
                kk = kappa0[b_, n_][:, None, None, :] + dk[..., None]
                yy = (kk * vm.pow(2)).expand(mm, G, K, S - 2).contiguous()
                q = integrate_controls(ss, hh, aa, yy, dt=dt, clamp=clamp_friction,
                                       a_max=a_max, kappa_max=kappa_max, mu=mu,
                                       margin=margin, min_speed=min_speed)
                d = min_rel_distance(q, ld[b_, n_][:, None, None, :, :],
                                     skip_first=skip_first)          # [m, G, K]
                ok = d >= r_need
                # ⭐ MINIMAL INTERVENTION, ORDERED: among the clear members prefer
                # the largest sigma (least progress given up), then the smallest
                # |dkappa| (least steering imposed). A pure "max sigma" rule would
                # happily pick a violent swerve to keep 1 % more speed.
                score = torch.where(ok, sg.expand_as(d) - 1e-6 * dk.abs().expand_as(d),
                                    torch.full_like(d, -1.0))
                flat = score.reshape(score.shape[0], -1).argmax(dim=-1)
                gi, ki = flat // K, flat % K
                got = ok.reshape(ok.shape[0], -1).any(dim=-1)
                best_sigma[b_, n_] = torch.where(got, sig_grid[gi], best_sigma[b_, n_])
                best_dkappa[b_, n_] = torch.where(got, dk_grid[ki], best_dkappa[b_, n_])
                margin_achieved[b_, n_] = torch.where(
                    got, torch.full_like(sig_grid[gi], float(rung)),
                    margin_achieved[b_, n_])
                via_lateral[b_, n_] = via_lateral[b_, n_] | got
                solved[b_, n_] = solved[b_, n_] | got

    # ⛔ an UNSOLVED candidate keeps its own controls and is LABELLED, never
    # silently half-retracted: a partial retraction that still collides is worse
    # than an honest "this one cannot be fixed", because it looks like a fix.
    unsolved = has_bn & (~solved)
    best_sigma = torch.where(unsolved, torch.ones_like(best_sigma), best_sigma)
    best_dkappa = torch.where(unsolved, torch.zeros_like(best_dkappa), best_dkappa)

    # ---- re-integrate at the chosen (sigma, dkappa), uniformly ------------- #
    sg = best_sigma
    ss = s0 * sg
    hh = h0
    aa = a0 * sg[..., None]
    vm = (v_mid0 * sg[..., None]).clamp_min(min_speed)
    yy = (kappa0 + best_dkappa[..., None]) * vm.pow(2)
    q_uniform = integrate_controls(ss, hh, aa, yy, dt=dt, clamp=clamp_friction,
                                   a_max=a_max, kappa_max=kappa_max, mu=mu,
                                   margin=margin, min_speed=min_speed)

    # the round-trip control: an untouched candidate must come back to itself
    untouched = (sg == 1.0) & (best_dkappa == 0.0)
    if bool(untouched.any()):
        rt = (q_uniform[untouched] - p[untouched]).norm(dim=-1).amax()
    else:
        rt = torch.zeros((), dtype=torch.float64, device=dev)

    # ⭐ the SHIPPED form: identity where nothing was retracted
    keep = untouched | (~has_bn)
    out = torch.where(keep[..., None, None], p, q_uniform)

    reason = torch.full((B, N), REASON_CLEAR, dtype=torch.long, device=dev)
    reason = torch.where(~has_bn, torch.full_like(reason, REASON_NO_AGENT), reason)
    moved = has_bn & (~untouched)
    reason = torch.where(moved & (~via_lateral),
                         torch.full_like(reason, REASON_LONGITUDINAL), reason)
    reason = torch.where(moved & via_lateral,
                         torch.full_like(reason, REASON_LATERAL), reason)
    reason = torch.where(unsolved,
                         torch.full_like(reason, REASON_UNAVOIDABLE), reason)
    info = {"reason": reason, "sigma": best_sigma.to(paths.dtype),
            "dkappa": best_dkappa.to(paths.dtype),
            "margin_achieved_m": margin_achieved.to(paths.dtype),
            "solved": solved, "via_lateral": via_lateral,
            "roundtrip_max_m": rt.to(paths.dtype),
            "uniform": q_uniform.to(paths.dtype),
            "margin_rungs_m": torch.tensor(rungs, dtype=paths.dtype),
            "r_need_m": torch.tensor(float(r_needs[0]), dtype=paths.dtype),
            "r_need_floor_m": torch.tensor(float(r_needs[-1]), dtype=paths.dtype),
            "enabled": torch.ones((), dtype=torch.bool)}
    return out.to(paths.dtype), info


def assert_no_contact(paths: Tensor, lead: Tensor, has_agent: Tensor | None = None,
                      *, r_m: float = CONTACT_R_M) -> dict:
    """Re-derive the SCORER's own contact flag on ``paths`` and report the rate.

    ⭐ **It re-derives with ``rewards._collision`` itself**, not with this module's
    :func:`min_rel_distance`, and not with the projection's bookkeeping. A control
    that checks the arithmetic it just performed cannot see that it performed it on
    the wrong object (RETRACTION #30); a control that re-implements the predicate it
    is checking cannot see that both share a convention error.
    """
    from ..rl.rewards import _collision
    ctx = {"lead_path": lead.expand_as(paths) if lead.shape != paths.shape else lead,
           "ego_radius_m": r_m / 2.0, "obs_radius_m": r_m / 2.0,
           "obstacles": None}
    hit = (_collision(paths, ctx) < 0)
    if has_agent is not None:
        hit = hit & has_agent.reshape(-1, *([1] * (hit.dim() - 1))).bool()
    d = min_rel_distance(paths.to(torch.float64), lead.to(torch.float64))
    return {"contact_rate": float(hit.double().mean()),
            "contact_count": int(hit.sum()),
            "windows_with_contact": int(hit.any(dim=-1).sum()) if hit.dim() > 1 else int(hit.any()),
            "min_clearance_m": float(d.min()),
            "predicate": "tanitad.rl.rewards._collision (moving-lead, swept, relative frame)"}
