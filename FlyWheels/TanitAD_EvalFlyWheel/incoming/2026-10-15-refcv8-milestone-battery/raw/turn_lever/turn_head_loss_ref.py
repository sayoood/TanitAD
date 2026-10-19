"""REFERENCE implementation of the pre-registered L_turn term (PREREG_DRAFT_R8_TURN_REFINE_LOSS.md sec. 6).

Design-only artifact: NOT wired into any trainer. It exists so the pre-registration's pseudo-diff is a function
that has been run against ANALYTIC targets (literal expected values, never an expression over this code), plus a
mutation that reintroduces a sign error and must go RED. CPU only.

    python turn_head_loss_ref.py        -> prints one line per check and exits 1 on any failure

Definitions (frozen by the pre-registration):
  * slots 0..3 of the 8-slot plan = 0.5 / 1.0 / 1.5 / 2.0 s (horizons [5,10,15,20] ticks at 0.1 s, config.json
    `horizons`); chord k runs from slot k-1 (origin for k = 0) to slot k -- the SAME chords `heading_drift.py`
    scores ("segment k = chord ending at instant k").
  * e_h[k]  = |wrap(h_plan[k] - h_gt[k])|                     chord-heading error [rad]
  * e_i[k]  = |wrap(dh_plan[k] - dh_gt[k])|, dh[k] = h[k]-h[k-1]  heading-INCREMENT error = curvature x arc [rad]
              (no division by arc length, so it does not blow up at low speed)
  * a chord counts only if BOTH plan and GT chords are >= min_chord_m long and the slot is valid
  * per-window weight w_win = clamp((|yaw_rate_gt| - 0.02) / (0.15 - 0.02), 0, cap): 0 on the instrument's
    'straight' bin, 1 at the 'sharp' edge, capped
  * L_turn = [sum w e_h / sigma_h + w_inc * sum w e_i / sigma_h] / sum(w * valid)
"""
from __future__ import annotations

import math
import sys

import torch

SIGMA_H_RAD = math.radians(2.0)
YAW_STRAIGHT = 0.02
YAW_SHARP = 0.15


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def window_weight(yaw_rate_gt: torch.Tensor, v0: torch.Tensor, cap: float = 3.0,
                  v_min: float = 2.0) -> torch.Tensor:
    """[B] -> [B]; 0 on straights (< 0.02 rad/s) and on crawling windows (v0 < 2 m/s, the instrument's 'undef')."""
    w = ((yaw_rate_gt.abs() - YAW_STRAIGHT) / (YAW_SHARP - YAW_STRAIGHT)).clamp(0.0, cap)
    return w * (v0 >= v_min).to(w.dtype)


def turn_head_loss(plan: torch.Tensor, gt: torch.Tensor, slot_valid: torch.Tensor, w_win: torch.Tensor, *,
                   n_slots: int = 4, min_chord_m: float = 0.5, w_inc: float = 0.5,
                   sigma_h: float = SIGMA_H_RAD, _mutate_sign: bool = False) -> tuple[torch.Tensor, int]:
    """plan / gt [B, S, 2] (ego frame at NOW, x fwd, y left, metres); slot_valid [B, S]; w_win [B].
    Returns (scalar loss, number of windows with non-zero weight AND >= 1 valid chord)."""
    b = plan.shape[0]
    z = plan.new_zeros(b, 1, 2)
    q = torch.cat([z, plan[:, :n_slots].to(torch.float32)], 1)
    g = torch.cat([z, gt[:, :n_slots].to(torch.float32)], 1)
    dp, dg = q[:, 1:] - q[:, :-1], g[:, 1:] - g[:, :-1]
    lp, lg = torch.linalg.vector_norm(dp, dim=-1), torch.linalg.vector_norm(dg, dim=-1)
    ok = (lp >= min_chord_m) & (lg >= min_chord_m) & slot_valid[:, :n_slots].to(torch.bool)
    # atan2 on a gated chord only: replace short chords by a unit vector so no 1/|d| gradient can reach them
    one = torch.tensor([1.0, 0.0], dtype=dp.dtype)
    dp_s = torch.where(ok[..., None], dp, one.expand_as(dp))
    dg_s = torch.where(ok[..., None], dg, one.expand_as(dg))
    hp = torch.atan2(dp_s[..., 1], dp_s[..., 0])
    hg = torch.atan2(dg_s[..., 1], dg_s[..., 0])
    e_h = _wrap(hp + hg if _mutate_sign else hp - hg).abs()
    inc_p, inc_g = hp[:, 1:] - hp[:, :-1], hg[:, 1:] - hg[:, :-1]
    e_i = _wrap(inc_p - inc_g).abs()
    ok_i = ok[:, 1:] & ok[:, :-1]
    w = w_win.to(torch.float32)[:, None]
    okf, okif = ok.to(torch.float32), ok_i.to(torch.float32)
    num = (w * e_h * okf).sum() + float(w_inc) * (w[:, :1] * e_i * okif).sum()
    den = (w * okf).sum().clamp_min(1.0)
    n_rows = int(((w_win > 0) & ok.any(-1)).sum())
    return num / (den * float(sigma_h)), n_rows


# ----------------------------------------------------------------------------------------------------------------- #
# analytic fixtures                                                                                                   #
# ----------------------------------------------------------------------------------------------------------------- #
T_SLOTS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)


def arc(kappa: float, v: float) -> torch.Tensor:
    """[1, 8, 2] points of a constant-curvature, constant-speed path at the 8 slot times (left turn for kappa > 0)."""
    pts = []
    for t in T_SLOTS:
        s = v * t
        if abs(kappa) < 1e-12:
            pts.append((s, 0.0))
        else:
            pts.append((math.sin(kappa * s) / kappa, (1.0 - math.cos(kappa * s)) / kappa))
    return torch.tensor([pts], dtype=torch.float32)


def main() -> int:
    fails = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal fails
        fails += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    sv = torch.ones(1, 8)
    v, kgt = 8.0, 1.0 / 30.0              # 8 m/s on a 30 m radius: yaw rate 0.2667 rad/s (a 'sharp' window)
    yaw = torch.tensor([v * kgt])
    w1 = window_weight(yaw, torch.tensor([v]))
    # literal: (0.26667 - 0.02) / 0.13 = 1.897436 (<= cap 3)
    check("weight literal", abs(float(w1) - 1.8974359) < 1e-5, f"w = {float(w1):.7f}, expected 1.8974359")
    check("weight straight = 0", float(window_weight(torch.tensor([0.01]), torch.tensor([8.0]))) == 0.0,
          "yaw 0.01 rad/s -> 0")
    check("weight crawl = 0", float(window_weight(torch.tensor([0.5]), torch.tensor([1.0]))) == 0.0,
          "v0 1 m/s -> 0")

    gt = arc(kgt, v)
    l0, n0 = turn_head_loss(gt.clone(), gt, sv, w1)
    check("plan == GT -> exactly 0", float(l0) == 0.0 and n0 == 1, f"L = {float(l0):.3e}, rows = {n0}")

    st = arc(0.0, v)
    ls, _ = turn_head_loss(st, st, sv, torch.tensor([1.0]))
    check("straight plan == straight GT -> 0", float(ls) == 0.0, f"L = {float(ls):.3e}")

    # under-rotated plan: curvature 0.7 x GT. A circular-arc chord's direction is the MEAN of its end tangents, so
    # chord k (arc length s_{k-1} -> s_k) has heading kappa (s_{k-1} + s_k) / 2. Uniform ds = 4 m (0.5 s at 8 m/s):
    #   e_h[k] = 0.3 * kgt * (s_{k-1} + s_k) / 2 = 0.3/30 * {2, 6, 10, 14} = {0.02, 0.06, 0.10, 0.14} rad
    #   e_i[k] = 0.3 * kgt * ds = 0.3/30 * 4 = 0.04 rad for k = 1..3
    # L = [w * (0.02+0.06+0.10+0.14) + 0.5 * w * 3 * 0.04] / (w * 4) / sigma = (0.32 + 0.06) / 4 / sigma
    expected = (0.32 + 0.06) / 4.0 / math.radians(2.0)          # = 2.7215495
    pu = arc(0.7 * kgt, v)
    lu, _ = turn_head_loss(pu, gt, sv, w1)
    check("0.7x under-rotation literal", abs(float(lu) - 2.7215495) < 1e-4,
          f"L = {float(lu):.7f}, expected {expected:.7f} (literal 2.7215495)")

    # gradient direction: parameterise the plan by its curvature; one SGD step must move kappa TOWARD kgt
    kp = torch.tensor(0.7 * kgt, requires_grad=True)
    s = torch.tensor([v * t for t in T_SLOTS])
    pts = torch.stack([torch.sin(kp * s) / kp, (1 - torch.cos(kp * s)) / kp], -1)[None]
    lk, _ = turn_head_loss(pts, gt, sv, w1)
    lk.backward()
    check("gradient increases rotation of an under-rotated plan", float(kp.grad) < 0.0,
          f"dL/dkappa = {float(kp.grad):.4f} (< 0 => descent raises kappa toward GT)")

    # the weight is a WINDOW weight: a straight window contributes nothing even with a wrong plan
    lz, nz = turn_head_loss(pu, gt, sv, torch.tensor([0.0]))
    check("zero-weight window -> 0", float(lz) == 0.0 and nz == 0, f"L = {float(lz):.3e}, rows = {nz}")

    # crawling plan: every chord < 0.5 m -> no valid chord, finite zero gradient (no atan2 blow-up)
    cp = (gt * 0.01).clone().requires_grad_(True)
    lc, nc = turn_head_loss(cp, gt * 0.01, sv, torch.tensor([1.0]))
    lc.backward()
    check("short chords gated, finite grad", float(lc) == 0.0 and bool(torch.isfinite(cp.grad).all()),
          f"L = {float(lc):.3e}, grad finite = {bool(torch.isfinite(cp.grad).all())}")

    # MUTATION: the sign error (h_plan + h_gt) must make the plan == GT case NON-zero -> this check must go RED
    lm, _ = turn_head_loss(gt.clone(), gt, sv, w1, _mutate_sign=True)
    check("mutation (sign flip) is detected by the plan==GT literal", float(lm) > 1e-3,
          f"mutated L(plan==GT) = {float(lm):.4f} (> 0 => the literal-0 check would be RED)")

    print(f"{'ALL PASS' if fails == 0 else str(fails) + ' FAIL'} ({__file__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
