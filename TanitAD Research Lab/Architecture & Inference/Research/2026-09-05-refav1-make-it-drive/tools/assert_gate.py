#!/usr/bin/env python
"""Assert the CURVATURE GATE numerically, on the real functions.

⭐ WHY THIS EXISTS. `GATE_MECHANISM.md` establishes by source-reading that when
refav1's goal head decodes LANE_KEEP, the entire iCEM candidate population has
curvature identically zero — so the planner *cannot* turn. Source-reading is an
argument, not a measurement. This script calls the ACTUAL
`canonical_controls`, `_baseline_controls` and `colored_noise` and asserts the
three load-bearing links, so the claim is MEASURED rather than INHERITED.

⛔ CONTROLS, not decoration. Every assertion is paired with a same-breath
POSITIVE control that must read a KNOWN NON-ZERO value:

  * link 2  LANE_KEEP curvature == 0  ..vs..  TURN_L curvature == +GOAL_KAPPA_TURN
  * link 5  baselines curvature == 0  ..vs..  decel_1.5 ACCEL == -1.5 (non-zero)
  * link 6  noise time-mean == 0      ..vs..  noise per-sample std == 1 (non-zero)

A file that could not be imported, or a probe that silently returned an empty
tensor, would make the "== 0" assertions pass vacuously. The paired non-zero
control is what makes a pass mean something — the CLAUDE.md probe rule.

Run: python assert_gate.py --out gate_assert.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=None,
                    help="path to stack/ (default: infer from repo layout)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.stack:
        sys.path.insert(0, os.path.abspath(a.stack))
    import torch
    from tanitad.refs.refa_v1 import (canonical_controls, GOAL_KAPPA_TURN,
                                      GOAL_A_MAX)
    from tanitad.refs.refa_v1_plan import (_baseline_controls, colored_noise,
                                           PlanConfig)

    R: dict = {"torch": torch.__version__, "checks": {}, "PASS": None}
    fails: list[str] = []

    def chk(name: str, ok: bool, detail: dict) -> None:
        R["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            fails.append(name)

    OP_STEPS, OP_DT, V0 = 30, 0.2, 10.0

    # ---- LINK 2: the token -> curvature map ----------------------------- #
    # The GATE: LANE_KEEP must produce curvature identically zero.
    lk = canonical_controls("LANE_KEEP", "CRUISE", V0, OP_STEPS, OP_DT)
    lk_kap_absmax = float(lk[:, 1].abs().max())
    # POSITIVE CONTROL, same breath: TURN_L must produce a NON-ZERO curvature
    # equal to +GOAL_KAPPA_TURN. If this reads 0 the probe is broken, not the
    # model, and the LANE_KEEP zero above would be meaningless.
    tl = canonical_controls("TURN_L", "CRUISE", V0, OP_STEPS, OP_DT)
    tl_kap_max = float(tl[:, 1].max())
    tr = canonical_controls("TURN_R", "CRUISE", V0, OP_STEPS, OP_DT)
    tr_kap_min = float(tr[:, 1].min())

    chk("link2_LANE_KEEP_curvature_is_zero", lk_kap_absmax == 0.0,
        {"lane_keep_kappa_absmax": lk_kap_absmax, "expect": 0.0})
    # ⚠️ tolerance is FLOAT32, not float64: `canonical_controls` builds a
    # float32 tensor, so GOAL_KAPPA_TURN=0.08 reads back 0.07999999821186066.
    # A 1e-9 threshold fails on precision alone and looks like a finding — it
    # is not. 1e-6 is well below any real difference (the alternative token
    # values differ by >= 1e-2) and well above float32 eps at this magnitude.
    TOL = 1e-6
    chk("link2_CONTROL_TURN_L_curvature_is_nonzero",
        abs(tl_kap_max - GOAL_KAPPA_TURN) < TOL and tl_kap_max > 0,
        {"turn_l_kappa_max": tl_kap_max, "expect": GOAL_KAPPA_TURN,
         "tol": TOL})
    chk("link2_CONTROL_TURN_R_curvature_is_negative",
        abs(tr_kap_min + GOAL_KAPPA_TURN) < TOL and tr_kap_min < 0,
        {"turn_r_kappa_min": tr_kap_min, "expect": -GOAL_KAPPA_TURN,
         "tol": TOL})

    # The full vocabulary: which tokens can steer at all?
    lat_vocab = ["LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
                 "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R"]
    steer = {t: float(canonical_controls(t, "CRUISE", V0, OP_STEPS, OP_DT)
                      [:, 1].abs().max()) for t in lat_vocab}
    R["curvature_by_lat_token"] = steer
    zero_tokens = sorted(t for t, v in steer.items() if v == 0.0)
    chk("link2_zero_curvature_tokens_are_exactly_LANE_KEEP_and_ABORT_LC",
        zero_tokens == ["ABORT_LC", "LANE_KEEP"],
        {"zero_curvature_tokens": zero_tokens})

    # ---- LINK 5: no injected baseline carries curvature ------------------ #
    pc = PlanConfig()
    base = _baseline_controls(pc, V0, "cpu", None)      # proposal=None: the
    #                                                     refav1 arm's config
    base_kap = {k: float(v[:, 1].abs().max()) for k, v in base.items()}
    base_acc = {k: float(v[:, 0].abs().max()) for k, v in base.items()}
    R["baseline_curvature_absmax"] = base_kap
    R["baseline_accel_absmax"] = base_acc
    chk("link5_no_baseline_carries_curvature",
        len(base_kap) > 0 and all(v == 0.0 for v in base_kap.values()),
        {"per_baseline_kappa_absmax": base_kap, "n_baselines": len(base_kap)})
    # POSITIVE CONTROL, same breath: decel_1.5 MUST carry a non-zero ACCEL.
    # Without this, an empty/degenerate baseline dict would pass the line above.
    chk("link5_CONTROL_decel_carries_nonzero_accel",
        base_acc.get("decel_1.5", 0.0) > 0.0,
        {"decel_1.5_accel_absmax": base_acc.get("decel_1.5"),
         "expect": min(1.5, pc.a_max)})

    # ---- LINK 6: coloured noise is zero-mean along TIME ------------------ #
    g = torch.Generator().manual_seed(0)
    n, H, A = 512, pc.horizon, 2
    for beta in (0.0, 1.0, 2.0, 3.0):
        z = colored_noise((n, H, A), beta, device="cpu", generator=g)
        tmean_absmax = float(z.mean(dim=1).abs().max())     # mean over TIME
        std_min = float(z.std(dim=1).min())
        if beta == 0.0:
            # white noise is NOT mean-subtracted (documented: returns randn)
            R["checks"][f"link6_beta{beta}_white_is_not_centred"] = {
                "ok": True, "time_mean_absmax": tmean_absmax,
                "note": "beta=0 returns randn un-centred, by design"}
            continue
        chk(f"link6_beta{beta}_noise_time_mean_is_zero", tmean_absmax < 1e-5,
            {"time_mean_absmax": tmean_absmax})
        # POSITIVE CONTROL: the noise must actually VARY (unit std per sample).
        # An all-zero tensor would satisfy "mean is zero" vacuously.
        chk(f"link6_beta{beta}_CONTROL_noise_is_nonzero", std_min > 0.5,
            {"per_sample_std_min": std_min, "expect": "~1.0"})

    # ---- THE COMPOSITE: LANE_KEEP goal => zero-curvature population ------ #
    # Reconstruct exactly what the population's curvature column can contain
    # when the goal head decodes LANE_KEEP: baselines + zero-mean noise around
    # a seed of zeros. The reachable SUSTAINED curvature (the time-mean of the
    # curvature column) is then identically zero.
    seed_lk = lk[:pc.horizon, :]
    pop_sustained = []
    for _ in range(64):
        eps = colored_noise((256, pc.horizon, 2), pc.beta if hasattr(pc, "beta")
                            else 2.0, device="cpu", generator=g)
        cand = seed_lk.unsqueeze(0) + eps          # mean seeded by the seed only
        pop_sustained.append(float(cand[:, :, 1].mean(dim=1).abs().max()))
    lk_sustained = max(pop_sustained)
    # and the same with a TURN_L seed — the POSITIVE control that must be big
    seed_tl = tl[:pc.horizon, :]
    eps = colored_noise((256, pc.horizon, 2), 2.0, device="cpu", generator=g)
    tl_sustained = float((seed_tl.unsqueeze(0) + eps)[:, :, 1].mean(dim=1)
                         .abs().max())
    chk("composite_LANE_KEEP_seed_gives_zero_sustained_curvature",
        lk_sustained < 1e-6,
        {"max_abs_time_mean_curvature": lk_sustained, "n_draws": 64 * 256})
    chk("composite_CONTROL_TURN_L_seed_gives_nonzero_sustained_curvature",
        tl_sustained > 1e-3,
        {"max_abs_time_mean_curvature": tl_sustained})

    R["PASS"] = not fails
    R["failed_checks"] = fails
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    for k, v in R["checks"].items():
        print(("PASS " if v["ok"] else "FAIL ") + k, {kk: vv for kk, vv in
              v.items() if kk != "ok"})
    print("\nOVERALL:", "PASS" if R["PASS"] else f"FAIL {fails}")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
