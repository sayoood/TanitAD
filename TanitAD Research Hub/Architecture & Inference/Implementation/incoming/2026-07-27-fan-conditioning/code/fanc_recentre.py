#!/usr/bin/env python3
"""S4 -- CONDITIONING THE **REAL** FAN, in the deployed regime.

The static-anchor study (fanc_anchors.py) answers the CoverNet question, but it
answers it about a static anchor set whose ceiling is ~3x worse than the fan that
actually ships. This script asks the same question of the REAL fan, which is the
regime the v5 retrain would live in.

The shipped fan proposes a ~17 m/s speed distribution in EVERY window (S1:
slope on v0 = -0.129, GT's = +1.000). A `v0`-conditioned anchor set would, at
minimum, RE-CENTRE that distribution on the ego's actual speed. That is a pure
REALLOCATION of the same N candidates -- unlike the clip, which can only delete
-- so it CAN move the ceiling, in either direction.

Three conditioning transforms, all applied to the real per-candidate
trajectories, all preserving N and the fan's shape diversity:

  T_shift  along-track offset so the fan's median implied speed equals v0
  T_scale  along-track scale   so the fan's median implied speed equals v0
  T_affine shift+scale so median -> v0 AND the speed spread is a fitted
           multiple of the reachable band (a_max * t), i.e. a fully
           state-conditioned longitudinal envelope

All transform parameters are fitted OUT-OF-FOLD (episode-disjoint) where they
are fitted at all; T_shift and T_scale are parameter-free given v0.

Usage:  python fanc_recentre.py
"""
from __future__ import annotations

import json

import numpy as np

from fanc_common import (OUT, T_HORIZON, WP_STEPS, ade, ci_paired, ci_single,
                         eid_str, episode_folds, load_refc_fan,
                         pick_nearest_to, r4)

ARMS = ("xl", "base", "small")
A_MAX_GRID = (1.0, 1.5, 2.0, 2.5, 4.0)


def times() -> np.ndarray:
    """Waypoint times in seconds: steps (5,10,15,20) at 10 Hz -> .5 .. 2.0 s."""
    return np.asarray(WP_STEPS, dtype=float) * (T_HORIZON / max(WP_STEPS))


def t_shift(fan, v0):
    """Rigid along-track shift so the fan's MEDIAN implied speed becomes v0."""
    t = times()[None, None, :]
    med = np.median(fan[:, :, -1, 0] / T_HORIZON, axis=1)      # [W]
    out = fan.copy()
    out[..., 0] += (v0 - med)[:, None, None] * t
    return out


def t_scale(fan, v0, eps=1e-3):
    """Multiplicative along-track scale so the median implied speed becomes v0."""
    med = np.median(fan[:, :, -1, 0] / T_HORIZON, axis=1)      # [W]
    k = np.where(np.abs(med) < eps, 1.0, v0 / np.where(np.abs(med) < eps, 1.0, med))
    out = fan.copy()
    out[..., 0] *= k[:, None, None]
    return out


def t_affine(fan, v0, a_max):
    """Shift the centre to v0 AND compress the speed spread to the +-a_max band.

    The candidate speed offsets around the fan median are rescaled so their 5-95
    span matches the physically reachable span 2*a_max*T. This is the full
    state-conditioned longitudinal envelope: centre AND width both come from the
    state, and only the fan's residual shape diversity is retained.
    """
    s = fan[:, :, -1, 0] / T_HORIZON                             # [W,N]
    med = np.median(s, axis=1, keepdims=True)
    off = s - med                                                # [W,N]
    span = np.percentile(off, 95, axis=1, keepdims=True) - \
        np.percentile(off, 5, axis=1, keepdims=True)
    target = 2.0 * a_max * T_HORIZON
    k = np.where(span < 1e-6, 1.0, target / np.maximum(span, 1e-6))
    new_s = v0[:, None] + off * k                                # [W,N]
    t = times()[None, None, :]
    out = fan.copy()
    # replace the along-track profile by a constant-speed ramp at the new speed,
    # preserving each candidate's along-track SHAPE deviation from its own ramp
    old_ramp = (s[:, :, None]) * t
    shape = fan[..., 0] - old_ramp
    out[..., 0] = new_s[:, :, None] * t + shape
    return out


def score(fan, gt, refs):
    """Ceiling + realised (nearest-to-reference) for a fan [W,N,4,2]."""
    err = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)     # [W,N]
    out = {"ceiling": err.min(1)}
    W = len(gt)
    for k, R in refs.items():
        d = np.linalg.norm(fan - R[:, None], axis=-1).mean(-1)
        idx = d.argmin(1)
        out[f"realised_{k}"] = err[np.arange(W), idx]
    return out, err


def main() -> None:
    res = {"_stream": "2026-07-27-fan-conditioning",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "_what": "v0-conditioning transforms applied to the REAL REF-C fans"}

    for arm in ARMS:
        d = load_refc_fan(arm)
        fan = d["fan"].numpy().astype(np.float64)
        gt = d["gt"].numpy().astype(np.float64)
        v0 = d["v0"].numpy().astype(np.float64)
        cv = d["cv"].numpy().astype(np.float64)
        sel = d["sel"].numpy()
        eid = eid_str(d)
        W = len(gt)
        refc_pick = fan[np.arange(W), sel]

        refs = {"cv": cv, "refc_pick": refc_pick, "oracle_gt": gt}
        base, base_err = score(fan, gt, refs)
        as_trained = base_err[np.arange(W), sel]

        arm_res = {
            "n_anchors": int(fan.shape[1]),
            "as_trained": ci_single(as_trained, eid),
            "baseline_ceiling": ci_single(base["ceiling"], eid),
            "baseline_realised": {k: ci_single(base[f"realised_{k}"], eid)
                                  for k in refs},
            "reference_own_ade": {k: r4(ade(R, gt).mean()) for k, R in refs.items()},
            "_fidelity_oracle_ref_equals_ceiling": bool(
                np.allclose(base["realised_oracle_gt"], base["ceiling"])),
            "transforms": {},
        }

        variants = {"T_shift": t_shift(fan, v0), "T_scale": t_scale(fan, v0)}
        for a in A_MAX_GRID:
            variants[f"T_affine_a{a:g}"] = t_affine(fan, v0, a)
        # NEGATIVE CONTROL: re-centre on a SHUFFLED v0. A transform that helps
        # because it uses the STATE must not help when the state is destroyed.
        rng = np.random.default_rng(20260727)
        v0_shuf = rng.permutation(v0)
        variants["NC_shift_shuffled_v0"] = t_shift(fan, v0_shuf)
        variants["NC_affine_shuffled_v0"] = t_affine(fan, v0_shuf, 2.5)

        for name, f2 in variants.items():
            s2, _ = score(f2, gt, refs)
            e = {"ceiling": ci_single(s2["ceiling"], eid),
                 "dCeiling_vs_baseline": ci_paired(s2["ceiling"],
                                                   base["ceiling"], eid),
                 "realised": {}, "dRealised_vs_baseline": {},
                 "slope_fan_speed_on_v0": r4(np.polyfit(
                     v0, (f2[:, :, -1, 0] / T_HORIZON).mean(1), 1)[0])}
            for k in refs:
                e["realised"][k] = ci_single(s2[f"realised_{k}"], eid)
                e["dRealised_vs_baseline"][k] = ci_paired(
                    s2[f"realised_{k}"], base[f"realised_{k}"], eid)
            # vs the AS-TRAINED selector, the number the bars are set against
            e["dRealised_refc_pick_vs_as_trained"] = ci_paired(
                s2["realised_refc_pick"], as_trained, eid)
            arm_res["transforms"][name] = e

        res[f"refc_{arm}"] = arm_res
        print(f"\n=== REF-C-{arm.upper()} (N={fan.shape[1]}) "
              f"as-trained {arm_res['as_trained']['mean']:.4f} "
              f"ceiling {arm_res['baseline_ceiling']['mean']:.4f} "
              f"fidelity={arm_res['_fidelity_oracle_ref_equals_ceiling']}")
        print(f"{'transform':24s} {'slope':>7s} {'ceiling':>9s} {'dCeil':>9s} "
              f"{'real(cv)':>9s} {'real(refc)':>11s} {'d vs A0':>9s} sep")
        print(f"{'(baseline, no transform)':24s} {'-0.129':>7s} "
              f"{arm_res['baseline_ceiling']['mean']:9.4f} {'-':>9s} "
              f"{arm_res['baseline_realised']['cv']['mean']:9.4f} "
              f"{arm_res['baseline_realised']['refc_pick']['mean']:11.4f}")
        for name, e in arm_res["transforms"].items():
            dd = e["dRealised_refc_pick_vs_as_trained"]
            print(f"{name:24s} {e['slope_fan_speed_on_v0']:+7.3f} "
                  f"{e['ceiling']['mean']:9.4f} "
                  f"{e['dCeiling_vs_baseline']['delta']:+9.4f} "
                  f"{e['realised']['cv']['mean']:9.4f} "
                  f"{e['realised']['refc_pick']['mean']:11.4f} "
                  f"{dd['delta']:+9.4f} {'SEP' if dd['separated'] else '-'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fanc_recentre.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT / "fanc_recentre.json")


if __name__ == "__main__":
    main()
