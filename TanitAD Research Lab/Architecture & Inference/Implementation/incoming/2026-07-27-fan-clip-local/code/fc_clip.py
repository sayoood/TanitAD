#!/usr/bin/env python3
"""E-CLIP -- the pre-registered LONGITUDINAL-BAND CLIP on the frozen fan.

Two halves, because the artifact situation forces it (FAN_CLIP_LOCAL.md S2):

  HALF A -- the EXACT band, on the REF-C anchored fans (xl/base/small), which are
            the ONLY fans in the repo carrying per-candidate TRAJECTORIES on the
            canonical 881-window val deployment. Route: the as-trained selector.
  HALF B -- the SURROGATE band, on v4's fan, whose per-candidate trajectories are
            NOT staged. Clip variable = `C2_wm_ref_proximity` (mean distance to
            the world model's roll-out under the OBSERVED action), which is
            ground-truth-free and maps to a longitudinal acceleration budget
            exactly (see THETA_PER_A below). Routes: as-trained AND
            imagination-consistency, under BOTH scorer world models.

BAND PROVENANCE -- physics and configuration ONLY.
    s        = a candidate's 2 s along-track displacement
    band     = |s - v0*T| <= 0.5 * a_max * T^2     (T = 2 s)  AND  s >= 0
    a_max sweeps a fixed physical grid. The named anchor a_max = 2.5 m/s^2 is
    `FlagshipV15Head.cfg.sel_accel_max` (stack/tanitad/models/flagship_v15.py:139
    -- "the 2 s reachable-speed clamp"), a CONFIG constant of the model under
    test. NO held-out error, NO ground-truth future pose, and NO val-set
    statistic other than the observed v0 (an INPUT, available at deploy time)
    enters the band. A band chosen from held-out error would be a privileged
    intervention; this one cannot be, by construction.

CONTROLS (all pre-registered, all reported at every band):
    random-on-survivors  -- how much of any gain is the CLIP, not the ROUTE
    anti-clip            -- keep ONLY the candidates the band rejects
    oracle clip          -- positive control: must drive every route to
                            oracle_in_fan, or the masking plumbing is broken
    oracle survival      -- does the best-in-fan candidate survive the band?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import fc_common as F

OUT = Path(__file__).resolve().parents[1] / "raw"

# ---- the physical grid, frozen before any number was computed --------------
A_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0,
          20.0, 30.0, 50.0, float("inf")]
A_NAMED = {2.5: "FlagshipV15Head.cfg.sel_accel_max (config default)",
           2.0: "the v5 harness getattr fallback",
           1.5: "comfortable AV longitudinal accel (engineering)",
           8.0: "hard emergency braking, dry asphalt (engineering)"}

# mean over j=1..20 of 0.5 * a * (j*dt)^2  with dt=0.1  ==  0.7175 * a
_MEAN_J2 = float(np.mean(np.arange(1, 21) ** 2))          # 143.5
THETA_PER_A = 0.5 * (0.1 ** 2) * _MEAN_J2                 # 0.7175 m per m/s^2

BARS = {"as_trained_v4": 0.8563,
        "v5_sec7.2_CONFIRM_beat_by_more_than_A4": 0.7706,
        "CONFIRM_barA_in_sample_ceiling": 0.4907,
        "STRONG_deployed_v1": 0.4271}


# --------------------------------------------------------------------------- #
def summarise(fan_err, keep, routes, eid, tag):
    """One band row: every route, both controls, oracle survival, cost of clip."""
    W, N = fan_err.shape
    n_surv = keep.sum(dim=1)
    oracle_idx = fan_err.argmin(dim=1)
    surv_oracle = keep.gather(1, oracle_idx[:, None]).squeeze(1)
    row = {
        "_tag": tag,
        "frac_candidates_removed": F.r4(1.0 - keep.float().mean()),
        "mean_survivors": F.r4(n_surv.float().mean()),
        "median_survivors": int(n_surv.median()),
        "n_windows_no_survivor_fallback": int((n_surv == 0).sum()),
        "oracle_survival_frac": F.r4(surv_oracle.float().mean()),
        "oracle_in_survivors_ade": F.r4(F.min_masked(fan_err, keep).mean()),
    }
    per_window = {}
    for name, cost in routes.items():
        pick = F.argmin_masked(cost, keep)
        v = F.ade_of_pick(fan_err, pick)
        per_window[name] = v
        row[name] = F.r4(v.mean())
    rnd = F.random_masked(keep, F.SEED)
    per_window["ctrl_random_on_survivors"] = F.ade_of_pick(fan_err, rnd)
    row["ctrl_random_on_survivors"] = F.r4(
        per_window["ctrl_random_on_survivors"].mean())
    return row, per_window


def rank_of_candidate(base_rank: torch.Tensor) -> torch.Tensor:
    """`base_rank[w, r]` is the candidate INDEX at rank r (an argsort), NOT the
    rank of candidate c. Selecting with it directly picks garbage -- it did here
    on the first run and produced a constant 40.1611 for the as-trained route.
    This inverts the permutation so `out[w, c]` is candidate c's rank, and the
    caller asserts the unclipped route reproduces the committed 0.8563.
    """
    return base_rank.argsort(dim=1)


def band_keep(term_along, v0, a_max, a_dec=None, allow_reverse=False):
    """The EXACT physical band on 2 s along-track displacement.

    `a_dec` (default = `a_max`) lets the deceleration side be widened
    independently, so the null can be tested against an ASYMMETRIC band
    (comfortable acceleration + emergency braking) as well as a symmetric one.
    """
    if a_max == float("inf"):
        return torch.ones_like(term_along, dtype=torch.bool)
    a_dec = a_max if a_dec is None else a_dec
    centre = (v0 * F.T_HORIZON)[:, None]
    up = 0.5 * a_max * F.T_HORIZON ** 2
    dn = 0.5 * a_dec * F.T_HORIZON ** 2
    keep = (term_along <= centre + up) & (term_along >= centre - dn)
    if not allow_reverse:
        keep &= term_along >= 0.0
    return keep


# =========================================================================== #
# HALF A -- the EXACT band on REF-C fans
# =========================================================================== #
def half_a() -> dict:
    R = {"_read": "EXACT 2 s along-track band. Route = the as-trained selector "
                  "(argmax anchor logits) restricted to survivors. This is the "
                  "band V5 S7.2 pre-registered, on the fans whose trajectories "
                  "are actually in the repo.",
         "_band": "|s - v0*T| <= 0.5*a_max*T^2 and s >= 0, T=2 s",
         "_theta_provenance": A_NAMED}
    for arm in ("xl", "base", "small"):
        d = F.load_refc_fan(arm)
        fan, gt, v0 = d["fan"].float(), d["gt"].float(), d["v0"].float()
        err = (fan - gt[:, None]).norm(dim=-1).mean(dim=-1)          # [W, N]
        ta = fan[:, :, -1, 0]                                        # [W, N]
        eid = F.eid_of(torch.as_tensor([int(x) for x in d["eid"]]))
        # as-trained route: HIGHER logit = better, so cost = -logit
        routes = {"route_as_trained": -d["logits"].float()}
        base_v = F.ade_of_pick(err, d["sel"].long())

        rows, anti_rows = [], []
        for a in A_GRID:
            keep = band_keep(ta, v0, a)
            row, pw = summarise(err, keep, routes, eid, f"a_max={a}")
            row["a_max_ms2"] = None if a == float("inf") else a
            row["paired_vs_unclipped_as_trained"] = F.ci_paired(
                pw["route_as_trained"], base_v, eid)
            row["single_ci_route_as_trained"] = F.ci_single(
                pw["route_as_trained"], eid)
            rows.append(row)
            if a != float("inf"):
                anti, _ = summarise(err, ~keep, routes, eid, f"ANTI a_max={a}")
                anti["a_max_ms2"] = a
                anti_rows.append(anti)

        # ASYMMETRIC band -- comfortable acceleration + emergency braking, and a
        # reverse-permitting variant. Tests whether the null depends on the band
        # SHAPE rather than only on its width.
        asym = []
        for a_acc, a_dec, rev in ((2.5, 8.0, False), (1.5, 8.0, False),
                                  (2.5, 2.5, True), (2.0, 4.0, False)):
            keep = band_keep(ta, v0, a_acc, a_dec=a_dec, allow_reverse=rev)
            row, pw = summarise(err, keep, routes, eid,
                                f"a_acc={a_acc} a_dec={a_dec} rev={rev}")
            row |= {"a_acc": a_acc, "a_dec": a_dec, "allow_reverse": rev}
            row["paired_vs_unclipped_as_trained"] = F.ci_paired(
                pw["route_as_trained"], base_v, eid)
            asym.append(row)

        # positive control -- an ORACLE clip must reach oracle_in_fan
        okeep = torch.zeros_like(err, dtype=torch.bool)
        okeep.scatter_(1, err.argmin(dim=1)[:, None], True)
        octrl, _ = summarise(err, okeep, routes, eid, "ORACLE-CLIP positive ctrl")

        # descriptive census of how absurd the fan actually is
        implied_v = ta / F.T_HORIZON
        R[arm] = {
            "n_anchors": int(fan.shape[1]),
            "unclipped_as_trained_ade": F.r4(base_v.mean()),
            "unclipped_oracle_in_fan": F.r4(err.min(dim=1).values.mean()),
            "absurdity_census": {
                "_read": "implied mean speed = s / 2 s. The comparison speed is "
                         "the MAX OBSERVED v0 over this deployment (an INPUT, "
                         "not held-out error).",
                "max_observed_v0_ms": F.r4(v0.max()),
                "frac_cand_implied_v_over_max_observed_v0": F.r4(
                    (implied_v > v0.max()).float().mean()),
                "frac_cand_implied_v_over_40ms_144kmh": F.r4(
                    (implied_v > 40.0).float().mean()),
                "frac_cand_reverse_s_negative": F.r4((ta < 0).float().mean()),
                "frac_cand_outside_a2.5_band": F.r4(
                    1.0 - band_keep(ta, v0, 2.5).float().mean()),
            },
            "sweep": rows,
            "asymmetric_bands": asym,
            "anti_clip": anti_rows,
            "positive_control_oracle_clip": octrl,
        }
    return R


# =========================================================================== #
# HALF B -- the SURROGATE band on v4's fan, BOTH routes, BOTH scorer WMs
# =========================================================================== #
def half_b() -> dict:
    R = {"_read": "v4's per-candidate trajectories are NOT staged, so the band "
                  "is applied to `C2_wm_ref_proximity` = mean_j ||cand_j - "
                  "wm_ref_j||, the distance to the world model's roll-out under "
                  "the OBSERVED action. Ground-truth-free. A candidate that "
                  "deviates from that reference by a constant relative "
                  "acceleration a has mean offset THETA_PER_A * a, so the SAME "
                  "physical a_max grid maps onto a threshold in metres.",
          "_theta_per_a_m_per_ms2": round(THETA_PER_A, 6),
          "_theta_provenance": A_NAMED,
          "_degeneracy_warning":
              "as theta -> 0 the survivor set collapses to argmin C2, so ANY "
              "route degenerates to arm C2. The `mean_survivors` and "
              "`ctrl_random_on_survivors` columns are what separate a real "
              "route effect from that collapse and must be read with every row."}
    for tag in ("v4", "v1"):
        D = F.load_v5(tag)
        err = D["fan_err4"].float()
        eid = F.eid_of(D["ep"])
        c2 = D["costs"]["C2_wm_ref_proximity"].float()
        rank_of = rank_of_candidate(D["base_rank"]).float()
        routes = {
            "route_as_trained": rank_of,
            "route_imag_consistency": D["costs"]["A1_imag_consistency"].float(),
            # reference rule, NOT a pre-registered route: the committed
            # training-free C2 arm. Reported so the clipped routes can be read
            # against the best realisable rule already known on this fan.
            "ref_route_C2_wm_proximity": c2,
        }
        base_v = F.ade_of_pick(err, D["picks"]["A0_as_trained"])
        # SELF-TEST: the unclipped as-trained route must BE the committed pick
        _all = torch.ones_like(err, dtype=torch.bool)
        assert torch.equal(F.argmin_masked(rank_of, _all).long(),
                           D["picks"]["A0_as_trained"].long()), \
            "as-trained route does not reproduce the committed A0 pick"

        rows, anti_rows = [], []
        for a in A_GRID:
            theta = float("inf") if a == float("inf") else THETA_PER_A * a
            keep = (c2 <= theta) if a != float("inf") else torch.ones_like(
                c2, dtype=torch.bool)
            row, pw = summarise(err, keep, routes, eid, f"a_max={a}")
            row["a_max_ms2"] = None if a == float("inf") else a
            row["theta_m"] = None if a == float("inf") else round(theta, 4)
            for rn in routes:
                row[f"paired_{rn}_vs_unclipped_as_trained"] = F.ci_paired(
                    pw[rn], base_v, eid)
                row[f"single_ci_{rn}"] = F.ci_single(pw[rn], eid)
            rows.append(row)
            if a != float("inf"):
                anti, _ = summarise(err, ~keep, routes, eid, f"ANTI a_max={a}")
                anti["a_max_ms2"] = a
                anti_rows.append(anti)

        # secondary GT-free clip variable: the IMAGINED kinematic magnitude
        a3 = D["costs"]["A3_imag_kinematic"].float()
        a3_rows = []
        for q in (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00):
            if q >= 1.0:
                keep = torch.ones_like(a3, dtype=torch.bool)
            else:
                thr = a3.quantile(q, dim=1, keepdim=True)
                keep = a3 <= thr
            row, pw = summarise(err, keep, routes, eid, f"A3 keep-q={q}")
            row["keep_quantile"] = q
            a3_rows.append(row)

        # breadth clip on the model's own prior (extends V5 S5.2 to both routes)
        rank_rows = []
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            keep = rank_of < n
            row, pw = summarise(err, keep, routes, eid, f"top-{n} by base_rank")
            row["n_kept"] = n
            rank_rows.append(row)

        okeep = torch.zeros_like(err, dtype=torch.bool)
        okeep.scatter_(1, err.argmin(dim=1)[:, None], True)
        octrl, _ = summarise(err, okeep, routes, eid, "ORACLE-CLIP positive ctrl")

        R[f"scorer_{tag}"] = {
            "unclipped_as_trained_ade": F.r4(base_v.mean()),
            "unclipped_imag_consistency_ade": F.r4(
                F.ade_of_pick(err, D["picks"]["A1_imag_consistency"]).mean()),
            "unclipped_oracle_in_fan": F.r4(err.min(dim=1).values.mean()),
            "sweep_C2_band": rows,
            "anti_clip_C2_band": anti_rows,
            "sweep_A3_imagined_accel_quantile": a3_rows,
            "sweep_base_rank_breadth": rank_rows,
            "positive_control_oracle_clip": octrl,
        }
    return R


# =========================================================================== #
# HALF C -- the phi-free BOUND: how good is a rule's r-th choice?
# =========================================================================== #
def half_c() -> dict:
    """Any clip that only REMOVES candidates leaves the rule picking its r-th
    choice for some per-window r. This profiles E[ade | rank = r] for both
    routes, so the question "could removing the implausible tail rescue A1?"
    gets an answer that does not depend on the missing trajectories."""
    R = {"_read": "E[ade_0_2s | the rule's r-th ranked candidate]. If NO r is "
                  "competitive, no tail-removal clip can rescue that rule."}
    for tag in ("v4", "v1"):
        D = F.load_v5(tag)
        err = D["fan_err4"].float()
        eid = F.eid_of(D["ep"])
        out = {}
        for name, cost in (("A1_imag_consistency",
                            D["costs"]["A1_imag_consistency"].float()),
                           ("as_trained_base_rank",
                            rank_of_candidate(D["base_rank"]).float())):
            order = cost.argsort(dim=1)                    # [W, N] best -> worst
            prof = err.gather(1, order)                    # [W, N] by rank
            means = prof.mean(dim=0)
            best_r = int(means.argmin())
            out[name] = {
                "ade_at_rank": {f"r{r+1}": F.r4(means[r])
                                for r in (0, 1, 2, 3, 4, 7, 15, 31, 63, 127, 255)},
                "best_rank_1based": best_r + 1,
                "ade_at_best_rank": F.r4(means[best_r]),
                "ci_at_best_rank": F.ci_single(prof[:, best_r].numpy(), eid),
                "n_ranks_below_0.4907": int((means < 0.4907).sum()),
                "n_ranks_below_0.4271": int((means < 0.4271).sum()),
            }
        R[f"scorer_{tag}"] = out
    return R


def main() -> None:
    t0 = time.time()
    R = {"_experiment": "E-CLIP -- longitudinal-band clip on the frozen fan "
                        "(LOCAL, dev box, no GPU needed, no pod)",
         "_evidence_class": "MEASURED (ours, dev box)",
         "_estimator": "episode_cluster_bootstrap / paired_episode_cluster_"
                       "bootstrap (taniteval/ci.py), B=2000, unit = episode "
                       "cluster, 40 clusters / 881 windows. "
                       "overlapping_holdout_se NEVER used.",
         "_bars": BARS,
         "_a_grid": [None if a == float("inf") else a for a in A_GRID]}
    R["HALF_A_exact_band_refc_fans"] = half_a()
    R["HALF_B_surrogate_band_v4_fan"] = half_b()
    R["HALF_C_rank_profile_bound"] = half_c()
    R["_wall_s"] = round(time.time() - t0, 2)
    (OUT / "fc_clip.json").write_text(json.dumps(R, indent=2))
    print("wrote", OUT / "fc_clip.json", "in", R["_wall_s"], "s")


if __name__ == "__main__":
    main()
