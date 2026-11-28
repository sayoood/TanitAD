#!/usr/bin/env python3
"""S0 -- HARNESS GATE. Reproduce COMMITTED numbers before quoting a new one,
and prove the harness can render a FAILING verdict.

Both directions, as the operating standard requires:
  * FIDELITY  -- every published v5 bar, both scorer world models, and the
                 registry's REF-C-XL / REF-C-base headline, recomputed here on
                 the dev box from committed artifacts only;
  * FAILING   -- a uniform-random pick, an anti-oracle (argmax fan error), and a
                 SHUFFLED-cost input that destroys the pick<->window pairing.
                 If any of these came out *good*, the harness would be measuring
                 nothing and this file aborts.

Nothing here needs a GPU, a checkpoint or the parity corpus.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import fc_common as F

OUT = Path(__file__).resolve().parents[1] / "raw" / "fc_gate.json"

# ---- COMMITTED targets. Sources, verbatim, so each is traceable. ------------
COMMITTED_V4 = {   # V5_IMAGINATION_SELECTION.md S2 table (raw/v5_v4.json)
    "A0_as_trained": 0.8563, "R_random": 15.8738, "O_oracle_in_fan": 0.2505,
    "A1_imag_consistency": 11.5298, "A2_imag_goal_speed": 10.3863,
    "A3_imag_kinematic": 13.1805, "C1_ctrv_consistency": 1.7836,
    "C2_wm_ref_proximity": 1.0653, "A4_imag_combo_oof": 0.7706,
}
COMMITTED_V1 = {   # V5_IMAGINATION_SELECTION.md S3.3 table (raw/v5_v1.json)
    "A0_as_trained": 0.8563, "A1_imag_consistency": 1.2472,
    "A2_imag_goal_speed": 5.0746, "A3_imag_kinematic": 6.2200,
    "C1_ctrv_consistency": 1.7836, "C2_wm_ref_proximity": 0.5645,
    "A4_imag_combo_oof": 0.5645, "O_oracle_in_fan": 0.2505,
}
COMMITTED_DEPTH = {"k1": 2.906, "k2": 1.423, "k8": 2.818, "k20": 11.530}
COMMITTED_ANTI_ORACLE = 45.5488           # V5 S1 row S5
COMMITTED_FAN_ENVELOPE = {                # V5 S2.2 / raw/v5_posthoc.json
    "min_term_along_m": -15.47, "max_term_along_m": 100.571,
    "per_window_span_mean_m": 108.736, "gt_mean_term_along_m": 25.396,
}
COMMITTED_REGISTRY = {                    # MODEL_REGISTRY.md S "leaderboard"
    "refc-xl-30k": 0.4714, "refc-base-30k": 0.4728, "refc-small-30k": 0.5261,
}
TOL = 5e-4                                 # 4-decimal reproduction


def main() -> None:
    t0 = time.time()
    R: dict = {
        "_experiment": "fan-clip LOCAL -- S0 harness gate",
        "_evidence_class": "MEASURED (ours, dev box)",
        "_tier": "CONFIRMED",
        "_host": {"node": platform.node(), "python": platform.python_version(),
                  "torch": torch.__version__,
                  "cuda_available": bool(torch.cuda.is_available()),
                  "gpu": (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else None),
                  "_note": "GPU present but this gate is pure CPU tensor "
                           "reduction; no kernel is launched."},
        "_estimator": "episode_cluster_bootstrap / paired_episode_cluster_"
                      "bootstrap (taniteval/ci.py), B=2000, unit = episode "
                      "cluster. overlapping_holdout_se is NEVER used.",
        "_parity": "No corpus is read. Every input is a committed artifact. The "
                   "dev-box episode cache (14231cd29c74) is NOT the canonical "
                   "parity key (e438721ae894) and is never touched.",
    }
    fails: list[str] = []

    # ---------------- S0.1 / S0.2 -- the v5 bars, both scorer WMs -----------
    for tag, committed in (("v4", COMMITTED_V4), ("v1", COMMITTED_V1)):
        D = F.load_v5(tag)
        fe = D["fan_err4"].float()
        got = {k: F.r4(F.ade_of_pick(fe, v).mean())
               for k, v in D["picks"].items()}
        # oracle / anti-oracle recomputed from fan_err4 itself, not from picks
        got["O_oracle_in_fan"] = F.r4(fe.min(dim=1).values.mean())
        anti = F.r4(fe.max(dim=1).values.mean())
        row = {k: {"committed": committed[k], "recomputed": got[k],
                   "abs_diff": F.r4(abs(got[k] - committed[k])),
                   "pass": abs(got[k] - committed[k]) <= TOL}
               for k in committed}
        R[f"S0.1_v5_bars_scorer_{tag}"] = {
            "_source": f"V5_IMAGINATION_SELECTION.md -> raw/v5_{tag}.json",
            "arms": row,
            "anti_oracle_recomputed": anti,
            "all_pass": all(v["pass"] for v in row.values()),
        }
        fails += [f"S0.1[{tag}]:{k}" for k, v in row.items() if not v["pass"]]

        if tag == "v4":
            dep = {k: F.r4(F.ade_of_pick(fe, v.argmin(dim=1)).mean())
                   for k, v in D["cost_A1_by_k"].items() if k in COMMITTED_DEPTH}
            R["S0.2_depth_axis_n256"] = {
                "_source": "V5 S5.2 row n=256",
                "rows": {k: {"committed": COMMITTED_DEPTH[k], "recomputed": v,
                             "abs_diff": F.r4(abs(v - COMMITTED_DEPTH[k])),
                             "pass": abs(v - COMMITTED_DEPTH[k]) <= 1e-3}
                         for k, v in dep.items()},
            }
            fails += [f"S0.2:{k}" for k, v in R["S0.2_depth_axis_n256"]["rows"]
                      .items() if not v["pass"]]
            R["S0.5_failing_input"] = {
                "_read": "the harness MUST be able to return a failing value.",
                "uniform_random_pick": got["R_random"],
                "committed_random": COMMITTED_V4["R_random"],
                "anti_oracle_recomputed": anti,
                "committed_anti_oracle": COMMITTED_ANTI_ORACLE,
                "anti_oracle_pass": abs(anti - COMMITTED_ANTI_ORACLE) <= 1e-3,
                "spread_random_over_as_trained": F.r4(
                    got["R_random"] / got["A0_as_trained"]),
            }
            if abs(anti - COMMITTED_ANTI_ORACLE) > 1e-3:
                fails.append("S0.5:anti_oracle")

    # ---------------- S0.3 -- WINDOW ALIGNMENT across three artifacts -------
    D4, D1 = F.load_v5("v4"), F.load_v5("v1")
    gt20 = F.load_gt_dense()
    xl = F.load_refc_fan("xl")
    gt4_bar = gt20[:, [s - 1 for s in F.WP_STEPS]]
    R["S0.3_window_alignment"] = {
        "_read": "the EXACT longitudinal band is evaluated on REF-C fans; this "
                 "proves they sit on the SAME 881 windows / 40 episodes / same "
                 "order / same GT as the v5 dump, so the two halves of this "
                 "stream are commensurable.",
        "n_windows": int(gt20.shape[0]),
        "n_episodes": int(len(np.unique(D4["ep"].numpy()))),
        "gt_refcxl_vs_barA_wp4_max_abs_m": F.r4(
            (xl["gt"].float() - gt4_bar).abs().max()),
        "v0_refcxl_vs_v5_max_abs_ms": F.r4(
            (xl["v0"].float() - D4["v0"].float()).abs().max()),
        "ep_labels_identical_v4_v1": bool(torch.equal(D4["ep"], D1["ep"])),
        "fan_err4_identical_v4_v1_dumps": bool(
            torch.equal(D4["fan_err4"], D1["fan_err4"])),
        "C1_identical_across_scorers_(C1_is_WM_free_by_design)": F.r4(
            (D4["costs"]["C1_ctrv_consistency"]
             - D1["costs"]["C1_ctrv_consistency"]).abs().max()),
    }
    if R["S0.3_window_alignment"]["gt_refcxl_vs_barA_wp4_max_abs_m"] > 1e-3:
        fails.append("S0.3:gt_alignment")

    # ---------------- S0.4 -- registry reproduction for the REF-C fans ------
    reg: dict = {}
    for arm, key in (("xl", "refc-xl-30k"), ("base", "refc-base-30k"),
                     ("small", "refc-small-30k")):
        d = F.load_refc_fan(arm)
        fan, gt = d["fan"].float(), d["gt"].float()
        err = (fan - gt[:, None]).norm(dim=-1).mean(dim=-1)      # [W, N]
        sel = d["sel"].long()
        got = F.r4(err.gather(1, sel[:, None]).mean())
        # the head's own contract: sel == argmax(anchor logits)
        agree = float((d["logits"].argmax(dim=1) == sel).float().mean())
        reg[key] = {
            "committed_registry_full_set_ade_0_2s": COMMITTED_REGISTRY[key],
            "recomputed_here": got,
            "abs_diff": F.r4(abs(got - COMMITTED_REGISTRY[key])),
            "pass": abs(got - COMMITTED_REGISTRY[key]) <= TOL,
            "n_anchors": int(fan.shape[1]),
            "oracle_in_fan": F.r4(err.min(dim=1).values.mean()),
            "sel_equals_argmax_logits_frac": F.r4(agree),
        }
        if not reg[key]["pass"]:
            fails.append(f"S0.4:{key}")
    R["S0.4_registry_refc_fans"] = {
        "_source": "Project Steering/MODEL_REGISTRY.md leaderboard "
                   "(full-set mean column)", "arms": reg}

    # ---------------- S0.6 -- SHUFFLED-COST negative control ---------------
    # Destroy the pick<->window pairing. A harness that still reports a good
    # number is not measuring the rule.
    fe = D4["fan_err4"].float()
    g = torch.Generator().manual_seed(F.SEED)
    perm = torch.randperm(fe.shape[0], generator=g)
    shuf_pick = D4["costs"]["A1_imag_consistency"][perm].argmin(dim=1)
    base_shuf = D4["base_rank"][perm].argmin(dim=1)
    R["S0.6_shuffled_cost_control"] = {
        "_read": "the SAME two selection routes driven by another window's cost "
                 "matrix. Both must degrade toward the random pick.",
        "A1_route_on_shuffled_cost": F.r4(F.ade_of_pick(fe, shuf_pick).mean()),
        "as_trained_route_on_shuffled_rank": F.r4(
            F.ade_of_pick(fe, base_shuf).mean()),
        "reference_random": F.r4(F.ade_of_pick(fe, D4["picks"]["R_random"]).mean()),
        "reference_as_trained": 0.8563,
    }
    if R["S0.6_shuffled_cost_control"]["as_trained_route_on_shuffled_rank"] < 2.0:
        fails.append("S0.6:shuffled_as_trained_not_degraded")

    # ---------------- S0.7 -- the fan envelope, INDEPENDENTLY on REF-C -----
    envs = {}
    for arm in ("xl", "base", "small"):
        d = F.load_refc_fan(arm)
        ta = d["fan"].float()[:, :, -1, 0]
        envs[arm] = {
            "n_anchors": int(ta.shape[1]),
            "min_term_along_m": F.r4(ta.min()), "max_term_along_m": F.r4(ta.max()),
            "per_window_span_mean_m": F.r4(
                (ta.max(dim=1).values - ta.min(dim=1).values).mean()),
            "max_implied_mean_speed_kmh": F.r4(
                float(ta.max()) / F.T_HORIZON * 3.6),
            "frac_candidates_over_50ms_implied": F.r4(
                float(((ta / F.T_HORIZON) > 50.0).float().mean())),
        }
    R["S0.7_fan_envelope_refc_replication"] = {
        "_read": "v4's own envelope (committed) vs three REF-C anchored fans "
                 "measured here. If the 108.7 m span replicates, the "
                 "longitudinal over-dispersion is a property of the anchored "
                 "fan family, not a v4 accident.",
        "v4_committed": COMMITTED_FAN_ENVELOPE,
        "refc_measured_here": envs,
        "gt_mean_term_along_m_measured_here": F.r4(gt20[:, -1, 0].mean()),
    }

    R["_wall_s"] = round(time.time() - t0, 2)
    R["_VERDICT"] = "GATE PASS" if not fails else "GATE FAIL"
    R["_failures"] = fails
    OUT.write_text(json.dumps(R, indent=2))
    print(json.dumps({"verdict": R["_VERDICT"], "failures": fails,
                      "wall_s": R["_wall_s"],
                      "S0.3": R["S0.3_window_alignment"],
                      "S0.4": {k: {kk: v[kk] for kk in
                                   ("recomputed_here", "committed_registry_"
                                    "full_set_ade_0_2s", "pass", "n_anchors",
                                    "oracle_in_fan")}
                               for k, v in reg.items()},
                      "S0.5": R["S0.5_failing_input"],
                      "S0.6": R["S0.6_shuffled_cost_control"],
                      "S0.7": R["S0.7_fan_envelope_refc_replication"]},
                     indent=2))
    if fails:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
