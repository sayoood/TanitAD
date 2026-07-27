"""The BEFORE/AFTER envelope table, on the SAME corpus and the SAME function.

Upgrades the sequential loop's out-of-envelope fractions from `INHERITED` to
`MEASURED` by recomputing them here from the **committed** per-window tensors of
the 30 k gate, with the identical ``taniteval.ood.envelope_fractions`` the
pseudo-simulation grid is judged by. Without this the "12.3 % -> 0 %" claim would
be comparing my number against somebody else's, which the operating standard
forbids for a decision-grade claim.

Also reproduces the committed corridor headline numbers (v4 K=185 overall
**0.6388** / junction **0.8432**, REF-C base **0.5833**) as the reproduce-before-
you-quote check.

CPU only. No GPU, no pod, no model, no corpus — the committed dumps are enough.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
sys.path.insert(0, str(_REPO / "taniteval"))

from taniteval import clhorizon as CH                              # noqa: E402
from taniteval import ood as _ood                                  # noqa: E402
from taniteval import pseudosim as PS                              # noqa: E402

_CP = (_REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
       / "incoming" / "2026-07-26-v4-30k-gate" / "coprimary")

DUMPS = [("flagship_v4_30k", 185, "corridor_v4_30k_K185_perwindow_K185.pt"),
         ("flagship_v4_30k", 20, "corridor_v4_30k_K185_perwindow_K20.pt"),
         ("refc_base_30k", 185, "corridor_refcbase_30k_K185_perwindow_K185.pt"),
         ("refc_base_30k", 20, "corridor_refcbase_30k_K185_perwindow_K20.pt")]

# MEASURED, committed: GATE_30K_RESULTS.md 6.3. Named so a drift goes red.
COMMITTED = {("flagship_v4_30k", 185): (0.6388, 0.8432),
             ("refc_base_30k", 185): (0.5833, 0.7027)}


def main():
    out = {
        "_what": "BEFORE (sequential rollout) vs AFTER (pseudo-simulation grid) "
                 "out-of-envelope fractions, same corpus, same function",
        "_evidence_class": "MEASURED (ours; recomputed from the COMMITTED "
                           "per-window tensors of the 30 k gate)",
        "_function": "taniteval.ood.envelope_fractions",
        "_envelope": {"lat_max_m": _ood.ENV_LAT_MAX,
                      "yaw_max_deg": _ood.ENV_YAW_MAX},
        "sequential_rollout": [], "pseudo_simulation": None,
        "reproduction_check": [],
    }
    for arm, K, f in DUMPS:
        pw = torch.load(_CP / f, weights_only=False)
        fr = _ood.envelope_fractions(pw["lat"].numpy(), pw["yaw"].numpy())
        out["sequential_rollout"].append({
            "arm": arm, "K": K, "horizon_s": round(K * 0.1, 1),
            "n_windows": int(pw["lat"].shape[0]),
            "frac_steps_lat_over_3m": fr["frac_steps_lat_over_3m"],
            "frac_steps_yaw_over_12deg": fr["frac_steps_yaw_over_12deg"],
            "frac_steps_any": fr["frac_steps_any"],
            "frac_windows_any_step_out_of_envelope":
                fr["frac_windows_any_step_out_of_envelope"],
            "verdict_class": _ood.verdict_class(_ood._verdict_string(
                False, fr["frac_windows_any_step_out_of_envelope"],
                fr["frac_steps_any"])),
        })
        if (arm, K) in COMMITTED:
            res = CH.corridor_from_perwindow(_CP / f, K=K)
            ov = round(float(res["overall"]["corridor_departure_rate"]["mean"]), 4)
            ju = round(float(res["junction"]["corridor_departure_rate"]["mean"]), 4)
            exp_ov, exp_ju = COMMITTED[(arm, K)]
            out["reproduction_check"].append({
                "arm": arm, "K": K,
                "corridor_departure_rate_overall": ov, "committed_overall": exp_ov,
                "corridor_departure_rate_junction": ju, "committed_junction": exp_ju,
                "matches": bool(abs(ov - exp_ov) < 5e-5 and abs(ju - exp_ju) < 5e-5),
            })

    grid = PS.default_grid()
    proof = PS.assert_grid_in_envelope(grid)
    out["pseudo_simulation"] = {
        "grid": grid.describe(),
        "n_grid_points": proof["n_grid_points"],
        "frac_steps_lat_over_3m": proof["EXTRAPOLATION_frac_steps_lat_over_3m"],
        "frac_steps_yaw_over_12deg": proof["EXTRAPOLATION_frac_steps_yaw_over_12deg"],
        "frac_steps_any": proof["EXTRAPOLATION_frac_steps_any"],
        "frac_windows_any_step_out_of_envelope":
            proof["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"],
        "verdict_class": _ood.verdict_class(proof["EXTRAPOLATION_VERDICT"]),
        "falsifier": proof["falsifier"],
    }
    assert all(r["matches"] for r in out["reproduction_check"]), \
        "committed corridor numbers did NOT reproduce — refuse to quote anything new"
    p = _HERE.parent / "artifacts" / "before_after_envelope.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[before_after] wrote {p}")


if __name__ == "__main__":
    main()
