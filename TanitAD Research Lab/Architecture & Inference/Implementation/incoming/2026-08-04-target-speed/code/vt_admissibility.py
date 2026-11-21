#!/usr/bin/env python3
"""D-VT1 step 5 — run the PI's admissibility check as a COMPUTATION, on the real
signals, and emit the verdict a gate can act on.

`stack/tanitad/eval/goal_admissibility.py` is the instrument (12 tests, including
a reproduction of the nav echo). This script applies it to the three candidate
target-speed signals and writes `raw/vt_admissibility.json`.

⛔ The verdict for a signal depends on WHICH SIDE it sits on, and the instrument
refuses to guess: `supplied_at_inference` has no default. The same evidence about
`vt_oracle` therefore yields INADMISSIBLE when supplied and ADMISSIBLE as a label
— which is exactly the PI's 2026-08-03 ruling, expressed as code rather than as a
paragraph somebody has to remember.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.goal_admissibility import (audit_goal_signal,      # noqa: E402
                                             echo_score,
                                             horizon_disjoint,
                                             incremental_information,
                                             situation_disjoint)
from tanitad.lake.vocab import vtarget_band                           # noqa: E402
from tanitad.lake.vtarget import VT_GUARD_STEPS, VT_LOOK_HI           # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from vt_leak_audit import past_block                                  # noqa: E402

#: the situation classifier's outputs, by symbol. The PI's second clause forbids
#: ALL of these inside a goal path at inference — posterior, argmax, embedding, or
#: any feature derived from them.
SIT_OUTPUTS = ("sit_posterior", "sit_argmax", "sit_embedding", "sit_logits")


def main(labels_json: Path, out_json: Path):
    d = json.load(open(labels_json, encoding="utf-8"))
    rows = [r for r in d["rows"]
            if r["vt_guarded_valid"] and r["vt_oracle_valid"]]
    eid = np.array([r["eid"] for r in rows])
    past = past_block(rows)
    y = np.array([r["dv_2s"] for r in rows])
    ell = 7                                   # a representative window origin

    h_oracle = horizon_disjoint(ell + 1, ell + VT_LOOK_HI, ell,
                                ell + VT_GUARD_STEPS + 1)
    h_guard = horizon_disjoint(ell + VT_GUARD_STEPS + 1, ell + VT_LOOK_HI, ell,
                               ell + VT_GUARD_STEPS + 1)
    inc_o = incremental_information(
        y, past, np.array([[r["vt_oracle"]] for r in rows]), eid)
    inc_g = incremental_information(
        y, past, np.array([[r["vt_guarded"]] for r in rows]), eid)
    prov_ok = situation_disjoint(["pooled", "v0"], SIT_OUTPUTS)
    prov_bad = situation_disjoint(["pooled", "v0", "sit_posterior"], SIT_OUTPUTS)

    band_g = np.array([r["band_guarded"] for r in rows])
    band_v0 = np.array([vtarget_band(r["v0"]) for r in rows])
    echo_self = echo_score(band_g, band_g)
    echo_v0 = echo_score(band_v0, band_g)

    out = {
        "_what": ("the PI's admissibility check, run as a computation on the "
                  "real val40 signals"),
        "_instrument": "stack/tanitad/eval/goal_admissibility.py",
        "_ruling": ("Sayed 2026-08-03: labels may use ego / other agents / maps "
                    "/ future poses; INFERENCE is the constrained side. And a "
                    "goal input must not carry the situation classifier's "
                    "output in any form."),
        "n_windows": len(rows), "n_episodes": int(len(set(eid.tolist()))),
        "echo_demonstrations": {
            "supplied_band_scored_on_band": {
                **{k: v for k, v in echo_self.items() if k != "mapping"},
                "_why_this_is_here": (
                    "the trap, on the real 637 tokens: an arm FED the target-"
                    "speed band and SCORED on target-speed band accuracy earns "
                    "1.000000 by construction. That is the flagship-v1 route "
                    "head's 1.0000 exactly. ⛔ Never score a head on a quantity "
                    "it is fed."),
            },
            "current_speed_band_predicts_target_band": {
                **{k: v for k, v in echo_v0.items() if k != "mapping"},
                "_why_this_is_here": (
                    "the FREE baseline in echo terms: how much of the target-"
                    "speed band is already determined by the CURRENT speed band, "
                    "which the model holds anyway. A goal head must beat this or "
                    "it is a dead parameter."),
            },
        },
        "verdicts": {
            "vt_oracle_SUPPLIED": audit_goal_signal(
                name="vt_oracle (supplied at inference)",
                supplied_at_inference=True, horizon=h_oracle, increment=inc_o,
                provenance=prov_ok),
            "vt_oracle_AS_LABEL": audit_goal_signal(
                name="vt_oracle (offline label)", supplied_at_inference=False,
                horizon=h_oracle, increment=inc_o, provenance=prov_ok),
            "vt_guarded_SUPPLIED": audit_goal_signal(
                name="vt_guarded (supplied at inference)",
                supplied_at_inference=True, horizon=h_guard, increment=inc_g,
                provenance=prov_ok),
            "vt_guarded_AS_LABEL": audit_goal_signal(
                name="vt_guarded (offline label)", supplied_at_inference=False,
                horizon=h_guard, increment=inc_g, provenance=prov_ok),
            "vt_predicted_from_image_and_v0": audit_goal_signal(
                name="v_hat_target = f(pooled, v0), trained on vt_guarded",
                supplied_at_inference=False, horizon=h_guard,
                increment=None, provenance=prov_ok),
            "COUNTEREXAMPLE_goal_carrying_the_situation_classifier": (
                audit_goal_signal(
                    name="v_hat_target = f(pooled, v0, sit_posterior)",
                    supplied_at_inference=False, horizon=h_guard,
                    increment=None, provenance=prov_bad)),
        },
        "_shared_trunk_note": (
            "`pooled` feeds the goal head AND, in any fused arm, the situation "
            "head. That is a shared ENCODER, not a shared signal: attributability "
            "comes from the situation classifier's OUTPUT being absent from the "
            "goal path's graph, which is what `situation_disjoint` records. The "
            "same argument `RefCModel._goal_provenance` already makes for S6. ⚠️ "
            "It is an argument about the graph, not a measurement — if a future "
            "arm trains the two heads jointly with a shared loss, re-examine it."),
        "_what_this_check_CANNOT_do": [
            "detect a statistical leak that is not functional (echo_score)",
            "detect autocorrelation across a disjoint boundary (horizon_disjoint) "
            "— which is why incremental_information is not optional",
            "detect laundering through a learned trunk (situation_disjoint takes "
            "DECLARED provenance)",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["verdicts"].items():
        print(f"  {k:52s} {v['verdict']:14s} {v['failures'] or ''}")
    print(f"echo: fed-band scored on band = "
          f"{echo_self['functional_agreement']} (bijection="
          f"{echo_self['bijection']}); v0-band -> target-band = "
          f"{echo_v0['functional_agreement']}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
