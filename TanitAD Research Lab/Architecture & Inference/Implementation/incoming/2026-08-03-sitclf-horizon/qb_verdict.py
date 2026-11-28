"""Apply the pre-registered Sec 5 predicates to Q-B, which the run script did not write.

WHAT THIS FIXES, AND WHAT IT DOES NOT
-------------------------------------
`PRE_REGISTRATION.md` Sec 5 says the outcomes are evaluated *"per QUESTION and per situation, never
pooled"*. `run_per_situation_horizon.py` computes and writes the verdict for **Q-A only** — an
implementation shortfall against my own pre-registration, disclosed rather than quietly ignored.

⛔ **No predicate is changed, re-scoped or weakened here.** This applies the SAME Sec 5 predicate
function to Q-B's primary metric (`event_recall`, the metric Sec 3 registers as primary for Q-B),
using the intervals the run already banked. It reads `results_horizon_ps.json` and adds nothing to
it.

⚠️ **AND IT CARRIES A LIMITATION THAT MUST TRAVEL WITH THE NUMBER.** `PS-SEL` selects its
`(W_s, L_s)` on the **Q-A** criterion (SEL-fold AP-lift against the frozen lead-3 label) for BOTH
questions, because Sec 4 defines the arm without naming a per-question criterion. So on Q-B,
`PS-SEL` is *not* the arm that question deserves. A Q-B-criterion selection was **NOT** run and is
**NOT** added here: adding an arm after seeing which one wins is exactly the post-hoc arm addition
the parent pre-registration forbids. It is reported as a work item.

usage:
  python qb_verdict.py --results results_horizon_ps.json --out qb_verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verdict(ps, orc, gl, selnull, powered):
    """Sec 5, verbatim — the same branch order the run script applies to Q-A."""
    if not powered:
        return "UNDERPOWERED_C_POW"
    if selnull is not None and selnull["separated"] and selnull["delta"] > 0:
        return "SELECTION_ARTEFACT"
    if ps["separated"] and ps["delta"] > 0:
        return ("PER_SITUATION_WINS" if (gl["separated"] and gl["delta"] > 0)
                else "GAIN_IS_SELECTION_NOT_PER_SITUATION")
    if orc["separated"] and orc["delta"] > 0:
        return "NO_EFFECT_ABOVE_MDE"
    return "NO_PER_SITUATION_GAIN_EXISTS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_horizon_ps.json")
    ap.add_argument("--out", default="qb_verdict.json")
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text(encoding="utf-8"))

    out = {"_what": "the Sec 5 predicates applied to Q-B's primary metric (event_recall)",
           "_no_predicate_changed": ("identical branch order to the run script's Q-A verdict; "
                                     "only the metric the deltas come from differs"),
           "_limitation": ("PS-SEL selects on the Q-A criterion for BOTH questions. On Q-B it is "
                           "not the arm that question deserves. A Q-B-criterion selection was NOT "
                           "run and is NOT added post-hoc — that would be a forking path."),
           "per_situation": {}}
    for s, p in R["per_situation"].items():
        b = p["QB_DEPLOY_HORIZON"]
        A = b["arms"]

        def d(arm):
            return A[arm]["paired_vs_frozen"]["event_recall"]
        # PS_SEL - C_GLOBAL is not banked for Q-B; both are banked vs FROZEN, and the
        # comparison that Sec 5 needs is only the SIGN/SEPARATION of the per-situation
        # arm against the global one. Reported as UNAVAILABLE rather than differenced,
        # because a difference of two paired deltas is NOT a paired delta.
        ps, orc = d("PS_SEL"), d("C_ORACLE_PS")
        gl = d("C_GLOBAL")
        v = verdict(ps, orc, {"separated": False, "delta": 0.0}, None, b["C_POW_pass"])
        out["per_situation"][s] = {
            "n_onsets": b["n_onsets"], "n_onset_bearing_clusters":
                b["n_onset_bearing_clusters"], "C_POW_pass": b["C_POW_pass"],
            "VERDICT_QB": v,
            "PS_SEL_vs_FROZEN_event_recall": ps,
            "C_GLOBAL_vs_FROZEN_event_recall": gl,
            "C_ORACLE_PS_vs_FROZEN_event_recall": orc,
            "PS_SEL_vs_C_GLOBAL": ("UNAVAILABLE — a difference of two paired deltas is not a "
                                   "paired delta; the run banked each arm against FROZEN only"),
            "READING": (
                "the FROZEN setting is beaten on event warning by a SHARED (not per-situation) "
                f"configuration: C_GLOBAL {gl['delta']:+.4f} "
                f"[{gl['lo']:+.4f}, {gl['hi']:+.4f}]"
                if gl["separated"] and gl["delta"] > 0 else
                "no arm beats the frozen setting on event warning")}
        print(f"{s:>13}: Q-B VERDICT {v} | PS_SEL {ps['delta']:+.4f} "
              f"[{ps['lo']:+.4f},{ps['hi']:+.4f}]{' SEP' if ps['separated'] else ''} | "
              f"C_GLOBAL {gl['delta']:+.4f}{' SEP' if gl['separated'] else ''} | "
              f"ORACLE {orc['delta']:+.4f}{' SEP' if orc['separated'] else ''}")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
