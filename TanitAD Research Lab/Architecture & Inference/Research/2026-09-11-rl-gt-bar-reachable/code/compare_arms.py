#!/usr/bin/env python3
"""Compare the >=GT arm against its reproduction control, field by field.

⛔ THE CONTROL IS THE POINT, NOT THE ARM. `p1-grpo-2k-repro` re-runs the banked
`p1-grpo-2k` with the SAME seed, steps, batch and reward and the bar OFF. If it
does not reproduce the banked numbers exactly, the "one key differs" claim about
the ON arm is worthless — a change elsewhere in the tree could account for the
whole difference. So this script reports the CONTROL's agreement first and marks
the comparison INADMISSIBLE when it fails.

⛔ TIER: T0, training-side. These are training-side readouts on a NON-PARITY
corpus, one seed, no paired episode-cluster bootstrap. Nothing here is a
capability claim; that requires T1 (`EVAL_DOCTRINE.md`).
"""

from __future__ import annotations

import argparse
import json
import os

FIELDS = ("done", "steps", "final_loss", "veto_rate_mean", "frac_above_bar_mean",
          "use_gt_bar", "noise_mode", "w_anchor", "train_mode_forward")
READOUTS = ("R1_fan_reward", "R2_fan_collision_rate", "R3_sel_ade_m")


def load(d: str) -> dict:
    out = {}
    for name in ("pilot_summary", "readout_before", "readout_after"):
        p = os.path.join(d, name + ".json")
        # ⛔ ASSERT ON THE ARTIFACT. A missing file is the evidence that a stage
        # did not finish — never an exit code that a shell can overwrite.
        out[name] = (json.load(open(p, encoding="utf-8"))
                     if os.path.isfile(p) else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banked", required=True, help="the 2026-09-10 p1-grpo-2k dir")
    ap.add_argument("--repro", required=True, help="bar OFF, same seed — the CONTROL")
    ap.add_argument("--gtbar", required=True, help="bar ON — the arm")
    ap.add_argument("--reg", default=None, help="the hackable-reward regression arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = {"banked": load(a.banked), "repro": load(a.repro), "gtbar": load(a.gtbar)}
    if a.reg:
        arms["reg"] = load(a.reg)

    rec: dict = {
        "_what": "P-RC21 >=GT truncation arm vs its bar-OFF reproduction control",
        "_tier": ("T0 training-side; NON-PARITY corpus; ONE SEED; no paired "
                  "episode-cluster bootstrap. NOT a capability claim."),
        "_evidence_class": "MEASURED (ours)",
        "missing_artifacts": [k for k, v in arms.items()
                              if v["pilot_summary"] is None],
    }

    def summ(k):
        s = arms[k]["pilot_summary"]
        return None if s is None else {f: s.get(f) for f in FIELDS}

    rec["summary_fields"] = {k: summ(k) for k in arms}

    def readouts(k):
        out = {}
        for phase in ("readout_before", "readout_after"):
            r = arms[k][phase]
            out[phase] = (None if r is None else
                          {m: r[m]["mean"] for m in READOUTS if m in r})
        s = arms[k]["pilot_summary"]
        out["pilot_delta"] = None if s is None else s.get("pilot_delta")
        return out

    rec["readouts"] = {k: readouts(k) for k in arms}

    # --- THE CONTROL, FIRST ------------------------------------------------- #
    b, r = arms["banked"]["pilot_summary"], arms["repro"]["pilot_summary"]
    ctrl = {}
    if b and r:
        for f in ("final_loss", "veto_rate_mean", "steps"):
            ctrl[f] = {"banked": b.get(f), "repro": r.get(f),
                       "identical": b.get(f) == r.get(f)}
        for m in READOUTS:
            for phase in ("readout_before", "readout_after"):
                bb, rr = arms["banked"][phase], arms["repro"][phase]
                if bb and rr:
                    ctrl[f"{phase}.{m}"] = {
                        "banked": bb[m]["mean"], "repro": rr[m]["mean"],
                        "identical": bb[m]["mean"] == rr[m]["mean"]}
    rec["reproduction_control"] = ctrl
    rec["CONTROL_REPRODUCES_EXACTLY"] = bool(ctrl) and all(
        v["identical"] for v in ctrl.values())

    # --- then, and only then, the pair -------------------------------------- #
    off, on = arms["repro"]["pilot_summary"], arms["gtbar"]["pilot_summary"]
    pair = {}
    if off and on:
        pair["differs_by"] = sorted(
            f for f in FIELDS if off.get(f) != on.get(f))
        for m in READOUTS:
            a_off, a_on = arms["repro"]["readout_after"], arms["gtbar"]["readout_after"]
            b_off, b_on = arms["repro"]["readout_before"], arms["gtbar"]["readout_before"]
            if a_off and a_on:
                pair[m] = {"bar_off_after": a_off[m]["mean"],
                           "bar_on_after": a_on[m]["mean"],
                           "delta_on_minus_off": a_on[m]["mean"] - a_off[m]["mean"],
                           "before_identical": (b_off and b_on and
                                                b_off[m]["mean"] == b_on[m]["mean"])}
        pair["frac_above_bar_mean_ON"] = on.get("frac_above_bar_mean")
        pair["THE_MASK_WAS_REACHED"] = on.get("frac_above_bar_mean") is not None
    rec["pair"] = pair
    rec["_admissible"] = (rec["CONTROL_REPRODUCES_EXACTLY"]
                          and pair.get("THE_MASK_WAS_REACHED") is True)
    rec["_admissibility_note"] = (
        "_admissible == True means only that the PAIR IS CLEAN (the control "
        "reproduced and the mask was reached). It does NOT make the deltas a "
        "result: one seed, no paired bootstrap, and the reward audit's own "
        "verdict still gates whether any P1 number may be quoted.")

    for k in ("banked", "repro", "gtbar", "reg"):
        s = arms.get(k, {}).get("pilot_summary") if k in arms else None
        if s:
            rec.setdefault("reward_audit", {})[k] = {
                "empty_ctx": s.get("reward_audit", {}).get("verdict"),
                "empty_ctx_dead": s.get("reward_audit", {}).get("dead_components"),
                "empty_ctx_tied": s.get("reward_audit", {}).get("tied_at_max"),
                "exercising": (s.get("reward_audit_exercising") or {}).get("verdict"),
                "exercising_dead": (s.get("reward_audit_exercising")
                                    or {}).get("dead_components"),
                "exercising_tied": (s.get("reward_audit_exercising")
                                    or {}).get("tied_at_max"),
            }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in
                      ("missing_artifacts", "CONTROL_REPRODUCES_EXACTLY", "pair",
                       "reward_audit", "_admissible") if k in rec}, indent=1))
    print(f"ZZCOMPARE-WRITTEN-{a.out}ZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
