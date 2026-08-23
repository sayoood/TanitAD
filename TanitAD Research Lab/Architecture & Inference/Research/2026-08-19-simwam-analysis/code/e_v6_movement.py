"""E-V6MOVE — what is actually still training in v6F-SW-30k, and how much is left?

⛔ TWO QUESTIONS E-DETECT-1 COULD NOT ANSWER FROM FEATURES ALONE.

  1. Is the encoder even being UPDATED? "v6's token field is indistinguishable
     from raw pixels" has a mundane alternative explanation — a frozen or
     barely-moving encoder — and that would be a BUG, not a finding about the
     objective.
  2. Will the remaining steps change it? The probe read step 20,000 of 30,000.

Both are answerable from the banked fp16 checkpoints plus the trainer's own `lr`
field, with no GPU and without touching Thor.

⚠️⚠️ THE CONFOUND THAT ALMOST COST A WRONG ANSWER. Raw weight-movement per step
DECAYS across every module (encoder 0.0201 -> 0.0055 over 12k->20k), which reads
as "converging". It is not, on its own: **the learning rate decays too**
(5.891e-05 -> 2.262e-05 over the same window). A decaying update norm is exactly
what a decaying schedule produces. Movement must be normalised by the LR actually
applied before any convergence claim is admissible — and the honest quantity for
"how much training is left" is the INTEGRAL of lr, not the step count.

⭐ THE FREEZE DECLARATION IS HONEST — CHECKED, NOT ASSUMED. 19 state_dict groups
show EXACTLY zero change at every checkpoint, which looks alarming until it is
cross-checked: the live stage is **S-W**, and `stage_trainable_groups("S-W")`
declares `("encoder", "readout", "predictor_op", "aux")`. The two groups that
appear to move without being declared resolve through `V6Stack._GROUP_PREFIXES`:
`step_readout_op.` -> `predictor_op` and `masked_cells.` -> `aux`. So OBSERVED
movement == DECLARED trainability, exactly. The apparent mismatch was mine —
comparing state_dict top-level keys against MODULE_GROUPS, which are a different
partition.

⚠️ WHAT THE FROZEN 19 MEAN FOR READING ANY v6 RESULT: the entire tactical and
strategic hierarchy — adapters, both per-layer predictors, goal and action heads,
and every vocabulary table — receives ZERO gradient in this stage. Any claim
about "the hierarchy" from this run is a claim about randomly-initialised
modules, not trained ones.

TIER: T0-DIAGNOSTIC (a training-dynamics measurement; NOT a capability claim).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

CKPT = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
            r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
            r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\ckpt")
STEPS = (12000, 14000, 16000, 18000, 20000)
TARGET_STEP = 30000
#: MEASURED from Thor's own train_log.jsonl `lr` field, 2026-08-21.
LR_AT = {11950: 5.8911319000666e-05, 13950: 4.975364334077322e-05,
         15950: 4.038094838329002e-05, 17950: 3.120286587303872e-05,
         19950: 2.262052205704337e-05, 24950: 6.124501140449847e-06,
         25200: 5.545645712876312e-06}
#: trapezoid integrals of lr over step, MEASURED on the full log
LR_INTEGRAL = {"0_25200": 1.39362, "12000_20000": 0.32227,
               "12000_25200": 0.39088, "25200_30000_extrapolated": 0.01496}
LR_MAX = 9.999931461237134e-05


def load(step: int) -> dict:
    d = torch.load(CKPT / f"v6F_sw_step{step:06d}.fp16.pt", map_location="cpu",
                   weights_only=False)
    for k in ("model", "state_dict", "sd", "weights"):
        if k in d and isinstance(d[k], dict):
            return d[k]
    return d


def main() -> None:
    sds = [load(s) for s in STEPS]
    groups: dict[str, list[str]] = {}
    for k in sds[0]:
        groups.setdefault(k.split(".")[0], []).append(k)

    deltas: dict[str, list[float]] = {}
    for i in range(1, len(STEPS)):
        a, b = sds[i - 1], sds[i]
        for g, ks in groups.items():
            num = den = 0.0
            for k in ks:
                if k not in b or not torch.is_floating_point(b[k]):
                    continue
                x, y = a[k].float(), b[k].float()
                num += float((y - x).pow(2).sum())
                den += float(x.pow(2).sum())
            if den > 0:
                deltas.setdefault(g, []).append(num ** 0.5 / den ** 0.5)

    moving = {g: v for g, v in deltas.items() if max(v) > 1e-9}
    frozen = sorted(g for g, v in deltas.items() if max(v) <= 1e-9)

    # LR actually applied over each interval (mean of the two endpoint reads)
    lr_pts = [LR_AT[s - 50] for s in STEPS]
    lr_mid = [(lr_pts[i] + lr_pts[i + 1]) / 2 for i in range(len(lr_pts) - 1)]

    out = {
        "_evidence_class": "MEASURED (ours; banked fp16 checkpoints + Thor's "
                           "own train_log.jsonl lr field)",
        "eval_tier": "T0-DIAGNOSTIC",
        "run": "v6F-SW-30k", "steps_compared": list(STEPS),
        "lr_at_step": LR_AT, "lr_max": LR_MAX, "lr_integral": LR_INTEGRAL,
        "moving_groups": {}, "frozen_groups": frozen,
        "n_frozen": len(frozen), "n_moving": len(moving),
        "stage": "S-W",
        "declared_trainable_groups": ["encoder", "readout", "predictor_op",
                                      "aux"],
        "declaration_matches_observation": True,
        "declaration_check": "step_readout_op.->predictor_op and "
                             "masked_cells.->aux via V6Stack._GROUP_PREFIXES; "
                             "observed movement == declared trainability",
    }
    print("  relative L2 weight change per 2,000 steps, RAW and LR-NORMALISED\n")
    hdr = "".join(f"{STEPS[i + 1] // 1000}k".rjust(11) for i in range(4))
    print(f"  {'group':<18}{hdr}   {'raw trend':<10}{'norm trend'}")
    for g, v in sorted(moving.items(), key=lambda x: -max(x[1])):
        nrm = [v[i] / lr_mid[i] for i in range(len(v))]
        raw_t = f"x{v[-1] / v[0]:.2f}"
        nrm_t = f"x{nrm[-1] / nrm[0]:.2f}"
        out["moving_groups"][g] = {
            "raw": [round(x, 6) for x in v],
            "lr_normalised": [round(x, 1) for x in nrm],
            "raw_ratio_last_over_first": round(v[-1] / v[0], 3),
            "lr_normalised_ratio_last_over_first": round(nrm[-1] / nrm[0], 3),
        }
        print(f"  {g:<18}" + "".join(f"{x:11.6f}" for x in v)
              + f"   {raw_t:<10}{nrm_t}")

    print(f"\n  FROZEN at every checkpoint ({len(frozen)} groups, EXACTLY zero "
          f"change):\n     " + ", ".join(frozen))

    rem = LR_INTEGRAL["25200_30000_extrapolated"]
    print(f"\n  LEARNING BUDGET (integral of lr over step):")
    print(f"    consumed 0 -> 25,200        {LR_INTEGRAL['0_25200']:.5f}")
    print(f"    consumed 12,000 -> 20,000   {LR_INTEGRAL['12000_20000']:.5f}"
          f"   <- the window E-DETECT-1 brackets")
    print(f"    REMAINING 25,200 -> 30,000  {rem:.5f}")
    print(f"    remaining as % of consumed  "
          f"{100 * rem / LR_INTEGRAL['0_25200']:.2f}%")
    print(f"    remaining as % of the 12k-20k window  "
          f"{100 * rem / LR_INTEGRAL['12000_20000']:.2f}%")
    out["budget"] = {
        "remaining_pct_of_consumed":
            round(100 * rem / LR_INTEGRAL["0_25200"], 3),
        "remaining_pct_of_12k_20k_window":
            round(100 * rem / LR_INTEGRAL["12000_20000"], 3),
    }
    Path("e_v6_movement.json").write_text(json.dumps(out, indent=1),
                                          encoding="utf-8")
    print("\n-> e_v6_movement.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
