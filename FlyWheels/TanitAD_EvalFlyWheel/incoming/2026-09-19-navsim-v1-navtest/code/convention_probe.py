#!/usr/bin/env python3
"""Does the NAVSIM v1 paper ROUND or TRUNCATE its printed values? (any python)

WHY THIS EXISTS. E1 MEASURED that [N2] (arXiv 2506.04218v3, the pseudo-simulation / navhard
paper) **truncates**: all 8 of its Table 2 CV S1 sub-metrics match E1's own numbers under
truncation to 1 dp, only 4 of 8 under rounding. W3's pre-registered verdict rule was written
around ROUNDING ("20.64 reads 20.6"), and the two conventions disagree on other values
(a true 20.58 prints 20.6 rounded, 20.5 truncated). ⛔ Assuming either one would be exactly the
"a correct formula under the wrong convention reads like an answer" trap.

WHAT CAN ACTUALLY BE TESTED for [N1] (arXiv 2406.15349v2, the v1 paper):

1. **Internal** — the only test that uses the paper alone. Table 2 prints three TransFuser seeds
   (83.3 / 84.0 / 84.4) and the text states their standard deviation as "+- 0.56". Recomputing the
   SAMPLE std from those printed seeds gives 0.5568: rounding prints 0.56, truncation prints 0.55.
   ⚠️ The paper computed its std from the TRUE seed values, not the printed ones, so this is
   evidence, not proof.
2. **Against the INHERITED live leaderboard** (NAVSIM_PROTOCOL.md 6.1, retrieved 2026-08-23; the
   paper says Table 3 IS that leaderboard). Four cells carry 4 dp there, so each one discriminates.
   ⚠️ The leaderboard is a MOVING TARGET two years newer than the paper: a disagreement can mean
   "different run", not "different convention". That is why the cells are reported individually.

The output is a JSON record; the verdict rule does NOT consume it automatically. It exists so the
amended rule can say WHY it reports both readings instead of assuming one.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from decimal import ROUND_HALF_UP, Decimal


def round_dp(x: float, dp: int) -> float:
    """Half-UP rounding at `dp` decimals (the convention a paper's 'rounded' would mean)."""
    q = Decimal(1).scaleb(-dp)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def trunc_dp(x: float, dp: int) -> float:
    """Truncation toward zero at `dp` decimals."""
    f = 10 ** dp
    return math.trunc(x * f) / f


def cell(name: str, true_value: float, printed: float, dp: int, source: str) -> dict:
    r, t = round_dp(true_value, dp), trunc_dp(true_value, dp)
    verdict = ("UNINFORMATIVE (both conventions print the same)" if r == t else
               ("ROUNDING" if r == printed else ("TRUNCATION" if t == printed else
                                                 "NEITHER — the two numbers are not the same run")))
    return {"cell": name, "true_value": true_value, "printed": printed, "dp": dp,
            "rounded": r, "truncated": t, "implies": verdict, "true_value_source": source}


def main() -> int:
    LB = "INHERITED: HF navtest leaderboard via NAVSIM_PROTOCOL.md 6.1 (retrieved 2026-08-23)"
    seeds = [83.3, 84.0, 84.4]                       # [N1] Table 2 A1/A2/A3, printed
    sd = statistics.stdev(seeds)                     # sample std (ddof=1)
    internal = {"what": "[N1] Table 2 three TransFuser seeds -> the text's '+- 0.56'",
                "printed_seeds": seeds, "sample_std_from_printed": round(sd, 6),
                "population_std_from_printed": round(statistics.pstdev(seeds), 6),
                "printed_std": 0.56, "dp": 2,
                "rounded": round_dp(sd, 2), "truncated": trunc_dp(sd, 2),
                "implies": ("ROUNDING" if round_dp(sd, 2) == 0.56 and trunc_dp(sd, 2) != 0.56
                            else ("TRUNCATION" if trunc_dp(sd, 2) == 0.56 else "UNINFORMATIVE")),
                "caveat": ("the paper's std came from the TRUE seeds, not the printed ones; a true "
                           "std anywhere in [0.560, 0.570) would also truncate to 0.56")}
    cells = [
        cell("Tab.3 Constant Velocity PDMS", 20.6517, 20.6, 1, LB),
        cell("Tab.3 TransFuser PDMS (3 seeds)", 83.8822, 83.9, 1, LB),
        cell("Tab.3 LTF PDMS (3 seeds)", 83.5239, 83.5, 1, LB),
        cell("Tab.3 Ego Status MLP PDMS (3 seeds)", 66.3989, 66.4, 1, LB),
        cell("Tab.3 TransFuser +- (3 seeds)", 0.4477, 0.4, 1, LB),
        cell("Tab.3 LTF +- (3 seeds)", 0.552, 0.6, 1, LB),
        cell("Tab.3 Ego Status MLP +- (3 seeds)", 0.9406, 0.9, 1, LB),
    ]
    tally = {}
    for c in cells:
        tally[c["implies"].split(" ")[0]] = tally.get(c["implies"].split(" ")[0], 0) + 1
    out = {
        "question": "does [N1] arXiv 2406.15349v2 ROUND or TRUNCATE its printed table values?",
        "primary": {"library_key": "2406.15349",
                    "sha256": "d3bc66d321cfceccc4f431113dea3f0f481d74d29dd283a33e204a4326f0403c"},
        "prior": ("E1 MEASURED that [N2] arXiv 2506.04218v3 TRUNCATES (8/8 truncating vs 4/8 "
                  "rounding on its Table 2 CV S1 column, n = 450). Same group, different paper: a "
                  "prior, never proof for [N1]."),
        "internal_test": internal,
        "leaderboard_cells": cells,
        "tally_over_leaderboard_cells": tally,
        "verdict": None,
        "consequence": None,
    }
    inf = [c for c in cells if c["implies"] in ("ROUNDING", "TRUNCATION")]
    n_r = sum(1 for c in inf if c["implies"] == "ROUNDING")
    n_t = sum(1 for c in inf if c["implies"] == "TRUNCATION")
    out["verdict"] = (
        f"UNSETTLED for [N1]. The one test internal to the paper implies {internal['implies']}; "
        f"the {len(inf)} informative leaderboard cells split {n_r} ROUNDING / {n_t} TRUNCATION, and "
        f"that source is a moving target two years newer than the paper, so a single disagreeing "
        f"cell is as likely to be a different run as a different convention. ⛔ No convention may "
        f"be assumed.")
    out["consequence"] = (
        "SPEC AMENDMENT A1: the verdict reports BOTH readings of the measured value at the table's "
        "printed precision — round-half-up and truncate — and a cell counts as REPRODUCED only if "
        "BOTH read the published digits. One-sided agreement is reported as "
        "REPRODUCED_UNDER_<convention> with the other reading printed beside it, never as a bare "
        "REPRODUCED, and never chosen for being the flattering one.")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "raw", "print_convention_probe.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
