"""Q6 — T-FLIP and T-ZERO: do the instruments EXIST, and CAN THEY FAIL?

⛔ "An acceptance test that cannot fail is not an acceptance test." Four checks
in one night were green forever because their expected value was an expression
over the code under test (`CLAUDE.md`, 2026-09-07). So every row below is a
CONSTRUCTED input with a LITERAL expected verdict, including a DELIBERATE
REGRESSION that must go RED.

T-FLIP is a GATE (`acceptance_panel` gates on it). T-ZERO is a DIAGNOSTIC by
SPEC §10.5 — it carries NO bar by design — so for T-ZERO the right question is
not "can it fail" but "can it DISCRIMINATE": a nav-echo class and a
scene-learned class must come out different, and the turn classes must be
excluded from the summary.

⚠️ The instrument's REACHABILITY is a separate question from its correctness,
and it is checked last: a verdict function nothing calls is a brick.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)

from taniteval import refcv6_acceptance as A       # noqa: E402


def _arm(fed_mean, fed_lo, d_delta, d_lo, *, sep=True, nw=200, ne=40,
         powered=True, flipped=True):
    blk = {
        "powered": powered,
        "n_informative_windows": nw, "n_informative_episodes": ne,
        "paired_true_minus_shuffled": {"delta": d_delta, "lo": d_lo,
                                       "hi": d_lo + 0.2, "separated": sep,
                                       "n_windows": nw},
    }
    if flipped:
        blk["flipped"] = {
            "follows_FED_command": {"mean": fed_mean, "lo": fed_lo,
                                    "hi": min(1.0, fed_lo + 0.2),
                                    "n_windows": nw, "n_episodes": ne},
            "follows_TRUE_command": {"mean": 1 - fed_mean, "lo": 0.0,
                                     "hi": 0.3},
        }
    return blk


def main():
    res = {"module": A.__file__, "bars": {
        "TFLIP_BAR_FOLLOWS_FED": A.TFLIP_BAR_FOLLOWS_FED,
        "TFLIP_BAR_TRUE_MINUS_SHUFFLED": A.TFLIP_BAR_TRUE_MINUS_SHUFFLED,
        "TFLIP_TODAY_FOLLOWS_FED": A.TFLIP_TODAY_FOLLOWS_FED}}

    cases = [
        # name,                                          arm, expected verdict
        ("GREEN clears both bars",
         _arm(0.72, 0.61, 0.55, 0.44), "PASS"),
        ("REGRESSION today's refcv5-v2 numbers (0.205 / 0.099)",
         _arm(0.205, 0.150, 0.099, 0.050), "FAIL"),
        ("REGRESSION fed just under the bar (lo 0.49)",
         _arm(0.60, 0.49, 0.55, 0.44), "FAIL"),
        ("REGRESSION delta just under the bar (lo 0.37)",
         _arm(0.72, 0.61, 0.45, 0.37), "FAIL"),
        ("REGRESSION winner's curse: mean 0.72 but CI lo 0.20",
         _arm(0.72, 0.20, 0.55, 0.44), "FAIL"),
        ("REGRESSION delta clears its bar but is NOT separated",
         _arm(0.72, 0.61, 0.55, 0.44, sep=False), "FAIL"),
        ("ABSENCE no flipped block",
         _arm(0.72, 0.61, 0.55, 0.44, flipped=False), "NOT_RUN"),
        ("UNDERPOWERED 20 windows",
         _arm(0.72, 0.61, 0.55, 0.44, nw=20, ne=3), "UNPOWERED"),
        ("UNPOWERED flag off",
         _arm(0.72, 0.61, 0.55, 0.44, powered=False), "UNPOWERED"),
    ]
    rows = []
    for name, arm, expect in cases:
        got = A.tflip_verdict(arm)["verdict"]
        rows.append({"case": name, "expected": expect, "got": got,
                     "agrees": got == expect})
    res["TFLIP_cases"] = rows
    res["TFLIP_can_fail"] = any(r["got"] == "FAIL" for r in rows)
    res["TFLIP_can_pass"] = any(r["got"] == "PASS" for r in rows)
    res["TFLIP_all_as_expected"] = all(r["agrees"] for r in rows)

    # ------------------------------------------------------------- T-ZERO
    true_ = {
        "FOLLOW_LANE": {"n": 3629, "recall": 0.80},   # scene-learned
        "STOP_POINT": {"n": 327, "recall": 0.40},     # nav echo (constructed)
        "TURN_L": {"n": 275, "recall": 0.60},         # excluded by design
        "LANE_CHANGE_R": {"n": 15, "recall": 0.50},   # under min_n
    }
    zero_echo = {
        "FOLLOW_LANE": {"recall": 0.78},
        "STOP_POINT": {"recall": 0.00},
        "TURN_L": {"recall": 0.00},
        "LANE_CHANGE_R": {"recall": 0.00},
    }
    zero_scene = {k: {"recall": v["recall"]} for k, v in true_.items()}
    r_echo = A.tzero_nonturn_report(true_, zero_echo)
    r_scene = A.tzero_nonturn_report(true_, zero_scene)
    res["TZERO"] = {
        "echo_summary": r_echo["summary"],
        "scene_summary": r_scene["summary"],
        "discriminates": (r_echo["summary"]["min_retained_fraction"]
                          != r_scene["summary"]["min_retained_fraction"]),
        "turn_excluded_from_summary": (
            r_echo["classes"]["TURN_L"]["is_turn_class"] is True
            and r_echo["summary"]["n_classes_summarised"] == 2),
        "under_min_n_excluded": (
            r_echo["classes"]["LANE_CHANGE_R"]["powered"] is False),
        "has_pass_fail_bar": any(
            k in r_echo for k in ("verdict", "gates", "bars")),
        "classes": r_echo["classes"],
    }

    # ------------------------------------------------------- REACHABILITY
    panel = A.acceptance_panel(
        tflip=A.tflip_verdict(_arm(0.205, 0.150, 0.099, 0.050)),
        tzero=r_echo, obedience=None)
    res["PANEL"] = {"keys": sorted(panel),
                    "verdict": panel.get("verdict") or panel.get("summary"),
                    "raw": panel}
    p = OUT / "q6_tflip_tzero.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    for r in rows:
        mark = "OK " if r["agrees"] else "XX "
        print(f"{mark}{r['case']:55s} expect={r['expected']:9s} got={r['got']}")
    print("\nT-ZERO discriminates:", res["TZERO"]["discriminates"],
          "| turn excluded:", res["TZERO"]["turn_excluded_from_summary"],
          "| has bar:", res["TZERO"]["has_pass_fail_bar"])
    print("echo :", res["TZERO"]["echo_summary"]["median_retained_fraction"],
          res["TZERO"]["echo_summary"]["min_retained_fraction"])
    print("scene:", res["TZERO"]["scene_summary"]["median_retained_fraction"],
          res["TZERO"]["scene_summary"]["min_retained_fraction"])
    print("\nPANEL keys:", sorted(panel))
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
