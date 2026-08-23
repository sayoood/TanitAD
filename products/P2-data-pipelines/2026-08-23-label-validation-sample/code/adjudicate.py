"""Independent geometric adjudication of the S2 strategic labels.

⚠️ WHAT THIS IS AND IS NOT. These rules are a SECOND OPINION computed from ego
poses alone. They are not ground truth and they cannot see the road, the lane
markings, the signs or the other agents — so a disagreement is a FLAG FOR HUMAN
REVIEW, never a verdict that the label is wrong. The frames in the report are
what let a human settle it.

⭐ THE HORIZON CORRECTION THAT CHANGED EVERYTHING. My first pass judged g_str on
a 4 s window and found 6 "mismatches" — TURN_LEFT clips with ~0 deg of yaw.
Every one dissolved on the STRATEGIC horizon: the turns began 4.8-11.2 s after
the anchor. HIERARCHY_VOCABULARY.md §3 puts g_str at 8-30 s. Judging a
strategic label on a tactical window manufactures false positives, so the rules
below use the whole available horizon and REPORT THE ONSET TIME.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
TURN_DEG = 25.0          # the yaw a token-worthy turn must reach
DV_CRUISE = 2.0          # m/s gain that makes RESUME_CRUISE true
DV_REDUCE = 1.0          # m/s loss that makes REDUCE_TO true


def judge_g(tok, g):
    peak, onset = g["peak_yaw_after_key_deg"], g["t_yaw_onset_25deg_s"]
    if tok == "TURN_LEFT":
        if peak >= TURN_DEG:
            return "AGREE", f"left turn {peak:+.0f}deg begins t+{onset}s"
        return "FLAG", f"no left turn on the horizon (peak {peak:+.0f}deg)"
    if tok == "TURN_RIGHT":
        if peak <= -TURN_DEG:
            return "AGREE", f"right turn {peak:+.0f}deg begins t+{onset}s"
        if abs(peak) >= TURN_DEG:
            return "FLAG", (f"SIGN CONFLICT: labelled RIGHT, largest yaw is "
                            f"{peak:+.0f}deg (left)")
        return "FLAG", f"no right turn on the horizon (peak {peak:+.0f}deg)"
    if tok == "STRAIGHT_THROUGH":
        return (("AGREE", f"straight, peak {peak:+.0f}deg")
                if abs(peak) < TURN_DEG else
                ("FLAG", f"{peak:+.0f}deg turn labelled straight"))
    if tok in ("FOLLOW_MAIN_ROAD", "NONE_ABSTAIN"):
        if abs(peak) < TURN_DEG:
            return "AGREE", f"no junction manoeuvre (peak {peak:+.0f}deg)"
        return "FLAG", (f"{peak:+.0f}deg turn at t+{onset}s — a junction "
                        f"manoeuvre labelled "
                        f"{'no-goal' if tok == 'NONE_ABSTAIN' else 'corridor-follow'}")
    if tok == "STOP_AT":
        return (("AGREE", "reaches v=0 on the horizon")
                if g["comes_to_a_stop"] else
                ("FLAG", f"never stops (v_min {g['v_min_future_mps']:.2f})"))
    return "UNJUDGED", "no geometric rule for this token"


def judge_a(tok, abst, g):
    if abst:
        return "ABSTAIN", "declined — no a_str target (refuted lane-change gate)"
    dv, vmin = g["dv_key_to_end_mps"], g["v_min_future_mps"]
    vkey, vend = g["v_at_key_mps"], g["v_end_mps"]
    if tok == "PREPARE_STOP":
        if vkey < 1.0 and dv > DV_CRUISE:
            return "FLAG", (f"INVERTED: accelerating away from rest "
                            f"({vkey:.2f}->{vend:.2f} m/s), not preparing to stop")
        return (("AGREE", f"decelerates / stops (v_min {vmin:.2f})")
                if (dv < 0 or g["comes_to_a_stop"]) else
                ("FLAG", f"speed rises {dv:+.2f} m/s and never stops"))
    if tok == "RESUME_CRUISE":
        return (("AGREE", f"accelerates {dv:+.2f} m/s")
                if dv >= DV_CRUISE else
                ("FLAG", f"no acceleration ({dv:+.2f} m/s)"))
    if tok == "REDUCE_TO":
        return (("AGREE", f"slows {dv:+.2f} m/s")
                if dv <= -DV_REDUCE else
                ("FLAG", f"does not slow ({dv:+.2f} m/s)"))
    if tok == "HOLD_CORRIDOR":
        if g["comes_to_a_stop"] and vkey > 3.0:
            return "FLAG", (f"comes to a FULL STOP from {vkey:.2f} m/s — "
                            f"a stop labelled 'hold corridor'")
        return "AGREE", f"no stop event (v_min {vmin:.2f})"
    return "UNJUDGED", "no geometric rule for this token"


def main() -> None:
    rows = json.loads((OUT / "sample_slim.json").read_text(encoding="utf-8"))
    tally = {"g": {}, "a": {}}
    for r in rows:
        g = r["geometry"]
        gv, gw = judge_g(r["g_str"]["token"], g)
        av, aw = judge_a(r["a_str"]["token"], r["a_str"]["abstain"], g)
        r["verdict"] = {"g_str": {"verdict": gv, "why": gw},
                        "a_str": {"verdict": av, "why": aw}}
        tally["g"][gv] = tally["g"].get(gv, 0) + 1
        tally["a"][av] = tally["a"].get(av, 0) + 1

    (OUT / "adjudicated.json").write_text(json.dumps(rows, indent=1),
                                          encoding="utf-8")
    n = len(rows)
    print(f"n = {n} clips\n")
    print("g_str:", tally["g"])
    print("a_str:", tally["a"])
    agree_g = tally["g"].get("AGREE", 0)
    agree_a = tally["a"].get("AGREE", 0)
    print(f"\ng_str agreement: {agree_g}/{n} = {agree_g/n:.1%}")
    print(f"a_str agreement: {agree_a}/{n} = {agree_a/n:.1%}")
    print("\n--- FLAGGED ---")
    for r in rows:
        for fam in ("g_str", "a_str"):
            if r["verdict"][fam]["verdict"] == "FLAG":
                tok = r[fam]["token"]
                print(f"  {r['clip_id'][:8]}  {fam}={tok:17s} "
                      f"{r['verdict'][fam]['why']}")


if __name__ == "__main__":
    main()
