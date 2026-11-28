#!/usr/bin/env python3
"""recover_next_combine.py -- RECOVER-NEXT: the three per-arm reads (raw/RECOVER_READ_{RAPR,RTL,RP01}.json, any that exist) -> raw/RECOVER_NEXT.json.

Card: dagland/recover_next/PREREG_RECOVER_NEXT_FROZEN.md (sha256 01046fe6...), sections 3 and 4.  Nothing here is a new rule: it applies the card's attribution (Q1), the RAPR secondary and the section-4 next-arm table to numbers
the reader already wrote.  R's values are READ from dagland/recover_read/raw/RECOVER_READ.json (primary.share_closed, label), never typed.

Per arm (X in RAPR, RTL, RP01): the RECOVER-FT label, the composite status (DOMINANT counted or not), the M1 share closed, the guards.
Attribution Q1 (the HARMS question; defined when R's label is HARMS): blend harmed iff RTL's guards PASS at BOTH seeds; dose harmed iff RP01's guards PASS at BOTH seeds; reported "both" when both pass,
"neither" when neither does, "blend" / "dose" when only that arm's guards pass; "INCOMPLETE" when RTL or RP01 has no usable read.  Reported with RTL's and RP01's M1 share closed.
RAPR secondary: RAPR's M1 share closed strictly exceeds R's at each seed (an undefined share counts 0, as the reader's share_mean does).
Next arm (card section 4), the FIRST row in table order that matches; every matching row is listed; a case no row covers is reported UNCOVERED (the combiner never invents an arm).
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
DL = "E:/Projects/TanitAD-artifacts/dagland"
R_JSON = f"{DL}/recover_read/raw/RECOVER_READ.json"
CARD = f"{DL}/recover_next/PREREG_RECOVER_NEXT_FROZEN.md"
ARMS = ("RAPR", "RTL", "RP01")
SEEDS = ("s0", "s1")
M1 = "M1 crawl share"
NEXT = {
    "row1": "any XC DOMINANT, counted: evidence videos + the PI scale signal; a third seed pair of X on fresh inits.",
    "row2": "RAPRC fails D2 on M1 alone and RAPR's M1 closure exceeds R's at both seeds: dose up, RAPR at --recover-p 0.5 (2 runs, ~1.3 h).",
    "row3": "RAPRC fails D2 on a non-M1 row: the a-priori law with whichever target / dose the attribution says does not harm (RTL or RP01 setting).",
    "row4": ("RAPR's M1 closure does not exceed R's at either seed: the lever is not the size of the training deviation; the level-from-image auxiliary (a head regressing the logged speed from the current frame tokens, "
             "fed back as a feature; RECOVER-FT section 8), as a PI-decision item (architecture change)."),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def usable(j, arm):
    """A per-arm read is usable iff it exists, names this arm, was not refused and carries the blocks the rules read.  -> (bool, reason)."""
    if j is None:
        return False, "ABSENT"
    if j.get("r_arm") != arm:
        return False, f"WRONG ARM: the file says r_arm={j.get('r_arm')!r}"
    if j.get("refused"):
        return False, f"REFUSED: {j.get('refused')}"
    if not all(k in j for k in ("label", "primary", "guards", "composite")):
        return False, "INCOMPLETE: label / primary / guards / composite missing"
    return True, None


def share(j, sk):
    """M1 share closed of one seed; an undefined share counts 0 (the reader's share_mean convention)."""
    v = ((j.get("primary") or {}).get("share_closed") or {}).get(sk)
    return 0.0 if v is None else float(v)


def arm_summary(j, arm):
    ok, why = usable(j, arm)
    if not ok:
        return {"arm": arm, "usable": False, "why": why}
    comp, g = j.get("composite") or {}, j["guards"]
    return {"arm": arm, "usable": True, "label": j["label"], "primary_PASS": j["primary"]["PASS"], "guards_PASS": g["PASS"], "floor_PASS": (j.get("floor") or {}).get("PASS"),
            "freeze_reading": (j.get("frame_freeze") or {}).get("reading"), "composite_status": comp.get("status"), "composite_counts": comp.get("counts"), "composite_attribution_vs_C0C": comp.get("attribution"),
            "m1_share_closed": {sk: j["primary"]["share_closed"].get(sk) for sk in SEEDS}, "m1_mean_share_closed": j["primary"].get("mean_share_closed"),
            "guards_per_seed": {sk: {"ade6_delta": v.get("ade6_delta"), "ade6_within_guard": v.get("ade6_within_guard"), "non_crawl_separated_worse": v.get("non_crawl_separated_worse")} for sk, v in (g.get("per_seed") or {}).items()},
            "controls_failed": j.get("controls_failed"), "reader_sha256": j.get("reader_sha256")}


def guards_pass_both_seeds(j):
    """The arm's guards hold at EACH of the two seeds: ADE6 R-C0 point within the guard and no separated-worse non-crawl row (re-derived per seed from guards.per_seed, and equal to guards.PASS)."""
    per = (j.get("guards") or {}).get("per_seed") or {}
    if set(per) != set(SEEDS):
        return None
    return bool(all(per[sk]["ade6_within_guard"] and not per[sk]["non_crawl_separated_worse"] for sk in SEEDS))


def attribution_q1(rtl, rp01, r_label):
    """Q1.  ``rtl`` / ``rp01`` = per-arm read JSONs (or None).  -> dict with ``answer`` in blend | dose | both | neither | INCOMPLETE | NOT APPLICABLE."""
    out = {"question": "Q1 (committed HARMS question): did the rejoin blend or the dose harm?", "R_label": r_label,
           "rule": "blend harmed iff RTL's guards PASS at both seeds; dose harmed iff RP01's guards PASS at both seeds; both / neither otherwise"}
    ur, ud = usable(rtl, "RTL"), usable(rp01, "RP01")
    out["RTL_usable"], out["RP01_usable"] = ur, ud
    out["RTL_m1_share_closed"] = {sk: (None if not ur[0] else (rtl["primary"]["share_closed"] or {}).get(sk)) for sk in SEEDS}
    out["RP01_m1_share_closed"] = {sk: (None if not ud[0] else (rp01["primary"]["share_closed"] or {}).get(sk)) for sk in SEEDS}
    out["RTL_m1_mean_share_closed"] = rtl["primary"].get("mean_share_closed") if ur[0] else None
    out["RP01_m1_mean_share_closed"] = rp01["primary"].get("mean_share_closed") if ud[0] else None
    if r_label != "HARMS":
        out["answer"], out["note"] = "NOT APPLICABLE", f"R's label is {r_label!r}, not HARMS: Q1 is the question asked of a HARMS verdict"
        return out
    if not (ur[0] and ud[0]):
        out["answer"], out["note"] = "INCOMPLETE", "both RTL and RP01 reads are needed (" + "; ".join(f"{n}: {u[1]}" for n, u in (("RTL", ur), ("RP01", ud)) if not u[0]) + ")"
        out["blend_harmed"] = None if not ur[0] else bool(guards_pass_both_seeds(rtl))
        out["dose_harmed"] = None if not ud[0] else bool(guards_pass_both_seeds(rp01))
        return out
    gb, gd = guards_pass_both_seeds(rtl), guards_pass_both_seeds(rp01)
    if gb is None or gd is None:
        out["answer"], out["note"] = "INCOMPLETE", "guards.per_seed does not carry exactly s0 and s1"
        return out
    out["blend_harmed"], out["dose_harmed"] = bool(gb), bool(gd)
    out["RTL_guards_pass_per_seed"] = {sk: bool(rtl["guards"]["per_seed"][sk]["ade6_within_guard"] and not rtl["guards"]["per_seed"][sk]["non_crawl_separated_worse"]) for sk in SEEDS}
    out["RP01_guards_pass_per_seed"] = {sk: bool(rp01["guards"]["per_seed"][sk]["ade6_within_guard"] and not rp01["guards"]["per_seed"][sk]["non_crawl_separated_worse"]) for sk in SEEDS}
    out["answer"] = "both" if (gb and gd) else ("blend" if gb else ("dose" if gd else "neither"))
    return out


def rapr_secondary(rapr, r):
    """RAPR's M1 share closed strictly exceeds R's at EACH seed.  R's shares come from R's own read (``primary.share_closed``)."""
    ok, why = usable(rapr, "RAPR")
    out = {"rule": "RAPR's M1 share closed exceeds R's (0.220 / 0.132 in the RECOVER-FT read, read from its JSON) at each seed", "R_m1_share_closed": {sk: share(r, sk) for sk in SEEDS}}
    if not ok:
        out.update({"usable": False, "why": why, "exceeds_at": None, "exceeds_at_both": None, "exceeds_at_neither": None})
        return out
    ex = {sk: bool(share(rapr, sk) > share(r, sk)) for sk in SEEDS}
    out.update({"usable": True, "RAPR_m1_share_closed": {sk: share(rapr, sk) for sk in SEEDS}, "exceeds_per_seed": ex, "exceeds_at_both": bool(all(ex.values())), "exceeds_at_neither": bool(not any(ex.values())),
                "exceeds_at_exactly_one": bool(sum(ex.values()) == 1)})
    return out


def d2_profile(comp):
    """What the composite's Read A failed on, per the by-seed blocks: M1 alone (D2 fails on M1 crawl share only; D1 and D3 hold) / a non-M1 row / neither."""
    by = (comp or {}).get("by_seed") or {}
    if set(by) != set(SEEDS):
        return None
    rows = {sk: list(by[sk]["separated_worse_rows"]) for sk in SEEDS}
    non_m1 = bool(any(x != M1 for r in rows.values() for x in r))
    m1_alone = bool(not non_m1 and any(M1 in r for r in rows.values()) and all(by[sk]["D1"] and by[sk]["D3"] for sk in SEEDS))
    return {"separated_worse_rows": rows, "fails_D2_on_M1_alone": m1_alone, "fails_D2_on_a_non_M1_row": non_m1,
            "D1_D3_hold_at_both_seeds": bool(all(by[sk]["D1"] and by[sk]["D3"] for sk in SEEDS))}


def next_arm(arms, attribution, secondary):
    """Card section 4: the first matching row in table order.  ``arms`` = {arm: read JSON or None}."""
    matching, facts = [], {}
    counted = [a for a in ARMS if usable(arms.get(a), a)[0] and (arms[a].get("composite") or {}).get("counts") is True]
    facts["XC_dominant_counted"] = counted
    if counted:
        matching.append("row1")
    rapr = arms.get("RAPR")
    prof = d2_profile(rapr.get("composite")) if usable(rapr, "RAPR")[0] else None
    facts["RAPRC_d2_profile"] = prof
    facts["RAPR_closure_exceeds_R"] = {k: secondary.get(k) for k in ("exceeds_at_both", "exceeds_at_neither", "exceeds_at_exactly_one")}
    if prof and prof["fails_D2_on_M1_alone"] and secondary.get("exceeds_at_both"):
        matching.append("row2")
    if prof and prof["fails_D2_on_a_non_M1_row"]:
        matching.append("row3")
    if secondary.get("exceeds_at_neither"):
        matching.append("row4")
    sel = matching[0] if matching else None
    out = {"matching_rows": matching, "selected_row": sel, "facts": facts}
    if sel is None:
        out["selected"] = None
        out["status"] = "UNCOVERED: no row of card section 4 matches"
        out["why_uncovered"] = ("RAPR has no usable read" if not usable(rapr, "RAPR")[0] else
                                ("RAPR's closure exceeds R's at exactly one seed (row 2 needs both, row 4 needs neither)" if secondary.get("exceeds_at_exactly_one") else
                                 "RAPRC fails a term the table does not name (D1 / D3, or a DOMINANT that is not counted), or the composite was not computed"))
        out["rule_zero"] = "the Master Mind names the next arm; this is a gap in the card's table, reported, not filled"
        return out
    out["status"] = "SELECTED" if len(matching) == 1 else "SELECTED (first of several matching rows, table order)"
    out["selected"] = NEXT[sel]
    if sel == "row3":
        a = attribution.get("answer")
        out["row3_setting"] = {"blend": "target logged (the RTL setting)", "dose": "--recover-p 0.1 (the RP01 setting)", "both": "target logged AND --recover-p 0.1 (both settings)",
                               "neither": "NO setting is known not to harm: the attribution is 'neither'", "INCOMPLETE": "pending: the attribution needs both RTL and RP01",
                               "NOT APPLICABLE": "attribution not applicable"}.get(a, a)
    if sel == "row1":
        out["row1_arms"] = counted
    return out


def combine(arms, r_json):
    """``arms`` = {arm: per-arm read JSON or None}; ``r_json`` = R's RECOVER-FT read."""
    summ = {a: arm_summary(arms.get(a), a) for a in ARMS}
    attr = attribution_q1(arms.get("RTL"), arms.get("RP01"), r_json.get("label"))
    sec = rapr_secondary(arms.get("RAPR"), r_json)
    return {"per_arm": summ, "R_reference": {"label": r_json.get("label"), "m1_share_closed": {sk: share(r_json, sk) for sk in SEEDS}, "mean_share_closed": r_json["primary"].get("mean_share_closed"),
                                            "source": R_JSON},
            "attribution_q1": attr, "rapr_secondary": sec, "next_arm": next_arm(arms, attr, sec)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=f"{PK}/raw", help="where RECOVER_READ_<ARM>.json are, and where RECOVER_NEXT.json is written")
    ap.add_argument("--r-json", default=R_JSON)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = json.load(open(a.r_json, encoding="utf-8"))
    arms, ins = {}, {}
    for arm in ARMS:
        p = f"{a.raw}/RECOVER_READ_{arm}.json"
        arms[arm] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
        ins[arm] = {"path": p, "sha256": sha256_file(p) if os.path.exists(p) else None}
    res = {"card": "PREREG_RECOVER_NEXT_FROZEN.md sections 3 and 4", "card_sha256": sha256_file(CARD) if os.path.exists(CARD) else None, "tier": "T1-kin",
           "estimator": "each arm's read: paired episode-cluster bootstrap, B=2000, seed 0 (answers another draw of EPISODES only); the combiner adds no interval",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "combiner_sha256": sha256_file(f"{HERE}/recover_next_combine.py"),
           "inputs": {**ins, "R": {"path": a.r_json, "sha256": sha256_file(a.r_json)}}}
    res.update(combine(arms, r))
    out = a.out or f"{a.raw}/RECOVER_NEXT.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ARMS", {k: (v.get("label"), v.get("composite_status")) if v["usable"] else v["why"] for k, v in res["per_arm"].items()})
    print("Q1", res["attribution_q1"]["answer"], "| RAPR secondary exceeds at both", res["rapr_secondary"].get("exceeds_at_both"), "| NEXT", res["next_arm"]["status"], res["next_arm"]["selected_row"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
