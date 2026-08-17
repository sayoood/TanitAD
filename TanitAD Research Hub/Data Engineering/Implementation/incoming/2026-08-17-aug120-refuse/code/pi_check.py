"""Re-run the PI's 19 adjudicated clips against the RE-FUSED labels.

⭐ THE CHEAPEST VALIDATION AVAILABLE, AND THE ONLY ONE WITH A HUMAN IN IT.
The PI adjudicated 19 `S1_LANE_TARGET` rows of the S2 v1 labels on 2026-08-16
(`review/PI_VERDICTS_2026-08-16.json`, PRIMARY SOURCE). His notes name THREE
distinct defects, and each maps to a different fix in this re-fuse:

  1. "prepare lane change wrong" / "No Lane change here"  -> the geometric
     lane-change gate (§LC): LANE_TARGET / PREPARE_LANE_CHANGE must be GONE.
  2. "no agents wrong"                                    -> the C77 census:
     the record must no longer render an absent measurement as a negative.
  3. "3 lane wrong" / "there are two lanes, one ego lane and one for oncoming"
                                                          -> `lane_phrase`:
     the count is B1-scoped to the EGO CARRIAGEWAY and the prose must say so.

⛔ A VERDICT IS NOT RE-EARNED BY ASSERTION. Each complaint is checked as a
PREDICATE over the new record, per clip, and a clip only counts as addressed
when EVERY complaint the PI actually raised on it is addressed. A clip whose
note raised two defects and had one fixed is reported as PARTIAL, never as a
win.

⚠️ AND IT IS NOT A RE-ADJUDICATION. Only the PI can say whether the NEW label
is right. What is measurable here is whether the SPECIFIC THING HE OBJECTED TO
is still present — that is a fact about the artifact, and it is all this
claims.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
S2DIR = os.path.join(
    REPO, "TanitAD Research Hub", "Data Engineering", "Implementation",
    "incoming", "2026-08-16-s2-v1-labels")

#: The PI's complaint vocabulary -> the predicate that decides it, MEASURED on
#: the note text. Ordered; a note may raise several.
COMPLAINTS = (
    ("lane_change", re.compile(r"lane\s*change|lane change|prepare", re.I)),
    ("no_agents", re.compile(r"no agents?\b|not agent", re.I)),
    ("lane_count", re.compile(r"\d+\s*lane|one lane|two lanes|lanes", re.I)),
)


def load_jsonl(p):
    return {r["clip_id"]: r for r in
            (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--a3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    verdicts = json.load(open(os.path.join(S2DIR, "review",
                                           "PI_VERDICTS_2026-08-16.json"),
                              encoding="utf-8"))["verdicts"]
    s2v1 = load_jsonl(os.path.join(S2DIR, "labels", "s2_labels_aug120.jsonl"))
    s2v2 = load_jsonl(os.path.join(S2DIR, "review", "labels_v2",
                                   "s2_labels_aug120.jsonl"))

    def load(d):
        out = {}
        for p in glob.glob(os.path.join(d, "*.json")):
            if os.path.basename(p).startswith("_"):
                continue
            r = json.load(open(p, encoding="utf-8"))
            out[r["clip_id"]] = r
        return out
    a0, a3 = load(a.a0), load(a.a3)

    # ---- ⭐ the independent cross-check, PER CLIP not per histogram -------- #
    # ph1_fuse + s2_derive and the S2 label builder + s2_derive are DIFFERENT
    # code paths onto the same module. If they agree per clip, the re-fuse
    # reproduces the corrected S2 v2 label set rather than merely resembling it.
    xs = {"n": 0, "g_str_agree": 0, "a_str_agree": 0, "g_str_diff": [],
          "a_str_diff": []}
    for cid, rec in a3.items():
        if cid not in s2v2:
            continue
        xs["n"] += 1
        gv = (rec["vocab"]["g_str"] or {}).get("token")
        av = (rec["vocab"]["a_str"] or {}).get("token")
        gr = (s2v2[cid].get("g_str") or {}).get("token")
        ar = (s2v2[cid].get("a_str") or {}).get("token")
        xs["g_str_agree"] += int(gv == gr)
        xs["a_str_agree"] += int(av == ar)
        if gv != gr:
            xs["g_str_diff"].append({"clip": cid, "fused": gv, "s2v2": gr})
        if av != ar:
            xs["a_str_diff"].append({"clip": cid, "fused": av, "s2v2": ar})

    rows, tally = [], {"addressed": 0, "partial": 0, "not_addressed": 0,
                       "n_graded_wrong": 0, "n_graded_correct": 0,
                       "n_ungraded": 0}
    for cid, v in verdicts.items():
        if v.get("section") != "S1_LANE_TARGET":
            continue
        note = v.get("note") or ""
        raised = [name for name, rx in COMPLAINTS if rx.search(note)]
        old = s2v1.get(cid, {})
        new = a3.get(cid, {})
        old_g = (old.get("g_str") or {}).get("token")
        old_a = (old.get("a_str") or {}).get("token")
        new_g = ((new.get("vocab") or {}).get("g_str") or {}).get("token")
        new_a = ((new.get("vocab") or {}).get("a_str") or {}).get("token")
        desc0 = (a0.get(cid) or {}).get("scenario_description", "")
        desc3 = new.get("scenario_description", "")
        cen = ((new.get("perception") or {}).get("census") or {})
        checks = {
            "lane_change": {
                "was": [t for t in (old_g, old_a)
                        if t in ("LANE_TARGET", "PREPARE_LANE_CHANGE")],
                "now": [t for t in (new_g, new_a)
                        if t in ("LANE_TARGET", "PREPARE_LANE_CHANGE")],
            },
            "no_agents": {
                "was": "no agents" in desc0,
                "now": "no agents" in desc3,
                "census_state": cen.get("state"),
                "n_agent_tracks": cen.get("n_agents"),
            },
            "lane_count": {
                "was": desc0.split(";")[0] if desc0 else None,
                "now": desc3.split(";")[0] if desc3 else None,
                "scope_named": "-lane-ego-carriageway" in desc3
                               or "UNCLEAR" in desc3,
            },
        }
        resolved = {
            "lane_change": not checks["lane_change"]["now"],
            "no_agents": (not checks["no_agents"]["now"]
                          and cen.get("state") == "measured"),
            "lane_count": checks["lane_count"]["scope_named"],
        }
        rel = {k: resolved[k] for k in raised} or {}
        status = ("addressed" if rel and all(rel.values())
                  else "not_addressed" if rel and not any(rel.values())
                  else "partial" if rel else "no_complaint_text")
        tally[status] = tally.get(status, 0) + 1
        if v.get("v") == "wrong":
            tally["n_graded_wrong"] += 1
        elif v.get("v") == "correct":
            tally["n_graded_correct"] += 1
        else:
            tally["n_ungraded"] += 1
        rows.append({
            "clip_id": cid, "pi_verdict": v.get("v"), "pi_note": note,
            "complaints_raised": raised, "status": status,
            "old_g_str": old_g, "old_a_str": old_a,
            "new_g_str": new_g, "new_a_str": new_a,
            "s2v2_g_str": (s2v2.get(cid, {}).get("g_str") or {}).get("token"),
            "checks": checks, "resolved": resolved,
            "old_scenario": desc0, "new_scenario": desc3,
        })

    out = {"class": "MEASURED",
           "source": "review/PI_VERDICTS_2026-08-16.json (PRIMARY)",
           "scope_note": (
               "Only the S1_LANE_TARGET section was adjudicated; every other "
               "section is v=null and MUST NOT be read as agreement."),
           "claim_limit": (
               "This measures whether the SPECIFIC DEFECT THE PI NAMED is "
               "still present in the new record. It is NOT a re-adjudication "
               "— only the PI can grade the new label."),
           "n_rows": len(rows), "tally": tally,
           "cross_check_vs_s2_labels_v2": xs, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"tally": tally, "cross_check": {
        k: v for k, v in xs.items() if not k.endswith("_diff")}}, indent=1))
    print("PI_CHECK_DONE ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
