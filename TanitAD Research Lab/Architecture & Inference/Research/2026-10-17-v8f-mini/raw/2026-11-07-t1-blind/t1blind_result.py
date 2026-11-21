"""t1blind_result: writes RESULT_T1BLIND.md / .json from raw/READ_T1BLIND.json and raw/GATES.json (no new numbers: a formatter). Four families, STRATEGIC listed as PAUSED (PI item 25)."""
import os
import sys
import json
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
ORDER = ("PRIMARY", "crawl", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fmt(r):
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args(argv)
    RAW_, OUT_ = a.raw, a.out_dir
    R = json.load(open(f"{RAW_}/READ_T1BLIND.json"))
    G = json.load(open(f"{RAW_}/GATES.json"))
    out = {"card": "PREREG_T1_BLIND.md", "tier": R["tier"], "outcome": R["outcome"], "outcome_reading_beside": R["outcome_reading_beside"], "outcome_rule_reading": R["outcome_rule_reading"], "next_committed_by_card": R["next_committed_by_card"],
           "F_m": R["F_m"], "estimator": R["estimator"], "gates_PASS": G["PASS"], "read_json": f"{RAW_}/READ_T1BLIND.json", "read_json_sha256": sha(f"{RAW_}/READ_T1BLIND.json"), "gates_json_sha256": sha(f"{RAW_}/GATES.json"),
           "strategic_family": R["strategic_family"], "seeds": {}}
    md = ["# RESULT_T1BLIND (T1-kin, eval-141, scene `reindexed`)", "",
          f"**Outcome by the card's section 5: {R['outcome']}.**  Tier T1-kin. Estimator: paired episode-cluster bootstrap B 2,000 seed 0 over 141 episodes / 2,399 windows. F = {R['F_m']} m (training-run spread of M2b on ADE6).",
          f"Reading of the card's disjunction: {R['outcome_rule_reading']}.", "",
          "## Gates (card section 3): " + ("all PASS" if G["PASS"] else "NOT ALL PASS"), "",
          "| seed | G-ID | G-MEAN (frames received / equal to the pinned mean) | G-EGO (tick-0 windows equal / compared) |", "|---|---|---|---|"]
    for s in ("s0", "s1"):
        g = G["seeds"][s]
        md.append(f"| {s} | {'PASS' if g['G-ID'] else 'FAIL'} | {'PASS' if g['G-MEAN']['PASS'] else 'FAIL'} ({g['G-MEAN']['frames_received']:,} / {g['G-MEAN']['frames_equal_mean_bitwise']:,}) | "
                  f"{'PASS' if g['G-EGO']['PASS'] else 'FAIL'} ({g['G-EGO']['tick0_windows_equal']:,} / {g['G-EGO']['tick0_windows_compared']:,}) |")
    md += ["", "G-EGO compares the tick-0 ego inputs (the logged history) window for window; ticks 1 and 2 are the executed motion, which differs when the plan differs (the effect under test).", "",
           "## The two deciding rows, B = BLIND1 - RIC0", "", "| seed | ADE6 B (m) [95 % CI] | status | point above F | M1 crawl share B [95 % CI] | status |", "|---|---|---|---|---|---|"]
    for s in ("s0", "s1"):
        d = R["seeds"][s]
        c = d["conditions"]
        md.append(f"| {s} | {fmt(d['rows_BLIND1_minus_RIC0']['ADE6'])} | {d['status_BLIND1_minus_RIC0']['ADE6']} | {c['ade6_point_above_F']} | {fmt(d['rows_BLIND1_minus_RIC0']['M1 crawl share'])} | {d['status_BLIND1_minus_RIC0']['M1 crawl share']} |")
        out["seeds"][s] = {"ADE6_B": d["rows_BLIND1_minus_RIC0"]["ADE6"], "M1_B": d["rows_BLIND1_minus_RIC0"]["M1 crawl share"], "conditions": c,
                           "separated_better_rows": d["separated_better_rows"], "separated_worse_rows": d["separated_worse_rows"], "status_BLIND1_minus_RIC0": d["status_BLIND1_minus_RIC0"],
                           "status_BLIND1_minus_P10": d["status_BLIND1_minus_P10"], "status_RIC0_minus_P10": d["status_RIC0_minus_P10"]}
    md += ["", f"Alternative readings, beside: {json.dumps(R['outcome_reading_beside'])}", "", "## All 19 rows, per seed (status: better / worse = CI excludes 0 on that side of the row's direction; n.s. = CI contains 0)", ""]
    for s in ("s0", "s1"):
        d = R["seeds"][s]
        md += [f"### {s}", "", "| family | row | BLIND1 - RIC0 [CI] | status | BLIND1 - P10 [CI] | status | RIC0 - P10 status |", "|---|---|---|---|---|---|---|"]
        labs = sorted(d["rows_BLIND1_minus_RIC0"], key=lambda l: (ORDER.index(d["family_of"][l]), list(d["rows_BLIND1_minus_RIC0"]).index(l)))
        for lab in labs:
            fam = d["family_of"][lab]
            tag = fam + (" (PAUSED, decides nothing)" if fam == "STRATEGIC" else "")
            md.append(f"| {tag} | {lab} | {fmt(d['rows_BLIND1_minus_RIC0'][lab])} | {d['status_BLIND1_minus_RIC0'][lab]} | {fmt(d['rows_BLIND1_minus_P10'][lab])} | {d['status_BLIND1_minus_P10'][lab]} | {d['status_RIC0_minus_P10'][lab]} |")
        md += ["", f"Separated-better in B: {', '.join(d['separated_better_rows']) or 'none'}.  Separated-worse in B: {', '.join(d['separated_worse_rows']) or 'none'}.", ""]
    md += ["## Next, committed by the card for this outcome", "", R["next_committed_by_card"], "",
           "STRATEGIC family: PAUSED (PI item 25), listed above, decides nothing. The estimator answers another draw of EPISODES only; BLIND1 is a deterministic function of C0, so the seed question is read by the two seeds side by side.", "",
           f"Artifacts: `raw/READ_T1BLIND.json` (sha256 {out['read_json_sha256']}), `raw/GATES.json` (sha256 {out['gates_json_sha256']}), `raw/GATE_ID_s*.json`, rollouts `F:/TanitAD-scratch/motion_c/t1_blind/out/`."]
    open(f"{OUT_}/RESULT_T1BLIND.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    json.dump(out, open(f"{OUT_}/RESULT_T1BLIND.json", "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote RESULT_T1BLIND.md / .json:", R["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
