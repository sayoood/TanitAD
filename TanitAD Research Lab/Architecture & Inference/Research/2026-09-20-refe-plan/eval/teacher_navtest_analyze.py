#!/usr/bin/env python3
"""P4 readout (eval/PREREG_P4_TEACHER_NAVTEST.md §6-§7): teacher vs human (paired) on navtest, the registered verdict,
sub-scores, zero rate, the four families, and the review's dirty-fan scene-type splits. Written BEFORE any teacher
score existed; it reads only harness CSVs (whose status must have been PASS) and the banked per-token inputs.

    python eval/teacher_navtest_analyze.py --teacher-csv <csv> --human-csv <csv> --seam <teacher seam.npz>
        --rows-dir <dir with rows_s*of*.jsonl> --out <result.json> [--families]

Estimator: eval/proxy_eval.py::boot -- paired log-cluster bootstrap over the logs, 10,000 resamples, 95 %, seed
20260927 (answers "another draw of LOGS"; the teacher is deterministic given device and rate).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import proxy_eval as PE  # noqa: E402

EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
C1_KNOWN = os.path.join(PKG, "raw", "2026-10-04-navtest-final", "frame_control_navtest_full.json")
GOAL_CENSUS = os.path.join(PKG, "raw", "2026-10-01-goal-trigger", "goal_census_navtest_full.json")
LANE_CENSUS = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "lane_census.jsonl")
D = "E:/Projects/TanitAD/data/refe_navtest"
STUDENT_DEPLOYED_CSV = f"{D}/score/refe_navtest_final_routefix/refe_navtest_final_routefix.csv"   # A9 adopted
STUDENT_DEPLOYED_SEAM = f"{D}/seams/refe_navtest_final_routefix.npz"
STUDENT_REVIEW_CSV = f"{D}/score/refe_navtest_final/refe_navtest_final.csv"     # the run REVIEW_7's fan census is of
SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
EV6E = f"{SP}/ev6e"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SUB = {"PDMS": "score", "NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
       "TTC": "time_to_collision_within_bound", "C": "comfort", "DDC": "driving_direction_compliance"}
PAPER_TEACHER = {"PDMS": 95.8, "NC": 99.8, "DAC": 99.9, "EP": 91.5, "TTC": 99.1, "C": 99.0}      # Table 4, PUBLISHED
PAPER_HUMAN = {"PDMS": 94.8, "NC": 100.0, "DAC": 100.0, "EP": 87.5, "TTC": 100.0, "C": 99.9}    # Table 4, PUBLISHED


def per_token(csv_p: str) -> dict:
    """{token: {sub: value x100}} from a harness CSV; invalid rows are dropped (and counted by the caller)."""
    out = {}
    for t, r in PE.read_csv(csv_p).items():
        if str(r.get("valid", "True")) != "True" or r.get("score") in (None, ""):
            continue
        out[t] = {k: 100.0 * float(r[c]) for k, c in SUB.items()}
    return out


def ci(vals: dict, tl: dict) -> dict:
    if not vals:
        return {"n": 0}
    m, lo, hi = PE.boot(vals, tl)
    return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals),
            "n_logs": len({tl[t] for t in vals})}


def block(arm: dict, toks: list, tl: dict) -> dict:
    out = {k: ci({t: arm[t][k] for t in toks}, tl) for k in SUB}
    out["zero_rate_pct"] = ci({t: 100.0 * (arm[t]["PDMS"] == 0.0) for t in toks}, tl)
    return out


def paired(a: dict, b: dict, toks: list, tl: dict) -> dict:
    out = {k: ci({t: a[t][k] - b[t][k] for t in toks}, tl) for k in SUB}
    out["zero_rate_pct"] = ci({t: 100.0 * ((a[t]["PDMS"] == 0.0) - (b[t]["PDMS"] == 0.0)) for t in toks}, tl)
    out["better_worse_tied_PDMS"] = [sum(a[t]["PDMS"] > b[t]["PDMS"] for t in toks),
                                     sum(a[t]["PDMS"] < b[t]["PDMS"] for t in toks),
                                     sum(a[t]["PDMS"] == b[t]["PDMS"] for t in toks)]
    return out


def verdict(pdms: dict, dac: dict) -> dict:
    """PREREG §6, point estimates, x100; the interval flag is reported and does not change the verdict."""
    p, d = pdms["mean"], dac["mean"]
    if p < 93.0 or d < 98.0:
        v = "TEACHER CAPS"
    elif p >= 94.0 and d >= 99.0:
        v = "TEACHER DOES NOT CAP"
    else:
        miss = [f"PDMS {p:.2f} (needs >= 94.0)"] if p < 94.0 else []
        miss += [f"DAC {d:.2f} (needs >= 99.0)"] if d < 99.0 else []
        v = "PARTIAL: " + "; ".join(miss)
    sens = [f"{name} threshold {th} inside CI {c['ci95']}" for name, c, ths in
            (("PDMS", pdms, (93.0, 94.0)), ("DAC", dac, (98.0, 99.0))) for th in ths
            if c["ci95"][0] <= th <= c["ci95"][1]]
    return {"verdict": v, "sensitive_to_log_sampling": sens,
            "rule": "CAPS iff PDMS < 93.0 OR DAC < 98.0; DOES NOT CAP iff PDMS >= 94.0 AND DAC >= 99.0; else PARTIAL"}


def split(name: str, groups, arms: dict, tl: dict) -> dict:
    res = {}
    for lab, toks in groups:
        if not toks:
            continue
        row = {"tokens": len(toks)}
        for an, arm in arms.items():
            row[an] = {"PDMS": ci({t: arm[t]["PDMS"] for t in toks}, tl),
                       "DAC": round(float(np.mean([arm[t]["DAC"] for t in toks])), 3),
                       "NC": round(float(np.mean([arm[t]["NC"] for t in toks])), 3),
                       "EP": round(float(np.mean([arm[t]["EP"] for t in toks])), 3),
                       "zero_rate_pct": round(float(np.mean([100.0 * (arm[t]["PDMS"] == 0) for t in toks])), 3)}
        res[lab] = row
    return res


def families(seam: str, toks: set, label: str, out_dir: str):
    z = np.load(seam)
    tt = z["token"]
    idx = [i for i, t in enumerate(tt) if str(t) in toks]
    sub = os.path.join(out_dir, f"{label}_seam.npz")
    np.savez(sub, token=tt[idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx], sampling=z["sampling"],
             arm=z["arm"])
    out = os.path.join(out_dir, f"{label}_families.json")
    with open(os.path.join(out_dir, f"{label}_families.log"), "w") as lf:
        subprocess.call([PY_T, "families6.py", "--seam", sub, "--inputs", EXPORT, "--stage", "1", "--label", label,
                         "--out", out, "--n-boot", "2000"], cwd=EV6E, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                        stdout=lf, stderr=subprocess.STDOUT)
    return json.load(open(out, encoding="utf-8")) if os.path.exists(out) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-csv", required=True)
    ap.add_argument("--human-csv", required=True)
    ap.add_argument("--seam", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: exp[t]["log_name"] for t in exp}
    T, H = per_token(a.teacher_csv), per_token(a.human_csv)
    S = per_token(STUDENT_DEPLOYED_CSV) if os.path.exists(STUDENT_DEPLOYED_CSV) else {}
    SR = per_token(STUDENT_REVIEW_CSV) if os.path.exists(STUDENT_REVIEW_CSV) else {}
    seam_toks = [str(t) for t in np.load(a.seam)["token"]]
    known = set(json.load(open(C1_KNOWN, encoding="utf-8")).get("over_bar", {}))
    both = [t for t in seam_toks if t in T and t in H]
    primary = sorted(t for t in both if t not in known)
    alltok = sorted(both)
    res = {"what": "P4: our DriveRL teacher protocol (released checkpoint_2400.pt, rank 0, 10 Hz, CPU) on NAVSIM v1.1 "
                   "navtest, paired with the logged human on the same tokens",
           "tier": "NAVSIM v1.1 PDMS = non-reactive pseudo-simulation of a 4 s open-loop plan against logged agents",
           "estimator": "paired log-cluster bootstrap over logs (proxy_eval.boot), 10,000 resamples, 95 %, seed 20260927",
           "inputs": {"teacher_csv": a.teacher_csv, "human_csv": a.human_csv, "seam": a.seam,
                      "student_deployed_csv": STUDENT_DEPLOYED_CSV, "student_review_csv": STUDENT_REVIEW_CSV},
           "counts": {"seam_rows": len(seam_toks), "teacher_valid": len(T), "human_valid": len(H),
                      "scored_both": len(both), "known_c1_36_in_set": len([t for t in both if t in known]),
                      "primary": len(primary), "invalid_teacher_rows": len([t for t in seam_toks if t not in T])}}
    res["teacher"] = {"primary": block(T, primary, tl), "all_tokens": block(T, alltok, tl)}
    res["human"] = {"primary": block(H, primary, tl), "all_tokens": block(H, alltok, tl)}
    res["teacher_minus_human"] = {"primary": paired(T, H, primary, tl), "all_tokens": paired(T, H, alltok, tl)}
    res["verdict"] = verdict(res["teacher"]["primary"]["PDMS"], res["teacher"]["primary"]["DAC"])
    res["verdict_all_tokens_reported"] = verdict(res["teacher"]["all_tokens"]["PDMS"],
                                                 res["teacher"]["all_tokens"]["DAC"])
    c2 = res["human"]["all_tokens"]["PDMS"]
    res["c2_human_control"] = {"human_PDMS_all": c2, "known_full_navtest": [94.55, [93.80, 95.24]],
                               "pass": bool(c2["mean"] >= 93.0 and (len(alltok) < 12000 or
                                                                    abs(c2["mean"] - 94.55) <= 0.25))}
    res["paper_table4_PUBLISHED"] = {"teacher": PAPER_TEACHER, "human": PAPER_HUMAN,
                                     "ours_minus_paper_teacher_point": {k: round(res["teacher"]["primary"][k]["mean"]
                                                                                 - v, 3)
                                                                        for k, v in PAPER_TEACHER.items()}}
    # the student on the same tokens (paired), for context
    if S:
        st = [t for t in primary if t in S]
        res["student_deployed_routefix"] = {"primary_subset": block(S, st, tl),
                                            "teacher_minus_student": paired(T, S, st, tl)}
    # ---- per-token side inputs
    rows = {}
    for p in glob.glob(os.path.join(a.rows_dir, "rows_s*of*.jsonl")):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "plan8" in r:
                rows[r["token"]] = r
    gc = json.load(open(GOAL_CENSUS, encoding="utf-8"))["rows"]
    share = {}
    for ln in open(LANE_CENSUS, encoding="utf-8"):
        r = json.loads(ln)
        nc = np.asarray(r["nc"][1:65], dtype=float)
        dac = np.asarray(r["dac"][1:65], dtype=float)
        share[r["token"]] = float(np.mean(nc * dac == 1.0))

    def bearing_of(g):
        if not g or len(g) < 4:
            return None
        return math.degrees(abs(math.atan2(g[3], max(g[2], 1e-3))))

    arms = {"teacher": T, "human": H}
    if SR:
        arms["student_review_run"] = SR
    P = [t for t in primary if (not SR or t in SR)]
    dr = {t: (gc[t]["ego_to_route_m"] if gc.get(t, {}).get("ego_to_route_m") is not None else 999.0) for t in P}
    sb = {t: bearing_of(gc.get(t, {}).get("goal")) for t in P}
    tb = {t: bearing_of(rows.get(t, {}).get("goal")) for t in P}
    v0 = {t: rows[t]["v0"] for t in P if t in rows}

    def bins(vals, edges, labels):
        return [(lab, [t for t in P if vals.get(t) is not None and lo <= vals[t] < hi]) for lab, (lo, hi) in
                zip(labels, edges)]

    res["splits"] = {
        "note": "scene types exactly as REVIEW_7 / gap_dirty_fans.py define them (goal census of the pre-A9 student goal; "
                "fan safe share over the 64 proposals of refe_navtest_final from lane_census); student_review_run = "
                "refe_navtest_final, the run whose fan the census describes",
        "by_route_distance": split("route", [("on_route<=20m", [t for t in P if dr[t] <= 20]),
                                             ("off_route>20m", [t for t in P if dr[t] > 20])], arms, tl),
        "by_far_goal_bearing_deg_review": split("b", bins(sb, [(0, 10), (10, 30), (30, 60), (60, 181)],
                                                          ["<10", "10-30", "30-60", ">=60"]), arms, tl),
        "by_teacher_goal_bearing_deg": split("tb", bins(tb, [(0, 10), (10, 30), (30, 60), (60, 181)],
                                                        ["<10", "10-30", "30-60", ">=60"]), arms, tl),
        "by_v0_mps": split("v0", bins(v0, [(0, 1), (1, 5), (5, 10), (10, 99)], ["<1", "1-5", "5-10", ">=10"]),
                           arms, tl),
        "by_student_fan_safe_share": split("fan", [("dirty<0.5", [t for t in P if t in share and share[t] < 0.5]),
                                                   ("mid", [t for t in P if t in share and 0.5 <= share[t] < 0.95]),
                                                   ("clean>=0.95", [t for t in P if t in share and share[t] >= 0.95])],
                                           arms, tl)}
    # inheritance: is the student's zero pick more likely where the TEACHER itself fails?
    if SR:
        tf = [t for t in P if T[t]["NC"] * T[t]["DAC"] == 0]
        tp = [t for t in P if T[t]["NC"] * T[t]["DAC"] > 0]
        dirty = [t for t in P if t in share and share[t] < 0.5]
        zero_s = [t for t in P if SR[t]["PDMS"] == 0]
        res["inheritance"] = {
            "teacher_NCxDAC_fail_tokens": len(tf),
            "student_zero_rate_where_teacher_fails_pct": round(100 * np.mean([SR[t]["PDMS"] == 0 for t in tf]), 2)
            if tf else None,
            "student_zero_rate_where_teacher_passes_pct": round(100 * np.mean([SR[t]["PDMS"] == 0 for t in tp]), 2),
            "dirty_fan_rate_where_teacher_fails_pct": round(100 * np.mean([t in dirty for t in tf]), 2) if tf else None,
            "dirty_fan_rate_where_teacher_passes_pct": round(100 * np.mean([t in dirty for t in tp]), 2),
            "teacher_fail_rate_on_dirty_fans_pct": round(100 * np.mean([T[t]["NC"] * T[t]["DAC"] == 0 for t in dirty]),
                                                         2) if dirty else None,
            "share_of_student_zero_picks_on_teacher_fail_tokens_pct":
                round(100 * len(set(zero_s) & set(tf)) / max(1, len(zero_s)), 2),
            "student_zero_picks": len(zero_s), "dirty_fan_tokens": len(dirty)}
    res["teacher_row_diagnostics"] = {
        "history_clamped": sum(1 for r in rows.values() if r.get("history_clamped")),
        "route_corrected_changed_length": sum(1 for r in rows.values()
                                              if r.get("corrected_route_n") != r.get("raw_route_n")),
        "plan_vs_exact_max_m": max((r["plan_vs_exact_max_m"] for r in rows.values()), default=None)}
    if a.families:
        fam = {"teacher": families(a.seam, set(primary), "p4_teacher", out_dir)}
        if os.path.exists(STUDENT_DEPLOYED_SEAM):
            fam["student_deployed"] = families(STUDENT_DEPLOYED_SEAM, set(primary), "p4_student_deployed", out_dir)
        fam["strategic"] = "UNAVAILABLE: NAVSIM scores no strategic decision / route-setting label (n = 0 by design)"
        res["families_files"] = {k: (os.path.join(out_dir, f"p4_{k}_families.json") if k != "strategic" else None)
                                 for k in fam}
        res["families_summary"] = {k: (v.get("families") if isinstance(v, dict) else v) for k, v in fam.items()}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    t, h = res["teacher"]["primary"], res["teacher_minus_human"]["primary"]
    print(f"ZZP4 {res['verdict']['verdict']} | teacher PDMS {t['PDMS']['mean']:.2f} {t['PDMS']['ci95']} "
          f"DAC {t['DAC']['mean']:.2f} {t['DAC']['ci95']} | teacher-human PDMS {h['PDMS']['mean']:+.2f} "
          f"{h['PDMS']['ci95']} | n={len(primary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
