"""⚠️ EXPLORATORY, NOT CONFIRMATORY. Warmup's scores are already known, so nothing here is a
score-blind test; it is a DIAGNOSIS of why the 5.0 m/s threshold inverted, and it proposes the
replacement stratum that must then be pre-registered and tested where no score has been seen.

Three questions, all arm-independent or paired, all with 0 GPU and no new scoring:
  Q1  Does STOP still beat CV on the scenes where the clause did NOT fire?
      -> if STOP still wins there, the clause is NOT what makes stopping win.
  Q2  WHY does the clause fire MORE at speed (32.5 % vs 8.3 %)? Mechanism: the clause tests
      MASKED progress -- raw progress zeroed wherever a multiplicative metric fails -- so if
      fast scenes are where moving arms violate DAC/TLC/DDC, every compliant proposal is a
      slow one and the best masked progress stays under 5 m.
  Q3  Is the clause-fired split a BETTER stratum than the speed split? Reported as the
      separation it produces, not asserted.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import statistics as st
import sys

RAW = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                   "2026-09-19-navsim-refcv4b-bridge/raw")
HERE = pathlib.Path(__file__).resolve().parent
HEX = re.compile(r"^[0-9a-f]{8,}$")
ARMS = {"STOP": "score_STOP_zero.csv", "CV": "score_CV_official.csv",
        "A1": "score_A1_ego_cmd.csv", "ECHO": "score_ECHO_ha0_ext.csv",
        "A2": "score_A2_vision_pure.csv", "A4": "score_A4_blind_ego_cmd.csv"}
MULTI = ["no_at_fault_collisions_stage_two", "drivable_area_compliance_stage_two",
         "driving_direction_compliance_stage_two", "traffic_light_compliance_stage_two"]


def read(fn):
    out = {}
    for r in csv.DictReader(open(RAW / fn, newline="", encoding="utf-8")):
        t = (r.get("token") or "")
        if HEX.match(t):
            out[t] = r
    return out


def f(row, col):
    try:
        return float(row.get(col, ""))
    except (TypeError, ValueError):
        return None


def main() -> int:
    rows = {a: read(fn) for a, fn in ARMS.items()}
    toks = set(rows["STOP"])
    for a in rows:
        toks &= set(rows[a])
    tbl = [ln.split(",") for ln in
           (HERE / "warmup_token_v0.csv").read_text().splitlines()[1:]]
    speeds = {r[0]: float(r[3]) for r in tbl}
    # ⛔ the CLUSTER key. 204 synthetic scenes derive from 16 originals, so any interval over
    # scenes is pseudo-replication: the effective n is 16, not 204.
    clust = {r[0]: r[2] for r in tbl}
    toks = sorted(toks & set(speeds))
    fired = {t for t in toks
             if all(f(rows[a][t], "ego_progress_stage_two") == 1.0 for a in rows)}
    print(f"n={len(toks)}  fired={len(fired)}  not_fired={len(toks)-len(fired)}")

    out = {"_class": "EXPLORATORY on warmup (scores already known) -- NOT a confirmatory test",
           "n": len(toks), "n_fired": len(fired), "n_not_fired": len(toks) - len(fired)}

    # ---- Q1: STOP vs CV, PAIRED CLUSTER bootstrap, per clause stratum -------------------
    import random
    random.seed(20260920)

    def cluster_boot(sel, B=10000):
        """Resample ORIGINAL SCENES with replacement, not synthetic scenes."""
        by = collections.defaultdict(list)
        for t in sel:
            by[clust[t]].append(f(rows["STOP"][t], "score") - f(rows["CV"][t], "score"))
        keys = list(by)
        if len(keys) < 2:
            return None, None, len(keys)
        means = []
        for _ in range(B):
            draw = [x for k in random.choices(keys, k=len(keys)) for x in by[k]]
            means.append(st.mean(draw))
        means.sort()
        return means[int(0.025 * B)], means[int(0.975 * B)], len(keys)

    q1 = {}
    for name, sel in (("clause_FIRED", [t for t in toks if t in fired]),
                      ("clause_NOT_fired", [t for t in toks if t not in fired]),
                      ("ALL", toks)):
        d = [f(rows["STOP"][t], "score") - f(rows["CV"][t], "score") for t in sel]
        lo, hi, nk = cluster_boot(sel)
        q1[name] = {
            "n_scenes": len(d), "n_CLUSTERS": nk,
            "STOP_mean": round(st.mean(f(rows["STOP"][t], "score") for t in sel), 4),
            "CV_mean": round(st.mean(f(rows["CV"][t], "score") for t in sel), 4),
            "mean_STOP_minus_CV": round(st.mean(d), 4),
            "CI95_cluster_bootstrap": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "separated_from_zero": bool(lo is not None and (lo > 0 or hi < 0)),
            "STOP_wins_on": sum(1 for x in d if x > 0),
            "CV_wins_on": sum(1 for x in d if x < 0),
            "ties": sum(1 for x in d if x == 0),
        }
    out["Q1_stop_vs_cv_by_clause"] = q1
    out["_estimator"] = ("paired bootstrap over ORIGINAL-SCENE clusters (B=10,000, seed "
                         "20260920). ⛔ NOT over scenes: 204 synthetic scenes come from 16 "
                         "originals, so a per-scene interval is pseudo-replication.")

    # ---- Q2: mechanism -- are fast scenes where MOVING arms violate a multiplier? -------
    q2 = {}
    for sname, ssel in (("FAST_ge5", [t for t in toks if speeds[t] >= 5.0]),
                        ("SLOW_lt5", [t for t in toks if speeds[t] < 5.0])):
        ent = {"n": len(ssel),
               "clause_fires": round(sum(1 for t in ssel if t in fired) / len(ssel), 4)}
        for a in ("CV", "A1"):          # two arms that actually move
            for col in MULTI:
                vals = [f(rows[a][t], col) for t in ssel]
                vals = [v for v in vals if v is not None]
                ent[f"{a}.{col.replace('_stage_two','')}"] = round(st.mean(vals), 4)
        q2[sname] = ent
    out["Q2_multiplier_compliance_by_speed"] = q2

    # ---- Q3: which split separates STOP's advantage more? -------------------------------
    def gap(sel):
        return st.mean(f(rows["STOP"][t], "score") - f(rows["CV"][t], "score") for t in sel)
    sp_fast = [t for t in toks if speeds[t] >= 5.0]
    sp_slow = [t for t in toks if speeds[t] < 5.0]
    cl_y = [t for t in toks if t in fired]
    cl_n = [t for t in toks if t not in fired]
    out["Q3_separation"] = {
        "speed_split_|gap_fast - gap_slow|": round(abs(gap(sp_fast) - gap(sp_slow)), 4),
        "clause_split_|gap_fired - gap_notfired|": round(abs(gap(cl_y) - gap(cl_n)), 4),
    }

    print(json.dumps(out, indent=1))
    (HERE / "clause_stratum.json").write_bytes(json.dumps(out, indent=1).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
