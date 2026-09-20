"""How often does NavSim warmup's <= 5 m ego-progress clause actually fire, and does it
explain STOP's win?

`pdm_scorer.py` sets EP to 1 for EVERY proposal when the best rule-compliant proposal's
progress is <= 5 m. That is a claim about the PROTOCOL, so it is checkable from the banked
per-scene CSVs with no scoring and no GPU.

Detection: the all-zero STOP plan has zero displacement, so its EP can only be 1.0 on a
scene where the clause fired. CONTROL: on exactly those scenes EVERY other arm must also
read 1.0 -- the clause is arm-independent or it is not this clause.

/!\ The CSVs carry 3 AGGREGATE rows (`extended_pdm_score_*`) beside the scene rows. A first
run of this census included them and reported 206 "scenes"; the scene count is 204, which is
also the n the bridge RESULT reports. Rows are filtered to hex tokens.
"""
from __future__ import annotations

import csv, json, pathlib, re, statistics as st, sys

RAW = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                   "2026-09-19-navsim-refcv4b-bridge/raw")
HEX = re.compile(r"^[0-9a-f]{8,}$")
ARMS = {"STOP": "score_STOP_zero.csv", "CV": "score_CV_official.csv",
        "A1": "score_A1_ego_cmd.csv", "ECHO": "score_ECHO_ha0_ext.csv",
        "A2": "score_A2_vision_pure.csv", "A4": "score_A4_blind_ego_cmd.csv"}
COL = "ego_progress_stage_two"


def read(fn):
    out, dropped = {}, 0
    for r in csv.DictReader(open(RAW / fn, newline="", encoding="utf-8")):
        t = (r.get("token") or "")
        if not HEX.match(t):
            dropped += 1
            continue
        try:
            out[t] = float(r.get(COL, ""))
        except (TypeError, ValueError):
            pass
    return out, dropped


def main() -> int:
    eps, drops = {}, {}
    for a, fn in ARMS.items():
        if not (RAW / fn).exists():
            print(f"MISSING {fn}"); continue
        eps[a], drops[a] = read(fn)
    assert all(len(v) > 0 for v in eps.values()), "a control read zero rows"
    com = sorted(set.intersection(*(set(v) for v in eps.values())))
    fire = [t for t in com if abs(eps["STOP"][t] - 1.0) < 1e-9]
    rest = [t for t in com if t not in set(fire)]
    ctrl = {a: all(abs(eps[a][t] - 1.0) < 1e-9 for t in fire) for a in eps}
    res = {
        "_what": "how often the <= 5 m ego-progress clause fires on NavSim warmup stage 2",
        "_evidence_class": "MEASURED (ours, from the banked per-scene CSVs; no scoring, no GPU)",
        "_source": str(RAW), "_column": COL,
        "n_scenes": len(com), "aggregate_rows_dropped": drops,
        "clause_fires": len(fire), "clause_fires_pct": round(100 * len(fire) / len(com), 2),
        "_control_every_arm_reads_1_on_firing_scenes": ctrl,
        "remaining": {a: {"mean": round(st.mean([eps[a][t] for t in rest]), 4),
                          "median": round(st.median([eps[a][t] for t in rest]), 4),
                          "n_exactly_0": sum(1 for t in rest if eps[a][t] < 1e-9),
                          "n_exactly_1": sum(1 for t in rest if abs(eps[a][t] - 1) < 1e-9)}
                      for a in eps},
        "n_remaining": len(rest),
    }
    print(json.dumps(res, indent=1))
    (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
     else pathlib.Path("ep_clause_census.json")).write_text(
        json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
