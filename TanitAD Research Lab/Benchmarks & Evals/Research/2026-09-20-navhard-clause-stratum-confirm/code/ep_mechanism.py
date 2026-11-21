"""WHY does a zero-displacement plan earn EP median 0.195 where the <= 5 m clause is SILENT?

This is the question the clause result left behind (f3fdcc9): the clause is an amplifier, not the
cause, so `D-NAVSIM-STOP-1`'s second mechanism owns the effect and has been UNEXPLAINED since
d86dccb.

⭐ PREDICTION DERIVED FROM SOURCE, STATED BEFORE COMPUTING. `pdm_scorer._calculate_progress`
projects END minus START onto the centerline:

    start = ego_coords[p, 0, CENTER];  end = ego_coords[p, -1, CENTER]
    progress = centerline.project([start, end])[1] - [0]     # clipped at 0

`ego_coords[p, 0]` is the first SIMULATED tick, and the ego is propagated by a controller from its
initial velocity — it cannot stop instantly. ⇒ a "stop" plan still travels its BRAKING DISTANCE,
which is real centerline progress. Braking distance goes as v0^2 / 2a.

⇒ PREDICTED: STOP's EP on clause-NOT-fired scenes RISES with |v0|, and should track v0^2 at least
as well as v0. ⛔ If EP is FLAT in v0, this mechanism is wrong and the 0.195 is something else.

⚠️ EXPLORATORY: both venues' scores are already known, so this is a mechanism test with a
source-derived directional prediction, NOT a pre-registered confirmation. Its strength is that the
prediction comes from the code and is checked on TWO venues, not that the data were unseen.
⛔ CPU only, no model pass.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
HEX = re.compile(r"^[0-9a-f]{8,}$")
WARM = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                    "2026-09-19-navsim-refcv4b-bridge/raw")
WARM_ARMS = {"STOP": "score_STOP_zero.csv", "CV": "score_CV_official.csv",
             "A1": "score_A1_ego_cmd.csv", "ECHO": "score_ECHO_ha0_ext.csv",
             "A2": "score_A2_vision_pure.csv", "A4": "score_A4_blind_ego_cmd.csv"}
NAVH = pathlib.Path("D:/Projects/TanitAD/taniteval/results/bench/navsim_v2/"
                    "navhard_two_stage/20260920T082848Z-navsim_v2-none-06e257/scores")
COL = "ego_progress_stage_two"


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["token"]: r for r in csv.DictReader(fh)
                if HEX.match(r.get("token") or "")}


def num(row, col):
    try:
        return float(row.get(col, ""))
    except (TypeError, ValueError):
        return None


def venue(name, arms, table):
    tbl = [l.split(",") for l in (HERE / table).read_text().splitlines()[1:]]
    spd = {r[0]: float(r[3]) for r in tbl}
    toks = sorted(set.intersection(*(set(a) for a in arms.values())) & set(spd))
    fired = {t for t in toks if all(num(arms[a][t], COL) == 1.0 for a in arms)}
    sel = [t for t in toks if t not in fired]
    rows = [(spd[t], num(arms["STOP"][t], COL)) for t in sel]
    rows = [(s, e) for s, e in rows if e is not None]
    rows.sort()
    print(f"\n{name}: {len(sel)} clause-NOT-fired scenes, STOP EP median "
          f"{st.median(e for _, e in rows):.4f}")
    print("  v0 quintile      n    mean|v0|   STOP EP mean   EP median   EP==0")
    n = len(rows)
    out = []
    for q in range(5):
        lo, hi = q * n // 5, (q + 1) * n // 5
        c = rows[lo:hi]
        eps = [e for _, e in c]
        out.append({"q": q + 1, "n": len(c),
                    "mean_v0": round(st.mean(s for s, _ in c), 3),
                    "ep_mean": round(st.mean(eps), 4),
                    "ep_median": round(st.median(eps), 4),
                    "frac_ep_zero": round(sum(1 for e in eps if e == 0.0) / len(eps), 3)})
        print(f"     Q{q+1:<10} {len(c):>5}    {out[-1]['mean_v0']:>7.3f}   "
              f"{out[-1]['ep_mean']:>11.4f}   {out[-1]['ep_median']:>9.4f}   "
              f"{out[-1]['frac_ep_zero']:>5.3f}")
    mono = all(out[i]["ep_mean"] <= out[i + 1]["ep_mean"] for i in range(4))
    ratio = (out[4]["ep_mean"] / out[0]["ep_mean"]) if out[0]["ep_mean"] else None
    print(f"  monotone rising in v0: {mono}   Q5/Q1 = "
          + (f"{ratio:.2f}x" if ratio else "n/a"))
    return {"n": len(sel), "quintiles": out, "monotone": mono,
            "q5_over_q1": round(ratio, 3) if ratio else None}


def main() -> int:
    res = {}
    res["warmup"] = venue("WARMUP", {a: read(WARM / f) for a, f in WARM_ARMS.items()},
                          "warmup_token_v0.csv")
    res["navhard"] = venue("NAVHARD", {p.stem: read(p) for p in sorted(NAVH.glob("*.csv"))},
                           "navhard_token_v0.csv")
    both = res["warmup"]["monotone"] and res["navhard"]["monotone"]
    print("\n" + ("=> PREDICTION HELD on both venues: a stopping plan's EP is BRAKING DISTANCE"
                  if both else
                  "=> PREDICTION DID NOT HOLD on both venues; the braking-distance mechanism is "
                  "not sufficient"))
    res["_verdict"] = both
    (HERE / "ep_mechanism.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
