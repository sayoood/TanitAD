"""How many digits of the bootstrap CI are REAL?

The self-test reproduced every point estimate exactly and disagreed with the banked CI in the 3rd
decimal. That is Monte-Carlo error from a different RNG stream, not a defect -- but it means the
banked interval was REPORTED to 4 dp and is not STABLE to 4 dp.

⛔ The wrong fix is to tune the seed until the numbers match, which fits the instrument to the
fixture. The right fix is to measure the spread across independent streams and report only the
digits it supports.
"""
from __future__ import annotations

import collections
import csv
import pathlib
import random
import re
import statistics as st

RAW = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                   "2026-09-19-navsim-refcv4b-bridge/raw")
HERE = pathlib.Path(__file__).resolve().parent
HEX = re.compile(r"^[0-9a-f]{8,}$")
ARMS = {"STOP": "score_STOP_zero.csv", "CV": "score_CV_official.csv",
        "A1": "score_A1_ego_cmd.csv", "ECHO": "score_ECHO_ha0_ext.csv",
        "A2": "score_A2_vision_pure.csv", "A4": "score_A4_blind_ego_cmd.csv"}


def read(fn):
    return {r["token"]: r for r in csv.DictReader(open(RAW / fn, newline="", encoding="utf-8"))
            if HEX.match(r.get("token") or "")}


def main() -> int:
    arms = {a: read(fn) for a, fn in ARMS.items()}
    tbl = [ln.split(",") for ln in
           (HERE / "warmup_token_v0.csv").read_text().splitlines()[1:]]
    clust = {r[0]: r[2] for r in tbl}
    toks = sorted(set.intersection(*(set(a) for a in arms.values())) & set(clust))
    fired = {t for t in toks
             if all(float(arms[a][t]["ego_progress_stage_two"]) == 1.0 for a in arms)}
    sel = [t for t in toks if t not in fired]
    by = collections.defaultdict(list)
    for t in sel:
        by[clust[t]].append(float(arms["STOP"][t]["score"]) - float(arms["CV"][t]["score"]))
    keys = list(by)
    point = st.mean(x for k in keys for x in by[k])
    print(f"not-fired stratum: {len(sel)} scenes, {len(keys)} clusters, point {point:.4f}")

    for B in (10_000,):
        los, his = [], []
        for seed in range(12):                      # 12 independent streams
            rng = random.Random(1000 + seed)
            ms = []
            for _ in range(B):
                ms.append(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by[k]))
            ms.sort()
            los.append(ms[int(0.025 * B)])
            his.append(ms[int(0.975 * B)])
        print(f"\nB={B:,}  over 12 streams")
        print(f"  lo: mean {st.mean(los):.4f}  sd {st.stdev(los):.5f}  "
              f"range [{min(los):.4f}, {max(los):.4f}]  spread {max(los)-min(los):.4f}")
        print(f"  hi: mean {st.mean(his):.4f}  sd {st.stdev(his):.5f}  "
              f"range [{min(his):.4f}, {max(his):.4f}]  spread {max(his)-min(his):.4f}")
        print(f"  => lo > 0 in {sum(1 for x in los if x > 0)}/12 streams "
              f"(the VERDICT, which is what must be stable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
