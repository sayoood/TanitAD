"""Effective cluster count for the navhard confirmation's strata.

An episode/scene-clustered bootstrap resamples clusters as EQUAL draws, so unequal cluster sizes
make the real power lower than the cluster COUNT suggests. The right statistic is the
inverse-Simpson / Kish effective count:

    n_eff = (sum n_i)^2 / sum n_i^2        (= k exactly when all clusters are equal)

The TrainingFlyWheel measured this for the perception bar (19 non-empty clusters -> n_eff 12.60,
66 %) and for the eligible pool (60 -> 56.89, 95 %), and did not take navhard. This bounds the
power of MY OWN headline, "CONFIRMED at 435 clusters" (f3fdcc9), so it is mine to check.

⛔ No re-run and no re-scoring: the cluster sizes are already in the banked join table and the
scored CSVs.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
SCORES = pathlib.Path("D:/Projects/TanitAD/taniteval/results/bench/navsim_v2/"
                      "navhard_two_stage/20260920T082848Z-navsim_v2-none-06e257/scores")
HEX = re.compile(r"^[0-9a-f]{8,}$")


def n_eff(sizes) -> float:
    tot = sum(sizes)
    return (tot * tot) / sum(s * s for s in sizes) if tot else 0.0


def main() -> int:
    tbl = [l.split(",") for l in
           (HERE / "navhard_token_v0.csv").read_text().splitlines()[1:]]
    clust = {r[0]: r[2] for r in tbl}

    arms = {}
    for p in sorted(SCORES.glob("*.csv")):
        d = {}
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                t = (r.get("token") or "")
                if HEX.match(t):
                    d[t] = r
        arms[p.stem] = d
    toks = sorted(set.intersection(*(set(a) for a in arms.values())) & set(clust))

    def ep(row, col):
        try:
            return float(row.get(col, ""))
        except (TypeError, ValueError):
            return None

    fired = {t for t in toks
             if all(ep(arms[a][t], "ego_progress_stage_two") == 1.0 for a in arms)}

    print(f"arms {sorted(arms)}   scenes {len(toks)}")
    print(f"{'stratum':<18} {'scenes':>7} {'clusters':>9} {'n_eff':>8} {'%':>6}  "
          f"{'min':>4} {'med':>4} {'max':>4}")
    out = {}
    for name, sel in (("clause_NOT_fired", [t for t in toks if t not in fired]),
                      ("clause_FIRED", [t for t in toks if t in fired]),
                      ("ALL", toks)):
        c = collections.Counter(clust[t] for t in sel)
        sizes = sorted(c.values())
        e = n_eff(sizes)
        out[name] = {"scenes": len(sel), "clusters": len(sizes), "n_eff": round(e, 2),
                     "pct_of_clusters": round(100 * e / len(sizes), 1),
                     "min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]}
        print(f"{name:<18} {len(sel):>7} {len(sizes):>9} {e:>8.2f} "
              f"{100*e/len(sizes):>5.1f}%  {sizes[0]:>4} {sizes[len(sizes)//2]:>4} "
              f"{sizes[-1]:>4}")

    ref = out["clause_NOT_fired"]
    print(f"\nPRIMARY stratum: {ref['clusters']} clusters -> n_eff {ref['n_eff']} "
          f"({ref['pct_of_clusters']}%)")
    print("=> the cluster count " + ("OVERSTATES power materially"
          if ref["pct_of_clusters"] < 80 else "is a fair summary; clusters are near-balanced"))
    (HERE / "neff.json").write_text(json.dumps(
        {"_what": "inverse-Simpson effective cluster count per navhard stratum",
         "_formula": "(sum n_i)^2 / sum n_i^2", "strata": out}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
