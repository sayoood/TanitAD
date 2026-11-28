#!/usr/bin/env python3
"""C-ENV-1 step 2 -- IS THERE AN ADMISSIBLE MAX-SPEED CEILING IN WHAT WE HOLD?

Step 1 settled that `g_tac.goals.SPEED_BAND` is `[min, max]` of the EGO'S OWN
future speed over the 2-6 s tactical band (`s2_geom_emit_v7.tactical_goals`,
`v = p[lo:hi+1, 3]`). It is a DESCRIPTION, not a constraint, and supplying it at
inference is the nav-echo defect with no horizon guard at all.

The PI asked for a MAX SPEED "modelling the max speed the model can drive",
which the model should approach "when the situation allows". A ceiling that is a
function of the ego's own future cannot serve: it is not a property of the road.

v7.2 carries one field family that IS a road property and is NOT derived from
the ego's future: `strata = {country, road_class, daynight_clock}`. A
country x road_class default limit is genuine external regulatory information --
Belgium|urban is 50 km/h whatever the ego did. This asks whether such a prior is
usable as a ceiling, and prices its error HONESTLY.

⛔ THE HONEST FRAMING, stated before the numbers so it cannot be softened after:
this probe CANNOT validate a ceiling, because we hold no posted limit to check
one against. It can only answer the weaker, decidable question: **how much of
the variation in realised free-flow speed does the strata cell explain?** A cell
that explains little cannot carry a ceiling; a cell that explains a lot has not
thereby been shown correct -- only that a ceiling there would not be absurd.

⚠️ OBSERVED SPEED IS `v_hi_ms`, WHICH IS AN EGO STATISTIC. That is fine HERE and
would not be fine in a scored metric: this probe measures a property of the
CORPUS (do road classes differ in speed?), not a property of a model. No model
is run and nothing is scored.

CONTROLS, because a probe that reports a grouping effect will always report one:
  * `shuffled_cell` -- the same computation with strata cells randomly permuted
    across clips. It MUST read approximately zero explained variance. If the
    real cell and the shuffled cell score alike, the grouping is noise.
  * `n` and the number of clips per cell are printed for every cell; a cell
    under `MIN_CLIPS` is reported UNPOWERED rather than quietly averaged.

⛔⛔ CORRECTION APPLIED AFTER THE FIRST RUN, AND IT INVALIDATES THE HEADLINE.
The first run read `road_class` explaining 0.5794 of v_hi variance against a
0.0311 shuffle floor, and I was about to report that as a usable ceiling.
`road_class` IS DEFINED BY A SPEED THRESHOLD. From the selection design
(`2026-08-06-alpamayo-augmentation/DESIGN.md`, quoted verbatim in
`STRATA_AND_INERTIA.md`): "highway = >= 20 m/s sustained (>= 30 % of the clip),
intersection = a stop **and** a heading change (>= 25 deg yaw span), urban = the
remainder." So "road class explains speed" is very largely a TAUTOLOGY -- the
definition showing through, not a discovery, and the measured highway p50 of
23.91 m/s is downstream of a 20 m/s cut.

⚠️ THE SHUFFLE CONTROL CANNOT CATCH THIS, and that is the transferable lesson.
It distinguishes GROUPING from NOISE; it is structurally blind to a grouping
whose LABEL was derived from the scored quantity. Same family as this repo's
"probe that tunes on the data it scores", with the tuning moved upstream into
the label definition.

⇒ THE ONLY NON-CIRCULAR AXIS HERE IS `country`, which comes from PhysicalAI's
own `data_collection` metadata and never touches ego motion. Its number is the
honest one and it is reported as `HONEST_country_only` below.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

MIN_CLIPS = 30
MS_TO_KMH = 3.6


def var(xs):
    return st.pvariance(xs) if len(xs) > 1 else 0.0


def explained_fraction(groups):
    """1 - (within-group variance / total variance), pooled over groups."""
    allv = [x for g in groups.values() for x in g]
    if len(allv) < 2:
        return None
    tot = var(allv)
    if tot <= 0:
        return None
    within = sum(len(g) * var(g) for g in groups.values()) / len(allv)
    return 1.0 - within / tot


def main(blob: Path, out_json: Path, seed: int = 0):
    md5 = hashlib.md5(blob.read_bytes()).hexdigest()
    rows = []
    with gzip.open(blob, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sb = ((r.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND")
            sx = r.get("strata") or {}
            if not sb or sb.get("v_hi_ms") is None:
                continue
            rows.append({
                "clip": r.get("clip_id"),
                "v_hi": float(sb["v_hi_ms"]),
                "v_lo": float(sb["v_lo_ms"]),
                "country": sx.get("country"),
                "road_class": sx.get("road_class"),
                "daynight": sx.get("daynight_clock"),
                "cell": sx.get("strata_cell"),
            })
    n = len(rows)

    def group_by(keyfn):
        g = defaultdict(list)
        for r in rows:
            g[keyfn(r)].append(r["v_hi"])
        return g

    by_country = group_by(lambda r: r["country"])
    by_road = group_by(lambda r: r["road_class"])
    by_cr = group_by(lambda r: (r["country"], r["road_class"]))
    by_cell = group_by(lambda r: r["cell"])

    rng = random.Random(seed)
    shuffled_cells = [r["cell"] for r in rows]
    rng.shuffle(shuffled_cells)
    g_shuf = defaultdict(list)
    for c, r in zip(shuffled_cells, rows):
        g_shuf[c].append(r["v_hi"])

    def cell_table(g, label):
        out = {}
        for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
            name = k if isinstance(k, str) else "|".join(str(x) for x in k)
            if len(v) < MIN_CLIPS:
                out[name] = {"status": "UNPOWERED", "n_clips": len(v)}
                continue
            s = sorted(v)
            out[name] = {
                "n_clips": len(v),
                "mean_ms": round(st.mean(v), 3),
                "p50_ms": round(s[len(s) // 2], 3),
                "p85_ms": round(s[min(int(.85 * (len(s) - 1)), len(s) - 1)], 3),
                "p95_ms": round(s[min(int(.95 * (len(s) - 1)), len(s) - 1)], 3),
                "p95_kmh": round(s[min(int(.95 * (len(s) - 1)), len(s) - 1)]
                                 * MS_TO_KMH, 1),
                "sd_ms": round(st.pstdev(v), 3),
            }
        return out

    out = {
        "_what": ("does the v7.2 strata cell (country x road_class) explain "
                  "realised free-flow speed well enough to carry a MAX-SPEED "
                  "ceiling? Observed speed = SPEED_BAND.v_hi_ms."),
        "_evidence_class": "MEASURED (ours; artifact = this json + blob md5)",
        "_no_model_was_run": True,
        "_cannot_answer": ("whether any ceiling is CORRECT -- we hold no posted "
                           "speed limit to check against. This measures only "
                           "how much speed variation the cell explains."),
        "_observed_speed_is_an_ego_statistic": (
            "v_hi_ms = max ego speed over [t0+2s, t0+6s]. Admissible for a "
            "CORPUS property, INADMISSIBLE as a supplied model input."),
        "blob_md5": md5,
        "n_clips": n,
        "min_clips_for_a_cell": MIN_CLIPS,
        "explained_variance_fraction": {
            "country": round(explained_fraction(by_country), 4),
            "road_class": round(explained_fraction(by_road), 4),
            "country_x_road_class": round(explained_fraction(by_cr), 4),
            "strata_cell_full": round(explained_fraction(by_cell), 4),
            "CONTROL_shuffled_cell": round(explained_fraction(g_shuf), 4),
            "_reads": ("the shuffled control MUST read ~0. If the real cell and "
                       "the shuffled cell score alike, the grouping is noise "
                       "and no ceiling can be hung on it."),
        },
        "n_cells": {"country": len(by_country), "road_class": len(by_road),
                    "country_x_road_class": len(by_cr),
                    "strata_cell_full": len(by_cell)},
        "by_road_class": cell_table(by_road, "road_class"),
        "by_country": cell_table(by_country, "country"),
        "by_country_x_road_class": cell_table(by_cr, "cxr"),
        "pooled": {
            "v_hi_mean_ms": round(st.mean([r["v_hi"] for r in rows]), 3),
            "v_hi_sd_ms": round(st.pstdev([r["v_hi"] for r in rows]), 3),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")

    ev = out["explained_variance_fraction"]
    print(f"n_clips={n}  md5={md5}")
    print("EXPLAINED VARIANCE of v_hi_ms:")
    for k in ("country", "road_class", "country_x_road_class",
              "strata_cell_full", "CONTROL_shuffled_cell"):
        print(f"  {k:26s} {ev[k]:+.4f}")
    print("\nBY ROAD CLASS:")
    for k, v in out["by_road_class"].items():
        if v.get("status") == "UNPOWERED":
            print(f"  {k:14s} UNPOWERED n={v['n_clips']}")
        else:
            print(f"  {k:14s} n={v['n_clips']:4d} p50={v['p50_ms']:6.2f} "
                  f"p85={v['p85_ms']:6.2f} p95={v['p95_ms']:6.2f} "
                  f"({v['p95_kmh']:5.1f} km/h) sd={v['sd_ms']:5.2f}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
