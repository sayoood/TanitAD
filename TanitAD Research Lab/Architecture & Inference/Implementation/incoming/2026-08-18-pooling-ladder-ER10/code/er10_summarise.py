"""E-R1-0 — render the banked ladder JSONs into the tables the report quotes.

⛔ THIS FILE COMPUTES NOTHING. It only reads `raw/er10_*.json`. Every number it
prints exists in a committed artifact, so any table in the report can be
re-derived without re-running a fit. A summariser that recomputes is a second
implementation and a place for the two to disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw"
ARMS = ["p40", "p10", "p4", "p1", "cells"]
RATIO = {"p40": "40:1", "p10": "10:1", "p4": "4:1", "p1": "1:1",
         "cells": "40:1 (DEPLOYED, learned proj)"}


def load(tag):
    p = RAW / f"er10_{tag}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def table_r2(d, key="r2_ceiling_mean", sdkey="r2_ceiling_sd", title=""):
    arms = [a for a in ARMS if a in d["arms"]]
    print(f"\n### {title}")
    print("| rung | " + " | ".join(f"**{RATIO[a]}**" for a in arms) + " |")
    print("|---|" + "---|" * len(arms))
    tg = list(d["arms"][arms[0]]["targets"])
    for t in tg:
        cells = []
        for a in arms:
            r = d["arms"][a]["targets"].get(t)
            if r is None:
                cells.append("—")
                continue
            v = r.get(key)
            s = r.get(sdkey)
            cells.append(f"{v:.4f}" + (f" ±{s:.4f}" if s else ""))
        n = d["arms"][arms[0]]["targets"][t]["n_eval"]
        print(f"| `{t}` (n={n}) | " + " | ".join(cells) + " |")


def table_seed0(d, field, title, fmt="%+.4f"):
    arms = [a for a in ARMS if a in d["arms"]]
    print(f"\n### {title}")
    print("| rung | " + " | ".join(f"**{RATIO[a]}**" for a in arms) + " |")
    print("|---|" + "---|" * len(arms))
    for t in list(d["arms"][arms[0]]["targets"]):
        cells = []
        for a in arms:
            r = d["arms"][a]["targets"].get(t)
            if r is None:
                cells.append("—")
                continue
            ps = r["per_seed"]
            k = sorted(ps)[0]
            v = ps[k].get(field)
            if v is None:
                cells.append("n/a")
                continue
            cells.append(fmt % v
                         + (" ⛔DEG" if ps[k].get("K1_DEGENERATE") else "")
                         + (" **PASS**" if field == "K1_delta"
                            and ps[k]["K1_PASSES"] else ""))
        print(f"| `{t}` | " + " | ".join(cells) + " |")


def table_deltas(d):
    if "deltas_vs_p40" not in d:
        return
    print("\n### Δ r²_ceiling vs the DEPLOYED 40:1 arm — paired episode-cluster "
          "bootstrap, per projection seed")
    print("| rung | arm | Δr² (mean over seeds) | ALL seeds separated & >0 | "
          "ALL seeds separated & >0 AFTER partialling `v0` | Δ MAE seed0 |")
    print("|---|---|---|---|---|---|")
    for t, row in d["deltas_vs_p40"].items():
        for arm, r in row.items():
            s0 = sorted(r["delta_mae_per_seed"])[0]
            mae = r["delta_mae_per_seed"][s0]
            print(f"| `{t}` | {RATIO.get(arm, arm)} | "
                  f"{r['delta_r2c_mean']:+.5f} | "
                  f"{'✅ YES' if r['ALL_SEEDS_SEPARATED_AND_POSITIVE'] else '⛔ no'} | "
                  f"{'✅ YES' if r['ALL_SEEDS_SEPARATED_POSITIVE_PARTIAL_V0'] else '⛔ no'} | "
                  f"{mae['delta']:+.4f} [{mae['lo']:+.4f}, {mae['hi']:+.4f}]"
                  f"{' sep' if mae['separated'] else ''} |")


def main() -> int:
    which = sys.argv[1:] or ["main", "null", "proxyv0", "pc_dist", "pc_local",
                             "pc_local2", "dino", "gate"]
    for tag in which:
        d = load(tag)
        if d is None:
            print(f"\n## {tag}: ABSENT")
            continue
        print(f"\n\n## {tag} — {d['arm_label']}")
        print(f"* tier `{d['eval_tier']}` · seeds {d['proj_seeds']} · "
              f"intercept_col `{d['ridge_intercept_col']}` · "
              f"n_boot {d['n_boot']} · estimator `{d['estimator']}` · "
              f"wall {d['wall_s']} s")
        if d.get("oracle"):
            print(f"* PLANTED ORACLE `{d['oracle']}` amp {d['oracle_amp_rel']}× "
                  f"token sd ({d['oracle_token_sd']:.6f}); "
                  f"PC-2OBJ cells {json.dumps(d['pc_2obj_cell_geometry'])}")
        if "reproduction_gate" in d:
            g = d["reproduction_gate"]
            print(f"* ⛔ REPRODUCTION GATE vs `{Path(g['against']).name}`: "
                  f"**{'PASS' if g['PASSED'] else 'FAIL'}** "
                  f"({len(g['checks'])} checks)")
        table_r2(d, title="r²_ceiling = corr(pred, truth)² — mean ± sd over "
                          "projection seeds")
        if any("r2_ceiling_partial_v0" in s
               for a in d["arms"].values()
               for t in a["targets"].values()
               for s in t["per_seed"].values()):
            print("\n### r²_ceiling with `v0` PARTIALLED OUT (seed 0) — the "
                  "C92 trivial-proxy control")
            table_seed0(d, "r2_ceiling_partial_v0", "", fmt="%.4f")
        table_seed0(d, "K1_delta", "K1 vs C-CONST (negative = beats the "
                                   "constant); ⛔DEG = C97 degeneracy guard")
        table_seed0(d, "pred_sd_over_gt_sd", "⛔ C97 GUARD — pred_sd / gt_sd "
                    "(a PASS below the floor is NOT quotable)", fmt="%.4f")
        table_deltas(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
