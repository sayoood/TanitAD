"""RL3 — render rl_main.json into the report tables (markdown to stdout)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TARGETS = ("ego_v0", "lead_present", "lead_gap", "lead_closing",
           "n_agents_grid", "n_agents_any")


def seed_stats(td: dict):
    rows = [v for k, v in td.items() if k.startswith("seed")]
    if not rows:
        return None
    r2 = np.array([r["r2_ceiling"] for r in rows], float)
    pv = [r.get("r2_ceiling_partial_v0") for r in rows]
    pv = np.array([x for x in pv if x is not None], float)
    k1 = [bool(r["K1_PASSES"]) for r in rows]
    dg = [bool(r["K1_DEGENERATE"]) for r in rows]
    edge = [bool(r["alpha_at_grid_edge"]) for r in rows]
    return {"r2c_mean": float(r2.mean()), "r2c_min": float(r2.min()),
            "r2c_max": float(r2.max()),
            "pv_mean": float(pv.mean()) if pv.size else None,
            "k1_pass": sum(k1), "k1_degen": sum(dg), "n_seeds": len(rows),
            "alpha_edge": sum(edge),
            "n_eval": rows[0]["n_eval"], "n_train": rows[0]["n_train"],
            "auc": rows[0].get("auc")}


def main():
    d = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    print(f"wall_s {d['wall_s']}  split {d['split']}  rp_dim {d['rp_dim']}")
    arm_order = ["refc_base_pooled", "refc_xl_pooled",
                 "refc_base_pooled_seq", "refc_xl_pooled_seq"]
    print("\n## MAIN LADDER — r2_ceiling mean [min..max] over 5 RP seeds "
          "(v0-partial r² in parens)\n")
    hdr = "| target | n_ev | " + " | ".join(arm_order) + " | v0_proxy |"
    print(hdr)
    print("|" + "---|" * (len(arm_order) + 3))
    for t in TARGETS:
        cells = []
        n_ev = None
        for a in arm_order:
            s = seed_stats(d["arms"][a]["targets"][t])
            n_ev = s["n_eval"]
            pv = f" (pv {s['pv_mean']:.4f})" if s["pv_mean"] is not None else ""
            stamp = ""
            if s["k1_degen"]:
                stamp += f" ⚠K1_DEGEN {s['k1_degen']}/{s['n_seeds']}"
            if s["k1_pass"] < s["n_seeds"]:
                stamp += f" K1 {s['k1_pass']}/{s['n_seeds']}"
            auc = f" AUC {s['auc']}" if s.get("auc") else ""
            cells.append(f"{s['r2c_mean']:.5f} "
                         f"[{s['r2c_min']:.5f}..{s['r2c_max']:.5f}]"
                         f"{pv}{auc}{stamp}")
        pr = d["v0_proxy"]["targets"][t]
        cells.append(f"{float(np.corrcoef([0])[0][0]) if False else ''}"
                     f"{pr['r2_ceiling']:.5f}")
        print(f"| {t} | {n_ev} | " + " | ".join(cells) + " |")

    print("\n## CONTROLS (seed0) — PLANT / NOISE / YPERM r2_ceiling\n")
    print("| target | arm | PLANT | NOISE | YPERM |")
    print("|---|---|---|---|---|")
    for t in TARGETS:
        for a in ("refc_base_pooled", "refc_xl_pooled",
                  "refc_base_pooled_seq", "refc_xl_pooled_seq"):
            td = d["arms"][a]["targets"][t]
            print(f"| {t} | {a} | {td['PLANT']['r2_ceiling']:.5f} | "
                  f"{td['NOISE']['r2_ceiling']:.5f} | "
                  f"{td['YPERM']['r2_ceiling']:.5f} |")

    print("\n## PAIRED DELTAS (r2_ceiling, arm A − arm B; per-seed "
          "median delta with the seed-range of CIs)\n")
    for key, per_seed in d["paired_deltas"].items():
        for kind in ("raw", "partial_v0"):
            vals = [v[kind] for v in per_seed.values() if v.get(kind)]
            if not vals:
                continue
            dd = np.array([v["delta"] for v in vals])
            lo = min(v["lo"] for v in vals)
            hi = max(v["hi"] for v in vals)
            sep = sum(1 for v in vals if v["separated"])
            print(f"{key} [{kind}]  delta med {np.median(dd):+.5f} "
                  f"(seeds {dd.min():+.5f}..{dd.max():+.5f})  CI-envelope "
                  f"[{lo:+.5f}, {hi:+.5f}]  separated {sep}/{len(vals)}")


if __name__ == "__main__":
    main()
