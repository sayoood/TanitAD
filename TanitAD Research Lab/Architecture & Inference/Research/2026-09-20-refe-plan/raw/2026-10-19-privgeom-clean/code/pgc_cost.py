"""PRIVGEOM-CLEAN cost: extraction time per log MEASURED on 2 of the 76 (G-LIVE), projected to the 76; the fit / gate stages measured in the SEEN rehearsal and scaled by the set ratio.  Label-free.
All timings are wall / process time on a LOADED box (many other jobs were running); the SAFE realised rate is the quiet-box floor.

    python pgc_cost.py --out run/cost.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = "C:/Users/Admin/par18/pgclean"
SAFE_RUN = "C:/Users/Admin/par18/privgeom/run"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cov = json.load(open(f"{ROOT}/run/coverage.json", encoding="utf-8"))
    n_sets, n_tok, n_logs = cov["n_sets"], cov["n_tokens"], cov["n_unseen_resolved"]
    rows = []
    for p in sorted(glob.glob(f"{SAFE_RUN}/scalars/*.timing.json")):
        t = json.load(open(p, encoding="utf-8"))
        rows.append((t["n_sets"], len(t["tokens"]), t["t_build_s"] + sum(x["t_parts_s"] + x["t_prep_s"] for x in t["tokens"]) + sum(s["t_score_s"] + s["t_geom_s"] for s in t["sets"])))
    R = np.array(rows, float)
    g = json.load(open(f"{ROOT}/run/glive/glive.json", encoding="utf-8"))
    G = np.array([(r["n_sets"], r["n_tokens"], r["t_build_s"] + sum(r["t_parts_s_per_token"]) + sum(r["t_prep_s_per_token"]) + sum(r["t_score_s_per_set"]) + sum(r["t_geom_s_per_set"]), r["wall_s"], r["cpu_s"]) for r in g["records"]], float)
    model = 0.20993873 * n_logs + 1.36141208 * n_tok + 0.52827370 * n_sets
    proj = {"SAFE cost model (20-set fit)": model, "SAFE realised s/set (41 logs, quiet-ish box)": float(R[:, 2].sum() / R[:, 0].sum() * n_sets),
            "G-LIVE component s/set (2 logs, LOADED box)": float(G[:, 2].sum() / G[:, 0].sum() * n_sets), "G-LIVE process CPU s/set (2 logs, LOADED box)": float(G[:, 4].sum() / G[:, 0].sum() * n_sets)}
    out = {"schema": "privgeom-clean-cost-v1", "evidence": "MEASURED (G-LIVE on 2 of the 76; PRIVGEOM-SAFE's realised timings on the 41); the projection is ESTIMATED (linear in sets)",
           "unseen": {"n_logs": n_logs, "n_sets": n_sets, "n_tokens": n_tok, "largest_log_sets": cov["sets_per_log_min_med_max"][2]},
           "glive": {"logs": g["logs"], "sets": G[:, 0].tolist(), "tokens": G[:, 1].tolist(), "component_s": G[:, 2].tolist(), "wall_s": G[:, 3].tolist(), "cpu_s": G[:, 4].tolist(),
                     "s_per_set_all_in_component": float(G[:, 2].sum() / G[:, 0].sum()), "s_per_set_all_in_cpu": float(G[:, 4].sum() / G[:, 0].sum()), "cpu_over_wall": float(G[:, 4].sum() / G[:, 3].sum())},
           "safe_realised": {"n_logs": len(R), "sets": float(R[:, 0].sum()), "tokens": float(R[:, 1].sum()), "component_s": float(R[:, 2].sum()), "cpu_h": float(R[:, 2].sum() / 3600), "s_per_set_all_in": float(R[:, 2].sum() / R[:, 0].sum())},
           "extraction_projection_s": proj, "extraction_projection_cpu_h": {k: v / 3600 for k, v in proj.items()}, "extraction_wall_h_two_workers": {k: v / 3600 / 2 for k, v in proj.items()},
           "extraction_cpu_h_range": [min(proj.values()) / 3600, max(proj.values()) / 3600], "cap_cpu_h_PRIVGEOM_SAFE": 16.0}
    fits = sorted(glob.glob(f"{ROOT}/run_rehearsal/fits/*.json"))
    if fits:
        sec = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8")).get("seconds") for p in fits}
        sets_seen = 4941
        fit_tasks = {k: v for k, v in sec.items() if not k.startswith("plantlabel")}
        out["rehearsal_fit_seconds_on_41_logs"] = sec
        out["fit_projection_76"] = {"scale": n_sets / sets_seen, "n_fit_tasks_in_the_read": 15, "plantlabel_tasks_in_gate": 4,
                                   "total_fit_seconds_if_all_tasks_present": float(sum(v for v in sec.values() if v) * n_sets / sets_seen), "n_tasks_measured": len(sec)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("extraction_cpu_h_range", "extraction_wall_h_two_workers", "extraction_projection_cpu_h")}, indent=1))
    if fits:
        print(out["fit_projection_76"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
