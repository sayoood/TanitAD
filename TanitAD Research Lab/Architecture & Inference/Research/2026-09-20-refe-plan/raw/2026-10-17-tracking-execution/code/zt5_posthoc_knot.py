#!/usr/bin/env python3
"""zt5_posthoc_knot.py -- PREREG_ZT5 Amendment 2 (REPORTED ONLY, decides nothing): F_c split into F_c8 (tangent at the
last knot only) and F_c17 (knots 1-7 only), as DAC rescue on the navtest 144 tracker-made picks (POST-HOC) and on the
dev tracker-made pick / M-fan units.  Gates: base sim DAC reproduces the unit's recorded base (0) and full F_c
reproduces the recorded CF-H on every unit.

    PYTHONPATH=<navsim>;<refe>;<this dir> navsim-crun python zt5_posthoc_knot.py --c3 <c3 dir> --nav ../data/nav \
        --dev ../data/dev --out ../data/posthoc_knot.json
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt5_lib as L  # noqa: E402
import zt5_sim as Z  # noqa: E402

MC_ROOT = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest")
SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets"


def fix_knots(p8, knots):
    p = np.array(p8, np.float64, copy=True)
    tau, ok = L.tangents(p)
    m = np.zeros(8, bool)
    m[list(knots)] = True
    m &= ok
    p[m, 2] = p[m, 2] + L.wrap(tau[m] - p[m, 2])
    return p


def run_unit(p8, ego, dm, idcs):
    out = {}
    for name, plan in (("base", p8), ("Fc", L.fix_heading_tangent(p8)), ("Fc8", fix_knots(p8, [7])),
                       ("Fc17", fix_knots(p8, range(7)))):
        out[name] = float(Z.dac(Z.simulate(Z.raw_states(plan[None], ego), ego), ego, dm, idcs)[0])
    return out


def summarize(rows, key):
    if not rows:
        return {"n": 0}
    import zero_tail as zt
    logs = sorted({r["log"] for r in rows})
    li = np.array([logs.index(r["log"]) for r in rows])
    b = zt.Boot(li, len(logs))
    res = {"n": len(rows), "n_logs": len(logs),
           "gate_base_matches": all(r["dac"]["base"] == r["exp_base"] for r in rows),
           "gate_Fc_matches_recorded_H": all(r["dac"]["Fc"] == r["exp_H"] for r in rows)}
    for k in ("Fc", "Fc8", "Fc17"):
        x = np.array([r["dac"][k] for r in rows], float)
        res[f"rescue_pct_{k}"] = b.ratio(x, np.ones_like(x), 100.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", required=True)
    ap.add_argument("--nav", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    T = np.load(Path(a.c3) / "table.npz")
    ti = {str(t): i for i, t in enumerate(T["token"])}
    nav_rows = []
    for r in (json.loads(x) for f in sorted(glob.glob(f"{a.nav}/nav_*.jsonl")) for x in open(f, encoding="utf-8")):
        if r["kind"] != "TM":
            continue
        mc = pickle.loads(lzma.decompress(open(MC_ROOT / r["log"] / "unknown" / r["token"] / "metric_cache.pkl", "rb").read()))
        i = ti[r["token"]]
        p8 = T["proposals"][i, T["pick"][i]].astype(np.float64)
        nav_rows.append({"log": r["log"], "token": r["token"], "exp_base": r["cf"]["base"], "exp_H": r["cf"]["H"],
                         "dac": run_unit(p8, mc.ego_state, mc.drivable_area_map, Z.drivable_idx(mc.drivable_area_map))})
    print(f"navtest done {len(nav_rows)} {time.time() - t0:.0f}s", flush=True)
    import navtrain_scenarios as NS
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    import navsim_pdm_targets as PT
    units = [json.loads(x) for f in sorted(glob.glob(f"{a.dev}/dev_*of2_units.jsonl")) for x in open(f, encoding="utf-8")]
    units = [u for u in units if u["kind"] in ("pick", "fan")]
    want = {}
    for u in units:
        want.setdefault((u["log"], u["token"], u["step"], u["rank"]), []).append(u)
    sets = {}
    for fp in sorted(glob.glob(f"{SETS}/onpolicy_*.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            i = ln.index('"token": "') + 10
            tok = ln[i:ln.index('"', i)]
            if not any(k[1] == tok for k in want):
                continue
            d = json.loads(ln)
            k = (d["log_name"], d["token"], int(d["step"]), int(d["rank"]))
            if k in want:
                sets[k] = np.concatenate([np.asarray(d["traj"], np.float64), np.asarray(d["yaw"], np.float64)[..., None]], -1)
    dbs = NS.index_dbs()
    dev_rows = {"pick": [], "fan": []}
    seen = set()
    by_log = {}
    for k in want:
        by_log.setdefault(k[0], []).append(k)
    for lg, ks in sorted(by_log.items()):
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(dbs[lg], sorted({k[1] for k in ks}),
                                                                       history_rows=40, future_rows=100)}
        for k in ks:
            sc = scs[k[1]]
            ego = sc.initial_ego_state
            dm = PDMDrivableMap.from_simulation(sc.map_api, ego, 100)
            idcs = Z.drivable_idx(dm)
            P8 = PT.executed_plans8(sets[k], repair=True)
            for u in want[k]:
                tag = (k, u["j"], u["kind"])
                if tag in seen:
                    continue
                seen.add(tag)
                dev_rows[u["kind"]].append({"log": lg, "token": k[1], "exp_base": u["cf"]["base"], "exp_H": u["cf"]["H"],
                                            "dac": run_unit(P8[u["j"]], ego, dm, idcs)})
    res = {"amendment": "PREREG_ZT5 Amendment 2 -- REPORTED ONLY, decides nothing",
           "navtest_144_POSTHOC": summarize(nav_rows, "nav"),
           "dev_tracker_made_picks": summarize(dev_rows["pick"], "pick"),
           "dev_fan_tracker_made": summarize(dev_rows["fan"], "fan"), "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
