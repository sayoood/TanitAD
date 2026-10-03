#!/usr/bin/env python3
"""CPU only, NO model: the planner's OWN `_goal_for` under every `goal_fix` mode, for every navtest token.

The planner object is built without its network (`REFePlanner.__new__` + the attributes `_goal_for` reads), so the code
path measured is the deployed one, not a re-implementation. Three checks ride on it:
  REGRESSION   goal_fix=None must reproduce goal_census_navtest_full.json (2026-10-01, the pre-patch code) EXACTLY.
  CROSS-CHECK  goal_fix="pdm_route" vs the goal re-derived from NAVSIM's own metric-cache `centerline` (built by NAVSIM's
               PDM-Closed in another venv and process, pdm_route_goal_probe.json) -- an independent route to the same goal.
  SCOPE        which tokens' goals change under each mode -> exactly the tokens a GPU confirmation must forward.
    python goal_fix_census.py            (REFE_DRIVE=E: while the external drive carries that letter)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
DRV = os.environ.get("REFE_DRIVE", "D:")
sys.path[:0] = [str(PKG / "eval"), str(PKG / "refe"), str(PKG / "code")]
MODES = (None, "pdm_route", "navgoal_straight", "navgoal_arc")


def fix(s):
    return s.replace("D:/", f"{DRV}/").replace("D:\\", f"{DRV}\\") if isinstance(s, str) else s


def main() -> int:
    import eval_checkpoint as EC
    out_p = HERE / "goal_fix_census.json"
    if os.environ.get("REFE_GF_CHILD") != "1":
        env = {k: fix(v) for k, v in EC.env_driverl().items()}
        env.update(REFE_GF_CHILD="1", REFE_DRIVE=DRV,
                   PYTHONPATH=os.pathsep.join([env.get("PYTHONPATH", ""), str(PKG / "refe"), str(PKG / "eval"), str(PKG / "code")]))
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=str(PKG / "eval"), env=env)
    import gzip
    import navtrain_scenarios as NS
    import planner as PLM
    import refe_navtest_seam as SEAM
    from model import REFeConfig
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    SEAM.W3_EXPORT, SEAM.TEST_DB_DIR = fix(SEAM.W3_EXPORT), fix(SEAM.TEST_DB_DIR)
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    old = json.load(open(HERE / "goal_census_navtest_full.json", encoding="utf-8"))["rows"]
    toks = list(exp)
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)

    def bare(mode):
        p = PLM.REFePlanner.__new__(PLM.REFePlanner)
        p.cfg = REFeConfig.for_backbone("vitl16")
        p.horizon_s, p.min_speed_mps = 12.0, 5.0
        p.sanitize_goal, p.goal_fix, p.driving_command, p.goal_diag = False, mode, None, None
        p._init, p._route_poly, p._route_fix_diag = None, None, None
        p._scenario, p._log_hint = None, None               # read by `initialize` (frame lookup only)
        return p
    planners = {m: bare(m) for m in MODES}
    rows = {}
    for li, (lg, ts) in enumerate(sorted(by_log.items())):
        try:
            for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), ts, history_rows=1, future_rows=80):
                tok = sc._initial_lidar_token
                ego = sc.get_ego_state_at_iteration(0)
                r = {"log": lg}
                for m, p in planners.items():
                    p.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                       mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                    p.goal_diag = None
                    g = p._goal_for(ego)
                    r[str(m)] = {"goal": [float(x) for x in g.reshape(-1).tolist()], "diag": p.goal_diag}
                rows[tok] = r
        except Exception as e:  # noqa: BLE001
            if not any("error" in r for r in rows.values()):
                import traceback
                traceback.print_exc()                          # the FIRST failure in full; the rest are counted
            for t in ts:
                rows.setdefault(t, {"log": lg, "error": repr(e)[:300]})
        if len(sys.argv) > 1 and sys.argv[1] == "--one-log":
            break
        if (li + 1) % 10 == 0 or li + 1 == len(by_log):
            print(f"  [{li + 1}/{len(by_log)}] rows {len(rows)}", flush=True)
    ok = {t: r for t, r in rows.items() if "error" not in r}
    if not ok:
        print("ZZGFC_FAIL every row errored"); return 1
    # REGRESSION: None vs the pre-patch census
    reg = [float(np.abs(np.asarray(r["None"]["goal"]) - np.asarray(old[t]["goal"])).max())
           for t, r in ok.items() if "goal" in old.get(t, {})]
    summ = {"n_tokens": len(toks), "n_rows": len(rows), "n_error": len(rows) - len(ok),
            "regression_none_vs_prepatch": {"n": len(reg), "max_abs": max(reg) if reg else None,
                                            "n_nonzero": int(sum(x > 0 for x in reg))}}
    for m in MODES[1:]:
        d = np.array([float(np.abs(np.asarray(r[m]["goal"]) - np.asarray(r["None"]["goal"])).max()) for r in ok.values()])
        ch = [t for t, r in ok.items() if np.abs(np.asarray(r[m]["goal"]) - np.asarray(r["None"]["goal"])).max() > 0]
        summ[m] = {"n_goal_changed": len(ch), "n_changed_over_1m": int((d > 1.0).sum()), "max_change_m": float(d.max())}
        if m == "pdm_route":
            summ[m]["n_route_ids_changed"] = int(sum(bool((r[m]["diag"] or {}).get("changed")) for r in ok.values()))
        else:
            summ[m]["n_triggered"] = int(sum(bool((r[m]["diag"] or {}).get("triggered")) for r in ok.values()))
        summ[m]["changed_tokens"] = ch
    # CROSS-CHECK vs NAVSIM's own metric-cache centerline goal
    pp = HERE / "pdm_route_goal_probe.json"
    if pp.exists():
        P = json.load(open(pp, encoding="utf-8"))["rows"]
        cc = {}
        for grp in ("old_trigger", "new_only", "control_untriggered_300"):
            e = [float(np.linalg.norm(np.asarray(ok[t]["pdm_route"]["goal"]) - np.asarray(P[t]["goal_pdm"])))
                 for t in P if P[t].get("group") == grp and t in ok and "goal_pdm" in P[t]]
            if e:
                cc[grp] = {"n": len(e), "median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)),
                           "max_m": float(max(e)), "n_over_5m": int(sum(x > 5 for x in e))}
        summ["crosscheck_pdm_route_vs_metric_cache_centerline"] = cc
    json.dump({"summary": summ, "rows": rows}, open(out_p, "w", encoding="utf-8", newline="\n"), indent=0)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "changed_tokens"} if isinstance(v, dict) else v)
                      for k, v in summ.items()}, indent=1))
    print("ZZGFC_DONE", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
