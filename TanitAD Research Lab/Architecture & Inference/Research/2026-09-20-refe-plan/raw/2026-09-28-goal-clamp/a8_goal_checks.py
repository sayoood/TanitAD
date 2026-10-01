#!/usr/bin/env python3
"""SPEC_NAVTEST Amendment 8 -- the CPU checks that come BEFORE any GPU stage (no model output is produced here):

  PARITY  the patched planner with the switch at its DEFAULT (OFF) returns a goal BIT-IDENTICAL to the base planner's on
          every token checked (the 1,123 selection tokens + the 422 fresh tokens), and the default really is OFF.
          Snapshot 018's eval and every run after the landing use planner.py's defaults, so this is the landing gate.
  GATE A  (census re-check) with the switch ON, the trigger fires on EVERY fresh token and on NO token the census puts
          <= 20 m; the trigger distance equals the census value; an untriggered token's ON goal is bit-identical to
          OFF. The fallback's kind / route length / goal are recorded per token (the model never sees them here).
  FRAMES  every fresh token resolves to 4 camera frames that exist on disk (zip members checked).

The model is replaced by nn.Identity() in BOTH planners (the goal path never touches it; building ViT-L twice would
only cost memory). Scenarios come from navtrain_scenarios exactly as eval/refe_navtest_seam.py builds them.
    python a8_goal_checks.py --root <package root with the PATCHED planner> --base <dir holding the BASE planner.py>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=str(HERE / "a8_goal_checks.json"))
    a = ap.parse_args()
    root, base = Path(a.root).resolve(), Path(a.base).resolve()
    if os.environ.get("REFE_A8C_CHILD") != "1":
        sys.path.insert(0, str(PKG / "eval"))
        import eval_checkpoint as EC
        env = EC.env_driverl()
        env["REFE_A8C_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(root / "refe"), str(root / "eval"),
                                             str(root / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), "--root", str(root), "--base", str(base),
                                "--out", str(Path(a.out).resolve())], cwd=str(root / "eval"), env=env)
    for p in (root / "refe", root / "eval", root / "code"):
        sys.path.insert(0, str(p))
    import numpy as np
    import torch
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    new = load("planner_a8_new", root / "refe" / "planner.py")
    old = load("planner_a8_base", base / "planner.py")
    for m in (new, old):
        m.REFe = lambda cfg: torch.nn.Identity()                       # the goal path never touches the model
    sha = {"new": hashlib.sha256((root / "refe" / "planner.py").read_bytes()).hexdigest(),
           "base": hashlib.sha256((base / "planner.py").read_bytes()).hexdigest()}
    frames_root = "D:/Projects/TanitAD/data/refe_navtest/frames"
    mk = lambda M, **kw: M.REFePlanner(checkpoint=None, images_root=frames_root, db_dir=SEAM.TEST_DB_DIR,  # noqa: E731
                                       backbone="vitl16", device="cpu", **kw)
    p_old, p_off, p_on = mk(old), mk(new), mk(new, sanitize_goal=True)
    default_off = (p_off.sanitize_goal is False and new.REFePlanner.SANITIZE_GOAL is False)
    census = json.load(open(HERE / "route_cover_census_navtest_full.json", encoding="utf-8"))["rows"]
    fresh = json.load(open(HERE / "amendment8_fresh_set.json", encoding="utf-8"))
    sel = json.load(open(PKG / "raw" / "2026-09-28-m6b-tangent" / "tokens_1123.json", encoding="utf-8"))
    exp = SEAM.json.load(SEAM.gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = sorted(set(sel["tokens"]) | set(fresh["fresh_tokens"]))
    fresh_set = set(fresh["fresh_tokens"])
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    rows, zips = {}, {}
    t0 = time.time()
    for li, (lg, lt) in enumerate(sorted(by_log.items())):
        for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), lt, history_rows=1,
                                             future_rows=80):
            tok = sc._initial_lidar_token
            ego = sc.get_ego_state_at_iteration(0)
            goals = {}
            for tag, p in (("base", p_old), ("off", p_off), ("on", p_on)):
                p._scenario = sc
                p.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                   mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                if tag == "on":
                    p.driving_command = exp[tok]["ego_statuses"][-1]["driving_command"]
                    p.goal_diag = None
                goals[tag] = p._goal_for(ego)
            r = {"log": lg, "fresh": tok in fresh_set,
                 "off_equals_base": bool(torch.equal(goals["off"], goals["base"])),
                 "on_equals_off": bool(torch.equal(goals["on"], goals["off"])),
                 "goal_off": [float(x) for x in goals["off"].reshape(-1)],
                 "goal_on": [float(x) for x in goals["on"].reshape(-1)], "diag": p_on.goal_diag}
            if tok in fresh_set:
                paths = p_off.frames.resolve(p_off._log_hint, int(ego.time_point.time_us)) or []
                ok = len(paths) == 4
                for q in paths:
                    if "::" in q:
                        zp, mem = q.split("::", 1)
                        if zp not in zips:
                            zips[zp] = set(zipfile.ZipFile(zp).namelist()) if os.path.exists(zp) else set()
                        ok &= mem in zips[zp]
                    else:
                        ok &= os.path.exists(q)
                r["frames_ok"], r["n_frames"] = bool(ok), len(paths)
            rows[tok] = r
        if (li + 1) % 10 == 0:
            print(f"  [{li + 1}/{len(by_log)}] {len(rows)} tokens, {time.time() - t0:.0f} s", flush=True)
    missing = sorted(set(toks) - set(rows))
    cen = {t: census[t]["ego_to_route_m"] for t in rows}
    trig = {t: bool((rows[t]["diag"] or {}).get("triggered")) for t in rows}
    dist_err = max(abs((rows[t]["diag"] or {}).get("ego_to_route_m", 1e9) - cen[t]) for t in rows) if rows else None
    fr = [t for t in rows if rows[t]["fresh"]]
    kinds: dict = {}
    for t in fr:
        k = (rows[t]["diag"] or {}).get("fallback")
        kinds[k] = kinds.get(k, 0) + 1
    p2_on = [float(np.hypot(*rows[t]["goal_on"][2:4])) for t in fr]
    checks = {
        "PARITY default is OFF (instance and class)": default_off,
        "PARITY off == base, bit-identical, every token": bool(rows) and all(r["off_equals_base"] for r in rows.values()),
        "GATE A trigger fires on EVERY fresh token": bool(fr) and all(trig[t] for t in fr),
        "GATE A trigger == (census > 20 m) on every token checked": all(trig[t] == (cen[t] > 20.0) for t in rows),
        "GATE A trigger distance == census distance (max abs diff <= 1e-6 m)": dist_err is not None and dist_err <= 1e-6,
        "GATE A untriggered: ON goal bit-identical to OFF": all(rows[t]["on_equals_off"] for t in rows if not trig[t]),
        "GATE A triggered: ON goal differs from OFF": all(not rows[t]["on_equals_off"] for t in rows if trig[t]),
        "FRAMES every fresh token has 4 frames on disk": bool(fr) and all(rows[t].get("frames_ok") for t in fr),
        "every requested token was built": not missing,
    }
    out = {"sha256_planner": sha, "n_tokens": len(rows), "n_fresh": len(fr), "missing": missing[:20],
           "n_missing": len(missing), "trigger_distance_max_abs_diff_m": dist_err, "fresh_fallback_kinds": kinds,
           "fresh_on_goal_p2_m": {"median": float(np.median(p2_on)) if p2_on else None,
                                  "max": float(np.max(p2_on)) if p2_on else None},
           "fresh_off_goal_p2_m_median": float(np.median([np.hypot(*rows[t]["goal_off"][2:4]) for t in fr])) if fr else None,
           "checks": checks, "failed": [k for k, v in checks.items() if not v], "seconds": round(time.time() - t0, 1),
           "rows": rows}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "checks")}, default=float))
    print("ZZA8_CHECKS_" + ("OK" if not out["failed"] else "FAIL"))
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
