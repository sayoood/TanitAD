#!/usr/bin/env python3
"""SPEC_NAVTEST Amendment 8, validity gate (b): the unit test, with its mutation arms. No model output.

  B1  a COVERING route leaves the goal bit-identical (real scenario, token 014369205e025f0c: the ego's roadblock IS
      in its route, 0.7 m away; the goal is 34 / 68 m ahead)
  B2  the recorded 373.5 m case is REPLACED (token 3efebf87894a552e, log 2021.08.30.13.45.25_veh-40_01116_01336):
      triggered, trigger distance within 1 m of the traced 373.5 m, new goal p2 < 200 m and != the old one
  B3  a synthetic Y junction (start lane +x, successors LEFT / STRAIGHT / RIGHT): LEFT and RIGHT pick OPPOSITE
      successors; STRAIGHT and UNKNOWN pick the straight one; with no lane in the radius -> the straight-route goal
  B4  the switch's default is OFF
MUTATIONS (the planner's SOURCE is edited on a temp copy and re-imported; each must turn its target check RED):
  M1  delete the trigger check (`triggered = True`)            -> B1 RED
  M2  a trigger that never fires (`triggered = False`)         -> B2 RED
  M3  LEFT takes the most CLOCKWISE successor (max -> min)       -> B3 RED
  M4  the default switched ON (`SANITIZE_GOAL = True`)          -> B4 RED
    python a8_selftest.py --root <package root with the patched planner>   -> ZZA8_SELFTEST_OK / _FAIL + .json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
COVER = ("2021.05.25.14.16.10_veh-35_00083_00485", "014369205e025f0c")
FAR = ("2021.08.30.13.45.25_veh-40_01116_01336", "3efebf87894a552e")
MUTATIONS = {"M1": ("        triggered = d > GOAL_SANITIZE_D_M\n", "        triggered = True\n", "B1"),
             "M2": ("        triggered = d > GOAL_SANITIZE_D_M\n", "        triggered = False\n", "B2"),
             "M3": ("        return max(succs, key=lambda e: (dh(e), str(getattr(e, \"id\", \"\"))))\n",
                    "        return min(succs, key=lambda e: (dh(e), str(getattr(e, \"id\", \"\"))))\n", "B3"),
             "M4": ("    SANITIZE_GOAL = False\n", "    SANITIZE_GOAL = True\n", "B4")}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _P:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)


class _Path:
    def __init__(self, pts):
        self.discrete_path = [_P(x, y) for x, y in pts]


class _Edge:
    def __init__(self, eid, pts, outgoing=()):
        self.id, self.baseline_path, self.outgoing_edges = eid, _Path(pts), list(outgoing)


def _arc(x0, y0, h0, turn, length=25.0, n=11):
    """points along a circular arc of total heading change `turn` rad starting at (x0, y0) with heading h0"""
    pts = []
    for i in range(n):
        s = length * i / (n - 1)
        if abs(turn) < 1e-9:
            pts.append((x0 + s * math.cos(h0), y0 + s * math.sin(h0)))
        else:
            R = length / turn
            th = h0 + turn * i / (n - 1)
            pts.append((x0 + R * (math.sin(th) - math.sin(h0)), y0 - R * (math.cos(th) - math.cos(h0))))
    return pts


def run_checks(PL, SEAM, NS, M, A, PlannerInitialization, torch, scen):
    out = {}
    mk = lambda **kw: PL.REFePlanner(checkpoint=None, images_root="", db_dir=SEAM.TEST_DB_DIR,  # noqa: E731
                                     backbone="vitl16", device="cpu", **kw)
    off, on = mk(), mk(sanitize_goal=True)

    def goal(p, sc, cmd=(0, 1, 0, 0)):
        p._scenario = sc
        p.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                           mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
        p.driving_command = list(cmd)
        p.goal_diag = None
        return p._goal_for(sc.get_ego_state_at_iteration(0))
    g0, g1 = goal(off, scen["cover"]), goal(on, scen["cover"])
    d1 = on.goal_diag or {}
    out["B1"] = {"ok": bool(torch.equal(g0, g1)) and d1.get("triggered") is False,
                 "ego_to_route_m": d1.get("ego_to_route_m"), "goal": [round(float(x), 3) for x in g1.reshape(-1)]}
    f0, f1 = goal(off, scen["far"]), goal(on, scen["far"])
    d2 = on.goal_diag or {}
    p2_old, p2_new = float(torch.linalg.norm(f0.reshape(-1)[2:4])), float(torch.linalg.norm(f1.reshape(-1)[2:4]))
    out["B2"] = {"ok": d2.get("triggered") is True and abs(d2.get("ego_to_route_m", 0.0) - 373.5) < 1.0
                 and p2_new < 200.0 and not torch.equal(f0, f1),
                 "ego_to_route_m": d2.get("ego_to_route_m"), "p2_old_m": p2_old, "p2_new_m": p2_new,
                 "fallback": d2.get("fallback"), "route_len_m": d2.get("route_len_m")}
    # B3: a synthetic Y junction in the ego's own frame (anchor at the origin, heading +x)
    left = _Edge("L", _arc(40.0, 0.0, 0.0, +0.8))
    straight = _Edge("S", _arc(40.0, 0.0, 0.0, 0.0))
    right = _Edge("R", _arc(40.0, 0.0, 0.0, -0.8))
    start = _Edge("start", [(-5.0 + 5.0 * i, 0.0) for i in range(10)], outgoing=[right, straight, left])

    class _Map:
        def __init__(self, objs):
            self.objs = objs

        def get_proximal_map_objects(self, pt, r, layers):
            return {layers[0]: list(self.objs)}
    anchor = M.FrameRow(token=b"", token_hex="", timestamp_us=0, scene_token=None, x=0.0, y=0.0, yaw=0.0,
                        vx=5.0, vy=0.0, ax=0.0, ay=0.0, yaw_rate=0.0)
    pick = {c: PL.fallback_route(_Map([start]), anchor, c, M)[1].get("edges", [None, None])[1]
            for c in ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")}
    none_poly, none_diag = PL.fallback_route(_Map([]), anchor, "LEFT", M)
    out["B3"] = {"ok": pick == {"LEFT": "L", "STRAIGHT": "S", "RIGHT": "R", "UNKNOWN": "S"}
                 and none_poly is None and none_diag.get("fallback") == "straight", "picks": pick,
                 "no_lane": none_diag}
    out["B4"] = {"ok": off.sanitize_goal is False and PL.REFePlanner.SANITIZE_GOAL is False,
                 "default": off.sanitize_goal}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=str(HERE / "a8_selftest.json"))
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if os.environ.get("REFE_A8S_CHILD") != "1":
        sys.path.insert(0, str(PKG / "eval"))
        import eval_checkpoint as EC
        env = EC.env_driverl()
        env["REFE_A8S_CHILD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(root / "refe"), str(root / "eval"),
                                             str(root / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), "--root", str(root), "--out",
                                str(Path(a.out).resolve())], cwd=str(root / "eval"), env=env)
    for p in (root / "refe", root / "eval", root / "code"):
        sys.path.insert(0, str(p))
    import torch
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    import augment_routes as A
    from nuplan.planning.script import driverl_runtime_map_features as M
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    scen = {}
    for key, (lg, tok) in (("cover", COVER), ("far", FAR)):
        scen[key] = next(iter(NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), [tok],
                                                         history_rows=1, future_rows=80)))
    src = (root / "refe" / "planner.py").read_text(encoding="utf-8")
    res = {"planner_sha256": hashlib.sha256((root / "refe" / "planner.py").read_bytes()).hexdigest()}
    PL = load("planner_a8_selftest", root / "refe" / "planner.py")
    PL.REFe = lambda cfg: torch.nn.Identity()                           # the goal path never touches the model
    res["clean"] = run_checks(PL, SEAM, NS, M, A, PlannerInitialization, torch, scen)
    res["mutations"] = {}
    tmp = Path(tempfile.mkdtemp(prefix="a8_mut_"))
    for mid, (old, new, target) in MUTATIONS.items():
        n = src.count(old)
        if n != 1:
            res["mutations"][mid] = {"applied": False, "count": n}
            continue
        f = tmp / f"planner_{mid}.py"
        f.write_text(src.replace(old, new), encoding="utf-8")
        PLm = load(f"planner_a8_{mid}", f)
        PLm.REFe = lambda cfg: torch.nn.Identity()
        r = run_checks(PLm, SEAM, NS, M, A, PlannerInitialization, torch, scen)
        res["mutations"][mid] = {"applied": True, "target": target, "target_red": not r[target]["ok"],
                                 "result": {k: v["ok"] for k, v in r.items()}}
    clean_ok = all(v["ok"] for v in res["clean"].values())
    mut_ok = all(m.get("applied") and m.get("target_red") for m in res["mutations"].values())
    res["failed"] = ([f"clean {k}" for k, v in res["clean"].items() if not v["ok"]] +
                     [f"mutation {k} did not go RED" for k, m in res["mutations"].items()
                      if not (m.get("applied") and m.get("target_red"))])
    for k, v in res["clean"].items():
        print(f"  [{'PASS' if v['ok'] else 'FAIL'}] {k} {json.dumps({kk: vv for kk, vv in v.items() if kk != 'ok'}, default=float)[:220]}")
    for k, m in res["mutations"].items():
        print(f"  [{'PASS' if m.get('applied') and m.get('target_red') else 'FAIL'}] {k} goes RED on {m.get('target')}: "
              f"{m.get('result')}")
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    ok = clean_ok and mut_ok
    print("ZZA8_SELFTEST_" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
