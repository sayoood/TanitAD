#!/usr/bin/env python3
"""Re-roll the LANE-RANK twins of a bank with the augmented route HELD FIXED over the rollout (P6 D7, PI option 3, 2026-10-16).

WHY. The banked rank-K twins were rolled out while the teacher re-ranked its lane against the CURRENT ego every step, so they are
perturbations of the logged route (D7 check (c) 1/34, 0/7, 0/4). This re-rolls EXACTLY the same (log, token, K) rows -- same frame,
same rank, same teacher, same harness (`build_teacher_rollouts.make_row`) -- with `route_lane_rank_patch.held_route()` active.

WHAT CHANGES IN A ROW, AND WHAT MUST NOT.
  changes : traj, traj_world, aug.divergence_m (recomputed against the frame's rank-0 row, `augment_search.divergence_m`), aug.route_hold
  asserted UNCHANGED (else the row is REFUSED): goal (the step-0 goal: the held route IS the step-0 route), image, ego, origin_world, step, sim_hz
  goal-horizon twins and rank-0 rows are never touched (rank 0 delegates before any hold logic).

    run    --bank <dir with targets_rank1.jsonl> --rank0 <targets_rank0.jsonl> --out <dir> --arm held|legacy [--log-shard i/N] [--limit N] [--resume]
    merge  --bank <dir> --reroll-glob '<dir>/rr_s*/targets_reroll.jsonl' --out-dir <new dir>    replace every lane-rank row; REFUSES if one is missing
    check  --bank <dir>                                                                        ZZTWIN_HOLD_OK iff EVERY lane-rank row was re-rolled held

`--arm legacy` (REFE_ROUTE_HOLD=0) is the CONTROL: it must reproduce the banked trajectory (the failure), proving the harness is the bank's.
POD: `source /workspace/teacher_env.sh; export REFE_SIM_HZ=10 REFE_NUPLAN_DB_ROOT=... NUPLAN_DATA_ROOT=...`; --code-dir must hold the PATCHED
route_lane_rank_patch.py + build_teacher_rollouts.py (they shadow the tree's copies; their sha256 is printed and written to every stats line).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


GOAL_TOL_M = 1e-3


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def divergence_m(traj_a, traj_b) -> float:
    """`augment_search.divergence_m`, verbatim: max per-step displacement between two ego-frame trajectories (m)."""
    if traj_a is None or traj_b is None or len(traj_a) != len(traj_b):
        return float("inf")
    return max(math.dist(p[:2], q[:2]) for p, q in zip(traj_a, traj_b))


def lane_rows(bank):
    for fp in sorted(Path(bank).glob("targets_rank1.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                if '"lane_rank"' not in ln:
                    continue
                r = json.loads(ln)
                a = r.get("aug")
                if isinstance(a, dict) and a.get("kind") == "lane_rank":
                    yield r


def rank0_trajs(path, want: set) -> dict:
    """{(log, token): traj} for the wanted frames, by a substring pre-filter (the rank-0 file is ~350 MB)."""
    toks = {t for _, t in want}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            i = ln.find('"token": "')
            if i < 0:
                continue
            tok = ln[i + 10:i + 26]
            if tok not in toks:
                continue
            r = json.loads(ln)
            k = (r["log_name"], r["token"])
            if k in want and not r.get("aug") and int(r.get("rank", 0)) == 0:
                out[k] = r["traj"]
    return out


def key(r):
    return (r["log_name"], r["token"], int(r["aug"]["value"]))


def cmd_run(a) -> int:
    os.environ["REFE_ROUTE_HOLD"] = "1" if a.arm == "held" else "0"
    sys.path[:0] = [a.code_dir, a.refe_dir, a.pkg_code_dir]
    import route_lane_rank_patch as P
    import build_teacher_rollouts as BR
    import navtrain_scenarios as NS
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    shas = {"route_lane_rank_patch": _sha(P.__file__), "build_teacher_rollouts": _sha(BR.__file__)}
    print(f"  code: {P.__file__} {BR.__file__} {shas}", flush=True)
    if not hasattr(P, "held_route") or "held_route" not in Path(BR.__file__).read_text(encoding="utf-8"):
        print("  REFUSING: the patched route_lane_rank_patch / build_teacher_rollouts are not the ones on sys.path"); return 2
    if not P.install(1):
        print("  REFUSING: route patch did not install"); return 2
    from nuplan.planning.script import driverl_runtime_map_features as M
    if not getattr(M._choose_route_edge, "__route_hold__", False):
        print("  REFUSING: the live _choose_route_edge is not the held wrapper"); return 2
    print(f"  arm={a.arm} hold_enabled={P.hold_enabled()} sim_hz={NS.SIM_HZ}", flush=True)
    if NS.SIM_HZ != 10:
        print("  REFUSING: REFE_SIM_HZ must be 10 (the bank's rate)"); return 5

    rows = list(lane_rows(a.bank))
    by_log = {}
    for r in rows:
        by_log.setdefault(r["log_name"], []).append(r)
    names = sorted(by_log)
    if a.log_shard:
        i, n = (int(x) for x in a.log_shard.split("/"))
        names = names[i::n]
    os.makedirs(a.out, exist_ok=True)
    out_p, st_p = Path(a.out) / "targets_reroll.jsonl", Path(a.out) / "reroll_stats.jsonl"
    done = set()
    if a.resume and out_p.exists():
        for ln in open(out_p, encoding="utf-8"):
            if ln.strip():
                done.add(key(json.loads(ln)))
    todo = [r for lg in names for r in by_log[lg] if key(r) not in done]
    if a.limit:
        todo = todo[:a.limit]
    want = {(r["log_name"], r["token"]) for r in todo}
    t0 = time.time()
    r0 = rank0_trajs(a.rank0, want) if a.rank0 else {}
    print(f"  {len(rows):,} lane-rank rows in bank; this shard {len(todo):,} to re-roll; rank-0 refs {len(r0):,} ({time.time()-t0:.1f}s)", flush=True)

    BR.TEACHER_DEVICE = a.device
    planner = BR.build_planner(a.device)
    dbs = NS.index_dbs()
    W, S = open(out_p, "a", encoding="utf-8"), open(st_p, "a", encoding="utf-8")
    n_ok = n_bad = 0
    t_start = time.time()
    by_log_todo = {}
    for r in todo:
        by_log_todo.setdefault(r["log_name"], []).append(r)
    for lg, rs in by_log_todo.items():
        cams, cam_ts = BR.camera_arrays(dbs[lg])
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(dbs[lg], sorted({r["token"] for r in rs}))}
        for r in rs:
            k = int(r["aug"]["value"])
            sc = scs.get(r["token"])
            rec = {"log_name": lg, "token": r["token"], "k": k, "arm": a.arm, "code": shas}
            if sc is None:
                rec["error"] = "scenario not built"; S.write(json.dumps(rec) + "\n"); n_bad += 1; continue
            controller = BR.build_controller(sc, a.device)
            P.set_rank(k)
            planner._feature_builder.config.route_goal_horizon_s = float(a.goal_horizon)
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            t1 = time.time()
            try:
                new = BR.make_row(sc, planner, controller, int(r.get("step", 0)), cams, cam_ts, 1)
            except Exception as exc:                                   # noqa: BLE001
                BR._reset_planner_state(planner)
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"; S.write(json.dumps(rec) + "\n"); n_bad += 1; continue
            rec["rollout_s"] = round(time.time() - t1, 3)
            hold = getattr(BR.rollout_from_frame, "last_hold", None)
            rec["hold"] = None if hold is None else {kk: (len(v) if kk in ("rec", "chain_ids") and v is not None else v)
                                                     for kk, v in hold.items()}
            if new is None:
                rec["error"] = "make_row returned None"; S.write(json.dumps(rec) + "\n"); n_bad += 1; continue
            inv = {f: new[f] == r[f] for f in ("image", "ego", "step", "sim_hz")}
            # MEASURED in the probe: 2/74 rows' origin_world differ from the bank in the last bits (rows built on another machine);
            # a real change (another frame) is metres, so this is a tolerance, recorded per row
            rec["origin_max_abs_diff"] = max(abs(x - y) for x, y in zip(new["origin_world"], r["origin_world"]))
            inv["origin_world"] = rec["origin_max_abs_diff"] <= 1e-6
            # the step-0 goal: exact on the bank's own platform; a cross-platform (Windows-built row, Linux re-roll) ulp drift is
            # tolerated up to GOAL_TOL_M and RECORDED, a real change (another route at step 0) is metres and is refused
            rec["goal_max_abs_diff"] = max(abs(x - y) for x, y in zip(new["goal"], r["goal"])) if len(new["goal"]) == len(r["goal"]) else float("inf")
            inv["goal"] = rec["goal_max_abs_diff"] <= GOAL_TOL_M if a.goal_horizon == 12.0 else True   # a lever arm CHANGES the goal by design
            rec["invariants"] = inv
            rec["traj_vs_bank_max_m"] = round(max(math.dist(p[:2], q[:2]) for p, q in zip(new["traj"], r["traj"])), 6)
            ref = r0.get((lg, r["token"]))
            rec["div_bank"] = r["aug"].get("divergence_m")
            rec["div_new"] = None if ref is None else round(divergence_m(new["traj"], ref), 4)
            rec["div_bank_recomputed"] = None if ref is None else round(divergence_m(r["traj"], ref), 4)
            if a.arm == "held" and (hold is None or hold.get("chain_ids") is None):
                rec["error"] = "REFUSED: the held arm's rollout did not engage the route hold"
            elif not all(inv.values()):
                rec["error"] = "REFUSED: a frame invariant changed: " + ",".join(f for f, v in inv.items() if not v)
            if "error" in rec:
                S.write(json.dumps(rec) + "\n"); n_bad += 1; continue
            out = dict(r)
            out["traj"], out["traj_world"] = new["traj"], new["traj_world"]
            if a.goal_horizon != 12.0:                                   # LEVER ARM ONLY: the row carries the goal the teacher received
                out["goal"] = new["goal"]
            out["aug"] = dict(r["aug"])
            if rec["div_new"] is not None:
                out["aug"]["divergence_m"] = rec["div_new"]
            out["aug"]["route_hold"] = "step0" if a.arm == "held" else "per_step_rerank"
            if a.goal_horizon != 12.0:
                out["aug"]["goal_horizon_s"] = float(a.goal_horizon)
            W.write(json.dumps(out) + "\n"); W.flush()
            S.write(json.dumps(rec) + "\n"); S.flush()
            n_ok += 1
    W.close(); S.close()
    wall = time.time() - t_start
    print(f"ZZREROLL_DONE arm={a.arm} ok={n_ok} bad={n_bad} wall_s={wall:.1f} s_per_rollout={wall / max(n_ok + n_bad, 1):.3f}", flush=True)
    return 0 if n_bad == 0 else 1


def cmd_merge(a) -> int:
    """Write <out-dir>/targets_rank0.jsonl (copied) + targets_rank1.jsonl with every lane-rank row replaced by its held re-roll."""
    rr = {}
    for fp in sorted(glob.glob(a.reroll_glob)):
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                if r["aug"].get("route_hold") != "step0":
                    print(f"  REFUSING: {fp} holds a row that was not re-rolled with the route held"); return 2
                rr[key(r)] = ln if ln.endswith("\n") else ln + "\n"
    src, dst = Path(a.bank), Path(a.out_dir)
    dst.mkdir(parents=True, exist_ok=False)
    n = rep = miss = 0
    with open(src / "targets_rank1.jsonl", encoding="utf-8") as fi, open(dst / "targets_rank1.jsonl", "w", encoding="utf-8") as fo:
        for ln in fi:
            if '"lane_rank"' in ln:
                r = json.loads(ln)
                if isinstance(r.get("aug"), dict) and r["aug"].get("kind") == "lane_rank":
                    k = key(r)
                    if k not in rr:
                        miss += 1
                        continue
                    ln = rr.pop(k); rep += 1
            fo.write(ln); n += 1
    if miss or rr:
        print(f"ZZMERGE_REFUSED missing_reroll={miss} unused_reroll={len(rr)} -- {dst} is INCOMPLETE, delete it"); return 3
    import shutil
    for extra in sorted(src.iterdir()):
        if extra.name != "targets_rank1.jsonl" and extra.is_file():
            shutil.copyfile(extra, dst / extra.name)
    print(f"ZZMERGE_OK rows={n} lane_rank_replaced={rep} -> {dst}")
    return 0


def cmd_check(a) -> int:
    """The chain guard: every lane-rank twin in the bank must carry aug.route_hold == "step0" (a stale or regenerated bank goes RED)."""
    n = held = 0
    for r in lane_rows(a.bank):
        n += 1
        held += r["aug"].get("route_hold") == "step0"
    ok = n > 0 and held == n
    print(f"ZZTWIN_HOLD_{'OK' if ok else 'BAD'} lane_rank_rows={n} route_hold_step0={held}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--bank", required=True); r.add_argument("--rank0", default=None); r.add_argument("--out", required=True)
    r.add_argument("--arm", choices=("held", "legacy"), required=True); r.add_argument("--device", default="cpu")
    r.add_argument("--log-shard", default=None); r.add_argument("--limit", type=int, default=0); r.add_argument("--resume", action="store_true")
    r.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    r.add_argument("--goal-horizon", type=float, default=12.0, help="LEVER ARM ONLY (default 12.0 = the bank's base horizon)")
    r.add_argument("--refe-dir", required=True); r.add_argument("--pkg-code-dir", required=True)
    m = sp.add_parser("merge")
    m.add_argument("--bank", required=True); m.add_argument("--reroll-glob", required=True); m.add_argument("--out-dir", required=True)
    c = sp.add_parser("check")
    c.add_argument("--bank", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "merge": cmd_merge, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
