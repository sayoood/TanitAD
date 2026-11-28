#!/usr/bin/env python3
"""P6 lane-rank twin bank, option 3-C (commit filter).  PREREG_3C.md (sha256 e8365f34...) + PREREG_3C.addendum1.md.

For EVERY lane-rank twin of the input bank: ONE 10 s CPU-teacher rollout (route HELD at step 0, goal horizon 6 s, REFE_SIM_HZ=10), f(t) = d_logged /
(d_logged + d_twin) at 1..10 s against the twin / logged PDM centrelines (`probe/long_rollout.py`, verbatim math), KEEP iff f(10 s) >= 0.75.  A kept row is the
input row with traj/traj_world = the FIRST 4 s of that SAME rollout (built by the unchanged `build_teacher_rollouts.make_row`, fed the cached rollout instead of
rolling a second time), goal = the goal the teacher received, aug += {route_hold: step0, goal_horizon_s: 6.0, commit_f10, reroll: 3c} and divergence_m recomputed
(addendum 1).  Every other row of the bank passes through BYTE-IDENTICAL; a dropped twin is absent.

    run    --bank <dir w/ targets_rank1.jsonl> --rank0 <targets_rank0.jsonl> --out <shard dir> --log-shard i/N --mc <parts root>
           [--mc-twin <root>] [--mc-logged <root>] [--chunk 32] [--limit N] [--mutate-no-hold  (I4 control ONLY)]
           resume-safe (twin keys already in <shard>/audit.jsonl are skipped), append-only, flushed per row, rows written BEFORE their audit line.
    merge  --bank <dir> --shards '<out>/shards/s*' --out-dir <new dir>      REFUSES unless every lane twin has exactly one audit record
    progress --out <out>                                                     counts audit lines by outcome (a script, not a model turn)

Needs (pod): the teacher env of ms_run.sh; --code-dir holds the PATCHED route_lane_rank_patch.py + build_teacher_rollouts.py (their sha256 go into every shard).
`run` imports nuplan lazily, so `merge` / `progress` / the unit tests run anywhere with plain python3.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

GOAL_HORIZON_S = 6.0          # PREREG step 1
SPAN_S = 10.0                 # PREREG step 1
WINDOW_S = 4.0                # PREREG step 3 (the kept row is the first 4 s)
KEEP_F10 = 0.75               # PREREG step 3
CHUNK_DEFAULT = 32            # addendum 1, item 3
YIELD_GUARD_K1 = 0.20         # PREREG "Yield guard"
YIELD_GUARD_MIN_N = 30        # addendum 1, item 4
PREREG_SHA = "e8365f34fd5f11a23a1b5e0fceda14dc14426277ae09db3e5335526c1bd6cde3"
GOAL_TOL_NOTE = "goal changes by design (horizon 6 s != the bank's 12 s); recorded as goal_max_abs_diff, never judged"


# ---------------------------------------------------------------------------------------------------------------- plain helpers (no nuplan)
def sha256_file(p, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def key_of(r) -> tuple:
    return (r["log_name"], r["token"], int(r["aug"]["value"]))


def is_lane_row(r) -> bool:
    a = r.get("aug")
    return isinstance(a, dict) and a.get("kind") == "lane_rank"


def lane_lines(path):
    """Yield (parsed row) for the lane-rank rows of a bank file, in file order (substring pre-filter, then a real parse)."""
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if '"lane_rank"' not in ln:
                continue
            r = json.loads(ln)
            if is_lane_row(r):
                yield r


def divergence_m(traj_a, traj_b) -> float:
    """`augment_search.divergence_m`, verbatim (as in reroll_twins.py): max per-step displacement between two ego-frame trajectories (m)."""
    if traj_a is None or traj_b is None or len(traj_a) != len(traj_b):
        return float("inf")
    return max(math.dist(p[:2], q[:2]) for p, q in zip(traj_a, traj_b))


def rank0_trajs(path, want: set) -> dict:
    """{(log, token): traj} of the rank-0 rows for the wanted frames (`reroll_twins.rank0_trajs`, verbatim)."""
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


def plan_shards(keys: list, n_shards: int, chunk: int):
    """Deterministic shard plan.  keys = lane-twin keys in INPUT order.  Units are (log, chunk-of-`chunk` twins of that log, in input order); units are
    assigned to shards by LPT (largest first, least-loaded shard, ties to the lowest index).  Returns (units, assign) with assign[i] = unit indices of shard i
    sorted by (log, chunk index) so a shard handles a log's chunks back to back.  Every key lands in exactly one unit of exactly one shard."""
    by_log: dict = {}
    for k in keys:
        by_log.setdefault(k[0], []).append(k)
    units = []
    for lg in sorted(by_log):
        ks = by_log[lg]
        for ci in range(0, len(ks), chunk):
            units.append((lg, ci // chunk, ks[ci:ci + chunk]))
    order = sorted(range(len(units)), key=lambda u: (-len(units[u][2]), units[u][0], units[u][1]))
    load = [0] * n_shards
    assign = [[] for _ in range(n_shards)]
    for u in order:
        i = min(range(n_shards), key=lambda j: (load[j], j))
        assign[i].append(u)
        load[i] += len(units[u][2])
    for i in range(n_shards):
        assign[i].sort(key=lambda u: (units[u][0], units[u][1]))
    return units, assign


def _repair_tail(path: Path) -> None:
    """A kill mid-write leaves a partial last line; truncate it so the next append starts on a clean line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as fh:
        data = fh.read()
        if data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        fh.truncate(cut)


def _read_jsonl_tolerant(path: Path) -> list:
    """Parse a jsonl; a malformed LAST line (killed writer) is ignored, a malformed earlier line is an error."""
    out = []
    if not path.exists():
        return out
    lines = path.read_bytes().split(b"\n")
    for n, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            if n >= len(lines) - 2:
                continue
            raise
    return out


# ------------------------------------------------------------------------------------------------------------------------------------- run
def cmd_run(a) -> int:
    arm = "legacy" if a.mutate_no_hold else "held"
    os.environ["REFE_ROUTE_HOLD"] = "0" if arm == "legacy" else "1"          # read at call time by route_lane_rank_patch.hold_enabled()
    if a.goal_horizon != GOAL_HORIZON_S and not a.mutate_no_hold:
        print(f"  REFUSING: --goal-horizon {a.goal_horizon} != the pre-registered {GOAL_HORIZON_S}"); return 2
    sys.path[:0] = [a.code_dir, a.refe_dir, a.pkg_code_dir]
    import numpy as np
    import shapely
    import route_lane_rank_patch as P
    import build_teacher_rollouts as BR
    import navtrain_scenarios as NS
    import online_labeller as OL
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    shas = {"route_lane_rank_patch": sha256_file(P.__file__)[:16], "build_teacher_rollouts": sha256_file(BR.__file__)[:16],
            "reroll_bank_3c": sha256_file(__file__)[:16]}
    print(f"  code: {P.__file__} {BR.__file__} {shas}", flush=True)
    if not hasattr(P, "held_route") or "held_route" not in Path(BR.__file__).read_text(encoding="utf-8"):
        print("  REFUSING: the patched route_lane_rank_patch / build_teacher_rollouts are not the ones on sys.path"); return 2
    if not P.install(1):
        print("  REFUSING: route patch did not install"); return 2
    from nuplan.planning.script import driverl_runtime_map_features as M
    if not getattr(M._choose_route_edge, "__route_hold__", False):
        print("  REFUSING: the live _choose_route_edge is not the held wrapper"); return 2
    if P.hold_enabled() != (arm == "held"):
        print(f"  REFUSING: arm={arm} but hold_enabled()={P.hold_enabled()}"); return 2
    if NS.SIM_HZ != 10:
        print("  REFUSING: REFE_SIM_HZ must be 10 (the bank's rate)"); return 5

    bank_file = Path(a.bank) / "targets_rank1.jsonl"
    keys = [key_of(r) for r in lane_lines(bank_file)]
    if len(set(keys)) != len(keys):
        print(f"  REFUSING: the input bank holds duplicate lane-twin keys ({len(keys) - len(set(keys))})"); return 2
    n_shards, i_shard = 1, 0
    if a.log_shard:
        i_shard, n_shards = (int(x) for x in a.log_shard.split("/"))
    units, assign = plan_shards(keys, n_shards, a.chunk)
    my_units = [units[u] for u in assign[i_shard]]
    my_keys = [k for _, _, ks in my_units for k in ks]
    in_sha = sha256_file(bank_file)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"i": i_shard, "N": n_shards, "chunk": a.chunk, "arm": arm, "input_sha256": in_sha, "input_bytes": bank_file.stat().st_size,
            "n_assigned": len(my_keys), "code": shas, "prereg_sha256": PREREG_SHA, "goal_horizon_s": a.goal_horizon, "span_s": SPAN_S,
            "keep_f10": KEEP_F10, "mc_twin": str(a.mc_twin or a.mc), "mc_logged": str(a.mc_logged or a.mc)}
    mp = out / "shard_meta.json"
    if mp.exists():
        old = json.loads(mp.read_text(encoding="utf-8"))
        for f in ("i", "N", "chunk", "arm", "input_sha256", "goal_horizon_s", "span_s", "keep_f10", "code"):
            if old.get(f) != meta[f]:
                print(f"  REFUSING to resume: shard_meta.{f} differs ({old.get(f)!r} vs {meta[f]!r}) -- another configuration wrote this shard"); return 2
    else:
        mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ap_, rp_ = out / "audit.jsonl", out / "rows.jsonl"
    _repair_tail(ap_); _repair_tail(rp_)
    done = {key_of_audit(r) for r in _read_jsonl_tolerant(ap_)}
    todo_keys = [k for k in my_keys if k not in done]
    if a.limit:
        todo_keys = todo_keys[:a.limit]
    todo_set = set(todo_keys)
    rows_by_key = {}
    toks = {k[1] for k in todo_keys}
    with open(bank_file, encoding="utf-8") as fh:
        for ln in fh:
            if '"lane_rank"' not in ln:
                continue
            i = ln.find('"token": "')
            if i >= 0 and ln[i + 10:i + 26] not in toks:
                continue
            r = json.loads(ln)
            if is_lane_row(r) and key_of(r) in todo_set:
                rows_by_key[key_of(r)] = r
    want = {(k[0], k[1]) for k in todo_keys}
    t0 = time.time()
    r0 = rank0_trajs(a.rank0, want) if a.rank0 else {}
    print(f"  shard {i_shard}/{n_shards} arm={arm}: {len(my_keys):,} assigned, {len(done):,} already audited, {len(todo_keys):,} to do; "
          f"rank-0 refs {len(r0):,} ({time.time() - t0:.1f}s)", flush=True)
    if not todo_keys:
        (out / "DONE").write_text(json.dumps({"n_audit": len(done), "wall_s": 0}), encoding="utf-8"); return 0

    # ---- the one-rollout shim: make_row calls rollout_from_frame(sc, planner, controller, i, span4); it is served the first span4 states of the cached 10 s rollout.
    orig_roll = BR.rollout_from_frame
    cache = {"sc": None, "i": None, "poses": None, "served": 0}

    def shim(scenario, planner_, controller_, i, span, *args, **kw):
        if cache["poses"] is None or scenario is not cache["sc"] or i != cache["i"] or span > len(cache["poses"]):
            raise RuntimeError("3C: make_row asked for a rollout that is not the cached 10 s one -- refusing to roll a second time")
        cache["served"] += 1
        return cache["poses"][:span]
    BR.rollout_from_frame = shim         # NB: orig_roll sets `rollout_from_frame.last_goal / .last_hold` through the module global => on `shim`, which make_row reads

    BR.TEACHER_DEVICE = "cpu"
    planner = BR.build_planner("cpu")
    cfg = planner._feature_builder.config
    dbs = NS.index_dbs()
    fa, fr = open(ap_, "a", encoding="utf-8"), open(rp_, "a", encoding="utf-8")
    counts: dict = {}
    t_start = time.time()
    by_log: dict = {}
    for k in todo_keys:
        by_log.setdefault(k[0], []).append(k)
    for lg in sorted(by_log):
        lg_keys = by_log[lg]
        toks_lg = sorted({k[1] for k in lg_keys})
        scs, cams, cam_ts, setup_err = {}, None, None, None
        try:
            if lg not in dbs:
                raise KeyError("no navtrain DB for this log")
            cams, cam_ts = BR.camera_arrays(dbs[lg])
            scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(dbs[lg], toks_lg)}
        except Exception as exc:                                                         # noqa: BLE001
            setup_err = f"log_setup_failed: {type(exc).__name__}: {str(exc)[:100]}"
        for k in lg_keys:
            r = rows_by_key[k]
            rec = {"log_name": k[0], "token": k[1], "k": k[2], "step": r.get("step"), "arm": arm, "v0": r["ego"][6], "f_at_s": None, "f10_raw": None,
                   "d_twin_at_s": None, "commit": None, "drop_reason": None, "hold_misses": None, "hold_hits": None, "rollout_s": None}
            row_out = None
            try:
                row_out = _one_twin(rec, r, scs.get(k[1]), setup_err, cams, cam_ts, r0, a, arm, planner, cfg, BR, NS, OL, P, PlannerInitialization,
                                    np, shapely, cache, shim, orig_roll)
            except Exception as exc:                                                     # noqa: BLE001 -- a bug must not lose the shard: audited, loud
                BR._reset_planner_state(planner)
                rec["drop_reason"] = f"internal_error: {type(exc).__name__}: {str(exc)[:140]}"
            finally:
                cache.update(sc=None, i=None, poses=None)
            if row_out is not None:
                fr.write(json.dumps(row_out) + "\n"); fr.flush()          # row BEFORE its audit line: audit presence => row exists
            fa.write(json.dumps(rec) + "\n"); fa.flush()
            tag = "kept" if rec["drop_reason"] is None else rec["drop_reason"].split(":")[0]
            counts[tag] = counts.get(tag, 0) + 1
            if sum(counts.values()) % 10 == 0:
                print(f"  {sum(counts.values())}/{len(todo_keys)} {counts} {time.time() - t_start:.0f}s", flush=True)
    fa.close(); fr.close()
    wall = time.time() - t_start
    n_audit = len(done) + sum(counts.values())
    if not a.limit:
        (out / "DONE").write_text(json.dumps({"n_audit": n_audit, "wall_s": round(wall, 1), "counts": counts}), encoding="utf-8")
    print(f"ZZ3C_SHARD_DONE shard={i_shard}/{n_shards} arm={arm} audited_now={sum(counts.values())} counts={counts} wall_s={wall:.1f}", flush=True)
    return 0


def key_of_audit(rec) -> tuple:
    return (rec["log_name"], rec["token"], int(rec["k"]))


def _one_twin(rec, r, sc, setup_err, cams, cam_ts, r0, a, arm, planner, cfg, BR, NS, OL, P, PlannerInitialization, np, shapely, cache, shim, orig_roll):
    """Audit one twin (mutates `rec`).  Returns the kept bank row, or None (rec['drop_reason'] says why)."""
    k = rec["k"]
    if setup_err:
        rec["drop_reason"] = setup_err; return None
    if sc is None:
        rec["drop_reason"] = "missing_scenario"; return None
    tw_p = OL.parts_path(a.mc_twin or a.mc, rec["log_name"], rec["token"], {"kind": "lane_rank", "value": k})
    lg_p = OL.parts_path(a.mc_logged or a.mc, rec["log_name"], rec["token"], None)
    if not tw_p.is_file() or not lg_p.is_file():
        rec["drop_reason"] = "missing_parts:" + ("twin" if not tw_p.is_file() else "") + ("+" if (not tw_p.is_file() and not lg_p.is_file()) else "") \
            + ("logged" if not lg_p.is_file() else "")
        return None
    dt = float(sc.database_interval)
    if abs(dt - 0.1) > 1e-9:
        rec["drop_reason"] = f"unexpected_dt:{dt}"; return None
    span10 = int(round(SPAN_S / dt))
    step = int(r.get("step", 0))
    controller = BR.build_controller(sc, "cpu")
    P.set_rank(k)
    cfg.route_goal_horizon_s = a.goal_horizon                                        # route_goal_min_speed_mps is left at the config default
    planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    shim.last_goal = None; shim.last_hold = None
    t0 = time.time()
    try:
        poses = orig_roll(sc, planner, controller, step, span10)
    except Exception as exc:                                                         # noqa: BLE001
        BR._reset_planner_state(planner)
        rec["drop_reason"] = f"rollout_error: {type(exc).__name__}: {str(exc)[:120]}"; return None
    rec["rollout_s"] = round(time.time() - t0, 2)
    if poses is None:
        rec["drop_reason"] = "scenario_too_short_10s"; return None
    hold = getattr(shim, "last_hold", None)
    rec["hold_misses"] = None if hold is None else hold["misses"]
    rec["hold_hits"] = None if hold is None else hold["hits"]
    # ---- f(t): probe/long_rollout.py, verbatim math
    twl, lgl = OL.load_parts(tw_p)[3].linestring, OL.load_parts(lg_p)[3].linestring
    xy = shapely.points(np.array([[p.rear_axle.x, p.rear_axle.y] for p in poses], dtype=np.float64))
    d_lg, d_tw = shapely.distance(lgl, xy), shapely.distance(twl, xy)
    f = d_lg / np.maximum(d_lg + d_tw, 1e-6)
    per_s = int(round(1.0 / dt))
    idx = [min(per_s * s - 1, len(f) - 1) for s in range(1, int(SPAN_S) + 1)]
    rec["f_at_s"] = [round(float(f[i]), 3) for i in idx]
    rec["d_twin_at_s"] = [round(float(d_tw[i]), 2) for i in idx]
    rec["f10_raw"] = float(f[idx[-1]])
    rec["commit"] = bool(rec["f_at_s"][-1] >= KEEP_F10)
    if not rec["commit"]:
        rec["drop_reason"] = "not_committed"; return None
    # ---- a committing twin: build its row from the SAME rollout's first 4 s
    if arm == "held" and (hold is None or hold.get("chain_ids") is None):
        rec["drop_reason"] = "hold_not_engaged"; return None
    if cams is None:
        rec["drop_reason"] = "no_camera_channel"; return None
    cache.update(sc=sc, i=step, poses=poses, served=0)
    new = BR.make_row(sc, planner, controller, step, cams, cam_ts, 1)
    if new is None:
        rec["drop_reason"] = "make_row_none"; return None
    if cache["served"] != 1:
        rec["drop_reason"] = f"internal_error: make_row served {cache['served']} cached rollouts"; return None
    inv = {fld: new[fld] == r[fld] for fld in ("image", "ego", "step", "sim_hz")}
    rec["origin_max_abs_diff"] = max(abs(x - y) for x, y in zip(new["origin_world"], r["origin_world"]))
    inv["origin_world"] = rec["origin_max_abs_diff"] <= 1e-6                         # 2/74 rows differ in the last bits (MEASURED in the probe): a tolerance, recorded
    rec["goal_max_abs_diff"] = max(abs(x - y) for x, y in zip(new["goal"], r["goal"])) if len(new["goal"]) == len(r["goal"]) else None
    if not all(inv.values()):
        rec["drop_reason"] = "invariant_changed:" + ",".join(fld for fld, v in inv.items() if not v); return None
    ref = r0.get((rec["log_name"], rec["token"]))
    if ref is None:
        rec["drop_reason"] = "no_rank0_ref"; return None
    div_new = divergence_m(new["traj"], ref)
    if not math.isfinite(div_new):
        rec["drop_reason"] = "rank0_ref_shape_mismatch"; return None
    out = dict(r)
    out["traj"], out["traj_world"], out["goal"] = new["traj"], new["traj_world"], new["goal"]
    aug = dict(r["aug"])
    aug["divergence_m"] = round(div_new, 4)
    aug["route_hold"] = "step0" if arm == "held" else "per_step_rerank"
    aug["goal_horizon_s"] = float(a.goal_horizon)
    aug["commit_f10"] = rec["f_at_s"][-1]
    aug["reroll"] = "3c"
    out["aug"] = aug
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- merge
def _load_shards(pattern: str):
    sdirs = sorted(Path(p) for p in glob.glob(pattern))
    audit, metas = {}, []
    rows: dict = {}
    for sd in sdirs:
        if not (sd / "shard_meta.json").exists():
            raise SystemExit(f"ZZ3C_MERGE_REFUSED {sd} has no shard_meta.json")
        metas.append(json.loads((sd / "shard_meta.json").read_text(encoding="utf-8")))
        for rec in _read_jsonl_tolerant(sd / "audit.jsonl"):
            kk = key_of_audit(rec)
            if kk in audit:
                a0, b0 = audit[kk], rec
                if (a0["f_at_s"], a0["commit"], a0["drop_reason"]) != (b0["f_at_s"], b0["commit"], b0["drop_reason"]):
                    raise SystemExit(f"ZZ3C_MERGE_REFUSED conflicting audit records for {kk}")
                continue
            audit[kk] = rec
        if (sd / "rows.jsonl").exists():
            for ln in (sd / "rows.jsonl").read_bytes().split(b"\n"):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue                                       # a killed writer's partial last line: its twin has no audit line, coverage check catches it
                kk = key_of(r)
                if kk in rows and rows[kk] != ln:
                    raise SystemExit(f"ZZ3C_MERGE_REFUSED two different kept rows for {kk}")
                rows[kk] = ln
    return sdirs, metas, audit, rows


def cmd_merge(a) -> int:
    src, dst = Path(a.bank), Path(a.out_dir)
    in_file = src / "targets_rank1.jsonl"
    in_sha = sha256_file(in_file)
    sdirs, metas, audit, rows = _load_shards(a.shards)
    if not sdirs:
        print("ZZ3C_MERGE_REFUSED no shard directories match"); return 3
    m0 = metas[0]
    for m in metas:
        for f in ("N", "chunk", "arm", "input_sha256", "goal_horizon_s", "span_s", "keep_f10", "code"):
            if m[f] != m0[f]:
                print(f"ZZ3C_MERGE_REFUSED shard metas disagree on {f}"); return 3
    if m0["arm"] != "held":
        print(f"ZZ3C_MERGE_REFUSED shards were written by arm={m0['arm']!r} (a control arm, never a bank)"); return 3
    if m0["input_sha256"] != in_sha:
        print("ZZ3C_MERGE_REFUSED shards were run on another input bank (sha256 differs)"); return 3
    if len(sdirs) != m0["N"] or sorted(m["i"] for m in metas) != list(range(m0["N"])):
        print(f"ZZ3C_MERGE_REFUSED found shards {sorted(m['i'] for m in metas)} of N={m0['N']}"); return 3

    in_keys = [key_of(r) for r in lane_lines(in_file)]
    if len(set(in_keys)) != len(in_keys):
        print("ZZ3C_MERGE_REFUSED duplicate lane-twin keys in the input"); return 3
    in_set = set(in_keys)
    missing = [k for k in in_keys if k not in audit]
    unused = [k for k in audit if k not in in_set]
    if missing or unused:
        print(f"ZZ3C_MERGE_REFUSED coverage: {len(missing)} lane twins have no audit record, {len(unused)} audit records are not in the input "
              f"(first missing: {missing[:3]})"); return 3

    kept_keys = set()
    for k in in_keys:
        rec = audit[k]
        if rec["commit"] is True and rec["drop_reason"] is None:
            if k not in rows:
                print(f"ZZ3C_MERGE_REFUSED {k} is audited as kept but has no row"); return 3
            r = json.loads(rows[k])
            ag = r["aug"]
            ok = (ag.get("reroll") == "3c" and ag.get("route_hold") == "step0" and ag.get("goal_horizon_s") == GOAL_HORIZON_S
                  and ag.get("commit_f10") == rec["f_at_s"][-1] and rec["f_at_s"][-1] >= KEEP_F10)       # (c'1), asserted on every kept row
            if not ok:
                print(f"ZZ3C_MERGE_REFUSED kept row {k} fails its own provenance / (c'1) check: {ag}"); return 3
            kept_keys.add(k)
        elif rec["commit"] is False and rec["drop_reason"] != "not_committed":
            print(f"ZZ3C_MERGE_REFUSED {k}: commit False with reason {rec['drop_reason']!r}"); return 3
        elif rec["commit"] is None and rec["drop_reason"] is None:
            print(f"ZZ3C_MERGE_REFUSED {k}: an unevaluated twin with no drop reason"); return 3

    dst.mkdir(parents=True, exist_ok=False)
    part = dst / "targets_rank1.jsonl.partial"
    n_in = n_out = n_other = 0
    with open(in_file, "rb") as fi, open(part, "wb") as fo:
        for ln in fi:
            n_in += 1
            if b'"lane_rank"' in ln:
                r = json.loads(ln)
                if is_lane_row(r):
                    k = key_of(r)
                    if k in kept_keys:
                        fo.write(rows[k] + b"\n"); n_out += 1
                    continue
            fo.write(ln); n_out += 1; n_other += 1
    # ---- the report
    by_rank: dict = {}
    for k in in_keys:
        rec = audit[k]
        b = by_rank.setdefault(str(k[2]), {"in": 0, "evaluated": 0, "committed": 0, "kept": 0, "dropped_by_reason": {}, "committed_but_dropped": 0,
                                           "f10_round_vs_raw_disagree": 0})
        b["in"] += 1
        if rec["f_at_s"] is not None:
            b["evaluated"] += 1
            b["f10_round_vs_raw_disagree"] += int((rec["f10_raw"] >= KEEP_F10) != (rec["f_at_s"][-1] >= KEEP_F10))
        b["committed"] += int(bool(rec["commit"]))
        if k in kept_keys:
            b["kept"] += 1
        else:
            reason = (rec["drop_reason"] or "UNKNOWN").split(":")[0]
            b["dropped_by_reason"][reason] = b["dropped_by_reason"].get(reason, 0) + 1
            b["committed_but_dropped"] += int(bool(rec["commit"]))
    for b in by_rank.values():
        b["keep_share_all"] = round(b["kept"] / b["in"], 4) if b["in"] else None
        b["keep_share_evaluated"] = round(b["kept"] / b["evaluated"], 4) if b["evaluated"] else None
    k1 = by_rank.get("1")
    guard = {"rule": f"K=1 keep share (evaluated basis) < {YIELD_GUARD_K1} => STOP (PREREG_3C); needs K=1 evaluated n >= {YIELD_GUARD_MIN_N}",
             "k1_evaluated": k1["evaluated"] if k1 else 0, "k1_share_evaluated": k1["keep_share_evaluated"] if k1 else None,
             "k1_share_all": k1["keep_share_all"] if k1 else None}
    if not k1 or k1["evaluated"] < YIELD_GUARD_MIN_N:
        guard["status"] = "NOT_EVALUATED (K=1 evaluated n < %d)" % YIELD_GUARD_MIN_N
        tripped = False
    else:
        tripped = k1["keep_share_evaluated"] < YIELD_GUARD_K1
        guard["status"] = "TRIPPED" if tripped else "HOLDS"
        if (k1["keep_share_all"] < YIELD_GUARD_K1) != tripped:
            guard["note"] = "the two denominators DISAGREE on tripping: the Master Mind decides (addendum 1, item 4)"
    secs = sorted(r["rollout_s"] for r in audit.values() if r.get("rollout_s") is not None)
    report = {"prereg_sha256": PREREG_SHA, "input": {"path": str(in_file), "sha256": in_sha, "lines": n_in, "lane_rank_rows": len(in_keys), "other_rows": n_other},
              "output": {"lines": n_out, "dropped_twins": len(in_keys) - len(kept_keys), "kept_twins": len(kept_keys)},
              "by_rank": by_rank, "yield_guard": guard, "shards": len(sdirs), "code": m0["code"],
              "rollout_s": {"n": len(secs), "sum": round(sum(secs), 1), "mean": round(sum(secs) / len(secs), 2) if secs else None,
                            "median": secs[len(secs) // 2] if secs else None, "p90": secs[int(0.9 * (len(secs) - 1))] if secs else None,
                            "max": secs[-1] if secs else None}}
    with open(dst / "audit_all.jsonl", "w", encoding="utf-8") as fh:
        for k in in_keys:
            fh.write(json.dumps(audit[k]) + "\n")
    if n_out != n_in - report["output"]["dropped_twins"]:
        print("ZZ3C_MERGE_REFUSED line-count identity failed"); part.unlink(); return 3
    if tripped:
        part.rename(dst / "targets_rank1.jsonl.REJECTED_YIELD_GUARD")
        (dst / "STOP_YIELD_GUARD.txt").write_text(json.dumps(guard, indent=1), encoding="utf-8")
        (dst / "merge_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"ZZ3C_YIELD_GUARD_TRIPPED {guard} -- NO bank is handed to P6; see {dst}"); return 4
    part.rename(dst / "targets_rank1.jsonl")
    report["output"]["sha256"] = sha256_file(dst / "targets_rank1.jsonl")
    extras = {}
    for extra in sorted(src.iterdir()):
        if extra.name != "targets_rank1.jsonl" and extra.is_file() and not extra.name.startswith("."):
            shutil.copyfile(extra, dst / extra.name)
            extras[extra.name] = {"sha256_src": sha256_file(extra), "sha256_dst": sha256_file(dst / extra.name)}
            if extras[extra.name]["sha256_src"] != extras[extra.name]["sha256_dst"]:
                print(f"ZZ3C_MERGE_REFUSED copy of {extra.name} differs"); return 3
    report["copied_unchanged"] = extras
    (dst / "merge_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"ZZ3C_MERGE_OK lines_in={n_in} lines_out={n_out} kept={len(kept_keys)} dropped={report['output']['dropped_twins']} yield_guard={guard['status']} -> {dst}")
    return 0


# -------------------------------------------------------------------------------------------------------------------------------- progress
def cmd_progress(a) -> int:
    n = 0
    reasons: dict = {}
    per_k: dict = {}
    first = last = None
    shards = done = 0
    for sd in sorted(glob.glob(str(Path(a.out) / "shards" / "s*"))):
        shards += 1
        done += int((Path(sd) / "DONE").exists())
        ap_ = Path(sd) / "audit.jsonl"
        if not ap_.exists():
            continue
        st = ap_.stat()
        first = st.st_ctime if first is None else min(first, st.st_ctime)
        last = st.st_mtime if last is None else max(last, st.st_mtime)
        for rec in _read_jsonl_tolerant(ap_):
            n += 1
            tag = "kept" if rec["drop_reason"] is None else rec["drop_reason"].split(":")[0]
            reasons[tag] = reasons.get(tag, 0) + 1
            kk = per_k.setdefault(str(rec["k"]), {"audited": 0, "kept": 0})
            kk["audited"] += 1; kk["kept"] += int(tag == "kept")
    total = a.total
    out = {"audited": n, "total": total, "shards_seen": shards, "shards_done": done, "by_outcome": reasons, "per_k": per_k, "now_utc": time.strftime("%FT%TZ", time.gmtime())}
    if a.started_epoch and n:
        el = time.time() - a.started_epoch
        out["rate_per_s"] = round(n / el, 3)
        out["eta_h"] = round((total - n) / (n / el) / 3600, 2) if total else None
    print(json.dumps(out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--bank", required=True); r.add_argument("--rank0", default=None); r.add_argument("--out", required=True)
    r.add_argument("--mc", required=True); r.add_argument("--mc-twin", default=None); r.add_argument("--mc-logged", default=None)
    r.add_argument("--log-shard", default=None); r.add_argument("--chunk", type=int, default=CHUNK_DEFAULT); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--goal-horizon", type=float, default=GOAL_HORIZON_S)
    r.add_argument("--mutate-no-hold", action="store_true", help="I4 CONTROL ONLY: REFE_ROUTE_HOLD=0; the shards are refused by merge")
    r.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    r.add_argument("--refe-dir", required=True); r.add_argument("--pkg-code-dir", required=True)
    m = sp.add_parser("merge")
    m.add_argument("--bank", required=True); m.add_argument("--shards", required=True); m.add_argument("--out-dir", required=True)
    g = sp.add_parser("progress")
    g.add_argument("--out", required=True); g.add_argument("--total", type=int, default=44409); g.add_argument("--started-epoch", type=float, default=None)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "merge": cmd_merge, "progress": cmd_progress}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
