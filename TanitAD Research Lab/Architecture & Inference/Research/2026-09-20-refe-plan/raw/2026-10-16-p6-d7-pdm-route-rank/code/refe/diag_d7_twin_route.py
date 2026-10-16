#!/usr/bin/env python3
"""D7 verification harness (POD ONLY: needs the NAVSIM stack, nuPlan, the DriveRL fork and a nuPlan DB).  Checks that lane-rank twin parts are on THEIR OWN route.

    select   --bank B --out D --seed 0 --n1 51 --n2 14 --n0 12     a seeded mini-bank (D/targets_rank0.jsonl + D/targets_rank1.jsonl) of real bank rows, so the REAL cmd_build runs on a small set
    verify   --bank D --k K --twin-cache T --logged-cache L ...     checks (a) (b) (c) on the twin parts a REAL `build_metric_cache.py build --lane-rank K` wrote
    compare  --bank D --a A --b B [--k K] ...                       check (d): two parts caches (rank-0 rows, or the rank-K twins) hold the same parts -- canon modulo the nondeterministic track_id + D9's score_plans equality

PRE-REGISTERED BEFORE THE FIRST RUN (2026-10-16, so no bar moves after the data):
  (a) DISTINCT   p6_prep.route_digest(twin) != route_digest(logged) on >= 0.9 of the sampled twins (the chain's own bar, P6_D7_MIN_DISTINCT).
  (b) SPEC LANE  the twin centreline EQUALS (all points, atol 1e-9) the centreline PDM-Closed's own search (`_get_discrete_centerline`, UNPATCHED NAVSIM) yields from DriveRL's rank-K lane --
                 DriveRL's OWN `_choose_route_edge` under route_lane_rank_patch.set_rank(K), on the interior lanes of PDM's starting roadblock with DriveRL's anchor -- on ALL checked twins
                 (zero mismatches), AND the verifier reproduces the LOGGED centreline from PDM's own starting lane on 100 % (its own control).  A lane-ID comparison is NOT used: adjacent
                 nuPlan lanes can share a start and a long prefix (MEASURED: 64280 / 64281), so 'the first lane whose baseline is a prefix of the centreline' is ambiguous.
  (c) SEMANTIC   among twins with divergence_m >= 0.5: the banked teacher trajectory (traj_world) is closer (mean point-to-polyline distance over its poses) to the TWIN centreline than to
                 the LOGGED centreline for >= 0.75 of them.  (Chance is 0.5; a twin identical to the logged route scores 0.0 because "closer" is strict.)
  AMENDMENT 1 (same day, AFTER run 1, disclosed): run 1 compared lane IDs ("the first route lane whose baseline is a prefix of the centreline") and read b = 15/51.  That comparator is
                 AMBIGUOUS (lanes 64280 / 64281 share their first 204 points), so (b) was redefined as the centreline-level equality above.  (a) and (c) and their bars (0.9, 0.75) were NOT changed.
                 Run 1's numbers are kept verbatim in raw/verify_k1_run1_laneid.json.
  (d) CONTROL    rank-0 parts rebuilt with the fixed code vs the existing cache, >= 10 tokens: all six parts' `canon` and `route_digest` equal (zero differences).
  AMENDMENT 2 (same day, AFTER the first control run, disclosed): run 1 of (d) read 0/12 "all six equal" because `obs` differs in the nuPlan DB's integer `track_id` metadata -- which also
                 differs between two builds by the SAME fixed code (chain-style vs standalone) and which no scored quantity reads.  (d) therefore compares canon MODULO track_id, and adds D9's
                 own functional contract (score_plans equal, every array, seeded random proposals).  The raw numbers of run 1 are kept in raw/control_rank0_run1_raw_canon.json.
  (e) MUTATION   with the Pool initializer removed (REFE_D7_MUTATE) the build FAILS LOUD; with initializer AND guard removed (P6's defect) (a) AND (b) must FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))
DZ = os.environ.get("DZ_ROOT", "/workspace/dz/DriveZero/DriveRL")


def _rows(bank, only_lane_k=None, rank0=False):
    for fp in sorted(Path(bank).glob("targets_rank*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                aug = r.get("aug")
                lane = isinstance(aug, dict) and aug.get("kind") == "lane_rank"
                if only_lane_k is not None and not (lane and int(aug["value"]) == int(only_lane_k)):
                    continue
                if rank0 and (aug or int(r.get("rank", 0)) != 0):
                    continue
                yield r


# ------------------------------------------------------------------------------------------------ select
def cmd_select(a) -> int:
    rng = random.Random(a.seed)
    by = {1: {}, 2: {}}
    bank = Path(a.bank)
    with open(bank / "targets_rank1.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            aug = r.get("aug")
            if isinstance(aug, dict) and aug.get("kind") == "lane_rank" and int(aug["value"]) in by:
                by[int(aug["value"])].setdefault(r["log_name"], []).append(r)
    logs2 = sorted(by[2])
    logs = sorted(l for l in by[1] if l in by[2])
    rng.shuffle(logs)
    chosen1, chosen2 = [], []
    for lg in logs:
        if len(chosen1) < a.n1:
            rs = sorted(by[1][lg], key=lambda r: r["token"])
            rng.shuffle(rs)
            chosen1 += rs[:3]
        if len(chosen2) < a.n2:
            rs = sorted(by[2][lg], key=lambda r: r["token"])
            rng.shuffle(rs)
            chosen2 += rs[:1]
    chosen1, chosen2 = chosen1[: a.n1], chosen2[: a.n2]
    # rank-0 rows of 12 tokens in logs already chosen (the (d) control + the logged cache for lane-distinct)
    want0 = {(r["log_name"], r["token"]) for r in chosen1}
    rank0, seen = [], set()
    with open(bank / "targets_rank0.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("aug") or int(r.get("rank", 0)) != 0:
                continue
            k = (r["log_name"], r["token"])
            if k in want0 and k not in seen:
                seen.add(k)
                rank0.append(r)
    rng.shuffle(rank0)
    rank0 = rank0[: a.n0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "targets_rank0.jsonl", "w", encoding="utf-8") as fh:
        for r in rank0:
            fh.write(json.dumps(r) + "\n")
    with open(out / "targets_rank1.jsonl", "w", encoding="utf-8") as fh:
        for r in chosen1 + chosen2:
            fh.write(json.dumps(r) + "\n")
    print(f"ZZSELECT rank1={len(chosen1)} rank2={len(chosen2)} rank0={len(rank0)} logs1={len({r['log_name'] for r in chosen1})} logs2={len({r['log_name'] for r in chosen2})} "
          f"div>=0.5: r1={sum(r['aug']['divergence_m'] >= 0.5 for r in chosen1)} r2={sum(r['aug']['divergence_m'] >= 0.5 for r in chosen2)}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ geometry helpers
def _cl_xy(discrete_path):
    import numpy as np
    return np.array([[s.x, s.y] for s in discrete_path], dtype=np.float64)


def _same(a, b, atol=1e-9) -> bool:
    import numpy as np
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol, rtol=0.0))


def _mean_dist_to(linestring, xy):
    import numpy as np
    import shapely
    return float(np.mean(shapely.distance(linestring, shapely.points(np.asarray(xy, dtype=np.float64)[:, :2]))))


def _route_world(route_local, anchor, max_m=40.0):
    """DriveRL's step-0 route polyline (anchor frame) -> (world xy of the first `max_m` metres from the anchor, world xy of all points)."""
    import math
    import numpy as np
    r = np.asarray(route_local, dtype=np.float64)
    c, s_ = math.cos(anchor.yaw), math.sin(anchor.yaw)
    w = np.stack([anchor.x + c * r[:, 0] - s_ * r[:, 1], anchor.y + s_ * r[:, 0] + c * r[:, 1]], 1)
    keep = np.hypot(r[:, 0], r[:, 1]) <= max_m
    return (w[keep] if keep.sum() >= 2 else w[:2]), w


# ------------------------------------------------------------------------------------------------ verify (a) (b) (c)
def cmd_verify(a) -> int:
    import numpy as np
    import shapely
    import build_metric_cache as B
    import navsim_pdm_targets as PT
    import online_labeller as OL
    import p6_prep
    import route_lane_rank_patch as P
    sys.path.insert(0, f"{DZ}/src")
    from driverl.nuplan.route_correction import correct_route_roadblock_ids
    from nuplan.planning.script import driverl_runtime_map_features as M
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    import pdm_route_rank as R

    k = int(a.k)
    assert R.installed_rank() is None, "the verifier must run on UNPATCHED NAVSIM: it reconstructs PDM-Closed's own centrelines"
    P.install(k)                       # the oracle serves DriveRL's rank-k route; it never builds PDM parts
    rows = {}
    for r in _rows(a.bank, only_lane_k=k):
        rows[(r["log_name"], r["token"])] = r
    if a.limit:
        rows = dict(list(rows.items())[: a.limit])
    by_log = {}
    for lg, tk in rows:
        by_log.setdefault(lg, set()).add(tk)
    S = PT._setup()
    pdm = S["proc"]._pdm_closed
    out, t0 = [], time.time()
    for lg, toks in sorted(by_log.items()):
        scs = B._scenarios(a.db_root, a.map_root, lg, toks)
        for tk in sorted(toks):
            r = rows[(lg, tk)]
            aug = r["aug"]
            rec = {"log": lg, "token": tk, "k": k, "divergence_m": aug.get("divergence_m"), "bank_lanes": aug.get("lanes")}
            sc = scs.get(tk)
            tw_p, lg_p = OL.parts_path(a.twin_cache, lg, tk, aug), OL.parts_path(a.logged_cache, lg, tk, None)
            if sc is None or not tw_p.is_file() or not lg_p.is_file():
                rec.update(error=f"missing scenario={sc is not None} twin_file={tw_p.is_file()} logged_file={lg_p.is_file()}")
                out.append(rec)
                continue
            try:
                tw, lgp = OL.load_parts(tw_p), OL.load_parts(lg_p)
                planner_input, init = S["proc"]._get_planner_inputs(sc)
                ego_state, _ = planner_input.history.current_state
                map_api = init.map_api
                anchor = R.anchor_from_ego(ego_state, M)
                # ---- PDM side, UNPATCHED: exactly the steps compute_planner_trajectory runs before _update_proposal_manager
                pdm.initialize(init)
                pdm._iteration = 0
                pdm._route_roadblock_correction(ego_state)
                pdm._drivable_area_map = PDMDrivableMap.from_simulation(pdm._map_api, ego_state, pdm._map_radius)
                lane0 = pdm._get_starting_lane(ego_state)                                  # NAVSIM's own choice
                blk = pdm._route_roadblock_dict[lane0.get_roadblock_id()]
                P.set_rank(k)
                exp_lane = M._choose_route_edge(list(blk.interior_edges), anchor, None)     # DriveRL's OWN patched function, on PDM's starting roadblock
                P.set_rank(0)
                r0_pdm = M._choose_route_edge(list(blk.interior_edges), anchor, None)       # DriveRL's rank-0 lane of the SAME roadblock
                cl_lane0, cl_exp = _cl_xy(pdm._get_discrete_centerline(lane0)), _cl_xy(pdm._get_discrete_centerline(exp_lane))
                tw_xy, lg_xy = _cl_xy(tw[3].discrete_path), _cl_xy(lgp[3].discrete_path)
                # ---- the TEACHER's own route at step 0 (DriveRL's start roadblock, its corrected route)
                rids = correct_route_roadblock_ids(ego_state, map_api, [str(i) for i in init.route_roadblock_ids])
                blocks = [b for b in (M._route_roadblock_for_id(map_api, i) for i in rids) if b is not None]
                first = blocks[M._route_start_index(blocks, anchor)]
                P.set_rank(k)
                rk_t = M._choose_route_edge(list(first.interior_edges), anchor, None)
                rp_k = M._sample_route_points_from_map(map_api=map_api, route_roadblock_ids=rids, anchor=anchor)
                P.set_rank(0)
                r0_t = M._choose_route_edge(list(first.interior_edges), anchor, None)
                rp_0 = M._sample_route_points_from_map(map_api=map_api, route_roadblock_ids=rids, anchor=anchor)
                P.set_rank(k)
                rk40, rk_all = _route_world(rp_k, anchor)
                _, r0_all = _route_world(rp_0, anchor)
                traj = np.asarray(r["traj_world"], dtype=np.float64)
                tw_ls, lg_ls = tw[3].linestring, lgp[3].linestring
                n_lanes = len([e for e in blk.interior_edges if M._edge_centerline_coords(e)])
                rec.update(
                    pdm_block=str(blk.id), pdm_lane0=str(lane0.id), expected_lane=str(exp_lane.id), driverl_r0_lane_in_pdm_block=str(r0_pdm.id), n_lanes_pdm_block=n_lanes,
                    driverl_start_block=str(first.id), driverl_rk_lane=str(rk_t.id), driverl_r0_lane=str(r0_t.id), start_block_agrees=bool(first.id == blk.id),
                    harness_reproduces_logged_centerline=_same(lg_xy, cl_lane0),                 # the verifier's own control: must be 100 %
                    digest_differs=p6_prep.route_digest(tw) != p6_prep.route_digest(lgp),
                    spec_ok=_same(tw_xy, cl_exp),                                                # (b): twin centreline == PDM's search from DriveRL's rank-k lane
                    twin_equals_logged_centerline=_same(tw_xy, lg_xy),
                    expected_equals_pdm_lane0=bool(exp_lane.id == lane0.id),                     # the twin cannot differ: rank-k lane IS PDM's own lane
                    pdm_lane0_equals_driverl_r0=bool(lane0.id == r0_pdm.id),
                    clamped=bool(k > n_lanes - 1),
                    teacher_lane_equals_expected=bool(rk_t.id == exp_lane.id), teacher_rk_equals_r0=bool(rk_t.id == r0_t.id),
                    twin_centerline_len=float(tw[3].length), logged_centerline_len=float(lgp[3].length),
                    # (b2) the TEACHER'S OWN step-0 route polyline (first 40 m) vs the two centrelines
                    route_k_equals_route_0=bool(rk_all.shape == r0_all.shape and np.allclose(rk_all, r0_all, atol=1e-3)),
                    d_route_to_twin=_mean_dist_to(tw_ls, rk40), d_route_to_logged=_mean_dist_to(lg_ls, rk40),
                    # (c) the banked teacher trajectory vs the two centrelines, and vs the teacher's OWN rank-k route
                    d_twin=_mean_dist_to(tw_ls, traj), d_logged=_mean_dist_to(lg_ls, traj),
                    d_traj_to_route_k=_mean_dist_to(shapely.linestrings(rk_all), traj), d_traj_to_route_0=_mean_dist_to(shapely.linestrings(r0_all), traj))
                rec["closer_to_twin"] = bool(rec["d_twin"] < rec["d_logged"])
                rec["route_closer_to_twin"] = bool(rec["d_route_to_twin"] < rec["d_route_to_logged"])
            except Exception as e:                                           # noqa: BLE001
                import traceback
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                rec["tb"] = traceback.format_exc()[-600:]
            out.append(rec)
    ok = [r for r in out if "error" not in r]

    def frac(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 4) if rs else None

    def block(rs):
        hi_ = [r for r in rs if (r["divergence_m"] or 0) >= 0.5]
        dist_ = [r for r in rs if not r["expected_equals_pdm_lane0"]]
        return {"n": len(rs), "digest_differs": frac(rs, "digest_differs"), "spec_ok": frac(rs, "spec_ok"), "twin_equals_logged": frac(rs, "twin_equals_logged_centerline"),
                "expected_equals_pdm_lane0": frac(rs, "expected_equals_pdm_lane0"), "teacher_lane_equals_expected": frac(rs, "teacher_lane_equals_expected"),
                "n_distinct_expected": len(dist_), "route_closer_to_twin_among_distinct": frac(dist_, "route_closer_to_twin"),
                "n_div_ge_0_5": len(hi_), "traj_closer_to_twin_div_ge_0_5": frac(hi_, "closer_to_twin")}
    hi = [r for r in ok if (r["divergence_m"] or 0) >= 0.5]
    summ = {"k": k, "n_requested": len(rows), "n_checked": len(ok), "n_error": len(out) - len(ok),
            "harness_reproduces_logged_centerline_frac": frac(ok, "harness_reproduces_logged_centerline"),
            "all": block(ok), "start_block_agrees": block([r for r in ok if r["start_block_agrees"]]), "start_block_disagrees": block([r for r in ok if not r["start_block_agrees"]]),
            "start_block_agrees_frac": frac(ok, "start_block_agrees"), "clamped_frac": frac(ok, "clamped"), "pdm_lane0_equals_driverl_r0_frac": frac(ok, "pdm_lane0_equals_driverl_r0"),
            "c_mean_d_twin": round(float(np.mean([r["d_twin"] for r in hi])), 3) if hi else None, "c_mean_d_logged": round(float(np.mean([r["d_logged"] for r in hi])), 3) if hi else None,
            "teacher_traj_mean_dist_to_own_route_k": round(float(np.mean([r["d_traj_to_route_k"] for r in ok])), 3) if ok else None,
            "teacher_traj_mean_dist_to_route_0": round(float(np.mean([r["d_traj_to_route_0"] for r in ok])), 3) if ok else None,
            "twin_centerline_shorter_than_20m": sum(r["twin_centerline_len"] < 20.0 for r in ok), "wall_s": round(time.time() - t0, 1)}
    summ["a_ok"] = bool(summ["all"]["digest_differs"] is not None and summ["all"]["digest_differs"] >= 0.9)
    summ["b_ok"] = bool(ok and summ["n_error"] == 0 and summ["all"]["spec_ok"] == 1.0 and summ["harness_reproduces_logged_centerline_frac"] == 1.0)
    summ["c_ok"] = bool(summ["all"]["traj_closer_to_twin_div_ge_0_5"] is not None and summ["all"]["traj_closer_to_twin_div_ge_0_5"] >= 0.75)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "twins": out}, indent=1, default=str), encoding="utf-8")
    print("ZZD7V_SUMMARY " + json.dumps(summ, default=str), flush=True)
    print(f"ZZD7V_{'PASS' if summ['a_ok'] and summ['b_ok'] and summ['c_ok'] else 'FAIL'} a={summ['a_ok']} b={summ['b_ok']} c={summ['c_ok']}", flush=True)
    return 0 if summ["a_ok"] and summ["b_ok"] and summ["c_ok"] else 1


# ------------------------------------------------------------------------------------------------ compare / control (d)
IGNORE_KEYS = ("track_id",)


def _canon_ign(o, depth=0, seen=None) -> str:
    """p6_prep.canon, except that the nuPlan DB's integer `track_id` is skipped.  MEASURED 2026-10-16: it differs between two builds of the SAME token by the SAME code
    (189 vs 3275 for the same track_token; the pickles differ by a few dozen bytes) and nothing the scorer reads depends on it (score_plans equal on every array)."""
    import hashlib
    import re
    seen = seen if seen is not None else set()
    if o is None or isinstance(o, (bool, int, float, str, bytes)):
        return repr(o)
    if depth > 6 or id(o) in seen:
        return "<" + type(o).__name__ + ">"
    seen = seen | {id(o)}
    import numpy as np
    if isinstance(o, np.ndarray) and o.dtype != object:
        return f"nd:{o.dtype.str}:{o.shape}:{hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest()}"
    if isinstance(o, np.generic):
        return repr(o.item())
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(_canon_ign(x, depth + 1, seen) for x in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(sorted(_canon_ign(k, depth + 1, seen) + ":" + _canon_ign(v, depth + 1, seen) for k, v in o.items() if k not in IGNORE_KEYS)) + "}"
    if isinstance(o, (set, frozenset)):
        return "s{" + ",".join(sorted(_canon_ign(x, depth + 1, seen) for x in o)) + "}"
    if hasattr(o, "__dict__"):
        return type(o).__name__ + _canon_ign(vars(o), depth + 1, seen)
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(o))


def cmd_compare(a) -> int:
    """Two parts caches, same keys, built by (supposedly) the same code: every part identical (canon, modulo track_id), the route digest identical, and D9's FUNCTIONAL contract:
    score_plans(A) == score_plans(B), every array np.array_equal, on seeded random proposal sets.  --k omitted: the rank-0 rows of --bank (logged paths); --k K: the rank-K twin paths."""
    import numpy as np
    import online_labeller as OL
    import p6_prep
    import navsim_pdm_targets as PT
    PT._setup()
    k = a.k
    keys = {}
    for r in _rows(a.bank, only_lane_k=k, rank0=(k is None)):
        keys[(r["log_name"], r["token"])] = r.get("aug") if k is not None else None
    rng = np.random.default_rng(0)
    res = []
    for (lg, tk), aug in sorted(keys.items()):
        pa, pb = OL.parts_path(a.a, lg, tk, aug), OL.parts_path(a.b, lg, tk, aug)
        if not (pa.is_file() and pb.is_file()):
            res.append({"log": lg, "token": tk, "error": f"A={pa.is_file()} B={pb.is_file()}"})
            continue
        x, y = OL.load_parts(pa), OL.load_parts(pb)
        names = ("traj", "ego", "obs", "centerline", "lanes", "dam")
        eq = {nm: _canon_ign(u) == _canon_ign(v) for nm, u, v in zip(names, x, y)}
        eq_raw = {nm: p6_prep.canon(u) == p6_prep.canon(v) for nm, u, v in zip(names, x, y)}
        n_eq = 0
        for _ in range(a.n_sets):
            props = np.stack([np.stack([np.linspace(0.5, 30.0, 20) * (0.8 + 0.4 * rng.random()), rng.normal(0, 1.0, 20).cumsum() * 0.1, np.zeros(20)], -1) for _ in range(8)])
            plans8 = PT.executed_plans8(props, repair=True)
            s1, s2 = PT.score_plans(x, plans8), PT.score_plans(y, plans8)
            n_eq += int(all(np.array_equal(s1[kk], s2[kk]) for kk in s1))
        res.append({"log": lg, "token": tk, "canon_equal_modulo_track_id": eq, "all_six_equal_modulo_track_id": all(eq.values()), "canon_equal_raw": eq_raw,
                    "route_digest_equal": p6_prep.route_digest(x) == p6_prep.route_digest(y), "score_equal_sets": n_eq, "n_sets": a.n_sets,
                    "file_bytes_equal": pa.read_bytes() == pb.read_bytes()})
    ok = [r for r in res if "error" not in r]
    summ = {"k": k, "n_keys": len(res), "n_compared": len(ok), "n_error": len(res) - len(ok),
            "all_six_equal_modulo_track_id": sum(r["all_six_equal_modulo_track_id"] for r in ok), "route_digest_equal": sum(r["route_digest_equal"] for r in ok),
            "score_equal_all_sets": sum(r["score_equal_sets"] == r["n_sets"] for r in ok),
            "five_parts_except_obs_equal_raw": sum(all(v for kk, v in r["canon_equal_raw"].items() if kk != "obs") for r in ok),
            "obs_equal_raw": sum(r["canon_equal_raw"]["obs"] for r in ok), "file_bytes_equal": sum(r["file_bytes_equal"] for r in ok)}
    summ["ok"] = bool(len(ok) >= a.min_keys and summ["n_error"] == 0 and summ["all_six_equal_modulo_track_id"] == len(ok) and summ["route_digest_equal"] == len(ok)
                      and summ["score_equal_all_sets"] == len(ok))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "keys": res}, indent=1), encoding="utf-8")
    print("ZZD7C_SUMMARY " + json.dumps(summ), flush=True)
    print("ZZD7C_PASS" if summ["ok"] else "ZZD7C_FAIL", flush=True)
    return 0 if summ["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--bank", required=True); s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--n1", type=int, default=51); s.add_argument("--n2", type=int, default=14); s.add_argument("--n0", type=int, default=12)
    v = sub.add_parser("verify")
    v.add_argument("--bank", required=True); v.add_argument("--k", type=int, required=True)
    v.add_argument("--twin-cache", required=True); v.add_argument("--logged-cache", required=True)
    v.add_argument("--db-root", default=None); v.add_argument("--map-root", default=None)
    v.add_argument("--out", required=True); v.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare", aliases=["control"])
    c.add_argument("--bank", required=True); c.add_argument("--a", required=True); c.add_argument("--b", required=True); c.add_argument("--out", required=True)
    c.add_argument("--k", type=int, default=None); c.add_argument("--n-sets", type=int, default=4); c.add_argument("--min-keys", type=int, default=10)
    a = ap.parse_args(argv)
    return {"select": cmd_select, "verify": cmd_verify, "compare": cmd_compare, "control": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
