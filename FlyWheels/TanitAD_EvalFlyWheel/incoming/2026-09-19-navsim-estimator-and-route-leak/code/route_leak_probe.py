#!/usr/bin/env python3
"""E3 Part B (iii) — is NavSim's `driving_command` (and nuPlan's route) derived from the
EXPERT'S FUTURE PATH? An EMPIRICAL probe on the local OpenScene test metadata + nuPlan maps.

Run with the NavSim C: runtime (Python 3.9, nuPlan + the pinned devkit importable):

    C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -u code/route_leak_probe.py \
        --frames-cache <scratch>/frames_cache.pkl --out raw/route_leak_probe.json

FOUR PROBES, each a different MECHANISM (a second probe is a different mechanism, not the
same command twice):

(a) REPRODUCTION — re-run OpenScene's OWN `get_driving_command` (the banked source at
    OpenDriveLab/OpenScene@7286074, `raw/source_probes/…driving_command.py`) on the stored
    (current pose, stored `roadblock_ids`, map). If it reproduces the stored command, the
    command is a function of (pose NOW, route, map) and of nothing else.
(b) ROUTE PROVENANCE — per nuPlan scene, does the stored route coincide with the roadblocks
    the logged ego ACTUALLY traverses afterwards (visited, in order)? And does it equal
    nuPlan-devkit's own `get_roadblock_ids_from_trajectory` on the FUTURE ego path (the
    function nuPlan documents as "route roadblock ids extracted from expert trajectory")?
(c) AGREEMENT MATRIX — stored command vs the logged future path (lateral offset at 20 m of
    driven arc; lateral / heading change at fixed horizons), with n.
(d) COUNTERFACTUAL — recompute the command from a route built from the ego's FUTURE path, and
    from a NO-FUTURE route (greedy straight-ahead successor at every roadblock). The
    difference between the two, on the frames where the stored command says TURN, is the
    part of the command that only the expert's future can supply.

Memory discipline: ONE CITY MAP at a time, released before the next; system free RAM checked.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
import types
from collections import Counter, defaultdict

import numpy as np

MAPS_ROOT = os.environ.setdefault("NUPLAN_MAPS_ROOT", "C:/Users/Admin/navsim-crun/data/maps")
MAP_VERSION = "nuplan-maps-v1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OS_SRC = os.path.join(PKG, "raw", "source_probes",
                      "OpenScene_7286074_DriveEngine_process_data_helpers_driving_command.py")
CMD = ("left", "forward", "right", "unknown")


def _cmd_name(onehot) -> str:
    a = list(onehot)
    if sum(a) != 1:
        return f"invalid{a}"
    return CMD[a.index(1)]


def _free_mb() -> float:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 20
    except Exception:
        return float("nan")


def load_openscene_command_fn():
    spec = importlib.util.spec_from_file_location("openscene_driving_command_7286074", OS_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# geometry on the logged ego path                                              #
# --------------------------------------------------------------------------- #
def to_ego(x0, y0, yaw0, x, y):
    dx, dy = x - x0, y - y0
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    return dx * c - dy * s, dx * s + dy * c


def point_at_arc(rows, i, dist):
    """Logged ego position after `dist` metres of driven arc from frame i (linear interp)."""
    s = 0.0
    for j in range(i, len(rows) - 1):
        a, b = rows[j], rows[j + 1]
        seg = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        if s + seg >= dist and seg > 0:
            t = (dist - s) / seg
            return a["x"] + t * (b["x"] - a["x"]), a["y"] + t * (b["y"] - a["y"]), j + 1
        s += seg
    return None


def classify_lat(lat, thr):
    return "left" if lat >= thr else ("right" if lat <= -thr else "forward")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def future_features(rows, i):
    r0 = rows[i]
    out = {}
    p = point_at_arc(rows, i, 20.0)
    if p is not None:
        lon, lat = to_ego(r0["x"], r0["y"], r0["yaw"], p[0], p[1])
        out["lat20_egoyaw"] = lat
        # path-tangent frame: heading of the driven path over its first ~2 m
        q = point_at_arc(rows, i, 2.0)
        if q is not None:
            th = math.atan2(q[1] - r0["y"], q[0] - r0["x"])
            out["lat20_pathtan"] = to_ego(r0["x"], r0["y"], th, p[0], p[1])[1]
    for h in (2, 4, 6, 8):                     # 1, 2, 3, 4 s at 2 Hz
        if i + h < len(rows):
            rh = rows[i + h]
            out[f"lat_{h / 2:.0f}s"] = to_ego(r0["x"], r0["y"], r0["yaw"], rh["x"], rh["y"])[1]
            out[f"dyaw_{h / 2:.0f}s"] = math.degrees(wrap(rh["yaw"] - r0["yaw"]))
    return out


RULES = {
    "arc20m_lat2m_egoyaw": ("lat20_egoyaw", 2.0, "lat"),
    "arc20m_lat2m_pathtangent": ("lat20_pathtan", 2.0, "lat"),
    "arc20m_lat1m_egoyaw": ("lat20_egoyaw", 1.0, "lat"),
    "arc20m_lat3m_egoyaw": ("lat20_egoyaw", 3.0, "lat"),
    "t4s_lat2m": ("lat_4s", 2.0, "lat"),
    "t4s_lat1m": ("lat_4s", 1.0, "lat"),
    "t2s_lat1m": ("lat_2s", 1.0, "lat"),
    "t4s_dyaw15deg": ("dyaw_4s", 15.0, "yaw"),
    "t4s_dyaw10deg": ("dyaw_4s", 10.0, "yaw"),
    "t3s_dyaw15deg": ("dyaw_3s", 15.0, "yaw"),
}


def agreement(pairs):
    """pairs: list of (stored_cmd, predicted_cmd). Matrix + accuracy on known commands."""
    m = Counter(pairs)
    known = [(s, p) for s, p in pairs if s in ("left", "forward", "right")]
    acc = (sum(1 for s, p in known if s == p) / len(known)) if known else None
    per = {}
    for c in ("left", "forward", "right"):
        rows = [(s, p) for s, p in known if s == c]
        per[c] = {"n": len(rows), "recall": (round(sum(1 for s, p in rows if p == c) / len(rows), 4)
                                             if rows else None)}
    return {"n": len(pairs), "n_known_command": len(known),
            "accuracy": round(acc, 4) if acc is not None else None, "per_stored_class": per,
            "matrix_stored_x_predicted": {f"{s}->{p}": c for (s, p), c in sorted(m.items())}}


# --------------------------------------------------------------------------- #
# map-side routes                                                              #
# --------------------------------------------------------------------------- #
class _EgoLike:
    """Minimal stand-in exposing what get_roadblock_ids_from_trajectory reads."""

    def __init__(self, x, y):
        from nuplan.common.actor_state.state_representation import Point2D
        self.rear_axle = types.SimpleNamespace(point=Point2D(x, y))


def future_route(map_api, rows, i, max_frames):
    """nuPlan's OWN expert-trajectory route (nuplan_scenario.py:171-177 semantics) on the
    logged FUTURE path from frame i."""
    from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory
    ego = [_EgoLike(r["x"], r["y"]) for r in rows[i:i + max_frames]]
    return get_roadblock_ids_from_trajectory(map_api, ego)


def straight_route(map_api, x, y, yaw, depth=30):
    """A NO-FUTURE route: from the roadblock containing the ego, repeatedly take the
    outgoing edge whose lanes keep the heading straightest. Uses the map and the pose NOW
    only — the counterfactual 'navigation that does not know where the expert went'."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    pt = Point2D(x, y)
    cands = map_api.get_all_map_objects(pt, SemanticMapLayer.ROADBLOCK) or \
        map_api.get_all_map_objects(pt, SemanticMapLayer.ROADBLOCK_CONNECTOR)
    if not cands:
        return []

    def end_heading(block):
        hs = []
        for lane in block.interior_edges:
            dp = lane.baseline_path.discrete_path
            if len(dp) >= 2:
                hs.append(dp[-1].heading)
        return hs

    def start_heading(block):
        hs = []
        for lane in block.interior_edges:
            dp = lane.baseline_path.discrete_path
            if dp:
                hs.append(dp[0].heading)
        return hs

    # pick the containing block whose lanes best match the ego heading
    def fit(block):
        hs = start_heading(block) + end_heading(block)
        return min((abs(wrap(h - yaw)) for h in hs), default=9.0)

    cur = min(cands, key=fit)
    route = [cur.id]
    head = yaw
    for _ in range(depth):
        outs = list(cur.outgoing_edges)
        if not outs:
            break
        hs_cur = end_heading(cur)
        if hs_cur:
            head = hs_cur[int(np.argmin([abs(wrap(h - head)) for h in hs_cur]))]

        def turn(block):
            hs = end_heading(block)
            return min((abs(wrap(h - head)) for h in hs), default=9.0)

        nxt = min(outs, key=turn)
        if nxt.id in route:
            break
        route.append(nxt.id)
        cur = nxt
    return route


def contained_roadblocks(map_api, x, y):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    pt = Point2D(x, y)
    ids = [b.id for b in map_api.get_all_map_objects(pt, SemanticMapLayer.ROADBLOCK)]
    ids += [b.id for b in map_api.get_all_map_objects(pt, SemanticMapLayer.ROADBLOCK_CONNECTOR)]
    return ids


def _route_prov_summary(rp):
    if not rp:
        return {"n_scenes_checked": 0}
    def frac(pred):
        return round(float(np.mean([bool(pred(p)) for p in rp])), 4)
    located = [p for p in rp if p["ego_block_index_in_route_at_scene_start"] is not None]
    return {
        "n_scenes_checked": len(rp),
        "frac_ego_inside_route_at_scene_start": frac(
            lambda p: p["ego_block_index_in_route_at_scene_start"] is not None),
        "ego_block_index_at_scene_start": {
            "n": len(located),
            "frac_index_0": (round(float(np.mean([p["ego_block_index_in_route_at_scene_start"] == 0
                                                  for p in located])), 4) if located else None),
            "median": (float(np.median([p["ego_block_index_in_route_at_scene_start"]
                                        for p in located])) if located else None)},
        "frac_scenes_whose_route_includes_PAST_blocks": frac(
            lambda p: p["n_route_blocks_before_ego_visited_in_PAST"] > 0),
        "mean_frac_route_ahead_visited_in_FUTURE": round(float(np.mean([
            p["frac_ahead_visited_in_FUTURE"] for p in rp
            if p["frac_ahead_visited_in_FUTURE"] is not None])), 4),
        "frac_scenes_route_ahead_FULLY_visited_in_future": frac(
            lambda p: p["frac_ahead_visited_in_FUTURE"] == 1.0),
        "frac_scenes_route_ahead_fully_visited_OR_unvisited_only_beyond_log_end": frac(
            lambda p: p["frac_ahead_visited_in_FUTURE"] == 1.0
            or p["n_never_visited"] == p["n_never_visited_beyond_last_logged_block"]),
        "NULL_CONTROL_frac_straight_route_fully_visited_or_beyond_log_end": frac(
            lambda p: p["NULL_straight_route_fully_visited_or_beyond_log_end"]),
        "frac_future_visits_in_route_order": frac(lambda p: p["future_visits_in_route_order"]),
        "frac_route_ahead_equals_nuplan_expert_trajectory_route_prefix": frac(
            lambda p: p["route_ahead_equals_expert_trajectory_route_prefix"]),
        "route_len_median": float(np.median([p["route_len"] for p in rp])),
        "first10": rp[:10],
    }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-cache", required=True)
    ap.add_argument("--cluster-maps", default=os.path.join(PKG, "raw", "cluster_maps"))
    ap.add_argument("--synthetic", nargs="*", default=[],
                    help="pickles of per-synthetic-scene meta from two_stage_synthetic_census.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--navtest-sample", type=int, default=1500)
    ap.add_argument("--future-route-frames", type=int, default=120,
                    help="frames of logged future used to build the FUTURE route in (d)")
    ap.add_argument("--expert-route-frames", type=int, default=400,
                    help="frames of logged future fed to nuPlan's expert-trajectory route in (b)")
    ap.add_argument("--synthetic-sample", type=int, default=600)
    ap.add_argument("--min-free-mb", type=float, default=3000.0)
    ap.add_argument("--cities", nargs="*", default=None,
                    help="SMOKE ONLY: restrict the map probes to these cities")
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args(argv)
    t0 = time.time()
    rnd = random.Random(a.seed)

    with open(a.frames_cache, "rb") as f:
        cache = pickle.load(f)
    frames = cache["frames"]
    n_frames = sum(len(v) for v in frames.values())
    print(f"[leak] frames read: {n_frames} across {len(frames)} logs", flush=True)

    # evaluated tokens per split (stage 1), from the census cluster maps
    split_tokens = {}
    for s in ("navhard_two_stage", "warmup_two_stage", "navtest"):
        p = os.path.join(a.cluster_maps, f"{s}.json")
        with open(p, encoding="utf-8") as f:
            split_tokens[s] = set(json.load(f)["token_to_log_name"])
    tok_index = {r["token"]: (lg, r["i"]) for lg, rows in frames.items() for r in rows}

    # ---- (c) agreement matrices: numpy only, every frame ---------------------- #
    feats = {}
    for lg, rows in frames.items():
        for r in rows:
            feats[r["token"]] = future_features(rows, r["i"])
    stored = {r["token"]: _cmd_name(r["cmd"]) for rows in frames.values() for r in rows}
    agree = {}
    for scope, toks in (("all_frames", list(stored)),
                        ("navtest_stage1", sorted(split_tokens["navtest"])),
                        ("navhard_stage1", sorted(split_tokens["navhard_two_stage"])),
                        ("warmup_stage1", sorted(split_tokens["warmup_two_stage"]))):
        agree[scope] = {"base_rate": dict(Counter(stored[t] for t in toks)), "rules": {}}
        for rule, (key, thr, kind) in RULES.items():
            pairs = []
            for t in toks:
                v = feats[t].get(key)
                if v is None:
                    continue
                pred = classify_lat(v, thr)
                pairs.append((stored[t], pred))
            agree[scope]["rules"][rule] = agreement(pairs)
            agree[scope]["rules"][rule]["n_unreached_or_no_future"] = len(toks) - len(pairs)
    print(f"[leak] (c) agreement matrices done ({time.time() - t0:.0f}s)", flush=True)

    # ---- sample for the map probes -------------------------------------------- #
    nav_sample = sorted(split_tokens["navtest"])
    rnd.shuffle(nav_sample)
    probe = set(split_tokens["navhard_two_stage"]) | set(split_tokens["warmup_two_stage"]) | \
        set(nav_sample[:a.navtest_sample])
    by_city = defaultdict(list)
    for t in probe:
        lg, i = tok_index[t]
        by_city[frames[lg][0]["map"]].append(t)
    syn_meta = {}
    for p in a.synthetic:
        with open(p, "rb") as f:
            m = pickle.load(f)
        keys = sorted(m)
        rnd.shuffle(keys)
        for k in keys[:a.synthetic_sample]:
            syn_meta[k] = m[k]
    syn_by_city = defaultdict(list)
    for k, m in syn_meta.items():
        syn_by_city[m["map"]].append(k)

    # per nuPlan scene: the route, and the logged path from the scene's first frame
    scenes = {}
    for lg, rows in frames.items():
        for r in rows:
            key = (lg, r["scene_token"])
            sc = scenes.setdefault(key, {"first_i": r["i"], "routes": set(), "n": 0})
            sc["routes"].add(r["rb"])
            sc["n"] += 1
    n_scene_multi_route = sum(1 for s in scenes.values() if len(s["routes"]) > 1)

    OS = load_openscene_command_fn()
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    rec_a, rec_d_future, rec_d_straight, rec_syn = [], [], [], []
    route_prov = []
    timing = {}
    for city in sorted(set(list(by_city) + list(syn_by_city))):
        if a.cities and city not in a.cities:
            continue
        free = _free_mb()
        if free == free and free < a.min_free_mb:
            print(f"[leak] SKIP city {city}: free RAM {free:.0f} MB < {a.min_free_mb}", flush=True)
            timing[city] = {"skipped_low_ram_mb": free}
            continue
        tc = time.time()
        map_api = get_maps_api(MAPS_ROOT, MAP_VERSION, city)
        # (a) + (d) on the stage-1 probe frames
        for t in sorted(by_city.get(city, [])):
            lg, i = tok_index[t]
            rows = frames[lg]
            r = rows[i]
            pose = StateSE2(r["x"], r["y"], r["yaw"])
            rec = {"token": t, "log": lg, "stored": stored[t],
                   "split": ("navhard" if t in split_tokens["navhard_two_stage"] else
                             "warmup" if t in split_tokens["warmup_two_stage"] else "navtest")}
            try:
                rec["recomputed"] = _cmd_name(OS.get_driving_command(pose, map_api, list(r["rb"])))
            except Exception as e:
                rec["recomputed"] = f"ERROR:{type(e).__name__}"
            try:
                fr = future_route(map_api, rows, i, a.future_route_frames)
                rec["future_route_len"] = len(fr)
                rec["from_future_route"] = (_cmd_name(OS.get_driving_command(pose, map_api, fr))
                                            if fr else "no-route")
            except Exception as e:
                rec["from_future_route"] = f"ERROR:{type(e).__name__}"
            try:
                sr = straight_route(map_api, r["x"], r["y"], r["yaw"])
                rec["straight_route_len"] = len(sr)
                rec["from_straight_route"] = (_cmd_name(OS.get_driving_command(pose, map_api, sr))
                                              if sr else "no-route")
            except Exception as e:
                rec["from_straight_route"] = f"ERROR:{type(e).__name__}"
            rec_a.append(rec)
        # (b) route provenance per nuPlan scene in this city — per-frame roadblock
        #     membership computed ONCE per log, then reused by every scene of the log
        member = {}
        for lg, rows in frames.items():
            if rows and rows[0]["map"] == city:
                member[lg] = [contained_roadblocks(map_api, r["x"], r["y"]) for r in rows]
        for (lg, sct), sc in scenes.items():
            rows = frames[lg]
            if rows[0]["map"] != city or len(sc["routes"]) != 1:
                continue
            route = list(dict.fromkeys(next(iter(sc["routes"]))))
            if not route:
                continue
            i0 = sc["first_i"]
            rset = set(route)
            mem = member[lg]
            # where along the route is the ego at the scene's first frame?
            cur_idx = min((route.index(b) for b in mem[i0] if b in rset), default=None)
            first_seen = {}
            for k, bl in enumerate(mem):
                for b in bl:
                    if b in rset and b not in first_seen:
                        first_seen[b] = k
            past = [b for b in route if b in first_seen and first_seen[b] < i0]
            fut_order = sorted(((first_seen[b], route.index(b)) for b in route
                                if b in first_seen and first_seen[b] >= i0))
            ranks = [rk for _, rk in fut_order]
            in_order = all(x < y for x, y in zip(ranks, ranks[1:]))
            ahead = route[cur_idx:] if cur_idx is not None else route
            ahead_visited = [b for b in ahead if b in first_seen and first_seen[b] >= i0]
            # blocks never visited: are they simply beyond the end of the recorded segment?
            last_in = max((route.index(b) for b in mem[-1] if b in rset), default=None)
            never = [b for b in route if b not in first_seen]
            never_after_log_end = [b for b in never
                                   if last_in is not None and route.index(b) > last_in]
            try:
                fr = future_route(map_api, rows, i0, a.expert_route_frames)
            except Exception:
                fr = None
            fr_d = list(dict.fromkeys(fr)) if fr else []
            m_ = min(len(ahead), len(fr_d))
            # NULL CONTROL: the same test on a NO-FUTURE route (straight-ahead successors
            # from the scene-start pose). A property every route has is not evidence.
            try:
                sr = straight_route(map_api, rows[i0]["x"], rows[i0]["y"], rows[i0]["yaw"])
            except Exception:
                sr = []
            sset = set(sr)
            s_seen = {}
            for k, bl in enumerate(mem[i0:]):
                for b in bl:
                    if b in sset and b not in s_seen:
                        s_seen[b] = k
            s_last = max((sr.index(b) for b in mem[-1] if b in sset), default=None)
            s_never = [b for b in sr if b not in s_seen]
            s_ok = bool(sr) and all(s_last is not None and sr.index(b) > s_last for b in s_never)
            route_prov.append({
                "log": lg, "scene_token": sct, "route_len": len(route),
                "ego_block_index_in_route_at_scene_start": cur_idx,
                "n_route_blocks_before_ego_visited_in_PAST": len(past),
                "n_route_blocks_ahead": len(ahead),
                "frac_ahead_visited_in_FUTURE": (round(len(ahead_visited) / len(ahead), 4)
                                                 if ahead else None),
                "future_visits_in_route_order": bool(in_order),
                "n_never_visited": len(never),
                "n_never_visited_beyond_last_logged_block": len(never_after_log_end),
                "ego_in_route_at_log_end": last_in is not None,
                "route_ahead_equals_expert_trajectory_route_prefix": (
                    bool(m_ > 0 and ahead[:m_] == fr_d[:m_])),
                "expert_route_len": len(fr_d),
                "NULL_straight_route_len": len(sr),
                "NULL_straight_route_fully_visited_or_beyond_log_end": s_ok,
                "n_future_frames": len(rows) - i0})
        # stage-2 synthetic scenes: is the command recomputed for the PERTURBED start?
        for k in sorted(syn_by_city.get(city, [])):
            m = syn_meta[k]
            x, y, yaw = m["cur_pose"][:3]
            pose = StateSE2(x, y, yaw)
            rec = {"synthetic": k, "stored": _cmd_name(m["cur_cmd"]),
                   "in_global_frame": m["cur_in_global_frame"]}
            try:
                rec["recomputed_at_synthetic_pose"] = _cmd_name(
                    OS.get_driving_command(pose, map_api, list(m["cur_rb"])))
            except Exception as e:
                rec["recomputed_at_synthetic_pose"] = f"ERROR:{type(e).__name__}"
            rec_syn.append(rec)
        timing[city] = {"s": round(time.time() - tc, 1), "free_mb_before": round(free),
                        "n_stage1": len(by_city.get(city, [])),
                        "n_synthetic": len(syn_by_city.get(city, []))}
        print(f"[leak] city {city}: {timing[city]}  ({time.time() - t0:.0f}s)", flush=True)
        del map_api
        try:
            get_maps_api.cache_clear()
        except Exception:
            pass
        gc.collect()

    def rate(recs, a_key, b_key="stored", subset=None):
        rr = [r for r in recs if (subset is None or subset(r))
              and not str(r.get(a_key, "")).startswith(("ERROR", "no-route"))]
        ok = sum(1 for r in rr if r[a_key] == r[b_key])
        return {"n": len(rr), "agree": ok, "rate": round(ok / len(rr), 4) if rr else None,
                "n_error_or_no_route": sum(1 for r in recs if (subset is None or subset(r))
                                           and str(r.get(a_key, "")).startswith(("ERROR", "no-route")))}

    turning = (lambda r: r["stored"] in ("left", "right"))
    out = {
        "_what": __doc__.split("\n")[0],
        "evidence_class": "MEASURED (local OpenScene test metadata + nuPlan maps v1.0 + OpenScene@7286074 "
                          "get_driving_command + nuplan-devkit get_roadblock_ids_from_trajectory)",
        "frames_read": n_frames, "logs_read": len(frames),
        "n_probe_stage1_frames": len(rec_a),
        "n_nuplan_scenes": len(scenes),
        "n_nuplan_scenes_with_more_than_one_route": n_scene_multi_route,
        "a_reproduction": {
            "all": rate(rec_a, "recomputed"),
            "turning_only": rate(rec_a, "recomputed", subset=turning),
            "by_split": {s: rate(rec_a, "recomputed", subset=lambda r, s=s: r["split"] == s)
                         for s in ("navhard", "warmup", "navtest")},
            "matrix": agreement([(r["stored"], r["recomputed"]) for r in rec_a]),
        },
        "d_counterfactual": {
            "from_FUTURE_route": {"all": rate(rec_a, "from_future_route"),
                                  "turning_only": rate(rec_a, "from_future_route", subset=turning),
                                  "matrix": agreement([(r["stored"], r["from_future_route"])
                                                       for r in rec_a])},
            "from_NO_FUTURE_straight_route": {
                "all": rate(rec_a, "from_straight_route"),
                "turning_only": rate(rec_a, "from_straight_route", subset=turning),
                "matrix": agreement([(r["stored"], r["from_straight_route"]) for r in rec_a])},
        },
        "b_route_provenance": _route_prov_summary(route_prov),
        "c_agreement": agree,
        "stage2_synthetic": {
            "n": len(rec_syn),
            "stored_command_distribution": dict(Counter(r["stored"] for r in rec_syn)),
            "recomputed_at_synthetic_pose_agrees": rate(rec_syn, "recomputed_at_synthetic_pose"),
            "in_global_frame_values": dict(Counter(r["in_global_frame"] for r in rec_syn)),
        },
        "timing": timing, "elapsed_s": round(time.time() - t0, 1),
        "records_stage1": rec_a, "records_route_provenance": route_prov,
        "records_synthetic_first200": rec_syn[:200],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    summ = {k: out[k] for k in ("n_probe_stage1_frames", "n_nuplan_scenes",
                                "n_nuplan_scenes_with_more_than_one_route")}
    summ["a"] = out["a_reproduction"]["all"]
    summ["a_turning"] = out["a_reproduction"]["turning_only"]
    summ["d_future"] = out["d_counterfactual"]["from_FUTURE_route"]["turning_only"]
    summ["d_straight"] = out["d_counterfactual"]["from_NO_FUTURE_straight_route"]["turning_only"]
    summ["b"] = {k: v for k, v in out["b_route_provenance"].items() if k != "first10"}
    summ["syn"] = out["stage2_synthetic"]["recomputed_at_synthetic_pose_agrees"]
    print(json.dumps(summ, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
