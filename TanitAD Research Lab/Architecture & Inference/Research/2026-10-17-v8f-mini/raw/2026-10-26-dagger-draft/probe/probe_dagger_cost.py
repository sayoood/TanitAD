#!/usr/bin/env python3
"""PREREG_V8F_DAGGER feasibility probe (dev-box CPU, <= 4 min, CUDA_VISIBLE_DEVICES=-1).

WHAT IT MEASURES: wall time only, plus identity controls of a prototype route-paired DAgger item builder.
  1. the matcher (route point r') on SYNTHETIC geometry: straight path, stationary tie, lateral cut, heading cut, end edge, loop.
  2. the cost of generating own states: an m = 10 T1-kin rollout of M2b (plain, --limit causal) for TWO forwards per window
     (ticks 0 and 1; tick 2's state is reached by committing tick 1's plan, its own plan is never needed).
  3. the cost of matching each own state and building its route-paired training item, and three identities of that item.

WHAT IT DOES NOT DO: it never stores an own state, never reads a label statistic, never reports an off-route count or a lag
distribution. The rollouts run on eval-141 windows ONLY because the TRAIN cache is not on the dev box; they are used for TIMING
ONLY. The matching constants were fixed in THRESHOLDS_FIXED.txt before this file ran (its sha256 is printed first).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFF_ROUTE_LAT_M, OFF_ROUTE_HEAD_DEG, ARC_TIE_M, H = 3.0, 30.0, 0.05, 30


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def match_route(pg, t0, own, j):
    """r' for an own pose ``own`` = (x, y, yaw) reached at clock row ``j``: project onto the logged polyline pg[t0..T-1] (world xy);
    r' = the row whose cumulative arc is nearest the foot's arc; rows within ARC_TIE_M of that arc are tied -> nearest in time to j, then earlier.
    Returns (r', e_lat, dpsi_deg, reason) with reason in {"ok", "off_lat", "off_head", "end_edge"}."""
    P = np.asarray(pg, np.float64)[t0:, :3]
    xy = P[:, :2]
    seg = np.diff(xy, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(L)])
    q = np.asarray(own[:2], np.float64)
    if len(seg):
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(L > 0, ((q - xy[:-1]) * seg).sum(1) / np.where(L > 0, L * L, 1.0), 0.0)
        u = np.clip(u, 0.0, 1.0)
        foot = xy[:-1] + u[:, None] * seg
        dist = np.hypot(*(foot - q).T)
        k = int(np.argmin(dist))                                   # first minimum (earlier segment) on exact ties
        e_lat, s_star = float(dist[k]), float(arc[k] + u[k] * L[k])
    else:
        e_lat, s_star = float(np.hypot(*(xy[0] - q))), 0.0
    da = np.abs(arc - s_star)
    tied = np.nonzero(da <= da.min() + ARC_TIE_M)[0]
    rows = tied + t0
    r = int(rows[np.lexsort((rows, np.abs(rows - j)))[0]])         # nearest in time to j, then the earlier row
    dpsi = math.degrees(abs(wrap(float(own[2]) - float(pg[r][2]))))
    if e_lat > OFF_ROUTE_LAT_M:
        return r, e_lat, dpsi, "off_lat"
    if dpsi > OFF_ROUTE_HEAD_DEG:
        return r, e_lat, dpsi, "off_head"
    if r > len(pg) - 31:
        return r, e_lat, dpsi, "end_edge"
    return r, e_lat, dpsi, "ok"


def synthetic_tests():
    out = {}
    T = 80
    pg = np.zeros((T, 4)); pg[:, 0] = np.arange(T) * 2.0                     # straight, 10 m/s
    r, e, dp, why = match_route(pg, 5, (40.9, 0.4, 0.0), 30)
    out["straight"] = (r == 20 and abs(e - 0.4) < 1e-9 and why == "ok")
    pg2 = pg.copy(); pg2[20:40, 0] = 40.0; pg2[40:, 0] = 40.0 + np.arange(T - 40) * 2.0     # stationary rows 20..39 at x = 40
    r, e, dp, why = match_route(pg2, 5, (40.0, 0.0, 0.0), 35)
    out["stationary_tie_time_nearest"] = (r == 35 and why == "ok")
    r_first = 20
    out["stationary_tie_differs_from_first_min"] = (r != r_first)
    r, e, dp, why = match_route(pg, 5, (40.0, 3.2, 0.0), 30)
    out["lateral_cut"] = (why == "off_lat")
    r, e, dp, why = match_route(pg, 5, (40.0, 0.0, math.radians(40.0)), 30)
    out["heading_cut"] = (why == "off_head")
    r, e, dp, why = match_route(pg, 5, (2.0 * 60, 0.0, 0.0), 70)
    out["end_edge"] = (why == "end_edge" and r == 60)
    th = np.linspace(0, 2 * math.pi, T)                                         # a loop: the path returns to its start heading the other way at the crossing
    pg3 = np.zeros((T, 4)); pg3[:, 0], pg3[:, 1], pg3[:, 2] = 30 * np.sin(th), 30 * (1 - np.cos(th)), th
    r, e, dp, why = match_route(pg3, 0, (pg3[10, 0], pg3[10, 1], pg3[10, 2]), 10)
    out["loop_on_leg"] = (r == 10 and why == "ok")
    out["pass"] = all(out.values())
    return out


def reanchor(P, own_now, log_now):
    """Own world poses P [..., 3] -> the same poses relative to own_now, placed at log_now (a rigid motion: relative geometry is exact)."""
    P = np.asarray(P, np.float64)
    c, s = math.cos(-own_now[2]), math.sin(-own_now[2])
    dx, dy = P[..., 0] - own_now[0], P[..., 1] - own_now[1]
    rx, ry = dx * c - dy * s, dx * s + dy * c
    C, S_ = math.cos(log_now[2]), math.sin(log_now[2])
    return np.stack([log_now[0] + rx * C - ry * S_, log_now[1] + rx * S_ + ry * C, P[..., 2] - own_now[2] + log_now[2]], -1)


def dagger_item(D, ds_log, ds_own, proxy, st, r, a_t, a_s):
    """The route-paired DAgger item: the LOGGED training item at (ci, r') with its ego tensors replaced by the own state at row st.j."""
    it = ds_log.item(st.ci, r, a_t, a_s)                                     # images, labels, targets, limit, act: logged at r'
    proxy.d = st.d
    ow = ds_own.item(st.ci, st.j, a_t, a_s)                                  # the own state exactly as the T1 rollout builds it
    own_now, log_now = np.asarray(st.d["pose"][st.j][:3], np.float64), np.asarray(it["pose_now"], np.float64)
    out = dict(it)
    out["p10f"], out["kin3"], out["p10_t"], out["base"] = ow["p10f"], ow["kin3"], ow["p10_t"], ow["base"]
    out["frame_ok"] = it["frame_ok"] & ow["frame_ok"]
    out["pose"] = reanchor(np.asarray(st.d["pose"])[np.clip(np.arange(st.j - ds_own.hmax + 1, st.j + 1), 0, st.T - 1)][:, :3], own_now, log_now).astype(np.float32)
    out["pose_t"] = reanchor(ow["pose_t"], own_now, log_now).astype(np.float32)
    out["pose_s"] = reanchor(ow["pose_s"], own_now, log_now).astype(np.float32)
    out["pose_now"] = it["pose_now"]
    # nav progress: anchored at the logged NOW (r'), history differences from the OWN travelled arc (ORC's rule: same token and validity as NOW)
    idx = np.clip(np.arange(st.j - ds_own.hmax + 1, st.j + 1), 0, st.T - 1)
    xy = np.asarray(st.d["pose"], np.float64)[idx, :2]
    back = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    back = back[-1] - back                                                   # own arc from each history frame to NOW
    nav = np.array(it["nav"], np.float32, copy=True)
    same = (nav[:, :3].argmax(-1) == nav[-1, :3].argmax(-1)) & (nav[:, 4] == nav[-1, 4])
    nav[same, 3] = np.clip(nav[-1, 3] + back[same] / 100.0, 0.0, 3.0)
    out["nav"] = nav
    return out, ow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--contiguous", action="store_true", help="the first N grid windows in grid order (clip-ordered, ~17 per clip), as a TRAIN pass reads them")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"thresholds_sha256": hashlib.sha256(open(os.path.join(HERE, "THRESHOLDS_FIXED.txt"), "rb").read()).hexdigest()}
    print("THRESHOLDS_FIXED.txt sha256", res["thresholds_sha256"], flush=True)
    res["synthetic"] = synthetic_tests()
    print("synthetic matcher tests", res["synthetic"], flush=True)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    sys.path[:0] = [os.path.join(a.tree, "stack", "scripts")]
    import v8f_t1_eval as T1
    D, M = T1.D, T1.M
    PK = os.path.join(a.tree, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw")
    t_load = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(a.tracks)
    J = D.load_join(os.path.join(PK, "join_eval141.npz"))
    store = D.ClipStore(a.cache, "eval", lru=8, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(os.path.join(PK, "eval_grid_windows.json"), J)
    wins = wins[:a.n] if a.contiguous else [wins[i] for i in np.linspace(0, len(wins) - 1, a.n).round().astype(int)]
    wins.sort(key=lambda w: (w[0], w[1]))                                   # clip-ordered, as a TRAIN bank pass would run
    proxy = T1._OneClipStore()
    kw = dict(hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    ds_own = D.V8FMiniWindows(proxy, J, [], **kw)
    ds_log = D.V8FMiniWindows(store, J, [], **kw)
    res["load_s"] = time.time() - t_load
    mp = T1.model_planner(model, torch.device("cpu"))
    t_roll, t_match, t_item, n_own, n_ok, t_fwd, t_k0 = 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0
    ident = {"tick0_ego_bitexact": True, "pose_now_reanchor_max_m": 0.0, "rel_pose_t_max": 0.0}
    B = 32
    for b0 in range(0, len(wins), B):
        chunk = wins[b0:b0 + B]
        t1 = time.time()
        states = T1.make_states(store, tracks, J, chunk, "deadreckon")
        snaps = []
        for n in range(2):                                                   # ticks 0 and 1: two forwards per window
            a_t, a_s = T1.ages_for(n * 10, "periodic")
            items = [T1.build_item(ds_own, proxy, st, n, a_t, a_s, "logged", "causal") for st in states]
            full = D.collate(items)
            batch = {k: full[k] for k in T1.MODEL_KEYS}
            tf = time.time()
            with torch.no_grad():
                plan = mp(batch)
            t_fwd += time.time() - tf
            tk = time.time()
            if n == 0:                                                       # identity control: at tick 0 the own state IS the logged state
                for st in states:
                    log0 = ds_log.item(st.ci, st.t0, 0, 0)
                    proxy.d = st.d
                    own0 = ds_own.item(st.ci, st.t0, 0, 0)
                    for k in ("p10f", "kin3", "p10_t", "base", "pose_now"):
                        if not np.array_equal(np.asarray(log0[k]), np.asarray(own0[k]), equal_nan=True):
                            ident["tick0_ego_bitexact"] = False
            t_k0 += time.time() - tk
            for st, pl in zip(states, plan):
                st.commit_plan(pl.astype(np.float64), 10)
            snaps.append([(st, st.j) for st in states])
        t_roll += time.time() - t1
        # tick 2's state is the state after the second commit; tick 1's state was the state after the first commit (own rows <= t0 + 10 of the same track)
        for st in states:
            for j in (st.t0 + 10, st.t0 + 20):
                n_own += 1
                t2 = time.time()
                pg = store.get(str(J["clip"][st.ci]))["pose"]
                own = np.asarray(st.d["pose"][j][:3], np.float64)
                r, e, dp, why = match_route(pg, st.t0, own, j)
                t_match += time.time() - t2
                if why != "ok":
                    continue
                n_ok += 1
                t3 = time.time()
                jj = st.j
                st.j = j                                                     # build the item for the own state at row j (rows > j exist but item() reads only <= j)
                it, ow = dagger_item(D, ds_log, ds_own, proxy, st, r, 0, 0)
                st.j = jj
                t_item += time.time() - t3
                ident["pose_now_reanchor_max_m"] = max(ident["pose_now_reanchor_max_m"], float(np.abs(reanchor(np.asarray(st.d["pose"][j][:3])[None], own, it["pose_now"])[0, :2] - it["pose_now"][:2]).max()))
                rel_own = reanchor(np.asarray(ow["pose_t"], np.float64)[None], own, np.zeros(3))[0]          # pose_t in the own NOW frame
                rel_dag = reanchor(np.asarray(it["pose_t"], np.float64)[None], np.asarray(it["pose_now"], np.float64), np.zeros(3))[0]
                ident["rel_pose_t_max"] = max(ident["rel_pose_t_max"], float(np.abs(rel_own - rel_dag).max()))
    res.update({"n_windows": len(wins), "n_own_states": n_own, "rollout_s": t_roll, "match_s": t_match, "item_s": t_item,
                "rollout_ms_per_window_2fwd": 1000 * t_roll / len(wins), "forward_ms_per_window_2fwd": 1000 * t_fwd / len(wins), "tick0_control_ms_per_window": 1000 * t_k0 / len(wins),
                "generation_ms_per_window_excl_control": 1000 * (t_roll - t_k0) / len(wins), "contiguous": bool(a.contiguous), "match_ms_per_own_state": 1000 * t_match / max(n_own, 1),
                "item_ms_per_own_state": 1000 * t_item / max(n_ok, 1), "threads": a.threads, "identities": ident,
                "note": "eval-141 windows used for TIMING ONLY; no off-route count, lag or label statistic is recorded"})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "synthetic"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
