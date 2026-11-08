#!/usr/bin/env python3
"""A2 real-data stage 1: build metric-cache PARTS (D9 format) + a seeded 64-proposal fan for N nuPlan trainval states (navtest-excluded by BOTH gate keys).

The states are nuPlan TRAINVAL DBs on the dev box (E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval).  navtrain is a SCENE FILTER over trainval; the filter file is
NOT on this box, so "these are navtrain scenes" is UNVERIFIED -- what IS enforced is navtest-disjointness (log gate + token gate).
The fan is NOT REFe's: it is a seeded perturbation of the ground-truth future + PDM-Closed's own plan (8 speed profiles x 8 lateral profiles), built to EXERCISE every term.
Rates measured on it are therefore term-sensitivity checks, not training-label rates; the rates that matter (REFe's own fan on navtrain) are specified for the pod.

  python build_scenes.py --out <dir> [--n 16] [--seed 20261109]       (venv: driverl-eval; detached; ~0.5-1 min/scene warm, ~45 s per city cold)
"""
import argparse
import hashlib
import json
import lzma
import os
import pickle
import sqlite3
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
NAVSIM = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
TIP = os.environ.get("A2_TIP_REFE", "F:/a2_scratch/tip_refe")
A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [NAVSIM, TIP, os.path.join(A2, "overlay", "refe")]
import numpy as np  # noqa: E402

DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
MAPS = "F:/Archive/devbox-C/navsim/data/maps"
MAP_NAMES = {"las_vegas": "us-nv-las-vegas-strip", "boston": "us-ma-boston", "pittsburgh": "us-pa-pittsburgh-hazelwood", "singapore": "sg-one-north"}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def select(n, seed):
    import behaviour_gate as BG
    out, h = [], (lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    for name in sorted((f for f in os.listdir(DB) if f.endswith(".db")), key=h):
        if len(out) >= n:
            break
        if BG.is_navtest(name[:-3], "0" * 16)[0]:
            continue
        con = sqlite3.connect(f"file:{DB}/{name}?mode=ro", uri=True)
        try:
            cnt = con.execute("select count(*) from lidar_pc").fetchone()[0]
            if cnt < 500:
                continue
            lo, hi = 40, cnt - 260
            r = lo + int(h(name + ":row")[:8], 16) % (hi - lo)
            tok, ts = con.execute("select token, timestamp from lidar_pc order by timestamp limit 1 offset ?", (r,)).fetchone()
            loc = con.execute("select location from log").fetchone()[0]
            mv = con.execute("select map_version from log").fetchone()[0]
        finally:
            con.close()
        BG.assert_not_navtest(name[:-3], tok.hex())
        out.append(dict(db=f"{DB}/{name}", log=name[:-3], row=int(r), token=tok.hex(), ts=int(ts), map_name=MAP_NAMES.get(loc, mv), location=loc))
    return out


def fan_from(gt20, ref20):
    """gt20/ref20 [20,3] ego-frame poses at 0.2..4.0 s -> [64,20,3]: 8 speed profiles (time warps of the GT path) x 8 lateral profiles."""
    t = 0.2 * np.arange(1, 21)
    full_t = np.concatenate([[0.0], t])
    gt = np.concatenate([np.zeros((1, 3)), gt20])
    path_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(gt[:, :2], axis=0).T))])
    ext_t = np.concatenate([full_t, full_t[-1] + 0.2 * np.arange(1, 21)])             # extrapolate the GT path straight on for speed-ups
    v_end = max(path_s[-1] - path_s[-5], 1e-3) / 0.8
    ext_s = np.concatenate([path_s, path_s[-1] + v_end * 0.2 * np.arange(1, 21)])
    ext_xy = np.concatenate([gt[:, :2], gt[-1, :2] + (gt[-1, :2] - gt[-5, :2]) / 0.8 * 0.2 * np.arange(1, 21)[:, None]])
    ext_h = np.concatenate([gt[:, 2], np.full(20, gt[-1, 2])])

    def warp(g):                                     # g(t) = time-warp: positions at warped time
        tw = np.clip(g, 0.0, ext_t[-1])
        x = np.interp(tw, ext_t, ext_xy[:, 0]); y = np.interp(tw, ext_t, ext_xy[:, 1]); h = np.interp(tw, ext_t, np.unwrap(ext_h))
        return np.stack([x, y, h], -1)
    warps = [0.0 * t, 0.3 * t, 0.6 * t, 0.85 * t, 1.0 * t, 1.2 * t, t + 0.9 * np.sin(2 * np.pi * t / 1.6) / (2 * np.pi / 1.6) * 0.6,
             np.where(t < 2.0, 0.2 * t, 0.4 + 1.8 * (t - 2.0))]            # stop, slow x3, GT, fast, oscillating speed, brake-then-go
    lats = [lambda t: 0 * t - 1.5, lambda t: 0 * t - 0.75, lambda t: 0 * t, lambda t: 0 * t + 0.75, lambda t: 0 * t + 1.5,
            lambda t: 1.5 * t / 4.0, lambda t: -1.5 * t / 4.0, lambda t: 1.0 * np.sin(2 * np.pi * t / 3.0)]
    out = []
    for w in warps:
        base = warp(w)
        hd = base[:, 2]
        for lat in lats:
            off = lat(t)
            p = base.copy()
            p[:, 0] += -np.sin(hd) * off
            p[:, 1] += np.cos(hd) * off
            out.append(p)
    fan = np.stack(out)
    fan[..., 2] = wrap(fan[..., 2])
    return fan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
    import navsim_pdm_targets as PT
    import online_labeller as OL
    import build_metric_cache as BM
    states = select(a.n, a.seed)
    json.dump(states, open(os.path.join(a.out, "states.json"), "w"), indent=1)
    t00 = time.time()
    for k, st in enumerate(states):
        t0 = time.time()
        if os.path.isfile(os.path.join(a.out, "parts", st["log"], f"{st['token']}.pkl.xz")) and os.path.isfile(os.path.join(a.out, f"gt_{k:02d}.npy")):
            continue                                                      # resumable
        try:
            sc = NuPlanScenario(data_root="E:/Projects/TanitAD/data/nuplan", log_file_load_path=st["db"], initial_lidar_token=st["token"], initial_lidar_timestamp=st["ts"],
                                scenario_type="unknown", map_root=MAPS, map_version="nuplan-maps-v1.0", map_name=st["map_name"],
                                scenario_extraction_info=ScenarioExtractionInfo("a2", 10.5, 0.0, 0.5), ego_vehicle_parameters=get_pacifica_parameters())
            parts = PT.metric_cache_parts(sc)
            ego0 = sc.initial_ego_state
            fut = sc.get_ego_future_trajectory(0, 4.0, 20)
            fut = list(fut)[-20:]
            c, s_, h0 = np.cos(-ego0.rear_axle.heading), np.sin(-ego0.rear_axle.heading), ego0.rear_axle.heading
            gt = np.array([[(e.rear_axle.x - ego0.rear_axle.x) * c - (e.rear_axle.y - ego0.rear_axle.y) * s_,
                            (e.rear_axle.x - ego0.rear_axle.x) * s_ + (e.rear_axle.y - ego0.rear_axle.y) * c, wrap(e.rear_axle.heading - h0)] for e in fut])
            fan = fan_from(gt, gt)
            for attempt in range(5):                                         # F: occasionally refuses os.replace for a moment (WinError 32)
                try:
                    BM.write_parts(os.path.join(a.out, "parts", st["log"], f"{st['token']}.pkl.xz"), parts)
                    break
                except PermissionError:
                    time.sleep(2.0)
            np.save(os.path.join(a.out, f"fan_{k:02d}.npy"), fan)
            np.save(os.path.join(a.out, f"gt_{k:02d}.npy"), gt)
            print(f"scene {k} {st['log']} {st['location']} done {time.time() - t0:.1f}s (total {time.time() - t00:.0f}s) v0={ego0.dynamic_car_state.rear_axle_velocity_2d.x:.2f}", flush=True)
        except Exception as e:                                           # noqa: BLE001
            import traceback
            print(f"scene {k} {st['log']} FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    open(os.path.join(a.out, "BUILD_DONE.txt"), "w").write("done\n")


if __name__ == "__main__":
    main()
