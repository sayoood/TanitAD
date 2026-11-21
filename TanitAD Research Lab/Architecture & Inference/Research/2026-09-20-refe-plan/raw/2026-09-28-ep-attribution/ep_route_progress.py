#!/usr/bin/env python3
"""EXPLORATORY (W3's 200 SELECTION tokens -- never a claim): does NAVSIM's EXACT EP numerator, computed from admissible
inputs, recover the EP head's share of the selection gap?

NAVSIM v1 EP (navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py, read 2026-09-28):
  raw_i = max(0, centerline.project(end centre) - centerline.project(start centre)) of the LQR-SIMULATED trajectory
  (`_calculate_progress`, :398-412); EP_i = raw_i * multi_i / max_j(raw_j * multi_j) over {PDM-Closed, agent}, or 1
  when that max <= progress_distance_threshold = 5 m (`_aggregate_scores`, :165-173). The PDM-Closed reference is
  PRIVILEGED, so it is replaced here by three admissible references.

progress_i: every candidate of the E-6 table simulated with the VENDORED scorer simulator
(`refe/navsim_dac.py::simulated_states`, LQR + kinematic bicycle), its bounding-box CENTRE projected onto the route
polyline that `REFePlanner._goal_for` builds (the goal path's own object, i.e. navigation input; Amendment 8's
sanitised route where its trigger fires, as the planner now runs it). No map-based DAC, no GT.
References (EP_hat = clip(progress_i / ref, 0, 1); = 1 for all when ref <= 5 m, NAVSIM's own threshold rule):
  (a) scene_max  the max progress over the scene's valid candidates
  (b) cv_v0      the progress of the constant-velocity rollout at v0 (measured v0 at t0 is admissible), simulated alike
  (c) v5x4       max(v0, 5 m/s) x 4 s
Re-pick with navsim_v1 (NC*DAC*(5EP+5TTC+2C)/12 over the SHIPPED heads, EP replaced). CONTROL: EP_hat = the true EP
must reproduce ep_attribution.json's true_head.EP (017 89.44 / 018 89.83).
Decision (coordinator, fixed before the run): a variant that recovers >= +3.4 at 017 AND >= +3.1 at 018 over the
shipped pick earns a DRAFT amendment (confirmation on a7confirm_ep015's 923 fresh tokens); otherwise an elimination.

    python ep_route_progress.py            -> ep_route_progress.json (+ progress cache npz)
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
PT = Path("D:/Projects/TanitAD/data/refe_navtest/proptable")
SUBS = ["NC", "DAC", "EP", "TTC", "C", "DDC"]
SNAPS = ("017", "018")
BAR = {"017": 3.4, "018": 3.1}


def main() -> int:
    import eval_checkpoint as EC
    if os.environ.get("REFE_EPR_CHILD") != "1":
        env = EC.env_driverl()
        env["REFE_EPR_CHILD"] = "1"
        env["OMP_NUM_THREADS"] = "2"
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(PKG / "refe"), str(PKG / "eval"),
                                             str(PKG / "code")])
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__)], cwd=str(PKG / "eval"), env=env)
    for p in (PKG / "refe", PKG / "eval", PKG / "code"):
        sys.path.insert(0, str(p))
    from scipy.stats import spearmanr
    from shapely.geometry import LineString, Point
    import augment_routes as A
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import planner as PL
    import refe_navtest_seam as SEAM
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.script import driverl_runtime_map_features as M
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    tabs = {e: np.load(PT / f"sub200_ep{e}" / "table.npz", allow_pickle=True) for e in SNAPS}
    toks = [str(t) for t in tabs[SNAPS[0]]["token"]]
    assert all([str(t) for t in tabs[e]["token"]] == toks for e in SNAPS)
    cache_p = HERE / "ep_route_progress_cache.npz"
    vp = get_pacifica_parameters()
    if cache_p.exists():
        c = np.load(cache_p)
        prog = {e: c[f"prog_{e}"] for e in SNAPS}
        cvp, v0s, trig = c["cv"], c["v0"], c["trig"]
    else:
        prog = {e: np.full((len(toks), 64), np.nan) for e in SNAPS}
        cvp, v0s, trig = np.full(len(toks), np.nan), np.full(len(toks), np.nan), np.zeros(len(toks), bool)
        by_log: dict = {}
        for i, t in enumerate(toks):
            by_log.setdefault(exp[t]["log_name"], []).append(t)
        t0 = time.time()
        for li, (lg, lt) in enumerate(sorted(by_log.items())):
            for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db"), lt, history_rows=1,
                                                 future_rows=80):
                t = sc._initial_lidar_token
                i = toks.index(t)
                ego = sc.get_ego_state_at_iteration(0)
                anchor = A._anchor_from_ego(ego)
                poly, _ = A._route_with_lane_rank(sc.map_api, list(sc.get_route_roadblock_ids() or []), anchor, 0)
                if poly is not None and PL.ego_to_polyline_m(poly) > PL.GOAL_SANITIZE_D_M:   # Amendment 8's route
                    c_ = np.asarray(exp[t]["ego_statuses"][-1]["driving_command"], float)
                    cmd = PL.DRIVING_COMMANDS[int(np.argmax(c_))] if c_.max() > 0 else "UNKNOWN"
                    fb, _d = PL.fallback_route(sc.map_api, anchor, cmd, M)
                    trig[i] = True
                    if fb is not None:
                        poly = fb
                if poly is None:
                    continue
                line = LineString(np.asarray(poly, np.float64))
                x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
                ch, sh = np.cos(-h0), np.sin(-h0)

                def progress(props8):
                    sim = ND.simulated_states(props8, ego)
                    cen = state_array_to_coords_array(sim, vp)[:, :, int(BBCoordsIndex.CENTER)]   # [P, 41, 2] global
                    out = []
                    for p in cen:
                        loc = [((x - x0) * ch - (y - y0) * sh, (x - x0) * sh + (y - y0) * ch) for x, y in (p[0], p[-1])]
                        out.append(max(0.0, line.project(Point(*loc[1])) - line.project(Point(*loc[0]))))
                    return np.array(out)
                v = ego.dynamic_car_state.rear_axle_velocity_2d
                v0 = float(np.hypot(v.x, v.y))
                v0s[i] = v0
                tt = 0.5 * np.arange(1, 9)
                cvp[i] = progress(np.stack([np.stack([v0 * tt, np.zeros(8), np.zeros(8)], -1)]))[0]
                for e in SNAPS:
                    prog[e][i] = progress(np.asarray(tabs[e]["proposals"][i], np.float64))
            if (li + 1) % 5 == 0:
                print(f"  [{li + 1}/{len(by_log)}] {time.time() - t0:.0f} s", flush=True)
        np.savez(cache_p, token=np.array(toks), cv=cvp, v0=v0s, trig=trig, **{f"prog_{e}": prog[e] for e in SNAPS})
    att = json.load(open(HERE / "ep_attribution.json", encoding="utf-8"))["snapshots"]
    res = {"_label": "EXPLORATORY (W3's 200 selection tokens): the exact NAVSIM EP numerator from admissible inputs",
           "ep_definition": "pdm_scorer.py:398-412 raw progress (LQR-simulated bbox centre, projected on the route "
                            "centerline, clipped >= 0); :165-173 normalised by the max over {PDM-Closed, agent} "
                            "(privileged, replaced here), = 1 when that max <= 5 m",
           "n_tokens": len(toks), "tokens_without_route": int(np.isnan(prog[SNAPS[0]]).all(1).sum()),
           "amendment8_route_used_on": int(trig.sum()), "snapshots": {}}
    for e in SNAPS:
        T = tabs[e]
        P, S, V = T["pdms"], T["sub"], T["valid"].astype(bool)
        L = T["logits"].astype(np.float64)
        n, ar = P.shape[0], np.arange(P.shape[0])
        sig = 1 / (1 + np.exp(-L))
        g = {k: sig[..., SUBS.index(k)] for k in SUBS}
        true_ep = S[..., SUBS.index("EP")]

        def pick(ep):
            agg = g["NC"] * g["DAC"] * (5 * ep + 5 * g["TTC"] + 2 * g["C"]) / 12
            return float(P[ar, np.where(V, agg, -np.inf).argmax(1)].mean() * 100)
        shipped = float(P[ar, T["pick"]].mean() * 100)
        pr = np.nan_to_num(prog[e], nan=0.0)
        refs = {"a_scene_max": np.where(V, pr, -np.inf).max(1),
                "b_cv_v0": np.nan_to_num(cvp, nan=0.0),
                "c_v5x4": np.maximum(np.nan_to_num(v0s, nan=0.0), 5.0) * 4.0}
        r = {"shipped_pick": shipped, "control_true_EP": pick(true_ep),
             "control_reference_true_head_EP": att[e]["true_head"]["EP"],
             "true_EP_gain": att[e]["true_head"]["EP"] - shipped, "bar": BAR[e], "variants": {}}
        r["control_ok"] = abs(r["control_true_EP"] - r["control_reference_true_head_EP"]) < 1e-9
        for name, ref in refs.items():
            eh = np.clip(pr / np.maximum(ref[:, None], 1e-9), 0.0, 1.0)
            eh[ref <= 5.0] = 1.0
            rho = [spearmanr(eh[i][V[i]], true_ep[i][V[i]]).correlation for i in range(n)
                   if V[i].sum() > 2 and np.ptp(true_ep[i][V[i]]) > 0 and np.ptp(eh[i][V[i]]) > 0]
            mae = float(np.mean(np.abs(eh[V] - true_ep[V])))
            pk = pick(eh)
            r["variants"][name] = {"pick_pdms": pk, "gain": pk - shipped, "passes_bar": pk - shipped >= BAR[e],
                                   "within_scene_spearman_median": float(np.nanmedian(rho)) if rho else None,
                                   "n_scenes_rho": len(rho), "mae_vs_true_EP": mae}
        rho_raw = [spearmanr(pr[i][V[i]], true_ep[i][V[i]]).correlation for i in range(n)
                   if V[i].sum() > 2 and np.ptp(true_ep[i][V[i]]) > 0 and np.ptp(pr[i][V[i]]) > 0]
        r["raw_progress_within_scene_spearman_median"] = float(np.nanmedian(rho_raw))
        r["EP_head_within_scene_spearman_median"] = att[e]["EP_head_within_scene_spearman_median"]
        res["snapshots"][e] = r
    passing = [v for v in res["snapshots"]["017"]["variants"]
               if all(res["snapshots"][e]["variants"][v]["passes_bar"] for e in SNAPS)]
    res["variants_passing_both_bars"] = passing
    res["decision"] = ("DRAFT an amendment (confirmation on a7confirm_ep015's 923 fresh tokens)" if passing else
                       "ELIMINATED: no admissible reference recovers half of the true-EP gain at both snapshots")
    res["at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(res, open(HERE / "ep_route_progress.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    for e in SNAPS:
        r = res["snapshots"][e]
        print(e, f"shipped {r['shipped_pick']:.2f} control {r['control_true_EP']:.2f} (ref {r['control_reference_true_head_EP']:.2f}) "
                 f"ok {r['control_ok']}; raw-progress rho {r['raw_progress_within_scene_spearman_median']:.3f}; EP head rho "
                 f"{r['EP_head_within_scene_spearman_median']:.3f}")
        for k, v in r["variants"].items():
            print(f"   {k}: pick {v['pick_pdms']:.2f} gain {v['gain']:+.2f} (bar {r['bar']}) rho {v['within_scene_spearman_median']} "
                  f"mae {v['mae_vs_true_EP']:.3f}")
    print("DECISION:", res["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
