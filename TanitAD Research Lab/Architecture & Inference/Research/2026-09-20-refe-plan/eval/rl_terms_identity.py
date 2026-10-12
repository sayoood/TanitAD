#!/usr/bin/env python3
"""G-REWARD-ID: does the drivable-excess gap g(t) of refe/rl_terms_label.py vanish EXACTLY when NAVSIM's own DAC passes?  (RL-S1 part 2, priority 1)

    identity under test:   (g(t) > 0 at some executed step)   <=>   (NAVSIM v1.1 DAC = 0)

Data: the navtest metric cache REBUILT from the nuPlan DBs through our scenario builder (the route eval/validate_pdm_targets.py already validated against
NAVSIM's real cache: DAC 100.000 % exact on 19,500 trajectories), scored with NAVSIM's OWN PDMSimulator + PDMScorer on [PDM-Closed, 64 hypotheses, human].
Plans = the final model's 64 hypotheses (proptable/navtest_final/proposals.npz, NAVSIM grid) + the human future.  Tokens: every k-th row of lane_census.jsonl
(k = floor(rows / n)), a deterministic stride -- registered here BEFORE any number is read.

CONTROLS (all measured in the same run, on the same scenes):
  C0  vendored executed states == the real PDMScorer._states, bit for bit
  C1  our bbox corners / centres == navsim state_array_to_coords_array (get_pacifica_parameters())
  C2  per-STEP:  (g > 0)  ==  PDMScorer._ego_areas[:, :, NON_DRIVABLE_AREA]      (a finer identity than the per-proposal one)
  C3  the scorer's DAC (this run) == the census DAC from NAVSIM's real navtest cache
  C4  K inputs: offset == Point.distance(centerline.linestring); dpsi == PDMPath.interpolate(project); in_intersection == dam.is_in_layer;
      multiple_lanes == _ego_areas[:, :, MULTIPLE_LANES]
DELIBERATE-REGRESSION ARMS (must make the identity FAIL -- they are what a mutation of the labeller would do):
  R1  the 8-pose plan instead of the executed states     R2  the untracked 41-step interpolation of the plan (no LQR)
  R3  bbox CENTRE instead of the 4 corners               R4  MIN over corners instead of MAX
    python eval/rl_terms_identity.py [--n 400] [--workers 12] [--out eval/raw/rl_terms/identity_navtest.json]   (re-launches itself in the navsim v1.1 venv)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DRV = os.environ.get("REFE_DRIVE", "E:")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = f"{DRV}/Archive/devbox-C/navsim/data/maps"
DB = f"{DRV}/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
PROPS = f"{DRV}/Projects/TanitAD/data/refe_navtest/proptable/navtest_final/proposals.npz"
CENSUS = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "lane_census.jsonl")
OUT = os.path.join(HERE, "raw", "rl_terms", "identity_navtest.json")
ARMS = ("exact", "R1_eight_pose", "R2_untracked_41", "R3_centre_only", "R4_min_corner")
TOLS = (0.0, 1e-9, 1e-6, 1e-3)


def one_token(sc, plans, census_dac):
    """All measurements for one scene.  Returns a dict of small arrays / counters (no polygons)."""
    import shapely
    from shapely.geometry import Point
    import navsim_dac as ND
    import navsim_pdm_targets as PT
    import rl_terms_label as RT
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, EgoAreaIndex
    parts = PT.metric_cache_parts(sc)
    _, ego, _, centerline, _, dam = parts
    lab = PT.score_plans(parts, plans)                                          # NAVSIM's own scorer, [PDM-Closed, 65 plans]
    scorer = PT._S["scorer"]
    n = len(plans)
    r = {"dac": np.asarray(lab["dac"]), "n": n}
    # ---- C0 / C1
    states = RT.executed_states(parts, plans)                                   # [65, 41, 11]
    ref = scorer._states[1:]
    r["c0_equal"] = bool(np.array_equal(states, ref))
    r["c0_maxabs"] = float(np.abs(states - ref).max())
    vp = get_pacifica_parameters()
    r["params_ok"] = bool(abs(vp.length - RT.PACIFICA["length"]) < 1e-9 and abs(vp.width - RT.PACIFICA["width"]) < 1e-9
                          and abs(vp.rear_axle_to_center - RT.PACIFICA["rear_axle_to_center"]) < 1e-9)
    nv = state_array_to_coords_array(ref, vp)
    mine = RT.bbox_coords(states)
    r["c1_corner_centre_maxabs"] = float(max(np.abs(nv[:, :, int(BBCoordsIndex.CENTER)] - mine["centers"]).max(),
                                             _corner_set_maxabs(nv[:, :, :4], mine["corners"])))
    # ---- the gap, exact arm
    geo = RT.scene_geometry(parts, sc, "DKL")
    g = RT.corner_gap(mine["corners"], geo.drivable)                           # [65, 41]
    r["gmax"] = g.max(axis=-1)
    r["nd_gap_steps"] = int((g > 0).sum())
    r["d"] = RT.RW.drivable_excess(g)
    # ---- C2: per-step identity against the scorer's own flags
    nd = scorer._ego_areas[1:, :, int(EgoAreaIndex.NON_DRIVABLE_AREA)]
    r["c2_steps"] = int(nd.size)
    r["c2_mismatch"] = int(((g > 0) != nd).sum())
    r["c2_nd_steps"] = int(nd.sum())
    r["dis"] = [{"proposal": int(i), "gmax": float(g[i].max()), "dac": float(lab["dac"][i]), "g_pos_steps": int((g[i] > 0).sum()),
                 "scorer_nd_steps": int(nd[i].sum()), "min_positive_g": float(g[i][g[i] > 0].min()) if (g[i] > 0).any() else 0.0}
                for i in range(n) if (g[i].max() > 0) != (lab["dac"][i] == 0)]
    # ---- the regression arms
    x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
    arm_g = {
        "R1_eight_pose": RT.corner_gap(ND.corners_global(plans, x0, y0, h0), geo.drivable).max(axis=-1),
        "R2_untracked_41": RT.corner_gap(ND.corners_global(ND.at_10hz(plans), x0, y0, h0), geo.drivable).max(axis=-1),
        "R3_centre_only": RT.corner_gap(mine["centers"][..., None, :], geo.drivable).max(axis=-1),
        "R4_min_corner": RT.point_gap(mine["corners"], geo.drivable).min(axis=-1).max(axis=-1),
    }
    r["arm_gmax"] = {k: np.asarray(v) for k, v in arm_g.items()}
    # ---- C3 against the census (NAVSIM's real navtest cache)
    r["c3_equal"] = int((np.asarray(lab["dac"]) == np.asarray(census_dac)).sum()) if census_dac is not None else -1
    # ---- C4: K inputs against the scorer's / PDMPath's own functions
    cl_ref = centerline.linestring
    c = mine["centers"]
    off_ref = np.array([[Point(*c[i, t]).distance(cl_ref) for t in range(c.shape[1])] for i in range(n)])
    ct = geo.centerline.trace(c, states[..., 2])
    r["c4_offset_maxabs"] = float(np.abs(ct["offset"] - off_ref).max())
    sref = np.asarray(centerline.project(shapely.points(c.reshape(-1, 2))))
    hp = centerline.interpolate(sref, as_array=True)[..., 2].reshape(c.shape[:2])
    dref = np.degrees(RT.wrap_angle(states[..., 2] - hp))
    r["c4_dpsi_maxabs_deg"] = float(np.abs(RT.wrap_angle(np.radians(ct["dpsi_deg"] - dref))).max() * 180 / np.pi)
    in_ref = np.array([[dam.is_in_layer(_P2D(*c[i, t]), _LAYER("INTERSECTION")) for t in range(c.shape[1])] for i in range(n)])
    in_mine = RT.inside_any(c, geo.intersection)
    r["c4_intersection_mismatch"] = int((in_ref != in_mine).sum())
    mult = RT.multiple_lanes(mine["corners"], geo.lanes)
    r["c4_multiple_lanes_mismatch"] = int((mult != scorer._ego_areas[1:, :, int(EgoAreaIndex.MULTIPLE_LANES)]).sum())
    # ---- K / L on the real scene (distribution + the scene labels)
    t = RT.terms_from_states(states, geo, "DKL")
    r["k_mean64"] = float(t["k"][:64].mean())
    r["k_raw_mean64"] = float(t["k_raw"][:64].mean())
    r["lc_executed64"] = int(t["lc_executed"][:64].sum())
    r["dither64"] = int(t["dither"][:64].sum())
    r["label_fires"] = bool(t["label_fires"])
    r["human_lc"] = bool(geo.lane_ctx["human_lc"])
    r["route_needs"] = bool(geo.lane_ctx["route_needs"])
    r["start_lane"] = not geo.lane_ctx.get("none", False)
    r["intersection_frac"] = float(in_mine[:64].mean())
    r["d_mean64"] = float(r["d"][:64].mean())
    return r


def _lex(x):
    """[..., 4, 2] corners -> the same rows ordered lexicographically by (x, y), so two orderings of one box compare equal."""
    o = np.lexsort((x[..., 1], x[..., 0]), axis=-1)
    return np.take_along_axis(x, o[..., None], axis=-2)


def _corner_set_maxabs(nv4, mine4):
    """navsim returns [FL, RL, RR, FR]; ours [FL, FR, RL, RR]: compare as SETS."""
    return float(np.abs(_lex(np.asarray(nv4)) - _lex(np.asarray(mine4))).max())


def _P2D(x, y):
    from nuplan.common.actor_state.state_representation import Point2D
    return Point2D(x, y)


def _LAYER(name):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    return getattr(SemanticMapLayer, name)


_W: dict = {}


def work(job):
    import navtrain_scenarios as NS
    log, toks = job
    out, errs = [], []
    try:
        scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(f"{DB}/{log}.db", [t["token"] for t in toks], map_root=MAPS,
                                                                                 history_rows=40, future_rows=100)}
    except Exception as e:                                                       # noqa: BLE001
        return [], [f"{log}: build {type(e).__name__}: {str(e)[:100]}"] * len(toks), len(toks)
    skipped = 0
    for t in toks:
        sc = scs.get(t["token"])
        if sc is None:
            skipped += 1
            continue
        try:
            r = one_token(sc, t["plans"], t["census_dac"])
            r["token"], r["log"] = t["token"], log
            out.append(r)
        except Exception:                                                        # noqa: BLE001
            errs.append(f"{t['token']}: {traceback.format_exc()[-300:]}")
    return out, errs, skipped


def cm(pred_off, truth_off):
    """2x2: rows = prediction (g > tol somewhere), columns = NAVSIM DAC == 0.  Counts only."""
    p, t = np.asarray(pred_off, bool), np.asarray(truth_off, bool)
    return {"pred_off_dac0": int((p & t).sum()), "pred_off_dac1": int((p & ~t).sum()),
            "pred_ok_dac0": int((~p & t).sum()), "pred_ok_dac1": int((~p & ~t).sum()), "n": int(p.size)}


def main() -> int:
    if os.environ.get("REFE_RLTID_CHILD") != "1":
        env = dict(os.environ, REFE_RLTID_CHILD="1", PYTHONPATH=f"{V11};{os.path.join(PKG, 'refe')}", PYTHONIOENCODING="utf-8",
                   NUPLAN_MAPS_ROOT=MAPS, NUPLAN_MAP_VERSION="nuplan-maps-v1.0", REFE_SIM_HZ="10", OMP_NUM_THREADS="1")
        return subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    P = np.load(PROPS)
    idx = {str(t): i for i, t in enumerate(P["token"])}
    rows = []
    for line in open(CENSUS, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r and r["token"] in idx and r["token"] in E:
            rows.append({"token": r["token"], "log": r["log"], "dac": r["dac"][1:66]})
    stride = max(1, len(rows) // a.n)
    sel = rows[::stride][:a.n]
    by: dict = {}
    for r in sel:
        plans = np.concatenate([P["proposals"][idx[r["token"]]].astype(np.float64), np.asarray(E[r["token"]]["human_future_poses"], np.float64)[None]], 0)
        by.setdefault(r["log"], []).append({"token": r["token"], "plans": plans, "census_dac": r["dac"]})
    jobs = sorted(by.items())
    tok_digest = hashlib.sha256("\n".join(sorted(r["token"] for r in sel)).encode()).hexdigest()
    print(f"census rows usable {len(rows)}, stride {stride}, tokens {len(sel)} in {len(jobs)} logs, workers {a.workers}", flush=True)
    res, errs, skipped, t0 = [], [], 0, time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for o, e, s in pool.imap_unordered(work, jobs):
            res += o
            errs += e
            skipped += s
            print(f"  {len(res)} tokens, {len(errs)} errors, {time.time() - t0:.0f} s", flush=True)
    res.sort(key=lambda r: r["token"])
    out = {"tokens": len(res), "skipped_by_scenario_guard": skipped, "errors": errs[:20], "n_errors": len(errs), "stride": stride,
           "token_list_sha256": tok_digest, "seconds": round(time.time() - t0, 1),
           "code_sha256": {"rl_terms_label.py": hashlib.sha256(open(os.path.join(PKG, "refe", "rl_terms_label.py"), "rb").read()).hexdigest(),
                           "rl_terms_identity.py": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}}
    dac0 = np.concatenate([r["dac"] == 0 for r in res]).reshape(len(res), -1)                  # [tokens, 65]
    gm = np.stack([r["gmax"] for r in res])
    for name, sl in (("hypotheses_64", slice(0, 64)), ("all_65_incl_human", slice(0, 65))):
        sub = {"proposals": int(dac0[:, sl].size), "dac0_proposals": int(dac0[:, sl].sum())}
        sub["exact_g_gt_0"] = cm(gm[:, sl] > 0, dac0[:, sl])
        sub["tolerance_sweep"] = {f"g>{t:g}": cm(gm[:, sl] > t, dac0[:, sl]) for t in TOLS}
        sub["regression_arms"] = {k: cm(np.stack([r["arm_gmax"][k] for r in res])[:, sl] > 0, dac0[:, sl]) for k in ARMS[1:]}
        out[name] = sub
    # D by DAC outcome (hypotheses)
    d = np.stack([r["d"] for r in res])[:, :64]
    out["D_by_dac"] = {"dac0": {"n": int(dac0[:, :64].sum()), "mean": float(d[dac0[:, :64]].mean()) if dac0[:, :64].any() else None,
                                "min": float(d[dac0[:, :64]].min()) if dac0[:, :64].any() else None},
                       "dac1": {"n": int((~dac0[:, :64]).sum()), "max": float(d[~dac0[:, :64]].max()), "all_exactly_zero": bool((d[~dac0[:, :64]] == 0).all())}}
    # disagreements, diagnosed
    dis = [dict(token=r["token"], **x) for r in res for x in r["dis"]]
    out["disagreements_n"] = len(dis)
    out["disagreements_first_25"] = dis[:25]
    dh = d[dac0[:, :64]]
    gh = gm[:, :64][dac0[:, :64]]
    q = lambda v: {k: float(np.quantile(v, p)) for k, p in (("min", 0), ("p10", .1), ("p50", .5), ("p90", .9), ("max", 1))} if len(v) else None
    out["graded_signal"] = {"D_among_dac0_hypotheses": q(dh), "gmax_m_among_dac0_hypotheses": q(gh),
                            "share_dac0_with_D_lt_0.01": float((dh < 0.01).mean()) if len(dh) else None}
    n_dac0 = dac0[:, :64].sum(axis=1)
    all0, all1 = n_dac0 == 64, n_dac0 == 0
    d64 = np.stack([r["d"] for r in res])[:, :64]
    out["tie_breaking_P3"] = {"scenes": len(res), "all_64_dac0": int(all0.sum()), "all_64_dac1": int(all1.sum()), "mixed": int((~all0 & ~all1).sum()),
                              "all_dac0_scenes_where_D_varies_gt_1e-6": int((d64[all0].std(axis=1) > 1e-6).sum()) if all0.any() else 0,
                              "all_dac1_scenes_D_identically_zero": bool((d64[all1] == 0).all()) if all1.any() else None}
    c = {"c0_states_bit_equal_scenes": f"{sum(r['c0_equal'] for r in res)}/{len(res)}", "c0_maxabs": max(r["c0_maxabs"] for r in res),
         "pacifica_params_match_nuplan": all(r["params_ok"] for r in res), "c1_corner_centre_maxabs_m": max(r["c1_corner_centre_maxabs"] for r in res),
         "c2_steps_compared": sum(r["c2_steps"] for r in res), "c2_step_mismatches": sum(r["c2_mismatch"] for r in res),
         "c2_nondrivable_steps": sum(r["c2_nd_steps"] for r in res),
         "c3_scorer_dac_equals_census": f"{sum(r['c3_equal'] for r in res)}/{sum(r['n'] for r in res)}",
         "c4_offset_maxabs_m": max(r["c4_offset_maxabs"] for r in res), "c4_dpsi_maxabs_deg": max(r["c4_dpsi_maxabs_deg"] for r in res),
         "c4_intersection_mismatches": sum(r["c4_intersection_mismatch"] for r in res),
         "c4_multiple_lanes_mismatches": sum(r["c4_multiple_lanes_mismatch"] for r in res)}
    out["controls"] = c
    n = len(res)
    out["scene_stats"] = {"scenes": n, "start_on_lane": sum(r["start_lane"] for r in res), "human_lane_change_label_i": sum(r["human_lc"] for r in res),
                          "route_needs_label_iii": sum(r["route_needs"] for r in res), "label_fires_i_or_iii": sum(r["label_fires"] for r in res),
                          "lc_executed_of_64": float(np.mean([r["lc_executed64"] for r in res])), "dither_of_64": float(np.mean([r["dither64"] for r in res])),
                          "mean_k_suppressed": float(np.mean([r["k_mean64"] for r in res])), "mean_k_raw": float(np.mean([r["k_raw_mean64"] for r in res])),
                          "mean_d": float(np.mean([r["d_mean64"] for r in res])), "mean_intersection_frac": float(np.mean([r["intersection_frac"] for r in res]))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("disagreements_first_25",)}, indent=1)[:6000])
    ex = out["hypotheses_64"]["exact_g_gt_0"]
    print("ZZRLTID", json.dumps({"tokens": n, "hyp_n": ex["n"], "confusion": ex, "disagreements_all65": out["disagreements_n"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
