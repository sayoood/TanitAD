"""S0-D (R-sim) driver: the agent LIST through NAVSIM's own scorer (at-fault NC and TTC), arms R-sim-true / -t0 / -cv / C-SIM-NONE.

  python s0d_rsim.py --smoke 2 --out <dir>              counts and timings ONLY (+ the R-sim-true identity control); never gains
  python s0d_rsim.py --out <dir>                        FULL read, held-out covered sets (1,343 sets, 13 logs) = the deciding column
  python s0d_rsim.py --out <dir> --all                  all 3,958 covered sets (47 logs); the held-out column is reported inside it
  python s0d_rsim.py --out <dir> --assemble-only        re-assemble the result from <dir>/parts (no scoring)

Per-log checkpoint files go to <out>/parts/<log>.npz; a re-run skips finished logs (the full read can be split over several
processes in time; ONE process at a time).  Environment = s0b_cid.py exactly: driverl-eval venv (DriveRL nuPlan fork) + navsim 3e8291b,
CUDA_VISIBLE_DEVICES=-1.  Reuses s0b_run / s0b_readout UNCHANGED (covered_sets, set_gains, oracle_gains, column_report, gfid).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
HERE = Path(__file__).resolve().parent
S0B = Path(os.environ.get("S0D_S0B_DIR", str(HERE.parent)))            # the S0-B hold: s0b_run / s0b_readout / s0b_raster / ref/
V1 = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
MAP_ROOT = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
sys.path[:0] = [str(HERE), str(S0B), V1, str(S0B / "ref" / "refe")]   # same order as s0b_cid.py; venv nuplan, NOT the F: devkit
os.environ.setdefault("NUPLAN_MAPS_ROOT", MAP_ROOT)
import numpy as np  # noqa: E402

ARMS = ("R-sim-true", "R-sim-t0", "R-sim-cv", "C-SIM-NONE", "R-sim-t0v")      # index 3 = NONE (monotone check); t0v is secondary
ARM_MODE = {"R-sim-true": "true", "R-sim-t0": "t0", "R-sim-cv": "cv", "C-SIM-NONE": "none", "R-sim-t0v": "t0v"}
GO_T0, MOTION_CV = 0.25, 0.50
SMOKE_SEED = 20261114


# ----------------------------------------------------------------------------------------------- mechanics (pure, unit-tested)
def outcome_row(identity_ok, none_ok, c_t0, c_cv, c_t0v=None):
    """The committed table, mechanically. Returns (row, reasons).  c_* = capture of O_NC+TTC (None if the denominator is 0)."""
    why = []
    if not identity_ok:
        why.append("R-sim-true is not identical to truth")
    if not none_ok:
        why.append("C-SIM-NONE violates the monotone NC check")
    if why:
        return "INVALID", why
    if c_t0 is None or c_cv is None:
        return "NOT RUN", ["denominator is zero"]
    if c_t0 >= GO_T0:
        return "S0D-GO", []
    if c_t0v is not None and c_t0v >= GO_T0:
        return "S0D-STOPFLAG", []                                     # pose-only fails, but a moving/stopped flag on the t0 pose clears 0.25
    if c_cv >= MOTION_CV:
        return "S0D-MOTION", []
    return "S0D-NULL", []


def identity_check(nc_flag_true, ttc_flag_true, nc_truth, ttc_truth):
    """R-sim-true flags vs truth (nc==0 / ttc==0 from the fit pack), proposal by proposal. -> n mismatching proposals."""
    return int(((nc_flag_true != (nc_truth == 0)) | (ttc_flag_true != (ttc_truth == 0))).sum())


def monotone_none(nc_flag_none, nc_flag_true):
    """Removing agents cannot create an agent collision.  Per set: NC-flag COUNT of NONE <= TRUE's, and (stricter) the NONE
    flags are a subset of TRUE's per proposal.  -> (n_count_violations, n_subset_violations) over sets."""
    cnt = int((nc_flag_none.sum(axis=1) > nc_flag_true.sum(axis=1)).sum())
    sub = int((nc_flag_none & ~nc_flag_true).any(axis=1).sum())
    return cnt, sub


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)
    except Exception:                                               # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------- NAVSIM glue
def base_parts(sc):
    """PDM-Closed trajectory etc. for a scenario: PT.metric_cache_parts minus the observation (which the arms build)."""
    import navsim_pdm_targets as PT
    S = PT._setup()
    proc = S["proc"]
    planner_input, planner_initialization = proc._get_planner_inputs(sc)          # real scenario: the hook is NOT applied here
    proc._pdm_closed.initialize(planner_initialization)
    pdm_traj = proc._pdm_closed.compute_planner_trajectory(planner_input)
    return (pdm_traj, sc.initial_ego_state, proc._pdm_closed._centerline, list(proc._pdm_closed._route_lane_dict.keys()),
            proc._pdm_closed._drivable_area_map)


def prepare_sim(parts5, plans8):
    """The proposal simulation of PT.score_plans, done ONCE per token (it does not depend on the observation)."""
    import navsim_pdm_targets as PT
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    S = PT._setup()
    pdm_traj, ego = parts5[0], parts5[1]
    states = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for p in np.asarray(plans8, dtype=np.float32):
        it = transform_trajectory(Trajectory(p, S["samp8"]), ego)
        states.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    return S["sim"].simulate_proposals(np.stack(states, 0), ego)


def score_obs(sim, parts5, obs):
    """NAVSIM's scorer on an observation -> (nc (M,), ttc (M,)) of the proposals (PDM-Closed dropped), as PT.score_plans reads them."""
    import navsim_pdm_targets as PT
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    sc = PT._setup()["scorer"]
    sc.score_proposals(sim.copy(), obs, parts5[2], parts5[3], parts5[4])
    nc = sc._multi_metrics[MultiMetricIndex.NO_COLLISION][1:].copy()
    ttc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.TTC])[1:].copy()
    return nc, ttc


def run_token(sc, props, truth_nc, truth_ttc, verify_unhooked=False):
    """-> dict: flags (4,2,64) bool [arm, nc/ttc], raw_mismatch (true arm raw nc/ttc vs truth), timings."""
    import navsim_pdm_targets as PT
    from s0d_hook import HookedScenario
    proc = PT._setup()["proc"]
    tm = {}
    t = time.time()
    parts5 = base_parts(sc)
    tm["pdm_closed"] = time.time() - t
    t = time.time()
    plans8 = PT.executed_plans8(props)
    sim = prepare_sim(parts5, plans8)
    tm["sim"] = time.time() - t
    cache = {}
    flags = np.zeros((len(ARMS), 2, 64), dtype=bool)
    out = {"timings": tm}
    for ai, arm in enumerate(ARMS):
        t = time.time()
        obs = proc._interpolate_gt_observation(HookedScenario(sc, ARM_MODE[arm], cache))
        tm[f"obs_{arm}"] = time.time() - t
        t = time.time()
        nc, ttc = score_obs(sim, parts5, obs)
        tm[f"score_{arm}"] = time.time() - t
        flags[ai, 0], flags[ai, 1] = nc == 0, ttc == 0
        if arm == "R-sim-true":
            out["raw_mismatch"] = int((nc != truth_nc).sum() + (ttc != truth_ttc).sum())
            out["raw_max_abs_diff"] = float(max(np.abs(nc - truth_nc).max(), np.abs(ttc - truth_ttc).max()))
            if verify_unhooked:                                      # the UNHOOKED path: NAVSIM's own obs, PT.score_plans
                pdm_traj, ego, centerline, lanes, dam = parts5
                obs_u = proc._interpolate_gt_observation(sc)
                nc_u, ttc_u = score_obs(sim, parts5, obs_u)
                sp = PT.score_plans((pdm_traj, ego, obs_u, centerline, lanes, dam), plans8)
                out["hooked_vs_unhooked_mismatch"] = int((nc != nc_u).sum() + (ttc != ttc_u).sum())
                out["split_vs_score_plans_mismatch"] = int((nc_u != np.asarray(sp["nc"])).sum() + (ttc_u != np.asarray(sp["ttc"])).sum())
    out["flags"] = flags
    return out


def iter_log(db, tokens):
    import navtrain_scenarios as NS
    for sc in NS.build_scenarios_for_log(db, list(tokens), map_root=MAP_ROOT, history_rows=40, future_rows=100):
        yield str(sc.token), sc


# ----------------------------------------------------------------------------------------------- data
def load_pack():
    import s0b_run as RUN
    sha = RUN.sha256_file(RUN.PACK)
    assert sha.startswith(RUN.PACK_SHA_PREFIX), f"fit pack sha mismatch {sha}"
    z = np.load(RUN.PACK, allow_pickle=False)
    pk = {k: z[k] for k in ("log", "token", "traj_xy", "yaw", "logits", "pdm", "pdm_names", "is_heldout", "has_pdm")}
    found = [l.strip() for l in open(S0B / "found_logs.txt", encoding="utf-8") if l.strip()]
    rows = RUN.covered_sets(pk, found)
    return RUN, pk, sha, rows


def population(pk, rows, all_sets, selftest=False):
    ho = pk["is_heldout"][rows].astype(bool)
    sel = [j for j in range(len(rows)) if all_sets or ho[j]]
    if selftest:                                                     # code-path self-test only: ONE log, its first 6 sets
        lg0 = str(pk["log"][rows[sel[0]]])
        sel = [j for j in sel if str(pk["log"][rows[j]]) == lg0][:6]
    by_log = {}
    for j in sel:
        by_log.setdefault(str(pk["log"][rows[j]]), []).append(j)
    return sel, by_log


def process_log_to_part(RUN, pk, rows, names, log, js, parts_dir):
    """Score every set of one log; write parts/<log>.npz.  Returns (n_ok, seconds)."""
    t_log = time.time()
    db = RUN.db_path(log)
    toks = [str(pk["token"][rows[j]]) for j in js]
    jj = {tk: j for tk, j in zip(toks, js)}
    ix = {n: i for i, n in enumerate(names)}
    got, flags, rawm, secs = [], [], [], []
    status = {}
    for tok, sc in iter_log(db, toks):
        j = jj[tok]
        r = rows[j]
        t = time.time()
        try:
            props = np.concatenate([pk["traj_xy"][r], pk["yaw"][r][..., None]], axis=-1).astype(np.float64)
            pdm = pk["pdm"][r].astype(np.float64)
            res = run_token(sc, props, pdm[:, ix["nc"]], pdm[:, ix["ttc"]])
            got.append(j); flags.append(res["flags"]); rawm.append(res["raw_mismatch"])
            status[tok] = "ok"
        except Exception as e:                                       # noqa: BLE001
            status[tok] = f"error {type(e).__name__}: {str(e)[:200]}"
            print("ERROR", log, tok, traceback.format_exc()[-400:], flush=True)
        secs.append(time.time() - t)
    for tk in toks:
        status.setdefault(tk, "guard_rejected")
    Path(parts_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(parts_dir) / f"{log}.npz", js=np.array(got, dtype=np.int64),
                        flags=np.packbits(np.array(flags, dtype=bool).reshape(len(got), -1), axis=1) if got else np.zeros((0, 1), np.uint8),
                        raw_mismatch=np.array(rawm, dtype=np.int64), n_requested=len(js), secs=np.array(secs),
                        status=np.array(json.dumps(status)))
    return len(got), time.time() - t_log


def load_parts(parts_dir, by_log, n_sets, pos):
    """pos: {row-list index j -> position in the population}; parts store j."""
    S = n_sets
    flags = np.zeros((S, len(ARMS), 2, 64), dtype=bool)
    have = np.zeros(S, bool)
    rawm = np.zeros(S, dtype=np.int64)
    status, missing = {}, []
    for log in sorted(by_log):
        p = Path(parts_dir) / f"{log}.npz"
        if not p.exists():
            missing.append(log)
            continue
        z = np.load(p, allow_pickle=False)
        js = np.array([pos[int(j)] for j in z["js"]], dtype=np.int64)
        if len(js):
            fl = np.unpackbits(z["flags"], axis=1, count=len(ARMS) * 2 * 64).reshape(len(js), len(ARMS), 2, 64).astype(bool)
            flags[js], have[js], rawm[js] = fl, True, z["raw_mismatch"]
        for tk, s in json.loads(str(z["status"])).items():
            key = "error" if s.startswith("error") else s
            status[key] = status.get(key, 0) + 1
    return flags, have, rawm, status, missing


# ----------------------------------------------------------------------------------------------- smoke
def smoke(args):
    RUN, pk, sha, rows = load_pack()
    names = [str(n) for n in pk["pdm_names"]]
    ix = {n: i for i, n in enumerate(names)}
    sel, by_log = population(pk, rows, all_sets=False)
    rng = np.random.default_rng(SMOKE_SEED)
    pick = sorted(rng.choice(len(sel), size=min(args.smoke, len(sel)), replace=False).tolist())
    per, wall0 = [], time.time()
    for k in pick:
        j = sel[k]
        r = rows[j]
        log, tok = str(pk["log"][r]), str(pk["token"][r])
        rec = {"log": log, "token": tok}
        t = time.time()
        try:
            got = next(iter_log(RUN.db_path(log), [tok]), None)
            rec["scenario_build_s"] = round(time.time() - t, 2)
            if got is None:
                rec["error"] = "guard_rejected"
                per.append(rec); continue
            sc = got[1]
            props = np.concatenate([pk["traj_xy"][r], pk["yaw"][r][..., None]], axis=-1).astype(np.float64)
            pdm = pk["pdm"][r].astype(np.float64)
            t = time.time()
            res = run_token(sc, props, pdm[:, ix["nc"]], pdm[:, ix["ttc"]], verify_unhooked=True)
            rec["run_token_s"] = round(time.time() - t, 2)
            rec["timings_s"] = {a: round(b, 3) for a, b in res["timings"].items()}
            fl = res["flags"]
            rec["R_sim_true_flags_vs_truth_mismatch"] = identity_check(fl[0, 0][None], fl[0, 1][None], pdm[:, ix["nc"]][None], pdm[:, ix["ttc"]][None])
            rec["R_sim_true_raw_nc_ttc_mismatch"] = res["raw_mismatch"]
            rec["R_sim_true_raw_max_abs_diff"] = res["raw_max_abs_diff"]
            rec["hooked_vs_unhooked_mismatch"] = res["hooked_vs_unhooked_mismatch"]
            rec["split_scorer_vs_PT_score_plans_mismatch"] = res["split_vs_score_plans_mismatch"]
            rec["n_proposals_with_flag_differing_from_true"] = {a: int((fl[i] != fl[0]).any(axis=0).sum()) for i, a in enumerate(ARMS) if i}
            rec["monotone_none_violations(count,subset)"] = monotone_none(fl[3, 0][None], fl[0, 0][None])
        except Exception as e:                                       # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec["tb"] = traceback.format_exc()[-700:]
        per.append(rec)
        print(json.dumps(rec), flush=True)
    ok = [p for p in per if "run_token_s" in p]
    n_logs, n_sets = len(by_log), len(sel)
    rep = {"smoke": True, "heldout_sets": n_sets, "heldout_logs": n_logs, "tokens": per, "n_ok": len(ok),
           "peak_rss_mb": peak_rss_mb(), "wall_s": round(time.time() - wall0, 1),
           "identity_exact_on_all_smoke_tokens": bool(ok) and all(p["R_sim_true_flags_vs_truth_mismatch"] == 0 and p["R_sim_true_raw_nc_ttc_mismatch"] == 0
                                                                  and p["hooked_vs_unhooked_mismatch"] == 0 for p in ok)}
    if ok:
        warm = ok[-1]["run_token_s"] + ok[-1]["scenario_build_s"]
        mean = float(np.mean([p["run_token_s"] + p["scenario_build_s"] for p in ok]))
        rep["ESTIMATED_per_set_s_warm_last_token"] = round(warm, 2)
        rep["ESTIMATED_per_set_s_mean"] = round(mean, 2)
        rep["ESTIMATED_heldout_cpu_h_range"] = [round(warm * n_sets / 3600, 2), round(mean * n_sets / 3600, 2)]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "smoke_report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "tokens"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------------------- full read
def assemble(args, RUN, pk, sha, rows):
    import s0b_readout as RO
    sel, by_log = population(pk, rows, args.all, args.selftest)
    out = Path(args.out)
    pos = {j: i for i, j in enumerate(sel)}
    flags, have, rawm, status, missing = load_parts(out / "parts", by_log, len(sel), pos)
    if missing:
        print("MISSING LOG PARTS (run is incomplete):", len(missing), missing[:5])
        return 3
    names = [str(n) for n in pk["pdm_names"]]
    ix = {n: i for i, n in enumerate(names)}
    R = np.array([rows[j] for j in sel])
    logs_all = np.array([str(pk["log"][r]) for r in R])
    sel_mod = RO.load_sel_common(os.environ.get("S0B_SEL_DIR", str(S0B / "ref")))
    rule = sel_mod.agg_torch(pk["logits"][R].astype(np.float32), lane=False)
    p6 = np.argmax(rule, axis=-1)
    pdm = pk["pdm"][R].astype(np.float64)
    nc_t, ttc_t, pd_t = pdm[..., ix["nc"]], pdm[..., ix["ttc"]], pdm[..., ix["pdms"]]
    ho = pk["is_heldout"][R].astype(bool)
    result = {"pack_sha256": sha, "population": "all_covered" if args.all else "heldout_covered", "n_requested_sets": len(sel),
              "n_scored_sets": int(have.sum()), "status_counts": status, "columns": {}, "outcome": {}}
    ident_bad = identity_check(flags[have, 0, 0], flags[have, 0, 1], nc_t[have], ttc_t[have])
    raw_bad = int(rawm[have].sum())
    cnt_v, sub_v = monotone_none(flags[have, 3, 0], flags[have, 0, 0])
    result["controls"] = {"R_sim_true_flag_mismatch_proposals": ident_bad, "R_sim_true_raw_nc_ttc_mismatches": raw_bad,
                          "C_SIM_NONE_count_violations_sets": cnt_v, "C_SIM_NONE_subset_violations_sets": sub_v}
    cols = [("heldout_covered", have & ho)] + ([("all_covered", have)] if args.all else [])
    for cname, cm in cols:
        m = np.nonzero(cm)[0]
        if len(m) == 0:
            result["columns"][cname] = {"n_sets": 0}
            result["outcome"][cname] = {"row": "NOT RUN", "reasons": ["no sets"]}
            continue
        lg = logs_all[m]
        O = RO.oracle_gains(rule[m], pd_t[m], nc_t[m], ttc_t[m], p6[m])
        gains = {k: {} for k in ("both", "nc_only", "ttc_only")}
        gfid, nflag = {}, {}
        for ai, arm in enumerate(ARMS):
            g = RO.set_gains(rule[m], pd_t[m], p6[m], flags[m, ai, 0], flags[m, ai, 1])
            for k in gains:
                gains[k][arm] = g[k]
            gfid[arm] = RO.gfid(flags[m, ai, 0] | flags[m, ai, 1], nc_t[m], ttc_t[m], rule[m])
            nflag[arm] = {"nc_flag_count": int(flags[m, ai, 0].sum()), "ttc_flag_count": int(flags[m, ai, 1].sum()),
                          "all_flagged_sets": int(g["all_flagged"].sum())}
        ident_gain_exact = bool(np.array_equal(gains["both"]["R-sim-true"], O["both"]))
        rep = {}
        for k, oname in (("both", "both"), ("nc_only", "nc_only"), ("ttc_only", "ttc_only")):
            rep[k] = RO.column_report(lg, gains[k], O[oname])
        c = rep["both"]
        ident_ok = ident_bad == 0 and raw_bad == 0 and ident_gain_exact
        none_ok = cnt_v == 0 and sub_v == 0
        row, why = outcome_row(ident_ok, none_ok, c["R-sim-t0"]["capture"], c["R-sim-cv"]["capture"], c["R-sim-t0v"]["capture"])
        result["columns"][cname] = {"n_sets": int(len(m)), "n_logs": int(len(set(lg.tolist()))),
                                    "O": {"NC+TTC": float(O["both"].mean()), "NC": float(O["nc_only"].mean()), "TTC": float(O["ttc_only"].mean())},
                                    "arms_both": {a: c[a] for a in ARMS}, "arms_nc_only_vs_O_NC": {a: rep["nc_only"][a] for a in ARMS},
                                    "arms_ttc_only_vs_O_TTC": {a: rep["ttc_only"][a] for a in ARMS}, "O_ci": c["O_NC+TTC"],
                                    "g_fid_vs_truth": gfid, "flag_counts": nflag, "identity_gain_equals_oracle_per_set": ident_gain_exact}
        result["outcome"][cname] = {"row": row, "reasons": why}
    result["DECIDING_COLUMN"] = "heldout_covered"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "s0d_result.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out / "s0d_flags.npz", rows=R, ok=have, logs=logs_all,
                        **{f"{a}__nc": np.packbits(flags[:, i, 0], axis=1) for i, a in enumerate(ARMS)},
                        **{f"{a}__ttc": np.packbits(flags[:, i, 1], axis=1) for i, a in enumerate(ARMS)})
    with open(out / "S0D_RESULT.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(result))
    if args.selftest:
        print("FULLPATH-SELFTEST OK (numbers not printed; delete the out dir)", sorted(result["columns"]), result["controls"])
    else:
        print("DONE", {k: v["row"] for k, v in result["outcome"].items()}, result["controls"])
    return 0


def render_md(res):
    L = ["# S0-D (R-sim) result (MEASURED, dev-box CPU; NAVSIM scorer on modified agent lists; PDMS x 100 vs P6's pick)", "",
         f"pack sha256 {res['pack_sha256'][:16]}..., sets scored {res['n_scored_sets']}/{res['n_requested_sets']}, status {res['status_counts']}",
         f"controls: {res['controls']}", ""]
    for c, col in res["columns"].items():
        if not col.get("n_sets"):
            L += [f"## {c}: no sets", ""]
            continue
        L += [f"## {c} (n={col['n_sets']}, logs={col['n_logs']})  outcome: **{res['outcome'][c]['row']}** {res['outcome'][c]['reasons']}",
              f"oracle O: {col['O']}", "", "| arm | gain [95% CI] | capture vs O_NC+TTC [95% CI] | flags nc/ttc | G-FID all P/R | G-FID top5 P/R |", "|---|---|---|---|---|---|"]
        for a, r in col["arms_both"].items():
            g, f = col["g_fid_vs_truth"][a], col["flag_counts"][a]
            pr = lambda d: f"{d['precision']:.3f}/{d['recall']:.3f}" if d["precision"] is not None and d["recall"] is not None else "n/a"  # noqa: E731
            L.append(f"| {a} | {r['gain']:+.4f} {r['gain_ci']} | {r['capture']} {r['capture_ci']} | {f['nc_flag_count']}/{f['ttc_flag_count']} | {pr(g['all'])} | {pr(g['top5'])} |")
        L.append("")
    return "\n".join(L) + "\n"


def full(args):
    RUN, pk, sha, rows = load_pack()
    names = [str(n) for n in pk["pdm_names"]]
    sel, by_log = population(pk, rows, args.all, args.selftest)
    if not args.assemble_only:
        t_all = time.time()
        parts = Path(args.out) / "parts"
        for li, log in enumerate(sorted(by_log)):
            if (parts / f"{log}.npz").exists():
                continue
            n_ok, sec = process_log_to_part(RUN, pk, rows, names, log, by_log[log], parts)
            print(f"[{li + 1}/{len(by_log)}] {log} sets={len(by_log[log])} ok={n_ok} {sec:.0f}s total={time.time() - t_all:.0f}s peak_rss_mb={peak_rss_mb()}", flush=True)
    return assemble(args, RUN, pk, sha, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="all 3,958 covered sets (default: the 1,343 held-out sets)")
    ap.add_argument("--selftest", action="store_true", help="code-path self-test on ONE log / 6 sets; prints no gain or capture")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(smoke(a) if a.smoke else full(a))


if __name__ == "__main__":
    main()
