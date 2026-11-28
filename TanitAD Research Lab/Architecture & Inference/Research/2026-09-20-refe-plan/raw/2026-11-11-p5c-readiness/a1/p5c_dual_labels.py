"""P5-C DUAL pod label driver: Tc20s (exactly as p5c_labels.py) AND P5-R(lambda*) per bank row, in one pass.  CPU only.  No git writes.

Why: PREREG_P5C 4.1 labels the bank with Tc20s.  If the pod's full-table TB-6 on Tc20s reads rank-0 FAIL, PREREG_P5R 10 makes P5-R(lambda*)
the label instead.  Both tables come out of ONE pod run so that the decision never costs a second labelling pass.

Tc20s fields are the SAME fields p5c_labels.py writes, with the same code path (copied lines, NOT imported: p5c_labels.py is sha256-pinned):
    {"log","token","rank","traj_sha256","tc20s":[20][3],"knot_maxdev","knot_checks"}   or   {...,"error"}
P5-R fields are ADDED, prefixed "p5r_" (the Tc20s warm start is the P5-R warm start, exactly as the P5-R EVAL run: d0 = Tc20s - traj, lam list [lam*]):
    p5r [20][3], p5r_sim8, p5r_E_maxxy, p5r_S1_finite, p5r_S2_yaw_branch, p5r_S3_dev, p5r_S5_lambda_reached,
    p5r_J, p5r_J_track, p5r_J_reg (= lam*||g||^2), p5r_iters, p5r_accepted, p5r_stop, p5r_calls, p5r_props, p5r_t_solve, p5r_t_score
    or p5r_error (row has NO p5r label: it is a raw fallback for the P5-R table).  A Tc20s error implies p5r_error "no tc20s warm start".
Raw fallbacks are therefore counted PER LABEL (done record: counts.err / counts.p5r_err).

modes
  label    python p5c_dual_labels.py label --shard 0 --nshard 16 --out DIR        resumable like p5c_labels.py (done_<shard>.txt)
  repair   python p5c_dual_labels.py repair --rank 1 --n 300 --out FILE.json     sim8 of raw / Tc20s / P5-R on the fixed-rule sample
           (lowest sha256("log|token") among rows of that rank whose log has a local DB: the SAME rule as p5c_labels.py repair).
env (pod): P5C_PKG  P5C_BANKDIR  P5C_DBROOT  NUPLAN_MAPS_ROOT  P5R_LAMBDA_JSON (default <P5R_DIR>/lambda_star.json)  P5R_DIR (default: this directory)
"""
import os, sys, json, time, hashlib, argparse, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")      # the SAME env defaults as p5c_labels.py (its Tc20s bits depend on the BLAS thread setting, see THREADS below); MKL/OPENBLAS are NOT touched here
os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("P5C_PKG", "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
for p in (PKG + "/eval", PKG + "/refe", PKG + "/code"):
    sys.path.insert(0, p)
sys.path.insert(0, os.environ.get("P5R_DIR", HERE))
BANKDIR = os.environ.get("P5C_BANKDIR", "C:/Users/Admin/p5c_scratch/bank")
DBROOT = os.environ.get("P5C_DBROOT", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
LAMBDA_JSON = os.environ.get("P5R_LAMBDA_JSON", os.path.join(os.environ.get("P5R_DIR", HERE), "lambda_star.json"))


def traj_sha(traj):
    """sha256 of the row's own traj as float64 little-endian bytes of the [20,3] array (identical to p5c_labels.traj_sha)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(traj, np.float64)).tobytes()).hexdigest()


def load_rows():
    rows = []
    for rk in (0, 1):
        with open(f"{BANKDIR}/targets_rank{rk}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["log_name"], r["token"], int(r["rank"]), r["traj"]))
    return rows


def lam_star():
    d = json.load(open(LAMBDA_JSON, encoding="utf-8"))
    assert d.get("status") == "OK", d.get("status")
    return float(d["lambda_star"])


try:
    from threadpoolctl import threadpool_limits
except Exception:  # noqa
    threadpool_limits = None
# THREADS (MEASURED 2026-11-12): p5c_labels.py's Tc20s bits depend on the BLAS thread setting (env unset + OMP 2 reproduces the banked dev-box labels
# bit for bit; OPENBLAS/MKL = 1 differs at 1e-14 on 28 of 40 rows), and the P5-R EVAL run (p5r_run.py) ran with OMP/MKL/OPENBLAS = 1 BEFORE numpy was imported.
# So: Tc20s runs in the caller's env, untouched; the P5-R part runs single-threaded.  MM ruling R-C (2026-11-12): threadpoolctl is OPTIONAL (the pod's venv-teacher has none and
# nothing may be installed there).  Mechanism that pins the P5-R part, recorded in the DONE record as `thread_pin`:
#   "threadpoolctl"  threadpoolctl is importable: threadpool_limits(P5R_BLAS_THREADS, default 1, user_api="blas") around every P5-R row
#   "env"            threadpoolctl absent: the labeller REQUIRES OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and OMP_NUM_THREADS all equal to "1" in its env (set before numpy was imported by the
#                    caller; the pod runner exports them) and REFUSES (exit 2, ZZDUAL_REFUSE) otherwise
P5R_BLAS_THREADS = int(os.environ.get("P5R_BLAS_THREADS", "1"))
THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
BLAS_LIMIT_MODE = "threadpoolctl" if threadpool_limits is not None else "env"


def thread_pin(environ=None, have_threadpoolctl=None):
    """-> {"mechanism": "threadpoolctl" | "env", "env": {var: value}}.  Raises RuntimeError (the refusal) when threadpoolctl is absent and any of the three BLAS env vars is not exactly "1"."""
    environ = os.environ if environ is None else environ
    have = (threadpool_limits is not None) if have_threadpoolctl is None else have_threadpoolctl
    seen = {v: environ.get(v) for v in THREAD_ENV_VARS}
    if have:
        return {"mechanism": "threadpoolctl", "env": seen}
    bad = {v: x for v, x in seen.items() if x != "1"}
    if bad:
        raise RuntimeError("threadpoolctl is not importable, so the P5-R part can only be single-threaded through the environment: OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and "
                           f"OMP_NUM_THREADS must all be '1' before python starts; got {seen} (offending: {bad}). Refusing: a different BLAS thread count moves P5-R labels (MEASURED up to 4e-4).")
    return {"mechanism": "env", "env": seen}


def p5r_row(S, R, sanity, ego, polys, traj, tc20, lam):
    """P5-R for one row, warm-started from Tc20s exactly as p5r_run.stage_solve did on EVAL.  -> dict of p5r_* fields (raises on failure)."""
    if threadpool_limits is not None and P5R_BLAS_THREADS > 0:       # R-C: without threadpoolctl the env pin checked by thread_pin() is the mechanism
        with threadpool_limits(limits=P5R_BLAS_THREADS, user_api="blas"):
            return _p5r_row(S, R, sanity, ego, polys, traj, tc20, lam)
    return _p5r_row(S, R, sanity, ego, polys, traj, tc20, lam)


def _p5r_row(S, R, sanity, ego, polys, traj, tc20, lam):
    ND = S.ND
    T = ND.to_navsim(traj)
    cnt = R.Counter(S.real(ego))
    t1 = time.perf_counter()
    sol = R.solve(traj, tc20 - traj, [lam], cnt, ND.to_navsim)
    t_solve = time.perf_counter() - t1
    z = sol[lam]
    sc, tm = S.score(ego, polys, T, [z["label"]])
    san = sanity(z, traj)
    return {"p5r": [[float(v) for v in p] for p in z["label"]], "p5r_sim8": bool(sc[0]["sim8"]), "p5r_E_maxxy": float(sc[0]["E_maxxy"]),
            "p5r_S1_finite": san["S1_finite"], "p5r_S2_yaw_branch": san["S2_yaw_branch"], "p5r_S3_dev": float(S.s3(ego, z["label"], z["s"])),
            "p5r_S5_lambda_reached": san["S5_lambda_reached"], "p5r_J": float(z["J"]), "p5r_J_track": float(z["J_track"]), "p5r_J_reg": float(z["J_reg"]),
            "p5r_iters": int(z["iters"]), "p5r_accepted": int(z["accepted"]), "p5r_stop": str(z["stop"]), "p5r_calls": int(cnt.calls), "p5r_props": int(cnt.props),
            "p5r_t_solve": float(t_solve), "p5r_t_score": float(tm["t_score"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("label", "repair"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    try:
        PIN = thread_pin()
    except RuntimeError as e:
        print("ZZDUAL_REFUSE thread_pin", e, flush=True)
        return 2

    import p5_target_labels as P
    import navsim_dac as ND
    import p5r_run as RUN
    import p5r_labels as R
    RUN.DBROOT = DBROOT                       # Sim() indexes the DBs from this module global
    S = RUN.Sim()
    NS, dbs = S.NS, S.dbs                     # == NS.index_dbs(db_root=DBROOT), the index p5c_labels.py builds
    assert S.ND is ND
    LAM = lam_star()
    t_start = time.time()
    rows = load_rows()

    if a.mode == "label":
        os.makedirs(a.out, exist_ok=True)
        tag = f"s{a.shard}of{a.nshard}"
        by = {}
        for lg, tok, rk, traj in rows:
            by.setdefault(lg, []).append((tok, rk, traj))
        mylogs = [lg for i, lg in enumerate(sorted(by)) if i % a.nshard == a.shard]
        done_fp = f"{a.out}/done_{tag}.txt"
        done = set(open(done_fp, encoding="utf-8").read().split("\n")) if os.path.exists(done_fp) else set()
        outf = open(f"{a.out}/labels_{tag}.jsonl", "a", encoding="utf-8", newline="\n")
        cnt = {"ok": 0, "err": 0, "nodb": 0, "p5r_ok": 0, "p5r_err": 0}
        tsum = {"tc20s": 0.0, "p5r": 0.0}
        print(f"START {tag} logs={len(mylogs)} already_done={len(done)} lam*={LAM} thread_pin={PIN['mechanism']} env={PIN['env']}", flush=True)
        for li, lg in enumerate(mylogs):
            if lg in done:
                continue
            items = by[lg]
            db = dbs.get(lg, "")
            recs = []
            if not db or not os.path.exists(db):
                for tok, rk, traj in items:
                    recs.append({"log": lg, "token": tok, "rank": rk, "traj_sha256": traj_sha(traj), "error": "no local DB", "p5r_error": "no local DB"})
                    cnt["nodb"] += 1
                    cnt["p5r_err"] += 1
            else:
                try:
                    scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(db, sorted({t for t, _, _ in items}))}
                except Exception as e:  # noqa
                    scs = {}
                    print("SCEN-FAIL", lg, repr(e)[:150], flush=True)
                egos = {}
                for tok, rk, traj in items:
                    base = {"log": lg, "token": tok, "rank": rk, "traj_sha256": traj_sha(traj)}
                    t_a = time.perf_counter()
                    # ---- Tc20s: the lines of p5c_labels.py `label`, unchanged
                    try:
                        if tok not in scs:
                            raise KeyError("token not built by build_scenarios_for_log")
                        if tok not in egos:
                            egos[tok] = scs[tok].get_ego_state_at_iteration(0)
                        t20 = np.asarray(traj, np.float64)
                        ctl = {"knot_exact": True, "knot_checks": 0}
                        tc20, _, _, _ = P.iterate20(ND, t20, ND.to_navsim(t20), egos[tok], ctl, None, "cubic")
                        if not ctl["knot_exact"] or not np.isfinite(tc20).all() or tc20.shape != (20, 3):
                            raise ValueError("control/finite/shape check failed")
                        base["tc20s"] = [[float(v) for v in p] for p in tc20]
                        base["knot_maxdev"] = float(ctl.get("knot_maxdev", 0.0)); base["knot_checks"] = int(ctl["knot_checks"])
                        cnt["ok"] += 1
                    except Exception as e:  # noqa
                        base["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                        cnt["err"] += 1
                    tsum["tc20s"] += time.perf_counter() - t_a
                    # ---- P5-R (lam*), warm-started from Tc20s; its own error never touches the Tc20s fields
                    t_b = time.perf_counter()
                    if "tc20s" not in base:
                        base["p5r_error"] = "no tc20s warm start"
                        cnt["p5r_err"] += 1
                    else:
                        try:
                            polys = ND.drivable_polygons(scs[tok].map_api, egos[tok])
                            base.update(p5r_row(S, R, RUN.sanity, egos[tok], polys, t20, tc20, LAM))
                            cnt["p5r_ok"] += 1
                        except Exception as e:  # noqa
                            base["p5r_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                            cnt["p5r_err"] += 1
                    tsum["p5r"] += time.perf_counter() - t_b
                    recs.append(base)
            for r in recs:
                outf.write(json.dumps(r) + "\n")
            outf.flush()
            with open(done_fp, "a", encoding="utf-8", newline="\n") as f:
                f.write(lg + "\n")
            if li % 10 == 0:
                print(f"[{li + 1}/{len(mylogs)}] {cnt} {time.time() - t_start:.0f}s", flush=True)
        outf.close()
        n_rows = sum(len(by[l]) for l in mylogs)
        done_rec = {"shard": a.shard, "nshard": a.nshard, "logs": len(mylogs), "rows_expected": n_rows, "counts": cnt, "lambda_star": LAM, "blas_limit_mode": BLAS_LIMIT_MODE, "thread_pin": PIN["mechanism"], "thread_env": PIN["env"], "p5r_blas_threads": P5R_BLAS_THREADS,
                    "seconds": round(time.time() - t_start, 1), "seconds_tc20s_part": round(tsum["tc20s"], 1), "seconds_p5r_part": round(tsum["p5r"], 1)}
        json.dump(done_rec, open(f"{a.out}/DONE_{tag}.json", "w", newline="\n"), indent=1)
        print("ZZLABEL_DONE", json.dumps(done_rec), flush=True)
        return 0

    # ---- repair: sim8 of raw / Tc20s / P5-R, same sample rule as p5c_labels.py repair
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    pp = get_pacifica_parameters()
    cand = [(hashlib.sha256(f"{lg}|{tok}".encode()).hexdigest(), lg, tok, rk, traj) for lg, tok, rk, traj in rows if rk == a.rank and lg in dbs]
    cand.sort(key=lambda x: x[0])
    sel = cand[: a.n]
    by = {}
    for h, lg, tok, rk, traj in sel:
        by.setdefault(lg, []).append((tok, traj))
    res_rows, errs = [], []
    for li, (lg, items) in enumerate(sorted(by.items())):
        scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(dbs[lg], [t for t, _ in items])}
        for tok, traj in items:
            try:
                sc = scs[tok]; ego = sc.get_ego_state_at_iteration(0)
                polys = ND.drivable_polygons(sc.map_api, ego)
                t20 = np.asarray(traj, np.float64); T = ND.to_navsim(t20)
                ctl = {"knot_exact": True, "knot_checks": 0}
                tc20, _, _, _ = P.iterate20(ND, t20, T, ego, ctl, None, "cubic")
                pr = p5r_row(S, R, RUN.sanity, ego, polys, t20, tc20, LAM)
                Pn = np.stack([T, ND.to_navsim(tc20), ND.to_navsim(np.asarray(pr["p5r"], np.float64))])
                sim = ND.simulated_states(Pn, ego)
                coords = state_array_to_coords_array(sim, pp)
                ci = [i for i in range(coords.shape[2]) if i != int(BBCoordsIndex.CENTER)]
                dac = ND.violations(polys, coords[:, :, ci])
                res_rows.append({"log": lg, "token": tok, "raw_sim8": bool(dac[0]), "tc20s_sim8": bool(dac[1]), "p5r_sim8": bool(dac[2]), "p5r_sim8_scored_alone": pr["p5r_sim8"]})
            except Exception as e:  # noqa
                errs.append({"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:120]}"})
        if li % 20 == 0:
            print(f"[{li + 1}/{len(by)}] rows={len(res_rows)} err={len(errs)} {time.time() - t_start:.0f}s", flush=True)
    n = len(res_rows)
    raw = np.array([r["raw_sim8"] for r in res_rows], int); tc = np.array([r["tc20s_sim8"] for r in res_rows], int); pr_ = np.array([r["p5r_sim8"] for r in res_rows], int)
    res = {"rank": a.rank, "rule": "lowest sha256('log|token') among rows of that rank whose log has a local DB", "n_requested": a.n, "n": n, "n_errors": len(errs), "errors": errs[:5],
           "raw_sim8_fail_rate": float(raw.mean()) if n else None, "tc20s_sim8_fail_rate": float(tc.mean()) if n else None, "p5r_sim8_fail_rate": float(pr_.mean()) if n else None,
           "n_p5r_batch_vs_alone_differs": int(sum(r["p5r_sim8"] != r["p5r_sim8_scored_alone"] for r in res_rows)),
           "note": "the gates recompute r_k with PREREG_P5C 4.3's estimator (seed 20260927) from `rows`; p5r_sim8 plays the role of tc20s_sim8 in --label p5r mode",
           "thread_pin": PIN["mechanism"], "thread_env": PIN["env"], "seconds": round(time.time() - t_start, 1), "rows": res_rows}
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)
    print("ZZREPAIR", json.dumps({k: v for k, v in res.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
