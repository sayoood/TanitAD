"""SLOWWM planner read, the run (PREREG_V8F_SLOWWM). The ladder runner (``ladder_plantest/run_lp.py``) unit for unit, with ``lp_slowwm.prepare_arm_read`` in place of ``lp_lib.prepare_arm``: FLAT1 at level 1, forms c1 and
c3, units p1 (PLANTEST form: seed_mode prior, lambda 0, ALL windows, goals true and rand), tune (PLANTEST2 form: goal true, seed_mode hold, TUNE windows, every lambda of plantest2_lib.LAMBDAS) and score (SCORE windows,
goals true and rand, at the lambda each family chose on TUNE). Same window sets, planner settings, lambda grid, choice rule and unit file names (with the arm's name) as the ladder run. Finished units are skipped.
A `meta` unit per arm stores the window arrays the report needs (so the report loads no model).
Usage: python run_lp_slowwm.py <n_workers> "<name>|<run_dir>|<basis_npz>|<ubasis_npz>|<units>" ...       units = all | meta  (an R35H arm: basis and ubasis empty; units meta reads only its windows)
       -> RUNS (env SLOWWM_PLAN_RUNS, default F:/TanitAD-scratch/slowwm/plan_runs)."""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lp_slowwm as S  # noqa: E402
import numpy as np  # noqa: E402

RUNS = os.environ.get("SLOWWM_PLAN_RUNS", "F:/TanitAD-scratch/slowwm/plan_runs")
GOALS = ("true", "rand")
FORMS = ("c1", "c3")
PL = "flat1"
_CTX = {}


def lam_s(l):
    return f"{l:g}"


def path(job):
    if job[0] == "meta":
        return f"{RUNS}/meta_{job[1]}.npz"
    if job[0] == "p1":
        _, arm, form, goal = job[:4]
        return f"{RUNS}/p1_{arm}_{form}_{PL}_{goal}.npz"
    if job[0] == "tune":
        _, arm, form, lam = job[:4]
        return f"{RUNS}/tune_{arm}_{form}_{PL}_L{lam_s(lam)}.npz"
    _, arm, form, goal, lam = job[:5]
    return f"{RUNS}/score_{arm}_{form}_{PL}_{goal}_L{lam_s(lam)}.npz"


def choice_path(arm, form):
    return f"{RUNS}/choice_{arm}_{form}_{PL}.json"


def wait_memory(min_gb=6.0, max_wait_s=3600):
    t0 = time.time()
    while S.LR.free_gb() < min_gb:
        if time.time() - t0 > max_wait_s:
            raise MemoryError("free physical memory stayed below the 6 GB guard")
        time.sleep(30)


def ctx_for(arm, spec):
    if arm not in _CTX:
        _CTX.clear()
        _CTX[arm] = S.prepare_arm_read(arm, spec["run_dir"], spec["basis"] or None, spec["ubasis"] or None)
    return _CTX[arm]


def work(job_spec):
    job, spec = job_spec
    import plantest_lib as P
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        wait_memory()
        arm, form = job[1], job[2] if len(job) > 2 else None
        ctx = ctx_for(arm, spec)
        t = time.time()
        if job[0] == "meta":
            r = S.meta_of(ctx)
        else:
            sel = S.LP.sel_arrays(ctx, form)
            if job[0] == "p1":
                idx = sel["all"]
                r = S.LP.unit3(ctx, PL, job[3], form, idx, 0.0, seed_mode="prior", collect=True)
            elif job[0] == "tune":
                idx = sel["tune"]
                assert set(idx.tolist()) <= set(ctx.tune_idx.tolist()) and set(idx.tolist()).isdisjoint(set(ctx.score_idx.tolist())), "TUNE read touched a SCORE window"
                r = S.LP.unit3(ctx, PL, "true", form, idx, job[3], seed_mode="hold", collect=False)
                m = P.window_metrics(r["plan_raw"], ctx.A[idx], ctx.v0[idx])
                r["acc_mae_mean"], r["yaw_mae_mean"], r["n"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean()), np.array(len(idx))
            else:
                idx = sel["score"]
                r = S.LP.unit3(ctx, PL, job[3], form, idx, job[4], seed_mode="hold", collect=True)
            r["plan_raw"] = r["plan_raw"].astype(np.float32)
            r["basis_kind"], r["basis_sha"] = np.array(ctx.basis_kind), np.array(ctx.basis_sha)
        os.makedirs(RUNS, exist_ok=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-900:]


def jobs_a(arms):
    import plantest2_lib as Q
    J = [(("meta", a["name"]), a) for a in arms]
    for f in FORMS:
        for a in arms:
            if a["units"] != "all":
                continue
            J += [(("p1", a["name"], f, g), a) for g in GOALS]
            J += [(("tune", a["name"], f, l), a) for l in Q.LAMBDAS]
    return J


def choose(arm, form):
    """run_lp.choose's rule: lambda per family from the TUNE units only (plantest2_lib.choose_lambda)."""
    import plantest2_lib as Q
    tab, ch = {}, {}
    for l in Q.LAMBDAS:
        z = np.load(path(("tune", arm, form, l)))
        tab[lam_s(l)] = dict(acc_mae=float(z["acc_mae_mean"]), yaw_mae=float(z["yaw_mae_mean"]), n=int(z["n"]))
    for fam, q in Q.FAMS.items():
        ch[fam] = Q.choose_lambda({l: tab[lam_s(l)][q] for l in Q.LAMBDAS})
    json.dump(dict(choice=ch, tune_table=tab, split_salt=Q.SALT), open(choice_path(arm, form), "w"), indent=1)
    return ch


def jobs_b(arms):
    J = []
    for f in FORMS:
        for a in arms:
            if a["units"] != "all":
                continue
            ch = choose(a["name"], f)
            for lam in sorted(set(ch.values())):
                J += [(("score", a["name"], f, g, lam), a) for g in GOALS]
    return J


def run_stage(pool, J, label):
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        print(time.strftime("%H:%M:%S"), label, job, round(dt, 1), "s", st[:900], flush=True)
        bad += st.startswith("ERR")
    return bad


def parse_arm(s):
    f = s.split("|")
    assert len(f) == 5 and f[4] in ("all", "meta"), "arm: name|run_dir|basis_npz|ubasis_npz|all-or-meta"
    return dict(name=f[0], run_dir=f[1], basis=f[2], ubasis=f[3], units=f[4])


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    arms = [parse_arm(a) for a in sys.argv[2:]]
    assert arms
    status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), arms=[a["name"] for a in arms], n_workers=n)
    with mp.get_context("spawn").Pool(n, maxtasksperchild=6) as pool:
        ea = run_stage(pool, jobs_a(arms), "A")
        eb = run_stage(pool, jobs_b(arms), "B") if ea == 0 else None
    status.update(errors_A=ea, errors_B=eb, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    tag = "_".join(a["name"] for a in arms)
    json.dump(status, open(os.path.join(HERE, f"run_lp_slowwm_{tag}.status.json"), "w"), indent=1)
    sys.exit(0 if (ea == 0 and eb == 0) else 1)
