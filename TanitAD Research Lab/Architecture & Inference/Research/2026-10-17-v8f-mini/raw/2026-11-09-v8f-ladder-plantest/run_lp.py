"""LADDER PLANTEST / PLANTEST2 full run. Resumable: a unit whose npz exists under RUNS is skipped. Tier T0. CPU, a pool of single-thread workers (<= 2).
Waves (priority order, IMPL_CHOICES section 5); inside a wave: stage A = PLANTEST units + the TUNE grid, then the lambda choice per (arm, form, planner, family) on TUNE only, then stage B = SCORE units.
Usage: python run_lp.py <n_workers> [wave ...]   waves: default order w1 w2 w4 w3 w5. The wrapper job_run.cmd writes LADDER_PLANTEST_RUN.done LAST (after the report)."""
import json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lp_lib  # noqa: F401  (puts the gates' folders and the ladder tree on sys.path; the spawned workers import this module too)
import numpy as np

RUNS = "F:/TanitAD-scratch/ladder_plantest/runs"
GOALS = ("true", "rand")
WAVES = {                                   # name: (arms, form, kinds)
    "w1": (("R35_s0", "R35_s1", "C0_s0", "C0_s1"), "c1", ("flat1",)),
    "w2": (("R35_s0", "R35_s1", "C0_s0", "C0_s1"), "c3", ("flat1",)),
    "w3": (("R35H_s0", "R35H_s1"), "both", ("flat1",)),
    "w4": (("C0_s0", "C0_s1"), "c1", ("td3",)),
    "w5": (("C0_s0", "C0_s1"), "c3", ("td3",)),
}
_CTX = {}


def lam_s(l):
    return f"{l:g}"


def forms_of(f):
    return ("c1", "c3") if f == "both" else (f,)


def path(job):
    kind = job[0]
    if kind == "p1":
        _, arm, form, pl, goal = job
        return f"{RUNS}/p1_{arm}_{form}_{pl}_{goal}.npz"
    if kind == "tune":
        _, arm, form, pl, lam = job
        return f"{RUNS}/tune_{arm}_{form}_{pl}_L{lam_s(lam)}.npz"
    _, arm, form, pl, goal, lam = job
    return f"{RUNS}/score_{arm}_{form}_{pl}_{goal}_L{lam_s(lam)}.npz"


def choice_path(arm, form, pl):
    return f"{RUNS}/choice_{arm}_{form}_{pl}.json"


def jobs_a(wave):
    import plantest2_lib as Q
    arms, form, kinds = WAVES[wave]
    J = []
    for f in [x for x in ("c1", "c3") if x in forms_of(form)]:
        for pl in kinds:
            for arm in arms:
                J += [("p1", arm, f, pl, g) for g in GOALS]
                J += [("tune", arm, f, pl, l) for l in Q.LAMBDAS]
    return J


def choose(arm, form, pl):
    """lambda per family from the TUNE units only (plantest2_lib.choose_lambda: lowest mean MAE, a tie goes to the smaller lambda)."""
    import plantest2_lib as Q
    tab, ch = {}, {}
    for l in Q.LAMBDAS:
        z = np.load(path(("tune", arm, form, pl, l)))
        tab[lam_s(l)] = dict(acc_mae=float(z["acc_mae_mean"]), yaw_mae=float(z["yaw_mae_mean"]), n=int(z["n"]))
    for fam, q in Q.FAMS.items():
        ch[fam] = Q.choose_lambda({l: tab[lam_s(l)][q] for l in Q.LAMBDAS})
    json.dump(dict(choice=ch, tune_table=tab, split_salt=Q.SALT), open(choice_path(arm, form, pl), "w"), indent=1)
    return ch


def jobs_b(wave):
    arms, form, kinds = WAVES[wave]
    J = []
    for f in [x for x in ("c1", "c3") if x in forms_of(form)]:
        for pl in kinds:
            for arm in arms:
                ch = choose(arm, f, pl)
                for lam in sorted(set(ch.values())):
                    J += [("score", arm, f, pl, g, lam) for g in GOALS]
    return J


def wait_memory(min_gb=6.0, max_wait_s=3600):
    import lp_lib as L
    t0 = time.time()
    while L.LR.free_gb() < min_gb:
        if time.time() - t0 > max_wait_s:
            raise MemoryError("free physical memory stayed below the 6 GB guard")
        time.sleep(30)


def work(job):
    import lp_lib as L
    import plantest_lib as P
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        wait_memory()
        arm, form, pl = job[1], job[2], job[3]
        if arm not in _CTX:
            _CTX.clear()
            _CTX[arm] = L.prepare_arm(arm)
        ctx = _CTX[arm]
        if pl == "td3" and not ctx.td3_defined:
            raise RuntimeError("TD3 is not defined on this built model: " + L.td3_reason(ctx.built))
        sel = L.sel_arrays(ctx, form)
        t = time.time()
        if job[0] == "p1":
            idx = sel["all"]
            r = L.unit3(ctx, pl, job[4], form, idx, 0.0, seed_mode="prior", collect=True)
        elif job[0] == "tune":
            idx = sel["tune"]
            assert set(idx.tolist()) <= set(ctx.tune_idx.tolist()) and set(idx.tolist()).isdisjoint(set(ctx.score_idx.tolist())), "TUNE read touched a SCORE window"
            r = L.unit3(ctx, pl, "true", form, idx, job[4], seed_mode="hold", collect=False)
            m = P.window_metrics(r["plan_raw"], ctx.A[idx], ctx.v0[idx])
            r["acc_mae_mean"], r["yaw_mae_mean"], r["n"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean()), np.array(len(idx))
        else:
            idx = sel["score"]
            r = L.unit3(ctx, pl, job[4], form, idx, job[5], seed_mode="hold", collect=True)
        r["plan_raw"] = r["plan_raw"].astype(np.float32)
        os.makedirs(RUNS, exist_ok=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-900:]


def run_stage(pool, J, label):
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        print(time.strftime("%H:%M:%S"), label, job, round(dt, 1), "s", st[:900], flush=True)
        bad += st.startswith("ERR")
    return bad


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    waves = [w for w in sys.argv[2:] if w in WAVES] or ["w1", "w2", "w4", "w3", "w5"]
    status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), waves={})
    errs = 0
    with mp.get_context("spawn").Pool(n, maxtasksperchild=6) as pool:
        for w in waves:
            ea = run_stage(pool, jobs_a(w), f"{w}A")
            eb = None
            if ea == 0:
                eb = run_stage(pool, jobs_b(w), f"{w}B")
            status["waves"][w] = dict(errors_A=ea, errors_B=eb, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            errs += (ea or 0) + (eb or 0) + (1 if eb is None else 0)
            json.dump(status, open(f"{HERE}/run_lp.status.json", "w"), indent=1)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(status, open(f"{HERE}/run_lp.status.json", "w"), indent=1)
    sys.exit(1 if errs else 0)
