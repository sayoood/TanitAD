"""PREREG_V8F_PLANTEST2: the full read. Stage A: the TUNE grid (G-TRUE, 6 lambdas). Stage B (after tune_choice.json): SCORE units at the chosen lambdas (G-TRUE, G-RAND, planner-seed replicate).
Usage: python run_plantest2.py <n_workers>. Units are saved under runs/ (resumable). Writes run_plantest2.status.json at the end (the .cmd wrapper writes PLANTEST2.done)."""
import json, os, sys, time, itertools, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

TAGS = ("M2b_s0", "M2b_s1")
KINDS = ("td3", "flat1")
RUNS = f"{HERE}/runs"
_CTX = {}


def lam_s(l):
    return f"{l:g}"


def path(job):
    if job[0] == "tune":
        _, tag, kind, lam = job
        return f"{RUNS}/tune_{tag}_{kind}_L{lam_s(lam)}.npz"
    _, tag, kind, goal, lam, pseed = job
    return f"{RUNS}/score_{tag}_{kind}_{goal}_L{lam_s(lam)}_p{pseed}.npz"


def jobs_a():
    import plantest2_lib as Q
    return [("tune", t, k, l) for t in TAGS for k in KINDS for l in Q.LAMBDAS]


def jobs_b(choice):
    J = []
    for t in TAGS:
        for k in KINDS:
            lams = sorted(set(choice[f"{t}|{k}|{f}"] for f in ("LONGITUDINAL", "LATERAL")))
            for l in lams:
                J += [("score", t, k, "true", l, 0), ("score", t, k, "rand", l, 0), ("score", t, k, "true", l, 1)]
    return J


def work(job):
    import plantest2_real as R2
    import plantest_lib as P
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        tag = job[1]
        if tag not in _CTX:
            _CTX.clear()
            _CTX[tag] = R2.prepare2(tag)
        ctx = _CTX[tag]
        t = time.time()
        if job[0] == "tune":
            r = R2.unit2(ctx, job[2], "true", ctx.tune_idx, job[3], seed=0, collect=False)
            m = P.window_metrics(r["plan_raw"], ctx.A[ctx.tune_idx], ctx.v0[ctx.tune_idx])
            r["acc_mae_mean"], r["yaw_mae_mean"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean())
        else:
            r = R2.unit2(ctx, job[2], job[3], ctx.score_idx, job[4], seed=job[5], collect=True)
        os.makedirs(RUNS, exist_ok=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-800:]


def choose_from_runs():
    import plantest2_lib as Q
    tab, choice = {}, {}
    for t in TAGS:
        for k in KINDS:
            row = {}
            for l in Q.LAMBDAS:
                z = np.load(path(("tune", t, k, l)))
                row[l] = dict(acc_mae=float(z["acc_mae_mean"]), yaw_mae=float(z["yaw_mae_mean"]))
            tab[f"{t}|{k}"] = {lam_s(l): v for l, v in row.items()}
            for fam, q in Q.FAMS.items():
                choice[f"{t}|{k}|{fam}"] = Q.choose_lambda({l: v[q] for l, v in row.items()})
    json.dump(dict(choice=choice, tune_table=tab, split_salt=Q.SALT), open(f"{HERE}/tune_choice.json", "w"), indent=1)
    return choice


def run_stage(pool, J, label):
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        print(time.strftime("%H:%M:%S"), label, job, round(dt, 1), "s", st[:900], flush=True)
        bad += st.startswith("ERR")
    return bad


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    status = dict(stage_a_errors=None, stage_b_errors=None, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with mp.get_context("spawn").Pool(n, maxtasksperchild=4) as pool:
        A = jobs_a()
        status["stage_a_errors"] = run_stage(pool, A, "A")
        if status["stage_a_errors"] == 0:
            choice = choose_from_runs()
            print("CHOICE", json.dumps(choice), flush=True)
            B = jobs_b(choice)
            B.sort(key=lambda j: (j[5], j[1], j[2]))              # deciding units (planner seed 0) first, replicate last
            status["stage_b_errors"] = run_stage(pool, B, "B")
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(status, open(f"{HERE}/run_plantest2.status.json", "w"))
    sys.exit(1 if (status["stage_a_errors"] or status["stage_b_errors"] or status["stage_b_errors"] is None) else 0)
