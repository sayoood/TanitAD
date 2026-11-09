"""PREREG_V8F_PLANTEST3: the full run. Stage 0: decoder gate + KIN-CEILING per training seed. Stage A: FLAT1 TUNE grid (4 arms x 3 lambdas x 2 seeds, G-TRUE). Stage B (after tune_choice.json):
B1 FLAT1 SCORE (G-TRUE, G-RAND at the chosen lambdas), B2 TD3 DEC-END SCORE, B3 planner-seed replicates (G-TRUE, planner seed 1).
Usage: python run_plantest3.py <n_workers>. Every unit is its own file under runs/ and is skipped when present (resumable). RAM guard before each unit. The .cmd wrapper writes PLANTEST3.done last."""
import ctypes, json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

TAGS = ("M2b_s0", "M2b_s1")
ARMS = ("end", "path")
FORMS = ("plain", "shrink")
LAMS = (0.0, 0.1, 1.0)
FAMS = ("LONGITUDINAL", "LATERAL")
RUNS = f"{HERE}/runs"
MIN_FREE_GB = 6.0
_CTX = {}


def lam_s(l):
    return f"{l:g}"


def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def wait_ram(min_gb=None, poll=15, beat=60):
    """Wait while free physical memory < min_gb (the label job must never be starved); heartbeat line every `beat` seconds."""
    min_gb = MIN_FREE_GB if min_gb is None else min_gb
    t0 = last = time.time()
    while True:
        f = free_gb()
        if f >= min_gb:
            return
        if time.time() - last >= beat:
            print(time.strftime("%H:%M:%S"), f"RAM guard: free {f:.1f} GB < {min_gb} GB, waiting {time.time() - t0:.0f} s", flush=True)
            last = time.time()
        time.sleep(poll)


def path(job):
    k = job[0]
    if k == "pre":
        return f"{RUNS}/pre_{job[1]}.json"
    if k == "tune":
        _, tag, arm, form, lam = job
        return f"{RUNS}/tune_{tag}_{arm}_{form}_L{lam_s(lam)}.npz"
    _, tag, kind, arm, form, goal, lam, pseed = job
    return f"{RUNS}/score_{tag}_{kind}_{arm}_{form}_{goal}_L{lam_s(lam)}_p{pseed}.npz"


def jobs_pre():
    return [("pre", t) for t in TAGS]


def jobs_a():
    return [("tune", t, a, f, l) for a in ARMS for f in FORMS for t in TAGS for l in LAMS]


def td3_pick(choice, tune_table, tag, fam):
    """the form FLAT1 DEC-END chose for (tag, family): lower TUNE G-TRUE MAE at its own chosen lambda; tie to plain. Returns (form, lam)."""
    q = "acc_mae" if fam == "LONGITUDINAL" else "yaw_mae"
    best = None
    for f in FORMS:
        lam = choice[f"{tag}|end|{f}|{fam}"]
        m = tune_table[f"{tag}|end|{f}"][lam_s(lam)][q]
        if best is None or m < best[0]:
            best = (m, f, lam)
    return best[1], best[2]


def jobs_b(choice, tune_table):
    B1, B2, B3 = [], [], []
    for a in ARMS:
        for f in FORMS:
            for t in TAGS:
                for l in sorted(set(choice[f"{t}|{a}|{f}|{fam}"] for fam in FAMS)):
                    B1 += [("score", t, "flat1", a, f, "true", l, 0), ("score", t, "flat1", a, f, "rand", l, 0)]
                    B3.append(("score", t, "flat1", a, f, "true", l, 1))
    td3 = set()
    for t in TAGS:
        for fam in FAMS:
            td3.add((t,) + td3_pick(choice, tune_table, t, fam))
    for t, f, l in sorted(td3):
        B2 += [("score", t, "td3", "end", f, "true", l, 0), ("score", t, "td3", "end", f, "rand", l, 0)]
        B3.append(("score", t, "td3", "end", f, "true", l, 1))
    return B1, B2, B3


def work(job):
    import plantest3_real as R3
    import plantest_lib as P
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        wait_ram()
        tag = job[1]
        if tag not in _CTX:
            _CTX.clear()
            _CTX[tag] = R3.prepare3(tag)
        ctx = _CTX[tag]
        t = time.time()
        os.makedirs(RUNS, exist_ok=True)
        if job[0] == "pre":
            rec = R3.gate(ctx)
            dec = R3.get_dec(ctx)
            import torch
            torch.save(dict(W1=dec.W1, b1=dec.b1, W2=dec.W2, b2=dec.b2, ystd=dec.ystd, sha256=dec.digest()), f"{RUNS}/decoder_{tag}.pt")
            for donor in (False, True):
                kp = R3.kin_unit(ctx, ctx.score_idx, donor=donor)
                np.savez(f"{RUNS}/kin_{tag}_{'rand' if donor else 'true'}.npz", plan_raw=kp, idx=np.asarray(ctx.score_idx))
            rec["wall_s"] = time.time() - t
            tmp = path(job) + ".tmp"
            json.dump(rec, open(tmp, "w"), indent=1)
            os.replace(tmp, path(job))
            return job, time.time() - t, "ok"
        if job[0] == "tune":
            _, _, arm, form, lam = job
            r = R3.unit3(ctx, "flat1", "true", ctx.tune_idx, lam, arm, form, seed=0, collect=False)
            m = P.window_metrics(r["plan_raw"], ctx.A[ctx.tune_idx], ctx.v0[ctx.tune_idx])
            r["acc_mae_mean"], r["yaw_mae_mean"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean())
        else:
            _, _, kind, arm, form, goal, lam, pseed = job
            r = R3.unit3(ctx, kind, goal, ctx.score_idx, lam, arm, form, seed=pseed, collect=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-800:]


def choose_from_runs():
    import plantest3_lib as L3
    import plantest2_lib as Q
    tab, choice = {}, {}
    for t in TAGS:
        for a in ARMS:
            for f in FORMS:
                row = {}
                for l in LAMS:
                    z = np.load(path(("tune", t, a, f, l)))
                    row[l] = dict(acc_mae=float(z["acc_mae_mean"]), yaw_mae=float(z["yaw_mae_mean"]))
                tab[f"{t}|{a}|{f}"] = {lam_s(l): v for l, v in row.items()}
                for fam, q in Q.FAMS.items():
                    choice[f"{t}|{a}|{f}|{fam}"] = Q.choose_lambda({l: v[q] for l, v in row.items()})
    td3 = {f"{t}|{fam}": list(td3_pick(choice, tab, t, fam)) for t in TAGS for fam in FAMS}
    json.dump(dict(choice=choice, tune_table=tab, td3_pick=td3, split_salt=Q.SALT, lambdas=list(LAMS)), open(f"{HERE}/tune_choice.json", "w"), indent=1)
    return choice, tab


def run_stage(pool, J, label):
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        print(time.strftime("%H:%M:%S"), label, job, round(dt, 1), "s", st[:900], flush=True)
        bad += st.startswith("ERR")
    return bad


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    status = dict(stage_pre_errors=None, stage_a_errors=None, stage_b_errors=None, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with mp.get_context("spawn").Pool(n, maxtasksperchild=4) as pool:
        status["stage_pre_errors"] = run_stage(pool, jobs_pre(), "PRE")
        status["stage_a_errors"] = run_stage(pool, jobs_a(), "A")
        if status["stage_a_errors"] == 0:
            choice, tab = choose_from_runs()
            print("CHOICE", json.dumps(choice), flush=True)
            B1, B2, B3 = jobs_b(choice, tab)
            status["stage_b_errors"] = 0
            for lab, J in (("B1", B1), ("B2", B2), ("B3", B3)):
                status["stage_b_errors"] += run_stage(pool, J, lab)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(status, open(f"{HERE}/run_plantest3.status.json", "w"))
    sys.exit(1 if (status["stage_pre_errors"] or status["stage_a_errors"] or status["stage_b_errors"] or status["stage_b_errors"] is None) else 0)
