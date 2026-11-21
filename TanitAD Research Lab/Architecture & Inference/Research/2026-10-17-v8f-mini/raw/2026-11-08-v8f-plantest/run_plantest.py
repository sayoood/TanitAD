"""PREREG_V8F_PLANTEST: the full read. A pool of single-thread workers makes every (model, planner, goal, planner-seed) unit and saves it under runs/ (resumable: a unit whose npz exists is skipped).
Usage: python run_plantest.py <n_workers>. Writes PLANTEST_RUN.done with rc= at the end."""
import os, sys, time, itertools, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

TAGS = ("M2b_s0", "M2b_s1")
RAND_IDX = np.linspace(0, 1975, 256).astype(int)          # beside: 256 windows spread over the deciding set (untrained stacks)
_CTX = {}


def jobs():
    J = []
    for tag in TAGS:                                                              # deciding: both seeds, all three planners, three goals, planner seed 0
        for kind, goal in itertools.product(("td3", "flat", "flat1"), ("true", "rand", "stay")):
            J.append((tag, kind, goal, 0))
    for tag in TAGS:                                                              # beside: planner-seed replicate (seed 1) under G-TRUE
        for kind in ("td3", "flat"):
            J.append((tag, kind, "true", 1))
    for r in (0, 1):                                                              # beside: untrained stacks
        for kind, goal in itertools.product(("td3", "flat"), ("true", "rand", "stay")):
            J.append((f"rand{r}", kind, goal, 0))
    return J


def path(job):
    tag, kind, goal, seed = job
    return f"{HERE}/runs/{tag}_{kind}_{goal}_s{seed}.npz"


def work(job):
    import plantest_real as R
    tag, kind, goal, seed = job
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        if tag not in _CTX:
            _CTX[tag] = R.prepare("M2b_s0", random_r=int(tag[4:])) if tag.startswith("rand") else R.prepare(tag)
            _CTX[tag].tag = tag
        ctx = _CTX[tag]
        t = time.time()
        r = R.unit(ctx, kind, goal, idx=RAND_IDX if tag.startswith("rand") else None, seed=seed)
        os.makedirs(f"{HERE}/runs", exist_ok=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-800:]


def phase(job):
    return "main" if (job[3] == 0 and job[1] in ("td3", "flat") and not job[0].startswith("rand")) else "beside"


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    J = jobs()
    order = {"td3": 0, "flat": 1, "flat1": 2}
    J.sort(key=lambda j: (phase(j) != "main", j[0].startswith("rand"), j[3], order[j[1]]))
    for f in ("PLANTEST_RUN.done", "PLANTEST_MAIN.done"):
        if os.path.exists(f"{HERE}/{f}"):
            os.remove(f"{HERE}/{f}")
    bad = 0
    for ph in ("main", "beside"):
        Jp = [j for j in J if phase(j) == ph]
        badp = 0
        with mp.get_context("spawn").Pool(n, maxtasksperchild=2) as pool:      # recycle workers: bounds the resident memory (the box is RAM-tight)
            for job, dt, st in pool.imap_unordered(work, Jp, chunksize=1):
                print(time.strftime("%H:%M:%S"), job, round(dt, 1), "s", st[:900], flush=True)
                badp += st.startswith("ERR")
        bad += badp
        open(f"{HERE}/PLANTEST_{'MAIN' if ph == 'main' else 'RUN'}.done", "w").write("rc=" + str(1 if badp else 0) + " errors=" + str(badp) + " phase=" + ph)
