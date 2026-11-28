"""B3 planner run (BESIDE_RULES_SLOWWM.md): the deciding read's units (``../read/run_lp_slowwm.py``: p1 / tune / score, contexts c1 and c3, FLAT1 level 1) with ONLY the level-1 goal cost changed to ||P_S (z_pred - goal)||^2. SLOW arms only.
Same window sets, planner settings, lambda grid, choice rule and unit file names as the deciding run (its ``jobs_a`` / ``jobs_b`` / ``choose`` / ``path`` are used as they are, pointed at this run's own folder). Finished units are skipped;
every unit writes its npz atomically. A ``meta`` unit per arm stores the window arrays the report needs.
Usage: python top64_run.py <n_workers> [--smoke N] "<name>|<run_dir>|<basis_npz>|<ubasis_npz>|<projector_npz>" ...
  --smoke N : the first N windows of every unit (the first N windows of the TUNE and of the SCORE halves), written to <runs>_smoke; used for the wall-time estimate, never for a result.
  runs folder: env SLOWWM_B3_RUNS (default F:/TanitAD-scratch/slowwm/beside_top64_runs)."""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
B3_RUNS = os.environ.get("SLOWWM_B3_RUNS", "F:/TanitAD-scratch/slowwm/beside_top64_runs")
os.environ["SLOWWM_B3_RUNS"] = B3_RUNS
os.environ["SLOWWM_PLAN_RUNS"] = B3_RUNS                      # run_lp_slowwm.RUNS is read from this at import (also in every spawned worker)
import numpy as np  # noqa: E402
import top64_lib as T  # noqa: E402
import run_lp_slowwm as RL  # noqa: E402

S = T.S
assert RL.RUNS == B3_RUNS
_CTX = {}


def _log(*a):
    """Worker progress goes to a file: a spawned worker's stdout does not reach the parent's log on this box."""
    os.makedirs(B3_RUNS, exist_ok=True)
    with open(os.path.join(B3_RUNS, "worker.log"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M:%S") + f" pid {os.getpid()} " + " ".join(str(x) for x in a) + "\n")


def parse_arm(s):
    f = s.split("|")
    assert len(f) == 5, "arm: name|run_dir|basis_npz|ubasis_npz|projector_npz"
    if S.kind_of(f[0]) != "slow":
        raise ValueError(f"B3 runs on SLOW arms only, not {f[0]}")
    return dict(name=f[0], run_dir=f[1], basis=f[2], ubasis=f[3], proj=f[4], units="all")


def ctx_for(arm, spec):
    if arm not in _CTX:
        _CTX.clear()
        ctx = S.prepare_arm_read(arm, spec["run_dir"], spec["basis"], spec["ubasis"])
        proj = T.verify_projector(T.load_projector(spec["proj"]), ctx, spec["run_dir"])
        ctx.b3_proj = proj
        _CTX[arm] = ctx
    return _CTX[arm]


def work(job_spec):
    """RL.work with the restricted cost installed around the planner call."""
    job, spec = job_spec
    import plantest_lib as P
    if os.path.exists(RL.path(job)):
        return job, 0.0, "skip"
    try:
        _log("begin", job)
        RL.wait_memory()
        arm, form = job[1], job[2] if len(job) > 2 else None
        ctx = ctx_for(arm, spec)
        _log("ctx ready", job)
        maxw = int(os.environ.get("SLOWWM_B3_MAXWIN", "0")) or None
        _log("start", job, "maxwin", maxw, "runs", RL.RUNS)
        t = time.time()
        if job[0] == "meta":
            r = S.meta_of(ctx)
        else:
            sel = S.LP.sel_arrays(ctx, form)
            cut = (lambda ix: ix[:maxw]) if maxw else (lambda ix: ix)
            with T.restricted_cost(ctx.st, ctx.b3_proj["P"]):
                if job[0] == "p1":
                    idx = cut(sel["all"])
                    r = S.LP.unit3(ctx, RL.PL, job[3], form, idx, 0.0, seed_mode="prior", collect=True)
                elif job[0] == "tune":
                    idx = cut(sel["tune"])
                    assert set(idx.tolist()) <= set(ctx.tune_idx.tolist()) and set(idx.tolist()).isdisjoint(set(ctx.score_idx.tolist())), "TUNE read touched a SCORE window"
                    r = S.LP.unit3(ctx, RL.PL, "true", form, idx, job[3], seed_mode="hold", collect=False)
                    m = P.window_metrics(r["plan_raw"], ctx.A[idx], ctx.v0[idx])
                    r["acc_mae_mean"], r["yaw_mae_mean"], r["n"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean()), np.array(len(idx))
                else:
                    idx = cut(sel["score"])
                    r = S.LP.unit3(ctx, RL.PL, job[3], form, idx, job[4], seed_mode="hold", collect=True)
            r["plan_raw"] = r["plan_raw"].astype(np.float32)
            r["basis_kind"], r["basis_sha"] = np.array(ctx.basis_kind), np.array(ctx.basis_sha)
            r["b3_rank"], r["b3_projector_sha256"] = np.array(ctx.b3_proj["rank"]), np.array(T.sha256_file(spec["proj"]))
        os.makedirs(B3_RUNS, exist_ok=True)
        tmp = RL.path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, RL.path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-900:]


def run_stage(pool, J, label):
    """RL.run_stage maps ITS OWN ``work`` (the full-width one): this one maps ours."""
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        print(time.strftime("%H:%M:%S"), label, job, round(dt, 1), "s", st[:900], flush=True)
        bad += st.startswith("ERR")
    return bad


def main(argv):
    import multiprocessing as mp
    smoke = None
    if "--smoke" in argv:
        i = argv.index("--smoke")
        smoke = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
        os.environ["SLOWWM_B3_RUNS"] = os.environ["SLOWWM_PLAN_RUNS"] = RL.RUNS = globals()["B3_RUNS"] = B3_RUNS + "_smoke"
        os.environ["SLOWWM_B3_MAXWIN"] = str(smoke)
    n = int(argv[0])
    arms = [parse_arm(a) for a in argv[1:]]
    assert arms
    status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), arms=[a["name"] for a in arms], n_workers=n, runs=B3_RUNS, smoke_windows=smoke, rules_sha256=T.RULES_SHA)
    t0 = time.time()
    with mp.get_context("spawn").Pool(n, maxtasksperchild=6) as pool:
        ea = run_stage(pool, RL.jobs_a(arms), "A")                # RL's jobs carry the arm dict handed in (ours, with ``proj``)
        eb = run_stage(pool, RL.jobs_b(arms), "B") if ea == 0 else None
    status.update(errors_A=ea, errors_B=eb, finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1))
    tag = "_".join(a["name"] for a in arms) + ("_smoke" if smoke else "")
    json.dump(status, open(os.path.join(HERE, f"top64_run_{tag}.status.json"), "w"), indent=1)
    return 0 if (ea == 0 and eb == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
