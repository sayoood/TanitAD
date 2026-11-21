"""R35H's wave (w3) run beside the main run_lp.py process (Master Mind, 2026-11-10 ~22:55, before any R35H SCORE unit exists).

Why: the main run (`job_run_r35h_first.cmd`, started 2026-11-09 19:37) put w3 first, but two of its TUNE units (R35H_s1 c3 lambda 0.03 and
0.1) hit the 6 GB memory guard at 06:37, so w3's SCORE stage did not run and the main process moved on to w1, w2, w4. This runner calls
run_lp's OWN functions (jobs_a, jobs_b, work, run_stage) for w3 only: same code, same env (env_lp.cmd, 1 thread per worker), same
units, same file names. Finished units are skipped by `work`. It writes its own log (run_lp_w3par.log) and status
(run_lp_w3par.status.json) and never touches run_lp.status.json. w3 shares no unit with w4 / w5, so the two processes never write the
same file. Usage: python run_lp_w3par.py <n_workers>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_lp as R  # noqa: E402

if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    st = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), wave="w3", n_workers=n)
    with mp.get_context("spawn").Pool(n, maxtasksperchild=6) as pool:
        ea = R.run_stage(pool, R.jobs_a("w3"), "w3A")
        eb = R.run_stage(pool, R.jobs_b("w3"), "w3B") if ea == 0 else None
    st.update(errors_A=ea, errors_B=eb, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(st, open(os.path.join(HERE, "run_lp_w3par.status.json"), "w"), indent=1)
    sys.exit(0 if (ea == 0 and eb == 0) else 1)
