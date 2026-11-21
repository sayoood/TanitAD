"""SCORER1 v3 suite runner (detached). Runs every refe/test_*.py (+ 4 eval tests that read launch_gate/scorer_finetune)
as `python test_x.py` with CUDA hidden, per-test timeout 2400 s, N workers, one result line per test.
usage: run_suite_v3.py <tree_root> <out_dir> <workers> [timeout_s]
Result line: `name rc=N secs=S last-line`; ZZ token line preferred as last-line. out_dir/ALLDONE is written at the end."""
import os, subprocess, sys, threading, time, glob, queue

TREE, OUT, NW = sys.argv[1], sys.argv[2], int(sys.argv[3])
TMO = int(sys.argv[4]) if len(sys.argv) > 4 else 2400
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
P = os.path.join(TREE, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-09-20-refe-plan")
REFE, EVAL = os.path.join(P, "refe"), os.path.join(P, "eval")
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
TMP = os.path.join(OUT, "tmp"); os.makedirs(TMP, exist_ok=True)
_t = TMP.replace("\\", "/")
assert os.path.isdir(REFE) and _t.startswith("F:/") and _t.count("/") >= 3, (REFE, _t)

LONG = ["p6_chain", "p7_chain", "hnscreen_chain", "rls1_queue", "p6_levers", "hn_content_fp", "ego_yaw", "rl_safeobj",
        "p2rep_chain", "launch_gate", "hn_screen_wiring", "dump_scores", "r0e_b9b10", "s1c", "session_reserve", "carve_feed"]
names = sorted(os.path.basename(f)[5:-3] for f in glob.glob(os.path.join(REFE, "test_*.py")))
order = [n for n in LONG if n in names] + [n for n in names if n not in LONG]
jobs = [("refe", n, REFE) for n in order]
for n in ("aux_navtest_forward", "hn_screen_read", "r0e_diagnostics", "safeobj_outcome"):
    if os.path.isfile(os.path.join(EVAL, f"test_{n}.py")):
        jobs.append(("eval", n, EVAL))
q = queue.Queue()
for j in jobs:
    q.put(j)
lock = threading.Lock()
RES = os.path.join(OUT, "results.txt")


def work():
    while True:
        try:
            grp, n, cwd = q.get_nowait()
        except queue.Empty:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", SCORER1_TMP=TMP, TEMP=TMP, TMP=TMP, PYTHONIOENCODING="utf-8",
                   PYTHONPATH=";".join([REFE, EVAL]) if grp == "eval" else REFE)
        env.pop("PYTHONHOME", None)
        log = os.path.join(OUT, "logs", f"{grp}.test_{n}.log")
        t0 = time.time()
        with open(log, "wb") as lf:
            p = subprocess.Popen([PY, f"test_{n}.py"], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=TMO)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(); rc = 124
        txt = open(log, "rb").read().decode("utf-8", "replace").splitlines()
        txt = [l for l in txt if l.strip()]
        zz = [l for l in txt if l.lstrip().startswith("ZZ") and ("PASS" in l or "FAIL" in l)]
        last = (zz[-1] if zz else (txt[-1] if txt else ""))[:170]
        with lock, open(RES, "a", encoding="utf-8") as f:
            f.write(f"{grp}/test_{n}.py rc={rc} secs={int(time.time()-t0)} {last}\n")


ts = [threading.Thread(target=work) for _ in range(NW)]
[t.start() for t in ts]; [t.join() for t in ts]
open(os.path.join(OUT, "ALLDONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
