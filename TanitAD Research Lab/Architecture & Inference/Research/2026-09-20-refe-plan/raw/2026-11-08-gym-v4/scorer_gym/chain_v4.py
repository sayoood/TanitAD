"""M-GYM-4 chain (detached). Stages, markers in F:/tanitad_scratch/mgym4/progress.txt:
 TESTS_GREEN (4 parallel test processes, all must print TESTS_DONE fails=0, else TESTS_NOT_GREEN_CHAIN_ABORTED and NOTHING fresh is simulated)
 -> MAIN_DONE (fixes-on run, 4 workers) -> DET_DONE (fresh-process determinism) -> TTCOFF_DONE -> SUMMARY_DONE -> ALL_DONE"""
import os, subprocess, sys, time, json, re
PY = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
V4 = r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v4\scorer_gym"
W = r"F:\tanitad_scratch\mgym4"
PROG = W + r"\progress.txt"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
def mark(s):
    open(PROG, "a").write(f"{time.strftime('%H:%M:%S')} {s}\n")
def run(args, out):
    return subprocess.Popen([PY, "-I"] + args, stdout=open(out, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
os.makedirs(W, exist_ok=True)
# ---- tests: 4 parallel groups
GROUPS = [
    ["test_fix1_oracle_lexicographic_and_regression_arm", "test_fix1_control_identical_when_argmax_arm_is_clean", "test_fix4_all_tie_cell_label_free_and_regression_arm",
     "test_fix6_ladder_known_function_and_regression_arms", "test_v4_state_selection_seeded_disjoint_navtest_free"],
    ["test_fix3_map_prescreen_flags_main10_and_regression_arm", "test_fix3_prescreen_does_not_flag_clean_state"],
    ["test_fix2_ttc_corridor_clears_stall15_false_alarms_and_regression_arm", "test_fix2_corridor_keeps_stall10_and_stall7_collisions_flagged"],
    ["test_control_fixes_off_reproduces_pilot_v2_returns", "test_fix6_ladder_policy_reproduces_pilot_v3_q3_when_fixes_off", "test_v4_deterministic_same_process_fixes_on"],
]
ps = []
for i, grp in enumerate(GROUPS):
    ps.append((run([V4 + r"\tests\test_v4.py"] + grp, W + rf"\tests_v4_g{i}.out"), i))
for p, i in ps:
    p.wait()
ok = all("TESTS_DONE fails=0" in open(W + rf"\tests_v4_g{i}.out", errors="ignore").read() for _, i in ps)
if not ok:
    mark("TESTS_NOT_GREEN_CHAIN_ABORTED"); sys.exit(1)
mark("TESTS_GREEN")
import hashlib
def sha(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()
pin = {"scorer_gym/" + f: sha(os.path.join(V4, f)) for f in ("gym_env.py", "run_v4.py", "summarize_v4.py", "summarize_v2.py", "determinism_child_v4.py", "chain_v4.py")}
pin["scorer_gym/tests/test_v4.py"] = sha(os.path.join(V4, "tests", "test_v4.py"))
for f in ("PREREG_M_GYM_4.md", "PREREG_M_GYM_4_STATES.json"):
    pin[f] = sha(os.path.join(os.path.dirname(V4), f))
json.dump(dict(files=pin, pinned_before="RUN A (fresh states), after tests green", python=PY), open(os.path.join(os.path.dirname(V4), "CODE_PIN.json"), "w"), indent=1)
mark("CODE_PINNED")
# ---- main run
p = run([V4 + r"\run_v4.py", "main", "--workers", "4", "--out", W + r"\run_main"], W + r"\run_main.log"); p.wait()
mark(f"MAIN_DONE rc={p.returncode}")
# ---- determinism: two fresh processes, each repeating in-process
d = [run([V4 + r"\determinism_child_v4.py", "0"], W + rf"\det{i}.txt") for i in (1, 2)]
for q in d: q.wait()
def parse(f):
    return json.loads([l for l in open(f, errors="ignore") if l.startswith("JSON")][0][4:])
try:
    a, b = parse(W + r"\det1.txt"), parse(W + r"\det2.txt")
    same = all(a[m][0] == a[m][1] == b[m][0] == b[m][1] for m in ("log", "idm"))
    mark(f"DET_DONE bit_identical={same}")
except Exception as e:
    mark(f"DET_DONE ERROR {e!r}")
# ---- ttc-off truth run
p = run([V4 + r"\run_v4.py", "ttcoff", "--workers", "4", "--out", W + r"\run_ttcoff"], W + r"\run_ttcoff.log"); p.wait()
mark(f"TTCOFF_DONE rc={p.returncode}")
p = run([V4 + r"\summarize_v4.py", W + r"\run_main", W + r"\run_ttcoff", W + r"\results_v4.json"], W + r"\summarize_v4.out"); p.wait()
mark(f"SUMMARY_DONE rc={p.returncode}")
mark("ALL_DONE")
