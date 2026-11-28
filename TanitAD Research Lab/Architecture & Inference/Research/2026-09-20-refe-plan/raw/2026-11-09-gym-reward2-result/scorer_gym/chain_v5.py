"""REWARD-1 chain (detached). Markers in F:/tanitad_scratch/mgym5/progress.txt; the LAST line is ALL_DONE or an ABORT line.
 TESTS_GREEN -> CODE_PINNED -> GREPRO_PASS (used states only; fail => ABORT, NO DRAW) -> DRAWN (seed 20261110) -> STATES_PINNED
 -> MAIN_DONE (20 full-arm states, FLOOR 0.20) -> LADDER_B0_DONE (states 20..159) -> [LADDER_BLOCK lo:hi ... while < 5 eligible and < 240]
 -> DET_DONE -> TTCOFF_DONE -> SUMMARY_DONE -> ALL_DONE.
CPU only, <= 4 processes at a time. A red test, failed G-REPRO, or any non-zero stage => `ABORT <reason>` and exit (nothing fresh simulated before the tests were green)."""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
PY = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
V5 = r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v5\scorer_gym"
P5 = os.path.dirname(V5)
W = r"F:\tanitad_scratch\mgym5"
PROG = W + r"\progress.txt"
FLOOR = "0.20"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")


def mark(s):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")


def run(args, out):
    return subprocess.Popen([PY, "-I"] + args, stdout=open(out, "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)


def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


def abort(why):
    mark("ABORT " + why)
    sys.exit(1)


def stage(args, out, name):
    p = run(args, out)
    p.wait()
    if p.returncode != 0:
        abort(f"{name} rc={p.returncode} (see {out})")
    mark(f"{name} rc=0")


def main():
    os.makedirs(W, exist_ok=True)
    open(W + r"\chain.pid", "w", encoding="utf-8").write(str(os.getpid()))
    mark(f"CHAIN_START pid={os.getpid()}")
    # ---------------- tests: at most 4 processes at a time
    T4, T5 = V5 + r"\tests\test_v4.py", V5 + r"\tests\test_v5.py"
    GROUPS = [
        (T4, ["test_fix1_oracle_lexicographic_and_regression_arm", "test_fix1_control_identical_when_argmax_arm_is_clean", "test_fix4_all_tie_cell_label_free_and_regression_arm",
              "test_fix6_ladder_known_function_and_regression_arms", "test_v4_state_selection_seeded_disjoint_navtest_free"]),
        (T4, ["test_fix3_map_prescreen_flags_main10_and_regression_arm", "test_fix3_prescreen_does_not_flag_clean_state"]),
        (T4, ["test_fix2_ttc_corridor_clears_stall15_false_alarms_and_regression_arm", "test_fix2_corridor_keeps_stall10_and_stall7_collisions_flagged"]),
        (T4, ["test_control_fixes_off_reproduces_pilot_v2_returns", "test_fix6_ladder_policy_reproduces_pilot_v3_q3_when_fixes_off", "test_v4_deterministic_same_process_fixes_on"]),
        (T5, ["test_card_constants", "test_qfree_floor_value_and_regression_arm", "test_eligibility_rule_and_regression_arms", "test_b3_verdict_and_regression_arms",
              "test_block_extension_and_regression_arms"]),
        (T5, ["test_floor_reachable_from_flag_to_crawl_return_and_regression_arm"]),
        (T5, ["test_selection_v5_structure_with_dummy_seed"]),
    ]
    pending, running, outs = list(enumerate(GROUPS)), [], {}
    while pending or running:
        while pending and len(running) < 4:
            i, (f, names) = pending.pop(0)
            outs[i] = W + rf"\tests_v5_g{i}.out"
            running.append((run([f] + names, outs[i]), i))
        time.sleep(2)
        running = [(p, i) for p, i in running if p.poll() is None]
    ok = all("TESTS_DONE fails=0" in open(outs[i], errors="ignore", encoding="utf-8").read() for i in outs)
    if not ok:
        abort("TESTS_NOT_GREEN (nothing fresh simulated; fix the code and relaunch)")
    mark("TESTS_GREEN")
    pin = {"scorer_gym/" + f: sha(os.path.join(V5, f)) for f in sorted(os.listdir(V5)) if f.endswith(".py")}
    for f in sorted(os.listdir(V5 + r"\tests")):
        if f.endswith(".py"):
            pin["scorer_gym/tests/" + f] = sha(V5 + "\\tests\\" + f)
    for f in ("PREREG_REWARD_1.md",):
        pin[f] = sha(os.path.join(P5, f))
    json.dump(dict(files=pin, pinned_before="G-REPRO, the draw and every fresh rollout, after tests green", python=PY), open(os.path.join(P5, "CODE_PIN.json"), "w", encoding="utf-8"), indent=1)
    mark("CODE_PINNED")
    # ---------------- G-REPRO on used states only: BEFORE any fresh state is drawn
    p = run([V5 + r"\grepro_v5.py", W + r"\grepro_v5.json"], W + r"\grepro_v5.out"); p.wait()
    os.makedirs(P5 + r"\raw", exist_ok=True)
    for f in ("grepro_v5.json", "grepro_v5.out"):
        if os.path.exists(W + "\\" + f):
            shutil.copy(W + "\\" + f, P5 + "\\raw\\" + f)
    if p.returncode != 0:
        abort(f"G_REPRO_FAIL rc={p.returncode} (NO DRAW; see raw/grepro_v5.out)")
    mark("GREPRO_PASS")
    # ---------------- the draw (seed 20261110), once
    states = P5 + r"\PREREG_REWARD_1_STATES.json"
    if os.path.exists(states):
        mark("DRAW_REUSED (list already drawn and pinned; never redrawn)")
    else:
        stage([V5 + r"\draw_v5.py"], W + r"\draw_v5.out", "DRAWN")
    pin = json.load(open(os.path.join(P5, "CODE_PIN.json"), encoding="utf-8"))
    pin["files"]["PREREG_REWARD_1_STATES.json"] = sha(states)
    json.dump(pin, open(os.path.join(P5, "CODE_PIN.json"), "w", encoding="utf-8"), indent=1)
    mark("STATES_PINNED sha=" + sha(states))
    # ---------------- fresh rollouts
    stage([V5 + r"\run_v5.py", "main", "--floor", FLOOR, "--workers", "4", "--out", W + r"\run_main"], W + r"\run_main.log", "MAIN_DONE")
    lad = []
    lo, hi = 20, 160
    while True:
        d = W + rf"\ladder_{lo}_{hi}"
        stage([V5 + r"\run_v5.py", "ladder", "--floor", FLOOR, "--workers", "4", "--range", f"{lo}:{hi}", "--out", d], d + ".log", f"LADDER_{lo}_{hi}_DONE")
        lad.append(d)
        q = W + r"\next_block.out"
        stage([V5 + r"\next_block_v5.py", W + r"\run_main"] + lad, q, "NEXT_BLOCK_CHECK")
        info = json.loads([l for l in open(q, encoding="utf-8") if l.startswith("JSON")][0][4:])
        mark(f"B3_SO_FAR {json.dumps(info)}")
        if info["next_block"] is None:
            break
        lo, hi = info["next_block"]
    # ---------------- determinism (two fresh processes), ttc-off truth run, summary
    d = [run([V5 + r"\determinism_child_v5.py", "0", FLOOR], W + rf"\det{i}.txt") for i in (1, 2)]
    for q_ in d:
        q_.wait()
    mark("DET_DONE")
    stage([V5 + r"\run_v5.py", "ttcoff", "--floor", FLOOR, "--workers", "4", "--out", W + r"\run_ttcoff"], W + r"\run_ttcoff.log", "TTCOFF_DONE")
    stage([V5 + r"\summarize_v5.py", W + r"\run_main", W + r"\run_ttcoff", W + r"\results_v5.json", W + r"\det1.txt", W + r"\det2.txt"] + lad, W + r"\summarize_v5.out", "SUMMARY_DONE")
    shutil.copy(W + r"\results_v5.json", P5 + r"\raw\results_v5.json")
    mark("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        mark("ABORT exception " + traceback.format_exc().replace("\n", " | ")[-600:])
        sys.exit(2)
