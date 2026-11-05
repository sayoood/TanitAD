"""Mutation proof for the arm-CL package: each mutation deletes ONE safety line in a COPY of the package (never the real one), regenerates the copy's box-2 files, runs the named tests on the copy and requires them RED
(the clean copy must be GREEN first). usage: python -I tests/mutations_cl.py   -> logs/mutation_cl.log. NO network (the tests carry netguard)."""
import os, shutil, subprocess, sys, tempfile, time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TMP = tempfile.mkdtemp(prefix="cl_mut_")
LOG = os.path.join(PKG, "logs", "mutation_cl.log")
MUTS = [  # (label, file, old, new, tests that must go RED)
    ("the CL_PI_GO refusal of the chain launcher is deleted", "cl_lib.sh", ': "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this script refuses to touch box 2 without the go of the PI}"\n', "", ["t_nothing_contacts_box2_without_the_pi_go"],
     [("launch_cl.sh", ': "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"' + chr(10), "")]),
    ("the CL_PI_GO refusal of the pull is deleted", "pull_cl.sh", ': "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"\n', "", ["t_nothing_contacts_box2_without_the_pi_go"]),
    ("the python-process precondition is deleted", "cl_lib.sh", '  [ "$(echo "$pre" | sed -n 1p)" = "0" ] || die "a python process is running on box 2"\n', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the box-2 re-hash of a shipped stack file is deleted", "cl_lib.sh", '''    [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }\n''', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the fresh-telemetry check is deleted", "cl_lib.sh", '''  [[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"\n''', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the lock is released on a failed pull too", "pull_cl.sh", 'if [ "$bad" = 0 ] && [ -f "$LOCK" ]', 'if [ -f "$LOCK" ]', ["t_shell_static_waker_and_pull_scripts"]),
    ("the detach control is not read", "cl_check.template.py", '"head_grad_at_w0_detach_control", "control_repeat_total_diff"', '"control_repeat_total_diff"', ["t_gate_negatives_each_named"]),
    ("the shipped stack files are not bound at preflight", "cl_check.template.py", "    bad += stack_problems()\n", "", ["t_checker_preflight_pin_and_mutated_argv"]),
    ("the chain guard also ends on a probe marker", "mk_cl_package.py", '["CHAIN_CL.done", "CHAIN_CL.STOPPED"]', '["CHAIN_CL.done", "CHAIN_CL.STOPPED", "WPROBE.done"]', ["t_guard_scripts_parse_stop_on_heat_and_ignore_stale_markers"]),
    ("the checker's two W-placeholder refusals are both deleted", "cl_check.template.py", "return w != PLACEHOLDER and float(w) > 0.0", "return float(w.replace('__W__', '1')) > 0.0", ["t_placeholder_state_refuses_the_chain_and_allows_the_probe"],
     [("cl_check.template.py", "    if PLACEHOLDER in toks:", "    if False:")]),     # two layers of the checker's refusal deleted together (either alone leaves the other: the placeholder is refused more than once on purpose)
]


def fresh(name):
    root = os.path.join(TMP, name); shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(PKG, root + "/cl_launch", ignore=shutil.ignore_patterns("__pycache__", "logs"))
    os.makedirs(root + "/cl_launch/logs"); os.makedirs(root + "/build"); shutil.copy(PKG + "/../build/SHA_CHANGES.txt", root + "/build/")
    shutil.copytree(PKG + "/../cl_prereg", root + "/cl_prereg"); return root


def run(root, tests):
    subprocess.run([PY, "-I", root + "/cl_launch/mk_cl_package.py"], capture_output=True, text=True)
    r = subprocess.run([PY, "-I", root + "/cl_launch/tests/test_cl_launch.py"] + tests, capture_output=True, text=True, env={**os.environ, "CL_TEST_WORK": TMP + "/work_" + str(time.time_ns())}, timeout=900)
    return r.returncode, r.stdout


def main():
    out = [f"mutation proof {time.strftime('%Y-%m-%d %H:%M:%S')}  package {PKG}"]
    names = sorted({t for m in MUTS for t in m[4]})
    rc, o = run(fresh("clean"), names); out.append(f"CLEAN copy: rc {rc} ({'GREEN' if rc == 0 else 'RED: ' + o[-300:]})")
    bad = rc != 0
    for i, (label, f, old, new, tests, *extra) in enumerate(MUTS):
        root = fresh(f"m{i}"); skipped = False
        for f_, o_, n_ in [(f, old, new)] + (extra[0] if extra else []):
            p = f"{root}/cl_launch/{f_}"; s = open(p, encoding="utf-8", newline="").read()
            if s.count(o_) != 1:
                out.append(f"M{i} {label}: SKIPPED, the anchor appears {s.count(o_)} times in {f_}"); bad = skipped = True; break
            open(p, "w", encoding="utf-8", newline="").write(s.replace(o_, n_))
        if skipped:
            continue
        rc, o = run(root, tests); red = rc != 0 and "FAIL" in o
        out.append(f"M{i} {label}: {'RED (caught) ' if red else 'GREEN (NOT caught) '}{[l for l in o.splitlines() if l.startswith('FAIL')][:1]}"); bad = bad or not red
    os.makedirs(os.path.dirname(LOG), exist_ok=True); open(LOG, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
