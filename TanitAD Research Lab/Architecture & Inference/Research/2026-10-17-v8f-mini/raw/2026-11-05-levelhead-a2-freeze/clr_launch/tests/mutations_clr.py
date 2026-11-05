"""Mutation proof for the arm-CLR package: each mutation deletes ONE safety line in a COPY of the package (never the real one), regenerates the copy's box-2 files, runs the named tests on the copy and requires them RED
(the clean copy must be GREEN first). usage: python -I tests/mutations_clr.py   -> logs/mutation_clr.log. NO network (the tests carry netguard)."""
import os, shutil, subprocess, sys, tempfile, time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TMP = tempfile.mkdtemp(prefix="clr_mut_")
LOG = os.path.join(PKG, "logs", "mutation_clr.log")
MUTS = [  # (label, file, old, new, tests that must go RED)
    ("the CLR_PI_GO refusal of the chain launcher is deleted", "clr_lib.sh", ': "${CLR_PI_GO:?set CLR_PI_GO to the PI go time (HH:MM): this script refuses to touch box 2 without the go of the PI}"\n', "", ["t_nothing_contacts_box2_without_the_pi_go"],
     [("launch_clr.sh", ': "${CLR_PI_GO:?set CLR_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"' + chr(10), "")]),
    ("the CLR_PI_GO refusal of the pull is deleted", "pull_clr.sh", ': "${CLR_PI_GO:?set CLR_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"\n', "", ["t_nothing_contacts_box2_without_the_pi_go"]),
    ("the python-process precondition is deleted", "clr_lib.sh", '  [ "$(echo "$pre" | sed -n 1p)" = "0" ] || die "a python process is running on box 2"\n', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the box-2 re-hash of a shipped stack file is deleted", "clr_lib.sh", '''    [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }\n''', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the fresh-telemetry check is deleted", "clr_lib.sh", '''  [[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"\n''', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the lock is released on a failed pull too", "pull_clr.sh", 'if [ "$bad" = 0 ] && [ -f "$LOCK" ]', 'if [ -f "$LOCK" ]', ["t_shell_static_waker_and_pull_scripts"]),
    ("the detach control is not read", "clr_check.template.py", '"head_grad_at_w0_detach_control", "control_repeat_total_diff"', '"control_repeat_total_diff"', ["t_gate_negatives_each_named"]),
    ("the shipped stack files are not bound at preflight", "clr_check.template.py", "    bad += stack_problems()\n", "", ["t_checker_preflight_pin_and_mutated_argv"]),
    ("the chain guard also ends on a probe marker", "mk_clr_package.py", '["CHAIN_CLR.done", "CHAIN_CLR.STOPPED"]', '["CHAIN_CLR.done", "CHAIN_CLR.STOPPED", "WPROBE.done"]', ["t_guard_scripts_parse_stop_on_heat_and_ignore_stale_markers"]),
    ("the recovery-lever read of the gate is deleted", "clr_check.template.py", '    p += recover_problems(lvs, sc.get("recover_probe") or {}, info, s)\n', "", ["t_gate_negatives_each_named"]),
    ("the seed-0 level-share window is deleted", "clr_check.template.py", "    elif s == 0 and not SHARE_LO <= share <= SHARE_HI:", "    elif False:", ["t_gate_negatives_each_named"]),
    ("the recovery tokens are dropped from the checker's literal argv", "clr_check.template.py", "    toks = [t for kv in base + rec for t in kv]\n", "    toks = [t for kv in base for t in kv]\n", ["t_argv_is_raprp01_plus_exactly_out_and_the_two_level_tokens"]),
    ("the post check that the recovery perturbed samples (KR1) is deleted", "clr_check.template.py", "        if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):", "        if False:", ["t_post_finish_stopped_and_fail_markers"]),
    ("the local check of the frozen card's sha256 is deleted", "clr_local_check.py", '        if sha(card) != pins["card_sha256"]:', "        if False:", ["t_local_check_chain_refusals"]),
    ("the local check against the card's FROZEN_SHA record is deleted", "clr_local_check.py", '        elif sha(card) not in fz.read_text(encoding="utf-8"):', "        elif False:", ["t_local_check_chain_refusals"]),
    ("the launcher ignores a local refusal", "clr_lib.sh", '''[ "${PIPESTATUS[0]}" = 0 ] || { echo "[$(ts)] ZZCLR-REFUSED by the local check (nothing contacted, no lock taken)"; exit 6; }''', "", ["t_nothing_contacts_box2_without_the_pi_go"]),
    ("the pull step no longer waits for CHAIN DONE", "wake_pull_clr.sh", '''  case "$out" in *"CHAIN DONE"*) break ;; esac\n''', "", ["t_wake_pull_chain_on_stand_ins"]),
]


def fresh(name):
    root = os.path.join(TMP, name); shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(PKG, root + "/clr_launch", ignore=shutil.ignore_patterns("__pycache__", "logs"))
    os.makedirs(root + "/clr_launch/logs")
    shutil.copytree(PKG + "/../cl_prereg", root + "/cl_prereg"); shutil.copytree(PKG + "/../a2_prereg", root + "/a2_prereg"); return root


def run(root, tests):
    subprocess.run([PY, "-I", root + "/clr_launch/mk_clr_package.py"], capture_output=True, text=True)
    r = subprocess.run([PY, "-I", root + "/clr_launch/tests/test_clr_launch.py"] + tests, capture_output=True, text=True, env={**os.environ, "CLR_TEST_WORK": TMP + "/work_" + str(time.time_ns())}, timeout=900)
    return r.returncode, r.stdout


def main():
    out = [f"mutation proof {time.strftime('%Y-%m-%d %H:%M:%S')}  package {PKG}"]
    names = sorted({t for m in MUTS for t in m[4]})
    rc, o = run(fresh("clean"), names); out.append(f"CLEAN copy: rc {rc} ({'GREEN' if rc == 0 else 'RED: ' + o[-300:]})")
    bad = rc != 0
    for i, (label, f, old, new, tests, *extra) in enumerate(MUTS):
        root = fresh(f"m{i}"); skipped = False
        for f_, o_, n_ in [(f, old, new)] + (extra[0] if extra else []):
            p = f"{root}/clr_launch/{f_}"; s = open(p, encoding="utf-8", newline="").read()
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
