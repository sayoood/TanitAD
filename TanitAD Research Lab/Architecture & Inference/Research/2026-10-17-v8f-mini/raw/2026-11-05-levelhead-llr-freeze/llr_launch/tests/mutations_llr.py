"""Mutation proof for the arm-LLR package: each mutation deletes ONE safety line in a COPY of the package (never the real one), regenerates the copy's box-2 files, runs the named tests on the copy and requires them RED
(the clean copy must be GREEN first). usage: python -I tests/mutations_llr.py   -> logs/mutation_llr.log. NO network (the tests carry netguard)."""
import os, shutil, subprocess, sys, tempfile, time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TMP = tempfile.mkdtemp(prefix="llr_mut_")
LOG = os.path.join(PKG, "logs", "mutation_llr.log")
GO = ': "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"' + chr(10)
LEADZ = '"head_grad_at_w0_detach_control", "control_repeat_total_diff", "control_repeat_grad_diff", "control_allfalse_term", "control_allfalse_total_diff", "control_allfalse_head_grad_diff", "control_prior_grad_diff"):\n        if lp.get(k) != 0.0:\n            p.append(f"lead_head_probe.{k}'
MUTS = [  # (label, file, old, new, tests that must go RED[, extra [(file, old, new)]])
    ("the LLR_PI_GO refusal of the chain launcher is deleted", "llr_lib.sh", ': "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to touch box 2 without the go of the PI}"\n', "", ["t_nothing_contacts_box2_without_the_pi_go"],
     [("launch_llr.sh", GO, "")]),
    ("the LLR_PI_GO refusal of the probe launcher is deleted", "launch_wleadprobe.sh", GO, "", ["t_nothing_contacts_box2_without_the_pi_go"], [("llr_lib.sh", ': "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to touch box 2 without the go of the PI}"\n', "")]),
    ("the LLR_PI_GO refusal of the pull is deleted", "pull_llr.sh", ': "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"\n', "", ["t_nothing_contacts_box2_without_the_pi_go"]),
    ("the LLR_PI_GO refusal of the probe pull is deleted", "pull_wleadprobe.sh", ': "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"\n', "", ["t_wprobe_waker_pull_and_wake_pull_on_stand_ins"]),
    ("the python-process precondition is deleted", "llr_lib.sh", '  [ "$(echo "$pre" | sed -n 1p)" = "0" ] || die "a python process is running on box 2"\n', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the box-2 re-hash of a shipped stack file is deleted", "llr_lib.sh", '''    [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }\n''', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the fresh-telemetry check is deleted", "llr_lib.sh", '''  [[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"\n''', "", ["t_launchers_dry_run_on_a_fake_box"]),
    ("the launcher ignores a local refusal", "llr_lib.sh", '''[ "${PIPESTATUS[0]}" = 0 ] || { echo "[$(ts)] ZZLLR-REFUSED by the local check (nothing contacted, no lock taken)"; exit 6; }''', "", ["t_nothing_contacts_box2_without_the_pi_go"]),
    ("the lock is released on a failed pull too", "pull_llr.sh", 'if [ "$bad" = 0 ] && [ -f "$LOCK" ]', 'if [ -f "$LOCK" ]', ["t_shell_static_waker_and_pull_scripts"]),
    ("the probe pull releases the lock on a refused reading", "pull_wleadprobe.sh", 'if [ "$rc" = 0 ] && [ -f "$LOCK" ]', 'if [ -f "$LOCK" ]', ["t_wprobe_waker_pull_and_wake_pull_on_stand_ins"]),
    ("the chain pull step no longer waits for CHAIN DONE", "wake_pull_llr.sh", '''  case "$out" in *"CHAIN DONE"*) break ;; esac\n''', "", ["t_wake_pull_chain_on_stand_ins"]),
    ("a CLR folder path survives in the library (the CLR clone's launch failure)", "llr_lib.sh", "H2=D:/tanitad/llr;", "H2=D:/tanitad/clr;", ["t_no_cl_or_clr_folder_paths_in_llr_scripts"]),
    ("the chain guard also ends on a probe marker", "mk_llr_package.py", '["CHAIN_LLR.done", "CHAIN_LLR.STOPPED"], "chain")', '["CHAIN_LLR.done", "CHAIN_LLR.STOPPED", "WLPROBE.done"], "chain")', ["t_guard_scripts_parse_stop_on_heat_and_ignore_stale_markers"]),
    ("the lead probe's detach control is not read", "llr_check.template.py", LEADZ, LEADZ.replace('"head_grad_at_w0_detach_control", ', ""), ["t_gate_negatives_each_named"]),
    ("the lead term's gap row is not read", "llr_check.template.py", '"head_grad_diff_presence_row", "head_grad_diff_gap_row", reach,', '"head_grad_diff_presence_row", reach,', ["t_gate_negatives_each_named"]),
    ("the lead output's reach into O is not read", "llr_check.template.py", '"head_grad_diff_gap_row", reach, "plan_moves_with_token",', '"head_grad_diff_gap_row", "plan_moves_with_token",', ["t_gate_negatives_each_named"]),
    ("the lead head's vision-only proof is not read", "llr_check.template.py", '    if not (isinstance(vo, dict) and vo.get("source_problems") == [] and vo.get("keys_read") == ["frame_ok"]):', "    if False:", ["t_gate_negatives_each_named"]),
    ("the BUILT wiring is not read back", "llr_check.template.py", '    if lvs.get("aux_cond") != wiring:\n        p.append(f"built aux_cond', '    if False:\n        p.append(f"built aux_cond', ["t_gate_negatives_each_named"]),
    ("the fresh key set is not compared with the wiring's", "llr_check.template.py", " and fresh_want is not None and sorted(fresh) == fresh_want):", "):", ["t_gate_negatives_each_named"]),
    ("the lead label coverage (>= 50 % of the clips) is not read", "llr_check.template.py", ' and ll["n_clips_with_labels"] >= LEAD_MIN_CLIP_COVERAGE * ll["n_clips"]):', "):", ["t_gate_negatives_each_named"]),
    ("the lead lever is not read from the BUILT model", "llr_check.template.py", '        p += lead_lever_problems(lvs, info, sc, toks, wdecl_lead, WIRING)\n', "        pass\n", ["t_gate_negatives_each_named"]),
    ("the shipped stack files are not bound at preflight", "llr_check.template.py", "    bad += stack_problems()\n", "", ["t_checker_preflight_pin_and_mutated_argv"]),
    ("the recovery-lever read of the gate is deleted", "llr_check.template.py", '    p += recover_problems(lvs, sc.get("recover_probe") or {}, info, s)\n', "", ["t_gate_negatives_each_named"]),
    ("the seed-0 lead-share window is deleted", "llr_check.template.py", '    elif s == 0 and WIRING == "add" and not SHARE_LO <= lsh <= SHARE_HI:', "    elif False:", ["t_gate_negatives_each_named"]),
    ("the lead-share window also gates the token wiring", "llr_check.template.py", 'elif s == 0 and WIRING == "add" and not', "elif s == 0 and not", ["t_gate_negatives_each_named"]),
    ("the recovery tokens are dropped from the checker's literal argv", "llr_check.template.py", "    toks = [t for kv in base + rec for t in kv]\n", "    toks = [t for kv in base for t in kv]\n", ["t_argv_is_clr_plus_exactly_the_lead_tokens"]),
    ("the wiring token is dropped from the checker's literal argv", "llr_check.template.py", ', "--aux-cond", WIRING]\n', "]\n", ["t_argv_is_clr_plus_exactly_the_lead_tokens"]),
    ("the post check that the recovery perturbed samples (KR1) is deleted", "llr_check.template.py", "        if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):", "        if False:", ["t_post_finish_stopped_and_fail_markers"]),
    ("the post check that the run's label file is the gate's is deleted", "llr_check.template.py", '        if gtok and (sl.get("sha256") != gtok.get("lead_labels_sha256") or not sl.get("sha256")):', "        if False:", ["t_post_finish_stopped_and_fail_markers"]),
    ("a token for another wiring is accepted as cached", "llr_check.template.py", ' and b.get("aux_cond") == WIRING and', " and", ["t_gate_pass_token_binding_and_cache"]),
    ("the local check of the frozen card's / ruling's sha256 is deleted", "llr_local_check.py", '        if sha(f) != pins[fkey]:', "        if False:", ["t_local_check_states_and_refusals"]),
    ("the local check against the FROZEN_SHA record is deleted", "llr_local_check.py", '        elif sha(f) not in fz.read_text(encoding="utf-8"):', "        elif False:", ["t_local_check_states_and_refusals"]),
    ("the local check of CLR's wiring decision is deleted", "llr_local_check.py", "                if mapped is not None and mapped != wiring:", "                if False:", ["t_local_check_states_and_refusals"]),
    ("the local check that the files are landed is deleted", "llr_local_check.py", '    if not pins.get("landed") or not lt.exists() or lt.read_text(encoding="utf-8").strip() != pins["tip_commit"]:', "    if False:", ["t_local_check_states_and_refusals"]),
    ("the local check that W_lead is near the probe's is deleted", "llr_local_check.py", "                if not raw > 0 or abs(float(wl) - raw) > W_TOL * raw:", "                if False:", ["t_local_check_states_and_refusals"]),
    ("the local check that the probe was read under the frozen wiring is deleted", "llr_local_check.py", '                if rec.get("aux_cond") != pins["wiring"]:', "                if False:", ["t_local_check_states_and_refusals"]),
    ("freeze_llr.py lets a ruling contradict the card's mapping", "freeze_llr.py", "        if given is not None and given != mapped:", "        if False:", ["t_freeze_helper_and_probe_reader"]),
    ("the probe reader does not require the frozen wiring", "w_lead_probe_read.py", '    if lp.get("aux_cond") != fz["WIRING"] or (sc.get("levers") or {}).get("aux_cond") != fz["WIRING"]:', "    if False:", ["t_freeze_helper_and_probe_reader"]),
]


def fresh(name):
    root = os.path.join(TMP, name); shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(PKG, root + "/llr_launch", ignore=shutil.ignore_patterns("__pycache__", "logs"))
    os.makedirs(root + "/llr_launch/logs")
    for d in ("cl_prereg", "a2_prereg", "clr_read", "lead_build"):
        src = PKG + "/../" + d
        if os.path.isdir(src):
            shutil.copytree(src, root + "/" + d, ignore=shutil.ignore_patterns("__pycache__", "logs", "code", "tests", "patches", "runner", "mutwork", "pt"), dirs_exist_ok=True)
    shutil.copytree(PKG + "/../clr_launch/box2_files/argv", root + "/clr_launch/box2_files/argv")        # CLR's real generated argv: the base the LLR argv tests compare with
    return root


def run(root, tests):
    subprocess.run([PY, "-I", root + "/llr_launch/mk_llr_package.py"], capture_output=True, text=True)
    r = subprocess.run([PY, "-I", root + "/llr_launch/tests/test_llr_launch.py"] + tests, capture_output=True, text=True, env={**os.environ, "LLR_TEST_WORK": TMP + "/work_" + str(time.time_ns())}, timeout=1500)
    return r.returncode, r.stdout


def main():
    out = [f"mutation proof {time.strftime('%Y-%m-%d %H:%M:%S')}  package {PKG}"]
    names = sorted({t for m in MUTS for t in m[4]})
    rc, o = run(fresh("clean"), names); out.append(f"CLEAN copy: rc {rc} ({'GREEN' if rc == 0 else 'RED: ' + o[-300:]})")
    bad = rc != 0
    for i, (label, f, old, new, tests, *extra) in enumerate(MUTS):
        root = fresh(f"m{i}"); skipped = False
        for f_, o_, n_ in [(f, old, new)] + (extra[0] if extra else []):
            p = f"{root}/llr_launch/{f_}"; s = open(p, encoding="utf-8", newline="").read()
            if s.count(o_) != 1:
                out.append(f"M{i} {label}: SKIPPED, the anchor appears {s.count(o_)} times in {f_}"); bad = skipped = True; break
            open(p, "w", encoding="utf-8", newline="").write(s.replace(o_, n_))
        if skipped:
            continue
        rc, o = run(root, tests); red = rc != 0 and "FAIL" in o and "FileNotFoundError" not in o and "ModuleNotFoundError" not in o      # RED for an infrastructure reason is not a catch
        out.append(f"M{i} {label}: {'RED (caught) ' if red else 'GREEN (NOT caught) '}{[l for l in o.splitlines() if l.startswith('FAIL')][:1]}"); bad = bad or not red
        print(out[-1][:200], flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True); open(LOG, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n"); print("\n".join(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
