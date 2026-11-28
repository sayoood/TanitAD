"""Mutation audit of the launch gate ITSELF: each mutation disables one refusal in a scratch copy of
`launch_gate.py` / `sup_refcv7.sh`, and `stack/tests/test_launch_gate.py` must go RED -- via the
test that names the refusal. (CLAUDE.md: a guard is certified only by reintroducing its defect.)

    python gate_self_mutation.py --src <dir holding stack/{scripts,tests,ops}> \
        --tanitad <tree whose stack/ provides tanitad> --json <out.json>

Verdicts per mutation: CAUGHT (the named test failed), MISCREDITED (red, but not via the named
test), ESCAPED (green with the defect installed). Every anchor must occur EXACTLY ONCE, or the
audit refuses (a rotted anchor would mutate nothing and report clean).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: (key, file, anchor, replacement, the test(s) that must fail)
MUTATIONS = [
    ("hmac_always_ok", "stack/scripts/launch_gate.py",
     '    return len(got) == 64 and hmac.compare_digest(sign(body, key), got)',
     '    return True',
     ["test_REFUSED_when_the_token_is_hand_edited",
      "test_REFUSED_when_the_argv_AND_the_token_are_edited_consistently"]),
    ("verify_skips_argv", "stack/scripts/launch_gate.py",
     '    if b.get("argv_sha256") != a_sha or b.get("argv") != argv:',
     '    if False:',
     ["test_REFUSED_when_the_argv_changed_after_the_pass"]),
    ("verify_skips_tree", "stack/scripts/launch_gate.py",
     '        if tsha != b.get("tree_sha256"):',
     '        if False:',
     ["test_REFUSED_when_one_code_file_changed"]),
    ("verify_skips_data", "stack/scripts/launch_gate.py",
     '    if d_bad:\n        reasons.append(f"data differs',
     '    if False:\n        reasons.append(f"data differs',
     ["test_REFUSED_when_a_data_input_changed"]),
    ("finalize_ignores_FAIL", "stack/scripts/launch_gate.py",
     '    if any(v["status"] in ("FAIL", "ERROR") for v in checks.values()) or reasons:',
     '    if False:',
     ["test_one_FAIL_makes_a_FAIL_token_and_DEMOTES_the_old_PASS"]),
    ("finalize_missing_is_pass", "stack/scripts/launch_gate.py",
     '    elif verdict_missing:',
     '    elif False:',
     ["test_missing_evidence_is_INCOMPLETE_never_PASS"]),
    ("finalize_accepts_arms", "stack/scripts/launch_gate.py",
     '            if ev.get("arm"):\n                why.append(f"arm {ev[\'arm\']}")',
     '            if False:\n                why.append(f"arm {ev[\'arm\']}")',
     ["test_an_ARM_can_never_produce_a_PASS"]),
    ("live_ignores_absent_terms", "stack/scripts/launch_gate.py",
     '            if present == 0:',
     '            if False:',
     ["test_G_LIVE_the_F3_defect_as_shipped_FAILS_naming_cascade"]),
    ("clock_ignores_violations", "stack/scripts/launch_gate.py",
     '        if not (d <= tol):\n            n_viol += 1',
     '        if False:\n            n_viol += 1',
     ["test_G_CLOCK_the_true_clock_passes_and_the_historical_one_does_not"]),
    ("drivort_allowed", "stack/scripts/launch_gate.py",
     '    for names, why in prof.get("forbidden_levers", ()):',
     '    for names, why in ():',
     ["test_refcv7_REFUSES_every_DrivoR_T_flag_today_and_after_the_rename"]),
    ("needs_pi_is_pass", "stack/scripts/launch_gate.py",
     '        verdict = "PI-DECISION"',
     '        verdict = "PASS"',
     ["test_PI_DECISION_is_its_own_verdict_and_never_verifies"]),
    ("exec_skips_verify", "stack/scripts/launch_gate.py",
     '    _emit_verify(a, ok, reasons, info, "exec")\n    if not ok:\n        return 5',
     '    _emit_verify(a, ok, reasons, info, "exec")\n    if False:\n        return 5',
     ["test_exec_REFUSES_on_its_own_and_never_starts_the_trainer"]),
    ("supervisor_skips_verify", "stack/ops/sup_refcv7.sh",
     '  if [ "$vrc" -ne 0 ] || [ "$verdict" != "MATCH" ]; then',
     '  if false; then',
     ["test_supervisor_REFUSES_when_the_argv_changed_after_the_pass"]),
    ("supervisor_accepts_train_cmd", "stack/ops/sup_refcv7.sh",
     'if [ -n "${TRAIN_CMD:-}" ]; then          # ⛔ G1',
     'if false; then          # ⛔ G1',
     ["test_supervisor_REFUSES_a_free_form_TRAIN_CMD"]),
    # ---- added 2026-09-26 with the coordinator's later rulings ---------------------------------
    ("finalize_skips_profile_rules", "stack/scripts/launch_gate.py",
     '    own_refusals, pi_pending = profile_argv_rules(ctx.prof, ctx.argv, ctx.options, pmap)',
     '    own_refusals, pi_pending = [], []',
     ["test_a_standalone_finalize_cannot_mint_a_token_that_run_would_refuse",
      "test_NEW1_the_ruled_prior_PASSES_the_refused_ones_FAIL_and_the_tau_sha_is_in_the_token"]),
    ("pi_pending_ignored", "stack/scripts/launch_gate.py",
     '    elif needs_pi or pi_pending:',
     '    elif needs_pi:',
     ["test_an_OPEN_value_decision_makes_the_token_PI_DECISION_until_it_is_recorded"]),
    ("refused_prior_modes_allowed", "stack/scripts/launch_gate.py",
     '        if v is not None and (not v or v[0] in bad):',
     '        if False:',
     ["test_NEW1_residual_prior_off_is_not_NEW1_and_only_ha0_ext_pose_is_refcv7"]),
    ("tau_stamp_not_required", "stack/scripts/launch_gate.py",
     '    if not ok:\n        return [f"FIX-4: config.json carries no',
     '    if False:\n        return [f"FIX-4: config.json carries no',
     ["test_SPEC7_the_tau_FILE_stamp_path_sha256_tau_must_be_in_config_json"]),
    ("tau_labels_content_unchecked", "stack/scripts/launch_gate.py",
     '                if got != lab["sha256"]:',
     '                if False:',
     ["test_FIX4_the_BANKED_tau_record_binds_the_split_by_path_AND_label_content"]),
    ("suite_skip_is_a_pass", "stack/scripts/launch_gate.py",
     '    return {t: r for t, r in (res.get("skipped") or {}).items() if any(p.search(r) for p in pats)}',
     '    return {}',
     ["test_G_SUITE_a_skip_for_a_missing_HF_cache_is_NOT_a_pass"]),
    ("ga_reach_ignores_missing_rows", "stack/scripts/launch_gate.py",
     '        if miss:\n            missing_rows.append((r["step"], miss))',
     '        if False:\n            missing_rows.append((r["step"], miss))',
     ["test_G_LIVE_ga_reach_every_logged_row_every_trainable_group"]),
    ("overfit_ignores_the_launch_argv", "stack/scripts/launch_gate.py",
     '    if argv_sha and str(r_argv) != argv_sha:',
     '    if False:',
     ["test_G_MAP_OVERFIT_record_is_a_launch_prerequisite_bound_to_the_launch"]),
    ("overfit_presence_floor_off", "stack/scripts/launch_gate.py",
     '        if not (_finite_num(nc) and nc >= mo["min_cells"]):',
     '        if False:',
     ["test_G_MAP_OVERFIT_every_class_present_at_its_registered_bar_and_the_must_fail_arms"]),
    ("unwire_arm_is_a_noop", "stack/scripts/launch_gate.py",
     '            setattr(dec, attr, None)',
     '            pass',
     ["test_the_unwire_arm_removes_a_BUILT_selection_term_or_reports_that_it_could_not"]),
    # ---- added 2026-09-27 (G-SUITE consume, the import closure, A6-A10) ------------------------
    ("suite_ignores_the_commit", "stack/scripts/launch_gate.py",
     '    if bound.get("commit") != commit:',
     '    if False:',
     ["test_G_SUITE_a_verdict_is_BOUND_to_the_commit_and_the_tree_it_ran_on"]),
    ("suite_ignores_new_failures", "stack/scripts/launch_gate.py",
     '    outside = sorted(fails - pinned - flaky_ok)',
     '    outside = []',
     ["test_G_SUITE_a_NEW_failure_outside_the_pinned_list_FAILS"]),
    ("flaky_admitted_uncited", "stack/scripts/launch_gate.py",
     '        if not rel or not _HEX64.match(sha):',
     '        if False:',
     ["test_G_SUITE_a_FLAKY_failure_is_admitted_ONLY_with_its_record_cited"]),
    ("probe_callable_skips_backward", "stack/scripts/launch_gate.py",
     '        loss.backward()\n        return loss',
     '        return loss',
     ["test_the_grad_unreachable_probe_gets_a_callable_that_RUNS_the_backward"]),
    ("suite_pin_unchecked", "stack/scripts/launch_gate.py",
     '    if got != want:\n        return None, (f"the pinned env-failure list',
     '    if False:\n        return None, (f"the pinned env-failure list',
     ["test_G_SUITE_the_pinned_env_list_is_content_bound_by_the_profile"]),
    ("pinned_skip_is_a_pass", "stack/scripts/launch_gate.py",
     '            if not any(ip.fullmatch(tid) and rp.search(why) for ip, rp in pats):',
     '            if False:',
     ["test_G_SUITE_PINNED_every_test_passes_or_skips_for_a_REGISTERED_reason"]),
    ("import_closure_blind", "stack/scripts/launch_gate.py",
     '        if eager or (top in strict and unguarded):',
     '        if False:',
     ["test_G_HYG_import_closure_eager_missing_FAILS_lazy_and_guarded_are_reported"]),
    ("live_pres_blind", "stack/scripts/launch_gate.py",
     '    bad = {k: v for k, v in pres.items() if not (v["frac"] < float(lp["max_frac"]))}',
     '    bad = {}',
     ["test_G_LIVE_PRES_both_slot_heads_below_half_at_the_end_of_the_smoke"]),
    ("box_overfit_arms_unchecked", "stack/scripts/launch_gate.py",
     '        elif any(res["criteria"][c] for c in crits):',
     '        elif False:',
     ["test_G_BOX_OVERFIT_the_registered_literals_and_the_must_fail_arms"]),
    ("box_overfit_trusts_the_record", "stack/scripts/launch_gate.py",
     '        out = {c: all(meets(k, op, lv) for k, op, lv in terms)',
     '        out = {c: bool(((a.get("criteria") or {}).get(c) or {}).get("pass", True))',
     ["test_G_BOX_OVERFIT_the_registered_literals_and_the_must_fail_arms"]),
    ("a6_levers_unchecked", "stack/scripts/launch_gate.py",
     '        if got != want:\n            why = (prof.get("required_values_why") or {}).get(f, "a recorded ruling")',
     '        if False:\n            why = (prof.get("required_values_why") or {}).get(f, "a recorded ruling")',
     ["test_SPEC_A6_A7_A8_every_NEW2_lever_has_ONE_admissible_value"]),
    # ---- added 2026-09-27 (SPEC_REFCV7 A11: the CODE CLOSURE; open items; the throwaway repo) --
    ("closure_ignores_a_changed_blob", "stack/scripts/launch_gate.py",
     '        elif lb != blob:\n            changed.append(rel)',
     '        elif False:\n            changed.append(rel)',
     ["test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
    ("closure_accepts_an_early_record", "stack/scripts/launch_gate.py",
     '    if c.get("binding") is not True:',
     '    if False:',
     ["test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
    ("closure_accepts_a_crashed_harness", "stack/scripts/launch_gate.py",
     '    if c.get("exit_status") != 0:',
     '    if False:',
     ["test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
    ("closure_blind_to_unrecorded_imports", "stack/scripts/launch_gate.py",
     '        miss = [x for x in need if x not in have]\n        det["static_eager_files"] = len(need)',
     '        miss = []\n        det["static_eager_files"] = len(need)',
     ["test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
    ("closure_blind_to_modules_that_appeared", "stack/scripts/launch_gate.py",
     '    appeared = [n for n in absent if _closure_resolve(tree, n, extra) is not None]',
     '    appeared = []',
     ["test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
    ("closure_env_mismatch_passes", "stack/scripts/launch_gate.py",
     '        diff = {k: (env.get(k), host_env.get(k)) for k in _CLOSURE_ENV_KEYS\n'
     '                if env.get(k) != host_env.get(k)}',
     '        diff = {}',
     ["test_closure_a_torch_timm_CUDA_cuDNN_mismatch_FAILS"]),
    ("map_overfit_job_skips_the_closure", "stack/scripts/launch_gate.py",
     '                           must_have_data_sha=[cw_sha] if cw_sha else [],\n'
     '                           git_dir=ctx.options.get("git_dir"), commit=ctx.commit)\n'
     '    reasons += cr',
     '                           must_have_data_sha=[cw_sha] if cw_sha else [],\n'
     '                           git_dir=ctx.options.get("git_dir"), commit=ctx.commit)\n'
     '    pass',
     ["test_G_MAP_OVERFIT_job_needs_its_closure_and_its_PASS_states_the_isolation_scope"]),
    ("open_items_ignored", "stack/scripts/launch_gate.py",
     '    open_items = open_item_reasons(ctx.prof)',
     '    open_items = []',
     ["test_the_profile_OPEN_ITEMS_block_a_PASS_until_each_is_closed"]),
    ("throwaway_repo_inherits_GIT_DIR", "stack/scripts/launch_gate.py",
     '    `commit`, index = HEAD. Idempotent; its HEAD is asserted by the caller."""\n'
     '    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV}',
     '    `commit`, index = HEAD. Idempotent; its HEAD is asserted by the caller."""\n'
     '    env = dict(os.environ)',
     ["test_G_SUITE_PINNED_git_reads_go_to_a_THROWAWAY_repo_never_the_shared_one"]),
    ("pinned_tests_inherit_GIT_DIR_and_MKL", "stack/scripts/launch_gate.py",
     '    env = {k: v for k, v in base.items() if k not in _GIT_ENV and k != "MKL_NUM_THREADS"}',
     '    env = dict(base)',
     ["test_G_SUITE_PINNED_git_reads_go_to_a_THROWAWAY_repo_never_the_shared_one"]),
    ("supervisor_inherits_the_callers_env", "stack/ops/sup_refcv7.sh",
     'unset CODE PYTHON GATE_TOKEN GATE_ARGV_FILE GATE_KEY_FILE TRAIN_CMD STEPS OUT_DIR',
     ': unset-removed',
     ["test_supervisor_takes_its_variables_from_the_MANIFEST_never_the_callers_environment"]),
    ("eval_list_builder_blind", "stack/scripts/launch_gate.py",
     '        miss = sorted(want - have)\n        det = {"source": "_eval_row_from_acc on the BUILT model",',
     '        miss = []\n        det = {"source": "_eval_row_from_acc on the BUILT model",',
     ["test_G_MAP_eval_keys_from_the_trainers_OWN_row_builder_on_the_BUILT_model"]),
    ("spelling_uses_the_2_constant", "stack/scripts/launch_gate.py",
     '    bands = tuple(fn(int(h["logits_hw"][0]))) if fn is not None else getattr(M, "BAND_KEYS", None)',
     '    bands = getattr(M, "BAND_KEYS", None)',
     ["test_G_MAP_spelling_is_checked_at_the_DECLARED_extent_not_the_2_constant"]),
    ("skip_registry_ignores_the_test_id", "stack/scripts/launch_gate.py",
     '            if not any(ip.fullmatch(tid) and rp.search(why) for ip, rp in pats):',
     '            if not any(rp.search(why) for ip, rp in pats):',
     ["test_G_SUITE_PINNED_a_registered_skip_admits_ITS_test_only"]),
    ("pinned_env_skips_the_full_secret_scan", "stack/scripts/launch_gate.py",
     '               SECRET_SCAN_FULL="1")',
     '               )',
     ["test_G_SUITE_PINNED_git_reads_go_to_a_THROWAWAY_repo_never_the_shared_one"]),
    ("live_pres_counts_aux_layers", "stack/scripts/launch_gate.py",
     '            (aux if re.search(r"\\.aux(\\[|\\.|$)", pth) else heads)[pth] = d',
     '            heads[pth] = d',
     ["test_G_LIVE_PRES_deep_supervision_aux_layers_never_count_as_a_head"]),
    ("abort_trusts_the_kill", "stack/scripts/launch_gate.py",
     '            try:\n                p.kill()\n            except Exception as e:',
     '            try:\n                pass\n            except Exception as e:',
     ["test_an_ABORT_is_VERIFIED_the_child_tree_is_gone_or_the_evidence_says_it_is_not"]),
    ("refcv7_skip_registry_emptied", "stack/scripts/launch_gate.py",
     '        suite_pinned_skip_ok=(',
     '        suite_pinned_skip_ok=(), _struck=(',
     ["test_G_SUITE_PINNED_the_refcv7_registry_admits_exactly_the_five_measured_skips"]),
    ("closure_run_from_a_hand_list", "stack/scripts/closure_run.py",
     '        for name, mod in list(sys.modules.items()):',
     '        for name, mod in []:',
     ["test_closure_run_reads_the_modules_from_SYS_MODULES_not_from_a_list"]),
    ("closure_run_ignores_code_it_opened", "stack/scripts/closure_run.py",
     '        if rp.suffix == ".py" and _under(rp, self.tree):',
     '        if False:',
     ["test_closure_run_records_the_imported_tree_modules_FROM_THE_PROCESS"]),
    ("closure_run_blind_to_lookups", "stack/scripts/closure_run.py",
     '        if self.on:\n            self.names.add(fullname)',
     '        if False:\n            self.names.add(fullname)',
     ["test_closure_run_records_the_imported_tree_modules_FROM_THE_PROCESS",
      "test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
    ("closure_run_records_a_crash_as_success", "stack/scripts/closure_run.py",
     '        rec.status, rec.error = 1, f"{type(e).__name__}: {e}"[:600]\n        rec.write(1, rec.error)',
     '        rec.status, rec.error = 0, None\n        rec.write(0, None)',
     ["test_closure_run_a_CRASHED_harness_still_writes_its_record_with_the_exit_status"]),
    ("closure_run_ignores_data_roots", "stack/scripts/closure_run.py",
     '        for root in self.roots:\n            if _under(rp, root):',
     '        for root in []:\n            if _under(rp, root):',
     ["test_closure_run_records_the_imported_tree_modules_FROM_THE_PROCESS"]),
    ("closure_run_always_binding", "stack/scripts/closure_run.py",
     '        self.binding = bool(a.binding)',
     '        self.binding = True',
     ["test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11"]),
]


def run_tests(root: Path, tanitad_tree: Path) -> tuple[int, set[str], str]:
    env = dict(os.environ)
    # the tree under test FIRST; anything the caller put on PYTHONPATH (Thor: the --no-deps
    # pytest side folder) after it
    extra = [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    env["PYTHONPATH"] = os.pathsep.join([str(tanitad_tree / "stack"),
                                         str(tanitad_tree / "taniteval"), *extra])
    env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
               OMP_NUM_THREADS="4")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
                        "tests/test_launch_gate.py"], cwd=str(root / "stack"), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    failed = {ln.split("::", 1)[1].split(" ")[0].split("[")[0]
              for ln in p.stdout.splitlines() if ln.startswith("FAILED ") and "::" in ln}
    tail = ([ln for ln in p.stdout.splitlines() if ln.strip()] or ["<none>"])[-1]
    return p.returncode, failed, tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tanitad", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    src, tt = Path(a.src), Path(a.tanitad)
    work = Path(tempfile.mkdtemp(prefix="gsm_"))
    files = ("stack/scripts/launch_gate.py", "stack/scripts/closure_run.py",
             "stack/tests/test_launch_gate.py", "stack/ops/sup_refcv7.sh",
             "stack/ops/launch_gate_thor_env_failures.json")
    for f in files:
        (work / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f, work / f)
    for key, f, anchor, _rep, _t in MUTATIONS:
        n = (src / f).read_text(encoding="utf-8").count(anchor)
        if n != 1:
            raise SystemExit(f"REGISTRY ROT: anchor of {key} occurs {n} times in {f}")
    rc0, failed0, tail0 = run_tests(work, tt)
    out = {"baseline": {"rc": rc0, "failed": sorted(failed0), "tail": tail0}, "mutations": []}
    if rc0 != 0:
        out["verdict"] = "UNREADABLE: the suite is red without any mutation"
    else:
        for key, f, anchor, rep, tests in MUTATIONS:
            orig = (src / f).read_text(encoding="utf-8")
            (work / f).write_text(orig.replace(anchor, rep, 1), encoding="utf-8", newline="\n")
            rc, failed, tail = run_tests(work, tt)
            (work / f).write_text(orig, encoding="utf-8", newline="\n")
            named = sorted(set(tests) & failed)
            verdict = "CAUGHT" if named else ("MISCREDITED" if rc != 0 else "ESCAPED")
            out["mutations"].append({"key": key, "file": f, "verdict": verdict,
                                     "named_tests_failed": named, "all_failed": sorted(failed),
                                     "tail": tail})
            print(f"{verdict:12s} {key}  {named or sorted(failed)[:3]}", flush=True)
        c = {}
        for m in out["mutations"]:
            c[m["verdict"]] = c.get(m["verdict"], 0) + 1
        out["counts"] = c
        out["verdict"] = ("ALL CAUGHT" if c.get("CAUGHT", 0) == len(MUTATIONS) else
                          "NOT ALL CAUGHT")
    out["scratch"] = str(work)   # left in place (5 small files): no recursive delete of a path
    out["n_mutations"] = len(MUTATIONS)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(out.get("counts"), out["verdict"], "->", a.json)
    return 0 if out["verdict"] == "ALL CAUGHT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
