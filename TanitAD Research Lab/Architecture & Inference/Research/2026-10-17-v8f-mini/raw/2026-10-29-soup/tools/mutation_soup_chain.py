#!/usr/bin/env python3
"""mutation_soup_chain.py -- the mutation pass over the runner's key guards: soup_chain.sh and soup_chain_lib.py (zero training, CPU only, a throw-away world per test, one pytest at a time).

Each mutant is the runner (soup_chain.sh, soup_chain_lib.py, wake_soup.sh, soup_chain.cmd copied into its own directory) with ONE textual change (the anchor must occur exactly once in its file).  The self-test
(tests/test_soup_chain.py, SOUP_CHAIN_DIR=<mutant dir>) is run in two stages:
    1. TARGETED: only the tests designed to kill that mutant (-k), with -x;
    2. FULL: if it survived the targeted stage, the whole file minus the three read-only production-fact tests, with -x.
KILLED iff some test fails.  A survivor of both stages is a finding (a test to add) or a NAMED EQUIVALENT mutant (EQUIVALENT below, each with its reason).  The unmutated runner must pass the full file first.

    python mutation_soup_chain.py --check-anchors
    python mutation_soup_chain.py --work <scratch dir> --out ../evidence/mutation_chain.json [--only name,name] [--skip-baseline]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
W = os.path.dirname(HERE)
TEST = f"{W}/tests/test_soup_chain.py"
FILES = ["soup_chain.sh", "soup_chain_lib.py", "wake_soup.sh", "soup_chain.cmd"]
FULL_DESELECT = ["shipped_plan", "shipped_tool_digest", "real_tree_hashes"]
TESTS_SHA = {f: hashlib.sha256(open(f"{W}/tests/{f}", "rb").read()).hexdigest() for f in ("test_soup_chain.py", "soup_chain_stub_tool.py", "soup_chain_stub_reader.py")}

# (name, file, old, new, targeted -k expression)
SH, LB = "soup_chain.sh", "soup_chain_lib.py"
M = [
    # ---- the PREREG: both from the environment, and the file is the pinned one
    ("prereg_needs_only_one", SH, 'if [ -z "$PREREG" ] || [ -z "$PREREG_SHA" ]; then PRE_WHY=', 'if [ -z "$PREREG" ] && [ -z "$PREREG_SHA" ]; then PRE_WHY=', "prereg_and_its_sha or unset_prereg"),
    ("prereg_sha_format_unchecked", SH, 'elif ! [[ "$PREREG_SHA" =~ ^[0-9a-f]{64}$ ]]; then', "elif false; then", "prereg_and_its_sha"),
    ("prereg_file_unchecked", SH, 'elif [ ! -f "$(u "$PREREG")" ]; then', "elif false; then", "prereg_and_its_sha"),
    ("prereg_hash_unchecked", SH, 'elif [ "$(fsha "$PREREG")" != "$PREREG_SHA" ]; then', "elif false; then", "prereg_and_its_sha"),
    ("prereg_real_run_continues", SH, '|| { [ "$MODE" = pins ] || stop "$PRE_WHY"; }', '|| { [ "$MODE" = pins ] || say "$PRE_WHY"; }', "prereg_and_its_sha or unset_prereg"),
    ("prereg_changed_before_reader", SH, '[ -z "$PREREG" ] || [ "$(fsha "$PREREG")" = "$PREREG_SHA" ] || stop "the PREREG changed ($1)"', "true", "prereg_that_changes"),
    # ---- the dry run
    ("dry_not_ready_exits_zero", SH, 'echo "ZZSOUP-DRY-NOT-READY $DRYBAD"; exit 1', 'echo "ZZSOUP-DRY-NOT-READY $DRYBAD"; exit 0', "dry_run_reports"),
    ("dry_always_ok", SH, 'if [ -z "$DRYBAD" ]; then echo "ZZSOUP-DRY-OK"; exit 0; fi', 'if true; then echo "ZZSOUP-DRY-OK"; exit 0; fi', "dry_run_reports"),
    ("dry_parse_failure_ignored", SH, '*) DRYBAD="${DRYBAD:+$DRYBAD; }step $tag does not parse" ;; esac', "*) : ;; esac", "dry_run_reports"),
    ("dry_pins_mode_ignores_prereg_in_dry", SH, '[ "$MODE" = dry ] && { say "a real run would STOP here: $PRE_WHY"; DRYBAD="PREREG: $PRE_WHY"; }', '[ "$MODE" = dry ] && { say "a real run would STOP here: $PRE_WHY"; }', "dry_run_reports"),
    # ---- gates and readiness
    ("c_gate_off", SH, '[ "$CF0" -ge "$CMIN" ] || stop "C: has', '[ "$CF0" -ge "$CMIN" ] || say "C: has', "c_drive_gate"),
    ("plan_invalid_ignored", SH, '|| stop "the plan $PLANm is invalid:', '|| say "the plan $PLANm is invalid:', "invalid_plan"),
    ("readiness_ignored", SH, '|| stop "a checkpoint is not ready (section 13a):', '|| say "a checkpoint is not ready (section 13a):', "readiness_stops"),
    ("reader_files_missing_ignored", SH, '|| stop "a reader file is missing', '|| say "a reader file is missing', "missing_reader"),
    ("li_reader_sha_unchecked", SH, '[ "$(fsha "$LIREADER")" = "$LIREADER_SHA" ] || stop', '[ "$(fsha "$LIREADER")" = "$LIREADER_SHA" ] || say', "changed_input"),
    ("pins_verify_ignored", SH, '|| stop "PINS.txt: $(echo "$PV"', '|| say "PINS.txt: $(echo "$PV"', "changed_input or pinned_chain_script"),
    ("pins_write_ignores_failure", SH, '|| stop "cannot write $PINSm"; exit 0', "|| true; exit 0", "pins_file_that_cannot or write_pins"),
    ("lib_arg_missing_from_pins", SH, '--lib "$LIB" --chain "$Hm/soup_chain.sh"', '--lib "$LIB"', "write_pins_covers"),
    ("chain_arg_missing_from_pins", SH, ' --chain "$Hm/soup_chain.sh"', "", "write_pins_covers"),
    ("templates_missing_from_pins", SH, '"${TMPLARGS[@]}" "${EXPT[@]}")', '"${EXPT[@]}")', "write_pins_covers"),
    ("test_world_pins_nothing_else", SH, 'EXPT=(); [ "$TESTWORLD" = 1 ] && EXPT=(--expect-tree-digest none)', "EXPT=()", "write_pins_covers or dry_run_prints"),
    # ---- per-step discipline
    ("step_pin_compare_off", SH, '[ -n "$pinned" ] && [ "$sha" = "$pinned" ] || stop "step $tag:', '[ -n "$pinned" ] && [ "$sha" = "$pinned" ] || say "step $tag:', "changing_during"),
    ("skip_never", SH, "  if why=$(", '  why=skipoff; if false && why=$(', "resume_skips"),
    ("skip_ignores_command", SH, '--tool-digest "$TOOLD" --cmd-sha "$cs" < /dev/null 2>&1); then say "skip', '--tool-digest "$TOOLD" --cmd-sha "$sha" < /dev/null 2>&1); then say "skip', "resume_skips"),
    ("done_record_without_command_sha", SH, '--before "$sha" --after "$sha2" --tool-digest "$TOOLD" --cmd-sha "$cs"', '--before "$sha" --after "$sha2" --tool-digest "$TOOLD" --cmd-sha "$sha"', "real_run_executes"),
    ("done_record_without_tool_digest", SH, '--before "$sha" --after "$sha2" --tool-digest "$TOOLD"', '--before "$sha" --after "$sha2" --tool-digest "$sha"', "real_run_executes"),
    ("done_record_after_is_before", SH, '--after "$sha2"', '--after "$pinned"', "done_record_is_wrong"),
    ("aside_off", SH, '  [ $any -eq 1 ] && aside "${OUTS[@]}"', "  true", "incomplete_earlier"),
    ("aside_deletes_instead", SH, 'for f in "$@"; do [ -e "$(u "$f")" ] && mv "$(u "$f")" "$w/"; done', 'for f in "$@"; do [ -e "$(u "$f")" ] && rm -rf "$(u "$f")"; done', "incomplete_earlier"),
    ("guards_wait_removed", SH, '  guards_wait "$tag"\n  nostop; pin_check "before $tag"', '  nostop; pin_check "before $tag"', "stays_low"),
    ("guard_ram_ignored", SH, '--ram-min-mb "$RAMMIN"', "--ram-min-mb 0", "stays_low"),
    ("guard_disk_ignored", SH, '--disk-min-gb "$FMIN"', "--disk-min-gb 0", "stays_low"),
    ("guard_c_ignored", SH, '--c-min-mb "$CMIN" 2>&1', "--c-min-mb 0 2>&1", "stays_low"),
    ("guard_disk_is_c", SH, '--disk "${OUTm:0:3}"', '--disk "C:/"', "disk_guard_looks or stays_low"),
    ("guard_wait_never_gives_up", SH, '[ $(( $(date +%s) - t0 )) -ge "$GUARDWAIT_S" ] && stop', '[ $(( $(date +%s) - t0 )) -ge "$GUARDWAIT_S" ] && return 0 #', "stays_low"),
    ("guard_recovery_unannounced", SH, '[ $w -gt 0 ] && say "$what: guards hold again ($out)"; return 0', "return 0", "low_guard_waits"),
    ("nostop_blind", SH, 'nostop() { [ "$MODE" = real ] || return 0; [ -e "$LOGu/STOP" ] && stop', 'nostop() { [ "$MODE" = real ] || return 0; [ -e "$LOGu/STOP_" ] && stop', "stop_file"),
    ("stale_stop_survives_restart", SH, '  rm -f "$LOGu/STOP"\n', "", "stop_file"),
    ("tree_not_rechecked_after", SH, '  pin_check "after $tag"\n', "", "changing_during"),
    ("t0_tool_not_rechecked", SH, '[ "$(fsha "$MINI")" = "$MINISHA" ] || stop "the T0 tool', '[ "$(fsha "$MINI")" = "$MINISHA" ] || say "the T0 tool', "t0_tool_changing"),
    ("tree_digest_not_rechecked", SH, 'pin_check() { [ "$(tree_digest)" = "$TOOLD" ] || stop "the tool tree changed ($1)";', 'pin_check() { [ "$(tree_digest)" = "$TOOLD" ] || say "the tool tree changed ($1)";', "changing_during"),
    ("ckpt_change_during_ignored", SH, '[ "$sha2" = "$sha" ] || stop "$ck changed during', '[ "$sha2" = "$sha" ] || say "$ck changed during', "changing_during"),
    ("exit_code_ignored", SH, 'if [ $rc -ne 0 ]; then stop "step $tag failed', 'if false; then stop "step $tag failed', "failed_step_stops"),
    ("outputs_unchecked", SH, '|| stop "step $tag: $o"', "|| true", "exits_zero"),
    ("argv_not_recorded", SH, '''    case "$kind" in causal|logged) printf '%s\\n' "${CMD[@]}" > "$OUTu/argv/$tag.argv" ;; esac''', "    :", "real_run_executes"),
    ("segv_not_retried", SH, "if [ $rc -eq 139 ] && [ $att -eq 1 ]; then", "if false; then", "native_crash"),
    ("segv_retry_keeps_partial_output", SH, '''mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; aside "${OUTS[@]}"; continue''', '''mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; continue''', "native_crash"),
    ("final_verification_off", SH, '[ "$n" -eq "$NSTEPS" ] || stop "$n of $NSTEPS steps have a verifying done record"', '[ "$n" -eq "$NSTEPS" ] || say "$n of $NSTEPS steps have a verifying done record"', "stops_verifying"),
    ("manifest_failure_ignored", SH, '|| stop "manifest: $(tr', '|| say "manifest: $(tr', "manifest_that_cannot"),
    ("tmp_on_c", SH, 'export TMP="$OUTm/_tmp" TEMP="$OUTm/_tmp"', ":", "real_run_executes"),
    ("cuda_not_hidden", SH, "export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8", "export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8", "real_run_executes"),
    ("cmd_sha_is_newline_joined", SH, "printf '%s\\0' \"$@\" | sha256sum", "printf '%s\\n' \"$@\" | sha256sum", "real_run_executes"),
    # ---- the reader step
    ("reader_prereg_not_passed", SH, '--prereg "${PREREGm:-UNSET}"', '--prereg "UNSET"', "real_run_executes"),
    ("reader_prereg_sha_not_passed", SH, '--prereg-sha "${PREREG_SHA:-UNSET}"', '--prereg-sha "UNSET"', "real_run_executes"),
    ("reader_manifest_wrong", SH, '--manifest "$OUTm/runs_soup.json"', '--manifest "$OUTm/runs.json"', "real_run_executes"),
    ("reader_builder_sha_wrong", SH, '--soup-build-sha "$BSHA"', '--soup-build-sha "$RSHA"', "real_run_executes"),
    ("reader_li_sha_wrong", SH, '--li-reader-sha "$LIREADER_SHA"', '--li-reader-sha "$RSHA"', "real_run_executes"),
    ("reader_keeps_the_tool_tree_environment", SH, "env -u TANITEVAL_STACK_OVERRIDE -u PYTHONPATH OMP_NUM_THREADS=2", "env OMP_NUM_THREADS=2", "real_run_executes"),
    ("reader_threads", SH, 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RCMD[@]}"', 'OMP_NUM_THREADS=8 MKL_NUM_THREADS=2 "${RCMD[@]}"', "real_run_executes"),
    ("reader_refused_mislabelled", SH, "2) echo 'REFUSED, nothing written'", "2) echo 'outcome written'", "reader_exit_code"),
    ("reader_unplaced_mislabelled", SH, "3) echo 'UNPLACED, written'", "3) echo 'outcome written'", "reader_exit_code"),
    ("reader_crash_mislabelled", SH, "*) echo 'reader crashed'", "*) echo 'outcome written'", "reader_exit_code"),
    ("reader_changed_not_detected", SH, '[ "$(fsha "$READER")" = "$RSHA" ] || stop "the soup reader changed', '[ "$(fsha "$READER")" = "$RSHA" ] || say "the soup reader changed', "real_run_executes or reader_or_builder_that_changes"),
    # ---- the lock, the world, the log
    ("lock_alive_not_recognised", SH, '''grep -q " $op "; then echo "ZZSOUP-STOP another instance''', '''grep -q " X$op "; then echo "ZZSOUP-STOP another instance''', "only_one_instance"),
    ("stale_lock_kept", SH, '    rm -rf "$LOGu/chain.lock"; mkdir "$LOGu/chain.lock" ||', '    mkdir "$LOGu/chain.lock" ||', "only_one_instance"),
    ("lock_not_released", SH, "  trap 'rm -rf \"$LOGu/chain.lock\"' EXIT\n", "", "real_run_executes"),
    ("log_not_rotated", SH, '[ -s "$LOGu/chain.out" ] && mv "$LOGu/chain.out" "$LOGu/chain.out.prev_$(date +%m%d%H%M%S)"', "true", "resume_skips"),
    ("world_not_validated", SH, '|| { echo "ZZSOUP-STOP SOUP_WORLD=$WORLD is not a valid throw-away world: refused, nothing was run"; exit 1; }', "|| true", "world_marker"),
    ("stop_without_marker", SH, '''echo "ZZSOUP-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }''', '''touch "$LOGu/STOP"; exit 1; }''', "failed_step_stops or prereg_and_its_sha"),
    ("stop_without_stop_file", SH, '''echo "ZZSOUP-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }''', '''echo "ZZSOUP-STOP $*" >> "$LOGu/chain.out"; exit 1; }''', "failed_step_stops"),
    ("step_marker_missing", SH, '''echo "ZZSOUP-STEP-DONE $n/$NSTEPS $tag ($kind)" >> "$LOGu/chain.out"; tail -1 "$LOGu/chain.out"''', ":", "real_run_executes"),
    # ---- the lib (unit tests, and the runs they drive)
    ("world_token_unchecked", LB, 'open(mk, encoding="utf-8").read().split("\\n")[0].strip() != WORLD_TOKEN', "False", "world_ok"),
    ("world_marker_unchecked", LB, "if not os.path.isfile(mk) or", "if False or", "world_ok"),
    ("world_not_a_dir_unchecked", LB, 'if not w or not os.path.isdir(w):', "if not w:", "world_ok"),
    ("world_production_unchecked", LB, "if r == b or r.startswith(b + \"/\") or b.startswith(r + \"/\"):", "if False:", "world_ok"),
    ("world_inside_production_allowed", LB, 'if r == b or r.startswith(b + "/") or b.startswith(r + "/"):', 'if r == b or b.startswith(r + "/"):', "world_ok"),
    ("world_above_production_allowed", LB, 'if r == b or r.startswith(b + "/") or b.startswith(r + "/"):', 'if r == b or r.startswith(b + "/"):', "world_ok"),
    ("world_production_list_short", LB, "for bad in (OUT_PROD, HERE, TREE_PROD):", "for bad in (OUT_PROD, HERE):", "world_ok"),
    ("plan_py_unchecked", LB, "or tok[0] != py:", "or False:", "parse_plan"),
    ("plan_short_line_allowed", LB, "if len(tok) < 8 or", "if len(tok) < 1 or", "parse_plan"),
    ("plan_tool_dir_unchecked", LB, 'if not tool.startswith(tree + "/stack/scripts/") or', "if False or", "parse_plan"),
    ("plan_tool_name_unchecked", LB, "or os.path.basename(tool) not in (T1_TOOL, T0_TOOL) or", "or False or", "parse_plan"),
    ("plan_subcommand_unchecked", LB, 'or tok[2] != "run":', "or False:", "parse_plan"),
    ("plan_ckpt_name_unchecked", LB, 'if not ckpt.endswith("/ckpt_last.pt"):', "if False:", "parse_plan"),
    ("plan_duplicate_tag_allowed", LB, "if tag in seen or not re.match", "if not re.match", "parse_plan"),
    ("plan_tag_charset_unchecked", LB, "if tag in seen or not re.match(", "if tag in seen or False and not re.match(", "parse_plan"),
    ("plan_limit_unchecked", LB, 'if kind not in ("causal", "logged") or od != out or', "if od != out or", "parse_plan"),
    ("plan_out_dir_unchecked", LB, 'if kind not in ("causal", "logged") or od != out or', 'if kind not in ("causal", "logged") or', "parse_plan"),
    ("plan_dump_ticks_unchecked", LB, 'or norm(_flag(tok, "--dump-ticks", ln_no)) != out + "/ticks":', "or False:", "parse_plan"),
    ("plan_dump_heads_optional", LB, 'if "--dump-heads" not in tok:', "if False:", "parse_plan"),
    ("plan_ages_value_unchecked", LB, 'if fresh and _flag(tok, "--ages", ln_no) != "fresh":', "if False:", "parse_plan"),
    ("plan_t0_dir_unchecked", LB, 'if not od.startswith(out + "/t0"):', "if False:", "parse_plan"),
    ("plan_dev_kind_wrong", LB, '"t0dev" if tag.endswith("-dev") else "t0std"', '"t0std" if tag.endswith("-dev") else "t0dev"', "parse_plan"),
    ("plan_crlf_allowed", LB, 'if "\\r" in text:', "if False:", "parse_plan"),
    ("plan_empty_allowed", LB, "if not steps:", "if False:", "parse_plan"),
    ("plan_comment_lines_parsed", LB, 'if not ln.strip() or ln.startswith("#"):', "if not ln.strip():", "parse_plan"),
    ("expected_t1_files", LB, '"files": [f"{o}/t1v8f_{t}.json", f"{o}/ticks/ticks_{t}.npz", f"{o}/argv/{t}.argv"]', '"files": [f"{o}/t1v8f_{t}.json", f"{o}/ticks/ticks_{t}.npz"]', "expected_outputs or manifest or real_run"),
    ("expected_t0_done_name", LB, '"done": f"{o}/done_t0_{t}.txt"', '"done": f"{o}/done_{t}.txt"', "expected_outputs or real_run"),
    ("expected_proj_not_listed", LB, '"extra": [f"{d}/eval_{t}_proj.npz"]', '"extra": []', "expected_outputs"),
    ("dump_count_threshold", LB, "if ex[\"dumps\"] and n_dumps(ex[\"dumps\"]) < NDUMP:", "if ex[\"dumps\"] and n_dumps(ex[\"dumps\"]) < NDUMP - 1:", "expected_outputs"),
    ("dump_name_pattern", LB, '''re.match(r"^ep.*\\.npz$", f)''', '''re.match(r"^.*\\.npz$", f)''', "expected_outputs"),
    ("empty_output_accepted", LB, "if not os.path.isfile(f) or os.path.getsize(f) == 0:", "if not os.path.isfile(f):", "expected_outputs"),
    ("done_before_not_checked", LB, '("ckpt_sha256_before", ckpt_sha), ', "", "step_done"),
    ("done_after_not_checked", LB, '("ckpt_sha256_after", ckpt_sha), ', "", "step_done"),
    ("done_ckpt_not_checked", LB, 'for k, want in (("ckpt_sha256", ckpt_sha), ', "for k, want in (", "step_done"),
    ("done_tool_not_checked", LB, '("tool_digest", tool_digest), ', "", "step_done"),
    ("done_command_not_checked", LB, '("cmd_sha256", cmd), ', "", "step_done"),
    ("done_tag_not_checked", LB, '("tag", step["tag"])', '("tag", d.get("tag"))', "step_done"),
    ("done_smoke_accepted", LB, 'if d.get("smoke") is not False or d.get("max_windows") is not None:', 'if d.get("max_windows") is not None:', "step_done"),
    ("done_window_limit_accepted", LB, 'if d.get("smoke") is not False or d.get("max_windows") is not None:', 'if d.get("smoke") is not False:', "step_done"),
    ("done_outputs_not_rechecked", LB, "    probs = step_problems(step)\n    return (False, probs[0]) if probs else (True, \"complete\")", '    return (True, "complete")', "step_done"),
    ("done_record_before_default", LB, '"ckpt_sha256_before": before or ckpt_sha,', '"ckpt_sha256_before": before or tool_digest,', "step_done"),
    ("done_record_not_a_full_run", LB, '"smoke": False, "max_windows": None}', '"smoke": True, "max_windows": None}', "step_done or real_run"),
    ("ram_compare_off", LB, "if ram < ram_min_mb:", "if False:", "guard_hooks"),
    ("disk_compare_off", LB, "if free < disk_min_gb:", "if False:", "guard_hooks"),
    ("c_compare_off", LB, "if cfree < c_min_mb:", "if False:", "guard_hooks"),
    ("ram_hook_in_production", LB, '    if not world_ok():\n        return None\n    fp = os.path.join(os.environ["SOUP_WORLD"]', '    fp = os.path.join(os.environ.get("SOUP_WORLD", "")', "guard_hooks"),
    ("ready_soup_sha_unchecked", LB, 'if r.get("output_sha256") != got:', "if False:", "ready"),
    ("ready_basin_unchecked", LB, 'if (r.get("K-BASIN") or {}).get("PASS") is not True:', "if False:", "ready"),
    ("ready_basin_truthy", LB, 'if (r.get("K-BASIN") or {}).get("PASS") is not True:', 'if not (r.get("K-BASIN") or {}).get("PASS"):', "ready"),
    ("ready_member_sha_unchecked", LB, "            if got != want:", "            if False:", "ready"),
    ("ready_missing_file_unchecked", LB, "if not os.path.isfile(ck):", "if False:", "ready"),
    ("ready_unknown_ckpt_accepted", LB, "else:\n            probs.append(f\"{ck}: neither a soup", "elif False:\n            probs.append(f\"{ck}: neither a soup", "ready"),
    ("pins_no_ckpt_entries", LB, '        ents.append(f"ckpt={ck} {sha256_file(ck)}")', "        pass", "pins_check or write_pins"),
    ("pins_no_t0_tool", LB, '    ents.append(f"tool={t0} {sha256_file(t0)}")\n', "", "pins_check or write_pins"),
    ("pins_tree_digest_not_expected", LB, "    if expect_tree_digest is not None and have.get", "    if False and have.get", "pins_check"),
    ("pins_union_is_intersection", LB, "for k in sorted(set(have) | set(want))", "for k in sorted(set(have) & set(want))", "pins_check"),
    ("pins_missing_file_ok", LB, 'if not os.path.isfile(pins_path):\n        return [f"no PINS file at {pins_path}"]', "if False:\n        pass", "pins_check"),
    ("pins_default_digest_off", LB, "    return a.expect_tree_digest or TOOL_DIGEST", "    return a.expect_tree_digest", "pins_check or dry_run_prints or write_pins or pins_write_refuses"),
    ("pins_write_digest_unchecked", LB, "if _expect_digest(a) is not None and tree_digest(a.tree) != _expect_digest(a):", "if False:", "pins_write_refuses_wrong_digest"),
    ("digest_ignores_taniteval", LB, 'for r in ("stack/tanitad", "taniteval"):', 'for r in ("stack/tanitad",):', "tree_digest"),
    ("digest_ignores_t1_tool", LB, '    paths.add("stack/scripts/v8f_t1_eval.py")\n', "", "tree_digest"),
    ("digest_counts_non_python", LB, 'if f.endswith(".py"):', "if True:", "tree_digest"),
    ("manifest_no_dev_t0", LB, '            arms[r][s]["t0_dev"] = t0(f"M2b-{stem}_{s}-dev")', "            pass", "manifest"),
    ("manifest_dev_stems_swapped", LB, '(("CTRL", "dagoacc"), ("A", "a"))', '(("CTRL", "a"), ("A", "dagoacc"))', "manifest"),
    ("manifest_soup_dir", LB, '"soup_dir": os.path.dirname(one(f"{name}_m10_causal_heads")["ckpt"])', '"soup_dir": os.path.dirname(one(f"{name}_m10_causal_heads")["out"])', "manifest"),
    ("manifest_fresh_is_standard", LB, '"t0_fresh": t0(f"M2b-{name.lower()}-fresh")}', '"t0_fresh": t0(f"M2b-{name.lower()}")}', "manifest"),
    ("manifest_logged_is_causal", LB, '"t1_logged": t1(f"{name}_m10_logged_heads")', '"t1_logged": t1(f"{name}_m10_causal_heads")', "manifest"),
    ("manifest_xinit_included", LB, '        if sp.get("expect"):\n            continue\n        role, s', '        role, s', "manifest"),
    ("manifest_steps_unchecked", LB, "        pr = step_problems(s)\n        if pr:", "        pr = None\n        if pr:", "manifest_from_the_plan"),
    ("manifest_done_unchecked", LB, "        if not os.path.isfile(expected(s)[\"done\"]):", "        if False:", "manifest"),
    ("manifest_seed_not_generic", LB, 'seeds = [str(x) for x in li_runs.get("seeds", [0, 1])]', 'seeds = ["0", "1"]', "manifest"),
    ("manifest_mutates_li_runs", LB, 'arms = {r: json.loads(json.dumps(li_runs["arms"][r])) for r in ("CTRL", "REP", "A", "Arep")}', 'arms = {r: li_runs["arms"][r] for r in ("CTRL", "REP", "A", "Arep")}', "manifest"),
    ("parsecheck_runs_the_tool", LB, "    raise Cap()\\n", "    pass\\n", "parsecheck_cuts"),
    ("pinsha_wrong_key", LB, 'ln.rpartition(" ")[0] == f"ckpt={norm(a.ckpt)}"', 'ln.rpartition(" ")[0].startswith("ckpt=")', "real_run_executes or changing_during"),
    ("cmd_sha_without_terminator", LB, 'return hashlib.sha256(b"".join(a.encode() + b"\\0" for a in argv)).hexdigest()', 'return hashlib.sha256(b"\\0".join(a.encode() for a in argv)).hexdigest()', "cmd_sha"),
    ("fmt_quotes_everything", LB, 'f"\'{a}\'" if re.search(r"[^A-Za-z0-9_./:=,@%+-]", a) else a', 'f"\'{a}\'"', "cmd_sha_is"),
    # ---- the waker
    ("waker_prose_matches", "wake_soup.sh", "grep -aq '^ZZSOUP-DONE'", "grep -aq 'ZZSOUP-DONE'", "waker"),
    ("waker_ignores_stop", "wake_soup.sh", "if grep -aq '^ZZSOUP-STOP' \"$O\" 2>/dev/null; then echo \"CHAIN STOP: ", "if false; then echo \"CHAIN STOP: ", "waker"),
    ("waker_dead_chain_unseen", "wake_soup.sh", '! tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -q " $pid "', "false", "waker"),
    ("waker_reads_rotated_logs", "wake_soup.sh", "L=${SOUP_LOG:-$H/logs}; O=$L/chain.out;", "L=${SOUP_LOG:-$H/logs}; O=$L/chain.out*;", "waker"),
    ("waker_launcher_log_ignores_lock", "wake_soup.sh", 'if [ ! -e "$L/chain.lock" ] && grep', "if true && grep", "waker"),
    ("waker_never_times_out", "wake_soup.sh", 'echo "WAKER TIMEOUT"; tail -5 "$O" 2>/dev/null | cut -c1-260; exit 5', 'echo "WAKER TIMEOUT"; tail -5 "$O" 2>/dev/null | cut -c1-260; exit 0', "waker"),
    # ---- the launcher
    ("launcher_sets_the_prereg", "soup_chain.cmd", "rem   set SOUP_PREREG=E:/Projects/TanitAD/Project Steering/PREREG_V8F_SOUP.md", "set SOUP_PREREG=E:/Projects/TanitAD/Project Steering/PREREG_V8F_SOUP.md", "launcher"),
    ("launcher_appends_its_log", "soup_chain.cmd", "logs/soup_chain.launch.out 2>&1", "logs/soup_chain.launch.out 2>&1 >>", "launcher"),
]

# A mutant that cannot change behaviour is NAMED with the reason; the pass requires every other mutant killed.
EQUIVALENT = {
    "done_record_after_is_before": "`--after \"$sha2\"` -> `--after \"$pinned\"`: at that line `$sha2 == $sha` (the step stops 'changed during' otherwise) and `$sha == $pinned` (the step stops 'hashes to ... PINS.txt has' before it starts), so the same value is written",
    "world_not_a_dir_unchecked": "`os.path.isdir(w)` is redundant in world_ok: when w is not a directory, `os.path.join(w, marker)` cannot be a file, so the very next check (`os.path.isfile(mk)`) returns False anyway",
}


def apply(src, old, new, name):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"mutant {name}: anchor occurs {n} times (need exactly 1): {old[:100]!r}")
    return src.replace(old, new, 1)


def pytest_run(mut_dir, tmp, k=None, full=False):
    env = dict(os.environ, SOUP_CHAIN_DIR=mut_dir, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    for key in [x for x in env if x.startswith("SOUP_") and x != "SOUP_CHAIN_DIR"]:
        del env[key]
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", tmp, TEST]
    if k:
        cmd += ["-k", k]
    if full:
        cmd += ["-k", " and ".join(f"not {d}" for d in FULL_DESELECT)]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=3600, encoding="utf-8", errors="replace")
    failed = [ln.split(" - ")[0].replace("FAILED ", "").strip() for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
    r = {"rc": p.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": (p.stdout.strip().splitlines() or [""])[-1]}
    if p.returncode != 0:
        r["fail_output"] = [ln[:300] for ln in p.stdout.splitlines()][-60:]                       # why it failed: a kill must be explainable, not just counted
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--check-anchors", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--merge", default="", help="an earlier --out json: its verdicts are kept for every mutant not re-run now (the new verdicts replace the old; the file records which pass each verdict is from)")
    a = ap.parse_args(argv)
    srcs = {f: open(f"{W}/{f}", encoding="utf-8", newline="").read() for f in FILES}
    names = [m[0] for m in M]
    if a.check_anchors:
        bad = [(n, f, srcs[f].count(o)) for n, f, o, nw, k in M if srcs[f].count(o) != 1 or o == nw]
        dup = sorted({n for n in names if names.count(n) > 1})
        print(f"{len(M)} mutants; bad anchors: {bad}; duplicate names: {dup}")
        return 0 if not bad and not dup else 1
    if not (a.work and a.out):
        raise SystemExit("--work and --out are required")
    os.makedirs(f"{a.work}/pytmp", exist_ok=True)
    res = {"tests_sha256": TESTS_SHA, "runner_sha256": {f: hashlib.sha256(srcs[f].encode("utf-8")).hexdigest() for f in FILES}, "baseline": None, "mutants": {}, "equivalent_named": {}}

    def mk(name, f=None, old=None, new=None):
        d = f"{a.work}/mutants/{name}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for ff in FILES:
            with open(f"{d}/{ff}", "w", encoding="utf-8", newline="") as fh:
                fh.write(apply(srcs[ff], old, new, name) if ff == f else srcs[ff])
        return d
    if not a.skip_baseline:
        res["baseline"] = pytest_run(mk("_baseline"), f"{a.work}/pytmp/000f", full=True)
        print("baseline:", res["baseline"], flush=True)
        if res["baseline"]["rc"] != 0:
            json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
            print("ZZ-MUTATION-STOP baseline does not pass")
            return 2
    only = set(filter(None, a.only.split(",")))
    for idx, (name, f, old, new, k) in enumerate(M, 1):
        if only and name not in only:
            continue
        d = mk(name, f, old, new)
        # the pytest basetemp has the SAME short length for every mutant and for the baseline: a name-dependent path length once 'killed' 14 mutants through MAX_PATH alone (see tests lp())
        r = pytest_run(d, f"{a.work}/pytmp/{idx:03d}t", k=k)
        r["stage"] = "targeted"
        if r["rc"] == 0:
            r2 = pytest_run(d, f"{a.work}/pytmp/{idx:03d}f", full=True)
            r2["stage"] = "full"
            r = {**r2, "targeted_seconds": r["seconds"], "stage": "full"}
        r["killed"] = r["rc"] != 0
        r["tests_sha256"] = TESTS_SHA
        res["mutants"][name] = r
        print(f"{'KILLED  ' if r['killed'] else 'SURVIVED'} {name} [{r['stage']}]: {r['failed'][:1]} ({r['seconds']}s)", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    if a.merge:
        prev = json.load(open(a.merge, encoding="utf-8"))
        assert prev["runner_sha256"] == res["runner_sha256"], "the runner changed since the merged pass: its verdicts are about a different runner"
        merged = {**prev["mutants"], **res["mutants"]}
        res["merged_from"] = {"file": a.merge.replace("\\", "/"), "n_kept_verdicts": len([n for n in prev["mutants"] if n not in res["mutants"]]), "kept_verdicts_tests_sha256": prev.get("tests_sha256"), "baseline": prev.get("baseline")}
        res["mutants"] = merged
    surv = sorted(n for n, r in res["mutants"].items() if not r["killed"])
    res["n_mutants"], res["n_killed"] = len(res["mutants"]), sum(r["killed"] for r in res["mutants"].values())
    res["equivalent_named"] = {n: EQUIVALENT[n] for n in surv if n in EQUIVALENT}
    res["survivors"] = [n for n in surv if n not in EQUIVALENT]
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"ZZ-MUTATION-DONE {res['n_killed']}/{res['n_mutants']} killed; named equivalent={sorted(res['equivalent_named'])}; UNEXPLAINED survivors={res['survivors']}", flush=True)
    return 0 if not res["survivors"] else 1


if __name__ == "__main__":
    sys.exit(main())
