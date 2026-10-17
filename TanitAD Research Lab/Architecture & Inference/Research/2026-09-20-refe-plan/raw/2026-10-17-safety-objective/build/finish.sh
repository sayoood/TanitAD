#!/bin/bash
# finish.sh : the automatic tail of the SAFEOBJ build.  Run DETACHED right after sweep.sh was started; it waits for the sweep (C:/so/logs_t_DONE) and then
#   1. HARDENING: t2 = a copy of the swept tree t + the hardened test_rl_safeobj.py (build/test_rl_safeobj.template.py via gen_safeobj_test.py: the eps-shift check and mutation M27).
#      The ONLY file that differs between t and t2 is refe/test_rl_safeobj.py, and the ONLY test that reads it is refe/test_rl_r0c.py (imports it) -> both are rerun on t2, two processes (the cap).
#   2. FINAL tree = t2 if both pass, else t (HARDENING.md says which, with the reason).
#   3. TEST_TABLE.md, deliverables/ + diffs/ + BLOBS.txt (make_deliverables.sh), a LANDING CHECK (the diffs applied with `patch -p1` to a copy of the pristine refe/ + eval/, then cmp with the final tree), logs/.
#   usage: finish.sh      (no arguments; writes C:/so/logs_t_FINISH_DONE when everything is written)
HERE="$(cd "$(dirname "$0")" && pwd)"; PKG="$(dirname "$HERE")"
PY="${PYTHON:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
rm -f C:/so/logs_t_FINISH_DONE
while [ ! -f C:/so/logs_t_DONE ]; do sleep 20; done
# ---- 1. the hardened tree
rm -rf C:/so/t2 C:/so/logs_t3 C:/so/logs_t4 C:/so/logs_t5 C:/so/logs_tb2
mkdir -p C:/so/t2 && cp -r C:/so/t/. C:/so/t2/
find C:/so/t2 -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
"$PY" -X utf8 "$HERE/gen_safeobj_test.py" "C:/so/t2/$PKR" > "$PKG/logs/gen_hardened_test.out" 2>&1
bash "$HERE/run_tests.sh" C:/so/t2 C:/so/logs_t3 refe/test_rl_safeobj.py &
P1=$!
bash "$HERE/run_tests.sh" C:/so/t2 C:/so/logs_t4 refe/test_rl_r0c.py &
P2=$!
wait $P1 $P2
# ---- 1b. test_launch_gate needs a POSIX `bash` that CreateProcess does not find on this box (see bashshim.py): rerun it under the shim on BOTH trees (patched = t2: it does not read the content of test_rl_safeobj.py)
bash "$HERE/run_tests_shim.sh" C:/so/t2 C:/so/logs_t5 refe/test_launch_gate.py &
P3=$!
bash "$HERE/run_tests_shim.sh" C:/so/b C:/so/logs_tb2 refe/test_launch_gate.py &
P4=$!
wait $P3 $P4
OK3=$(grep -c "^test_rl_safeobj rc=0 .*tok=ZZTEST_PASS" C:/so/logs_t3/SUMMARY.txt)
OK4=$(grep -c "^test_rl_r0c rc=0 .*tok=ZZTEST_PASS 26" C:/so/logs_t4/SUMMARY.txt)
if [ "$OK3" = "1" ] && [ "$OK4" = "1" ]; then
  FINAL=C:/so/t2; export SO_BASE_RERUN_DIRS="logs_tb,logs_tb2"; export SO_PATCHED_DIRS="logs_t1,logs_t2,logs_t3,logs_t4,logs_t5"
  { echo "# HARDENING (eps) : APPLIED"; echo; echo "The final deliverables are the swept tree + one extra check in refe/test_rl_safeobj.py: \`U_eps_shifts_the_violation_and_the_term_by_exactly_eps_through_epo_con_term_and_through_train_losses\` and mutation M27 (the call site passes a literal 0.0 for the slack); M16 (eps ignored by the term) now also goes RED behaviourally, not only on the pin."; echo "Reason: eps = 0 in every registered arm, so no earlier behavioural check read the slack (M16 was '[PIN ONLY]' in the first sweep)."; echo "Evidence: logs/sweep/logs_t3 (test_rl_safeobj on the hardened tree) and logs/sweep/logs_t4 (test_rl_r0c, which imports it)."; echo "No trainer, gate or chain line changed; the trainer blob is the one the whole sweep ran on."; } > "$PKG/HARDENING.md"
else
  FINAL=C:/so/t; export SO_BASE_RERUN_DIRS="logs_tb,logs_tb2"; export SO_PATCHED_DIRS="logs_t1,logs_t2,logs_t5"
  { echo "# HARDENING (eps) : NOT APPLIED"; echo; echo "The hardened test_rl_safeobj.py (eps-shift check + M27) did NOT pass on the hardened tree (test_rl_safeobj ok=$OK3, test_rl_r0c ok=$OK4), so the deliverables are the swept tree WITHOUT it."; echo "Logs: logs/sweep/logs_t3, logs/sweep/logs_t4.  The hardened template is build/test_rl_safeobj.template.py; either fix the check in build/test_rl_safeobj.template.py (the hardened template, which apply_all.sh uses) or regenerate the un-hardened test from build/test_rl_safeobj.template.pre_hardening.py, which is what the deliverables were built from."; echo "The un-hardened test passed in the sweep (logs/sweep/logs_t2).  M16 stays '[PIN ONLY]'."; } > "$PKG/HARDENING.md"
fi
# ---- 3. table, deliverables, landing check
"$PY" -X utf8 "$HERE/make_test_table.py" "$FINAL" "$PKG" > "$PKG/logs/make_test_table.out" 2>&1
bash "$HERE/make_deliverables.sh" "C:/so/b/$PKR" "$FINAL/$PKR" "$PKG" > "$PKG/logs/make_deliverables.out" 2>&1
( cd "$HERE" && "$PY" -X utf8 verify_blobs.py "$FINAL/$PKR" > "$PKG/logs/verify_blobs.json" 2>&1; echo "verify_blobs rc=$?" >> "$PKG/logs/verify_blobs.json" )
L=C:/so/land; rm -rf "$L"; mkdir -p "$L"; cp -r "C:/so/b/$PKR/refe" "C:/so/b/$PKR/eval" "$L/"; find "$L" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
{
  cd "$L" && for d in "$PKG"/diffs/*.diff; do
    if patch -p1 -s --no-backup-if-mismatch < "$d" > /dev/null 2>&1; then echo "applied $(basename "$d")"; else echo "FAILED $(basename "$d")"; fi
  done
  for f in refe/scorer_finetune.py refe/launch_gate.py refe/gate_checks.py refe/rls1_checks.py refe/rls1_pod.sh refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_epo.py refe/test_launch_gate.py refe/test_rls1_chain.py refe/test_rls1_queue.py refe/test_dump_scores.py refe/test_aux_s1_wiring.py refe/test_rl_safeobj.py eval/rls1d_arms.json; do
    if cmp -s "$L/$f" "$FINAL/$PKR/$f" && cmp -s "$L/$f" "$PKG/deliverables/$f"; then echo "LANDING-OK  $f"; else echo "LANDING-DIFF $f"; fi
  done
} > "$PKG/logs/landing_check.txt" 2>&1
mkdir -p "$PKG/logs/sweep"; for d in logs_t1 logs_t2 logs_t3 logs_t4 logs_t5 logs_tb logs_tb2 logs_b logs_b2; do [ -d C:/so/$d ] && { mkdir -p "$PKG/logs/sweep/$d"; cp C:/so/$d/*.log C:/so/$d/SUMMARY.txt "$PKG/logs/sweep/$d/" 2>/dev/null; }; done
cp C:/so/logs_t_procs.txt "$PKG/logs/sweep/procs.txt" 2>/dev/null
date > C:/so/logs_t_FINISH_DONE
