#!/bin/bash
# finish_round4.sh : the tail of round 4 (the gate re-reads the arm directory; the pod precheck lists the S-reader files).  Run AFTER build/sweep_round4.sh has written C:/so/logs_r4_DONE, with the final tree
# C:/so/t8 (a fresh build of the pristine tip C:/so/b by build/apply_all.sh) and its second fresh build C:/so/t9 on disk.  Regenerates TEST_TABLE.md, deliverables/, diffs/, BLOBS.txt,
# logs/verify_blobs.json, the landing check and logs/round4/.
HERE="$(cd "$(dirname "$0")" && pwd)"; PKG="$(dirname "$HERE")"
PY="${PYTHON:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
FINAL=C:/so/t8
mkdir -p "$PKG/logs/round4" "$PKG/logs/sweep"
# what differs between the round-3 tree (t6) and the round-4 tree (t8), and between two fresh builds of round 4 (t8, t9: must be empty)
diff -rq -x __pycache__ "C:/so/t6/$PKR" "$FINAL/$PKR" > "$PKG/logs/round4/t6_vs_t8.txt" 2>&1
( cd "C:/so/t6/$PKR" && for f in eval/safeobj_read.py refe/rls1_checks.py refe/rls1_pod.sh refe/test_rl_safeobj.py refe/test_rls1_chain.py refe/test_rls1_queue.py eval/test_safeobj_read.py eval/test_safeobj_gate.py; do diff -u --label "a/$f" --label "b/$f" "$f" "$FINAL/$PKR/$f"; done ) > "$PKG/logs/round4/t6_vs_t8.diff" 2>&1
diff -rq -x __pycache__ "$FINAL/$PKR" "C:/so/t9/$PKR" > "$PKG/logs/round4/t8_vs_t9.txt" 2>&1; echo "diff -rq rc=$? (0 = byte-identical)" >> "$PKG/logs/round4/t8_vs_t9.txt"
export SO_BASE_RERUN_DIRS="logs_tb,logs_tb2,logs_tb4"
export SO_PATCHED_DIRS="logs_t1,logs_t2,logs_t3,logs_t4,logs_t5,logs_t9,logs_t10,logs_t12,logs_t14,logs_r3_1,logs_r3_2,logs_r4_1,logs_r4_2"
"$PY" -X utf8 "$HERE/make_test_table.py" "$FINAL" "$PKG" > "$PKG/logs/make_test_table.out" 2>&1
bash "$HERE/make_deliverables.sh" "C:/so/b/$PKR" "$FINAL/$PKR" "$PKG" > "$PKG/logs/make_deliverables.out" 2>&1
( cd "$HERE" && "$PY" -X utf8 verify_blobs.py "$FINAL/$PKR" > "$PKG/logs/verify_blobs.json" 2>&1; echo "verify_blobs rc=$?" >> "$PKG/logs/verify_blobs.json" )
L=C:/so/land5; rm -rf "$L"; mkdir -p "$L"; cp -r "C:/so/b/$PKR/refe" "C:/so/b/$PKR/eval" "$L/"; find "$L" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
{
  cd "$L" && for d in "$PKG"/diffs/*.diff; do
    if patch -p1 -s --no-backup-if-mismatch < "$d" > /dev/null 2>&1; then echo "applied $(basename "$d")"; else echo "FAILED $(basename "$d")"; fi
  done
  for f in refe/scorer_finetune.py refe/launch_gate.py refe/gate_checks.py refe/rls1_checks.py refe/rls1_pod.sh refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_epo.py refe/test_launch_gate.py refe/test_rls1_chain.py refe/test_rls1_queue.py refe/test_dump_scores.py refe/test_aux_s1_wiring.py refe/test_rl_safeobj.py eval/rls1d_arms.json eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py; do
    if cmp -s "$L/$f" "$FINAL/$PKR/$f" && cmp -s "$L/$f" "$PKG/deliverables/$f"; then echo "LANDING-OK  $f"; else echo "LANDING-DIFF $f"; fi
  done
} > "$PKG/logs/landing_check.txt" 2>&1
for d in logs_r4_1 logs_r4_2; do [ -d C:/so/$d ] && { mkdir -p "$PKG/logs/sweep/$d"; cp C:/so/$d/*.log C:/so/$d/SUMMARY.txt "$PKG/logs/sweep/$d/" 2>/dev/null; }; done
cp C:/so/logs_r4_procs.txt "$PKG/logs/round4/procs.txt" 2>/dev/null
cp C:/so/build_t8.log "$PKG/logs/round4/build_t8.log" 2>/dev/null; cp C:/so/build_t9.log "$PKG/logs/round4/build_t9.log" 2>/dev/null
date > C:/so/logs_round4_FINISH_DONE
