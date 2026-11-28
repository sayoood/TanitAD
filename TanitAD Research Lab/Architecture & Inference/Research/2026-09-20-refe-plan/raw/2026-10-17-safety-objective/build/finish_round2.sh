#!/bin/bash
# finish_round2.sh : the tail of round 2 (queue-test fix + S-reader).  Run AFTER logs_t9 (test_rls1_queue on C:/so/t4), logs_t10 (test_safeobj_read + test_rl_safeobj on C:/so/t4) and
# logs_t11 (test_launch_gate under bashshim.py on C:/so/t4) exist (logs_t6 / logs_t7: the earlier runs on C:/so/t3, kept as history; logs_t6 = the queue test failing at its SECOND wrong assertion, the arm list 'R1d R1x').  Regenerates TEST_TABLE.md, deliverables/, diffs/, BLOBS.txt, logs/verify_blobs.json, the landing check and logs/round2/.
HERE="$(cd "$(dirname "$0")" && pwd)"; PKG="$(dirname "$HERE")"
PY="${PYTHON:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
FINAL=C:/so/t4
mkdir -p "$PKG/logs/round2" "$PKG/logs/sweep"
( diff -rq -x __pycache__ "C:/so/t2/$PKR" "$FINAL/$PKR"; echo "--- t (first sweep tree) vs t4:"; diff -rq -x __pycache__ "C:/so/t/$PKR" "$FINAL/$PKR" ) > "$PKG/logs/round2/t2_vs_t4.txt" 2>&1
export SO_BASE_RERUN_DIRS="logs_tb,logs_tb2,logs_tb4"; export SO_PATCHED_DIRS="logs_t1,logs_t2,logs_t3,logs_t4,logs_t5,logs_t9,logs_t10,logs_t12,logs_t14"
"$PY" -X utf8 "$HERE/make_test_table.py" "$FINAL" "$PKG" > "$PKG/logs/make_test_table.out" 2>&1
bash "$HERE/make_deliverables.sh" "C:/so/b/$PKR" "$FINAL/$PKR" "$PKG" > "$PKG/logs/make_deliverables.out" 2>&1
( cd "$HERE" && "$PY" -X utf8 verify_blobs.py "$FINAL/$PKR" > "$PKG/logs/verify_blobs.json" 2>&1; echo "verify_blobs rc=$?" >> "$PKG/logs/verify_blobs.json" )
L=C:/so/land3; rm -rf "$L"; mkdir -p "$L"; cp -r "C:/so/b/$PKR/refe" "C:/so/b/$PKR/eval" "$L/"; find "$L" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
{
  cd "$L" && for d in "$PKG"/diffs/*.diff; do
    if patch -p1 -s --no-backup-if-mismatch < "$d" > /dev/null 2>&1; then echo "applied $(basename "$d")"; else echo "FAILED $(basename "$d")"; fi
  done
  for f in refe/scorer_finetune.py refe/launch_gate.py refe/gate_checks.py refe/rls1_checks.py refe/rls1_pod.sh refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_epo.py refe/test_launch_gate.py refe/test_rls1_chain.py refe/test_rls1_queue.py refe/test_dump_scores.py refe/test_aux_s1_wiring.py refe/test_rl_safeobj.py eval/rls1d_arms.json eval/safeobj_read.py eval/test_safeobj_read.py; do
    if cmp -s "$L/$f" "$FINAL/$PKR/$f" && cmp -s "$L/$f" "$PKG/deliverables/$f"; then echo "LANDING-OK  $f"; else echo "LANDING-DIFF $f"; fi
  done
} > "$PKG/logs/landing_check.txt" 2>&1
for d in logs_t6 logs_t7 logs_t9 logs_t10 logs_t12 logs_t13 logs_t14 logs_tb3 logs_tb4; do [ -d C:/so/$d ] && { mkdir -p "$PKG/logs/sweep/$d"; cp C:/so/$d/*.log C:/so/$d/SUMMARY.txt "$PKG/logs/sweep/$d/" 2>/dev/null; }; done
cp C:/so/drv/reader_test*.log "$PKG/logs/round2/" 2>/dev/null
date > C:/so/logs_round2_FINISH_DONE
