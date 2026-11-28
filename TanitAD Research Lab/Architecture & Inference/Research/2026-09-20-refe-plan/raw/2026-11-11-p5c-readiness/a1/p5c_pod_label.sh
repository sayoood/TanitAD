#!/usr/bin/env bash
# P5-C pod label runner, DUAL version (PREREG_P5C 4.1 / 4.2 + the CONDITIONAL amendment A1 for P5-R).  FAIL-CLOSED: any unmet precondition prints "ZZP5CL_FAIL <stage>" and exits non-zero.
# Run ON THE POD, AFTER P7's last step (the pod's CPUs and DBs are free then).  WRITTEN AND `bash -n` CHECKED ONLY; NEVER RUN BY ITS AUTHOR.
# Usage:  bash p5c_pod_label.sh            (env: P5CL_NSHARD=16  P5CL_SHIP=/workspace/data/p5c_ship  P5CL_WORK=/workspace/data/p5c_label)
# One labelling pass writes Tc20s AND P5-R(lam*) per row.  Both gate sets and (where the 4.3 policy allows) both tables are built.  The R4 order picks the label:
#   Tc20s gates ZZTB_ALL_PASS                                                   -> ZZP5CL_LABEL tc20s
#   Tc20s fails ONLY by TB-6 rank-0 / RK0 (the P5-C trigger) and P5-R gates pass -> ZZP5CL_LABEL p5r   (A1 must be frozen and the PI told before any launch: MM R2/R3)
#   trigger fired and P5-R gates fail                                           -> ZZP5CL_LABEL p5t    (next lever P5-T; exit 3, no table)
#   any other Tc20s gate failure                                                -> ZZP5CL_FAIL gates   (an instrument/pipeline fault, not a label verdict)
# Resumable: the driver skips logs listed in done_<shard>.txt, so a re-run with the SAME P5CL_NSHARD continues.
# Prints ZZP5CL_DONE (+ P5C_LABEL_TABLE_SHA256=<hex> of the SELECTED label) or ZZP5CL_FAIL <stage>.  Writes nothing under /workspace/refe-plan-p7 and never reads it.
set -u

NSHARD="${P5CL_NSHARD:-16}"
SHIP="${P5CL_SHIP:-/workspace/data/p5c_ship}"          # where POD_SHIP_LIST.txt's files land (flat, plus dev_labels/ and p5r_eval/)
WORK="${P5CL_WORK:-/workspace/data/p5c_label}"
SRC_TREE=/workspace/refe-plan-vdac                      # NEVER /workspace/refe-plan-p7
BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7
DBROOT=/workspace/data/navtrain_dbs
MAPS=/workspace/data/nuplan-maps/nuplan-maps-v1.0
PY=/workspace/venv-teacher/bin/python
EXPECT_ROWS=150603
EVAL_N=2400                                             # P5-R EVAL rows (1,200 per rank) that TB-1 must reproduce

# ---- pins (sha256) -----------------------------------------------------------------------------
PIN_BANK_R0=a5690f8186f940eccacf68f45c7b49bb788ad3190966c8fc99c40912b8384a24
PIN_BANK_R1=8b6ccd1b0b85e9f7af12e7d0897efbf8dd9b750f9b50b653d9ac070658f927aa
PIN_P5C_LABELS=77724709088044d19e132cde5eb8a6e549665af37a8a79bcfd245f1ad2a8f933          # p5c_labels.py (reference; the dual labeller copies its Tc20s lines)
PIN_DUAL=07e84b1cfd26d72c4fef7ea5557b07460e39f808a6bbba6a4d0df582ea7bd197                # p5c_dual_labels.py
PIN_P5R_LABELS=84e1cac32510fb277ad7d1a876059152dbded5076b4fd1928f5ade330b6c468d          # p5r_labels.py (PREREG_P5R solver)
PIN_P5R_RUN=58faf7a76cb1840f4a8ed76ee6eaa8bb1d8c835f20009cb7a2749990cc7feddb             # p5r_run.py (imported: Sim, sanity)
PIN_LAMBDA=83988ddac582a8a1cd37e9e11745e195abe8012bd093a4dfd72acbb608bdd467              # lambda_star.json (lam* 3.0)
PIN_EVAL_R0=a576bf2a80d6ca7f39217ff58c8e8baa3b179352a6dddbc81ac4aad97679d5fd             # p5r_eval/eval_r0.jsonl
PIN_EVAL_R1=1d2762b19deae138b0ef9906e75d3921ddf43f4c5a949dfb88dc00bfbf8c37b0             # p5r_eval/eval_r1.jsonl
PIN_VERDICT=7ae694a3bb754d8b7ddbd355f0b16928498fefd91cb4d0b71c996d9587a17e9e             # p5r_eval/verdict.json (P5-R's dev-box EVAL values: recorded as R-TOL-3 context only, not gated)
PIN_P5_TARGET=2b31af64d6f6bcff7ffc0a59d524528b48ad8fceb44033e9cc845b201326c8d0           # eval/p5_target_labels.py (git blob 294644df)
PIN_P4_TRACKER=193c024b107fd2b73fdb9541f805a1000a1c882660f7cd2684a5bc70335609d5          # eval/p4_tracker_comp.py (imported by p5_target_labels)
PIN_TABLE_MOD=9592b62cc9428d215d6c75cff36987a0b66d6274ba89fd119df9457d2c869646           # traj_label_table.py (wiring hold copy)
PIN_GATES=d600a83042c3e66087417f69a6c1b5f13cdada29ab4d624ddcf7a47b12c5a633               # p5c_table_gates.py (with --label p5r)
PIN_DEV_S0=7f8d58cc967117ef9d66ed98c6c04d7df044cc98cdc6ea889ae43199bccb5bd3              # dev_labels/labels_s0of2.jsonl (TB-0 control)
PIN_DEV_S1=2dea6e4326ea38feb7ec4ff06c3daaabff5632d700b948ebf3ace8b81083f2a1              # dev_labels/labels_s1of2.jsonl

fail() { echo "ZZP5CL_FAIL $1 ${2:-}"; exit 1; }
sha_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
pin() {  # pin <file> <expected-sha256> <stage>
  [ -f "$1" ] || fail "$3" "missing $1"
  local got; got="$(sha_of "$1")"
  [ "$got" = "$2" ] || fail "$3" "sha256 of $1 is $got, expected $2"
}

echo "P5CL start $(date -u +%FT%TZ) nshard=$NSHARD ship=$SHIP work=$WORK"
case "$SRC_TREE $SHIP $WORK" in *refe-plan-p7*) fail guard_p7_path ;; esac
case "$NSHARD" in ''|*[!0-9]*) fail bad_nshard ;; esac
[ "$NSHARD" -ge 1 ] || fail bad_nshard

# ---- 1. refuse while a trainer runs (exact argv fields, never pgrep -f / pkill -f) --------------
BUSY="$(ps -eo args= | awk '$1 ~ /(^|\/)python[0-9.]*$/ { for (i = 2; i <= NF; i++) if ($i ~ /(^|\/)(train\.py|scorer_finetune\.py)$/) { print $0; break } }')"
[ -z "$BUSY" ] || fail guard_trainer_running "$(echo "$BUSY" | head -1 | cut -c1-160)"

# ---- 2. pins -------------------------------------------------------------------------------------
[ -x "$PY" ] || fail venv_python "$PY not executable"
[ -d "$BANK" ] || fail bank_dir "$BANK"
[ -d "$DBROOT" ] || fail db_root "$DBROOT"
[ -d "$MAPS" ] || fail maps_root "$MAPS"
[ -d "$SRC_TREE/refe" ] && [ -d "$SRC_TREE/code" ] || fail src_tree "$SRC_TREE/refe or /code missing"
pin "$BANK/targets_rank0.jsonl" "$PIN_BANK_R0" pin_bank_rank0
pin "$BANK/targets_rank1.jsonl" "$PIN_BANK_R1" pin_bank_rank1
pin "$SHIP/p5c_labels.py"                 "$PIN_P5C_LABELS" pin_p5c_labels
pin "$SHIP/p5c_dual_labels.py"            "$PIN_DUAL"       pin_dual_labels
pin "$SHIP/p5r_labels.py"                 "$PIN_P5R_LABELS" pin_p5r_labels
pin "$SHIP/p5r_run.py"                    "$PIN_P5R_RUN"    pin_p5r_run
pin "$SHIP/lambda_star.json"              "$PIN_LAMBDA"     pin_lambda_star
pin "$SHIP/p5r_eval/eval_r0.jsonl"        "$PIN_EVAL_R0"    pin_eval_r0
pin "$SHIP/p5r_eval/eval_r1.jsonl"        "$PIN_EVAL_R1"    pin_eval_r1
pin "$SHIP/p5r_eval/verdict.json"         "$PIN_VERDICT"    pin_verdict
pin "$SHIP/p5_target_labels.py"           "$PIN_P5_TARGET"  pin_p5_target_labels
pin "$SHIP/p4_tracker_comp.py"            "$PIN_P4_TRACKER" pin_p4_tracker_comp
pin "$SHIP/traj_label_table.py"           "$PIN_TABLE_MOD"  pin_table_module
pin "$SHIP/p5c_table_gates.py"            "$PIN_GATES"      pin_gates
pin "$SHIP/dev_labels/labels_s0of2.jsonl" "$PIN_DEV_S0"     pin_dev_labels_s0
pin "$SHIP/dev_labels/labels_s1of2.jsonl" "$PIN_DEV_S1"     pin_dev_labels_s1
for f in TB1_control_1018.json repair_rank0.json repair_rank1.json; do [ -f "$SHIP/$f" ] || fail ship_missing "$SHIP/$f"; done

# ---- 3. package dir: refe/ + code/ from refe-plan-vdac, eval/ from the shipped, pinned files -----
PKG="$WORK/pkg"
mkdir -p "$WORK" || fail mkdir_work
rm -rf "${PKG:?}" 2>/dev/null          # PKG is this script's own re-creatable copy (literal path under $WORK)
mkdir -p "$PKG/eval" || fail mkdir_pkg
( cd "$SRC_TREE" && tar --exclude='__pycache__' -cf - refe code ) | ( cd "$PKG" && tar -xf - ) || fail pkg_copy
cp "$SHIP/p5_target_labels.py" "$SHIP/p4_tracker_comp.py" "$PKG/eval/" || fail pkg_eval
[ -f "$PKG/refe/navsim_dac.py" ] && [ -f "$PKG/refe/navtrain_scenarios.py" ] || fail pkg_refe_incomplete
pin "$PKG/eval/p5_target_labels.py" "$PIN_P5_TARGET" pin_pkg_p5_target_labels

# P5R_DIR: where p5c_dual_labels.py finds p5r_run.py / p5r_labels.py (it imports them, never copies).  THREADS (MM ruling R-C): the dual labeller runs the Tc20s part in THIS env and the
# P5-R part single-threaded: under threadpoolctl when the venv has it, otherwise by THIS env (all three of OPENBLAS/MKL/OMP_NUM_THREADS = 1, which the labeller then REQUIRES and
# records as thread_pin in its DONE record).  A changed BLAS thread count moves Tc20s at 1e-14 and P5-R at ~4e-4 (MEASURED 2026-11-12, see HANDOFF).
export P5C_PKG="$PKG" P5C_BANKDIR="$BANK" P5C_DBROOT="$DBROOT" NUPLAN_MAPS_ROOT="$MAPS" REFE_SIM_HZ=10 P5R_DIR="$SHIP" P5R_LAMBDA_JSON="$SHIP/lambda_star.json" P5R_BLAS_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ "$OMP_NUM_THREADS/$MKL_NUM_THREADS/$OPENBLAS_NUM_THREADS" = "1/1/1" ] || fail thread_env "OMP/MKL/OPENBLAS_NUM_THREADS must all be 1: $OMP_NUM_THREADS/$MKL_NUM_THREADS/$OPENBLAS_NUM_THREADS"
# the shards import the pod's venv-teacher stack: prove it BEFORE launching N processes.  threadpoolctl is OPTIONAL (R-C): the probe records whether it is importable; without it the
# labeller pins the P5-R part through the three env vars exported above.
"$PY" -c "import sys, nuplan, scipy; sys.path[:0] = ['$SHIP', '$PKG/eval', '$PKG/refe', '$PKG/code']; import navsim_dac, navtrain_scenarios, p5_target_labels, p5r_labels, p5r_run
try:
    import threadpoolctl; print('ZZPROBE thread_pin=threadpoolctl')
except ImportError:
    print('ZZPROBE thread_pin=env (threadpoolctl absent; OPENBLAS/MKL/OMP_NUM_THREADS=1 required and exported)')" \
  >"$WORK/import_probe.log" 2>&1 || fail venv_imports "see $WORK/import_probe.log"
grep -h -- ZZPROBE "$WORK/import_probe.log"

# ---- 4. N shards in parallel at nice 19 ----------------------------------------------------------
OUT="$WORK/out"
mkdir -p "$OUT" || fail mkdir_out
PIDS=""
i=0
while [ "$i" -lt "$NSHARD" ]; do
  nice -n 19 "$PY" "$SHIP/p5c_dual_labels.py" label --shard "$i" --nshard "$NSHARD" --out "$OUT" >"$OUT/shard_${i}.log" 2>&1 &
  PIDS="$PIDS $!"
  i=$((i + 1))
done
echo "P5CL launched $NSHARD shards: $PIDS"
BADRC=""
for p in $PIDS; do wait "$p" || BADRC="$BADRC $p"; done
[ -z "$BADRC" ] || fail shard_rc "nonzero exit for pids:$BADRC (see $OUT/shard_*.log; re-run resumes)"

# ---- 5. every shard log must carry its ZZLABEL_DONE; row count must be exact -----------------------
i=0
while [ "$i" -lt "$NSHARD" ]; do
  grep -q -- "ZZLABEL_DONE" "$OUT/shard_${i}.log" || fail shard_not_done "shard $i has no ZZLABEL_DONE"
  [ -f "$OUT/labels_s${i}of${NSHARD}.jsonl" ] || fail shard_no_output "shard $i"
  i=$((i + 1))
done
NLINES="$(cat "$OUT"/labels_s*of"${NSHARD}".jsonl | wc -l | tr -d ' ')"
[ "$NLINES" = "$EXPECT_ROWS" ] || fail row_count "label records $NLINES != $EXPECT_ROWS (a resumed shard may have duplicates: the gate would catch it, but stop here)"
for f in "$OUT"/labels_s*of*.jsonl; do case "$f" in *of"${NSHARD}".jsonl) ;; *) fail stale_other_nshard "$f" ;; esac; done

# ---- 5b. the P5-R repair sample (A1 point 4: r_k recomputed with P5-R labels): 300 rows per rank, sim8 of raw / Tc20s / P5-R -------------
REP="$WORK/repair"
mkdir -p "$REP" || fail mkdir_repair
for k in 0 1; do
  nice -n 19 "$PY" "$SHIP/p5c_dual_labels.py" repair --rank "$k" --n 300 --out "$REP/repair_rank${k}.json" >"$REP/repair_rank${k}.log" 2>&1 &
done
wait
for k in 0 1; do
  [ -s "$REP/repair_rank${k}.json" ] || fail repair_missing "rank $k (see $REP/repair_rank${k}.log)"
  grep -q -- "ZZREPAIR" "$REP/repair_rank${k}.log" || fail repair_not_done "rank $k"
done

# ---- 6. table gates TB-0..TB-7 + the 4.3 rank-1 policy: Tc20s (PREREG_P5C as frozen) and P5-R (A1), BOTH always ---------
GATES="$WORK/gates"
mkdir -p "$GATES/tc20s" "$GATES/p5r" || fail mkdir_gates
"$PY" "$SHIP/p5c_table_gates.py" --bank "$BANK" --labels "$OUT"/labels_s*of"${NSHARD}".jsonl \
  --control "$SHIP/dev_labels/labels_s0of2.jsonl" "$SHIP/dev_labels/labels_s1of2.jsonl" \
  --tb1 "$SHIP/TB1_control_1018.json" --repair0 "$SHIP/repair_rank0.json" --repair1 "$SHIP/repair_rank1.json" \
  --expect-rows "$EXPECT_ROWS" --tc20s-tol 1e-9 --code-file "$PKG/eval/p5_target_labels.py" "$SHIP/p5c_dual_labels.py" "$SHIP/p5c_table_gates.py" "$SHIP/traj_label_table.py" \
  --out-dir "$GATES/tc20s" 2>&1 | tee "$GATES/tc20s/gates_stdout.txt"
"$PY" "$SHIP/p5c_table_gates.py" --label p5r --bank "$BANK" --labels "$OUT"/labels_s*of"${NSHARD}".jsonl \
  --control "$SHIP/dev_labels/labels_s0of2.jsonl" "$SHIP/dev_labels/labels_s1of2.jsonl" \
  --eval-control "$SHIP/p5r_eval/eval_r0.jsonl" "$SHIP/p5r_eval/eval_r1.jsonl" --eval-n "$EVAL_N" --lambda-star "$SHIP/lambda_star.json" --p5r-verdict "$SHIP/p5r_eval/verdict.json" --tc20s-tol 1e-9 \
  --tb1 "$SHIP/TB1_control_1018.json" --repair0 "$REP/repair_rank0.json" --repair1 "$REP/repair_rank1.json" \
  --expect-rows "$EXPECT_ROWS" --code-file "$PKG/eval/p5_target_labels.py" "$SHIP/p5c_dual_labels.py" "$SHIP/p5r_labels.py" "$SHIP/p5r_run.py" "$SHIP/p5c_table_gates.py" "$SHIP/traj_label_table.py" \
  --out-dir "$GATES/p5r" 2>&1 | tee "$GATES/p5r/gates_stdout.txt"
TC_LINE="$(grep -- '^ZZTB_' "$GATES/tc20s/gates_stdout.txt" | tail -1)"
P5R_LINE="$(grep -- '^ZZTB_' "$GATES/p5r/gates_stdout.txt" | tail -1)"
[ -n "$TC_LINE" ] || fail gates_tc20s_no_verdict "no ZZTB line"
[ -n "$P5R_LINE" ] || fail gates_p5r_no_verdict "no ZZTB line"
echo "ZZP5CL_GATES tc20s: $TC_LINE"
echo "ZZP5CL_GATES p5r: $P5R_LINE"

# ---- 6b. the R4 order (MM_RULINGS_P5R.md; PREREG_P5R 10): which label does P5-C use -----------------------------------
# classify <label> <ZZTB line> sets CLASS:
#   pass       ZZTB_ALL_PASS
#   rank1only  the ONLY failing token is TB-6 (no STOP token) AND that label's gates.json shows TB-6 rank 0 PASS, rank 1 FAIL, rank_policy.decision == "rank0":
#              PREREG_P5C 4.2 "rank-1 FAIL => rank-0 only (4.3)": the label PASSES with policy rank0 (MM ruling R-D); build_table then runs for it
#   trigger    the only failing tokens are TB-6 / TB-6-rank0-STOP / RK0-STOP with a STOP token present (TB-6 rank-0 FAIL or RK0): the P5-C trigger
#   fault      anything else (an instrument / pipeline fault, not a label verdict)
classify() {
  local lab="$1" line="$2" tok other=0 stop=0 tb6=0
  CLASS=fault
  if [ "$line" = "ZZTB_ALL_PASS" ]; then CLASS=pass; return 0; fi
  for tok in ${line#ZZTB_FAIL}; do
    case "$tok" in
      TB-6) tb6=1 ;;
      TB-6-rank0-STOP|RK0-STOP) stop=1 ;;
      *) other=1 ;;
    esac
  done
  [ "$other" = 0 ] || return 0
  if [ "$stop" = 1 ]; then CLASS=trigger; return 0; fi
  if [ "$tb6" = 1 ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); t=d['gates']['TB-6']['detail']; sys.exit(0 if (t['rank0']['pass'] is True and t['rank1']['pass'] is False and d['rank_policy']['decision'] == 'rank0') else 1)" "$GATES/$lab/gates.json" 2>/dev/null; then
    CLASS=rank1only
  fi
  return 0
}
classify tc20s "$TC_LINE"; TC_CLASS="$CLASS"
classify p5r "$P5R_LINE"; P5R_CLASS="$CLASS"
echo "ZZP5CL_CLASS tc20s=$TC_CLASS p5r=$P5R_CLASS"
case "$TC_CLASS" in
  pass|rank1only) TRIGGER=0 ;;
  trigger)        TRIGGER=1 ;;
  *)              fail gates "Tc20s gates failed outside the P5-C trigger: $TC_LINE" ;;
esac
if [ "$TRIGGER" = 0 ]; then
  SEL=tc20s
elif [ "$P5R_CLASS" = pass ] || [ "$P5R_CLASS" = rank1only ]; then
  SEL=p5r
else
  SEL=p5t
fi
echo "ZZP5CL_LABEL $SEL"
if [ "$SEL" = p5t ]; then
  echo "ZZP5CL_STOP both_labels_fail (Tc20s: $TC_LINE | P5-R: $P5R_LINE): next lever is P5-T, PREREG_P5R 10"
  exit 3
fi
[ "$SEL" != p5r ] || echo "ZZP5CL_NOTE p5r selected by the R4 order: amendment A1 must be FROZEN and the PI told (MM R2) BEFORE any P5-C launch; the R-B tolerances (R-TOL-1/2/3) decided TB-0/TB-1"

# ---- 7. build + verify the table(s) with the rank policy each gate set decided ------------------------------------------
build_table() {   # build_table <label: tc20s|p5r> <label-name>
  lab="$1"; lname="$2"; gdir="$GATES/$1"
  policy="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['rank_policy']['decision'])" "$gdir/gates.json")" || fail "rank_policy_read_$lab"
  case "$policy" in
    all)   merged="$gdir/merged_labels_all.jsonl" ;;
    rank0) merged="$gdir/merged_labels_rank0only.jsonl" ;;
    *)     return 2 ;;      # STOP or UNDECIDED: no table for this label
  esac
  [ -f "$merged" ] || fail "merged_missing_$lab" "$merged"
  table="$WORK/P5C_label_table_${lab}_${policy}.json.gz"
  "$PY" "$SHIP/traj_label_table.py" build --bank-dir "$BANK" --labels "$merged" --out "$table" --require-coverage 1.0 --label-name "$lname" >"$WORK/table_build_$lab.log" 2>&1 || fail "table_build_$lab" "see $WORK/table_build_$lab.log"
  "$PY" "$SHIP/traj_label_table.py" verify --table "$table" --bank-dir "$BANK" >"$WORK/table_verify_$lab.json" 2>&1 || fail "table_verify_$lab" "see $WORK/table_verify_$lab.json"
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['rows']==$EXPECT_ROWS and d['covered']==$EXPECT_ROWS and not d['refused'], d" "$WORK/table_verify_$lab.json" || fail "table_verify_counts_$lab"
  tsha="$(sha_of "$table")"
  echo "$tsha  $table" > "$WORK/P5C_LABEL_TABLE_SHA256_${lab}.txt"
  echo "ZZP5CL_TABLE $lab rank_policy=$policy sha256=$tsha table=$table"
  case "$lab" in tc20s) TABLE_SHA_TC="$tsha" ;; p5r) TABLE_SHA_P5R="$tsha" ;; esac
  return 0
}
TABLE_SHA_TC=""; TABLE_SHA_P5R=""
case "$TC_CLASS" in pass|rank1only) build_table tc20s Tc20s || true ;; esac
case "$P5R_CLASS" in pass|rank1only) build_table p5r P5R || true ;; esac
if [ "$SEL" = tc20s ]; then SELSHA="$TABLE_SHA_TC"; else SELSHA="$TABLE_SHA_P5R"; fi
[ -n "$SELSHA" ] || fail "table_selected_$SEL" "the selected label has no table (rank policy STOP or UNDECIDED)"
echo "P5C_LABEL_TABLE_SHA256=$SELSHA  label=$SEL"
echo "$SELSHA  label=$SEL" > "$WORK/P5C_LABEL_TABLE_SHA256.txt"
echo "ZZP5CL_DONE"
exit 0
