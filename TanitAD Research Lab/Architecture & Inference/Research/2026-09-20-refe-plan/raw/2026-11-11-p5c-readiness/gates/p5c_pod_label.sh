#!/usr/bin/env bash
# P5-C pod label runner (PREREG_P5C.md 4.1 / 4.2).  FAIL-CLOSED: any unmet precondition prints "ZZP5CL_FAIL <stage>" and exits non-zero.
# Run ON THE POD, AFTER P7's last step (the pod's CPUs and DBs are free then).  WRITTEN AND `bash -n` CHECKED ONLY; NEVER RUN BY ITS AUTHOR.
# Usage:  bash p5c_pod_label.sh            (env: P5CL_NSHARD=16  P5CL_SHIP=/workspace/data/p5c_ship  P5CL_WORK=/workspace/data/p5c_label)
# Resumable: the driver skips logs listed in done_<shard>.txt, so a re-run with the SAME P5CL_NSHARD continues.
# Prints ZZP5CL_DONE (+ P5C_LABEL_TABLE_SHA256=<hex>) or ZZP5CL_FAIL <stage>.  Writes nothing under /workspace/refe-plan-p7 and never reads it.
set -u

NSHARD="${P5CL_NSHARD:-16}"
SHIP="${P5CL_SHIP:-/workspace/data/p5c_ship}"          # where POD_SHIP_LIST.txt's files land (flat, plus dev_labels/)
WORK="${P5CL_WORK:-/workspace/data/p5c_label}"
SRC_TREE=/workspace/refe-plan-vdac                      # NEVER /workspace/refe-plan-p7
BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7
DBROOT=/workspace/data/navtrain_dbs
MAPS=/workspace/data/nuplan-maps/nuplan-maps-v1.0
PY=/workspace/venv-teacher/bin/python
EXPECT_ROWS=150603

# ---- pins (sha256) -----------------------------------------------------------------------------
PIN_BANK_R0=a5690f8186f940eccacf68f45c7b49bb788ad3190966c8fc99c40912b8384a24
PIN_BANK_R1=8b6ccd1b0b85e9f7af12e7d0897efbf8dd9b750f9b50b653d9ac070658f927aa
PIN_P5C_LABELS=77724709088044d19e132cde5eb8a6e549665af37a8a79bcfd245f1ad2a8f933          # p5c_labels.py
PIN_P5_TARGET=2b31af64d6f6bcff7ffc0a59d524528b48ad8fceb44033e9cc845b201326c8d0           # eval/p5_target_labels.py (git blob 294644df)
PIN_P4_TRACKER=193c024b107fd2b73fdb9541f805a1000a1c882660f7cd2684a5bc70335609d5          # eval/p4_tracker_comp.py (imported by p5_target_labels)
PIN_TABLE_MOD=9592b62cc9428d215d6c75cff36987a0b66d6274ba89fd119df9457d2c869646           # traj_label_table.py (wiring hold copy)
PIN_GATES=c1260394b783923072172245fab7c2afe886ea13e2ce824fce61356c1ee0181d               # p5c_table_gates.py
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

export P5C_PKG="$PKG" P5C_BANKDIR="$BANK" P5C_DBROOT="$DBROOT" NUPLAN_MAPS_ROOT="$MAPS" REFE_SIM_HZ=10
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# the shards import the pod's venv-teacher stack: prove it BEFORE launching 16 processes
"$PY" -c "import sys, nuplan, scipy; sys.path[:0] = ['$PKG/eval', '$PKG/refe', '$PKG/code']; import navsim_dac, navtrain_scenarios, p5_target_labels" \
  >"$WORK/import_probe.log" 2>&1 || fail venv_imports "see $WORK/import_probe.log"

# ---- 4. N shards in parallel at nice 19 ----------------------------------------------------------
OUT="$WORK/out"
mkdir -p "$OUT" || fail mkdir_out
PIDS=""
i=0
while [ "$i" -lt "$NSHARD" ]; do
  nice -n 19 "$PY" "$SHIP/p5c_labels.py" label --shard "$i" --nshard "$NSHARD" --out "$OUT" >"$OUT/shard_${i}.log" 2>&1 &
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
# any leftover shard files from another N would be double-counted by the glob below
for f in "$OUT"/labels_s*of*.jsonl; do case "$f" in *of"${NSHARD}".jsonl) ;; *) fail stale_other_nshard "$f" ;; esac; done

# ---- 6. table gates TB-0..TB-7 + the 4.3 rank-1 policy ---------------------------------------------
GATES="$WORK/gates"
mkdir -p "$GATES" || fail mkdir_gates
"$PY" "$SHIP/p5c_table_gates.py" --bank "$BANK" --labels "$OUT"/labels_s*of"${NSHARD}".jsonl \
  --control "$SHIP/dev_labels/labels_s0of2.jsonl" "$SHIP/dev_labels/labels_s1of2.jsonl" \
  --tb1 "$SHIP/TB1_control_1018.json" --repair0 "$SHIP/repair_rank0.json" --repair1 "$SHIP/repair_rank1.json" \
  --expect-rows "$EXPECT_ROWS" --code-file "$PKG/eval/p5_target_labels.py" "$SHIP/p5c_labels.py" "$SHIP/p5c_table_gates.py" "$SHIP/traj_label_table.py" \
  --out-dir "$GATES" 2>&1 | tee "$GATES/gates_stdout.txt"
grep -q -- "^ZZTB_ALL_PASS$" "$GATES/gates_stdout.txt" || fail gates "$(grep -- '^ZZTB_FAIL' "$GATES/gates_stdout.txt" | head -1)"

# ---- 7. build + verify the table with the rank policy the gates decided ----------------------------
POLICY="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['rank_policy']['decision'])" "$GATES/gates.json")"
case "$POLICY" in
  all)   MERGED="$GATES/merged_labels_all.jsonl" ;;
  rank0) MERGED="$GATES/merged_labels_rank0only.jsonl" ;;
  *)     fail "rank_policy_$POLICY" "no table is built for this decision (STOP or UNDECIDED)" ;;
esac
[ -f "$MERGED" ] || fail merged_missing "$MERGED"
TABLE="$WORK/P5C_label_table_${POLICY}.json.gz"
"$PY" "$SHIP/traj_label_table.py" build --bank-dir "$BANK" --labels "$MERGED" --out "$TABLE" --require-coverage 1.0 --label-name Tc20s >"$WORK/table_build.log" 2>&1 || fail table_build "see $WORK/table_build.log"
"$PY" "$SHIP/traj_label_table.py" verify --table "$TABLE" --bank-dir "$BANK" >"$WORK/table_verify.json" 2>&1 || fail table_verify "see $WORK/table_verify.json"
"$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['rows']==$EXPECT_ROWS and d['covered']==$EXPECT_ROWS and not d['refused'], d" "$WORK/table_verify.json" || fail table_verify_counts
TSHA="$(sha_of "$TABLE")"
echo "P5C_LABEL_TABLE_SHA256=$TSHA  rank_policy=$POLICY  table=$TABLE"
echo "$TSHA  $TABLE" > "$WORK/P5C_LABEL_TABLE_SHA256.txt"
echo "ZZP5CL_DONE"
exit 0
