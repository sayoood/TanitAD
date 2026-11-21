#!/usr/bin/env bash
# RJ5 chain C on the POD (PREREG_P5C_A3 section 7, the MM's scope addition 2026-11-15): rolls the 13,949 rank-0 keys over 192 logs of chain C with the RJ5 teacher override, then the pod R0 control on the 36 bulk control logs.
# Run ON THE POD, ONLY AFTER P7's last step.  WRITTEN AND `bash -n` CHECKED, TESTED IN A STUB SANDBOX ONLY; NEVER RUN BY ITS AUTHOR.
#   bash rj5c_pod.sh [--dry-run]          (any other argument is REFUSED; in particular `--fetch`: the chain's delete loop removes only DBs it fetched, so without --fetch no pod DB can ever be deleted)
# Ship dir (A3 section 7; nothing under refe-plan-p7):  $RJ5C_SHIP (default /workspace/data/rj5c_ship) = rj5_roll.py  rj5_chain.py  patch/build_teacher_rollouts.py  logs_fetch_C.txt  control36_logs.txt
# REFUSES (exit 1, `ZZRJ5C_REFUSE <why>`): --fetch / any argument; a trainer on the pod (exact argv fields, never pgrep -f); a shipped file whose sha256 is not A3 section 7's pin (the 4 ship files, the pod bank rank-0 file, the control-log list);
#   a missing teacher_env.sh / venv-teacher python / DB dir; a DB dir that is not the pod's navtrain_dbs; a non-empty scratch --db-dir; fewer cgroup CPUs than --procs (read from the cgroup quota, never nproc); a missing DB for any chain-C or control log.
# Output: $RJ5C_OUT (default /workspace/data/rj5c_run): run_fetch_C/{rows,batches,...} (RJ5 + the chain's own every-20th R0), run_control36/{rows,...} (R0 on the 36 bulk control logs).  Resumable: re-running the same command resumes.
# Prints ZZRJ5C_DONE.  The pod R0 control reads (A3 section 7): [H] pod R0 == bank bit-for-bit where the dev box split by 1 ULP; rows whose bank teacher_device is None can never reproduce on a CPU.
set -u

SHIP="${RJ5C_SHIP:-/workspace/data/rj5c_ship}"
OUT="${RJ5C_OUT:-/workspace/data/rj5c_run}"
PKGSRC=/workspace/refe-plan-vdac                 # NEVER /workspace/refe-plan-p7
PKG="${RJ5C_PKG:-/workspace/data/rj5c_pkg}"      # the copy of refe-plan-vdac the roller imports
BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7
DBS=/workspace/data/navtrain_dbs
TENV=/workspace/teacher_env.sh
PY=/workspace/venv-teacher/bin/python
PROCS=14
BATCH_LOGS=48
CONTROL_EVERY=20

# ---- pins (sha256; A3 section 7 and section 2) ---------------------------------------------------
PIN_ROLL=e43733e2af4e34e10b5df3b4071eb0b74c03402ecf8de6e6b76dc83ea9d69ffe        # rj5_roll.py (blob 7c2fe686)
PIN_CHAIN=a2702a01c4ca887900e1321348c0f1fc25c69e5e001bd4d52f183c10ff12a801       # rj5_chain.py (blob e6064314)
PIN_PATCH=4c3d58697be8569a0bc361b109c425542d6495d0cb2d7adae77ca8da08aed251       # patch/build_teacher_rollouts.py (blob 791fa8b1; empty overrides => bit-identical to 049b0290's build)
PIN_LOGS_C=07d3ea80226ad54dc441e7922d53f80289c49a9795eecc695d5f72e29c1a1db1      # logs_fetch_C.txt (blob 3a6beb44)
PIN_BANK_R0=a5690f8186f940eccacf68f45c7b49bb788ad3190966c8fc99c40912b8384a24     # the pod's own train_grow_cmd_3c_p7/targets_rank0.jsonl
PIN_CTL36=b4e4f80fda1d45e5581428941a0713ca53e980215b0220274b96fa4bda5095ca                                                          # control36_logs.txt: the 36 bulk control logs of rj5path/control_by_log_bulk.json (sha256 MEASURED from control_by_log_bulk.json: 36 entries = 35 distinct logs)

refuse() { echo "ZZRJ5C_REFUSE $*"; exit 1; }
sha_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
pin() { [ -f "$1" ] || refuse "missing $1 ($3)"; [ "$(sha_of "$1")" = "$2" ] || refuse "sha256 of $1 is $(sha_of "$1"), A3 section 7 pins $2 ($3)"; }
cpu_quota() {
  local cg="${RJ5C_CGROUP_ROOT:-/sys/fs/cgroup}" q="" per=""
  { read -r q per < "$cg/cpu.max"; } 2> /dev/null
  case "$q" in ''|max|*[!0-9]*) ;; *) case "$per" in ''|*[!0-9]*|0) ;; *) echo $(( q / per > 0 ? q / per : 1 )); return 0 ;; esac ;; esac
  q=""; per=""
  { read -r q < "$cg/cpu/cpu.cfs_quota_us"; read -r per < "$cg/cpu/cpu.cfs_period_us"; } 2> /dev/null
  case "$q" in ''|*[!0-9]*|0) return 1 ;; esac
  case "$per" in ''|*[!0-9]*|0) return 1 ;; esac
  echo $(( q / per > 0 ? q / per : 1 ))
}

DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --fetch|--fetch=*|--fetch-script|--fetch-script=*) refuse "$a: the pod chain never fetches (A3 section 7: without --fetch no pod DB can be deleted)" ;;
    *) refuse "unknown argument '$a' (usage: rj5c_pod.sh [--dry-run])" ;;
  esac
done
case "$SHIP $OUT $PKG" in *refe-plan-p7*) refuse "a path names refe-plan-p7" ;; esac

# ---- 1. no trainer on the pod (exact argv fields, never pgrep -f / pkill -f) --------------------
PSCMD="${RJ5C_PS:-ps -eo args=}"
BUSY="$($PSCMD 2> /dev/null | awk '$1 ~ /(^|\/)python[0-9.]*$/ { for (i = 2; i <= NF; i++) if ($i ~ /(^|\/)(train\.py|scorer_finetune\.py)$/) { print $0; break } }')"
[ -z "$BUSY" ] || refuse "a trainer is running on the pod: $(echo "$BUSY" | head -1 | cut -c1-160)"

# ---- 2. pins -------------------------------------------------------------------------------------
pin "$SHIP/rj5_roll.py" "$PIN_ROLL" rj5_roll.py
pin "$SHIP/rj5_chain.py" "$PIN_CHAIN" rj5_chain.py
pin "$SHIP/patch/build_teacher_rollouts.py" "$PIN_PATCH" "TF-N2 patch"
pin "$SHIP/logs_fetch_C.txt" "$PIN_LOGS_C" logs_fetch_C.txt
pin "$BANK/targets_rank0.jsonl" "$PIN_BANK_R0" "the pod's own bank"

pin "$SHIP/control36_logs.txt" "$PIN_CTL36" control36_logs.txt
[ -f "$TENV" ] || refuse "missing $TENV"
[ -x "$PY" ] || refuse "$PY is not executable"
[ -d "$DBS" ] || refuse "missing the pod's DB dir $DBS"
[ -d "$PKGSRC/refe" ] && [ -d "$PKGSRC/code" ] || refuse "$PKGSRC/refe or /code missing"
NLOGS="$(wc -l < "$SHIP/logs_fetch_C.txt" | tr -d ' ')"
[ "$NLOGS" = 192 ] || refuse "logs_fetch_C.txt lists $NLOGS logs, A3 section 7 says 192"
NCTL="$(wc -l < "$SHIP/control36_logs.txt" | tr -d ' ')"
[ "$NCTL" = 35 ] || refuse "control36_logs.txt lists $NCTL logs: the 36 control entries of control_by_log_bulk.json are 35 distinct logs (one log has two entries)"
while read -r lg; do [ -z "$lg" ] || [ -f "$DBS/$lg.db" ] || refuse "no pod DB for log $lg ($DBS/$lg.db)"; done < "$SHIP/logs_fetch_C.txt"
while read -r lg; do [ -z "$lg" ] || [ -f "$DBS/$lg.db" ] || refuse "no pod DB for control log $lg"; done < "$SHIP/control36_logs.txt"
Q="$(cpu_quota)" || refuse "no readable cgroup CPU quota (never nproc)"
[ "$Q" -ge "$PROCS" ] || refuse "the cgroup grants $Q CPUs, --procs $PROCS needs at least that many"

# ---- 3. env (A3 section 7), the package copy, the scratch --db-dir ------------------------------
# shellcheck disable=SC1090
source "$TENV"
[ -n "${DZ_ROOT:-}" ] || refuse "teacher_env.sh did not set DZ_ROOT"
export DRIVERL_EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 REFE_SIM_HZ=10
unset DRIVERL_EVAL_ROUTE_LANE_RANK
mkdir -p "$OUT" || refuse "cannot create $OUT"
if [ "$DRY" = 1 ]; then echo "DRY: copy $PKGSRC/{refe,code,eval} to $PKG"; else
  rm -rf "${PKG:?}" 2> /dev/null            # this script's own re-creatable copy (literal path)
  mkdir -p "$PKG" && ( cd "$PKGSRC" && tar --exclude='__pycache__' -cf - refe code eval 2> /dev/null ) | ( cd "$PKG" && tar -xf - ) || refuse "package copy failed"
fi
export PYTHONPATH="$SHIP/patch:$PKG/refe:$PKG/code:$PKG/eval:$DZ_ROOT/nuplan-devkit"
EMPTY="$OUT/empty_db_dir"
[ "$DRY" = 1 ] || mkdir -p "$EMPTY"
if [ "$DRY" != 1 ] && [ -n "$(ls -A "$EMPTY" 2> /dev/null)" ]; then refuse "the scratch --db-dir $EMPTY is not empty"; fi

run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi; "$@"; }

# ---- 4. chain C: RJ5, the chain's own every-20th-log R0 control, NO --fetch ----------------------
echo "ZZRJ5C_START chain C: $NLOGS logs, procs $PROCS, quota $Q, PYTHONPATH $PYTHONPATH"
run "$PY" "$SHIP/rj5_chain.py" --bank "$BANK/targets_rank0.jsonl" --logs-file "$SHIP/logs_fetch_C.txt" --work "$OUT/run_fetch_C" --db-dir "$EMPTY" --held-db-dir "$DBS" \
  --procs "$PROCS" --batch-logs "$BATCH_LOGS" --arms RJ5 --control-every "$CONTROL_EVERY" > "$OUT/chain_C.log" 2>&1 || refuse "chain C exited non-zero (see $OUT/chain_C.log; re-running resumes)"
# ---- 5. the pod R0 control: R0 re-rolls of the 36 bulk control logs (the identity control against the bank) ----------------------
echo "ZZRJ5C_START pod R0 control on $NCTL logs"
run "$PY" "$SHIP/rj5_chain.py" --bank "$BANK/targets_rank0.jsonl" --logs-file "$SHIP/control36_logs.txt" --work "$OUT/run_control36" --db-dir "$EMPTY" --held-db-dir "$DBS" \
  --procs "$PROCS" --batch-logs "$BATCH_LOGS" --arms R0 --control-every 0 > "$OUT/control36.log" 2>&1 || refuse "the R0 control exited non-zero (see $OUT/control36.log; re-running resumes)"
echo "ZZRJ5C_DONE rows: $OUT/run_fetch_C/rows (RJ5 + R0 every ${CONTROL_EVERY}th log), $OUT/run_control36/rows (R0, 36 logs)"
exit 0
