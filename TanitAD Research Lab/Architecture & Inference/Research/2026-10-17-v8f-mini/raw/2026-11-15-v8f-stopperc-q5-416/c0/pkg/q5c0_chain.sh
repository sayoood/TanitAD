#!/usr/bin/env bash
# q5c0_chain.sh -- Q5 control C0 + the S2 E block (AMENDMENT_7_Q5_416.md section 4 C0, section 2.5 step 2).  Detached bash on the dev box; the waiter and the chain are this one script.
#   0 dev box: the eval npz (s0p_extract_eval.py, CPU, 2 threads) must have sha256 0c25fe72..fb39 (skipped if the file is already right)
#   1 wait: C:/Users/Admin/actsens_state/DONE.txt exists  (a STOP.txt there = report and exit rc 12, "ZZ Q5C0 FAIL ZZ actsens-stop"), the shared dev-box busy marker is absent,
#     box 2's D:\tanitad\BOX2_BUSY.txt is absent, box 2's GPU is under the idle bar and no python.exe of D:\tanitad runs.  Then claim the busy marker with noclobber.
#   2 ship s0p_lib.py + s0p_extract_train.py (S0-P chain folder, unedited) to D:\tanitad\stopperc\q5c0\s0p (sha256 verified); pre-flight; --dry-run for S1 and S2 (sample sha 843e4017 / 32ad96be)
#   3 per sample (S1 then S2), K=4 shards one at a time: disk check, detached job via launch.ps1, pull with sha256 verify, verify against the shard's own MANIFEST, delete box 2's copy (our own output)
#   4 assemble each sample into F:/TanitAD-scratch/q5_416/run/c0/<S>, release the busy marker, c0check -> c0/C0_RESULT.json (PASS/FAIL per check)
# Sentinels: the LAST line of $STATE/chain.log is "ZZ Q5C0 DONE ZZ" or "ZZ Q5C0 FAIL ZZ <why>" (also in $STATE/DONE.txt / STOP.txt).  No other line of the log carries "ZZ".
# rc: 2 state collision / standing STOP; 4 dry-run refused; 5 job or transfer failed; 6 disk-low; 7 pulled copy differs; 8 package incomplete / test hook in a real run; 9 eval or assemble failed;
#     10 C0 check FAILED (result file written); 11 timeout or unreadable probe; 12 predecessor stopped / box busy / wait timeout.
# Resume: after a STOP is inspected rerun with Q5C0_RESUME=1 (STOP renamed, never deleted); finished stages (done/*.done) are skipped; a box-2 job that runs is WAITED on, never started twice.
export PATH=/usr/bin:/mingw64/bin:/bin:$PATH
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
STATE="${Q5C0_STATE:-C:/Users/Admin/q5c0/state}"
BASE="${Q5C0_BASE:-F:/TanitAD-scratch/q5_416/run/c0}"
PYV="${Q5C0_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
QTOOL="$HERE/q5c0_tool.py"
S0PDIR="${Q5C0_S0P_DIR:-E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p}"
export S0P_PY="$PYV"
PRED="${Q5C0_PRED_STATE:-C:/Users/Admin/actsens_state}"
BUSY_DEV="${Q5C0_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${Q5C0_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"
EVAL_NPZ="${Q5C0_EVAL_NPZ:-$BASE/s0p_eval.npz}"; EVAL_SHA="0c25fe722f3359ee49ec6c31affa6761a4cd6f3ecb1e63d4ccc475ae2935fb39"
NT="${Q5C0_N_TOTAL:-16000}"; K=4
SHARD_BYTES="${Q5C0_SHARD_BYTES:-900000000}"; B2_EXTRA_BYTES=1000000000; F_FLOOR_MB="${Q5C0_F_FLOOR_MB:-20480}"
GPU_IDLE_MIB="${Q5C0_GPU_IDLE_MIB:-3500}"
mkdir -p "$STATE/done" || exit 2
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
final() {   # final DONE|FAIL <why> <rc>: the last line of the chain log and the sentinel file
  if [ "$1" = DONE ]; then echo "ZZ Q5C0 DONE ZZ" | tee -a "$STATE/chain.log" > "$STATE/DONE.txt"
  else echo "ZZ Q5C0 FAIL ZZ $2" | tee -a "$STATE/chain.log" > "$STATE/STOP.txt"; fi
}
refuse_early() { final FAIL "$1 rc=$2"; echo "REFUSED: $1" >&2; exit "$2"; }
if [ "${Q5C0_DRY:-0}" != 1 ]; then
  for h in Q5C0_N_TOTAL Q5C0_N_POS Q5C0_SHARD_BYTES Q5C0_S0P_STATE Q5C0_REF Q5C0_STATE Q5C0_BASE Q5C0_PRED_STATE Q5C0_BUSY_DEV Q5C0_EVAL_NPZ Q5C0_S0P_DIR Q5C0_F_FLOOR_MB Q5C0_GPU_IDLE_MIB; do
    [ -z "${!h:-}" ] || refuse_early "test-hook-in-real-run ($h is set without Q5C0_DRY=1)" 8
  done
fi
if [ -s "$STATE/STOP.txt" ] && [ "${Q5C0_RESUME:-0}" != 1 ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with Q5C0_RESUME=1" >&2; exit 2; fi
[ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
for f in s0p_lib.py s0p_extract_train.py s0p_extract_eval.py; do [ -f "$S0PDIR/$f" ] || refuse_early "package-incomplete ($f missing in $S0PDIR)" 8; done
[ -f "$QTOOL" ] || refuse_early "package-incomplete (q5c0_tool.py)" 8
export S0P_DRY="${Q5C0_DRY:-0}"
( source "$S0PDIR/chain/s0p_common.sh" ) > /dev/null 2> "$STATE/transport.err" || refuse_early "transport-not-fail-closed ($(tr '\n' ' ' < "$STATE/transport.err" | head -c 300))" 8
export S0P_SRC="$S0PDIR"
source "$S0PDIR/chain/s0p_common.sh"
c_free_mb() { df -m "$BASE" 2>/dev/null | awk 'NR==2{print $4}'; }    # the dev-box volume that receives the pulls and the assembled dirs (F:)
B2_Q="D:/tanitad/stopperc/q5c0"; B2_S0P="$B2_Q/s0p"; B2_OUT="$B2_Q/out"; B2_MD="$B2_Q/jobs"
mkdir -p "$BASE" "$BASE/_pull" "$ASIDE_DEV" "$STATE/jobs" || refuse_early "cannot-create-dirs" 8
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
OWN_MARK=""
release_box() { if [ -n "$OWN_MARK" ] && [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.q5c0.$(date +%Y%m%d_%H%M%S).txt"; OWN_MARK=""; fi; }
trap 'release_box' EXIT
stop() { local d=""; [ -s "$STATE/stage_detail.txt" ] && d=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 400)"; log "stop at stage $1: $2 (rc $3)"; release_box; final FAIL "$1 $2 rc=$3$d"; exit "$3"; }
fail() { printf '%s\n' "$STAGE: $3" | head -c 600 > "$STATE/stage_detail.txt"; stop "$STAGE" "$2" "$1"; }

# ---------------------------------------------------------------- 0 the eval npz (dev box, CPU)
STAGE=eval
if [ -e "$STATE/done/eval.done" ] && [ "$(sha256sum "$EVAL_NPZ" 2>/dev/null | cut -d' ' -f1)" = "$EVAL_SHA" ]; then log "skip eval (sha256 right)"; else
  if [ "$(sha256sum "$EVAL_NPZ" 2>/dev/null | cut -d' ' -f1)" != "$EVAL_SHA" ]; then
    [ "${Q5C0_DRY:-0}" != 1 ] || fail 8 test-hook-in-real-run "eval npz missing or wrong in a dry run: no real extraction in test mode"
    log "start eval extraction (s0p_extract_eval.py, CPU 2 threads, ~1.5 min MEASURED)"
    ( cd "$S0PDIR" && "$PYV" s0p_extract_eval.py --out "$EVAL_NPZ.part" ) > "$STATE/eval_extract.log" 2>&1 || fail 9 eval-failed "s0p_extract_eval.py rc!=0 (see $STATE/eval_extract.log)"
    mv "$EVAL_NPZ.part" "$EVAL_NPZ"
  fi
  [ "$(sha256sum "$EVAL_NPZ" | cut -d' ' -f1)" = "$EVAL_SHA" ] || fail 9 eval-sha-mismatch "eval npz sha256 != $EVAL_SHA"
  echo "eval $(date '+%F %T') $EVAL_SHA" > "$STATE/done/eval.done"; log "eval npz sha256 equals the banked one"
fi

# ---------------------------------------------------------------- 1 wait + claim
NEED_BOX=0
for s in S1 S2; do [ -e "$STATE/done/asm_$s.done" ] || NEED_BOX=1; done
[ -e "$STATE/done/pre.done" ] || NEED_BOX=1
b2_py_count() { local enc; enc="$("$PYV" -I "$QTOOL" psenc --log "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*tanitad*' } | Measure-Object).Count")"
  b2_ssh "powershell -NoProfile -EncodedCommand $enc" 2>/dev/null | grep -vE "$NOISE" | tr -d '\r' | tail -1; }
b2_gpu_mib() { b2_ssh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | grep -vE "$NOISE" | tr -d '\r ' | tail -1; }
box_free_now() {   # 0 if box 2 may be used now; echoes why not otherwise
  local g p
  [ ! -e "$BUSY_DEV" ] || { echo "dev busy marker: $(head -c 80 "$BUSY_DEV")"; return 1; }
  ! b2_exists "D:/tanitad/BOX2_BUSY.txt" || { echo "box 2 D:/tanitad/BOX2_BUSY.txt exists"; return 1; }
  g="$(b2_gpu_mib)"; case "$g" in ''|*[!0-9]*) echo "gpu probe unreadable ($g)"; return 1;; esac
  [ "$g" -lt "$GPU_IDLE_MIB" ] || { echo "gpu ${g} MiB >= $GPU_IDLE_MIB"; return 1; }
  p="$(b2_py_count)"; case "$p" in ''|*[!0-9]*) echo "python probe unreadable ($p)"; return 1;; esac
  [ "$p" -eq 0 ] || { echo "$p python.exe of D:\\tanitad running"; return 1; }
  return 0
}
if [ "$NEED_BOX" = 1 ]; then
  if [ "${Q5C0_RESUME:-0}" = 1 ] && [ -e "$BUSY_DEV" ] && head -c 10 "$BUSY_DEV" | grep -q '^q5c0 chain'; then
    opid="$(awk '{print $3}' "$BUSY_DEV")"; ps -p "$opid" > /dev/null 2>&1 || { mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.q5c0_stale.$(date +%Y%m%d_%H%M%S).txt"; log "moved aside our own stale busy marker (pid $opid gone)"; }
  fi
  n=0
  while :; do
    if [ -s "$PRED/STOP.txt" ]; then echo "$(date '+%F %T') actsens STOP: $(head -c 300 "$PRED/STOP.txt")" > "$STATE/wait_report.txt"; final FAIL "actsens-stop (predecessor $PRED/STOP.txt appeared; see $STATE/wait_report.txt) rc=12"; exit 12; fi
    if [ -s "$PRED/DONE.txt" ]; then why="$(box_free_now)" && break; else why="actsens DONE.txt absent"; fi
    [ "$n" -lt "${Q5C0_WAIT_POLLS:-2160}" ] || { final FAIL "wait-timeout (last: $why) rc=12"; exit 12; }
    n=$((n+1)); [ $((n % 10)) -eq 1 ] && log "waiting (poll $n): $why"; sleep "${Q5C0_WAIT_POLL_S:-120}"
  done
  if ! ( set -o noclobber; echo "q5c0 chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then final FAIL "busy-marker-taken ($BUSY_DEV) rc=12"; exit 12; fi
  OWN_MARK="$(cat "$BUSY_DEV")"; log "claimed $BUSY_DEV"
fi

# ---------------------------------------------------------------- 2 ship + pre-flight + dry runs
STAGE=pre
dry_sample() {  # dry_sample S1|S2 : dry-run job, check its log, pull the full sample.json
  local s="$1" gid r sd="$B2_OUT/$1"
  gid="$(date +%Y%m%d_%H%M%S)"
  "$PYV" -I "$QTOOL" mkjob --mode dry --sample "$s" --gid "$gid" --out "$STATE/jobs/dry_$s.cmd" || fail 8 package-incomplete "render dry job $s"
  put_verified "$STATE/jobs/dry_$s.cmd" "$B2_MD/dry_$s.cmd" || fail 5 job-or-transfer-failed "put dry_$s.cmd"
  b2_launch "$B2_MD/dry_$s.cmd" "$B2_MD/dry_$s.log" > /dev/null
  r="$(b2_wait "$B2_MD/${s}DRY_$gid.done" "$B2_MD/${s}DRY_$gid.fail" "${Q5C0_POLLS_DRY:-60}")"
  [ "$r" != TIMEOUT ] || fail 11 box2-timeout-or-unreadable "dry job $s did not finish ($B2_MD/dry_$s.log)"
  b2_cat "$B2_MD/dry_$s.log" > "$STATE/dry_$s.log"
  [ "$r" = DONE ] || fail 4 dry-run-refused "dry job $s failed: $(tail -c 300 "$STATE/dry_$s.log" | tr '\r\n' '  ')"
  "$PYV" -I "$QTOOL" check-dry --log "$STATE/dry_$s.log" --sample "$s" > "$STATE/sample_sha_$s.txt" 2> "$STATE/stage_detail.txt" || fail 4 dry-run-refused "dry-run asserts for $s: $(cat "$STATE/stage_detail.txt")"
  rm -f "$STATE/stage_detail.txt"
  pull_verified "$sd/sample.json" "$STATE/sample_$s.json" || fail 7 pull-sha-mismatch "dry-run sample.json of $s"
  say "dry run $s ok: sample sha256 $(cat "$STATE/sample_sha_$s.txt")"
}
pre() {
  local f
  b2_mkdir "$B2_Q" "$B2_S0P" "$B2_OUT" "$B2_MD"
  for f in s0p_lib.py s0p_extract_train.py; do put_verified "$S0PDIR/$f" "$B2_S0P/$f" || fail 5 job-or-transfer-failed "put $f: sha256 on box 2 differs"; done
  for f in "D:/tanitad/repo/stack/scripts/v8f_t1_eval.py" "D:/tanitad/venv/Scripts/python.exe" "D:/tanitad/runs/v8f_mini_full/join_train.npz" "D:/tanitad/cache_full/native_tracks.pt" "D:/tanitad/cache_full/train/clips" "D:/tanitad/cache_full/eval/manifest.jsonl" "D:/tanitad/launch.ps1"; do
    b2_exists "$f" || fail 4 preflight-refused "$f is missing on box 2"
  done
  # code of record drift probe (diagnosis only, never a gate): the sha256 of the two files whose bits decide E
  { for f in D:/tanitad/repo/stack/scripts/v8f_t1_eval.py D:/tanitad/repo/stack/tanitad/data/v8f_mini_data.py; do echo "$f $(b2_sha256 "$f")"; done; } > "$STATE/repo_probe.txt"
  dry_sample S1; dry_sample S2
}
if [ -e "$STATE/done/pre.done" ]; then log "skip pre (done)"; else log "start pre"; pre; echo "pre $(date '+%F %T')" > "$STATE/done/pre.done"; log "done pre"; fi

# ---------------------------------------------------------------- 3 shards
PULLD="$BASE/_pull"
shard() {   # shard S i
  local s="$1" i="$2" RUNF="$B2_MD/RUN_${1}SH$2.txt" SD="$B2_OUT/$1/shard_$2" gid r f free need cmb d bytes names ssha
  STAGE="${s}shard$i"
  [ -e "$STATE/done/${s}_shard_$i.done" ] && [ -d "$PULLD/$s/shard_$i" ] && { log "skip $s shard $i (pulled and verified)"; return 0; }
  log "start $s shard $i"; ssha="$(cat "$STATE/sample_sha_$s.txt")"
  rm -rf "${PULLD:?}/$s/.partial_$i"
  gid=""
  if b2_exists "$RUNF"; then gid="$(b2_cat "$RUNF" | head -n 1 | tr -d ' ')"; case "$gid" in ''|*[!0-9A-Za-z_]*) fail 11 box2-timeout-or-unreadable "unreadable $RUNF";; esac; say "$s shard $i: a job (gid $gid) was launched earlier; waiting for its marker"; fi
  if [ -z "$gid" ]; then
    free="$(b2_free_bytes)"; case "$free" in ''|*[!0-9]*) fail 11 box2-timeout-or-unreadable "box 2 D: free unreadable ($free)";; esac
    need=$((SHARD_BYTES + B2_EXTRA_BYTES)); [ "$free" -ge "$need" ] || fail 6 disk-low "box 2 D: free $free < $need (shard + 1 GB)"
    cmb="$(c_free_mb)"; case "$cmb" in ''|*[!0-9]*) fail 11 box2-timeout-or-unreadable "dev free unreadable";; esac
    [ "$cmb" -ge $((F_FLOOR_MB + SHARD_BYTES / 1048576)) ] || fail 6 disk-low "dev free $cmb MB < $((F_FLOOR_MB + SHARD_BYTES / 1048576)) MB"
    gid="$(date +%Y%m%d_%H%M%S)"
    "$PYV" -I "$QTOOL" mkjob --mode shard --sample "$s" --i "$i" --k "$K" --gid "$gid" --out "$STATE/jobs/${s}sh$i.$gid.cmd" || fail 8 package-incomplete "render shard job"
    printf '%s\n' "$gid" > "$STATE/jobs/run_${s}sh$i.txt"
    put_verified "$STATE/jobs/run_${s}sh$i.txt" "$RUNF" || fail 5 job-or-transfer-failed "put $RUNF"
    put_verified "$STATE/jobs/${s}sh$i.$gid.cmd" "$B2_MD/${s}sh$i.cmd" || fail 5 job-or-transfer-failed "put ${s}sh$i.cmd"
    b2_launch "$B2_MD/${s}sh$i.cmd" "$B2_MD/${s}sh$i.log" > /dev/null; say "launched $s shard $i (gid $gid)"
  fi
  r="$(b2_wait "$B2_MD/${s}SH${i}_$gid.done" "$B2_MD/${s}SH${i}_$gid.fail" "${Q5C0_POLLS_SHARD:-120}")"; say "$s shard $i marker=$r"
  [ "$r" != TIMEOUT ] || fail 11 box2-timeout-or-unreadable "$s shard $i did not finish ($B2_MD/${s}sh$i.log); rerun with Q5C0_RESUME=1 to keep waiting"
  if [ "$r" != DONE ]; then b2_remove_file "$RUNF"; fail 5 job-or-transfer-failed "$s shard $i job failed: $(b2_cat "$B2_MD/${s}SH${i}_$gid.fail" | head -c 200 | tr '\n' ' ') (log $B2_MD/${s}sh$i.log)"; fi
  b2_cat "$B2_MD/${s}sh$i.log" > "$STATE/${s}sh$i.box2.log"                 # box 2's own stdout of the job (elapsed_s etc.)
  d="$PULLD/$s/.partial_$i"; mkdir -p "$d"
  names="E.npy v40.npy v160.npy mean.npy y.npy clip.npy ws.npy eid.npy meta.json sample.json MANIFEST.sha256"
  for f in $names; do pull_verified "$SD/$f" "$d/$f" || fail 7 pull-sha-mismatch "$s shard $i file $f differs from box 2's (3 attempts)"; done
  "$PYV" -I "$QTOOL" verify-shard --dir "$d" --i "$i" --k "$K" --sample "$s" --sample-sha "$ssha" > "$STATE/${s}_shard_$i.verify.json" 2> "$STATE/stage_detail.txt" || fail 7 pull-sha-mismatch "$s shard $i fails its manifest/meta check: $(cat "$STATE/stage_detail.txt")"
  rm -f "$STATE/stage_detail.txt"
  mv "$d" "$PULLD/$s/shard_$i" || fail 7 pull-sha-mismatch "cannot move the verified shard into place"
  echo "$s shard$i $(date '+%F %T') $(cat "$STATE/${s}_shard_$i.verify.json")" > "$STATE/done/${s}_shard_$i.done"
  bytes="$(b2_dir_bytes "$SD")"; case "$bytes" in ''|*[!0-9]*) bytes=unknown;; esac
  b2_remove_dir "$SD"; b2_exists "$SD" && fail 5 job-or-transfer-failed "box-2 copy $SD still exists after the delete"
  b2_remove_file "$RUNF"
  echo "$(date '+%F %T') deleted box-2 $SD ($bytes bytes): our own output, verified pulled copy $PULLD/$s/shard_$i, re-creatable from the seed" >> "$STATE/deletions.log"
  log "$s shard $i pulled, verified; box-2 copy deleted ($bytes bytes)"
}
for s in S1 S2; do
  [ -e "$STATE/done/asm_$s.done" ] && continue
  mkdir -p "$PULLD/$s"
  for ((i=0; i<K; i++)); do shard "$s" "$i"; done
  STAGE="asm_$s"
  "$PYV" -I "$QTOOL" assemble --pull "$PULLD/$s" --sample "$s" --k "$K" --out "$BASE/$s" --sample-json "$STATE/sample_$s.json" > "$STATE/asm_$s.json" 2> "$STATE/stage_detail.txt" || fail 9 assemble-failed "$s: $(cat "$STATE/stage_detail.txt")"
  rm -f "$STATE/stage_detail.txt"; echo "asm $s $(date '+%F %T')" > "$STATE/done/asm_$s.done"; log "assembled $BASE/$s"
done

# ---------------------------------------------------------------- 4 release box 2 (cleanup of our own leftovers there), the C0 check
if [ -n "$OWN_MARK" ]; then
  for s in S1 S2; do b2_exists "$B2_OUT/$s/sample.json" && { b2_remove_file "$B2_OUT/$s/sample.json"; echo "$(date '+%F %T') deleted box-2 $B2_OUT/$s/sample.json (dry-run copy; pulled to $STATE/sample_$s.json)" >> "$STATE/deletions.log"; }; b2_ssh "rmdir $(w "$B2_OUT/$s")" > /dev/null 2>&1; done
  b2_ssh "rmdir $(w "$B2_OUT")" > /dev/null 2>&1
fi
release_box
STAGE=c0check
"$PYV" -I "$QTOOL" c0check --state "$STATE" --base "$BASE" --out "$BASE/C0_RESULT.json" > "$STATE/c0check.out" 2> "$STATE/stage_detail.txt"; rc=$?
if [ "$rc" -ne 0 ]; then
  fail 10 c0-check-failed "C0 $(head -c 20 "$STATE/c0check.out") (rc $rc); $BASE/C0_RESULT.json holds the per-check result; $(head -c 200 "$STATE/stage_detail.txt")"
fi
rm -f "$STATE/stage_detail.txt"
log "C0 PASS: $BASE/C0_RESULT.json sha256 $(sha256sum "$BASE/C0_RESULT.json" | cut -d' ' -f1)"
# own temp, C0 passed: the verified shard pulls are inside the assembled dirs (concatenation asserted equal to the full sample) and re-creatable from the seed on box 2; delete by literal path, log size.
# (on FAIL they are kept for diagnosis)
pbytes="$(du -sb "${PULLD:?}" 2>/dev/null | cut -f1)"
if rm -rf "${PULLD:?}"; then echo "$(date '+%F %T') deleted $PULLD (${pbytes:-unknown} bytes): own shard pulls, C0 PASSED, contents are in $BASE/S1 and $BASE/S2" >> "$STATE/deletions.log"; else echo "$(date '+%F %T') DELETE REFUSED $PULLD (${pbytes:-unknown} bytes): left for the MM" >> "$STATE/deletions.log"; fi
final DONE "" 0
exit 0
