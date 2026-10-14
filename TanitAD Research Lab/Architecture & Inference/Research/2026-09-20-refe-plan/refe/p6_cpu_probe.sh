#!/bin/bash
# P6 CPU PROBE (eval/P6_ON_11VCPU.md, PI 2026-10-14 "find a solution with our 11 cpu pod"): does the online PDM labeller keep up on THIS pod's cgroup cores?
# CPU only.  No GPU, no HF_TOKEN, no network.  Run it ONCE in a NON-TRAINING WINDOW (after S1, before the P2 replicate).  <= ~60 min, bounded by P6CPU_BUDGET_MIN.
#
#   P6CPU_PKG=/workspace/refe-plan-p6 nohup setsid bash /workspace/refe-plan-p6/refe/p6_cpu_probe.sh > /workspace/data/refe_p6cpu/stdout.log 2>&1 < /dev/null &
#   then read:  /workspace/data/refe_p6cpu/probe.log  (markers ZZP6CPU_*)   and   /workspace/data/refe_p6cpu/verdict.json
#   the ONE verdict line:  ZZP6CPU_FIT | ZZP6CPU_MARGINAL | ZZP6CPU_SHORT   (ZZP6CPU_VOID = the measurement is invalid: an identity check failed or a stage produced nothing;
#                          ZZP6CPU_REFUSED = it did not run because the pod is not quiet).
#
# What it does, in order (each stage is skipped, and logged, when the time budget is spent):
#   0  REFUSES if any chain lock is held ( flock -n 9 ) 9<> file  -- NEVER a read-only fd: it is EBADF on NFSv4, test_flock_rw.py --, if the GPU is busy or cannot be asked,
#      if another probe runs, or if the cgroup already burns more than P6CPU_MAX_BG cores (a loaded pod makes every number below void).
#   1  environment: cgroup cores (never nproc), CPU model, memory limit.
#   2  ONE pass over the training bank: the EXACT D9 workload (distinct tokens, lane-twin keys) and a sample of whole logs (their real tokens per log).
#   3  D9 on the sample into a SCRATCH dir (never refe_navtrain/metric_cache): CPU-s per token, from the cgroup's own counter.
#   4  for each label-worker count W in P6CPU_LWS (7 8 9 10): (a) diag_p6_labeller_identity.py == the gate's own instrument (identity MUST pass, else the number is void),
#      (b) the same pool instrumented: CPU-s/set, RSS, threads, cgroup throttling, (c) service latency of a 4-set micro-batch, (d) the trainer's clock (GPU stall).
#   5  A/B at the chain's default W: the chain's OMP_NUM_THREADS=4 against 1 BLAS thread (is the pool oversubscribing the quota?).
#   6  disk: sizes of what exists, and a write test of P6's NEW bytes (P6CPU_DD_GB) that stops at the first ENOSPC/EDQUOT and reports the GB proven.
#   7  the verdict line, against the gate's own bars (6.735 sets/s = 1.5 x 4.49) and the need (4.49).
# It never changes a gate value, never writes outside $DATA/refe_p6cpu and its dd file, never uses pkill, and removes its scratch (P6CPU_KEEP=1 keeps it).
# Tests: refe/test_p6_cpu_probe.py (ZZTEST_PASS n).  Every child is launched with fd 200 closed (the probe holds its flock on 200).
set -u
# the probe needs no token: whatever the pod's shell exported is dropped before any child exists
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"; DATA=$WORK/data
PKG="${P6CPU_PKG:-$(dirname "$HERE")}"
PR="${P6CPU_DIR:-$DATA/refe_p6cpu}"; SCR="$PR/scratch"; L="$PR/probe.log"
BUDGET_S=$(( ${P6CPU_BUDGET_MIN:-60} * 60 ))
BANK="${P6CPU_BANK:-$DATA/refe_navtrain/train_grow}"; DBS="${P6CPU_DBS:-$DATA/navtrain_dbs}"
MC="$SCR/metric_cache"; SB="$SCR/bank_sample"
# the chain locks.  RL-S1, S1-AUX and the P2 seed replicate are the ones that matter after S1; P2 and P6 itself are listed because a probe beside them is as wrong.
LOCKS="${P6CPU_LOCKS:-$DATA/rls1/chain.lock $DATA/s1aux/chain.lock $DATA/refe_p2rep/chain.lock $DATA/refe_p2/chain.lock $DATA/refe_p6/chain.lock}"
LWS="${P6CPU_LWS:-7 8 9 10}"; NSETS="${P6CPU_SETS:-256}"
CGROOT="${P6CPU_CG_ROOT:-/sys/fs/cgroup}"
mkdir -p "$PR"
log() { echo "$(date -u +%FT%T) $*" | tee -a "$L"; }
refuse() { log "ZZP6CPU_REFUSED $*"; exit 1; }
left_s() { echo $(( BUDGET_S - SECONDS )); }
# a stage runs only when enough budget is left; otherwise it is logged as skipped (never silently dropped)
have_time() { if [ "$(left_s)" -ge "$1" ]; then return 0; fi; log "ZZP6CPU_BUDGET skip=$2 (left $(left_s)s < need $1s)"; return 1; }
cleanup() {
  rm -f "$DATA/.p6cpu_dd" 2> /dev/null
  [ "${P6CPU_KEEP:-0}" = 1 ] || rm -rf "$SCR" 2> /dev/null
}

# ---- 0. refusals.  (a) our own lock  (b) the chains' locks  (c) the GPU  (d) a loaded cgroup
exec 200>"$PR/probe.lock"
flock -n 200 || { echo "ZZP6CPU_REFUSED another p6_cpu_probe.sh holds $PR/probe.lock"; exit 1; }
# a stage file left by an EARLIER run must never reach this run's verdict (probe.log is append-only and kept); only AFTER our own lock is held, so a second probe cannot
# delete the running one's files
rm -rf "$SCR" 2> /dev/null
rm -f "$PR"/lab_w*.json "$PR"/lab_w*.log "$PR"/pool_*.json "$PR"/pool_*.log "$PR"/d9.json "$PR"/d9_build.out "$PR"/dd.json "$PR"/bank_sample.json "$PR"/bank_sample.out "$PR"/verdict.json "$PR"/disk.txt 2> /dev/null
# the cleanup trap is armed ONLY now: a probe that lost the lock race must not delete the running probe's scratch or dd file on its way out
trap cleanup EXIT
for lk in $LOCKS; do
  # -e first: a lock file that never existed means that chain never ran, and the rw open below would CREATE it
  if [ -e "$lk" ] && ! ( flock -n 9 ) 9<> "$lk"; then refuse "the chain lock $lk is held: a chain is running; the probe runs only between chains"; fi
done
# rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never pgrep -f: it matches its own command line.
gpu_busy() {
  local nv="${P6CPU_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null 200>&-)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
gpu_busy && refuse "a process is on the GPU (or nvidia-smi cannot say): the probe never adds load to a pod that is training"
[ -f "$WORK/teacher_env.sh" ] || refuse "$WORK/teacher_env.sh is missing (the NAVSIM interpreter is DRIVERL_EVAL_PYTHON there)"
source "$WORK/teacher_env.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN
PY="${DRIVERL_EVAL_PYTHON:-}"; [ -n "$PY" ] || refuse "DRIVERL_EVAL_PYTHON is empty after sourcing teacher_env.sh"
for f in refe/p6_cpu_probe.py refe/diag_p6_labeller_identity.py refe/build_metric_cache.py refe/p6_prep.py refe/online_labeller.py refe/navsim_pdm_targets.py \
         refe/navtrain_scenarios.py splits/navtrain.yaml; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: set P6CPU_PKG to the shipped P6 tree"
done
HELP="$PKG/refe/p6_cpu_probe.py"; [ -f "$HELP" ] || HELP="$HERE/p6_cpu_probe.py"
export P6CPU_PKG="$PKG"
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
log "ZZP6CPU_START pkg=$PKG budget_min=$((BUDGET_S / 60)) lws='$LWS' sets=$NSETS locks_checked='$LOCKS'"

# ---- 1. environment.  The cgroup quota, NEVER nproc (96 shown on an 11-core quota, MEASURED 2026-10-13)
CG="$("$PY" "$HELP" cores --root "$CGROOT" 200>&-)"; CORES_F="$(echo "$CG" | awk '{print $2}')"; CORES_SRC="$(echo "$CG" | cut -d' ' -f3-)"
CORES="${CORES_F%%.*}"; [ -n "$CORES" ] && [ "$CORES" -ge 1 ] 2> /dev/null || CORES=1
CW="${P6CPU_CACHE_WORKERS:-$(( CORES < 1 ? 1 : (CORES > 24 ? 24 : CORES) ))}"                  # == p6_pod.sh
LWD="${P6CPU_LW_DEFAULT:-$(( CORES - 4 < 1 ? 1 : (CORES - 4 > 16 ? 16 : CORES - 4) ))}"       # == p6_pod.sh's LW
CPUM="$(grep -m1 'model name' /proc/cpuinfo 2> /dev/null | cut -d: -f2- | sed 's/^ *//')"; MHZ="$(grep -m1 'cpu MHz' /proc/cpuinfo 2> /dev/null | cut -d: -f2- | sed 's/^ *//')"
TPC="$(lscpu 2> /dev/null | grep -m1 'Thread(s) per core' | awk '{print $NF}')"
log "ZZP6CPU_ENV cores=$CORES_F ($CORES_SRC) cache_workers=$CW lw_default=$LWD nproc=$(nproc) cpu='${CPUM:-unknown}' mhz=${MHZ:-unknown} threads_per_core=${TPC:-unknown}"
log "ZZP6CPU_CGROUP $("$PY" "$HELP" cg --root "$CGROOT" 200>&- | cut -c1-600)"
# (d) a loaded cgroup voids every number: measure what the cgroup burns with NO work of ours
QS="${P6CPU_QUIET_S:-5}"
Q0="$("$PY" "$HELP" cg --root "$CGROOT" 200>&-)"; sleep "$QS" 200>&-; Q1="$("$PY" "$HELP" cg --root "$CGROOT" 200>&-)"
BG="$("$PY" - "$Q0" "$Q1" "$QS" 200>&- <<'EOF'
import json, sys
a, b = (json.loads(x.split(" ", 1)[1]) for x in sys.argv[1:3])
print(round((b["usage_s"] - a["usage_s"]) / float(sys.argv[3]), 3) if "usage_s" in a and "usage_s" in b else "unknown")
EOF
)"
[ -n "$BG" ] || BG=unknown
log "ZZP6CPU_QUIET bg_cores=$BG (max ${P6CPU_MAX_BG:-1.0})"
if [ "$BG" = unknown ]; then log "ZZP6CPU_WARN no cgroup CPU counter readable: the pod's background load is UNVERIFIED"
elif awk -v a="$BG" -v b="${P6CPU_MAX_BG:-1.0}" 'BEGIN{exit !(a > b)}'; then refuse "the cgroup burns $BG cores with nothing of ours running (> ${P6CPU_MAX_BG:-1.0}): not a quiet window"; fi
MAPS="${NUPLAN_MAPS_ROOT:-}"
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DBS" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
[ -d "$BANK" ] && [ -d "$DBS" ] || refuse "bank $BANK or DBs $DBS missing"
mkdir -p "$SCR"
SIMHZ_ENV=(); [ -n "${P6CPU_SIM_HZ:-}" ] && SIMHZ_ENV=("REFE_SIM_HZ=$P6CPU_SIM_HZ")           # unset == what p6_pod.sh does today (navtrain_scenarios defaults to 20 Hz)
log "ZZP6CPU_SIMHZ effective=${P6CPU_SIM_HZ:-unset(chain default)}"

# ---- 2. the bank, ONE pass: the exact D9 workload and a sample of whole logs
if have_time 600 bank; then
  timeout "${P6CPU_SCAN_TMO:-420}" "$PY" "$HELP" sample-bank --bank "$BANK" --db-root "$DBS" --logs "${P6CPU_D9_LOGS:-12}" --out "$SB" --max-scan-s "${P6CPU_SCAN_S:-300}" \
    --json "$PR/bank_sample.json" > "$PR/bank_sample.out" 2>&1 200>&-
  grep -q '^ZZBANK_SAMPLE ' "$PR/bank_sample.out" || refuse "the bank scan produced nothing ($(tail -n 1 "$PR/bank_sample.out" | cut -c1-200))"
  log "$(grep '^ZZBANK_SAMPLE ' "$PR/bank_sample.out" | cut -c1-500)"
fi

# ---- 3. D9 on the sample, into a SCRATCH dir.  Same environment as p6_pod.sh's build: OMP_NUM_THREADS=1, nice 10, spawn pool of CW workers
D9OK=0
if [ -s "$SB/targets_rank0.jsonl" ] && have_time 420 d9; then
  G0="$("$PY" "$HELP" cg --root "$CGROOT" 200>&-)"; T0=$SECONDS
  ( exec env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 "${SIMHZ_ENV[@]}" nice -n 10 timeout "${P6CPU_D9_TMO:-900}" "$PY" build_metric_cache.py build \
      --targets "$SB" --out "$MC" --db-root "$DBS" --map-root "$MAPS" --workers "$CW" ) > "$PR/d9_build.out" 2>&1 200>&-
  G1="$("$PY" "$HELP" cg --root "$CGROOT" 200>&-)"; D9S=$((SECONDS - T0))
  if grep -q '^ZZMETRIC_CACHE_DONE .* failed=0 ' "$PR/d9_build.out"; then
    "$PY" "$HELP" d9-summary --before "$G0" --after "$G1" --wall "$D9S" --cw "$CW" --bank-json "$PR/bank_sample.json" --cache "$MC" --out "$PR/d9.json" 200>&- | tee -a "$L"
    D9OK=1
  else
    log "ZZP6CPU_D9_FAILED $(tail -n 1 "$PR/d9_build.out" | cut -c1-300)"
  fi
fi

# ---- 4. the pool, per label-worker count
TOKS=""; [ "$D9OK" = 1 ] && TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n 128 200>&- | sed -n 's/^ZZTOKENS //p')"
if [ -z "$TOKS" ]; then log "ZZP6CPU_NO_PARTS no parts to label: the pool stages cannot run"; fi
POOL_ENV=(env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK "${SIMHZ_ENV[@]}")
if [ -n "$TOKS" ]; then
  for W in $LWS; do
    [ "$W" -le "$((CORES + 1))" ] || { log "ZZP6CPU_SKIP lw=$W (> cores+1)"; continue; }
    have_time 420 "lw$W" || continue
    # (a) the gate's own instrument, with the chain's exported OMP_NUM_THREADS=4
    "${POOL_ENV[@]}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 timeout "${P6CPU_LAB_TMO:-900}" "$PY" diag_p6_labeller_identity.py --cache "$MC" --db-root "$DBS" --map-root "$MAPS" \
      --tokens "$TOKS" --workers "$W" --n-sets "$NSETS" --out "$PR/lab_w$W.json" > "$PR/lab_w$W.log" 2>&1 200>&-
    log "ZZP6CPU_LW w=$W gate: $(grep -E '^ZZLAB_(IDENTITY|THROUGHPUT)' "$PR/lab_w$W.log" | tr '\n' ' ')"
    # (b) instrumented, same pool, same OMP; (c) unqueued service latency; (d) the trainer's clock
    for M in thru latency paced; do
      NS="$NSETS"; [ "$M" = latency ] && NS=96
      "${POOL_ENV[@]}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 timeout "${P6CPU_POOL_TMO:-600}" "$PY" "$HELP" pool --cache "$MC" --tokens "$TOKS" --workers "$W" --n-sets "$NS" --mode "$M" \
        --n-micro "${P6CPU_PACED_MICRO:-60}" --out "$PR/pool_${M}_w$W.json" > "$PR/pool_${M}_w$W.log" 2>&1 200>&-
      log "ZZP6CPU_POOL w=$W mode=$M $(grep '^ZZPOOL_OK' "$PR/pool_${M}_w$W.log" | cut -c1-260)"
    done
  done
  # ---- 5. A/B at the chain's default W: 4 BLAS/OpenMP threads per worker (the chain exports OMP_NUM_THREADS=4) against 1
  if [ -s "$PR/lab_w$LWD.json" ] && have_time 240 ab; then
    "${POOL_ENV[@]}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 timeout "${P6CPU_POOL_TMO:-600}" "$PY" "$HELP" pool --cache "$MC" --tokens "$TOKS" \
      --workers "$LWD" --n-sets "$NSETS" --mode thru --out "$PR/pool_thru_w${LWD}_omp1.json" > "$PR/pool_thru_w${LWD}_omp1.log" 2>&1 200>&-
    log "ZZP6CPU_AB w=$LWD omp1 $(grep '^ZZPOOL_OK' "$PR/pool_thru_w${LWD}_omp1.log" | cut -c1-260)"
  fi

  # ---- 5b. the latency lever, measured on THIS pod: each 64-proposal set split into 2 jobs of 32 (labels unchanged, M-LAB chunk_probe: max |dPDMS| = 0.0)
  if [ -s "$PR/lab_w$LWD.json" ] && have_time 240 chunk2; then
    for M in latency paced; do
      NS="$NSETS"; [ "$M" = latency ] && NS=96
      "${POOL_ENV[@]}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 timeout "${P6CPU_POOL_TMO:-600}" "$PY" "$HELP" pool --cache "$MC" --tokens "$TOKS" --workers "$LWD" --n-sets "$NS" --mode "$M" \
        --n-micro "${P6CPU_PACED_MICRO:-60}" --chunks 2 --out "$PR/pool_${M}_w${LWD}_c2.json" > "$PR/pool_${M}_w${LWD}_c2.log" 2>&1 200>&-
      log "ZZP6CPU_CHUNK2 w=$LWD mode=$M $(grep '^ZZPOOL_OK' "$PR/pool_${M}_w${LWD}_c2.log" | cut -c1-260)"
    done
  fi
fi

# ---- 6. disk.  P6's NEW bytes against what is really writable (df does not see a pod volume's quota): sizes of what exists, then a write test
du_one() { local d="$1" s; [ -e "$d" ] || { echo "absent"; return; }; s="$(timeout "${P6CPU_DU_TMO:-90}" du -sk "$d" 2> /dev/null 200>&- | cut -f1)"; echo "${s:-timeout}"; }
if have_time 300 disk; then
  {
    echo "ZZP6CPU_DISK df_workspace_kb(UNRELIABLE): $(df -k "$DATA" 2> /dev/null | tail -n 1 | awk '{print "size="$2" used="$3" avail="$4}')"
    for d in refe_navtrain/train_grow refe_navtrain/train_grow_cmd refe_navtrain/onpolicy refe_navtrain/metric_cache refe_sft2 refe_runs refe_p2 rls1 s1aux refe_p2rep refe_p6 navsim_logs navsim_logs_dl backbones; do
      echo "ZZP6CPU_DU $d kb=$(du_one "$DATA/$d")"
    done
    echo "ZZP6CPU_DU navtrain_dbs files=$(ls "$DBS" 2> /dev/null | wc -l) navtrain_pixels shards=$(ls -d "$DATA"/navtrain_pixels/.done_* 2> /dev/null | wc -l)"
    [ "${P6CPU_DU_PIX:-0}" = 1 ] && echo "ZZP6CPU_DU navtrain_pixels kb=$(du_one "$DATA/navtrain_pixels")"
  } 200>&- | tee -a "$L" > "$PR/disk.txt"
fi
# P6's NEW persistent bytes, ESTIMATED in eval/P6_ON_11VCPU.md section 5: 44-85 GB (high case: D9 parts 27 + lane twins <= 22.8, NAVSIM metadata <= 25, run <= 7.5, bank copy ~1, scratch ~2); 96 GB = the high case + ~7 GB the P2 seed replicate writes beside it + a margin (72 with the metadata freed: set P6CPU_DD_GB)
DDGB="${P6CPU_DD_GB:-96}"; DDMB="${P6CPU_DD_MB:-$((DDGB * 1024))}"; DDCH="${P6CPU_DD_CHUNK_MB:-1024}"; DDMAX="${P6CPU_DD_MAX_S:-600}"
if have_time 120 dd; then
  F="$DATA/.p6cpu_dd"; : > "$F"; T0=$SECONDS; MB=0; STOP=target
  while [ "$MB" -lt "$DDMB" ]; do
    [ $((SECONDS - T0)) -ge "$DDMAX" ] && { STOP=time_cap; break; }
    [ "$(left_s)" -le 60 ] && { STOP=budget; break; }
    dd if=/dev/zero of="$F" bs=1M count="$DDCH" oflag=append conv=notrunc,fsync 2>> "$L" 200>&- || { STOP=write_error; break; }
    MB=$(( $(stat -c %s "$F" 2> /dev/null || echo 0) / 1048576 ))
  done
  DS=$((SECONDS - T0)); rm -f "$F"
  printf '{"mb_target": %s, "mb_ok": %s, "gb_target": %s, "gb_ok": %s, "seconds": %s, "mb_s": %s, "stopped_by": "%s"}\n' "$DDMB" "$MB" "$((DDMB / 1024))" "$((MB / 1024))" "$DS" "$(( DS > 0 ? MB / DS : 0 ))" "$STOP" > "$PR/dd.json"
  log "ZZP6CPU_DD $(cat "$PR/dd.json")"
fi

# ---- 7. the verdict, against the gate's own bars.  One line.
"$PY" "$HELP" verdict --dir "$PR" --cores "$CORES_F" --lw-default "$LWD" --workers "$(echo $LWS | tr ' ' ',')" 200>&- | tee -a "$L"
log "ZZP6CPU_DONE elapsed=${SECONDS}s"
