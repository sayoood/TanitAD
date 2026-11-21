#!/usr/bin/env bash
# pod_latency_run.sh <TREE> <CKPT> [<CKPT> ...]      (runs ON THE POD under nohup, started by the lead; the builder did NOT run it)
#   nohup bash pod_latency_run.sh /workspace/v8proof/tree_leadsup /workspace/v8proof/runs/<chain>/<ARM>/ckpt_last.pt > /workspace/v8proof/runs/latency_proxy.nohup 2>&1 &
# Measures the LATENCY PROXY (latency_proxy.py) of each checkpoint on the pod GPU.  A proxy: not Thor, not a product latency, the image encoder is not included.
# Steps: (0) argument and file checks, a start marker made with noclobber (a second start is refused), the self-check runner on cpu;  (1) slot wait: fewer than 3 GPU compute processes
# (nvidia-smi --query-compute-apps=pid) AND more than 28 GB free container memory (memory.max minus the anon line of memory.stat), two probes 45 s apart, polled every 60 s, at most 6 h,
# then a STOP marker;  (2) one latency_proxy.py run per checkpoint, in the foreground, into /workspace/v8proof/runs/latency_proxy/<chain>_<ARM>/.
# Output folder: /workspace/v8proof/runs/latency_proxy (new; an existing folder or start marker refuses the start).  End markers LATENCY_DONE.txt / LATENCY_STOP.txt in it; the last log line is
# `ZZ LATENCY DONE` or `ZZ LATENCY STOP: <reason>`.  DRY=1 prints every command and runs nothing.  This script never stops, starts or touches another job, and never removes or moves a file.
# Env (all optional): LAT_W (default /workspace/v8proof)  LAT_PY (default /workspace/venv/bin/python)  LAT_TRACKS LAT_GRID LAT_JOIN LAT_EVALC (inputs as pod_imagplan_run.sh)
#   LAT_LIBDIRS (colon separated folders with imag_plan.py v8f_t1_eval_imagplan.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py)  LAT_TIMEOUT (s per checkpoint, default 3600)
#   LAT_EXTRA (extra arguments of latency_proxy.py, e.g. "--n 100")  LAT_MAX_GPU_JOBS (3)  LAT_MIN_FREE_GB (28)  LAT_PROBE_GAP_S (45)  LAT_POLL_S (60)  LAT_MAX_WAIT_S (21600)
#   for tests off the pod: LAT_MEMMAX_FILE LAT_MEMSTAT_FILE LAT_GPU_COUNT
set -u
DRY=${DRY:-0}
[ $# -ge 2 ] || { echo "REFUSED: usage: pod_latency_run.sh <TREE> <CKPT> [<CKPT> ...]" >&2; exit 2; }
TREE=$1; shift
CKPTS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
W=${LAT_W:-/workspace/v8proof}
PY=${LAT_PY:-/workspace/venv/bin/python}
MAXGPU=${LAT_MAX_GPU_JOBS:-3}; MINGB=${LAT_MIN_FREE_GB:-28}; GAP=${LAT_PROBE_GAP_S:-45}; POLL=${LAT_POLL_S:-60}; MAXWAIT=${LAT_MAX_WAIT_S:-21600}; TMO=${LAT_TIMEOUT:-3600}
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
INP=$W/data/inputs; PINS=$W/hierrep/pins
TRACKS=${LAT_TRACKS:-$INP/native_tracks.pt}; GRID=${LAT_GRID:-$PINS/eval_grid_windows.json}; JOINE=${LAT_JOIN:-$PINS/join_eval141.npz}; EVALC=${LAT_EVALC:-$W/data/cache_pod}
RUNS=$W/runs
OUT=$RUNS/latency_proxy
START=$RUNS/latency_proxy.STARTED
LOG=$OUT/latency.log

refuse() { echo "REFUSED: $*" >&2; exit 2; }
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
stop() { # reason: marker, then the last log line
  echo "$*" > "$OUT/LATENCY_STOP.txt"
  say "ZZ LATENCY STOP: $*"
  exit 1
}
dry() { echo "DRY: $*"; }

# ---- 0. checks before anything is created
for c in "${CKPTS[@]}"; do [[ "$c" =~ ^/[A-Za-z0-9_./-]+$ ]] || refuse "checkpoint path must be absolute and plain ([A-Za-z0-9_./-]): $c"; done
[[ "$TREE" =~ ^/[A-Za-z0-9_./-]+$ ]] || refuse "tree path must be absolute and plain: $TREE"
if [ "$DRY" != 1 ]; then
  [ -d "$TREE" ] || refuse "no tree $TREE"
  for c in "${CKPTS[@]}"; do [ -f "$c" ] || refuse "no checkpoint $c"; done
  for f in "$TRACKS" "$GRID" "$JOINE"; do [ -f "$f" ] || refuse "missing input $f"; done
  [ -d "$EVALC" ] || refuse "missing eval cache $EVALC"
  [ -x "$PY" ] || refuse "no python at $PY"
fi
SCRIPT=$TREE/stack/scripts/v8f_t1_eval.py
if [ "$DRY" != 1 ]; then
  [ -f "$SCRIPT" ] || refuse "no pinned tool $SCRIPT"
  got=$(sha256sum "$SCRIPT" | cut -c1-64)
  [ "$got" = "$PINNED" ] || refuse "pinned tool sha256 $got != $PINNED"
fi
LIBS="$HERE:$HERE/../imagplan:$HERE/../distrule:$HERE/../imaghead:${LAT_LIBDIRS:-}"
findlib() { local IFS=:; for d in $LIBS; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }; done; return 1; }
LIBPATH=""
if [ "$DRY" != 1 ]; then
  for f in imag_plan.py v8f_t1_eval_imagplan.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py; do findlib "$f" > /dev/null || refuse "$f not found next to this script or in ../imagplan ../distrule ../imaghead LAT_LIBDIRS"; done
  [ -f "$HERE/latency_proxy.py" ] && [ -f "$HERE/run_checks_pod.py" ] || refuse "latency_proxy.py and run_checks_pod.py must sit next to this script ($HERE)"
  LIBPATH=$(for f in imag_plan.py v8f_t1_eval_imagplan.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py; do dirname "$(findlib "$f")"; done | sort -u | paste -sd: -)
else
  LIBPATH="<folders of the five tool files>"
fi
NAMES=()
for c in "${CKPTS[@]}"; do
  n=$(basename "$(dirname "$(dirname "$c")")")_$(basename "$(dirname "$c")")
  n=$(printf '%s' "$n" | tr -c 'A-Za-z0-9_' '_')
  for e in "${NAMES[@]:-}"; do [ "$e" != "$n" ] || refuse "two checkpoints map to the same output name $n"; done
  NAMES+=("$n")
done

# ---- start marker (noclobber) and the output folder
if [ "$DRY" = 1 ]; then
  dry "(set -C; : > $START)   # the start marker; an existing marker refuses a second start"
  dry "mkdir $OUT   # new folder; an existing one refuses"
else
  [ -d "$RUNS" ] || refuse "no runs folder $RUNS"
  [ ! -e "$OUT" ] || refuse "output exists: $OUT (a second start is refused)"
  if ! ( set -C; : > "$START" ) 2>/dev/null; then refuse "start marker exists: $START (a second start is refused)"; fi
  mkdir "$OUT" || refuse "cannot create $OUT"
  say "start: tree $TREE; checkpoints ${#CKPTS[@]}; names ${NAMES[*]}; proxy on the pod GPU (not Thor, not a product latency; the image encoder is not included)"
fi

# ---- 0b. the self-check runner (cpu, pure python, about a minute)
CHK="PYTHONPATH=$HERE timeout 600 $PY $HERE/run_checks_pod.py --work-dir $OUT/checks"
if [ "$DRY" = 1 ]; then dry "$CHK > $OUT/checks.log   # must end with: ZZ CHECKS-POD DONE <n_pass> 0"
else
  PYTHONPATH="$HERE" timeout 600 "$PY" "$HERE/run_checks_pod.py" --work-dir "$OUT/checks" > "$OUT/checks.log" 2>&1
  last=$(tail -n 1 "$OUT/checks.log" | tr -d '\r')
  case "$last" in
    "ZZ CHECKS-POD DONE "*" 0") say "check runner: $last" ;;
    *) stop "the cpu check runner did not end with n_fail 0 (last line: $last)" ;;
  esac
fi

# ---- 1. the slot probe
probe() { # prints "<gpu jobs> <free GB>" ; returns 0 when a slot is free
  local ng mm anon free_b free_gb
  if [ -n "${LAT_GPU_COUNT:-}" ]; then ng=$LAT_GPU_COUNT
  else ng=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true); fi
  [[ "$ng" =~ ^[0-9]+$ ]] || ng=99
  local MM=${LAT_MEMMAX_FILE:-/sys/fs/cgroup/memory.max} MS=${LAT_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
  mm=$(tr -d ' \r\n' < "$MM" 2>/dev/null); anon=$(awk '$1=="anon"{print $2}' "$MS" 2>/dev/null | head -n 1)
  if [ "$mm" = max ]; then free_b=$(awk '$1=="MemAvailable:"{print $2*1024}' /proc/meminfo 2>/dev/null | head -n 1)
  elif [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then free_b=$((mm - anon))
  else free_b=0; fi
  [[ "$free_b" =~ ^[0-9]+$ ]] || free_b=0
  free_gb=$(awk -v b="$free_b" 'BEGIN{printf "%.1f", b/1073741824}')
  echo "$ng $free_gb"
  [ "$ng" -lt "$MAXGPU" ] && awk -v b="$free_b" -v m="$MINGB" 'BEGIN{exit !(b/1073741824 > m)}'
}
wait_slot() {
  local t0 now r1 r2
  t0=$(date +%s)
  while :; do
    r1=$(probe); if [ $? -eq 0 ]; then
      sleep "$GAP"
      r2=$(probe); if [ $? -eq 0 ]; then say "slot free: probe 1 [$r1], probe 2 [$r2] (GPU jobs, free GB)"; return 0; fi
      say "slot taken at the second probe [$r2]"
    else say "no slot [$r1] (GPU jobs, free GB; need fewer than $MAXGPU jobs and more than $MINGB GB)"; fi
    now=$(date +%s)
    [ $((now - t0)) -lt "$MAXWAIT" ] || return 1
    sleep "$POLL"
  done
}
if [ "$DRY" = 1 ]; then
  dry "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]'   # GPU jobs, need fewer than $MAXGPU"
  dry "free = memory.max - anon line of memory.stat (/sys/fs/cgroup), need more than $MINGB GB; two probes ${GAP} s apart, polled every ${POLL} s, at most ${MAXWAIT} s, then a STOP marker"
fi

# ---- 2. one run per checkpoint
fails=0
i=0
for c in "${CKPTS[@]}"; do
  name=${NAMES[$i]}; i=$((i + 1))
  RUNOUT=$OUT/$name
  ARGS="--tree $TREE --ckpt $c --out-dir $RUNOUT --tracks $TRACKS --eval-cache $EVALC --eval-grid $GRID --eval-join $JOINE --device cuda --lib-dirs $LIBPATH ${LAT_EXTRA:-}"
  if [ "$DRY" = 1 ]; then
    dry "(slot wait, as above)"
    dry "sha256sum $c | cut -c1-64   # recorded in the log"
    dry "PYTHONPATH=$TREE/stack:$HERE:$LIBPATH timeout $TMO $PY $HERE/latency_proxy.py $ARGS >> $OUT/$name.log 2>&1   # exit 0 = done, 2 = a self-check failed, 4 = output exists"
    dry "test -f $RUNOUT/LATENCY_PROXY.json && test -f $RUNOUT/LATENCY_PROXY.md"
    continue
  fi
  wait_slot || stop "no slot within ${MAXWAIT} s before $name"
  say "run $name: ckpt sha256 $(sha256sum "$c" | cut -c1-64)"
  # shellcheck disable=SC2086
  PYTHONPATH="$TREE/stack:$HERE:$LIBPATH" timeout "$TMO" "$PY" "$HERE/latency_proxy.py" $ARGS >> "$OUT/$name.log" 2>&1
  rc=$?
  say "run $name: rc $rc; last log line: $(tail -n 1 "$OUT/$name.log" | tr -d '\r' | cut -c1-200)"
  if [ "$rc" != 0 ] || [ ! -f "$RUNOUT/LATENCY_PROXY.json" ] || [ ! -f "$RUNOUT/LATENCY_PROXY.md" ]; then
    fails=$((fails + 1))
    say "run $name FAILED (rc $rc); see $OUT/$name.log"
  fi
done

if [ "$DRY" = 1 ]; then
  dry "echo done > $OUT/LATENCY_DONE.txt   # only when every run wrote LATENCY_PROXY.json and .md"
  dry "last log line: ZZ LATENCY DONE   (or: ZZ LATENCY STOP: <reason>)"
  exit 0
fi
if [ "$fails" -gt 0 ]; then stop "$fails of ${#CKPTS[@]} runs did not write LATENCY_PROXY.json and .md (see the per-run logs)"; fi
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ) runs ${#CKPTS[@]}" > "$OUT/LATENCY_DONE.txt"
say "ZZ LATENCY DONE"
exit 0
