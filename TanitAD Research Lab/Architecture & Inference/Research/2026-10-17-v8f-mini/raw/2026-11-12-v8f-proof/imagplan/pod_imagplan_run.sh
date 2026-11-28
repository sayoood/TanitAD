#!/usr/bin/env bash
# pod_imagplan_run.sh <NEWARM> <QUEUE> <BASEARM> <MODE> [TREE]      (runs ON THE POD, started by the lead; card PREREG_V8F_PROOF_ADD40; the builder did NOT run it)
# One T1-kin rollout of the checkpoint /workspace/v8proof/runs/<QUEUE>/<BASEARM>/ckpt_last.pt through the IMAGPLAN wrapper, with the eval argv of the chain's dump step
# (as pod_distrule_run.sh), written to /workspace/v8proof/runs/imagplan/t1/<NEWARM>/ in the layout pull_spatial_t1.sh pulls (Q=imagplan):
#   dump_<NEWARM>_m10_causal_heads/ep*.npz  ticks/ticks_<tag>.npz  t1v8f_<tag>.json  T1_DONE ;  the imag log and the run record go into ticks/ (they travel in the same tar).
# MODE: off | cand-imag | cand-cv | cand-label.   TREE (optional 5th argument, default $IP_TREE or $IP_W/tree_leadsup): the tree that holds the pinned tool.
# Options (env, all optional; each reaches a flag of the wrapper): IP_CAND_SET ("1.0" for control p1)  IP_CONST_GAP (metres, p1: 100)  IP_SHIFT_M (p6: -20, cand-imag only)
#   IP_FAMILY (scale|decel, default scale; decel = note ADD40_N3, IP_CAND_SET then holds d values, default the card set; IP_FLOOR must stay card)
#   IP_FLOOR (card|cap-only, default card)  IP_FALLBACK (zero|none, default zero; none = identity probe p1 only)  IP_NOISE (seed >= 0: control p3)  IP_P45=1 (write the p4/p5 log)
# Env for the machine: IP_W (default /workspace/v8proof) IP_PY IP_TIMEOUT (s, default 2700) IP_MAX_WINDOWS (smoke: a subset, then no T1_DONE) IP_DEVICE (default cuda)
#   IP_DRY=1 (all checks, print the command, run nothing)  IP_MIN_FREE_GB (default 20)  IP_MEMMAX_FILE IP_MEMSTAT_FILE (cgroup files, for a test off the pod)  IP_GPU_COUNT (tests; else nvidia-smi)
#   IP_MAX_GPU_JOBS (default 3: refuse when this many GPU jobs already run)
# Files next to this script (copy the folder as it is): imag_plan.py v8f_t1_eval_imagplan.py and the three sibling tools dist_rule.py v8f_t1_eval_rule.py imag_dump.py
#   (found next to this script, or in ../distrule and ../imaghead, or in IP_LIBDIRS = colon-separated folders).
# Refuses (rc 2) on: bad arguments, an output folder that exists, a missing checkpoint or input or tool file, a pinned tool whose sha256 differs, under 20 GB free container memory
# (/sys/fs/cgroup/memory.max minus the anon line of memory.stat), 3 or more GPU jobs already running. Starts exactly ONE rollout process, in the foreground; never touches another job, git or any other folder.
set -u
NEW=${1:-}; Q=${2:-}; BASE=${3:-}; MODE=${4:-}; TREE_ARG=${5:-}
die() { echo "REFUSED: $*" >&2; exit 2; }
[ $# -ge 4 ] && [ $# -le 5 ] || die "usage: pod_imagplan_run.sh <NEWARM> <QUEUE> <BASEARM> <MODE> [TREE]"
[[ "$NEW" =~ ^[A-Za-z0-9_]+$ ]] || die "NEWARM must match [A-Za-z0-9_]+ (got '$NEW')"
[[ "$Q" =~ ^[A-Za-z0-9_]+$ ]] && [[ "$BASE" =~ ^[A-Za-z0-9_]+$ ]] || die "QUEUE and BASEARM must match [A-Za-z0-9_]+"
case "$MODE" in off|cand-imag|cand-cv|cand-label|cand-imagkin) ;; *) die "MODE must be off|cand-imag|cand-cv|cand-label|cand-imagkin (got '$MODE')";; esac
HERE=$(cd "$(dirname "$0")" && pwd)
W=${IP_W:-/workspace/v8proof}
PY=${IP_PY:-/workspace/venv/bin/python}
TREE=${TREE_ARG:-${IP_TREE:-$W/tree_leadsup}}
FLOOR=${IP_FLOOR:-card}; FALLBACK=${IP_FALLBACK:-zero}
case "$FLOOR" in card|cap-only) ;; *) die "IP_FLOOR must be card|cap-only";; esac
case "$FALLBACK" in zero|none) ;; *) die "IP_FALLBACK must be zero|none";; esac
FAMILY=${IP_FAMILY:-scale}
case "$FAMILY" in scale|decel) ;; *) die "IP_FAMILY must be scale|decel";; esac
[ "$FAMILY" != decel ] || [ "$FLOOR" = card ] || die "IP_FLOOR applies to IP_FAMILY=scale only"
TMO=${IP_TIMEOUT:-2700}; DEV=${IP_DEVICE:-cuda}; MINGB=${IP_MIN_FREE_GB:-20}; MAXGPU=${IP_MAX_GPU_JOBS:-3}
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
CK=$W/runs/$Q/$BASE/ckpt_last.pt
INP=$W/data/inputs; PINS=$W/hierrep/pins
TRACKS=${IP_TRACKS:-$INP/native_tracks.pt}; GRID=${IP_GRID:-$PINS/eval_grid_windows.json}; JOINE=${IP_JOIN:-$PINS/join_eval141.npz}; EVALC=${IP_EVALC:-$W/data/cache_pod}
OUT=$W/runs/imagplan/t1/$NEW
TAG=${NEW}_m10_causal_heads

# 1. memory first (the pod container is capped; free shows the host)
MM=${IP_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${IP_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
[ -r "$MM" ] && [ -r "$MS" ] || die "cannot read $MM / $MS: container memory unknown"
mmax=$(tr -d ' \r\n' < "$MM"); anon=$(awk '$1=="anon"{print $2}' "$MS" | head -n 1)
[[ "$anon" =~ ^[0-9]+$ ]] || die "no numeric anon line in $MS"
if [ "$mmax" = max ]; then
  avail=$(awk '$1=="MemAvailable:"{print $2*1024}' /proc/meminfo 2>/dev/null | head -n 1)
  [[ "$avail" =~ ^[0-9]+$ ]] || die "memory.max is 'max' and /proc/meminfo has no MemAvailable"
  free_b=$avail
else
  [[ "$mmax" =~ ^[0-9]+$ ]] || die "memory.max is not numeric ('$mmax')"
  free_b=$((mmax - anon))
fi
free_gb=$(awk -v b="$free_b" 'BEGIN{printf "%.1f", b/1073741824}')
awk -v b="$free_b" -v m="$MINGB" 'BEGIN{exit !(b/1073741824 >= m)}' || die "free container memory $free_gb GB < $MINGB GB (memory.max $mmax, anon $anon)"
echo "memory: free $free_gb GB (memory.max $mmax, anon $anon)"

# 2. GPU jobs: refuse when MAXGPU or more compute processes already run
if [ -n "${IP_GPU_COUNT:-}" ]; then ngpu=$IP_GPU_COUNT
else ngpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true); fi
[[ "$ngpu" =~ ^[0-9]+$ ]] || ngpu=0
[ "$ngpu" -lt "$MAXGPU" ] || die "$ngpu GPU jobs already run (limit $MAXGPU): not adding a fourth"
echo "gpu jobs: $ngpu"

# 3. inputs, tool files and the output
[ -e "$OUT" ] && die "output exists: $OUT (remove it by hand if it is a failed run)"
[ -f "$CK" ] || die "no checkpoint $CK"
for f in "$TRACKS" "$GRID" "$JOINE"; do [ -f "$f" ] || die "missing input $f"; done
[ -d "$EVALC" ] || die "missing eval cache $EVALC"
SCRIPT=$TREE/stack/scripts/v8f_t1_eval.py
[ -f "$SCRIPT" ] || die "no pinned tool $SCRIPT"
got=$(sha256sum "$SCRIPT" | cut -c1-64)
[ "$got" = "$PINNED" ] || die "pinned tool sha256 $got != $PINNED"
LIBS="$HERE:$HERE/../distrule:$HERE/../imaghead:${IP_LIBDIRS:-}"
findlib() { local IFS=:; for d in $LIBS; do [ -n "$d" ] && [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }; done; return 1; }
for f in imag_plan.py v8f_t1_eval_imagplan.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py; do findlib "$f" > /dev/null || die "$f not found next to this script or in ../distrule, ../imaghead, IP_LIBDIRS"; done
[ -f "$HERE/v8f_t1_eval_imagplan.py" ] || die "v8f_t1_eval_imagplan.py must sit next to this script ($HERE)"
LIBPATH=$(for f in imag_plan.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py; do dirname "$(findlib "$f")"; done | sort -u | paste -sd: -)

# 4. the command: the chain's dump argv + the wrapper flags
ARGV=(--tree "$TREE" --imag-plan "$MODE")
if [ "$MODE" != off ]; then
  ARGV+=(--imag-log "$OUT/ticks/imag_log_$NEW.npz" --floor "$FLOOR" --fallback "$FALLBACK" --cand-family "$FAMILY")
  [ -z "${IP_CAND_SET:-}" ] || ARGV+=(--cand-set "$IP_CAND_SET")
  [ -z "${IP_CONST_GAP:-}" ] || ARGV+=(--const-gap "$IP_CONST_GAP")
  [ -z "${IP_SHIFT_M:-}" ] || ARGV+=(--imag-shift-m "$IP_SHIFT_M")
  [ -z "${IP_NOISE:-}" ] || ARGV+=(--noise-after-tick "$IP_NOISE")
  [ "${IP_P45:-0}" != 1 ] || ARGV+=(--p45-log "$OUT/ticks/p45_$NEW.npz")
else
  for v in IP_FAMILY IP_CAND_SET IP_CONST_GAP IP_SHIFT_M IP_NOISE; do [ -z "${!v:-}" ] || die "$v with MODE off would change nothing"; done
fi
ARGV+=(run --ckpt "$CK" --tag "$TAG" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$OUT"
  --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$OUT/ticks" --dump-heads --device "$DEV")
[ -z "${IP_MAX_WINDOWS:-}" ] || ARGV+=(--max-windows "$IP_MAX_WINDOWS")
if [ "${IP_DRY:-0}" = 1 ]; then echo "DRY: PYTHONPATH=$TREE/stack:$HERE:$LIBPATH timeout $TMO $PY $HERE/v8f_t1_eval_imagplan.py ${ARGV[*]}"; exit 0; fi

mkdir -p "$OUT/ticks" || die "cannot create $OUT"
ckpt_sha=$(sha256sum "$CK" | cut -c1-64)
echo "new arm $NEW from $Q/$BASE mode $MODE floor $FLOOR fallback $FALLBACK ckpt sha256 $ckpt_sha started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$OUT/ticks/imagplan_run_$NEW.txt"
echo "argv: ${ARGV[*]}" >> "$OUT/ticks/imagplan_run_$NEW.txt"
echo "sha256 wrapper $(sha256sum "$HERE/v8f_t1_eval_imagplan.py" | cut -c1-64) imag_plan $(sha256sum "$(findlib imag_plan.py)" | cut -c1-64) pinned $got" >> "$OUT/ticks/imagplan_run_$NEW.txt"
PYTHONPATH="$TREE/stack:$HERE:$LIBPATH" timeout "$TMO" "$PY" "$HERE/v8f_t1_eval_imagplan.py" "${ARGV[@]}" >> "$OUT/t1.log" 2>&1
rc=$?
echo "rc $rc finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/ticks/imagplan_run_$NEW.txt"
[ $rc = 0 ] || { echo "FAILED rc=$rc (see $OUT/t1.log)" >&2; exit 1; }
if [ -z "${IP_MAX_WINDOWS:-}" ]; then
  n=$(ls "$OUT/dump_$TAG" 2>/dev/null | grep -c -- '^ep.*npz$')
  [ "$n" = 141 ] || { echo "FAILED: dump holds $n ep files, not 141" >&2; exit 1; }
  [ "$MODE" = off ] || [ -f "$OUT/ticks/imag_log_$NEW.npz" ] || { echo "FAILED: no imag log" >&2; exit 1; }
  touch "$OUT/T1_DONE"
fi
echo "DONE $NEW"
