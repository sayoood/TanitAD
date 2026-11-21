#!/usr/bin/env bash
# pod_distrule_run.sh <NEWARM> <QUEUE> <BASEARM> <MODE>      (runs ON THE POD, started by the lead; card PREREG_V8F_PROOF_ADD26)
# One T1-kin rollout of the checkpoint /workspace/v8proof/runs/<QUEUE>/<BASEARM>/ckpt_last.pt through the DISTRULE wrapper, with the eval argv of the chain's dump step
# (chain2/chain_arms.sh t1dump), written to /workspace/v8proof/runs/distrule/t1/<NEWARM>/ in the layout pull_spatial_t1.sh pulls (Q=distrule):
#   dump_<NEWARM>_m10_causal_heads/ep*.npz  ticks/ticks_<tag>.npz  t1v8f_<tag>.json  T1_DONE ;  the rule log and the run record go into ticks/ (they travel in the same tar).
# MODE: label | head | never | push | plumb | off.   Naming for the lead: <BASE>RL_s<seed> label, <BASE>RH_s<seed> head, <BASE>RN_s<seed> never, <BASE>RO_s<seed> off,
#   <BASE>RP_s<seed> push, <BASE>RM_s<seed> plumb.
# Refuses (rc 2) on: bad arguments, an output folder that exists, a missing checkpoint or input, a pinned tool whose sha256 differs, under 20 GB free container memory
# (/sys/fs/cgroup/memory.max minus the anon line of memory.stat). Starts exactly ONE rollout process, in the foreground; never touches another job, git or any other folder.
# Env (all optional): DR_W (default /workspace/v8proof) DR_PY DR_TREE (default $DR_W/tree_leadsup) DR_FLOOR (card|cap-only, default card) DR_TIMEOUT (s, default 2700)
#   DR_MAX_WINDOWS (smoke: a subset, then no T1_DONE) DR_DEVICE (default cuda) DR_DRY=1 (all checks, print the command, run nothing)
#   DR_MEMMAX_FILE DR_MEMSTAT_FILE (the cgroup files, for a test off the pod) DR_MIN_FREE_GB (default 20)
set -u
NEW=${1:-}; Q=${2:-}; BASE=${3:-}; MODE=${4:-}
die() { echo "REFUSED: $*" >&2; exit 2; }
[ $# = 4 ] || die "usage: pod_distrule_run.sh <NEWARM> <QUEUE> <BASEARM> <MODE>"
[[ "$NEW" =~ ^[A-Za-z0-9_]+$ ]] || die "NEWARM must match [A-Za-z0-9_]+ (got '$NEW')"
[[ "$Q" =~ ^[A-Za-z0-9_]+$ ]] && [[ "$BASE" =~ ^[A-Za-z0-9_]+$ ]] || die "QUEUE and BASEARM must match [A-Za-z0-9_]+"
case "$MODE" in label|head|never|push|plumb|off|mix) ;; *) die "MODE must be label|head|never|push|plumb|off (got '$MODE')";; esac
HERE=$(cd "$(dirname "$0")" && pwd)
W=${DR_W:-/workspace/v8proof}
PY=${DR_PY:-/workspace/venv/bin/python}
TREE=${DR_TREE:-$W/tree_leadsup}
FLOOR=${DR_FLOOR:-card}
case "$FLOOR" in card|cap-only) ;; *) die "DR_FLOOR must be card|cap-only";; esac
TMO=${DR_TIMEOUT:-2700}
DEV=${DR_DEVICE:-cuda}
MINGB=${DR_MIN_FREE_GB:-20}
PINNED=61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397
CK=$W/runs/$Q/$BASE/ckpt_last.pt
INP=$W/data/inputs; PINS=$W/hierrep/pins
TRACKS=${DR_TRACKS:-$INP/native_tracks.pt}; GRID=${DR_GRID:-$PINS/eval_grid_windows.json}; JOINE=${DR_JOIN:-$PINS/join_eval141.npz}; EVALC=${DR_EVALC:-$W/data/cache_pod}
OUT=$W/runs/distrule/t1/$NEW
TAG=${NEW}_m10_causal_heads

# 1. memory first (the pod container is capped; free shows the host)
MM=${DR_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${DR_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
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

# 2. inputs and the output
[ -e "$OUT" ] && die "output exists: $OUT (remove it by hand if it is a failed run)"
[ -f "$CK" ] || die "no checkpoint $CK"
for f in "$TRACKS" "$GRID" "$JOINE"; do [ -f "$f" ] || die "missing input $f"; done
[ -d "$EVALC" ] || die "missing eval cache $EVALC"
SCRIPT=$TREE/stack/scripts/v8f_t1_eval.py
[ -f "$SCRIPT" ] || die "no pinned tool $SCRIPT"
got=$(sha256sum "$SCRIPT" | cut -c1-64)
[ "$got" = "$PINNED" ] || die "pinned tool sha256 $got != $PINNED"
[ -f "$HERE/${DR_RULEPY:-v8f_t1_eval_rule.py}" ] && [ -f "$HERE/dist_rule.py" ] || die "${DR_RULEPY:-v8f_t1_eval_rule.py} / dist_rule.py not next to this script ($HERE)"

# 3. the command: the chain's dump argv + the wrapper flags
if [ "$MODE" = mix ]; then set -- $DR_MIX; ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3"); else ARGV=(--tree "$TREE" --dist-rule "$MODE"); fi
[ "$MODE" = off ] || ARGV+=(--rule-log "$OUT/ticks/rule_log_$NEW.npz" --floor "$FLOOR")
ARGV+=(run --ckpt "$CK" --tag "$TAG" --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$OUT"
  --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$OUT/ticks" --dump-heads --device "$DEV")
[ -z "${DR_MAX_WINDOWS:-}" ] || ARGV+=(--max-windows "$DR_MAX_WINDOWS")
if [ "${DR_DRY:-0}" = 1 ]; then echo "DRY: PYTHONPATH=$TREE/stack timeout $TMO $PY $HERE/${DR_RULEPY:-v8f_t1_eval_rule.py} ${ARGV[*]}"; exit 0; fi

mkdir -p "$OUT/ticks" || die "cannot create $OUT"
ckpt_sha=$(sha256sum "$CK" | cut -c1-64)
echo "new arm $NEW from $Q/$BASE mode $MODE floor $FLOOR ckpt sha256 $ckpt_sha started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$OUT/ticks/rule_run_$NEW.txt"
echo "argv: ${ARGV[*]}" >> "$OUT/ticks/rule_run_$NEW.txt"
echo "sha256 wrapper $(sha256sum "$HERE/${DR_RULEPY:-v8f_t1_eval_rule.py}" | cut -c1-64) dist_rule $(sha256sum "$HERE/dist_rule.py" | cut -c1-64) pinned $got" >> "$OUT/ticks/rule_run_$NEW.txt"
PYTHONPATH="$TREE/stack" timeout "$TMO" "$PY" "$HERE/${DR_RULEPY:-v8f_t1_eval_rule.py}" "${ARGV[@]}" >> "$OUT/t1.log" 2>&1
rc=$?
echo "rc $rc finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/ticks/rule_run_$NEW.txt"
[ $rc = 0 ] || { echo "FAILED rc=$rc (see $OUT/t1.log)" >&2; exit 1; }
if [ -z "${DR_MAX_WINDOWS:-}" ]; then
  n=$(ls "$OUT/dump_$TAG" 2>/dev/null | grep -c -- '^ep.*npz$')
  [ "$n" = 141 ] || { echo "FAILED: dump holds $n ep files, not 141" >&2; exit 1; }
  [ "$MODE" = off ] || [ -f "$OUT/ticks/rule_log_$NEW.npz" ] || { echo "FAILED: no rule log" >&2; exit 1; }
  touch "$OUT/T1_DONE"
fi
echo "DONE $NEW"
