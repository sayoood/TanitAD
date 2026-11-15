#!/usr/bin/env bash
# pod_imagplan_seq.sh <NAME> <QUEUE> <BASEARM> <TREE> [RUNS] : runs ON THE POD (nohup), written by the lead. Card PREREG_V8F_PROOF_ADD40 with note N1:
# the runs of ONE checkpoint, serial, each through pod_imagplan_run.sh (which refuses below 20 GB of free container memory and with 3 GPU jobs running;
# such a refusal is waited out, up to 6 h per run). Every run uses --floor cap-only (note N1 A1). New arm = <tag>_<BASEARM>.
# RUNS (optional, default "all") = space-separated tags of:
#   IPo   off (the pinned tool through the wrapper, the reference of this environment)
#   IPi   cand-imag with the p4/p5 log            IPid1 cand-imag, candidate set {1.0}, fallback none (p1, first part)
#   IPid2 cand-imag, constant gap 100 m (p1, second part)       IPn cand-imag with noise after the tick (p3)
#   IPc   cand-cv        IPl cand-label (p2)        IPs cand-imag with every imagined gap shifted by -20 m (p6)
# Then the dump compares that exist for the runs made (p1: IPid1 and IPid2 against IPo; p3: IPn against IPi; firing checks: IPi against IPo, IPs against IPi).
# Deletes nothing, stops nothing, starts one rollout at a time. Last log line: ZZ SEQ <NAME> DONE or ZZ SEQ STOP ...
set -u
NAME=${1:?name}; Q=${2:?queue}; BASE=${3:?base arm}; TREE=${4:?tree}; RUNS=${5:-all}
W=/workspace/v8proof; D=$W/imagplan; R=$W/runs/imagplan; LOG=$R/seq_$NAME.log
mkdir -p "$R"
export IP_FLOOR=cap-only IP_LIBDIRS=$W/distrule:$W/probe
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> "$LOG"; }
want() { [ "$RUNS" = all ] && return 0; case " $RUNS " in *" $1 "*) return 0;; esac; return 1; }
one() { # tag mode [VAR=value ...]
  local tag=$1 mode=$2; shift 2
  local new=${tag}_$BASE n=0 rc
  want "$tag" || return 0
  [ -f "$R/t1/$new/T1_DONE" ] && { say "$new has T1_DONE, kept"; return 0; }
  [ -e "$R/t1/$new" ] && { say "ZZ SEQ STOP $new: output folder exists without T1_DONE"; return 9; }
  while :; do
    env "$@" bash "$D/pod_imagplan_run.sh" "$new" "$Q" "$BASE" "$mode" "$TREE" > "$R/run_$new.out" 2>&1; rc=$?
    [ $rc = 0 ] && { say "$new done ($mode $*)"; return 0; }
    if [ $rc = 2 ] && [ ! -e "$R/t1/$new" ] && grep -q "REFUSED: .*\(free container memory\|GPU jobs already run\)" "$R/run_$new.out"; then
      n=$((n+1)); [ $n -gt 72 ] && { say "ZZ SEQ STOP $new: no slot after 6 h"; return 8; }
      [ $n = 1 ] && say "$new waits for a slot: $(tail -n 1 "$R/run_$new.out" | cut -c1-160)"
      sleep 300; continue
    fi
    say "ZZ SEQ STOP $new rc=$rc: $(tail -n 2 "$R/run_$new.out" | cut -c1-200 | tr '\n' ';')"; return 7
  done
}
dd() { echo "$R/t1/${1}_$BASE/dump_${1}_${BASE}_m10_causal_heads"; }
cmpd() { [ -d "$(dd "$1")" ] && [ -d "$(dd "$2")" ] || return 0; say "cmp $1 against $2 ($3): $(/workspace/venv/bin/python "$W/distrule/cmp_dumps.py" "$(dd "$1")" "$(dd "$2")" 2>&1 | tail -n 1 | cut -c1-200)"; }
say "seq $NAME start: $Q/$BASE tree $TREE runs $RUNS"
one IPo off || exit 1
one IPi cand-imag IP_P45=1 || exit 1
one IPid1 cand-imag IP_CAND_SET=1.0 IP_FALLBACK=none || exit 1
one IPid2 cand-imag IP_CONST_GAP=100 || exit 1
one IPn cand-imag IP_NOISE=0 || exit 1
one IPc cand-cv || exit 1
one IPl cand-label || exit 1
one IPs cand-imag IP_SHIFT_M=-20 || exit 1
cmpd IPid1 IPo "p1 first part, must be equal"
cmpd IPid2 IPo "p1 second part, must be equal"
cmpd IPn IPi "p3, must be equal"
cmpd IPi IPo "firing check, expected to differ"
cmpd IPs IPi "firing check, expected to differ"
say "ZZ SEQ $NAME DONE"
touch "$R/SEQ_${NAME}_DONE.txt"
