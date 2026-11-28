#!/usr/bin/env bash
# pod_imagplan_seq4.sh <NAME> <QUEUE> <BASEARM> <TREE> "<RUNS>" : runs ON THE POD (nohup), written by the lead. Card PREREG_V8F_PROOF_ADD40 with notes N1 to N4.
# Fourth form: the tool folder is IP_D (default /workspace/v8proof/imagplan_k: the tool with the source cand-imagkin of note N4 and the second diagnostic of the p4/p5 log);
# every candidate run uses the family of note N3 (IP_FAMILY=decel). The runs of ONE checkpoint, serial, each through pod_imagplan_run.sh (which refuses below 20 GB of free
# container memory and with 3 GPU jobs running; such a refusal is waited out, up to 6 h per run).
# Env: IP_SFX (suffix of every run but IPo), IP_REF_SFX (suffix of IPo; default = IP_SFX). New arm = <tag><suffix>_<BASEARM>. RUNS = space-separated tags, run in THIS order:
#   IPo    off (the pinned tool through the wrapper)
#   IPi    cand-imag with the p4/p5 log        IPid1  cand-imag, set {0}, fallback none      IPid3  cand-imag, constant gap 1e9 m     IPn   cand-imag, noise after the tick
#   IPs    cand-imag, imagined gaps shifted by -20 m          IPc  cand-cv          IPl  cand-label
#   IPk    cand-imagkin with the p4/p5 log     IPkid1 cand-imagkin, set {0}, fallback none   IPkid3 cand-imagkin, constant gap 1e9 m  IPkn  cand-imagkin, noise after the tick
#   IPks   cand-imagkin, imagined level shifted by -20 m
# Then the dump compares for the runs that exist. Deletes nothing, stops nothing, starts one rollout at a time. Last log line: ZZ SEQ <NAME> DONE or ZZ SEQ STOP ...
set -u
NAME=${1:?name}; Q=${2:?queue}; BASE=${3:?base arm}; TREE=${4:?tree}; RUNS=${5:?runs}
SFX=${IP_SFX:-}; RSFX=${IP_REF_SFX-$SFX}
W=/workspace/v8proof; D=${IP_D:-$W/imagplan_k}; R=$W/runs/imagplan; LOG=$R/seq_$NAME.log
mkdir -p "$R"
export IP_FAMILY=decel IP_LIBDIRS=$W/distrule:$W/probe
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> "$LOG"; }
tn() { case "$1" in IPo) echo "$1$RSFX";; *) echo "$1$SFX";; esac; }
# The anchored-arm launcher of note ADD37_N3 needs a free GPU slot with 28 GB of memory after chain actcons_c0 has ended: while actcons_c0 is done and actcons_c1 has not
# been started (and no not-started marker exists), wait (up to 60 min) before a run is started.
yield_slot() { local n=0; while [ -f $W/chains/actcons_c0/actcons_c0_DONE.txt ] && [ ! -f $W/runs/actcons_z/ACTCONS_C1_STARTED.txt ] && [ ! -f $W/runs/actcons_z/ACTCONS_C_NOT_STARTED.txt ] && [ ! -f $W/chains/actcons_c0/actcons_c0_STOP.txt ]; do
    n=$((n+1)); [ $n = 1 ] && say "yields the slot to the launcher of chain actcons_c1"; [ $n -gt 60 ] && { say "yield bound reached, going on"; break; }; sleep 60; done; }
one() { # tag mode [VAR=value ...]
  local tag=$1 mode=$2; shift 2
  local new n=0 rc
  new=$(tn "$tag")_$BASE
  [ -f "$R/t1/$new/T1_DONE" ] && { say "$new has T1_DONE, kept"; return 0; }
  [ -e "$R/t1/$new" ] && { say "ZZ SEQ STOP $new: output folder exists without T1_DONE"; return 9; }
  yield_slot
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
run_tag() {
  case "$1" in
    IPo)    one IPo off ;;
    IPi)    one IPi cand-imag IP_P45=1 ;;
    IPid1)  one IPid1 cand-imag IP_CAND_SET=0 IP_FALLBACK=none ;;
    IPid3)  one IPid3 cand-imag IP_CONST_GAP=1000000000 ;;
    IPn)    one IPn cand-imag IP_NOISE=0 ;;
    IPs)    one IPs cand-imag IP_SHIFT_M=-20 ;;
    IPc)    one IPc cand-cv ;;
    IPl)    one IPl cand-label ;;
    IPk)    one IPk cand-imagkin IP_P45=1 ;;
    IPkid1) one IPkid1 cand-imagkin IP_CAND_SET=0 IP_FALLBACK=none ;;
    IPkid3) one IPkid3 cand-imagkin IP_CONST_GAP=1000000000 ;;
    IPkn)   one IPkn cand-imagkin IP_NOISE=0 ;;
    IPks)   one IPks cand-imagkin IP_SHIFT_M=-20 ;;
    *)      say "ZZ SEQ STOP unknown tag $1"; return 6 ;;
  esac
}
dd() { local t; t=$(tn "$1"); echo "$R/t1/${t}_$BASE/dump_${t}_${BASE}_m10_causal_heads"; }
cmpd() { [ -d "$(dd "$1")" ] && [ -d "$(dd "$2")" ] || return 0; say "cmp $(tn "$1") against $(tn "$2") ($3): $(/workspace/venv/bin/python "$W/distrule/cmp_dumps.py" "$(dd "$1")" "$(dd "$2")" 2>&1 | tail -n 1 | cut -c1-200)"; }
say "seq $NAME start (family decel, tool folder $D): $Q/$BASE tree $TREE runs '$RUNS' suffix '$SFX' ref suffix '$RSFX' imag_plan $(sha256sum "$D/imag_plan.py" | cut -c1-16) wrapper $(sha256sum "$D/v8f_t1_eval_imagplan.py" | cut -c1-16)"
for t in $RUNS; do run_tag "$t" || exit 1; done
cmpd IPid1 IPo "p1 first part, must be equal"
cmpd IPid3 IPo "p1b, must be equal"
cmpd IPn IPi "p3, must be equal"
cmpd IPi IPo "firing check, expected to differ"
cmpd IPs IPi "firing check, expected to differ"
cmpd IPkid1 IPo "p1 first part of cand-imagkin, must be equal"
cmpd IPkid3 IPo "p1b of cand-imagkin, must be equal"
cmpd IPkn IPk "p3 of cand-imagkin, must be equal"
cmpd IPk IPo "firing check, expected to differ"
cmpd IPks IPk "firing check, expected to differ"
cmpd IPk IPi "cand-imagkin against cand-imag, expected to differ"
say "ZZ SEQ $NAME DONE"
touch "$R/SEQ_${NAME}_DONE.txt"
