#!/usr/bin/env bash
# pod_imagplan_seq_q.sh (note PREREG_V8F_PROOF_ADD40_N7; form of pod_imagplan_seq6.sh with the tags IPo IPc IPl IPcq IPcqid IPcq0 only) <NAME> <QUEUE> <BASEARM> <TREE> "<RUNS>" : runs ON THE POD (nohup), written by the lead. Card PREREG_V8F_PROOF_ADD40 with notes N1 to N4.
# Sixth form = pod_imagplan_seq5.sh (memory guard) with the off run started WITHOUT the family variable (the run script refuses it with mode off); IP_MAX_WINDOWS, if set, reaches every run (note N5: p3 on the first 600 windows). Reason (MEASURED 2026-11-15T23:28:17Z): the run IPknx_LFT_s0 (noise after the tick: it copies the tokens of every
# window) was killed by the container's memory limit beside two trainers (20 GB and 5 GB resident); the trainers survived. A tool run must never endanger a trainer:
#   * a run starts only with IP_NEED_GB (default 24) GB of the memory limit free, a noise run (IPn, IPkn) only with IP_NEED_NOISE_GB (default 40);
#   * while a run is alive the free memory is read every second; below IP_FLOOR_GB (default 8) the run of THIS sequence is stopped (TERM, then KILL) and the sequence ends;
#   * the peak resident memory of every run is written to the log.
# Tool folder IP_D (default /workspace/v8proof/imagplan_q), IP_FAMILY=decel, serial runs through pod_imagplan_run_q.sh, tags (this form: IPo IPc IPl IPcq IPcqid IPcq0 only; IPcq = cand-cvq, IPcqid = cand-cvq with the constant 1e9, IPcq0 = with 0):
#   IPo off | IPi cand-imag + p4/p5 log | IPid1 cand-imag set {0} no fallback | IPid3 cand-imag constant gap 1e9 | IPn cand-imag noise | IPs cand-imag shift -20 | IPc cand-cv | IPl cand-label
#   IPk cand-imagkin + p4/p5 log | IPkid1 | IPkid3 | IPkn | IPks (the same options with cand-imagkin).
# Env: IP_SFX (suffix of every run but IPo), IP_REF_SFX (suffix of IPo; default = IP_SFX). Deletes nothing; stops only its own run. Last log line: ZZ SEQ <NAME> DONE or ZZ SEQ STOP ...
set -u
NAME=${1:?name}; Q=${2:?queue}; BASE=${3:?base arm}; TREE=${4:?tree}; RUNS=${5:?runs}
SFX=${IP_SFX:-}; RSFX=${IP_REF_SFX-$SFX}
W=/workspace/v8proof; D=${IP_D:-$W/imagplan_q}; R=$W/runs/imagplan; LOG=$R/seq_$NAME.log
NEED=${IP_NEED_GB:-24}; NEED_NOISE=${IP_NEED_NOISE_GB:-40}; FLOOR=${IP_FLOOR_GB:-8}
mkdir -p "$R"
export IP_FAMILY=decel IP_LIBDIRS=$W/distrule:$W/probe
# the banked head of the checkpoint: LS_* in qhead_dose_ls/heads, LSD_* in qhead_dose/heads (override the folder with IP_QHEAD_DIR)
case "$BASE" in LS_s*) QDIR=${IP_QHEAD_DIR:-$W/qhead_dose_ls/heads};; LSD_s*) QDIR=${IP_QHEAD_DIR:-$W/qhead_dose/heads};; *) QDIR=${IP_QHEAD_DIR:-/nonexistent};; esac
QFILE=$QDIR/$BASE.qhead.npz
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> "$LOG"; }
tn() { case "$1" in IPo) echo "$1$RSFX";; *) echo "$1$SFX";; esac; }
freegb() { local a m; a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max); echo $(( (m - a) / 1000000000 )); }
yield_slot() { local n=0; while [ -f $W/chains/actcons_c0/actcons_c0_DONE.txt ] && [ ! -f $W/runs/actcons_z/ACTCONS_C1_STARTED.txt ] && [ ! -f $W/runs/actcons_z/ACTCONS_C_NOT_STARTED.txt ] && [ ! -f $W/chains/actcons_c0/actcons_c0_STOP.txt ]; do
    n=$((n+1)); [ $n = 1 ] && say "yields the slot to the launcher of chain actcons_c1"; [ $n -gt 60 ] && { say "yield bound reached, going on"; break; }; sleep 60; done; }
one() { # tag mode need_gb [VAR=value ...]
  local tag=$1 mode=$2 need=$3; shift 3
  local new n=0 rc pid peak=0 f rss pat
  new=$(tn "$tag")_$BASE
  [ -f "$R/t1/$new/T1_DONE" ] && { say "$new has T1_DONE, kept"; return 0; }
  [ -e "$R/t1/$new" ] && { say "ZZ SEQ STOP $new: output folder exists without T1_DONE"; return 9; }
  pat="v8f_t1_eval_imagplan_q.py .*--out-dir $R/t1/$new "
  while :; do
    yield_slot
    f=$(freegb)
    if [ "$f" -lt "$need" ]; then n=$((n+1)); [ $n = 1 ] && say "$new waits for $need GB of free memory (now $f GB)"; [ $n -gt 360 ] && { say "ZZ SEQ STOP $new: no $need GB of free memory in 6 h"; return 8; }; sleep 60; continue; fi
    env "$@" bash "$D/pod_imagplan_run_q.sh" "$new" "$Q" "$BASE" "$mode" "$TREE" > "$R/run_$new.out" 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      f=$(freegb)
      rss=$(ps -o rss= -p "$(pgrep -d, -f "$pat" 2>/dev/null)" 2>/dev/null | sort -n | tail -n 1); rss=${rss:-0}; [ "$rss" -gt "$peak" ] && peak=$rss
      if [ "$f" -lt "$FLOOR" ]; then
        pkill -TERM -f "$pat"; sleep 3; pkill -KILL -f "$pat"; wait $pid 2>/dev/null
        say "ZZ SEQ STOP $new: memory guard (free memory $f GB below $FLOOR GB; this sequence stopped its own run; peak resident $((peak / 1048576)) GB; partial folder kept)"; return 5
      fi
      sleep 1
    done
    wait $pid; rc=$?
    [ $rc = 0 ] && { say "$new done ($mode $*; peak resident $((peak / 1048576)) GB)"; return 0; }
    if [ $rc = 2 ] && [ ! -e "$R/t1/$new" ] && grep -q "REFUSED: .*\(free container memory\|GPU jobs already run\)" "$R/run_$new.out"; then
      n=$((n+1)); [ $n -gt 360 ] && { say "ZZ SEQ STOP $new: no slot after 6 h"; return 8; }
      [ $n = 1 ] && say "$new waits for a slot: $(tail -n 1 "$R/run_$new.out" | cut -c1-160)"
      sleep 60; continue
    fi
    say "ZZ SEQ STOP $new rc=$rc (peak resident $((peak / 1048576)) GB): $(tail -n 2 "$R/run_$new.out" | cut -c1-200 | tr '\n' ';')"; return 7
  done
}
run_tag() {
  case "$1" in
    IPo)    one IPo off $NEED -u IP_FAMILY ;;
    IPc)    one IPc cand-cv $NEED ;;
    IPl)    one IPl cand-label $NEED ;;
    IPcq)   one IPcq cand-cvq $NEED IP_QHEAD_FILE=$QFILE ;;
    IPcqid) one IPcqid cand-cvq $NEED IP_QHEAD_FILE=$QFILE IP_QHEAD_CONST=1000000000 ;;
    IPcq0)  one IPcq0 cand-cvq $NEED IP_QHEAD_FILE=$QFILE IP_QHEAD_CONST=0 ;;
    *)      say "ZZ SEQ STOP unknown tag $1"; return 6 ;;
  esac
}
dd() { local t; t=$(tn "$1"); echo "$R/t1/${t}_$BASE/dump_${t}_${BASE}_m10_causal_heads"; }
cmpd() { [ -f "$R/t1/$(tn "$1")_$BASE/T1_DONE" ] && [ -f "$R/t1/$(tn "$2")_$BASE/T1_DONE" ] && [ -d "$(dd "$1")" ] && [ -d "$(dd "$2")" ] || return 0
  say "cmp $(tn "$1") against $(tn "$2") ($3): $(/workspace/venv/bin/python "$W/distrule/cmp_dumps.py" "$(dd "$1")" "$(dd "$2")" 2>&1 | tail -n 1 | cut -c1-200)"; }
say "seq $NAME start (family decel, tool folder $D, memory guard: start at $NEED GB free, noise runs at $NEED_NOISE GB, stop below $FLOOR GB): $Q/$BASE tree $TREE runs '$RUNS' suffix '$SFX' ref suffix '$RSFX' imag_plan $(sha256sum "$D/imag_plan.py" | cut -c1-16) wrapper $(sha256sum "$D/v8f_t1_eval_imagplan_q.py" | cut -c1-16) qhead_model $(sha256sum "$D/qhead_model.py" | cut -c1-16)"
st=0
for t in $RUNS; do run_tag "$t" || { st=1; break; }; done
cmpd IPcqid IPc "q1, must be equal"
cmpd IPcq0 IPc "q2 firing check, expected to differ"
cmpd IPcq IPc "cand-cvq against cand-cv, expected to differ"
cmpd IPc IPo "firing check, expected to differ"
cmpd IPl IPo "firing check, expected to differ"
[ $st = 0 ] || { say "ZZ SEQ $NAME ENDED AFTER A STOP (compares above are for the runs that exist)"; touch "$R/SEQ_${NAME}_STOPPED.txt"; exit 1; }
say "ZZ SEQ $NAME DONE"
touch "$R/SEQ_${NAME}_DONE.txt"
