#!/usr/bin/env bash
# wait_actlevel.sh : dev-box waiter (script polling, no model turns). Card PREREG_V8F_PROOF_ADD37, note N6; pod launcher actlevel/ops/pod_actlevel_auto.sh.
# Waits for chains actlevel_0 and actlevel_1 (RLVa, RLVb, RLAh at bases 0 and 1; bound 30 h), then: pod CPU imagination dumps (idempotent), pull with sha256,
# reader of card ADD36 (controls first), pairs against RLFT and RLAm (reads of record, actcons/read_rl) and against RLA3 (actcons/read_f), T1-kin pulls to F:,
# family and headway reads with the second-version readers. One Python job at a time, each only with 2.5 GB of commit memory free.
# It starts no GPU job and deletes nothing. Last line: ZZ ACTLEVEL READS DONE | ZZ ACTLEVEL NOTHING TO READ | ZZ ACTLEVEL NOT STARTED | ZZ POD-IDLE ...
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=$H/imagplan/pull_t1
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
SEEDS=0,1; S1=0; S2=1
has() { echo "$S" | grep -q "^$1\$"; }
k=0; idle=0
while :; do
  k=$((k+1))
  S=$(pod 60 'cd /workspace/v8proof; for c in actlevel_0 actlevel_1; do d=chains/$c; if [ -f $d/${c}_DONE.txt ]; then echo $c=DONE; elif [ -f $d/${c}_STOP.txt ]; then echo $c=STOP; elif [ -d $d ]; then echo $c=RUN; else echo $c=NONE; fi; done; [ -f runs/actlevel_z/ACTLEVEL_NOT_STARTED.txt ] && echo AUTO=REFUSED; grep -aq "auto end" runs/actlevel_z/actlevel_auto.log 2>/dev/null && echo AUTO=END; echo GPUJOBS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); echo "LAST $(tail -n 1 runs/actlevel_z/actlevel_auto.log 2>/dev/null | cut -c1-110)"')
  [ $((k % 3)) = 1 ] && echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-330)"
  if [ -n "$S" ]; then
    n=0
    for b in 0 1; do
      if has actlevel_$b=DONE || has actlevel_$b=STOP; then n=$((n+1))
      elif has AUTO=REFUSED && has actlevel_$b=NONE; then n=$((n+1)); fi
    done
    [ $n = 2 ] && break
    if has GPUJOBS=0 && echo "$S" | grep -q "=RUN\$"; then idle=$((idle+1)); else idle=0; fi
    [ $idle -ge 4 ] && { echo "ZZ POD-IDLE actlevel: a chain is in the running state and the pod shows no GPU job on four polls in a row ($(ts))"; exit 3; }
  fi
  [ "$k" -gt 360 ] && { echo "[$(ts)] time limit; reading what exists"; break; }
  sleep 300
done
echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-330)"
pod 60 'cd /workspace/v8proof; cat runs/actlevel_z/ACTLEVEL_NOT_STARTED.txt 2>/dev/null | cut -c1-240; grep -a "SELFCHECK\|dry gate\|chain started\|pins hold\|queue of card\|slot free" runs/actlevel_z/actlevel_auto.log 2>/dev/null | cut -c1-300; grep -a "ARM DONE\|STOP" chains/actlevel_0/chain.log chains/actlevel_1/chain.log 2>/dev/null | cut -c1-200'
if has AUTO=REFUSED && has actlevel_0=NONE && has actlevel_1=NONE; then echo "ZZ ACTLEVEL NOT STARTED: the launcher refused ($(ts))"; exit 0; fi
D=$H/imaghead/dump; R=$H/actlevel/read; mkdir -p "$D" "$R" "$PULL"
freemb() { powershell.exe -NoProfile -Command "\$o = Get-CimInstance Win32_OperatingSystem; [int](\$o.FreeVirtualMemory / 1024)" 2>/dev/null | tr -d '\r'; }
gate() { local n=0 f; while true; do f=$(freemb); [ "${f:-0}" -ge 2560 ] && return 0; n=$((n+1)); [ "$n" -gt 30 ] && { echo "[$(ts)] ZZ ACTLEVEL STOP: under 2.5 GB commit free for 30 min"; exit 8; }; sleep 60; done; }
dump() { # chain arm... : pod CPU dumps (idempotent), then pull with sha256 from the SHA lines
  local q=$1 out try; shift
  for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
    out=$(timeout 1500 bash "$P" "bash -s -- $q tree_leadsup $*" < $H/imaghead/ops/pod_imag_dump_arm.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r')
    echo "$out" | cut -c1-330 >> "$R/pod_dump.out"
    echo "$out" | grep -q "DUMP-NOT-RUN" || break
    echo "[$(ts)] pod memory low, dump of $* later"; sleep 600
  done
  echo "$out" | grep "c2b\|^   c1\|no checkpoint" | cut -c1-300
  echo "$out" | grep "^   SHA " | while read -r _ a s1 s2; do
    for pr in "DUMP_$a.npz:$s1" "DUMP_$a.npz.json:$s2"; do f=${pr%%:*}; s=${pr##*:}
      [ -f "$D/$f" ] && [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && continue
      timeout 300 bash $C /workspace/v8proof/runs/imag_z/dump/$f "$D/$f" < /dev/null > /dev/null 2>&1; [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
    done
  done
}
have() { local a; for a in "$@"; do [ -f "$D/DUMP_${a}_s$S1.npz" ] && [ -f "$D/DUMP_${a}_s$S2.npz" ] || return 1; done; return 0; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
RW=$W/actlevel/read; DW=$W/imaghead/dump; REC=$W/actcons/read_rl; RF=$W/actcons/read_f
rd() { have "$1" || { echo "[$(ts)] $1: dumps of both bases not present, no read"; return 1; }
  gate; ( cd $H/imaghead && "$PY" imag_read_r2.py read --dump "$DW/DUMP_$1_s$S1.npz" "$DW/DUMP_$1_s$S2.npz" --arm "$1" --seeds $SEEDS --out-dir "$RW/$1" > "$R/$1.log" 2>&1 ); echo "[$(ts)] read $1 rc $? : $(tail -n 1 "$R/$1.log" | cut -c1-260)"; }
pr() { # a-arm b-arm dir-of-b-read
  [ -f "$R/$1/RESULT_IMAG_$1.json" ] && [ -f "$3/RESULT_IMAG_$2.json" ] || { echo "[$(ts)] pair $1 minus $2: a read is missing"; return 1; }
  gate; ( cd $H/imaghead && "$PY" imag_read_r2.py pair --a-dir "$RW/$1" --b-dir "$3" --a-arm "$1" --b-arm "$2" --out-dir "$RW/PAIR_$1_$2" > "$R/PAIR_$1_$2.log" 2>&1 ); echo "[$(ts)] pair $1 minus $2 rc $? : $(tail -n 1 "$R/PAIR_$1_$2.log" | cut -c1-300)"; }
pull() { local q=$1 a; shift; for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done; }
havet() { local a; for a in "$@"; do [ -f "$PULL/${a}_s$S1/PULLED.txt" ] && [ -f "$PULL/${a}_s$S2/PULLED.txt" ] || return 1; done; return 0; }
dump actlevel_0 RLVa_s0 RLVb_s0 RLAh_s0
dump actlevel_1 RLVa_s1 RLVb_s1 RLAh_s1
for a in RLVa RLVb RLAh; do rd $a; done
for a in RLVa RLVb RLAh; do pr $a RLFT "$REC/RLFT"; pr $a RLAm "$REC/RLAm"; pr $a RLA3 "$RF/RLA3"; done
pull actlevel_0 RLVa_s0 RLVb_s0 RLAh_s0
pull actlevel_1 RLVa_s1 RLVb_s1 RLAh_s1
PAIRS=""; for a in RLVa RLVb RLAh; do havet $a && PAIRS="${PAIRS:+$PAIRS,}$a:RLFT,$a:RLAm"; done
if [ -n "$PAIRS" ]; then
  gate; ( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds $SEEDS --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw_r2" > "$R/hw_r2.log" 2>&1 ); echo "[$(ts)] headway r2 rc $? : $(tail -n 1 "$R/hw_r2.log" | cut -c1-200)"
  gate; ( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds $SEEDS --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3_r2" > "$R/q3_r2.log" 2>&1 ); echo "[$(ts)] families r2 rc $? : $(tail -n 1 "$R/q3_r2.log" | cut -c1-200)"
  echo "ZZ ACTLEVEL READS DONE $(ts) pairs $PAIRS"
else echo "ZZ ACTLEVEL NOTHING TO READ: the T1-kin dumps of the arms at seeds $SEEDS are not all pulled ($(ts))"; fi
