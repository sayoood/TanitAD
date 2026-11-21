#!/usr/bin/env bash
# wait_actcons_g.sh <f|e> : dev-box waiter (script polling, no model turns). Card PREREG_V8F_PROOF_ADD37 with notes N4 and N5; pod launcher pod_actcons_g_auto.sh.
#   f : waits for chain actcons_f (dose arms RLA3 0.001 and RLA2 0.01 at bases 0 and 1), then the reads at seeds 0,1 into actcons/read_f/ (pairs against the reads of
#       record of RLFT and RLAm in actcons/read_rl/). Bound 14 h.
#   e : waits for chains actcons_e2 and actcons_e3 (RLFT, RLAm, RLA3, RLA2 on the RL bases of seeds 2 and 3, chains imag_b2 and imag_b3), then the reads at seeds 2,3
#       into actcons/read_g/. Bound 30 h.
# Both: pod CPU imagination dumps (idempotent), pull with sha256, reader of card ADD36, pairs, T1-kin pulls, family and headway reads. It starts no GPU job and deletes
# nothing. It also ends with ZZ POD-IDLE if a chain is in the running state while the pod shows no GPU job on four polls in a row (20 min).
# Last line: ZZ ACTCONS-G <stage> READS DONE | ZZ POD-IDLE ... | ZZ ACTCONS-G <stage> NOTHING TO READ.
ST=${1:?stage f or e}
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
case $ST in f) LIM=168; SEEDS=0,1; S1=0; S2=1; RN=read_f ;; e) LIM=360; SEEDS=2,3; S1=2; S2=3; RN=read_g ;; *) echo "stage f or e"; exit 2 ;; esac
has() { echo "$S" | grep -q "^$1\$"; }
k=0; idle=0
while :; do
  k=$((k+1))
  S=$(pod 60 'cd /workspace/v8proof; for c in imag_b2 imag_b3 actcons_f actcons_e2 actcons_e3; do d=chains/$c; if [ -f $d/${c}_DONE.txt ]; then echo $c=DONE; elif [ -f $d/${c}_STOP.txt ]; then echo $c=STOP; elif [ -d $d ]; then echo $c=RUN; else echo $c=NONE; fi; done; [ -f runs/actcons_z/ACTCONS_G_NOT_STARTED.txt ] && echo AUTO=REFUSED; grep -aq "auto end" runs/actcons_z/actcons_g_auto.log && echo AUTO=END; echo GPUJOBS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); echo "LAST $(tail -n 1 runs/actcons_z/actcons_g_auto.log 2>/dev/null | cut -c1-110)"')
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-330)"
  if [ -n "$S" ]; then
    if [ $ST = f ]; then
      { has actcons_f=DONE || has actcons_f=STOP; } && break
      has AUTO=REFUSED && has actcons_f=NONE && break
    else
      n=0
      for b in 2 3; do
        if has actcons_e$b=DONE || has actcons_e$b=STOP || has imag_b$b=STOP; then n=$((n+1))
        elif { has AUTO=REFUSED || has AUTO=END; } && has actcons_e$b=NONE; then n=$((n+1)); fi
      done
      [ $n = 2 ] && break
    fi
    if has GPUJOBS=0 && echo "$S" | grep -q "=RUN\$"; then idle=$((idle+1)); else idle=0; fi
    [ $idle -ge 4 ] && { echo "ZZ POD-IDLE $ST: a chain is in the running state and the pod shows no GPU job on four polls in a row ($(ts))"; exit 3; }
  fi
  [ "$k" -gt $LIM ] && { echo "ZZ ACTCONS-G $ST TIME LIMIT; reading what exists"; break; }
  sleep 300
done
pod 60 'cd /workspace/v8proof; cat runs/actcons_z/ACTCONS_G_NOT_STARTED.txt 2>/dev/null | cut -c1-240; grep -a "SELFCHECK\|dry gate\|chain started\|left out\|is done" runs/actcons_z/actcons_g_auto.log | cut -c1-300; grep -a "ARM DONE\|STOP" chains/imag_b2/chain.log chains/imag_b3/chain.log chains/actcons_f/chain.log chains/actcons_e2/chain.log chains/actcons_e3/chain.log 2>/dev/null | cut -c1-200'
D=$H/imaghead/dump; R=$H/actcons/$RN; mkdir -p "$D" "$R"
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
RW=$W/actcons/$RN; DW=$W/imaghead/dump; REC=$W/actcons/read_rl
rd() { have "$1" || { echo "[$(ts)] $1: dumps of both bases not present, no read"; return 1; }
  ( cd $H/imaghead && "$PY" imag_read.py read --dump "$DW/DUMP_$1_s$S1.npz" "$DW/DUMP_$1_s$S2.npz" --arm "$1" --seeds $SEEDS --out-dir "$RW/$1" > "$R/$1.log" 2>&1 ); echo "[$(ts)] read $1 rc $? : $(tail -n 1 "$R/$1.log" | cut -c1-260)"; }
pr() { # a-arm b-arm [dir of b's read]
  local bd=${3:-$RW/$2}
  [ -f "$R/$1/RESULT_IMAG_$1.json" ] && [ -f "$bd/RESULT_IMAG_$2.json" ] || { echo "[$(ts)] pair $1 minus $2: a read is missing"; return 1; }
  ( cd $H/imaghead && "$PY" imag_read.py pair --a-dir "$RW/$1" --b-dir "$bd" --a-arm "$1" --b-arm "$2" --out-dir "$RW/PAIR_$1_$2" > "$R/PAIR_$1_$2.log" 2>&1 ); echo "[$(ts)] pair $1 minus $2 rc $? : $(tail -n 1 "$R/PAIR_$1_$2.log" | cut -c1-300)"; }
pull() { local q=$1 a; shift; for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done; }
havet() { local a; for a in "$@"; do [ -f "$PULL/${a}_s$S1/PULLED.txt" ] && [ -f "$PULL/${a}_s$S2/PULLED.txt" ] || return 1; done; return 0; }
hw() { ( cd $H/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$1" --seeds $SEEDS --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw" > "$R/hw.log" 2>&1 ); echo "[$(ts)] headway rc $? : $(tail -n 1 "$R/hw.log" | cut -c1-200)"; }
q3() { ( cd $H/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$1" --seeds $SEEDS --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3" > "$R/q3.log" 2>&1 ); echo "[$(ts)] families rc $? : $(tail -n 1 "$R/q3.log" | cut -c1-200)"; }
if [ $ST = f ]; then
  dump actcons_f RLA3_s0 RLA2_s0 RLA3_s1 RLA2_s1
  rd RLA3; rd RLA2
  pr RLA3 RLFT "$REC/RLFT"; pr RLA3 RLAm "$REC/RLAm"; pr RLA2 RLFT "$REC/RLFT"; pr RLA2 RLAm "$REC/RLAm"
  pull actcons_f RLA3_s0 RLA2_s0 RLA3_s1 RLA2_s1
  PAIRS=""; havet RLA3 && PAIRS="RLA3:RLFT,RLA3:RLAm"; havet RLA2 && PAIRS="${PAIRS:+$PAIRS,}RLA2:RLFT,RLA2:RLAm"
else
  dump imag_b2 RL_s2; dump imag_b3 RL_s3
  dump actcons_e2 RLFT_s2 RLAm_s2 RLA3_s2 RLA2_s2
  dump actcons_e3 RLFT_s3 RLAm_s3 RLA3_s3 RLA2_s3
  rd RL; rd RLFT; rd RLAm; rd RLA3; rd RLA2
  pr RLAm RLFT; pr RLFT RL; pr RLA3 RLFT; pr RLA3 RLAm; pr RLA2 RLFT; pr RLA2 RLAm
  pull actcons_e2 RLFT_s2 RLAm_s2 RLA3_s2 RLA2_s2
  pull actcons_e3 RLFT_s3 RLAm_s3 RLA3_s3 RLA2_s3
  PAIRS=""
  if havet RLFT; then for a in RLAm RLA3 RLA2; do havet $a && PAIRS="${PAIRS:+$PAIRS,}$a:RLFT"; done; fi
fi
if [ -n "$PAIRS" ]; then q3 "$PAIRS"; hw "$PAIRS"; echo "ZZ ACTCONS-G $ST READS DONE $(ts) pairs $PAIRS"
else echo "ZZ ACTCONS-G $ST NOTHING TO READ: the T1-kin dumps of the arms at seeds $SEEDS are not all pulled ($(ts))"; fi
