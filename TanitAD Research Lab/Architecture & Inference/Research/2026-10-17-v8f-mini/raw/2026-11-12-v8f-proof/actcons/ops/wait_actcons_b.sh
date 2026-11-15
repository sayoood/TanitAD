#!/usr/bin/env bash
# wait_actcons_b.sh : dev-box waiter (script polling, no model turns; bounded 30 h). Card PREREG_V8F_PROOF_ADD37 with note N3 (the term on base RL).
# Waits for the pod chains actcons_b0 and actcons_b1 (or their stop / not-started marker), then: pod CPU imagination dumps of the eight arms (idempotent),
# pull with sha256, reads with the tool of card ADD36, pairs with the arm names given, T1-kin pulls, family and headway reads. It starts no GPU job.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof; ls chains/actcons_b0/actcons_b0_DONE.txt chains/actcons_b0/actcons_b0_STOP.txt chains/actcons_b1/actcons_b1_DONE.txt chains/actcons_b1/actcons_b1_STOP.txt runs/actcons_z/ACTCONS_B_NOT_STARTED.txt 2>/dev/null | sed 's|.*/||'; tail -n 1 runs/actcons_z/actcons_b_auto.log 2>/dev/null | cut -c1-150; grep -a 'ARM DONE\|train ' chains/actcons_b0/chain.log chains/actcons_b1/chain.log 2>/dev/null | tail -n 1 | cut -c1-110")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-420)"
  echo "$S" | grep -q "ACTCONS_B_NOT_STARTED\|actcons_b0_STOP\|actcons_b1_DONE\|actcons_b1_STOP" && break
  [ "$k" -gt 360 ] && { echo "ZZ ACTCONS-B TIME LIMIT (30 h); reading what exists"; break; }
  sleep 300
done
pod 60 "cd /workspace/v8proof; cat runs/actcons_z/ACTCONS_B_NOT_STARTED.txt chains/actcons_b0/actcons_b0_STOP.txt chains/actcons_b1/actcons_b1_STOP.txt 2>/dev/null | cut -c1-240; grep -a 'SELFCHECK\|dry gate\|chain started' runs/actcons_z/actcons_b_auto.log | cut -c1-300; grep -a 'ARM DONE\|STOP' chains/actcons_b0/chain.log chains/actcons_b1/chain.log 2>/dev/null | cut -c1-120"
D=$H/imaghead/dump; R=$H/actcons/read_rl; mkdir -p "$D" "$R"
dump() { # chain arm... : pod CPU dumps (idempotent), then pull with sha256 from the SHA lines
  local q=$1 out try; shift
  for try in 1 2 3 4 5 6 7 8 9; do
    out=$(timeout 1500 bash "$P" "bash -s -- $q tree_leadsup $*" < $H/imaghead/ops/pod_imag_dump_arm.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r')
    echo "$out" | cut -c1-330 >> "$R/pod_dump.out"
    echo "$out" | grep -q "DUMP-NOT-RUN" || break
    echo "[$(ts)] pod memory low, dump of $* later"; sleep 600
  done
  echo "$out" | grep "c2b\|^   c1" | cut -c1-300
  echo "$out" | grep "^   SHA " | while read -r _ a s1 s2; do
    for pr in "DUMP_$a.npz:$s1" "DUMP_$a.npz.json:$s2"; do f=${pr%%:*}; s=${pr##*:}
      [ -f "$D/$f" ] && [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && continue
      timeout 300 bash $C /workspace/v8proof/runs/imag_z/dump/$f "$D/$f" < /dev/null; [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
    done
  done
}
have() { local a; for a in "$@"; do [ -f "$D/DUMP_${a}_s0.npz" ] && [ -f "$D/DUMP_${a}_s1.npz" ] || return 1; done; return 0; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
RW=$W/actcons/read_rl; DW=$W/imaghead/dump
rd() { have "$1" || { echo "[$(ts)] $1: dumps of both bases not present, no read"; return 1; }
  ( cd $H/imaghead && "$PY" imag_read.py read --dump "$DW/DUMP_$1_s0.npz" "$DW/DUMP_$1_s1.npz" --arm "$1" --seeds 0,1 --out-dir "$RW/$1" > "$R/$1.log" 2>&1 ); echo "[$(ts)] read $1 rc $? : $(tail -n 1 "$R/$1.log" | cut -c1-260)"; }
pr() { [ -f "$R/$1/RESULT_IMAG_$1.json" ] && [ -f "$R/$2/RESULT_IMAG_$2.json" ] || return 1
  ( cd $H/imaghead && "$PY" imag_read.py pair --a-dir "$RW/$1" --b-dir "$RW/$2" --a-arm "$1" --b-arm "$2" --out-dir "$RW/PAIR_$1_$2" > "$R/PAIR_$1_$2.log" 2>&1 ); echo "[$(ts)] pair $1 minus $2 rc $? : $(tail -n 1 "$R/PAIR_$1_$2.log" | cut -c1-300)"; }
pull() { local q=$1 a; shift; for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done; }
havet() { local a; for a in "$@"; do [ -f "$PULL/${a}_s0/PULLED.txt" ] && [ -f "$PULL/${a}_s1/PULLED.txt" ] || return 1; done; return 0; }
hw() { ( cd $H/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw_$1" > "$R/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$R/hw_$1.log" | cut -c1-200)"; }
q3() { ( cd $H/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3_$1" > "$R/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$R/q3_$1.log" | cut -c1-200)"; }
dump actcons_b0 RLFT_s0 RLA_s0 RLAm_s0 RLFTb_s0
dump actcons_b1 RLFT_s1 RLA_s1 RLAm_s1 RLFTb_s1
rd RLFT; rd RLA; rd RLAm; rd RLFTb
pr RLA RLFT; pr RLAm RLFT; pr RLFTb RLFT; pr RLA RLAm
pull actcons_b0 RLFT_s0 RLA_s0 RLAm_s0 RLFTb_s0
pull actcons_b1 RLFT_s1 RLA_s1 RLAm_s1 RLFTb_s1
if havet RLFT RLA RLAm RLFTb; then q3 rl "RLA:RLFT,RLAm:RLFT,RLFTb:RLFT"; hw rl "RLA:RLFT,RLAm:RLFT,RLFTb:RLFT"; elif havet RLFT RLA; then q3 rl "RLA:RLFT"; hw rl "RLA:RLFT"; else echo "[$(ts)] T1-kin dumps of RLFT and RLA not all pulled: no family read"; fi
echo "ZZ ACTCONS-B READS DONE $(ts)"
