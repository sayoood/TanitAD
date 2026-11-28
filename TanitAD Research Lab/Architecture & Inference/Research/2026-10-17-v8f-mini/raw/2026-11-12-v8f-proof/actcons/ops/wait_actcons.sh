#!/usr/bin/env bash
# wait_actcons.sh <stage> : dev-box waiter (script polling, no model turns; bounded 24 h). Card PREREG_V8F_PROOF_ADD37 with note N1.
# stage 1: once LA_s1 (chain actcons_a) and LFT_s1 (chain leadcond_a) are done: imagined-lead dumps on the pod CPU, pull, reads of LFT and LA and their pair, family reads.
# stage 2: once chain actcons_a and chain leadcond_a have ended: the same for LFTb and LA01 and all pairs. It starts no training and no GPU job on the pod.
ST=${1:?stage 1 or 2}
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
  S=$(pod 60 "cd /workspace/v8proof; echo \"LA1=\$(grep -a -c 'ARM DONE LA_s1' chains/actcons_a/chain.log 2>/dev/null) LFT1=\$(grep -a -c 'ARM DONE LFT_s1' chains/leadcond_a/chain.log 2>/dev/null) \$(ls chains/actcons_a/actcons_a_DONE.txt chains/actcons_a/actcons_a_STOP.txt chains/leadcond_a/leadcond_a_DONE.txt chains/leadcond_a/leadcond_a_STOP.txt runs/actcons_z/ACTCONS_NOT_STARTED.txt runs/actcons_z/ACTCONS_STARTED.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' ')\"; tail -n 1 chains/actcons_a/chain.log 2>/dev/null | cut -c1-100; tail -n 1 runs/actcons_z/actcons_auto.log | cut -c1-160")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-420)"
  echo "$S" | grep -q "ACTCONS_NOT_STARTED\|actcons_a_STOP" && { echo "ZZ ACTCONS STAGE $ST: the chain did not start or stopped; reading what exists"; break; }
  if [ "$ST" = 1 ]; then echo "$S" | grep -q "LA1=[1-9]" && echo "$S" | grep -q "LFT1=[1-9]" && break
  else echo "$S" | grep -q "actcons_a_DONE" && echo "$S" | grep -q "leadcond_a_DONE\|leadcond_a_STOP" && break; fi
  [ "$k" -gt 288 ] && { echo "ZZ ACTCONS STAGE $ST TIME LIMIT (24 h); reading what exists"; break; }
  sleep 300
done
D=$H/imaghead/dump; R=$H/actcons/read; mkdir -p "$D" "$R"
dump() { # chain arm... : pod CPU dumps (idempotent), then pull with sha256 from the SHA lines
  local q=$1 out try; shift
  for try in 1 2 3 4 5 6; do
    out=$(timeout 900 bash "$P" "bash -s -- $q tree_leadsup $*" < $H/imaghead/ops/pod_imag_dump_arm.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r')
    echo "$out" | cut -c1-330 >> "$R/pod_dump_stage$ST.out"
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
rd() { have "$1" || { echo "[$(ts)] $1: dumps of both bases not present, no read"; return 1; }
  ( cd $H/imaghead && "$PY" imag_read.py read --dump "$D/DUMP_$1_s0.npz" "$D/DUMP_$1_s1.npz" --arm "$1" --seeds 0,1 --out-dir "$R/$1" > "$R/$1.log" 2>&1 ); echo "[$(ts)] read $1 rc $? : $(tail -n 1 "$R/$1.log" | cut -c1-260)"; }
pr() { [ -f "$R/$1/RESULT_IMAG_$1.json" ] && [ -f "$R/$2/RESULT_IMAG_$2.json" ] || return 1
  ( cd $H/imaghead && "$PY" imag_read.py pair --a-dir "$R/$1" --b-dir "$R/$2" --out-dir "$R/PAIR_$1_$2" > "$R/PAIR_$1_$2.log" 2>&1 ); echo "[$(ts)] pair $1 minus $2 rc $? : $(tail -n 1 "$R/PAIR_$1_$2.log" | cut -c1-300)"; }
pull() { local q=$1 a; shift; for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done; }
havet() { local a; for a in "$@"; do [ -f "$PULL/${a}_s0/PULLED.txt" ] && [ -f "$PULL/${a}_s1/PULLED.txt" ] || return 1; done; return 0; }
hw() { ( cd $H/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$R/hw_$1" > "$R/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$R/hw_$1.log" | cut -c1-200)"; }
q3() { ( cd $H/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$R/q3_$1" > "$R/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$R/q3_$1.log" | cut -c1-200)"; }
if [ "$ST" = 1 ]; then
  dump leadcond_a LFT_s0 LFT_s1; dump actcons_a LA_s0 LA_s1
  rd LFT; rd LA; pr LA LFT
  pull leadcond_a LFT_s0 LFT_s1; pull actcons_a LA_s0 LA_s1
  havet LFT LA && { q3 s1 "LA:LFT"; hw s1 "LA:LFT"; }
else
  dump leadcond_a LFT_s0 LFT_s1 LFTb_s0 LFTb_s1; dump actcons_a LA_s0 LA_s1 LA01_s0 LA01_s1
  rd LFT; rd LFTb; rd LA; rd LA01; pr LA LFT; pr LFTb LFT; pr LA01 LFT; pr LA LA01
  pull leadcond_a LFT_s0 LFT_s1 LFTb_s0 LFTb_s1; pull actcons_a LA_s0 LA_s1 LA01_s0 LA01_s1
  if havet LFT LFTb LA LA01; then q3 s2 "LA:LFT,LA01:LFT,LFTb:LFT"; hw s2 "LA:LFT,LA01:LFT,LFTb:LFT"; elif havet LFT LA; then q3 s2 "LA:LFT"; hw s2 "LA:LFT"; fi
fi
echo "ZZ ACTCONS STAGE $ST READS DONE $(ts)"
