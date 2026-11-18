#!/usr/bin/env bash
# wait_actcons_e.sh : dev-box waiter (script polling, no model turns; bounded 30 h). Card PREREG_V8F_PROOF_ADD37 with note N4 (the low weight on RL bases of new seeds).
# Waits for the pod chains actcons_e2 and actcons_e3 (or a stop / not-started marker), then: pod CPU imagination dumps of RL_s2, RL_s3 (chain imag_b) and of RLFT and
# RLAm at both new bases (idempotent), pull with sha256, reads with the tool of card ADD36 (seeds 2,3), the pair RLAm minus RLFT, T1-kin pulls, family and headway
# reads of RLAm against RLFT. It starts no GPU job. Form of wait_actcons_c.sh. Deletes nothing. Last line: ZZ ACTCONS-E READS DONE.
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
  S=$(pod 60 "cd /workspace/v8proof; ls chains/imag_b/imag_b_STOP.txt chains/actcons_e2/actcons_e2_DONE.txt chains/actcons_e2/actcons_e2_STOP.txt chains/actcons_e3/actcons_e3_DONE.txt chains/actcons_e3/actcons_e3_STOP.txt runs/actcons_z/ACTCONS_E_NOT_STARTED.txt 2>/dev/null | sed 's|.*/||'; tail -n 1 runs/actcons_z/actcons_e_auto.log 2>/dev/null | cut -c1-150")
  echo "[$(ts)] $(echo "$S" | tr '\n' ';' | cut -c1-420)"
  echo "$S" | grep -q "ACTCONS_E_NOT_STARTED\|imag_b_STOP\|actcons_e3_DONE\|actcons_e3_STOP" && break
  [ "$k" -gt 360 ] && { echo "ZZ ACTCONS-E TIME LIMIT (30 h); reading what exists"; break; }
  sleep 300
done
pod 60 "cd /workspace/v8proof; cat runs/actcons_z/ACTCONS_E_NOT_STARTED.txt chains/imag_b/imag_b_STOP.txt chains/actcons_e2/actcons_e2_STOP.txt chains/actcons_e3/actcons_e3_STOP.txt 2>/dev/null | cut -c1-240; grep -a 'SELFCHECK\|dry gate\|chain started\|arms file' runs/actcons_z/actcons_e_auto.log | cut -c1-300; grep -a 'ARM DONE\|STOP' chains/imag_b/chain.log chains/actcons_e2/chain.log chains/actcons_e3/chain.log 2>/dev/null | cut -c1-200"
D=$H/imaghead/dump; R=$H/actcons/read_e; mkdir -p "$D" "$R"
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
have() { local a; for a in "$@"; do [ -f "$D/DUMP_${a}_s2.npz" ] && [ -f "$D/DUMP_${a}_s3.npz" ] || return 1; done; return 0; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
RW=$W/actcons/read_e; DW=$W/imaghead/dump
rd() { have "$1" || { echo "[$(ts)] $1: dumps of both new bases not present, no read"; return 1; }
  ( cd $H/imaghead && "$PY" imag_read.py read --dump "$DW/DUMP_$1_s2.npz" "$DW/DUMP_$1_s3.npz" --arm "$1" --seeds 2,3 --out-dir "$RW/$1" > "$R/$1.log" 2>&1 ); echo "[$(ts)] read $1 rc $? : $(tail -n 1 "$R/$1.log" | cut -c1-260)"; }
pr() { [ -f "$R/$1/RESULT_IMAG_$1.json" ] && [ -f "$R/$2/RESULT_IMAG_$2.json" ] || return 1
  ( cd $H/imaghead && "$PY" imag_read.py pair --a-dir "$RW/$1" --b-dir "$RW/$2" --a-arm "$1" --b-arm "$2" --out-dir "$RW/PAIR_$1_$2" > "$R/PAIR_$1_$2.log" 2>&1 ); echo "[$(ts)] pair $1 minus $2 rc $? : $(tail -n 1 "$R/PAIR_$1_$2.log" | cut -c1-300)"; }
pull() { local q=$1 a; shift; for a in "$@"; do [ -f "$PULL/$a/PULLED.txt" ] || Q=$q bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-200; done; }
havet() { local a; for a in "$@"; do [ -f "$PULL/${a}_s2/PULLED.txt" ] && [ -f "$PULL/${a}_s3/PULLED.txt" ] || return 1; done; return 0; }
hw() { ( cd $H/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 2,3 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw_$1" > "$R/hw_$1.log" 2>&1 ); echo "[$(ts)] headway $1 rc $? : $(tail -n 1 "$R/hw_$1.log" | cut -c1-200)"; }
q3() { ( cd $H/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$2" --seeds 2,3 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3_$1" > "$R/q3_$1.log" 2>&1 ); echo "[$(ts)] families $1 rc $? : $(tail -n 1 "$R/q3_$1.log" | cut -c1-200)"; }
dump imag_b RL_s2 RL_s3
dump actcons_e2 RLFT_s2 RLAm_s2
dump actcons_e3 RLFT_s3 RLAm_s3
rd RL; rd RLFT; rd RLAm
pr RLAm RLFT; pr RLFT RL
pull actcons_e2 RLFT_s2 RLAm_s2
pull actcons_e3 RLFT_s3 RLAm_s3
if havet RLFT RLAm; then q3 e "RLAm:RLFT"; hw e "RLAm:RLFT"; else echo "[$(ts)] T1-kin dumps of RLFT and RLAm at the new bases not all pulled: no family read"; fi
echo "ZZ ACTCONS-E READS DONE $(ts)"
