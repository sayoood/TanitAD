#!/bin/bash
# post_chain.sh: everything after the rollouts, with NO model turn: wait for both lanes -> verify -> pack -> Read A rows + S-POS (2 procs) -> t1kf + S-POS repro (2 procs)
#                -> C1 t1kf + tacdk (2 procs) -> T0 beside -> controls -> build -> cleanup (tar + remove the per-episode dump dirs this task created).  Ends with ZZPOST-DONE.
# Re-runnable: each stage logs to logs/post_*.log; a failed stage does not stop the later ones (the build step reports what is missing).
PK=/e/Projects/TanitAD-artifacts/daghacc
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMP=$PK/pt_tmp TEMP=$PK/pt_tmp
cd $PK
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $PK/logs/post_chain.log; }
say "post_chain start"
for i in $(seq 1 240); do
  if grep -q ZZLANE-DONE $PK/logs/lane0.log 2>/dev/null && grep -q ZZLANE-DONE $PK/logs/lane1.log 2>/dev/null; then break; fi
  sleep 15
done
grep -q ZZLANE-DONE $PK/logs/lane0.log && grep -q ZZLANE-DONE $PK/logs/lane1.log || { say "STOP: lanes not done after 60 min"; echo ZZPOST-STOP >> $PK/logs/post_chain.log; exit 1; }
say "lanes done"
$PY -I post/pack_daghacc.py > logs/post_pack.log 2>&1; say "pack rc=$?"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p raw
( $PY -I post/readA_daghacc.py DAGHACC DAGHACCSTRICT raw/readA_DAGHACC.json > logs/post_readA.log 2>&1; say "readA rc=$?" ) &
( $PY -I post/spos_daghacc.py raw/spos_daghacc.json > logs/post_spos.log 2>&1; say "spos rc=$?" ) &
wait
( bash post/run_t1kf_daghacc.sh > logs/post_t1kf.log 2>&1; say "t1kf rc=$? (see raw/t1kf_DAGHACC/run.log)" ) &
( $PY -I post/spos_daghacc.py raw/spos_repro.json --repro-dag > logs/post_spos_repro.log 2>&1; say "spos repro rc=$?" ) &
wait
( bash post/run_c1_t1kf_daghacc.sh > logs/post_c1_t1kf.log 2>&1; say "c1 t1kf rc=$?" ) &
( $PY -I post/tacdk_daghacc.py > logs/post_tacdk.log 2>&1; say "tacdk rc=$?" ) &
wait
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
bash post/t0_beside.sh > logs/post_t0.log 2>&1; say "t0 beside rc=$?"
$PY -I post/controls_daghacc.py > logs/post_controls.log 2>&1; say "controls rc=$?"
$PY -I post/build_daghacc.py > logs/post_build.log 2>&1; say "build rc=$?"
# cleanup: only what this task created (per-episode dump dirs), after a verified tar
$PY -I post/tar_dumps.py > logs/post_tar.log 2>&1; say "tar rc=$?"
echo ZZPOST-DONE >> $PK/logs/post_chain.log
