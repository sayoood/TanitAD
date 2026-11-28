#!/bin/bash
# R8-WISEFT stage 2 chain (dev box): after the stage-1 waiter reads ADMISSIBLE with all controls passing, ship + launch the ONE
# registered navtest roll of theta_lambda* behind the Thor GPU flock, score it on the Thor CPU backend (separate scores dir), finalize,
# pull, apply the registered navtest rule.  If stage 1 reads FAILED, exit cleanly with NOSTAGE2 (nothing is rolled).  Markers ZZWS2<STAGE>ZZ.
set -u
W=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft; LOG=$W/wiseft_stage2.log; S1LOG=$W/wiseft_waiter.log
X=/home/nvidia/r8ns/navtest_read; NX=/home/nvidia/r8ns/code/navtest_read; WC=/home/nvidia/r8ns/navtrain_sel/wiseft/code
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mark() { echo "ZZWS2$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
until grep -q 'ZZWWWEN[D]ZZ' "$S1LOG" 2>/dev/null; do
  grep -qE 'ZZWWW(ABORTED|TIMEOUT)ZZ' "$S1LOG" 2>/dev/null && { mark S1FAIL; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { mark TIMEOUT; exit 1; }
  sleep 120
done
TAG=$("$PY" -c "import json;r=json.load(open('C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/WISEFT_READ.json'));s=r['stage1'];print(s['tag'] if s['verdict']=='ADMISSIBLE' and r['controls']['all_pass'] else 'NONE')" 2>/dev/null)
[ -n "$TAG" ] || { mark READERR; exit 2; }
[ "$TAG" = NONE ] && { mark NOSTAGE2 "stage 1 did not select a lambda; no navtest read (registered)"; mark END; exit 0; }
mark SELECTED "tag=$TAG"
f=r8nt_wiseft_LAUNCH.sh; scp -q $W/code/$f tanitad-thor:$WC/$f
a=$(md5sum $W/code/$f | cut -c1-32); b=$(ssh -n tanitad-thor "md5sum $WC/$f" | cut -c1-32)
[ ${#a} -eq 32 ] && [ "$a" = "$b" ] || { mark SHIPBAD; exit 3; }
ssh -n tanitad-thor "cd $X && nohup setsid flock -o /home/nvidia/refcv7_post/thor_gpu.lock bash $WC/$f $TAG > $X/wiseft_$TAG.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 30; mark LAUNCHED
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZALLDON[E]' $X/wiseft_$TAG.out; grep -c 'ZZREFUS[E]' $X/wiseft_$TAG.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark REFUSED "$(ssh -n tanitad-thor "grep -o 'ZZREFUSE_[a-z0-9_]*' $X/wiseft_$TAG.out")"; exit 4; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && break
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { mark TIMEOUT "$r"; exit 4; }
  sleep 180
done
mark ROLLED
O=$X/wiseft_l$TAG; B=$O/bridge_navtest; S=$O/scores_navtest
A="--arm R8_LEGAL=$B/seam_R8_LEGAL.npz --arm R8_LEGAL_s1=$B/seam_R8_LEGAL_s1.npz --arm PRIOR_ha0p=$B/seam_PRIOR_ha0p.npz"
ssh -n tanitad-thor "python3 $NX/r8nt_score.py jobs --prep $X/prep --scores $S --tag w$TAG $A --out $O/score_jobs.jsonl" >> "$LOG" 2>&1
n=$(ssh -n tanitad-thor "wc -l < $O/score_jobs.jsonl" 2>/dev/null); [ "${n:-0}" = "12" ] || { mark JOBSBAD "n=$n"; exit 5; }
ssh -n tanitad-thor "mkdir -p $S && cd $S && OMP_NUM_THREADS=6 nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $O/score_jobs.jsonl --state $O/pool_state.json > $O/pool.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 60; mark POOLSTARTED
while :; do
  z=$(ssh -n tanitad-thor "tail -c 40 $O/pool.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)
  d=$(echo "$z" | tr -d Z | cut -d- -f1); fl=$(echo "$z" | tr -d Z | cut -d- -f2); al=$(echo "$z" | tr -d Z | cut -d- -f3); q=$(echo "$z" | tr -d Z | cut -d- -f4)
  [ -n "$z" ] && [ "$al" = "0" ] && [ "$q" = "0" ] && { mark POOLDONE "$z"; break; }
  [ $(( $(date +%s) - t0 )) -gt 50400 ] && { mark POOLTIMEOUT "$z"; exit 6; }
  sleep 90
done
[ "$fl" = "0" ] && [ "$d" = "12" ] || { mark POOLFAIL "$z"; exit 6; }
ssh -n tanitad-thor "python3 $NX/r8nt_score.py finalize --prep $X/prep --scores $S --tag w$TAG $A --order-ref $X/controls/STOP_navtest.csv" > $W/stage2_finalize.out 2>&1
nok=$(ssh -n tanitad-thor "grep -l '\"status\": *\"PASS\"' $S/r8nt_w${TAG}_*/r8nt_w${TAG}_*.counts.json | wc -l" 2>/dev/null)
[ "$nok" = "3" ] || { mark FINALIZEBAD "pass=$nok"; exit 7; }
mkdir -p $W/stage2
ssh -n tanitad-thor "cd $O && tar -cf - scores_navtest/r8nt_w${TAG}_R8_LEGAL scores_navtest/r8nt_w${TAG}_R8_LEGAL_s1 scores_navtest/r8nt_w${TAG}_PRIOR_ha0p bridge_navtest/*.manifest.json cuda_controls.json RT_ASSERT.json INDEX.json post_gate.log" | tar -xf - -C $W/stage2
"$PY" -X utf8 C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/code/wiseft_read_navtest.py $TAG > $W/stage2/WISEFT_NAVTEST_READ.out 2>&1
[ -s $W/stage2/WISEFT_NAVTEST_READ.json ] && mark READOK "$(grep -o 'QQWISE2.*' $W/stage2/WISEFT_NAVTEST_READ.out | cut -c1-600)" || mark READBAD
mark END
