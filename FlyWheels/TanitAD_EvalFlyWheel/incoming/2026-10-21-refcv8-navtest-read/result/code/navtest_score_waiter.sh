#!/bin/bash
# Waiter (dev box): when the Thor navtest roll prints its done marker, score navtest on the Thor CPU backend (HANDOFF.md 4a),
# finalize, pull the navtest parts, then run bars8 (N1/N4 + strata) and the pre-registered gate eval locally.
# Markers (opaque, disjoint from what is grepped): ZZNTW<STAGE>ZZ in $LOG.  Assert on artifacts, never on exit codes.
set -u
LOG=/c/Users/Admin/refcv8_pf/navtest_read/navtest_score_waiter.log
X=/home/nvidia/r8ns/navtest_read; NX=/home/nvidia/r8ns/code/navtest_read; O=$X/full; B=$O/bridge_navtest; S=$O/scores_navtest
LOC=/c/Users/Admin/refcv8_pf/navtest_read/raw
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mark() { echo "ZZNTW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START
# 1. wait for the navtest roll (or a refusal)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZNAVTESTDON[E]' $X/full.out; grep -c 'ZZREFUS[E]_' $X/full.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  if [ "${2:-0}" != "0" ] && [ -n "${2:-}" ]; then mark REFUSED "$r"; exit 1; fi
  if [ "${1:-0}" != "0" ] && [ -n "${1:-}" ]; then mark ROLLDONE "$r"; break; fi
  sleep 180
done
# 2. jobs + pool (only once the lever pool has nothing alive, so <= 4 scorers total on the CPU)
while :; do
  alive=$(ssh -n tanitad-thor "tail -c 30 /home/nvidia/r8ns/navtrain_sel/lever/pool.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1 | cut -d- -f3)
  [ "${alive:-1}" = "0" ] && break
  sleep 60
done
mark LEVERPOOLIDLE
A="--arm R8_LEGAL=$B/seam_R8_LEGAL.npz --arm R8_LEGAL_s1=$B/seam_R8_LEGAL_s1.npz --arm R8_PRIV=$B/seam_R8_PRIV.npz --arm R8_RT=$B/seam_R8_RT.npz --arm R8_RT_s1=$B/seam_R8_RT_s1.npz --arm PRIOR_ha0p=$B/seam_PRIOR_ha0p.npz"
ssh -n tanitad-thor "python3 $NX/r8nt_score.py jobs --prep $X/prep --scores $S --tag s48852e $A --out $O/score_jobs.jsonl" >> "$LOG" 2>&1
n=$(ssh -n tanitad-thor "wc -l < $O/score_jobs.jsonl" 2>/dev/null)
[ "${n:-0}" = "24" ] || { mark JOBSBAD "n=$n"; exit 2; }
mark JOBSOK "n=$n"
ssh -n tanitad-thor "mkdir -p $S && cd $S && OMP_NUM_THREADS=6 nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $O/score_jobs.jsonl --state $O/pool_state.json > $O/pool.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 60
mark POOLSTARTED
# 3. wait for the pool: done-fail-alive-queued
while :; do
  z=$(ssh -n tanitad-thor "tail -c 40 $O/pool.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)
  d=$(echo "$z" | tr -d Z | cut -d- -f1); f=$(echo "$z" | tr -d Z | cut -d- -f2); al=$(echo "$z" | tr -d Z | cut -d- -f3); q=$(echo "$z" | tr -d Z | cut -d- -f4)
  if [ -n "$z" ] && [ "$al" = "0" ] && [ "$q" = "0" ]; then mark POOLDONE "$z"; break; fi
  sleep 120
done
[ "$f" = "0" ] && [ "$d" = "24" ] || { mark POOLFAIL "$z"; exit 3; }
# 4. finalize
ssh -n tanitad-thor "python3 $NX/r8nt_score.py finalize --prep $X/prep --scores $S --tag s48852e $A --order-ref $X/controls/STOP_navtest.csv" > "$LOC/../finalize.out" 2>&1
grep -o 'QQFIN.*' "$LOC/../finalize.out" | tail -1 >> "$LOG"
nok=$(ssh -n tanitad-thor "grep -l '\"status\": *\"PASS\"' $S/r8nt_s48852e_*/r8nt_s48852e_*.counts.json | wc -l" 2>/dev/null)
[ "$nok" = "6" ] || { mark FINALIZEBAD "pass=$nok"; exit 4; }
mark FINALIZEOK "6/6 counts PASS"
# 5. pull the navtest parts (bridge seams + scores + INDEX)
mkdir -p "$LOC/full"
ssh -n tanitad-thor "cd $O && tar -cf - bridge_navtest scores_navtest INDEX.json 2>/dev/null" | tar -xf - -C "$LOC/full"
ls "$LOC/full/scores_navtest" | grep -c r8nt_s48852e_ > /dev/null && mark PULLED "$(du -sh $LOC/full | cut -f1)"
# 6. bars8 (navtest bars; navhard/warmup summaries come later) + the pre-registered gate eval
P=C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim
PYTHONPATH="$P/code;C:/lgs/Lfb47a58376/taniteval" "$PY" -X utf8 $P/code/navsim_bars8.py --run-dir "$LOC/full" --tag step48852_ema \
  --out-dir "$LOC/bars" --city-census C:/Users/Admin/navsim8/navtest_city_census.json --ship-r8-passed 1 > "$LOC/../bars8.out" 2>&1
[ -s "$LOC/bars/NAVSIM_BARS_step48852_ema.json" ] && mark BARSOK || mark BARSBAD
SC="$LOC/full/scores_navtest"; BRL="$LOC/full/bridge_navtest"
"$PY" -X utf8 /c/Users/Admin/refcv8_pf/navtest_read/code/r8_gate_eval.py --rt-csv $SC/r8nt_s48852e_R8_RT/r8nt_s48852e_R8_RT.csv \
  --rt-csv-s1 $SC/r8nt_s48852e_R8_RT_s1/r8nt_s48852e_R8_RT_s1.csv --prior-csv $SC/r8nt_s48852e_PRIOR_ha0p/r8nt_s48852e_PRIOR_ha0p.csv \
  --seam-rt $BRL/seam_R8_RT.npz --seam-rt-s1 $BRL/seam_R8_RT_s1.npz --seam-prior $BRL/seam_PRIOR_ha0p.npz \
  --inputs D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz --label NAVTEST_RESULT \
  --out "$LOC/gate/R8_GATE_NAVTEST.json" > "$LOC/../gate.out" 2>&1
[ -s "$LOC/gate/R8_GATE_NAVTEST.json" ] && mark GATEOK || mark GATEBAD
mark END
