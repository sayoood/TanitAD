#!/bin/bash
# nhwu_waiter.sh (dev box, Git Bash) -- refcv8 EMA-48,852 warmup (N3) + navhard (N2) scoring on the Thor CPU backend, then parse8.
#   1. wait for ZZALLDONE in the roll's full.out (a ZZREFUSE_ stops us);
#   2. wait until NO other NavSim pool on Thor is live (navtest + navtrain-lever pool tails ZZd-f-alive-q-GBZZ, live thor_pool
#      processes) AND the navtest-score waiter is past its pool (terminal marker in its log) -- so <= 4 scorers on Thor's CPU;
#      fallback: 3 h after ZZALLDONE with no live pool at all;
#   3. re-stage + verify /dev/shm/navsim (nhwu_stage.sh -> ZZSTG9-0ZZ);
#   4. build jobs (nhwu_jobs.py: manifest / refcv7-fingerprint / token-set / staged-sha gates; both splits on the accepted
#      AgentInput fallback with their dev-box references);
#   5. thor_pool (nice 19 by the pool, <= 4 scorers, OMP_NUM_THREADS=6): warmup arms first, then navhard arms x 6 shards;
#   6. assert on ARTIFACTS (nhwu_counts.py): warmup counts.json PASS / 220 rows, every navhard shard PASS, rows sum 5,912;
#   7. finalize navhard per arm on Thor (finalize_navhard_arm.py -> merge_shards.py navhard, devkit tail verbatim);
#   8. pull bridges + scores; assemble the flat scores_navhard/ parse8 reads (byte copies, sha256-recorded);
#   9. parse8 both splits -> nhwu/summaries/summary_{warmup,navhard}.json; nhwu_assert.py -> summaries/NHWU_ASSERT.json;
#  10. (beside) bars8 with both summaries into nhwu/bars/ iff the navtest run has been pulled (raw/full/scores_navtest).
# Markers (opaque, disjoint from every pattern grepped): ZZNHW<STAGE>ZZ in $LOG.  Never reads an exit code through a pipe.
set -u
H=C:/Users/Admin/refcv8_pf/navtest_read/nhwu; LOG=$H/nhwu_waiter.log
NTLOG=C:/Users/Admin/refcv8_pf/navtest_read/navtest_score_waiter.log
X=/home/nvidia/r8ns/navtest_read; O=$X/full; TH=/home/nvidia/r8ns/nhwu; R=$TH/s48852e
TPY=/home/nvidia/venvs/navsim-cpu/bin/python; TC=/home/nvidia/navsim-thor/code/thor
YAML=/home/nvidia/navsim-thor/src/navsim-v2/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml
OREF=$TH/ship/ref/order_ref_navhard_R7_A1_s50400_devbox.csv
SW=$O/scores_warmup; SN=$O/scores_navhard_thor
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
P=C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim
F=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/floors
WU_IN=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json
NH_IN=C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json
L=$H/raw/full; SUM=$H/summaries; HP=/c/Users/Admin/refcv8_pf/navtest_read/nhwu; LP=$HP/raw/full
mark() { echo "ZZNHW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
rsh() { ssh -n -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 tanitad-thor "$@"; }
mkdir -p "$L" "$SUM"
mark START "pid=$$"
# ---- 1. the roll ----
while :; do
  r=$(rsh "grep -c 'ZZALLDON[E]' $X/full.out; grep -c 'ZZREFUS[E]_' $X/full.out; grep -c 'ZZSKI[P]_' $X/full.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  if [ -n "${2:-}" ] && [ "$2" != "0" ]; then mark REFUSED "alldone/refuse/skip=$r"; exit 1; fi
  if [ -n "${1:-}" ] && [ "$1" != "0" ]; then mark ROLLDONE "alldone/refuse/skip=$r"; break; fi
  sleep 180
done
[ "${3:-0}" = "0" ] || mark SKIPSEEN "a split printed a skip marker: $r"
# ---- 2. no other pool live; the navtest waiter past its pool ----
busy() { case "$1" in ZZnoneZZ) return 1;; ZZ[0-9]*ZZ) local s; s=$(echo "$1" | tr -d Z); [ "$(echo "$s" | cut -d- -f3)" != 0 ] || [ "$(echo "$s" | cut -d- -f4)" != 0 ];; *) return 0;; esac; }
t0=$(date +%s)
while :; do
  q=$(rsh "for f in $O/pool.out /home/nvidia/r8ns/navtrain_sel/lever/pool.out; do if [ -f \$f ]; then m=\$(tail -c 40 \$f | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1); echo \${m:-ZZemptyZZ}; else echo ZZnoneZZ; fi; done; ps -eo args | grep -c 'thor_poo[l].py'" 2>/dev/null | tr '\n' ' ')
  set -- $q
  nt=${1:-x}; lv=${2:-x}; np=${3:-x}
  ntdone=0; grep -qE 'ZZNTW(POOLDONE|POOLFAIL|JOBSBAD|REFUSED|END)ZZ' "$NTLOG" 2>/dev/null && ntdone=1
  el=$(( $(date +%s) - t0 ))
  if [ "$#" = 3 ] && ! busy "$nt" && ! busy "$lv" && [ "$np" = "0" ]; then
    if [ $ntdone = 1 ]; then mark POOLSIDLE "navtest=$nt lever=$lv thor_pool_procs=$np"; break; fi
    if [ $el -gt 10800 ]; then mark POOLSIDLEFALLBACK "navtest waiter not terminal after ${el}s; navtest=$nt lever=$lv procs=$np"; break; fi
  fi
  sleep 120
done
# ---- 3. stage ----
st=$(rsh "bash $TH/nhwu_stage.sh" 2>/dev/null | grep -oE 'ZZSTG[0-9]+-[0-9]+ZZ' | tail -1)
[ "$st" = "ZZSTG9-0ZZ" ] || { mark STAGEBAD "$st"; exit 2; }
mark STAGED "$st"
# ---- 4. jobs (arms = what the roll emitted; PRIOR_ha0p iff its seam exists) ----
arms() { rsh "cd $O/bridge_$1 && ls seam_*.npz" 2>/dev/null | sed -n 's/^seam_\(.*\)\.npz$/\1/p' | tr '\n' ' '; }
WARMS=$(arms warmup); NARMS=$(arms navhard)
mark ARMS "warmup=[$WARMS] navhard=[$NARMS]"
for need in R8_LEGAL R8_LEGAL_s1 R8_BLIND; do case " $WARMS " in *" $need "*) ;; *) mark ARMSBAD "warmup lacks $need"; exit 3;; esac; done
for need in R8_LEGAL R8_LEGAL_s1; do case " $NARMS " in *" $need "*) ;; *) mark ARMSBAD "navhard lacks $need"; exit 3;; esac; done
AW=""; for a in $WARMS; do AW="$AW --arm $a=$O/bridge_warmup/seam_$a.npz"; done
AN=""; for a in $NARMS; do AN="$AN --arm $a=$O/bridge_navhard/seam_$a.npz"; done
rsh "mkdir -p $SW $SN/_shards $R; $TPY $TH/nhwu_jobs.py --split warmup $AW --seam-dir /dev/shm/navsim/seams/s48852e --out-root $SW --exp-tag nhw48852 --jobs-out $R/jobs_wu.jsonl --report $R/jobs_wu_report.json; $TPY $TH/nhwu_jobs.py --split navhard $AN --seam-dir /dev/shm/navsim/seams/s48852e --out-root $SN --exp-tag nhn48852 --jobs-out $R/jobs_nh.jsonl --report $R/jobs_nh_report.json; cat $R/jobs_wu.jsonl $R/jobs_nh.jsonl > $R/jobs.jsonl" >> "$LOG" 2>&1
v=$(rsh "grep -c '\"verdict\": \"PAS[S]\"' $R/jobs_wu_report.json $R/jobs_nh_report.json" 2>/dev/null | tr '\n' ' ')
nw=$(echo $WARMS | wc -w); nn=$(echo $NARMS | wc -w); want=$(( nw + 6 * nn ))
nj=$(rsh "wc -l < $R/jobs.jsonl" 2>/dev/null)
case "$v" in *"jobs_wu_report.json:1 "*"jobs_nh_report.json:1 "*) ;; *) mark JOBSBAD "reports=$v"; exit 4;; esac
[ "${nj:-0}" = "$want" ] || { mark JOBSBAD "n=$nj want=$want"; exit 4; }
mark JOBSOK "n=$nj (warmup $nw arms + navhard $nn arms x 6)"
# ---- 5. pool ----
timeout 90 ssh -n tanitad-thor "cd $R; OMP_NUM_THREADS=6 nohup setsid python3 $TC/thor_pool.py --jobs $R/jobs.jsonl --state $R/pool_state.json > $R/pool.out 2>&1 < /dev/null &" > /dev/null 2>&1
sleep 90
pc=$(rsh "ps -eo args | grep -c 'nhwu/s48852[e]/jobs.jsonl'" 2>/dev/null)
mark POOLSTARTED "pool_procs=$pc"
dead=0
while :; do
  z=$(rsh "tail -c 40 $R/pool.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)
  al=$(echo "$z" | tr -d Z | cut -d- -f3); qq=$(echo "$z" | tr -d Z | cut -d- -f4)
  if [ -n "$z" ] && [ "$al" = "0" ] && [ "$qq" = "0" ]; then mark POOLDONE "$z"; break; fi
  pc=$(rsh "ps -eo args | grep -c 'nhwu/s48852[e]/jobs.jsonl'" 2>/dev/null)
  if [ "$pc" = "0" ]; then dead=$((dead+1)); else dead=0; fi
  [ $dead -ge 3 ] && { mark POOLDEAD "last=$z"; exit 5; }
  sleep 120
done
# ---- 6. artifacts: counts.json per job ----
cc=$(rsh "python3 $TH/nhwu_counts.py --sw $SW --sn $SN --warms '$WARMS' --narms '$NARMS' --wu-report $R/jobs_wu_report.json --nh-report $R/jobs_nh_report.json --out $R/counts_check.json" 2>/dev/null)
case "$cc" in *"QQCNT []"*) mark COUNTSOK;; *) mark COUNTSBAD "$(echo "$cc" | cut -c1-600)"; exit 6;; esac
# ---- 7. finalize navhard (Thor, navsim-cpu venv) ----
for a in $NARMS; do
  rsh "cd $R; PYTHONPATH=/home/nvidia/navsim-thor/code/stubs $TPY $TC/finalize_navhard_arm.py --arm $a --shard-root $SN/_shards --out $SN --order-ref $OREF --yaml $YAML" >> "$LOG" 2>&1
done
fz=$(rsh "grep -l '\"status\": \"PAS[S]\"' $SN/*_THOR/score_*_THOR__navhard_two_stage.counts.json | wc -l" 2>/dev/null)
[ "$fz" = "$nn" ] || { mark FINALIZEBAD "pass=$fz want=$nn"; exit 7; }
mark FINALIZEOK "$fz/$nn"
# ---- 8. pull + assemble ----
rsh "cd $O && tar -cf - bridge_warmup bridge_navhard scores_warmup scores_navhard_thor post_gate_warmup.log post_gate_navhard.log bridge_warmup.log bridge_navhard.log warmup_done.txt navhard_done.txt INDEX.json 2>/dev/null" | tar -xf - -C "$LP"
rsh "cd $TH && tar -cf - s48852e stage_check.json" | tar -xf - -C "$HP/raw"
"$PY" -X utf8 "$H/code/nhwu_assemble.py" --full "$L" --arms "$NARMS" --warms "$WARMS" --out-json "$H/raw/ASSEMBLE.json" >> "$LOG" 2>&1
grep -q '"verdict": "PASS"' "$H/raw/ASSEMBLE.json" 2>/dev/null || { mark ASSEMBLEBAD "see raw/ASSEMBLE.json"; exit 8; }
mark PULLED "$(du -sh "$LP" | cut -f1)"
# ---- 9. parse8 + assert ----
lab=$("$PY" -c "import json;print(json.load(open(r'$L/INDEX.json'))['label'])" 2>/dev/null)
[ "$lab" = "RESULT" ] || { mark LABELBAD "INDEX label=$lab"; exit 9; }
"$PY" -X utf8 $P/code/parse8.py --split warmup_two_stage --scores "$L/scores_warmup" --bridge "$L/bridge_warmup" --floors $F/warmup_two_stage \
  --inputs $WU_IN --label RESULT --out "$SUM/summary_warmup.json" > "$H/raw/parse8_warmup.out" 2>&1
"$PY" -X utf8 $P/code/parse8.py --split navhard_two_stage --scores "$L/scores_navhard" --bridge "$L/bridge_navhard" --floors $F/navhard_two_stage \
  --inputs $NH_IN --csv-suffix __navhard_two_stage --label RESULT --out "$SUM/summary_navhard.json" > "$H/raw/parse8_navhard.out" 2>&1
"$PY" -X utf8 "$H/code/nhwu_assert.py" --summaries "$SUM" --out "$SUM/NHWU_ASSERT.json" >> "$LOG" 2>&1
grep -q '"verdict": "PASS"' "$SUM/NHWU_ASSERT.json" 2>/dev/null && mark SUMMARIESOK || { mark SUMMARIESBAD "see summaries/NHWU_ASSERT.json"; exit 10; }
# ---- 10. bars8 beside (needs the navtest run pulled by the navtest waiter) ----
NR=C:/Users/Admin/refcv8_pf/navtest_read/raw/full
if [ -d "$NR/scores_navtest" ]; then
  PYTHONPATH="$P/code;C:/lgs/Lfb47a58376/taniteval" "$PY" -X utf8 $P/code/navsim_bars8.py --run-dir "$NR" --tag step48852_ema --out-dir "$H/bars" \
    --city-census C:/Users/Admin/navsim8/navtest_city_census.json --ship-r8-passed 1 \
    --navhard-summary "$SUM/summary_navhard.json" --warmup-summary "$SUM/summary_warmup.json" > "$H/raw/bars8_with_summaries.out" 2>&1
  [ -s "$H/bars/NAVSIM_BARS_step48852_ema.json" ] && mark BARSOK || mark BARSBAD "see raw/bars8_with_summaries.out"
else
  mark BARSSKIPPED "navtest run not pulled ($NR/scores_navtest absent)"
fi
mark END
