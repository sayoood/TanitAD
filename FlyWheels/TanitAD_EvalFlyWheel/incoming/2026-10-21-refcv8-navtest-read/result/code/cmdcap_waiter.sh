#!/bin/bash
# G_CMDCAP (R8-GATE-NAVTEST AMENDMENT 1, 9cd2347) on navtest: after the navtest waiter has pulled the bridge seams, build the
# R8PATH_MINPROG seams locally (build_lever_seams_navtest.py, K1-K6 controls must pass), ship them to Thor, score them in a SEPARATE
# scores dir (never the bar read's scores_navtest), finalize, pull, run r8_gate_eval.py with all three gates.  Markers ZZCCW<STAGE>ZZ.
set -u
LOG=/c/Users/Admin/refcv8_pf/navtest_read/cmdcap_waiter.log
LOC=/c/Users/Admin/refcv8_pf/navtest_read/raw; NTW=/c/Users/Admin/refcv8_pf/navtest_read/navtest_score_waiter.log
X=/home/nvidia/r8ns/navtest_read; NX=/home/nvidia/r8ns/code/navtest_read; O=$X/full; BC=$O/bridge_cmdcap; SC=$O/scores_cmdcap
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; INP=D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz
mark() { echo "ZZCCW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START
t0=$(date +%s)
until grep -q 'ZZNTWPULLE[D]' "$NTW" 2>/dev/null; do
  grep -q 'ZZNTW\(REFUSED\|JOBSBAD\|POOLFAIL\|FINALIZEBAD\)ZZ' "$NTW" 2>/dev/null && { mark UPSTREAMFAIL; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && { mark TIMEOUT; exit 1; }
  sleep 120
done
mark UPSTREAMPULLED
# 1. build (local, CPU) -- the K1-K6 controls are asserted by the builder's own exit code AND re-read from its report
mkdir -p "$LOC/cmdcap"
"$PY" -X utf8 /c/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/build_lever_seams_navtest.py "$LOC/full/bridge_navtest" "$INP" "$LOC/cmdcap" > "$LOC/cmdcap/build.out" 2>&1
ok=$("$PY" -c "import json;r=json.load(open('C:/Users/Admin/refcv8_pf/navtest_read/raw/cmdcap/lever_seams_build_report.json'));print(all(v['pass'] for v in r['controls'].values()))" 2>/dev/null)
[ "$ok" = "True" ] && [ -s "$LOC/cmdcap/thor_seams/seam_R8PATH_MINPROG.npz" ] && [ -s "$LOC/cmdcap/thor_seams/seam_R8PATH_MINPROG_s1.npz" ] || { mark BUILDBAD "controls=$ok"; exit 2; }
mark BUILDOK
# 2. ship (md5 verified)
ssh -n tanitad-thor "mkdir -p $BC"
for f in seam_R8PATH_MINPROG.npz seam_R8PATH_MINPROG.manifest.json seam_R8PATH_MINPROG_s1.npz seam_R8PATH_MINPROG_s1.manifest.json; do
  scp -q "$LOC/cmdcap/thor_seams/$f" "tanitad-thor:$BC/$f"
  a=$(md5sum "$LOC/cmdcap/thor_seams/$f" | cut -c1-32); b=$(ssh -n tanitad-thor "md5sum $BC/$f" | cut -c1-32)
  [ ${#a} -eq 32 ] && [ "$a" = "$b" ] || { mark SHIPBAD "$f"; exit 3; }
done
mark SHIPOK
# 3. jobs + pool (separate state/out files)
A="--arm R8PATH_MINPROG=$BC/seam_R8PATH_MINPROG.npz --arm R8PATH_MINPROG_s1=$BC/seam_R8PATH_MINPROG_s1.npz"
ssh -n tanitad-thor "python3 $NX/r8nt_score.py jobs --prep $X/prep --scores $SC --tag s48852e $A --out $O/score_jobs_cmdcap.jsonl" >> "$LOG" 2>&1
n=$(ssh -n tanitad-thor "wc -l < $O/score_jobs_cmdcap.jsonl" 2>/dev/null)
[ "${n:-0}" = "8" ] || { mark JOBSBAD "n=$n"; exit 4; }
ssh -n tanitad-thor "mkdir -p $SC && cd $SC && OMP_NUM_THREADS=6 nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $O/score_jobs_cmdcap.jsonl --state $O/pool_cmdcap_state.json > $O/pool_cmdcap.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 60
mark POOLSTARTED
while :; do
  z=$(ssh -n tanitad-thor "tail -c 40 $O/pool_cmdcap.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)
  d=$(echo "$z" | tr -d Z | cut -d- -f1); f=$(echo "$z" | tr -d Z | cut -d- -f2); al=$(echo "$z" | tr -d Z | cut -d- -f3); q=$(echo "$z" | tr -d Z | cut -d- -f4)
  if [ -n "$z" ] && [ "$al" = "0" ] && [ "$q" = "0" ]; then mark POOLDONE "$z"; break; fi
  sleep 90
done
[ "$f" = "0" ] && [ "$d" = "8" ] || { mark POOLFAIL "$z"; exit 5; }
# 4. finalize + assert on the counts artifacts
ssh -n tanitad-thor "python3 $NX/r8nt_score.py finalize --prep $X/prep --scores $SC --tag s48852e $A --order-ref $X/controls/STOP_navtest.csv" > "$LOC/cmdcap/finalize.out" 2>&1
nok=$(ssh -n tanitad-thor "grep -l '\"status\": *\"PASS\"' $SC/r8nt_s48852e_*/r8nt_s48852e_*.counts.json | wc -l" 2>/dev/null)
[ "$nok" = "2" ] || { mark FINALIZEBAD "pass=$nok"; exit 6; }
mark FINALIZEOK
# 5. pull + the three-gate eval
ssh -n tanitad-thor "cd $O && tar -cf - scores_cmdcap" | tar -xf - -C "$LOC/cmdcap"
SN="$LOC/full/scores_navtest"; SCL="$LOC/cmdcap/scores_cmdcap"; BRL="$LOC/full/bridge_navtest"
"$PY" -X utf8 /c/Users/Admin/refcv8_pf/navtest_read/code/r8_gate_eval.py --rt-csv $SN/r8nt_s48852e_R8_RT/r8nt_s48852e_R8_RT.csv \
  --rt-csv-s1 $SN/r8nt_s48852e_R8_RT_s1/r8nt_s48852e_R8_RT_s1.csv --prior-csv $SN/r8nt_s48852e_PRIOR_ha0p/r8nt_s48852e_PRIOR_ha0p.csv \
  --seam-rt $BRL/seam_R8_RT.npz --seam-rt-s1 $BRL/seam_R8_RT_s1.npz --seam-prior $BRL/seam_PRIOR_ha0p.npz --inputs "$INP" \
  --cap-csv $SCL/r8nt_s48852e_R8PATH_MINPROG/r8nt_s48852e_R8PATH_MINPROG.csv \
  --cap-csv-s1 $SCL/r8nt_s48852e_R8PATH_MINPROG_s1/r8nt_s48852e_R8PATH_MINPROG_s1.csv \
  --label NAVTEST_RESULT --out "$LOC/gate/R8_GATE_NAVTEST_3gates.json" > "$LOC/cmdcap/gate3.out" 2>&1
[ -s "$LOC/gate/R8_GATE_NAVTEST_3gates.json" ] && mark GATEOK "$(grep -o 'QQGATE.*' $LOC/cmdcap/gate3.out | cut -c1-600)" || mark GATEBAD
mark END
