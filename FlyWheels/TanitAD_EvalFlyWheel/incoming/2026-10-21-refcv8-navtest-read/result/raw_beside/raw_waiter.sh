#!/bin/bash
# RAW weights 48,852 BESIDE (A-R8-NS1 (i): never a bar): after raw_beside.out reads ZZALLDONE and no thor_pool is alive, score R8_LEGAL
# and R8_RT of the RAW roll in a SEPARATE scores dir, finalize, pull, and read RAW - EMA paired on the same tokens.  Markers ZZRWW<STAGE>ZZ.
set -u
LOG=/c/Users/Admin/refcv8_pf/navtest_read/raw_waiter.log; LOC=/c/Users/Admin/refcv8_pf/navtest_read/raw
X=/home/nvidia/r8ns/navtest_read; NX=/home/nvidia/r8ns/code/navtest_read; O=$X/raw_beside; B=$O/bridge_navtest; S=$O/scores_navtest
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mark() { echo "ZZRWW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZALLDON[E]' $X/raw_beside.out; grep -c 'ZZREFUS[E]' $X/raw_beside.out; ps -eo args | grep -c '^python3 /home/nvidia/navsim-thor/code/thor/thor_po[o]l.py' || true" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark REFUSED "$r"; exit 1; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && [ "${3:-1}" = "0" ] && break
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && { mark TIMEOUT "$r"; exit 1; }
  sleep 180
done
mark READY
A="--arm R8_LEGAL=$B/seam_R8_LEGAL.npz --arm R8_RT=$B/seam_R8_RT.npz"
ssh -n tanitad-thor "python3 $NX/r8nt_score.py jobs --prep $X/prep --scores $S --tag s48852r $A --out $O/score_jobs_raw.jsonl" >> "$LOG" 2>&1
n=$(ssh -n tanitad-thor "wc -l < $O/score_jobs_raw.jsonl" 2>/dev/null); [ "${n:-0}" = "8" ] || { mark JOBSBAD "n=$n"; exit 2; }
ssh -n tanitad-thor "mkdir -p $S && cd $S && OMP_NUM_THREADS=6 nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $O/score_jobs_raw.jsonl --state $O/pool_state.json > $O/pool.out 2>&1 < /dev/null &" < /dev/null > /dev/null 2>&1 &
sleep 60; mark POOLSTARTED
while :; do
  z=$(ssh -n tanitad-thor "tail -c 40 $O/pool.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)
  d=$(echo "$z" | tr -d Z | cut -d- -f1); f=$(echo "$z" | tr -d Z | cut -d- -f2); al=$(echo "$z" | tr -d Z | cut -d- -f3); q=$(echo "$z" | tr -d Z | cut -d- -f4)
  [ -n "$z" ] && [ "$al" = "0" ] && [ "$q" = "0" ] && { mark POOLDONE "$z"; break; }
  sleep 90
done
[ "$f" = "0" ] && [ "$d" = "8" ] || { mark POOLFAIL "$z"; exit 3; }
ssh -n tanitad-thor "python3 $NX/r8nt_score.py finalize --prep $X/prep --scores $S --tag s48852r $A --order-ref $X/controls/STOP_navtest.csv" > "$LOC/raw_finalize.out" 2>&1
nok=$(ssh -n tanitad-thor "grep -l '\"status\": *\"PASS\"' $S/r8nt_s48852r_*/r8nt_s48852r_*.counts.json | wc -l" 2>/dev/null)
[ "$nok" = "2" ] || { mark FINALIZEBAD "pass=$nok"; exit 4; }
mkdir -p "$LOC/raw_beside"; ssh -n tanitad-thor "cd $O && tar -cf - scores_navtest bridge_navtest/seam_R8_LEGAL.manifest.json bridge_navtest/seam_R8_RT.manifest.json cuda_controls.json INDEX.json" | tar -xf - -C "$LOC/raw_beside"
"$PY" -X utf8 - > "$LOC/raw_beside/RAW_VS_EMA.out" 2>&1 <<'EOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("b8r", "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/navsim_bars8.py")
B = importlib.util.module_from_spec(spec); sys.modules["b8r"] = B; spec.loader.exec_module(B)
L = "C:/Users/Admin/refcv8_pf/navtest_read/raw"; cfg = {**B.DEFAULT_CFG}
rd = lambda p: B.read_scores_csv(p)
raw = {a: rd(f"{L}/raw_beside/scores_navtest/r8nt_s48852r_{a}/r8nt_s48852r_{a}.csv") for a in ("R8_LEGAL", "R8_RT")}
ema = {a: rd(f"{L}/full/scores_navtest/r8nt_s48852e_{a}/r8nt_s48852e_{a}.csv") for a in ("R8_LEGAL", "R8_RT", "PRIOR_ha0p")}
toks = sorted(ema["R8_LEGAL"].index); assert all(sorted(d.index) == toks for d in list(raw.values()) + list(ema.values()))
ti = B.load_inputs(B.DEFAULT_INPUTS).loc[toks]; cl = [str(x) for x in ti["log"]]
res = {"label": "BESIDE (RAW weights 48,852, never a bar; A-R8-NS1 (i))", "tier": "NavSim v1.1 open-loop PDMS, navtest, Thor",
       "PDMS_x100": {**{f"RAW_{a}": round(100 * float(d["PDMS"].mean()), 4) for a, d in raw.items()},
                     **{f"EMA_{a}": round(100 * float(d["PDMS"].mean()), 4) for a, d in ema.items()}},
       "RAW_minus_EMA_R8_LEGAL": B.pair_record(raw["R8_LEGAL"], ema["R8_LEGAL"], toks, cl, cfg),
       "RAW_minus_EMA_R8_RT": B.pair_record(raw["R8_RT"], ema["R8_RT"], toks, cl, cfg),
       "RAW_R8_LEGAL_minus_PRIOR": B.pair_record(raw["R8_LEGAL"], ema["PRIOR_ha0p"], toks, cl, cfg)}
json.dump(res, open(f"{L}/raw_beside/RAW_VS_EMA.json", "w"), indent=1)
print("QQRAW " + json.dumps({"PDMS": res["PDMS_x100"], "raw_minus_ema_legal": res["RAW_minus_EMA_R8_LEGAL"]["delta_x100"]}) + " QQ")
EOF
[ -s "$LOC/raw_beside/RAW_VS_EMA.json" ] && mark READOK "$(grep -o 'QQRAW.*' $LOC/raw_beside/RAW_VS_EMA.out | cut -c1-400)" || mark READBAD
mark END
