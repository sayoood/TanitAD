#!/bin/bash
# rh_read_rk_chain.sh -- STAGED, NOT STARTED.  The ROLLHIST read of the arm RK (KINLAB: RL's argv with `--dagger-label speed_v1` -> `kin_v1` = the own-kinematics label on matcher-ok own states, the RH bank, p = 0.1, swap mode; card PREREG_V8F_KINLAB, commit 0ca6f49d; the ROLLHIST card sha256 781eea87 section 3 binds unless the KINLAB card changes it).
# Same readers, cells, floors, estimator and tier as rh_read_chain.sh (the finished PARTIAL read, which this script does not touch): RK - B0 under the form of record (RH_AKIND=RK swaps the treated arm in rh_common.PAIRS), with
# RK - RH beside (no bar) and ATTR = RK - RL under C* (R0C10XHb20).  B0's, RH's and RL's dumps are REUSED (out/t1/I10B0_*, out/t1/I10RH_*, out/t1/I10RL_*); only RK is pulled, rolled (6 forms x 2 seeds = 12 rollouts, ~6.4 min each on CPU, ESTIMATED) and T0-evaluated.
# CPU only, 2 threads, ONE process at a time, every step gated on dev-box free commit >= 5.5 GB (wait_commit.py) and F: free >= 6 GB.  Resumable: out/steps_rk/<name>.ok.   Outputs only under out/read_rk/ and out/t1/I10RK_*, out/t0/eval_RK_*, out/read_rk/MECH3_KINLAB.json.
#   0  gates: the PARTIAL read's files are present (RESULT_ROLLHIST.json, refs readers, STRAT_FLOOR_UEA.json, the RH, RL and B0 dumps with T1_DONE, the RL checkpoints and T0 evals), C_EXEC.json pass, tree blobcheck, the reader selftest under RH_AKIND=RK
#   1  pull RK_s0 / RK_s1 from the pod (read only; the pod's own sha256sum, path /root/rollhist/runs/<arm>/train/ckpt_last.pt, only after that arm's ARM_DONE), check each against it
#   2  12 rollouts (rh_roll.py RK <form> <seed>)         (K-REPLAY is not repeated: it is a property of the rollout instrument, which is unchanged and passed in the PARTIAL read)
#   4  G-T0: the T0 E0 eval of RK_s0 / RK_s1, then the guard (RK - B0 at T0 ADE6, <= +0.0705 m), then MECH3 (mech_rk.py: RK - RL at T0 ADE6, paired, per seed; MECH3-YES iff the interval is below 0 at both seeds; beside RK - B0, RL - B0, RK - RH, RH - B0)
#   5  readers: headway (U, C, F), families (q3), STRATEGIC (rh_strat.py, frozen ADD66 reader); the stage-A references and the one-seed STRATEGIC floor are the PARTIAL read's (UE.A property), reused read-only
#   6  rh_read.py -> out/read_rk/RESULT_ROLLHIST_RK.json   (one committed outcome name per the card's table + the one-line answer "does the kinematic label hold G-T0 and keep P2/P3?")
# Env: RH_SEEDS (default "0,1").  Last line of the log: "ZZ ROLLHIST-READ-RK DONE ZZ" or "ZZ ROLLHIST-READ-RK FAIL ZZ <why>".  Never `| tee`; every line goes to the log with >> "$LOG".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export RH_AKIND=RK
E=F:/TanitAD-scratch/rollhist/read
PY=${RH_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
O=$E/out; STEPS=$O/steps_rk; LOG=$O/rh_read_rk_chain.log; RD=$O/read_rk
SEEDS=${RH_SEEDS:-0,1}
export TMP=$E/tmp TEMP=$E/tmp
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
V=F:/TanitAD-scratch/v8proof
P=/c/Users/Admin/lorapod/pssh_cmd.sh; PC=/c/Users/Admin/lorapod/pcat.sh
cd "$E" || exit 1
mkdir -p "$O" "$STEPS" "$E/tmp" "$RD" "$E/ckpts"
fail() { echo "ZZ ROLLHIST-READ-RK FAIL ZZ $1" >> "$LOG"; echo "FAIL $1" > "$O/ROLLHIST_READ_RK.done"; exit 1; }
gate() { $PY $WC 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"
         fk=$(df -k /f | awk 'NR==2{print int($4/1048576)}'); [ "${fk:-0}" -ge 6 ] || fail "F: free ${fk} GB < 6 GB before: $1"; }
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[rollhist read RK chain] start $(date) seeds $SEEDS tree $E/tree (43319227) treated arm RK" >> "$LOG"
IFS=, read -ra SL <<< "$SEEDS"
# ---- 0 gates: the PARTIAL read is finished and its reusable parts are there (read only)
need() { [ -s "$1" ] || fail "gate0: missing $1 (the finished PARTIAL read's file)"; }
need $O/read/RESULT_ROLLHIST.json; need $O/read/refs_q3/RESULT_Q3.json; need $O/read/refs_hw/RESULT_HEADWAY.json; need $O/read/STRAT_FLOOR_UEA.json
for s in "${SL[@]}"; do
  for f in R0Hb20 R0C10XHb20; do [ -e "$O/t1/I10RH_${f}_s$s/T1_DONE" ] || fail "gate0: RH dump $f s$s missing (the RK - RH beside pair)"; done
  for f in R0Hb20 R0C10XHb20; do [ -e "$O/t1/I10RL_${f}_s$s/T1_DONE" ] || fail "gate0: RL dump $f s$s missing (the ATTR pair RK - RL needs R0C10XHb20; RL's two beside forms are kept equal to RH's)"; done
  for f in R0Hb20 R0C10Hb20 R0G50Hb20 R0G0Hb20 R0KHb20 R0C10XHb20; do [ -e "$O/t1/I10B0_${f}_s$s/T1_DONE" ] || fail "gate0: B0 dump $f s$s missing (the comparator is the existing B0)"; done
  for k in RH RL B0; do [ -s "$E/ckpts/${k}_s$s/ckpt_last.pt" ] || fail "gate0: checkpoint ${k}_s$s missing (provenance check)"; done
  [ -s "$O/t0/eval_RH_s$s.npz" ] && [ -s "$O/t0/eval_RL_s$s.npz" ] && [ -s "$O/t0/eval_B0_s$s.npz" ] || fail "gate0: T0 eval of RH_s$s / RL_s$s / B0_s$s missing"
done
$PY -c "import json,sys; d=json.load(open('$E/C_EXEC.json')); sys.exit(0 if d.get('PASS') is True and not d.get('smoke_windows') else 1)" >> "$LOG" 2>&1 || fail "gate0: C_EXEC.json missing or not PASS"
step blobcheck "tree blobs equal the commit" bash -c "$PY $E/blobcheck.py $E/tree 43319227 | head -1 | grep -q 'bad 0 '"
step readertests "rh_read.py selftest under RH_AKIND=RK (K-SELF, K-PLANT, floor, seeds, harms, mutation)" $PY -u $E/rh_read.py --selftest
# ---- 1 the RK checkpoints from the pod (read only)
pull_arm() { # ARM
  local arm=$1 rp=/root/rollhist/runs/$1/train/ckpt_last.pt a b
  mkdir -p "$E/ckpts/$arm"
  [ "$(timeout 300 bash $P "test -e /root/rollhist/runs/$arm/ARM_DONE && echo yes" 2>/dev/null | tr -cd 'a-z')" = yes ] || { echo "the pod arm $arm has no ARM_DONE yet"; return 1; }
  a=$(timeout 300 bash $P "sha256sum $rp | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
  [ ${#a} = 64 ] || { echo "no pod sha256 for $arm (checkpoint not there yet?)"; return 1; }
  timeout 900 bash $PC "$rp" "$E/ckpts/$arm/ckpt_last.pt" < /dev/null > /dev/null 2>&1
  b=$(sha256sum "$E/ckpts/$arm/ckpt_last.pt" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "PULL-MISMATCH $arm pod ${a:0:12} local ${b:0:12}"; return 1; }
  grep -v " $arm/ckpt_last.pt\$" "$E/ckpts/SHA256_PODS.txt" 2>/dev/null > "$E/ckpts/SHA256_PODS.tmp"; echo "$a  $arm/ckpt_last.pt" >> "$E/ckpts/SHA256_PODS.tmp"; mv "$E/ckpts/SHA256_PODS.tmp" "$E/ckpts/SHA256_PODS.txt"
  echo "pulled $arm sha256 ${a:0:16}"
}
for s in "${SL[@]}"; do step "pull_RK_s$s" "pull and check RK_s$s" pull_arm RK_s$s; done
# ---- 2 rollouts (RK only)
for s in "${SL[@]}"; do for f in R0Hb20 R0C10Hb20 R0G50Hb20 R0G0Hb20 R0KHb20 R0C10XHb20; do
  step "ro_RK_${f}_s$s" "rollout RK $f seed $s" $PY -u $E/rh_roll.py RK $f $s
done; done
# ---- 4 G-T0
arms=(); for s in "${SL[@]}"; do arms+=(RK_s$s); done
step t0evals "G-T0 T0 evals ${arms[*]}" $PY -u $E/rh_t0.py run "${arms[@]}"
step gt0 "G-T0 guard (RK - B0)" $PY -u $E/rh_t0.py read --seeds "$SEEDS" --out $RD/GT0_ROLLHIST_RK.json
step mech3 "MECH3 (RK - RL at T0 ADE6, paired, per seed)" $PY -I -u $E/mech_rk.py $E $RD/MECH3_KINLAB.json
# ---- 5 readers (frozen: headway_read_r2.py, q3_read_r3.py; B0 and RH dumps reused)
PAIRS=$($PY -c "import sys; sys.path.insert(0,'$E'); import rh_common as C; assert C.A_KIND=="RK"; print(','.join('I10%s:I10%s'%(a,b) for a,b in C.PAIRS.values()))")
export HEADWAY_Q3DIR=$E/code/q3 HIERREP_STAGEB=$V/pod/stageB TANITEVAL_ROOT=$E/tree/taniteval HEADWAY_JOIN=F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz HEADWAY_BANK=$V/res/cache/frames.npz
SP=I10RK_R0Hb20
if [ ${#SL[@]} = 1 ]; then SPREAD="--spread-arm I10UEA_R0C10XHb20 --spread-seeds 0,1"; else SPREAD="--spread-arm $SP --spread-seeds $SEEDS"; fi
step headway "headway reader U C F" bash -c "cd $E/code/headway && $PY -u headway_read_r2.py read --t1-root $O/t1 --pairs '$PAIRS' --seeds $SEEDS $SPREAD --out-dir $RD/hw"
step q3 "family reader (LONGITUDINAL / LATERAL / TACTICAL cells)" bash -c "cd $E/code/q3 && $PY -u q3_read_r3.py read --t1-root $O/t1 --pairs '$PAIRS' --seeds $SEEDS $SPREAD --same-sign --out-dir $RD/q3"
step strat "STRATEGIC family (frozen ADD66 reader, per-seed mode)" $PY -u $E/rh_strat.py --out $RD --seeds "$SEEDS"
# ---- 6 the read
BR=F:/TanitAD-scratch/v8proof/integ10k/read_boundtier/pod
step read "rh_read.py (the outcome and the one-line RK answer)" $PY -u $E/rh_read.py --q3 $RD/q3/RESULT_Q3.json --hw $RD/hw/RESULT_HEADWAY.json --strat $RD/READ_STRAT_ROLLHIST.json --gt0 $RD/GT0_ROLLHIST_RK.json --strat-floor $O/read/STRAT_FLOOR_UEA.json --refs-q3 $O/read/refs_q3/RESULT_Q3.json --refs-hw $O/read/refs_hw/RESULT_HEADWAY.json --add68-q3 $BR/RESULT_Q3.json --add68-hw $BR/RESULT_HEADWAY.json --rh-result $O/read/RESULT_ROLLHIST.json --seeds "$SEEDS" --out $RD/RESULT_ROLLHIST_RK.json
echo "DONE" > "$O/ROLLHIST_READ_RK.done"
echo "ZZ ROLLHIST-READ-RK DONE ZZ" >> "$LOG"
