#!/bin/bash
# rh_read_s3_chain.sh -- STAGED, NOT STARTED.  The SEED3 dev-box read (card PREREG_V8F_SEED3 sha256 75f3b191): the ONE-seed s2 read of B0_s2 / RH_s2 / KI_s2, decisive reads G2 and P1-s2, the beside reads, and the card's name (s3_name.py).
# Built from rh_read_ki_chain.sh.  Same readers (headway_read_r2.py, q3_read_r3.py, rh_strat.py: all frozen, unedited), cells, estimator (paired episode-cluster bootstrap B 2,000 seed 0), tier (T1-kin/hb0.2) and executor as the finished RH / RA / RK / KI reads, which this script does not touch.
# What is new and only new: seed 2 in the pairs, the cell floors F_c TAKEN UNCHANGED from the finished two-seed reads (rh_read.py --floors-from: out/read/RESULT_ROLLHIST.json for the RH pass, out/read_ki/RESULT_ROLLHIST_KI.json for the KI pass; sha256 pinned below),
# the G-T0 bar 0.1174 with 0.0705 printed beside (rh_t0.py --bar --beside-bar), MECH4 at s2 (mech_s3.py), the name (s3_name.py).
# CPU only, 2 threads, ONE process at a time, every step gated on dev-box free commit >= 5.5 GB (wait_commit.py 5.5 60) and F: free >= 6 GB.  Resumable: out/steps_s3/<name>.ok.  Outputs: out/read_s3/, out/t1/I10{B0,RH,KI}_*_s2, out/t0/eval_{B0,RH,KI}_s2.*
#   0  gates: the finished reads' files are present, the two floor files have their PINNED sha256, the stage-A spread dumps are present, C_EXEC.json pass, tree blobcheck
#      (RH_S3_DRY=1: print the pairs and the plan and STOP before the pull; nothing is pulled, rolled or written under steps_s3)
#   1  pull B0_s2 / RH_s2 / KI_s2 from the pod (read only; the pod's own sha256sum of /root/rollhist/runs/<arm>/train/ckpt_last.pt, only after that arm's ARM_DONE), each checked against it
#   2  18 rollouts (3 arms x 6 forms) at T1-kin/hb0.2 (rh_roll.py <kind> <form> 2)
#   3  T0 evals of the three checkpoints; G-T0 (RH_s2 - B0_s2 = G2, and KI_s2 - B0_s2): bar 0.1174, 0.0705 beside; mech_s3.py (MECH4 at s2, the three-seed list, the s0/s1 G-T0 points printed only)
#   4  readers, once with RH_AKIND=RH (RH_s2 - B0_s2: P1-s2, P2, P3, P4, beside) and once with RH_AKIND=KI (KI_s2 - B0_s2, KI_s2 - RH_s2 beside): headway, q3 (spread arm = the stage-A UEA dumps, as the KI chain's one-seed branch), strat (--seeds 2)
#   5  rh_read.py --seeds 2 --floors-from (both passes), then s3_name.py -> out/read_s3/RESULT_SEED3.json (the name; every input value)
# Needs out/read_s3/LEVER_ACTS_S3.json (written by the MM from the pod's records; keys in s3_name.LEVER_KEYS, every one true).  Absent -> the name is SEED3-BLOCKED, never a guess.
# Three-seed pooled reads are NOT computed (the card freezes F_c for two seeds only; see s3_name.py).
# Last line of the log: "ZZ SEED3-READ DONE ZZ" or "ZZ SEED3-READ FAIL ZZ <why>".  Never `| tee`; every line goes to the log with >> "$LOG".  Single quotes inside every python -c string.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
unset RH_AKIND
E=F:/TanitAD-scratch/rollhist/read
PY=${RH_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
O=$E/out; STEPS=$O/steps_s3; LOG=$O/rh_read_s3_chain.log; RD=$O/read_s3
S=2
export TMP=$E/tmp TEMP=$E/tmp
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
V=F:/TanitAD-scratch/v8proof
P=/c/Users/Admin/lorapod/pssh_cmd.sh; PC=/c/Users/Admin/lorapod/pcat.sh
FLOOR_RH=$O/read/RESULT_ROLLHIST.json;        SHA_RH=5613818e5748b8ee38fa6c88165bf5699dd549f04128c4af8f068643e0d99c41
FLOOR_KI=$O/read_ki/RESULT_ROLLHIST_KI.json;  SHA_KI=a12fef1c111721e8175bc4c4c70078489f3ef52a0cc6993493ae6210fb731bfa
FORMS="R0Hb20 R0C10Hb20 R0G50Hb20 R0G0Hb20 R0KHb20 R0C10XHb20"
cd "$E" || exit 1
mkdir -p "$O" "$STEPS" "$E/tmp" "$RD" "$E/ckpts"
fail() { echo "ZZ SEED3-READ FAIL ZZ $1" >> "$LOG"; echo "FAIL $1" > "$O/ROLLHIST_READ_S3.done"; exit 1; }
gate() { $PY $WC 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"
         fk=$(df -k /f | awk 'NR==2{print int($4/1048576)}'); [ "${fk:-0}" -ge 6 ] || fail "F: free ${fk} GB < 6 GB before: $1"; }
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[SEED3 read chain] start $(date) seed $S arms B0_s2 RH_s2 KI_s2 tree $E/tree (43319227) dry=${RH_S3_DRY:-0}" >> "$LOG"
# ---- 0 gates (read only)
need() { [ -s "$1" ] || fail "gate0: missing $1 (a finished read's file)"; }
need $FLOOR_RH; need $FLOOR_KI; need $O/read_ki/MECH4_KINPUT.json; need $O/read/GT0_ROLLHIST.json; need $O/read_ki/GT0_ROLLHIST_KI.json
[ "$(sha256sum $FLOOR_RH | cut -d' ' -f1)" = "$SHA_RH" ] || fail "gate0: the RH floor file changed (sha256 is not the pinned ${SHA_RH:0:12})"
[ "$(sha256sum $FLOOR_KI | cut -d' ' -f1)" = "$SHA_KI" ] || fail "gate0: the KI floor file changed (sha256 is not the pinned ${SHA_KI:0:12})"
for s in 0 1; do
  [ "$(ls $O/t1/I10UEA_R0C10XHb20_s$s/dump_I10UEA_R0C10XHb20_s${s}_m10_causal_heads 2>/dev/null | wc -l)" = 141 ] || fail "gate0: stage-A UEA dump R0C10XHb20 s$s (141 files) missing (the readers' spread arm)"
done
$PY -c "import json,sys; d=json.load(open('$E/C_EXEC.json')); sys.exit(0 if d.get('PASS') is True and not d.get('smoke_windows') else 1)" >> "$LOG" 2>&1 || fail "gate0: C_EXEC.json missing or not PASS"
# the pairs the read would use, per treated arm (printed in the log; the dry walk stops after this)
for K in RH KI; do
  echo "[pairs $K] $(RH_AKIND=$K $PY -c "import sys; sys.path.insert(0,'$E'); import rh_common as C; assert C.A_KIND=='$K'; print(' '.join('%s:%s'%(a,b) for a,b in C.PAIRS.values()))")" >> "$LOG"
done
echo "[plan] pull B0_s2 RH_s2 KI_s2 | rollouts: B0 RH KI x {$FORMS} seed $S (18) | T0: B0_s2 RH_s2 KI_s2 | floors: $FLOOR_RH ${SHA_RH:0:12} ; $FLOOR_KI ${SHA_KI:0:12}" >> "$LOG"
if [ -n "$RH_S3_DRY" ]; then
  for K in RH KI; do echo "PAIRS $K: $(RH_AKIND=$K $PY -c "import sys; sys.path.insert(0,'$E'); import rh_common as C; assert C.A_KIND=='$K'; print(' '.join('I10%s:I10%s'%(a,b) for a,b in C.PAIRS.values()))")"; done
  echo "PLAN: pull B0_s2 RH_s2 KI_s2; rollouts B0 RH KI x $FORMS at seed $S; T0 evals; floors pinned ${SHA_RH:0:12} ${SHA_KI:0:12}"
  echo "DRY WALK STOPS BEFORE THE PULL"; echo "ZZ SEED3-READ DRY STOP ZZ" >> "$LOG"; exit 0
fi
step blobcheck "tree blobs equal the commit" bash -c "$PY $E/blobcheck.py $E/tree 43319227 | head -1 | grep -q 'bad 0 '"
step readertests "rh_read.py selftest under RH_AKIND=RH and RH_AKIND=KI (K-SELF, K-PLANT, floor, seeds, harms, S3 floors, mutation)" bash -c "RH_AKIND=RH $PY -u $E/rh_read.py --selftest && RH_AKIND=KI $PY -u $E/rh_read.py --selftest"
step nametest "planted check of the name (five names, precedence, STRADDLE) and of mech_s3.py" $PY -I -u $E/plant_s3_check.py $E/tmp/plant_s3_chain
# ---- 1 the s2 checkpoints from the pod (read only)
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
for k in B0 RH KI; do step "pull_${k}_s$S" "pull and check ${k}_s$S" pull_arm ${k}_s$S; done
# ---- 2 rollouts: 3 arms x 6 forms
for k in B0 RH KI; do for f in $FORMS; do
  step "ro_${k}_${f}_s$S" "rollout $k $f seed $S" $PY -u $E/rh_roll.py $k $f $S
done; done
# ---- 3 G-T0 and MECH4 (T0)
step t0evals "T0 evals B0_s2 RH_s2 KI_s2" $PY -u $E/rh_t0.py run B0_s$S RH_s$S KI_s$S
step gt0_RH "G2: RH_s2 - B0_s2 at T0 ADE6, bar 0.1174, 0.0705 beside" env RH_AKIND=RH $PY -I -u $E/rh_t0.py read --seeds $S --bar 0.1174 --beside-bar 0.0705 --out $RD/GT0_S3_RH.json
step gt0_KI "KI_s2 - B0_s2 at T0 ADE6, bar 0.1174, 0.0705 beside" env RH_AKIND=KI $PY -I -u $E/rh_t0.py read --seeds $S --bar 0.1174 --beside-bar 0.0705 --out $RD/GT0_S3_KI.json
step mech "MECH4 at s2 (KI_s2 - RH_s2), three-seed list, s0/s1 G-T0 printed only" $PY -I -u $E/mech_s3.py $E $RD/MECH4_S3.json
# ---- 4 readers (frozen: headway_read_r2.py, q3_read_r3.py, rh_strat.py over strat_roll_read.py), one pass per treated arm
export HEADWAY_Q3DIR=$E/code/q3 HIERREP_STAGEB=$V/pod/stageB TANITEVAL_ROOT=$E/tree/taniteval HEADWAY_JOIN=F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz HEADWAY_BANK=$V/res/cache/frames.npz
SPREAD="--spread-arm I10UEA_R0C10XHb20 --spread-seeds 0,1"
for K in RH KI; do
  k=$(echo $K | tr 'A-Z' 'a-z'); D=$RD/$k; mkdir -p "$D"
  PAIRS=$(RH_AKIND=$K $PY -c "import sys; sys.path.insert(0,'$E'); import rh_common as C; assert C.A_KIND=='$K'; print(','.join('I10%s:I10%s'%(a,b) for a,b in C.PAIRS.values()))")
  [ -n "$PAIRS" ] || fail "empty pair list for $K"
  step headway_$K "headway reader U C F ($K pass, seed $S)" bash -c "cd $E/code/headway && RH_AKIND=$K $PY -u headway_read_r2.py read --t1-root $O/t1 --pairs '$PAIRS' --seeds $S $SPREAD --out-dir $D/hw"
  step q3_$K "family reader ($K pass, seed $S)" bash -c "cd $E/code/q3 && RH_AKIND=$K $PY -u q3_read_r3.py read --t1-root $O/t1 --pairs '$PAIRS' --seeds $S $SPREAD --same-sign --out-dir $D/q3"
  step strat_$K "STRATEGIC family (frozen ADD66 reader, one-seed mode, $K pass)" env RH_AKIND=$K $PY -u $E/rh_strat.py --out $D --seeds "$S"
done
# ---- 5 the reads and the name
step read_RH "rh_read.py RH pass: floors from the RH read, seed 2" env RH_AKIND=RH $PY -u $E/rh_read.py --q3 $RD/rh/q3/RESULT_Q3.json --hw $RD/rh/hw/RESULT_HEADWAY.json --strat $RD/rh/READ_STRAT_ROLLHIST.json --gt0 $RD/GT0_S3_RH.json --floors-from $FLOOR_RH --seeds $S --out $RD/RESULT_S3_RH.json
step read_KI "rh_read.py KI pass: floors from the KI read, seed 2" env RH_AKIND=KI $PY -u $E/rh_read.py --q3 $RD/ki/q3/RESULT_Q3.json --hw $RD/ki/hw/RESULT_HEADWAY.json --strat $RD/ki/READ_STRAT_ROLLHIST.json --gt0 $RD/GT0_S3_KI.json --floors-from $FLOOR_KI --seeds $S --out $RD/RESULT_S3_KI.json
[ -s "$RD/LEVER_ACTS_S3.json" ] || echo "[note] $RD/LEVER_ACTS_S3.json is absent: the name will read SEED3-BLOCKED until the MM writes it (then re-run s3_name.py only: rm $STEPS/name.ok)" >> "$LOG"
step name "s3_name.py: the card's section 4 name -> RESULT_SEED3.json" $PY -I -u $E/s3_name.py --rh $RD/RESULT_S3_RH.json --ki $RD/RESULT_S3_KI.json --gt0-rh $RD/GT0_S3_RH.json --gt0-ki $RD/GT0_S3_KI.json --mech $RD/MECH4_S3.json --lever $RD/LEVER_ACTS_S3.json --out $RD/RESULT_SEED3.json
echo "DONE" > "$O/ROLLHIST_READ_S3.done"
echo "ZZ SEED3-READ DONE ZZ" >> "$LOG"
