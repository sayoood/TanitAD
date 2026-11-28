#!/bin/bash
# DERIVED by k2dc/make_wboth_kit.py from the WISE-ENC script of the same role: WISE-BOTH = S_plan (20 prefixes) + core.encoder, alpha 0.5,
# pre-registered in e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM2.md (landed 8153ee7d, before any WISE-ENC result). Runs only if WISE-ENC fails (d).
# e2wboth_stageB_launch.sh -- MM 4a3d05, 2026-11-17: E2-WISE-BOTH STAGE B (planner clauses c/d/e) on model_wise_both_a050.pt of K2DC and K0 = e2wise_stageB_launch.sh with A/B/CK changed and clause (a) report-only.
# Pre-registered: e2/e2raw/PREREG_E2WISE_2026-11-17.md (ad9f1cbe). Guards as E2-TAIL (dump ckpt_md5 == the WISE file md5, step 6000).
#   Clause (a) on the WISE-BOTH weights is REPORTED, not a gate (prereg s3.2: stage A then B). REFUSES unless stage A ended DONE.
#   READ TREE = read_tree_9697fa4df004_k2dc_raw1c48ee8f (built for E2-RAW: K2DC's E2 read tree + the landed R9E2_CKPT_NAME roll 38161299 / env d4bdcf07).
#   SEEDS read/dump_{K2,K0} (hard links of stage A's WISE-BOTH DEV dumps; ckpt_md5 == the WISE file's md5, step 6000 checked) and read/EVAL_{K2,K0}.json
#     (stage A's FIT-threshold EVALs). K0's roll, mapsel and camera-blind run FRESH on the WISE-BOTH weights.
#   LAUNCH_DRY=1: every guard, then the roll's own dry run (full output to $R/logs/e2wboth_stageB_dry.txt); no seeding, no launch.
set -u
R=/home/nvidia/r8ns/r9e2; A=$R/runs/E2WISE_BOTH/stageA; B=$R/runs/E2WISE_BOTH/stageB; CK=model_wise_both_a050.pt
TREE=$R/read_tree_9697fa4df004_k2dc_raw1c48ee8f; ENVF=$R/inputs/r9e2_read_env.thor.sh_d4bdcf07
ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb; ENV_BLOB=d4bdcf07cf337758ee0770f72091a23e6bf4eb34
die() { echo "ZZE2WBOTHB-REFUSED $*"; exit 3; }
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
grep -q ' DONE stage A' $A/STATUS.txt || die "stage A has not ended DONE ($A/STATUS.txt)"
for X in K2 K0; do [ -s $A/EVAL_${X}wboth_fitthr.json ] && [ -s $A/devdump_$X/dump.json ] || die "stage A output missing for $X"; done
PASS=$(python3 -c 'import json,sys
p=lambda f: json.load(open(f,encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]["point"]
k2,k0=p(sys.argv[1]),p(sys.argv[2]); print(("A-PASS" if k2>k0 else "A-FAIL"), round(k2,6), round(k0,6))' $A/EVAL_K2wboth_fitthr.json $A/EVAL_K0wboth_fitthr.json)
echo "ZZE2WBOTHB-CLAUSE-A $PASS (reported, not a gate: PREREG_E2WISE s3.2 runs stage B after stage A regardless; (d) is the primary readout)"
for X in K2 K0; do
  case $X in K2) RUN=$R/runs/r9e2-K2DC;; K0) RUN=$R/runs/r9e2-K0;; esac
  [ -s $RUN/$CK ] && [ -s $RUN/config.json ] || die "$RUN lacks $CK or config.json"
  python3 -c 'import json,sys,hashlib
d=json.load(open(sys.argv[1],encoding="utf-8")); h=hashlib.md5(open(sys.argv[2],"rb").read()).hexdigest()
sys.exit(0 if d.get("ckpt_md5")==h and int(d.get("step",-1))==6000 else 1)' $A/devdump_$X/dump.json $RUN/$CK || die "$X DEV dump is not the WISE-BOTH file at step 6000"
done
[ -d $TREE/stack ] || die "no $TREE (built for E2-RAW)"
[ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die "tree roll blob != $ROLL_BLOB"
[ "$(b $ENVF)" = $ENV_BLOB ] || die "$ENVF blob != $ENV_BLOB"
[ ! -e $B ] || die "$B already exists (a read is never overwritten)"
if [ "${LAUNCH_DRY:-0}" = 1 ]; then
  R9E2_DRY=1 bash -c "export R9E2_TREE=$TREE R9E2_OUT=$B R9E2_CKPT_NAME=$CK; source $ENVF K2DC && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $R/logs/e2wboth_stageB_dry.txt 2>&1
  echo "ZZE2WBOTHB-DRY rc $? lines $(wc -l < $R/logs/e2wboth_stageB_dry.txt) ckpt-sites $(grep -o -- "--ckpt [^ ]*" $R/logs/e2wboth_stageB_dry.txt | sort | uniq -c | tr '\n' ' ') model_ema.pt $(grep -c 'model_ema[.]pt' $R/logs/e2wboth_stageB_dry.txt) require-ema $(grep -c -- --require-ema $R/logs/e2wboth_stageB_dry.txt)"
  exit 0
fi
mkdir -p $B/read || die "cannot create $B"
for X in K2 K0; do cp -al $A/devdump_$X $B/read/dump_$X && ln $A/EVAL_${X}wboth_fitthr.json $B/read/EVAL_$X.json || die "seeding $X failed"; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SEEDED-FROM-STAGE-A read/dump_{K2,K0} (WISE-BOTH DEV dumps) + read/EVAL_{K2,K0}.json (FIT-threshold EVALs) from $A; $PASS" >> $B/ROLL_STATUS.txt
LOG=$R/logs/e2wboth_stageB_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup bash -c "export R9E2_TREE=$TREE R9E2_OUT=$B R9E2_CKPT_NAME=$CK; source $ENVF K2DC && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $LOG 2>&1 < /dev/null &
sleep 5
echo "ZZE2WBOTHB-STARTED pid $! log $LOG"
head -n 5 $LOG | cut -c1-220
