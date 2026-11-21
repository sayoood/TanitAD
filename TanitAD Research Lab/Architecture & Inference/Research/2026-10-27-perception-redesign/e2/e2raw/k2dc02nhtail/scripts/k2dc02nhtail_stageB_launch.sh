#!/bin/bash
# k2dc02nhtail_stageB_launch.sh -- K2DC02nh-TAIL STAGE B (planner clauses c/d/e, the lever rule's planner rows) on K2DC02nh's TAIL-EMA weights. Pre-registered:
#   e2/e2raw/PREREG_K2DC02NH_2026-11-20.md. = e2tail_stageB_launch.sh with K2 = r9e2-K2DC02nh (staging run dir runs/E2TAIL_K2DC02nh/run_K2DC02nh) and the K0 slot SEEDED, not rolled.
#   REFUSES unless stage A ended ' DONE stage A' AND clause (a) passed: K2DC02nht fitted primary > K0t fitted primary (point; K0t = the banked E2-TAIL EVAL_K0tail_fitthr.json).
#   READ TREE = read_tree_<K2S_TRAIN_COMMIT>_k2dc (K2DC02nh's EMA-read tree; roll blob 38161299, env d4bdcf07, both verified here, == E2-TAIL's pins).
#   ROLL = the stock r9e2_roll.sh with R9E2_CKPT_NAME=model_ema_tail5k6k.pt. It rolls K0 then K2; K0's stages are SKIPPED when their outputs exist, so the K0 slot is SEEDED:
#     $CMP = the K2DC02-tail read's cmp/ ($W0/cmp, PINNED) must hold E2_K0_pf (complete: r9e2_dev_filter.json, dump_s0, dump_s1) = the K0-TAIL planner roll, which EXISTS (881 of the 3,524 pinned files). ⛔ Nothing is re-rolled
#     (PREREG_K2DC02S1 sec. 2 item 4): the launcher REFUSES unless k2dc02nhtail_pin_check.sh passes, and it COPIES the roll into $B and re-hashes the copy against the pin.
#     Also seeded (small): E2_K0_mapsel/trace_s{0,1}.npz (from E2-TAIL's dev-box pull, sha256 vs its thor.sha256), read/dump_K0/dump.json (HEADER-ONLY stub) and read/EVAL_K0.json
#     (the banked E2-TAIL one): all so that the roll skips K0's GPU stages. The K0 slot's full dump / rasters are assembled on the dev box from E2-TAIL's pull (k2dc02nhtail_s6c_pull_score.sh).
#   Seeds also read/dump_K2 (hard link of stage A's DEV dump, ckpt_md5 == the tail's md5 and step 6000 checked) and read/EVAL_K2.json (stage A's FIT-threshold EVAL).
#   LAUNCH_DRY=1: every guard, then the roll's own dry run (full output to $R/logs/k2dc02nhtail_stageB_dry.txt); no seeding, no launch.
# Differences from k2dc02tail_stageB_launch.sh (PREREG_K2DC02TAIL sec. 2 item 3 and PREREG_K2DC02S1 sec. 2 items 4, 5): ARM K2DC02nh; the FIRST guard is that the trained arm exists; the K0 slot is the PINNED, EXISTING K0t roll of the
# K2DC02-tail read (W0/cmp/E2_K0_pf, no comparator re-roll), re-hashed against K2DC02S1_CMP_PIN.sha256 by k2dc02nhtail_pin_check.sh before anything reads it, and COPIED (cp -a, never hard-linked: the pinned
# files must stay untouched) into this read's stageB, then the copy is re-hashed against the pin; K0_CMP / K0_SEED overrides are removed (a pin is not overridable); the tree is read_tree_<K2S_TRAIN_COMMIT>_k2dc.
# DRY: K2S_DRY=1 or first arg DRY (= LAUNCH_DRY=1).
set -u
if [ "${K2S_DRY:-0}" = 1 ] || [ "${1:-}" = DRY ]; then LAUNCH_DRY=1; fi
TC=${K2S_TRAIN_COMMIT:-}
R=/home/nvidia/r8ns/r9e2; W0=$R/runs/E2TAIL_K2DC02; W=$R/runs/E2TAIL_K2DC02nh; A=$W/stageA; B=$W/stageB; RUN=$W/run_K2DC02nh; CK=model_ema_tail5k6k.pt
CMP=$W0/cmp; K0SEED=$W0/inputs/k0seed; PINMAN=$W0/K2DC02S1_CMP_PIN.sha256      # K0SEED: trace_s{0,1}.npz, dump_K0.dump.json, EVAL_K0tail_fitthr.json (the K2DC02 tail's shipped, sha256-checked seeds; read only)
TREE=$R/read_tree_${TC}_k2dc; ENVF=$R/inputs/r9e2_read_env.thor.sh_d4bdcf07
ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb; ENV_BLOB=d4bdcf07cf337758ee0770f72091a23e6bf4eb34
die() { echo "ZZK2DC02NHTAILB-REFUSED $*"; exit 3; }
[ -s $R/runs/r9e2-K2DC02nh/model_ema.pt ] && [ -s $R/runs/r9e2-K2DC02nh/config.json ] || die "guard 1: $R/runs/r9e2-K2DC02nh lacks model_ema.pt / config.json: K2DC02nh has not trained"
[ ${#TC} = 12 ] || die "K2S_TRAIN_COMMIT (the 12-char commit K2DC02nh trained from) is required, got '$TC'"
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
grep -q ' DONE stage A' $A/STATUS.txt || die "stage A has not ended DONE ($A/STATUS.txt)"
[ -s $A/EVAL_K2tail_fitthr.json ] && [ -s $A/devdump_K2/dump.json ] || die "stage A output missing for K2"
[ -s $K0SEED/EVAL_K0tail_fitthr.json ] || die "no $K0SEED/EVAL_K0tail_fitthr.json (the banked E2-TAIL K0 fitted EVAL, shipped from e2/e2raw/e2tail/stageA/)"
PASS=$(python3 -c 'import json,sys
p=lambda f: json.load(open(f,encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]["point"]
k2,k0=p(sys.argv[1]),p(sys.argv[2]); print(("A-PASS" if k2>k0 else "A-FAIL"), round(k2,6), round(k0,6))' $A/EVAL_K2tail_fitthr.json $K0SEED/EVAL_K0tail_fitthr.json)
case "$PASS" in A-PASS*) echo "ZZK2DC02NHTAILB-CLAUSE-A $PASS";; *) die "clause (a) on tail weights: $PASS -- PREREG_K2DC02NH sec. 4 item 5 (TAIL-VOID / diagnose), stage B NOT run";; esac
[ -s $RUN/$CK ] && [ -s $RUN/config.json ] || die "$RUN lacks $CK or config.json"
python3 -c 'import json,sys,hashlib
d=json.load(open(sys.argv[1],encoding="utf-8")); h=hashlib.md5(open(sys.argv[2],"rb").read()).hexdigest()
sys.exit(0 if d.get("ckpt_md5")==h and int(d.get("step",-1))==6000 else 1)' $A/devdump_K2/dump.json $RUN/$CK || die "K2 DEV dump is not the tail file at step 6000"
[ -d $TREE/stack ] || die "no $TREE"
bash "$(cd "$(dirname "$0")" && pwd)/k2dc02nhtail_pin_check.sh" > /dev/null || die "pin check of the K2DC02 tail's comparator rolls REFUSED (run k2dc02nhtail_pin_check.sh to see why): nothing is read"
[ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die "tree roll blob != $ROLL_BLOB"
[ "$(b $ENVF)" = $ENV_BLOB ] || die "$ENVF blob != $ENV_BLOB"
for f in r9e2_dev_filter.json dump_s0/manifest.json dump_s1/manifest.json windows_s0.json windows_s1.json; do [ -s $CMP/E2_K0_pf/$f ] || die "pinned K0-tail planner roll missing: $CMP/E2_K0_pf/$f (the K2DC02-tail read's roll, pinned in $PINMAN)"; done
for f in trace_s0.npz trace_s1.npz dump_K0.dump.json; do [ -s $K0SEED/$f ] || die "no $K0SEED/$f (ship from E2-TAIL's dev-box pull, sha256-checked against its thor.sha256)"; done
[ ! -e $B ] || die "$B already exists (a read is never overwritten)"
FREE=$(df -BM --output=avail /home/nvidia | tail -1 | tr -dc 0-9); [ "${FREE:-0}" -gt $((15360 + 800)) ] || die "free ${FREE} MB: stage B needs ~0.8 GB above the 15 GB floor"
if [ "${LAUNCH_DRY:-0}" = 1 ]; then
  R9E2_DRY=1 bash -c "export R9E2_TREE=$TREE R9E2_OUT=$B R9E2_CKPT_NAME=$CK; source $ENVF K2DC02nh && export R9E2_K2_RUN=$RUN && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $R/logs/k2dc02nhtail_stageB_dry.txt 2>&1
  echo "ZZK2DC02NHTAILB-DRY rc $? lines $(wc -l < $R/logs/k2dc02nhtail_stageB_dry.txt) ckpt-sites $(grep -o -- "--ckpt [^ ]*" $R/logs/k2dc02nhtail_stageB_dry.txt | sort | uniq -c | tr '\n' ' ') model_ema.pt $(grep -c 'model_ema[.]pt' $R/logs/k2dc02nhtail_stageB_dry.txt) require-ema $(grep -c -- --require-ema $R/logs/k2dc02nhtail_stageB_dry.txt)"
  exit 0
fi
mkdir -p $B/read $B/E2_K0_mapsel || die "cannot create $B"
cp -a $CMP/E2_K0_pf $B/E2_K0_pf && grep ' cmp/E2_K0_pf/' $PINMAN | sed 's| cmp/E2_K0_pf/| E2_K0_pf/|' | ( cd $B && sha256sum -c --quiet ) && cp -p $K0SEED/trace_s0.npz $K0SEED/trace_s1.npz $B/E2_K0_mapsel/ && mkdir -p $B/read/dump_K0 && cp -p $K0SEED/dump_K0.dump.json $B/read/dump_K0/dump.json \
  && cp -p $K0SEED/EVAL_K0tail_fitthr.json $B/read/EVAL_K0.json && cp -al $A/devdump_K2 $B/read/dump_K2 && ln $A/EVAL_K2tail_fitthr.json $B/read/EVAL_K2.json || die "seeding failed (cp -al needs one filesystem); $B left for inspection"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SEEDED K0 slot: E2_K0_pf (a copy of the pinned $CMP/E2_K0_pf, re-hashed against the pin), E2_K0_mapsel/trace_s{0,1}.npz, read/dump_K0/dump.json (HEADER-ONLY STUB), read/EVAL_K0.json (banked E2-TAIL K0 fitted EVAL); K2 slot: read/dump_K2 + read/EVAL_K2.json from stage A; $PASS" >> $B/ROLL_STATUS.txt
LOG=$R/logs/k2dc02nhtail_stageB_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup bash -c "export R9E2_TREE=$TREE R9E2_OUT=$B R9E2_CKPT_NAME=$CK; source $ENVF K2DC02nh && export R9E2_K2_RUN=$RUN && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $LOG 2>&1 < /dev/null &
sleep 5
echo "ZZK2DC02NHTAILB-STARTED pid $! log $LOG"
head -n 5 $LOG | cut -c1-220
