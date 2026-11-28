#!/bin/bash
# e2raw_stageB_launch.sh -- MM 4a3d05, 2026-11-16: E2-RAW STAGE B (planner clauses c/d/e) on the RAW ckpt.pt of K2DC and K0.
# Pre-registered: dagland/e2/PREREG_E2RAW_2026-11-16.md (repo e2/e2raw/, landed 232f8738). Switch landed 1c48ee8f (R9E2_CKPT_NAME, R9E2_OUT).
#   REFUSES unless stage A ended ' DONE stage A' AND clause (a) passed: K2raw fitted primary > K0raw fitted primary (point) -- the prereg runs stage B only then.
#   READ TREE = read_tree_9697fa4df004_k2dc hard-linked (cp -al) + EXACTLY the landed r9e2_roll.sh (blob 38161299) and r9e2_read_env.thor.sh (d4bdcf07),
#     written with cp --remove-destination so the original tree's files are never touched.
#   SEEDS read/dump_{K2,K0} (hard links of stage A's raw DEV dumps, weights_are_ema false, step 6000 checked) and read/EVAL_{K2,K0}.json (stage A's FIT-threshold
#     EVALs: the verdict scores clause (a) from them, r9e2_inputs.py:65 / r9_e2_verdict.py:185). K0's roll, mapsel and camera-blind are run FRESH on raw weights.
#   LAUNCH_DRY=1: every guard, then `R9E2_DRY=1 r9e2_roll.sh` (prints only); no tree build, no seeding, no launch.
set -u
R=/home/nvidia/r8ns/r9e2; A=$R/runs/E2RAW_K2DC/stageA; B=$R/runs/E2RAW_K2DC/stageB
OLDT=$R/read_tree_9697fa4df004_k2dc; TREE=$R/read_tree_9697fa4df004_k2dc_raw1c48ee8f
ROLLF=$R/inputs/r9e2_roll.sh_38161299; ENVF=$R/inputs/r9e2_read_env.thor.sh_d4bdcf07
ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb; ENV_BLOB=d4bdcf07cf337758ee0770f72091a23e6bf4eb34
die() { echo "ZZE2RAWB-REFUSED $*"; exit 3; }
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
grep -q ' DONE stage A' $A/STATUS.txt || die "stage A has not ended DONE ($A/STATUS.txt)"
for X in K2 K0; do [ -s $A/EVAL_${X}raw_fitthr.json ] && [ -s $A/devdump_$X/dump.json ] || die "stage A output missing for $X"; done
PASS=$(python3 -c 'import json,sys
p=lambda f: json.load(open(f,encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]["point"]
k2,k0=p(sys.argv[1]),p(sys.argv[2]); print(("A-PASS" if k2>k0 else "A-FAIL"), round(k2,6), round(k0,6))' $A/EVAL_K2raw_fitthr.json $A/EVAL_K0raw_fitthr.json)
case "$PASS" in A-PASS*) echo "ZZE2RAWB-CLAUSE-A $PASS";; *) die "clause (a) on raw weights: $PASS -- prereg: stage B is NOT run (outcome 2)";; esac
for X in K2 K0; do python3 -c 'import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); sys.exit(0 if d.get("weights_are_ema") is False and int(d.get("step",-1))==6000 else 1)' $A/devdump_$X/dump.json || die "$X DEV dump is not raw step 6000"; done
for d in $R/runs/r9e2-K0 $R/runs/r9e2-K2DC; do [ -s $d/ckpt.pt ] && [ -s $d/config.json ] || die "$d lacks ckpt.pt or config.json"; done
[ "$(b $ROLLF)" = $ROLL_BLOB ] || die "$ROLLF blob != $ROLL_BLOB"
case "$(b $ENVF)" in $ENV_BLOB*) ;; *) die "$ENVF blob is not $ENV_BLOB";; esac
grep -q 'R9E2_OUT="${R9E2_OUT:-' $ENVF || die "env does not let R9E2_OUT be overridden"
[ -d $OLDT/stack ] || die "no $OLDT"
[ ! -e $B ] || die "$B already exists (a read is never overwritten)"
if [ "${LAUNCH_DRY:-0}" = 1 ]; then
  [ -d $TREE/stack ] || { echo "ZZE2RAWB-DRY-GUARDS-OK (tree not built yet; the roll dry run needs it)"; exit 0; }
  R9E2_DRY=1 bash -c "export R9E2_TREE=$TREE R9E2_OUT=$B R9E2_CKPT_NAME=ckpt.pt; source $ENVF K2DC && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" | tail -n 8; exit 0
fi
if [ ! -d $TREE/stack ]; then
  cp -al $OLDT $TREE || die "cp -al $OLDT failed"
  cp --remove-destination $ROLLF $TREE/stack/ops/r9e2_read/r9e2_roll.sh && cp --remove-destination $ENVF $TREE/stack/ops/r9e2_read/r9e2_read_env.thor.sh || die "tree file replace failed"
  echo "read_tree_9697fa4df004_k2dc hard-linked + r9e2_roll.sh $ROLL_BLOB + r9e2_read_env.thor.sh $ENV_BLOB (landed 1c48ee8f) -- built $(date -u +%Y-%m-%dT%H:%M:%SZ) by e2raw_stageB_launch.sh" > $TREE.provenance
fi
[ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die "tree roll blob wrong"
[ "$(b $OLDT/stack/ops/r9e2_read/r9e2_roll.sh)" = b4d141b9a540901fb44e3c3f644d11ff8e1535e5 ] || die "ORIGINAL tree's roll changed: the hard-link replace leaked"
mkdir -p $B/read || die "cannot create $B"
for X in K2 K0; do cp -al $A/devdump_$X $B/read/dump_$X && ln $A/EVAL_${X}raw_fitthr.json $B/read/EVAL_$X.json || die "seeding $X failed"; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SEEDED-FROM-STAGE-A read/dump_{K2,K0} (raw DEV dumps) + read/EVAL_{K2,K0}.json (FIT-threshold EVALs) from $A; $PASS" >> $B/ROLL_STATUS.txt
LOG=$R/logs/e2raw_stageB_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup bash -c "export R9E2_TREE=$TREE R9E2_OUT=$B R9E2_CKPT_NAME=ckpt.pt; source $ENVF K2DC && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $LOG 2>&1 < /dev/null &
sleep 5
echo "ZZE2RAWB-STARTED pid $! log $LOG"
head -n 5 $LOG | cut -c1-220
