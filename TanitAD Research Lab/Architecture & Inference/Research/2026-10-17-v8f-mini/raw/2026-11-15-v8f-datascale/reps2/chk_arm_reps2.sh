#!/bin/bash
# chk_arm_reps2.sh <mint|run> <UE_s2|UE_s3> A -- the ONE place the checker arguments of the two REPS2 arms are written (mint and run bind the same files). Pod side. Stage A only.
# FORK of chk_arm_reps1.sh (UE_s1: tree $D/tree, commit cdbff971, no tac flags): the same tree and commit, the checker is the patched copy $D/blockreps2/study/dscale10k_check.py (arms UE_s2, UE_s3 added;
# patch_checker_reps2.py), and --rep-ref-argv argv/UE_s0.A.json is bound into the token at mint and re-checked by the runner.
# DS_ROOT / DS_PY / DS_CHK / DS_TREE / DS_RP / DS_GD / DS_REFARGV / DS_EVIDENCE / DS_DRYREC override for a dry run (dry_reps2); in the real run none is set.
set -u
CMD=$1; ARM=$2; STAGE=$3
D=${DS_ROOT:-/root/dscale10k}
cd $D || exit 2
RP=${DS_RP:-$D/blockreps2}; PY=${DS_PY:-/root/venv/bin/python}; CHK=${DS_CHK:-$RP/study/dscale10k_check.py}
[ "$STAGE" = A ] || { echo "chk_arm_reps2.sh serves stage A only"; exit 2; }
case $ARM in
  UE_s2|UE_s3) TREE=${DS_TREE:-/root/dscale10k/tree}; COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb;;
  *) echo "chk_arm_reps2.sh serves UE_s2 and UE_s3"; exit 2;;
esac
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$RP/study:$D/study" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DS10K_TMP=$D/tmp PYTHONSAFEPATH=1
mkdir -p $D/tmp
ARGV=$D/argv/$ARM.A.json; BASE=${DS_BASEARGV:-$D/argv/B_s0.A.json}; GD=${DS_GD:-$D/gate/tokens/$ARM.A}; REFARGV=${DS_REFARGV:-$D/argv/UE_s0.A.json}
EVID=${DS_EVIDENCE:-$D/gate/evidence_$ARM.json}; DRYREC=${DS_DRYREC:-$D/dry/$ARM.A.dry.json}
FILES="--evidence $EVID --dry $DRYREC --merge-record $D/merged/merge_record.json --union-release $D/merged/v9_labels_union.npz --tenk-v9 $D/in/labels/v9_labels_datascale10k.npz --rep-ref-argv $REFARGV"
COMMON="--arm $ARM --stage $STAGE --argv $ARGV --root $TREE --commit $COMMIT --gate-dir $GD --py $PY --gpu 0"
if [ "$CMD" = mint ]; then
  MORE="--corpus-join /root/strat1/inputs/join_train.npz --union-join $D/merged/join_union.npz --ue-dry $D/dry/U_s0.A.dry.json"
  exec $PY $CHK mint $COMMON --base-argv $BASE $FILES $MORE --eval-ids /root/ops/clips141.txt --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ')
else
  exec $PY $CHK run $COMMON $FILES
fi
