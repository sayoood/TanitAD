#!/bin/bash
# chk_arm_reps1.sh <mint|run> <UE_s1|UM_s1> A -- the ONE place the checker arguments of the two REPS1 arms are written (mint and run bind the same files). Pod side. Stage A only.
# FORK of block3/chk_arm3.sh (UE_s1: tree $D/tree, commit cdbff971, no tac flags) and blockum/chk_armum.sh (UM_s1: tree $D/tree_um, commit 648d4ca3, + --tacsrc-list --ue-argv --b-dry), each arm on the tree
# its seed-0 twin trained on. The checker is the patched copy $D/blockreps1/study/dscale10k_check.py (arms UE_s1, UM_s1 added; see patch_checker_reps1.py). UM_s1's --ue-argv is argv/UE_s1.A.json.
# DS_ROOT / DS_PY / DS_CHK / DS_TREE override for a dry run.
set -u
CMD=$1; ARM=$2; STAGE=$3
D=${DS_ROOT:-/root/dscale10k}
cd $D || exit 2
RP=${DS_RP:-$D/blockreps1}; PY=${DS_PY:-/root/venv/bin/python}; CHK=${DS_CHK:-$RP/study/dscale10k_check.py}
[ "$STAGE" = A ] || { echo "chk_arm_reps1.sh serves stage A only"; exit 2; }
case $ARM in
  UE_s1) TREE=${DS_TREE:-$D/tree};    COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb;;
  UM_s1) TREE=${DS_TREE:-$D/tree_um}; COMMIT=648d4ca3cab588d738f153872473bd3cb6f5e40e;;
  *) echo "chk_arm_reps1.sh serves UE_s1 and UM_s1"; exit 2;;
esac
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$RP/study:$D/study" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DS10K_TMP=$D/tmp PYTHONSAFEPATH=1
mkdir -p $D/tmp
ARGV=$D/argv/$ARM.A.json; BASE=$D/argv/B_s0.A.json; GD=$D/gate/tokens/$ARM.A
FILES="--evidence $D/gate/evidence_$ARM.json --dry $D/dry/$ARM.A.dry.json --merge-record $D/merged/merge_record.json --union-release $D/merged/v9_labels_union.npz --tenk-v9 $D/in/labels/v9_labels_datascale10k.npz"
[ $ARM = UM_s1 ] && FILES="$FILES --tacsrc-list $D/mix/bset_clips.txt --ue-argv $D/argv/UE_s1.A.json --b-dry $D/dry/B_s0.A.dry.json"
COMMON="--arm $ARM --stage $STAGE --argv $ARGV --root $TREE --commit $COMMIT --gate-dir $GD --py $PY --gpu 0"
if [ "$CMD" = mint ]; then
  MORE="--corpus-join /root/strat1/inputs/join_train.npz --union-join $D/merged/join_union.npz --ue-dry $D/dry/U_s0.A.dry.json"
  exec $PY $CHK mint $COMMON --base-argv $BASE $FILES $MORE --eval-ids /root/ops/clips141.txt --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ')
else
  exec $PY $CHK run $COMMON $FILES
fi
