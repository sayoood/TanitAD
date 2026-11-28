#!/bin/bash
# FORK of block3/chk_arm3.sh for the conditional arm X_s0 (the 10,000 data-scale clips alone; card 2.3 row X_s0, section 6 DATA-HURTS). Changes: blockx/FORK_NOTES.md. NOT launched: the MM asks the PI first.
# (block 3: chk_arm3.sh = block 2's chk_arm.sh with UE_s0 (union set) and BL_s0 (B set) added, and the MM-ruled checker from $D/block3/study (mint passes --ue-dry = U_s0's dry record to C-STEPS))
# chk_arm.sh <mint|run> <ARM> <A|B>  -- the ONE place the checker's arguments of an arm are written (mint and run must bind the same files). Pod side.
# Stage B reads runs/<ARM>/argv_B.json (the template with the arm's own init checkpoint, bank and bank sha256 filled in by the chain) and runs/<ARM>/B_selfcheck/selfcheck.json
# (the trainer's own --selfcheck-only on that argv). DS_ROOT / DS_PY / DS_CHK / DS_TREE can be overridden for a dry run; the commit is fixed.
set -u
CMD=$1; ARM=$2; STAGE=$3
D=${DS_ROOT:-/root/dscale10k}; B3=${DS_BX:-$D/blockx}; PY=${DS_PY:-/root/venv/bin/python}; CHK=${DS_CHK:-$B3/study/dscale10k_check.py}; TREE=${DS_TREE:-$D/tree}
COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$B3/study:$D/study" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DS10K_TMP=$D/tmp
mkdir -p $D/tmp
if [ "$STAGE" = A ]; then
  ARGV=$D/argv/$ARM.A.json; BASE=$D/argv/B_s0.A.json; GD=$D/gate/tokens/$ARM.A; SC=""
else
  ARGV=$D/runs/$ARM/argv_B.json; BASE=$D/argv/B_s0.B.tmpl.json; GD=$D/gate/tokens/$ARM.B; SC="--selfcheck $D/runs/$ARM/B_selfcheck/selfcheck.json"
fi
FILES="--evidence $D/gate/evidence_$ARM.json --dry $D/dry/$ARM.A.dry.json $SC"
case $ARM in
  U_s0|U_s1|UE_s0|X_s0) FILES="$FILES --merge-record $D/merged/merge_record.json --union-release $D/merged/v9_labels_union.npz --tenk-v9 $D/in/labels/v9_labels_datascale10k.npz";;
  R_s0) FILES="$FILES --merge-record $D/merged/merge_record.json --union-release $D/merged/v9_labels_union.npz --tenk-v9 $D/in/labels/v9_labels_datascale10k.npz --r-release $D/merged/v9_labels_R_s0.npz";;
esac
COMMON="--arm $ARM --stage $STAGE --argv $ARGV --root $TREE --commit $COMMIT --gate-dir $GD --py $PY --gpu 0"
if [ "$CMD" = mint ]; then
  MORE=""
  case $ARM in
    U_s0|U_s1|UE_s0|R_s0) MORE="--corpus-join /root/strat1/inputs/join_train.npz --union-join $D/merged/join_union.npz";;
  esac
  [ "$ARM" = X_s0 ] && MORE="--corpus-join $D/merged/join_union.npz --union-join $D/merged/join_X.npz"      # X: the UNION join sits in the corpus slot, the X join in the union slot (blockx study fork)
  [ "$ARM" = R_s0 ] && MORE="--corpus-join /root/strat1/inputs/join_train.npz --union-join $D/merged/join_R_s0.npz"
  exec $PY $CHK mint $COMMON --base-argv $BASE $FILES $MORE --eval-ids /root/ops/clips141.txt --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ')
else
  exec $PY $CHK run $COMMON $FILES
fi
