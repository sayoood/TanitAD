#!/bin/bash
# chk_arm_s3.sh <mint|run> TAG   (TAG = B0_s2 | RH_s2 | KI_s2)   Pod side. THIN COPY of chk_arm_ki.sh (itself of chk_arm_rh.sh) for the three SEED3 arms. Differences, and only these:
#   * the checker is $HERE/dscale10k_check_s3.py (dscale10k_check_ki.py + the arms B0_s2 / RH_s2 / KI_s2 and U_s2); the template is $HERE/U_s2.B.tmpl.json (the kit's: no U_s2.B.tmpl.json exists under /root/dscale10k/argv);
#   * the tree / commit / G-SUITE are the arm's own (s3_env.sh s3_arm_env): B0 / RH at tree_cdb 97fe0a56, KI at tree_ki 4097a89e;
#   * the pair: RH and KI against the finished-by-then B0_s2; B0_s2 (trained FIRST, before the bank exists) against runs/B0_s2/pair_rh_skeleton.json = RH_s2's argv with the bank sha256 spelled @BANKSHA@ (s3_argv.py);
#   * the readout declaration (--readout-token) is a CLI argument of the RH and KI mint and run; B0 takes none; the UE_s2 stage-A evidence / dry record are $DS/gate/evidence_UE_s2.json and $DS/dry/UE_s2.A.dry.json.
# DRY mint (mint only): DRY_GD = a scratch gate dir containing /dry_, DRY_SC = a finished selfcheck.json of this arm; `run` REFUSES when either is set.
set -u
CMD=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/s3_env.sh
s3_arm_env $TAG || exit 1
rh_readout_flags; export_py
seed=${TAG##*_s}; kind=${TAG%%_*}; D=$R/runs/$TAG; GD=$D/gate; SC=$D/selfcheck/selfcheck.json
[ "$seed" = 2 ] || { echo "REFUSED: $TAG is not a seed-2 arm"; exit 1; }
if [ -n "${DRY_GD:-}" ] || [ -n "${DRY_SC:-}" ]; then
  [ "$CMD" = mint ] || { echo "REFUSED: DRY_GD / DRY_SC are set: the dry form is mint only"; exit 1; }
  case "${DRY_GD:-}" in */dry_*) ;; *) echo "REFUSED: DRY_GD must be a path containing /dry_ (got '${DRY_GD:-}')"; exit 1;; esac
  [ -s "${DRY_SC:-}" ] || { echo "REFUSED: DRY_SC is not a file"; exit 1; }
  GD=$DRY_GD; SC=$DRY_SC
fi
CHK=${DS_CHK:-$HERE/dscale10k_check_s3.py}
case $kind in B0) PAIRARGV=$R/runs/B0_s2/pair_rh_skeleton.json;; RH|KI) PAIRARGV=$R/runs/B0_s2/argv.json;; *) echo "REFUSED: $TAG"; exit 1;; esac
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$R/study" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${S3_OMP:-4} MKL_NUM_THREADS=${S3_OMP:-4} HF_HUB_DISABLE_XET=1
mkdir -p $GD
EV=$GD/evidence_$TAG.json
if [ "$CMD" = mint ] || [ ! -s $EV ]; then
  $PY $R/rh_evidence.py --ue-evidence $DS/gate/evidence_UE_s2.json --gsuite $GSUITE --commit $COMMIT --out $EV > $GD/evidence.log 2>&1 || { echo "REFUSED: evidence: $(tail -c 300 $GD/evidence.log)"; exit 1; }
fi
COMMON="--arm $TAG --stage B --argv $D/argv.json --root $TREE --commit $COMMIT --gate-dir $GD --py $PY --gpu 0"
FILES="--evidence $EV --dry $DS/dry/UE_s2.A.dry.json --selfcheck $SC --merge-record $DS/merged/merge_record.json --union-release $DS/merged/v9_labels_union.npz --tenk-v9 $DS/in/labels/v9_labels_datascale10k.npz"
FILES="$FILES --template $HERE/U_s2.B.tmpl.json --pair-argv $PAIRARGV --pair-check $TREE/stack/scripts/rollhist_pair_check.py"
[ "$kind" = B0 ] || FILES="$FILES --readout-token $DECL_TOKEN"
cd $DS                  # the DSCALE10K merge record lists in/manifest_10k.jsonl RELATIVE to /root/dscale10k (C-MERGE re-hashes it); every other path here is absolute
if [ "$CMD" = mint ]; then
  exec $PY $CHK mint $COMMON --base-argv $DS/argv/B_s0.B.tmpl.json $FILES --corpus-join /root/strat1/inputs/join_train.npz --union-join $DS/merged/join_union.npz \
       --eval-ids /root/ops/clips141.txt --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ')
else
  exec $PY $CHK run $COMMON $FILES      # REFUSES (rc 2) without a token that binds this exact argv, commit, data, tree, trainer, checker, template and pair; then starts the trainer under that argv
fi
