#!/bin/bash
# chk_arm_rh.sh <mint|run> TAG   (TAG = RH_s0 | B0_s0 | RH_s1 | B0_s1)   Pod side. The ONE place the checker's arguments of a ROLLHIST arm are written, so `mint` and `run` bind the same files
# (the pattern of chk_arm.sh / chk_arm3.sh: the stage-B arm script gates the trainer on the token the checker minted). Checker = dscale10k_check_rh.py (this build), sourced from $R/study/
# together with dscale10k_merge.py (byte-identical to /root/dscale10k/study/dscale10k_merge.py). Needs RH_READOUT in the environment (rh_env.sh rh_readout_flags sets DECL_TOKEN).
# Inputs: $R/runs/TAG/argv.json (filled), the pair's argv.json, $R/runs/TAG/selfcheck/selfcheck.json, $R/bank/sK (RH), $R/gsuite_result_rh.json, the DSCALE10K stage-A evidence/dry of UE_sK.
set -u
CMD=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rh_env.sh
rh_readout_flags; export_py
seed=${TAG##*_s}; kind=${TAG%%_*}; D=$R/runs/$TAG; GD=$D/gate
CHK=${DS_CHK:-$R/study/dscale10k_check_rh.py}
[ "$kind" = RH ] && PAIR=B0_s$seed || PAIR=RH_s$seed
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$R/study" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_DISABLE_XET=1
mkdir -p $GD
# the evidence: UE_sK's stage-A evidence (hyg / reach / clock / G-EVAL / G-CKPT: INHERITED, labelled so inside the file) + THIS commit's G-SUITE
EV=$GD/evidence_$TAG.json
if [ "$CMD" = mint ] || [ ! -s $EV ]; then
  $PY $HERE/rh_evidence.py --ue-evidence $DS/gate/evidence_UE_s$seed.json --gsuite $R/gsuite_result_rh.json --commit $COMMIT --out $EV > $GD/evidence.log 2>&1 || { echo "REFUSED: evidence: $(tail -c 300 $GD/evidence.log)"; exit 1; }
fi
COMMON="--arm $TAG --stage B --argv $D/argv.json --root $TREE --commit $COMMIT --gate-dir $GD --py $PY --gpu 0"
FILES="--evidence $EV --dry $DS/dry/UE_s$seed.A.dry.json --selfcheck $D/selfcheck/selfcheck.json --merge-record $DS/merged/merge_record.json --union-release $DS/merged/v9_labels_union.npz --tenk-v9 $DS/in/labels/v9_labels_datascale10k.npz"
FILES="$FILES --template $DS/argv/U_s$seed.B.tmpl.json --pair-argv $R/runs/$PAIR/argv.json --pair-check $TREE/stack/scripts/rollhist_pair_check.py"
[ "$kind" = RH ] && FILES="$FILES --readout-token $DECL_TOKEN"          # the declaration is a CLI argument of the RH mint (and of its run); B0 takes none
if [ "$CMD" = mint ]; then
  exec $PY $CHK mint $COMMON --base-argv $DS/argv/B_s0.B.tmpl.json $FILES --corpus-join /root/strat1/inputs/join_train.npz --union-join $DS/merged/join_union.npz \
       --eval-ids /root/ops/clips141.txt --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ')
else
  exec $PY $CHK run $COMMON $FILES      # REFUSES (rc 2) without a token that binds this exact argv, commit, data, tree, trainer, checker, template and pair; then starts the trainer under that argv
fi
