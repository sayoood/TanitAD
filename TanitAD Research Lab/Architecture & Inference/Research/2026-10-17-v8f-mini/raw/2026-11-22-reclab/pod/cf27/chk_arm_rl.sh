#!/bin/bash
# chk_arm_rl.sh <mint|run> TAG   (TAG = RL_s0 | RL_s1)   Pod side. THIN COPY of chk_arm_rh.sh (sha256 b699452b..., Amendment 6's `cd $DS` kept) for the ROLLHIST next arm RL (Amendment 7 DRAFT).
# Differences from chk_arm_rh.sh, and only these:
#   * the checker is $R/study/dscale10k_check_rl.py (dscale10k_check_rhk.py 1bd28255 + the RL kind; patch_rl.py): the RL kind is pinned to RH's argv plus exactly the two tokens `--dagger-label speed_v1` (cf27dbb7 form: rl_env.sh, gsuite_result_rl.json);
#   * kind RL is an RH-like arm: the pair is the EXISTING B0_s$seed (no new B0), and the readout declaration is a CLI argument of the mint and of the run.
# Everything else (evidence, files, `cd $DS`, the template, the pair check, the base argv) is byte for byte the RH call.
set -u
CMD=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rl_env.sh
rh_readout_flags; export_py
seed=${TAG##*_s}; kind=${TAG%%_*}; D=$R/runs/$TAG; GD=$D/gate; SC=$D/selfcheck/selfcheck.json
# DRY MINT (DRY_MINT_PLAN.md): `mint` only, before the arm exists. DRY_GD = a scratch gate dir, DRY_SC = a finished selfcheck.json of the SAME seed's RH arm (the p = 0.3 selfcheck needs the GPU). `run` REFUSES when either is set,
# so a dry token can never start a trainer; the dry token lives only in DRY_GD (named dry_*), never in runs/$TAG/gate.
if [ -n "${DRY_GD:-}" ] || [ -n "${DRY_SC:-}" ]; then
  [ "$CMD" = mint ] || { echo "REFUSED: DRY_GD / DRY_SC are set: the dry form is mint only"; exit 1; }
  case "${DRY_GD:-}" in */dry_*) ;; *) echo "REFUSED: DRY_GD must be a path containing /dry_ (got '${DRY_GD:-}')"; exit 1;; esac
  [ -s "${DRY_SC:-}" ] || { echo "REFUSED: DRY_SC is not a file"; exit 1; }
  GD=$DRY_GD; SC=$DRY_SC
fi
[ "$kind" = RL ] || { echo "REFUSED: $TAG is not an RL arm"; exit 1; }
CHK=${DS_CHK:-$R/study/dscale10k_check_rl.py}
PAIR=B0_s$seed
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$R/study" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_DISABLE_XET=1
mkdir -p $GD
EV=$GD/evidence_$TAG.json
if [ "$CMD" = mint ] || [ ! -s $EV ]; then
  $PY $HERE/rh_evidence.py --ue-evidence $DS/gate/evidence_UE_s$seed.json --gsuite $GSUITE_RL --commit $COMMIT --out $EV > $GD/evidence.log 2>&1 || { echo "REFUSED: evidence: $(tail -c 300 $GD/evidence.log)"; exit 1; }
fi
COMMON="--arm $TAG --stage B --argv $D/argv.json --root $TREE --commit $COMMIT --gate-dir $GD --py $PY --gpu 0"
FILES="--evidence $EV --dry $DS/dry/UE_s$seed.A.dry.json --selfcheck $SC --merge-record $DS/merged/merge_record.json --union-release $DS/merged/v9_labels_union.npz --tenk-v9 $DS/in/labels/v9_labels_datascale10k.npz"
FILES="$FILES --template $DS/argv/U_s$seed.B.tmpl.json --pair-argv $R/runs/$PAIR/argv.json --pair-check $TREE/stack/scripts/rollhist_pair_check.py"
FILES="$FILES --readout-token $DECL_TOKEN"
cd $DS                  # Amendment 6: the DSCALE10K merge record lists in/manifest_10k.jsonl RELATIVE to /root/dscale10k (C-MERGE re-hashes it); every other path here is absolute
if [ "$CMD" = mint ]; then
  exec $PY $CHK mint $COMMON --base-argv $DS/argv/B_s0.B.tmpl.json $FILES --corpus-join /root/strat1/inputs/join_train.npz --union-join $DS/merged/join_union.npz \
       --eval-ids /root/ops/clips141.txt --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ')
else
  exec $PY $CHK run $COMMON $FILES      # REFUSES (rc 2) without a token that binds this exact argv, commit, data, tree, trainer, checker, template and pair; then starts the trainer under that argv
fi
