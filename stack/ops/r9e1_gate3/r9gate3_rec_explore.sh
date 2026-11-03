#!/usr/bin/env bash
# exploratory (non-token) run of the gate's own record job on the shipped tip tree, arm $1 (study R9_STUDY: e1 = default; root / argv prefix / profile from r9gate3_study.sh NEXT TO THIS SCRIPT)
ARM=$1; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/r9gate3_study.sh" || { echo "ZZR9E1G3-${ARM}-rec-explore-REFUSEDZZ"; exit 10; }    # R9_ROOT R9_PREFIX R9_PROFILE ... (one definition; refuses an unknown study)
R=$R9_ROOT; T=$R/launch_tree; SHA=$(cat $R/launch_tree.commit); C12=${SHA:0:12}
ARGV=$T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json
REC12="$(cat $R/gate3/REC_C12 2>/dev/null || echo $R9_DEFAULT_REC12)"; OV=$R/ovf_${ARM}_$REC12
# the record KIND follows the arm's argv (E1_OWN_RECORDS): a --r9-map-banded arm binds its /2-banded map record
MREC=$OV/run_map/g_map_overfit.json; MCLO=$OV/map_closure.json; BREC=$OV/run_box/gbo_binding.json; BCLO=$OV/box_closure.json
grep -q '"--r9-map-banded"' "$ARGV" 2>/dev/null && { MREC=$OV/run_map-banded/g_map_overfit.json; MCLO=$OV/map-banded_closure.json; }
refuse() { LOGF=$R/gate3_explore_rec_${C12}_$ARM.log; { echo "REFUSED: $*"; echo "rc=10"; } > "$LOGF"; echo "ZZR9E1G3-${ARM}-rec-explore-REFUSEDZZ"; exit 10; }
[ -n "$REC12" ] || refuse "no $R/gate3/REC_C12 stamp and study $R9_STUDY has no default record commit (r9gate3_restart.sh writes it from its third argument)"
# >>> E1-REUSE record pick (byte-identical in r9gate3_thor_gate.sh and r9gate3_rec_explore.sh; tests/test_launch_gate_e1rig_reuse.py runs this block)
# An arm bound to a BASE arm's PASSING record (r9e1_bind.sh --reuse-from) has $OV/run_<kind>/reuse_binding.json INSTEAD of a run record; the gate's G-MAP-OVERFIT /
# G-BOX-OVERFIT judges read it through the same --map-overfit-record / --box-overfit-record option. Exactly ONE must exist per kind: both = ambiguous (no silent preference).
for _k in map box; do
  if [ "$_k" = map ]; then _run="$MREC"; else _run="$BREC"; fi; _ru="$OV/run_$_k/reuse_binding.json"
  if [ -e "$_ru" ] && [ -e "$_run" ]; then refuse "both a run record ($_run) and a reuse binding ($_ru) exist for the $_k kind: ambiguous, remove the stale one"; fi
  if [ -e "$_ru" ]; then if [ "$_k" = map ]; then MREC="$_ru"; else BREC="$_ru"; fi
  elif [ ! -e "$_run" ]; then refuse "neither a run record ($_run) nor a reuse binding ($_ru) exists for the $_k kind"; fi
done
# <<< E1-REUSE record pick
PY=/home/nvidia/venvs/tanitad-train/bin/python
cd /tmp && PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $PY $T/stack/scripts/launch_gate.py run \
  --profile "$R9_PROFILE" --checks G-MAP-OVERFIT,G-BOX-OVERFIT --tree $T --commit $SHA \
  --argv-file $T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json --out-dir $R/gate3_explore/rec_${C12}_$ARM \
  --map-overfit-record $MREC --map-overfit-closure $MCLO \
  --box-overfit-record $BREC --box-overfit-closure $BCLO \
  > $R/gate3_explore_rec_${C12}_$ARM.log 2>&1
echo "rc=$?" >> $R/gate3_explore_rec_${C12}_$ARM.log
