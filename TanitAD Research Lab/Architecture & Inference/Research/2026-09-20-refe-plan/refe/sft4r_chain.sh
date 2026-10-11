#!/bin/bash
# SFT-4R chain (eval/PREREG_SFT4R.md). Starts after sft4r_relabel.sh logged ZZRELABEL_SFT4R_DONE.
#   nohup bash sft4r_chain.sh > /workspace/data/refe_sft4r/chain.out 2>&1 &
# Order, every step fail-closed:
#  1. the PDM labels are COMPLETE (pdm_label_complete.py: train recomputed from the sets, held-out --expect 3137) --
#     SFT-4 launched on 146,185 of 174,912 because "complete" was never computed (PREREG_SFT4 Amendment 4);
#  2. lane labels: train >= 2,109 rows (SFT-4's count at launch), held-out lane + teacher-lane present;
#  3. G-EGR: hf_egress.py gate (a real push that must verify, and a regression push the server must refuse);
#  4. the PDM labels leave the pod BEFORE training (they cost ~11 h CPU; STOP_CONSEQUENCES: they were lost once);
#  5. preflight PASS, then BOTH deliberate regressions must turn the gate RED: empty PDM -> G9, MUTATE_G3 -> G3;
#  6. the run, in its own process group, with the egress watcher beside it (pushes every checkpoint, halts the run
#     after two consecutive failed pushes plus the next checkpoint -- PREREG_SFT4R "Changed 3").
# Auth: the pod's own HF login, provisioned by the PI. No token is read, passed or written here.
set -u
D=/workspace/data/refe_sft2
OUT=/workspace/data/refe_sft4r
RUN=/workspace/refe-sft4r/run
L=$OUT/chain.log
REPO=Sayood/tanitad-refe-sft4r
mkdir -p $OUT
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: $*"; exit 1; }
source /workspace/teacher_env.sh
HPY=${HF_PY:-$DRIVERL_EVAL_PYTHON}
cd $RUN
log "chain armed"

# 1. PDM completeness
grep -q "^ZZPDM_TRAIN_EXIT 0" $D/pdm_train.log 2>/dev/null || refuse "PDM train relabel did not exit 0"
grep -q "^ZZPDM_HELDOUT_EXIT 0" $D/pdm_heldout.log 2>/dev/null || refuse "PDM held-out relabel did not exit 0"
PY_C=$DRIVERL_EVAL_PYTHON
$PY_C pdm_label_complete.py --log $D/pdm_train.log --out $D/train_pdm --sets /workspace/data/refe_navtrain/onpolicy/sets \
  --json $OUT/pdm_complete_train.json >> $L 2>&1 || refuse "train PDM labels INCOMPLETE ($(tail -1 $L))"
$PY_C pdm_label_complete.py --log $D/pdm_heldout.log --out $D/heldout_pdm --sets /workspace/data/refe_heldout/sets \
  --expect 3137 --json $OUT/pdm_complete_heldout.json >> $L 2>&1 || refuse "held-out PDM labels INCOMPLETE ($(tail -1 $L))"
log "PDM COMPLETE: train $(cat $D/train_pdm/*.jsonl | wc -l) rows, held-out $(cat $D/heldout_pdm/*.jsonl | wc -l) rows"

# 2. lane labels
NL=$(cat $D/train_teacher_lane/*.jsonl 2>/dev/null | wc -l)
[ "$NL" -ge 2109 ] || refuse "train teacher-lane rows $NL < 2,109 (SFT-4's count at launch)"
for d in heldout_lane heldout_teacher_lane; do
  [ "$(cat $D/$d/*.jsonl 2>/dev/null | wc -l)" -gt 0 ] || refuse "$d empty"
done
log "lane labels: train $NL, held-out lane $(cat $D/heldout_lane/*.jsonl | wc -l), teacher $(cat $D/heldout_teacher_lane/*.jsonl | wc -l)"

# 3. G-EGR
$HPY -c "import huggingface_hub, torch" 2>>$L || refuse "huggingface_hub/torch not importable by $HPY"
$HPY hf_egress.py gate --repo $REPO --out $OUT/gate --chain-log $L --egress-log $OUT/egress.jsonl > $OUT/gate.log 2>&1
grep -q " ZZEGR_PASS " $OUT/gate.log || refuse "G-EGR did not pass ($(tail -1 $OUT/gate.log))"
log "G-EGR PASS"

# 4. the PDM labels leave the pod first
mkdir -p $OUT/pdm_egress
tar -C $D -cf $OUT/pdm_egress/train_pdm.tar train_pdm && tar -C $D -cf $OUT/pdm_egress/heldout_pdm.tar heldout_pdm \
  || refuse "tar of the PDM labels failed"
( cd $OUT/pdm_egress && sha256sum train_pdm.tar heldout_pdm.tar > MANIFEST_pdm_sha256.txt
  cp $OUT/pdm_complete_train.json $OUT/pdm_complete_heldout.json . )
log "PDM egress staged: $(du -sb $OUT/pdm_egress | cut -f1) bytes"
$HPY hf_egress.py push --repo Sayood/tanitad-refe-navtrain-banks --repo-type dataset --prefix pdm_labels_sft4r \
  --tag pdm_labels --chain-log $L --egress-log $OUT/egress.jsonl \
  --files $OUT/pdm_egress/train_pdm.tar $OUT/pdm_egress/heldout_pdm.tar $OUT/pdm_egress/MANIFEST_pdm_sha256.txt \
          $OUT/pdm_egress/pdm_complete_train.json $OUT/pdm_egress/pdm_complete_heldout.json >> $OUT/gate.log 2>&1
grep -q "ZZEGRESS_OK pdm_labels" $L || refuse "the PDM label push did not verify"

# 5. preflight + both deliberate regressions
MODE=preflight bash $RUN/sft4r_pod.sh
grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "preflight did not pass"
log "preflight PASS"
MODE=mutate bash $RUN/sft4r_pod.sh
grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate.log && grep -q "\"G9_pdm_targets_in_bank\": {\"train_pdm\": 0" $OUT/mutate.log \
  || refuse "the empty-PDM mutation did NOT fail G9"
log "G9 mutation RED as required"
MODE=mutate_g3 bash $RUN/sft4r_pod.sh
grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_g3.log && grep -q "\"G3_copies_equal_model_at_step0\": {\"ok\": false" $OUT/mutate_g3.log \
  || refuse "the MUTATE_G3 regression did NOT fail G3"
log "G3 mutation RED as required"

# 6. run + watcher
log "run launch: PDM train rows $(cat $D/train_pdm/*.jsonl | wc -l), lane rows $NL"
set -m
MODE=run bash $RUN/sft4r_pod.sh &
TPID=$!
set +m
sleep 2
PG=$(ps -o pgid= $TPID | tr -d ' ')
if [ "$PG" != "$TPID" ]; then kill $TPID; refuse "run is not its own process group (pgid $PG, pid $TPID)"; fi
$HPY hf_egress.py watch --repo $REPO --out $OUT --stage $OUT/egress_stage --halt-pid $TPID --chain-log $L \
  --egress-log $OUT/egress.jsonl --log $OUT/sft.log $L $OUT/preflight.log $OUT/mutate.log $OUT/mutate_g3.log \
  $OUT/gate/g_egr_result.json $OUT/pdm_complete_train.json $OUT/pdm_complete_heldout.json > $OUT/watch.log 2>&1 &
WPID=$!
log "run pid/pgid $TPID, watcher pid $WPID"
wait $TPID; RRC=$?
wait $WPID; WRC=$?
log "RUN EXIT $RRC ($(tail -1 $OUT/sft.log)); WATCH EXIT $WRC ($(tail -1 $OUT/watch.log))"
[ $RRC -eq 0 ] && [ $WRC -eq 0 ] && log "ZZSFT4R_CHAIN_DONE" || log "ZZSFT4R_CHAIN_FAIL"
