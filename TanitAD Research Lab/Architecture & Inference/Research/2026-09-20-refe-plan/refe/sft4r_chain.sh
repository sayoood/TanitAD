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
PY_C=$DRIVERL_EVAL_PYTHON
# The LAST exit marker decides: the relabel logs and chain.log are APPENDED across attempts, so an earlier attempt's
# "0" must never vouch for a later attempt that failed.  last_exit <marker> <log>
last_exit() { grep "^$1 " "$2" 2>/dev/null | tail -1 | cut -d' ' -f2-; }
# Gate JSON is PARSED, never grepped. json.dumps keeps insertion order and scorer_finetune.py writes G3's "ok" LAST,
# so the earlier grep for '{"ok": false' could never match (MEASURED on SFT-4's
# raw/2026-10-04-sft4-smoke/r2_smoke2_g3mut.log) and the chain would have refused every launch at step 5.
#   gate_is <log> <gate> <key> <json literal>   -> rc 0 iff the log's single preflight row has gates.<gate>.<key> == it
gate_is() {
  $PY_C - "$@" <<'PYEOF'
import json, sys
log, gate, key, want = sys.argv[1:5]
rows = [json.loads(ln) for ln in open(log, encoding="utf-8", errors="replace") if ln.startswith('{"event": "preflight"')]
if len(rows) != 1:
    sys.exit("%s: %d preflight rows, want 1" % (log, len(rows)))
got = json.dumps(rows[0]["gates"].get(gate, {}).get(key, "<absent>"))
if got != want:
    sys.exit("%s: %s.%s = %s, want %s" % (log, gate, key, got, want))
PYEOF
}

# 1. PDM completeness
[ "$(last_exit ZZPDM_TRAIN_EXIT $D/pdm_train.log)" = "0" ] \
  || refuse "PDM train relabel's last exit is '$(last_exit ZZPDM_TRAIN_EXIT $D/pdm_train.log)', not 0"
[ "$(last_exit ZZPDM_HELDOUT_EXIT $D/pdm_heldout.log)" = "0" ] \
  || refuse "PDM held-out relabel's last exit is '$(last_exit ZZPDM_HELDOUT_EXIT $D/pdm_heldout.log)', not 0"
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
# PREREG_SFT4R "Changed 2" trains on the rows written by launch with a 2,109 floor, so a non-zero lane rc is RECORDED,
# not refused -- refusing on it would move the registered rule. A missing marker means the relabel never finished.
log "lane relabel last exit (rank0 rank1): '$(last_exit ZZLANE_TRAIN_EXIT $D/tl_train.log)'"

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
N0=$(wc -l < $L)      # only THIS push's lines may vouch for it (chain.log is appended across attempts)
$HPY hf_egress.py push --repo Sayood/tanitad-refe-navtrain-banks --repo-type dataset --prefix pdm_labels_sft4r \
  --tag pdm_labels --chain-log $L --egress-log $OUT/egress.jsonl \
  --files $OUT/pdm_egress/train_pdm.tar $OUT/pdm_egress/heldout_pdm.tar $OUT/pdm_egress/MANIFEST_pdm_sha256.txt \
          $OUT/pdm_egress/pdm_complete_train.json $OUT/pdm_egress/pdm_complete_heldout.json >> $OUT/gate.log 2>&1
tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK pdm_labels " || refuse "the PDM label push did not verify"

# 5. preflight + both deliberate regressions
MODE=preflight bash $RUN/sft4r_pod.sh
grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "preflight did not pass"
log "preflight PASS"
MODE=mutate bash $RUN/sft4r_pod.sh
grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate.log || refuse "the empty-PDM mutation did not fail the preflight"
m=$(gate_is $OUT/mutate.log G9_pdm_targets_in_bank train_pdm 0 2>&1) || refuse "the empty-PDM mutation did not empty the train PDM bank ($m)"
m=$(gate_is $OUT/mutate.log G9_pdm_targets_in_bank ok false 2>&1) || refuse "the empty-PDM mutation did NOT fail G9 ($m)"
log "G9 mutation RED as required"
MODE=mutate_g3 bash $RUN/sft4r_pod.sh
grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_g3.log || refuse "the MUTATE_G3 regression did not fail the preflight"
m=$(gate_is $OUT/mutate_g3.log G3_copies_equal_model_at_step0 ok false 2>&1) || refuse "the MUTATE_G3 regression did NOT fail G3 ($m)"
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
