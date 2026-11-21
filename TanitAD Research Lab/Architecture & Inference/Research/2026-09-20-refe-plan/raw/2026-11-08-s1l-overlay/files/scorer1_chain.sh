#!/bin/bash
# SCORER1 chain (eval/PREREG_SCORER1.md + AMENDMENT_1): C0, C0r and S1 from u400, 400 updates each, on the A40 pod AFTER P7.  NOT PLACED ON THE POD by its author; the Master Mind ships it with the landed tree.
#   SCORER1_HF_TOKEN_FILE=<the pod's HF token file> nohup setsid bash /workspace/refe-plan-scorer1/refe/scorer1_chain.sh > /workspace/data/refe_scorer1/chain.out 2>&1 < /dev/null &
# Pattern: refe/sft4r_chain.sh (same fail-closed stages, markers keyed by the SHA-256 of SHIP_MANIFEST.json, REFUSED lines in chain.log), with the SCORER1 differences:
#  * HUMAN GATES, checked first (card section 8): $ROOT/STAGE0_READING must hold Z1 or INCONCLUSIVE (on Z2 the arms do NOT launch); $ROOT/P6_SPEED_READING must hold REFUTED or NOT_PROVEN
#    (on HOLDS u400 is the wrong base and the Master Mind goes back to the PI).  Both files are written BY HAND by the Master Mind after reading the evidence; the chain only refuses without them.
#  * WAIT FOR P7: the P7 chain's end marker ZZP7_DONE in $DATA/refe_p7/chain.log AND its chain.lock free (and P6's, and every other chain's), then the GPU idle on 3 consecutive polls.  No GPU load is added to a pod that trains.
#  * inputs AUDITED, never rebuilt: u400 by sha256 (8b4dba48...), the navtest lists by sha256, the SFT-4R PDM / lane labels and the held-out sets (the pod currently lacks refe_sft2 and refe_heldout:
#    restore them first, see HANDOFF "what must be pulled"), then calibrate (lambda_p) -> per arm: preflight -> (S1) both deliberate regressions must turn the preflight RED -> LAUNCH GATE (profile refe_scorer1)
#    -> train -> check -> bank off the pod.  Order C0, C0r, S1; the arm's marker makes a re-run skip what is done.  A token is never hand-written: launch_gate.py mints it, `verify` re-measures it.
# Markers: $OUT/stage/<name>.ok = date + tree sha.  SCORER1_STOP_AFTER=<stage> ends the chain after that stage (a dry run).
set -u
WS=${SCORER1_WS:-/workspace}
DATA=$WS/data
PKG=${SCORER1_PKG:-$WS/refe-plan-scorer1}
RUN=$PKG/refe
D=$DATA/refe_sft2
ROOT=$DATA/refe_scorer1
S=$ROOT/stage
L=$ROOT/chain.log
P7LOG=$DATA/refe_p7/chain.log
REPO=Sayood/tanitad-refe-scorer1
HFTOK=${SCORER1_HF_TOKEN_FILE:-}
CKPT=$ROOT/base/model_sft4_B_u400.pt
CKPT_SHA=8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d
NTL=$ROOT/navtest_ids/navtest_logs.json;  NTL_SHA=79daa3c2ba3fbf1c6d45d0e3b630f43bb26a073160a34ca5f9c98553c0bf43e0
NTT=$ROOT/navtest_ids/navtest_tokens.txt.gz; NTT_SHA=e61613ce129bb67dc654a5a575de2889a22fcfcbc64e4ec389a38295b8df9c7a
export TANITAD_GATE_KEY_FILE=${TANITAD_GATE_KEY_FILE:-$ROOT/launch_gate.key}      # the HMAC key lives on the volume, OUTSIDE the shipped tree and anything the egress pushes
mkdir -p $ROOT $S
STAGE=start
TREE=""
export SCORER1_WS=$WS
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: stage=$STAGE: $*"; exit 1; }
stop_after() { if [ "${SCORER1_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (SCORER1_STOP_AFTER)"; exit 0; fi; }
tree_sha() { sha256sum "$PKG/SHIP_MANIFEST.json" 2>/dev/null | cut -c1-64; }
stage_done() { [ -f "$S/$1.ok" ] && [ "$(sed -n 2p "$S/$1.ok")" = "$TREE" ]; }
stage_mark() { { date -u +%FT%T; echo "$TREE"; } > "$S/$1.ok"; }
# gate JSON is PARSED, never grepped:  gate_is <log> <gate-key-prefix> <key> <json literal>  -> rc 0 iff the log's single preflight row has that gate's field equal to it
gate_is() {
  $PY_C - "$@" <<'PYEOF'
import json, sys
log, gate, key, want = sys.argv[1:5]
rows = [json.loads(ln) for ln in open(log, encoding="utf-8", errors="replace") if ln.startswith('{"event": "preflight"')]
if len(rows) != 1:
    sys.exit("%s: %d preflight rows, want 1" % (log, len(rows)))
g = next((v for k, v in rows[0]["gates"].items() if k.split("_", 1)[0] == gate), None)
got = json.dumps("<absent>" if g is None else g.get(key, "<absent>"))
if got != want:
    sys.exit("%s: %s.%s = %s, want %s" % (log, gate, key, got, want))
PYEOF
}
gpu_busy() {          # rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never pgrep -f.
  local nv="${SCORER1_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }        # READ-WRITE fd: an exclusive flock on a read-only fd is EBADF on nfs4
others_hold() {
  local c
  for c in refe_p2 refe_p2rep rls1 s1aux refe_p6 refe_p7; do
    if lock_held "$DATA/$c"; then echo "the $c chain holds $DATA/$c/chain.lock"; return 2; fi
  done
  return 0
}
p7_over() {           # P7's END MARKER: ZZP7_DONE as the LAST ZZP7_ line of its chain.log (a later REFUSED/FAILED line voids an older DONE), and nobody holds its lock
  [ -f "$P7LOG" ] && [ "$(grep -o 'ZZP7_[A-Z_]*' "$P7LOG" | grep -v 'ZZP7_WAITING\|ZZP7_WAIT_OK\|ZZP7_START' | tail -n 1)" = "ZZP7_DONE" ] && ! lock_held "$DATA/refe_p7"
}
wait_clear() {        # BLOCK until P7 is over, no chain holds the pod and the GPU was idle on SCORER1_IDLE_POLLS consecutive polls
  local poll="${SCORER1_POLL_S:-30}" need="${SCORER1_IDLE_POLLS:-3}" max="${SCORER1_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why=""; rc=0
    if ! p7_over; then why="P7 has not written ZZP7_DONE (or still holds its lock)"; rc=2; fi
    if [ "$rc" = 0 ]; then why="$(others_hold)"; rc=$?; fi
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZSCORER1_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (SCORER1_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
arm_argv() { ARGS="$(ARM=$1 MODE=argv SCORER1_WS=$WS bash $RUN/scorer1_pod.sh)" && [ -n "$ARGS" ] || refuse "scorer1_pod.sh ARM=$1 MODE=argv printed nothing"; }
# >>> lane manifest (PREREG_SCORER1 AMENDMENT 4): every arm trains on ONE lane draw.  File names + sha256, pinned once before the first arm and re-checked at the start of every arm; the row count in the inputs stage cannot see a different draw.
LANE_DIR=$D/train_teacher_lane
LANE_MAN=$ROOT/train_teacher_lane.manifest.sha256
lane_list() { (cd "$LANE_DIR" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
lane_pin() {
  [ -s "$LANE_MAN" ] && return 0
  lane_list > "$LANE_MAN.tmp"
  [ -s "$LANE_MAN.tmp" ] || refuse "$LANE_DIR holds no files: no lane draw to pin"
  mv "$LANE_MAN.tmp" "$LANE_MAN"
}
lane_check() {          # uses $ARGS (arm_argv)
  [ -s "$LANE_MAN" ] || refuse "no lane manifest at $LANE_MAN: the inputs stage writes it before the first arm"
  lane_list | cmp -s - "$LANE_MAN" || refuse "$LANE_DIR differs from the manifest pinned before the first arm: the arms would not share one lane draw"
  case " $ARGS " in *" --teacher-lane-train $LANE_DIR "*) ;; *) refuse "this arm's argv does not read --teacher-lane-train $LANE_DIR";; esac
}
# <<< lane manifest
prof_of() { if [ "$1" = "S1L" ]; then echo refe_scorer1l; else echo refe_scorer1; fi; }      # S1L (AMENDMENT_2/3) has its own launch-gate profile
token_ok() {          # $1 = arm, uses $ARGS.  ZZ: `launch_gate.py verify` RE-MEASURES the argv, the code tree, the checkpoint sha and the data and checks the HMAC.
  local tok
  tok="$(ls "$ROOT/lg_$1"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY_C" "$RUN/launch_gate.py" verify --profile "$(prof_of "$1")" --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_$1.json" >> "$ROOT/token_verify_$1.log" 2>&1 200>&-
}

# ---- 0. precheck: the lock, the human gates, the package
exec 200>"$ROOT/chain.lock"
flock -n 200 || { echo "REFUSED: another scorer1 chain holds $ROOT/chain.lock"; exit 1; }
STAGE=precheck
[ -n "$HFTOK" ] || refuse "SCORER1_HF_TOKEN_FILE is not set (hf_egress.py reads the token file in place; no default path in source)"
TREE="$(tree_sha)"
log "chain armed (ZZSCORER1_START; scorer1_chain.sh md5 $(md5sum "$0" | cut -c1-12), tree ${TREE:0:12})"
[ -n "$TREE" ] || refuse "no $PKG/SHIP_MANIFEST.json: ship the landed tree with its manifest (launch_gate.py ship-manifest --commit <40-hex>)"
[ -r "$HFTOK" ] && grep -qE 'hf_[A-Za-z0-9]{8,}' "$HFTOK" || refuse "$HFTOK is missing, unreadable or holds no hf_ token"
r0="$(cat $ROOT/STAGE0_READING 2> /dev/null | tr -d '[:space:]')"
case "$r0" in Z1|INCONCLUSIVE) : ;; *) refuse "$ROOT/STAGE0_READING is '$r0': the arms launch only on Z1 or INCONCLUSIVE (card section 1/8.2; on Z2 the next lever is the navtrain->navtest shift, not this run)" ;; esac
r6="$(cat $ROOT/P6_SPEED_READING 2> /dev/null | tr -d '[:space:]')"
case "$r6" in REFUTED|NOT_PROVEN) : ;; *) refuse "$ROOT/P6_SPEED_READING is '$r6': the chain is armed only after H-P6-SCORER-SPEED-1 reads REFUTED or NOT_PROVEN (card section 8.3; on HOLDS the Master Mind returns to the PI)" ;; esac
[ -f "$WS/teacher_env.sh" ] || refuse "$WS/teacher_env.sh is missing"
source $WS/teacher_env.sh
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
PY_C="${DRIVERL_EVAL_PYTHON:-}"
[ -n "$PY_C" ] && [ -x "$PY_C" ] || refuse "teacher_env.sh did not set a runnable DRIVERL_EVAL_PYTHON"
HPY=${HF_PY:-$PY_C}
for f in refe/scorer_finetune.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/pdm_label_complete.py refe/scorer1_pod.sh refe/scorer1_chain.sh \
         refe/test_pair_loss.py refe/test_scorer1_gate.py refe/test_scorer1_mutation.py refe/test_suite_nesting.py refe/test_launch_gate.py refe/train.py refe/model.py refe/ckpt_io.py refe/planner.py splits/navtrain.yaml SHIP_MANIFEST.json; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f is missing from the shipped package"
done
env -u HF_TOKEN "$PY_C" - "$PKG" "$RUN" > "$ROOT/tree_check.log" 2>&1 200>&- <<'PYEOF' || refuse "the shipped tree is not its SHIP_MANIFEST.json: $(tail -n 1 "$ROOT/tree_check.log" | cut -c1-300)"
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import gate_token as GT
pkg = Path(sys.argv[1])
man = GT.read_json(pkg / GT.SHIP_MANIFEST)
tree = GT.tree_manifest(pkg)
if man.get("files") != tree:
    sys.exit("tree differs from the manifest: %s" % GT.tree_diff(man.get("files") or {}, tree))
print("ZZSCORER1_TREE_OK %d files commit %s" % (len(tree), str(man.get("commit"))[:12]))
PYEOF
log "ZZSCORER1_PRECHECK_OK ($(tail -n 1 $ROOT/tree_check.log))"; stop_after precheck

# ---- 1. the inputs, AUDITED (read-only)
STAGE=inputs
if ! stage_done inputs; then
  for p in "$DATA/refe_navtrain/onpolicy/sets" "$DATA/refe_heldout/sets" "$DATA/refe_heldout/heldout_logs.txt" "$DATA/refe_navtrain/train_grow" "$DATA/navtrain_pixels" \
           "$D/train_pdm" "$D/heldout_pdm" "$D/train_teacher_lane" "$D/heldout_teacher_lane" "$D/heldout_lane" "$CKPT" "$NTL" "$NTT"; do
    [ -e "$p" ] || refuse "$p is missing: restore it first (the pod lacked refe_sft2/, refe_heldout/ and the u400 checkpoint on 2026-11-08; HANDOFF 'what must be pulled')"
  done
  [ "$(sha256sum "$CKPT" | cut -c1-64)" = "$CKPT_SHA" ] || refuse "$CKPT is not u400 (sha256 8b4dba48...)"
  [ "$(sha256sum "$NTL" | cut -c1-64)" = "$NTL_SHA" ] || refuse "$NTL is not the registered navtest log list"
  [ "$(sha256sum "$NTT" | cut -c1-64)" = "$NTT_SHA" ] || refuse "$NTT is not the registered navtest token list"
  # the PDM labels' completeness is recomputed from the sets by the trainer's data event and by G-DVB (train_sets == 174,912 == pdm_train): no log is trusted here
  [ "$(cat $D/train_pdm/*.jsonl | wc -l)" -ge 174912 ] || refuse "train PDM labels hold $(cat $D/train_pdm/*.jsonl | wc -l) rows, want >= 174,912"
  [ "$(cat $D/heldout_pdm/*.jsonl | wc -l)" = 3137 ] || refuse "held-out PDM labels hold $(cat $D/heldout_pdm/*.jsonl | wc -l) rows, want 3,137"
  [ "$(cat $D/train_teacher_lane/*.jsonl | wc -l)" -ge 2109 ] || refuse "train teacher-lane rows < 2,109"
  stage_mark inputs
fi
lane_pin
log "lane manifest sha256 $(sha256sum "$LANE_MAN" | cut -c1-64)"; sed 's/^/lane_manifest /' "$LANE_MAN" | head -n 200 >> $L
log "ZZSCORER1_INPUTS_OK"; stop_after inputs

# ---- 2. G-EGR (the run must be able to leave the pod), a real disk write test
STAGE=egr
$HPY -c "import huggingface_hub, torch" 2>> $L || refuse "huggingface_hub/torch not importable by $HPY"
if ! stage_done egr; then
  $HPY $RUN/hf_egress.py gate --repo $REPO --out $ROOT/gate --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" > $ROOT/gate.log 2>&1 200>&-
  grep -q " ZZEGR_PASS " $ROOT/gate.log || refuse "G-EGR did not pass ($(tail -1 $ROOT/gate.log | cut -c1-300)); the HF repo $REPO must exist"
  [ -f $ROOT/gate/g_egr_result.json ] || refuse "G-EGR wrote no g_egr_result.json"
  stage_mark egr
fi
log "ZZSCORER1_EGR_OK"; stop_after egr
STAGE=dd
if ! stage_done dd; then
  DDMB=${SCORER1_DD_MB:-10240}                 # 3 arms x (model_sft_A/B 2.6 GB + states) + staging
  dd if=/dev/zero of=$ROOT/dd_test.tmp bs=1M count=$DDMB conv=fsync > $ROOT/dd_test.log 2>&1 200>&- ; ddrc=$?
  rm -f $ROOT/dd_test.tmp
  [ $ddrc -eq 0 ] || refuse "the ${DDMB} MB write test failed: $(tail -n 1 $ROOT/dd_test.log)"
  stage_mark dd
fi
log "ZZSCORER1_DD_OK"; stop_after dd

# ---- 3. the pod: P7 over, nothing else holds it, GPU idle.  Then lambda_p.
STAGE=gpu_wait
wait_clear
STAGE=calibrate
if ! stage_done calibrate; then
  ARM=S1 MODE=calibrate bash $RUN/scorer1_pod.sh 200>&-
  grep -q "^ZZPAIR_CALIBRATED" $ROOT/calibrate/calibrate.log || refuse "lambda_p calibration failed ($(tail -n 2 $ROOT/calibrate/calibrate.log | tr '\n' ' ' | cut -c1-300))"
  python3 -c 'import json,math,sys; d=json.load(open(sys.argv[1])); w=d["pair_w"]; assert w and math.isfinite(w) and w>0 and d["micro_batches"]==50 and all(v>0 for v in d["sets_with_pair_share"].values()), d' \
    $ROOT/calibrate/pair_weight.json || refuse "pair_weight.json is not a finite positive lambda_p over 50 micro-batches with every head's share of sets with a pair > 0"
  log "lambda_p = $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pair_w"])' $ROOT/calibrate/pair_weight.json)"
  stage_mark calibrate
fi
log "ZZSCORER1_CALIBRATED"; stop_after calibrate
# ---- 3b. S1L only (SCORER1_ARMS names it): its own lambda_p, measured by the same rule on the FOUR-head loss; lives in $ROOT/calibrate_s1l (S1's calibrate/ is untouched)
case " ${SCORER1_ARMS:-C0 C0r S1} " in *" S1L "*)
  STAGE=calibrate_s1l
  if ! stage_done calibrate_s1l; then
    ARM=S1L MODE=calibrate bash $RUN/scorer1_pod.sh 200>&-
    grep -q "^ZZPAIR_CALIBRATED" $ROOT/calibrate_s1l/calibrate.log || refuse "S1L lambda_p calibration failed ($(tail -n 2 $ROOT/calibrate_s1l/calibrate.log | tr '
' ' ' | cut -c1-300))"
    python3 -c 'import json,math,sys; d=json.load(open(sys.argv[1])); w=d["pair_w"]; assert w and math.isfinite(w) and w>0 and d["micro_batches"]==50 and sorted(d["heads"])==["DAC","NC","TTC","lane"] and all(v>0 for v in d["sets_with_pair_share"].values()), d' \n      $ROOT/calibrate_s1l/pair_weight.json || refuse "S1L pair_weight.json is not a finite positive lambda_p over 50 micro-batches and the four heads with a share of sets with a pair > 0 (lane included)"
    log "S1L lambda_p = $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pair_w"])' $ROOT/calibrate_s1l/pair_weight.json)"
    stage_mark calibrate_s1l
  fi
  log "ZZSCORER1_CALIBRATED_S1L";;
esac

# ---- 4. per arm: preflight (+ S1's two regressions) -> LAUNCH GATE -> train -> check -> bank
for ARMN in ${SCORER1_ARMS:-C0 C0r S1}; do
  LA=$(echo "$ARMN" | tr 'A-Z' 'a-z'); OUT=$ROOT/$LA; STAGE="preflight_$ARMN"
  arm_argv "$ARMN"
  lane_check; log "ZZSCORER1_LANE_DRAW_OK $ARMN"
  if ! stage_done "preflight_$LA"; then
    wait_clear
    rm -f $OUT/preflight.jsonl $OUT/PREFLIGHT_PASS.json
    ARM=$ARMN MODE=preflight bash $RUN/scorer1_pod.sh 200>&-
    grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "$ARMN preflight did not pass ($(tail -n 2 $OUT/preflight.log | tr '\n' ' ' | cut -c1-300))"
    if [ "$ARMN" = "S1" ] || [ "$ARMN" = "S1L" ]; then
      ARM=$ARMN MODE=mutate_nobalance bash $RUN/scorer1_pod.sh 200>&-
      grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_nobalance.log || refuse "the NOBALANCE mutation did not fail the preflight"
      m=$(gate_is $OUT/mutate_nobalance.log G16 ok false 2>&1) || refuse "the NOBALANCE mutation did NOT turn G16 RED ($m)"
      ARM=$ARMN MODE=mutate_droploss bash $RUN/scorer1_pod.sh 200>&-
      grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_droploss.log || refuse "the DROP_LOSS mutation did not fail the preflight"
      m=$(gate_is $OUT/mutate_droploss.log G16 ok false 2>&1) || refuse "the DROP_LOSS mutation did NOT turn G16 RED ($m)"
      log "$ARMN mutations RED as required"
    fi
    stage_mark "preflight_$LA"
  fi
  log "ZZSCORER1_PREFLIGHT_OK $ARMN"; stop_after "preflight_$LA"
  STAGE="gate_$ARMN"
  if ! stage_done "gate_$LA"; then
    rm -rf "$ROOT/lg_$ARMN"; mkdir -p "$ROOT/lg_$ARMN"
    env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY_C" $RUN/launch_gate.py run --profile "$(prof_of "$ARMN")" --pkg "$PKG" --args "$ARGS" --out-dir "$ROOT/lg_$ARMN" \
      --gate-preflight-json $OUT/preflight.jsonl --gate-egr-json $ROOT/gate/g_egr_result.json > $ROOT/launch_gate_$ARMN.log 2>&1 200>&-
    echo "ZZEXIT $?" >> $ROOT/launch_gate_$ARMN.log
    token_ok "$ARMN" || refuse "the launch gate did not mint a PASS token that verifies for $ARMN: $(grep -E 'REASON|FAIL' $ROOT/launch_gate_$ARMN.log | head -n 3 | cut -c1-300 | tr '\n' ' ')"
    stage_mark "gate_$LA"
  fi
  log "ZZSCORER1_GATE_OK $ARMN"; stop_after "gate_$LA"
  STAGE="train_$ARMN"
  if stage_done "trainstart_$LA" && ! stage_done "train_$LA"; then
    k=1; while [ -e "$OUT.killed_$k" ]; do k=$((k + 1)); done
    refuse "a previous attempt of $ARMN STARTED training and did not finish; scorer_finetune.py has no resume. Archive it (mv $OUT $OUT.killed_$k), re-run: the chain restarts the arm at update 0 under a fresh preflight and token"
  fi
  if ! stage_done "train_$LA"; then
    wait_clear
    token_ok "$ARMN" || refuse "the launch-gate token does not verify for $ARMN: scorer_finetune.py is NOT started ($(tail -n 2 $ROOT/token_verify_$ARMN.log | tr '\n' ' ' | cut -c1-300))"
    stage_mark "trainstart_$LA"
    log "run launch $ARMN (ZZSCORER1_TRAIN_START)"
    set -m
    ARM=$ARMN MODE=run bash $RUN/scorer1_pod.sh 200>&- &
    TPID=$!
    set +m
    wait $TPID; RRC=$?
    log "RUN EXIT $ARMN $RRC ($(tail -1 $OUT/sft.log))"
    env -u HF_TOKEN "$PY_C" - "$OUT" > "$OUT/run_check.log" 2>&1 200>&- <<'PYEOF'
import json, os, sys
out = sys.argv[1]
rows = [json.loads(ln) for ln in open(os.path.join(out, "sft.jsonl"), encoding="utf-8") if ln.startswith("{")]
done = [r for r in rows if r.get("event") == "done"]
start = [r for r in rows if r.get("event") == "start"]
bad = []
if "ZZSFT_DONE" not in open(os.path.join(out, "sft.log"), encoding="utf-8", errors="replace").read():
    bad.append("no ZZSFT_DONE in sft.log")
if len(done) != 1 or done[0].get("updates") != 400:
    bad.append("done event %s, want updates 400" % ([d.get("updates") for d in done],))
if len(start) != 1 or start[0].get("updates") != 400 or start[0].get("train_sets") != 174912:
    bad.append("start event %s, want updates 400 on 174912 sets" % ([(s.get("updates"), s.get("train_sets")) for s in start],))
for m in ("model_sft_A.pt", "model_sft_B.pt"):
    if not os.path.isfile(os.path.join(out, m)):
        bad.append("missing " + m)
if bad:
    sys.exit("; ".join(bad))
print("ZZSCORER1_RUN_OK updates 400")
PYEOF
    RCC=$?
    log "run check $ARMN: $(tail -n 1 $OUT/run_check.log | cut -c1-300)"
    [ $RRC -eq 0 ] && [ $RCC -eq 0 ] || { log "ZZSCORER1_CHAIN_FAIL $ARMN"; exit 1; }
    # bank the arm off the pod BEFORE the next arm (a verified remote listing, not the push call)
    N0=$(wc -l < $L)
    $HPY $RUN/hf_egress.py push --repo $REPO --prefix "$LA" --max-push-bytes 6e9 --tag "arm_$LA" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" \
      --files $OUT/model_sft_B.pt $OUT/sft.log $OUT/sft.jsonl $OUT/preflight.log $ROOT/launch_gate_$ARMN.log $OUT/run_check.log >> $ROOT/gate.log 2>&1 200>&-
    tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_$LA " || refuse "the $ARMN egress push did not verify: the arm lives on the pod only"
    stage_mark "train_$LA"
  fi
  log "ZZSCORER1_ARM_DONE $ARMN"
done
log "ZZSCORER1_CHAIN_DONE"
