#!/bin/bash
# S1ZSM on P6 chain (eval/PREREG_S1ZSM_P6.md, frozen 2026-11-10): ONE arm, S1ZSM = S1Z's recipe (C0 + L_twin on NC,DAC,TTC, radius 1.0 m, top-K 8) + ONE lever, --twin-speed-neutral (speed band 0.15 + F/S balancing, seed
# 20261107), from P6's model_final.pt, 400 updates, on the L40 pod.  NOT PLACED ON THE POD by its author; the Master Mind ships it with the landed tree.
#   S1ZSM_HF_TOKEN_FILE=<the pod's HF token file> nohup setsid bash /workspace/refe-plan-s1zsm/refe/s1zsm_chain.sh > /workspace/data/refe_s1zsm/chain.out 2>&1 < /dev/null &
# Pattern: refe/s1z_chain.sh (same fail-closed stages, markers keyed by the SHA-256 of SHIP_MANIFEST.json, REFUSED lines in chain.log), with the S1ZSM differences (the S1Z ones kept):
#  * OWN POD DIRECTORY: package $WS/refe-plan-s1zsm, data $DATA/refe_s1zsm.  gate_token.TREE_ROOTS = (refe, code, splits): every file under refe/ is in the tree sha, so shipping this tree over a pod dir that holds a minted
#    token (refe-plan-p6, -p7, -scorer1p6) would VOID that token.  Never copy this refe/ into any of them.
#  * NO HUMAN GATES; the chain's gates are the CARD'S SHA-256 (the file $ROOT/PREREG_S1ZSM_P6.md must be the frozen bytes) and the launch-gate PASS token (a token is never hand-written).
#  * NO PAUSING OR RESUMING OF P7 is done here.  Like s1z_chain.sh the chain waits for NO chain lock held and the GPU idle on 3 consecutive polls; the Master Mind pauses P7 (card section 7) outside this script.
#    Two locks are added to the list the S1Z chain checks: refe_s1z and refe_rep (their chains could otherwise start an arm between two of ours).
#  * THE DATA PIN is launch_gate.SCORER1P6_DATA_PIN (the same P6 label set); while it is None the chain refuses (and so does the gate).
#  * ORDER: lambda_z RE-MEASURED once for the speed-neutral loss (MODE=calibrate; --pair-calibrate --twin-heads --twin-speed-neutral is never in a launch argv, and never fed to the gate; S1Z's 0.026321 is never reused: a
#    pair_weight.json that does not say speed_neutral true is REFUSED, at the precheck and after the measurement), the trainer's --preflight (G1-G18 incl. the real-bank probes and the extended speed-neutral G18, 6 outputs),
#    the FIVE deliberate regressions each of which must fail the preflight AND turn G18 RED (the speed band removed, the balancing removed, L_twin dropped from the loss, S1Z's radius filter removed, S1Z's top-K mining
#    removed), the launch gate, the run, the run check (400 updates, the pinned set count, every head's share of sets with a kept pair > 0, kept F == kept S, none outside the band), and the HF egress under the prefix
#    s1zsm/ .  The marker of each stage makes a re-run skip what is done.
# Markers: $OUT/stage/<name>.ok = date + tree sha.  S1ZSM_STOP_AFTER=<stage> ends the chain after that stage (a dry run: S1ZSM_STOP_AFTER=inputs stops after the inputs audit).  End markers in chain.log: ZZS1ZSM_CHAIN_DONE, ZZS1ZSM_CHAIN_FAIL, REFUSED.
set -u
WS=${S1ZSM_WS:-/workspace}
DATA=$WS/data
PKG=${S1ZSM_PKG:-$WS/refe-plan-s1zsm}
RUN=$PKG/refe
ROOT=$DATA/refe_s1zsm
DUMP=$DATA/p6base_labels
S=$ROOT/stage
L=$ROOT/chain.log
REPO=${S1ZSM_HF_REPO:-Sayood/tanitad-refe-scorer1p6}          # the repo S1Z banks to (an existing repo, G-EGR passes on it); the arm is banked under the prefix s1zsm/
HFTOK=${S1ZSM_HF_TOKEN_FILE:-}
CKPT=$DATA/refe_runs/p6_paper_faithful/model_final.pt
CKPT_SHA=968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1
CARD=$ROOT/PREREG_S1ZSM_P6.md; CARD_SHA=ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8
NTL=$ROOT/navtest_ids/navtest_logs.json;  NTL_SHA=79daa3c2ba3fbf1c6d45d0e3b630f43bb26a073160a34ca5f9c98553c0bf43e0
NTT=$ROOT/navtest_ids/navtest_tokens.txt.gz; NTT_SHA=e61613ce129bb67dc654a5a575de2889a22fcfcbc64e4ec389a38295b8df9c7a
export TANITAD_GATE_KEY_FILE=${TANITAD_GATE_KEY_FILE:-$ROOT/launch_gate.key}      # the HMAC key lives on the volume, OUTSIDE the shipped tree and anything the egress pushes
mkdir -p $ROOT $S
STAGE=start
TREE=""
export S1ZSM_WS=$WS
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: stage=$STAGE: $*"; exit 1; }
stop_after() { if [ "${S1ZSM_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (S1ZSM_STOP_AFTER)"; exit 0; fi; }
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
  local nv="${S1ZSM_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }        # READ-WRITE fd: an exclusive flock on a read-only fd is EBADF on nfs4
others_hold() {
  local c
  for c in refe_p2 refe_p2rep rls1 s1aux refe_p6 refe_p7 refe_scorer1p6 refe_s1z refe_rep; do
    if lock_held "$DATA/$c"; then echo "the $c chain holds $DATA/$c/chain.lock"; return 2; fi
  done
  return 0
}
wait_clear() {        # BLOCK until no chain holds the pod (P7 is PAUSED for this run: its lock must be free) and the GPU was idle on S1ZSM_IDLE_POLLS consecutive polls
  local poll="${S1ZSM_POLL_S:-30}" need="${S1ZSM_IDLE_POLLS:-3}" max="${S1ZSM_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why=""; rc=0
    why="$(others_hold)"; rc=$?
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZS1ZSM_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (S1ZSM_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
arm_argv() { ARGS="$(ARM=$1 MODE=argv S1ZSM_WS=$WS bash $RUN/s1zsm_pod.sh)" && [ -n "$ARGS" ] || refuse "s1zsm_pod.sh ARM=$1 MODE=argv printed nothing (or refused a pair_weight.json that is not speed-neutral)"; }
token_ok() {          # $1 = arm, uses $ARGS.  ZZ: `launch_gate.py verify` RE-MEASURES the argv, the code tree, the checkpoint sha and the data and checks the HMAC.
  local tok
  tok="$(ls "$ROOT/lg_$1"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY_C" "$RUN/launch_gate.py" verify --profile refe_s1zsm_p6 --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_$1.json" >> "$ROOT/token_verify_$1.log" 2>&1 200>&-
}

# ---- 0. precheck: the lock, the human gates, the package
exec 200>"$ROOT/chain.lock"
flock -n 200 || { echo "REFUSED: another S1ZSM chain holds $ROOT/chain.lock"; exit 1; }
STAGE=precheck
[ -n "$HFTOK" ] || refuse "S1ZSM_HF_TOKEN_FILE is not set (hf_egress.py reads the token file in place; no default path in source)"
TREE="$(tree_sha)"
log "chain armed (ZZS1ZSM_START; s1zsm_chain.sh md5 $(md5sum "$0" | cut -c1-12), tree ${TREE:0:12})"
[ -n "$TREE" ] || refuse "no $PKG/SHIP_MANIFEST.json: ship the landed tree with its manifest (launch_gate.py ship-manifest --commit <40-hex>)"
[ -r "$HFTOK" ] && grep -qE 'hf_[A-Za-z0-9]{8,}' "$HFTOK" || refuse "$HFTOK is missing, unreadable or holds no hf_ token"
# the card, by bytes (replaces the u400 chain's human gates STAGE0_READING / P6_SPEED_READING: card section 1).  A card with other bytes is not the frozen experiment.
[ -f "$CARD" ] || refuse "$CARD is missing: copy the frozen PREREG_S1ZSM_P6.md there"
[ "$(sha256sum "$CARD" | cut -c1-64)" = "$CARD_SHA" ] || refuse "$CARD is not the frozen PREREG_S1ZSM_P6.md (sha256 ${CARD_SHA:0:12}...)"
[ -f "$WS/teacher_env.sh" ] || refuse "$WS/teacher_env.sh is missing"
source $WS/teacher_env.sh
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
PY_C="${DRIVERL_EVAL_PYTHON:-}"
[ -n "$PY_C" ] && [ -x "$PY_C" ] || refuse "teacher_env.sh did not set a runnable DRIVERL_EVAL_PYTHON"
HPY=${HF_PY:-$PY_C}
for f in refe/scorer_finetune.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/pdm_label_complete.py refe/s1z_pod.sh refe/s1z_chain.sh refe/s1z_layer.py refe/test_s1z_loss.py refe/test_s1z_gate.py refe/test_s1z_preflight.py refe/test_s1z_layer.py \
         refe/s1zsm_pod.sh refe/s1zsm_chain.sh refe/s1zsm_layer.py refe/test_s1zsm_loss.py refe/test_s1zsm_gate.py refe/test_s1zsm_preflight.py refe/test_s1zsm_layer.py \
         refe/test_pair_loss.py refe/test_pair_loss_p6.py refe/test_scorer1_gate.py refe/test_scorer1p6_gate.py refe/test_scorer1p6_preflight.py refe/test_scorer1_mutation.py refe/test_scorer1_layer.py refe/scorer1_layer.py refe/test_suite_nesting.py refe/test_launch_gate.py refe/train.py refe/model.py refe/ckpt_io.py refe/planner.py splits/navtrain.yaml SHIP_MANIFEST.json; do
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
print("ZZS1ZSM_TREE_OK %d files commit %s" % (len(tree), str(man.get("commit"))[:12]))
PYEOF
PWF=$ROOT/calibrate_s1zsm/pair_weight.json
if [ -f "$PWF" ]; then          # S1Z's 0.026321 (measured for the loss WITHOUT the speed-neutral step) must never be reused: the file must SAY it was measured speed-neutral, with the card's band, balancing and seed
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("speed_neutral") is True and d.get("sn_dv")==0.15 and d.get("sn_balance") is True and d.get("sn_seed")==20261107 and d.get("lever")=="twin", d' "$PWF" > /dev/null 2>&1 \
    || refuse "$PWF was not measured with --twin-speed-neutral (speed_neutral true, sn_dv 0.15, sn_balance true, sn_seed 20261107): lambda_z of another loss is never reused; move it aside and re-calibrate"
fi
log "ZZS1ZSM_PRECHECK_OK ($(tail -n 1 $ROOT/tree_check.log))"; stop_after precheck

# ---- 1. the inputs, AUDITED (read-only)
STAGE=inputs
if ! stage_done inputs; then
  for p in "$DUMP/train_sets" "$DUMP/heldout_sets" "$DUMP/logs/heldout_logs.txt" "$DUMP/labels_manifest.json" "$DUMP/labels.DONE" "$DUMP/train_pdm" "$DUMP/heldout_pdm" "$DATA/refe_navtrain/train_grow_cmd_3c_p7" "$DATA/refe_navtrain/train_grow/calib_table.json" "$DATA/navtrain_pixels" "$CKPT" "$NTL" "$NTT"; do
    [ -e "$p" ] || refuse "$p is missing: run Jobs A (P6 fan dump on the pod) and B (PDM labelling) and place the P6 checkpoint first (launch_gate.S1Z_DUMP is the UNVERIFIED layout)"
  done
  [ "$(sha256sum "$CKPT" | cut -c1-64)" = "$CKPT_SHA" ] || refuse "$CKPT is not P6's model_final.pt (sha256 968a469d...)"
  [ "$(sha256sum "$NTL" | cut -c1-64)" = "$NTL_SHA" ] || refuse "$NTL is not the registered navtest log list"
  [ "$(sha256sum "$NTT" | cut -c1-64)" = "$NTT_SHA" ] || refuse "$NTT is not the registered navtest token list"
  # THE DATA PIN: the exact label counts, set in launch_gate.SCORER1P6_DATA_PIN once the labels exist.  While it is None the gate refuses every arm, and so does this stage.
  PIN="$($PY_C -c 'import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; p = L.SCORER1P6_DATA_PIN; print("UNSET" if p is None else "%d %d" % (p["train_sets"], p["heldout_sets"]))' "$RUN" 2> /dev/null)"
  case "$PIN" in UNSET|"") refuse "launch_gate.SCORER1P6_DATA_PIN is unset: the exact P6 label counts are not pinned (fill it from Jobs A + B)" ;; esac
  NTR=${PIN% *}; NHO=${PIN#* }
  # the labels' completeness is recomputed from the sets by the trainer's data event and by G-DVB (train_sets == pin == pdm_train): no log is trusted here
  [ "$(cat $DUMP/train_pdm/*.jsonl | wc -l)" -ge "$NTR" ] || refuse "train PDM labels hold $(cat $DUMP/train_pdm/*.jsonl | wc -l) rows, want >= $NTR (the pin)"
  [ "$(cat $DUMP/heldout_pdm/*.jsonl | wc -l)" = "$NHO" ] || refuse "held-out PDM labels hold $(cat $DUMP/heldout_pdm/*.jsonl | wc -l) rows, want $NHO (the pin)"
  stage_mark inputs
fi
PIN="$($PY_C -c 'import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; p = L.SCORER1P6_DATA_PIN; print("UNSET" if p is None else "%d %d" % (p["train_sets"], p["heldout_sets"]))' "$RUN" 2> /dev/null)"
case "$PIN" in UNSET|"") refuse "launch_gate.SCORER1P6_DATA_PIN is unset: the exact P6 label counts are not pinned" ;; esac
NTR=${PIN% *}; NHO=${PIN#* }
log "ZZS1ZSM_INPUTS_OK"; stop_after inputs

# ---- 2. G-EGR (the run must be able to leave the pod), a real disk write test
STAGE=egr
$HPY -c "import huggingface_hub, torch" 2>> $L || refuse "huggingface_hub/torch not importable by $HPY"
if ! stage_done egr; then
  $HPY $RUN/hf_egress.py gate --repo $REPO --out $ROOT/gate --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" > $ROOT/gate.log 2>&1 200>&-
  grep -q " ZZEGR_PASS " $ROOT/gate.log || refuse "G-EGR did not pass ($(tail -1 $ROOT/gate.log | cut -c1-300)); the HF repo $REPO must exist"
  [ -f $ROOT/gate/g_egr_result.json ] || refuse "G-EGR wrote no g_egr_result.json"
  stage_mark egr
fi
log "ZZS1ZSM_EGR_OK"; stop_after egr
STAGE=dd
if ! stage_done dd; then
  DDMB=${S1Z_DD_MB:-10240}                 # 3 arms x (model_sft_A/B 2.6 GB + states) + staging
  dd if=/dev/zero of=$ROOT/dd_test.tmp bs=1M count=$DDMB conv=fsync > $ROOT/dd_test.log 2>&1 200>&- ; ddrc=$?
  rm -f $ROOT/dd_test.tmp
  [ $ddrc -eq 0 ] || refuse "the ${DDMB} MB write test failed: $(tail -n 1 $ROOT/dd_test.log)"
  stage_mark dd
fi
log "ZZS1ZSM_DD_OK"; stop_after dd

# ---- 3. the pod: P7 over, nothing else holds it, GPU idle.  Then lambda_z.
STAGE=gpu_wait
wait_clear
STAGE=calibrate
if ! stage_done calibrate; then
  ARM=S1ZSM MODE=calibrate bash $RUN/s1zsm_pod.sh 200>&-
  grep -q "^ZZPAIR_CALIBRATED" $ROOT/calibrate_s1zsm/calibrate.log || refuse "lambda_z calibration failed ($(tail -n 2 $ROOT/calibrate_s1zsm/calibrate.log | tr '\n' ' ' | cut -c1-300))"
  python3 -c 'import json,math,sys; d=json.load(open(sys.argv[1])); w=d["twin_w"]; assert w and math.isfinite(w) and w>0 and d["micro_batches"]==50 and d["lever"]=="twin" and sorted(d["heads"])==["DAC","NC","TTC"] and d["ade"]==1.0 and d["topk"]==8 and d["speed_neutral"] is True and d["sn_dv"]==0.15 and d["sn_balance"] is True and d["sn_seed"]==20261107 and all(v>0 for v in d["sets_with_pair_share"].values()), d' \
    $ROOT/calibrate_s1zsm/pair_weight.json || refuse "pair_weight.json is not a finite positive lambda_z of the SPEED-NEUTRAL loss over 50 micro-batches at radius 1.0 / top-K 8 / band 0.15 / seed 20261107 with every head's share of sets with a kept pair > 0"
  log "lambda_z (speed-neutral) = $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["twin_w"])' $ROOT/calibrate_s1zsm/pair_weight.json)"
  stage_mark calibrate
fi
log "ZZS1ZSM_CALIBRATED"; stop_after calibrate

# ---- 4. per arm: preflight (+ the three regressions) -> LAUNCH GATE -> train -> check -> bank
for ARMN in S1ZSM; do
  LA=$(echo "$ARMN" | tr 'A-Z' 'a-z'); OUT=$ROOT/$LA; STAGE="preflight_$ARMN"
  arm_argv "$ARMN"
  if ! stage_done "preflight_$LA"; then
    wait_clear
    rm -f $OUT/preflight.jsonl $OUT/PREFLIGHT_PASS.json
    ARM=$ARMN MODE=preflight bash $RUN/s1zsm_pod.sh 200>&-
    grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "$ARMN preflight did not pass ($(tail -n 2 $OUT/preflight.log | tr '\n' ' ' | cut -c1-300))"
    # the FIVE deliberate regressions of the card (section 2): each must fail the preflight AND turn G18 RED
    for MUT in noband nosnbalance droploss noradius notopk; do
      ARM=S1ZSM MODE=mutate_$MUT bash $RUN/s1zsm_pod.sh 200>&-
      grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_$MUT.log || refuse "the $MUT mutation did not fail the preflight"
      m=$(gate_is $OUT/mutate_$MUT.log G18 ok false 2>&1) || refuse "the $MUT mutation did NOT turn G18 RED ($m)"
    done
    log "S1ZSM mutations RED as required (speed band removed, F/S balancing removed, L_twin dropped from the loss, radius filter removed, top-K mining removed)"
    stage_mark "preflight_$LA"
  fi
  log "ZZS1ZSM_PREFLIGHT_OK $ARMN"; stop_after "preflight_$LA"
  STAGE="gate_$ARMN"
  if ! stage_done "gate_$LA"; then
    rm -rf "$ROOT/lg_$ARMN"; mkdir -p "$ROOT/lg_$ARMN"
    env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY_C" $RUN/launch_gate.py run --profile refe_s1zsm_p6 --pkg "$PKG" --args "$ARGS" --out-dir "$ROOT/lg_$ARMN" \
      --gate-preflight-json $OUT/preflight.jsonl --gate-egr-json $ROOT/gate/g_egr_result.json > $ROOT/launch_gate_$ARMN.log 2>&1 200>&-
    echo "ZZEXIT $?" >> $ROOT/launch_gate_$ARMN.log
    token_ok "$ARMN" || refuse "the launch gate did not mint a PASS token that verifies for $ARMN: $(grep -E 'REASON|FAIL' $ROOT/launch_gate_$ARMN.log | head -n 3 | cut -c1-300 | tr '\n' ' ')"
    stage_mark "gate_$LA"
  fi
  log "ZZS1ZSM_GATE_OK $ARMN"; stop_after "gate_$LA"
  STAGE="train_$ARMN"
  if stage_done "trainstart_$LA" && ! stage_done "train_$LA"; then
    k=1; while [ -e "$OUT.killed_$k" ]; do k=$((k + 1)); done
    refuse "a previous attempt of $ARMN STARTED training and did not finish; scorer_finetune.py has no resume. Archive it (mv $OUT $OUT.killed_$k), re-run: the chain restarts the arm at update 0 under a fresh preflight and token"
  fi
  if ! stage_done "train_$LA"; then
    wait_clear
    token_ok "$ARMN" || refuse "the launch-gate token does not verify for $ARMN: scorer_finetune.py is NOT started ($(tail -n 2 $ROOT/token_verify_$ARMN.log | tr '\n' ' ' | cut -c1-300))"
    stage_mark "trainstart_$LA"
    log "run launch $ARMN (ZZS1ZSM_TRAIN_START)"
    set -m
    ARM=$ARMN MODE=run bash $RUN/s1zsm_pod.sh 200>&- &
    TPID=$!
    set +m
    wait $TPID; RRC=$?
    log "RUN EXIT $ARMN $RRC ($(tail -1 $OUT/sft.log))"
    env -u HF_TOKEN "$PY_C" - "$OUT" "$NTR" > "$OUT/run_check.log" 2>&1 200>&- <<'PYEOF'
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
if len(start) != 1 or start[0].get("updates") != 400 or start[0].get("train_sets") != int(sys.argv[2]):
    bad.append("start event %s, want updates 400 on the pinned %s sets" % ([(s.get("updates"), s.get("train_sets")) for s in start], sys.argv[2]))
for m in ("model_sft_A.pt", "model_sft_B.pt"):
    if not os.path.isfile(os.path.join(out, m)):
        bad.append("missing " + m)
ts = (done[0].get("twin_share") if done else None) or {}
if sorted(ts) != ["DAC", "NC", "TTC"] or not all(v.get("share_with_pair", 0) > 0 and v.get("sets", 0) > 0 for v in ts.values()):
    bad.append("the done event's twin_share is %s: every head must have a share of sets with a kept pair > 0" % (ts,))
elif not all(v.get("kept_F", -1) == v.get("kept_S", -2) and v.get("kept_F", 0) > 0 and v.get("out_of_band", -1) == 0 and v.get("unbalanced_sets", -1) == 0 and v.get("pairs") == 2 * v.get("kept_F", 0) for v in ts.values()):
    bad.append("the done event's twin_share is %s: every head must keep F == S pairs, none outside the speed band" % (ts,))
if not any(r.get("event") == "data" and (r.get("twin") or {}).get("speed_neutral") is True for r in rows):
    bad.append("the data event does not say speed_neutral true")
if bad:
    sys.exit("; ".join(bad))
print("ZZS1ZSM_RUN_OK updates 400 twin_share " + json.dumps({h: round(v["share_with_pair"], 4) for h, v in ts.items()}))
PYEOF
    RCC=$?
    log "run check $ARMN: $(tail -n 1 $OUT/run_check.log | cut -c1-300)"
    [ $RRC -eq 0 ] && [ $RCC -eq 0 ] || { log "ZZS1ZSM_CHAIN_FAIL $ARMN"; exit 1; }
    # bank the arm off the pod BEFORE the next arm (a verified remote listing, not the push call)
    N0=$(wc -l < $L)
    $HPY $RUN/hf_egress.py push --repo $REPO --prefix "s1zsm" --max-push-bytes 6e9 --tag "arm_$LA" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" \
      --files $OUT/model_sft_B.pt $OUT/sft.log $OUT/sft.jsonl $OUT/preflight.log $ROOT/launch_gate_$ARMN.log $OUT/run_check.log >> $ROOT/gate.log 2>&1 200>&-
    tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_$LA " || refuse "the $ARMN egress push did not verify: the arm lives on the pod only"
    stage_mark "train_$LA"
  fi
  log "ZZS1ZSM_ARM_DONE $ARMN"
done
log "ZZS1ZSM_CHAIN_DONE"
