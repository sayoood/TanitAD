#!/bin/bash
# VDAC chain (PREREG_VDAC_P6 + A1 + A2; the H48 list = amendment A3): ONE arm per invocation, HA-s0 | HA-s1 | HA-SHUF = the drivable head on P6's frozen trunk (S1c's recipe), the 48 logs of H48 excluded from its labels, from P6's model_final.pt, on the L40 pod.
# NOT PLACED ON THE POD by its author; the Master Mind ships it with the landed tree.
#   bash /workspace/refe-plan-vdac/refe/vdac_chain.sh HA-s0|HA-s1|HA-SHUF        (VDAC_HF_TOKEN_FILE=<the pod's HF token file> in the environment; refe/vdac_pause_p7.sh runs S1a = HA-s0 then HA-SHUF, S1b = HA-s1, each inside ONE P7 pause)
# Pattern: refe/div_chain.sh (same fail-closed stages and the SAME wait functions, markers keyed by the SHA-256 of SHIP_MANIFEST.json), with the VDAC differences:
#  * OWN POD DIRECTORY: package $WS/refe-plan-vdac, data $DATA/refe_vdac.  gate_token.TREE_ROOTS = (refe, code, splits): shipping this refe/ over another chain's package dir (one that holds a minted token) would VOID that token.
#  * THE FROZEN INPUTS, BY BYTES: PREREG_VDAC_P6.md (5594c2ae...), A1 (8f121bdf...), A2 (c3d56358...) in $ROOT, and the H48 list (splits/h48_logs.txt, sha256 2ae13935...) from the shipped tree, copied to $ROOT/h48_logs.txt and re-hashed.
#  * THE EXCLUSION IS CHECKED THREE TIMES: (1) the preflight's own data event (`aux_excl`: 0 H48 rows reach the loss, filter == the file's logs, > 0 labelled frames excluded) is PARSED here; (2) the DELIBERATE-REGRESSION MUTANT (MODE=mutate: the filter removed) must be rejected by the trainer
#    (rc 4, an `aux-excl` refusal), or the chain does not train; (3) the real run's own data event is parsed again.  Then the LAUNCH GATE (profile refe_vdac_p6; a PASS token minted by launch_gate.py, never hand-written).
#  * AFTER THE RUN: the run check (updates, the h48x forward's frame count == the audit's excluded frames), the POST-TRAINING GATE (vdac_postgate.py: every head parameter has Adam state and a non-zero first moment), the EXPORT of the H48 sets' P6 proposals and NAVSIM sub-scores from the pod
#    label bank (vdac_export_h48.py, once per pod: the S1 read runs on CPU locally), and the HF egress under vdac/<arm> with the LFS sha256 of aux_drv.pt recorded.
#  * NO PAUSING OR RESUMING OF P7 here (refe/vdac_pause_p7.sh does that).  The chain waits for NO other chain lock held and the GPU idle on 3 consecutive polls.
# FAILURE LINES in chain_<arm>.log start with `REFUSED: stage=` or `ZZVDAC_CHAIN_FAIL`; a success path never prints the bare word REFUSED (a watcher matched a success line of DIV's chain that carried that word as a failure).
# Markers: $ROOT/stage_<arm>/<name>.ok = date + tree sha.  VDAC_STOP_AFTER=<stage> ends the chain after that stage (a dry run).
set -u
ARMN=${1:-}
HEADN=2440450          # the head's BUILT parameter count the post-training gate must find in Adam state: 2,440,450 at cell 2.0, 3,423,490 at cell 1.0 (80 x 64 queries; gate_checks.AUX_HEAD_PARAMS_BY_CELL).  vdac_postgate.py's own default is the 2 m count, so every 1 m arm MUST pass it (A6: HC-s0 / HC-SHUF would have failed the post-training gate on it)
case "$ARMN" in
  HA-s0)   HAUPD=10000; SHUFSEED=none ;;
  HA-s1)   HAUPD=10000; SHUFSEED=none ;;
  HA-SHUF) HAUPD=2000;  SHUFSEED=0 ;;
  HB-s0)   HAUPD=10000; SHUFSEED=none ;;          # VDAC A5 (DRAFT): the LoRA trunk copy; aux_drv.pt is aux_drv/2 (vdac_postgate.py judges the LoRA Adam state too)
  HB-SHUF) HAUPD=2000;  SHUFSEED=0 ;;
  HC-s0)   HAUPD=10000; SHUFSEED=none; HEADN=3423490 ;;          # 1 m cells + boundary weight
  HC-SHUF) HAUPD=2000;  SHUFSEED=0; HEADN=3423490 ;;
  HBC-s0)  HAUPD=10000; SHUFSEED=none; HEADN=3423490 ;;          # VDAC A6 (DRAFT): HB's copy + HC's 1 m head and boundary weight; aux_drv.pt is aux_drv/2 (LoRA 3,145,728 + head 3,423,490 both judged)
  HBC-SHUF) HAUPD=2000;  SHUFSEED=0; HEADN=3423490 ;;
  *) echo "REFUSED: stage=start: unknown arm '$ARMN' (usage: vdac_chain.sh HA-s0 | HA-s1 | HA-SHUF | HB-s0 | HB-SHUF | HC-s0 | HC-SHUF | HBC-s0 | HBC-SHUF)"; exit 2 ;;
esac
PROFILE=refe_vdac_p6
PCARD_NAME=PREREG_VDAC_P6.md;    PCARD_SHA=5594c2ae777a53ed4a8a873ca374c12943eadc74385cb07671c2ac90a3292ae4
A1_NAME=PREREG_VDAC_P6_A1.md;    A1_SHA=8f121bdf9d6f19acec8927f12c50d3f7be2832c70c7f1fe40140d5eb8cc4ed70
A2_NAME=PREREG_VDAC_P6_A2.md;    A2_SHA=c3d563586be6479ea11455f92d7a5295c9e1658e2d7ba1cc77b27e87e5c8b417
H48_SHA=2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8
STORE_KEYS=103139
LA=$(echo "$ARMN" | tr 'A-Z' 'a-z')
WS=${VDAC_WS:-/workspace}
DATA=$WS/data
PKG=${VDAC_PKG:-$WS/refe-plan-vdac}
RUN=$PKG/refe
ROOT=$DATA/refe_vdac
PROOT=$DATA/refe_scorer1p6
DUMP=$DATA/p6base_labels
LABELS=$DATA/aux_drivable/label_store
OUT=$ROOT/$LA
S=$ROOT/stage_$LA
L=$ROOT/chain_$LA.log
REPO=${VDAC_HF_REPO:-Sayood/tanitad-refe-scorer1p6}          # an existing PRIVATE repo (G-EGR passes on it); the arm is banked under the prefix vdac/<arm lower>
HFTOK=${VDAC_HF_TOKEN_FILE:-}
CKPT=$DATA/refe_runs/p6_paper_faithful/model_final.pt
CKPT_SHA=968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1
NTL=$PROOT/navtest_ids/navtest_logs.json;  NTL_SHA=79daa3c2ba3fbf1c6d45d0e3b630f43bb26a073160a34ca5f9c98553c0bf43e0
NTT=$PROOT/navtest_ids/navtest_tokens.txt.gz; NTT_SHA=e61613ce129bb67dc654a5a575de2889a22fcfcbc64e4ec389a38295b8df9c7a
H48F=$ROOT/h48_logs.txt
NAVTEST_JSON=${AUX_NAVTEST:-$DATA/aux_drivable/navtest.json}; NAVTEST_SHA=3ba1c96488f157fd2ca7c57c83abe439b4c9680ebdb1eadf65c34ac323d42d44      # the aux navtest list G-SUITE's E10 reads (same pin as s1aux_pod.sh)
navtest_sha() { tr -d '\r' < "$1" 2> /dev/null | sha256sum | cut -c1-64; }
EXPORT=$ROOT/export/h48_proposals_subscores.npz
export TANITAD_GATE_KEY_FILE=${TANITAD_GATE_KEY_FILE:-$ROOT/launch_gate.key}      # the HMAC key lives on the volume, OUTSIDE the shipped tree and anything the egress pushes
mkdir -p $ROOT $S
STAGE=start
TREE=""
export VDAC_WS=$WS SCORER1P6_WS=$WS
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: stage=$STAGE: $*"; exit 1; }
stop_after() { if [ "${VDAC_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (VDAC_STOP_AFTER)"; exit 0; fi; }
tree_sha() { sha256sum "$PKG/SHIP_MANIFEST.json" 2>/dev/null | cut -c1-64; }
stage_done() { [ -f "$S/$1.ok" ] && [ "$(sed -n 2p "$S/$1.ok")" = "$TREE" ]; }
stage_mark() { { date -u +%FT%T; echo "$TREE"; } > "$S/$1.ok"; }
gpu_busy() {          # rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never pgrep -f.
  local nv="${VDAC_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }        # READ-WRITE fd: an exclusive flock on a read-only fd is EBADF on nfs4
others_hold() {
  local c
  for c in refe_p2 refe_p2rep rls1 s1aux refe_p6 refe_p7 refe_scorer1p6 refe_s1z refe_s1zsm refe_rep refe_dose refe_div; do
    if lock_held "$DATA/$c"; then echo "the $c chain holds $DATA/$c/chain.lock"; return 2; fi
  done
  return 0
}
wait_clear() {        # BLOCK until no chain holds the pod (P7 is PAUSED for this run: its lock must be free) and the GPU was idle on VDAC_IDLE_POLLS consecutive polls
  local poll="${VDAC_POLL_S:-30}" need="${VDAC_IDLE_POLLS:-3}" max="${VDAC_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why=""; rc=0
    why="$(others_hold)"; rc=$?
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZVDAC_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (VDAC_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
arm_argv() { ARGS="$(ARM=$1 MODE=argv VDAC_WS=$WS bash $RUN/vdac_pod.sh)" && [ -n "$ARGS" ] || refuse "vdac_pod.sh ARM=$1 MODE=argv printed nothing"; }
token_ok() {          # $1 = arm, uses $ARGS.  `launch_gate.py verify` RE-MEASURES the argv, the code tree, the checkpoint sha and the data and checks the HMAC.
  local tok
  tok="$(ls "$ROOT/lg_$1"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY_C" "$RUN/launch_gate.py" verify --profile $PROFILE --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_$1.json" >> "$ROOT/token_verify_$1.log" 2>&1 200>&-
}

# the exclusion block of a preflight / run log's data event, PARSED (never grepped):  excl_is <log> <shuffle seed | none>  -> rc 0 iff the data event carries the `aux_excl` block of THIS list with 0 labelled H48 rows reaching the loss, the filter == the file,
# > 0 labelled frames excluded; and (HA-SHUF) the derangement block of the registered seed with 0 fixed points over all store keys.  Prints the counts.
excl_is() {
  $PY_C - "$1" "$2" "$H48_SHA" "$STORE_KEYS" <<'PYEOF'
import json, sys
log, shuf, h48, keys = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(ln) for ln in open(log, encoding="utf-8", errors="replace") if ln.startswith('{"event": "data"')]
if len(rows) != 1:
    sys.exit("%s: %d data events, want 1" % (log, len(rows)))
d = rows[0]
b = d.get("aux_excl")
if not isinstance(b, dict):
    sys.exit("%s: the data event carries no aux_excl block (--aux-drv-exclude-logs was not built)" % log)
bad = []
if b.get("sha256") != h48: bad.append("aux_excl.sha256 %r is not the H48 list's" % (b.get("sha256"),))
if b.get("reached_loss_rows") != 0 or b.get("reached_loss_frames") != 0: bad.append("H48 rows reaching the loss: %r rows / %r frames (must be 0)" % (b.get("reached_loss_rows"), b.get("reached_loss_frames")))
if b.get("filter_equals_file") is not True: bad.append("aux_excl.filter_equals_file %r" % (b.get("filter_equals_file"),))
if not (isinstance(b.get("excluded_labelled_frames"), int) and b["excluded_labelled_frames"] > 0): bad.append("aux_excl.excluded_labelled_frames %r (must be > 0)" % (b.get("excluded_labelled_frames"),))
a = d.get("aux_drv") or {}
if shuf == "none":
    if a.get("shuffle_seed") is not None: bad.append("aux_drv.shuffle_seed %r on a non-control arm" % (a.get("shuffle_seed"),))
else:
    if a.get("shuffle_seed") != int(shuf): bad.append("aux_drv.shuffle_seed %r != %s" % (a.get("shuffle_seed"), shuf))
    if a.get("derangement_fixed_points") != 0 or a.get("shuffle_keys") != keys: bad.append("derangement: %r fixed points over %r keys (want 0 over %d)" % (a.get("derangement_fixed_points"), a.get("shuffle_keys"), keys))
if bad:
    sys.exit("%s: %s" % (log, "; ".join(bad)))
print("H48 logs %s: bank rows %s, frames %s, labelled frames excluded %s, reaching the loss %s" % (b["logs"], b["excluded_rows"], b["excluded_frames"], b["excluded_labelled_frames"], b["reached_loss_rows"]))
PYEOF
}

# ---- 0. precheck: the lock, the cards, the H48 list, the package
exec 200>"$ROOT/chain.lock"
flock -n 200 || { echo "REFUSED: stage=precheck: another VDAC chain holds $ROOT/chain.lock"; exit 1; }
STAGE=precheck
[ -n "$HFTOK" ] || refuse "VDAC_HF_TOKEN_FILE is not set (hf_egress.py reads the token file in place; no default path in source)"
TREE="$(tree_sha)"
log "chain armed (ZZVDAC_START $ARMN, $HAUPD updates, shuffle $SHUFSEED; vdac_chain.sh md5 $(md5sum "$0" | cut -c1-12), tree ${TREE:0:12})"
[ -n "$TREE" ] || refuse "no $PKG/SHIP_MANIFEST.json: ship the landed tree with its manifest (launch_gate.py ship-manifest --commit <40-hex>)"
[ -r "$HFTOK" ] && grep -qE 'hf_[A-Za-z0-9]{8,}' "$HFTOK" || refuse "$HFTOK is missing, unreadable or holds no hf_ token"
# the frozen card and its two amendments, by bytes: a card with other bytes is not the frozen experiment this arm runs (ship them into $ROOT)
for c in "$PCARD_NAME:$PCARD_SHA" "$A1_NAME:$A1_SHA" "$A2_NAME:$A2_SHA"; do
  [ -f "$ROOT/${c%%:*}" ] || refuse "$ROOT/${c%%:*} is missing: ship the frozen card into $ROOT"
  [ "$(sha256sum "$ROOT/${c%%:*}" | cut -c1-64)" = "${c##*:}" ] || refuse "$ROOT/${c%%:*} is not the frozen ${c%%:*} (sha256 ${c##*:})"
done
# the H48 list (amendment A3): the shipped file by bytes, copied to the path the argv names, re-hashed there
[ -f "$PKG/splits/h48_logs.txt" ] || refuse "$PKG/splits/h48_logs.txt is missing from the shipped package"
[ "$(sha256sum "$PKG/splits/h48_logs.txt" | cut -c1-64)" = "$H48_SHA" ] || refuse "$PKG/splits/h48_logs.txt is not the H48 list (sha256 $H48_SHA)"
cp "$PKG/splits/h48_logs.txt" "$H48F"
[ "$(sha256sum "$H48F" | cut -c1-64)" = "$H48_SHA" ] && [ "$(grep -c . "$H48F")" = 48 ] || refuse "$H48F is not the 48-log H48 list after the copy"
[ -f "$WS/teacher_env.sh" ] || refuse "$WS/teacher_env.sh is missing"
source $WS/teacher_env.sh
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
PY_C="${DRIVERL_EVAL_PYTHON:-}"
[ -n "$PY_C" ] && [ -x "$PY_C" ] || refuse "teacher_env.sh did not set a runnable DRIVERL_EVAL_PYTHON"
HPY=${HF_PY:-$PY_C}
NEED="refe/scorer_finetune.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/pdm_label_complete.py refe/scorer1p6_pod.sh refe/vdac_pod.sh refe/vdac_chain.sh refe/test_vdac_gate.py refe/vdac_layer.py refe/vdac_postgate.py refe/vdac_export_h48.py refe/aux_hb.py refe/vdac_probes.py refe/test_vdac_hb.py \
      refe/div_layer.py refe/s1zsm_layer.py refe/s1z_layer.py refe/scorer1_layer.py refe/aux_drivable_head.py refe/aux_label_store.py refe/test_aux_drivable_head.py refe/test_aux_s1_wiring.py refe/test_s1c.py refe/test_rl_reward.py refe/test_rl_epo.py refe/test_launch_gate.py \
      refe/train.py refe/model.py refe/ckpt_io.py refe/planner.py splits/navtrain.yaml splits/h48_logs.txt SHIP_MANIFEST.json"
for f in $NEED; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f is missing from the shipped package"
done
env -u HF_TOKEN "$PY_C" - "$PKG" "$RUN" > "$ROOT/tree_check_$LA.log" 2>&1 200>&- <<'PYEOF' || refuse "the shipped tree is not its SHIP_MANIFEST.json: $(tail -n 1 "$ROOT/tree_check_$LA.log" | cut -c1-300)"
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import gate_token as GT
pkg = Path(sys.argv[1])
man = GT.read_json(pkg / GT.SHIP_MANIFEST)
tree = GT.tree_manifest(pkg)
if man.get("files") != tree:
    sys.exit("tree differs from the manifest: %s" % GT.tree_diff(man.get("files") or {}, tree))
print("ZZVDAC_TREE_OK %d files commit %s" % (len(tree), str(man.get("commit"))[:12]))
PYEOF
log "ZZVDAC_PRECHECK_OK ($(tail -n 1 $ROOT/tree_check_$LA.log))"; stop_after precheck

# ---- 1. the inputs, AUDITED (read-only): P6's label dump, bank, checkpoint, navtest lists, and the label store (every shard re-hashed)
STAGE=inputs
if ! stage_done inputs; then
  for p in "$DUMP/train_sets" "$DUMP/heldout_sets" "$DUMP/logs/heldout_logs.txt" "$DUMP/labels_manifest.json" "$DUMP/labels.DONE" "$DUMP/train_pdm" "$DUMP/heldout_pdm" "$DATA/refe_navtrain/train_grow_cmd_3c_p7" "$DATA/refe_navtrain/train_grow/calib_table.json" "$DATA/navtrain_pixels" "$CKPT" "$NTL" "$NTT" "$LABELS/manifest.json"; do
    [ -e "$p" ] || refuse "$p is missing: P6's inputs and the aux label store must be on the pod"
  done
  [ "$(sha256sum "$CKPT" | cut -c1-64)" = "$CKPT_SHA" ] || refuse "$CKPT is not P6's model_final.pt (sha256 968a469d...)"
  [ "$(sha256sum "$NTL" | cut -c1-64)" = "$NTL_SHA" ] || refuse "$NTL is not the registered navtest log list"
  [ "$(sha256sum "$NTT" | cut -c1-64)" = "$NTT_SHA" ] || refuse "$NTT is not the registered navtest token list"
  PIN="$($PY_C -c 'import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; p = L.SCORER1P6_DATA_PIN; print("UNSET" if p is None else "%d %d" % (p["train_sets"], p["heldout_sets"]))' "$RUN" 2> /dev/null)"
  case "$PIN" in UNSET|"") refuse "launch_gate.SCORER1P6_DATA_PIN is unset: the exact P6 label counts are not pinned" ;; esac
  NTR=${PIN% *}; NHO=${PIN#* }
  [ "$(cat $DUMP/train_pdm/*.jsonl | wc -l)" -ge "$NTR" ] || refuse "train PDM labels hold $(cat $DUMP/train_pdm/*.jsonl | wc -l) rows, want >= $NTR (the pin)"
  [ "$(cat $DUMP/heldout_pdm/*.jsonl | wc -l)" = "$NHO" ] || refuse "held-out PDM labels hold $(cat $DUMP/heldout_pdm/*.jsonl | wc -l) rows, want $NHO (the pin)"
  # the label store: the pinned store sha256, every shard re-hashed against its manifest (content, never presence)
  env -u HF_TOKEN "$PY_C" - "$RUN" "$LABELS" > "$ROOT/store_check.log" 2>&1 200>&- <<'PYEOF' || refuse "the aux label store is not the pinned one: $(tail -n 1 "$ROOT/store_check.log" | cut -c1-300)"
import sys
sys.path.insert(0, sys.argv[1])
import aux_drivable_head as AUX
src = AUX.AuxLabelSource(sys.argv[2], max_logs=1)
v = src.verify_all()
if v["mismatch"] != 0 or v["files"] != src.manifest["counts"]["logs"] or src.manifest["counts"]["stored_keys"] != AUX.PINNED_STORE_KEYS:
    sys.exit("store verification: %s" % (v,))
print("ZZVDAC_STORE_OK %d shards, %d frames, sha256 %s" % (v["files"], src.manifest["counts"]["stored_keys"], src.manifest["store_sha256"]))
PYEOF
  stage_mark inputs
fi
log "ZZVDAC_INPUTS_OK ($(tail -n 1 $ROOT/store_check.log | cut -c1-120))"; stop_after inputs

# ---- 2. G-EGR (the run must be able to leave the pod), a real disk write test
STAGE=egr
$HPY -c "import huggingface_hub, torch" 2>> $L || refuse "huggingface_hub/torch not importable by $HPY"
if ! stage_done egr; then
  $HPY $RUN/hf_egress.py gate --repo $REPO --out $ROOT/gate --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" > $ROOT/gate_$LA.log 2>&1 200>&-
  grep -q " ZZEGR_PASS " $ROOT/gate_$LA.log || refuse "G-EGR did not pass ($(tail -1 $ROOT/gate_$LA.log | cut -c1-300)); the HF repo $REPO must exist"
  [ -f $ROOT/gate/g_egr_result.json ] || refuse "G-EGR wrote no g_egr_result.json"
  stage_mark egr
fi
log "ZZVDAC_EGR_OK"; stop_after egr
STAGE=dd
if ! stage_done dd; then
  DDMB=${VDAC_DD_MB:-8192}                 # one arm: model_sft_A/B 2.6 GB + states + the head + the score dump + the H48 forward + staging
  dd if=/dev/zero of=$ROOT/dd_test_$LA.tmp bs=1M count=$DDMB conv=fsync > $ROOT/dd_test_$LA.log 2>&1 200>&- ; ddrc=$?
  rm -f $ROOT/dd_test_$LA.tmp
  [ $ddrc -eq 0 ] || refuse "the ${DDMB} MB write test failed: $(tail -n 1 $ROOT/dd_test_$LA.log)"
  stage_mark dd
fi
log "ZZVDAC_DD_OK"; stop_after dd

# ---- 3. the pod: P7 over, nothing else holds it, GPU idle.
STAGE=gpu_wait
wait_clear

# ---- 4. the arm: preflight + the exclusion audit + the NOFILTER mutant -> LAUNCH GATE -> train -> check -> post-training gate -> export -> bank
STAGE="preflight_$ARMN"
arm_argv "$ARMN"
if ! stage_done "preflight_$LA"; then
  wait_clear
  rm -f $OUT/preflight.jsonl $OUT/PREFLIGHT_PASS.json
  ARM=$ARMN MODE=preflight bash $RUN/vdac_pod.sh 200>&-
  grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "$ARMN preflight did not pass ($(tail -n 2 $OUT/preflight.log | tr '\n' ' ' | cut -c1-300))"
  m=$(excl_is $OUT/preflight.log $SHUFSEED 2>&1) || refuse "the $ARMN preflight's exclusion audit failed: $m"
  log "ZZVDAC_EXCL $ARMN $m"
  # the deliberate regression: the exclusion filter REMOVED on the real argv.  The trainer must reject it (rc 4, an `aux-excl` refusal) and must NOT pass the preflight; a mutant that passes means the audit cannot see a missing filter: no training
  ARM=$ARMN MODE=mutate bash $RUN/vdac_pod.sh 200>&-; mrc=$?
  [ "$mrc" = 4 ] && grep -q "^REFUSED: aux-excl:" $OUT/mutate_exclude.log || refuse "the NOFILTER mutant was NOT rejected by the trainer's exclusion audit (rc $mrc): the audit cannot see a removed filter, so $ARMN is not trained ($(tail -n 2 $OUT/mutate_exclude.log | tr '\n' ' ' | cut -c1-200))"
  ! grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/mutate_exclude.log || refuse "the NOFILTER mutant passed the preflight: no training"
  log "$ARMN NOFILTER mutant rejected as required, rc $mrc ($(grep -m1 '^REFUSED: aux-excl:' $OUT/mutate_exclude.log | sed 's/^REFUSED: //' | cut -c1-160))"
  # G19 / G20 (card section 7): parsed from the REAL preflight, re-judged from their raw fields by vdac_probes (never the gate's own ok); the bank the head draws from is printed here, BEFORE training
  g=$(env -u HF_TOKEN "$PY_C" - "$OUT/preflight.jsonl" "$RUN" 2>&1 200>&- <<'PYEOF'
import json, sys
sys.path.insert(0, sys.argv[2])
import vdac_probes as VP
rows = [json.loads(ln) for ln in open(sys.argv[1], encoding="utf-8", errors="replace") if ln.startswith('{"event": "preflight"')]
if not rows:
    sys.exit("no preflight event in %s" % sys.argv[1])
G = rows[-1].get("gates") or {}
g19, g20 = G.get("G19_vdac_head_disjoint_from_ego_command"), G.get("G20_vdac_label_aligned_with_targets")
if g19 is None or g20 is None:
    sys.exit("the preflight carries no G19/G20 gate (the head's disjointness and the label alignment were NOT proved)")
bad = VP.judge_g19(g19) + VP.judge_g20(g20)
if bad:
    sys.exit("; ".join(bad))
print("G19 head grad wrt ego %s, command %s (control %.3g / %.3g); G20 %d frames, real %.4f, y-flip %.4f, shift1 %.4f; bank %s" % (g19["head_grad_ego"], g19["head_grad_command"], g19["control_grad_ego"], g19["control_grad_command"], g20["n_frames"], g20["frac_real"], g20["frac_yflip"], g20["frac_shift1"], g20.get("bank")))
PYEOF
  ) || refuse "$ARMN preflight: G19/G20 not proved: $(echo "$g" | tail -n 1 | cut -c1-400)"
  log "ZZVDAC_G19_G20 $ARMN $g"
  # G19's deliberate regression on the real model: the ego token routed into the head's context must turn the disjointness probe RED, and the preflight must not pass
  ARM=$ARMN MODE=mutate_ego bash $RUN/vdac_pod.sh 200>&-; erc=$?
  ! grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/mutate_ego.log || refuse "the EGO-INTO-HEAD mutant passed the preflight: G19 cannot see a head that reads ego, so $ARMN is not trained"
  env -u HF_TOKEN "$PY_C" - $OUT/mutate_ego.log $RUN 200>&- <<'PYEOF' || refuse "the EGO-INTO-HEAD mutant was not rejected by G19 specifically (rc $erc): $(tail -n 2 $OUT/mutate_ego.log | tr '
' ' ' | cut -c1-300)"
import json, sys
sys.path.insert(0, sys.argv[2])
import vdac_probes as VP
rows = [json.loads(ln) for ln in open(sys.argv[1], encoding="utf-8", errors="replace") if ln.startswith('{"event": "preflight"')]
g19 = ((rows[-1].get("gates") or {}).get("G19_vdac_head_disjoint_from_ego_command")) if rows else None
if not isinstance(g19, dict) or not any("NOT disjoint" in r for r in VP.judge_g19(g19)):
    sys.exit("G19 under the mutant: %s" % (g19,))
print("mutant read head grad wrt ego %s" % g19.get("head_grad_ego"))
PYEOF
  log "$ARMN EGO-INTO-HEAD mutant rejected by G19 as required, rc $erc"
  stage_mark "preflight_$LA"
fi
log "ZZVDAC_PREFLIGHT_OK $ARMN"; stop_after "preflight_$LA"
STAGE="gate_$ARMN"
if ! stage_done "gate_$LA"; then
  [ "$(navtest_sha "$NAVTEST_JSON")" = "$NAVTEST_SHA" ] || refuse "the navtest list $NAVTEST_JSON is absent or its CR-stripped bytes are not sha256 ${NAVTEST_SHA:0:12}: the live G-SUITE (E10) cannot read it"
  rm -rf "$ROOT/lg_$ARMN"; mkdir -p "$ROOT/lg_$ARMN"
  env -u HF_TOKEN OMP_NUM_THREADS=4 AUX_LABEL_STORE="$LABELS" AUX_NAVTEST="$NAVTEST_JSON" "$PY_C" $RUN/launch_gate.py run --profile $PROFILE --arm-name "$ARMN" --pkg "$PKG" --args "$ARGS" --out-dir "$ROOT/lg_$ARMN" \
    --gate-preflight-json $OUT/preflight.jsonl --gate-egr-json $ROOT/gate/g_egr_result.json > $ROOT/launch_gate_$ARMN.log 2>&1 200>&-
  echo "ZZEXIT $?" >> $ROOT/launch_gate_$ARMN.log
  token_ok "$ARMN" || refuse "the launch gate did not mint a PASS token that verifies for $ARMN: $(grep -E 'REASON|FAIL' $ROOT/launch_gate_$ARMN.log | head -n 3 | cut -c1-300 | tr '\n' ' ')"
  stage_mark "gate_$LA"
fi
log "ZZVDAC_GATE_OK $ARMN"; stop_after "gate_$LA"
STAGE="train_$ARMN"
if stage_done "trainstart_$LA" && ! stage_done "train_$LA"; then
  k=1; while [ -e "$OUT.killed_$k" ]; do k=$((k + 1)); done
  refuse "a previous attempt of $ARMN STARTED training and did not finish; scorer_finetune.py has no resume. Archive it (mv $OUT $OUT.killed_$k), re-run: the chain restarts the arm at update 0 under a fresh preflight and token"
fi
if ! stage_done "train_$LA"; then
  wait_clear
  token_ok "$ARMN" || refuse "the launch-gate token does not verify for $ARMN: scorer_finetune.py is NOT started ($(tail -n 2 $ROOT/token_verify_$ARMN.log | tr '\n' ' ' | cut -c1-300))"
  stage_mark "trainstart_$LA"
  log "run launch $ARMN (ZZVDAC_TRAIN_START)"
  set -m
  ARM=$ARMN MODE=run bash $RUN/vdac_pod.sh 200>&- &
  TPID=$!
  set +m
  wait $TPID; RRC=$?
  log "RUN EXIT $ARMN $RRC ($(tail -1 $OUT/sft.log))"
  env -u HF_TOKEN "$PY_C" - "$OUT" "$HAUPD" "$H48_SHA" > "$OUT/run_check.log" 2>&1 200>&- <<'PYEOF'
import json, math, os, sys
out, upd, h48 = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [json.loads(ln) for ln in open(os.path.join(out, "sft.jsonl"), encoding="utf-8") if ln.startswith("{")]
ev = lambda n: [r for r in rows if r.get("event") == n]
done, start, data, h48x, saved = ev("done"), ev("start"), ev("data"), ev("h48x_forward"), ev("aux_saved")
bad = []
if "ZZSFT_DONE" not in open(os.path.join(out, "sft.log"), encoding="utf-8", errors="replace").read():
    bad.append("no ZZSFT_DONE in sft.log")
# the REAL run's own data event: the exclusion block of THIS list, nothing of H48 reaching the loss
b = (data[0].get("aux_excl") if len(data) == 1 else None) or {}
if len(data) != 1 or b.get("sha256") != h48 or b.get("reached_loss_rows") != 0 or b.get("filter_equals_file") is not True or not (isinstance(b.get("excluded_labelled_frames"), int) and b["excluded_labelled_frames"] > 0):
    bad.append("the run's data event aux_excl block is %s (want sha256 %s, 0 rows reaching the loss, the filter == the file, > 0 labelled frames excluded)" % (b, h48))
n = data[0].get("train_sets") if len(data) == 1 else None
want = min(upd, int(math.ceil(n / 8 * 2 / 1))) if isinstance(n, int) and n > 0 else None      # batch 8, --epochs 2, accum 1
if want != upd:
    bad.append("the schedule reaches %s updates on %s sets, the arm needs %d" % (want, n, upd))
print("ZZVDAC_UPDATES done=%s start=%s want=%d (--max-updates is the stop that binds, --epochs 2 only supplies the draws)" % ([d.get("updates") for d in done], [s.get("updates") for s in start], upd))
if len(done) != 1 or done[0].get("updates") != upd:
    bad.append("done event %s, want updates %d" % ([d.get("updates") for d in done], upd))
if len(start) != 1 or start[0].get("updates") != upd:
    bad.append("start event %s, want updates %d" % ([s.get("updates") for s in start], upd))
if len(saved) != 1 or not os.path.isfile(os.path.join(out, "aux_drv.pt")):
    bad.append("the head was not saved (aux_saved events %d, aux_drv.pt present %s)" % (len(saved), os.path.isfile(os.path.join(out, "aux_drv.pt"))))
# the head's forward on the frames it never trained on: one row per frame of the excluded logs this bank holds == the audit's excluded frames
if len(h48x) != 1 or h48x[0].get("frames") != b.get("excluded_frames") or not os.path.isfile(os.path.join(out, "head_pred_h48x.npz")):
    bad.append("the H48 forward is %s, the audit excluded %s frames" % ([x.get("frames") for x in h48x], b.get("excluded_frames")))
if not os.path.isfile(os.path.join(out, "scores", "head_pred_update%d.npz" % upd)):
    bad.append("no held-out head prediction file scores/head_pred_update%d.npz" % upd)
for m in ("model_sft_A.pt", "model_sft_B.pt"):
    if not os.path.isfile(os.path.join(out, m)):
        bad.append("missing " + m)
if bad:
    sys.exit("; ".join(bad))
print("ZZVDAC_RUN_OK updates %d on %s sets, %s H48 frames forwarded" % (upd, n, h48x[0].get("frames")))
PYEOF
  RCC=$?
  log "run check $ARMN: $(tail -n 1 $OUT/run_check.log | cut -c1-300)"
  [ $RRC -eq 0 ] && [ $RCC -eq 0 ] || { log "ZZVDAC_CHAIN_FAIL $ARMN (run rc $RRC, run check rc $RCC)"; exit 1; }
  # the post-training gate: every head parameter holds Adam state and a non-zero first moment, at the arm's step count
  env -u HF_TOKEN "$PY_C" $RUN/vdac_postgate.py "$OUT" --updates $HAUPD --n-params $HEADN > "$OUT/postgate.log" 2>&1 200>&-
  PGRC=$?
  log "post-training gate $ARMN rc $PGRC: $(tail -n 1 $OUT/postgate.log | cut -c1-300)"
  [ $PGRC -eq 0 ] && grep -q "^ZZVDAC_POSTGATE_OK" $OUT/postgate.log || { log "ZZVDAC_CHAIN_FAIL $ARMN (post-training gate)"; exit 1; }
  stage_mark "train_$LA"
fi
log "ZZVDAC_TRAIN_DONE $ARMN"; stop_after "train_$LA"

# ---- 5. the export of the H48 sets' P6 proposals and NAVSIM sub-scores (CPU, once per pod; the bank and the PDM labels are P6's, nothing is trained)
STAGE=export
if [ ! -f "$ROOT/stage_export.ok" ] || [ "$(sed -n 2p "$ROOT/stage_export.ok")" != "$TREE" ]; then
  mkdir -p "$ROOT/export"
  env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY_C" $RUN/vdac_export_h48.py --h48 "$H48F" --h48-sha256 $H48_SHA --train-sets $DUMP/train_sets --train-pdm $DUMP/train_pdm --heldout-sets $DUMP/heldout_sets --heldout-pdm $DUMP/heldout_pdm \
    --ckpt "$CKPT" --out "$EXPORT" > "$ROOT/export/export.log" 2>&1 200>&-
  grep -q "^ZZVDAC_EXPORT_OK" "$ROOT/export/export.log" || refuse "the H48 export failed: $(tail -n 1 $ROOT/export/export.log | cut -c1-300)"
  { date -u +%FT%T; echo "$TREE"; } > "$ROOT/stage_export.ok"
fi
log "ZZVDAC_EXPORT_OK ($(tail -n 1 $ROOT/export/export.log | cut -c1-200))"; stop_after export

# ---- 6. bank the arm off the pod (a verified remote listing, not the push call): the head, its predictions on H48, the held-out predictions, the logs, and (once) the export
STAGE=egress
if ! stage_done "egress_$LA"; then
  N0=$(wc -l < $L)
  $HPY $RUN/hf_egress.py push --repo $REPO --prefix "vdac/$LA" --max-push-bytes 6e9 --tag "arm_vdac_$LA" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" \
    --files $OUT/aux_drv.pt $OUT/head_pred_h48x.npz $OUT/scores/head_pred_update$HAUPD.npz $OUT/sft.log $OUT/sft.jsonl $OUT/preflight.log $OUT/mutate_exclude.log $ROOT/launch_gate_$ARMN.log $OUT/run_check.log $OUT/postgate.log >> $ROOT/gate_$LA.log 2>&1 200>&-
  tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_vdac_$LA " || refuse "the $ARMN egress push did not verify: the arm lives on the pod only"
  N1=$(wc -l < $L)
  $HPY $RUN/hf_egress.py push --repo $REPO --prefix "vdac/export" --max-push-bytes 6e9 --tag "arm_vdac_export_$LA" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" \
    --files $EXPORT $EXPORT.json >> $ROOT/gate_$LA.log 2>&1 200>&-
  tail -n +$((N1 + 1)) $L | grep -q " ZZEGRESS_OK arm_vdac_export_$LA " || refuse "the export push did not verify: the S1 read's input lives on the pod only"
  # the LFS sha256 of the banked head (what the hub holds) next to the sha256 of the pod's file: both recorded, and they must agree (the token file is read in place, never printed)
  LOCSHA=$(sha256sum $OUT/aux_drv.pt | cut -c1-64)
  env -u HF_TOKEN "$HPY" - "$REPO" "vdac/$LA/aux_drv.pt" "$HFTOK" "$LOCSHA" "$OUT/lfs_sha.json" >> $L 2>&1 200>&- <<'PYEOF' || log "ZZVDAC_LFS_SHA_UNVERIFIED $ARMN: the hub's LFS sha256 could not be read or differs; the pod file's sha256 is $LOCSHA (verify by sha256 after the pull)"
import json, re, sys
from huggingface_hub import HfApi
repo, path, tokfile, loc, outp = sys.argv[1:6]
tok = re.search(r"hf_[A-Za-z0-9]+", open(tokfile, encoding="utf-8").read()).group(0)
info = HfApi(token=tok).get_paths_info(repo, [path], expand=True)
lfs = getattr(info[0], "lfs", None)
sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
json.dump({"repo": repo, "path": path, "lfs_sha256": sha, "pod_sha256": loc, "equal": sha == loc}, open(outp, "w"))
print("ZZVDAC_LFS_SHA %s %s pod %s equal %s" % (path, sha, loc, sha == loc))
if sha != loc:
    sys.exit("the hub's LFS sha256 %s != the pod file's %s" % (sha, loc))
PYEOF
  stage_mark "egress_$LA"
fi
log "ZZVDAC_ARM_DONE $ARMN"
log "ZZVDAC_CHAIN_DONE $ARMN"
