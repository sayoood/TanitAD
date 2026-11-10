#!/bin/bash
# DIV chain (PREREG_DIV_P6): ONE arm per invocation, DV0 or DV1 = C0's argv + `--split-half <h>` (DV0: half 0, C0's seed 0; DV1: half 1, C0r's seed 1), 400 updates, from P6's model_final.pt, on the L40 pod.  NOT PLACED ON THE POD by its author; the Master Mind ships it with the landed tree.
#   bash /workspace/refe-plan-div/refe/div_chain.sh DV0|DV1        (DIV_HF_TOKEN_FILE=<the pod's HF token file> in the environment; refe/div_pause_p7.sh runs DV0 then DV1 inside ONE P7 pause)
# Pattern: refe/dose_chain.sh (same fail-closed stages and the SAME wait functions, markers keyed by the SHA-256 of SHIP_MANIFEST.json, REFUSED lines in the arm's log), with the DIV differences:
#  * OWN POD DIRECTORY: package $WS/refe-plan-div, data $DATA/refe_div.  gate_token.TREE_ROOTS = (refe, code, splits): shipping this refe/ over refe-plan-p6, -p7, -scorer1p6, -s1z, -s1zsm, -rep or -dose (a dir that holds a minted token) would VOID that token.
#  * THE PARENT'S INPUTS: C0's card (PREREG_SCORER1P6.md, by bytes) AND the DIV card (PREREG_DIV_P6.md, by bytes, shipped to $ROOT, sha256 e5fb2f8e...), the navtest lists, the checkpoint and the label dump are C0's.  No lambda: C0 names no pair / twin lever.
#  * THE SPLIT IS CHECKED THREE TIMES: (1) the preflight's own data event (the half, the rule, 0 kept rows off the half, both halves >= 25,000 sets) is printed with both halves' set and log counts and PARSED here; (2) the DELIBERATE-REGRESSION MUTANT
#    (mutate_nofilter: the split filter removed) must be REFUSED by the trainer (rc 4, `REFUSED: split:`, no preflight pass), or the chain refuses to train; (3) the real run's own data event is parsed again.
#  * NO PAUSING OR RESUMING OF P7 here (refe/div_pause_p7.sh does that).  The chain waits for NO chain lock held (the parents', refe_rep and refe_dose included) and the GPU idle on 3 consecutive polls.
#  * ORDER: precheck, inputs, G-EGR, dd, wait, preflight (G1-G11 on the real bank + the split block), the mutant, the LAUNCH GATE (profile refe_div_p6; a PASS token minted by launch_gate.py, never hand-written), the run, the run check
#    (updates == min(400, ceil(split sets / 64)); the card: a half whose epoch is shorter stops at the epoch and that is stated), the HF egress under the prefix div/<arm> with the LFS sha256 recorded, ZZDIV_CHAIN_DONE <arm>.
# Markers: $ROOT/stage_<arm>/<name>.ok = date + tree sha.  DIV_STOP_AFTER=<stage> ends the chain after that stage (a dry run).
set -u
ARMN=${1:-}
case "$ARMN" in
  DV0) PARENT=C0;  HALF=0; PROOT_NAME=refe_scorer1p6; PCARD=PREREG_SCORER1P6.md; CARD_SHA=adb5aafd4fb8abdfd1a53d9cc5ff865a89bd20f9008c701cf5d308dac595abed; PROFILE=refe_div_p6; PPOD=scorer1p6_pod.sh ;;
  DV1) PARENT=C0r; HALF=1; PROOT_NAME=refe_scorer1p6; PCARD=PREREG_SCORER1P6.md; CARD_SHA=adb5aafd4fb8abdfd1a53d9cc5ff865a89bd20f9008c701cf5d308dac595abed; PROFILE=refe_div_p6; PPOD=scorer1p6_pod.sh ;;
  *) echo "REFUSED: unknown arm '$ARMN' (usage: div_chain.sh DV0 | DV1)"; exit 2 ;;
esac
DCARD_NAME=PREREG_DIV_P6.md; DCARD_SHA=e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac
SPLIT_MIN=25000
LA=$(echo "$ARMN" | tr 'A-Z' 'a-z')
WS=${DIV_WS:-/workspace}
DATA=$WS/data
PKG=${DIV_PKG:-$WS/refe-plan-div}
RUN=$PKG/refe
ROOT=$DATA/refe_div
PROOT=$DATA/$PROOT_NAME
DUMP=$DATA/p6base_labels
OUT=$ROOT/$LA
S=$ROOT/stage_$LA
L=$ROOT/chain_$LA.log
REPO=${DIV_HF_REPO:-Sayood/tanitad-refe-scorer1p6}          # an existing repo (G-EGR passes on it); the arm is banked under the prefix div/<arm lower>
HFTOK=${DIV_HF_TOKEN_FILE:-}
CKPT=$DATA/refe_runs/p6_paper_faithful/model_final.pt
CKPT_SHA=968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1
CARD=$PROOT/$PCARD
NTL=$PROOT/navtest_ids/navtest_logs.json;  NTL_SHA=79daa3c2ba3fbf1c6d45d0e3b630f43bb26a073160a34ca5f9c98553c0bf43e0
NTT=$PROOT/navtest_ids/navtest_tokens.txt.gz; NTT_SHA=e61613ce129bb67dc654a5a575de2889a22fcfcbc64e4ec389a38295b8df9c7a
DCARD=$ROOT/$DCARD_NAME
export TANITAD_GATE_KEY_FILE=${TANITAD_GATE_KEY_FILE:-$ROOT/launch_gate.key}      # the HMAC key lives on the volume, OUTSIDE the shipped tree and anything the egress pushes
mkdir -p $ROOT $S
STAGE=start
TREE=""
export DIV_WS=$WS SCORER1P6_WS=$WS
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: stage=$STAGE: $*"; exit 1; }
stop_after() { if [ "${DIV_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (DIV_STOP_AFTER)"; exit 0; fi; }
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
  local nv="${DIV_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }        # READ-WRITE fd: an exclusive flock on a read-only fd is EBADF on nfs4
others_hold() {
  local c
  for c in refe_p2 refe_p2rep rls1 s1aux refe_p6 refe_p7 refe_scorer1p6 refe_s1z refe_s1zsm refe_rep refe_dose; do
    if lock_held "$DATA/$c"; then echo "the $c chain holds $DATA/$c/chain.lock"; return 2; fi
  done
  return 0
}
wait_clear() {        # BLOCK until no chain holds the pod (P7 is PAUSED for this run: its lock must be free) and the GPU was idle on DIV_IDLE_POLLS consecutive polls
  local poll="${DIV_POLL_S:-30}" need="${DIV_IDLE_POLLS:-3}" max="${DIV_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why=""; rc=0
    why="$(others_hold)"; rc=$?
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZDIV_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (DIV_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
arm_argv() { ARGS="$(ARM=$1 MODE=argv DIV_WS=$WS bash $RUN/div_pod.sh)" && [ -n "$ARGS" ] || refuse "div_pod.sh ARM=$1 MODE=argv printed nothing"; }
token_ok() {          # $1 = arm, uses $ARGS.  `launch_gate.py verify` RE-MEASURES the argv, the code tree, the checkpoint sha and the data and checks the HMAC.
  local tok
  tok="$(ls "$ROOT/lg_$1"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY_C" "$RUN/launch_gate.py" verify --profile $PROFILE --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_$1.json" >> "$ROOT/token_verify_$1.log" 2>&1 200>&-
}


# the split of a preflight / run log's data event, PARSED (never grepped):  split_is <log> <half>  -> rc 0 iff the log's data event carries the split block of that half with 0 kept rows off the half, the filter built,
# both halves >= SPLIT_MIN sets and sets == train_sets; prints the counts (the two halves' set and log counts are reported before any update is trained)
split_is() {
  $PY_C - "$1" "$2" "$SPLIT_MIN" <<'PYEOF'
import json, sys
log, half, floor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(ln) for ln in open(log, encoding="utf-8", errors="replace") if ln.startswith('{"event": "data"')]
if len(rows) != 1:
    sys.exit("%s: %d data events, want 1" % (log, len(rows)))
d = rows[0]
b = d.get("split")
if not isinstance(b, dict):
    sys.exit("%s: the data event carries no split block (--split-half was not built)" % log)
bad = []
if b.get("half") != half: bad.append("split.half %r != %d" % (b.get("half"), half))
if b.get("rule") != "int(sha256(log_name utf-8).hexdigest(), 16) % 2": bad.append("split.rule %r" % (b.get("rule"),))
if b.get("filter_built") is not True: bad.append("split.filter_built %r" % (b.get("filter_built"),))
if b.get("rows_off_half") != 0: bad.append("split.rows_off_half %r != 0" % (b.get("rows_off_half"),))
if b.get("sets") != d.get("train_sets"): bad.append("split.sets %r != train_sets %r" % (b.get("sets"), d.get("train_sets")))
hs, hl = b.get("half_sets") or {}, b.get("half_logs") or {}
if sorted(hs) != ["0", "1"] or not all(isinstance(v, int) and v >= floor for v in hs.values()): bad.append("half_sets %r: a half below %d sets is a launch blocker" % (hs, floor))
if b.get("sets") != hs.get(str(half)) or b.get("logs") != hl.get(str(half)): bad.append("kept %r sets / %r logs are not half %d's %r / %r" % (b.get("sets"), b.get("logs"), half, hs.get(str(half)), hl.get(str(half))))
if bad:
    sys.exit("%s: %s" % (log, "; ".join(bad)))
print("half=%d sets=%d logs=%d | half0 sets=%s logs=%s | half1 sets=%s logs=%s | pre_split_sets=%s | heldout_sets=%s" % (half, b["sets"], b["logs"], hs["0"], hl["0"], hs["1"], hl["1"], b.get("pre_split_sets"), d.get("heldout_sets")))
PYEOF
}

# ---- 0. precheck: the lock, the card, the package
exec 200>"$ROOT/chain.lock"
flock -n 200 || { echo "REFUSED: another DIV chain holds $ROOT/chain.lock"; exit 1; }
STAGE=precheck
[ -n "$HFTOK" ] || refuse "DIV_HF_TOKEN_FILE is not set (hf_egress.py reads the token file in place; no default path in source)"
TREE="$(tree_sha)"
log "chain armed (ZZDIV_START $ARMN of $PARENT, half $HALF; div_chain.sh md5 $(md5sum "$0" | cut -c1-12), tree ${TREE:0:12})"
[ -n "$TREE" ] || refuse "no $PKG/SHIP_MANIFEST.json: ship the landed tree with its manifest (launch_gate.py ship-manifest --commit <40-hex>)"
[ -r "$HFTOK" ] && grep -qE 'hf_[A-Za-z0-9]{8,}' "$HFTOK" || refuse "$HFTOK is missing, unreadable or holds no hf_ token"
# the parent's card, by bytes: a card with other bytes is not the frozen experiment this arm builds on
[ -f "$CARD" ] || refuse "$CARD is missing: the parent's frozen $PCARD must be on the pod"
[ "$(sha256sum "$CARD" | cut -c1-64)" = "$CARD_SHA" ] || refuse "$CARD is not the frozen $PCARD (sha256 ${CARD_SHA:0:12}...)"
# the DIV card, by bytes: a card with other bytes is not the frozen experiment this arm runs (ship PREREG_DIV_P6.md into $ROOT; its sha256 is in the .sha256 file beside it)
[ -f "$DCARD" ] || refuse "$DCARD is missing: ship the frozen $DCARD_NAME into $ROOT"
[ "$(sha256sum "$DCARD" | cut -c1-64)" = "$DCARD_SHA" ] || refuse "$DCARD is not the frozen $DCARD_NAME (sha256 ${DCARD_SHA:0:12}...)"
[ -f "$WS/teacher_env.sh" ] || refuse "$WS/teacher_env.sh is missing"
source $WS/teacher_env.sh
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
PY_C="${DRIVERL_EVAL_PYTHON:-}"
[ -n "$PY_C" ] && [ -x "$PY_C" ] || refuse "teacher_env.sh did not set a runnable DRIVERL_EVAL_PYTHON"
HPY=${HF_PY:-$PY_C}
NEED="refe/scorer_finetune.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/pdm_label_complete.py refe/scorer1p6_pod.sh refe/div_pod.sh refe/div_chain.sh refe/test_div_gate.py refe/div_layer.py refe/s1zsm_layer.py \
      refe/test_pair_loss.py refe/test_pair_loss_p6.py refe/test_scorer1_gate.py refe/test_scorer1p6_gate.py refe/test_scorer1p6_preflight.py refe/test_scorer1_mutation.py refe/test_scorer1_layer.py refe/scorer1_layer.py refe/test_suite_nesting.py refe/test_launch_gate.py refe/train.py refe/model.py refe/ckpt_io.py refe/planner.py splits/navtrain.yaml SHIP_MANIFEST.json"
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
print("ZZDIV_TREE_OK %d files commit %s" % (len(tree), str(man.get("commit"))[:12]))
PYEOF
log "ZZDIV_PRECHECK_OK ($(tail -n 1 $ROOT/tree_check_$LA.log))"; stop_after precheck

# ---- 1. the inputs, AUDITED (read-only): the parent's label dump, bank, checkpoint and navtest lists
STAGE=inputs
if ! stage_done inputs; then
  for p in "$DUMP/train_sets" "$DUMP/heldout_sets" "$DUMP/logs/heldout_logs.txt" "$DUMP/labels_manifest.json" "$DUMP/labels.DONE" "$DUMP/train_pdm" "$DUMP/heldout_pdm" "$DATA/refe_navtrain/train_grow_cmd_3c_p7" "$DATA/refe_navtrain/train_grow/calib_table.json" "$DATA/navtrain_pixels" "$CKPT" "$NTL" "$NTT"; do
    [ -e "$p" ] || refuse "$p is missing: the parent's inputs must be on the pod (DIV reads C0's label dump, bank, checkpoint and navtest lists)"
  done
  [ "$(sha256sum "$CKPT" | cut -c1-64)" = "$CKPT_SHA" ] || refuse "$CKPT is not P6's model_final.pt (sha256 968a469d...)"
  [ "$(sha256sum "$NTL" | cut -c1-64)" = "$NTL_SHA" ] || refuse "$NTL is not the registered navtest log list"
  [ "$(sha256sum "$NTT" | cut -c1-64)" = "$NTT_SHA" ] || refuse "$NTT is not the registered navtest token list"
  PIN="$($PY_C -c 'import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; p = L.SCORER1P6_DATA_PIN; print("UNSET" if p is None else "%d %d" % (p["train_sets"], p["heldout_sets"]))' "$RUN" 2> /dev/null)"
  case "$PIN" in UNSET|"") refuse "launch_gate.SCORER1P6_DATA_PIN is unset: the exact P6 label counts are not pinned" ;; esac
  NTR=${PIN% *}; NHO=${PIN#* }
  [ "$(cat $DUMP/train_pdm/*.jsonl | wc -l)" -ge "$NTR" ] || refuse "train PDM labels hold $(cat $DUMP/train_pdm/*.jsonl | wc -l) rows, want >= $NTR (the pin)"
  [ "$(cat $DUMP/heldout_pdm/*.jsonl | wc -l)" = "$NHO" ] || refuse "held-out PDM labels hold $(cat $DUMP/heldout_pdm/*.jsonl | wc -l) rows, want $NHO (the pin)"
  stage_mark inputs
fi
log "ZZDIV_INPUTS_OK"; stop_after inputs

# ---- 2. G-EGR (the run must be able to leave the pod), a real disk write test
STAGE=egr
$HPY -c "import huggingface_hub, torch" 2>> $L || refuse "huggingface_hub/torch not importable by $HPY"
if ! stage_done egr; then
  $HPY $RUN/hf_egress.py gate --repo $REPO --out $ROOT/gate --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" > $ROOT/gate_$LA.log 2>&1 200>&-
  grep -q " ZZEGR_PASS " $ROOT/gate_$LA.log || refuse "G-EGR did not pass ($(tail -1 $ROOT/gate_$LA.log | cut -c1-300)); the HF repo $REPO must exist"
  [ -f $ROOT/gate/g_egr_result.json ] || refuse "G-EGR wrote no g_egr_result.json"
  stage_mark egr
fi
log "ZZDIV_EGR_OK"; stop_after egr
STAGE=dd
if ! stage_done dd; then
  DDMB=${DIV_DD_MB:-6144}                 # one arm: model_sft_A/B 2.6 GB + states + staging
  dd if=/dev/zero of=$ROOT/dd_test_$LA.tmp bs=1M count=$DDMB conv=fsync > $ROOT/dd_test_$LA.log 2>&1 200>&- ; ddrc=$?
  rm -f $ROOT/dd_test_$LA.tmp
  [ $ddrc -eq 0 ] || refuse "the ${DDMB} MB write test failed: $(tail -n 1 $ROOT/dd_test_$LA.log)"
  stage_mark dd
fi
log "ZZDIV_DD_OK"; stop_after dd

# ---- 3. the pod: P7 over, nothing else holds it, GPU idle.
STAGE=gpu_wait
wait_clear

# ---- 4. the arm: preflight + the split audit + the NOFILTER mutant -> LAUNCH GATE -> train -> check -> bank
STAGE="preflight_$ARMN"
arm_argv "$ARMN"
if ! stage_done "preflight_$LA"; then
  wait_clear
  rm -f $OUT/preflight.jsonl $OUT/PREFLIGHT_PASS.json
  ARM=$ARMN MODE=preflight bash $RUN/div_pod.sh 200>&-
  grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "$ARMN preflight did not pass ($(tail -n 2 $OUT/preflight.log | tr '\n' ' ' | cut -c1-300))"
  m=$(split_is $OUT/preflight.log $HALF 2>&1) || refuse "the $ARMN preflight's split audit failed: $m"
  log "ZZDIV_SPLIT $ARMN $m"
  # the deliberate regression: the split filter REMOVED on the real argv.  The trainer must REFUSE it (rc 4, `REFUSED: split:`) and must NOT pass the preflight; a mutant that passes means the audit cannot see a missing filter: no training
  ARM=$ARMN MODE=mutate_nofilter bash $RUN/div_pod.sh 200>&-; mrc=$?
  [ "$mrc" = 4 ] && grep -q "^REFUSED: split:" $OUT/mutate_nofilter.log || refuse "the NOFILTER mutant was NOT refused by the trainer's split audit (rc $mrc): the audit cannot see a removed filter, so $ARMN is not trained ($(tail -n 2 $OUT/mutate_nofilter.log | tr '\n' ' ' | cut -c1-200))"
  ! grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/mutate_nofilter.log || refuse "the NOFILTER mutant passed the preflight: no training"
  log "$ARMN NOFILTER mutant REFUSED as required ($(grep -m1 '^REFUSED: split:' $OUT/mutate_nofilter.log | cut -c1-160))"
  stage_mark "preflight_$LA"
fi
log "ZZDIV_PREFLIGHT_OK $ARMN"; stop_after "preflight_$LA"
STAGE="gate_$ARMN"
if ! stage_done "gate_$LA"; then
  rm -rf "$ROOT/lg_$ARMN"; mkdir -p "$ROOT/lg_$ARMN"
  env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY_C" $RUN/launch_gate.py run --profile $PROFILE --pkg "$PKG" --args "$ARGS" --out-dir "$ROOT/lg_$ARMN" \
    --gate-preflight-json $OUT/preflight.jsonl --gate-egr-json $ROOT/gate/g_egr_result.json > $ROOT/launch_gate_$ARMN.log 2>&1 200>&-
  echo "ZZEXIT $?" >> $ROOT/launch_gate_$ARMN.log
  token_ok "$ARMN" || refuse "the launch gate did not mint a PASS token that verifies for $ARMN: $(grep -E 'REASON|FAIL' $ROOT/launch_gate_$ARMN.log | head -n 3 | cut -c1-300 | tr '\n' ' ')"
  stage_mark "gate_$LA"
fi
log "ZZDIV_GATE_OK $ARMN"; stop_after "gate_$LA"
STAGE="train_$ARMN"
if stage_done "trainstart_$LA" && ! stage_done "train_$LA"; then
  k=1; while [ -e "$OUT.killed_$k" ]; do k=$((k + 1)); done
  refuse "a previous attempt of $ARMN STARTED training and did not finish; scorer_finetune.py has no resume. Archive it (mv $OUT $OUT.killed_$k), re-run: the chain restarts the arm at update 0 under a fresh preflight and token"
fi
if ! stage_done "train_$LA"; then
  wait_clear
  token_ok "$ARMN" || refuse "the launch-gate token does not verify for $ARMN: scorer_finetune.py is NOT started ($(tail -n 2 $ROOT/token_verify_$ARMN.log | tr '\n' ' ' | cut -c1-300))"
  stage_mark "trainstart_$LA"
  log "run launch $ARMN (ZZDIV_TRAIN_START)"
  set -m
  ARM=$ARMN MODE=run bash $RUN/div_pod.sh 200>&- &
  TPID=$!
  set +m
  wait $TPID; RRC=$?
  log "RUN EXIT $ARMN $RRC ($(tail -1 $OUT/sft.log))"
  env -u HF_TOKEN "$PY_C" - "$OUT" "$HALF" "$SPLIT_MIN" > "$OUT/run_check.log" 2>&1 200>&- <<'PYEOF'
import json, math, os, sys
out, half, floor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(ln) for ln in open(os.path.join(out, "sft.jsonl"), encoding="utf-8") if ln.startswith("{")]
done = [r for r in rows if r.get("event") == "done"]
start = [r for r in rows if r.get("event") == "start"]
data = [r for r in rows if r.get("event") == "data"]
bad = []
if "ZZSFT_DONE" not in open(os.path.join(out, "sft.log"), encoding="utf-8", errors="replace").read():
    bad.append("no ZZSFT_DONE in sft.log")
# the REAL run's own data event: the split block of this half, nothing off the half, both halves above the floor
b = (data[0].get("split") if len(data) == 1 else None) or {}
hs = b.get("half_sets") or {}
if len(data) != 1 or b.get("half") != half or b.get("rows_off_half") != 0 or b.get("filter_built") is not True or b.get("sets") != data[0].get("train_sets") or sorted(hs) != ["0", "1"] or min(hs.values() or [0]) < floor:
    bad.append("the run's data event split block is %s (want half %d, 0 rows off the half, both halves >= %d sets)" % (b, half, floor))
# 400 updates, or the half's epoch when it is shorter: ceil(sets / batch 8 / accum 8) (the card: a shorter epoch stops the run there, and that is stated)
n = data[0].get("train_sets") if len(data) == 1 else None
want = min(400, int(math.ceil(n / 8 / 8))) if isinstance(n, int) and n > 0 else None
if want is not None and want < 400:
    print("NOTE: the half's epoch is shorter than 400 updates: the run stops at update %d (%d sets), as the card allows" % (want, n))
if len(done) != 1 or done[0].get("updates") != want:
    bad.append("done event %s, want updates %s" % ([d.get("updates") for d in done], want))
if len(start) != 1 or start[0].get("updates") != want or start[0].get("train_sets") != n:
    bad.append("start event %s, want updates %s on %s sets" % ([(s.get("updates"), s.get("train_sets")) for s in start], want, n))
for m in ("model_sft_A.pt", "model_sft_B.pt"):
    if not os.path.isfile(os.path.join(out, m)):
        bad.append("missing " + m)
if bad:
    sys.exit("; ".join(bad))
print("ZZDIV_RUN_OK updates %s half %d sets %s" % (want, half, n))
PYEOF
  RCC=$?
  log "run check $ARMN: $(tail -n 1 $OUT/run_check.log | cut -c1-300)"
  [ $RRC -eq 0 ] && [ $RCC -eq 0 ] || { log "ZZDIV_CHAIN_FAIL $ARMN"; exit 1; }
  # bank the arm off the pod (a verified remote listing, not the push call)
  N0=$(wc -l < $L)
  $HPY $RUN/hf_egress.py push --repo $REPO --prefix "div/$LA" --max-push-bytes 6e9 --tag "arm_div_$LA" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" \
    --files $OUT/model_sft_B.pt $OUT/sft.log $OUT/sft.jsonl $OUT/preflight.log $OUT/mutate_nofilter.log $ROOT/launch_gate_$ARMN.log $OUT/run_check.log >> $ROOT/gate_$LA.log 2>&1 200>&-
  tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_div_$LA " || refuse "the $ARMN egress push did not verify: the arm lives on the pod only"
  # the LFS sha256 of the banked model (what the hub holds) next to the sha256 of the pod's file: both recorded, and they must agree (the token file is read in place, never printed)
  LOCSHA=$(sha256sum $OUT/model_sft_B.pt | cut -c1-64)
  env -u HF_TOKEN "$HPY" - "$REPO" "div/$LA/model_sft_B.pt" "$HFTOK" "$LOCSHA" "$OUT/lfs_sha.json" >> $L 2>&1 200>&- <<'PYEOF' || log "ZZDIV_LFS_SHA_UNVERIFIED $ARMN: the hub's LFS sha256 could not be read or differs; the pod file's sha256 is $LOCSHA (verify by sha256 after the pull)"
import json, re, sys
from huggingface_hub import HfApi
repo, path, tokfile, loc, outp = sys.argv[1:6]
tok = re.search(r"hf_[A-Za-z0-9]+", open(tokfile, encoding="utf-8").read()).group(0)
info = HfApi(token=tok).get_paths_info(repo, [path], expand=True)
lfs = getattr(info[0], "lfs", None)
sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
json.dump({"repo": repo, "path": path, "lfs_sha256": sha, "pod_sha256": loc, "equal": sha == loc}, open(outp, "w"))
print("ZZDIV_LFS_SHA %s %s pod %s equal %s" % (path, sha, loc, sha == loc))
if sha != loc:
    sys.exit("the hub's LFS sha256 %s != the pod file's %s" % (sha, loc))
PYEOF
  stage_mark "train_$LA"
fi
log "ZZDIV_ARM_DONE $ARMN"
log "ZZDIV_CHAIN_DONE $ARMN"
