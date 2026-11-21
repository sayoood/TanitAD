#!/bin/bash
# SEED-REPLICATE chain (PREREG_SCORER1P6 O5, PREREG_S1Z_P6 Z1/Z5): ONE replicate arm per invocation, from P6's model_final.pt, 400 updates, on the L40 pod.  NOT PLACED ON THE POD by its author; the Master Mind ships it with the landed tree.
#   bash /workspace/refe-plan-rep/refe/rep_chain.sh <C0r2|S1E1|S1Z1|S1ZSM1>        (REP_HF_TOKEN_FILE=<the pod's HF token file> in the environment; nohup setsid ... > /workspace/data/refe_rep/chain_<arm>.out 2>&1 < /dev/null &)
#   C0r2 = C0 with data-order seed 2, S1E1 = S1E with seed 1, S1Z1 = S1Z with seed 1, S1ZSM1 = S1ZSM (PREREG_S1ZSM_P6) with seed 1 (its speed-neutral generator seed 20261107 and its measured speed-neutral lambda_z stay the parent's).  Each differs from its parent in --seed and --out ONLY (rep_pod.sh derives the argv from the parent's pod script; test_rep_gate.py proves it).
# Pattern: refe/s1z_chain.sh / scorer1p6_chain.sh (same fail-closed stages and the SAME wait functions, markers keyed by the SHA-256 of SHIP_MANIFEST.json, REFUSED lines in the arm's log), with the replicate differences:
#  * OWN POD DIRECTORY: package $WS/refe-plan-rep, data $DATA/refe_rep.  gate_token.TREE_ROOTS = (refe, code, splits): shipping this refe/ over refe-plan-p6, -p7, -scorer1p6 or -s1z (a dir that holds a minted token) would VOID that token.
#  * THE PARENT'S INPUTS: the card (by bytes), the navtest lists, the checkpoint and the label dump are the parent's; the parent's measured lambda (S1E: refe_scorer1p6/calibrate_s1e, S1Z: refe_s1z/calibrate_s1z) is READ, never re-measured
#    (a re-measured lambda would move a second lever), so there is NO calibrate stage.  The data pin is launch_gate.SCORER1P6_DATA_PIN.  No lane, no human gate.
#  * NO PAUSING OR RESUMING OF P7 here (refe/rep_pause_p7.sh does that).  The chain waits for NO chain lock held (the parents' included) and the GPU idle on 3 consecutive polls.
#  * ORDER: precheck, inputs, G-EGR, dd, wait, the arm's preflight (G1-G16 / G1-G18 on the real bank), the parent's deliberate regressions (S1E1: nobalance epbalance droploss each failing the preflight and turning G16 RED;
#    S1Z1: noradius notopk droploss each turning G18 RED; S1ZSM1: noband nosnbalance droploss noradius notopk each turning G18 RED; C0r2: none, C0 has none), the LAUNCH GATE (a PASS token minted by launch_gate.py, never hand-written), the run, the run check (400 updates, the pinned set count,
#    S1Z1 also every head's twin share > 0), the HF egress under the prefix rep/<arm>, ZZREP_CHAIN_DONE <arm>.
# Markers: $ROOT/stage_<arm>/<name>.ok = date + tree sha.  REP_STOP_AFTER=<stage> ends the chain after that stage (a dry run).
set -u
ARMN=${1:-}
case "$ARMN" in
  C0r2) PARENT=C0;  PROOT_NAME=refe_scorer1p6; PCARD=PREREG_SCORER1P6.md; CARD_SHA=adb5aafd4fb8abdfd1a53d9cc5ff865a89bd20f9008c701cf5d308dac595abed; PROFILE=refe_rep_p6;     FAM=p6;  PPOD=scorer1p6_pod.sh ;;
  S1E1) PARENT=S1E; PROOT_NAME=refe_scorer1p6; PCARD=PREREG_SCORER1P6.md; CARD_SHA=adb5aafd4fb8abdfd1a53d9cc5ff865a89bd20f9008c701cf5d308dac595abed; PROFILE=refe_rep_p6;     FAM=p6;  PPOD=scorer1p6_pod.sh ;;
  S1Z1) PARENT=S1Z; PROOT_NAME=refe_s1z;      PCARD=PREREG_S1Z_P6.md;     CARD_SHA=8c7b6b697b94ccf1a4781a727ed28c10ed8b11878c33eeb2d016a507b36efa69; PROFILE=refe_rep_s1z_p6; FAM=s1z; PPOD=s1z_pod.sh ;;
  S1ZSM1) PARENT=S1ZSM; PROOT_NAME=refe_s1zsm; PCARD=PREREG_S1ZSM_P6.md; CARD_SHA=ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8; PROFILE=refe_rep_s1zsm_p6; FAM=s1zsm; PPOD=s1zsm_pod.sh ;;
  *) echo "REFUSED: unknown arm '$ARMN' (usage: rep_chain.sh C0r2 | S1E1 | S1Z1 | S1ZSM1)"; exit 2 ;;
esac
LA=$(echo "$ARMN" | tr 'A-Z' 'a-z')
WS=${REP_WS:-/workspace}
DATA=$WS/data
PKG=${REP_PKG:-$WS/refe-plan-rep}
RUN=$PKG/refe
ROOT=$DATA/refe_rep
PROOT=$DATA/$PROOT_NAME
DUMP=$DATA/p6base_labels
OUT=$ROOT/$LA
S=$ROOT/stage_$LA
L=$ROOT/chain_$LA.log
REPO=${REP_HF_REPO:-Sayood/tanitad-refe-scorer1p6}          # an existing repo (G-EGR passes on it); the arm is banked under the prefix rep/<arm>
HFTOK=${REP_HF_TOKEN_FILE:-}
CKPT=$DATA/refe_runs/p6_paper_faithful/model_final.pt
CKPT_SHA=968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1
CARD=$PROOT/$PCARD
NTL=$PROOT/navtest_ids/navtest_logs.json;  NTL_SHA=79daa3c2ba3fbf1c6d45d0e3b630f43bb26a073160a34ca5f9c98553c0bf43e0
NTT=$PROOT/navtest_ids/navtest_tokens.txt.gz; NTT_SHA=e61613ce129bb67dc654a5a575de2889a22fcfcbc64e4ec389a38295b8df9c7a
case "$ARMN" in S1E1) PW_FILE=$PROOT/calibrate_s1e/pair_weight.json ;; S1Z1) PW_FILE=$PROOT/calibrate_s1z/pair_weight.json ;; S1ZSM1) PW_FILE=$PROOT/calibrate_s1zsm/pair_weight.json ;; *) PW_FILE="" ;; esac
case "$FAM" in s1z|s1zsm) PWKEY=twin_w ;; *) PWKEY=pair_w ;; esac
export TANITAD_GATE_KEY_FILE=${TANITAD_GATE_KEY_FILE:-$ROOT/launch_gate.key}      # the HMAC key lives on the volume, OUTSIDE the shipped tree and anything the egress pushes
mkdir -p $ROOT $S
STAGE=start
TREE=""
export REP_WS=$WS SCORER1P6_WS=$WS S1Z_WS=$WS
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: stage=$STAGE: $*"; exit 1; }
stop_after() { if [ "${REP_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (REP_STOP_AFTER)"; exit 0; fi; }
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
  local nv="${REP_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }        # READ-WRITE fd: an exclusive flock on a read-only fd is EBADF on nfs4
others_hold() {
  local c
  for c in refe_p2 refe_p2rep rls1 s1aux refe_p6 refe_p7 refe_scorer1p6 refe_s1z refe_s1zsm; do
    if lock_held "$DATA/$c"; then echo "the $c chain holds $DATA/$c/chain.lock"; return 2; fi
  done
  return 0
}
wait_clear() {        # BLOCK until no chain holds the pod (P7 is PAUSED for this run: its lock must be free) and the GPU was idle on REP_IDLE_POLLS consecutive polls
  local poll="${REP_POLL_S:-30}" need="${REP_IDLE_POLLS:-3}" max="${REP_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why=""; rc=0
    why="$(others_hold)"; rc=$?
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZREP_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (REP_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
arm_argv() { ARGS="$(ARM=$1 MODE=argv REP_WS=$WS bash $RUN/rep_pod.sh)" && [ -n "$ARGS" ] || refuse "rep_pod.sh ARM=$1 MODE=argv printed nothing"; }
token_ok() {          # $1 = arm, uses $ARGS.  `launch_gate.py verify` RE-MEASURES the argv, the code tree, the checkpoint sha and the data and checks the HMAC.
  local tok
  tok="$(ls "$ROOT/lg_$1"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY_C" "$RUN/launch_gate.py" verify --profile $PROFILE --token "$tok" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_$1.json" >> "$ROOT/token_verify_$1.log" 2>&1 200>&-
}

# ---- 0. precheck: the lock, the card, the package
exec 200>"$ROOT/chain.lock"
flock -n 200 || { echo "REFUSED: another replicate chain holds $ROOT/chain.lock"; exit 1; }
STAGE=precheck
[ -n "$HFTOK" ] || refuse "REP_HF_TOKEN_FILE is not set (hf_egress.py reads the token file in place; no default path in source)"
TREE="$(tree_sha)"
log "chain armed (ZZREP_START $ARMN of $PARENT; rep_chain.sh md5 $(md5sum "$0" | cut -c1-12), tree ${TREE:0:12})"
[ -n "$TREE" ] || refuse "no $PKG/SHIP_MANIFEST.json: ship the landed tree with its manifest (launch_gate.py ship-manifest --commit <40-hex>)"
[ -r "$HFTOK" ] && grep -qE 'hf_[A-Za-z0-9]{8,}' "$HFTOK" || refuse "$HFTOK is missing, unreadable or holds no hf_ token"
# the parent's card, by bytes: a card with other bytes is not the frozen experiment this arm replicates
[ -f "$CARD" ] || refuse "$CARD is missing: the parent's frozen $PCARD must be on the pod"
[ "$(sha256sum "$CARD" | cut -c1-64)" = "$CARD_SHA" ] || refuse "$CARD is not the frozen $PCARD (sha256 ${CARD_SHA:0:12}...)"
[ -f "$WS/teacher_env.sh" ] || refuse "$WS/teacher_env.sh is missing"
source $WS/teacher_env.sh
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
PY_C="${DRIVERL_EVAL_PYTHON:-}"
[ -n "$PY_C" ] && [ -x "$PY_C" ] || refuse "teacher_env.sh did not set a runnable DRIVERL_EVAL_PYTHON"
HPY=${HF_PY:-$PY_C}
NEED="refe/scorer_finetune.py refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/hf_egress.py refe/pdm_label_complete.py refe/scorer1p6_pod.sh refe/rep_pod.sh refe/rep_chain.sh refe/test_rep_gate.py \
      refe/test_pair_loss.py refe/test_pair_loss_p6.py refe/test_scorer1_gate.py refe/test_scorer1p6_gate.py refe/test_scorer1p6_preflight.py refe/test_scorer1_mutation.py refe/test_scorer1_layer.py refe/scorer1_layer.py refe/test_suite_nesting.py refe/test_launch_gate.py refe/train.py refe/model.py refe/ckpt_io.py refe/planner.py splits/navtrain.yaml SHIP_MANIFEST.json"
[ "$FAM" = s1z ] || [ "$FAM" = s1zsm ] && NEED="$NEED refe/s1z_pod.sh refe/s1z_layer.py refe/test_s1z_loss.py refe/test_s1z_gate.py refe/test_s1z_preflight.py refe/test_s1z_layer.py"
[ "$FAM" = s1zsm ] && NEED="$NEED refe/s1zsm_pod.sh refe/s1zsm_layer.py refe/test_s1zsm_loss.py refe/test_s1zsm_gate.py refe/test_s1zsm_preflight.py refe/test_s1zsm_layer.py"
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
print("ZZREP_TREE_OK %d files commit %s" % (len(tree), str(man.get("commit"))[:12]))
PYEOF
log "ZZREP_PRECHECK_OK ($(tail -n 1 $ROOT/tree_check_$LA.log))"; stop_after precheck

# ---- 1. the inputs, AUDITED (read-only): the parent's label dump, bank, checkpoint, navtest lists and its measured lambda
STAGE=inputs
if ! stage_done inputs; then
  for p in "$DUMP/train_sets" "$DUMP/heldout_sets" "$DUMP/logs/heldout_logs.txt" "$DUMP/labels_manifest.json" "$DUMP/labels.DONE" "$DUMP/train_pdm" "$DUMP/heldout_pdm" "$DATA/refe_navtrain/train_grow_cmd_3c_p7" "$DATA/refe_navtrain/train_grow/calib_table.json" "$DATA/navtrain_pixels" "$CKPT" "$NTL" "$NTT"; do
    [ -e "$p" ] || refuse "$p is missing: the parent's inputs must be on the pod (the replicate reads the parent's label dump, bank, checkpoint and navtest lists)"
  done
  [ "$(sha256sum "$CKPT" | cut -c1-64)" = "$CKPT_SHA" ] || refuse "$CKPT is not P6's model_final.pt (sha256 968a469d...)"
  [ "$(sha256sum "$NTL" | cut -c1-64)" = "$NTL_SHA" ] || refuse "$NTL is not the registered navtest log list"
  [ "$(sha256sum "$NTT" | cut -c1-64)" = "$NTT_SHA" ] || refuse "$NTT is not the registered navtest token list"
  if [ -n "$PW_FILE" ]; then
    # the parent's measured lambda must exist, finite and positive; it is the ONLY value read from the parent's run dir, and it is not re-measured
    python3 -c 'import json,math,sys; d=json.load(open(sys.argv[1])); w=d[sys.argv[2]]; assert w and math.isfinite(w) and w>0, d; assert sys.argv[3] != "s1zsm" or (d.get("speed_neutral") is True and d.get("sn_dv")==0.15 and d.get("sn_balance") is True and d.get("sn_seed")==20261107 and d.get("lever")=="twin"), d' "$PW_FILE" "$PWKEY" "$FAM" \
      || refuse "$PW_FILE is missing or holds no finite positive lambda: the parent $PARENT has not calibrated, so $ARMN is not a replicate of it"
    log "lambda (the parent's, not re-measured) = $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PW_FILE" "$PWKEY") from $PW_FILE"
  fi
  PIN="$($PY_C -c 'import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; p = L.SCORER1P6_DATA_PIN; print("UNSET" if p is None else "%d %d" % (p["train_sets"], p["heldout_sets"]))' "$RUN" 2> /dev/null)"
  case "$PIN" in UNSET|"") refuse "launch_gate.SCORER1P6_DATA_PIN is unset: the exact P6 label counts are not pinned" ;; esac
  NTR=${PIN% *}; NHO=${PIN#* }
  [ "$(cat $DUMP/train_pdm/*.jsonl | wc -l)" -ge "$NTR" ] || refuse "train PDM labels hold $(cat $DUMP/train_pdm/*.jsonl | wc -l) rows, want >= $NTR (the pin)"
  [ "$(cat $DUMP/heldout_pdm/*.jsonl | wc -l)" = "$NHO" ] || refuse "held-out PDM labels hold $(cat $DUMP/heldout_pdm/*.jsonl | wc -l) rows, want $NHO (the pin)"
  stage_mark inputs
fi
PIN="$($PY_C -c 'import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; p = L.SCORER1P6_DATA_PIN; print("UNSET" if p is None else "%d %d" % (p["train_sets"], p["heldout_sets"]))' "$RUN" 2> /dev/null)"
case "$PIN" in UNSET|"") refuse "launch_gate.SCORER1P6_DATA_PIN is unset: the exact P6 label counts are not pinned" ;; esac
NTR=${PIN% *}; NHO=${PIN#* }
log "ZZREP_INPUTS_OK"; stop_after inputs

# ---- 2. G-EGR (the run must be able to leave the pod), a real disk write test
STAGE=egr
$HPY -c "import huggingface_hub, torch" 2>> $L || refuse "huggingface_hub/torch not importable by $HPY"
if ! stage_done egr; then
  $HPY $RUN/hf_egress.py gate --repo $REPO --out $ROOT/gate --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" > $ROOT/gate_$LA.log 2>&1 200>&-
  grep -q " ZZEGR_PASS " $ROOT/gate_$LA.log || refuse "G-EGR did not pass ($(tail -1 $ROOT/gate_$LA.log | cut -c1-300)); the HF repo $REPO must exist"
  [ -f $ROOT/gate/g_egr_result.json ] || refuse "G-EGR wrote no g_egr_result.json"
  stage_mark egr
fi
log "ZZREP_EGR_OK"; stop_after egr
STAGE=dd
if ! stage_done dd; then
  DDMB=${REP_DD_MB:-6144}                 # one arm: model_sft_A/B 2.6 GB + states + staging
  dd if=/dev/zero of=$ROOT/dd_test_$LA.tmp bs=1M count=$DDMB conv=fsync > $ROOT/dd_test_$LA.log 2>&1 200>&- ; ddrc=$?
  rm -f $ROOT/dd_test_$LA.tmp
  [ $ddrc -eq 0 ] || refuse "the ${DDMB} MB write test failed: $(tail -n 1 $ROOT/dd_test_$LA.log)"
  stage_mark dd
fi
log "ZZREP_DD_OK"; stop_after dd

# ---- 3. the pod: P7 over, nothing else holds it, GPU idle.  NO calibrate stage: the parent's lambda is read.
STAGE=gpu_wait
wait_clear

# ---- 4. the arm: preflight (+ the parent's regressions) -> LAUNCH GATE -> train -> check -> bank
STAGE="preflight_$ARMN"
arm_argv "$ARMN"
if ! stage_done "preflight_$LA"; then
  wait_clear
  rm -f $OUT/preflight.jsonl $OUT/PREFLIGHT_PASS.json
  ARM=$ARMN MODE=preflight bash $RUN/rep_pod.sh 200>&-
  grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/preflight.log || refuse "$ARMN preflight did not pass ($(tail -n 2 $OUT/preflight.log | tr '\n' ' ' | cut -c1-300))"
  MUTS=""; GATEK=""
  [ "$ARMN" = "S1E1" ] && { MUTS="nobalance epbalance droploss"; GATEK=G16; }
  [ "$ARMN" = "S1Z1" ] && { MUTS="noradius notopk droploss"; GATEK=G18; }
  [ "$ARMN" = "S1ZSM1" ] && { MUTS="noband nosnbalance droploss noradius notopk"; GATEK=G18; }
  for MUT in $MUTS; do
    ARM=$ARMN MODE=mutate_$MUT bash $RUN/rep_pod.sh 200>&-
    grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_$MUT.log || refuse "the $MUT mutation did not fail the preflight"
    m=$(gate_is $OUT/mutate_$MUT.log $GATEK ok false 2>&1) || refuse "the $MUT mutation did NOT turn $GATEK RED ($m)"
  done
  [ -z "$MUTS" ] || log "$ARMN mutations ($MUTS) RED as required ($GATEK)"
  stage_mark "preflight_$LA"
fi
log "ZZREP_PREFLIGHT_OK $ARMN"; stop_after "preflight_$LA"
STAGE="gate_$ARMN"
if ! stage_done "gate_$LA"; then
  rm -rf "$ROOT/lg_$ARMN"; mkdir -p "$ROOT/lg_$ARMN"
  env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY_C" $RUN/launch_gate.py run --profile $PROFILE --pkg "$PKG" --args "$ARGS" --out-dir "$ROOT/lg_$ARMN" \
    --gate-preflight-json $OUT/preflight.jsonl --gate-egr-json $ROOT/gate/g_egr_result.json > $ROOT/launch_gate_$ARMN.log 2>&1 200>&-
  echo "ZZEXIT $?" >> $ROOT/launch_gate_$ARMN.log
  token_ok "$ARMN" || refuse "the launch gate did not mint a PASS token that verifies for $ARMN: $(grep -E 'REASON|FAIL' $ROOT/launch_gate_$ARMN.log | head -n 3 | cut -c1-300 | tr '\n' ' ')"
  stage_mark "gate_$LA"
fi
log "ZZREP_GATE_OK $ARMN"; stop_after "gate_$LA"
STAGE="train_$ARMN"
if stage_done "trainstart_$LA" && ! stage_done "train_$LA"; then
  k=1; while [ -e "$OUT.killed_$k" ]; do k=$((k + 1)); done
  refuse "a previous attempt of $ARMN STARTED training and did not finish; scorer_finetune.py has no resume. Archive it (mv $OUT $OUT.killed_$k), re-run: the chain restarts the arm at update 0 under a fresh preflight and token"
fi
if ! stage_done "train_$LA"; then
  wait_clear
  token_ok "$ARMN" || refuse "the launch-gate token does not verify for $ARMN: scorer_finetune.py is NOT started ($(tail -n 2 $ROOT/token_verify_$ARMN.log | tr '\n' ' ' | cut -c1-300))"
  stage_mark "trainstart_$LA"
  log "run launch $ARMN (ZZREP_TRAIN_START)"
  set -m
  ARM=$ARMN MODE=run bash $RUN/rep_pod.sh 200>&- &
  TPID=$!
  set +m
  wait $TPID; RRC=$?
  log "RUN EXIT $ARMN $RRC ($(tail -1 $OUT/sft.log))"
  env -u HF_TOKEN "$PY_C" - "$OUT" "$NTR" "$FAM" > "$OUT/run_check.log" 2>&1 200>&- <<'PYEOF'
import json, os, sys
out, fam = sys.argv[1], sys.argv[3]
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
if fam in ("s1z", "s1zsm"):
    ts = (done[0].get("twin_share") if done else None) or {}
    if sorted(ts) != ["DAC", "NC", "TTC"] or not all(v.get("share_with_pair", 0) > 0 and v.get("sets", 0) > 0 for v in ts.values()):
        bad.append("the done event's twin_share is %s: every head must have a share of sets with a kept pair > 0" % (ts,))
    if fam == "s1zsm":      # PREREG_S1ZSM_P6: the speed-neutral loss ran as built (same check as s1zsm_chain.sh)
        if ts and not all(v.get("kept_F", -1) == v.get("kept_S", -2) and v.get("kept_F", 0) > 0 and v.get("out_of_band", -1) == 0 and v.get("unbalanced_sets", -1) == 0 and v.get("pairs") == 2 * v.get("kept_F", 0) for v in ts.values()):
            bad.append("the done event's twin_share is %s: every head must keep F == S pairs, none outside the speed band" % (ts,))
        if not any(r.get("event") == "data" and (r.get("twin") or {}).get("speed_neutral") is True for r in rows):
            bad.append("the data event does not say speed_neutral true")
if bad:
    sys.exit("; ".join(bad))
print("ZZREP_RUN_OK updates 400")
PYEOF
  RCC=$?
  log "run check $ARMN: $(tail -n 1 $OUT/run_check.log | cut -c1-300)"
  [ $RRC -eq 0 ] && [ $RCC -eq 0 ] || { log "ZZREP_CHAIN_FAIL $ARMN"; exit 1; }
  # bank the arm off the pod (a verified remote listing, not the push call)
  N0=$(wc -l < $L)
  $HPY $RUN/hf_egress.py push --repo $REPO --prefix "rep/$LA" --max-push-bytes 6e9 --tag "arm_rep_$LA" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$HFTOK" \
    --files $OUT/model_sft_B.pt $OUT/sft.log $OUT/sft.jsonl $OUT/preflight.log $ROOT/launch_gate_$ARMN.log $OUT/run_check.log >> $ROOT/gate_$LA.log 2>&1 200>&-
  tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_rep_$LA " || refuse "the $ARMN egress push did not verify: the arm lives on the pod only"
  stage_mark "train_$LA"
fi
log "ZZREP_ARM_DONE $ARMN"
log "ZZREP_CHAIN_DONE $ARMN"
