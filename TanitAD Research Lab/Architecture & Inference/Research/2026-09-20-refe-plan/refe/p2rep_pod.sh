#!/bin/bash
# P2REP: the TRAINING-SEED REPLICATE of P2 (P2-s1) on the A40 pod, AFTER RL-S1 and S1 (PI 2026-10-14: "Replicate p2").  Runbook: eval/P2REP_POD_RUNBOOK.md.  Registration: eval/PREREG_P2_REPLICATE.md
# (+ eval/PREREG_P2.md, eval/PREREG_P2_AMENDMENT_1.md for the recipe).  Derived from refe/p2r_pod.sh, which stays UNTOUCHED.
#
# THE ARM: P2's training argv CHARACTER FOR CHARACTER with exactly TWO changes -- `--out` is this arm's own run dir and `--seed 1` is added (P2 ran the parser default, seed 0).  Same init checkpoint,
# same data, same everything.  The launch-gate profile `refe_p2rep` PINS the whole argv, token for token (`refe_p2` binds neither --seed nor --out); refe/test_p2rep_chain.py diffs it against p2r_pod.sh.
#
# WHAT IT WILL NOT DO: build, carve, relabel, download or write into anything P2 built.  The inputs (the carved on-policy bank, refe_sft2/train_pdm, navtrain_pixels, the init checkpoint, the DINOv3
# weights, the teacher env) are shared with RL-S1 and S1: this chain VERIFIES them (refe/p2rep_checks.py + P2's own pdm_label_complete.py, read-only) and REFUSES if any is wrong.
#   p2r_pod.sh stage      here
#   env                   verify-only: /workspace/teacher_env.sh must exist (never built here)
#   weights               verify-only: sha256 of the ViT-L weights against code/pod_train.sh's pin (never downloaded)
#   restore (HF)          DROPPED: it downloads the init checkpoint and the two banks.  The init checkpoint is verified against p2r_sync.PINS; the banks against P2's own served-count definition
#   recover (HF)          CONVERTED: only THIS arm's run dir (ckpt_last.pt) comes back, from THIS arm's repo, into a scratch work dir (the PDM-label branch of `recover` cannot reach refe_sft2/)
#   carve                 DROPPED (it MOVES files inside the shared bank): replaced by the carve's invariants read-only -- served count 174,912 and zero rows of the 24 held-out logs
#   dbs, relabel          DROPPED (the labels exist); P2's pdm_label_complete.py re-audits them READ-ONLY (--expect 174912).  The navtrain DBs are counted for the log only: train.py never reads them
#   pix                   verify-only: pod_dataprep.sh's `.done_1..32` markers + 32 shard dirs
#   pdm_push              DROPPED (the labels were banked by P2; this chain never pushes them)
#   gate, launch_gate     KEPT, in this arm's own dir; launch-gate profile refe_p2rep (the pinned argv)
#   preflight, train      KEPT; the PASS token is VERIFIED before the preflight and before EVERY launch; the GPU is re-checked before each
#   final push            KEPT, to THIS arm's PRIVATE repo, remote-verified by model_final's sha256
#
# Needs: HF_TOKEN in the environment (write-scoped; read here, never echoed, never an argument, stripped from every child except p2r_sync.py) and the package shipped to /workspace/refe-plan-p2rep
# (a directory of its OWN, like refe-plan-rls1 / refe-plan-s1aux: every chain's launch gate re-measures its own tree).
# Start (detached):  nohup setsid bash /workspace/refe-plan-p2rep/refe/p2rep_pod.sh > /workspace/data/refe_p2rep/stdout.log 2>&1 < /dev/null &
# Start it while RL-S1 and S1 are RUNNING or FINISHED (their chains hold chain.lock from their first line): it WAITS until both locks are free, S1 logged ZZS1AUX_DONE and the GPU was idle on 3 consecutive
# polls; it never starts beside a training job.  A re-run after ANY interruption is safe: finished stages leave /workspace/data/refe_p2rep/stage/<name>.ok, inputs are re-verified by content on every start,
# and train.py --resume continues from ckpt_last.pt.
# P2REP_STOP_AFTER=<precheck|wait|verify|recover|gate|launch_gate> ends the chain after that stage (a dry run).
# P2REP_S1_ABANDONED=1   lets the chain start when S1's chain is not running and never logged ZZS1AUX_DONE (a killed arm, a refusal): only on the Master Mind's word.
# P2REP_P2_PROVENANCE_WAIVED=1  waives ONLY P2's own log markers (ZZP2R_DONE, weights.log, carve.log, pix.log); every content check stays.  Only on the Master Mind's word.
# Knobs: P2REP_POLL_S (30), P2REP_IDLE_POLLS (3), P2REP_WAIT_MAX_S (0 = unbounded), P2REP_DD_MB (8192), P2REP_SYNC_MIN (30), P2REP_HEAVY_MIN (30), P2REP_NVSMI.
# Every child is launched with fd 200 closed (the chain holds its flock on 200).  Markers are opaque ZZP2REP_* tokens; every stop is `ZZP2REP_REFUSED stage=<stage>: <why>` in chain.log.
set -u
WORK=/workspace; PKG=$WORK/refe-plan-p2rep; DATA=$WORK/data
P2=$DATA/refe_p2                                   # P2R's work dir: READ-ONLY here (its markers are checked, never written)
RLS1=$DATA/rls1; S1=$DATA/s1aux                    # their chain.lock / chain.log decide "upstream is over"
R=$DATA/refe_p2rep; S=$R/stage; L=$R/chain.log; WSTOP=$R/sync.stop; LG=$R/lg
RUN=$DATA/refe_runs/p2_joint_pdm_s1                # this arm's OWN run dir (P2's is refe_runs/p2_joint_pdm)
INIT=$DATA/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt
SETS=$DATA/refe_navtrain/onpolicy/sets
PDM=$DATA/refe_sft2/train_pdm; PDMLOG=$DATA/refe_sft2/pdm_train.log
OUT_REPO=Sayood/tanitad-refe-p2-joint-pdm-s1       # PRIVATE; created by the first push.  P2's repo (Sayood/tanitad-refe-p2-joint-pdm) is NEVER written: every p2r_sync.py call below names --repo "$OUT_REPO"
EXPECT_SERVED=174912
# THE TRAINING ARGUMENTS: refe/p2r_pod.sh's ARGS, character for character, with --out -> $RUN and `--seed 1` added before it.  The token is minted over EXACTLY this string (profile refe_p2rep pins it).
ARGS="--backbone vitl16 --targets /workspace/data/refe_navtrain/train_grow --scorer-targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --init-from /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --scorer-mode onpolicy --onpolicy-targets /workspace/data/refe_navtrain/onpolicy/sets --pdm-labels /workspace/data/refe_sft2/train_pdm --pdm-only --scorer-sees-trunk --epochs 3 --epoch-unit scenes --batch 4 --accum 64 --lr 5e-5 --amp bf16 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --seed 1 --out $RUN --resume"
# the launch-gate HMAC key lives on the persistent volume but OUTSIDE the shipped code tree and outside anything p2r_sync pushes (*.log / *.json only)
export TANITAD_GATE_KEY_FILE="$R/launch_gate.key"
mkdir -p "$R" "$S"
STAGE=start
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "ZZP2REP_REFUSED stage=$STAGE: $*"; exit 1; }
stage_done() { [ -f "$S/$1.ok" ]; }
stage_mark() { date -u +%FT%T > "$S/$1.ok"; }
stop_after() { if [ "${P2REP_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (P2REP_STOP_AFTER)"; exit 0; fi; }
# The gate JSON is PARSED, never grepped (p2r_pod.sh): all four gates must be present AND ok.   gate_pass <gate.json>   -> rc 0 iff PASS
gate_pass() {
  $PY_C - "$@" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
g = d.get("gates", {})
bad = [p for p in ("G1", "G2", "G3", "G4") if not any(k.startswith(p + "_") and v.get("ok") is True for k, v in g.items())]
if d.get("verdict") != "PASS" or bad:
    sys.exit("gate not PASS: verdict=%s missing_or_not_ok=%s" % (d.get("verdict"), bad))
PYEOF
}
# rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never pgrep -f: it matches its own command line.
gpu_busy() {
  local nv="${P2REP_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
# rc 0 = nothing upstream holds the pod; rc 2 = a chain still runs (WAIT); rc 1 = it never will be over (REFUSE).  PI order: P2 -> RL-S1 -> S1 -> P2-s1.
# ⛔ a lock probe takes a READ-WRITE fd (`9<>`): an exclusive flock on a read-only fd is EBADF on the pod's nfs4 and would wait here forever (refe/test_flock_rw.py scans this file).
upstream() {
  if [ -e "$P2/chain.lock" ] && ! ( flock -n 9 ) 9<> "$P2/chain.lock"; then echo "the P2 chain still holds $P2/chain.lock"; return 2; fi
  if [ -e "$RLS1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$RLS1/chain.lock"; then echo "the RL-S1 chain still holds $RLS1/chain.lock"; return 2; fi
  if [ -e "$S1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$S1/chain.lock"; then echo "the S1 chain still holds $S1/chain.lock"; return 2; fi
  if ! grep -q "ZZS1AUX_DONE" "$S1/chain.log" 2> /dev/null && [ "${P2REP_S1_ABANDONED:-}" != "1" ]; then
    echo "S1 is neither running nor finished: ZZS1AUX_DONE is not in $S1/chain.log (P2REP_S1_ABANDONED=1 only on the Master Mind's word)"; return 1; fi
  return 0
}
# BLOCK until upstream is over AND the GPU was idle on P2REP_IDLE_POLLS consecutive polls: this chain never adds load to a pod that is training (CLAUDE.md), and a gap between two upstream arms is not "free".
wait_clear() {
  local poll="${P2REP_POLL_S:-30}" need="${P2REP_IDLE_POLLS:-3}" max="${P2REP_WAIT_MAX_S:-0}" idle=0 waited=0 since=0 why rc
  while :; do
    why="$(upstream)"; rc=$?
    [ "$rc" = 1 ] && refuse "$why"
    if [ "$rc" = 0 ] && gpu_busy; then why="a process is on the GPU (or nvidia-smi cannot say)"; rc=2; fi
    if [ "$rc" = 0 ]; then
      idle=$((idle + 1))
      [ "$idle" -ge "$need" ] && return 0
    else
      idle=0
      if [ "$since" = 0 ] || [ "$since" -ge 600 ]; then log "ZZP2REP_WAITING $why (waited ${waited}s)"; since=0; fi
    fi
    [ "$max" -gt 0 ] && [ "$waited" -ge "$max" ] && refuse "waited ${waited}s for the pod (P2REP_WAIT_MAX_S): $why"
    sleep "$poll" 200>&-
    waited=$((waited + poll)); since=$((since + poll))
  done
}
# ⛔ THE LAUNCH-GATE TOKEN (CLAUDE.md, PI 2026-09-26, BINDING): train.py does not start unless this returns 0.  `launch_gate.py verify` RE-MEASURES the argv, the code tree and the data and checks the
# HMAC; it is never a grep of a log.   token_ok  -> rc 0 iff ZZGATE-MATCHZZ
token_ok() {
  tok="$(ls "$LG"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY" "$PKG/refe/launch_gate.py" verify --profile refe_p2rep --token "$tok" --args "$ARGS" --pkg "$PKG" \
    --json "$R/token_verify.json" >> "$R/token_verify.log" 2>&1 200>&-
}
on_exit() {
  rc=$?
  touch "$WSTOP" 2>/dev/null
  if [ -n "${WPID:-}" ]; then
    for _w in $(seq 1 36); do kill -0 "$WPID" 2>/dev/null || break; sleep 5 200>&-; done
    kill -0 "$WPID" 2>/dev/null && kill "$WPID" 2>/dev/null
  fi
  if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ] && [ -n "${HF_TOKEN:-}" ]; then       # one last push, so a REFUSED line reaches HF too
    ( cd "$PKG/refe" && "$PY" p2r_sync.py tick --repo "$OUT_REPO" --run "$RUN" --logs "$R" --chain-log "$L" --outbox "$R/outbox_exit" ) \
      >> "$R/exit_push.log" 2>&1 200>&-
  fi
  exit $rc
}

exec 200>"$R/chain.lock"
flock -n 200 || { echo "ZZP2REP_REFUSED another p2rep_pod.sh holds $R/chain.lock"; exit 1; }
trap on_exit EXIT
log "ZZP2REP_START chain (p2rep_pod.sh md5 $(md5sum "$0" | cut -c1-12), seed 1, run $RUN, repo $OUT_REPO)"

# ---- 0. precheck: cheap refusals FIRST (before the wait, so a mistake costs seconds, not a night)
STAGE=precheck
[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"
for f in refe/train.py refe/p2_gate.py refe/p2r_sync.py refe/p2r_gate_prep.py refe/p2r_carve.py refe/pdm_label_complete.py refe/onpolicy_relabel_pdm.py refe/select_bank_subset.py refe/hf_egress.py \
         refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/p2rep_checks.py refe/p2rep_pod.sh refe/p2r_pod.sh \
         refe/test_launch_gate.py refe/test_p2r_chain.py refe/test_p2rep_chain.py code/pod_train.sh splits/sft_heldout_logs.txt SHIP_MANIFEST.json; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship code/ refe/ splits/ SHIP_MANIFEST.json to $PKG first (a directory of its OWN, NOT the one RL-S1, S1 or P6 run from)"
done
[ -f "$WORK/teacher_env.sh" ] || refuse "$WORK/teacher_env.sh is missing: P2 built the teacher env on this pod; run code/pod_teacher_env.sh first (this chain never builds an env beside a training job)"
source "$WORK/teacher_env.sh"
PY="${DRIVERL_EVAL_PYTHON:-}"; PY_C="$PY"; export PKG PY_C
[ -n "$PY" ] || refuse "teacher_env.sh did not set DRIVERL_EVAL_PYTHON"
export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
# the shipped tree must be the manifest's, file for file (the launch gate re-measures the same tree: a drift is caught HERE, not after the wait)
env -u HF_TOKEN "$PY" p2rep_checks.py tree --pkg "$PKG" > "$R/tree_check.log" 2>&1 200>&-
grep -q "^ZZP2REP_TREE_OK" "$R/tree_check.log" || refuse "the shipped code tree is not its SHIP_MANIFEST.json: $(tail -n 1 "$R/tree_check.log" | cut -c1-400)"
log "ZZP2REP_PRECHECK_OK (tree $(tail -n 1 "$R/tree_check.log" | cut -c1-90))"; stop_after precheck

# ---- watcher, started BEFORE the wait so the waiting and the verification are visible on HF too: logs every P2REP_SYNC_MIN (30), checkpoints every P2REP_HEAVY_MIN (30), to THIS arm's repo; a failed push never stops the run
STAGE=egress
rm -f "$WSTOP"
"$PY" p2r_sync.py watch --repo "$OUT_REPO" --run "$RUN" --logs "$R" --chain-log "$L" --outbox "$R/outbox" --stop-file "$WSTOP" \
  --interval-min "${P2REP_SYNC_MIN:-30}" --heavy-min "${P2REP_HEAVY_MIN:-30}" >> "$R/sync.log" 2>&1 200>&- &
WPID=$!

# ---- 1. WAIT: RL-S1 and S1 are over (locks free, ZZS1AUX_DONE) and the GPU was idle on consecutive polls
STAGE=wait
wait_clear
log "ZZP2REP_WAIT_OK (upstream over, GPU idle)"; stop_after wait

# ---- 2. VERIFY the shared inputs (read-only; the replicate never builds, carves, relabels or downloads them).  Re-done on EVERY start: RL-S1 and S1 ran beside them.
STAGE=verify
WAIVE=""; [ "${P2REP_P2_PROVENANCE_WAIVED:-}" = "1" ] && WAIVE="--waive-p2-logs"
env -u HF_TOKEN "$PY" p2rep_checks.py inputs --data "$DATA" --pkg "$PKG" --p2-dir "$P2" --json "$R/inputs_verified.json" $WAIVE > "$R/inputs_check.log" 2>&1 200>&-
grep -q "^ZZP2REP_INPUTS_OK" "$R/inputs_check.log" || refuse "a shared input is not what P2 built: $(tail -n 1 "$R/inputs_check.log" | cut -c1-600)"
# the PDM labels are AUDITED by P2's own completeness check (p2r_pod.sh stage 4), read-only: it writes only its --json, here
[ -f "$PDMLOG" ] || refuse "$PDMLOG is missing: P2's relabel log (the completeness audit needs it)"
env -u HF_TOKEN "$PY" pdm_label_complete.py --log "$PDMLOG" --out "$PDM" --sets "$SETS" --expect "$EXPECT_SERVED" --json "$R/pdm_complete.json" > "$R/pdm_complete.out" 2>&1 200>&-
grep -q "^ZZPDM_COMPLETE" "$R/pdm_complete.out" || refuse "the PDM labels are not complete for the served bank: $(tail -n 1 "$R/pdm_complete.out" | cut -c1-300)"
inl="$(tail -n 1 "$R/inputs_check.log")"
log "ZZP2REP_INPUTS_OK ${inl#ZZP2REP_INPUTS_OK } | $(grep '^ZZPDM_COMPLETE' "$R/pdm_complete.out" | tail -n 1 | cut -c1-120)"; stop_after verify
# ⛔ never judge the disk with df (a pod volume's quota is invisible to it): WRITE to it.  The checkpoints and snapshots of one run are ~12 GB beyond the inputs.
DDMB="${P2REP_DD_MB:-8192}"
dd if=/dev/zero of="$R/.dd_test" bs=1M count="$DDMB" conv=fsync 2>> $L 200>&-
[ "$(stat -c %s "$R/.dd_test" 2>/dev/null)" = "$((DDMB * 1024 * 1024))" ] || { rm -f "$R/.dd_test"; refuse "dd write test of ${DDMB} MB failed"; }
rm -f "$R/.dd_test"
log "dd write test ${DDMB} MB ok"

# ---- 3. recover THIS arm's run dir from THIS arm's repo (ckpt_last.pt after an interruption on a pod whose run dir was lost).  A scratch work dir: nothing under refe_sft2/ or refe_navtrain/ can be written.
STAGE=recover
"$PY" p2r_sync.py recover --repo "$OUT_REPO" --work "$R/recover_work" --run "$RUN" > "$R/recover.log" 2>&1 200>&-
grep -q "^ZZRECOVER_OK" "$R/recover.log" || refuse "recover failed: $(tail -n 1 "$R/recover.log" | cut -c1-300)"
log "ZZP2REP_RECOVER $(grep '^ZZRECOVER_OK' "$R/recover.log" | tail -n 1 | cut -c1-80)"; stop_after recover

# ---- 4. the P2 gate (G1-G4) on a small bank where G3's mutation is NOT inert, in THIS arm's dir; the JSON is parsed
STAGE=gate
if ! stage_done gate; then
  rm -rf "$R/gate_sets" "$R/gate_pdm" "$R/gate"
  env -u HF_TOKEN "$PY" p2r_gate_prep.py --sets "$SETS" --pdm "$PDM" --out-sets "$R/gate_sets" --out-pdm "$R/gate_pdm" \
    > "$R/gate_prep.log" 2>&1 200>&- || refuse "gate subset build failed: $(tail -n 1 "$R/gate_prep.log" | cut -c1-200)"
  env -u HF_TOKEN OMP_NUM_THREADS=2 "$PY" p2_gate.py --init-from "$INIT" --onpolicy "$R/gate_sets" \
    --pdm-labels "$R/gate_pdm" --out "$R/gate" > "$R/gate.log" 2>&1 200>&-
  echo "ZZEXIT $?" >> "$R/gate.log"
  gate_pass "$R/gate/gate.json" 2>> $L || refuse "the P2 gate did not PASS (gate.log, gate/gate.json)"
  stage_mark gate
fi
log "ZZP2REP_GATE_OK (p2_gate G1-G4 PASS)"; stop_after gate

# ---- 5. THE LAUNCH GATE (profile refe_p2rep: G-HYG G-DVB G-LIVE G-EVAL G-CKPT G-SUITE + p2_gate's G1-G4 as a sub-check), on the REAL argv (P2's + --out + --seed 1, pinned) and the REAL data.  A PASS token or a
#      REFUSAL; the token is then VERIFIED (not grepped).  Route and contract: eval/REFE_LAUNCH_GATE.md.
STAGE=launch_gate
if ! stage_done launch_gate; then
  rm -rf "$LG" "$R/gate_bank"
  env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY" launch_gate.py run --profile refe_p2rep --pkg "$PKG" --args "$ARGS" --out-dir "$LG" \
    --gate-sets "$R/gate_sets" --gate-pdm "$R/gate_pdm" --gate-bank "$R/gate_bank" --p2-gate-json "$R/gate/gate.json" \
    > "$R/launch_gate.log" 2>&1 200>&-
  echo "ZZEXIT $?" >> "$R/launch_gate.log"
  token_ok || refuse "the launch gate did not mint a PASS token that verifies for this argv (launch_gate.log, lg/, token_verify.log)"
  stage_mark launch_gate
fi
token_ok || refuse "the launch-gate token no longer verifies (token_verify.log)"
cp "$(ls "$LG"/PASS_*.json | head -n 1)" "$R/launch_gate_PASS_token.json" 2>/dev/null          # a top-level *.json: the watcher pushes it with the logs
log "ZZP2REP_LAUNCH_GATE_OK (token $(basename "$(ls "$LG"/PASS_*.json | head -n 1)") verified for seed 1, run $RUN)"; stop_after launch_gate

# ---- 6. preflight on the REAL configuration, then the supervised run.  The GPU is re-checked first (the gate's own processes are over; nothing else may be on it), and before every relaunch.
STAGE=train
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
wait_clear
token_ok || refuse "no launch-gate token verifies for this argv: the preflight and the trainer do not start"
env -u HF_TOKEN "$PY" train.py $ARGS --preflight > "$R/preflight.log" 2>&1 200>&-
grep -q "^PREFLIGHT_OK" "$R/preflight.log" || refuse "train.py --preflight did not print PREFLIGHT_OK"
log "ZZP2REP_TRAIN_START (train.py preflight OK; seed 1; launching)"
for att in 1 2 3 4 5 6; do
  if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then break; fi
  wait_clear
  token_ok || refuse "the launch-gate token does not verify for this argv (attempt $att): train.py is NOT started"
  env -u HF_TOKEN "$PY" train.py $ARGS >> "$R/train.log" 2>&1 200>&-
  log "attempt $att exited rc $?"
  sleep 60 200>&-
done
if [ -f $RUN/summary.json ] && grep -q '"done": true' $RUN/summary.json; then log "ZZP2REP_TRAIN_DONE"
else log "ZZP2REP_FAILED (no done=true after the attempts)"; exit 1; fi

# ---- 7. the final push is the hard gate: success is the REMOTE listing (and model_final's sha256), not the push call
STAGE=final
touch "$WSTOP"
for _w in $(seq 1 120); do kill -0 "$WPID" 2> /dev/null || break; sleep 5 200>&-; done
"$PY" p2r_sync.py final --repo "$OUT_REPO" --run "$RUN" --logs "$R" --chain-log "$L" --outbox "$R/outbox_final" > "$R/final.log" 2>&1 200>&-
FRC=$?
fin="$(grep '^ZZP2R_FINAL_OK' "$R/final.log" | tail -n 1)"
if [ "$FRC" -eq 0 ] && [ -n "$fin" ]; then log "ZZP2REP_FINAL_OK ${fin#ZZP2R_FINAL_OK }"; log "ZZP2REP_DONE"; exit 0; fi
log "ZZP2REP_FINAL_FAILED (final.log): the checkpoint is on the pod only until a retry succeeds"
exit 1
