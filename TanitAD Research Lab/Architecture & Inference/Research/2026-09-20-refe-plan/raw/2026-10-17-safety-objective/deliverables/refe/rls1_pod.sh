#!/bin/bash
# RL-S1 on the P2 pod, AFTER P2 (PI order: P2 -> RL-S1 -> P6): frozen-fan dump -> PDM + rl_terms labels -> five scorer-only arms, each behind its own launch-gate
# PASS token -> RL-val lookup reads -> a PRIVATE HF repo.  Runbook: eval/RLS1_POD_RUNBOOK.md.  Spec: eval/PREREG_RL_S1.md (DRAFT) + eval/HANDOFF_RL_S1.md (the gate section's argv).
#
# TL-1 (PI 2026-10-12, "paper-pure, smaller pool"): the pool is 16,000 RL-train + 3,000 RL-val scenes, and THIS CHAIN produces the teacher-lane labels of both sides itself
# (stage 6: refe/rl_tlane_label.py per shard, pipelined with the PDM and rl_terms labellers; stage 6b verifies them EXACT).  Nothing is supplied from outside.
# Needs: HF_TOKEN in the environment (write-scoped; read here, never echoed, never an argument, stripped from every child except the egress ones) and the package
# shipped to /workspace/refe-plan-rls1 (a directory of its OWN: P2's token_ok re-measures /workspace/refe-plan, so adding files there while P2 trains would break it).
# Start (detached):  nohup setsid bash /workspace/refe-plan-rls1/refe/rls1_pod.sh > /workspace/data/rls1/stdout.log 2>&1 < /dev/null &
# Re-running after ANY interruption is safe: finished stages leave /workspace/data/rls1/stage/<name>.ok (an arm interrupted in training restarts at update 0: the trainer has no resume).
# RLS1_STOP_AFTER=<precheck|queue|env|ckpts|partition|plan|fancmp|tlane|dump|labels|join|bank|gate_<ARM>|arm_<ARM>> ends the chain after that stage (a dry run; gate_C0 stops after the first token).
# THE QUEUE (2026-10-14).  This chain never trains beside another chain, and never deadlocks with one.  AHEAD of it: P2, S1 (its chain.lock held), the P2 seed replicate once it is PAST its
# wait (ZZP2REP_WAIT_OK after its last ZZP2REP_START) and P6 once past its wait (ZZP6_WAIT_OK).  While any of them is alive this chain holds ONLY $R/chain.queue (the single-instance guard):
# S1 and the replicate re-check the RL-S1 chain.lock before their next GPU step, so holding it while waiting for them would be a DEADLOCK.  It takes chain.lock the moment nothing is ahead
# (every other chain reads that lock as "RL-S1 holds the pod": a replicate still waiting then waits on it), settles RLS1_SETTLE_S, and needs the GPU idle on RLS1_IDLE_POLLS (3) polls.
# Knobs: RLS1_QUEUE_POLL_S (5), RLS1_IDLE_POLL_S (30), RLS1_IDLE_POLLS (3), RLS1_SETTLE_S (20), RLS1_WAIT_MAX_S (0 = unbounded), RLS1_GPU_WAIT_MAX_S (1800, per arm), RLS1_P2_ABANDONED.
# RL-S1b (eval/rls1b_arms.json, profile refe_rls1b):  RLS1_ARMS="R0b C1 R1b"  -- R0b is read against C0's banked dumps, C1 runs, R1b is read against C1.  A kill of R0b (K3 ...) STOPS the chain after its
# evidence is banked: no C1, no R1b; the Master Mind decides (the generic "a killed arm refuses" rule below).
# RL-S1c (eval/rls1c_arms.json, profile refe_rls1c; raw/2026-10-15-rls1-adam-dose/PREREG_R0C_DRAFT.md):  RLS1_ARMS="R0c"  -- R0 with ONE lever, --epo-ref armA (the EPO term's KL anchor is arm A's detached policy, not the frozen base).
# NOT launched without a PI decision (~3.5 A40-h).  R0c is read against ITS OWN arm A, always (the same-pass control: the K3 / K4 monitor, the kill replay and rlval_vs_A0.json), never against C0 even when C0's dumps are on the pod;
# C0's banked dumps are the SECOND control (rlval_vs_C0.json) and a log line (ZZRLS1_R0C_NO_SECOND_CONTROL) says so when they are not there.  A kill of R0c STOPS the chain (the generic "a killed arm refuses" rule below).
# FAIL CLOSED (K-MON): an arm's kill monitor must print ZZRLS1_MONITOR_READY within RLS1_MON_READY_S (600) and stay alive while its trainer runs; otherwise the chain stops the trainer by its
# explicit process-group id, writes KILLED.json (code K-MON), banks the evidence and refuses.  Knobs: RLS1_MON_READY_S, RLS1_SUP_POLL_S (5), RLS1_STOP_GRACE_S (30).
# The fan-equivalence bounds (stage fancmp) are PINNED in this script, NOT inputs: FANCMP_ATOL 0.5 (catastrophe bound) and FANCMP_MEAN_ATOL 0.025 (runbook D-RLS1-5, DECIDED 2026-10-12).
# RLS1_FANCMP_ATOL / RLS1_FANCMP_MEAN_ATOL, if set, must EQUAL the pinned values or the stage refuses (a stale export cannot loosen the gate).
# RLS1_BANKED_SETS=<dir> overrides where the fan-equivalence gate (stage fancmp) finds the main run's banked step-10,075 sets (default: P2's restored+carved bank, see FANCMP below).
# RLS1_ARMS (refused at the precheck when it names an arm that is not in REGISTERED_ARMS) = "C0 R0 C1 R1 R2" (the default order: a control runs BEFORE its R arm, so K4 compares R with a finished C).  RLS1_ARMS="R0 R1 R2" is the PAIRED design
# (each R run's arm A is its matched control by construction; K4 then reads against that arm A): about 9 h of A40 less.  The choice is the Master Mind's (runbook D-RLS1-1).
# Every child is launched with fd 200 closed (the chain holds chain.lock on 200 once the queue cleared); fd 199 (chain.queue) is deliberately inherited by the trainer, monitor and watcher: a
# survivor of a killed chain keeps the single-instance guard, so a restart cannot start a second chain beside it (the ticker closes it).  Markers are opaque ZZ tokens.
set -u
WORK=/workspace; PKG=$WORK/refe-plan-rls1; DATA=$WORK/data
P2=$DATA/refe_p2                                   # P2R's work dir: its chain.lock and chain.log decide "P2 is over"
S1=$DATA/s1aux; P2REP=$DATA/refe_p2rep; P6D=$DATA/refe_p6      # the chains AHEAD of RL-S1 in the queue: their chain.lock / chain.log (read-only here)
R=$DATA/rls1; S=$R/stage; L=$R/chain.log; STOPF=$R/chain.stop
FINAL=$DATA/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt        # P2's restore put it here (md5-pinned); the frozen fan is ITS 64 proposals
BASE=$R/base/model_sft4_B_u400.pt                                          # the RL-S1 base (PREREG_RL_S1 OPEN-BASE DECIDED: variant A)
OUT_REPO=Sayood/tanitad-refe-rls1                                          # PRIVATE; created by the first push (hf_egress.py push: create_repo exist_ok, private, verified private)
FINAL_SHA256=4246a320be71ad0e315f8f298b7dadab3dbbba2c24f4fc30706e0129f5057e66
TARGETS=$DATA/refe_navtrain/train_grow; IMAGES=$DATA/navtrain_pixels; CALIB=$TARGETS/calib_table.json
PART=$PKG/raw/rl_pool/partition_runs; PNAME=A_no_trained_list_seed20261012_n16000
MANIFEST=$PART/$PNAME.manifest.json; TRAIN_TSV=$PART/$PNAME/rl_train_scenes.tsv; VAL_TSV=$PART/$PNAME/rl_val_scenes.tsv
# the arms this chain knows.  RLS1B-INSERTION-POINT-1 (FILLED 2026-10-14 from eval/rls1b_arms.json): the RL-S1b arms R0b and R1b (R0 / R1 with --epo-lam 0.25), argv in arm_args (-2), the control each reads
# against in the arm loop (-3).  RL-S1b is run as  RLS1_ARMS="R0b C1 R1b"  (Master Mind's order): R0b against C0's banked dumps, then C1, then R1b against C1.  Their gate profile is refe_rls1b (arm_profile).
# R0c (2026-10-15, from eval/rls1c_arms.json): R0 + --epo-ref armA, gate profile refe_rls1c, read against its own arm A (arm loop, "RL-S1c").  Run as  RLS1_ARMS="R0c".
# SAFEOBJ (2026-10-17, from eval/rls1d_arms.json; PREREG_SAFEOBJ.md): R0d / R1d = R0 / R1's argv + the constrained-EPO levers --epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0; R1o = seed 1 with --epo-con off NAMED.
# Gate profile refe_rls1d; each is read against its OWN arm A (the K3 / K4 monitor, the replay, rlval_vs_A0.json), the seed-matched C<n> is the SECOND control (read when its dumps are there, announced when not).
# Run in TWO STEPS, never as the one list "R0d R1d R1o":  (1) RLS1_ARMS="R0d R1d"  (a K3 kill of R0d or R1d STOPS the chain);  (2) the operator runs the reader  python eval/safeobj_read.py --arm-dir $R/arms/R0d
# --arm-dir $R/arms/R1d  (this chain never runs it: a bar-S record is the READER's output, never hand-written, and it can exist only once both arms have finished -- the chain itself rewrites an arm's kill_replay.json,
# rlval_vs_A0.json and scores/, so a record made earlier would be refused as stale);  (3) RLS1_ARMS="R1o".  R1o refuses (before any GPU work) unless R0d and R1d both carry a GENUINE safeobj_S.json reading S = PASS, every input unchanged since the read and a fresh
# read of the directory reproducing the record: rls1_checks.py safeobj-gate.  The one list trains R0d and R1d and then R1o's gate refuses (nothing of R1o is created).  R1o's monitor and replay are --observe-only.
REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"
SETS_TR=$R/sets_train; SETS_VA=$R/sets_val; PDM_TR=$R/pdm_train; PDM_VA=$R/pdm_val; TERMS_TR=$R/terms_train; TERMS_VA=$R/terms_val
TLANE_TR=$R/tlane_train; TLANE_VA=$R/tlane_heldout; VALLOGS=$R/rl_val_logs.txt
NSH="${RLS1_SHARDS:-4}"
# the launch-gate HMAC key lives on the persistent volume but OUTSIDE the shipped code tree and outside anything the egress pushes
export TANITAD_GATE_KEY_FILE="$R/launch_gate.key"
mkdir -p "$R" "$S"
log() { echo "$(date -u +%FT%T) $*" >> $L; }
refuse() { log "REFUSED: $*"; exit 1; }
stage_done() { [ -f "$S/$1.ok" ]; }
stage_mark() { date -u +%FT%T > "$S/$1.ok"; }
stop_after() { if [ "${RLS1_STOP_AFTER:-}" = "$1" ]; then log "stopping after $1 (RLS1_STOP_AFTER)"; exit 0; fi; }

# cores the cgroup grants (MEASURED 2026-10-11 on the P2 pod: cgroup v1, cfs 1105000/100000 = 11 cores while nproc said 96)
cgroup_cores() {
  local root="${RLS1_CG_ROOT:-/sys/fs/cgroup}" q per n
  n="$(nproc)"
  if { read -r q per < "$root/cpu.max"; } 2> /dev/null && [ "$q" != "max" ] && [ "${per:-0}" -gt 0 ]; then
    n=$(( q / per > 0 ? q / per : 1 ))
  elif { read -r q < "$root/cpu/cpu.cfs_quota_us"; read -r per < "$root/cpu/cpu.cfs_period_us"; } 2> /dev/null && [ "${q:-0}" -gt 0 ] && [ "${per:-0}" -gt 0 ]; then
    n=$(( q / per > 0 ? q / per : 1 ))
  fi
  echo "$n"
}
# rc 0 = a process is on the GPU, OR the GPU cannot be asked (what cannot be proven idle is busy).  Never pgrep -f: it matches its own command line.
gpu_busy() {
  local nv="${RLS1_NVSMI:-nvidia-smi}" apps
  apps="$("$nv" --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" || return 0
  [ -n "$(echo "$apps" | tr -d '[:space:]')" ]
}
# P2 must be OVER: its chain holds its flock for its whole life, and it must have logged ZZP2R_DONE (the PI's order, P2 -> RL-S1).  The GPU is NOT judged here any more: a chain that is
# queued behind S1 must be startable while S1 trains; queue_wait judges the GPU (3 idle polls) before this chain's first load.
p2_over() {
  if [ -e "$P2/chain.lock" ] && ! ( flock -n 9 ) 9<> "$P2/chain.lock"; then echo "the P2 chain still holds $P2/chain.lock"; return 1; fi
  if ! grep -q "ZZP2R_DONE" "$P2/chain.log" 2> /dev/null && [ "${RLS1_P2_ABANDONED:-}" != "1" ]; then
    echo "P2 never logged ZZP2R_DONE in $P2/chain.log (RLS1_P2_ABANDONED=1 only on the PI's word)"; return 1; fi
  return 0
}
# the launch-gate profile an arm is judged and its token minted under: refe_rls1b for the RL-S1b arms (C1 too when it runs inside an RL-S1b list), refe_rls1 for the rest.
arm_profile() {
  case "$1" in
    R0b|R1b) echo refe_rls1b ;;
    R0c) echo refe_rls1c ;;
    R0d|R1d|R1o) echo refe_rls1d ;;
    C1) case " ${RLS1_ARMS:-} " in *" R0b "*|*" R1b "*) echo refe_rls1b ;; *) echo refe_rls1 ;; esac ;;
    *) echo refe_rls1 ;;
  esac
}
arm_registered() { case " $REGISTERED_ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# ---- THE QUEUE.  ⛔ a lock probe takes a READ-WRITE fd (`9<>`): an exclusive flock on a read-only fd is EBADF on the pod's nfs4 (refe/test_flock_rw.py scans this file).
lock_held() { [ -e "$1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$1/chain.lock"; }
# rc 0 iff the chain at <dir> is ALIVE (its chain.lock is held) and its CURRENT attempt has logged <ok marker> after its last <start marker> (chain.log is appended across attempts, so a
# marker from an earlier attempt must not count).  Literal substring matches, never regexes.
chain_past() { lock_held "$1" && awk -v s="$2" -v k="$3" 'index($0, s) { ok = 0 } index($0, k) { ok = 1 } END { exit !ok }' "$1/chain.log" 2> /dev/null; }
# rc 0 (and the reason on stdout) iff a chain AHEAD of RL-S1 is alive.  S1 and P2: their lock.  The P2 seed replicate and P6: only once PAST their wait (still waiting, they wait on US).
ahead() {
  if lock_held "$P2"; then echo "the P2 chain holds $P2/chain.lock"; return 0; fi
  if lock_held "$S1"; then echo "the S1 chain holds $S1/chain.lock"; return 0; fi
  if chain_past "$P2REP" "ZZP2REP_START" "ZZP2REP_WAIT_OK"; then echo "the P2 seed replicate is past its wait (ZZP2REP_WAIT_OK) and holds $P2REP/chain.lock"; return 0; fi
  if chain_past "$P6D" " chain start (p6_pod.sh" "ZZP6_WAIT_OK"; then echo "P6 is past its wait (ZZP6_WAIT_OK) and holds $P6D/chain.lock"; return 0; fi
  return 1
}
# BLOCK until nothing is ahead, THEN hold chain.lock (fd 200) for the rest of the life, settle, and see the GPU idle on RLS1_IDLE_POLLS polls.  chain.lock is RELEASED again whenever something
# ahead shows up (a replicate that passed its wait inside the settle window): never wait for an upstream chain while holding the lock they read.
queue_wait() {
  local poll="${RLS1_QUEUE_POLL_S:-5}" gpoll="${RLS1_IDLE_POLL_S:-30}" need="${RLS1_IDLE_POLLS:-3}" settle="${RLS1_SETTLE_S:-20}" max="${RLS1_WAIT_MAX_S:-0}"
  local t0=$SECONDS held=0 idle=0 since=-1 tlock=0 tgpu=-1000000 why=""
  while :; do
    if why="$(ahead)"; then
      if [ "$held" = 1 ]; then exec 200>&-; held=0; log "ZZRLS1_QUEUE_YIELD chain.lock released: $why"; fi
      idle=0
      if [ "$since" -lt 0 ] || [ $((SECONDS - since)) -ge 600 ]; then log "ZZRLS1_QUEUED $why (waited $((SECONDS - t0))s)"; since=$SECONDS; fi
    elif [ "$held" = 0 ]; then
      exec 200>"$R/chain.lock"
      for _i in 1 2 3 4 5 6; do flock -n 200 && break; sleep 1 200>&-; done        # the other chains PROBE this lock (an exclusive try-lock for a few ms): a collision with a probe is not a second RL-S1
      flock -n 200 || refuse "another process holds $R/chain.lock although no chain is ahead in the queue"
      held=1; idle=0; tlock=$SECONDS
      log "ZZRLS1_QUEUE_LOCK nothing is ahead: chain.lock taken, settling ${settle}s before the GPU polls"
    elif [ $((SECONDS - tlock)) -ge "$settle" ] && [ $((SECONDS - tgpu)) -ge "$gpoll" ]; then
      tgpu=$SECONDS
      if gpu_busy; then idle=0; why="a process is on the GPU (or nvidia-smi cannot say)"
      else idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0; fi
    fi
    if [ "$max" -gt 0 ] && [ $((SECONDS - t0)) -ge "$max" ]; then refuse "waited $((SECONDS - t0))s in the queue (RLS1_WAIT_MAX_S): ${why:-the GPU never idled}"; fi
    sleep "$poll" 200>&-
  done
}
# the GPU idle on RLS1_IDLE_POLLS consecutive polls while THIS chain holds chain.lock (before every trainer); rc 1 after RLS1_GPU_WAIT_MAX_S.  nvidia-smi that cannot answer counts as busy.
gpu_idle_wait() {
  local gpoll="${RLS1_IDLE_POLL_S:-30}" need="${RLS1_IDLE_POLLS:-3}" max="${RLS1_GPU_WAIT_MAX_S:-1800}" idle=0 t0=$SECONDS
  while :; do
    if gpu_busy; then idle=0; else idle=$((idle + 1)); [ "$idle" -ge "$need" ] && return 0; fi
    [ $((SECONDS - t0)) -ge "$max" ] && return 1
    sleep "$gpoll" 200>&-
  done
}
# ---- FAIL CLOSED.  rc 0 iff <pid> exists and is not a zombie (kill -0 also answers for a child that has exited and was not waited for yet).
pid_alive() {
  local st
  kill -0 "$1" 2> /dev/null || return 1
  st="$(sed -n 's/^.*) \(.\).*$/\1/p' "/proc/$1/stat" 2> /dev/null)"
  [ "$st" != Z ]
}
# stop the trainer's OWN process group by its explicit id (== the trainer's pid, verified at launch): TERM, a grace period, KILL.  Never by name (pkill -f / pgrep -f match their own command line).
stop_trainer() {
  local pg="$1" i
  case "$pg" in ''|*[!0-9]*) return 1 ;; esac
  { [ "$pg" -gt 1 ] && [ "$pg" != "$$" ]; } || return 1
  kill -TERM -- "-$pg" 2> /dev/null
  for i in $(seq 1 "${RLS1_STOP_GRACE_S:-30}"); do pid_alive "$pg" || break; sleep 1 200>&-; done
  kill -KILL -- "-$pg" 2> /dev/null
  return 0
}
# supervise_arm <arm> <trainer pid = pgid> <monitor pid> <monitor log> <arm dir>: returns when the trainer is over (0) or after a K-MON stop (3).  The kill monitor is the ONLY thing that enforces K2 / K3 / K4 /
# K5 / K6 while the trainer runs; a monitor that is dead (rc 137 on 2026-10-14: SIGKILLed by the cgroup's OOM killer before it printed a line) or never READY means those rules are unenforceable.
# MON_RC carries the monitor's exit code when it was reaped here.  An existing KILLED.json (the monitor's own K-rule) is never overwritten.
supervise_arm() {
  local arm="$1" tp="$2" mp="$3" mlog="$4" ad="$5" ready_s="${RLS1_MON_READY_S:-600}" poll="${RLS1_SUP_POLL_S:-5}" t0=$SECONDS ready=0 why=""
  MON_RC=""
  while pid_alive "$tp"; do
    if [ "$ready" = 0 ] && grep -q "^ZZRLS1_MONITOR_READY" "$mlog" 2> /dev/null; then
      ready=1; log "arm $arm: kill monitor READY ($(grep '^ZZRLS1_MONITOR_READY' "$mlog" | tail -n 1 | cut -c1-60), after $((SECONDS - t0))s)"
    fi
    if ! pid_alive "$mp"; then
      wait "$mp"; MON_RC=$?
      if pid_alive "$tp"; then why="kill rules unenforceable: monitor died rc $MON_RC"; fi
      break
    fi
    if [ "$ready" = 0 ] && [ $((SECONDS - t0)) -ge "$ready_s" ]; then why="kill rules unenforceable: monitor not READY within ${ready_s}s"; break; fi
    sleep "$poll" 200>&-
  done
  [ -n "$why" ] || return 0
  stop_trainer "$tp"
  if pid_alive "$mp"; then kill -KILL "$mp" 2> /dev/null; fi
  [ -f "$ad/KILLED.json" ] || printf '{"arm": "%s", "code": "K-MON", "why": "%s", "at": "%s"}\n' "$arm" "$why" "$(date +%FT%T)" > "$ad/KILLED.json"
  log "ZZRLS1_KILL $arm K-MON $why"
  return 3
}
# THE argv of one arm: the gate section's tokens (eval/HANDOFF_RL_S1.md), in its order, plus the two run-kind additions --eval-every 250 (the prereg's 250-update
# checks) and --dump-scores (the RL-val read's input).  C0/C1 carry NO --epo* flag at all; R0/R1 the core reward; R2 = R0 + --epo-k --epo-l.
arm_args() {
  local arm="$1" seed epo="" ad="$R/arms/$1"
  case "$arm" in
    C0) seed=0 ;;
    C1) seed=1 ;;
    R0) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA" ;;
    R1) seed=1; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA" ;;
    R2) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-k --epo-l" ;;
    # RLS1B-INSERTION-POINT-2 (FILLED): R0 / R1 with --epo-lam 0.25 and nothing else (eval/rls1b_arms.json, token for token; launch_gate.py profile refe_rls1b).  An R2b is NOT here: it needs a PI decision.
    R0b) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-lam 0.25" ;;
    R1b) seed=1; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-lam 0.25" ;;
    # R0c: R0's argv with --epo-ref armA appended and nothing else (eval/rls1c_arms.json, token for token; launch_gate.py profile refe_rls1c, which requires the flag NAMED: its parser default `base` would reproduce R0).
    R0c) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-ref armA" ;;
    # SAFEOBJ (eval/rls1d_arms.json, token for token; launch_gate.py profile refe_rls1d, which requires every --epo-con* flag NAMED: the parser default `off` must never carry a lever)
    R0d) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0" ;;
    R1d) seed=1; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0" ;;
    R1o) seed=1; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-con off" ;;
    *) return 1 ;;
  esac
  echo "--ckpt $BASE --backbone vitl16 --targets $TARGETS --images $IMAGES --calib $CALIB --onpolicy $SETS_TR --heldout $SETS_VA --heldout-logs $VALLOGS --out $ad --pdm-labels-train $PDM_TR --pdm-labels-heldout $PDM_VA --require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 4 --workers 3 --seed $seed --eval-every 250 --dump-scores $ad/scores --lane-head --teacher-lane-train $TLANE_TR --teacher-lane-heldout $TLANE_VA${epo:+ $epo}"
}
# ⛔ THE LAUNCH-GATE TOKEN (CLAUDE.md, PI 2026-09-26, BINDING): scorer_finetune.py does not start unless this returns 0.  `launch_gate.py verify` RE-MEASURES the argv,
# the code tree and the data and checks the HMAC; it is never a grep of a log.   token_ok  -> rc 0 iff ZZGATE-MATCHZZ
token_ok() {
  tok="$(ls "$LG"/PASS_*.json 2> /dev/null | head -n 1)"
  [ -n "$tok" ] || return 1
  env -u HF_TOKEN "$PY" "$PKG/refe/launch_gate.py" verify --profile "$PROFILE" --token "$tok" --args "$ARGS" --pkg "$PKG" \
    --json "$ARMDIR/token_verify.json" >> "$ARMDIR/token_verify.log" 2>&1 200>&-
}
on_exit() {
  rc=$?
  touch "$STOPF" 2> /dev/null
  [ -n "${TICKPID:-}" ] && kill "$TICKPID" 2> /dev/null
  if [ -n "${PY:-}" ] && [ -d "$PKG/refe" ] && [ -n "${HF_TOKEN:-}" ]; then       # one last push, so a REFUSED line reaches HF too
    ( cd "$PKG/refe" && "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix logs --tag chain_exit --files "$L" --chain-log "$R/egress_exit.log" ) \
      >> "$R/exit_push.log" 2>&1 200>&-
  fi
  exit $rc
}

exec 199>"$R/chain.queue"
flock -n 199 || { echo "ZZRLS1_REFUSED another rls1_pod.sh is queued or running ($R/chain.queue is held; a survivor of an earlier chain holds it too)"; exit 1; }
trap on_exit EXIT
rm -f "$STOPF"
log "chain start (rls1_pod.sh md5 $(md5sum "$0" | cut -c1-12))"

# ---- 0. the cheap refusals, before anything is downloaded and BEFORE the queue (a mistake costs seconds, not the hours of a wait behind S1)
why="$(p2_over)" || refuse "$why"
[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"
for ARM in ${RLS1_ARMS:-C0 R0 C1 R1 R2}; do arm_registered "$ARM" || refuse "unknown arm '$ARM' in RLS1_ARMS (registered: $REGISTERED_ARMS)"; done
for f in refe/scorer_finetune.py refe/aux_drivable_head.py refe/rl_fan_dump.py refe/rl_partition.py refe/rl_pool_available.py refe/onpolicy_relabel_pdm.py refe/rl_terms_label.py \
         refe/pdm_label_complete.py refe/rl_reward.py refe/hf_egress.py refe/rls1_checks.py refe/train.py refe/model.py refe/ckpt_io.py \
         refe/launch_gate.py refe/gate_token.py refe/gate_checks.py refe/gate_fixture.py refe/test_launch_gate.py refe/test_rl_epo.py refe/test_rl_reward.py refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_safeobj.py \
         refe/rl_tlane_label.py refe/test_rl_tlane_label.py refe/onpolicy_relabel_teacher_lane.py refe/onpolicy_label.py refe/planner.py refe/scorer_gate.py \
         refe/score_proposals.py refe/route_rank.py refe/map_feature_cache.py refe/query_cache.py refe/navtrain_scenarios.py refe/build_scorer_targets.py code/augment_routes.py \
         eval/rl_val_eval.py eval/proxy_eval.py eval/rl_terms_join_identity.py eval/safeobj_read.py eval/test_safeobj_read.py eval/test_safeobj_gate.py SHIP_MANIFEST.json \
         code/pod_teacher_env.sh code/pod_dataprep.sh splits/navtrain.yaml splits/sft_heldout_logs.txt \
         splits/rl_fancmp_keys.tsv \
         raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000.manifest.json \
         raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000/rl_train_scenes.tsv \
         raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000/rl_val_scenes.tsv; do
  [ -f "$PKG/$f" ] || refuse "$PKG/$f missing: ship code/ refe/ splits/ eval/ raw/rl_pool/ SHIP_MANIFEST.json to $PKG first (a directory of its own, NOT the one P2 ran from)"
done
log "precheck ok (P2 over, the token is set, the package, the arm list)"; stop_after precheck
# ---- 0b. THE QUEUE: wait for every chain ahead of us, then hold chain.lock and see the GPU idle.  Nothing below adds load (the dd test, the checksums, the labellers, the GPU) before this returns.
queue_wait
log "ZZRLS1_QUEUE_CLEAR nothing is ahead, chain.lock held, GPU idle on ${RLS1_IDLE_POLLS:-3} polls"; stop_after queue
# ⛔ never judge the disk with df (a pod volume's quota is invisible to it): WRITE to it.
DDMB="${RLS1_DD_MB:-16384}"
dd if=/dev/zero of="$DATA/.dd_test" bs=1M count="$DDMB" conv=fsync 2>> $L 200>&-
[ "$(stat -c %s "$DATA/.dd_test" 2>/dev/null)" = "$((DDMB * 1024 * 1024))" ] || { rm -f "$DATA/.dd_test"; refuse "dd write test of ${DDMB} MB failed"; }
rm -f "$DATA/.dd_test"
log "dd write test ok (${DDMB} MB)"

# ---- 1. env (P2's teacher env is REUSED: it holds the NAVSIM stack, the maps and the DriveRL fork)
if ! [ -f "$WORK/teacher_env.sh" ]; then
  env -u HF_TOKEN bash "$PKG/code/pod_teacher_env.sh" > "$R/teacher_env.log" 2>&1 200>&-
  grep -q "^ZZOK teacher env ready" "$R/teacher_env.log" || refuse "pod_teacher_env.sh did not finish (teacher_env.log)"
fi
source "$WORK/teacher_env.sh"
PY="$DRIVERL_EVAL_PYTHON"; export PKG
export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1
[ -d "$REFE_BACKBONE_ROOT" ] || refuse "$REFE_BACKBONE_ROOT is missing: P2's weights stage did not run on this pod"
cd "$PKG/refe" || refuse "cannot cd $PKG/refe"
CORES="$(cgroup_cores)"
LW="${RLS1_LABEL_WORKERS:-$(( CORES - 1 < 1 ? 1 : (CORES - 1 > 16 ? 16 : CORES - 1) ))}"    # one core stays with the dump that overlaps the labels
# a ticker pushes chain.log every 30 min, so the hours of dump and labels are not invisible
if [ "${RLS1_TICK:-1}" = 1 ]; then
  ( while [ ! -f "$STOPF" ]; do sleep 1800 200>&- 199>&-; [ -f "$STOPF" ] && break
      "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix logs --tag chain_tick --files "$L" --chain-log "$R/egress_tick.log" 200>&- 199>&-; done ) >> "$R/ticker.log" 2>&1 200>&- 199>&- &
  TICKPID=$!
fi
log "env ok (cgroup cores $CORES, label workers $LW)"; stop_after env

# ---- 2. the two checkpoints, verified by size + md5 + sha256 (u400 is on the DEV BOX only until it is pushed to $OUT_REPO base/; model_final came with P2's restore)
env -u HF_TOKEN "$PY" rls1_checks.py verify-file --path "$FINAL" --pin model_final > "$R/ckpt_final.log" 2>&1 200>&-
grep -q "^ZZRLS1_FILE_OK" "$R/ckpt_final.log" || refuse "model_final.pt is not the pinned file: $(tail -n 1 "$R/ckpt_final.log" | cut -c1-250)"
if ! stage_done base; then
  mkdir -p "$R/base"
  if ! env -u HF_TOKEN "$PY" rls1_checks.py verify-file --path "$BASE" --pin u400 > "$R/ckpt_u400.log" 2>&1 200>&-; then
    log "u400 is not on the pod (or does not verify): fetching $OUT_REPO base/model_sft4_B_u400.pt"
    rm -f "$BASE"
    "$PY" -c "import sys; from huggingface_hub import hf_hub_download; hf_hub_download(sys.argv[1], 'base/model_sft4_B_u400.pt', local_dir=sys.argv[2])" \
      "$OUT_REPO" "$R" > "$R/ckpt_u400_fetch.log" 2>&1 200>&-
    env -u HF_TOKEN "$PY" rls1_checks.py verify-file --path "$BASE" --pin u400 > "$R/ckpt_u400.log" 2>&1 200>&-
  fi
  grep -q "^ZZRLS1_FILE_OK" "$R/ckpt_u400.log" || refuse "u400 (model_sft4_B_u400.pt) is not the pinned file: $(tail -n 1 "$R/ckpt_u400.log" | cut -c1-250); push it from the dev box to $OUT_REPO base/ (runbook section 1)"
  stage_mark base
fi
log "checkpoints verified (model_final + u400)"; stop_after ckpts

# ---- 3. the frozen partition: manifest, both TSVs, log-disjointness; the RL-val log list is the trainer's --heldout-logs
env -u HF_TOKEN "$PY" rls1_checks.py verify-partition --manifest "$MANIFEST" --train "$TRAIN_TSV" --val "$VAL_TSV" > "$R/partition_verify.log" 2>&1 200>&-
grep -q "^ZZRLS1_PARTITION_OK" "$R/partition_verify.log" || refuse "the partition is not the frozen one: $(tail -n 1 "$R/partition_verify.log" | cut -c1-250)"
env -u HF_TOKEN "$PY" rls1_checks.py val-logs --val "$VAL_TSV" --out "$VALLOGS" > "$R/val_logs.log" 2>&1 200>&-
grep -q "^ZZRLS1_VALLOGS_OK" "$R/val_logs.log" || refuse "the RL-val log list was not written (val_logs.log)"
log "partition verified ($(tail -n 1 "$R/partition_verify.log" | cut -c1-120))"; stop_after partition

# ---- 3b. (TL-1) the 7th (lane) output's labels are PRODUCED by this chain, not asked for: stage 6 runs refe/rl_tlane_label.py per shard (the SFT-4 recipe: the DriveRL Scorer on the
#          dumped proposals + the banked teacher trajectory, last heading held) and stage 6b verifies them EXACT (the `tlane-exact` mode of rls1_checks.py).  There is NO override of the label
#          directories: a label source this chain did not make cannot reach the trainer.  The labeller is smoke-tested on 3 RL-val sets right after the val dump (stage `tlane`).
NTRAIN=$(( $(wc -l < "$TRAIN_TSV") - 1 ))        # the verified partition's RL-train scenes (16,000): the planned-update check of every arm reads it
log "partition: $NTRAIN RL-train scenes"

# ---- 4. image coverage of the frozen partition on THIS pod (seconds, no GPU).  PREREG_RL_S1 section 8: a shortfall triggers the ONE admissible remedy
#          (rl_pool_available.py + the same rl_partition.py / seed / params), which is LOGGED and then STOPS the chain: a new partition_id must be written into
#          PREREG_RL_S1 and re-pinned in rls1_checks.py BEFORE any RL number exists.
if ! stage_done plan; then
  env -u HF_TOKEN "$PY" rl_fan_dump.py --plan-only --ckpt "$FINAL" --keys "$TRAIN_TSV" "$VAL_TSV" --partition-manifest "$MANIFEST" --targets "$TARGETS" \
    --images "$IMAGES" --calib "$CALIB" --out "$R/plan" > "$R/plan.log" 2>&1 200>&-
  prc=$?
  if [ "$prc" = 3 ]; then
    for f in raw/rl_pool/navtrain_token_table.tsv.gz raw/p2_r2_instrument/baseline/navtest_full_tokens.json; do
      [ -f "$PKG/$f" ] || refuse "image-coverage gap found and the remedy needs $PKG/$f (ship it)"
    done
    env -u HF_TOKEN "$PY" rl_pool_available.py --pool "$PKG/raw/rl_pool/navtrain_token_table.tsv.gz" --unavailable "$R"/plan/unavailable_scenes_*.tsv \
      --out "$R/partition_remedy/pool_available.tsv.gz" > "$R/remedy.log" 2>&1 200>&-
    env -u HF_TOKEN "$PY" rl_partition.py --pool "$R/partition_remedy/pool_available.tsv.gz" --navtest "$PKG/raw/p2_r2_instrument/baseline/navtest_full_tokens.json" \
      --heldout-logs "$PKG/splits/sft_heldout_logs.txt" --n-train 16000 --n-val 3000 --min-val-logs 60 --seed 20261012 --out "$R/partition_remedy" >> "$R/remedy.log" 2>&1 200>&-
    newid="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['partition_id'])" "$R/partition_remedy/rl_partition_manifest.json" 2> /dev/null)"
    log "ZZRLS1_REMEDY old partition_id 9995a39aeeac new ${newid:0:12} ($(cat "$R"/plan/unavailable_scenes_*.tsv | wc -l) rows incl. headers; $R/partition_remedy)"
    "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix remedy --tag partition_remedy --chain-log "$L" --files "$R/plan.log" "$R/remedy.log" \
      "$R/partition_remedy/rl_partition_manifest.json" 200>&-
    refuse "image-coverage gap on this pod: the remedy partition is in $R/partition_remedy (new id ${newid:0:12}); log OLD and NEW partition_id in PREREG_RL_S1 section 8, re-pin rls1_checks.PARTITION, re-ship, re-run"
  fi
  [ "$prc" = 0 ] || refuse "rl_fan_dump.py --plan-only exited $prc (plan.log): $(tail -n 1 "$R/plan.log" | cut -c1-200)"
  grep -q "^ZZRLFAN_DONE 0 0 0$" "$R/plan.log" || refuse "plan-only did not print ZZRLFAN_DONE 0 0 0 (plan.log)"
  stage_mark plan
fi
log "image coverage complete for all 19,000 scenes ($(grep '^ZZRLFAN_PLAN' "$R/plan.log" | tail -n 1))"; stop_after plan

# ---- 4b. THE FAN-EQUIVALENCE GATE (before the DBs, the labellers or ONE of the 19,000 scenes is dumped).  Does THIS pod's dumper (these images, this calibration table, this
#          torch / cv2 / cuDNN) reproduce the fan the main run banked at step 10,075?  Nothing else in the chain asks: every label is computed on whatever was dumped.
#          rl_fan_dump.py dumps a PINNED slice (splits/rl_fancmp_keys.tsv: 512 scenes outside the 24 SFT held-out logs, each with a banked rank-0 step-10,075 set) from model_final
#          and compares every set with the banked one under TWO ABSOLUTE bounds, BOTH PINNED below and BOTH required on EVERY set:
#            FANCMP_ATOL = 0.5       the CATASTROPHE bound: max |delta| over every x / y (m) and yaw (rad) value of the set.  The only bound that reads yaw.
#            FANCMP_MEAN_ATOL = 0.025   the MEAN bound: mean |delta| over the set's x / y values (yaw excluded), strictly greater fails.
#          BIT-IDENTITY IS THE WRONG BAR (MEASURED 2026-10-12, raw/rl_fan/RESULT_IDENTITY.md): the fan moves by up to 0.25 with the batch grouping alone, and dev box vs the A40's
#          banked sets differ by 0.25 on every key.  The max delta alone cannot separate a fault from that floor (round 1, raw/rl_fan/atol_calibration/verdict.json: CHOSEN_ATOL null);
#          the per-set MEAN can (round 2, raw/rl_fan/atol_calibration/round2/PREREG_ROUND2.md, PASS, verdict_round2.json: CHOSEN_MEAN_ATOL 0.025 = 2 x the null floor 0.0122 rounded up
#          to 0.005; weakest decision arm 7.2 x M; 0 of 1,509 null sets above M).  Runbook D-RLS1-5: DECIDED by the Master Mind 2026-10-12.
#          THE BOUNDS ARE NOT INPUTS.  They are constants here, cross-checked against verdict_round2.json by test_rls1_chain.py (check_fancmp: CHOSEN_MEAN_ATOL and
#          catastrophe_atol_kept), so a drifted number fails the suite.  RLS1_FANCMP_ATOL / RLS1_FANCMP_MEAN_ATOL, if exported, must equal them or the stage refuses.  The verdict is the
#          dumper's exit 0 AND both PASS markers, each echoing the bound it applied (a dumper that ran with another number is refused).  `fancmp.ok` records the bounds it passed under: a
#          marker from another pair is discarded and the stage runs again.  PASS is not 'the pod is equivalent': the pod's own floor (this A40 against the A40's banked sets) is
#          UNMEASURED; this run writes it (`delta_quantiles` and `mean_abs_delta_quantiles` in $R/fancmp/rl_fan_dump_fancmp_s0of1.manifest.json and the compare-sets line of fancmp.log):
#          BANK both, and tighten M from it (runbook D-RLS1-5).
#          The banked sets are P2's restored bank AFTER its carve (refe/p2r_pod.sh:22, SETS=$DATA/refe_navtrain/onpolicy/sets; the 24 held-out logs were moved to sets_full/, which is why
#          the slice excludes them).  That the directory is still on THIS pod is UNVERIFIED (never read from the pod): an absent or empty one is a refusal, not a skip.
FANCMP_KEYS=$PKG/splits/rl_fancmp_keys.tsv
FANCMP_KEYS_SHA256=4c613ff027e8984b924723ee2c6989cbed30a4f22002d5674eae0530ea30c55c
FANCMP_ATOL=0.5
FANCMP_MEAN_ATOL=0.025
BANKED="${RLS1_BANKED_SETS:-$DATA/refe_navtrain/onpolicy/sets}"
if stage_done fancmp && ! grep -qx "atol=$FANCMP_ATOL mean_atol=$FANCMP_MEAN_ATOL" "$S/fancmp.ok"; then
  log "fancmp.ok was not written under the pinned bounds (atol $FANCMP_ATOL, mean_atol $FANCMP_MEAN_ATOL): the fan-equivalence gate runs again"
  rm -f "$S/fancmp.ok"
fi
if ! stage_done fancmp; then
  [ -z "${RLS1_FANCMP_ATOL:-}" ] || awk -v a="$RLS1_FANCMP_ATOL" -v p="$FANCMP_ATOL" 'BEGIN { exit !(a ~ /^[0-9]*[.]?[0-9]+$/ && a + 0 == p + 0) }' || refuse "RLS1_FANCMP_ATOL='$RLS1_FANCMP_ATOL' is not the pinned catastrophe bound $FANCMP_ATOL: the fan-equivalence bounds are constants of this script (runbook D-RLS1-5), not inputs; unset it"
  [ -z "${RLS1_FANCMP_MEAN_ATOL:-}" ] || awk -v a="$RLS1_FANCMP_MEAN_ATOL" -v p="$FANCMP_MEAN_ATOL" 'BEGIN { exit !(a ~ /^[0-9]*[.]?[0-9]+$/ && a + 0 == p + 0) }' || refuse "RLS1_FANCMP_MEAN_ATOL='$RLS1_FANCMP_MEAN_ATOL' is not the pinned mean bound $FANCMP_MEAN_ATOL: the fan-equivalence bounds are constants of this script (runbook D-RLS1-5), not inputs; unset it"
  [ "$(sha256sum "$FANCMP_KEYS" | cut -c1-64)" = "$FANCMP_KEYS_SHA256" ] || refuse "$FANCMP_KEYS is not the pinned fan-equivalence slice (sha256 != $FANCMP_KEYS_SHA256): a different slice is a different gate"
  ls "$BANKED"/onpolicy_*.jsonl > /dev/null 2>&1 || refuse "no banked step-10,075 sets at $BANKED/onpolicy_*.jsonl (P2's restored bank; RLS1_BANKED_SETS=<dir> if it lives elsewhere): the fan-equivalence gate has nothing to compare with"
  NK=$(( $(wc -l < "$FANCMP_KEYS") - 1 ))
  rm -rf "$R/fancmp"
  env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" rl_fan_dump.py --ckpt "$FINAL" --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$FANCMP_KEYS" --targets "$TARGETS" \
    --images "$IMAGES" --calib "$CALIB" --out "$R/fancmp" --tag fancmp --batch 4 \
    --compare-sets "$BANKED" --compare-atol "$FANCMP_ATOL" --compare-mean-atol "$FANCMP_MEAN_ATOL" --compare-min-matched "$NK" >> "$R/fancmp.log" 2>&1 200>&-
  frc=$?
  { [ "$frc" = 0 ] && grep -qE "^ZZRLFAN_ATOL_PASS $NK [^ ]+ ${FANCMP_ATOL}\$" "$R/fancmp.log" && grep -qE "^ZZRLFAN_MEAN_ATOL_PASS $NK [^ ]+ ${FANCMP_MEAN_ATOL}\$" "$R/fancmp.log"; } || refuse "the fan-equivalence gate FAILED (rl_fan_dump.py exit $frc, want 0 and ZZRLFAN_ATOL_PASS $NK and ZZRLFAN_MEAN_ATOL_PASS $NK at the pinned bounds $FANCMP_ATOL / $FANCMP_MEAN_ATOL; log $R/fancmp.log, sets in $R/fancmp): $(grep -E '^(ZZRLFAN_ATOL_FAIL|ZZRLFAN_MEAN_ATOL_FAIL|REFUSED)' "$R/fancmp.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  stage_mark fancmp
  echo "atol=$FANCMP_ATOL mean_atol=$FANCMP_MEAN_ATOL" >> "$S/fancmp.ok"
fi
log "ZZRLS1_FANCMP_OK $(grep '^ZZRLFAN_ATOL_PASS' "$R/fancmp.log" | tail -n 1) $(grep '^ZZRLFAN_MEAN_ATOL_PASS' "$R/fancmp.log" | tail -n 1) (atol $FANCMP_ATOL, mean_atol $FANCMP_MEAN_ATOL, slice $FANCMP_KEYS_SHA256, banked $BANKED)"; stop_after fancmp

# ---- 5. the labellers must be ABLE to label before hours of them: NAVSIM's metric-cache processor, simulator and scorer import and build
#          (2026-10-11: navsim was missing from the venv and the relabel failed EVERY set).  The navtrain DBs are only needed here.
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DATA/navtrain_dbs" REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml"
if ! stage_done dbs; then
  env -u HF_TOKEN STAGES=dbs bash "$PKG/code/pod_dataprep.sh" > "$R/dbs.log" 2>&1 200>&-
  grep -q "^ZZOK dbs 1192 files ZZ" "$R/dbs.log" || refuse "navtrain DBs: want 1192 files (dbs.log)"
  stage_mark dbs
fi
env -u HF_TOKEN "$PY" -c "import navsim_pdm_targets as PT; PT._setup(); import rl_terms_label; print('ZZPDM_SETUP_OK')" > "$R/pdm_setup.log" 2>&1 200>&-
grep -q "^ZZPDM_SETUP_OK" "$R/pdm_setup.log" || refuse "the labellers cannot start: $(tail -n 1 "$R/pdm_setup.log" | cut -c1-200)"

# ---- 6. dump (GPU, model_final's frozen fan) and labels (CPU), pipelined per shard: dump shard i+1 while shard i is labelled.  Dump shard i and label shard i hold the
#          same logs (crc32 % n).  PDM labels on EVERY proposal (no --served-only), then rl_terms.  Both labellers resume by (key, ckpt_step).
dump_one() {      # dump_one <side> <i> <n> <tsv> <sets dir>
  local m="dump_$1_s$2of$3"
  stage_done "$m" && return 0
  env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" rl_fan_dump.py --ckpt "$FINAL" --expect-ckpt-sha256 "$FINAL_SHA256" --keys "$4" --partition-manifest "$MANIFEST" \
    --targets "$TARGETS" --images "$IMAGES" --calib "$CALIB" --out "$5" --tag fan --shard "$2/$3" >> "$R/$m.log" 2>&1 200>&-
  [ $? -eq 0 ] && grep -q "^ZZRLFAN_DONE [0-9]* [0-9]* 0$" "$R/$m.log" || refuse "dump $1 shard $2/$3 failed or has missing scenes ($m.log)"
  stage_mark "$m"
}
label_one() {     # label_one <side> <i> <n> <sets> <pdm> <terms> <tlane> <tsv>     (runs in a background subshell: it cannot exit the chain, the markers are the verdict)
  local m="label_$1_s$2of$3" f="onpolicy_rlfan_fan_s$2of$3.jsonl"
  stage_done "$m" && return 0
  ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
    exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" onpolicy_relabel_pdm.py --sets "$4" --out "$5" --files "$f" --workers "$LW" ) >> "$R/$m.pdm.log" 2>&1 200>&-
  echo "ZZPDM_EXIT $?" >> "$R/$m.pdm.log"
  env -u HF_TOKEN "$PY" pdm_label_complete.py --log "$R/$m.pdm.log" --out "$5" --sets "$4" --files "$f" --all-versions --json "$R/$m.pdm_complete.json" \
    > "$R/$m.pdm_complete.out" 2>&1 200>&-
  grep -q "^ZZPDM_COMPLETE" "$R/$m.pdm_complete.out" || { log "REFUSED: PDM labels of $m incomplete: $(tail -n 1 "$R/$m.pdm_complete.out" | cut -c1-200)"; return 1; }
  ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
    exec env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 19 "$PY" rl_terms_label.py --sets "$4" --out "$6" --files "$f" --workers "$LW" ) >> "$R/$m.terms.log" 2>&1 200>&-
  echo "ZZTERMS_EXIT $?" >> "$R/$m.terms.log"
  grep -q "^ZZRELABEL_RLT_DONE [0-9]* 0 0$" "$R/$m.terms.log" || { log "REFUSED: rl_terms of $m incomplete: $(tail -n 2 "$R/$m.terms.log" | tr '\n' ' ' | cut -c1-200)"; return 1; }
  # TL-1: the teacher-lane labels of this shard (the long pole: about 10.2 s per set, MEASURED).  REFE_SIM_HZ=10 as op_env.sh / pod_dataprep.sh (the bank's own rate).
  ( echo 1000 > /proc/self/oom_score_adj 2> /dev/null
    exec env -u HF_TOKEN OMP_NUM_THREADS=1 REFE_SIM_HZ=10 nice -n 19 "$PY" rl_tlane_label.py --sets "$4" --bank "$TARGETS" --out "$7" --keys "$8" --shard "$2/$3" \
      --repair-last-heading --workers "$LW" ) >> "$R/$m.tlane.log" 2>&1 200>&-
  echo "ZZTLANE_EXIT $?" >> "$R/$m.tlane.log"
  grep -q "^ZZTLANE_DONE [0-9]* [0-9]* 0$" "$R/$m.tlane.log" || { log "REFUSED: teacher-lane labels of $m incomplete: $(tail -n 2 "$R/$m.tlane.log" | tr '\n' ' ' | cut -c1-200)"; return 1; }
  stage_mark "$m"
}
LABEL_MARKS="label_val_s0of1"
( dump_one val 0 1 "$VAL_TSV" "$SETS_VA" ) 200>&-
stage_done dump_val_s0of1 || refuse "dump of RL-val did not finish"
# ---- 6a. THE TEACHER-LANE LABELLER MUST BE ABLE TO LABEL before hours of dump + labels (2026-10-11: every long stage gets a smoke).  3 RL-val sets through the REAL
#          rl_tlane_label.py (the DriveRL Scorer on THIS pod's DBs and maps), judged by the acceptance the trainer applies to the rows.
if ! stage_done tlane; then
  rm -rf "$R/tlane_smoke"
  env -u HF_TOKEN OMP_NUM_THREADS=1 REFE_SIM_HZ=10 "$PY" rl_tlane_label.py --sets "$SETS_VA" --bank "$TARGETS" --out "$R/tlane_smoke" --keys "$VAL_TSV" \
    --repair-last-heading --workers 1 --limit 3 > "$R/tlane_smoke.log" 2>&1 200>&-
  grep -q "^ZZTLANE_DONE 3 0 0$" "$R/tlane_smoke.log" || refuse "the teacher-lane labeller cannot label 3 RL-val sets (tlane_smoke.log): $(tail -n 1 "$R/tlane_smoke.log" | cut -c1-200)"
  env -u HF_TOKEN "$PY" rls1_checks.py tlane-check --dir "$R/tlane_smoke" --tsv "$VAL_TSV" --min-rows 3 > "$R/tlane_smoke_check.log" 2>&1 200>&-
  grep -q "^ZZRLS1_TLANE_OK" "$R/tlane_smoke_check.log" || refuse "the 3 smoke labels are not rows the trainer accepts (tlane_smoke_check.log): $(tail -n 1 "$R/tlane_smoke_check.log" | cut -c1-200)"
  stage_mark tlane
fi
log "teacher-lane labeller smoke ok (3 RL-val sets, rows accepted by the trainer's reader)"; stop_after tlane
( flock 201; label_one val 0 1 "$SETS_VA" "$PDM_VA" "$TERMS_VA" "$TLANE_VA" "$VAL_TSV" ) 201>"$R/label.lock" 200>&- &
LPIDS="$!"
for i in $(seq 0 $((NSH - 1))); do
  ( dump_one train "$i" "$NSH" "$TRAIN_TSV" "$SETS_TR" ) 200>&-
  stage_done "dump_train_s${i}of${NSH}" || refuse "dump of RL-train shard $i/$NSH did not finish"
  ( flock 201; label_one train "$i" "$NSH" "$SETS_TR" "$PDM_TR" "$TERMS_TR" "$TLANE_TR" "$TRAIN_TSV" ) 201>"$R/label.lock" 200>&- &
  LPIDS="$LPIDS $!"
  LABEL_MARKS="$LABEL_MARKS label_train_s${i}of${NSH}"
done
log "dump ok (val + $NSH train shards); waiting for the last labels"
wait $LPIDS
for m in $LABEL_MARKS; do stage_done "$m" || refuse "labels of $m did not finish ($m.pdm.log, $m.terms.log, chain.log); re-run to resume"; done
log "labels ok (PDM + rl_terms + teacher lane)"; stop_after dump; stop_after labels       # the two are pipelined: either stop name ends the chain here

# ---- 7. THE JOIN GATE (post-label): the dump IS the partition side, PDM labels and rl_terms on 100 % of the sets (C and R must train on the same sets), and G-REWARD-ID
#          (g > 0 <=> DAC = 0) with ZERO disagreements.  A number below 100 % is a refusal, never a footnote.
for side in train val; do
  if [ "$side" = train ]; then sd=$SETS_TR; pd=$PDM_TR; td=$TERMS_TR; tl=$TLANE_TR; tsv=$TRAIN_TSV; ex=16000; else sd=$SETS_VA; pd=$PDM_VA; td=$TERMS_VA; tl=$TLANE_VA; tsv=$VAL_TSV; ex=3000; fi
  # TL-1 (6b): the teacher-lane labels of this side are EXACT: one accepted row per dumped set (= per scene), none foreign, none doubled, the count equal to the partition's
  env -u HF_TOKEN "$PY" rls1_checks.py tlane-exact --dir "$tl" --sets "$sd" --tsv "$tsv" --expect "$ex" --json "$R/tlane_$side.json" > "$R/tlane_exact_$side.log" 2>&1 200>&-
  grep -q "^ZZRLS1_TLANE_OK" "$R/tlane_exact_$side.log" || refuse "teacher-lane labels ($side) are not exact: $(tail -n 1 "$R/tlane_exact_$side.log" | cut -c1-300)"
  env -u HF_TOKEN "$PY" rls1_checks.py join-gate --side "$side" --sets "$sd" --pdm "$pd" --terms "$td" --tsv "$tsv" --expect "$ex" --json "$R/join_$side.json" \
    > "$R/join_$side.log" 2>&1 200>&-
  grep -q "^ZZRLS1_JOIN_OK" "$R/join_$side.log" || refuse "join gate ($side): $(tail -n 1 "$R/join_$side.log" | cut -c1-300)"
done
log "join gate PASS (train + val: PDM, rl_terms and teacher-lane labels on 100 %, G-REWARD-ID clean)"; stop_after join

# ---- 8. bank the dump and the labels OFF the pod before any arm trains (3.3 h of GPU + ~63 CPU-h; they must not exist once)
if ! stage_done bank; then
  mkdir -p "$R/outbox"
  for d in sets_train sets_val pdm_train pdm_val terms_train terms_val tlane_train tlane_heldout; do
    tar -C "$R" -czf "$R/outbox/rlfan_$d.tar.gz" "$d" || refuse "tar of $d failed"
  done
  ( cd "$R/outbox" && sha256sum rlfan_*.tar.gz > MANIFEST_rlfan_sha256.txt )
  cp "$R/join_train.json" "$R/join_val.json" "$MANIFEST" "$R/outbox/"
  N0=$(wc -l < $L)           # only THIS push's lines may vouch for it (chain.log is appended across attempts)
  "$PY" hf_egress.py push --repo "$OUT_REPO" --prefix data --tag rlfan_data --chain-log "$L" --egress-log "$R/egress.jsonl" --max-push-bytes 12e9 \
    --files "$R"/outbox/rlfan_*.tar.gz "$R/outbox/MANIFEST_rlfan_sha256.txt" "$R/outbox/join_train.json" "$R/outbox/join_val.json" "$R/outbox/$PNAME.manifest.json" \
    >> "$R/bank_push.log" 2>&1 200>&-
  tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK rlfan_data " || refuse "the dump and labels were not banked to HF (bank_push.log)"
  stage_mark bank
fi
log "dump + labels banked on HF"; stop_after bank

# ---- 9. the arms, one at a time: preflight -> gate (PASS token) -> train (watcher + kill monitor beside it) -> RL-val reads -> egress.
#          K1 (lam_pg = 0 reproduces C) is the preflight's G11 and the G-SUITE's test_rl_epo.py; K2/K6 and K3/K4/K5 are the monitor's.
export OMP_NUM_THREADS=4
for ARM in ${RLS1_ARMS:-C0 R0 C1 R1 R2}; do
  arm_registered "$ARM" || refuse "unknown arm '$ARM'"
  stage_done "arm_$ARM" && { log "arm $ARM already done"; continue; }
  [ -f "$S/arm_${ARM}_killed" ] && stage_done "egress_$ARM" && refuse "arm $ARM was KILLED by a pre-registered rule ($(cat "$S/arm_${ARM}_killed")): the Master Mind decides; delete $S/arm_${ARM}_killed or drop the arm from RLS1_ARMS"
  ARMDIR=$R/arms/$ARM; LG=$R/lg/$ARM; ARGS="$(arm_args "$ARM")" || refuse "no argv for arm $ARM"
  PROFILE="$(arm_profile "$ARM")"
  # SAFEOBJ (PREREG_SAFEOBJ.md section 4): R1o -- the lever-off control -- runs only if BOTH constrained arms read bar S = PASS (the reader writes <arm>/safeobj_S.json); refused here, before any GPU work.
  OBS=""
  if [ "$ARM" = R1o ]; then
    OBS="--observe-only"                  # R1o's K3 is logged (ZZRLS1_K3_OBSERVED), never a kill: a kill at u500 would delete the matched control at the checks where the lever is read; K2 / K6 stay live
    env -u HF_TOKEN "$PY" rls1_checks.py safeobj-gate --dirs "$R/arms/R0d" "$R/arms/R1d" > "$R/safeobj_gate_R1o.log" 2>&1 200>&-
    grep -q "^ZZRLS1_SAFEOBJ_S_PASS" "$R/safeobj_gate_R1o.log" || refuse "R1o runs only if R0d and R1d both read bar S = PASS: $(tail -n 1 "$R/safeobj_gate_R1o.log" | cut -c1-300)"
  fi
  mkdir -p "$ARMDIR" "$LG"
  # TL-1: the pre-registered budget is --max-updates 1000.  On 16,000 scenes --epochs 2 would end at 500 (the trainer never complains), and the reads below name update1000.
  env -u HF_TOKEN "$PY" rls1_checks.py updates-check --args "$ARGS" --n-sets "$NTRAIN" > "$ARMDIR/updates.log" 2>&1 200>&-
  grep -q "^ZZRLS1_UPDATES_OK" "$ARMDIR/updates.log" || refuse "the planned optimizer updates of $ARM are not the pre-registered budget: $(tail -n 1 "$ARMDIR/updates.log" | cut -c1-250)"
  # RLS1B-INSERTION-POINT-3: KIND (A = a control run, B = an R run) and RD (the matched control a B run reads against) of an RL-S1b arm that does not follow the C<n> / R<n> naming.
  case "$ARM" in C*) KIND=A; RD="";; *) KIND=B; RD=C${ARM:1:1};; esac
  [ "$ARM" = R2 ] && RD=C0
  [ "$ARM" = R0c ] && RD=C0        # RL-S1c: C0 is R0c's SECOND control (the 9d read below); named here, not left to ${ARM:1:1}
  case "$ARM" in R0d) RD=C0;; R1d|R1o) RD=C1;; esac        # SAFEOBJ: the seed-matched control is the SECOND control (read when its dumps are there, announced when not); named here, not left to ${ARM:1:1}
  # the control the K4 / K5 monitor and the replay compare R with: the matched C run's arm A when that run finished, else this run's own arm A (paired by construction)
  if [ -z "$RD" ] || [ ! -f "$R/arms/$RD/scores/scores_update1000_A.npz" ]; then REFDIR="$ARMDIR/scores"; else REFDIR="$R/arms/$RD/scores"; fi
  # RL-S1c (PREREG_R0C_DRAFT.md section 2: "the control for every delta is the arm's own A"): R0c's monitor and replay reference is ITS OWN arm A whether or not C0's dumps are on the pod
  [ "$ARM" = R0c ] && REFDIR="$ARMDIR/scores"
  case "$ARM" in R0d|R1d|R1o) REFDIR="$ARMDIR/scores";; esac     # SAFEOBJ (PREREG_SAFEOBJ.md section 9: "K3 and K4 read against the arm's OWN A"): the monitor and replay reference is its own arm A, whether or not C<n>'s dumps are on the pod
  # RL-S1b: an R arm is read against ITS control's banked dumps (R0b -> C0, R1b -> C1), never silently against its own arm A (the fallback above is the original plan's paired design)
  if [ "$PROFILE" = refe_rls1b ] && [ "$KIND" = B ] && [ "$REFDIR" = "$ARMDIR/scores" ]; then
    refuse "RL-S1b arm $ARM is read against $RD, whose dumps $R/arms/$RD/scores/scores_update1000_A.npz are missing: run it first (RLS1_ARMS=\"R0b C1 R1b\") or restore them"
  fi
  # ---- 9a. preflight on the REAL configuration (G1-G11); the gate CONSUMES its preflight.jsonl, so it comes first
  if ! stage_done "pre_$ARM"; then
    rm -f "$ARMDIR/preflight.jsonl" "$ARMDIR/PREFLIGHT_PASS.json"
    env -u HF_TOKEN "$PY" scorer_finetune.py $ARGS --preflight > "$ARMDIR/preflight.log" 2>&1 200>&-
    grep -q "^ZZSFT_PREFLIGHT_PASS" "$ARMDIR/preflight.log" || refuse "scorer_finetune.py --preflight did not pass for $ARM (preflight.log)"
    stage_mark "pre_$ARM"
  fi
  # ---- 9b. the launch gate, for THIS arm: a PASS token or a refusal; the token is then VERIFIED (not grepped)
  if ! stage_done "gate_$ARM"; then
    rm -rf "$LG"; mkdir -p "$LG"
    env -u HF_TOKEN OMP_NUM_THREADS=4 "$PY" launch_gate.py run --profile "$PROFILE" --arm-name "$ARM" --pkg "$PKG" --args "$ARGS" --out-dir "$LG" \
      --gate-preflight-json "$ARMDIR/preflight.jsonl" > "$ARMDIR/launch_gate.log" 2>&1 200>&-
    echo "ZZEXIT $?" >> "$ARMDIR/launch_gate.log"
    token_ok || refuse "the launch gate did not mint a PASS token that verifies for $ARM (launch_gate.log, lg/, token_verify.log)"
    stage_mark "gate_$ARM"
  fi
  token_ok || refuse "the launch-gate token of $ARM no longer verifies (token_verify.log)"
  log "arm $ARM: preflight PASS, launch gate PASS (token verified)"; stop_after "gate_$ARM"
  # ---- 9c. train.  An interrupted attempt is archived, never appended to (the trainer has no resume).
  # (a KILLED arm never re-trains by itself: only the Master Mind removes stage/arm_<ARM>_killed; a deliberate re-run then also clears the arm's read and egress markers below)
  if ! stage_done "train_$ARM" && [ ! -f "$S/arm_${ARM}_killed" ]; then
    for f in sft.jsonl sft.log KILLED.json kill_checks.jsonl monitor.log; do [ -e "$ARMDIR/$f" ] && mv "$ARMDIR/$f" "$ARMDIR/$f.attempt$(date +%s)"; done
    rm -rf "$ARMDIR/scores" "$ARMDIR/egress_stage"
    rm -f "$S/read_$ARM.ok" "$S/egress_$ARM.ok"
    gpu_idle_wait || refuse "the GPU was not idle on ${RLS1_IDLE_POLLS:-3} polls within ${RLS1_GPU_WAIT_MAX_S:-1800}s before the trainer of $ARM (this chain holds chain.lock: a process outside the queue is on it)"
    token_ok || refuse "the launch-gate token does not verify for $ARM: scorer_finetune.py is NOT started"
    set -m
    ( env -u HF_TOKEN OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 "$PY" scorer_finetune.py $ARGS >> "$ARMDIR/sft.log" 2>&1 ) 200>&- &
    TPID=$!
    set +m
    sleep "${RLS1_LAUNCH_SETTLE_S:-2}" 200>&-
    PG="$(ps -o pgid= "$TPID" | tr -d ' ')"
    if [ "$PG" != "$TPID" ]; then kill "$TPID" 2> /dev/null; refuse "the trainer of $ARM is not its own process group (pgid $PG, pid $TPID)"; fi
    # the checkpoint watcher: pushes every settled scorer_states + the logs; two consecutive failed pushes + the next checkpoint HALT the run (PREREG_SFT4R Changed 3)
    "$PY" hf_egress.py watch --repo "$OUT_REPO-$(echo "$ARM" | tr 'A-Z' 'a-z')" --out "$ARMDIR" --stage "$ARMDIR/egress_stage" --halt-pid "$TPID" --chain-log "$L" \
      --egress-log "$ARMDIR/egress.jsonl" --log "$ARMDIR/sft.log" "$ARMDIR/sft.jsonl" "$L" > "$ARMDIR/watch.log" 2>&1 200>&- &
    WPID=$!
    # the kill monitor: K2 (non-finite loss), K6 (hours) on every arm; K3 / K4 (two consecutive 250-update checks) and K5 (R2) on the RL-val lookup of the R arms.  It is SUPERVISED below
    # (supervise_arm): it must print ZZRLS1_MONITOR_READY and stay alive while the trainer runs, or the trainer is stopped and the arm is K-MON.
    env -u HF_TOKEN OMP_NUM_THREADS=1 nice -n 10 "$PY" rls1_checks.py monitor --arm "$ARM" --out "$ARMDIR" --refdir "$REFDIR" --pdm "$PDM_VA" --terms "$TERMS_VA" \
      --sets "$SETS_VA" --tsv "$VAL_TSV" --pid "$TPID" ${OBS:-} > "$ARMDIR/monitor.log" 2>&1 200>&- &
    MPID=$!
    log "arm $ARM: trainer pid/pgid $TPID, watcher $WPID, monitor $MPID (control: $REFDIR)"
    supervise_arm "$ARM" "$TPID" "$MPID" "$ARMDIR/monitor.log" "$ARMDIR"
    wait "$TPID"; TRC=$?
    wait "$WPID"; WRC=$?
    if [ -n "${MON_RC:-}" ]; then MRC=$MON_RC; else wait "$MPID"; MRC=$?; fi
    log "arm $ARM: trainer rc $TRC, watcher rc $WRC, monitor rc $MRC"
    if [ -f "$ARMDIR/KILLED.json" ]; then
      echo "$(tr -d '\n' < "$ARMDIR/KILLED.json" | cut -c1-200)" > "$S/arm_${ARM}_killed"
      log "ZZRLS1_ARM_KILLED $ARM $(cat "$S/arm_${ARM}_killed")"
    elif [ "$TRC" -ne 0 ] || ! grep -q "^ZZSFT_DONE" "$ARMDIR/sft.log"; then
      refuse "arm $ARM: the trainer did not finish (rc $TRC; tail of sft.log: $(tail -n 1 "$ARMDIR/sft.log" | cut -c1-200))"
    else
      stage_mark "train_$ARM"
    fi
  fi
  # ---- 9d. the RL-val lookup reads + the retroactive kill replay (the monitor may have errored or missed a tag).  C arm: A0 vs C.  R arm: A0 vs its own arm A vs R,
  #          then R against its MATCHED control run when that run exists.
  if stage_done "train_$ARM" && ! stage_done "read_$ARM"; then
    env -u HF_TOKEN OMP_NUM_THREADS=1 "$PY" rls1_checks.py replay --arm "$ARM" --out "$ARMDIR" --refdir "$REFDIR" --pdm "$PDM_VA" --terms "$TERMS_VA" \
      --sets "$SETS_VA" --tsv "$VAL_TSV" ${OBS:-} > "$ARMDIR/replay.log" 2>&1 200>&-
    if grep -q "^ZZRLS1_REPLAY_KILL" "$ARMDIR/replay.log"; then
      echo "retroactive: $(tail -n 1 "$ARMDIR/replay.log" | cut -c1-200)" > "$S/arm_${ARM}_killed"; log "ZZRLS1_ARM_KILLED $ARM $(cat "$S/arm_${ARM}_killed")"
    else
      grep -q "^ZZRLS1_REPLAY_CLEAN" "$ARMDIR/replay.log" || refuse "the kill replay of $ARM did not run (replay.log)"
    fi
    RV="$PY $PKG/eval/rl_val_eval.py --pdm $PDM_VA --terms $TERMS_VA --sets $SETS_VA --rl-val-scenes $VAL_TSV --require-full --rule navsim_v1_lane"
    [ "$ARM" = R2 ] && RV="$RV --use-k --use-l"
    SC=$ARMDIR/scores
    if [ "$KIND" = A ]; then
      env -u HF_TOKEN $RV --scores A0=$SC/scores_update1000_base.npz C=$SC/scores_update1000_A.npz --base A0 --out "$ARMDIR/rlval_vs_A0.json" > "$ARMDIR/rlval.log" 2>&1 200>&-
    else
      env -u HF_TOKEN $RV --scores A0=$SC/scores_update1000_base.npz CTRL=$SC/scores_update1000_A.npz R=$SC/scores_update1000_B.npz --base A0 \
        --out "$ARMDIR/rlval_vs_A0.json" > "$ARMDIR/rlval.log" 2>&1 200>&-
      if [ -f "$R/arms/$RD/scores/scores_update1000_A.npz" ]; then
        env -u HF_TOKEN $RV --scores C=$R/arms/$RD/scores/scores_update1000_A.npz R=$SC/scores_update1000_B.npz --base C --out "$ARMDIR/rlval_vs_$RD.json" \
          >> "$ARMDIR/rlval.log" 2>&1 200>&-
      elif [ "$ARM" = R0c ]; then
        log "ZZRLS1_R0C_NO_SECOND_CONTROL arm R0c: C0's banked dumps $R/arms/$RD/scores/scores_update1000_A.npz are not on the pod, so rlval_vs_$RD.json was NOT produced: the read against its own arm A (rlval_vs_A0.json) stands alone; restore C0's scores (HF Sayood/tanitad-refe-rls1-c0) and run rl_val_eval.py for the second control"
      elif [ "$ARM" = R0d ] || [ "$ARM" = R1d ] || [ "$ARM" = R1o ]; then
        log "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL arm $ARM: $RD's banked dumps $R/arms/$RD/scores/scores_update1000_A.npz are not on the pod, so rlval_vs_$RD.json was NOT produced: the read against its own arm A (rlval_vs_A0.json) stands alone"
      fi
    fi
    grep -q "^ZZRLVAL_OK" "$ARMDIR/rlval.log" || refuse "rl_val_eval.py did not finish clean for $ARM (rlval.log)"
    stage_mark "read_$ARM"
  fi
  # ---- 9e. egress: the arm's checkpoint (A for a C arm, B for an R arm), logs, dumps, reads, token.  Success is the REMOTE listing, not the push call.
  if ! stage_done "egress_$ARM"; then
    FILES=""
    for f in "$ARMDIR/model_sft_$KIND.pt" "$ARMDIR"/sft.jsonl "$ARMDIR"/sft.log "$ARMDIR"/preflight.jsonl "$ARMDIR"/PREFLIGHT_PASS.json "$ARMDIR"/launch_gate.log \
             "$ARMDIR"/token_verify.json "$LG"/PASS_*.json "$ARMDIR"/KILLED.json "$ARMDIR"/kill_checks.jsonl "$ARMDIR"/monitor.log "$ARMDIR"/kill_replay.json "$ARMDIR"/rlval_*.json \
             "$ARMDIR"/scorer_states_last.pt "$ARMDIR"/scores/scores_*.npz; do
      [ -f "$f" ] && FILES="$FILES $f"
    done
    # an R run's arm A IS its matched control (same sets, order, steps and trunk forward, EPO never touches it): bank it too, for the full-navtest read of the control
    [ "$KIND" = B ] && [ -f "$ARMDIR/model_sft_A.pt" ] && FILES="$FILES $ARMDIR/model_sft_A.pt"
    N0=$(wc -l < $L)
    "$PY" hf_egress.py push --repo "$OUT_REPO-$(echo "$ARM" | tr 'A-Z' 'a-z')" --prefix final --tag "arm_$ARM" --chain-log "$L" --egress-log "$ARMDIR/egress.jsonl" \
      --max-push-bytes 12e9 --files $FILES >> "$ARMDIR/final_push.log" 2>&1 200>&-
    tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_$ARM " || refuse "the evidence of arm $ARM was not banked to HF (final_push.log): it is on the pod ONLY; do not delete the pod"
    stage_mark "egress_$ARM"
  fi
  [ -f "$S/arm_${ARM}_killed" ] && refuse "arm $ARM was KILLED ($(cat "$S/arm_${ARM}_killed")): its evidence is banked; the Master Mind decides before any further arm"
  stage_mark "arm_$ARM"
  log "ZZRLS1_ARM_DONE $ARM"; stop_after "arm_$ARM"
done
touch "$STOPF"
log "ZZRLS1_DONE arms: ${RLS1_ARMS:-C0 R0 C1 R1 R2}"
exit 0
