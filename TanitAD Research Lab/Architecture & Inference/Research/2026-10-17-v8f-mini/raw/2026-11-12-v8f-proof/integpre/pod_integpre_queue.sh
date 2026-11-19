#!/usr/bin/env bash
# pod_integpre_queue.sh : runs ON THE POD (nohup setsid), started by the lead through chain_ip1.sh. Card PREREG_V8F_PROOF_ADD44 (INTEG-PRE). Fails closed. DRY=1 prints the commands only (no check, no wait, no marker, no run).
# Derived from pod_qhead_queue.sh (card ADD42). Per checkpoint of integpre_arms.txt (RLAm and RLFT at bases 0 to 3; queue folders actcons_c0 c1 e2 e3):
#   1. the level-1 states of ALL training clips (extract_states.py --lo 0 --hi 4572, CPU) and the lead read of them (pod_lead_apply.py), the dose-step form of note ADD42_N2;
#   2. the head fit in the form of note ADD42_N2 (qhead_fit_n1.py: fit 3048 / select 762 / check 762 clips, early stopping on the select clips, held-out shift), CPU;
#   3. three rollouts through the banked rule tool: IR0_<BASE> (no rule, mode off), IRL_<BASE> (the rule fed by labels, mode label), IRQ_<BASE> (g_used = min(g_head, q_head), the quantile-head tool).
#   Control c3 after base RLAm_s0: IRH_RLAm_s0 (the head-fed rule, mode head) and IRQI_RLAm_s0 (the identity form q_head = g_head, P 0.5); their dumps must be equal (cmp_dumps: differing=0).
# A head whose check-clip coverage lies outside [0.15, 0.25] (record key h5_check_coverage_in_0.15_0.25 not true) is not read: MISSING_<BASE>.txt is written, the RQ rollout of that checkpoint is skipped, the base counts as missing.
# Before EVERY job (extract, lead read, fit, rollout) a SLOT WAIT = fewer than 3 GPU compute processes (nvidia-smi --query-compute-apps=pid) AND more than 28 GB of the container memory limit free (memory.max minus the
# anon line of memory.stat), both on two probes 45 s apart; poll every 60 s for at most 10 h, then a STOP marker. One GPU job at a time. No process-name test anywhere (no pgrep, no pkill, no ps | grep). No rm, no mv.
# Disk: before the first extract the expected volume of the state dumps is printed; the queue refuses if the pod volume has less than that plus 10 GB free (read with stat -f on the volume, not df).
# A second start is refused: $RUNF/INTEGPRE_STARTED.txt is created once, atomically (noclobber). End markers in $RUNF: INTEGPRE_DONE.txt (count of rollouts with T1_DONE) or INTEGPRE_STOP.txt (reason).
# Tiers: states, lead read and fits T0 (input preparation, no model is trained); rollouts T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
# Env: IP_W (/workspace/v8proof) IP_D ($IP_W/integpre: the pushed tools, arms file, states, heads, run folder) IP_TREE ($IP_W/tree_actcons) IP_PROBE ($IP_W/probe) IP_PACK ($IP_W/qhead_dose/pack/train_pack_all.npz)
#      IP_CACHE ($IP_W/data/cache_pod) IP_R ($IP_W/runs/distrule/t1) IP_PY IP_HI (4572) IP_MIN_FREE_GB (28) IP_MAX_GPU_JOBS (3) IP_PROBE_GAP_S (45) IP_POLL_S (60) IP_MAX_WAIT_S (36000) IP_VOL_MARGIN_GB (10)
#      IP_CG_MAX IP_CG_STAT IP_ARMS IP_FIT_ARGS DRY
set -u
W=${IP_W:-/workspace/v8proof}
D=${IP_D:-$W/integpre}
TREE=${IP_TREE:-$W/tree_actcons}
PROBE=${IP_PROBE:-$W/probe}
PACK=${IP_PACK:-$W/qhead_dose/pack/train_pack_all.npz}
CACHE=${IP_CACHE:-$W/data/cache_pod}
R=${IP_R:-$W/runs/distrule/t1}
PY=${IP_PY:-/workspace/venv/bin/python}
HI=${IP_HI:-4572}
MIN_FREE_GB=${IP_MIN_FREE_GB:-28}
MAX_GPU_JOBS=${IP_MAX_GPU_JOBS:-3}
PROBE_GAP_S=${IP_PROBE_GAP_S:-45}
POLL_S=${IP_POLL_S:-60}
MAX_WAIT_S=${IP_MAX_WAIT_S:-36000}
VOL_MARGIN_GB=${IP_VOL_MARGIN_GB:-10}
CG_MAX=${IP_CG_MAX:-/sys/fs/cgroup/memory.max}
CG_STAT=${IP_CG_STAT:-/sys/fs/cgroup/memory.stat}
FITARGS=${IP_FIT_ARGS:-}
DRY=${DRY:-0}
ARMS=${IP_ARMS:-$D/integpre_arms.txt}
RUNF=$D/run
ST=$D/states
HEADS=$D/heads
LOG=$RUNF/integpre_queue.log
RUN_PLAIN=$D/pod_ruleattr_run.sh      # banked run script, unchanged (modes off, label, head)
RUN_Q=$D/pod_integpre_run.sh          # derived: the argv line gets --head-p-min (the rule tool of the quantile head)
CTRL_BASE=RLAm_s0
CTRL_P=0.5
REF_STATE_BYTES=594171496             # MEASURED 2026-11-19: qhead_dose/states/LSD_s0_dose_train.npz (4,572 clips, D 640)
REF_STATE=$W/qhead_dose/states/LSD_s0_dose_train.npz
LEAD_BYTES=6332746                    # MEASURED: _dose_train_lead.npz 6,001,012 + .head.npz 331,734
ROLL_BYTES=30000000                   # about 25 MB per rollout (ESTIMATED from the ADD42 pulls) plus the logs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8

say() {
  local line="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] integpre: $*"
  echo "$line"
  if [ "$DRY" != 1 ] && [ -d "$RUNF" ]; then echo "$line" >> "$LOG"; fi
}
stop() {  # reason
  say "ZZ INTEGPRE STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$RUNF"; printf 'INTEGPRE_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$RUNF/INTEGPRE_STOP.txt"; fi
  exit 4
}
refuse() { echo "REFUSED: $*"; exit 2; }

gpu_jobs() {  # number of GPU compute processes; an unreadable list counts as busy
  local o rc
  o=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo 99; return; fi
  printf '%s\n' "$o" | grep -c .
}
free_gb() {  # free memory of the container in whole GB: memory.max minus the anon line of memory.stat
  local maxb anon freeb
  maxb=$(cat "$CG_MAX" 2>/dev/null || echo max)
  anon=$(awk '$1=="anon"{print $2}' "$CG_STAT" 2>/dev/null)
  if [ "$maxb" = max ] || [ -z "$maxb" ]; then
    freeb=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo 2>/dev/null)
  elif [ -z "$anon" ]; then
    echo 0; return
  else
    freeb=$((maxb - anon))
  fi
  freeb=${freeb:-0}
  echo $((freeb / 1073741824))
}
slot_probe() {
  local n g
  n=$(gpu_jobs); g=$(free_gb)
  say "slot probe: gpu_jobs=$n (need < $MAX_GPU_JOBS), container free=${g} GB (need > $MIN_FREE_GB)"
  [ "$n" -lt "$MAX_GPU_JOBS" ] && [ "$g" -gt "$MIN_FREE_GB" ]
}
slot_wait() {  # label: two probes PROBE_GAP_S apart must both find a slot
  if [ "$DRY" = 1 ]; then echo "DRY: slot wait before $1 (fewer than $MAX_GPU_JOBS GPU compute processes and more than $MIN_FREE_GB GB free, two probes ${PROBE_GAP_S} s apart)"; return 0; fi
  SECONDS=0
  while true; do
    if slot_probe; then
      sleep "$PROBE_GAP_S"
      if slot_probe; then say "slot free on two probes for $1"; return 0; fi
    fi
    [ "$SECONDS" -ge "$MAX_WAIT_S" ] && stop "no free slot for $1 after ${SECONDS} s (limit ${MAX_WAIT_S} s): never a fourth GPU job"
    say "waiting for a slot for $1 (${SECONDS} s of ${MAX_WAIT_S} s)"
    sleep "$POLL_S"
  done
}
sha_of() { grep -o '"ckpt_sha256": "[0-9a-f]*"' "$1" 2>/dev/null | head -n 1 | cut -d'"' -f4; }
pin() {  # file sha256 : a pinned input must have exactly this sha256
  local got
  got=$(sha256sum "$1" 2>/dev/null | cut -c1-64)
  [ "$got" = "$2" ] || stop "pinned file $1 has sha256 ${got:-none}, expected $2"
}

# ---- the arms
if [ "$DRY" = 1 ] && [ ! -f "$ARMS" ]; then
  ARMLINES="IRQ_RLAm_s0 actcons_c0 RLAm_s0 0 0.5
IRQ_RLFT_s0 actcons_c0 RLFT_s0 0 0.5
IRQ_RLAm_s1 actcons_c1 RLAm_s1 1 0.5
IRQ_RLFT_s1 actcons_c1 RLFT_s1 1 0.5
IRQ_RLAm_s2 actcons_e2 RLAm_s2 2 0.5
IRQ_RLFT_s2 actcons_e2 RLFT_s2 2 0.5
IRQ_RLAm_s3 actcons_e3 RLAm_s3 3 0.5
IRQ_RLFT_s3 actcons_e3 RLFT_s3 3 0.5"
else
  [ -f "$ARMS" ] || refuse "missing $ARMS"
  ARMLINES=$(grep -v '^[[:space:]]*#' "$ARMS" | grep -v '^[[:space:]]*$')
fi
NARMS=$(printf '%s\n' "$ARMLINES" | grep -c .)
[ "$NARMS" -ge 1 ] || refuse "no arms in $ARMS"

# ---- expected volume of the state dumps (printed before the first extract; refused if the volume has less than that plus the margin)
if [ -f "$REF_STATE" ]; then ONE=$(stat -c %s "$REF_STATE"); else ONE=$REF_STATE_BYTES; fi
EXPECT=$(( NARMS * (ONE + LEAD_BYTES) + (3 * NARMS + 2) * ROLL_BYTES ))
say "expected volume: $NARMS checkpoints x (state dump $ONE + lead read $LEAD_BYTES) + $((3 * NARMS + 2)) rollouts x $ROLL_BYTES = $EXPECT bytes ($((EXPECT / 1048576)) MB); margin ${VOL_MARGIN_GB} GB"
if [ "$DRY" = 1 ]; then
  echo "DRY: volume check: stat -f -c '%a %S' $W (free = available blocks x block size); refuse if free < expected + ${VOL_MARGIN_GB} GB"
else
  VOLFREE=$(stat -f -c '%a %S' "$W" 2>/dev/null | awk '{printf "%.0f", $1 * $2}')
  [[ "$VOLFREE" =~ ^[0-9]+$ ]] || refuse "cannot read the free space of $W with stat -f"
  [ "$VOLFREE" -ge $(( EXPECT + VOL_MARGIN_GB * 1073741824 )) ] || refuse "volume of $W has $VOLFREE bytes free, less than the expected $EXPECT plus ${VOL_MARGIN_GB} GB"
fi

if [ "$DRY" = 1 ]; then
  echo "DRY=1: commands only (no check, no wait, no marker, no run)"
else
  for m in INTEGPRE_DONE.txt INTEGPRE_STOP.txt INTEGPRE_STARTED.txt; do
    [ -e "$RUNF/$m" ] && refuse "$RUNF/$m exists (a second start is refused; the lead moves the marker if a restart is meant)"
  done
  mkdir -p "$RUNF" "$ST" "$HEADS" || refuse "cannot create folders under $D"
  ( set -o noclobber; printf 'INTEGPRE_STARTED %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUNF/INTEGPRE_STARTED.txt" ) 2>/dev/null || refuse "another start created $RUNF/INTEGPRE_STARTED.txt first"
  for f in pod_ruleattr_run.sh dist_rule.py cmp_dumps.py v8f_t1_eval_rule.py v8f_t1_eval_rule_mix.py v8f_t1_eval_rule_qhead.py qhead_model.py qhead_fit_n1.py; do
    [ -f "$D/$f" ] || stop "missing $D/$f"
  done
  for f in extract_states.py pod_lead_apply.py; do [ -f "$PROBE/$f" ] || stop "missing $PROBE/$f"; done
  [ -f "$PACK" ] || stop "missing train pack $PACK"
  [ -d "$CACHE/train/clips" ] || stop "missing token cache $CACHE/train/clips"
  [ -x "$PY" ] || stop "no python $PY"
  # pinned inputs (sha256 MEASURED 2026-11-19 on the pod and the dev box)
  pin "$PROBE/extract_states.py" f7cfb17e72a822a477d5e4f8855b55c788738fcfc03ac5220114949f6efbdbe8
  pin "$PROBE/pod_lead_apply.py" c9b1ecef2302100bbd52bfb37ff6ddbdb181c61dbf555a8c39cc94bd9eb97d3c
  pin "$PACK" 2c1a47a5dd831562d0b684b80c845becb5bfeb6f6342b1a1b1cfbcd903be869c
  pin "$D/qhead_fit_n1.py" 57e1242713e0b42eab05cf142b8d152fbb87db94354290ddd3f275f06292036b
  pin "$D/qhead_model.py" f6ff755e3dd4e21247c9d4034a39deb26ac09c6b9a7135b05534278df639de0b
  pin "$D/pod_ruleattr_run.sh" 7b7494135d3aeb96fc4ecdf38e62cbb9dea71c76dadf31d28aeab40dc115c134
  pin "$D/dist_rule.py" 40b5dd9173aeaf8c7d7a7d4bfbb52484a8f30b9a8480d3356f8fa0aee65746dc
  pin "$D/cmp_dumps.py" 49f727c42e708447821863e0b4bcde2afa24043c281a80283fbd1e2340ff22fa
  pin "$D/v8f_t1_eval_rule.py" 51e196758cd878fca261312ca09f53c1a57c0131b6b3c50bc56d5bd98c553a03
  pin "$D/v8f_t1_eval_rule_mix.py" 6d69688ef87d3b569547111c1f46e988d36b31f4ba08d802e44f09a722492dff
  pin "$D/v8f_t1_eval_rule_qhead.py" 21cf7ac38d344fd715eecc80c0322eedeca433a546955fa7542a397a5fbc8ae0
  if [ ! -f "$RUN_Q" ]; then
    sed -e 's|ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|ARGV=(--head-p-min "$4" $5 ${6:-} --tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3")|' "$RUN_PLAIN" > "$RUN_Q"
  fi
  [ "$(grep -c -- '--head-p-min' "$RUN_Q")" = 1 ] && bash -n "$RUN_Q" || stop "the derived run script is not as expected"
  say "queue start; run script $(sha256sum "$RUN_Q" | cut -c1-16) tool $(sha256sum "$D/v8f_t1_eval_rule_qhead.py" | cut -c1-16) fit $(sha256sum "$D/qhead_fit_n1.py" | cut -c1-16) arms $(sha256sum "$ARMS" | cut -c1-16) tree $TREE"
fi

# runs one command; a failure is a STOP (never a silent continue)
step() {  # label, logfile, command...
  local label=$1 lf=$2; shift 2
  if [ "$DRY" = 1 ]; then echo "DRY: $* > $lf 2>&1"; return 0; fi
  say "run: $label"
  "$@" > "$lf" 2>&1 < /dev/null || stop "$label failed: $(tail -n 2 "$lf" | cut -c1-200 | tr '\n' ' ')"
  say "$label: $(tail -n 1 "$lf" | cut -c1-220)"
}

# rollout: roll NEWNAME QUEUE BASE MODE P FLAG1 FLAG2   (MODE off | label | head use the banked run script; MODE qhead uses the derived one with the rule tool of the quantile head)
roll() {
  local NEW=$1 Q=$2 BASE=$3 MODE=$4 P=$5 F1=${6:-} F2=${7:-} rc
  if [ "$DRY" = 1 ]; then
    slot_wait "rollout $NEW"
    if [ "$MODE" = qhead ]; then
      echo "DRY: DR_W=$W DR_TREE=$TREE DR_FLOOR=cap-only DISTRULE_DIR=$D DR_MIX=\"head head head $P $F1 $F2\" DR_RULEPY=v8f_t1_eval_rule_qhead.py bash $RUN_Q $NEW $Q $BASE mix > $RUNF/run_$NEW.out"
    else
      echo "DRY: DR_W=$W DR_TREE=$TREE DR_FLOOR=cap-only DISTRULE_DIR=$D bash $RUN_PLAIN $NEW $Q $BASE $MODE > $RUNF/run_$NEW.out"
    fi
    return 0
  fi
  [ -e "$R/$NEW/T1_DONE" ] && { say "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && stop "output folder $R/$NEW exists without T1_DONE (the lead moves it)"
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || stop "no checkpoint $Q/$BASE"
  slot_wait "rollout $NEW"
  if [ "$MODE" = qhead ]; then
    DR_W=$W DR_TREE=$TREE DR_FLOOR=cap-only DISTRULE_DIR=$D DR_MIX="head head head $P $F1 $F2" DR_RULEPY=v8f_t1_eval_rule_qhead.py bash "$RUN_Q" "$NEW" "$Q" "$BASE" mix > "$RUNF/run_$NEW.out" 2>&1 < /dev/null; rc=$?
  else
    DR_W=$W DR_TREE=$TREE DR_FLOOR=cap-only DISTRULE_DIR=$D bash "$RUN_PLAIN" "$NEW" "$Q" "$BASE" "$MODE" > "$RUNF/run_$NEW.out" 2>&1 < /dev/null; rc=$?
  fi
  say "$NEW ($MODE, P $P $F1 on $Q/$BASE) rc $rc : $(tail -n 1 "$RUNF/run_$NEW.out" | cut -c1-160)"
  [ $rc -eq 0 ] && [ -e "$R/$NEW/T1_DONE" ] || stop "rollout $NEW failed (rc $rc, T1_DONE absent)"
}
cmpd() { "$PY" "$D/cmp_dumps.py" "$R/$1/dump_$1_m10_causal_heads" "$R/$2/dump_$2_m10_causal_heads" 2>&1 | tail -n 1; }

# ---- per checkpoint
nmiss=0
while read -r NEW Q BASE SEED P; do
  [ -n "${NEW:-}" ] || continue
  S="$ST/${BASE}_train.npz"; L="$ST/${BASE}_train_lead.npz"; CK="$W/runs/$Q/$BASE/ckpt_last.pt"
  if [ "$DRY" != 1 ]; then
    [ -f "$CK" ] || stop "missing checkpoint $CK"
  fi
  # 1. states of all training clips, then the lead read
  if [ "$DRY" != 1 ] && [ -f "$S.json" ]; then
    say "$BASE: states already written"
  else
    [ "$DRY" != 1 ] && [ -e "$S" ] && stop "$S exists without its record (partial file; the lead moves it)"
    slot_wait "states $BASE"
    step "states $BASE" "$ST/${BASE}_train.log" nice -n 19 "$PY" "$PROBE/extract_states.py" --tree "$TREE" --ckpt "$CK" --cache "$CACHE" --split train --lo 0 --hi "$HI" --out "$S"
    if [ "$DRY" != 1 ]; then
      grep -q "\"n_clips\": $HI" "$S.json" && grep -q '"lo": 0,' "$S.json" && grep -q "\"hi\": $HI" "$S.json" || stop "$BASE: record $S.json does not say lo 0, hi $HI, n_clips $HI"
      [ -n "$(sha_of "$S.json")" ] && [ "$(sha_of "$S.json")" = "$(sha256sum "$CK" | cut -c1-64)" ] || stop "$BASE: checkpoint sha256 in the states record differs from the checkpoint file"
    fi
  fi
  if [ "$DRY" != 1 ] && [ -f "$L.json" ]; then
    say "$BASE: lead read already written"
  else
    [ "$DRY" != 1 ] && [ -e "$L" ] && stop "$L exists without its record (partial file; the lead moves it)"
    slot_wait "lead read $BASE"
    step "lead read $BASE" "$ST/${BASE}_train_lead.log" nice -n 19 "$PY" "$PROBE/pod_lead_apply.py" --tree "$TREE" --ckpt "$CK" --states "$S" --out "$L"
    if [ "$DRY" != 1 ]; then
      [ -f "$L.json" ] && [ -f "$L.head.npz" ] || stop "$BASE: lead read wrote no record or head file"
      [ "$(sha_of "$L.json")" = "$(sha256sum "$CK" | cut -c1-64)" ] || stop "$BASE: checkpoint sha256 in the lead record differs from the checkpoint file"
    fi
  fi
  # 2. the head fit (form of note ADD42_N2)
  if [ "$DRY" != 1 ] && [ -f "$HEADS/$BASE.qhead.json" ]; then
    say "fit $BASE already written"
  else
    slot_wait "fit $BASE"
    step "fit $BASE" "$RUNF/fit_$BASE.out" "$PY" "$D/qhead_fit_n1.py" --arm "$BASE" --seed "$SEED" --states "$S" --pack "$PACK" --out-dir "$HEADS" --device cpu --fit-clips 3048 --select-clips 762 --check-clips 762 --lead "$L" --banked-head "$L.head.npz" $FITARGS
  fi
  HEAD_OK=1
  if [ "$DRY" != 1 ]; then
    [ -f "$HEADS/$BASE.qhead.json" ] && [ -f "$HEADS/$BASE.qhead.npz" ] || stop "fit $BASE wrote no head or record"
    if ! grep -q '"h5_check_coverage_in_0.15_0.25": true' "$HEADS/$BASE.qhead.json"; then
      HEAD_OK=0; nmiss=$((nmiss+1))
      printf 'MISSING %s: check-clip coverage outside [0.15, 0.25]; the head is not read and the base counts as missing\n' "$BASE" > "$RUNF/MISSING_$BASE.txt"
      say "MISSING $BASE: h5 is not true"
    fi
  fi
  # 3. the rollouts: no rule, label-fed rule, quantile-head rule
  roll "IR0_$BASE" "$Q" "$BASE" off 0
  roll "IRL_$BASE" "$Q" "$BASE" label 0
  if [ "$HEAD_OK" = 1 ]; then
    roll "IRQ_$BASE" "$Q" "$BASE" qhead "$P" --qhead "$HEADS/$BASE.qhead.npz"
  fi
  # control c3 on the candidate at base 0: the head-fed rule against the identity form of the quantile-head tool
  if [ "$BASE" = "$CTRL_BASE" ]; then
    roll "IRH_$BASE" "$Q" "$BASE" head 0
    roll "IRQI_$BASE" "$Q" "$BASE" qhead "$CTRL_P" --q-equals-head
    if [ "$DRY" = 1 ]; then
      echo "DRY: $PY $D/cmp_dumps.py $R/IRQI_$BASE/dump_IRQI_${BASE}_m10_causal_heads $R/IRH_$BASE/dump_IRH_${BASE}_m10_causal_heads  (c3: must read 'differing=0 ')"
    else
      c3=$(cmpd "IRQI_$BASE" "IRH_$BASE")
      say "c3 the identity form (q_head = g_head, P $CTRL_P) against the head-fed rule on $BASE: $c3"
      echo "$c3" | grep -q 'differing=0 ' || stop "C3 (0 files differing was required)"
      say "control c3 passed"
    fi
  fi
done <<EOF
$ARMLINES
EOF

if [ "$DRY" = 1 ]; then echo "DRY: at the end INTEGPRE_DONE.txt in $RUNF holds the count of rollouts with T1_DONE"; exit 0; fi
nd=0; tot=0
while read -r NEW Q BASE SEED P; do
  [ -n "${NEW:-}" ] || continue
  for t in IR0 IRL IRQ; do
    [ "$t" = IRQ ] && [ -f "$RUNF/MISSING_$BASE.txt" ] && continue
    tot=$((tot+1)); [ -e "$R/${t}_$BASE/T1_DONE" ] && nd=$((nd+1))
  done
done <<EOF
$ARMLINES
EOF
say "ZZ INTEGPRE QUEUE DONE: $nd of $tot rollouts with T1_DONE, $nmiss bases missing"
echo "$nd" > "$RUNF/INTEGPRE_DONE.txt"
