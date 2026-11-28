#!/bin/bash
# run_l7cap_thor.sh <seed> [train139|eval139]  -- H-L7-1 train-split fan capture on Thor. WRITTEN, NOT RUN (no compute on Thor was authorised for this task).
#
# Launch (the Thor GPU-lock convention; if you forget the flock the script re-execs itself under it, waiting up to $L7C_LOCK_WAIT_S):
#     flock -o /home/nvidia/refcv7_post/thor_gpu.lock bash run_l7cap_thor.sh 0 train139
# `flock -o` closes the lock fd in the child; every spawned child below ALSO carries `200>&-` (CLAUDE.md lock-fd trap).
#
# Everything is read from ./ (CODE=/home/nvidia/r8ns/l7cap/code, shipped by the Master Mind with md5 check) and from the md5-verified fanceil tree.
# OPAQUE MARKERS in $O/l7cap.log:  ZZL7C<NAME>ZZ  (never grep this file for text that appears in this script; parse the ZZ..ZZ tokens it prints).
#   START / WAIT / MD5OK / MD5BAD / BUSY / A6OK / A6BAD / ROLLEND / SCRUBOK / SCRUBBAD / OUT / END     Exit: 2 = an input md5 differs, 3 = GPU busy / smi failed / A6 bad, 4 = roll ok but the clip-id scrub did not read CLEAN (do not bank).
set -u
SEED=${1:?seed}; MODE=${2:-train139}
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
if [ "${L7C_LOCKED:-0}" != 1 ]; then
  echo "ZZL7CWAITZZ re-exec under flock $LOCK"
  exec flock -o -w "${L7C_LOCK_WAIT_S:-86400}" "$LOCK" env L7C_LOCKED=1 bash "$0" "$@"
fi
R=/home/nvidia/r8ns; F=$R/fanceil; L=$R/l7cap; CODE=$L/code; O=$L/out/${MODE}_s$SEED; mkdir -p "$O"
PY=/home/nvidia/venvs/tanitad-train/bin/python
CK=$R/ckpt/ckpt_ema_48852.pt; CFG=$R/ckpt/config.json
LOG=$O/l7cap.log
AGJ=/home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz
SAM=/home/nvidia/data/sam3_gt_v3
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION L7CAP_MUTATION
export R8_LAUNCH_TREE=$F/tree R8_EVAL_KIT=/home/nvidia PF_R8_CODE=$F/code/batt/code PF_NF_HOLD=$F/code/navfix
export R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$F/tree/stack:$F/tree/taniteval
export L7_STACK_SCRIPTS=$F/tree/stack/scripts
echo "ZZL7CSTARTZZ $(date -Is) seed $SEED mode $MODE pid $$" >> "$LOG"

# ---- 1. md5-verified inputs (immutable data/ckpt: literals measured earlier; code: literals frozen at ship time) -------------------------
bad=0
check() { f=$1; want=$2; got=$(md5sum "$f" 2>/dev/null | cut -c1-32)
  if [ ${#got} -ne 32 ] || [ ${#want} -ne 32 ] || [ "$got" != "$want" ]; then echo "ZZL7CMD5BADZZ $f got=$got want=$want" >> "$LOG"; bad=1; fi; }
check "$CK"  b4364029b68f29a33eeb4aebbbbd002b
check "$CFG" 4ce2c77230dc4e802e78b39ce05af916
check "$CODE/fan_roll_r8_l7cap.py"            d886b0a60b6a02d0b67b7f1605bc3bf1
check "$CODE/l7cap_hooks.py"                   53b5006e3bb0b7a67702238a7925364d
check "$CODE/TRAIN139_WINDOWS.json"            285fc22d119943299969618c8905dfb3
check "$CODE/a6_sizes_md5_sha12.json"          6b0b60ec56e6a7a98d42a7630cbae306
check "$CODE/make_a6cache_thor.py"             50719a553a523bd53979c40a41e7e134
check "$CODE/scrub_clip_ids.py"                db098ac21bb6aaee2f72bad4939291f9
if [ "$MODE" = train139 ]; then
  check "$CODE/refcv8_train_remap_thor_l7cap.json" 058eb283c2102720603dbb4d8a86662b
  check /home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz b45377a1f25263b5c0f3d318c126b1ac
  check /home/nvidia/data/refcv6_speed_max_v8_train.jsonl           bab41adf5d08aef89ebd4596ed48cead
  check /home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz 83e30bd3791f0767707eade5f4671759
else
  check "$F/inputs/windows_s1.json"                          a3a78dba889e70aa69a05271559794ad
  check "$F/code/fanceil/refcv8_eval_remap_thor_fanceil.json" 4dc5fb013cb1ed4533379816f0bc3a22
fi
check "$AGJ" 0c31a3a63d7205e9fa50e8ac4204ef46
[ $bad = 0 ] || exit 2
echo "ZZL7CMD5OKZZ ckpt config code windows remap labels agentjoin" >> "$LOG"

# ---- 2. GPU precheck, fail CLOSED (a failed nvidia-smi is not 'empty') -----------------------------------------------------------------
smi=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); smirc=$?
apps=$(printf '%s' "$smi" | grep -c '[0-9]')
pys=$(ps -u nvidia -o pid=,args= | awk '$2 ~ /python/' | grep -vE 'r8ns/nt/|code/nt_[a-z0-9_]+[.]py' | wc -l)
echo "precheck smi_rc=$smirc compute_apps=$apps other_python=$pys (other_python is informational)" >> "$LOG"
if [ "$smirc" != 0 ] || [ "$apps" != 0 ]; then echo "ZZL7CBUSYZZ $(date -Is)" >> "$LOG"; exit 3; fi

# ---- 3. mode-specific inputs and the roll -----------------------------------------------------------------------------------------------
cd "$O"
if [ "$MODE" = train139 ]; then
  "$PY" "$CODE/make_a6cache_thor.py" --windows "$CODE/TRAIN139_WINDOWS.json" --sizes "$CODE/a6_sizes_md5_sha12.json" --out "$L/a6cache" >> "$LOG" 2>&1 200>&-
  grep -q 'ZZL7CA6OKZZ' "$LOG" || { echo "ZZL7CA6BADZZ" >> "$LOG"; exit 3; }
  MAPARG="--l7-map-gt-root $SAM"; [ "${L7C_NO_MAP:-0}" = 1 ] && MAPARG=""
  "$PY" -X utf8 "$CODE/fan_roll_r8_l7cap.py" --mode train139 --pf-code "$F/code/pf/code" --train-remap-json "$CODE/refcv8_train_remap_thor_l7cap.json" \
      --ckpt "$CK" --config "$CFG" --seed "$SEED" --dump-dir "$O/dump" --out-json "$O/roll_s$SEED.json" --fan-meta-json "$O/fan_roll_s$SEED.json" \
      --registered-windows "$CODE/TRAIN139_WINDOWS.json" --l7-agent-join "$AGJ" $MAPARG --device cuda >> "$LOG" 2>&1 200>&-
else
  "$PY" -X utf8 "$CODE/fan_roll_r8_l7cap.py" --mode eval139 --pf-code "$F/code/pf/code" --remap-json "$F/code/fanceil/refcv8_eval_remap_thor_fanceil.json" \
      --ckpt "$CK" --config "$CFG" --seed "$SEED" --dump-dir "$O/dump" --windows-json "$F/inputs/windows_s1.json" --out-json "$O/roll_s$SEED.json" \
      --fan-meta-json "$O/fan_roll_s$SEED.json" --l7-agent-join "$AGJ" --l7-map-gt-root $SAM --device cuda --os-only >> "$LOG" 2>&1 200>&-
fi
rc=$?
echo "ZZL7CROLLENDZZ rc=$rc $(date -Is)" >> "$LOG"

# ---- 3b. clip-id rule: rewrite raw clip ids in this run's text artifacts (dump manifest, roll log) to clip-sha12-<12 hex> BEFORE anything is banked ----
"$PY" "$CODE/scrub_clip_ids.py" --root "$O" --json "$O/SCRUB_CLIP_IDS.json" >> "$LOG" 2>&1 200>&-
scr=0; [ "$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['CLEAN']))" "$O/SCRUB_CLIP_IDS.json" 2>/dev/null 200>&-)" = 1 ] && scr=1

# ---- 4. assert on the ARTIFACTS, not the status ------------------------------------------------------------------------------------------
fan=0; sc=0; lb=0; mt=0
[ -s "$O/dump/fan_extras.npz" ] && fan=1; [ -s "$O/dump/l7cap_scene.npz" ] && sc=1; [ -s "$O/dump/l7cap_labels.npz" ] && lb=1; [ -s "$O/fan_roll_s$SEED.json" ] && mt=1
if [ "$fan$sc$mt" = 111 ]; then
  (cd "$O" && md5sum dump/fan_extras.npz dump/l7cap_scene.npz dump/refcv6_extras.npz fan_roll_s$SEED.json roll_s$SEED.json $( [ $lb = 1 ] && echo dump/l7cap_labels.npz ) > MD5_L7C$SEED.txt) 200>&-
fi
echo "ZZL7COUTZZ fan=$fan scene=$sc labels=$lb meta=$mt scrub=$scr" >> "$LOG"
echo "ZZL7CENDZZ rc=$rc $(date -Is)" >> "$LOG"
if [ "$rc" = 0 ] && [ "$scr" != 1 ]; then exit 4; fi        # 4 = the roll worked but the clip-id scrub did not read CLEAN: do not bank
exit $rc
