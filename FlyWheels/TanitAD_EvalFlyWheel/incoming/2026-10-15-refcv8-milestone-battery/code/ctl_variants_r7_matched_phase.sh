#!/bin/sh
# ctl_variants_r7_matched_phase.sh -- the two ALTERNATIVE paths of the known-value control (PREREG_R7_MATCHED_PHASE_TURN.md section 2: "if the wrapper
# path fails but the native path passes (or vice versa), use the one that passes").  Run AFTER ctl_r7_matched_phase.sh read FAIL on the wrapper path.
#   native   : the launch tree's own run_battery_r7.py, only its stale BASELINES paths re-pointed (run_battery_r7_native.py), unpatched roll child.
#   ceiloff  : the wrapper path + the max-speed ceiling filter switched OFF in the main forward (R7M_CEIL_OFF=1) -- DIAGNOSTIC: tests the cause
#              found in the wrapper-path control (filter inert in the banked roll, acting in the launch-tree path).
#   e9noceil : the wrapper path + E9's final re-rank ignores the ceiling mask (R7M_E9_NO_CEIL=1) -- the MECHANISM-faithful emulation of the banked roll.
#   Same clips / seeds / lock discipline as the wrapper control (job refcv7-matched-ctl2).  Markers ZZR7MV...ZZ.
#   usage: nohup sh ctl_variants_r7_matched_phase.sh [MAX_CLIPS=4] [VARIANTS="native ceiloff"] < /dev/null > ... &
MAXC="${1:-4}"; VARS="${2:-native ceiloff}"
. "$(dirname "$0")/r7m_common.sh"
JOB=refcv7-matched-ctl2
mark VSTART "pid $$ max_clips $MAXC variants '$VARS'"
verify_ckpt 50400 || { mark VCKPTBAD "ckpt_50400.pt md5 != MD5SUMS"; mark VEND "ABORT"; exit 2; }
gpu_acquire "$JOB" || { mark VEND "ABORT no lock"; exit 3; }
STOPW="$KIT/chain/r7m_gpu_watch_$JOB.stop"; rm -f "$STOPW"
"$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/r7m_gpu_watch_$JOB.jsonl" --stop-file "$KITW/chain/r7m_gpu_watch_$JOB.stop" >> "$KIT/chain/r7m_lockwait_$JOB.log" 2>&1 &
trap ': > "$STOPW"; gpu_release "$JOB"' EXIT
for V in $VARS; do
  TAG="r7ctl${V}_step50400_smoke"
  mark VROLL "$V -> $TAG"
  case "$V" in
    native)
      "$PY" "$PKGW/code/run_battery_r7_native.py" --ckpt "$CKPT_DIR/ckpt_50400.pt" --config "$CFG7" --no-g0 --tag "$TAG" --infer-seeds 0,1 --lock-held \
          --skip-roll --out-root "$ARCW" --max-clips "$MAXC" >> "$KIT/chain/r7m_run_$TAG.log" 2>&1 ;;
    ceiloff)
      ( export R7M_CEIL_OFF=1; r7_pass "$TAG" "$CKPT_DIR/ckpt_50400.pt" --max-clips "$MAXC" ) ;;
    e9noceil)
      ( export R7M_E9_NO_CEIL=1; r7_pass "$TAG" "$CKPT_DIR/ckpt_50400.pt" --max-clips "$MAXC" ) ;;
    *) mark VUNKNOWN "$V" ; continue ;;
  esac
  mark VROLLDONE "$V: summary present: $(test -s "$ARCL/$TAG/battery_summary.json" && echo yes || echo no)"
  "$PY" -I -X utf8 "$PKGW/code/compare_r7_control.py" --new-root "$ARCW/$TAG" --banked-root "$R7KITW/battery/step50400_a6text" --seeds 0,1 \
      --out-json "$RAWW/KNOWN_VALUE_CONTROL_$V.json" --out-md "$RAWW/KNOWN_VALUE_CONTROL_$V.md" > "$KIT/chain/r7m_ctl_compare_$V.out" 2>&1
  if jq_ok "$RAWL/KNOWN_VALUE_CONTROL_$V.json" "j.get('verdict')"; then
    mark VVERDICT "$V: $("$PY" -I -X utf8 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['verdict'][:70])" "$RAWW/KNOWN_VALUE_CONTROL_$V.json")"
  else mark VNOVERDICT "$V: compare wrote no json (chain/r7m_ctl_compare_$V.out)"; fi
done
mark VEND "done"
