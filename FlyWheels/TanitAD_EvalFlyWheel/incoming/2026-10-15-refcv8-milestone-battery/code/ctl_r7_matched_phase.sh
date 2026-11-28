#!/bin/sh
# ctl_r7_matched_phase.sh -- the KNOWN-VALUE CONTROL of PREREG_R7_MATCHED_PHASE_TURN.md section 2.
#   Re-rolls refcv7-50,400 (ckpt_50400.pt, md5 b418d0fc...) through the path the milestones will use (run_battery_r7m.py = run_battery_r8.py + the
#   refcv7 remap; standalone pose clock; inference seeds 0 and 1) on a clip-restricted S2 subset, under the shared GPU lock (job refcv7-matched-ctl),
#   then compares the dumps with the BANKED step50400_a6text dumps on the same windows (compare_r7_control.py).
#   usage: nohup sh ctl_r7_matched_phase.sh [MAX_CLIPS=4] < /dev/null > ... &
#   artifacts: C:/Users/Admin/refcv8_eval_archive/battery/r7ctl_step50400_smoke/ (dumps)  and  raw/r7_matched_phase/KNOWN_VALUE_CONTROL.{json,md}
#   terminal marker: ZZR7MCTLENDZZ in E:/refcv8_eval_kit/chain/chain_r7_matched_phase.log
MAXC="${1:-4}"
. "$(dirname "$0")/r7m_common.sh"
TAG=r7ctl_step50400_smoke; JOB=refcv7-matched-ctl
mark CTLSTART "pid $$ max_clips $MAXC"
verify_ckpt 50400 || { mark CTLCKPTBAD "ckpt_50400.pt md5 != MD5SUMS"; mark CTLEND "ABORT"; exit 2; }
mark CTLCKPTOK "ckpt_50400.pt md5 verified"
gpu_acquire "$JOB" || { mark CTLEND "ABORT no lock"; exit 3; }
STOPW="$KIT/chain/r7m_gpu_watch_$JOB.stop"; rm -f "$STOPW"
"$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/r7m_gpu_watch_$JOB.jsonl" --stop-file "$KITW/chain/r7m_gpu_watch_$JOB.stop" >> "$KIT/chain/r7m_lockwait_$JOB.log" 2>&1 &
trap ': > "$STOPW"; gpu_release "$JOB"' EXIT
date +%s > "$KIT/chain/r7m_ctl_t0.txt"
mark CTLROLL "rolling $TAG"
r7_pass "$TAG" "$CKPT_DIR/ckpt_50400.pt" --max-clips "$MAXC"
mark CTLROLLDONE "battery returned; summary present: $(test -s "$ARCL/$TAG/battery_summary.json" && echo yes || echo no)"
# the dumps are the evidence; the panel/bars stage is a bonus (it needs >= 3 episodes to bootstrap) -- compare whatever dumps exist
"$PY" -I -X utf8 "$PKGW/code/compare_r7_control.py" --new-root "$ARCW/$TAG" --banked-root "$R7KITW/battery/step50400_a6text" --seeds 0,1 \
    --out-json "$RAWW/KNOWN_VALUE_CONTROL.json" --out-md "$RAWW/KNOWN_VALUE_CONTROL.md" > "$KIT/chain/r7m_ctl_compare.out" 2>&1
if jq_ok "$RAWL/KNOWN_VALUE_CONTROL.json" "j.get('verdict')"; then
  mark CTLVERDICT "$("$PY" -I -X utf8 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['verdict'][:60])" "$RAWW/KNOWN_VALUE_CONTROL.json")"
else mark CTLNOVERDICT "compare wrote no json (chain/r7m_ctl_compare.out)"; fi
mark CTLEND "done"
